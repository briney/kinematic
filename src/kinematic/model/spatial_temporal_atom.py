"""Spatial-Temporal atom encoder/decoder blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

import boltz.model.layers.initialize as init
from boltz.model.modules.transformersv2 import AtomTransformer
from boltz.model.modules.utils import LinearNoBias

from kinematic.model.temporal_attention import TemporalAttentionWithDecay


def _num_windows(n_items: int, window_size: int) -> int:
    """Return ceil(n_items / window_size) for positive inputs."""
    if n_items <= 0:
        raise ValueError(f"n_items must be positive, got {n_items}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    return (n_items + window_size - 1) // window_size


def _pad_or_trim_dim(
    x: torch.Tensor,
    target: int,
    dim: int,
    pad_value: float | bool = 0.0,
) -> torch.Tensor:
    """Pad or trim tensor ``x`` along ``dim`` to ``target`` length."""
    current = x.shape[dim]
    if current == target:
        return x

    if current > target:
        slices = [slice(None)] * x.dim()
        slices[dim] = slice(0, target)
        return x[tuple(slices)]

    pad_shape = list(x.shape)
    pad_shape[dim] = target - current
    pad = x.new_full(pad_shape, pad_value)
    return torch.cat((x, pad), dim=dim)


def _to_windowed_keys(
    keys: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    n_windows: int,
) -> torch.Tensor:
    """Normalize key tensor layouts to ``(B*NW, H, D)`` and align NW."""
    if keys.dim() == 4:
        if keys.shape[0] != batch_size:
            raise ValueError(
                f"Unexpected key batch dim {keys.shape[0]} != {batch_size}"
            )
        if keys.shape[2] == n_heads:
            key_windows = keys
        elif keys.shape[1] == n_heads:
            key_windows = keys.transpose(1, 2)
        else:
            raise ValueError(
                f"Could not infer key head dimension in shape {tuple(keys.shape)}"
            )
    elif keys.dim() == 3:
        if keys.shape[0] == batch_size:
            if keys.shape[1] % n_heads != 0:
                raise ValueError(
                    f"Key tensor shape {tuple(keys.shape)} is not divisible by n_heads={n_heads}"
                )
            key_windows = keys.reshape(
                batch_size,
                keys.shape[1] // n_heads,
                n_heads,
                keys.shape[2],
            )
        else:
            if keys.shape[1] != n_heads or keys.shape[0] % batch_size != 0:
                raise ValueError(
                    f"Could not infer key window layout from shape {tuple(keys.shape)}"
                )
            key_windows = keys.reshape(
                batch_size,
                keys.shape[0] // batch_size,
                n_heads,
                keys.shape[2],
            )
    else:
        raise ValueError(f"Unsupported key tensor rank: {keys.dim()}")

    key_windows = _pad_or_trim_dim(
        key_windows,
        target=n_windows,
        dim=1,
        pad_value=0.0,
    )
    return key_windows.reshape(batch_size * n_windows, n_heads, key_windows.shape[-1])


class SpatialTemporalAtomEncoder(nn.Module):
    """Atom-level encoder that interleaves spatial and temporal attention.

    Keeps an ``AtomTransformer`` for weight-loading compatibility, but drives
    its ``DiffusionTransformerLayer`` layers manually so that temporal attention
    can be inserted after each spatial layer.

    Sub-module names match ``AtomAttentionEncoder`` for weight loading.
    """

    def __init__(
        self,
        atom_s: int,
        token_s: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        atom_temporal_heads: int | None = None,
        structure_prediction: bool = True,
        activation_checkpointing: bool = False,
        causal: bool = False,
    ) -> None:
        super().__init__()
        atom_temporal_heads = atom_temporal_heads or atom_encoder_heads

        self.structure_prediction = structure_prediction
        if structure_prediction:
            self.r_to_q_trans = LinearNoBias(3, atom_s)
            init.final_init_(self.r_to_q_trans.weight)

        self.atom_encoder = AtomTransformer(
            dim=atom_s,
            dim_single_cond=atom_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            depth=atom_encoder_depth,
            heads=atom_encoder_heads,
            activation_checkpointing=activation_checkpointing,
        )

        self.atom_to_token_trans = nn.Sequential(
            LinearNoBias(atom_s, 2 * token_s),
            nn.ReLU(),
        )

        # Temporal attention layers (NEW — not loaded from Boltz-2)
        self.temporal_layers = nn.ModuleList([
            TemporalAttentionWithDecay(atom_s, atom_temporal_heads, causal)
            for _ in range(atom_encoder_depth)
        ])

    def forward(
        self,
        feats: dict,
        q: torch.Tensor,
        c: torch.Tensor,
        atom_enc_bias: torch.Tensor,
        to_keys,
        r: torch.Tensor,
        timestamps: torch.Tensor,
        T: int,
        multiplicity: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, object]:
        """Forward pass.

        Parameters
        ----------
        feats : dict
            Feature dict with ``atom_pad_mask``, ``atom_to_token``, etc.
        q : (B, M, atom_s)
            Atom query features from DiffusionConditioning.
        c : (B, M, atom_s)
            Atom conditioning features.
        atom_enc_bias : (B, K, W, H, depth*heads)
            Pre-computed atom encoder bias.
        to_keys : callable
            Key-gathering function from AtomEncoder.
        r : (B*T, M, 3)
            Noisy coordinates (one set per frame).
        timestamps : (B, T)
            Frame timestamps in nanoseconds.
        T : int
            Number of temporal frames.
        multiplicity : int
            Additional multiplicity (default 1, unused in standard Kinematic).

        Returns
        -------
        a : (B*T, N, 2*token_s)
            Aggregated token-level activations.
        q_skip : (B*T, M, atom_s)
            Atom-level skip connection for decoder.
        c_skip : (B*T, M, atom_s)
            Conditioning skip connection for decoder.
        to_keys : callable
            Key-gathering function (passed through).
        """
        B_orig = q.shape[0]
        M_orig = q.shape[1]
        D = q.shape[2]
        W = self.atom_encoder.attn_window_queries
        H = self.atom_encoder.attn_window_keys
        NW = _num_windows(M_orig, W)
        M = NW * W

        # --- Expand per frame ---
        if self.structure_prediction:
            q = q.repeat_interleave(T, 0)  # (B*T, M, D)
            q = q + self.r_to_q_trans(r)

        c = c.repeat_interleave(T, 0)  # (B*T, M, D)
        atom_mask = feats["atom_pad_mask"].bool()  # (B, M)
        atom_mask = atom_mask.repeat_interleave(T, 0)  # (B*T, M)
        q = _pad_or_trim_dim(q, target=M, dim=1, pad_value=0.0)
        c = _pad_or_trim_dim(c, target=M, dim=1, pad_value=0.0)
        atom_mask = _pad_or_trim_dim(atom_mask, target=M, dim=1, pad_value=False)

        B = B_orig * T  # effective batch

        # --- Window ---
        q_win = q.reshape(B * NW, W, D)
        c_win = c.reshape(B * NW, W, D)
        mask_win = atom_mask.reshape(B * NW, W)

        # --- Expand and split bias ---
        # atom_enc_bias: (B_orig, K, W, H, depth*heads)
        bias_expanded = atom_enc_bias.repeat_interleave(T, 0)  # (B*T, K, W, H, depth*heads)
        bias_expanded = _pad_or_trim_dim(
            bias_expanded, target=NW, dim=1, pad_value=0.0
        )
        bias_expanded = bias_expanded.reshape(B * NW, W, H, -1)  # (B*T*NW, W, H, depth*heads)
        layers = self.atom_encoder.diffusion_transformer.layers
        L = len(layers)
        heads_per_layer = bias_expanded.shape[-1] // L
        bias_split = bias_expanded.reshape(B * NW, W, H, L, heads_per_layer)

        # --- Build to_keys_new (same as AtomTransformer.forward) ---
        def to_keys_new(x: torch.Tensor) -> torch.Tensor:
            x_flat = x.reshape(B, M, -1)[:, :M_orig]
            keys = to_keys(x_flat)
            return _to_windowed_keys(
                keys,
                batch_size=B,
                n_heads=H,
                n_windows=NW,
            )

        # --- Layer loop: spatial → temporal ---
        for i, spatial_layer in enumerate(layers):
            bias_l = bias_split[:, :, :, i]  # (B*T*NW, W, H, heads)

            q_win = spatial_layer(
                q_win, c_win, bias_l, mask_win.float(), to_keys_new, multiplicity=1,
            )

            # Temporal attention: unwindow → reshape → temporal → reshape → rewindow
            q_flat = q_win.reshape(B, M, D)  # (B*T, M, D)
            q_temp = (
                q_flat
                .view(B_orig, T, M, D)
                .permute(0, 2, 1, 3)
                .reshape(B_orig * M, T, D)
            )
            q_temp = self.temporal_layers[i](q_temp, timestamps, n_spatial=M)
            q_flat = (
                q_temp
                .view(B_orig, M, T, D)
                .permute(0, 2, 1, 3)
                .reshape(B_orig * T, M, D)
            )
            q_win = q_flat.reshape(B * NW, W, D)

        # --- Unwindow ---
        q = q_win.reshape(B, M, D)[:, :M_orig]  # (B*T, M_orig, D)
        c = c[:, :M_orig]

        # --- Aggregate atom → token ---
        with torch.autocast("cuda", enabled=False):
            q_to_a = self.atom_to_token_trans(q).float()
            atom_to_token = feats["atom_to_token"].float()
            atom_to_token = atom_to_token.repeat_interleave(T, 0)
            atom_to_token_mean = atom_to_token / (
                atom_to_token.sum(dim=1, keepdim=True) + 1e-6
            )
            a = torch.bmm(atom_to_token_mean.transpose(1, 2), q_to_a)

        a = a.to(q)

        return a, q, c, to_keys


class SpatialTemporalAtomDecoder(nn.Module):
    """Atom-level decoder that interleaves spatial and temporal attention.

    Same pattern as encoder: keeps ``AtomTransformer`` for weight loading,
    drives layers manually with temporal attention interleaved.

    Sub-module names match ``AtomAttentionDecoder`` for weight loading.
    """

    def __init__(
        self,
        atom_s: int,
        token_s: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        atom_temporal_heads: int | None = None,
        activation_checkpointing: bool = False,
        causal: bool = False,
    ) -> None:
        super().__init__()
        atom_temporal_heads = atom_temporal_heads or atom_decoder_heads

        self.a_to_q_trans = LinearNoBias(2 * token_s, atom_s)
        init.final_init_(self.a_to_q_trans.weight)

        self.atom_decoder = AtomTransformer(
            dim=atom_s,
            dim_single_cond=atom_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            depth=atom_decoder_depth,
            heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
        )

        self.atom_feat_to_atom_pos_update = nn.Sequential(
            nn.LayerNorm(atom_s), LinearNoBias(atom_s, 3),
        )
        init.final_init_(self.atom_feat_to_atom_pos_update[1].weight)

        # Temporal attention layers (NEW — not loaded from Boltz-2)
        self.temporal_layers = nn.ModuleList([
            TemporalAttentionWithDecay(atom_s, atom_temporal_heads, causal)
            for _ in range(atom_decoder_depth)
        ])

    def forward(
        self,
        a: torch.Tensor,
        q: torch.Tensor,
        c: torch.Tensor,
        atom_dec_bias: torch.Tensor,
        feats: dict,
        to_keys,
        timestamps: torch.Tensor,
        T: int,
        multiplicity: int = 1,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        a : (B*T, N, 2*token_s)
            Token activations from transformer.
        q : (B*T, M, atom_s)
            Atom skip connection from encoder.
        c : (B*T, M, atom_s)
            Conditioning skip connection from encoder.
        atom_dec_bias : (B, K, W, H, depth*heads)
            Pre-computed atom decoder bias.
        feats : dict
            Feature dict with ``atom_pad_mask``, ``atom_to_token``, etc.
        to_keys : callable
            Key-gathering function.
        timestamps : (B, T)
        T : int
        multiplicity : int

        Returns
        -------
        r_update : (B*T, M, 3) — coordinate updates.
        """
        B_orig = a.shape[0] // T

        # Broadcast token → atom
        with torch.autocast("cuda", enabled=False):
            atom_to_token = feats["atom_to_token"].float()
            atom_to_token = atom_to_token.repeat_interleave(T, 0)
            a_to_q = self.a_to_q_trans(a.float())
            a_to_q = torch.bmm(atom_to_token, a_to_q)

        q = q + a_to_q.to(q)

        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(T, 0)

        # --- Decomposed atom decoder with temporal attention ---
        M_orig = q.shape[1]
        D = q.shape[2]
        W = self.atom_decoder.attn_window_queries
        H = self.atom_decoder.attn_window_keys
        B = B_orig * T
        NW = _num_windows(M_orig, W)
        M = NW * W
        q = _pad_or_trim_dim(q, target=M, dim=1, pad_value=0.0)
        c = _pad_or_trim_dim(c, target=M, dim=1, pad_value=0.0)
        atom_mask = _pad_or_trim_dim(atom_mask, target=M, dim=1, pad_value=False)

        # Window
        q_win = q.reshape(B * NW, W, D)
        c_win = c.reshape(B * NW, W, D)
        mask_win = atom_mask.reshape(B * NW, W)

        # Expand and split bias
        bias_expanded = atom_dec_bias.repeat_interleave(T, 0)
        bias_expanded = _pad_or_trim_dim(
            bias_expanded, target=NW, dim=1, pad_value=0.0
        )
        bias_expanded = bias_expanded.reshape(B * NW, W, H, -1)
        layers = self.atom_decoder.diffusion_transformer.layers
        L = len(layers)
        heads_per_layer = bias_expanded.shape[-1] // L
        bias_split = bias_expanded.reshape(B * NW, W, H, L, heads_per_layer)

        def to_keys_new(x: torch.Tensor) -> torch.Tensor:
            x_flat = x.reshape(B, M, -1)[:, :M_orig]
            keys = to_keys(x_flat)
            return _to_windowed_keys(
                keys,
                batch_size=B,
                n_heads=H,
                n_windows=NW,
            )

        # Layer loop: spatial → temporal
        for i, spatial_layer in enumerate(layers):
            bias_l = bias_split[:, :, :, i]

            q_win = spatial_layer(
                q_win, c_win, bias_l, mask_win.float(), to_keys_new, multiplicity=1,
            )

            # Temporal attention
            q_flat = q_win.reshape(B, M, D)
            q_temp = (
                q_flat
                .view(B_orig, T, M, D)
                .permute(0, 2, 1, 3)
                .reshape(B_orig * M, T, D)
            )
            q_temp = self.temporal_layers[i](q_temp, timestamps, n_spatial=M)
            q_flat = (
                q_temp
                .view(B_orig, M, T, D)
                .permute(0, 2, 1, 3)
                .reshape(B_orig * T, M, D)
            )
            q_win = q_flat.reshape(B * NW, W, D)

        # Unwindow
        q = q_win.reshape(B, M, D)[:, :M_orig]

        # Atom features → coordinate update
        r_update = self.atom_feat_to_atom_pos_update(q)
        return r_update
