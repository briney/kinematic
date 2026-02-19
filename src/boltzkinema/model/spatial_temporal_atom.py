"""Spatial-Temporal atom encoder/decoder blocks."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

import boltz.model.layers.initialize as init
from boltz.model.modules.encodersv2 import get_indexing_matrix, single_to_keys
from boltz.model.modules.transformersv2 import AtomTransformer
from boltz.model.modules.utils import LinearNoBias

from boltzkinema.model.temporal_attention import TemporalAttentionWithDecay


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
            Additional multiplicity (default 1, unused in standard BoltzKinema).

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
        M = q.shape[1]
        D = q.shape[2]
        W = self.atom_encoder.attn_window_queries
        H = self.atom_encoder.attn_window_keys

        # --- Expand per frame ---
        if self.structure_prediction:
            q = q.repeat_interleave(T, 0)  # (B*T, M, D)
            q = q + self.r_to_q_trans(r)

        c = c.repeat_interleave(T, 0)  # (B*T, M, D)
        atom_mask = feats["atom_pad_mask"].bool()  # (B, M)
        atom_mask = atom_mask.repeat_interleave(T, 0)  # (B*T, M)

        B = B_orig * T  # effective batch
        NW = M // W

        # --- Window ---
        q_win = q.view(B * NW, W, D)
        c_win = c.view(B * NW, W, D)
        mask_win = atom_mask.view(B * NW, W)

        # --- Expand and split bias ---
        # atom_enc_bias: (B_orig, K, W, H, depth*heads)
        bias_expanded = atom_enc_bias.repeat_interleave(T, 0)  # (B*T, K, W, H, depth*heads)
        bias_expanded = bias_expanded.view(B * NW, W, H, -1)  # (B*T*NW, W, H, depth*heads)
        layers = self.atom_encoder.diffusion_transformer.layers
        L = len(layers)
        heads_per_layer = bias_expanded.shape[-1] // L
        bias_split = bias_expanded.view(B * NW, W, H, L, heads_per_layer)

        # --- Build to_keys_new (same as AtomTransformer.forward) ---
        to_keys_new = lambda x: to_keys(x.view(B, NW * W, -1)).view(B * NW, H, -1)

        # --- Layer loop: spatial → temporal ---
        for i, spatial_layer in enumerate(layers):
            bias_l = bias_split[:, :, :, i]  # (B*T*NW, W, H, heads)

            q_win = spatial_layer(
                q_win, c_win, bias_l, mask_win.float(), to_keys_new, multiplicity=1,
            )

            # Temporal attention: unwindow → reshape → temporal → reshape → rewindow
            q_flat = q_win.view(B, M, D)  # (B*T, M, D)
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
            q_win = q_flat.view(B * NW, W, D)

        # --- Unwindow ---
        q = q_win.view(B, M, D)  # (B*T, M, D)

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
        M = q.shape[1]
        D = q.shape[2]
        W = self.atom_decoder.attn_window_queries
        H = self.atom_decoder.attn_window_keys
        B = B_orig * T
        NW = M // W

        # Window
        q_win = q.view(B * NW, W, D)
        c_win = c.view(B * NW, W, D)
        mask_win = atom_mask.view(B * NW, W)

        # Expand and split bias
        bias_expanded = atom_dec_bias.repeat_interleave(T, 0)
        bias_expanded = bias_expanded.view(B * NW, W, H, -1)
        layers = self.atom_decoder.diffusion_transformer.layers
        L = len(layers)
        heads_per_layer = bias_expanded.shape[-1] // L
        bias_split = bias_expanded.view(B * NW, W, H, L, heads_per_layer)

        to_keys_new = lambda x: to_keys(x.view(B, NW * W, -1)).view(B * NW, H, -1)

        # Layer loop: spatial → temporal
        for i, spatial_layer in enumerate(layers):
            bias_l = bias_split[:, :, :, i]

            q_win = spatial_layer(
                q_win, c_win, bias_l, mask_win.float(), to_keys_new, multiplicity=1,
            )

            # Temporal attention
            q_flat = q_win.view(B, M, D)
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
            q_win = q_flat.view(B * NW, W, D)

        # Unwindow
        q = q_win.view(B, M, D)

        # Atom features → coordinate update
        r_update = self.atom_feat_to_atom_pos_update(q)
        return r_update
