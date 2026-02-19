"""Spatial-Temporal token transformer block."""

from __future__ import annotations

import torch
import torch.nn as nn

from boltz.model.layers.attentionv2 import AttentionPairBias
from boltz.model.modules.transformersv2 import AdaLN, ConditionedTransitionBlock
from boltz.model.modules.utils import default

from kinematic.model.temporal_attention import TemporalAttentionWithDecay


class SpatialTemporalTokenTransformerBlock(nn.Module):
    """Single block: spatial attention → temporal attention → conditioned transition.

    Sub-module names match ``DiffusionTransformerLayer`` for weight loading.
    """

    def __init__(
        self,
        heads: int,
        dim: int = 768,
        dim_single_cond: int | None = None,
        post_layer_norm: bool = False,
        temporal_heads: int | None = None,
        causal: bool = False,
    ) -> None:
        super().__init__()
        dim_single_cond = default(dim_single_cond, dim)
        temporal_heads = temporal_heads or heads

        # Spatial attention (names match DiffusionTransformerLayer)
        self.adaln = AdaLN(dim, dim_single_cond)
        self.pair_bias_attn = AttentionPairBias(
            c_s=dim, num_heads=heads, compute_pair_bias=False,
        )

        self.output_projection_linear = nn.Linear(dim_single_cond, dim)
        nn.init.zeros_(self.output_projection_linear.weight)
        nn.init.constant_(self.output_projection_linear.bias, -2.0)
        self.output_projection = nn.Sequential(
            self.output_projection_linear, nn.Sigmoid(),
        )

        self.transition = ConditionedTransitionBlock(
            dim_single=dim, dim_single_cond=dim_single_cond,
        )

        if post_layer_norm:
            self.post_lnorm = nn.LayerNorm(dim)
        else:
            self.post_lnorm = nn.Identity()

        # Temporal attention (NEW — not loaded from Boltz-2)
        self.temporal_attn = TemporalAttentionWithDecay(dim, temporal_heads, causal)

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
        to_keys,
        multiplicity: int,
        timestamps: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        a : (B*T, N, D) — token activations.
        s : (B*T, N, D) — single conditioning.
        bias : (B, N, N, heads) — pair bias for this layer (unexpanded).
        mask : (B*T, N) — padding mask.
        to_keys : optional key-gathering function (None for token transformer).
        multiplicity : int — set to T for bias expansion in AttentionPairBias.
        timestamps : (B, T) — frame timestamps.
        T : int — number of temporal frames.
        """
        # --- Spatial attention ---
        b = self.adaln(a, s)

        k_in = b
        if to_keys is not None:
            k_in = to_keys(b)
            mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)

        b = self.pair_bias_attn(
            s=b, z=bias, mask=mask, multiplicity=multiplicity, k_in=k_in,
        )
        b = self.output_projection(s) * b
        a = a + b

        # --- Temporal attention ---
        BT, N, D = a.shape
        B = BT // T
        # (B*T, N, D) → (B, T, N, D) → (B, N, T, D) → (B*N, T, D)
        a = a.view(B, T, N, D).permute(0, 2, 1, 3).reshape(B * N, T, D)
        a = self.temporal_attn(a, timestamps, n_spatial=N)
        # (B*N, T, D) → (B, N, T, D) → (B, T, N, D) → (B*T, N, D)
        a = a.view(B, N, T, D).permute(0, 2, 1, 3).reshape(B * T, N, D)

        # --- Conditioned transition ---
        a = a + self.transition(a, s)
        a = self.post_lnorm(a)

        return a


class SpatialTemporalTokenTransformer(nn.Module):
    """Container for the 24-layer token transformer with temporal attention.

    Replaces ``DiffusionTransformer`` for the token-level transformer.
    Splits the concatenated pair bias and dispatches per-layer slices.
    """

    def __init__(
        self,
        depth: int,
        heads: int,
        dim: int = 768,
        dim_single_cond: int | None = None,
        temporal_heads: int | None = None,
        causal: bool = False,
        activation_checkpointing: bool = False,
        post_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        dim_single_cond = default(dim_single_cond, dim)
        self.activation_checkpointing = activation_checkpointing

        self.layers = nn.ModuleList([
            SpatialTemporalTokenTransformerBlock(
                heads=heads,
                dim=dim,
                dim_single_cond=dim_single_cond,
                post_layer_norm=post_layer_norm,
                temporal_heads=temporal_heads,
                causal=causal,
            )
            for _ in range(depth)
        ])

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
        to_keys=None,
        multiplicity: int = 1,
        timestamps: torch.Tensor | None = None,
        T: int = 1,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        a : (B*T, N, D)
        s : (B*T, N, D)
        bias : (B, N, N, depth*heads) — concatenated pair bias.
        mask : (B*T, N)
        to_keys : optional key-gathering function.
        multiplicity : int — T for bias expansion.
        timestamps : (B, T)
        T : int
        """
        # Split bias: (B, N, N, depth*heads) → (B, N, N, depth, heads)
        B_bias, N, _, D_total = bias.shape
        L = len(self.layers)
        bias = bias.view(B_bias, N, N, L, D_total // L)

        for i, layer in enumerate(self.layers):
            bias_l = bias[:, :, :, i]  # (B, N, N, heads)

            if self.activation_checkpointing and self.training:
                a = torch.utils.checkpoint.checkpoint(
                    layer, a, s, bias_l, mask, to_keys,
                    multiplicity, timestamps, T,
                    use_reentrant=False,
                )
            else:
                a = layer(a, s, bias_l, mask, to_keys, multiplicity, timestamps, T)

        return a
