"""TemporalAttentionWithDecay module."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TemporalAttentionWithDecay(nn.Module):
    """Multi-head attention across T frames per spatial position.

    Input shape: (B*N, T, C) where N is the number of spatial positions.
    The temporal decay bias encourages nearby frames to attend more strongly
    to each other: bias_ij^h = -exp(log_lambda_h) * |t_i - t_j|.

    The output projection is zero-initialized so that temporal attention
    starts as an identity operation (critical for loading Boltz-2 weights).
    """

    def __init__(self, dim: int, num_heads: int, causal: bool = False) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal

        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)

        # Output projection — zero-initialized so temporal starts as identity
        self.out_proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        # Learnable decay rates: geometric series from slow to fast decay
        log_lambda = torch.linspace(
            math.log(0.004), math.log(0.7), num_heads
        )
        self.log_lambda = nn.Parameter(log_lambda)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        n_spatial: int,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B*N, T, C)
            Input tensor with temporal dimension.
        timestamps : (B, T)
            Frame timestamps in nanoseconds.
        n_spatial : int
            Number of spatial positions N (used to derive B from B*N).

        Returns
        -------
        (B*N, T, C) — input + gated temporal attention output.
        """
        BN, T, C = x.shape
        B = BN // n_spatial
        H = self.num_heads
        d = self.head_dim

        # Pre-norm
        h = self.norm(x)

        # Q, K, V projections
        q = self.q_proj(h).view(BN, T, H, d)
        k = self.k_proj(h).view(BN, T, H, d)
        v = self.v_proj(h).view(BN, T, H, d)
        g = self.gate(h).sigmoid()

        # Temporal decay bias: -exp(log_lambda_h) * |t_i - t_j|
        # timestamps: (B, T) → dt: (B, T, T)
        dt = (timestamps[:, :, None] - timestamps[:, None, :]).abs()
        decay_rates = self.log_lambda.exp()  # (H,)
        bias = -decay_rates[None, :, None, None] * dt[:, None, :, :]  # (B, H, T, T)

        # Expand bias from (B, H, T, T) to (B*N, H, T, T)
        bias = bias.repeat_interleave(n_spatial, dim=0)

        # Attention
        with torch.autocast("cuda", enabled=False):
            attn = torch.einsum("bthd,bshd->bhts", q.float(), k.float())
            attn = attn / (d**0.5) + bias.float()

            if self.causal:
                causal_mask = torch.triu(
                    torch.ones(T, T, device=x.device, dtype=torch.bool),
                    diagonal=1,
                )
                attn = attn.masked_fill(causal_mask[None, None], float("-inf"))

            attn = attn.softmax(dim=-1)
            o = torch.einsum("bhts,bshd->bthd", attn, v.float()).to(v.dtype)

        o = o.reshape(BN, T, C)
        o = self.out_proj(g * o)

        return x + o
