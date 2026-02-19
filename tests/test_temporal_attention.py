"""Tests for TemporalAttentionWithDecay module."""

from __future__ import annotations

import math

import torch

from kinematic.model.temporal_attention import TemporalAttentionWithDecay


def test_temporal_attention_zero_init_is_identity() -> None:
    layer = TemporalAttentionWithDecay(dim=8, num_heads=2, causal=False)

    x = torch.randn(6, 5, 8)
    timestamps = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.5, 1.5, 2.5, 3.5, 4.5],
        ]
    )

    out = layer(x, timestamps=timestamps, n_spatial=3)
    assert torch.equal(out, x)


def test_temporal_attention_causal_blocks_future_information() -> None:
    layer_causal = TemporalAttentionWithDecay(dim=4, num_heads=1, causal=True)
    layer_noncausal = TemporalAttentionWithDecay(dim=4, num_heads=1, causal=False)

    for layer in (layer_causal, layer_noncausal):
        layer.norm = torch.nn.Identity()
        with torch.no_grad():
            layer.q_proj.weight.zero_()
            layer.q_proj.bias.zero_()
            layer.k_proj.weight.zero_()
            layer.v_proj.weight.copy_(torch.eye(4))
            layer.gate.weight.zero_()
            layer.out_proj.weight.copy_(torch.eye(4))
            layer.log_lambda.fill_(math.log(1e-6))

    x = torch.randn(1, 4, 4)
    x_future_changed = x.clone()
    x_future_changed[:, -1] += 100.0
    timestamps = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    out_causal_a = layer_causal(x, timestamps=timestamps, n_spatial=1)
    out_causal_b = layer_causal(x_future_changed, timestamps=timestamps, n_spatial=1)
    assert torch.allclose(out_causal_a[:, :3], out_causal_b[:, :3], atol=1e-6)

    out_noncausal_a = layer_noncausal(x, timestamps=timestamps, n_spatial=1)
    out_noncausal_b = layer_noncausal(x_future_changed, timestamps=timestamps, n_spatial=1)
    assert not torch.allclose(
        out_noncausal_a[:, :3], out_noncausal_b[:, :3], atol=1e-6
    )


def test_temporal_attention_decay_bias_prefers_nearby_frames() -> None:
    layer = TemporalAttentionWithDecay(dim=3, num_heads=1, causal=False)
    layer.norm = torch.nn.Identity()

    with torch.no_grad():
        layer.q_proj.weight.zero_()
        layer.q_proj.bias.zero_()
        layer.k_proj.weight.zero_()
        layer.v_proj.weight.copy_(torch.eye(3))
        layer.gate.weight.zero_()  # sigmoid(0) = 0.5
        layer.out_proj.weight.copy_(torch.eye(3))
        layer.log_lambda.fill_(math.log(1.0))

    # One-hot values so the attention weights are readable from the output delta.
    x = torch.eye(3).unsqueeze(0)  # (BN=1, T=3, C=3)
    timestamps = torch.tensor([[0.0, 1.0, 3.0]])

    out = layer(x, timestamps=timestamps, n_spatial=1)
    delta = out - x
    probs_query_t1 = (delta[0, 1] * 2.0).tolist()  # compensate for gate=0.5

    # Distances from t=1 are [1, 0, 2] so attention should satisfy p1 > p0 > p2.
    assert probs_query_t1[1] > probs_query_t1[0] > probs_query_t1[2]
