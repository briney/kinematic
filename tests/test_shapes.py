"""Tests for tensor shape consistency throughout the model."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("boltz")

from kinematic.model.spatial_temporal_atom import (
    SpatialTemporalAtomDecoder,
    SpatialTemporalAtomEncoder,
    _num_windows,
    _pad_or_trim_dim,
    _to_windowed_keys,
)


def test_num_windows_uses_ceil_division() -> None:
    assert _num_windows(32, 32) == 1
    assert _num_windows(33, 32) == 2
    assert _num_windows(65, 32) == 3


def test_pad_or_trim_dim_pads_and_trims() -> None:
    x = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
    padded = _pad_or_trim_dim(x, target=5, dim=1, pad_value=0.0)
    assert padded.shape == (1, 5, 2)
    assert torch.allclose(padded[:, :3], x)
    assert torch.allclose(padded[:, 3:], torch.zeros(1, 2, 2))

    trimmed = _pad_or_trim_dim(x, target=2, dim=1, pad_value=0.0)
    assert trimmed.shape == (1, 2, 2)
    assert torch.allclose(trimmed, x[:, :2])


def test_to_windowed_keys_aligns_window_count() -> None:
    # keys shape: (B*K, H, D) with B=1, K=2, H=4, D=3
    keys = torch.randn(2, 4, 3)
    out = _to_windowed_keys(keys, batch_size=1, n_heads=4, n_windows=3)
    assert out.shape == (3, 4, 3)
    assert torch.allclose(out[:2], keys)
    assert torch.allclose(out[2], torch.zeros_like(out[2]))


def test_atom_encoder_decoder_non_divisible_atom_count_smoke() -> None:
    torch.manual_seed(0)
    batch_size, n_frames, n_atoms, n_tokens = 1, 2, 5, 3
    atom_s, token_s = 16, 8
    window_q, window_k = 4, 6
    depth, heads = 1, 2

    encoder = SpatialTemporalAtomEncoder(
        atom_s=atom_s,
        token_s=token_s,
        atoms_per_window_queries=window_q,
        atoms_per_window_keys=window_k,
        atom_encoder_depth=depth,
        atom_encoder_heads=heads,
        atom_temporal_heads=heads,
    )
    decoder = SpatialTemporalAtomDecoder(
        atom_s=atom_s,
        token_s=token_s,
        atoms_per_window_queries=window_q,
        atoms_per_window_keys=window_k,
        atom_decoder_depth=depth,
        atom_decoder_heads=heads,
        atom_temporal_heads=heads,
    )

    feats = {
        "atom_pad_mask": torch.ones(batch_size, n_atoms, dtype=torch.bool),
        "atom_to_token": torch.randn(batch_size, n_atoms, n_tokens),
    }
    q = torch.randn(batch_size, n_atoms, atom_s)
    c = torch.randn(batch_size, n_atoms, atom_s)
    r = torch.randn(batch_size * n_frames, n_atoms, 3)
    timestamps = torch.tensor([[0.0, 1.0]])

    # Use fewer K windows than ceil(n_atoms / window_q) to exercise runtime padding.
    atom_enc_bias = torch.randn(batch_size, 1, window_q, window_k, depth * heads)
    atom_dec_bias = torch.randn(batch_size, 1, window_q, window_k, depth * heads)

    def to_keys(x: torch.Tensor) -> torch.Tensor:
        # x is (B, M, D). Return one window worth of keys to trigger key padding.
        return x[:, :window_k, :].mean(dim=1, keepdim=True).repeat(1, window_k, 1)

    a, q_skip, c_skip, to_keys_out = encoder(
        feats=feats,
        q=q,
        c=c,
        atom_enc_bias=atom_enc_bias,
        to_keys=to_keys,
        r=r,
        timestamps=timestamps,
        T=n_frames,
    )
    assert a.shape == (batch_size * n_frames, n_tokens, 2 * token_s)
    assert q_skip.shape == (batch_size * n_frames, n_atoms, atom_s)
    assert c_skip.shape == (batch_size * n_frames, n_atoms, atom_s)

    r_update = decoder(
        a=a,
        q=q_skip,
        c=c_skip,
        atom_dec_bias=atom_dec_bias,
        feats=feats,
        to_keys=to_keys_out,
        timestamps=timestamps,
        T=n_frames,
    )
    assert r_update.shape == (batch_size * n_frames, n_atoms, 3)
