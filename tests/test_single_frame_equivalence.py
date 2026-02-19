"""Tests for single-frame equivalence with Boltz-2."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("boltz")

from boltz.model.modules.encodersv2 import AtomAttentionDecoder, AtomAttentionEncoder
from boltz.model.modules.transformersv2 import DiffusionTransformerLayer

from kinematic.model.spatial_temporal_atom import (
    SpatialTemporalAtomDecoder,
    SpatialTemporalAtomEncoder,
)
from kinematic.model.spatial_temporal_transformer import (
    SpatialTemporalTokenTransformerBlock,
)


def _make_to_keys(window_q: int, window_k: int):
    def to_keys(x: torch.Tensor) -> torch.Tensor:
        bsz, n_atoms, dim = x.shape
        n_windows = n_atoms // window_q
        pooled = x.view(bsz, n_windows, window_q, dim).mean(dim=2)
        return pooled.unsqueeze(2).repeat(1, 1, window_k, 1).reshape(
            bsz, n_windows * window_k, dim
        )

    return to_keys


def test_token_transformer_block_matches_boltz_layer_for_single_frame() -> None:
    torch.manual_seed(0)
    bsz, n_tokens, dim, heads = 2, 5, 16, 4

    boltz_layer = DiffusionTransformerLayer(
        heads=heads,
        dim=dim,
        dim_single_cond=dim,
        post_layer_norm=False,
    )
    kinema_block = SpatialTemporalTokenTransformerBlock(
        heads=heads,
        dim=dim,
        dim_single_cond=dim,
        post_layer_norm=False,
        temporal_heads=heads,
        causal=False,
    )

    kinema_block.adaln.load_state_dict(boltz_layer.adaln.state_dict())
    kinema_block.pair_bias_attn.load_state_dict(boltz_layer.pair_bias_attn.state_dict())
    kinema_block.output_projection_linear.load_state_dict(
        boltz_layer.output_projection_linear.state_dict()
    )
    kinema_block.transition.load_state_dict(boltz_layer.transition.state_dict())

    a = torch.randn(bsz, n_tokens, dim)
    s = torch.randn(bsz, n_tokens, dim)
    bias = torch.randn(bsz, n_tokens, n_tokens, heads)
    mask = torch.ones(bsz, n_tokens)
    timestamps = torch.zeros(bsz, 1)

    out_boltz = boltz_layer(a, s, bias=bias, mask=mask, to_keys=None, multiplicity=1)
    out_kinema = kinema_block(
        a,
        s=s,
        bias=bias,
        mask=mask,
        to_keys=None,
        multiplicity=1,
        timestamps=timestamps,
        T=1,
    )

    assert torch.allclose(out_kinema, out_boltz, atol=1e-6)


def test_atom_encoder_decoder_match_boltz_for_single_frame() -> None:
    torch.manual_seed(1)
    bsz, n_tokens, n_atoms = 2, 4, 8
    atom_s, token_s = 16, 8
    window_q, window_k = 4, 6
    depth, heads = 2, 2

    boltz_encoder = AtomAttentionEncoder(
        atom_s=atom_s,
        token_s=token_s,
        atoms_per_window_queries=window_q,
        atoms_per_window_keys=window_k,
        atom_encoder_depth=depth,
        atom_encoder_heads=heads,
        structure_prediction=True,
    )
    kinema_encoder = SpatialTemporalAtomEncoder(
        atom_s=atom_s,
        token_s=token_s,
        atoms_per_window_queries=window_q,
        atoms_per_window_keys=window_k,
        atom_encoder_depth=depth,
        atom_encoder_heads=heads,
        atom_temporal_heads=heads,
        structure_prediction=True,
        causal=False,
    )

    boltz_decoder = AtomAttentionDecoder(
        atom_s=atom_s,
        token_s=token_s,
        attn_window_queries=window_q,
        attn_window_keys=window_k,
        atom_decoder_depth=depth,
        atom_decoder_heads=heads,
    )
    kinema_decoder = SpatialTemporalAtomDecoder(
        atom_s=atom_s,
        token_s=token_s,
        atoms_per_window_queries=window_q,
        atoms_per_window_keys=window_k,
        atom_decoder_depth=depth,
        atom_decoder_heads=heads,
        atom_temporal_heads=heads,
        causal=False,
    )

    kinema_encoder.atom_encoder.load_state_dict(boltz_encoder.atom_encoder.state_dict())
    kinema_encoder.atom_to_token_trans.load_state_dict(
        boltz_encoder.atom_to_token_trans.state_dict()
    )
    kinema_encoder.r_to_q_trans.load_state_dict(boltz_encoder.r_to_q_trans.state_dict())

    kinema_decoder.atom_decoder.load_state_dict(boltz_decoder.atom_decoder.state_dict())
    kinema_decoder.a_to_q_trans.load_state_dict(boltz_decoder.a_to_q_trans.state_dict())
    kinema_decoder.atom_feat_to_atom_pos_update.load_state_dict(
        boltz_decoder.atom_feat_to_atom_pos_update.state_dict()
    )

    feats = {
        "ref_pos": torch.randn(bsz, n_tokens, 3),
        "atom_pad_mask": torch.ones(bsz, n_atoms, dtype=torch.bool),
        "atom_to_token": torch.rand(bsz, n_atoms, n_tokens),
    }
    q = torch.randn(bsz, n_atoms, atom_s)
    c = torch.randn(bsz, n_atoms, atom_s)
    r = torch.randn(bsz, n_atoms, 3)  # B*T with T=1
    timestamps = torch.zeros(bsz, 1)
    n_windows = n_atoms // window_q
    atom_enc_bias = torch.randn(bsz, n_windows, window_q, window_k, depth * heads)
    atom_dec_bias = torch.randn(bsz, n_windows, window_q, window_k, depth * heads)
    to_keys = _make_to_keys(window_q=window_q, window_k=window_k)

    a_boltz, q_boltz, c_boltz, _ = boltz_encoder(
        feats=feats,
        q=q,
        c=c,
        atom_enc_bias=atom_enc_bias,
        to_keys=to_keys,
        r=r,
        multiplicity=1,
    )
    a_kinema, q_kinema, c_kinema, _ = kinema_encoder(
        feats=feats,
        q=q,
        c=c,
        atom_enc_bias=atom_enc_bias,
        to_keys=to_keys,
        r=r,
        timestamps=timestamps,
        T=1,
    )

    assert torch.allclose(a_kinema, a_boltz, atol=1e-6)
    assert torch.allclose(q_kinema, q_boltz, atol=1e-6)
    assert torch.allclose(c_kinema, c_boltz, atol=1e-6)

    r_boltz = boltz_decoder(
        a=a_boltz,
        q=q_boltz,
        c=c_boltz,
        atom_dec_bias=atom_dec_bias,
        feats=feats,
        to_keys=to_keys,
        multiplicity=1,
    )
    r_kinema = kinema_decoder(
        a=a_kinema,
        q=q_kinema,
        c=c_kinema,
        atom_dec_bias=atom_dec_bias,
        feats=feats,
        to_keys=to_keys,
        timestamps=timestamps,
        T=1,
    )

    assert torch.allclose(r_kinema, r_boltz, atol=1e-6)
