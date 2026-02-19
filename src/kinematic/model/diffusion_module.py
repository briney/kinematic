"""SpatialTemporalDiffusionModule."""

from __future__ import annotations

import torch
import torch.nn as nn

import boltz.model.layers.initialize as init
from boltz.model.modules.encodersv2 import SingleConditioning
from boltz.model.modules.utils import LinearNoBias

from kinematic.model.spatial_temporal_atom import (
    SpatialTemporalAtomDecoder,
    SpatialTemporalAtomEncoder,
)
from kinematic.model.spatial_temporal_transformer import (
    SpatialTemporalTokenTransformer,
)


class SpatialTemporalDiffusionModule(nn.Module):
    """Orchestrates the full score model for Kinematic.

    Replaces Boltz-2's ``DiffusionModule``. Sub-module names match
    ``DiffusionModule`` for weight loading.

    Pipeline:
      1. SingleConditioning(times, s_trunk, s_inputs) → s
      2. SpatialTemporalAtomEncoder → a (atom→token aggregation)
      3. a += s_to_a_linear(s)
      4. SpatialTemporalTokenTransformer → a (24-layer token transformer)
      5. a_norm → a
      6. SpatialTemporalAtomDecoder → r_update (token→atom broadcast)
    """

    def __init__(
        self,
        token_s: int,
        atom_s: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: float = 16.0,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        atom_temporal_heads: int | None = None,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        token_temporal_heads: int | None = None,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        causal: bool = False,
    ) -> None:
        super().__init__()

        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_attention_encoder = SpatialTemporalAtomEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            atom_temporal_heads=atom_temporal_heads,
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
            causal=causal,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * token_s), LinearNoBias(2 * token_s, 2 * token_s),
        )
        init.final_init_(self.s_to_a_linear[1].weight)

        self.token_transformer = SpatialTemporalTokenTransformer(
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            temporal_heads=token_temporal_heads,
            activation_checkpointing=activation_checkpointing,
            causal=causal,
        )

        self.a_norm = nn.LayerNorm(2 * token_s)

        self.atom_attention_decoder = SpatialTemporalAtomDecoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            atom_temporal_heads=atom_temporal_heads,
            activation_checkpointing=activation_checkpointing,
            causal=causal,
        )

    def forward(
        self,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        r_noisy: torch.Tensor,
        times: torch.Tensor,
        feats: dict,
        diffusion_conditioning: dict,
        timestamps: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        s_inputs : (B, N, token_s)
            Input single representation.
        s_trunk : (B, N, token_s)
            Trunk single representation.
        r_noisy : (B*T, M, 3)
            Scaled noisy coordinates.
        times : (B*T,)
            c_noise conditioning values per frame.
        feats : dict
            Feature dict for atom operations.
        diffusion_conditioning : dict
            Output from DiffusionConditioning: q, c, to_keys,
            atom_enc_bias, atom_dec_bias, token_trans_bias.
        timestamps : (B, T)
            Frame timestamps in nanoseconds.
        T : int
            Number of temporal frames.

        Returns
        -------
        r_update : (B*T, M, 3) — coordinate updates.
        """
        # 1. Single conditioning
        s, _normed_fourier = self.single_conditioner(
            times,
            s_trunk.repeat_interleave(T, 0),
            s_inputs.repeat_interleave(T, 0),
        )

        # 2. Atom attention encoder
        a, q_skip, c_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            q=diffusion_conditioning["q"].float(),
            c=diffusion_conditioning["c"].float(),
            atom_enc_bias=diffusion_conditioning["atom_enc_bias"].float(),
            to_keys=diffusion_conditioning["to_keys"],
            r=r_noisy,
            timestamps=timestamps,
            T=T,
        )

        # 3. Inject single conditioning into token activations
        a = a + self.s_to_a_linear(s)

        # 4. Token transformer
        mask = feats["token_pad_mask"].repeat_interleave(T, 0)
        a = self.token_transformer(
            a,
            s=s,
            bias=diffusion_conditioning["token_trans_bias"].float(),
            mask=mask.float(),
            multiplicity=T,
            timestamps=timestamps,
            T=T,
        )

        # 5. Layer norm
        a = self.a_norm(a)

        # 6. Atom attention decoder
        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            atom_dec_bias=diffusion_conditioning["atom_dec_bias"].float(),
            feats=feats,
            to_keys=to_keys,
            timestamps=timestamps,
            T=T,
        )

        return r_update
