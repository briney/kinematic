"""Top-level Kinematic model (nn.Module)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from boltz.model.modules.diffusion_conditioning import DiffusionConditioning
from boltz.model.modules.encodersv2 import RelativePositionEncoder

from kinematic.model.diffusion_module import SpatialTemporalDiffusionModule
from kinematic.model.edm import PerFrameEDM


class Kinematic(nn.Module):
    """Top-level Kinematic model.

    Wraps DiffusionConditioning (from Boltz-2) + the temporal score model
    + per-frame EDM preconditioning.

    Forward pipeline:
      1. rel_pos_encoder → relative position encoding (batch level)
      2. DiffusionConditioning → q, c, biases (batch level)
      3. Add per-frame noise (sigma=0 conditioning frames untouched)
      4. EDM scale_input → r_noisy
      5. Score model → r_update
      6. EDM combine_output → x_denoised
    """

    def __init__(
        self,
        token_s: int = 384,
        token_z: int = 128,
        atom_s: int = 128,
        atom_z: int = 128,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        atom_temporal_heads: int | None = None,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        token_temporal_heads: int | None = None,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        sigma_data: float = 16.0,
        dim_fourier: int = 256,
        atom_feature_dim: int = 128,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        causal: bool = False,
        use_no_atom_char: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
        fix_sym_check: bool = False,
        cyclic_pos_enc: bool = False,
    ) -> None:
        super().__init__()
        self.token_s = token_s
        self.sigma_data = sigma_data

        # --- Boltz-2 modules (shared across frames) ---
        self.diffusion_conditioning = DiffusionConditioning(
            token_s=token_s,
            token_z=token_z,
            atom_s=atom_s,
            atom_z=atom_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            token_transformer_depth=token_transformer_depth,
            token_transformer_heads=token_transformer_heads,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            atom_feature_dim=atom_feature_dim,
            conditioning_transition_layers=conditioning_transition_layers,
            use_no_atom_char=use_no_atom_char,
            use_atom_backbone_feat=use_atom_backbone_feat,
            use_residue_feats_atoms=use_residue_feats_atoms,
        )

        self.rel_pos_encoder = RelativePositionEncoder(
            token_z=token_z,
            fix_sym_check=fix_sym_check,
            cyclic_pos_enc=cyclic_pos_enc,
        )

        # --- Kinematic-specific modules ---
        self.edm = PerFrameEDM(sigma_data=sigma_data)

        self.score_model = SpatialTemporalDiffusionModule(
            token_s=token_s,
            atom_s=atom_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            sigma_data=sigma_data,
            dim_fourier=dim_fourier,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            atom_temporal_heads=atom_temporal_heads,
            token_transformer_depth=token_transformer_depth,
            token_transformer_heads=token_transformer_heads,
            token_temporal_heads=token_temporal_heads,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            conditioning_transition_layers=conditioning_transition_layers,
            activation_checkpointing=activation_checkpointing,
            causal=causal,
        )

    def _build_feats(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Merge batch-level keys into the feats dict expected by Boltz-2 modules.

        DiffusionConditioning and the score model require a feats dict with
        specific keys. This helper assembles them from the batch dict,
        combining reference structure features with batch-level tensors.
        """
        feats = dict(batch["feats"])  # shallow copy

        # Batch-level keys needed by encoder/decoder
        feats["atom_pad_mask"] = batch["atom_pad_mask"]
        feats["token_pad_mask"] = batch["token_pad_mask"]
        feats["atom_to_token"] = batch["atom_to_token"]

        return feats

    def forward(
        self, batch: dict[str, Any], add_noise: bool = True,
    ) -> dict[str, Any]:
        """Forward pass.

        Parameters
        ----------
        batch : dict
            Expected keys:
              - coords : (B, T, M, 3)
              - timestamps : (B, T)
              - sigma : (B, T) — per-frame noise levels (0 for conditioning)
              - conditioning_mask : (B, T) — bool
              - s_trunk : (B, N, token_s)
              - z_trunk : (B, N, N, token_z)
              - s_inputs : (B, N, token_s)
              - atom_pad_mask : (B, M)
              - token_pad_mask : (B, N)
              - atom_to_token : (B, M, N)
              - feats : dict of reference structure features
        add_noise : bool
            If True (default, training), sample noise and add to coords.
            If False (inference), treat coords as already at the noise
            level specified by sigma.

        Returns
        -------
        dict with:
          - x_denoised : (B, T, M, 3)
          - sigma : (B, T)
          - conditioning_mask : (B, T)
        """
        coords = batch["coords"]  # (B, T, M, 3)
        sigma = batch["sigma"]  # (B, T)
        B, T, M, _ = coords.shape

        feats = self._build_feats(batch)

        # 1. Relative position encoding (batch level, shared across frames)
        rel_pos_enc = self.rel_pos_encoder(feats)  # (B, N, N, token_z)

        # 2. Diffusion conditioning (batch level, shared across frames)
        q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
            self.diffusion_conditioning(
                batch["s_trunk"], batch["z_trunk"], rel_pos_enc, feats,
            )
        )

        # 3. Noise handling
        if add_noise:
            # Training: add per-frame noise (sigma=0 conditioning frames untouched)
            eps = torch.randn_like(coords)
            x_noisy = coords + sigma[..., None, None] * eps
        else:
            # Inference: coords are already at the correct noise level
            x_noisy = coords

        # 4. EDM scale input
        r_noisy = self.edm.scale_input(x_noisy, sigma)  # (B, T, M, 3)
        r_noisy = r_noisy.view(B * T, M, 3)

        # 5. Compute noise conditioning values
        times = self.edm.c_noise(sigma).view(B * T)  # (B*T,)

        # 6. Run score model
        diffusion_conditioning = {
            "q": q,
            "c": c,
            "to_keys": to_keys,
            "atom_enc_bias": atom_enc_bias,
            "atom_dec_bias": atom_dec_bias,
            "token_trans_bias": token_trans_bias,
        }

        r_update = self.score_model(
            s_inputs=batch["s_inputs"],
            s_trunk=batch["s_trunk"],
            r_noisy=r_noisy,
            times=times,
            feats=feats,
            diffusion_conditioning=diffusion_conditioning,
            timestamps=batch["timestamps"],
            T=T,
        )

        # 7. Reshape and apply EDM output combination
        r_update = r_update.view(B, T, M, 3)
        x_denoised = self.edm.combine_output(x_noisy, r_update, sigma)

        return {
            "x_denoised": x_denoised,
            "sigma": sigma,
            "conditioning_mask": batch["conditioning_mask"],
        }
