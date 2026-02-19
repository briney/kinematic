"""Batch collation with padding for variable-length trajectories.

Pads variable-size systems to batch max for atoms and tokens.
T (number of frames) is fixed per config.
"""

from __future__ import annotations

from typing import Any

import torch


class TrajectoryCollator:
    """Pads variable-size systems to batch max and constructs batch tensors.

    All systems in a batch are padded to:
      - max N_tokens across the batch
      - max N_atoms across the batch
    T (number of frames) is fixed per config.

    Padding uses zeros for coordinates/features, False for masks.
    """

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        B = len(samples)
        T = samples[0]["coords"].shape[0]
        max_atoms = max(s["coords"].shape[1] for s in samples)
        max_tokens = max(s["s_trunk"].shape[0] for s in samples)

        # Pre-allocate padded tensors
        batch: dict[str, Any] = {
            "coords": torch.zeros(B, T, max_atoms, 3),
            "timestamps": torch.stack([s["timestamps"] for s in samples]),
            "sigma": torch.stack([s["sigma"] for s in samples]),
            "conditioning_mask": torch.stack(
                [s["conditioning_mask"] for s in samples]
            ),
            "s_trunk": torch.zeros(B, max_tokens, samples[0]["s_trunk"].shape[-1]),
            "z_trunk": torch.zeros(
                B,
                max_tokens,
                max_tokens,
                samples[0]["z_trunk"].shape[-1],
            ),
            "s_inputs": torch.zeros(
                B, max_tokens, samples[0]["s_inputs"].shape[-1]
            ),
            "atom_pad_mask": torch.zeros(B, max_atoms, dtype=torch.bool),
            "observed_atom_mask": torch.zeros(B, max_atoms, dtype=torch.bool),
            "token_pad_mask": torch.zeros(B, max_tokens, dtype=torch.bool),
            "atom_to_token": torch.zeros(B, max_atoms, max_tokens),
            "mol_type_per_atom": torch.zeros(B, max_atoms, dtype=torch.long),
            "split": [s["split"] for s in samples],
            "feats": _collate_feats(samples, max_atoms, max_tokens),
        }

        # Optional bond data
        has_bonds = any("bond_indices" in s for s in samples)
        if has_bonds:
            max_bonds = max(
                s["bond_indices"].shape[0] for s in samples if "bond_indices" in s
            )
            batch["bond_indices"] = torch.zeros(B, max_bonds, 2, dtype=torch.long)
            batch["bond_lengths"] = torch.zeros(B, max_bonds)
            batch["bond_mask"] = torch.zeros(B, max_bonds, dtype=torch.bool)

        # Fill in per-sample data
        for i, s in enumerate(samples):
            na = s["coords"].shape[1]
            nt = s["s_trunk"].shape[0]

            batch["coords"][i, :, :na] = s["coords"]
            batch["s_trunk"][i, :nt] = s["s_trunk"]
            batch["z_trunk"][i, :nt, :nt] = s["z_trunk"]
            batch["s_inputs"][i, :nt] = s["s_inputs"]
            batch["atom_pad_mask"][i, :na] = s["atom_pad_mask"]
            batch["observed_atom_mask"][i, :na] = s["observed_atom_mask"]
            batch["token_pad_mask"][i, :nt] = s["token_pad_mask"]
            batch["atom_to_token"][i, :na, :nt] = s["atom_to_token"]
            batch["mol_type_per_atom"][i, :na] = s["mol_type_per_atom"]

            if has_bonds and "bond_indices" in s:
                nb = s["bond_indices"].shape[0]
                batch["bond_indices"][i, :nb] = s["bond_indices"]
                batch["bond_lengths"][i, :nb] = s["bond_lengths"]
                batch["bond_mask"][i, :nb] = True

        return batch


def _collate_feats(
    samples: list[dict[str, Any]],
    max_atoms: int,
    max_tokens: int,
) -> dict[str, torch.Tensor]:
    """Pad featurizer fields to (max_atoms, max_tokens)-compatible shapes.

    Handles the reference structure features needed by diffusion conditioning:
      - ref_pos: (B, max_atoms, 3)
      - mol_types: (B, max_atoms)
      - residue_indices: (B, max_atoms)
      - chain_ids: (B, max_atoms)
    """
    B = len(samples)
    feats_keys = samples[0]["feats"].keys()
    collated: dict[str, torch.Tensor] = {}

    for key in feats_keys:
        example = samples[0]["feats"][key]
        shape = example.shape
        dtype = example.dtype

        if len(shape) == 1:
            # Per-atom or per-token 1D tensor
            dim0 = shape[0]
            # Determine if this is atom-indexed or token-indexed
            # by checking if size matches any sample's atom or token count
            pad_dim = max_atoms  # default to atom padding
            padded = torch.zeros(B, pad_dim, dtype=dtype)
            for i, s in enumerate(samples):
                v = s["feats"][key]
                n = v.shape[0]
                padded[i, :n] = v
            collated[key] = padded

        elif len(shape) == 2:
            # (n_atoms/n_tokens, D) tensor
            dim0, dim1 = shape
            pad_dim = max_atoms
            padded = torch.zeros(B, pad_dim, dim1, dtype=dtype)
            for i, s in enumerate(samples):
                v = s["feats"][key]
                n = v.shape[0]
                padded[i, :n] = v
            collated[key] = padded

        else:
            # Higher-dimensional: just stack with zero-padding
            pad_shape = [B] + [
                max(s["feats"][key].shape[d] for s in samples)
                for d in range(len(shape))
            ]
            padded = torch.zeros(pad_shape, dtype=dtype)
            for i, s in enumerate(samples):
                v = s["feats"][key]
                slices = [i] + [slice(0, v.shape[d]) for d in range(len(shape))]
                padded[tuple(slices)] = v
            collated[key] = padded

    return collated
