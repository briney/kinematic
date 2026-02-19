"""Tests for batch collation behavior."""

from __future__ import annotations

import torch

from kinematic.data.collator import TrajectoryCollator


def _make_sample(*, n_atoms: int, n_tokens: int, has_bonds: bool) -> dict:
    sample = {
        "coords": torch.zeros(3, n_atoms, 3),
        "timestamps": torch.tensor([0.0, 0.1, 0.2]),
        "sigma": torch.tensor([1.0, 2.0, 3.0]),
        "conditioning_mask": torch.tensor([True, False, False]),
        "s_trunk": torch.zeros(n_tokens, 8),
        "z_trunk": torch.zeros(n_tokens, n_tokens, 4),
        "s_inputs": torch.zeros(n_tokens, 8),
        "atom_pad_mask": torch.ones(n_atoms, dtype=torch.bool),
        "observed_atom_mask": torch.ones(n_atoms, dtype=torch.bool),
        "token_pad_mask": torch.ones(n_tokens, dtype=torch.bool),
        "atom_to_token": torch.zeros(n_atoms, n_tokens),
        "mol_type_per_atom": torch.zeros(n_atoms, dtype=torch.long),
        "split": "train",
        "feats": {
            "ref_pos": torch.zeros(n_atoms, 3),
            "mol_types": torch.zeros(n_atoms, dtype=torch.long),
            "residue_indices": torch.arange(n_atoms, dtype=torch.long),
            "chain_ids": torch.zeros(n_atoms, dtype=torch.long),
        },
    }

    if has_bonds:
        sample["bond_indices"] = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        sample["bond_lengths"] = torch.tensor([1.5, 1.4], dtype=torch.float32)
    return sample


def test_collator_sets_bond_mask_for_real_vs_padded_bonds() -> None:
    collator = TrajectoryCollator()

    sample_with_bonds = _make_sample(n_atoms=4, n_tokens=2, has_bonds=True)
    sample_without_bonds = _make_sample(n_atoms=3, n_tokens=2, has_bonds=False)

    batch = collator([sample_with_bonds, sample_without_bonds])

    assert "bond_mask" in batch
    assert batch["bond_mask"].dtype == torch.bool
    assert batch["bond_mask"].shape == (2, 2)
    assert torch.equal(batch["bond_mask"][0], torch.tensor([True, True]))
    assert torch.equal(batch["bond_mask"][1], torch.tensor([False, False]))
