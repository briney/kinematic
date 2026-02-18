"""BoltzKinemaDataset for trajectory training data.

Each sample provides a T-frame trajectory segment with precomputed
trunk embeddings and per-frame noise levels for the noise-as-masking
training paradigm.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from boltzkinema.data.noise_masking import assign_noise
from boltzkinema.data.trunk_cache import load_trunk_embeddings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-system metadata loaded from the manifest
# ---------------------------------------------------------------------------

@dataclass
class SystemInfo:
    """Metadata for a single molecular system."""

    system_id: str
    dataset: str
    n_frames: int
    n_atoms: int
    n_tokens: int
    frame_dt_ns: float
    split: str
    coords_path: str
    trunk_cache_dir: str
    ref_path: str
    # Optional fields
    bond_indices_path: str | None = None
    feats_path: str | None = None
    # Lazily loaded data (not serialized)
    _coords: np.ndarray | None = field(default=None, repr=False, compare=False)
    _ref_data: dict[str, np.ndarray] | None = field(default=None, repr=False, compare=False)

    def load_coords(self, frame_indices: list[int]) -> torch.Tensor:
        """Load coordinates for given frame indices.

        Returns (T, n_atoms, 3) float32 tensor in Angstrom.
        """
        if self._coords is None:
            data = np.load(self.coords_path)
            self._coords = data["coords"]  # (total_frames, n_atoms, 3), Angstrom
        selected = self._coords[frame_indices]  # (T, n_atoms, 3)
        return torch.from_numpy(selected.copy()).float()

    def load_trunk(self) -> dict[str, torch.Tensor]:
        """Load precomputed trunk embeddings."""
        return load_trunk_embeddings(self.system_id, self.trunk_cache_dir)

    def load_ref_data(self) -> dict[str, np.ndarray]:
        """Load reference structure metadata."""
        if self._ref_data is None:
            self._ref_data = dict(np.load(self.ref_path, allow_pickle=False))
        return self._ref_data

    @property
    def atom_pad_mask(self) -> torch.Tensor:
        """(n_atoms,) bool — all True for actual atoms."""
        return torch.ones(self.n_atoms, dtype=torch.bool)

    @property
    def observed_atom_mask(self) -> torch.Tensor:
        """(n_atoms,) bool — True for atoms resolved in reference."""
        ref = self.load_ref_data()
        if "observed_atom_mask" in ref:
            # May come from the coords npz
            data = np.load(self.coords_path)
            if "observed_atom_mask" in data:
                return torch.from_numpy(data["observed_atom_mask"]).bool()
        return torch.ones(self.n_atoms, dtype=torch.bool)

    @property
    def token_pad_mask(self) -> torch.Tensor:
        """(n_tokens,) bool — all True for actual tokens."""
        return torch.ones(self.n_tokens, dtype=torch.bool)

    @property
    def mol_type_per_atom(self) -> torch.Tensor:
        """(n_atoms,) int — molecule type code per atom."""
        ref = self.load_ref_data()
        return torch.from_numpy(ref["mol_types"].astype(np.int64))

    @property
    def atom_to_token(self) -> torch.Tensor:
        """(n_atoms, n_tokens) float — soft assignment matrix."""
        ref = self.load_ref_data()
        if "atom_to_token" in ref:
            return torch.from_numpy(ref["atom_to_token"].astype(np.float32))
        # Build from residue indices: each atom maps to its residue token
        res_idx = ref["residue_indices"]  # (n_atoms,)
        n_tokens = self.n_tokens
        mat = np.zeros((self.n_atoms, n_tokens), dtype=np.float32)
        for i, ri in enumerate(res_idx):
            if 0 <= ri < n_tokens:
                mat[i, ri] = 1.0
        return torch.from_numpy(mat)


# ---------------------------------------------------------------------------
# SE(3) augmentation
# ---------------------------------------------------------------------------

def _random_rotation_matrix() -> torch.Tensor:
    """Sample a uniform random 3x3 rotation matrix."""
    # QR decomposition of random Gaussian matrix
    q, r = torch.linalg.qr(torch.randn(3, 3))
    # Ensure proper rotation (det = +1)
    q = q @ torch.diag(torch.sign(torch.diag(r)))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str | Path) -> list[SystemInfo]:
    """Load system manifest from JSON.

    Expected format: list of dicts with keys matching SystemInfo fields.
    """
    manifest_path = Path(manifest_path)
    with open(manifest_path) as f:
        entries = json.load(f)

    systems = []
    for entry in entries:
        systems.append(SystemInfo(
            system_id=entry["system_id"],
            dataset=entry["dataset"],
            n_frames=entry["n_frames"],
            n_atoms=entry["n_atoms"],
            n_tokens=entry["n_tokens"],
            frame_dt_ns=entry["frame_dt_ns"],
            split=entry["split"],
            coords_path=entry["coords_path"],
            trunk_cache_dir=entry["trunk_cache_dir"],
            ref_path=entry["ref_path"],
            bond_indices_path=entry.get("bond_indices_path"),
            feats_path=entry.get("feats_path"),
        ))
    return systems


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BoltzKinemaDataset(Dataset):
    """Trajectory dataset for BoltzKinema training.

    Each sample provides:
      - coords: (T, N_atoms, 3) in Angstrom
      - timestamps: (T,) in nanoseconds
      - sigma: (T,) per-frame noise levels
      - conditioning_mask: (T,) bool
      - s_trunk, z_trunk, s_inputs: precomputed trunk embeddings
      - atom_pad_mask, observed_atom_mask, token_pad_mask
      - atom_to_token, mol_type_per_atom
      - feats: dict of reference structure features
      - split, task

    Parameters
    ----------
    manifest_path : path to JSON manifest.
    trunk_cache_dir : directory with precomputed trunk .npz files.
    coords_dir : directory with coordinate .npz files.
    n_frames : number of frames per training sample.
    dataset_weights : sampling weight per dataset name.
    dt_ranges : per-dataset (dt_min_ns, dt_max_ns) ranges.
    noise_P_mean : EDM log-normal mean.
    noise_P_std : EDM log-normal std.
    sigma_data : EDM sigma_data.
    forecast_prob : probability of forecasting vs interpolation task.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        trunk_cache_dir: str | Path,
        coords_dir: str | Path,
        n_frames: int = 32,
        dataset_weights: dict[str, float] | None = None,
        dt_ranges: dict[str, list[float]] | None = None,
        noise_P_mean: float = -1.2,
        noise_P_std: float = 1.5,
        sigma_data: float = 16.0,
        forecast_prob: float = 0.5,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.dataset_weights = dataset_weights or {}
        self.dt_ranges = dt_ranges or {}
        self.noise_P_mean = noise_P_mean
        self.noise_P_std = noise_P_std
        self.sigma_data = sigma_data
        self.forecast_prob = forecast_prob

        # Load manifest and filter to training split
        all_systems = load_manifest(manifest_path)
        self.systems = [s for s in all_systems if s.split == "train"]
        logger.info("Loaded %d training systems from manifest", len(self.systems))

        # Build per-dataset system lists and sampling weights
        self._dataset_systems: dict[str, list[SystemInfo]] = {}
        for sys in self.systems:
            self._dataset_systems.setdefault(sys.dataset, []).append(sys)

        # Build cumulative weight distribution over datasets
        self._dataset_names: list[str] = []
        self._dataset_cum_weights: list[float] = []
        cumulative = 0.0
        for name, sys_list in self._dataset_systems.items():
            w = self.dataset_weights.get(name, 1.0)
            cumulative += w
            self._dataset_names.append(name)
            self._dataset_cum_weights.append(cumulative)

    def __len__(self) -> int:
        return len(self.systems)

    def _sample_system(self) -> SystemInfo:
        """Sample a system weighted by dataset ratios."""
        if not self._dataset_cum_weights:
            return random.choice(self.systems)

        # Weighted dataset selection
        r = random.random() * self._dataset_cum_weights[-1]
        for i, cw in enumerate(self._dataset_cum_weights):
            if r <= cw:
                break
        dataset_name = self._dataset_names[i]
        return random.choice(self._dataset_systems[dataset_name])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # 1. Sample a system (weighted by dataset ratios)
        system = self._sample_system()

        # 2. Sample inter-frame interval dt (log-uniform)
        dt_range = self.dt_ranges.get(system.dataset, [0.1, 10.0])
        dt_min, dt_max = dt_range[0], dt_range[1]
        log_dt = random.uniform(math.log(dt_min), math.log(dt_max))
        dt_ns = math.exp(log_dt)

        # 3. Convert dt to frame indices
        dt_frames = max(1, round(dt_ns / system.frame_dt_ns))
        max_start = system.n_frames - self.n_frames * dt_frames
        if max_start <= 0:
            # Trajectory too short for desired stride; reduce stride
            dt_frames = max(1, system.n_frames // self.n_frames)
            max_start = system.n_frames - self.n_frames * dt_frames
        start = random.randint(0, max(0, max_start))
        indices = [start + i * dt_frames for i in range(self.n_frames)]
        # Clamp indices to valid range
        indices = [min(i, system.n_frames - 1) for i in indices]

        # 4. Load coordinates for selected frames (already in Angstrom)
        coords = system.load_coords(indices)  # (T, n_atoms, 3)

        # 5. Random SE(3) augmentation
        R = _random_rotation_matrix()
        t = torch.randn(3) * 10.0  # random translation in Angstrom
        coords = coords @ R.T + t.unsqueeze(0).unsqueeze(0)

        # 6. Compute timestamps
        actual_dt_ns = dt_frames * system.frame_dt_ns
        timestamps = torch.tensor(
            [i * actual_dt_ns for i in range(self.n_frames)],
            dtype=torch.float32,
        )

        # 7. Noise-as-masking assignment
        sigma, conditioning_mask, task = assign_noise(
            n_frames=self.n_frames,
            P_mean=self.noise_P_mean,
            P_std=self.noise_P_std,
            sigma_data=self.sigma_data,
            forecast_prob=self.forecast_prob,
        )

        # 8. Load precomputed trunk embeddings
        trunk = system.load_trunk()

        # 9. Build feats dict from reference structure metadata
        ref_data = system.load_ref_data()
        feats = {
            "ref_pos": torch.from_numpy(ref_data["ref_coords"].astype(np.float32)),
            "mol_types": torch.from_numpy(ref_data["mol_types"].astype(np.int64)),
            "residue_indices": torch.from_numpy(
                ref_data["residue_indices"].astype(np.int64)
            ),
            "chain_ids": torch.from_numpy(ref_data["chain_ids"].astype(np.int64)),
        }

        # 10. Optional bond data
        sample: dict[str, Any] = {
            "coords": coords,
            "timestamps": timestamps,
            "sigma": sigma,
            "conditioning_mask": conditioning_mask,
            "task": task,
            "s_trunk": trunk["s_trunk"],
            "z_trunk": trunk["z_trunk"],
            "s_inputs": trunk["s_inputs"],
            "atom_pad_mask": system.atom_pad_mask,
            "observed_atom_mask": system.observed_atom_mask,
            "token_pad_mask": system.token_pad_mask,
            "atom_to_token": system.atom_to_token,
            "mol_type_per_atom": system.mol_type_per_atom,
            "split": system.split,
            "feats": feats,
        }

        if system.bond_indices_path is not None:
            bond_data = np.load(system.bond_indices_path)
            sample["bond_indices"] = torch.from_numpy(
                bond_data["bond_indices"].astype(np.int64)
            )
            sample["bond_lengths"] = torch.from_numpy(
                bond_data["bond_lengths"].astype(np.float32)
            )

        return sample
