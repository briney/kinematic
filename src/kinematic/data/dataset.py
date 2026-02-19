"""TrajectoryDataset for trajectory training data.

Each sample provides a T-frame trajectory segment with precomputed
trunk embeddings and per-frame noise levels for the noise-as-masking
training paradigm.
"""

from __future__ import annotations

import bisect
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

from kinematic.data.noise_masking import assign_noise
from kinematic.data.trunk_cache import load_trunk_embeddings

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
    _coords_meta: dict[str, np.ndarray] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _ref_data: dict[str, np.ndarray] | None = field(default=None, repr=False, compare=False)

    def _load_coords_data(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Load and cache coordinates NPZ payload."""
        if self._coords is None or self._coords_meta is None:
            with np.load(self.coords_path, allow_pickle=False) as data:
                self._coords = data["coords"]  # (total_frames, n_atoms, 3), Angstrom
                self._coords_meta = {
                    key: data[key] for key in data.files if key != "coords"
                }
        return self._coords, self._coords_meta

    def load_coords(self, frame_indices: list[int]) -> torch.Tensor:
        """Load coordinates for given frame indices.

        Returns (T, n_atoms, 3) float32 tensor in Angstrom.
        """
        self._coords, _ = self._load_coords_data()
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
        _, coords_meta = self._load_coords_data()
        if "observed_atom_mask" in coords_meta:
            return torch.from_numpy(coords_meta["observed_atom_mask"]).bool()
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

def _random_rotation_matrix(
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a uniform random 3x3 rotation matrix."""
    # QR decomposition of random Gaussian matrix
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    # Ensure proper rotation (det = +1)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
    q = q @ torch.diag(diag_sign)
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

class TrajectoryDataset(Dataset):
    """Trajectory dataset for Kinematic training.

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
    trunk_cache_dir : optional override directory for trunk cache files.
        Resolution precedence per system:
          1. manifest ``trunk_cache_dir`` (if it contains ``{system_id}_trunk.npz``)
          2. constructor override ``trunk_cache_dir``
    coords_dir : optional override directory for coordinate files.
        Resolution precedence per system:
          1. manifest ``coords_path``
          2. constructor override ``coords_dir`` + coords filename
    n_frames : number of frames per training sample.
    dataset_weights : sampling weight per dataset name.
    dt_ranges : per-dataset (dt_min_ns, dt_max_ns) ranges.
    noise_P_mean : EDM log-normal mean.
    noise_P_std : EDM log-normal std.
    sigma_data : EDM sigma_data.
    forecast_prob : probability of forecasting vs interpolation task.
    seed : base seed for deterministic idx-driven sampling.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        trunk_cache_dir: str | Path | None = None,
        coords_dir: str | Path | None = None,
        n_frames: int = 32,
        dataset_weights: dict[str, float] | None = None,
        dt_ranges: dict[str, list[float]] | None = None,
        noise_P_mean: float = -1.2,
        noise_P_std: float = 1.5,
        sigma_data: float = 16.0,
        forecast_prob: float = 0.5,
        seed: int = 0,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.dataset_weights = dataset_weights or {}
        self.dt_ranges = dt_ranges or {}
        self.noise_P_mean = noise_P_mean
        self.noise_P_std = noise_P_std
        self.sigma_data = sigma_data
        self.forecast_prob = forecast_prob
        self.seed = int(seed)
        self._epoch = 0

        self._manifest_path = Path(manifest_path).expanduser()
        self._manifest_dir = self._manifest_path.parent
        self._trunk_override_dir = self._normalize_override_dir(trunk_cache_dir)
        self._coords_override_dir = self._normalize_override_dir(coords_dir)

        # Load manifest and filter to training split
        all_systems = load_manifest(self._manifest_path)
        self.systems = [
            self._resolve_system_paths(s)
            for s in all_systems
            if s.split == "train"
        ]
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

    @staticmethod
    def _normalize_override_dir(path: str | Path | None) -> Path | None:
        if path is None:
            return None
        text = str(path).strip()
        if text == "":
            return None
        return Path(text).expanduser()

    def _resolve_manifest_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self._manifest_dir / path
        return path

    def _resolve_coords_path(self, system: SystemInfo) -> Path:
        attempts: list[Path] = []

        if system.coords_path:
            manifest_coords = self._resolve_manifest_path(system.coords_path)
            attempts.append(manifest_coords)
            if manifest_coords.exists():
                return manifest_coords

        if self._coords_override_dir is not None:
            basename = Path(system.coords_path).name if system.coords_path else ""
            if basename:
                by_basename = self._coords_override_dir / basename
                attempts.append(by_basename)
                if by_basename.exists():
                    return by_basename

            by_system_id = self._coords_override_dir / f"{system.system_id}_coords.npz"
            if not attempts or by_system_id != attempts[-1]:
                attempts.append(by_system_id)
            if by_system_id.exists():
                return by_system_id

        attempted = ", ".join(str(p) for p in attempts) if attempts else "<none>"
        raise FileNotFoundError(
            f"Could not resolve coords_path for system '{system.system_id}'. "
            f"Tried: {attempted}"
        )

    def _resolve_trunk_cache_dir(self, system: SystemInfo) -> Path:
        trunk_filename = f"{system.system_id}_trunk.npz"
        attempted_files: list[Path] = []

        candidates: list[Path] = []
        if system.trunk_cache_dir:
            candidates.append(self._resolve_manifest_path(system.trunk_cache_dir))
        if self._trunk_override_dir is not None:
            candidates.append(self._trunk_override_dir)

        seen: set[Path] = set()
        for candidate in candidates:
            candidate = candidate.resolve() if candidate.exists() else candidate
            if candidate in seen:
                continue
            seen.add(candidate)
            trunk_file = candidate / trunk_filename
            attempted_files.append(trunk_file)
            if trunk_file.exists():
                return candidate

        attempted = ", ".join(str(p) for p in attempted_files) if attempted_files else "<none>"
        raise FileNotFoundError(
            f"Could not resolve trunk cache for system '{system.system_id}'. "
            f"Expected file '{trunk_filename}'. Tried: {attempted}"
        )

    def _resolve_system_paths(self, system: SystemInfo) -> SystemInfo:
        system.coords_path = str(self._resolve_coords_path(system))
        system.trunk_cache_dir = str(self._resolve_trunk_cache_dir(system))

        ref_path = self._resolve_manifest_path(system.ref_path)
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Reference path does not exist for system '{system.system_id}': {ref_path}"
            )
        system.ref_path = str(ref_path)

        if system.bond_indices_path is not None:
            bond_path = self._resolve_manifest_path(system.bond_indices_path)
            if not bond_path.exists():
                raise FileNotFoundError(
                    f"Bond path does not exist for system '{system.system_id}': {bond_path}"
                )
            system.bond_indices_path = str(bond_path)

        return system

    def _seed_for_index(self, idx: int) -> int:
        # Mix base seed, epoch, and index so __getitem__ is deterministic
        # for a given (epoch, idx), independent of worker call order.
        mixed = (
            (self.seed + 1) * 0x9E3779B185EBCA87
            + (self._epoch + 1) * 0xC2B2AE3D27D4EB4F
            + (idx + 1) * 0x165667B19E3779F9
        )
        return mixed & ((1 << 63) - 1)

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic-but-varying sampling."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.systems)

    def _sample_system(self, idx: int, py_rng: random.Random) -> SystemInfo:
        """Select a system using deterministic idx-driven RNG."""
        if not self.systems:
            raise IndexError("Dataset is empty.")

        if not self._dataset_cum_weights:
            return self.systems[idx % len(self.systems)]

        # Weighted dataset selection
        r = py_rng.random() * self._dataset_cum_weights[-1]
        i = bisect.bisect_left(self._dataset_cum_weights, r)
        if i >= len(self._dataset_names):
            i = len(self._dataset_names) - 1
        dataset_name = self._dataset_names[i]
        systems = self._dataset_systems[dataset_name]
        return systems[py_rng.randrange(len(systems))]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += len(self.systems)
        if idx < 0 or idx >= len(self.systems):
            raise IndexError(f"Index out of range: {idx}")

        sample_seed = self._seed_for_index(idx)
        py_rng = random.Random(sample_seed)
        torch_generator = torch.Generator()
        torch_generator.manual_seed(sample_seed)

        # 1. Sample a system (weighted by dataset ratios)
        system = self._sample_system(idx, py_rng)

        # 2. Sample inter-frame interval dt (log-uniform)
        dt_range = self.dt_ranges.get(system.dataset, [0.1, 10.0])
        dt_min, dt_max = dt_range[0], dt_range[1]
        log_dt = py_rng.uniform(math.log(dt_min), math.log(dt_max))
        dt_ns = math.exp(log_dt)

        # 3. Convert dt to frame indices
        dt_frames = max(1, round(dt_ns / system.frame_dt_ns))
        max_start = system.n_frames - 1 - (self.n_frames - 1) * dt_frames
        if max_start < 0:
            # Trajectory too short for desired stride; reduce stride
            dt_frames = max(1, (system.n_frames - 1) // max(1, self.n_frames - 1))
            max_start = system.n_frames - 1 - (self.n_frames - 1) * dt_frames
        start = py_rng.randint(0, max(0, max_start))
        indices = [start + i * dt_frames for i in range(self.n_frames)]
        # Clamp indices to valid range
        indices = [min(i, system.n_frames - 1) for i in indices]

        # 4. Load coordinates for selected frames (already in Angstrom)
        coords = system.load_coords(indices)  # (T, n_atoms, 3)

        # 5. Random SE(3) augmentation
        R = _random_rotation_matrix(generator=torch_generator)
        t = torch.randn(3, generator=torch_generator) * 10.0
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
            py_rng=py_rng,
            torch_generator=torch_generator,
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
