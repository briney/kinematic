"""Dataset behavior tests."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import numpy as np
import pytest
import torch

from boltzkinema.data.dataset import BoltzKinemaDataset, SystemInfo


def _make_system(tmp_path, *, include_observed_mask: bool) -> tuple[SystemInfo, str]:
    coords_path = tmp_path / "coords.npz"
    ref_path = tmp_path / "ref.npz"

    payload = {
        "coords": np.zeros((3, 4, 3), dtype=np.float32),
    }
    if include_observed_mask:
        payload["observed_atom_mask"] = np.array([True, False, True, True], dtype=np.bool_)
    np.savez(coords_path, **payload)

    np.savez(
        ref_path,
        mol_types=np.zeros(4, dtype=np.int64),
        residue_indices=np.arange(4, dtype=np.int64),
    )

    system = SystemInfo(
        system_id="sys",
        dataset="toy",
        n_frames=3,
        n_atoms=4,
        n_tokens=2,
        frame_dt_ns=0.1,
        split="train",
        coords_path=str(coords_path),
        trunk_cache_dir="unused",
        ref_path=str(ref_path),
    )
    return system, str(coords_path)


def test_observed_atom_mask_loaded_from_coords_npz(tmp_path) -> None:
    system, _ = _make_system(tmp_path, include_observed_mask=True)
    mask = system.observed_atom_mask
    expected = torch.tensor([True, False, True, True])
    assert torch.equal(mask, expected)


def test_observed_atom_mask_falls_back_to_all_ones_when_missing(tmp_path) -> None:
    system, _ = _make_system(tmp_path, include_observed_mask=False)
    mask = system.observed_atom_mask
    assert torch.equal(mask, torch.ones(4, dtype=torch.bool))


def test_coords_npz_metadata_is_cached(tmp_path, monkeypatch) -> None:
    system, coords_path = _make_system(tmp_path, include_observed_mask=True)

    original_load = np.load
    calls = 0

    def wrapped_load(path, *args, **kwargs):
        nonlocal calls
        if str(path) == coords_path:
            calls += 1
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", wrapped_load)

    _ = system.observed_atom_mask
    _ = system.observed_atom_mask
    _ = system.load_coords([0, 1])

    assert calls == 1


def _write_ref_npz(path: Path, n_atoms: int) -> None:
    np.savez(
        path,
        ref_coords=np.zeros((n_atoms, 3), dtype=np.float32),
        mol_types=np.zeros(n_atoms, dtype=np.int64),
        residue_indices=np.arange(n_atoms, dtype=np.int64),
        chain_ids=np.zeros(n_atoms, dtype=np.int64),
    )


def _write_coords_npz(path: Path, n_frames: int, n_atoms: int) -> None:
    coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    # Frame index is encoded in atom-pair distance and survives SE(3) augmentation.
    for frame_idx in range(n_frames):
        coords[frame_idx, 1, 0] = float(frame_idx)
    np.savez(path, coords=coords)


def _write_trunk_npz(path: Path, n_tokens: int) -> None:
    np.savez(
        path,
        s_inputs=np.zeros((n_tokens, 4), dtype=np.float32),
        s_trunk=np.ones((n_tokens, 4), dtype=np.float32),
        z_trunk=np.zeros((n_tokens, n_tokens, 2), dtype=np.float32),
    )


def _write_manifest(path: Path, entry: dict) -> None:
    with open(path, "w") as f:
        json.dump([entry], f)


def _infer_sampled_frame_indices(coords: torch.Tensor) -> list[int]:
    rel = coords[:, 1] - coords[:, 0]
    norms = torch.linalg.norm(rel, dim=-1)
    return [int(round(v.item())) for v in norms]


def test_dataset_uses_manifest_paths_before_override_dirs(tmp_path) -> None:
    system_id = "sys"
    manifest_coords_dir = tmp_path / "manifest_coords"
    override_coords_dir = tmp_path / "override_coords"
    manifest_trunk_dir = tmp_path / "manifest_trunk"
    override_trunk_dir = tmp_path / "override_trunk"
    ref_dir = tmp_path / "refs"
    for d in (
        manifest_coords_dir,
        override_coords_dir,
        manifest_trunk_dir,
        override_trunk_dir,
        ref_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    manifest_coords = manifest_coords_dir / f"{system_id}_coords.npz"
    override_coords = override_coords_dir / f"{system_id}_coords.npz"
    ref_path = ref_dir / f"{system_id}_ref.npz"
    _write_coords_npz(manifest_coords, n_frames=6, n_atoms=2)
    _write_coords_npz(override_coords, n_frames=6, n_atoms=2)
    _write_ref_npz(ref_path, n_atoms=2)

    _write_trunk_npz(manifest_trunk_dir / f"{system_id}_trunk.npz", n_tokens=2)
    _write_trunk_npz(override_trunk_dir / f"{system_id}_trunk.npz", n_tokens=2)

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        {
            "system_id": system_id,
            "dataset": "toy",
            "n_frames": 6,
            "n_atoms": 2,
            "n_tokens": 2,
            "frame_dt_ns": 1.0,
            "split": "train",
            "coords_path": str(manifest_coords),
            "trunk_cache_dir": str(manifest_trunk_dir),
            "ref_path": str(ref_path),
        },
    )

    dataset = BoltzKinemaDataset(
        manifest_path=manifest_path,
        trunk_cache_dir=override_trunk_dir,
        coords_dir=override_coords_dir,
        n_frames=4,
    )
    system = dataset.systems[0]

    assert Path(system.coords_path) == manifest_coords
    assert Path(system.trunk_cache_dir) == manifest_trunk_dir


def test_dataset_falls_back_to_override_dirs_when_manifest_paths_missing(tmp_path) -> None:
    system_id = "sys"
    coords_override_dir = tmp_path / "coords_override"
    trunk_override_dir = tmp_path / "trunk_override"
    ref_dir = tmp_path / "refs"
    for d in (coords_override_dir, trunk_override_dir, ref_dir):
        d.mkdir(parents=True, exist_ok=True)

    coords_path = coords_override_dir / "missing_coords.npz"
    ref_path = ref_dir / f"{system_id}_ref.npz"
    _write_coords_npz(coords_path, n_frames=6, n_atoms=2)
    _write_ref_npz(ref_path, n_atoms=2)
    _write_trunk_npz(trunk_override_dir / f"{system_id}_trunk.npz", n_tokens=2)

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        {
            "system_id": system_id,
            "dataset": "toy",
            "n_frames": 6,
            "n_atoms": 2,
            "n_tokens": 2,
            "frame_dt_ns": 1.0,
            "split": "train",
            "coords_path": "missing_coords.npz",
            "trunk_cache_dir": "",
            "ref_path": str(ref_path),
        },
    )

    dataset = BoltzKinemaDataset(
        manifest_path=manifest_path,
        trunk_cache_dir=trunk_override_dir,
        coords_dir=coords_override_dir,
        n_frames=4,
    )
    system = dataset.systems[0]

    assert Path(system.coords_path) == coords_path
    assert Path(system.trunk_cache_dir) == trunk_override_dir


def test_dataset_fails_fast_when_paths_cannot_be_resolved(tmp_path) -> None:
    system_id = "sys"
    ref_path = tmp_path / "ref.npz"
    _write_ref_npz(ref_path, n_atoms=2)

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        {
            "system_id": system_id,
            "dataset": "toy",
            "n_frames": 6,
            "n_atoms": 2,
            "n_tokens": 2,
            "frame_dt_ns": 1.0,
            "split": "train",
            "coords_path": "does_not_exist_coords.npz",
            "trunk_cache_dir": "",
            "ref_path": str(ref_path),
        },
    )

    with pytest.raises(FileNotFoundError, match=system_id):
        BoltzKinemaDataset(manifest_path=manifest_path, n_frames=4)


def test_dataset_getitem_is_idx_deterministic_and_order_independent(tmp_path) -> None:
    system_id = "sys"
    coords_dir = tmp_path / "coords"
    trunk_dir = tmp_path / "trunk"
    ref_dir = tmp_path / "refs"
    for d in (coords_dir, trunk_dir, ref_dir):
        d.mkdir(parents=True, exist_ok=True)

    coords_path = coords_dir / f"{system_id}_coords.npz"
    ref_path = ref_dir / f"{system_id}_ref.npz"
    _write_coords_npz(coords_path, n_frames=10, n_atoms=2)
    _write_ref_npz(ref_path, n_atoms=2)
    _write_trunk_npz(trunk_dir / f"{system_id}_trunk.npz", n_tokens=2)

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        {
            "system_id": system_id,
            "dataset": "toy",
            "n_frames": 10,
            "n_atoms": 2,
            "n_tokens": 2,
            "frame_dt_ns": 1.0,
            "split": "train",
            "coords_path": str(coords_path),
            "trunk_cache_dir": str(trunk_dir),
            "ref_path": str(ref_path),
        },
    )

    dataset = BoltzKinemaDataset(
        manifest_path=manifest_path,
        n_frames=4,
        dt_ranges={"toy": [2.0, 2.0]},
        seed=7,
    )

    sample_a = dataset[0]
    sample_b = dataset[0]

    assert sample_a["task"] == sample_b["task"]
    assert torch.equal(sample_a["coords"], sample_b["coords"])
    assert torch.equal(sample_a["sigma"], sample_b["sigma"])
    assert torch.equal(sample_a["conditioning_mask"], sample_b["conditioning_mask"])
    assert torch.equal(sample_a["timestamps"], sample_b["timestamps"])


def test_dataset_reproducible_across_dataloader_worker_counts(tmp_path) -> None:
    system_id = "sys"
    coords_dir = tmp_path / "coords"
    trunk_dir = tmp_path / "trunk"
    ref_dir = tmp_path / "refs"
    for d in (coords_dir, trunk_dir, ref_dir):
        d.mkdir(parents=True, exist_ok=True)

    coords_path = coords_dir / f"{system_id}_coords.npz"
    ref_path = ref_dir / f"{system_id}_ref.npz"
    _write_coords_npz(coords_path, n_frames=10, n_atoms=2)
    _write_ref_npz(ref_path, n_atoms=2)
    _write_trunk_npz(trunk_dir / f"{system_id}_trunk.npz", n_tokens=2)

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        {
            "system_id": system_id,
            "dataset": "toy",
            "n_frames": 10,
            "n_atoms": 2,
            "n_tokens": 2,
            "frame_dt_ns": 1.0,
            "split": "train",
            "coords_path": str(coords_path),
            "trunk_cache_dir": str(trunk_dir),
            "ref_path": str(ref_path),
        },
    )

    dataset = BoltzKinemaDataset(
        manifest_path=manifest_path,
        n_frames=4,
        dt_ranges={"toy": [2.0, 2.0]},
        seed=11,
    )

    try:
        _lock = multiprocessing.get_context().Lock()
    except PermissionError:
        pytest.skip("multiprocessing DataLoader is not permitted in this environment")
    else:
        del _lock

    loader_single = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=False, num_workers=0
    )
    loader_multi = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=False, num_workers=2
    )

    sample_single = next(iter(loader_single))
    sample_multi = next(iter(loader_multi))

    assert sample_single["task"] == sample_multi["task"]
    assert torch.equal(sample_single["coords"], sample_multi["coords"])
    assert torch.equal(sample_single["sigma"], sample_multi["sigma"])
    assert torch.equal(
        sample_single["conditioning_mask"], sample_multi["conditioning_mask"]
    )


def test_dataset_max_start_includes_last_valid_index(tmp_path) -> None:
    system_id = "sys"
    coords_dir = tmp_path / "coords"
    trunk_dir = tmp_path / "trunk"
    ref_dir = tmp_path / "refs"
    for d in (coords_dir, trunk_dir, ref_dir):
        d.mkdir(parents=True, exist_ok=True)

    coords_path = coords_dir / f"{system_id}_coords.npz"
    ref_path = ref_dir / f"{system_id}_ref.npz"
    _write_coords_npz(coords_path, n_frames=10, n_atoms=2)
    _write_ref_npz(ref_path, n_atoms=2)
    _write_trunk_npz(trunk_dir / f"{system_id}_trunk.npz", n_tokens=2)

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        {
            "system_id": system_id,
            "dataset": "toy",
            "n_frames": 10,
            "n_atoms": 2,
            "n_tokens": 2,
            "frame_dt_ns": 1.0,
            "split": "train",
            "coords_path": str(coords_path),
            "trunk_cache_dir": str(trunk_dir),
            "ref_path": str(ref_path),
        },
    )

    dataset = BoltzKinemaDataset(
        manifest_path=manifest_path,
        n_frames=4,
        dt_ranges={"toy": [2.0, 2.0]},  # fixed dt_frames=2
        seed=1,
    )

    seen_starts: set[int] = set()
    for epoch in range(1024):
        dataset.set_epoch(epoch)
        sample = dataset[0]
        indices = _infer_sampled_frame_indices(sample["coords"])
        seen_starts.add(indices[0])
        if 3 in seen_starts:
            break

    # For n_total=10, T=4, dt=2:
    # max_start = 10 - 1 - (4 - 1) * 2 = 3
    assert 3 in seen_starts
