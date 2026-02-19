"""Tests for coordinate-unit inference helpers."""

from __future__ import annotations

import numpy as np

from kinematic.data.units import infer_coordinate_unit


def _make_linear_traj(*, spacing: float) -> np.ndarray:
    """Create a simple multi-frame chain trajectory with known length scale."""
    n_frames = 6
    n_atoms = 80
    base = np.zeros((n_atoms, 3), dtype=np.float32)
    base[:, 0] = np.arange(n_atoms, dtype=np.float32) * spacing

    rng = np.random.default_rng(0)
    frames = []
    for _ in range(n_frames):
        jitter = rng.normal(scale=0.02 * spacing, size=base.shape).astype(np.float32)
        frames.append(base + jitter)
    return np.stack(frames, axis=0)


def test_infer_coordinate_unit_nm() -> None:
    coords_nm = _make_linear_traj(spacing=0.15)
    unit, confidence, _ = infer_coordinate_unit(coords_nm)
    assert unit == "nm"
    assert confidence >= 0.65


def test_infer_coordinate_unit_angstrom() -> None:
    coords_angstrom = _make_linear_traj(spacing=1.5)
    unit, confidence, _ = infer_coordinate_unit(coords_angstrom)
    assert unit == "angstrom"
    assert confidence >= 0.65


def test_metadata_override_has_priority() -> None:
    coords_nm = _make_linear_traj(spacing=0.15)
    unit, confidence, reason = infer_coordinate_unit(
        coords_nm,
        metadata={"coordinates.unit": "angstrom"},
    )
    assert unit == "angstrom"
    assert confidence == 1.0
    assert reason.startswith("metadata:")
