"""Shared helpers for dataset preprocessing scripts."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mdtraj

logger = logging.getLogger(__name__)


def _load_preprocessing_ops() -> dict[str, Any]:
    """Import preprocessing ops lazily to keep module import lightweight."""
    from kinematic.data.preprocessing import (
        build_observation_mask,
        convert_trajectory,
        extract_atom_metadata,
        save_reference_structure,
    )

    return {
        "build_observation_mask": build_observation_mask,
        "convert_trajectory": convert_trajectory,
        "extract_atom_metadata": extract_atom_metadata,
        "save_reference_structure": save_reference_structure,
    }


def finalize_processed_system(
    *,
    system_id: str,
    dataset: str,
    traj: "mdtraj.Trajectory",
    output_dir: str | Path,
    ref_dir: str | Path,
    frame_dt_ns: float | None = None,
    split: str = "train",
) -> dict[str, Any]:
    """Convert aligned trajectory outputs into a manifest entry."""
    output_dir = Path(output_dir)
    ref_dir = Path(ref_dir)
    ops = _load_preprocessing_ops()

    coords_A = traj.xyz[0] * 10.0
    observed_mask = ops["build_observation_mask"](coords_A)
    coords_path = ops["convert_trajectory"](traj, system_id, output_dir, observed_mask)

    atoms = ops["extract_atom_metadata"](traj)
    ref_path = ops["save_reference_structure"](
        atoms,
        coords_A,
        ref_dir / f"{system_id}_ref.npz",
    )

    if frame_dt_ns is None:
        frame_dt_ns = float(traj.timestep) / 1000.0

    return {
        "system_id": system_id,
        "dataset": dataset,
        "n_frames": traj.n_frames,
        "n_atoms": traj.n_atoms,
        "n_tokens": len(set(a["residue_index"] for a in atoms)),
        "frame_dt_ns": float(frame_dt_ns),
        "split": split,
        "coords_path": str(coords_path),
        "trunk_cache_dir": "",
        "ref_path": str(ref_path),
    }


def collect_manifest_entries(
    systems: list[dict[str, Any]],
    preprocess_fn: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> list[dict[str, Any]]:
    """Run preprocessing for all systems and keep successful entries."""
    manifest_entries: list[dict[str, Any]] = []
    for system in systems:
        entry = preprocess_fn(system)
        if entry is not None:
            manifest_entries.append(entry)
    return manifest_entries


def write_manifest_entries(
    manifest_entries: list[dict[str, Any]],
    manifest_path: str | Path,
) -> Path:
    """Write manifest entries to a JSON file."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest_entries, f, indent=2)

    logger.info("Saved %d entries to %s", len(manifest_entries), manifest_path)
    return manifest_path
