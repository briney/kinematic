"""Octapeptides dataset preprocessing.

~1,100 8-residue peptides, 5 x 1 us each, ~8 ms total.
Force field: AMBER ff99SB-ildn, 300K, explicit TIP3P, 0.1M NaCl.
Format: topology.pdb + trajs/run001_protein.cmprsd.xtc + dataset.json.
4 fs timestep with hydrogen mass repartitioning.

All 5 replicas per system are treated as independent trajectories.
System size is very small (8 residues, ~60-100 heavy atoms).

Usage:
    python scripts/preprocess_octapeptides.py \\
        --input-dir data/raw/octapeptides \\
        --output-dir data/processed/coords \\
        --ref-dir data/processed/refs
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import numpy as np

from boltzkinema.data.preprocessing import (
    align_trajectory,
    build_observation_mask,
    convert_trajectory,
    extract_atom_metadata,
    remove_solvent,
    save_reference_structure,
)

logger = logging.getLogger(__name__)


def find_octapeptide_systems(input_dir: Path) -> list[dict]:
    """Discover octapeptide systems.

    Expected layout:
        input_dir/<peptide_id>/topology.pdb
        input_dir/<peptide_id>/trajs/run*_protein.cmprsd.xtc (or *.xtc)
        input_dir/<peptide_id>/dataset.json
    """
    systems = []
    for peptide_dir in sorted(input_dir.iterdir()):
        if not peptide_dir.is_dir():
            continue

        topology = peptide_dir / "topology.pdb"
        if not topology.exists():
            pdb_files = sorted(peptide_dir.glob("*.pdb"))
            if not pdb_files:
                continue
            topology = pdb_files[0]

        # Find trajectory files (multiple replicas)
        trajs_dir = peptide_dir / "trajs"
        if trajs_dir.is_dir():
            traj_files = sorted(trajs_dir.glob("*.xtc"))
        else:
            traj_files = sorted(peptide_dir.glob("*.xtc"))

        if not traj_files:
            continue

        peptide_id = peptide_dir.name

        # Each replica is treated as an independent trajectory
        for i, traj_file in enumerate(traj_files):
            systems.append({
                "system_id": f"octa_{peptide_id}_rep{i}",
                "peptide_id": peptide_id,
                "topology": topology,
                "trajectory": traj_file,
            })

    return systems


def preprocess_one(
    system: dict,
    output_dir: Path,
    ref_dir: Path,
) -> dict | None:
    """Preprocess a single octapeptide trajectory."""
    system_id = system["system_id"]
    logger.info("Processing %s", system_id)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            clean_traj = remove_solvent(
                system["topology"],
                system["trajectory"],
                Path(tmpdir) / "clean.xtc",
            )
            traj = align_trajectory(clean_traj, system["topology"])

        coords_A = traj.xyz[0] * 10.0
        observed_mask = build_observation_mask(coords_A)

        coords_path = convert_trajectory(traj, system_id, output_dir, observed_mask)

        atoms = extract_atom_metadata(traj)
        ref_path = save_reference_structure(
            atoms, coords_A, ref_dir / f"{system_id}_ref.npz"
        )

        return {
            "system_id": system_id,
            "dataset": "octapeptides",
            "n_frames": traj.n_frames,
            "n_atoms": traj.n_atoms,
            "n_tokens": len(set(a["residue_index"] for a in atoms)),
            "frame_dt_ns": float(traj.timestep) / 1000.0,
            "split": "train",
            "coords_path": str(coords_path),
            "trunk_cache_dir": "",
            "ref_path": str(ref_path),
        }

    except Exception:
        logger.exception("Failed to process %s", system_id)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Octapeptides dataset")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/processed/coords")
    parser.add_argument("--ref-dir", type=str, default="data/processed/refs")
    parser.add_argument(
        "--manifest-out", type=str, default="data/processed/octapeptides_manifest.json"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    systems = find_octapeptide_systems(Path(args.input_dir))
    logger.info("Found %d octapeptide systems", len(systems))

    manifest_entries = []
    for system in systems:
        entry = preprocess_one(system, Path(args.output_dir), Path(args.ref_dir))
        if entry is not None:
            manifest_entries.append(entry)

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest_entries, f, indent=2)
    logger.info("Saved %d entries to %s", len(manifest_entries), manifest_path)


if __name__ == "__main__":
    main()
