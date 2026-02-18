"""ATLAS dataset preprocessing.

Converts GROMACS .xtc trajectories + .gro topology to processed format.
~1,500 protein chains, 3x100ns trajectories each.

Pipeline:
  1. Solvent removal (MDAnalysis)
  2. Backbone alignment to frame 0 (mdtraj Kabsch)
  3. Build observation mask
  4. Unit conversion: nm -> A, ps -> ns
  5. Save coords .npz + reference structure .npz

Usage:
    python scripts/preprocess_atlas.py \\
        --input-dir data/raw/atlas \\
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


def find_atlas_systems(input_dir: Path) -> list[dict]:
    """Discover ATLAS systems from directory structure.

    Expected layout:
        input_dir/<chain_id>/<chain_id>_<replica>.gro
        input_dir/<chain_id>/<chain_id>_<replica>.xtc
    """
    systems = []
    for chain_dir in sorted(input_dir.iterdir()):
        if not chain_dir.is_dir():
            continue
        chain_id = chain_dir.name

        # Find topology files (.gro)
        gro_files = sorted(chain_dir.glob("*.gro"))
        if not gro_files:
            logger.warning("No .gro file found for %s, skipping", chain_id)
            continue
        topology = gro_files[0]

        # Find trajectory files (.xtc)
        xtc_files = sorted(chain_dir.glob("*.xtc"))
        for i, xtc in enumerate(xtc_files):
            systems.append({
                "system_id": f"atlas_{chain_id}_rep{i}",
                "chain_id": chain_id,
                "replica": i,
                "topology": topology,
                "trajectory": xtc,
            })

    return systems


def preprocess_one(
    system: dict,
    output_dir: Path,
    ref_dir: Path,
) -> dict | None:
    """Preprocess a single ATLAS system.

    Returns manifest entry dict, or None on failure.
    """
    system_id = system["system_id"]
    logger.info("Processing %s", system_id)

    try:
        # Step 1: Solvent removal
        with tempfile.TemporaryDirectory() as tmpdir:
            clean_traj = remove_solvent(
                system["topology"],
                system["trajectory"],
                Path(tmpdir) / "clean.xtc",
            )

            # Step 2: Frame alignment
            traj = align_trajectory(clean_traj, system["topology"])

        # Step 3: Build observation mask
        coords_A = traj.xyz[0] * 10.0  # first frame, nm -> A
        observed_mask = build_observation_mask(coords_A)

        # Step 4-5: Convert and save
        coords_path = convert_trajectory(
            traj, system_id, output_dir, observed_mask
        )

        # Save reference structure
        atoms = extract_atom_metadata(traj)
        ref_path = save_reference_structure(
            atoms, coords_A, ref_dir / f"{system_id}_ref.npz"
        )

        return {
            "system_id": system_id,
            "dataset": "atlas",
            "n_frames": traj.n_frames,
            "n_atoms": traj.n_atoms,
            "n_tokens": len(set(a["residue_index"] for a in atoms)),
            "frame_dt_ns": float(traj.timestep) / 1000.0,
            "split": "train",  # Assigned later by split script
            "coords_path": str(coords_path),
            "trunk_cache_dir": "",  # Filled after trunk precomputation
            "ref_path": str(ref_path),
        }

    except Exception:
        logger.exception("Failed to process %s", system_id)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ATLAS dataset")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/processed/coords")
    parser.add_argument("--ref-dir", type=str, default="data/processed/refs")
    parser.add_argument("--manifest-out", type=str, default="data/processed/atlas_manifest.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ref_dir = Path(args.ref_dir)

    systems = find_atlas_systems(input_dir)
    logger.info("Found %d ATLAS systems", len(systems))

    manifest_entries = []
    for system in systems:
        entry = preprocess_one(system, output_dir, ref_dir)
        if entry is not None:
            manifest_entries.append(entry)

    # Save manifest
    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest_entries, f, indent=2)
    logger.info(
        "Saved %d entries to %s", len(manifest_entries), manifest_path
    )


if __name__ == "__main__":
    main()
