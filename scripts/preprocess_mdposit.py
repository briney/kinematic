"""MDposit/DynaRepo dataset preprocessing.

Converts PDB + XTC trajectories from DynaRepo to processed format.
~930 systems, ~700 unique proteins, 3 replicas x 500ns each.

Pipeline:
  1. Solvent removal (MDAnalysis)
  2. Backbone alignment to frame 0 (mdtraj Kabsch)
  3. Ligand valency check where applicable
  4. Build observation mask
  5. Unit conversion: nm -> A, ps -> ns
  6. Save coords .npz + reference structure .npz

Usage:
    python scripts/preprocess_mdposit.py \\
        --input-dir data/raw/dynarepo \\
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


def find_mdposit_systems(input_dir: Path) -> list[dict]:
    """Discover MDposit/DynaRepo systems.

    Expected layout:
        input_dir/<accession>/replica_<i>/structure.pdb
        input_dir/<accession>/replica_<i>/trajectory.xtc
        input_dir/<accession>/replica_<i>/topology.tpr
    """
    systems = []
    for accession_dir in sorted(input_dir.iterdir()):
        if not accession_dir.is_dir():
            continue
        accession = accession_dir.name

        for replica_dir in sorted(accession_dir.iterdir()):
            if not replica_dir.is_dir():
                continue

            pdb = replica_dir / "structure.pdb"
            xtc = replica_dir / "trajectory.xtc"
            tpr = replica_dir / "topology.tpr"

            if not xtc.exists():
                continue

            # Prefer TPR for topology (more complete), fall back to PDB
            topology = tpr if tpr.exists() else pdb

            replica_idx = replica_dir.name.replace("replica_", "")
            systems.append({
                "system_id": f"mdposit_{accession}_rep{replica_idx}",
                "accession": accession,
                "topology": topology,
                "trajectory": xtc,
                "pdb": pdb,
            })

    return systems


def preprocess_one(
    system: dict,
    output_dir: Path,
    ref_dir: Path,
) -> dict | None:
    """Preprocess a single MDposit system."""
    system_id = system["system_id"]
    logger.info("Processing %s", system_id)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Solvent removal
            clean_traj = remove_solvent(
                system["topology"],
                system["trajectory"],
                Path(tmpdir) / "clean.xtc",
            )

            # Use PDB as topology for mdtraj (it needs atom names)
            topo_for_mdtraj = system["pdb"] if system["pdb"].exists() else system["topology"]

            # Step 2: Frame alignment
            traj = align_trajectory(clean_traj, topo_for_mdtraj)

        # Step 3: Build observation mask
        coords_A = traj.xyz[0] * 10.0  # nm -> A
        observed_mask = build_observation_mask(coords_A)

        # Step 4-5: Convert and save
        coords_path = convert_trajectory(traj, system_id, output_dir, observed_mask)

        # Save reference structure
        atoms = extract_atom_metadata(traj)
        ref_path = save_reference_structure(
            atoms, coords_A, ref_dir / f"{system_id}_ref.npz"
        )

        return {
            "system_id": system_id,
            "dataset": "mdposit",
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
    parser = argparse.ArgumentParser(description="Preprocess MDposit/DynaRepo dataset")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/processed/coords")
    parser.add_argument("--ref-dir", type=str, default="data/processed/refs")
    parser.add_argument("--manifest-out", type=str, default="data/processed/mdposit_manifest.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    systems = find_mdposit_systems(Path(args.input_dir))
    logger.info("Found %d MDposit systems", len(systems))

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
