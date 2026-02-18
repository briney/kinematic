"""CATH2 domains preprocessing.

~1,100 CATH domains, 50-200 amino acids, ~1 us each, ~41 ms total.
Force field: AMBER ff99SB-ildn, 300K, explicit TIP3P.
Format: topology.pdb + trajs/*.cmprsd.xtc + dataset.json per system.

NOTE: CATH1 (adaptive sampling) is excluded — its non-equilibrium
transition-state conformations conflict with noise-as-masking.

Pipeline:
  1. Load topology.pdb + compressed XTC trajectories
  2. Solvent removal
  3. Backbone alignment to frame 0
  4. Build observation mask
  5. Unit conversion: nm -> A, ps -> ns
  6. Save coords .npz + reference structure .npz

Usage:
    python scripts/preprocess_cath.py \\
        --input-dir data/raw/cath2 \\
        --output-dir data/processed/coords \\
        --ref-dir data/processed/refs
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import mdtraj
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


def find_cath2_systems(input_dir: Path) -> list[dict]:
    """Discover CATH2 systems.

    Expected layout (from BioEmu/MSR_cath2.zip):
        input_dir/<domain_id>/topology.pdb
        input_dir/<domain_id>/trajs/*.cmprsd.xtc (or *.xtc)
        input_dir/<domain_id>/dataset.json
    """
    systems = []
    for domain_dir in sorted(input_dir.iterdir()):
        if not domain_dir.is_dir():
            continue

        topology = domain_dir / "topology.pdb"
        if not topology.exists():
            # Try alternative names
            pdb_files = sorted(domain_dir.glob("*.pdb"))
            if not pdb_files:
                continue
            topology = pdb_files[0]

        # Find trajectory files
        trajs_dir = domain_dir / "trajs"
        if trajs_dir.is_dir():
            traj_files = sorted(trajs_dir.glob("*.xtc"))
        else:
            traj_files = sorted(domain_dir.glob("*.xtc"))

        if not traj_files:
            continue

        domain_id = domain_dir.name

        # Load dataset.json for metadata if available
        metadata = {}
        dataset_json = domain_dir / "dataset.json"
        if dataset_json.exists():
            with open(dataset_json) as f:
                metadata = json.load(f)

        for i, traj_file in enumerate(traj_files):
            systems.append({
                "system_id": f"cath2_{domain_id}_traj{i}",
                "domain_id": domain_id,
                "topology": topology,
                "trajectory": traj_file,
                "metadata": metadata,
            })

    return systems


def preprocess_one(
    system: dict,
    output_dir: Path,
    ref_dir: Path,
) -> dict | None:
    """Preprocess a single CATH2 domain trajectory."""
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

            # Step 2: Frame alignment
            traj = align_trajectory(clean_traj, system["topology"])

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
            "dataset": "cath2",
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
    parser = argparse.ArgumentParser(description="Preprocess CATH2 dataset")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/processed/coords")
    parser.add_argument("--ref-dir", type=str, default="data/processed/refs")
    parser.add_argument("--manifest-out", type=str, default="data/processed/cath2_manifest.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    systems = find_cath2_systems(Path(args.input_dir))
    logger.info("Found %d CATH2 systems", len(systems))

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
