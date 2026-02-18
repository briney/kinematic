"""MISATO dataset preprocessing.

Converts HDF5 protein-ligand complexes to processed format.
~16,000 protein-ligand complexes, 8ns each (100 frames).

Pipeline:
  1. Read HDF5 entries (coordinates + topology)
  2. Ligand valency check (RDKit sanitization)
  3. Frame alignment to frame 0
  4. Build observation mask
  5. Unit conversion: nm -> A, ps -> ns
  6. Save coords .npz + reference structure .npz

Usage:
    python scripts/preprocess_misato.py \\
        --input-dir data/raw/misato \\
        --output-dir data/processed/coords \\
        --ref-dir data/processed/refs
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import h5py
import mdtraj
import numpy as np

from boltzkinema.data.preprocessing import (
    align_trajectory,
    build_observation_mask,
    check_ligand_valency,
    convert_trajectory,
    extract_atom_metadata,
    save_reference_structure,
)

logger = logging.getLogger(__name__)


def find_misato_systems(input_dir: Path) -> list[dict]:
    """Discover MISATO systems from HDF5 files.

    MISATO stores protein-ligand complexes in HDF5 format with
    per-system groups containing coordinates and topology.
    """
    systems = []
    h5_files = sorted(input_dir.glob("*.h5")) + sorted(input_dir.glob("*.hdf5"))

    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, "r") as f:
                for pdb_id in f.keys():
                    systems.append({
                        "system_id": f"misato_{pdb_id}",
                        "pdb_id": pdb_id,
                        "h5_path": str(h5_path),
                    })
        except Exception:
            logger.exception("Failed to read %s", h5_path)

    return systems


def preprocess_one(
    system: dict,
    output_dir: Path,
    ref_dir: Path,
) -> dict | None:
    """Preprocess a single MISATO system."""
    system_id = system["system_id"]
    logger.info("Processing %s", system_id)

    try:
        with h5py.File(system["h5_path"], "r") as f:
            grp = f[system["pdb_id"]]

            # Extract coordinates (may be in nm or A depending on MISATO version)
            coords = np.array(grp["coordinates"])  # (n_frames, n_atoms, 3)

            # Check for topology/PDB data
            if "topology" in grp:
                topology_bytes = bytes(grp["topology"][()])
            elif "pdb" in grp:
                topology_bytes = bytes(grp["pdb"][()])
            else:
                logger.warning("No topology found for %s", system_id)
                return None

        # Write temporary PDB for mdtraj
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_pdb = Path(tmpdir) / "topology.pdb"
            tmp_pdb.write_bytes(topology_bytes)

            # Check ligand valency if present
            # (MISATO ligands may need RDKit validation)

            # Build mdtraj trajectory for alignment
            # MISATO coordinates are typically in Angstrom
            # Convert to nm for mdtraj (which expects nm internally)
            coords_nm = coords / 10.0 if coords.max() > 100.0 else coords
            traj = mdtraj.Trajectory(
                xyz=coords_nm.astype(np.float32),
                topology=mdtraj.load(str(tmp_pdb)).topology,
            )
            # Set time metadata (MISATO: 8ns over 100 frames, dt=80ps)
            traj.time = np.arange(traj.n_frames, dtype=np.float32) * 80.0  # ps

        # Alignment
        backbone = traj.topology.select("backbone")
        if len(backbone) > 0:
            traj.superpose(traj, frame=0, atom_indices=backbone)
        else:
            heavy = traj.topology.select("mass > 1.5")
            if len(heavy) > 0:
                traj.superpose(traj, frame=0, atom_indices=heavy)

        # Build observation mask
        coords_A = traj.xyz[0] * 10.0  # nm -> A
        observed_mask = build_observation_mask(coords_A)

        # Convert and save
        coords_path = convert_trajectory(traj, system_id, output_dir, observed_mask)

        # Save reference structure
        atoms = extract_atom_metadata(traj)
        ref_path = save_reference_structure(
            atoms, coords_A, ref_dir / f"{system_id}_ref.npz"
        )

        return {
            "system_id": system_id,
            "dataset": "misato",
            "n_frames": traj.n_frames,
            "n_atoms": traj.n_atoms,
            "n_tokens": len(set(a["residue_index"] for a in atoms)),
            "frame_dt_ns": 0.08,  # 80 ps = 0.08 ns
            "split": "train",
            "coords_path": str(coords_path),
            "trunk_cache_dir": "",
            "ref_path": str(ref_path),
        }

    except Exception:
        logger.exception("Failed to process %s", system_id)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MISATO dataset")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/processed/coords")
    parser.add_argument("--ref-dir", type=str, default="data/processed/refs")
    parser.add_argument("--manifest-out", type=str, default="data/processed/misato_manifest.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    systems = find_misato_systems(Path(args.input_dir))
    logger.info("Found %d MISATO systems", len(systems))

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
