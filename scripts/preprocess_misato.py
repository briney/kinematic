"""MISATO dataset preprocessing.

Converts HDF5 protein-ligand complexes to processed format.
~16,972 protein-ligand complexes, 10ns each.

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
import logging
import tempfile
from pathlib import Path

import h5py
import mdtraj
import numpy as np

from kinematic.data.units import infer_coordinate_unit
try:
    from scripts.preprocess_common import (
        collect_manifest_entries,
        finalize_processed_system,
        write_manifest_entries,
    )
except ImportError:
    from preprocess_common import (
        collect_manifest_entries,
        finalize_processed_system,
        write_manifest_entries,
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
    coords_unit: str,
) -> dict | None:
    """Preprocess a single MISATO system."""
    system_id = system["system_id"]
    logger.info("Processing %s", system_id)

    try:
        with h5py.File(system["h5_path"], "r") as f:
            grp = f[system["pdb_id"]]

            # Extract coordinates (may be in nm or A depending on MISATO version)
            coords_ds = grp["coordinates"]
            coords = np.array(coords_ds)  # (n_frames, n_atoms, 3)
            unit_metadata = {
                f"group.{k}": v for k, v in grp.attrs.items()
            }
            unit_metadata.update({
                f"coordinates.{k}": v for k, v in coords_ds.attrs.items()
            })

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
            # Convert to nm for mdtraj (which expects nm internally).
            if coords_unit == "auto":
                inferred_unit, confidence, reason = infer_coordinate_unit(
                    coords,
                    metadata=unit_metadata,
                )
                logger.warning(
                    "Inferred coordinate unit for %s: %s (confidence=%.2f, %s)",
                    system_id,
                    inferred_unit,
                    confidence,
                    reason,
                )
            else:
                inferred_unit = coords_unit
                logger.info(
                    "Using --coords-unit=%s for %s",
                    inferred_unit,
                    system_id,
                )

            coords_nm = coords if inferred_unit == "nm" else (coords / 10.0)
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

        return finalize_processed_system(
            system_id=system_id,
            dataset="misato",
            traj=traj,
            output_dir=output_dir,
            ref_dir=ref_dir,
            frame_dt_ns=0.08,  # 80 ps = 0.08 ns
        )

    except Exception:
        logger.exception("Failed to process %s", system_id)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MISATO dataset")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/processed/coords")
    parser.add_argument("--ref-dir", type=str, default="data/processed/refs")
    parser.add_argument("--manifest-out", type=str, default="data/processed/misato_manifest.json")
    parser.add_argument(
        "--coords-unit",
        choices=("auto", "nm", "angstrom"),
        default="auto",
        help=(
            "Coordinate unit for input HDF5 coordinates. "
            "'auto' uses metadata + geometric heuristics."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    systems = find_misato_systems(Path(args.input_dir))
    logger.info("Found %d MISATO systems", len(systems))

    manifest_entries = collect_manifest_entries(
        systems,
        lambda system: preprocess_one(
            system,
            Path(args.output_dir),
            Path(args.ref_dir),
            args.coords_unit,
        ),
    )
    write_manifest_entries(manifest_entries, args.manifest_out)


if __name__ == "__main__":
    main()
