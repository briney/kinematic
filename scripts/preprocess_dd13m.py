"""DD-13M dataset preprocessing for unbinding trajectories.

565 complexes, 26,612 metadynamics dissociation trajectories.
Coordinate trajectories at 10 ps intervals.

Pipeline:
  1. Read trajectory files
  2. Solvent removal if needed
  3. Frame alignment to frame 0
  4. Build observation mask
  5. Unit conversion: nm -> A, ps -> ns
  6. Save coords .npz + reference structure .npz

Usage:
    python scripts/preprocess_dd13m.py \\
        --input-dir data/raw/dd13m \\
        --output-dir data/processed/coords \\
        --ref-dir data/processed/refs
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import mdtraj
import numpy as np

from kinematic.data.preprocessing import (
    align_trajectory,
    remove_solvent,
)
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


def find_dd13m_systems(input_dir: Path) -> list[dict]:
    """Discover DD-13M systems from directory structure.

    Expected layout varies by DD-13M release; adapt to actual format.
    Looks for PDB + trajectory pairs.
    """
    systems = []

    # Look for trajectory directories
    for complex_dir in sorted(input_dir.iterdir()):
        if not complex_dir.is_dir():
            continue
        complex_id = complex_dir.name

        # Find topology (PDB)
        pdb_files = sorted(complex_dir.glob("*.pdb"))
        if not pdb_files:
            continue
        topology = pdb_files[0]

        # Find trajectory files
        traj_files = (
            sorted(complex_dir.glob("*.xtc"))
            + sorted(complex_dir.glob("*.dcd"))
            + sorted(complex_dir.glob("*.trr"))
        )

        for i, traj_file in enumerate(traj_files):
            systems.append({
                "system_id": f"dd13m_{complex_id}_traj{i}",
                "complex_id": complex_id,
                "topology": topology,
                "trajectory": traj_file,
            })

    return systems


def preprocess_one(
    system: dict,
    output_dir: Path,
    ref_dir: Path,
) -> dict | None:
    """Preprocess a single DD-13M trajectory."""
    system_id = system["system_id"]
    logger.info("Processing %s", system_id)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Attempt solvent removal
            try:
                clean_traj = remove_solvent(
                    system["topology"],
                    system["trajectory"],
                    Path(tmpdir) / "clean.xtc",
                )
                traj = align_trajectory(clean_traj, system["topology"])
            except Exception:
                # If solvent removal fails (e.g., already clean), load directly
                traj = mdtraj.load(
                    str(system["trajectory"]), top=str(system["topology"])
                )
                backbone = traj.topology.select("backbone")
                if len(backbone) > 0:
                    traj.superpose(traj, frame=0, atom_indices=backbone)

        # DD-13M: 10 ps intervals
        if traj.time is None or len(traj.time) == 0:
            traj.time = np.arange(traj.n_frames, dtype=np.float32) * 10.0  # 10 ps

        return finalize_processed_system(
            system_id=system_id,
            dataset="dd13m",
            traj=traj,
            output_dir=output_dir,
            ref_dir=ref_dir,
            frame_dt_ns=0.01,  # 10 ps = 0.01 ns
        )

    except Exception:
        logger.exception("Failed to process %s", system_id)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess DD-13M dataset")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/processed/coords")
    parser.add_argument("--ref-dir", type=str, default="data/processed/refs")
    parser.add_argument("--manifest-out", type=str, default="data/processed/dd13m_manifest.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    systems = find_dd13m_systems(Path(args.input_dir))
    logger.info("Found %d DD-13M systems", len(systems))

    manifest_entries = collect_manifest_entries(
        systems,
        lambda system: preprocess_one(
            system,
            Path(args.output_dir),
            Path(args.ref_dir),
        ),
    )
    write_manifest_entries(manifest_entries, args.manifest_out)


if __name__ == "__main__":
    main()
