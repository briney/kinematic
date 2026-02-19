"""MegaSim preprocessing with mutant subsampling.

271 wildtype proteins (2 force fields) + 21,458 point mutants.
Force field: AMBER ff14sb + ff99sb-disp, 295K.
Wildtype: 1.5 us/seed (last 1 us retained). Mutants: 1 us each.
Format: topology.pdb + trajs/*.xtc + dataset.json per system.

Mutant subsampling: select all 271 wildtypes + ~4,500 mutants
(top/bottom 10% by deltaG + diverse-position representatives).

Usage:
    python scripts/preprocess_megasim.py \\
        --input-dir data/raw/megasim \\
        --output-dir data/processed/coords \\
        --ref-dir data/processed/refs \\
        --subsample-mutants
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

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


def find_megasim_systems(
    input_dir: Path,
    subsample_mutants: bool = True,
) -> list[dict]:
    """Discover MegaSim systems with optional mutant subsampling.

    Expected layout (from megasim_wildtype_merge.zip / megasim_mutants_allatom.zip):
        input_dir/<system_id>/topology.pdb
        input_dir/<system_id>/trajs/*.xtc
        input_dir/<system_id>/dataset.json

    dataset.json should contain:
        - "type": "wildtype" or "mutant"
        - "parent_id": wildtype parent (for mutants)
        - "deltaG": experimental stability (for mutant subsampling)
        - "mutation": mutation description
    """
    systems = []
    all_mutants = []

    for system_dir in sorted(input_dir.iterdir()):
        if not system_dir.is_dir():
            continue

        topology = system_dir / "topology.pdb"
        if not topology.exists():
            pdb_files = sorted(system_dir.glob("*.pdb"))
            if not pdb_files:
                continue
            topology = pdb_files[0]

        trajs_dir = system_dir / "trajs"
        if trajs_dir.is_dir():
            traj_files = sorted(trajs_dir.glob("*.xtc"))
        else:
            traj_files = sorted(system_dir.glob("*.xtc"))

        if not traj_files:
            continue

        # Load metadata
        metadata = {}
        dataset_json = system_dir / "dataset.json"
        if dataset_json.exists():
            with open(dataset_json) as f:
                metadata = json.load(f)

        sys_type = metadata.get("type", "wildtype")
        system_id_base = system_dir.name

        entry = {
            "system_dir": system_dir,
            "system_id_base": system_id_base,
            "topology": topology,
            "traj_files": traj_files,
            "metadata": metadata,
            "sys_type": sys_type,
        }

        if sys_type == "wildtype":
            systems.append(entry)
        else:
            all_mutants.append(entry)

    # Mutant subsampling
    if subsample_mutants and all_mutants:
        selected = _subsample_mutants(all_mutants)
        systems.extend(selected)
        logger.info(
            "Subsampled %d/%d mutants", len(selected), len(all_mutants)
        )
    else:
        systems.extend(all_mutants)

    # Flatten: each trajectory file becomes a separate system entry
    flat_systems = []
    for entry in systems:
        for i, traj_file in enumerate(entry["traj_files"]):
            dataset_label = (
                "megasim_wt" if entry["sys_type"] == "wildtype" else "megasim_mut"
            )
            flat_systems.append({
                "system_id": f"megasim_{entry['system_id_base']}_traj{i}",
                "topology": entry["topology"],
                "trajectory": traj_file,
                "metadata": entry["metadata"],
                "dataset": dataset_label,
            })

    return flat_systems


def _subsample_mutants(
    all_mutants: list[dict],
    tail_fraction: float = 0.10,
    target_diverse: int = 2000,
) -> list[dict]:
    """Select ~4,500 mutants: top/bottom 10% by deltaG + diverse positions.

    Strategy:
      1. Collect all mutants with valid deltaG
      2. Take |z-score| > 1.28 (top/bottom ~10%)
      3. Add stratified random sample across mutation positions for diversity
    """
    # Collect deltaG values
    mutants_with_dg = []
    for m in all_mutants:
        dg = m["metadata"].get("deltaG")
        if dg is not None:
            mutants_with_dg.append((m, float(dg)))

    if not mutants_with_dg:
        logger.warning("No deltaG values found; returning all mutants")
        return all_mutants

    dg_values = np.array([dg for _, dg in mutants_with_dg])
    mean_dg = dg_values.mean()
    std_dg = dg_values.std() + 1e-8
    z_scores = (dg_values - mean_dg) / std_dg

    # Tail selection: |z| > 1.28 (~top/bottom 10%)
    tail_mask = np.abs(z_scores) > 1.28
    selected_set = set()
    for i, is_tail in enumerate(tail_mask):
        if is_tail:
            selected_set.add(i)

    logger.info("Tail-selected %d mutants by deltaG", len(selected_set))

    # Diverse position sampling
    position_groups: dict[str, list[int]] = {}
    for i, (m, _) in enumerate(mutants_with_dg):
        if i in selected_set:
            continue
        mutation = m["metadata"].get("mutation", "")
        # Group by position (e.g., "A42G" -> position "42")
        pos = "".join(c for c in mutation if c.isdigit())
        position_groups.setdefault(pos or "unknown", []).append(i)

    # Sample from each position group
    rng = np.random.default_rng(42)
    remaining_needed = max(0, target_diverse)
    positions = list(position_groups.keys())
    rng.shuffle(positions)

    for pos in positions:
        if remaining_needed <= 0:
            break
        indices = position_groups[pos]
        n_take = min(len(indices), max(1, remaining_needed // max(1, len(positions))))
        chosen = rng.choice(indices, size=min(n_take, len(indices)), replace=False)
        for idx in chosen:
            selected_set.add(idx)
            remaining_needed -= 1

    selected = [mutants_with_dg[i][0] for i in sorted(selected_set)]
    logger.info("Total selected mutants: %d", len(selected))
    return selected


def preprocess_one(
    system: dict,
    output_dir: Path,
    ref_dir: Path,
) -> dict | None:
    """Preprocess a single MegaSim trajectory."""
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

        return finalize_processed_system(
            system_id=system_id,
            dataset=system["dataset"],
            traj=traj,
            output_dir=output_dir,
            ref_dir=ref_dir,
        )

    except Exception:
        logger.exception("Failed to process %s", system_id)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MegaSim dataset")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/processed/coords")
    parser.add_argument("--ref-dir", type=str, default="data/processed/refs")
    parser.add_argument("--manifest-out", type=str, default="data/processed/megasim_manifest.json")
    parser.add_argument(
        "--subsample-mutants",
        action="store_true",
        help="Subsample mutants to ~4,500 (default for Phase 1 training)",
    )
    parser.add_argument(
        "--all-mutants",
        action="store_true",
        help="Keep all 21,458 mutants (for Phase 1.5 training)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    subsample = args.subsample_mutants or not args.all_mutants
    systems = find_megasim_systems(Path(args.input_dir), subsample_mutants=subsample)
    logger.info("Found %d MegaSim systems", len(systems))

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
