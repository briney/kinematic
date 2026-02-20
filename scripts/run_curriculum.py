"""Multi-phase training curriculum orchestrator.

Runs the Kinematic training curriculum sequentially:
    Phase 0: Monomer dynamics pretraining (CATH2 + ATLAS + Octapeptides)
    Phase 1: Full mixed training (equilibrium dynamics datasets)
    Phase 1.5: MegaSim mutant enrichment (optional, skipped by default)
    Phase 2: Unbinding fine-tuning (DD-13M)

Each phase automatically resolves the checkpoint path from the previous
phase's output_dir and max_steps.

Usage:
    # Run full curriculum (Phases 0, 1, 2):
    python scripts/run_curriculum.py

    # Include optional Phase 1.5:
    python scripts/run_curriculum.py --include-phase1.5

    # Start from Phase 1 (Phase 0 already complete):
    python scripts/run_curriculum.py --start-phase 1

    # Dry run (print commands without executing):
    python scripts/run_curriculum.py --dry-run

    # Custom accelerate args:
    python scripts/run_curriculum.py --num-processes 8
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class PhaseSpec:
    """Specification for a training phase."""

    name: str
    config_path: str
    depends_on: str | None = None  # phase name this depends on


# Curriculum phase definitions
PHASES: dict[str, PhaseSpec] = {
    "0": PhaseSpec(
        name="0",
        config_path="configs/train_phase0.yaml",
        depends_on=None,
    ),
    "1": PhaseSpec(
        name="1",
        config_path="configs/train_equilibrium.yaml",
        depends_on="0",
    ),
    "1.5": PhaseSpec(
        name="1.5",
        config_path="configs/train_mutant_enrichment.yaml",
        depends_on="1",
    ),
    "2": PhaseSpec(
        name="2",
        config_path="configs/train_unbinding.yaml",
        depends_on="1",  # defaults to Phase 1; updated to 1.5 if included
    ),
}


def _resolve_checkpoint_path(output_dir: str, max_steps: int) -> str:
    """Build the expected checkpoint path for a completed phase."""
    return os.path.join(output_dir, f"step_{max_steps}")


def _find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest checkpoint in an output directory.

    Looks for step_XXXXX directories and returns the one with highest step.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = []
    for d in output_path.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                step = int(d.name.split("step_")[1])
                checkpoints.append((step, str(d)))
            except (ValueError, IndexError):
                continue

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def _build_phase_command(
    phase: PhaseSpec,
    base_config: str | None,
    num_processes: int,
    resume_from: str | None = None,
) -> list[str]:
    """Build the accelerate launch command for a phase."""
    cmd = [
        "accelerate", "launch",
        f"--num_processes={num_processes}",
        "--mixed_precision=bf16",
        "-m", "kinematic", "train",
    ]

    # Config names are stems (e.g. "train_phase0"), not full paths
    if base_config:
        cmd.extend(["--base-config", Path(base_config).stem])

    cmd.extend(["--config", Path(phase.config_path).stem])

    # Override resume_from if we resolved a checkpoint from a prior phase
    if resume_from:
        cmd.append(f"resume_from={resume_from}")

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Kinematic multi-phase training curriculum"
    )
    parser.add_argument(
        "--start-phase",
        type=str,
        default="0",
        choices=["0", "1", "1.5", "2"],
        help="Phase to start from (default: 0)",
    )
    parser.add_argument(
        "--end-phase",
        type=str,
        default="2",
        choices=["0", "1", "1.5", "2"],
        help="Phase to end at, inclusive (default: 2)",
    )
    parser.add_argument(
        "--include-mutant-enrichment",
        action="store_true",
        default=False,
        help="Include optional Phase 1.5 (MegaSim mutant enrichment)",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/base.yaml",
        help="Shared base config (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Number of GPUs for accelerate (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print commands without executing",
    )
    args = parser.parse_args()

    # Build the sequence of phases to run
    phase_order = ["0", "1"]
    if args.include_mutant_enrichment:
        phase_order.append("1.5")
        # If Phase 1.5 is included, Phase 2 depends on 1.5 instead of 1
        PHASES["2"] = PhaseSpec(
            name="2",
            config_path="configs/train_unbinding.yaml",
            depends_on="1.5",
        )
    phase_order.append("2")

    # Filter to requested range
    all_phases = {p: i for i, p in enumerate(phase_order)}
    start_idx = all_phases.get(args.start_phase)
    end_idx = all_phases.get(args.end_phase)

    if start_idx is None:
        print(f"Error: start-phase '{args.start_phase}' not in curriculum sequence")
        sys.exit(1)
    if end_idx is None:
        print(f"Error: end-phase '{args.end_phase}' not in curriculum sequence")
        sys.exit(1)

    phases_to_run = phase_order[start_idx:end_idx + 1]

    # Check base config exists
    base_config = args.base_config
    if not Path(base_config).exists():
        base_config = None

    # Track completed phase checkpoints for auto-resolution
    completed_checkpoints: dict[str, str] = {}

    # If starting from a later phase, try to find checkpoints from earlier phases
    for phase_name in phase_order[:start_idx]:
        spec = PHASES[phase_name]
        cfg = OmegaConf.load(spec.config_path)
        if base_config:
            cfg = OmegaConf.merge(OmegaConf.load(base_config), cfg)
        output_dir = cfg.get("output_dir", f"checkpoints/phase{phase_name}/")
        ckpt = _find_latest_checkpoint(output_dir)
        if ckpt:
            completed_checkpoints[phase_name] = ckpt
            print(f"Found existing Phase {phase_name} checkpoint: {ckpt}")
        else:
            # Try the expected path from max_steps
            max_steps = cfg.get("max_steps", 0)
            expected = _resolve_checkpoint_path(output_dir, max_steps)
            if Path(expected).exists():
                completed_checkpoints[phase_name] = expected
                print(f"Found existing Phase {phase_name} checkpoint: {expected}")

    print(f"\nCurriculum: Phases {' -> '.join(phases_to_run)}")
    print(f"{'=' * 60}\n")

    # Run each phase
    for phase_name in phases_to_run:
        spec = PHASES[phase_name]

        # Resolve resume_from from the prior phase's checkpoint
        resume_from = None
        if spec.depends_on and spec.depends_on in completed_checkpoints:
            resume_from = completed_checkpoints[spec.depends_on]

        cmd = _build_phase_command(
            spec,
            base_config=base_config,
            num_processes=args.num_processes,
            resume_from=resume_from,
        )

        print(f"Phase {phase_name}: {spec.config_path}")
        if resume_from:
            print(f"  Resuming from: {resume_from}")
        print(f"  Command: {' '.join(cmd)}")

        if args.dry_run:
            print(f"  [DRY RUN] Skipping execution\n")
            # For dry run, simulate a checkpoint path for downstream phases
            cfg = OmegaConf.load(spec.config_path)
            if base_config:
                cfg = OmegaConf.merge(OmegaConf.load(base_config), cfg)
            output_dir = cfg.get("output_dir", f"checkpoints/phase{phase_name}/")
            max_steps = cfg.get("max_steps", 0)
            completed_checkpoints[phase_name] = _resolve_checkpoint_path(
                output_dir, max_steps
            )
            continue

        print(f"  Starting...\n")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\nPhase {phase_name} failed with return code {result.returncode}")
            sys.exit(result.returncode)

        # Find the checkpoint produced by this phase
        cfg = OmegaConf.load(spec.config_path)
        if base_config:
            cfg = OmegaConf.merge(OmegaConf.load(base_config), cfg)
        output_dir = cfg.get("output_dir", f"checkpoints/phase{phase_name}/")

        ckpt = _find_latest_checkpoint(output_dir)
        if ckpt:
            completed_checkpoints[phase_name] = ckpt
            print(f"\nPhase {phase_name} complete. Checkpoint: {ckpt}\n")
        else:
            print(f"\nWARNING: No checkpoint found in {output_dir} after Phase {phase_name}\n")

    print(f"{'=' * 60}")
    print("Curriculum complete!")
    for phase_name, ckpt in completed_checkpoints.items():
        print(f"  Phase {phase_name}: {ckpt}")


if __name__ == "__main__":
    main()
