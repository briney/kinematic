"""``kinematic train`` subcommand."""

from __future__ import annotations

import click

from kinematic.cli._config import compose_config
from kinematic.model.checkpoint_io import (
    has_unresolved_step_placeholder,
    resolve_checkpoint_path,
)


def _preflight_checkpoint_paths(cfg) -> None:
    """Fail fast on unresolved checkpoint placeholders in training config."""
    resume_from = cfg.get("resume_from")
    if not resume_from:
        return

    had_placeholder = has_unresolved_step_placeholder(resume_from)
    resolved = resolve_checkpoint_path(resume_from, auto_resolve_latest=True)
    cfg.resume_from = str(resolved)

    if had_placeholder:
        click.echo(f"[preflight] Resolved resume_from: {resume_from} -> {resolved}")


@click.command()
@click.option(
    "--config",
    "config_name",
    required=True,
    help="Config name without .yaml (e.g. train_phase0).",
)
@click.option(
    "--base-config",
    "base_config_name",
    default=None,
    help="Optional base config name (e.g. base).",
)
@click.option(
    "--config-dir",
    default=None,
    help="Override config directory path.",
)
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def train(config_name, base_config_name, config_dir, overrides):
    """Run training with Hydra-composed config.

    Accepts Hydra-style overrides as trailing arguments:

        kinematic train --config train_phase0 lr=2e-4 max_steps=50000
    """
    cfg = compose_config(
        config_name=config_name,
        base_config_name=base_config_name,
        overrides=list(overrides),
        config_dir=config_dir,
    )

    _preflight_checkpoint_paths(cfg)

    # Defer heavy imports to keep `kinematic --help` fast
    from kinematic.training.trainer import train as run_training

    run_training(cfg)
