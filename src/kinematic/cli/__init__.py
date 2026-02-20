"""Kinematic CLI — Click-based command-line interface."""

from __future__ import annotations

import click

from kinematic.cli.download import download_training_data
from kinematic.cli.train import train


@click.group()
@click.version_option(package_name="kinematic")
def cli() -> None:
    """Kinematic: Diffusion-based biomolecular trajectory generation."""


cli.add_command(train)
cli.add_command(download_training_data)
