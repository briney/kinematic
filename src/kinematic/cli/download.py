"""``kinematic download-training-data`` subcommand."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from kinematic.data.download import ALL_DATASETS, download_datasets


@click.command("download-training-data")
@click.option(
    "--datasets",
    default="all",
    help="Comma-separated dataset names or 'all'. "
    f"Available: {', '.join(ALL_DATASETS)}.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/raw"),
    show_default=True,
    help="Root output directory for raw data.",
)
def download_training_data(datasets: str, output_dir: Path) -> None:
    """Download training datasets."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if datasets == "all":
        names = list(ALL_DATASETS.keys())
    else:
        names = [d.strip() for d in datasets.split(",")]

    download_datasets(names, output_dir)
