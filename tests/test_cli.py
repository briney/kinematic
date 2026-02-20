"""Tests for the Click CLI interface."""

from __future__ import annotations

import subprocess
import sys

from click.testing import CliRunner

from kinematic.cli import cli


def test_cli_help():
    """Top-level --help shows group description and subcommands."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Kinematic" in result.output
    assert "train" in result.output
    assert "download-training-data" in result.output


def test_cli_version():
    """--version prints the package version."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower()


def test_train_help():
    """``kinematic train --help`` shows train options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output
    assert "--base-config" in result.output
    assert "--config-dir" in result.output


def test_download_help():
    """``kinematic download-training-data --help`` shows download options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["download-training-data", "--help"])
    assert result.exit_code == 0
    assert "--datasets" in result.output
    assert "--output-dir" in result.output


def test_python_m_kinematic_help():
    """``python -m kinematic --help`` works via __main__.py."""
    result = subprocess.run(
        [sys.executable, "-m", "kinematic", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "Kinematic" in result.stdout
    assert "train" in result.stdout
