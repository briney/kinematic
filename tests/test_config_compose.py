"""Tests for Hydra Compose API config loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig

from kinematic.cli._config import compose_config

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

# Skip all tests if configs directory doesn't exist (e.g. in CI without configs)
pytestmark = pytest.mark.skipif(
    not CONFIGS_DIR.is_dir(),
    reason="configs/ directory not found",
)


def test_compose_single_config():
    """Loading a single config by name returns a DictConfig."""
    cfg = compose_config("train_phase0", config_dir=str(CONFIGS_DIR))
    assert isinstance(cfg, DictConfig)
    # Phase 0 should have token_s defined
    assert "token_s" in cfg


def test_compose_with_base():
    """Base + phase config merge works correctly."""
    cfg = compose_config(
        "train_phase0",
        base_config_name="base",
        config_dir=str(CONFIGS_DIR),
    )
    assert isinstance(cfg, DictConfig)
    assert "token_s" in cfg


def test_compose_with_overrides():
    """CLI overrides are applied on top of config."""
    cfg = compose_config(
        "train_phase0",
        overrides=["token_s=999"],
        config_dir=str(CONFIGS_DIR),
    )
    assert cfg.token_s == 999


def test_compose_base_plus_overrides():
    """Base + phase + CLI overrides all merge correctly."""
    cfg = compose_config(
        "train_phase0",
        base_config_name="base",
        overrides=["token_s=777"],
        config_dir=str(CONFIGS_DIR),
    )
    assert cfg.token_s == 777


def test_compose_repeated_calls():
    """Repeated calls don't fail (GlobalHydra is cleared properly)."""
    cfg1 = compose_config("train_phase0", config_dir=str(CONFIGS_DIR))
    cfg2 = compose_config("train_phase0", config_dir=str(CONFIGS_DIR))
    assert cfg1.token_s == cfg2.token_s
