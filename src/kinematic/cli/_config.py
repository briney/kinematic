"""Hydra Compose API helper for config loading."""

from __future__ import annotations

import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf


def _resolve_config_dir(override: str | None = None) -> str:
    """Find the configs directory.

    Resolution order:
    1. Explicit override (from --config-dir)
    2. KINEMATIC_CONFIG_DIR environment variable
    3. cwd/configs/
    """
    if override:
        p = Path(override).resolve()
        if p.is_dir():
            return str(p)
        raise FileNotFoundError(f"Config directory not found: {p}")

    env = os.environ.get("KINEMATIC_CONFIG_DIR")
    if env:
        p = Path(env).resolve()
        if p.is_dir():
            return str(p)

    cwd_configs = Path.cwd() / "configs"
    if cwd_configs.is_dir():
        return str(cwd_configs.resolve())

    raise FileNotFoundError(
        "Cannot find configs directory. "
        "Run from the project root, set KINEMATIC_CONFIG_DIR, "
        "or pass --config-dir."
    )


def compose_config(
    config_name: str,
    base_config_name: str | None = None,
    overrides: list[str] | None = None,
    config_dir: str | None = None,
) -> DictConfig:
    """Load and merge YAML configs via Hydra Compose API + OmegaConf.

    Parameters
    ----------
    config_name : Name of the phase config (without .yaml).
    base_config_name : Optional base config name to merge under.
    overrides : CLI overrides in dotlist form (e.g. ["lr=2e-4"]).
    config_dir : Explicit config directory path.

    Returns
    -------
    Merged DictConfig ready for training.
    """
    resolved_dir = _resolve_config_dir(config_dir)
    overrides = overrides or []

    # Clear any previous Hydra state (safe for repeated calls)
    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=resolved_dir, version_base=None):
        if base_config_name:
            # Load base and phase separately, merge with OmegaConf.
            # Disable struct flag so phase keys not in base are accepted.
            base_cfg = compose(config_name=base_config_name)
            phase_cfg = compose(config_name=config_name)
            OmegaConf.set_struct(base_cfg, False)
            OmegaConf.set_struct(phase_cfg, False)
            cfg = OmegaConf.merge(base_cfg, phase_cfg)
        else:
            cfg = compose(config_name=config_name)
            OmegaConf.set_struct(cfg, False)

    # Apply CLI overrides last (highest priority)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg
