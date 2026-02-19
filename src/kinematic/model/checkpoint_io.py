"""Checkpoint loading utilities with suffix-based dispatch."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

import torch

_STEP_PLACEHOLDER_RE = re.compile(r"^step_[xX]+$")
_STEP_DIR_RE = re.compile(r"^step_(\d+)$")


def find_model_weights_file(checkpoint_path: str | Path) -> Path | None:
    """Resolve model weight file from a checkpoint path.

    If ``checkpoint_path`` is a file, return it.
    If it is a directory, search for Accelerate-style weight files.
    """
    path = Path(checkpoint_path)
    if path.is_file():
        return path
    if not path.is_dir():
        return None

    candidates = sorted(path.glob("pytorch_model*.bin")) + sorted(
        path.glob("model*.safetensors")
    )
    if not candidates:
        return None
    return candidates[0]


def has_unresolved_step_placeholder(checkpoint_path: str | Path) -> bool:
    """Return True if ``checkpoint_path`` contains a ``step_XXXXX``-style segment."""
    path = Path(os.path.expanduser(str(checkpoint_path)))
    return any(_STEP_PLACEHOLDER_RE.fullmatch(part) for part in path.parts)


def find_latest_step_checkpoint(parent_dir: str | Path) -> Path | None:
    """Find the latest ``step_<int>`` checkpoint directory under ``parent_dir``."""
    parent = Path(parent_dir)
    if not parent.is_dir():
        return None

    best: tuple[int, Path] | None = None
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        match = _STEP_DIR_RE.fullmatch(child.name)
        if match is None:
            continue
        step = int(match.group(1))
        if best is None or step > best[0]:
            best = (step, child)
    return None if best is None else best[1]


def resolve_checkpoint_path(
    checkpoint_path: str | Path,
    *,
    auto_resolve_latest: bool = True,
) -> Path:
    """Resolve checkpoint path, optionally replacing ``step_XXXXX`` with latest step dir.

    Raises
    ------
    ValueError
        If an unresolved placeholder is present and cannot be resolved.
    """
    path = Path(os.path.expanduser(str(checkpoint_path)))
    parts = path.parts

    placeholder_idx: int | None = None
    for idx, part in enumerate(parts):
        if _STEP_PLACEHOLDER_RE.fullmatch(part):
            placeholder_idx = idx
            break

    if placeholder_idx is None:
        return path

    parent = Path(*parts[:placeholder_idx]) if placeholder_idx > 0 else Path(".")
    suffix_parts = parts[placeholder_idx + 1:]

    if auto_resolve_latest:
        latest = find_latest_step_checkpoint(parent)
        if latest is not None:
            return latest.joinpath(*suffix_parts)

    raise ValueError(
        f"Checkpoint path contains unresolved placeholder segment: {checkpoint_path!s}. "
        f"Provide an explicit checkpoint path (e.g., {parent / 'step_12345'}) "
        "or create step_* directories so latest-step auto-resolution can run."
    )


def load_checkpoint_file(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    weights_only: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint file by extension.

    ``.safetensors`` files are loaded with ``safetensors.torch.load_file``.
    All other files are loaded with ``torch.load``.
    """
    path = Path(checkpoint_path)
    suffix = path.suffix.lower()

    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Checkpoint is a .safetensors file but `safetensors` is not installed. "
                "Install it with `pip install safetensors`."
            ) from exc

        device = str(map_location) if isinstance(map_location, torch.device) else map_location
        return load_file(str(path), device=device)

    state = torch.load(path, map_location=map_location, weights_only=weights_only)
    if not isinstance(state, dict):
        raise TypeError(
            f"Expected checkpoint to deserialize to a dict, got {type(state).__name__}"
        )
    return state


def load_model_state_dict(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load and normalize a model ``state_dict`` from checkpoint path."""
    state = load_checkpoint_file(checkpoint_path, map_location=map_location, weights_only=True)

    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise TypeError(
            f"Expected model state_dict to be a dict, got {type(state).__name__}"
        )
    return state
