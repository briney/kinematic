"""Boltz-2 weight extraction utilities."""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


# Prefix mapping rules: (boltz2_prefix, kinematic_prefix)
# Order matters — first match wins.
_PREFIX_RULES: list[tuple[str, str]] = [
    # DiffusionConditioning: same name in both models
    ("diffusion_conditioning.", "diffusion_conditioning."),
    # RelativePositionEncoder: renamed
    ("rel_pos.", "rel_pos_encoder."),
    # Score model: strip structure_module wrapper
    ("structure_module.score_model.", "score_model."),
]

# Boltz-2 prefixes we want to load (everything else is ignored)
_LOADABLE_PREFIXES: tuple[str, ...] = (
    "diffusion_conditioning.",
    "rel_pos.",
    "structure_module.score_model.",
)

# Temporal components that should NOT be loaded (they're new in Kinematic)
_TEMPORAL_PATTERNS: tuple[str, ...] = (
    "temporal_attn.",
    "temporal_layers.",
)


def _map_key(boltz2_key: str) -> str | None:
    """Map a Boltz-2 state_dict key to a Kinematic key.

    Returns None if the key should be skipped.
    """
    # Only load keys from relevant modules
    if not boltz2_key.startswith(_LOADABLE_PREFIXES):
        return None

    for src_prefix, dst_prefix in _PREFIX_RULES:
        if boltz2_key.startswith(src_prefix):
            return dst_prefix + boltz2_key[len(src_prefix):]

    return None


def load_boltz2_weights(
    model: torch.nn.Module,
    checkpoint_path: str,
    strict: bool = False,
) -> dict[str, Any]:
    """Load Boltz-2 pretrained weights into a Kinematic model.

    Maps Boltz-2 checkpoint keys to Kinematic parameter names and loads
    all spatial (non-temporal) weights. Temporal components remain at their
    randomly initialized values (zero output projections).

    Parameters
    ----------
    model : Kinematic
        The Kinematic model to load weights into.
    checkpoint_path : str
        Path to a Boltz-2 checkpoint (.ckpt or .pt file).
    strict : bool
        If True, raise an error for missing keys. Default False since
        temporal components are expected to be missing from Boltz-2.

    Returns
    -------
    dict with:
      - matched : list of (boltz2_key, kinema_key) pairs that were loaded
      - missing : list of kinema keys not found in checkpoint
      - skipped : list of boltz2 keys that were not mapped
      - temporal : list of temporal parameter names (verified zero-init)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Handle Lightning checkpoint format
    if "state_dict" in checkpoint:
        src_state = checkpoint["state_dict"]
    else:
        src_state = checkpoint

    # Build mapped state dict
    mapped_state: dict[str, torch.Tensor] = {}
    matched: list[tuple[str, str]] = []
    skipped: list[str] = []

    for boltz2_key, param in src_state.items():
        kinema_key = _map_key(boltz2_key)
        if kinema_key is None:
            skipped.append(boltz2_key)
            continue
        mapped_state[kinema_key] = param
        matched.append((boltz2_key, kinema_key))

    # Load into model
    model_state = model.state_dict()
    missing: list[str] = []

    for kinema_key in model_state:
        # Skip temporal components
        if any(pat in kinema_key for pat in _TEMPORAL_PATTERNS):
            continue
        # Skip EDM (pure math, no learned weights)
        if kinema_key.startswith("edm."):
            continue
        if kinema_key not in mapped_state:
            missing.append(kinema_key)

    # Perform the load
    load_result = model.load_state_dict(mapped_state, strict=False)

    n_loaded = len(matched)
    n_missing = len(missing)
    n_skipped = len(skipped)
    n_unexpected = len(load_result.unexpected_keys)

    logger.info(
        "Loaded %d Boltz-2 params, %d missing, %d skipped, %d unexpected",
        n_loaded, n_missing, n_skipped, n_unexpected,
    )

    if missing and strict:
        raise RuntimeError(
            f"Missing {len(missing)} keys in Kinematic model: {missing[:10]}..."
        )

    if missing:
        logger.warning("Missing non-temporal keys: %s", missing[:20])

    # Verify temporal components are zero-initialized
    temporal_params: list[str] = []
    for name, param in model.named_parameters():
        if any(pat in name for pat in _TEMPORAL_PATTERNS):
            temporal_params.append(name)
            if "out_proj.weight" in name:
                assert (param == 0).all(), (
                    f"Temporal output projection {name} should be zero-initialized "
                    f"but has non-zero values"
                )

    logger.info(
        "Verified %d temporal parameters (output projections are zero-initialized)",
        len(temporal_params),
    )

    return {
        "matched": matched,
        "missing": missing,
        "skipped": skipped,
        "temporal": temporal_params,
    }


def verify_temporal_zero_init(model: torch.nn.Module) -> bool:
    """Verify that all temporal output projections are zero-initialized.

    This is critical: temporal attention must start as identity so that
    the model initially behaves like Boltz-2.

    Parameters
    ----------
    model : Kinematic
        The model to verify.

    Returns
    -------
    bool
        True if all temporal output projections are zeros.
    """
    all_zero = True
    for name, param in model.named_parameters():
        if any(pat in name for pat in _TEMPORAL_PATTERNS):
            if "out_proj" in name and "weight" in name:
                if not (param == 0).all():
                    logger.error("Non-zero temporal output projection: %s", name)
                    all_zero = False
    return all_zero
