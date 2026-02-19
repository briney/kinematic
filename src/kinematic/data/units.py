"""Coordinate-unit inference helpers used by preprocessing scripts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _normalize_unit_label(value: Any) -> str | None:
    """Normalize metadata unit labels to ``'nm'`` or ``'angstrom'``."""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore")
    else:
        text = str(value)

    label = text.strip().lower().replace(" ", "").replace("-", "")
    if label in {"nm", "nanometer", "nanometers", "nanometre", "nanometres"}:
        return "nm"
    if label in {"a", "angstrom", "angstroms", "ang"}:
        return "angstrom"
    return None


def extract_length_unit_from_metadata(
    metadata: dict[str, Any] | None,
) -> tuple[str, str] | None:
    """Return ``(unit, key)`` if a known unit label exists in metadata."""
    if not metadata:
        return None

    preferred_keys = (
        "unit",
        "units",
        "coords_unit",
        "coordinate_unit",
        "length_unit",
    )

    # Prefer canonical unit key names when present.
    for key in preferred_keys:
        if key not in metadata:
            continue
        unit = _normalize_unit_label(metadata[key])
        if unit is not None:
            return unit, key

    # Fallback: scan all metadata values.
    for key, value in metadata.items():
        unit = _normalize_unit_label(value)
        if unit is not None:
            return unit, key
    return None


def _nearest_neighbor_median(coords: np.ndarray, max_atoms: int = 1024) -> float:
    """Median nearest-neighbor distance from a single frame."""
    frame = np.asarray(coords[0], dtype=np.float64)
    n_atoms = frame.shape[0]
    if n_atoms < 2:
        return 0.0

    if n_atoms > max_atoms:
        idx = np.linspace(0, n_atoms - 1, max_atoms, dtype=np.int64)
        frame = frame[idx]

    dist = np.linalg.norm(frame[:, None, :] - frame[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    nn = np.min(dist, axis=1)
    finite_nn = nn[np.isfinite(nn)]
    if finite_nn.size == 0:
        return 0.0
    return float(np.median(finite_nn))


def _radius_p95(coords: np.ndarray) -> float:
    """95th-percentile radial extent from per-frame centroids."""
    centers = np.nanmean(coords, axis=1, keepdims=True)
    radii = np.linalg.norm(coords - centers, axis=-1)
    return float(np.nanpercentile(radii, 95.0))


def infer_coordinate_unit(
    coords: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, float, str]:
    """Infer whether coordinates are in nanometers or Angstrom.

    Returns
    -------
    unit : ``'nm'`` or ``'angstrom'``
    confidence : float in ``[0, 1]``
    reason : short provenance/debug string
    """
    coords = np.asarray(coords)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(
            f"Expected coords shape (n_frames, n_atoms, 3), got {coords.shape}"
        )

    meta_match = extract_length_unit_from_metadata(metadata)
    if meta_match is not None:
        unit, key = meta_match
        return unit, 1.0, f"metadata:{key}"

    nn_median = _nearest_neighbor_median(coords)
    radius_p95 = _radius_p95(coords)

    nm_score = 0
    angstrom_score = 0

    # Bond-length proxy from nearest neighbors.
    if nn_median > 0:
        if nn_median <= 0.35:
            nm_score += 2
        elif nn_median >= 0.80:
            angstrom_score += 2

    # Coarse global extent proxy.
    if radius_p95 <= 8.0:
        nm_score += 1
    elif radius_p95 >= 25.0:
        angstrom_score += 1

    if nm_score > angstrom_score:
        margin = nm_score - angstrom_score
        confidence = min(0.95, 0.65 + 0.10 * margin)
        return (
            "nm",
            confidence,
            f"heuristic:nn={nn_median:.3f},radius95={radius_p95:.3f}",
        )

    if angstrom_score > nm_score:
        margin = angstrom_score - nm_score
        confidence = min(0.95, 0.65 + 0.10 * margin)
        return (
            "angstrom",
            confidence,
            f"heuristic:nn={nn_median:.3f},radius95={radius_p95:.3f}",
        )

    # Ambiguous: choose the bond-length scale closer to expected medians.
    if nn_median > 0:
        dist_nm = abs(math.log(nn_median) - math.log(0.14))
        dist_a = abs(math.log(nn_median) - math.log(1.40))
        if dist_nm < dist_a:
            return (
                "nm",
                0.55,
                f"ambiguous_nn_tiebreak:nn={nn_median:.3f},radius95={radius_p95:.3f}",
            )
        return (
            "angstrom",
            0.55,
            f"ambiguous_nn_tiebreak:nn={nn_median:.3f},radius95={radius_p95:.3f}",
        )

    return "angstrom", 0.50, f"ambiguous_default:radius95={radius_p95:.3f}"

