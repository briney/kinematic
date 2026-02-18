"""Trunk embedding I/O.

Handles saving and loading of precomputed Boltz-2 trunk embeddings
(s_inputs, s_trunk, z_trunk) stored as compressed float16 .npz files.
rel_pos_enc is NOT cached (recomputed at runtime per trunk cache rule).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def save_trunk_embeddings(
    system_id: str,
    s_inputs: torch.Tensor | np.ndarray,
    s_trunk: torch.Tensor | np.ndarray,
    z_trunk: torch.Tensor | np.ndarray,
    output_dir: str | Path,
) -> Path:
    """Save precomputed trunk embeddings as compressed float16 .npz.

    Parameters
    ----------
    system_id : unique system identifier.
    s_inputs : (N_tokens, 384) raw input embeddings.
    s_trunk : (N_tokens, 384) refined single representation.
    z_trunk : (N_tokens, N_tokens, 128) refined pair representation.
    output_dir : directory to write the cache file.

    Returns
    -------
    Path to the saved .npz file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _to_f16_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            return x.half().numpy()
        return x.astype(np.float16)

    out_path = output_dir / f"{system_id}_trunk.npz"
    np.savez_compressed(
        out_path,
        s_inputs=_to_f16_numpy(s_inputs),
        s_trunk=_to_f16_numpy(s_trunk),
        z_trunk=_to_f16_numpy(z_trunk),
    )
    logger.debug("Saved trunk cache: %s", out_path)
    return out_path


def load_trunk_embeddings(
    system_id: str,
    cache_dir: str | Path,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Load precomputed trunk embeddings from cache.

    Parameters
    ----------
    system_id : unique system identifier.
    cache_dir : directory containing trunk .npz files.
    dtype : target dtype for the returned tensors.

    Returns
    -------
    Dict with keys 's_inputs', 's_trunk', 'z_trunk' as torch tensors.
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{system_id}_trunk.npz"

    data = np.load(path)
    return {
        "s_inputs": torch.from_numpy(data["s_inputs"].astype(np.float32)).to(dtype),
        "s_trunk": torch.from_numpy(data["s_trunk"].astype(np.float32)).to(dtype),
        "z_trunk": torch.from_numpy(data["z_trunk"].astype(np.float32)).to(dtype),
    }
