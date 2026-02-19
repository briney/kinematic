"""Noise-as-masking sample construction.

Implements the per-frame noise assignment strategy:
conditioning frames get sigma=0 (clean), target frames get EDM
log-normal noise levels. Supports both forecasting and interpolation
tasks.
"""

from __future__ import annotations

import random

import torch


def assign_noise(
    n_frames: int,
    P_mean: float = -1.2,
    P_std: float = 1.5,
    sigma_data: float = 16.0,
    forecast_prob: float = 0.5,
    py_rng: random.Random | None = None,
    torch_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Assign per-frame noise levels using noise-as-masking.

    Parameters
    ----------
    n_frames : total number of frames T.
    P_mean : mean of the log-normal distribution for sigma.
    P_std : std of the log-normal distribution for sigma.
    sigma_data : EDM sigma_data (coordinate scale).
    forecast_prob : probability of choosing forecasting vs interpolation.
    py_rng : optional Python RNG for deterministic task sampling.
    torch_generator : optional torch Generator for deterministic sigma sampling.

    Returns
    -------
    sigma : (T,) float32 — per-frame noise level. 0.0 for conditioning frames.
    conditioning_mask : (T,) bool — True for conditioning (clean) frames.
    task : 'forecasting' or 'interpolation'.
    """
    if py_rng is None:
        py_rng = random

    # Sample noise levels from log-normal (EDM convention)
    log_sigma = torch.randn(n_frames, generator=torch_generator) * P_std + P_mean
    sigma = sigma_data * torch.exp(log_sigma)

    # Choose task type
    task = "forecasting" if py_rng.random() < forecast_prob else "interpolation"

    if task == "forecasting":
        # First frame is clean conditioning
        sigma[0] = 0.0
    else:
        # First and last frames are clean conditioning
        sigma[0] = 0.0
        sigma[-1] = 0.0

    conditioning_mask = sigma == 0.0
    return sigma, conditioning_mask, task
