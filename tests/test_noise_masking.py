"""Tests for noise-as-masking sample construction."""

from __future__ import annotations

import random

import torch

from kinematic.data.noise_masking import assign_noise


def test_assign_noise_forecasting_masks_only_first_frame() -> None:
    sigma, mask, task = assign_noise(
        n_frames=5,
        P_mean=0.0,
        P_std=0.0,
        sigma_data=2.0,
        forecast_prob=1.0,
    )

    assert task == "forecasting"
    assert sigma.tolist() == [0.0, 2.0, 2.0, 2.0, 2.0]
    assert mask.tolist() == [True, False, False, False, False]


def test_assign_noise_interpolation_masks_first_and_last_frames() -> None:
    sigma, mask, task = assign_noise(
        n_frames=6,
        P_mean=0.0,
        P_std=0.0,
        sigma_data=3.0,
        forecast_prob=0.0,
    )

    assert task == "interpolation"
    assert sigma.tolist() == [0.0, 3.0, 3.0, 3.0, 3.0, 0.0]
    assert mask.tolist() == [True, False, False, False, False, True]


def test_assign_noise_is_deterministic_with_seeded_generators() -> None:
    py_rng_a = random.Random(7)
    py_rng_b = random.Random(7)
    torch_gen_a = torch.Generator().manual_seed(11)
    torch_gen_b = torch.Generator().manual_seed(11)

    sigma_a, mask_a, task_a = assign_noise(
        n_frames=8,
        py_rng=py_rng_a,
        torch_generator=torch_gen_a,
    )
    sigma_b, mask_b, task_b = assign_noise(
        n_frames=8,
        py_rng=py_rng_b,
        torch_generator=torch_gen_b,
    )

    assert task_a == task_b
    assert torch.equal(sigma_a, sigma_b)
    assert torch.equal(mask_a, mask_b)
