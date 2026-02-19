"""Tests for EDM sampler schedule and endpoint behavior."""

from __future__ import annotations

import pytest
import torch

from kinematic.inference.sampler import EDMSampler


class _ConstantDenoiser(torch.nn.Module):
    """Dummy model returning a constant denoised prediction."""

    def __init__(self, pred_value: float = 7.0) -> None:
        super().__init__()
        self.pred_value = pred_value
        self.calls = 0
        self.sigmas: list[torch.Tensor] = []

    def forward(self, batch: dict, add_noise: bool = False) -> dict[str, torch.Tensor]:
        del add_noise
        self.calls += 1
        self.sigmas.append(batch["sigma"].detach().clone())
        return {"x_denoised": torch.full_like(batch["coords"], self.pred_value)}


class _LinearDenoiser(torch.nn.Module):
    """Dummy model with output dependent on current noisy input."""

    def __init__(self, scale: float = 0.2) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, batch: dict, add_noise: bool = False) -> dict[str, torch.Tensor]:
        del add_noise
        return {"x_denoised": batch["coords"] * self.scale}


def _make_dummy_inputs() -> dict[str, torch.Tensor]:
    n_cond, n_target, n_atoms = 2, 3, 5
    n_tokens, token_dim, token_z = 4, 8, 6
    return {
        "x_cond": torch.zeros(n_cond, n_atoms, 3),
        "t_cond": torch.tensor([0.0, 1.0]),
        "x_init": torch.randn(n_target, n_atoms, 3),
        "t_target": torch.tensor([2.0, 3.0, 4.0]),
        "s_trunk": torch.randn(n_tokens, token_dim),
        "z_trunk": torch.randn(n_tokens, n_tokens, token_z),
        "s_inputs": torch.randn(n_tokens, token_dim),
    }


def test_get_schedule_endpoint_and_length() -> None:
    sampler = EDMSampler(model=_ConstantDenoiser(), n_steps=6, sigma_min=0.01, sigma_max=2.0)
    sigmas = sampler.get_schedule()

    assert sigmas.shape == (7,)
    assert sigmas[-1].item() == pytest.approx(0.0, abs=1e-12)
    assert sigmas[-2].item() == pytest.approx(0.01, rel=1e-4)


def test_get_schedule_single_step() -> None:
    sampler = EDMSampler(model=_ConstantDenoiser(), n_steps=1, sigma_max=3.5)
    sigmas = sampler.get_schedule()
    assert torch.allclose(sigmas, torch.tensor([3.5, 0.0]))


def test_sample_runs_all_steps_and_reaches_zero_target() -> None:
    model = _ConstantDenoiser(pred_value=7.0)
    sampler = EDMSampler(
        model=model,
        sigma_min=0.05,
        sigma_max=1.0,
        rho=3.0,
        n_steps=4,
    )
    inputs = _make_dummy_inputs()

    out = sampler.sample(
        x_cond=inputs["x_cond"],
        t_cond=inputs["t_cond"],
        x_init=inputs["x_init"],
        t_target=inputs["t_target"],
        s_trunk=inputs["s_trunk"],
        z_trunk=inputs["z_trunk"],
        s_inputs=inputs["s_inputs"],
        feats={},
        mode="forecast",
    )

    # Regression for off-by-one bug: final step must run.
    assert model.calls == 4
    # Final update uses sigma_next=0, so output should equal model prediction exactly.
    assert torch.allclose(out, torch.full_like(out, 7.0), atol=1e-6)
    # Division denominator must stay positive for every model call.
    target_sigmas = [sigma[0, -len(inputs["t_target"]):] for sigma in model.sigmas]
    assert all(torch.all(s > 0) for s in target_sigmas)


def test_step_scale_changes_sampling_trajectory() -> None:
    inputs = _make_dummy_inputs()

    torch.manual_seed(123)
    sampler_no_churn = EDMSampler(
        model=_LinearDenoiser(scale=0.25),
        sigma_min=0.02,
        sigma_max=1.2,
        rho=3.0,
        n_steps=5,
        noise_scale=0.0,
        step_scale=0.0,
    )
    out_no_churn = sampler_no_churn.sample(
        x_cond=inputs["x_cond"],
        t_cond=inputs["t_cond"],
        x_init=inputs["x_init"],
        t_target=inputs["t_target"],
        s_trunk=inputs["s_trunk"],
        z_trunk=inputs["z_trunk"],
        s_inputs=inputs["s_inputs"],
        feats={},
        mode="forecast",
    )

    torch.manual_seed(123)
    sampler_with_churn = EDMSampler(
        model=_LinearDenoiser(scale=0.25),
        sigma_min=0.02,
        sigma_max=1.2,
        rho=3.0,
        n_steps=5,
        noise_scale=0.0,
        step_scale=1.5,
    )
    out_with_churn = sampler_with_churn.sample(
        x_cond=inputs["x_cond"],
        t_cond=inputs["t_cond"],
        x_init=inputs["x_init"],
        t_target=inputs["t_target"],
        s_trunk=inputs["s_trunk"],
        z_trunk=inputs["z_trunk"],
        s_inputs=inputs["s_inputs"],
        feats={},
        mode="forecast",
    )

    assert not torch.allclose(out_no_churn, out_with_churn)


def test_noise_scale_changes_sampling_when_churn_enabled() -> None:
    inputs = _make_dummy_inputs()

    torch.manual_seed(321)
    sampler_zero_noise = EDMSampler(
        model=_LinearDenoiser(scale=0.4),
        sigma_min=0.02,
        sigma_max=1.2,
        rho=3.0,
        n_steps=5,
        noise_scale=0.0,
        step_scale=1.5,
    )
    out_zero_noise = sampler_zero_noise.sample(
        x_cond=inputs["x_cond"],
        t_cond=inputs["t_cond"],
        x_init=inputs["x_init"],
        t_target=inputs["t_target"],
        s_trunk=inputs["s_trunk"],
        z_trunk=inputs["z_trunk"],
        s_inputs=inputs["s_inputs"],
        feats={},
        mode="forecast",
    )

    torch.manual_seed(321)
    sampler_with_noise = EDMSampler(
        model=_LinearDenoiser(scale=0.4),
        sigma_min=0.02,
        sigma_max=1.2,
        rho=3.0,
        n_steps=5,
        noise_scale=1.75,
        step_scale=1.5,
    )
    out_with_noise = sampler_with_noise.sample(
        x_cond=inputs["x_cond"],
        t_cond=inputs["t_cond"],
        x_init=inputs["x_init"],
        t_target=inputs["t_target"],
        s_trunk=inputs["s_trunk"],
        z_trunk=inputs["z_trunk"],
        s_inputs=inputs["s_inputs"],
        feats={},
        mode="forecast",
    )

    assert not torch.allclose(out_zero_noise, out_with_noise)
