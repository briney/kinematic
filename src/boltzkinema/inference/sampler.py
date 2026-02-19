"""EDM sampling loop for trajectory generation.

Implements the Karras et al. (2022) stochastic sampler adapted for
multi-frame denoising. Supports both 'forecast' (conditioning frames at
the start) and 'interpolate' (conditioning frames as endpoints) modes.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class EDMSampler:
    """EDM stochastic sampler for multi-frame denoising.

    Uses Karras et al. (2022) schedule with configurable parameters.

    Parameters
    ----------
    model : nn.Module
        BoltzKinema model instance.
    sigma_min : float
        Minimum noise level (absolute, coordinate units).
    sigma_max : float
        Maximum noise level (absolute, coordinate units).
    sigma_data : float
        Data standard deviation for EDM preconditioning.
    rho : float
        Schedule curvature parameter.
    n_steps : int
        Number of denoising steps.
    noise_scale : float
        Stochastic noise injection scale (S_noise in Karras).
    step_scale : float
        Step size scaling (S_churn in Karras).
    """

    def __init__(
        self,
        model: nn.Module,
        sigma_min: float = 0.0001,
        sigma_max: float = 160.0,
        sigma_data: float = 16.0,
        rho: float = 7.0,
        n_steps: int = 20,
        noise_scale: float = 1.75,
        step_scale: float = 1.5,
    ) -> None:
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.n_steps = n_steps
        self.noise_scale = noise_scale
        self.step_scale = step_scale

    def _get_churn_gamma(self, sigma_k: float) -> float:
        """Compute Karras churn factor for a denoising step."""
        del sigma_k  # sigma range gate not exposed; churn applies across schedule.
        if self.step_scale <= 0 or self.n_steps <= 0:
            return 0.0
        return min(self.step_scale / self.n_steps, math.sqrt(2.0) - 1.0)

    def get_schedule(self, n_steps: int | None = None) -> torch.Tensor:
        """Karras et al. EDM noise schedule.

        ``sigma_min``/``sigma_max`` are absolute sigma values in coordinate
        units (do not multiply by ``sigma_data`` again).

        Parameters
        ----------
        n_steps : int, optional
            Number of denoising steps. Defaults to ``self.n_steps``.

        Returns
        -------
        sigmas : (n_steps + 1,) tensor ending with 0.
        """
        if n_steps is None:
            n_steps = self.n_steps
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if n_steps == 1:
            return torch.tensor([self.sigma_max, 0.0], dtype=torch.float32)

        inv_rho = 1.0 / self.rho
        steps = torch.arange(n_steps)
        sigmas = (
            self.sigma_max ** inv_rho
            + steps / (n_steps - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)
        ) ** self.rho
        sigmas = F.pad(sigmas, (0, 1), value=0.0)

        # Sanity checks
        assert torch.isclose(sigmas[0], torch.tensor(self.sigma_max), rtol=1e-4, atol=1e-6)
        assert torch.isclose(sigmas[-2], torch.tensor(self.sigma_min), rtol=1e-4, atol=1e-6)
        return sigmas

    def _build_batch(
        self,
        x_cond: torch.Tensor,
        t_cond: torch.Tensor,
        x_target: torch.Tensor,
        t_target: torch.Tensor,
        sigma_k: float,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        s_inputs: torch.Tensor,
        feats: dict[str, Any],
        mode: str,
    ) -> dict[str, Any]:
        """Assemble a model-ready batch dict for one denoising step.

        Parameters
        ----------
        x_cond : (n_cond, M, 3) clean conditioning coordinates.
        t_cond : (n_cond,) conditioning timestamps in ns.
        x_target : (n_target, M, 3) current noisy target coordinates.
        t_target : (n_target,) target timestamps in ns.
        sigma_k : current noise level.
        s_trunk, z_trunk, s_inputs : precomputed trunk features.
        feats : atom-level feature dict.
        mode : 'forecast' or 'interpolate'.
        """
        device = x_cond.device

        if mode == "forecast":
            x_full = torch.cat([x_cond, x_target], dim=0).unsqueeze(0)
            sigma_full = torch.cat([
                torch.zeros(len(x_cond), device=device),
                torch.full((len(x_target),), sigma_k, device=device),
            ]).unsqueeze(0)
            t_full = torch.cat([t_cond, t_target]).unsqueeze(0)
            cond_mask = torch.cat([
                torch.ones(len(x_cond), dtype=torch.bool, device=device),
                torch.zeros(len(x_target), dtype=torch.bool, device=device),
            ]).unsqueeze(0)
        else:  # interpolate
            x_full = torch.cat([x_cond[0:1], x_target, x_cond[-1:]]).unsqueeze(0)
            sigma_full = torch.cat([
                torch.zeros(1, device=device),
                torch.full((len(x_target),), sigma_k, device=device),
                torch.zeros(1, device=device),
            ]).unsqueeze(0)
            t_full = torch.cat([t_cond[0:1], t_target, t_cond[-1:]]).unsqueeze(0)
            cond_mask = torch.cat([
                torch.ones(1, dtype=torch.bool, device=device),
                torch.zeros(len(x_target), dtype=torch.bool, device=device),
                torch.ones(1, dtype=torch.bool, device=device),
            ]).unsqueeze(0)

        batch = {
            "coords": x_full,
            "timestamps": t_full,
            "sigma": sigma_full,
            "conditioning_mask": cond_mask,
            "s_trunk": s_trunk.unsqueeze(0) if s_trunk.dim() == 2 else s_trunk,
            "z_trunk": z_trunk.unsqueeze(0) if z_trunk.dim() == 3 else z_trunk,
            "s_inputs": s_inputs.unsqueeze(0) if s_inputs.dim() == 2 else s_inputs,
            "feats": feats,
        }

        # Copy additional required keys from feats if present
        for key in ("atom_pad_mask", "token_pad_mask", "atom_to_token"):
            if key in feats:
                batch[key] = feats[key]

        return batch

    @torch.no_grad()
    def sample(
        self,
        x_cond: torch.Tensor,
        t_cond: torch.Tensor,
        x_init: torch.Tensor,
        t_target: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        s_inputs: torch.Tensor,
        feats: dict[str, Any],
        mode: str = "forecast",
    ) -> torch.Tensor:
        """Denoise target frames given conditioning frames.

        Parameters
        ----------
        x_cond : (n_cond, M, 3)
            Clean conditioning frame coordinates.
        t_cond : (n_cond,)
            Conditioning timestamps in ns.
        x_init : (n_target, M, 3)
            Initial noisy target coordinates (typically sampled from
            ``sigma_max * N(0,1)``).
        t_target : (n_target,)
            Target timestamps in ns.
        s_trunk, z_trunk, s_inputs :
            Precomputed trunk features (unbatched).
        feats : dict
            Atom-level features.
        mode : str
            ``'forecast'`` or ``'interpolate'``.

        Returns
        -------
        x_denoised : (n_target, M, 3)
            Denoised target coordinates.
        """
        sigmas = self.get_schedule().to(x_cond.device)
        x_target = x_init

        for k in range(self.n_steps):
            sigma_k = sigmas[k].item()
            sigma_next = sigmas[k + 1].item()
            if sigma_k <= 0:
                raise RuntimeError(
                    "Encountered non-positive sigma before final denoising update."
                )

            gamma = self._get_churn_gamma(sigma_k)
            sigma_hat = sigma_k * (1.0 + gamma)
            x_hat = x_target
            if gamma > 0:
                noise = torch.randn_like(x_target) * self.noise_scale
                noise_scale = math.sqrt(max(sigma_hat**2 - sigma_k**2, 0.0))
                x_hat = x_hat + noise * noise_scale

            # Build batch and run model
            batch = self._build_batch(
                x_cond, t_cond, x_hat, t_target,
                sigma_hat, s_trunk, z_trunk, s_inputs, feats, mode,
            )
            output = self.model(batch, add_noise=False)
            x_denoised_full = output["x_denoised"].squeeze(0)

            # Extract target predictions
            if mode == "forecast":
                x_pred = x_denoised_full[len(x_cond):]
            else:
                x_pred = x_denoised_full[1:-1]

            # EDM deterministic update step
            d = (x_hat - x_pred) / sigma_hat
            x_target = x_hat + (sigma_next - sigma_hat) * d

        return x_target
