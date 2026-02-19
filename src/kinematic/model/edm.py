"""Per-frame EDM preconditioning."""

from __future__ import annotations

import torch
import torch.nn as nn


class PerFrameEDM(nn.Module):
    """Per-frame EDM preconditioning where each frame has its own sigma.

    For conditioning frames (sigma=0):
      - c_skip(0) = 1, c_out(0) = 0  →  output = input (clean coords pass through)
      - c_noise(0) is clamped to avoid log(0)

    All broadcasting: sigma (B, T) → (B, T, 1, 1) for (B, T, M, 3) coords.
    """

    def __init__(self, sigma_data: float = 16.0) -> None:
        super().__init__()
        self.sigma_data = sigma_data

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute noise conditioning value. Clamped to handle sigma=0."""
        return torch.log((sigma / self.sigma_data).clamp(min=1e-20)) * 0.25

    def scale_input(
        self, x: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Scale noisy coordinates by c_in.

        Parameters
        ----------
        x : (B, T, M, 3)
        sigma : (B, T)

        Returns
        -------
        (B, T, M, 3)
        """
        return self.c_in(sigma[..., None, None]) * x

    def combine_output(
        self,
        x_noisy: torch.Tensor,
        f_out: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Combine noisy input with network output via skip/out weighting.

        Parameters
        ----------
        x_noisy : (B, T, M, 3)
        f_out : (B, T, M, 3)
        sigma : (B, T)

        Returns
        -------
        (B, T, M, 3) — denoised coordinates.
        """
        s = sigma[..., None, None]
        return self.c_skip(s) * x_noisy + self.c_out(s) * f_out

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """EDM loss weighting: (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2."""
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
