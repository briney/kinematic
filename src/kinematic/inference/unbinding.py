"""Auto-regressive unbinding trajectory generation.

Uses the causal variant of the model (trained on DD-13M) to generate
multiple stochastic unbinding trajectories one frame at a time.
"""

from __future__ import annotations

from typing import Any

import torch

from kinematic.inference.sampler import EDMSampler


class UnbindingGenerator:
    """Auto-regressive unbinding trajectory generation.

    Uses the causal variant of the model (trained on DD-13M).
    Each frame is generated conditioned on all previous frames.

    Parameters
    ----------
    sampler : EDMSampler
        Configured EDM sampler wrapping the model.
    dt_ps : float
        Time step in picoseconds.
    """

    def __init__(
        self,
        sampler: EDMSampler,
        dt_ps: float = 10.0,
    ) -> None:
        self.sampler = sampler
        self.dt_ps = dt_ps

    @torch.no_grad()
    def generate(
        self,
        complex_structure: torch.Tensor,
        total_ps: float,
        n_trajectories: int,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        s_inputs: torch.Tensor,
        feats: dict[str, Any],
    ) -> torch.Tensor:
        """Generate multiple unbinding trajectories.

        Parameters
        ----------
        complex_structure : (M, 3)
            Starting bound complex coordinates.
        total_ps : float
            Total simulation time in picoseconds.
        n_trajectories : int
            Number of independent trajectories to generate.
        s_trunk, z_trunk, s_inputs :
            Precomputed trunk embeddings (unbatched).
        feats : dict
            Atom-level features.

        Returns
        -------
        trajectories : (n_trajectories, n_frames, M, 3)
        """
        device = complex_structure.device
        n_steps = int(total_ps / self.dt_ps)
        all_trajs: list[torch.Tensor] = []

        for _traj_idx in range(n_trajectories):
            trajectory: list[torch.Tensor] = [complex_structure]

            for step in range(n_steps):
                context = torch.stack(trajectory)
                # Convert picosecond frame indices to nanoseconds
                context_times = (
                    torch.arange(len(context), device=device, dtype=torch.float32)
                    * self.dt_ps
                    / 1000.0
                )
                target_time = torch.tensor(
                    [len(context) * self.dt_ps / 1000.0], device=device,
                )

                x_init = (
                    torch.randn(1, *complex_structure.shape, device=device)
                    * self.sampler.sigma_max
                )

                x_next = self.sampler.sample(
                    x_cond=context,
                    t_cond=context_times,
                    x_init=x_init,
                    t_target=target_time,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    s_inputs=s_inputs,
                    feats=feats,
                    mode="forecast",
                )
                trajectory.append(x_next.squeeze(0))

            all_trajs.append(torch.stack(trajectory))

        return torch.stack(all_trajs)
