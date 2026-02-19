"""Hierarchical sampling: coarse forecast + fine interpolation.

Generates long trajectories in two stages:
  Stage 1: Coarse auto-regressive forecasting at large dt
  Stage 2: Fine interpolation between coarse anchor frames
"""

from __future__ import annotations

from typing import Any

import torch

from kinematic.inference.sampler import EDMSampler


class HierarchicalGenerator:
    """Generate long trajectories using coarse forecasting + fine interpolation.

    Stage 1: Coarse auto-regressive forecasting at large dt.
    Stage 2: Fine interpolation between coarse anchors.

    Parameters
    ----------
    sampler : EDMSampler
        Configured EDM sampler wrapping the model.
    coarse_dt_ns : float
        Coarse time step in nanoseconds.
    fine_dt_ns : float
        Fine time step in nanoseconds.
    generation_window : int
        Number of frames to generate per AR block.
    history_window : int
        Number of conditioning frames from history.
    """

    def __init__(
        self,
        sampler: EDMSampler,
        coarse_dt_ns: float = 5.0,
        fine_dt_ns: float = 0.1,
        generation_window: int = 40,
        history_window: int = 10,
    ) -> None:
        self.sampler = sampler
        self.coarse_dt_ns = coarse_dt_ns
        self.fine_dt_ns = fine_dt_ns
        self.generation_window = generation_window
        self.history_window = history_window

    @torch.no_grad()
    def generate(
        self,
        initial_structure: torch.Tensor,
        total_time_ns: float,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        s_inputs: torch.Tensor,
        feats: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate trajectory from initial structure.

        Parameters
        ----------
        initial_structure : (M, 3)
            Starting atom coordinates.
        total_time_ns : float
            Total simulation time in nanoseconds.
        s_trunk, z_trunk, s_inputs :
            Precomputed trunk embeddings (unbatched).
        feats : dict
            Atom-level features.

        Returns
        -------
        trajectory : (N_frames, M, 3)
            Full trajectory at fine resolution.
        timestamps : (N_frames,)
            Timestamps in nanoseconds.
        """
        device = initial_structure.device

        # === Stage 1: Coarse forecasting ===
        coarse_times = torch.arange(
            0, total_time_ns + self.coarse_dt_ns, self.coarse_dt_ns,
            device=device,
        )
        coarse_traj: list[torch.Tensor] = [initial_structure]

        idx = 0
        while idx < len(coarse_times) - 1:
            n_gen = min(self.generation_window, len(coarse_times) - 1 - idx)
            target_times = coarse_times[idx + 1 : idx + 1 + n_gen]

            # History context (sliding window)
            hist_start = max(0, len(coarse_traj) - self.history_window)
            history = torch.stack(coarse_traj[hist_start:])
            history_times = coarse_times[hist_start : idx + 1]

            # Sample initial noise for targets
            x_init = (
                torch.randn(n_gen, *initial_structure.shape, device=device)
                * self.sampler.sigma_max
            )

            x_gen = self.sampler.sample(
                x_cond=history,
                t_cond=history_times,
                x_init=x_init,
                t_target=target_times,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                s_inputs=s_inputs,
                feats=feats,
                mode="forecast",
            )

            for i in range(n_gen):
                coarse_traj.append(x_gen[i])
            idx += n_gen

        coarse_traj_t = torch.stack(coarse_traj)  # (N_coarse, M, 3)

        # === Stage 2: Fine interpolation ===
        n_interp = max(1, int(self.coarse_dt_ns / self.fine_dt_ns) - 1)
        fine_traj: list[torch.Tensor] = [coarse_traj_t[0]]
        fine_times: list[torch.Tensor] = [coarse_times[0:1]]

        for i in range(len(coarse_traj_t) - 1):
            t0 = coarse_times[i]
            t1 = coarse_times[i + 1]
            t_interp = torch.linspace(
                t0.item(), t1.item(), n_interp + 2, device=device,
            )[1:-1]

            anchors = torch.stack([coarse_traj_t[i], coarse_traj_t[i + 1]])
            anchor_times = torch.tensor([t0.item(), t1.item()], device=device)

            x_init = (
                torch.randn(n_interp, *initial_structure.shape, device=device)
                * self.sampler.sigma_max
            )

            x_interp = self.sampler.sample(
                x_cond=anchors,
                t_cond=anchor_times,
                x_init=x_init,
                t_target=t_interp,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                s_inputs=s_inputs,
                feats=feats,
                mode="interpolate",
            )

            for j in range(n_interp):
                fine_traj.append(x_interp[j])
                fine_times.append(t_interp[j : j + 1])
            fine_traj.append(coarse_traj_t[i + 1])
            fine_times.append(torch.tensor([t1.item()], device=device))

        trajectory = torch.stack(fine_traj)
        timestamps = torch.cat(fine_times)
        return trajectory, timestamps
