"""Loss functions for Kinematic training."""

from __future__ import annotations

import torch
import torch.nn as nn


class TrajectoryLoss(nn.Module):
    """Combined loss for Kinematic training.

    Components:
        1. L_struct: EDM-weighted structure reconstruction (MSE + bond + smooth lDDT)
        2. L_flex: Flexibility loss (RMSF + pairwise distance std + local)
        3. L_center: Ligand geometric center loss (unbinding only)

    Loss masking policy:
        - Conditioning frames (sigma=0) do not contribute to supervised terms.
        - Unobserved atoms/residues do not contribute to supervised terms.

    MD training:     L = L_struct + alpha_bond * L_bond + L_lddt + beta_flex * L_flex
    Unbinding:       L = L_struct + alpha_bond * L_bond + L_lddt + beta_center * L_center

    Parameters
    ----------
    sigma_data : EDM sigma_data parameter.
    alpha_bond : Weight for bond length loss.
    beta_flex : Weight for flexibility loss.
    beta_abs : Weight for absolute RMSF within flexibility loss.
    beta_rel_g : Weight for global relative flexibility.
    beta_rel_l : Weight for local relative flexibility.
    beta_center : Weight for ligand center loss (unbinding only).
    mol_weights : Per-molecule-type loss weights.
    """

    # Molecule type codes (matching dataset convention)
    MOL_PROTEIN = 0
    MOL_DNA = 1
    MOL_RNA = 2
    MOL_LIGAND = 3

    def __init__(
        self,
        sigma_data: float = 16.0,
        alpha_bond: float = 1.0,
        beta_flex: float = 1.0,
        beta_abs: float = 1.0,
        beta_rel_g: float = 4.0,
        beta_rel_l: float = 4.0,
        beta_center: float = 1.0,
        mol_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        self.alpha_bond = alpha_bond
        self.beta_flex = beta_flex
        self.beta_abs = beta_abs
        self.beta_rel_g = beta_rel_g
        self.beta_rel_l = beta_rel_l
        self.beta_center = beta_center

        defaults = {"protein": 1.0, "dna": 5.0, "rna": 5.0, "ligand": 10.0}
        if mol_weights is not None:
            defaults.update(mol_weights)
        self._mol_weight_map = defaults

        # Register as buffer so they move with .to(device)
        self.register_buffer(
            "_mol_weight_values",
            torch.tensor(
                [
                    defaults["protein"],  # 0
                    defaults["dna"],  # 1
                    defaults["rna"],  # 2
                    defaults["ligand"],  # 3
                ],
                dtype=torch.float32,
            ),
        )

        # lDDT thresholds
        self.register_buffer(
            "_lddt_thresholds",
            torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_mol_weights(self, mol_type_per_atom: torch.Tensor) -> torch.Tensor:
        """Map integer molecule type codes to per-atom loss weights.

        Parameters
        ----------
        mol_type_per_atom : (B, M) int tensor with codes 0-3.

        Returns
        -------
        (B, M) float tensor of per-atom weights.
        """
        return self._mol_weight_values[mol_type_per_atom]

    def _compute_local_flex(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        valid_atom_mask: torch.Tensor,
        residue_indices: torch.Tensor | None,
        target_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute local (intra-residue) flexibility loss.

        For each residue with >=2 valid atoms, compute pairwise distance
        standard deviation across target frames and penalize prediction error.

        Parameters
        ----------
        x_pred, x_gt : (B, T, M, 3)
        valid_atom_mask : (B, M) bool
        residue_indices : (B, M) int or None — maps each atom to its residue.
        target_frame_mask : (B, T) bool

        Returns
        -------
        Scalar loss tensor.
        """
        if residue_indices is None:
            return torch.tensor(0.0, device=x_pred.device)

        B, T, M, _ = x_pred.shape
        device = x_pred.device

        frame_w = target_frame_mask.float()
        n_target = frame_w.sum(dim=1, keepdim=True).clamp(min=1.0)
        frame_w = frame_w / n_target  # (B, T)

        total_loss = torch.tensor(0.0, device=device)
        n_residues = 0

        for b in range(B):
            mask_b = valid_atom_mask[b]  # (M,)
            res_b = residue_indices[b]  # (M,)

            # Get unique residues present in this sample
            valid_res = res_b[mask_b]
            if valid_res.numel() == 0:
                continue
            unique_res = valid_res.unique()

            for res_id in unique_res:
                # Atoms in this residue
                atom_idx = torch.where(mask_b & (res_b == res_id))[0]
                if atom_idx.shape[0] < 2:
                    continue

                n_a = atom_idx.shape[0]
                # Extract coords: (T, n_a, 3)
                pred_r = x_pred[b, :, atom_idx]  # (T, n_a, 3)
                gt_r = x_gt[b, :, atom_idx]  # (T, n_a, 3)

                # Pairwise distances within residue: (T, n_a, n_a)
                d_pred = torch.cdist(pred_r, pred_r)
                d_gt = torch.cdist(gt_r, gt_r)

                # Weighted mean and std across frames
                fw = frame_w[b]  # (T,)
                mean_pred = (d_pred * fw[:, None, None]).sum(dim=0, keepdim=True)
                mean_gt = (d_gt * fw[:, None, None]).sum(dim=0, keepdim=True)

                std_pred = torch.sqrt(
                    ((d_pred - mean_pred) ** 2 * fw[:, None, None]).sum(dim=0) + 1e-8
                )
                std_gt = torch.sqrt(
                    ((d_gt - mean_gt) ** 2 * fw[:, None, None]).sum(dim=0) + 1e-8
                )

                # Upper triangle only (avoid double-counting and diagonal)
                triu_mask = torch.triu(torch.ones(n_a, n_a, device=device), diagonal=1).bool()
                diff_sq = (std_pred - std_gt) ** 2
                total_loss = total_loss + diff_sq[triu_mask].mean()
                n_residues += 1

        if n_residues == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / n_residues

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def structure_loss(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        sigma: torch.Tensor,
        cond_mask: torch.Tensor,
        atom_mask: torch.Tensor,
        observed_mask: torch.Tensor,
        mol_weights: torch.Tensor,
    ) -> torch.Tensor:
        """EDM-weighted MSE on non-conditioning frames.

        Parameters
        ----------
        x_pred, x_gt : (B, T, M, 3)
        sigma : (B, T)
        cond_mask : (B, T) bool
        atom_mask : (B, M) bool
        observed_mask : (B, M) bool
        mol_weights : (B, M) per-atom weights

        Returns
        -------
        Scalar loss.
        """
        target_mask = ~cond_mask  # (B, T)
        valid_atom_mask = atom_mask & observed_mask  # (B, M)
        n_valid = valid_atom_mask.float().sum(dim=-1).clamp(min=1.0)  # (B,)

        # Per-atom squared error: (B, T, M)
        sq_err = ((x_pred - x_gt) ** 2).sum(dim=-1)

        # Apply atom mask and molecule-type weights
        weighted_err = (
            sq_err
            * mol_weights.unsqueeze(1)
            * valid_atom_mask.unsqueeze(1).float()
        )

        # Mean over atoms per frame: (B, T)
        per_frame_loss = weighted_err.sum(dim=-1) / n_valid.unsqueeze(1)

        # EDM loss weight (only for non-conditioning frames)
        edm_weight = (sigma**2 + self.sigma_data**2) / (
            sigma * self.sigma_data + 1e-8
        ) ** 2
        edm_weight = edm_weight * target_mask.float()

        n_target = target_mask.float().sum().clamp(min=1.0)
        return (edm_weight * per_frame_loss).sum() / n_target

    def bond_loss(
        self,
        x_pred: torch.Tensor,
        batch: dict[str, torch.Tensor],
        target_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Covalent bond length loss.

        Penalizes deviations from reference bond lengths on target frames.

        Parameters
        ----------
        x_pred : (B, T, M, 3)
        batch : must contain ``bond_indices`` (B, n_bonds, 2) and
                ``bond_lengths`` (B, n_bonds) if bonds are present.
        target_frame_mask : (B, T) bool

        Returns
        -------
        Scalar loss (0.0 if no bonds in batch).
        """
        if "bond_indices" not in batch:
            return torch.tensor(0.0, device=x_pred.device)

        idx_i = batch["bond_indices"][..., 0]  # (B, n_bonds)
        idx_j = batch["bond_indices"][..., 1]  # (B, n_bonds)
        ref_lengths = batch["bond_lengths"]  # (B, n_bonds)
        bond_mask = batch.get("bond_mask")

        B, T, M, _ = x_pred.shape

        # Gather atom positions for bonded pairs
        # idx_i/idx_j: (B, n_bonds) -> expand to (B, T, n_bonds)
        idx_i_exp = idx_i.unsqueeze(1).expand(B, T, -1)
        idx_j_exp = idx_j.unsqueeze(1).expand(B, T, -1)

        # Gather: (B, T, n_bonds, 3)
        xi = torch.gather(x_pred, 2, idx_i_exp.unsqueeze(-1).expand(-1, -1, -1, 3))
        xj = torch.gather(x_pred, 2, idx_j_exp.unsqueeze(-1).expand(-1, -1, -1, 3))

        pred_lengths = (xi - xj).norm(dim=-1)  # (B, T, n_bonds)

        if bond_mask is None:
            valid_bond_mask = torch.ones_like(ref_lengths, dtype=torch.bool)
        else:
            valid_bond_mask = bond_mask.bool()

        frame_mask = target_frame_mask.unsqueeze(-1).float()  # (B, T, 1)
        bond_weight = valid_bond_mask.unsqueeze(1).float()  # (B, 1, n_bonds)
        weight = frame_mask * bond_weight  # (B, T, n_bonds)
        sq = (pred_lengths - ref_lengths.unsqueeze(1)) ** 2
        n_denom = weight.sum()
        return (sq * weight).sum() / n_denom.clamp(min=1.0)

    def smooth_lddt_loss(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        atom_mask: torch.Tensor,
        observed_mask: torch.Tensor,
        target_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable lDDT approximation (AF3 Algorithm 27).

        For each atom pair within 15 Angstrom:
            score = mean over thresholds [0.5, 1, 2, 4] of
                sigmoid(threshold - |d_pred - d_gt|)

        Loss = 1 - mean(score)

        Parameters
        ----------
        x_pred, x_gt : (B, T, M, 3)
        atom_mask : (B, M) bool
        observed_mask : (B, M) bool
        target_frame_mask : (B, T) bool

        Returns
        -------
        Scalar loss.
        """
        B, T, M, _ = x_pred.shape

        # Pairwise distances: (B*T, M, M)
        d_pred = torch.cdist(
            x_pred.reshape(B * T, M, 3), x_pred.reshape(B * T, M, 3)
        )
        d_gt = torch.cdist(
            x_gt.reshape(B * T, M, 3), x_gt.reshape(B * T, M, 3)
        )

        # Valid atom mask
        valid_atom_mask = atom_mask & observed_mask  # (B, M)
        valid_bt = valid_atom_mask.unsqueeze(1).expand(B, T, M).reshape(B * T, M)

        # Frame mask: (B*T, 1, 1)
        frame_mask = target_frame_mask.reshape(B * T).float()[:, None, None]

        # Distance mask: within 15A, valid atoms, target frames, no diagonal
        dist_mask = (d_gt < 15.0)
        dist_mask = dist_mask & valid_bt.unsqueeze(-1) & valid_bt.unsqueeze(-2)
        dist_mask = dist_mask & (frame_mask > 0)
        diag = torch.eye(M, device=x_pred.device, dtype=torch.bool)
        dist_mask = dist_mask & ~diag

        # 4-threshold sigmoid scoring
        diff = torch.abs(d_pred - d_gt)  # (B*T, M, M)
        thresholds = self._lddt_thresholds.view(1, 1, 1, -1)
        scores = torch.sigmoid(thresholds - diff.unsqueeze(-1))  # (B*T, M, M, 4)
        score = scores.mean(dim=-1)  # (B*T, M, M)

        # Masked mean
        n_valid = dist_mask.float().sum().clamp(min=1.0)
        lddt = (score * dist_mask.float()).sum() / n_valid
        return 1.0 - lddt

    def flexibility_loss(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        atom_mask: torch.Tensor,
        observed_mask: torch.Tensor,
        mol_type_per_atom: torch.Tensor,
        target_frame_mask: torch.Tensor,
        residue_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Flexibility loss supervising ensemble distributional properties.

        Three components:
            1. Absolute RMSF: per-atom root-mean-square fluctuation
            2. Global relative: pairwise distance standard deviation
            3. Local relative: intra-residue distance std

        Parameters
        ----------
        x_pred, x_gt : (B, T, M, 3) — uses target frames only; needs >=2.
        atom_mask : (B, M) bool
        observed_mask : (B, M) bool
        mol_type_per_atom : (B, M) int
        target_frame_mask : (B, T) bool
        residue_indices : (B, M) int or None — for local flexibility.

        Returns
        -------
        Scalar loss.
        """
        B, T, M, _ = x_pred.shape
        device = x_pred.device

        if target_frame_mask.float().sum() < 2:
            return torch.tensor(0.0, device=device)

        valid_atom_mask = atom_mask & observed_mask  # (B, M)
        frame_w = target_frame_mask.float()  # (B, T)
        n_target = frame_w.sum(dim=1, keepdim=True).clamp(min=1.0)
        frame_w = frame_w / n_target  # normalized per-batch-element

        # ----------------------------------------------------------
        # 1. Absolute RMSF
        # ----------------------------------------------------------
        # Weighted mean position: (B, 1, M, 3)
        pred_mean = (x_pred * frame_w[:, :, None, None]).sum(dim=1, keepdim=True)
        gt_mean = (x_gt * frame_w[:, :, None, None]).sum(dim=1, keepdim=True)

        # RMSF: (B, M) — sqrt of weighted variance summed over xyz
        rmsf_pred = torch.sqrt(
            (((x_pred - pred_mean) ** 2) * frame_w[:, :, None, None])
            .sum(dim=1)
            .sum(dim=-1)
            + 1e-8
        )
        rmsf_gt = torch.sqrt(
            (((x_gt - gt_mean) ** 2) * frame_w[:, :, None, None])
            .sum(dim=1)
            .sum(dim=-1)
            + 1e-8
        )

        n_valid = valid_atom_mask.float().sum().clamp(min=1.0)
        l_abs = ((rmsf_pred - rmsf_gt) ** 2 * valid_atom_mask.float()).sum() / n_valid

        # ----------------------------------------------------------
        # 2. Global relative (pairwise distance std)
        # ----------------------------------------------------------
        # Pairwise distances: (B, T, M, M)
        d_pred = torch.cdist(
            x_pred.reshape(B * T, M, 3), x_pred.reshape(B * T, M, 3)
        ).reshape(B, T, M, M)
        d_gt = torch.cdist(
            x_gt.reshape(B * T, M, 3), x_gt.reshape(B * T, M, 3)
        ).reshape(B, T, M, M)

        # Weighted mean distance: (B, 1, M, M)
        mean_pred = (d_pred * frame_w[:, :, None, None]).sum(dim=1, keepdim=True)
        mean_gt = (d_gt * frame_w[:, :, None, None]).sum(dim=1, keepdim=True)

        # Weighted std: (B, M, M)
        std_pred = torch.sqrt(
            ((d_pred - mean_pred) ** 2 * frame_w[:, :, None, None]).sum(dim=1) + 1e-8
        )
        std_gt = torch.sqrt(
            ((d_gt - mean_gt) ** 2 * frame_w[:, :, None, None]).sum(dim=1) + 1e-8
        )

        # Distance-dependent weights: <=5A: 4x, (5,10]A: 2x, >10A: 1x
        mean_dist = mean_gt.squeeze(1)  # (B, M, M)
        gamma = torch.where(
            mean_dist <= 5.0,
            torch.tensor(4.0, device=device),
            torch.where(
                mean_dist <= 10.0,
                torch.tensor(2.0, device=device),
                torch.tensor(1.0, device=device),
            ),
        )

        pair_mask = valid_atom_mask.unsqueeze(-1) & valid_atom_mask.unsqueeze(-2)
        n_pairs = pair_mask.float().sum().clamp(min=1.0)
        l_rel_global = (
            gamma * (std_pred - std_gt) ** 2 * pair_mask.float()
        ).sum() / n_pairs

        # ----------------------------------------------------------
        # 3. Local relative (intra-residue)
        # ----------------------------------------------------------
        l_rel_local = self._compute_local_flex(
            x_pred, x_gt, valid_atom_mask, residue_indices, target_frame_mask
        )

        return (
            self.beta_abs * l_abs
            + self.beta_rel_g * l_rel_global
            + self.beta_rel_l * l_rel_local
        )

    def ligand_center_loss(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        ligand_mask: torch.Tensor,
        target_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Ligand geometric center loss for unbinding training.

        L_center = mean_t ||C(x_pred_t) - C(x_gt_t)||^2
        where C(x) = mean of ligand atom positions.

        Parameters
        ----------
        x_pred, x_gt : (B, T, M, 3)
        ligand_mask : (B, M) bool
        target_frame_mask : (B, T) bool

        Returns
        -------
        Scalar loss.
        """
        lig_mask = ligand_mask.unsqueeze(1).unsqueeze(-1).float()  # (B, 1, M, 1)
        n_lig = ligand_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1)
        n_lig = n_lig.unsqueeze(1)  # (B, 1, 1)

        center_pred = (x_pred * lig_mask).sum(dim=2) / n_lig  # (B, T, 3)
        center_gt = (x_gt * lig_mask).sum(dim=2) / n_lig  # (B, T, 3)

        sq = ((center_pred - center_gt) ** 2).sum(dim=-1)  # (B, T)
        n_target = target_frame_mask.float().sum().clamp(min=1.0)
        return (sq * target_frame_mask.float()).sum() / n_target

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        mode: str = "equilibrium",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total loss.

        Parameters
        ----------
        output : model output with keys ``x_denoised``, ``sigma``,
                 ``conditioning_mask``.
        batch : data batch with keys ``coords``, ``atom_pad_mask``,
                ``observed_atom_mask``, ``mol_type_per_atom``, plus optional
                bond data.
        mode : ``'equilibrium'`` or ``'unbinding'``.

        Returns
        -------
        (total_loss, loss_dict) where loss_dict contains per-component values.
        """
        x_pred = output["x_denoised"]
        x_gt = batch["coords"]
        sigma = output["sigma"]
        cond_mask = output["conditioning_mask"]
        atom_mask = batch["atom_pad_mask"]
        observed_mask = batch["observed_atom_mask"]
        target_frame_mask = ~cond_mask
        mol_weights = self._get_mol_weights(batch["mol_type_per_atom"])

        l_struct = self.structure_loss(
            x_pred, x_gt, sigma, cond_mask, atom_mask, observed_mask, mol_weights
        )
        l_bond = self.bond_loss(x_pred, batch, target_frame_mask)
        l_lddt = self.smooth_lddt_loss(
            x_pred, x_gt, atom_mask, observed_mask, target_frame_mask
        )

        total = l_struct + self.alpha_bond * l_bond + l_lddt

        loss_dict = {
            "loss": 0.0,  # filled below
            "l_struct": l_struct.item(),
            "l_bond": l_bond.item(),
            "l_lddt": l_lddt.item(),
        }

        if mode == "equilibrium":
            # Get residue indices from feats if available
            residue_indices = None
            if "feats" in batch and "residue_indices" in batch["feats"]:
                residue_indices = batch["feats"]["residue_indices"]

            l_flex = self.flexibility_loss(
                x_pred,
                x_gt,
                atom_mask,
                observed_mask,
                batch["mol_type_per_atom"],
                target_frame_mask,
                residue_indices=residue_indices,
            )
            total = total + self.beta_flex * l_flex
            loss_dict["l_flex"] = l_flex.item()

        elif mode == "unbinding":
            ligand_mask = batch["mol_type_per_atom"] == self.MOL_LIGAND
            l_center = self.ligand_center_loss(
                x_pred, x_gt, ligand_mask, target_frame_mask
            )
            total = total + self.beta_center * l_center
            loss_dict["l_center"] = l_center.item()

        loss_dict["loss"] = total.item()
        return total, loss_dict
