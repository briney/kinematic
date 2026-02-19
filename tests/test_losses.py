"""Tests for loss functions."""

from __future__ import annotations

import pytest
import torch

from boltzkinema.training.losses import BoltzKinemaLoss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, T, M = 2, 8, 20  # batch, frames, atoms


@pytest.fixture
def loss_fn() -> BoltzKinemaLoss:
    return BoltzKinemaLoss(sigma_data=16.0)


@pytest.fixture
def basic_batch() -> dict[str, torch.Tensor]:
    """Minimal batch dict for testing."""
    torch.manual_seed(42)
    coords = torch.randn(B, T, M, 3) * 5.0
    sigma = torch.rand(B, T) * 10.0 + 0.1
    cond_mask = torch.zeros(B, T, dtype=torch.bool)
    # First two frames are conditioning
    cond_mask[:, :2] = True

    mol_types = torch.zeros(B, M, dtype=torch.long)
    # Last 5 atoms are ligand
    mol_types[:, -5:] = 3

    return {
        "coords": coords,
        "atom_pad_mask": torch.ones(B, M, dtype=torch.bool),
        "observed_atom_mask": torch.ones(B, M, dtype=torch.bool),
        "mol_type_per_atom": mol_types,
        "feats": {
            "residue_indices": torch.arange(M).unsqueeze(0).expand(B, -1),
        },
    }


@pytest.fixture
def basic_output(basic_batch: dict) -> dict[str, torch.Tensor]:
    """Minimal model output dict."""
    torch.manual_seed(123)
    x_denoised = basic_batch["coords"] + torch.randn(B, T, M, 3) * 0.5
    sigma = torch.rand(B, T) * 10.0 + 0.1
    cond_mask = torch.zeros(B, T, dtype=torch.bool)
    cond_mask[:, :2] = True
    # Set conditioning frame sigma to 0
    sigma[cond_mask] = 0.0

    return {
        "x_denoised": x_denoised,
        "sigma": sigma,
        "conditioning_mask": cond_mask,
    }


# ---------------------------------------------------------------------------
# _get_mol_weights
# ---------------------------------------------------------------------------


class TestGetMolWeights:
    def test_default_weights(self, loss_fn: BoltzKinemaLoss):
        mol_types = torch.tensor([[0, 1, 2, 3]])  # protein, dna, rna, ligand
        weights = loss_fn._get_mol_weights(mol_types)
        expected = torch.tensor([[1.0, 5.0, 5.0, 10.0]])
        assert torch.allclose(weights, expected)

    def test_batch_shapes(self, loss_fn: BoltzKinemaLoss):
        mol_types = torch.zeros(B, M, dtype=torch.long)
        weights = loss_fn._get_mol_weights(mol_types)
        assert weights.shape == (B, M)

    def test_custom_weights(self):
        custom = {"protein": 2.0, "ligand": 20.0}
        fn = BoltzKinemaLoss(mol_weights=custom)
        mol_types = torch.tensor([[0, 3]])
        weights = fn._get_mol_weights(mol_types)
        expected = torch.tensor([[2.0, 20.0]])
        assert torch.allclose(weights, expected)


# ---------------------------------------------------------------------------
# structure_loss
# ---------------------------------------------------------------------------


class TestStructureLoss:
    def test_zero_for_perfect_prediction(self, loss_fn: BoltzKinemaLoss):
        """If x_pred == x_gt, structure loss should be 0."""
        coords = torch.randn(B, T, M, 3)
        sigma = torch.ones(B, T)
        cond_mask = torch.zeros(B, T, dtype=torch.bool)
        atom_mask = torch.ones(B, M, dtype=torch.bool)
        obs_mask = torch.ones(B, M, dtype=torch.bool)
        mol_w = torch.ones(B, M)

        loss = loss_fn.structure_loss(
            coords, coords, sigma, cond_mask, atom_mask, obs_mask, mol_w
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_conditioning_frames_excluded(self, loss_fn: BoltzKinemaLoss):
        """Conditioning frames should not contribute to loss."""
        torch.manual_seed(0)
        coords = torch.randn(B, T, M, 3)
        x_pred = coords + torch.randn_like(coords) * 2.0  # big error

        sigma = torch.ones(B, T)
        atom_mask = torch.ones(B, M, dtype=torch.bool)
        obs_mask = torch.ones(B, M, dtype=torch.bool)
        mol_w = torch.ones(B, M)

        # All frames are conditioning -> loss = 0
        cond_mask_all = torch.ones(B, T, dtype=torch.bool)
        loss_all_cond = loss_fn.structure_loss(
            x_pred, coords, sigma, cond_mask_all, atom_mask, obs_mask, mol_w
        )
        assert loss_all_cond.item() == pytest.approx(0.0, abs=1e-6)

        # Some frames conditioning -> lower loss than no conditioning
        cond_mask_half = torch.zeros(B, T, dtype=torch.bool)
        cond_mask_half[:, : T // 2] = True
        loss_half = loss_fn.structure_loss(
            x_pred, coords, sigma, cond_mask_half, atom_mask, obs_mask, mol_w
        )

        cond_mask_none = torch.zeros(B, T, dtype=torch.bool)
        loss_none = loss_fn.structure_loss(
            x_pred, coords, sigma, cond_mask_none, atom_mask, obs_mask, mol_w
        )

        # With half conditioning, per-frame loss is same but averaged over fewer frames
        # Both should be positive
        assert loss_half.item() > 0
        assert loss_none.item() > 0

    def test_unobserved_atoms_excluded(self, loss_fn: BoltzKinemaLoss):
        """Unobserved atoms should not contribute to loss."""
        torch.manual_seed(1)
        coords = torch.randn(B, T, M, 3)
        x_pred = coords.clone()
        # Add error to first 5 atoms only
        x_pred[:, :, :5] += 10.0

        sigma = torch.ones(B, T)
        cond_mask = torch.zeros(B, T, dtype=torch.bool)
        atom_mask = torch.ones(B, M, dtype=torch.bool)
        mol_w = torch.ones(B, M)

        # All atoms observed -> loss > 0
        obs_all = torch.ones(B, M, dtype=torch.bool)
        loss_all = loss_fn.structure_loss(
            x_pred, coords, sigma, cond_mask, atom_mask, obs_all, mol_w
        )
        assert loss_all.item() > 0

        # Only observe atoms 5+ (error atoms unobserved) -> loss = 0
        obs_partial = torch.ones(B, M, dtype=torch.bool)
        obs_partial[:, :5] = False
        loss_partial = loss_fn.structure_loss(
            x_pred, coords, sigma, cond_mask, atom_mask, obs_partial, mol_w
        )
        assert loss_partial.item() == pytest.approx(0.0, abs=1e-6)

    def test_edm_weighting_applied(self, loss_fn: BoltzKinemaLoss):
        """Higher sigma should get higher EDM weight."""
        torch.manual_seed(2)
        coords = torch.randn(1, 2, M, 3)
        x_pred = coords + 1.0  # uniform error
        cond_mask = torch.zeros(1, 2, dtype=torch.bool)
        atom_mask = torch.ones(1, M, dtype=torch.bool)
        obs_mask = torch.ones(1, M, dtype=torch.bool)
        mol_w = torch.ones(1, M)

        # Low sigma vs high sigma
        sigma_low = torch.tensor([[0.1, 0.1]])
        sigma_high = torch.tensor([[10.0, 10.0]])

        loss_low = loss_fn.structure_loss(
            x_pred, coords, sigma_low, cond_mask, atom_mask, obs_mask, mol_w
        )
        loss_high = loss_fn.structure_loss(
            x_pred, coords, sigma_high, cond_mask, atom_mask, obs_mask, mol_w
        )

        # EDM weight = (s^2 + sd^2) / (s * sd)^2
        # For sigma_data=16: low sigma -> very high weight, high sigma -> lower weight
        assert loss_low.item() > loss_high.item()

    def test_gradient_flows(self, loss_fn: BoltzKinemaLoss):
        """Gradients should flow back through x_pred."""
        coords = torch.randn(B, T, M, 3)
        x_pred = torch.randn(B, T, M, 3, requires_grad=True)
        sigma = torch.ones(B, T)
        cond_mask = torch.zeros(B, T, dtype=torch.bool)
        atom_mask = torch.ones(B, M, dtype=torch.bool)
        obs_mask = torch.ones(B, M, dtype=torch.bool)
        mol_w = torch.ones(B, M)

        loss = loss_fn.structure_loss(
            x_pred, coords, sigma, cond_mask, atom_mask, obs_mask, mol_w
        )
        loss.backward()
        assert x_pred.grad is not None
        assert x_pred.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# bond_loss
# ---------------------------------------------------------------------------


class TestBondLoss:
    def test_no_bonds_returns_zero(self, loss_fn: BoltzKinemaLoss):
        x_pred = torch.randn(B, T, M, 3)
        batch = {}  # no bond_indices
        target_mask = torch.ones(B, T, dtype=torch.bool)
        loss = loss_fn.bond_loss(x_pred, batch, target_mask)
        assert loss.item() == pytest.approx(0.0)

    def test_perfect_bonds_zero_loss(self, loss_fn: BoltzKinemaLoss):
        """If predicted bond lengths match reference, loss = 0."""
        x = torch.zeros(B, T, M, 3)
        # Place atom 0 at (0,0,0) and atom 1 at (1.5,0,0) -> distance 1.5
        x[:, :, 1, 0] = 1.5
        batch = {
            "bond_indices": torch.tensor([[[0, 1]], [[0, 1]]]),  # (B, 1, 2)
            "bond_lengths": torch.tensor([[1.5], [1.5]]),  # (B, 1)
        }
        target_mask = torch.ones(B, T, dtype=torch.bool)
        loss = loss_fn.bond_loss(x, batch, target_mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_bond_deviation_positive_loss(self, loss_fn: BoltzKinemaLoss):
        """Wrong bond lengths -> positive loss."""
        x = torch.zeros(B, T, M, 3)
        x[:, :, 1, 0] = 3.0  # actual distance = 3.0
        batch = {
            "bond_indices": torch.tensor([[[0, 1]], [[0, 1]]]),
            "bond_lengths": torch.tensor([[1.5], [1.5]]),  # ref = 1.5
        }
        target_mask = torch.ones(B, T, dtype=torch.bool)
        loss = loss_fn.bond_loss(x, batch, target_mask)
        assert loss.item() > 0

    def test_bond_conditioning_frames_excluded(self, loss_fn: BoltzKinemaLoss):
        """Only target frames contribute to bond loss."""
        x = torch.zeros(B, T, M, 3)
        x[:, :, 1, 0] = 3.0
        batch = {
            "bond_indices": torch.tensor([[[0, 1]], [[0, 1]]]),
            "bond_lengths": torch.tensor([[1.5], [1.5]]),
        }
        # All frames are conditioning (target_mask = False)
        target_mask = torch.zeros(B, T, dtype=torch.bool)
        loss = loss_fn.bond_loss(x, batch, target_mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# smooth_lddt_loss
# ---------------------------------------------------------------------------


class TestSmoothLddtLoss:
    def test_perfect_prediction_near_zero(self, loss_fn: BoltzKinemaLoss):
        """Perfect prediction -> lDDT is maximal -> loss is minimal.

        Note: smooth lDDT uses sigmoid approximation, so the score for a
        perfect prediction is mean(sigmoid([0.5, 1, 2, 4])) ~ 0.804,
        giving a floor loss of ~0.196. Verify it's near this floor.
        """
        torch.manual_seed(10)
        coords = torch.zeros(1, 4, 10, 3)
        for i in range(10):
            coords[:, :, i, 0] = float(i)

        atom_mask = torch.ones(1, 10, dtype=torch.bool)
        obs_mask = torch.ones(1, 10, dtype=torch.bool)
        target_mask = torch.ones(1, 4, dtype=torch.bool)

        loss = loss_fn.smooth_lddt_loss(coords, coords, atom_mask, obs_mask, target_mask)
        # sigmoid floor: 1 - mean(sigmoid([0.5, 1, 2, 4])) ≈ 0.196
        expected_floor = 1.0 - torch.sigmoid(torch.tensor([0.5, 1.0, 2.0, 4.0])).mean().item()
        assert loss.item() == pytest.approx(expected_floor, abs=0.01)

    def test_large_error_high_loss(self, loss_fn: BoltzKinemaLoss):
        """Large prediction error -> low lDDT -> high loss."""
        torch.manual_seed(11)
        coords_gt = torch.zeros(1, 4, 10, 3)
        for i in range(10):
            coords_gt[:, :, i, 0] = float(i)

        # Shuffle atoms drastically
        coords_pred = coords_gt.clone()
        coords_pred += torch.randn_like(coords_pred) * 10.0

        atom_mask = torch.ones(1, 10, dtype=torch.bool)
        obs_mask = torch.ones(1, 10, dtype=torch.bool)
        target_mask = torch.ones(1, 4, dtype=torch.bool)

        loss = loss_fn.smooth_lddt_loss(
            coords_pred, coords_gt, atom_mask, obs_mask, target_mask
        )
        assert loss.item() > 0.3  # should be substantially > 0

    def test_conditioning_frames_excluded(self, loss_fn: BoltzKinemaLoss):
        """All-conditioning target mask -> no valid pairs -> degenerate loss."""
        coords = torch.randn(1, 4, 10, 3) * 3.0
        atom_mask = torch.ones(1, 10, dtype=torch.bool)
        obs_mask = torch.ones(1, 10, dtype=torch.bool)
        target_mask = torch.zeros(1, 4, dtype=torch.bool)  # all conditioning

        loss = loss_fn.smooth_lddt_loss(
            coords + 5.0, coords, atom_mask, obs_mask, target_mask
        )
        # With no target frames, no valid pairs -> lddt defaults to 0 -> loss = 1
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_loss_bounded(self, loss_fn: BoltzKinemaLoss):
        """lDDT loss should be in [0, 1]."""
        torch.manual_seed(12)
        coords = torch.randn(1, 4, 10, 3) * 5.0
        pred = coords + torch.randn_like(coords) * 2.0
        atom_mask = torch.ones(1, 10, dtype=torch.bool)
        obs_mask = torch.ones(1, 10, dtype=torch.bool)
        target_mask = torch.ones(1, 4, dtype=torch.bool)

        loss = loss_fn.smooth_lddt_loss(pred, coords, atom_mask, obs_mask, target_mask)
        assert 0.0 <= loss.item() <= 1.0


# ---------------------------------------------------------------------------
# flexibility_loss
# ---------------------------------------------------------------------------


class TestFlexibilityLoss:
    def test_zero_for_identical_ensembles(self, loss_fn: BoltzKinemaLoss):
        """If pred and gt have identical distributions, flex loss ~ 0."""
        torch.manual_seed(20)
        coords = torch.randn(B, T, M, 3) * 3.0
        atom_mask = torch.ones(B, M, dtype=torch.bool)
        obs_mask = torch.ones(B, M, dtype=torch.bool)
        mol_types = torch.zeros(B, M, dtype=torch.long)
        target_mask = torch.ones(B, T, dtype=torch.bool)

        loss = loss_fn.flexibility_loss(
            coords, coords, atom_mask, obs_mask, mol_types, target_mask
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_fewer_than_2_target_frames(self, loss_fn: BoltzKinemaLoss):
        """With <2 target frames, flex loss = 0."""
        coords = torch.randn(B, T, M, 3)
        atom_mask = torch.ones(B, M, dtype=torch.bool)
        obs_mask = torch.ones(B, M, dtype=torch.bool)
        mol_types = torch.zeros(B, M, dtype=torch.long)

        # Only 1 target frame total across batch
        target_mask = torch.zeros(B, T, dtype=torch.bool)
        target_mask[0, 0] = True

        loss = loss_fn.flexibility_loss(
            coords, coords, atom_mask, obs_mask, mol_types, target_mask
        )
        assert loss.item() == pytest.approx(0.0)

    def test_positive_for_different_distributions(self, loss_fn: BoltzKinemaLoss):
        """Different pred/gt distributions -> positive flex loss."""
        torch.manual_seed(21)
        gt = torch.randn(B, T, M, 3) * 2.0
        # pred has much more variance
        pred = torch.randn(B, T, M, 3) * 10.0
        atom_mask = torch.ones(B, M, dtype=torch.bool)
        obs_mask = torch.ones(B, M, dtype=torch.bool)
        mol_types = torch.zeros(B, M, dtype=torch.long)
        target_mask = torch.ones(B, T, dtype=torch.bool)

        loss = loss_fn.flexibility_loss(
            pred, gt, atom_mask, obs_mask, mol_types, target_mask
        )
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# ligand_center_loss
# ---------------------------------------------------------------------------


class TestLigandCenterLoss:
    def test_zero_for_perfect_centers(self, loss_fn: BoltzKinemaLoss):
        """Same coords -> same centers -> loss = 0."""
        coords = torch.randn(B, T, M, 3)
        lig_mask = torch.zeros(B, M, dtype=torch.bool)
        lig_mask[:, -5:] = True
        target_mask = torch.ones(B, T, dtype=torch.bool)

        loss = loss_fn.ligand_center_loss(coords, coords, lig_mask, target_mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_shifted_ligand(self, loss_fn: BoltzKinemaLoss):
        """Shifted ligand center -> positive loss."""
        coords = torch.randn(B, T, M, 3)
        pred = coords.clone()
        pred[:, :, -5:] += 5.0  # shift ligand atoms

        lig_mask = torch.zeros(B, M, dtype=torch.bool)
        lig_mask[:, -5:] = True
        target_mask = torch.ones(B, T, dtype=torch.bool)

        loss = loss_fn.ligand_center_loss(pred, coords, lig_mask, target_mask)
        assert loss.item() > 0

    def test_conditioning_frames_excluded(self, loss_fn: BoltzKinemaLoss):
        """All conditioning -> loss = 0."""
        coords = torch.randn(B, T, M, 3)
        pred = coords.clone()
        pred[:, :, -5:] += 5.0

        lig_mask = torch.zeros(B, M, dtype=torch.bool)
        lig_mask[:, -5:] = True
        target_mask = torch.zeros(B, T, dtype=torch.bool)

        loss = loss_fn.ligand_center_loss(pred, coords, lig_mask, target_mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# forward (integration)
# ---------------------------------------------------------------------------


class TestForward:
    def test_equilibrium_mode(
        self,
        loss_fn: BoltzKinemaLoss,
        basic_output: dict,
        basic_batch: dict,
    ):
        total, loss_dict = loss_fn(basic_output, basic_batch, mode="equilibrium")
        assert total.item() > 0
        assert "loss" in loss_dict
        assert "l_struct" in loss_dict
        assert "l_bond" in loss_dict
        assert "l_lddt" in loss_dict
        assert "l_flex" in loss_dict
        assert loss_dict["loss"] == pytest.approx(total.item(), abs=1e-5)

    def test_unbinding_mode(
        self,
        loss_fn: BoltzKinemaLoss,
        basic_output: dict,
        basic_batch: dict,
    ):
        total, loss_dict = loss_fn(basic_output, basic_batch, mode="unbinding")
        assert total.item() > 0
        assert "l_center" in loss_dict
        assert "l_flex" not in loss_dict

    def test_gradient_flows_through_forward(
        self,
        loss_fn: BoltzKinemaLoss,
        basic_batch: dict,
    ):
        """Gradients flow from total loss to x_denoised."""
        torch.manual_seed(99)
        x = torch.randn(B, T, M, 3, requires_grad=True)
        sigma = torch.rand(B, T) * 5.0 + 0.1
        cond_mask = torch.zeros(B, T, dtype=torch.bool)
        cond_mask[:, 0] = True
        sigma[cond_mask] = 0.0

        output = {
            "x_denoised": x,
            "sigma": sigma,
            "conditioning_mask": cond_mask,
        }
        total, _ = loss_fn(output, basic_batch, mode="equilibrium")
        total.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_no_bonds_equilibrium(
        self,
        loss_fn: BoltzKinemaLoss,
        basic_output: dict,
        basic_batch: dict,
    ):
        """Equilibrium mode without bonds in batch works fine."""
        # Remove any bond data
        basic_batch.pop("bond_indices", None)
        basic_batch.pop("bond_lengths", None)

        total, loss_dict = loss_fn(basic_output, basic_batch, mode="equilibrium")
        assert total.item() > 0
        assert loss_dict["l_bond"] == pytest.approx(0.0)

    def test_with_bonds(self, loss_fn: BoltzKinemaLoss, basic_output: dict, basic_batch: dict):
        """Forward with bond data produces nonzero bond loss."""
        # Add bond data: atom 0 bonded to atom 1 with ref length 1.5
        basic_batch["bond_indices"] = torch.tensor([[[0, 1]], [[0, 1]]])
        basic_batch["bond_lengths"] = torch.tensor([[1.5], [1.5]])

        total, loss_dict = loss_fn(basic_output, basic_batch, mode="equilibrium")
        assert total.item() > 0
        # Bond loss should be non-zero since random coords unlikely to have exact length 1.5
        assert loss_dict["l_bond"] > 0

    def test_loss_dict_values_are_float(
        self,
        loss_fn: BoltzKinemaLoss,
        basic_output: dict,
        basic_batch: dict,
    ):
        """All loss_dict values should be Python floats (for logging)."""
        _, loss_dict = loss_fn(basic_output, basic_batch, mode="equilibrium")
        for k, v in loss_dict.items():
            assert isinstance(v, float), f"{k} is {type(v)}, expected float"


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_default_params(self):
        fn = BoltzKinemaLoss()
        assert fn.sigma_data == 16.0
        assert fn.alpha_bond == 1.0
        assert fn.beta_flex == 1.0
        assert fn.beta_abs == 1.0
        assert fn.beta_rel_g == 4.0
        assert fn.beta_rel_l == 4.0
        assert fn.beta_center == 1.0

    def test_custom_params(self):
        fn = BoltzKinemaLoss(
            sigma_data=10.0,
            alpha_bond=2.0,
            beta_flex=0.5,
            beta_center=3.0,
        )
        assert fn.sigma_data == 10.0
        assert fn.alpha_bond == 2.0
        assert fn.beta_flex == 0.5
        assert fn.beta_center == 3.0

    def test_device_transfer(self):
        """Buffers should transfer with .to()."""
        fn = BoltzKinemaLoss()
        # Just check that cpu works (GPU tests need hardware)
        fn = fn.to("cpu")
        assert fn._mol_weight_values.device.type == "cpu"
        assert fn._lddt_thresholds.device.type == "cpu"
