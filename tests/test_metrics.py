"""Tests for evaluation metrics."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from kinematic.evaluation.metrics import (
    ca_rmsf_correlation,
    interaction_map_similarity,
    pairwise_rmsd,
    physical_stability,
    ramachandran_scores,
    unbinding_precision_recall,
    w2_distance,
)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

N_FRAMES = 16
N_RES = 10
N_BACKBONE = 30  # 3 backbone atoms per residue
N_LIG = 5
M = N_BACKBONE + N_LIG  # 35 total atoms
N_TRAJ = 3


# ---------------------------------------------------------------------------
# TestPairwiseRmsd
# ---------------------------------------------------------------------------


class TestPairwiseRmsd:
    def test_zero_diagonal(self):
        """Diagonal of RMSD matrix should be zero (frame vs itself)."""
        torch.manual_seed(0)
        traj = torch.randn(N_FRAMES, M, 3)
        result = pairwise_rmsd(traj)
        diag = np.diag(result)
        np.testing.assert_allclose(diag, 0.0, atol=1e-3)

    def test_symmetric(self):
        """RMSD matrix should be symmetric."""
        torch.manual_seed(1)
        traj = torch.randn(N_FRAMES, M, 3) * 5.0
        result = pairwise_rmsd(traj)
        np.testing.assert_allclose(result, result.T, atol=1e-5)

    def test_static_trajectory_all_zeros(self):
        """All frames identical -> RMSD matrix is all zeros."""
        frame = torch.randn(1, M, 3)
        traj = frame.expand(N_FRAMES, -1, -1)
        result = pairwise_rmsd(traj)
        np.testing.assert_allclose(result, 0.0, atol=1e-3)

    def test_respects_mask(self):
        """Masking atoms should change the result."""
        torch.manual_seed(2)
        traj = torch.randn(N_FRAMES, M, 3) * 5.0
        mask_all = torch.ones(M, dtype=torch.bool)
        mask_half = torch.zeros(M, dtype=torch.bool)
        mask_half[:M // 2] = True

        r_all = pairwise_rmsd(traj, atom_mask=mask_all)
        r_half = pairwise_rmsd(traj, atom_mask=mask_half)
        # Results differ when using different atom subsets
        assert not np.allclose(r_all, r_half, atol=1e-3)

    def test_correct_shape(self):
        """Output shape should be (N_frames, N_frames)."""
        traj = torch.randn(N_FRAMES, M, 3)
        result = pairwise_rmsd(traj)
        assert result.shape == (N_FRAMES, N_FRAMES)


# ---------------------------------------------------------------------------
# TestCaRmsfCorrelation
# ---------------------------------------------------------------------------


class TestCaRmsfCorrelation:
    @staticmethod
    def _make_atom_names(n_res: int, n_lig: int) -> list[str]:
        """Create atom names: N, CA, C per residue + ligand atoms."""
        names = []
        for _ in range(n_res):
            names.extend(["N", "CA", "C"])
        for i in range(n_lig):
            names.append(f"L{i}")
        return names

    def test_identical_trajectory_perfect_correlation(self):
        """Identical pred/gt -> r ~ 1.0."""
        torch.manual_seed(10)
        traj = torch.randn(N_FRAMES, M, 3) * 3.0
        atom_names = self._make_atom_names(N_RES, N_LIG)
        res_idx = torch.tensor([i // 3 for i in range(N_BACKBONE)] + [-1] * N_LIG)

        r = ca_rmsf_correlation(traj, traj, atom_names, res_idx)
        assert r == pytest.approx(1.0, abs=0.01)

    def test_random_trajectories_low_correlation(self):
        """Independent random trajectories -> low |r|."""
        torch.manual_seed(11)
        pred = torch.randn(N_FRAMES, M, 3) * 3.0
        gt = torch.randn(N_FRAMES, M, 3) * 3.0
        atom_names = self._make_atom_names(N_RES, N_LIG)
        res_idx = torch.tensor([i // 3 for i in range(N_BACKBONE)] + [-1] * N_LIG)

        r = ca_rmsf_correlation(pred, gt, atom_names, res_idx)
        assert abs(r) < 0.8  # not perfectly correlated

    def test_returns_float(self):
        """Return type should be a Python float."""
        torch.manual_seed(12)
        traj = torch.randn(N_FRAMES, M, 3)
        atom_names = self._make_atom_names(N_RES, N_LIG)
        res_idx = torch.tensor([i // 3 for i in range(N_BACKBONE)] + [-1] * N_LIG)

        r = ca_rmsf_correlation(traj, traj, atom_names, res_idx)
        assert isinstance(r, float)

    def test_handles_few_cas(self):
        """With fewer than 2 CA atoms, returns 0.0."""
        traj = torch.randn(N_FRAMES, 3, 3)
        atom_names = ["N", "X", "C"]  # no CA
        res_idx = torch.tensor([0, 0, 0])

        r = ca_rmsf_correlation(traj, traj, atom_names, res_idx)
        assert r == 0.0


# ---------------------------------------------------------------------------
# TestW2Distance
# ---------------------------------------------------------------------------


class TestW2Distance:
    def test_identical_near_zero(self):
        """Identical trajectories -> W2 ~ 0."""
        torch.manual_seed(20)
        traj = torch.randn(N_FRAMES, M, 3) * 3.0
        w2 = w2_distance(traj, traj)
        assert w2 == pytest.approx(0.0, abs=0.01)

    def test_different_positive(self):
        """Different trajectories -> W2 > 0."""
        torch.manual_seed(21)
        pred = torch.randn(N_FRAMES, M, 3) * 3.0
        gt = torch.randn(N_FRAMES, M, 3) * 10.0
        w2 = w2_distance(pred, gt)
        assert w2 > 0.1

    def test_symmetric(self):
        """W2(a, b) == W2(b, a)."""
        torch.manual_seed(22)
        a = torch.randn(N_FRAMES, M, 3) * 3.0
        b = torch.randn(N_FRAMES, M, 3) * 5.0
        assert w2_distance(a, b) == pytest.approx(w2_distance(b, a), abs=0.01)

    def test_respects_mask(self):
        """Masking atoms should change the W2 result."""
        torch.manual_seed(23)
        pred = torch.randn(N_FRAMES, M, 3) * 3.0
        gt = torch.randn(N_FRAMES, M, 3) * 5.0
        mask = torch.zeros(M, dtype=torch.bool)
        mask[:10] = True

        w2_all = w2_distance(pred, gt)
        w2_masked = w2_distance(pred, gt, atom_mask=mask)
        assert w2_all != pytest.approx(w2_masked, abs=0.01)


# ---------------------------------------------------------------------------
# TestInteractionMapSimilarity
# ---------------------------------------------------------------------------


class TestInteractionMapSimilarity:
    @staticmethod
    def _make_mol_types() -> torch.Tensor:
        """Protein backbone + ligand atoms."""
        mt = torch.zeros(M, dtype=torch.long)
        mt[N_BACKBONE:] = 3  # ligand
        return mt

    def test_identical_perfect_correlation(self):
        """Identical pred/gt contact maps -> r ~ 1.0."""
        torch.manual_seed(30)
        # Place protein and ligand close together so contacts exist
        traj = torch.randn(N_FRAMES, M, 3) * 2.0
        mol_types = self._make_mol_types()

        r = interaction_map_similarity(traj, traj, mol_types)
        assert r == pytest.approx(1.0, abs=0.01)

    def test_no_ligand_returns_zero(self):
        """No ligand atoms -> returns 0.0."""
        torch.manual_seed(31)
        traj = torch.randn(N_FRAMES, M, 3)
        mol_types = torch.zeros(M, dtype=torch.long)  # all protein, no ligand

        r = interaction_map_similarity(traj, traj, mol_types)
        assert r == 0.0

    def test_respects_mask(self):
        """Masking out all ligand atoms -> 0.0."""
        torch.manual_seed(32)
        traj = torch.randn(N_FRAMES, M, 3) * 2.0
        mol_types = self._make_mol_types()
        mask = torch.ones(M, dtype=torch.bool)
        mask[N_BACKBONE:] = False  # mask out all ligands

        r = interaction_map_similarity(traj, traj, mol_types, atom_mask=mask)
        assert r == 0.0

    def test_returns_float(self):
        """Result should be a Python float."""
        torch.manual_seed(33)
        traj = torch.randn(N_FRAMES, M, 3) * 2.0
        mol_types = self._make_mol_types()

        r = interaction_map_similarity(traj, traj, mol_types)
        assert isinstance(r, float)


# ---------------------------------------------------------------------------
# TestPhysicalStability
# ---------------------------------------------------------------------------


class TestPhysicalStability:
    def test_import_error_without_openmm(self):
        """Should raise ImportError with clear message when openmm missing."""
        traj = torch.randn(N_FRAMES, M, 3)

        with patch.dict("sys.modules", {"openmm": None, "openmm.app": None, "openmm.unit": None}):
            with pytest.raises(ImportError, match="OpenMM is required"):
                physical_stability(traj, "dummy.pdb")


# ---------------------------------------------------------------------------
# TestRamachandranScores
# ---------------------------------------------------------------------------


class TestRamachandranScores:
    @staticmethod
    def _make_backbone_data(n_res: int):
        """Create backbone atom data for a protein chain.

        Places atoms in a roughly extended conformation so dihedrals
        are computable.
        """
        names = []
        res_indices = []
        mol_types_list = []
        coords = []

        for i in range(n_res):
            # N, CA, C in an extended conformation
            base_x = i * 3.8  # ~3.8 A per residue along x
            # N
            names.append("N")
            res_indices.append(i)
            mol_types_list.append(0)
            coords.append([base_x, 0.0, 0.0])
            # CA
            names.append("CA")
            res_indices.append(i)
            mol_types_list.append(0)
            coords.append([base_x + 1.47, 0.0, 0.0])
            # C
            names.append("C")
            res_indices.append(i)
            mol_types_list.append(0)
            coords.append([base_x + 2.47, 0.5, 0.0])

        atom_names = np.array(names)
        residue_indices = torch.tensor(res_indices)
        mol_types = torch.tensor(mol_types_list, dtype=torch.long)
        # (1, n_atoms, 3) trajectory with single frame, then expand
        frame = torch.tensor(coords, dtype=torch.float32)
        trajectory = frame.unsqueeze(0).expand(N_FRAMES, -1, -1).clone()
        # Add small noise for variation
        trajectory += torch.randn_like(trajectory) * 0.01

        return trajectory, atom_names, residue_indices, mol_types

    def test_returns_correct_keys(self):
        """Output dict should have the expected keys."""
        traj, names, res_idx, mt = self._make_backbone_data(N_RES)
        result = ramachandran_scores(traj, names, res_idx, mt)
        assert "fraction_favored" in result
        assert "fraction_allowed" in result
        assert "fraction_outlier" in result
        assert "n_residues" in result

    def test_fractions_sum_to_one(self):
        """Favored + allowed + outlier should sum to ~1.0."""
        traj, names, res_idx, mt = self._make_backbone_data(N_RES)
        result = ramachandran_scores(traj, names, res_idx, mt)
        if result["n_residues"] > 0:
            total = (
                result["fraction_favored"]
                + result["fraction_allowed"]
                + result["fraction_outlier"]
            )
            assert total == pytest.approx(1.0, abs=1e-6)

    def test_excludes_non_protein(self):
        """Non-protein atoms should not contribute residues."""
        traj, names, res_idx, mt = self._make_backbone_data(N_RES)
        # Mark all as ligand
        mt_lig = torch.full_like(mt, 3)
        result = ramachandran_scores(traj, names, res_idx, mt_lig)
        assert result["n_residues"] == 0

    def test_too_few_residues(self):
        """With < 3 complete residues, n_residues should be 0."""
        traj, names, res_idx, mt = self._make_backbone_data(2)
        result = ramachandran_scores(traj, names, res_idx, mt)
        assert result["n_residues"] == 0

    def test_respects_atom_mask(self):
        """Masking out residues reduces n_residues."""
        traj, names, res_idx, mt = self._make_backbone_data(N_RES)
        n_atoms = traj.shape[1]
        mask = torch.ones(n_atoms, dtype=torch.bool)

        result_all = ramachandran_scores(traj, names, res_idx, mt, atom_mask=mask)

        # Mask out last 3 residues (9 atoms)
        mask[-9:] = False
        result_masked = ramachandran_scores(traj, names, res_idx, mt, atom_mask=mask)

        assert result_masked["n_residues"] <= result_all["n_residues"]


# ---------------------------------------------------------------------------
# TestUnbindingPrecisionRecall
# ---------------------------------------------------------------------------


class TestUnbindingPrecisionRecall:
    @staticmethod
    def _make_ligand_mask() -> torch.Tensor:
        mask = torch.zeros(M, dtype=torch.bool)
        mask[N_BACKBONE:] = True
        return mask

    def test_identical_perfect_scores(self):
        """Identical pred/gt -> P=R=F1=1.0."""
        torch.manual_seed(40)
        traj = torch.randn(N_FRAMES, M, 3)
        lig_mask = self._make_ligand_mask()

        result = unbinding_precision_recall(traj, traj, lig_mask)
        assert result["precision"] == pytest.approx(1.0, abs=0.01)
        assert result["recall"] == pytest.approx(1.0, abs=0.01)
        assert result["f1"] == pytest.approx(1.0, abs=0.01)

    def test_disjoint_low_scores(self):
        """Very different trajectories -> low P, R, F1."""
        torch.manual_seed(41)
        pred = torch.randn(N_FRAMES, M, 3) * 2.0
        gt = torch.randn(N_FRAMES, M, 3) * 2.0 + 100.0  # far away
        lig_mask = self._make_ligand_mask()

        result = unbinding_precision_recall(pred, gt, lig_mask)
        assert result["precision"] == pytest.approx(0.0, abs=0.01)
        assert result["recall"] == pytest.approx(0.0, abs=0.01)
        assert result["f1"] == pytest.approx(0.0, abs=0.01)

    def test_f1_is_harmonic_mean(self):
        """F1 should be the harmonic mean of precision and recall."""
        torch.manual_seed(42)
        pred = torch.randn(N_FRAMES, M, 3) * 2.0
        gt = torch.randn(N_FRAMES, M, 3) * 2.5
        lig_mask = self._make_ligand_mask()

        result = unbinding_precision_recall(pred, gt, lig_mask, rmsd_threshold=5.0)
        p, r = result["precision"], result["recall"]
        if p + r > 0:
            expected_f1 = 2 * p * r / (p + r)
            assert result["f1"] == pytest.approx(expected_f1, abs=1e-6)

    def test_returns_correct_keys(self):
        """Output should have precision, recall, f1 keys."""
        traj = torch.randn(N_FRAMES, M, 3)
        lig_mask = self._make_ligand_mask()

        result = unbinding_precision_recall(traj, traj, lig_mask)
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result

    def test_no_ligand_atoms(self):
        """Empty ligand mask -> all zeros."""
        traj = torch.randn(N_FRAMES, M, 3)
        lig_mask = torch.zeros(M, dtype=torch.bool)

        result = unbinding_precision_recall(traj, traj, lig_mask)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCudaTensorSafety:
    def test_pairwise_rmsd_cuda_input(self):
        traj = torch.randn(6, 12, 3, device="cuda")
        out = pairwise_rmsd(traj)
        assert isinstance(out, np.ndarray)
        assert out.shape == (6, 6)

    def test_w2_distance_cuda_input(self):
        pred = torch.randn(6, 12, 3, device="cuda")
        gt = torch.randn(6, 12, 3, device="cuda")
        val = w2_distance(pred, gt, n_samples=256)
        assert isinstance(val, float)

    def test_interaction_map_similarity_cuda_input(self):
        pred = torch.randn(6, 12, 3, device="cuda")
        gt = torch.randn(6, 12, 3, device="cuda")
        mol_types = torch.zeros(12, dtype=torch.long, device="cuda")
        mol_types[-3:] = 3
        val = interaction_map_similarity(pred, gt, mol_types)
        assert isinstance(val, float)
