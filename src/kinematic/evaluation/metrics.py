"""Evaluation metrics: RMSF, W2-distance, IMS, etc."""

from __future__ import annotations

import numpy as np
import torch

# Molecule type constants (mirrored from preprocessing to avoid heavy deps)
MOL_TYPE_PROTEIN = 0
MOL_TYPE_LIGAND = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy_cpu(t: torch.Tensor) -> np.ndarray:
    """Detach tensor and convert to a NumPy array on CPU."""
    return t.detach().cpu().numpy()


def _compute_rmsf(trajectory: torch.Tensor) -> torch.Tensor:
    """Per-atom RMSF from a trajectory tensor.

    Parameters
    ----------
    trajectory : (T, M, 3) float tensor

    Returns
    -------
    (M,) RMSF values in the same units as input coordinates.
    """
    # Mean position across frames: (M, 3)
    mean_pos = trajectory.mean(dim=0)
    # Squared deviations summed over xyz, averaged over frames -> sqrt
    rmsf = torch.sqrt(
        ((trajectory - mean_pos.unsqueeze(0)) ** 2).sum(dim=-1).mean(dim=0) + 1e-8
    )
    return rmsf


def _dihedral_angle(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
) -> torch.Tensor:
    """Vectorized dihedral angle computation from four point sets.

    Parameters
    ----------
    p0, p1, p2, p3 : (..., 3) tensors representing the four atoms.

    Returns
    -------
    (...,) dihedral angles in radians, range [-pi, pi].
    """
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)

    # Normalize b2 for the atan2 computation
    b2_norm = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)

    m1 = torch.cross(n1, b2_norm, dim=-1)

    x = (n1 * n2).sum(dim=-1)
    y = (m1 * n2).sum(dim=-1)

    return torch.atan2(y, x)


# ---------------------------------------------------------------------------
# ATLAS Metrics
# ---------------------------------------------------------------------------


def pairwise_rmsd(
    trajectory: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
) -> np.ndarray:
    """Compute all-vs-all frame RMSD matrix for a trajectory.

    Parameters
    ----------
    trajectory : (N_frames, M, 3) coordinate tensor.
    atom_mask  : (M,) bool mask; ``True`` for real atoms.

    Returns
    -------
    (N, N) symmetric RMSD matrix as a numpy array.
    """
    traj = trajectory.detach().float()

    if atom_mask is not None:
        mask = atom_mask.detach().bool()
        traj = traj[:, mask]

    N, M, _ = traj.shape

    # (N, 1, M, 3) - (1, N, M, 3) -> (N, N, M, 3)
    diff = traj.unsqueeze(1) - traj.unsqueeze(0)
    # Mean over atoms and xyz, then sqrt -> (N, N)
    rmsd = torch.sqrt((diff ** 2).sum(dim=-1).mean(dim=-1) + 1e-8)
    return _to_numpy_cpu(rmsd)


def ca_rmsf_correlation(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    atom_names: np.ndarray | list[str],
    residue_indices: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
) -> float:
    """Pearson correlation of per-residue CA RMSF between predicted and GT.

    Parameters
    ----------
    pred_traj       : (T, M, 3) predicted trajectory.
    gt_traj         : (T, M, 3) ground-truth trajectory.
    atom_names      : (M,) array of atom name strings.
    residue_indices : (M,) integer residue index per atom.
    atom_mask       : (M,) bool mask for valid atoms.

    Returns
    -------
    Pearson r (float). Target: r > 0.7.
    """
    atom_names_arr = np.asarray(atom_names)

    # Build combined mask for CA atoms
    ca_mask = np.array([n.strip() == "CA" for n in atom_names_arr])
    if atom_mask is not None:
        ca_mask = ca_mask & _to_numpy_cpu(atom_mask.bool())

    if ca_mask.sum() < 2:
        return 0.0

    ca_idx = np.where(ca_mask)[0]

    pred_ca = pred_traj.detach().float()[:, ca_idx]
    gt_ca = gt_traj.detach().float()[:, ca_idx]

    rmsf_pred = _to_numpy_cpu(_compute_rmsf(pred_ca))
    rmsf_gt = _to_numpy_cpu(_compute_rmsf(gt_ca))

    # Pearson correlation
    r = float(np.corrcoef(rmsf_pred, rmsf_gt)[0, 1])
    if np.isnan(r):
        return 0.0
    return r


def w2_distance(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
    n_samples: int = 10000,
) -> float:
    """1D Wasserstein-2 distance between pairwise distance distributions.

    Computes pairwise inter-atom distances (upper triangle) across all frames,
    then approximates the W2 distance via sorted quantile matching.

    Parameters
    ----------
    pred_traj : (T_p, M, 3) predicted trajectory.
    gt_traj   : (T_g, M, 3) ground-truth trajectory.
    atom_mask : (M,) bool mask for valid atoms.
    n_samples : Number of quantile samples for W2 computation.

    Returns
    -------
    W2 distance (float, in Angstrom).
    """
    pred = pred_traj.detach().float()
    gt = gt_traj.detach().float()

    if atom_mask is not None:
        mask = atom_mask.detach().bool()
        pred = pred[:, mask]
        gt = gt[:, mask]

    M = pred.shape[1]
    if M < 2:
        return 0.0

    # Upper triangle indices
    idx_i, idx_j = torch.triu_indices(M, M, offset=1)

    def _flat_dists(traj: torch.Tensor) -> np.ndarray:
        # (T, M, M) pairwise distances
        d = torch.cdist(traj, traj)
        # Extract upper triangle for each frame and flatten
        return _to_numpy_cpu(d[:, idx_i, idx_j].reshape(-1))

    d_pred = _flat_dists(pred)
    d_gt = _flat_dists(gt)

    # Sorted quantile matching for W2
    q = np.linspace(0, 1, n_samples)
    q_pred = np.quantile(d_pred, q)
    q_gt = np.quantile(d_gt, q)

    w2 = float(np.sqrt(np.mean((q_pred - q_gt) ** 2)))
    return w2


# ---------------------------------------------------------------------------
# MISATO Metric
# ---------------------------------------------------------------------------


def interaction_map_similarity(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    mol_types: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
    contact_threshold: float = 4.5,
) -> float:
    """Pearson correlation of protein-ligand contact frequency maps.

    Parameters
    ----------
    pred_traj         : (T, M, 3) predicted trajectory.
    gt_traj           : (T, M, 3) ground-truth trajectory.
    mol_types         : (M,) integer mol type per atom.
    atom_mask         : (M,) bool mask for valid atoms.
    contact_threshold : Distance cutoff for contacts (Angstrom).

    Returns
    -------
    Pearson r between flattened contact frequency maps. Returns 0.0 if no
    protein-ligand pairs exist.
    """
    mt = mol_types.detach().long()
    prot_mask = mt == MOL_TYPE_PROTEIN
    lig_mask = mt == MOL_TYPE_LIGAND

    if atom_mask is not None:
        valid = atom_mask.detach().bool()
        prot_mask = prot_mask & valid
        lig_mask = lig_mask & valid

    n_prot = int(prot_mask.sum())
    n_lig = int(lig_mask.sum())
    if n_prot == 0 or n_lig == 0:
        return 0.0

    prot_idx = torch.where(prot_mask)[0]
    lig_idx = torch.where(lig_mask)[0]

    def _contact_freq(traj: torch.Tensor) -> np.ndarray:
        prot_coords = traj[:, prot_idx]  # (T, n_prot, 3)
        lig_coords = traj[:, lig_idx]  # (T, n_lig, 3)
        # (T, n_prot, n_lig) distances
        dists = torch.cdist(prot_coords.float(), lig_coords.float())
        contacts = (dists < contact_threshold).float().mean(dim=0)  # (n_prot, n_lig)
        return _to_numpy_cpu(contacts).ravel()

    freq_pred = _contact_freq(pred_traj.detach())
    freq_gt = _contact_freq(gt_traj.detach())

    if freq_pred.std() < 1e-10 or freq_gt.std() < 1e-10:
        return 0.0

    r = float(np.corrcoef(freq_pred, freq_gt)[0, 1])
    if np.isnan(r):
        return 0.0
    return r


# ---------------------------------------------------------------------------
# Long Trajectory QC
# ---------------------------------------------------------------------------


def physical_stability(
    trajectory: torch.Tensor,
    topology_path: str,
    frame_indices: list[int] | None = None,
    n_steps: int = 100,
) -> dict[str, float | list[float]]:
    """Assess physical plausibility via OpenMM energy minimization.

    For each selected frame, sets positions in an OpenMM system, runs a short
    energy minimization, and computes the RMSD between original and minimized
    coordinates.

    Parameters
    ----------
    trajectory    : (T, M, 3) coordinate tensor (Angstrom).
    topology_path : Path to topology file (PDB or similar).
    frame_indices : Which frames to test. Defaults to [0, T//2, T-1].
    n_steps       : Number of minimization steps per frame.

    Returns
    -------
    Dict with keys ``'mean_rmsd'``, ``'max_rmsd'``, ``'per_frame_rmsd'``.

    Raises
    ------
    ImportError
        If OpenMM is not installed.
    """
    try:
        from openmm import unit as omm_unit
        from openmm.app import PDBFile, ForceField, Simulation
        from openmm import LangevinIntegrator
    except ImportError:
        raise ImportError(
            "OpenMM is required for physical_stability but is not installed. "
            "Install it with: conda install -c conda-forge openmm"
        )

    traj = _to_numpy_cpu(trajectory.float())
    T = traj.shape[0]

    if frame_indices is None:
        frame_indices = [0, T // 2, T - 1]

    pdb = PDBFile(topology_path)
    forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    system = forcefield.createSystem(pdb.topology)
    integrator = LangevinIntegrator(
        300 * omm_unit.kelvin,
        1.0 / omm_unit.picoseconds,
        0.002 * omm_unit.picoseconds,
    )
    simulation = Simulation(pdb.topology, system, integrator)

    per_frame_rmsd: list[float] = []
    for fi in frame_indices:
        coords_ang = traj[fi]  # (M, 3) in Angstrom
        # OpenMM uses nanometers
        positions = coords_ang * 0.1  # Angstrom -> nm
        positions_omm = [
            (float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2]))
            * omm_unit.nanometers
            for i in range(positions.shape[0])
        ]
        simulation.context.setPositions(positions_omm)
        simulation.minimizeEnergy(maxIterations=n_steps)

        state = simulation.context.getState(getPositions=True)
        min_pos = state.getPositions(asNumpy=True).value_in_unit(omm_unit.nanometers)
        min_pos_ang = np.array(min_pos) * 10.0  # nm -> Angstrom

        rmsd = float(np.sqrt(np.mean((coords_ang - min_pos_ang) ** 2)))
        per_frame_rmsd.append(rmsd)

    return {
        "mean_rmsd": float(np.mean(per_frame_rmsd)),
        "max_rmsd": float(np.max(per_frame_rmsd)),
        "per_frame_rmsd": per_frame_rmsd,
    }


def ramachandran_scores(
    trajectory: torch.Tensor,
    atom_names: np.ndarray | list[str],
    residue_indices: torch.Tensor,
    mol_types: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
) -> dict[str, float | int]:
    """Compute Ramachandran favored/allowed/outlier fractions.

    Uses simplified rectangular regions for classification.

    Parameters
    ----------
    trajectory      : (T, M, 3) coordinate tensor.
    atom_names      : (M,) atom name strings.
    residue_indices : (M,) integer residue index per atom.
    mol_types       : (M,) integer mol type per atom.
    atom_mask       : (M,) bool mask for valid atoms.

    Returns
    -------
    Dict with ``'fraction_favored'``, ``'fraction_allowed'``,
    ``'fraction_outlier'``, ``'n_residues'``.
    """
    atom_names_arr = np.asarray(atom_names)
    mt = mol_types.detach().long()
    res_idx = residue_indices.detach().long()

    # Only protein atoms
    prot_mask = mt == MOL_TYPE_PROTEIN
    if atom_mask is not None:
        prot_mask = prot_mask & atom_mask.detach().bool()

    # Build per-residue backbone atom lookup: {res_id: {atom_name: global_idx}}
    backbone_atoms: dict[int, dict[str, int]] = {}
    for i in range(len(atom_names_arr)):
        if not prot_mask[i]:
            continue
        name = atom_names_arr[i].strip()
        if name in ("N", "CA", "C"):
            rid = int(res_idx[i])
            if rid not in backbone_atoms:
                backbone_atoms[rid] = {}
            backbone_atoms[rid][name] = i

    # Filter to residues with all three backbone atoms
    complete_residues = sorted(
        rid for rid, atoms in backbone_atoms.items()
        if {"N", "CA", "C"}.issubset(atoms.keys())
    )

    if len(complete_residues) < 3:
        return {
            "fraction_favored": 0.0,
            "fraction_allowed": 0.0,
            "fraction_outlier": 0.0,
            "n_residues": 0,
        }

    traj = trajectory.detach().float()
    # Average over frames for dihedral computation
    mean_coords = traj.mean(dim=0)  # (M, 3)

    phi_psi_list: list[tuple[float, float]] = []

    for k in range(1, len(complete_residues) - 1):
        rid_prev = complete_residues[k - 1]
        rid_curr = complete_residues[k]
        rid_next = complete_residues[k + 1]

        # Check chain continuity: C(prev)-N(curr) and C(curr)-N(next)
        c_prev = mean_coords[backbone_atoms[rid_prev]["C"]]
        n_curr = mean_coords[backbone_atoms[rid_curr]["N"]]
        c_curr = mean_coords[backbone_atoms[rid_curr]["C"]]
        n_next = mean_coords[backbone_atoms[rid_next]["N"]]

        if (c_prev - n_curr).norm() > 2.0:
            continue
        if (c_curr - n_next).norm() > 2.0:
            continue

        # Phi: C(i-1) - N(i) - CA(i) - C(i)
        phi = _dihedral_angle(
            c_prev,
            n_curr,
            mean_coords[backbone_atoms[rid_curr]["CA"]],
            c_curr,
        )

        # Psi: N(i) - CA(i) - C(i) - N(i+1)
        psi = _dihedral_angle(
            n_curr,
            mean_coords[backbone_atoms[rid_curr]["CA"]],
            c_curr,
            n_next,
        )

        phi_deg = float(phi) * 180.0 / np.pi
        psi_deg = float(psi) * 180.0 / np.pi
        phi_psi_list.append((phi_deg, psi_deg))

    n_residues = len(phi_psi_list)
    if n_residues == 0:
        return {
            "fraction_favored": 0.0,
            "fraction_allowed": 0.0,
            "fraction_outlier": 0.0,
            "n_residues": 0,
        }

    n_favored = 0
    n_allowed = 0
    n_outlier = 0

    for phi_deg, psi_deg in phi_psi_list:
        if _is_favored(phi_deg, psi_deg):
            n_favored += 1
        elif _is_allowed(phi_deg, psi_deg):
            n_allowed += 1
        else:
            n_outlier += 1

    return {
        "fraction_favored": n_favored / n_residues,
        "fraction_allowed": n_allowed / n_residues,
        "fraction_outlier": n_outlier / n_residues,
        "n_residues": n_residues,
    }


def _is_favored(phi: float, psi: float) -> bool:
    """Simplified rectangular favored regions."""
    # Alpha-helix: phi ~ -60, psi ~ -47
    if -100 <= phi <= -30 and -70 <= psi <= -10:
        return True
    # Beta-sheet: phi ~ -120, psi ~ 130
    if -180 <= phi <= -60 and 80 <= psi <= 180:
        return True
    # Left-handed alpha-helix: phi ~ 60, psi ~ 40
    if 30 <= phi <= 100 and 10 <= psi <= 70:
        return True
    return False


def _is_allowed(phi: float, psi: float) -> bool:
    """Simplified rectangular allowed (but not favored) regions."""
    # Broader alpha region
    if -120 <= phi <= -10 and -90 <= psi <= 10:
        return True
    # Broader beta region
    if -180 <= phi <= -40 and 60 <= psi <= 180:
        return True
    # Broader left-handed region
    if 10 <= phi <= 120 and -10 <= psi <= 90:
        return True
    # Bridge region
    if -180 <= phi <= -40 and -60 <= psi <= 60:
        return True
    return False


# ---------------------------------------------------------------------------
# DD-13M Metric
# ---------------------------------------------------------------------------


def unbinding_precision_recall(
    pred_trajs: torch.Tensor,
    gt_trajs: torch.Tensor,
    ligand_mask: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
    rmsd_threshold: float = 2.0,
) -> dict[str, float]:
    """Precision/recall/F1 for unbinding pathway frames.

    Computes frame-to-frame RMSD between predicted and ground-truth ligand
    coordinates and classifies frames as matched if RMSD < threshold.

    Parameters
    ----------
    pred_trajs     : (T_p, M, 3) predicted trajectory.
    gt_trajs       : (T_g, M, 3) ground-truth trajectory.
    ligand_mask    : (M,) bool mask for ligand atoms.
    atom_mask      : (M,) bool mask for valid atoms.
    rmsd_threshold : RMSD threshold for frame matching (Angstrom).

    Returns
    -------
    Dict with ``'precision'``, ``'recall'``, ``'f1'``.
    """
    lig = ligand_mask.detach().bool()
    if atom_mask is not None:
        lig = lig & atom_mask.detach().bool()

    n_lig = int(lig.sum())
    if n_lig == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lig_idx = torch.where(lig)[0]
    pred = pred_trajs.detach().float()[:, lig_idx]  # (T_p, n_lig, 3)
    gt = gt_trajs.detach().float()[:, lig_idx]  # (T_g, n_lig, 3)

    T_p = pred.shape[0]
    T_g = gt.shape[0]

    # (T_p, T_g, n_lig, 3) pairwise differences
    diff = pred.unsqueeze(1) - gt.unsqueeze(0)
    # (T_p, T_g) RMSD per frame pair
    rmsd_matrix = torch.sqrt((diff ** 2).sum(dim=-1).mean(dim=-1) + 1e-8)

    # Precision: fraction of pred frames with a GT neighbor within threshold
    min_per_pred = rmsd_matrix.min(dim=1).values  # (T_p,)
    precision = float((min_per_pred < rmsd_threshold).float().mean())

    # Recall: fraction of GT frames with a pred neighbor within threshold
    min_per_gt = rmsd_matrix.min(dim=0).values  # (T_g,)
    recall = float((min_per_gt < rmsd_threshold).float().mean())

    # F1: harmonic mean
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {"precision": precision, "recall": recall, "f1": f1}
