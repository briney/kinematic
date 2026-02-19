"""Trajectory preprocessing utilities.

Shared functions used by all dataset-specific preprocessing scripts.
Handles solvent removal, frame alignment, observation masks, ligand
valency checks, unit conversion (nm->A, ps->ns), and reference
structure extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import MDAnalysis as mda
import mdtraj
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Molecule type classification
# ---------------------------------------------------------------------------

_PROTEIN_RESNAMES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL",
    # Common protonation variants
    "HID", "HIE", "HIP", "CYX", "ASH", "GLH",
}
_DNA_RESNAMES = {"DA", "DT", "DC", "DG", "DA5", "DA3", "DT5", "DT3", "DC5", "DC3", "DG5", "DG3"}
_RNA_RESNAMES = {"A", "U", "C", "G", "A5", "A3", "U5", "U3", "C5", "C3", "G5", "G3"}
_SOLVENT_RESNAMES = {"HOH", "TIP3", "WAT", "SOL", "NA", "CL", "K", "MG", "ZN", "CA", "NA+", "CL-"}

# Integer codes for molecule types (matches Boltz-2 convention)
MOL_TYPE_PROTEIN = 0
MOL_TYPE_DNA = 1
MOL_TYPE_RNA = 2
MOL_TYPE_LIGAND = 3


def classify_mol_type(resname: str) -> int:
    """Classify a residue name into molecule type integer code."""
    resname = resname.strip().upper()
    if resname in _PROTEIN_RESNAMES:
        return MOL_TYPE_PROTEIN
    if resname in _DNA_RESNAMES:
        return MOL_TYPE_DNA
    if resname in _RNA_RESNAMES:
        return MOL_TYPE_RNA
    return MOL_TYPE_LIGAND


# ---------------------------------------------------------------------------
# Step 1: Solvent removal
# ---------------------------------------------------------------------------

_SOLVENT_SELECTION = "not (resname HOH TIP3 WAT SOL NA CL K MG ZN CA NA+ CL-)"


def remove_solvent(
    topology_path: str | Path,
    trajectory_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Remove water and ions, keeping protein/ligand/DNA/RNA.

    Parameters
    ----------
    topology_path : path to topology file (.gro, .pdb, .tpr)
    trajectory_path : path to trajectory file (.xtc, .trr, .dcd)
    output_path : path to write cleaned trajectory

    Returns
    -------
    Path to the cleaned trajectory file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    u = mda.Universe(str(topology_path), str(trajectory_path))
    non_solvent = u.select_atoms(_SOLVENT_SELECTION)
    logger.info(
        "Solvent removal: %d -> %d atoms",
        u.atoms.n_atoms,
        non_solvent.n_atoms,
    )

    with mda.Writer(str(output_path), non_solvent.n_atoms) as w:
        for ts in u.trajectory:
            w.write(non_solvent)

    return output_path


# ---------------------------------------------------------------------------
# Step 2: Frame alignment (remove rigid-body motion)
# ---------------------------------------------------------------------------


def align_trajectory(
    trajectory_path: str | Path,
    topology_path: str | Path,
) -> mdtraj.Trajectory:
    """Align all frames to frame 0 using Kabsch on backbone atoms.

    Parameters
    ----------
    trajectory_path : path to trajectory (.xtc, etc.)
    topology_path : path to topology (.pdb, .gro)

    Returns
    -------
    Aligned mdtraj.Trajectory (coordinates still in nm, times in ps).
    """
    traj = mdtraj.load(str(trajectory_path), top=str(topology_path))
    backbone = traj.topology.select("backbone")
    if len(backbone) == 0:
        # Fallback for non-protein systems: align on all heavy atoms
        backbone = traj.topology.select("mass > 1.5")
    traj.superpose(traj, frame=0, atom_indices=backbone)
    return traj


# ---------------------------------------------------------------------------
# Step 3: Observation mask
# ---------------------------------------------------------------------------


def build_observation_mask(coords: np.ndarray) -> np.ndarray:
    """Build boolean mask of atoms with valid (finite) reference coordinates.

    Parameters
    ----------
    coords : (n_atoms, 3) reference frame coordinates (Angstrom).

    Returns
    -------
    (n_atoms,) bool array — True for observed atoms.
    """
    return np.isfinite(coords).all(axis=-1).astype(np.bool_)


# ---------------------------------------------------------------------------
# Step 4: Ligand valency check
# ---------------------------------------------------------------------------


def check_ligand_valency(mol_block: str) -> bool:
    """Return True if the ligand passes RDKit sanitization."""
    from rdkit import Chem

    mol = Chem.MolFromMolBlock(mol_block, sanitize=False)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Step 5: Convert to processed format with canonical units
# ---------------------------------------------------------------------------

def convert_trajectory(
    traj: mdtraj.Trajectory,
    system_id: str,
    output_dir: str | Path,
    observed_mask: np.ndarray | None = None,
) -> Path:
    """Convert an aligned mdtraj trajectory to the processed .npz format.

    Applies unit conversions:
      - coordinates: nm -> Angstrom (×10)
      - timestamps: ps -> ns (÷1000)

    Saves ``{system_id}_coords.npz`` containing:
      - coords: (n_frames, n_atoms, 3) float32, Angstrom
      - times: (n_frames,) float32, ns
      - dt: float, ns per frame stride
      - observed_atom_mask: (n_atoms,) bool

    Parameters
    ----------
    traj : aligned mdtraj Trajectory (nm / ps units)
    system_id : unique identifier for this system
    output_dir : directory to write output
    observed_mask : optional pre-computed mask; if None, derived from frame 0

    Returns
    -------
    Path to the saved .npz file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Unit conversion
    coords_A = (traj.xyz * 10.0).astype(np.float32)       # nm -> A
    times_ns = (traj.time / 1000.0).astype(np.float32)    # ps -> ns
    dt_ns = float(traj.timestep) / 1000.0                 # ps -> ns

    if observed_mask is None:
        observed_mask = build_observation_mask(coords_A[0])

    out_path = output_dir / f"{system_id}_coords.npz"
    np.savez_compressed(
        out_path,
        coords=coords_A,
        times=times_ns,
        dt=np.float32(dt_ns),
        observed_atom_mask=observed_mask,
    )
    logger.info(
        "Saved %s: %d frames, %d atoms, dt=%.4f ns",
        out_path.name,
        coords_A.shape[0],
        coords_A.shape[1],
        dt_ns,
    )
    return out_path


def extract_atom_metadata(traj: mdtraj.Trajectory) -> list[dict[str, Any]]:
    """Extract per-atom metadata from an mdtraj topology.

    Returns a list of dicts with keys: name, element, residue_name,
    residue_index, chain_id, mol_type (int).
    """
    atoms = []
    for atom in traj.topology.atoms:
        atoms.append({
            "name": atom.name,
            "element": atom.element.symbol if atom.element is not None else "X",
            "residue_name": atom.residue.name,
            "residue_index": atom.residue.index,
            "chain_id": atom.residue.chain.index,
            "mol_type": classify_mol_type(atom.residue.name),
        })
    return atoms


def save_reference_structure(
    atoms: list[dict[str, Any]],
    ref_coords_A: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Save reference structure metadata for Boltz-2 tokenization.

    Parameters
    ----------
    atoms : per-atom metadata dicts from extract_atom_metadata
    ref_coords_A : (n_atoms, 3) first-frame coordinates in Angstrom
    output_path : where to write the .npz

    Returns
    -------
    Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = np.array([a["name"] for a in atoms], dtype="U4")
    elements = np.array([a["element"] for a in atoms], dtype="U2")
    residue_names = np.array([a["residue_name"] for a in atoms], dtype="U4")
    residue_indices = np.array([a["residue_index"] for a in atoms], dtype=np.int32)
    chain_ids = np.array([a["chain_id"] for a in atoms], dtype=np.int32)
    mol_types = np.array([a["mol_type"] for a in atoms], dtype=np.int32)

    np.savez_compressed(
        output_path,
        ref_coords=ref_coords_A.astype(np.float32),
        atom_names=names,
        elements=elements,
        residue_names=residue_names,
        residue_indices=residue_indices,
        chain_ids=chain_ids,
        mol_types=mol_types,
    )
    return output_path
