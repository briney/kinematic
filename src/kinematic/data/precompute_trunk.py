"""Precompute trunk embeddings (s_trunk, z_trunk) using frozen Boltz-2.

Runs the Boltz-2 trunk (InputEmbedder -> MSA -> Pairformer with recycling)
on each system's reference structure and caches (s_inputs, s_trunk, z_trunk)
as compressed float16 .npz files.

rel_pos_enc is NOT cached (recomputed at runtime per trunk cache rule).

Storage estimate per system (200-residue protein):
  - s_inputs: 200 x 384 x 2 bytes = 150 KB
  - s_trunk:  200 x 384 x 2 bytes = 150 KB
  - z_trunk:  200 x 200 x 128 x 2 bytes = 10 MB
  - Total: ~10.3 MB per system
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from kinematic.data.trunk_cache import save_trunk_embeddings

logger = logging.getLogger(__name__)


def load_boltz2_model(
    checkpoint_path: str | Path,
    device: str = "cuda",
) -> torch.nn.Module:
    """Load the Boltz-2 model from a checkpoint.

    Returns the model in eval mode on the specified device.
    """
    from boltz.main import Boltz2DiffusionParams, MSAModuleArgs, PairformerArgsV2, download_boltz2
    from boltz.model.models.boltz2 import Boltz2

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.info("Checkpoint not found at %s, downloading Boltz-2 weights...", checkpoint_path)
        cache_dir = checkpoint_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        download_boltz2(cache_dir)

    diffusion_params = Boltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs()

    model = Boltz2.load_from_checkpoint(
        str(checkpoint_path),
        strict=True,
        map_location=device,
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=False,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
    )
    model.eval()
    model.to(device)
    return model


def run_trunk(
    model: torch.nn.Module,
    feats: dict[str, torch.Tensor],
    recycling_steps: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute only the trunk portion of Boltz2.forward().

    Replicates the trunk logic from Boltz2.forward():
      1. input_embedder(feats) -> s_inputs
      2. s_init(s_inputs) -> s; z_init_1 + z_init_2 outer product -> z
      3. Add rel_pos, token_bonds to z
      4. Recycling loop: s_recycle, z_recycle, msa_module, pairformer_module
      5. Return s_inputs, s_trunk, z_trunk

    Parameters
    ----------
    model : Boltz2 model in eval mode.
    feats : feature dict from Boltz2Featurizer.
    recycling_steps : number of recycling iterations (Boltz-2 default: 3).

    Returns
    -------
    s_inputs : (N_tokens, token_s) raw input embeddings.
    s_trunk : (N_tokens, token_s) refined single representation.
    z_trunk : (N_tokens, N_tokens, token_z) refined pair representation.
    """
    # InputEmbedder
    s_inputs = model.input_embedder(feats)  # (1, N, 384)

    # Initial projections
    s_init = model.s_init(s_inputs)  # (1, N, 384)
    z_init = (
        model.z_init_1(s_inputs)[:, :, None]
        + model.z_init_2(s_inputs)[:, None, :]
    )  # (1, N, N, 128)

    # Relative position encoding (NOT cached, but needed for trunk computation)
    rel_pos_enc = model.rel_pos(feats)
    z_init = z_init + rel_pos_enc

    # Token bonds
    z_init = z_init + model.token_bonds(feats["token_bonds"].float())
    if hasattr(model, "bond_type_feature") and model.bond_type_feature:
        z_init = z_init + model.token_bonds_type(feats["type_bonds"].long())

    # Contact conditioning
    z_init = z_init + model.contact_conditioning(feats)

    # Compute masks
    mask = feats["token_pad_mask"].float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    # Recycling loop
    s = torch.zeros_like(s_init)
    z = torch.zeros_like(z_init)

    for _cycle in range(recycling_steps + 1):
        s = s_init + model.s_recycle(model.s_norm(s))
        z = z_init + model.z_recycle(model.z_norm(z))

        # MSA module
        z = z + model.msa_module(z, s_inputs, feats, use_kernels=False)

        # Pairformer
        s, z = model.pairformer_module(
            s, z, mask=mask, pair_mask=pair_mask, use_kernels=False
        )

    # Remove batch dim
    return s_inputs.squeeze(0), s.squeeze(0), z.squeeze(0)


def prepare_features_for_system(
    ref_path: str | Path,
    coords_path: str | Path,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Prepare Boltz-2 compatible features from a preprocessed system.

    This builds a minimal feature dict sufficient for the trunk computation.
    For full Boltz-2 feature compatibility, use the Boltz2Featurizer.

    Parameters
    ----------
    ref_path : path to reference structure .npz (from preprocessing).
    coords_path : path to coordinates .npz (for first-frame reference).
    device : target device.

    Returns
    -------
    Feature dict with keys needed by the trunk.
    """
    ref_data = np.load(ref_path, allow_pickle=False)
    np.load(coords_path, allow_pickle=False)  # validate coords file exists

    ref_coords = ref_data["ref_coords"]      # (n_atoms, 3) Angstrom
    residue_indices = ref_data["residue_indices"]
    chain_ids = ref_data["chain_ids"]
    mol_types = ref_data["mol_types"]
    atom_names = ref_data["atom_names"]        # (n_atoms,) str e.g. "CA"
    elements = ref_data["elements"]            # (n_atoms,) str e.g. "C"

    # Determine number of tokens (unique residues)
    n_tokens = int(residue_indices.max()) + 1
    n_atoms = len(ref_coords)

    # Pad atom count to multiple of Boltz-2 atom attention window size.
    # AtomEncoder does K = N // W then view(B, K, W, ...) which requires N % W == 0.
    ATOM_WINDOW_SIZE = 32
    n_atoms_padded = ((n_atoms - 1) // ATOM_WINDOW_SIZE + 1) * ATOM_WINDOW_SIZE

    # Build token-level features
    # For the trunk, we need at minimum:
    #   - token_pad_mask, token_bonds, residue_index, asym_id, entity_id,
    #     mol_type, res_type, ref_pos, atom_pad_mask, atom_to_token

    feats: dict[str, torch.Tensor] = {}

    # Token masks
    feats["token_pad_mask"] = torch.ones(1, n_tokens, device=device)

    # Token bonds (N x N x 1 adjacency — sequential bonds for polymers)
    token_bonds = torch.zeros(1, n_tokens, n_tokens, 1, device=device)
    for i in range(n_tokens - 1):
        # Check if consecutive tokens are in the same chain
        # Find atoms belonging to token i and i+1
        atoms_i = np.where(residue_indices == i)[0]
        atoms_i1 = np.where(residue_indices == i + 1)[0]
        if len(atoms_i) > 0 and len(atoms_i1) > 0:
            if chain_ids[atoms_i[0]] == chain_ids[atoms_i1[0]]:
                token_bonds[0, i, i + 1, 0] = 1.0
                token_bonds[0, i + 1, i, 0] = 1.0
    feats["token_bonds"] = token_bonds

    # Residue index (for relative position encoding)
    # Map: for each token, use the residue index
    token_res_idx = torch.zeros(1, n_tokens, dtype=torch.long, device=device)
    for tok in range(n_tokens):
        atom_indices = np.where(residue_indices == tok)[0]
        if len(atom_indices) > 0:
            token_res_idx[0, tok] = int(residue_indices[atom_indices[0]])
    feats["residue_index"] = token_res_idx

    # Chain/entity IDs
    token_asym = torch.zeros(1, n_tokens, dtype=torch.long, device=device)
    token_entity = torch.zeros(1, n_tokens, dtype=torch.long, device=device)
    token_mol_type = torch.zeros(1, n_tokens, dtype=torch.long, device=device)
    for tok in range(n_tokens):
        atom_indices = np.where(residue_indices == tok)[0]
        if len(atom_indices) > 0:
            token_asym[0, tok] = int(chain_ids[atom_indices[0]])
            token_entity[0, tok] = int(chain_ids[atom_indices[0]])
            token_mol_type[0, tok] = int(mol_types[atom_indices[0]])
    feats["asym_id"] = token_asym
    feats["entity_id"] = token_entity
    feats["sym_id"] = token_asym.clone()
    feats["mol_type"] = token_mol_type

    # Residue type (one-hot, num_classes=33 for Boltz-2)
    # For simplicity, use zeros (the trunk will still compute embeddings)
    num_res_types = 33
    feats["res_type"] = torch.zeros(
        1, n_tokens, num_res_types, device=device
    )

    # Token index
    feats["token_index"] = torch.arange(n_tokens, device=device).unsqueeze(0)

    # Atom features (padded to n_atoms_padded; padding stays zero)
    ref_pos = torch.zeros(1, n_atoms_padded, 3, device=device)
    ref_pos[0, :n_atoms] = torch.from_numpy(ref_coords).float().to(device)
    feats["ref_pos"] = ref_pos

    atom_pad_mask = torch.zeros(1, n_atoms_padded, device=device)
    atom_pad_mask[0, :n_atoms] = 1.0
    feats["atom_pad_mask"] = atom_pad_mask

    # Atom-to-token mapping
    atom_to_token = torch.zeros(1, n_atoms_padded, n_tokens, device=device)
    for i, ri in enumerate(residue_indices):
        if 0 <= ri < n_tokens:
            atom_to_token[0, i, ri] = 1.0
    feats["atom_to_token"] = atom_to_token

    # Contact conditioning (zeros = no contacts)
    feats["contact_conditioning"] = torch.zeros(
        1, n_tokens, n_tokens, device=device
    )
    feats["contact_threshold"] = torch.zeros(
        1, n_tokens, n_tokens, device=device
    )

    # MSA features (dummy single-sequence MSA)
    feats["msa"] = torch.zeros(1, 1, n_tokens, dtype=torch.long, device=device)
    feats["msa_mask"] = torch.ones(1, 1, n_tokens, dtype=torch.long, device=device)
    feats["msa_paired"] = torch.zeros(
        1, 1, n_tokens, dtype=torch.long, device=device
    )
    feats["has_deletion"] = torch.zeros(
        1, 1, n_tokens, dtype=torch.bool, device=device
    )
    feats["deletion_value"] = torch.zeros(1, 1, n_tokens, device=device)
    feats["deletion_mean"] = torch.zeros(1, n_tokens, device=device)
    feats["profile"] = torch.zeros(1, n_tokens, num_res_types, device=device)

    # Atom name/element features (for input embedder)
    # Boltz-2 encodes atom names as 4 characters, each ord(c)-32, one-hot 64
    name_chars = np.zeros((n_atoms_padded, 4), dtype=np.int64)
    for i, name in enumerate(atom_names):
        chars = [ord(c) - 32 for c in str(name).strip()]
        for j, c in enumerate(chars[:4]):
            name_chars[i, j] = c
    name_chars_t = torch.from_numpy(name_chars).to(device)
    feats["ref_atom_name_chars"] = torch.nn.functional.one_hot(
        name_chars_t, num_classes=64
    ).float().unsqueeze(0)  # (1, n_atoms_padded, 4, 64)

    # Boltz-2 encodes elements by atomic number, one-hot 128
    _element_to_z = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7,
        "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13,
        "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19,
        "Ca": 20, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
        "Zn": 30, "Se": 34, "Br": 35, "I": 53,
    }
    elem_idx = np.zeros(n_atoms_padded, dtype=np.int64)
    for i, e in enumerate(elements):
        elem_idx[i] = _element_to_z.get(str(e).strip(), 0)
    elem_t = torch.from_numpy(elem_idx).to(device)
    feats["ref_element"] = torch.nn.functional.one_hot(
        elem_t, num_classes=128
    ).float().unsqueeze(0)  # (1, n_atoms_padded, 128)

    feats["ref_charge"] = torch.zeros(1, n_atoms_padded, device=device)
    feats["ref_chirality"] = torch.zeros(
        1, n_atoms_padded, dtype=torch.long, device=device
    )
    ref_space_uid = torch.zeros(1, n_atoms_padded, dtype=torch.long, device=device)
    ref_space_uid[0, :n_atoms] = torch.from_numpy(
        residue_indices.astype(np.int64)
    ).to(device)
    feats["ref_space_uid"] = ref_space_uid

    atom_resolved_mask = torch.zeros(1, n_atoms_padded, device=device)
    atom_resolved_mask[0, :n_atoms] = 1.0
    feats["atom_resolved_mask"] = atom_resolved_mask

    # Disto features
    feats["disto_center"] = torch.zeros(1, n_tokens, 3, device=device)
    feats["token_resolved_mask"] = torch.ones(1, n_tokens, device=device)
    feats["token_disto_mask"] = torch.zeros(1, n_tokens, device=device)

    # Template features (dummy)
    tdim = 4  # Boltz-2 default template dimension
    feats["template_restype"] = torch.zeros(
        1, tdim, n_tokens, num_res_types, device=device
    )
    feats["template_frame_rot"] = torch.zeros(
        1, tdim, n_tokens, 3, 3, device=device
    )
    feats["template_frame_t"] = torch.zeros(
        1, tdim, n_tokens, 3, device=device
    )
    feats["template_cb"] = torch.zeros(1, tdim, n_tokens, 3, device=device)
    feats["template_ca"] = torch.zeros(1, tdim, n_tokens, 3, device=device)
    feats["template_mask_cb"] = torch.zeros(1, tdim, n_tokens, device=device)
    feats["template_mask_frame"] = torch.zeros(1, tdim, n_tokens, device=device)
    feats["template_mask"] = torch.zeros(1, tdim, n_tokens, device=device)
    feats["query_to_template"] = torch.zeros(
        1, tdim, n_tokens, dtype=torch.long, device=device
    )
    feats["visibility_ids"] = torch.zeros(1, tdim, n_tokens, device=device)

    # Method feature, modified, cyclic
    feats["method_feature"] = torch.zeros(
        1, n_tokens, dtype=torch.long, device=device
    )
    feats["modified"] = torch.zeros(1, n_tokens, dtype=torch.long, device=device)
    feats["cyclic_period"] = torch.zeros(1, n_tokens, device=device)

    return feats


def precompute_all(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    recycling_steps: int = 3,
) -> None:
    """Precompute trunk embeddings for all systems in a manifest.

    Parameters
    ----------
    manifest_path : path to JSON manifest.
    checkpoint_path : path to Boltz-2 checkpoint.
    output_dir : where to save trunk .npz files.
    device : compute device.
    recycling_steps : number of recycling iterations.
    """
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        entries = json.load(f)

    logger.info("Loading Boltz-2 model from %s", checkpoint_path)
    model = load_boltz2_model(checkpoint_path, device)

    n_skipped = 0
    for entry in tqdm(entries, desc="Precomputing trunk"):
        system_id = entry["system_id"]
        out_path = output_dir / f"{system_id}_trunk.npz"

        if out_path.exists():
            n_skipped += 1
            continue

        try:
            feats = prepare_features_for_system(
                entry["ref_path"],
                entry["coords_path"],
                device=device,
            )

            with torch.no_grad(), torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                s_inputs, s_trunk, z_trunk = run_trunk(
                    model, feats, recycling_steps
                )

            save_trunk_embeddings(
                system_id, s_inputs, s_trunk, z_trunk, output_dir
            )

        except Exception:
            logger.exception("Failed to precompute trunk for %s", system_id)

    if n_skipped:
        logger.info("Skipped %d existing trunk caches", n_skipped)

    # Update manifest with trunk_cache_dir
    for entry in entries:
        entry["trunk_cache_dir"] = str(output_dir)
    with open(manifest_path, "w") as f:
        json.dump(entries, f, indent=2)
    logger.info("Updated manifest with trunk_cache_dir")
