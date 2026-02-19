"""Inference entry point for trajectory generation.

Usage:
    python scripts/generate.py --config configs/inference.yaml --input structure.pdb

    # Override config parameters:
    python scripts/generate.py --config configs/inference.yaml --input structure.pdb \
        mode=unbinding total_frames=200

    # Hierarchical mode:
    python scripts/generate.py --config configs/inference.yaml --input structure.pdb \
        hierarchical=true
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from kinematic.model.checkpoint_io import (
    find_model_weights_file,
    has_unresolved_step_placeholder,
    load_model_state_dict,
    resolve_checkpoint_path,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from kinematic.inference.sampler import EDMSampler


# ------------------------------------------------------------------
# Inference config
# ------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """Inference configuration with defaults matching configs/inference.yaml."""

    # Model architecture
    token_s: int = 384
    token_z: int = 128
    atom_s: int = 128
    atom_z: int = 16
    atom_feature_dim: int = 128
    atoms_per_window_queries: int = 32
    atoms_per_window_keys: int = 128
    sigma_data: float = 16.0
    dim_fourier: int = 256
    atom_encoder_depth: int = 3
    atom_encoder_heads: int = 4
    atom_temporal_heads: int = 4
    token_transformer_depth: int = 24
    token_transformer_heads: int = 16
    token_temporal_heads: int = 16
    atom_decoder_depth: int = 3
    atom_decoder_heads: int = 4
    conditioning_transition_layers: int = 2
    causal: bool = False

    # Sampling
    sigma_min: float = 0.0001
    sigma_max: float = 160.0
    rho: float = 7.0
    n_denoise_steps: int = 20
    noise_scale: float = 1.75
    step_scale: float = 1.5

    # Generation
    mode: str = "equilibrium"  # equilibrium | unbinding
    n_frames: int = 32
    n_cond_frames: int = 4
    total_frames: int = 100
    dt_ns: float = 1.0

    # Hierarchical sampling
    hierarchical: bool = False
    coarse_dt_ns: float = 10.0
    fine_dt_ns: float = 0.1
    generation_window: int = 40
    history_window: int = 10

    # Unbinding-specific
    dt_ps: float = 10.0
    total_ps: float = 1000.0
    n_trajectories: int = 5

    # Paths
    checkpoint: str = "checkpoints/phase1/step_XXXXX"
    trunk_cache_dir: str = "data/processed/trunk_embeddings/"
    output_dir: str = "outputs/"

    # Device
    device: str = "cuda"
    dtype: str = "float32"

    # Seed
    seed: int = 42


def _normalize_legacy_inference_keys(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    """Map deprecated config keys to canonical schema with warnings."""
    cfg = dict(raw_cfg)

    if "coarse_n_frames" in cfg:
        coarse_n_frames = cfg.pop("coarse_n_frames")
        if "generation_window" not in cfg:
            cfg["generation_window"] = int(coarse_n_frames)
        logger.warning(
            "Config key 'coarse_n_frames' is deprecated; use 'generation_window'."
        )

    if "fine_interp_factor" in cfg:
        fine_interp_factor = float(cfg.pop("fine_interp_factor"))
        if fine_interp_factor <= 0:
            raise ValueError(
                f"fine_interp_factor must be > 0, got {fine_interp_factor}"
            )
        if "fine_dt_ns" not in cfg:
            coarse_dt = float(cfg.get("coarse_dt_ns", InferenceConfig.coarse_dt_ns))
            cfg["fine_dt_ns"] = coarse_dt / fine_interp_factor
        logger.warning(
            "Config key 'fine_interp_factor' is deprecated; use 'fine_dt_ns'."
        )

    return cfg


def _build_config(cfg: DictConfig, *, strict: bool = True) -> InferenceConfig:
    """Build ``InferenceConfig`` from OmegaConf with optional strict schema checks."""
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(
            f"Expected inference config to deserialize to a mapping, got {type(raw).__name__}"
        )
    normalized = _normalize_legacy_inference_keys(raw)

    config_fields = {f.name: f for f in fields(InferenceConfig)}
    unknown_keys = sorted(set(normalized) - set(config_fields))
    if strict and unknown_keys:
        raise ValueError(
            "Unknown inference config keys: "
            + ", ".join(unknown_keys)
            + ". Update keys to match InferenceConfig."
        )

    missing_required = sorted(
        name
        for name, f in config_fields.items()
        if f.default is MISSING
        and f.default_factory is MISSING
        and name not in normalized
    )
    if strict and missing_required:
        raise ValueError(
            "Missing required inference config keys: " + ", ".join(missing_required)
        )

    kwargs: dict[str, Any] = {}
    for name in config_fields:
        if name not in normalized:
            continue
        value = normalized[name]
        if isinstance(value, dict):
            value = {str(k): v for k, v in value.items()}
        kwargs[name] = value
    return InferenceConfig(**kwargs)


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def _build_model(config: InferenceConfig) -> torch.nn.Module:
    """Build and load trained Kinematic model."""
    from kinematic.model.kinematic import Kinematic

    model = Kinematic(
        token_s=config.token_s,
        token_z=config.token_z,
        atom_s=config.atom_s,
        atom_z=config.atom_z,
        atoms_per_window_queries=config.atoms_per_window_queries,
        atoms_per_window_keys=config.atoms_per_window_keys,
        atom_encoder_depth=config.atom_encoder_depth,
        atom_encoder_heads=config.atom_encoder_heads,
        atom_temporal_heads=config.atom_temporal_heads,
        token_transformer_depth=config.token_transformer_depth,
        token_transformer_heads=config.token_transformer_heads,
        token_temporal_heads=config.token_temporal_heads,
        atom_decoder_depth=config.atom_decoder_depth,
        atom_decoder_heads=config.atom_decoder_heads,
        sigma_data=config.sigma_data,
        dim_fourier=config.dim_fourier,
        atom_feature_dim=config.atom_feature_dim,
        conditioning_transition_layers=config.conditioning_transition_layers,
        causal=config.causal,
    )

    # Load trained checkpoint
    had_placeholder = has_unresolved_step_placeholder(config.checkpoint)
    ckpt_path = resolve_checkpoint_path(
        config.checkpoint,
        auto_resolve_latest=True,
    )
    if had_placeholder:
        logger.warning(
            "Resolved checkpoint placeholder: %s -> %s",
            config.checkpoint,
            ckpt_path,
        )

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            "Provide a valid path via --config or checkpoint=..., "
            "or point to a parent with step_* directories when using step_XXXXX."
        )

    # Support Accelerate checkpoints (pytorch_model.bin / model.safetensors)
    # and direct state_dict files
    ckpt_dir = Path(ckpt_path)
    if ckpt_dir.is_dir():
        # Accelerate checkpoint directory
        model_file = find_model_weights_file(ckpt_dir)
        if model_file is None:
            raise FileNotFoundError(
                f"No model weights found in checkpoint directory: {ckpt_path}"
            )
        state = load_model_state_dict(model_file, map_location="cpu")
    else:
        state = load_model_state_dict(ckpt_path, map_location="cpu")

    model.load_state_dict(state, strict=True)
    logger.info("Loaded model from %s", ckpt_path)

    return model


# ------------------------------------------------------------------
# Input loading
# ------------------------------------------------------------------

def _load_initial_structure(
    input_path: str,
    trunk_cache_dir: str,
) -> dict[str, Any]:
    """Load initial structure and precomputed trunk embeddings.

    Supports two input formats:

    1. NPZ file with preprocessed data (coords, ref_coords, etc.)
       - Must have a corresponding trunk cache file.

    2. Directory containing coords.npz + ref.npz + trunk.npz
       - Standard preprocessed system directory.

    Parameters
    ----------
    input_path : str
        Path to input structure (npz file or directory).
    trunk_cache_dir : str
        Directory containing precomputed trunk .npz files.

    Returns
    -------
    dict with keys:
      - initial_coords: (M, 3) tensor in Angstrom
      - s_trunk, z_trunk, s_inputs: trunk tensors
      - feats: atom-level features dict
      - atom_pad_mask, token_pad_mask, atom_to_token: masks
      - system_id: string identifier
    """
    from kinematic.data.trunk_cache import load_trunk_embeddings

    input_path = Path(os.path.expanduser(input_path))

    if input_path.is_dir():
        # Directory with preprocessed files
        coords_file = input_path / "coords.npz"
        ref_file = input_path / "ref.npz"
        system_id = input_path.name

        if not coords_file.exists():
            raise FileNotFoundError(f"coords.npz not found in {input_path}")
        if not ref_file.exists():
            raise FileNotFoundError(f"ref.npz not found in {input_path}")

        coords_data = np.load(coords_file)
        ref_data = dict(np.load(ref_file, allow_pickle=False))

        # Use first frame as initial structure
        initial_coords = torch.from_numpy(
            coords_data["coords"][0].astype(np.float32)
        )
    elif input_path.suffix == ".npz":
        system_id = input_path.stem
        data = np.load(input_path, allow_pickle=False)

        if "coords" in data:
            initial_coords = torch.from_numpy(data["coords"][0].astype(np.float32))
        else:
            raise KeyError(f"Expected 'coords' key in {input_path}")

        ref_data = {
            k: data[k] for k in data.files if k != "coords"
        }
    else:
        raise ValueError(
            f"Unsupported input format: {input_path.suffix}. "
            "Expected .npz file or preprocessed directory."
        )

    # Load trunk embeddings
    trunk = load_trunk_embeddings(system_id, trunk_cache_dir)

    # Build feats dict
    feats: dict[str, torch.Tensor] = {}
    if "ref_coords" in ref_data:
        feats["ref_pos"] = torch.from_numpy(ref_data["ref_coords"].astype(np.float32))
    if "mol_types" in ref_data:
        feats["mol_types"] = torch.from_numpy(ref_data["mol_types"].astype(np.int64))
    if "residue_indices" in ref_data:
        feats["residue_indices"] = torch.from_numpy(
            ref_data["residue_indices"].astype(np.int64)
        )
    if "chain_ids" in ref_data:
        feats["chain_ids"] = torch.from_numpy(ref_data["chain_ids"].astype(np.int64))

    M = initial_coords.shape[0]
    N = trunk["s_trunk"].shape[0]

    # Build masks
    atom_pad_mask = torch.ones(M, dtype=torch.bool)
    token_pad_mask = torch.ones(N, dtype=torch.bool)

    # Atom-to-token mapping
    if "atom_to_token" in ref_data:
        atom_to_token = torch.from_numpy(ref_data["atom_to_token"].astype(np.float32))
    elif "residue_indices" in ref_data:
        res_idx = ref_data["residue_indices"]
        mat = np.zeros((M, N), dtype=np.float32)
        for i, ri in enumerate(res_idx):
            if 0 <= ri < N:
                mat[i, ri] = 1.0
        atom_to_token = torch.from_numpy(mat)
    else:
        raise KeyError("Need 'atom_to_token' or 'residue_indices' in reference data")

    # Add batch-level keys to feats for model compatibility
    # (unsqueezed to batch dim=1 when passed to model)
    feats["atom_pad_mask"] = atom_pad_mask.unsqueeze(0)
    feats["token_pad_mask"] = token_pad_mask.unsqueeze(0)
    feats["atom_to_token"] = atom_to_token.unsqueeze(0)

    # Unsqueeze 1D/2D feats to have batch dim
    for key in ("ref_pos", "mol_types", "residue_indices", "chain_ids"):
        if key in feats:
            feats[key] = feats[key].unsqueeze(0)

    return {
        "initial_coords": initial_coords,
        "s_trunk": trunk["s_trunk"],
        "z_trunk": trunk["z_trunk"],
        "s_inputs": trunk["s_inputs"],
        "feats": feats,
        "atom_pad_mask": atom_pad_mask,
        "token_pad_mask": token_pad_mask,
        "atom_to_token": atom_to_token,
        "system_id": system_id,
    }


# ------------------------------------------------------------------
# Auto-regressive flat generation (non-hierarchical)
# ------------------------------------------------------------------

def _generate_flat(
    sampler: EDMSampler,
    initial_structure: torch.Tensor,
    config: InferenceConfig,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    s_inputs: torch.Tensor,
    feats: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Auto-regressive flat trajectory generation.

    Generates frames in windows, using the last n_cond_frames as
    conditioning for the next window.

    Returns
    -------
    trajectory : (N_frames, M, 3)
    timestamps : (N_frames,)
    """
    device = initial_structure.device
    trajectory: list[torch.Tensor] = [initial_structure]
    timestamps: list[float] = [0.0]

    generated = 0
    while generated < config.total_frames:
        n_gen = min(config.n_frames, config.total_frames - generated)

        # Build conditioning from recent history
        n_cond = min(config.n_cond_frames, len(trajectory))
        x_cond = torch.stack(trajectory[-n_cond:])
        t_cond = torch.tensor(timestamps[-n_cond:], device=device)

        # Target timestamps
        t_start = timestamps[-1] + config.dt_ns
        t_target = torch.tensor(
            [t_start + i * config.dt_ns for i in range(n_gen)],
            device=device,
        )

        # Sample noise
        x_init = (
            torch.randn(n_gen, *initial_structure.shape, device=device)
            * sampler.sigma_max
        )

        x_gen = sampler.sample(
            x_cond=x_cond,
            t_cond=t_cond,
            x_init=x_init,
            t_target=t_target,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            s_inputs=s_inputs,
            feats=feats,
            mode="forecast",
        )

        for i in range(n_gen):
            trajectory.append(x_gen[i])
            timestamps.append(t_target[i].item())
        generated += n_gen

    traj_tensor = torch.stack(trajectory)
    time_tensor = torch.tensor(timestamps, device=device)
    return traj_tensor, time_tensor


# ------------------------------------------------------------------
# Save trajectory
# ------------------------------------------------------------------

def _save_trajectory(
    trajectory: torch.Tensor,
    timestamps: torch.Tensor,
    output_dir: str,
    system_id: str,
) -> Path:
    """Save generated trajectory to NPZ file.

    Parameters
    ----------
    trajectory : (N_frames, M, 3) coordinates in Angstrom.
    timestamps : (N_frames,) timestamps in nanoseconds.
    output_dir : output directory path.
    system_id : system identifier for filename.

    Returns
    -------
    Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{system_id}_trajectory.npz"
    np.savez_compressed(
        out_path,
        coords=trajectory.cpu().numpy(),
        timestamps=timestamps.cpu().numpy(),
    )
    return out_path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Kinematic trajectory generation"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to inference YAML config",
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input structure (npz or directory)",
    )
    args, overrides = parser.parse_known_args()

    # Load config
    cfg = OmegaConf.load(args.config)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    config = _build_config(cfg)

    # Set seed
    torch.manual_seed(config.seed)

    # Device and dtype
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(config.dtype, torch.float32)

    logger.info("Device: %s, dtype: %s", device, dtype)

    # Load model
    model = _build_model(config)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    logger.info("Model loaded and set to eval mode")

    # Load input
    inputs = _load_initial_structure(args.input, config.trunk_cache_dir)
    initial_coords = inputs["initial_coords"].to(device=device, dtype=dtype)
    s_trunk = inputs["s_trunk"].to(device=device, dtype=dtype)
    z_trunk = inputs["z_trunk"].to(device=device, dtype=dtype)
    s_inputs = inputs["s_inputs"].to(device=device, dtype=dtype)
    feats = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
        for k, v in inputs["feats"].items()
    }
    system_id = inputs["system_id"]

    logger.info(
        "Input: %s (%d atoms, %d tokens)",
        system_id, initial_coords.shape[0], s_trunk.shape[0],
    )

    # Build sampler
    from kinematic.inference.sampler import EDMSampler

    sampler = EDMSampler(
        model=model,
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        sigma_data=config.sigma_data,
        rho=config.rho,
        n_steps=config.n_denoise_steps,
        noise_scale=config.noise_scale,
        step_scale=config.step_scale,
    )

    # Generate
    t0 = time.time()

    if config.mode == "unbinding":
        from kinematic.inference.unbinding import UnbindingGenerator

        generator = UnbindingGenerator(
            sampler=sampler,
            dt_ps=config.dt_ps,
        )
        trajectories = generator.generate(
            complex_structure=initial_coords,
            total_ps=config.total_ps,
            n_trajectories=config.n_trajectories,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            s_inputs=s_inputs,
            feats=feats,
        )
        # Save each trajectory separately
        for i in range(trajectories.shape[0]):
            n_frames = trajectories.shape[1]
            ts = torch.arange(n_frames, device=device) * config.dt_ps / 1000.0
            out_path = _save_trajectory(
                trajectories[i], ts, config.output_dir,
                f"{system_id}_unbind_{i}",
            )
            logger.info("Saved trajectory %d to %s", i, out_path)

    elif config.hierarchical:
        from kinematic.inference.hierarchical import HierarchicalGenerator

        generator = HierarchicalGenerator(
            sampler=sampler,
            coarse_dt_ns=config.coarse_dt_ns,
            fine_dt_ns=config.fine_dt_ns,
            generation_window=config.generation_window,
            history_window=config.history_window,
        )
        total_time_ns = config.total_frames * config.dt_ns
        trajectory, timestamps = generator.generate(
            initial_structure=initial_coords,
            total_time_ns=total_time_ns,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            s_inputs=s_inputs,
            feats=feats,
        )
        out_path = _save_trajectory(
            trajectory, timestamps, config.output_dir, system_id,
        )
        logger.info("Saved trajectory to %s", out_path)

    else:
        # Flat auto-regressive generation
        trajectory, timestamps = _generate_flat(
            sampler=sampler,
            initial_structure=initial_coords,
            config=config,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            s_inputs=s_inputs,
            feats=feats,
        )
        out_path = _save_trajectory(
            trajectory, timestamps, config.output_dir, system_id,
        )
        logger.info("Saved trajectory to %s", out_path)

    elapsed = time.time() - t0
    logger.info("Generation complete in %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
