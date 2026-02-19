"""Accelerate training loop."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.utils.data
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig

from boltzkinema.data.collator import BoltzKinemaCollator
from boltzkinema.data.dataset import BoltzKinemaDataset
from boltzkinema.model.boltzkinema import BoltzKinema
from boltzkinema.model.checkpoint_io import (
    find_model_weights_file,
    load_model_state_dict,
)
from boltzkinema.model.weight_loading import load_boltz2_weights
from boltzkinema.training.losses import BoltzKinemaLoss
from boltzkinema.training.scheduler import get_warmup_constant_scheduler

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Config dataclass — populated from OmegaConf DictConfig
# ------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training configuration with defaults matching WORKPLAN Phase 0."""

    # Model
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
    activation_checkpointing: bool = True
    freeze_diffusion_conditioning: bool = False
    causal: bool = False

    # Training
    training_mode: str = "equilibrium"
    lr: float = 1e-4
    warmup_steps: int = 200
    max_steps: int = 150_000
    max_epochs: int = 1000
    batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 4
    grad_clip: float = 10.0
    seed: int = 42
    n_frames: int = 32
    P_mean: float = -1.2
    P_std: float = 1.5
    forecast_prob: float = 0.5
    num_workers: int = 4

    # Loss
    alpha_bond: float = 1.0
    beta_flex: float = 1.0
    beta_abs: float = 1.0
    beta_rel_g: float = 4.0
    beta_rel_l: float = 4.0
    beta_center: float = 1.0
    mol_weights: dict[str, float] = field(
        default_factory=lambda: {
            "protein": 1.0,
            "dna": 5.0,
            "rna": 5.0,
            "ligand": 10.0,
        }
    )

    # Data
    dataset_weights: dict[str, float] = field(default_factory=dict)
    dt_ranges: dict[str, list[float]] = field(default_factory=dict)

    # Paths
    boltz2_checkpoint: str = "~/.boltz/boltz2_conf.ckpt"
    manifest_path: str = "data/processed/manifest.json"
    trunk_cache_dir: str = "data/processed/trunk_embeddings/"
    coords_dir: str = "data/processed/coords/"
    output_dir: str = "checkpoints/phase0/"
    log_every: int = 50
    save_every: int = 5000

    # Resume
    resume_from: str | None = None
    resume_optimizer: bool = True


def _validate_config(config: TrainConfig) -> list[str]:
    """Validate training config and return list of warnings.

    Raises ValueError for invalid configurations.
    """
    warnings = []

    # training_mode must be valid
    if config.training_mode not in ("equilibrium", "unbinding"):
        raise ValueError(
            f"Invalid training_mode: {config.training_mode!r}. "
            "Must be 'equilibrium' or 'unbinding'."
        )

    # Unbinding mode should use causal temporal attention
    if config.training_mode == "unbinding" and not config.causal:
        warnings.append(
            "training_mode='unbinding' typically requires causal=true "
            "for causal temporal attention (metadynamics data is time-ordered)"
        )

    # Causal mode is unusual for equilibrium training
    if config.training_mode == "equilibrium" and config.causal:
        warnings.append(
            "causal=true with training_mode='equilibrium' is unusual. "
            "Equilibrium training typically uses bidirectional temporal attention."
        )

    # resume_from path validation
    if config.resume_from:
        resume_path = Path(os.path.expanduser(config.resume_from))
        if not resume_path.exists():
            warnings.append(
                f"resume_from path does not exist: {config.resume_from}. "
                "Training will fail at checkpoint loading."
            )
        # Cross-phase resume should not carry optimizer state
        if config.resume_optimizer:
            # Check if output_dir differs from resume_from parent
            resume_phase = str(resume_path.parent)
            output_phase = os.path.expanduser(config.output_dir)
            if resume_phase != output_phase:
                warnings.append(
                    f"resume_optimizer=true but resuming from a different phase "
                    f"({resume_phase} -> {output_phase}). "
                    "Cross-phase transitions should use resume_optimizer=false "
                    "to start with a fresh optimizer."
                )

    # Dataset validation
    if not config.dataset_weights:
        warnings.append("No dataset_weights specified; all datasets equally weighted")

    # Step count sanity
    if config.max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {config.max_steps}")

    if config.warmup_steps >= config.max_steps:
        warnings.append(
            f"warmup_steps ({config.warmup_steps}) >= max_steps ({config.max_steps}); "
            "learning rate will never reach full value"
        )

    return warnings


def _estimate_memory_gb(
    config: TrainConfig,
    n_tokens: int,
    n_atoms: int,
    n_params: int = 215_000_000,
) -> float:
    """Quick memory estimate in GB for a single sample.

    Uses the model config dimensions plus representative N, M values
    to estimate peak GPU memory per sample during training.

    Parameters
    ----------
    config : TrainConfig
    n_tokens : Representative token count (N).
    n_atoms : Representative atom count (M).
    n_params : Approximate model parameter count.
    """
    T = config.n_frames
    N = n_tokens
    M = n_atoms
    d = 2 * config.token_s
    W_q, W_k = 32, 128
    nw = max(1, M // W_q)
    b = 2  # bf16 bytes

    # Atom layers (NOT checkpointed): full intermediates stored
    atom_depth = config.atom_encoder_depth + config.atom_decoder_depth
    atom_bytes = atom_depth * (
        8 * T * nw * W_q * config.atom_s * b
        + T * nw * config.atom_encoder_heads * W_q * W_k * b
        + T * nw * W_q * 4 * config.atom_s * b
        + 6 * M * T * config.atom_s * b
        + M * config.atom_temporal_heads * T * T * 4
    )

    # Token transformer (checkpointed): stored inputs + peak recompute
    token_bytes = (
        config.token_transformer_depth * T * N * d * b  # stored inputs
        + 5 * T * N * d * b
        + T * config.token_transformer_heads * N * N * b
        + T * N * 4 * d * b
        + 5 * N * T * d * b
        + N * config.token_temporal_heads * T * T * 4
    )

    # Backward overhead (gradient-of-activation storage + transient peaks)
    activation_gb = (atom_bytes + token_bytes) * 2.0 / (1024**3)

    # Fixed overhead: params (fp32) + grads (bf16) + optimizer (Adam fp32)
    fixed_gb = n_params * (4 + 2 + 8) / (1024**3)

    return activation_gb + fixed_gb


def _build_config(cfg: DictConfig) -> TrainConfig:
    """Build TrainConfig from an OmegaConf DictConfig.

    Handles type conversions for fields that OmegaConf may represent
    differently (e.g., DictConfig → plain dict).
    """
    kwargs: dict = {}
    for f in TrainConfig.__dataclass_fields__:
        if f in cfg:
            val = cfg[f]
            # Convert OmegaConf containers to plain Python
            if hasattr(val, "items"):
                val = {str(k): v for k, v in val.items()}
                # Convert nested lists (dt_ranges values)
                for k, v in val.items():
                    if hasattr(v, "__iter__") and not isinstance(v, str):
                        val[k] = list(v)
            kwargs[f] = val
    return TrainConfig(**kwargs)


def _build_model(config: TrainConfig) -> BoltzKinema:
    """Instantiate the BoltzKinema model from config."""
    return BoltzKinema(
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
        activation_checkpointing=config.activation_checkpointing,
        causal=config.causal,
    )


def _build_loss(config: TrainConfig) -> BoltzKinemaLoss:
    """Instantiate the loss module from config."""
    return BoltzKinemaLoss(
        sigma_data=config.sigma_data,
        alpha_bond=config.alpha_bond,
        beta_flex=config.beta_flex,
        beta_abs=config.beta_abs,
        beta_rel_g=config.beta_rel_g,
        beta_rel_l=config.beta_rel_l,
        beta_center=config.beta_center,
        mol_weights=config.mol_weights,
    )


def _build_dataset(config: TrainConfig) -> BoltzKinemaDataset:
    """Instantiate the training dataset from config."""
    return BoltzKinemaDataset(
        manifest_path=config.manifest_path,
        trunk_cache_dir=config.trunk_cache_dir,
        coords_dir=config.coords_dir,
        n_frames=config.n_frames,
        dataset_weights=config.dataset_weights,
        dt_ranges=config.dt_ranges,
        noise_P_mean=config.P_mean,
        noise_P_std=config.P_std,
        sigma_data=config.sigma_data,
        forecast_prob=config.forecast_prob,
        seed=config.seed,
    )


def _save_checkpoint(
    accelerator: Accelerator,
    config: TrainConfig,
    global_step: int,
) -> None:
    """Save an Accelerate checkpoint to output_dir/step_{global_step}."""
    save_dir = os.path.join(config.output_dir, f"step_{global_step}")
    accelerator.save_state(save_dir)
    # Also save a metadata file with the step count for easy resume
    if accelerator.is_main_process:
        meta_path = os.path.join(save_dir, "metadata.pt")
        torch.save({"global_step": global_step}, meta_path)
    accelerator.print(f"Checkpoint saved to {save_dir}")


def _load_checkpoint(
    accelerator: Accelerator,
    config: TrainConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> int:
    """Resume from a checkpoint directory.

    Supports two modes:
      - resume_optimizer=True: full resume (same phase, interrupted run)
      - resume_optimizer=False: model-only resume (new phase, fresh optimizer)

    Returns the global step to resume from.
    """
    resume_dir = config.resume_from
    accelerator.print(f"Resuming from {resume_dir}")

    if config.resume_optimizer:
        # Full resume: model + optimizer + scheduler state
        accelerator.load_state(resume_dir)
        # Read step count from metadata
        meta_path = os.path.join(resume_dir, "metadata.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location="cpu", weights_only=True)
            global_step = meta["global_step"]
        else:
            # Fallback: parse from directory name
            global_step = int(Path(resume_dir).name.split("step_")[-1])
        accelerator.print(f"Full resume from step {global_step}")
        return global_step
    else:
        # Model-only resume: load model weights, keep fresh optimizer/scheduler
        # We need to load the model state from the Accelerate checkpoint
        # Accelerate checkpoints store model in a pytorch_model.bin or
        # model.safetensors file inside the checkpoint directory
        model_file = find_model_weights_file(resume_dir)

        if model_file is not None:
            # Use Accelerate's unwrapped model for direct weight loading
            unwrapped = accelerator.unwrap_model(model)
            state = load_model_state_dict(model_file, map_location="cpu")
            unwrapped.load_state_dict(state, strict=True)
            accelerator.print("Loaded model weights (fresh optimizer)")
        else:
            # Fallback: full state load, then reset optimizer + scheduler
            accelerator.load_state(resume_dir)
            # Reset optimizer state
            for group in optimizer.param_groups:
                for p in group["params"]:
                    state = optimizer.state.get(p, {})
                    state.clear()
            # Reset scheduler to step 0
            scheduler.last_epoch = -1
            scheduler.step()
            accelerator.print(
                "Loaded full checkpoint; reset optimizer and scheduler"
            )

        return 0


# ------------------------------------------------------------------
# Main training function
# ------------------------------------------------------------------

def train(cfg: DictConfig) -> None:
    """Main training loop using HuggingFace Accelerate.

    Handles:
    - Multi-GPU distributed training (DDP)
    - Mixed precision (bf16)
    - Gradient accumulation
    - Gradient clipping
    - Checkpoint saving/resuming (full or model-only)
    - WandB logging

    Parameters
    ----------
    cfg : DictConfig
        OmegaConf config loaded from a YAML file.
    """
    config = _build_config(cfg)

    # Validate config before expensive initialization
    config_warnings = _validate_config(config)

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
    )

    set_seed(config.seed)

    # Print config warnings
    for warning in config_warnings:
        accelerator.print(f"WARNING: {warning}")

    # Initialize WandB (main process only)
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="boltzkinema",
            config={
                f.name: getattr(config, f.name)
                for f in config.__dataclass_fields__.values()
                if not isinstance(getattr(config, f.name), (dict,))
            },
        )

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = _build_model(config)

    # Load Boltz-2 pretrained weights (before any checkpoint resume)
    checkpoint_path = os.path.expanduser(config.boltz2_checkpoint)
    if os.path.exists(checkpoint_path):
        load_result = load_boltz2_weights(model, checkpoint_path)
        accelerator.print(
            f"Loaded {len(load_result['matched'])} Boltz-2 weights, "
            f"{len(load_result['temporal'])} temporal params verified"
        )
    else:
        accelerator.print(
            f"WARNING: Boltz-2 checkpoint not found at {checkpoint_path}, "
            "starting with random initialization"
        )

    # Freeze DiffusionConditioning if configured
    if config.freeze_diffusion_conditioning:
        for param in model.diffusion_conditioning.parameters():
            param.requires_grad = False
        accelerator.print("Froze DiffusionConditioning parameters")

    # Count parameters and estimate memory
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    accelerator.print(
        f"Parameters: {n_trainable / 1e6:.1f}M trainable / {n_total / 1e6:.1f}M total"
    )

    # Memory estimate (using representative token/atom counts)
    est_gb = _estimate_memory_gb(config, n_tokens=200, n_atoms=1500, n_params=n_total)
    accelerator.print(
        f"Estimated memory per sample: ~{est_gb:.1f} GB "
        f"(T={config.n_frames}, N~200, M~1500, checkpointing={config.activation_checkpointing})"
    )
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        max_batch = max(1, int(gpu_mem * 0.85 / est_gb))
        accelerator.print(
            f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.0f} GB), "
            f"estimated max batch_size ~{max_batch}"
        )

    # ------------------------------------------------------------------
    # Build dataset and dataloader
    # ------------------------------------------------------------------
    dataset = _build_dataset(config)
    accelerator.print(f"Training dataset: {len(dataset)} systems")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size_per_gpu,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=BoltzKinemaCollator(),
        pin_memory=True,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    scheduler = get_warmup_constant_scheduler(
        optimizer, warmup_steps=config.warmup_steps
    )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    criterion = _build_loss(config)

    # ------------------------------------------------------------------
    # Prepare with Accelerate
    # ------------------------------------------------------------------
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    global_step = 0
    if config.resume_from:
        global_step = _load_checkpoint(
            accelerator, config, model, optimizer, scheduler
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    os.makedirs(config.output_dir, exist_ok=True)
    accelerator.print(
        f"Starting training: mode={config.training_mode}, "
        f"max_steps={config.max_steps}, lr={config.lr}, "
        f"grad_accum={config.gradient_accumulation_steps}"
    )

    model.train()
    done = False

    for epoch in range(config.max_epochs):
        if done:
            break

        if hasattr(dataloader, "dataset") and hasattr(dataloader.dataset, "set_epoch"):
            dataloader.dataset.set_epoch(epoch)

        for batch in dataloader:
            with accelerator.accumulate(model):
                # Forward pass
                output = model(batch)

                # Compute loss
                loss, loss_dict = criterion(
                    output, batch, mode=config.training_mode
                )

                # Backward pass
                accelerator.backward(loss)

                # Gradient clipping (on sync steps)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config.grad_clip
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Step counting and logging (only after gradient sync)
            if accelerator.sync_gradients:
                global_step += 1

                # Logging
                if global_step % config.log_every == 0:
                    accelerator.log(loss_dict, step=global_step)
                    accelerator.log(
                        {"lr": scheduler.get_last_lr()[0]},
                        step=global_step,
                    )
                    accelerator.print(
                        f"step {global_step} | "
                        f"loss {loss_dict['loss']:.4f} | "
                        f"l_struct {loss_dict['l_struct']:.4f} | "
                        f"lr {scheduler.get_last_lr()[0]:.2e}"
                    )

                # Checkpointing
                if global_step % config.save_every == 0:
                    _save_checkpoint(accelerator, config, global_step)

                # Termination check
                if global_step >= config.max_steps:
                    done = True
                    break

    # Final checkpoint
    if global_step % config.save_every != 0:
        _save_checkpoint(accelerator, config, global_step)

    accelerator.print(f"Training complete at step {global_step}")
    accelerator.end_training()
