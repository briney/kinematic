"""Estimate GPU memory requirements for Kinematic training.

Computes per-sample and per-GPU memory based on model dimensions,
number of frames (T), tokens (N), and atoms (M).

Methodology:
  - Tensor sizes computed from architecture dimensions
  - Training memory accounts for forward activations, backward gradients,
    and optimizer states (Adam momentum + variance)
  - Activation checkpointing stores layer inputs at checkpoint boundaries
    but recomputes intermediates during backward; peak memory includes
    the recomputation working set of the most expensive layer
  - PyTorch CUDA allocator overhead (~10-15%) applied to final total

Usage:
    # Default config (T=32, N=200, M=1500):
    python scripts/estimate_memory.py

    # Custom dimensions:
    python scripts/estimate_memory.py --T 50 --N 300 --M 2000

    # With specific GPU and batch size:
    python scripts/estimate_memory.py --T 32 --N 200 --M 1500 --gpu-mem 80

    # Load dimensions from config:
    python scripts/estimate_memory.py --config configs/train_phase0.yaml --N 200 --M 1500
"""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf


def estimate_memory(
    T: int = 32,
    N: int = 200,
    M: int = 1500,
    batch_size: int = 1,
    token_s: int = 384,
    token_z: int = 128,
    atom_s: int = 128,
    atom_encoder_depth: int = 3,
    atom_encoder_heads: int = 4,
    atom_temporal_heads: int = 4,
    token_transformer_depth: int = 24,
    token_transformer_heads: int = 16,
    token_temporal_heads: int = 16,
    atom_decoder_depth: int = 3,
    atom_decoder_heads: int = 4,
    activation_checkpointing: bool = True,
    dtype_bytes: int = 2,
) -> dict[str, float]:
    """Estimate per-sample GPU memory in MB.

    Returns dict with memory breakdown.
    """
    B = batch_size
    bytes_to_mb = 1 / (1024 * 1024)
    d = 2 * token_s  # token transformer dim (768)
    W_q = 32  # atoms_per_window_queries
    W_k = 128  # atoms_per_window_keys
    n_windows = max(1, M // W_q)

    # =========================================================================
    # Coordinates: (B, T, M, 3) float32 — always in fp32 for precision
    # Multiple copies: clean coords, x_noisy, r_noisy, x_denoised
    # =========================================================================
    coords_mb = B * T * M * 3 * 4 * bytes_to_mb * 3

    # =========================================================================
    # Trunk embeddings (shared across frames, loaded once)
    # s_trunk (B, N, 384), z_trunk (B, N, N, 128), s_inputs (B, N, 384)
    # =========================================================================
    trunk_mb = (
        B * N * token_s * dtype_bytes  # s_trunk
        + B * N * N * token_z * dtype_bytes  # z_trunk
        + B * N * token_s * dtype_bytes  # s_inputs
    ) * bytes_to_mb

    # =========================================================================
    # DiffusionConditioning outputs (shared across frames)
    # =========================================================================
    conditioning_mb = (
        # q, c: (B, M, atom_s)
        2 * B * M * atom_s * dtype_bytes
        # atom_enc_bias: (B, n_windows, W_q, W_k, enc_depth*enc_heads)
        + B * n_windows * W_q * W_k * atom_encoder_depth * atom_encoder_heads * dtype_bytes
        # atom_dec_bias: (B, n_windows, W_q, W_k, dec_depth*dec_heads)
        + B * n_windows * W_q * W_k * atom_decoder_depth * atom_decoder_heads * dtype_bytes
        # token_trans_bias: (B, N, N, depth*heads)
        + B * N * N * token_transformer_depth * token_transformer_heads * dtype_bytes
        # rel_pos_enc: (B, N, N, token_z)
        + B * N * N * token_z * dtype_bytes
    ) * bytes_to_mb

    # =========================================================================
    # Per-frame activations: the core training memory cost
    #
    # During training, PyTorch stores intermediate tensors for backward.
    # Activation checkpointing applies ONLY to the token transformer
    # (per SpatialTemporalTokenTransformer code). Atom encoder/decoder
    # layers are NOT checkpointed and store ALL intermediates.
    #
    # Per-layer intermediates include: AdaLN output, QKV projections,
    # attention logits, attention output, gating, transition expansion.
    # We estimate ~8 tensors of layer-input size per sublayer.
    # =========================================================================

    atom_total_depth = atom_encoder_depth + atom_decoder_depth

    # --- Atom encoder/decoder: NOT checkpointed ---
    # Each layer has a spatial sublayer + temporal sublayer, both storing
    # full intermediates for backward.

    # Spatial sublayer intermediates per layer:
    #   ~8 tensors of (B*T * n_windows, W_q, atom_s) each
    #   + attention map: (B*T * n_windows, heads, W_q, W_k)
    #   + transition expansion: (B*T * n_windows, W_q, 4*atom_s)
    atom_spatial_per_layer = (
        8 * B * T * n_windows * W_q * atom_s * dtype_bytes
        + B * T * n_windows * atom_encoder_heads * W_q * W_k * dtype_bytes
        + B * T * n_windows * W_q * 4 * atom_s * dtype_bytes
    )

    # Temporal sublayer intermediates per layer:
    #   ~6 tensors of (B*M, T, atom_s) each
    #   + attention map in fp32: (B*M, atom_temp_heads, T, T) * 4 bytes
    atom_temporal_per_layer = (
        6 * B * M * T * atom_s * dtype_bytes
        + B * M * atom_temporal_heads * T * T * 4  # fp32 attention
    )

    atom_activation_bytes = atom_total_depth * (
        atom_spatial_per_layer + atom_temporal_per_layer
    )

    # --- Token transformer: CHECKPOINTED (per-layer) ---
    # With checkpointing: store only the input to each layer.
    # During backward, recompute one layer at a time.

    # Stored: 24 layer inputs of (B*T, N, d)
    token_stored = token_transformer_depth * B * T * N * d * dtype_bytes

    # Peak recompute working set (one layer):
    #   Spatial: QKV (3 tensors) + attention map + gating + transition
    #   Temporal: QKV + attention map + gating
    token_recompute_peak = (
        # Spatial sublayer
        5 * B * T * N * d * dtype_bytes  # QKV + adaln + gate
        + B * T * token_transformer_heads * N * N * dtype_bytes  # attn map
        + B * T * N * 4 * d * dtype_bytes  # transition expansion
        # Temporal sublayer
        + 5 * B * N * T * d * dtype_bytes  # QKV + norm + gate
        + B * N * token_temporal_heads * T * T * 4  # attn map (fp32)
    )

    token_activation_bytes = token_stored + token_recompute_peak

    # --- DiffusionConditioning (batch level, no T multiplier) ---
    # Runs once for the batch (not per-frame), so relatively small.
    # Includes its own AtomEncoder + TokenTransformer + AtomDecoder.
    # Estimated from: 24 token transformer layers at (B, N, 768)
    # plus 6 atom layers at (B, M, 128).
    conditioning_activation_bytes = (
        24 * 8 * B * N * d * dtype_bytes  # token layers intermediates
        + 6 * 8 * B * n_windows * W_q * atom_s * dtype_bytes  # atom layers
    )

    if activation_checkpointing:
        activation_bytes = (
            atom_activation_bytes
            + token_activation_bytes
            + conditioning_activation_bytes
        )
    else:
        # Without checkpointing: token transformer also stores all intermediates
        token_full_bytes = token_transformer_depth * (
            5 * B * T * N * d * dtype_bytes
            + B * T * token_transformer_heads * N * N * dtype_bytes
            + B * T * N * 4 * d * dtype_bytes
            + 5 * B * N * T * d * dtype_bytes
            + B * N * token_temporal_heads * T * T * 4
        )
        activation_bytes = (
            atom_activation_bytes
            + token_full_bytes
            + conditioning_activation_bytes
        )

    # During backward, PyTorch stores gradients of activations alongside
    # the saved activations. Additionally, there are transient peak
    # allocations during matmuls and attention. Apply ~2x factor to
    # account for backward gradient storage + transient peaks.
    backward_overhead = 2.0
    activation_bytes = int(activation_bytes * backward_overhead)

    activations_mb = activation_bytes * bytes_to_mb

    # =========================================================================
    # Model parameters + gradients + optimizer states
    # =========================================================================
    # Approximate parameter count from architecture:
    #   DiffusionConditioning: ~30M
    #   SingleConditioning: ~5M
    #   AtomEncoder (spatial): ~8M, AtomDecoder (spatial): ~8M
    #   TokenTransformer (24 layers, spatial): ~150M
    #   Temporal attention layers: ~15M (30 layers * ~0.5M each)
    #   Miscellaneous projections: ~5M
    n_params_approx = 215_000_000

    params_mb = n_params_approx * 4 * bytes_to_mb       # fp32 master weights
    grads_mb = n_params_approx * dtype_bytes * bytes_to_mb  # bf16 gradients
    optimizer_mb = n_params_approx * 8 * bytes_to_mb     # Adam momentum + variance (fp32)

    # =========================================================================
    # Total peak training memory
    # =========================================================================
    total_mb = (
        coords_mb + trunk_mb + conditioning_mb
        + activations_mb
        + params_mb + grads_mb + optimizer_mb
    )

    return {
        "coordinates": coords_mb,
        "trunk_embeddings": trunk_mb,
        "conditioning": conditioning_mb,
        "activations": activations_mb,
        "parameters": params_mb,
        "gradients": grads_mb,
        "optimizer_states": optimizer_mb,
        "total_mb": total_mb,
        "total_gb": total_mb / 1024,
    }


# Common GPU memory sizes
GPU_PROFILES = {
    "RTX4090": 24,
    "V100-32": 32,
    "A100-40": 40,
    "A6000": 48,
    "L40S": 48,
    "A100-80": 80,
    "H100": 80,
}


def recommend_batch_config(
    total_per_sample_gb: float,
    fixed_overhead_gb: float,
    gpu_mem_gb: float,
    target_effective: int = 4,
) -> dict[str, int | float]:
    """Recommend batch_size and gradient_accumulation for a GPU.

    Reserves ~15% for CUDA allocator overhead/fragmentation.
    Fixed overhead (params + optimizer) is paid once regardless of batch size.
    """
    usable_gb = gpu_mem_gb * 0.85
    available_for_samples = usable_gb - fixed_overhead_gb
    if available_for_samples <= 0:
        return {
            "batch_size_per_gpu": 0,
            "gradient_accumulation_steps": target_effective,
            "effective_batch_per_gpu": 0,
            "estimated_usage_gb": fixed_overhead_gb,
        }

    # Per-sample activation/data cost (total minus fixed)
    per_sample_variable = total_per_sample_gb - fixed_overhead_gb
    if per_sample_variable <= 0:
        per_sample_variable = total_per_sample_gb

    max_batch = max(1, int(available_for_samples / per_sample_variable))

    if max_batch >= target_effective:
        recommended_accum = 1
    elif max_batch >= 2:
        recommended_accum = max(1, target_effective // max_batch)
    else:
        recommended_accum = target_effective

    effective_batch = max_batch * recommended_accum
    estimated_gb = fixed_overhead_gb + max_batch * per_sample_variable

    return {
        "batch_size_per_gpu": max_batch,
        "gradient_accumulation_steps": recommended_accum,
        "effective_batch_per_gpu": effective_batch,
        "estimated_usage_gb": estimated_gb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate GPU memory for Kinematic training"
    )
    parser.add_argument("--T", type=int, default=32, help="Number of frames (default: 32)")
    parser.add_argument("--N", type=int, default=200, help="Number of tokens (default: 200)")
    parser.add_argument("--M", type=int, default=1500, help="Number of atoms (default: 1500)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--no-checkpointing", action="store_true", help="Disable activation checkpointing")
    parser.add_argument("--gpu-mem", type=float, default=None, help="GPU memory in GB (for recommendations)")
    parser.add_argument("--config", type=str, default=None, help="Load model dims from config YAML")
    parser.add_argument("--all-gpus", action="store_true", help="Show recommendations for all common GPUs")
    args = parser.parse_args()

    # Load model config if provided
    model_kwargs = {}
    if args.config:
        cfg = OmegaConf.load(args.config)
        for key in [
            "token_s", "token_z", "atom_s", "atom_encoder_depth",
            "atom_encoder_heads", "atom_temporal_heads",
            "token_transformer_depth", "token_transformer_heads",
            "token_temporal_heads", "atom_decoder_depth", "atom_decoder_heads",
        ]:
            if key in cfg:
                model_kwargs[key] = cfg[key]
        if "activation_checkpointing" in cfg:
            model_kwargs["activation_checkpointing"] = cfg.activation_checkpointing

    if args.no_checkpointing:
        model_kwargs["activation_checkpointing"] = False

    # Compute for comparison T values or single T
    t_values = [20, 32, 50] if args.T == 32 else [args.T]

    print("Kinematic Memory Estimation")
    print("=" * 70)
    print(f"  Tokens (N): {args.N}")
    print(f"  Atoms  (M): {args.M}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Activation checkpointing: {not args.no_checkpointing}")
    print()

    results = {}
    for t in t_values:
        results[t] = estimate_memory(
            T=t, N=args.N, M=args.M, batch_size=args.batch_size,
            **model_kwargs,
        )

    if len(t_values) > 1:
        # Multi-T comparison table
        print(f"{'Component':<30}", end="")
        for t in t_values:
            print(f"{'T=' + str(t):>12}", end="")
        print()
        print("-" * (30 + 12 * len(t_values)))

        display_keys = [
            ("coordinates", "Coordinates"),
            ("trunk_embeddings", "Trunk embeddings"),
            ("conditioning", "Conditioning outputs"),
            ("activations", "Activations (training)"),
            ("parameters", "Parameters (fp32)"),
            ("gradients", "Gradients"),
            ("optimizer_states", "Optimizer (Adam)"),
        ]

        for key, label in display_keys:
            print(f"{label:<30}", end="")
            for t in t_values:
                val_mb = results[t][key]
                if val_mb >= 1024:
                    print(f"{val_mb/1024:>10.1f}GB", end="")
                else:
                    print(f"{val_mb:>10.0f}MB", end="")
            print()

        print("-" * (30 + 12 * len(t_values)))
        print(f"{'TOTAL per sample':<30}", end="")
        for t in t_values:
            gb = results[t]["total_gb"]
            print(f"{gb:>10.1f}GB", end="")
        print()
    else:
        t = t_values[0]
        result = results[t]

        print(f"Frames (T): {t}")
        print()
        print(f"{'Component':<35} {'Memory':>10}")
        print("-" * 47)

        display_keys = [
            ("coordinates", "Coordinates"),
            ("trunk_embeddings", "Trunk embeddings"),
            ("conditioning", "Conditioning outputs"),
            ("activations", "Activations (training)"),
            ("parameters", "Parameters (fp32)"),
            ("gradients", "Gradients"),
            ("optimizer_states", "Optimizer (Adam)"),
        ]

        for key, label in display_keys:
            val_mb = result[key]
            if val_mb >= 1024:
                print(f"{label:<35} {val_mb/1024:>8.1f} GB")
            else:
                print(f"{label:<35} {val_mb:>8.0f} MB")

        print("-" * 47)
        print(f"{'TOTAL per sample':<35} {result['total_gb']:>8.1f} GB")

    # GPU recommendations
    print()

    # Compute fixed overhead (params + optimizer — constant regardless of batch)
    ref_t = 32 if 32 in results else t_values[0]
    ref_result = results[ref_t]
    fixed_gb = (ref_result["parameters"] + ref_result["gradients"] + ref_result["optimizer_states"]) / 1024

    if args.all_gpus:
        ref_gb = ref_result["total_gb"]
        print(f"GPU Recommendations (T={ref_t}, N={args.N}, M={args.M}):")
        print(f"{'GPU':<15} {'VRAM':>6} {'Batch':>6} {'Accum':>6} {'Eff.':>5} {'Est.Usage':>10}")
        print("-" * 52)
        for gpu_name, mem in sorted(GPU_PROFILES.items(), key=lambda x: x[1]):
            rec = recommend_batch_config(ref_gb, fixed_gb, mem)
            if rec["batch_size_per_gpu"] == 0:
                print(f"{gpu_name:<15} {mem:>4.0f}GB   {'OOM':>25}")
            else:
                print(
                    f"{gpu_name:<15} {mem:>4.0f}GB "
                    f"{rec['batch_size_per_gpu']:>5} "
                    f"{rec['gradient_accumulation_steps']:>5} "
                    f"{rec['effective_batch_per_gpu']:>4} "
                    f"{rec['estimated_usage_gb']:>8.1f}GB"
                )
    elif args.gpu_mem:
        ref_gb = ref_result["total_gb"]
        rec = recommend_batch_config(ref_gb, fixed_gb, args.gpu_mem)
        print(f"Recommendation for {args.gpu_mem:.0f}GB GPU (T={ref_t}):")
        print(f"  batch_size_per_gpu: {rec['batch_size_per_gpu']}")
        print(f"  gradient_accumulation_steps: {rec['gradient_accumulation_steps']}")
        print(f"  effective batch per GPU: {rec['effective_batch_per_gpu']}")
        print(f"  estimated VRAM: {rec['estimated_usage_gb']:.1f}GB / {args.gpu_mem:.0f}GB")
    else:
        ref_gb = ref_result["total_gb"]
        print(f"Recommended batch sizes (T={ref_t}):")
        for gpu_name in ["A6000", "A100-80", "H100"]:
            if gpu_name in GPU_PROFILES:
                mem = GPU_PROFILES[gpu_name]
                rec = recommend_batch_config(ref_gb, fixed_gb, mem)
                print(
                    f"  {gpu_name} ({mem}GB): "
                    f"batch_size={rec['batch_size_per_gpu']}, "
                    f"grad_accum={rec['gradient_accumulation_steps']}"
                )


if __name__ == "__main__":
    main()
