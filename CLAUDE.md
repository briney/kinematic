# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kinematic is a Python research package that extends Boltz-2 (an AlphaFold 3-like model, MIT licensed) with a Spatial-Temporal Diffusion Module for generating continuous-time, all-atom biomolecular trajectories. The frozen Boltz-2 trunk provides structural representations; only the temporal diffusion module is trained.

## Commands

### Install
```bash
pip install -e ".[dev]"
```
Requires the `boltz` package (Boltz-2) installed separately from source.

### Test
```bash
python -m pytest                              # all tests
python -m pytest tests/test_temporal_attention.py  # single test file
python -m pytest tests/test_shapes.py -k "test_name"  # single test
```
Some tests use `pytest.importorskip("boltz")` and skip automatically if boltz is unavailable.

### Lint
```bash
ruff check src scripts tests
ruff check --fix src scripts tests
```

### Train
```bash
accelerate launch scripts/train.py --base-config configs/base.yaml --config configs/train_phase0.yaml
accelerate launch scripts/train.py --config configs/train_phase0.yaml lr=2e-4  # CLI overrides
```
Config merge order: `base.yaml` → phase config → CLI overrides (OmegaConf).

### Inference
```bash
python scripts/generate.py --config configs/inference.yaml --input structure.pdb
```

## Architecture

### Source Layout
`src/` layout with all code under `src/kinematic/`. Five subpackages: `model/`, `data/`, `training/`, `inference/`, `evaluation/`.

### Model Hierarchy
```
Kinematic (kinematic.py)
├── rel_pos_encoder         ← Boltz-2 RelativePositionEncoder
├── diffusion_conditioning  ← Boltz-2 DiffusionConditioning
├── edm                     ← PerFrameEDM preconditioning
└── score_model             ← SpatialTemporalDiffusionModule (trainable)
    ├── atom_attention_encoder  ← SpatialTemporalAtomEncoder + TemporalAttentionWithDecay
    ├── token_transformer       ← SpatialTemporalTokenTransformer + TemporalAttentionWithDecay
    └── atom_attention_decoder  ← SpatialTemporalAtomDecoder + TemporalAttentionWithDecay
```

The **frozen Boltz-2 trunk** (InputEmbedder + MSA + Pairformer) runs once per system. Its outputs (`s_trunk`, `z_trunk`, `s_inputs`) are precomputed to float16 `.npz` files via `scripts/precompute_trunk.py`.

### Key Design Patterns

**Temporal Attention** (`temporal_attention.py`): Operates on `(B*N, T, C)` — for each spatial position, attends across frames with exponential decay bias. Output projection is **zero-initialized** so the model starts as single-frame Boltz-2.

**Noise-as-Masking** (`noise_masking.py`): Clean frames (σ=0) serve as conditioning; noised frames are denoising targets. Supports forecasting (first frame clean) and interpolation (first+last clean) tasks.

**EDM Preconditioning** (`edm.py`): Per-frame Karras et al. EDM scaling with `sigma_data=16.0` (Angstrom scale).

**Data Pipeline**: JSON manifest → `TrajectoryDataset` → `TrajectoryCollator` → batches. Trunk embeddings loaded from `.npz` cache, coordinates from per-system `.npz` files.

### Training Phases
- **Phase 0** (`train_phase0.yaml`): Temporal warmup on monomeric proteins, loads Boltz-2 weights
- **Phase 1** (`train_equilibrium.yaml`): Full mixed equilibrium training, resumes from Phase 0
- **Phase 2** (`train_unbinding.yaml`): Causal temporal attention fine-tune on unbinding data

## Critical Invariants

1. **Zero-initialized temporal output projections** — `TemporalAttentionWithDecay.out_proj.weight` must start as zeros so the model initially equals single-frame Boltz-2. Never change this initialization.
2. **Conditioning frames excluded from loss** — the `~conditioning_mask` gates loss computation. Changes to loss code must respect this.
3. **Trunk embeddings: float16 on disk, float32 in training** — `trunk_cache.py` saves float16 and loads as float32.
4. **Atom window padding uses ceiling division** — `_num_windows` in spatial_temporal_atom.py uses ceil to handle non-divisible atom counts.
5. **EDM sampler runs exactly `n_steps` model calls** — final step transitions to σ=0. Verified in `test_sampler.py`.
6. **Checkpoint loading dispatches by suffix** — `.safetensors` vs `.bin`/`.ckpt`/`.pt`. Always use `load_checkpoint_file` from `checkpoint_io.py`.

## Key Reference

`OVERVIEW.md` is the primary architecture specification (51 KB) covering math, pseudocode, data pipeline, training procedure, inference, and risk analysis.
