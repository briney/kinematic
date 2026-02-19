# Kinematic: Boltz-2 Based All-Atom Biomolecular Dynamics

**Version:** 1.0  
**Date:** February 18, 2026  
**Target Base Model:** Boltz-2 (MIT License, jwohlwend/boltz)  
**Original Paper:** Feng et al., 2026 — bioRxiv 10.64898/2026.02.15.705956

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Technical Analysis](#2-technical-analysis)
3. [Boltz-2 Component Mapping](#3-boltz-2-component-mapping)
4. [Architecture Specification](#4-architecture-specification)
5. [Data Pipeline](#5-data-pipeline)
6. [Training Procedure](#6-training-procedure)
7. [Inference Pipeline](#7-inference-pipeline)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [Risk Analysis & Mitigations](#9-risk-analysis--mitigations)
10. [Appendix: Key Equations](#10-appendix-key-equations)

---

## 1. Executive Summary

Kinematic is a diffusion-based generative model that predicts continuous-time, all-atom biomolecular trajectories. It extends an AlphaFold 3-like architecture with a novel **Spatial-Temporal Diffusion Module** that jointly models spatial relationships within frames and temporal dependencies across frames.

This plan describes a complete reimplementation using **Boltz-2** as the structural prediction backbone instead of Protenix. Boltz-2 is chosen because:

- **MIT license** — fully open for academic and commercial use
- **Architecturally equivalent** to AF3 with Input Embedder, MSA Module, Pairformer, and Diffusion Module
- **Pre-trained weights available** — can freeze trunk for transfer learning
- **Active codebase** with training infrastructure (Boltz-1 training code available; Boltz-2 training code forthcoming)
- **Superior or equivalent** to Protenix on structure prediction benchmarks

### What we are building

A PyTorch model (`Kinematic`) that:
1. **Freezes** the Boltz-2 Input Embedder + Pairformer trunk to extract single (`s`) and pair (`z`) representations
2. **Replaces** the standard single-frame Diffusion Module with a new **Spatial-Temporal Diffusion Module** that processes multi-frame trajectories
3. **Adds** temporal attention with physically grounded exponential-decay bias derived from Langevin dynamics
4. **Implements** noise-as-masking training for unified forecasting and interpolation
5. **Supports** hierarchical inference: coarse forecasting → fine interpolation

### What stays the same vs. what changes

| Component | Source | Status |
|-----------|--------|--------|
| Input Embedder (sequence → single/pair reps) | Boltz-2 pretrained | **Frozen** |
| MSA Module | Boltz-2 pretrained | **Frozen** |
| Pairformer (triangle updates + attention) | Boltz-2 pretrained | **Frozen** |
| Atom Spatial Encoder (atom → token aggregation) | Boltz-2 diffusion module | **Modified** — extended to spatial-temporal |
| Token Transformer (global self-attention) | Boltz-2 diffusion module | **Modified** — extended to spatial-temporal |
| Atom Spatial Decoder (token → atom coordinates) | Boltz-2 diffusion module | **Modified** — extended to spatial-temporal |
| Temporal Attention (exponential decay bias) | **New** | **New module** |
| Noise-as-Masking Training | **New** | **New training paradigm** |
| Hierarchical Sampling | **New** | **New inference strategy** |
| Flexibility / Unbinding Losses | **New** | **New loss terms** |

---

## 2. Technical Analysis

### 2.1 Overall Architecture

The model consists of three sequential stages:

**Stage A — Feature Extraction (Frozen AF3 Trunk):**
- Input: sequence, MSA, (optional) templates
- The Input Embedder produces:
  - Single representation: `s ∈ ℝ^{n × c_s}` (per-token features)
  - Pair representation: `z ∈ ℝ^{n × n × c_p}` (pairwise relationships)
  - Single input representation: `s_inputs ∈ ℝ^{n × c_s}` (raw input features)
- The Pairformer refines `s` and `z` through `N_cycle = 4` recycling iterations
- All parameters are **frozen** (initialized from pretrained Protenix/AF3)
- Representations are **precomputed** for training efficiency

**Stage B — Spatial-Temporal Diffusion Module (Trainable):**
- Input: noisy trajectory `{x̃_t}_{t=0}^{T-1} ∈ ℝ^{T × N_atoms × 3}`, timestamps `{t_i}`, noise levels `{σ_i}`, trunk representations `s`, `z`, `s_inputs`
- Architecture (from paper's Figure 1c):
  - **Atom Spatial-Temporal Encoder** (3 blocks): sequence-local atom attention + temporal attention → aggregation to token level
  - **Token Spatial-Temporal Transformer** (24 blocks): full self-attention over tokens + temporal attention over frames
  - **Atom Spatial-Temporal Decoder** (3 blocks): broadcast token activations back to atoms + temporal attention → coordinate updates
- Output: denoised trajectory `{x̂_t}_{t=0}^{T-1} ∈ ℝ^{T × N_atoms × 3}`

**Stage C — EDM Preconditioning:**
- Uses the standard EDM framework (Karras et al., 2022) with skip connections:
  ```
  D_θ(x; σ) = c_skip(σ) · x + c_out(σ) · F_θ(c_in(σ) · x; c_noise(σ))
  ```
- σ_data = 16 (data standard deviation)

### 2.2 The Spatial-Temporal Attention Mechanism

This is the core innovation. Each attention layer in the diffusion module is extended to operate in two modes:

**Spatial attention** (standard, within each frame):
- For atom-level: sequence-local attention within a window (as in AF3)
- For token-level: full self-attention over all tokens within a frame
- Uses pair representation `z` as attention bias

**Temporal attention** (new, across frames for each atom/token):
- For each atom/token position, attend across all T frames
- Uses exponential decay bias: `B_{ij} = -λ_h |t_i - t_j|`
- After softmax: `A_{ij} ∝ exp(q_i^T k_j / √d) · exp(-λ_h |t_i - t_j|)`
- Each head h has a learnable decay rate λ_h
- **Bidirectional** (not causal) for equilibrium MD — justified by time-reversal symmetry
- **Causal** for metadynamics/unbinding — history-dependent bias potential

**Head configuration:**
- Atom-level temporal attention: **4 heads**
- Token-level temporal attention: **16 heads**
- λ_h initialized following ALiBi geometric sequence: `[0.004, ..., 0.7]`

**Continuous time support:**
- Bias depends only on `|t_i - t_j|`, not absolute timestamps
- Training intervals: 0.08 ns to 100 ns (randomly sampled)
- Inference: any desired temporal resolution

### 2.3 Noise-as-Masking Training Paradigm

Instead of using a separate encoder for conditioning frames, the noise-as-masking paradigm reinterprets diffusion noise levels as information visibility:
- `σ = 0` → frame is fully visible (conditioning/"unmasked")
- `σ ~ p(σ)` → frame is a prediction target ("masked")
- `σ = σ_max` → frame is fully obscured

**Training sample construction:**
1. Sample a trajectory segment from training data
2. Sample inter-frame intervals Δt randomly (dataset-dependent ranges)
3. For each frame, sample independent noise level `σ_t ~ LogNormal(P_mean, P_std²)`
4. Designate conditioning frames by setting `σ_t = 0`:
   - **Forecasting task**: first frame clean, all others noised
   - **Interpolation task**: first and last frames clean, intermediates noised
5. Construct noisy trajectory: `x̃_t = x_t + σ_t · ε_t`
6. Feed entire sequence (clean + noisy) to Spatial-Temporal Diffusion Module
7. Compute loss only on non-conditioning frames

**Key advantage:** A single model architecture handles both forecasting and interpolation without separate encoders or conditioning mechanisms.

### 2.4 Hierarchical Sampling Strategy

To generate long trajectories (e.g., 1 μs) without catastrophic error accumulation:

**Stage 1 — Coarse-grained Forecasting:**
- Generate at large time intervals (Δt_coarse = 5–20 ns)
- Can be all-at-once or auto-regressive in blocks
- Example: 1 μs at 5 ns intervals → 200 coarse frames
- With 200 ns generation blocks → only 5 AR iterations (vs. 200 at fine resolution)

**Stage 2 — Fine-grained Interpolation:**
- For each coarse interval, fill in intermediate frames
- Both endpoint frames are fixed as clean anchors (σ = 0)
- Intermediate frames denoised simultaneously
- Geometrically constrained → no unbounded error growth

### 2.5 Training Data

| Dataset | Systems | Duration | Interval Range | Focus |
|---------|---------|----------|---------------|-------|
| ATLAS | ~1,500 chains | 450 μs total (3×100ns each) | 0.1–10 ns | Protein monomers |
| MISATO | ~16,000 complexes | 170 μs total (8ns each) | 0.08–0.8 ns | Protein-ligand |
| MDposit | 3,271 systems | 1,470 μs total (avg 457ns) | 0.1–100 ns | Diverse (monomer, multimer, nucleic acid, ligand) |
| DD-13M (unbinding) | 565 complexes, 26,612 trajectories | Variable | 10 ps steps | Ligand dissociation (metadynamics) |

**Total: >2,000 μs** of MD trajectories + metadynamics unbinding data.

### 2.6 Training Losses

**Main Loss — Structure Reconstruction (EDM-weighted denoising):**
```
L_struct = [σ² + σ²_data] / [σ · σ_data]² × (L_MSE + α_bond · L_bond) + L_smooth_lddt
```

Where:
- `L_MSE`: per-atom coordinate MSE with molecule-type weights (protein=1, DNA/RNA=5, ligand=10)
- `L_bond`: covalent bond length loss for ligand-protein connections
- `L_smooth_lddt`: differentiable lDDT approximation

**Flexibility Loss (for MD trajectory training):**
```
L_flex = β_abs · L^abs_flex + β_rel-g · L^rel-global_flex + β_rel-l · L^rel-local_flex
```

With ratio `1:4:4` for absolute RMSF, global relative (pairwise distance std), and local relative (intra-residue) components. Uses distance-dependent weights (≤5Å: 4×, 5–10Å: 2×, >10Å: 1×) and molecule-type weights (ligand-ligand: 10×, protein-ligand: 5×, protein-protein: 1×).

**Ligand Geometric Center Loss (for unbinding training):**
```
L_center = mean_t ||C(x̂_t) - C(x^GT_t)||²
```

**Total loss:**
- MD training: `L_total = L_struct + β_flex · L_flex`
- Unbinding training: `L_total = L_struct + β_center · L_center`

### 2.7 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1e-4 (fixed) |
| Warmup | 200 steps (linear) |
| Batch size | 32 |
| GPUs | 8 × NVIDIA RTX A6000 |
| Dataset resampling | ~1:1:1 ratio (ATLAS:MISATO:MDposit) |
| Noise distribution | `ln(σ) ~ N(P_mean, P²_std)` (EDM defaults) |
| Augmentation | Random rotation + translation (global, SE(3) invariant) |
| Frozen trunk embeddings | Precomputed and cached |
| σ_data | 16 |
| EDM inference steps | N=20 |
| σ_noise | 1.75 |
| Step scale η | 1.5 |

---

## 3. Boltz-2 Component Mapping

### 3.1 Architectural Correspondence

The AF3/Protenix architecture is structurally equivalent to Boltz-2's trunk. Here is the mapping:

| AF3/Protenix Component | Boltz-2 Equivalent | Notes |
|---------------------------|-------------------|-------|
| Input Embedder | `InputEmbedder` | Same function: sequence → s, z |
| MSA Module (optional) | `MSAModule` | Same: evolutionary info → pair updates |
| Pairformer (N_cycle recycling) | `PairformerModule` (with recycling) | Same: triangle updates + attention |
| Single representation `s` | `s` / `token_s` | Same dimensionality concept |
| Pair representation `z` | `z` / `token_z` | Same dimensionality concept |
| Single input `s_inputs` | `s_inputs` | Raw input features |
| Diffusion Conditioning | `DiffusionConditioning` | Produces q, c, biases from s, z |
| Atom Encoder (3 blocks) | `AtomEncoder` in diffusion module | Sequence-local atom attention |
| Token Transformer (24 blocks) | `TokenTransformer` in diffusion module | Global token self-attention |
| Atom Decoder (3 blocks) | `AtomDecoder` in diffusion module | Atom coordinate prediction |
| EDM preconditioning | Same EDM framework | Identical skip/scale functions |

### 3.2 Key Dimensional Parameters

From the Boltz-2 codebase and AF3 specification:

| Parameter | Typical Value | Description |
|-----------|--------------|-------------|
| `token_s` (c_s) | 384 | Single representation dimension |
| `token_z` (c_p) | 128 | Pair representation dimension |
| `atom_s` | 128 | Atom single representation |
| `atom_z` | 16 | Atom pair representation |
| Pairformer blocks | 48 | Number of pairformer blocks |
| Recycling cycles | 4 | Number of trunk recycling iterations |
| Diffusion atom encoder blocks | 3 | Atom-level spatial(-temporal) encoder |
| Diffusion token transformer blocks | 24 | Token-level spatial(-temporal) transformer |
| Diffusion atom decoder blocks | 3 | Atom-level spatial(-temporal) decoder |

### 3.3 What Must Be Modified in the Boltz-2 Diffusion Module

The standard Boltz-2 `AtomDiffusion` module processes a **single frame**. We must extend it to process **T frames simultaneously**:

**Per-block modifications:**

1. **Reshape tensors for temporal processing:**
   - Standard: `(B, N_atoms, C)` for atoms, `(B, N_tokens, C)` for tokens
   - Extended: `(B, T, N_atoms, C)` for atoms, `(B, T, N_tokens, C)` for tokens

2. **Add temporal attention after each spatial attention:**
   - Transpose to `(B × N, T, C)` — group by atom/token across frames
   - Apply multi-head attention with exponential decay bias
   - Transpose back to `(B × T, N, C)`

3. **Per-frame noise conditioning:**
   - Each frame has its own `σ_t` and timestamp `t_i`
   - The `DiffusionConditioning` outputs (q, c, biases) are shared across frames (from trunk)
   - But the EDM scaling (`c_in`, `c_skip`, `c_out`) is applied **per-frame** based on each frame's σ

4. **Continuous timestamp encoding:**
   - Temporal bias `B_{ij}^{(h)} = -λ_h |t_i - t_j|` computed from float timestamps
   - Added to attention logits before softmax

### 3.4 What to Extract from Boltz-2 Pretrained Weights

For weight initialization of the spatial components:

| Boltz-2 Module | Initialize To |
|---------------|---------------|
| `InputEmbedder` weights | Frozen, load directly |
| `MSAModule` weights | Frozen, load directly |
| `PairformerModule` weights | Frozen, load directly |
| `DiffusionConditioning` weights | Initialize spatial attention from pretrained, temporal from scratch |
| Atom encoder spatial attention | Load from Boltz-2 diffusion module |
| Token transformer spatial attention + pair bias | Load from Boltz-2 diffusion module |
| Atom decoder spatial attention | Load from Boltz-2 diffusion module |
| Temporal attention layers | **Random init** (output projections zero-initialized) |
| λ_h decay factors | **Geometric sequence** [0.004, ..., 0.7] following ALiBi |

The zero-initialization of temporal attention output projections ensures that at training start, the model behaves identically to the single-frame Boltz-2 predictor, allowing gradual learning of temporal dependencies.

---

## 4. Architecture Specification

### 4.1 Module Hierarchy

```
Kinematic (PyTorch Lightning Module)
├── BoltzTrunk (FROZEN)
│   ├── InputEmbedder
│   │   ├── SingleEmbedder
│   │   └── PairEmbedder
│   ├── MSAModule (optional)
│   └── PairformerModule
│       └── PairformerStack (48 blocks)
│           ├── TriangleMultiplication (outgoing + incoming)
│           ├── TriangleAttention (starting + ending)
│           └── PairTransition
│
├── DiffusionConditioning (partially frozen or fine-tuned)
│   ├── Boltz-2 DiffusionConditioning (→ q, c, biases)
│   └── NoiseConditioning (σ → noise embedding)
│
├── SpatialTemporalDiffusionModule (TRAINABLE)
│   ├── SpatialTemporalAtomEncoder (3 blocks)
│   │   ├── SpatialAtomAttention (from Boltz-2, init from pretrained)
│   │   ├── TemporalAtomAttention (NEW, 4 heads, zero-init output)
│   │   └── AtomToTokenAggregation
│   │
│   ├── SpatialTemporalTokenTransformer (24 blocks)
│   │   ├── AttentionPairBias (spatial, from Boltz-2, init from pretrained)
│   │   ├── TemporalAttentionWithDecay (NEW, 16 heads, zero-init output)
│   │   └── ConditionedTransitionBlock (from Boltz-2)
│   │
│   └── SpatialTemporalAtomDecoder (3 blocks)
│       ├── SpatialAtomAttention (from Boltz-2, init from pretrained)
│       ├── TemporalAtomAttention (NEW, 4 heads, zero-init output)
│       └── CoordinateUpdate
│
└── EDMPreconditioning
    ├── c_in, c_skip, c_out, c_noise (per-frame)
    └── Loss weighting
```

### 4.2 TemporalAttentionWithDecay — Detailed Specification

```python
class TemporalAttentionWithDecay(nn.Module):
    """
    Temporal attention with learnable exponential decay bias.
    
    For each atom/token position, attends across all T frames.
    Bias: B_ij^(h) = -λ_h * |t_i - t_j|
    After softmax: A_ij ∝ exp(QK^T/√d) * exp(-λ_h * |t_i - t_j|)
    
    Args:
        dim: input/output dimension
        n_heads: number of attention heads
        causal: if True, apply causal mask (for metadynamics training)
    """
    
    def __init__(self, dim: int, n_heads: int, causal: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.causal = causal
        
        # Q, K, V projections
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        # Output projection — ZERO INITIALIZED
        self.to_out = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.to_out.weight)
        
        # Learnable decay factors (one per head)
        # Initialize as geometric sequence [0.004, ..., 0.7]
        log_lambdas = torch.linspace(
            math.log(0.004), math.log(0.7), n_heads
        )
        self.log_lambda = nn.Parameter(log_lambdas)
    
    def forward(
        self,
        x: Tensor,        # (B, T, N, C) or reshaped to (B*N, T, C)
        timestamps: Tensor # (B, T) — continuous float timestamps in ns
    ) -> Tensor:
        B_N, T, C = x.shape  # already reshaped: B*N, T, C
        
        Q = self.to_q(x).reshape(B_N, T, self.n_heads, self.head_dim)
        K = self.to_k(x).reshape(B_N, T, self.n_heads, self.head_dim)
        V = self.to_v(x).reshape(B_N, T, self.n_heads, self.head_dim)
        
        # Attention scores
        # Q, K: (B*N, T, H, d) → einsum → (B*N, H, T, T)
        scores = torch.einsum('bthd,bshd->bhts', Q, K) / math.sqrt(self.head_dim)
        
        # Exponential decay bias
        lambda_h = torch.exp(self.log_lambda)  # (H,) — ensure positive
        # timestamps: (B, T) → need to broadcast per-atom
        # Δt_{ij} = |t_i - t_j|: (T, T)
        dt = torch.abs(timestamps.unsqueeze(-1) - timestamps.unsqueeze(-2))  # (B, T, T)
        # Need to handle B*N grouping — timestamps are same for all atoms
        # bias: (H, T, T)
        bias = -lambda_h.view(-1, 1, 1) * dt.unsqueeze(1)  # (B, H, T, T)
        
        scores = scores + bias  # broadcast over B*N
        
        # Optional causal mask (for metadynamics)
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device), diagonal=1
            ).bool()
            scores.masked_fill_(causal_mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        
        out = torch.einsum('bhts,bshd->bthd', attn, V)
        out = out.reshape(B_N, T, C)
        
        return self.to_out(out)
```

### 4.3 SpatialTemporalTransformerBlock — Detailed Specification

```python
class SpatialTemporalTransformerBlock(nn.Module):
    """
    One block of the Token Spatial-Temporal Transformer.
    
    Sequence per block:
    1. Spatial attention (AttentionPairBias from Boltz-2)
    2. Temporal attention with decay
    3. Conditioned transition (feed-forward)
    """
    
    def __init__(self, dim: int, n_spatial_heads: int, n_temporal_heads: int,
                 pair_dim: int, causal: bool = False):
        # Spatial attention — initialized from Boltz-2 pretrained weights
        self.spatial_attn = AttentionPairBias(
            dim=dim, n_heads=n_spatial_heads, pair_dim=pair_dim
        )
        self.spatial_norm = nn.LayerNorm(dim)
        
        # Temporal attention — NEW, zero-init output
        self.temporal_attn = TemporalAttentionWithDecay(
            dim=dim, n_heads=n_temporal_heads, causal=causal
        )
        self.temporal_norm = nn.LayerNorm(dim)
        
        # Transition — initialized from Boltz-2
        self.transition = ConditionedTransitionBlock(dim=dim, s_dim=dim)
    
    def forward(self, a, s, z, timestamps):
        """
        Args:
            a: token activations (B, T, N_tokens, C)
            s: single representation (B, N_tokens, C_s) — shared across frames
            z: pair representation (B, N_tokens, N_tokens, C_z) — shared across frames
            timestamps: (B, T) — continuous timestamps
        """
        B, T, N, C = a.shape
        
        # 1. Spatial attention (per-frame)
        a_flat = a.reshape(B * T, N, C)
        a_flat = a_flat + self.spatial_attn(self.spatial_norm(a_flat), s, z)
        a = a_flat.reshape(B, T, N, C)
        
        # 2. Temporal attention (per-token across frames)
        a_transposed = a.permute(0, 2, 1, 3).reshape(B * N, T, C)
        a_transposed = a_transposed + self.temporal_attn(
            self.temporal_norm(a_transposed), timestamps
        )
        a = a_transposed.reshape(B, N, T, C).permute(0, 2, 1, 3)
        
        # 3. Transition
        a_flat = a.reshape(B * T, N, C)
        a_flat = a_flat + self.transition(a_flat, s)
        a = a_flat.reshape(B, T, N, C)
        
        return a
```

### 4.4 Noise-as-Masking: Per-Frame Noise Conditioning

```python
class PerFrameEDMConditioning(nn.Module):
    """
    Apply EDM preconditioning independently per frame.
    
    Each frame has its own σ_t:
    - Conditioning frames: σ_t = 0 (clean)
    - Target frames: σ_t ~ p(σ)
    
    The spatial trunk representations (s, z) are SHARED across all frames.
    Only the noise-dependent scaling differs per frame.
    """
    
    def __init__(self, sigma_data: float = 16.0):
        super().__init__()
        self.sigma_data = sigma_data
    
    def scale_input(self, x: Tensor, sigma: Tensor) -> Tensor:
        """
        x: (B, T, N, 3) — noisy coordinates
        sigma: (B, T) — per-frame noise levels
        """
        c_in = 1.0 / torch.sqrt(sigma**2 + self.sigma_data**2)
        # Reshape for broadcasting: (B, T, 1, 1)
        return x * c_in.unsqueeze(-1).unsqueeze(-1)
    
    def combine_output(self, x_noisy: Tensor, f_out: Tensor, 
                       sigma: Tensor) -> Tensor:
        """
        EDM skip connection applied per-frame.
        
        x_out = c_skip(σ) * x_noisy + c_out(σ) * f_out
        """
        sd = self.sigma_data
        c_skip = sd**2 / (sd**2 + sigma**2)
        c_out = sigma * sd / torch.sqrt(sd**2 + sigma**2)
        
        c_skip = c_skip.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
        c_out = c_out.unsqueeze(-1).unsqueeze(-1)
        
        return c_skip * x_noisy + c_out * f_out
```

### 4.5 Full Forward Pass (Pseudocode)

```python
def forward(self, batch):
    """
    batch contains:
        - sequence, msa: input features
        - x_traj: ground truth trajectory (B, T, N_atoms, 3)
        - timestamps: (B, T) continuous timestamps in ns
        - sigma: (B, T) per-frame noise levels (0 for conditioning frames)
        - conditioning_mask: (B, T) bool — True for conditioning frames
    """
    # === STAGE A: Frozen trunk (or use precomputed) ===
    s, z, s_inputs = self.trunk(sequence, msa)  # precomputed
    
    # === Prepare noisy trajectory ===
    epsilon = torch.randn_like(x_traj)
    x_noisy = x_traj + sigma.unsqueeze(-1).unsqueeze(-1) * epsilon
    # Clean conditioning frames
    x_noisy[conditioning_mask] = x_traj[conditioning_mask]
    
    # === STAGE B: Spatial-Temporal Diffusion ===
    # Scale input per-frame
    r_noisy = self.edm.scale_input(x_noisy, sigma)  # (B, T, N_atoms, 3)
    
    # Diffusion conditioning (from trunk, shared across frames)
    q, c, biases = self.diff_conditioning(s, z, s_inputs, sigma)
    
    # Atom Encoder (3 blocks, spatial-temporal)
    a_tokens, skip_info = self.atom_encoder(r_noisy, q, c, biases, timestamps)
    
    # Token Transformer (24 blocks, spatial-temporal)
    a_tokens = self.token_transformer(a_tokens, s, z, timestamps)
    
    # Atom Decoder (3 blocks, spatial-temporal)
    r_update = self.atom_decoder(a_tokens, skip_info, timestamps)
    
    # === STAGE C: EDM output combination (per-frame) ===
    x_denoised = self.edm.combine_output(x_noisy, r_update, sigma)
    
    # === Loss (only on non-conditioning frames) ===
    loss_mask = ~conditioning_mask
    loss = self.compute_loss(x_denoised, x_traj, sigma, loss_mask)
    
    return x_denoised, loss
```

---

## 5. Data Pipeline

### 5.1 Data Sources and Acquisition

**ATLAS Dataset:**
- Source: https://www.dsimb.inserm.fr/ATLAS/
- Format: GROMACS trajectories (.xtc) + topology (.gro/.pdb)
- Preprocessing: Convert to coordinate arrays using MDAnalysis/MDTraj
- Force field: CHARMM36m + TIP3P, 150 mM NaCl
- Filtering: Remove structures with >40% sequence identity to test set

**MISATO Dataset:**
- Source: https://github.com/t7morgen/misato-dataset
- Format: HDF5 files with coordinates + topology
- Preprocessing: Discard first 2 ns equilibration; retain 8 ns (100 frames)
- Force field: Amber20 + TIP3P
- Filtering: Valency checks on small molecules

**MDposit (DynaRepo):**
- Source: https://dynarepo.github.io/ (previously MDposit)
- Format: Various MD formats
- Contains: proteins, multimers, protein-nucleic acid, protein-ligand
- Range: 2.47 ns to 5,350 ns per trajectory

**DD-13M (for unbinding fine-tuning):**
- Source: arXiv:2504.18367
- Format: Metadynamics dissociation trajectories
- 26,612 trajectories across 565 complexes

### 5.2 Preprocessing Pipeline

```
Raw MD Trajectories
        │
        ▼
┌─────────────────────┐
│  1. Solvent Removal  │  Remove water, ions (keep protein/ligand/DNA/RNA)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  2. Missing Residues │  Align to reference PDB, mark unresolved residues
│     Detection        │  Mask their loss terms during training
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  3. Valency Check    │  For ligand-containing systems
│     (Ligands)        │  Filter trajectories with chemical validity issues
└─────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  4. Tokenization            │  Use Boltz-2 tokenizer:
│     (Boltz-2 compatible)    │  - Proteins: per-residue tokens
│                             │  - Ligands: per-atom tokens
│                             │  - DNA/RNA: per-nucleotide tokens
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  5. Trunk Embedding         │  Run Boltz-2 trunk ONCE per system:
│     Precomputation          │  - s_inputs, s_trunk, z_trunk
│                             │  - Cache to disk (HDF5/NPZ)
│                             │  - Significant training speedup
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  6. Coordinate Processing   │  - Align all frames to frame 0
│                             │    (remove rigid-body motion)
│                             │  - Store as float32 arrays
│                             │  - Index: (system_id, frame_idx, atom_idx, xyz)
└─────────────────────────────┘
```

### 5.3 Training Sample Construction (DataLoader)

```python
class TrajectoryDataset(Dataset):
    """
    Dynamic sub-sampling of trajectory segments.
    
    For each sample:
    1. Pick a random system
    2. Pick a random Δt within the dataset's range
    3. Pick T consecutive frames at that Δt spacing
    4. Assign noise levels (noise-as-masking)
    5. Return trajectory segment + metadata
    """
    
    def __getitem__(self, idx):
        system = self.sample_system()  # weighted by dataset ratio
        
        # Sample inter-frame interval
        dt = self.sample_dt(system.dataset)  
        # ATLAS: 0.1-10 ns, MISATO: 0.08-0.8 ns, MDposit: 0.1-100 ns
        
        # Sample T frames
        n_frames = self.n_frames  # e.g., 50
        start_frame = random.randint(0, system.n_frames - n_frames * dt_frames)
        frame_indices = [start_frame + i * dt_frames for i in range(n_frames)]
        
        coords = system.get_coordinates(frame_indices)  # (T, N_atoms, 3)
        
        # Align all frames to first frame
        coords = self.align_to_first_frame(coords)
        
        # Random SE(3) augmentation
        R = random_rotation_matrix()
        t = random_translation()
        coords = coords @ R.T + t
        
        # Timestamps (continuous, in ns)
        timestamps = torch.tensor([i * dt for i in range(n_frames)])
        
        # Noise-as-masking assignment
        sigma = self.sample_noise_levels(n_frames)  # LogNormal
        task = random.choice(['forecasting', 'interpolation'])
        if task == 'forecasting':
            sigma[0] = 0.0  # first frame clean
        else:
            sigma[0] = 0.0   # first frame clean
            sigma[-1] = 0.0  # last frame clean
        
        conditioning_mask = (sigma == 0.0)
        
        # Load precomputed trunk embeddings
        s_trunk = system.load_trunk_s()
        z_trunk = system.load_trunk_z()
        s_inputs = system.load_s_inputs()
        
        return {
            'coords': coords,
            'timestamps': timestamps,
            'sigma': sigma,
            'conditioning_mask': conditioning_mask,
            's_trunk': s_trunk,
            'z_trunk': z_trunk,
            's_inputs': s_inputs,
            'atom_mask': system.atom_mask,
            'token_to_atom_map': system.token_to_atom_map,
            'molecule_types': system.molecule_types,  # for loss weighting
        }
```

### 5.4 Trunk Embedding Precomputation

This is a critical optimization. Since the trunk is frozen, we can precompute and cache all embeddings:

```python
def precompute_trunk_embeddings(boltz2_model, all_systems, output_dir):
    """
    Run Boltz-2 trunk once per system and cache outputs.
    
    For each system, stores:
    - s_inputs: (n_tokens, c_s) — raw input features
    - s_trunk: (n_tokens, c_s) — refined single representation
    - z_trunk: (n_tokens, n_tokens, c_z) — refined pair representation
    
    Storage estimate:
    - Per 200-residue protein: ~200 × 384 + 200² × 128 ≈ 5 MB
    - 20,000 systems: ~100 GB total
    """
    boltz2_model.eval()
    for system in tqdm(all_systems):
        features = prepare_boltz2_input(system)
        
        with torch.no_grad():
            # Run full trunk with recycling
            s, z, s_inputs = boltz2_model.run_trunk(features)
        
        # Save to disk
        path = output_dir / f"{system.id}_trunk.npz"
        np.savez_compressed(path,
            s_trunk=s.cpu().numpy().astype(np.float16),  # bfloat16 for space
            z_trunk=z.cpu().numpy().astype(np.float16),
            s_inputs=s_inputs.cpu().numpy().astype(np.float16),
        )
```

---

## 6. Training Procedure

### 6.1 Training Phases

**Phase 1 — Equilibrium MD Training (main phase):**
- Data: ATLAS + MISATO + MDposit (1:1:1 resampled)
- Loss: `L_struct + β_flex · L_flex`
- Temporal attention: bidirectional (time-reversal symmetry)
- Training frames: 20–50 frames per sample (memory-dependent)
- Inter-frame Δt: randomly sampled per dataset
- Duration: Until convergence (monitor W2-distance on validation set)

**Phase 2 — Unbinding Fine-tuning (optional):**
- Data: DD-13M metadynamics trajectories
- Loss: `L_struct + β_center · L_center`
- Temporal attention: **causal mask** (metadynamics is history-dependent)
- Frame step: 10 ps per iteration (auto-regressive)
- Initialize from Phase 1 checkpoint

### 6.2 Memory Management

The main memory bottleneck is the Spatial-Temporal Diffusion Module processing T frames simultaneously. Key strategies:

1. **Precomputed trunk embeddings** — eliminates trunk memory overhead during training
2. **Gradient checkpointing** — for the 24-block token transformer
3. **Mixed precision** (bfloat16) — following Boltz-2
4. **Dynamic frame count** — adjust T based on system size:
   - Small systems (<200 residues): T = 50 frames
   - Medium systems (200-500 residues): T = 30 frames
   - Large systems (>500 residues): T = 20 frames

**Memory estimate per sample (approximate):**
- Coordinates: T × N_atoms × 3 × 4 bytes
- Atom representations: T × N_atoms × 128 × 2 bytes (bf16)
- Token representations: T × N_tokens × 384 × 2 bytes (bf16)
- Temporal attention: T² × N_tokens × 16_heads × 2 bytes
- For T=50, N_tokens=200: ~4 GB per sample → batch size 8 per GPU

### 6.3 Training Loop Pseudocode

```python
def training_step(self, batch, batch_idx):
    coords = batch['coords']           # (B, T, N_atoms, 3)
    timestamps = batch['timestamps']    # (B, T)
    sigma = batch['sigma']              # (B, T)
    cond_mask = batch['conditioning_mask']  # (B, T)
    
    # Construct noisy trajectory
    eps = torch.randn_like(coords)
    x_noisy = coords + sigma[..., None, None] * eps
    x_noisy = torch.where(cond_mask[..., None, None], coords, x_noisy)
    
    # Forward pass
    x_denoised = self.model(
        x_noisy=x_noisy,
        timestamps=timestamps,
        sigma=sigma,
        s_trunk=batch['s_trunk'],
        z_trunk=batch['z_trunk'],
        s_inputs=batch['s_inputs'],
    )
    
    # === Structure Reconstruction Loss ===
    target_mask = ~cond_mask  # only compute loss on target frames
    
    # Per-atom MSE with molecule-type weighting
    weights = self.get_atom_weights(batch['molecule_types'])
    # weights: protein=1, DNA/RNA=5, ligand=10
    
    mse_per_frame = ((x_denoised - coords)**2).sum(dim=-1)  # (B, T, N_atoms)
    mse_per_frame = (mse_per_frame * weights).mean(dim=-1)   # (B, T)
    
    # EDM noise-level weighting
    edm_weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2
    edm_weight[cond_mask] = 0.0  # zero weight for conditioning frames
    
    l_struct = (edm_weight * mse_per_frame).sum() / target_mask.sum()
    
    # + Bond loss, smooth lDDT loss (per AF3)
    l_bond = self.bond_loss(x_denoised, batch)
    l_lddt = self.smooth_lddt_loss(x_denoised, batch)
    
    l_struct_total = l_struct + self.alpha_bond * l_bond + l_lddt
    
    # === Flexibility Loss ===
    # Computed on the ensemble of denoised frames
    l_flex = self.flexibility_loss(x_denoised, coords, batch)
    
    # === Total ===
    loss = l_struct_total + self.beta_flex * l_flex
    
    self.log('train/loss', loss)
    self.log('train/l_struct', l_struct)
    self.log('train/l_flex', l_flex)
    
    return loss
```

### 6.4 Flexibility Loss Implementation

```python
def flexibility_loss(self, x_pred, x_gt, batch):
    """
    Supervise distributional properties of the generated ensemble.
    
    x_pred, x_gt: (B, T, N_atoms, 3)
    """
    # --- Absolute flexibility (RMSF) ---
    pred_mean = x_pred.mean(dim=1, keepdim=True)
    gt_mean = x_gt.mean(dim=1, keepdim=True)
    
    rmsf_pred = ((x_pred - pred_mean)**2).mean(dim=1).sum(dim=-1).sqrt()  # (B, N)
    rmsf_gt = ((x_gt - gt_mean)**2).mean(dim=1).sum(dim=-1).sqrt()
    
    weights = self.get_atom_weights(batch['molecule_types'])
    l_abs = (weights * (rmsf_pred - rmsf_gt)**2).mean()
    
    # --- Global relative flexibility (pairwise distance std) ---
    # For Cα atoms (protein) and heavy atoms (ligand)
    ca_mask = batch['ca_mask']  # (B, N) bool
    
    # Pairwise distances: (B, T, N_ca, N_ca)
    ca_pred = x_pred[:, :, ca_mask]
    dist_pred = torch.cdist(ca_pred, ca_pred)  # (B, T, N_ca, N_ca)
    std_pred = dist_pred.std(dim=1)  # (B, N_ca, N_ca)
    
    ca_gt = x_gt[:, :, ca_mask]
    dist_gt = torch.cdist(ca_gt, ca_gt)
    std_gt = dist_gt.std(dim=1)
    
    # Distance-dependent weights
    mean_dist = dist_gt.mean(dim=1)
    gamma = torch.where(mean_dist <= 5.0, 4.0,
            torch.where(mean_dist <= 10.0, 2.0, 1.0))
    
    l_rel_global = (gamma * (std_pred - std_gt)**2).mean()
    
    # --- Local relative flexibility (intra-residue) ---
    # Computed per-residue for side chain atoms
    l_rel_local = self.compute_local_flex(x_pred, x_gt, batch)
    
    return self.beta_abs * l_abs + \
           self.beta_rel_g * l_rel_global + \
           self.beta_rel_l * l_rel_local
```

---

## 7. Inference Pipeline

### 7.1 Hierarchical Trajectory Generation

```python
def generate_trajectory(
    self,
    initial_structure: Tensor,  # (N_atoms, 3)
    total_time_ns: float = 1000.0,  # 1 μs
    coarse_dt_ns: float = 5.0,
    fine_dt_ns: float = 0.1,
    generation_window: int = 40,  # frames per AR block
    history_window: int = 10,     # conditioning frames from history
    n_diffusion_steps: int = 20,
    sigma_noise: float = 1.75,
    eta: float = 1.5,
) -> Tensor:
    """
    Generate a long trajectory using hierarchical sampling.
    
    Returns: (N_total_frames, N_atoms, 3)
    """
    # === Compute trunk embeddings once ===
    s_trunk, z_trunk, s_inputs = self.compute_trunk(initial_structure)
    
    # === Stage 1: Coarse-grained forecasting ===
    coarse_times = torch.arange(0, total_time_ns + coarse_dt_ns, coarse_dt_ns)
    coarse_traj = [initial_structure.unsqueeze(0)]  # list of (1, N, 3)
    
    idx = 0
    while idx < len(coarse_times) - 1:
        n_batch = min(generation_window, len(coarse_times) - 1 - idx)
        target_times = coarse_times[idx + 1 : idx + 1 + n_batch]
        
        # Select history context
        history = coarse_traj[-history_window:]
        history_times = coarse_times[max(0, idx + 1 - history_window) : idx + 1]
        
        x_cond = torch.cat(history, dim=0)  # (H, N, 3)
        t_all = torch.cat([history_times, target_times])
        
        # Initialize target frames from noise
        x_target = torch.randn(n_batch, *initial_structure.shape) * self.sigma_max
        
        # Denoise
        x_gen = self.sample_diffusion(
            x_cond=x_cond,
            x_target_init=x_target,
            t_cond=history_times,
            t_target=target_times,
            s_trunk=s_trunk, z_trunk=z_trunk, s_inputs=s_inputs,
            mode='forecast',
            n_steps=n_diffusion_steps,
        )
        
        coarse_traj.extend([x_gen[i:i+1] for i in range(n_batch)])
        idx += n_batch
    
    coarse_traj = torch.cat(coarse_traj, dim=0)  # (N_coarse, N, 3)
    
    # === Stage 2: Fine-grained interpolation ===
    n_interp = int(coarse_dt_ns / fine_dt_ns) - 1  # frames between coarse anchors
    fine_traj = [coarse_traj[0:1]]
    
    for i in range(len(coarse_traj) - 1):
        t_start = coarse_times[i]
        t_end = coarse_times[i + 1]
        t_interp = torch.linspace(t_start, t_end, n_interp + 2)[1:-1]
        
        x_anchors = torch.stack([coarse_traj[i], coarse_traj[i + 1]])
        
        # Initialize intermediate frames from noise
        x_interp_init = torch.randn(n_interp, *initial_structure.shape) * self.sigma_max
        
        x_interp = self.sample_diffusion(
            x_cond=x_anchors,
            x_target_init=x_interp_init,
            t_cond=torch.tensor([t_start, t_end]),
            t_target=t_interp,
            s_trunk=s_trunk, z_trunk=z_trunk, s_inputs=s_inputs,
            mode='interpolate',
            n_steps=n_diffusion_steps,
        )
        
        fine_traj.extend([x_interp[j:j+1] for j in range(n_interp)])
        fine_traj.append(coarse_traj[i+1:i+2])
    
    return torch.cat(fine_traj, dim=0)
```

### 7.2 EDM Sampling (Shared for Forecast and Interpolation)

```python
def sample_diffusion(self, x_cond, x_target_init, t_cond, t_target,
                     s_trunk, z_trunk, s_inputs, mode, n_steps=20):
    """
    EDM sampling with SDE solver.
    
    mode: 'forecast' or 'interpolate'
    """
    # Noise schedule (EDM)
    sigma_schedule = self.get_sigma_schedule(n_steps)  # decreasing
    
    x_target = x_target_init
    
    for k in range(n_steps - 1):
        sigma_k = sigma_schedule[k]
        sigma_next = sigma_schedule[k + 1]
        
        # Assemble full sequence
        if mode == 'forecast':
            x_full = torch.cat([x_cond, x_target], dim=0)
            sigma_full = torch.cat([
                torch.zeros(len(x_cond)),
                torch.full((len(x_target),), sigma_k)
            ])
            t_full = torch.cat([t_cond, t_target])
        else:  # interpolate
            x_full = torch.cat([x_cond[0:1], x_target, x_cond[1:2]], dim=0)
            sigma_full = torch.cat([
                torch.zeros(1),
                torch.full((len(x_target),), sigma_k),
                torch.zeros(1),
            ])
            t_full = torch.cat([t_cond[0:1], t_target, t_cond[1:2]])
        
        # Predict denoised coordinates
        x_denoised = self.model(
            x_noisy=x_full.unsqueeze(0),
            timestamps=t_full.unsqueeze(0),
            sigma=sigma_full.unsqueeze(0),
            s_trunk=s_trunk, z_trunk=z_trunk, s_inputs=s_inputs,
        ).squeeze(0)
        
        # Extract target predictions
        if mode == 'forecast':
            x_pred_target = x_denoised[len(x_cond):]
        else:
            x_pred_target = x_denoised[1:-1]
        
        # EDM update step
        d = (x_target - x_pred_target) / sigma_k
        x_target = x_pred_target + sigma_next * d
    
    return x_target
```

### 7.3 Unbinding Trajectory Generation (Auto-regressive)

```python
def generate_unbinding(self, complex_structure, n_trajectories=20,
                       total_ps=60, dt_ps=10):
    """
    Generate ligand unbinding trajectories.
    Uses causal temporal attention (model must be fine-tuned on DD-13M).
    """
    n_steps = total_ps // dt_ps
    
    all_trajectories = []
    for _ in range(n_trajectories):
        trajectory = [complex_structure.unsqueeze(0)]
        
        for step in range(n_steps):
            # Context: all previously generated frames
            context = torch.cat(trajectory, dim=0)
            context_times = torch.arange(len(context)) * dt_ps
            
            # Target: next frame
            target_time = torch.tensor([(len(context)) * dt_ps])
            x_init = torch.randn_like(complex_structure.unsqueeze(0)) * self.sigma_max
            
            x_next = self.sample_diffusion(
                x_cond=context,
                x_target_init=x_init,
                t_cond=context_times,
                t_target=target_time,
                mode='forecast',
                n_steps=20,
            )
            
            trajectory.append(x_next)
        
        all_trajectories.append(torch.cat(trajectory, dim=0))
    
    return torch.stack(all_trajectories)  # (n_traj, n_frames, N_atoms, 3)
```

---

## 8. Implementation Roadmap

### Phase 0: Environment Setup (Week 1)

- [ ] Install Boltz-2 from source
- [ ] Verify Boltz-2 inference works
- [ ] Extract and understand Boltz-2 diffusion module internals
- [ ] Set up training infrastructure (PyTorch Lightning, WandB logging)
- [ ] Identify exact weight names for trunk vs. diffusion components

### Phase 1: Core Architecture (Weeks 2–4)

- [ ] Implement `TemporalAttentionWithDecay` module
- [ ] Implement `SpatialTemporalTransformerBlock` (token level)
- [ ] Implement `SpatialTemporalAtomBlock` (atom level, encoder + decoder)
- [ ] Implement `PerFrameEDMConditioning`
- [ ] Implement `SpatialTemporalDiffusionModule` (full assembly)
- [ ] Write unit tests: single-frame output matches Boltz-2 (zero-init temporal)
- [ ] Write unit tests: multi-frame shapes correct, gradients flow

### Phase 2: Data Pipeline (Weeks 3–5, overlaps with Phase 1)

- [ ] Download and organize ATLAS dataset
- [ ] Download and organize MISATO dataset
- [ ] Download and organize MDposit/DynaRepo
- [ ] Implement trajectory preprocessing (solvent removal, alignment, valency checks)
- [ ] Implement Boltz-2 tokenization for all system types
- [ ] Run trunk embedding precomputation (batch job on GPU cluster)
- [ ] Implement `TrajectoryDataset` with dynamic sub-sampling
- [ ] Implement noise-as-masking sample construction
- [ ] Implement weighted dataset sampler (1:1:1 ratio)

### Phase 3: Training (Weeks 5–8)

- [ ] Implement all loss functions (L_struct, L_flex, L_bond, L_smooth_lddt)
- [ ] Implement training loop with EDM weighting
- [ ] Initial training run: small subset, verify loss decreases
- [ ] Memory profiling and optimization
- [ ] Full training on ATLAS + MISATO + MDposit
- [ ] Monitor: pairwise RMSD correlation, Cα RMSF correlation on ATLAS validation
- [ ] Ablation: hierarchical sampling vs. pure forecasting

### Phase 4: Evaluation (Weeks 8–10)

- [ ] ATLAS test set: pairwise RMSD, Cα RMSF, W2-distance
- [ ] ATLAS-OOD test set: ≤40% sequence identity
- [ ] MISATO test set: Interaction Map Similarity
- [ ] MISATO-OOD test set: physical stability metrics (1 μs trajectories)
- [ ] BioEmu benchmarks: cryptic pockets, local unfolding, domain motion
- [ ] Timing benchmarks: generation speed vs. conventional MD

### Phase 5: Unbinding Extension (Weeks 10–12)

- [ ] Download DD-13M dataset
- [ ] Implement causal masking variant
- [ ] Implement ligand geometric center loss
- [ ] Fine-tune on DD-13M from Phase 3 checkpoint
- [ ] Evaluate: precision/recall of unbinding pathways
- [ ] Case studies: RXRA, CERT1

### Phase 6: Polish & Release (Weeks 12–14)

- [ ] Clean API for trajectory generation
- [ ] Hierarchical sampling with configurable time scales
- [ ] Documentation and tutorials
- [ ] Benchmark suite
- [ ] Model checkpoint release

---

## 9. Risk Analysis & Mitigations

### 9.1 Boltz-2 ↔ Protenix Representation Differences

**Risk:** Boltz-2 and Protenix may produce different single/pair representations for the same input, meaning the Spatial-Temporal Diffusion Module trained on one may not transfer to the other.

**Mitigation:** Since the trunk is frozen and the diffusion module is trained from scratch on top of it, this is not a transfer issue — we are training our own diffusion module on Boltz-2 representations. The spatial attention weights are initialized from Boltz-2's own diffusion module (not Protenix), ensuring compatibility.

### 9.2 Boltz-2 Training Code Not Yet Released

**Risk:** Boltz-2 training infrastructure is not yet public. We need to implement our own training loop.

**Mitigation:** Boltz-1's training code IS available and provides the training patterns (PyTorch Lightning, data loading, loss computation). The key differences for Boltz-2 (DiffusionConditioning, template module) are well-documented in the DeepWiki analysis and paper. Our custom training loop only needs to handle the frozen trunk + trainable diffusion module pattern, which is simpler than full Boltz-2 training.

### 9.3 Memory Constraints for Multi-Frame Processing

**Risk:** Processing T frames simultaneously multiplies memory usage by T. For large systems + many frames, this may exceed GPU memory.

**Mitigation:**
- Precompute trunk embeddings (eliminates trunk memory)
- Dynamic frame count based on system size
- Gradient checkpointing in temporal attention
- Accumulate gradients across micro-batches
- Start with T=20 and scale up

### 9.4 Training Data Scale

**Risk:** >2,000 μs of MD data requires significant storage and preprocessing time.

**Mitigation:**
- Start with ATLAS (smallest, highest quality) for initial debugging
- Add MISATO for protein-ligand capability
- Add MDposit last for maximum diversity
- Use streaming data loading to avoid loading entire dataset into memory

### 9.5 Reproducing Physical Stability Over 1 μs

**Risk:** The hierarchical sampling strategy is critical for long trajectory stability. Incorrect implementation could lead to error accumulation.

**Mitigation:**
- Start by validating short trajectories (100 ns)
- Monitor physical stability metrics (RMSD after minimization, MolProbity, Ramachandran)
- Systematically vary coarse interval and evaluate stability
- The paper reports 5 ns coarse interval works well

---

## 10. Appendix: Key Equations

### A. EDM Preconditioning

```
c_in(σ)    = 1 / √(σ² + σ²_data)
c_skip(σ)  = σ²_data / (σ² + σ²_data)
c_out(σ)   = σ · σ_data / √(σ² + σ²_data)
c_noise(σ) = ln(σ) / 4

D_θ(x; σ)  = c_skip(σ) · x + c_out(σ) · F_θ(c_in(σ) · x; c_noise(σ))
```

### B. Training Loss Weight

```
λ(σ) = (σ² + σ²_data) / (σ · σ_data)²
```

### C. Temporal Attention Bias (Ornstein-Uhlenbeck Derived)

```
B_{ij}^{(h)} = -λ_h · |t_i - t_j|

After softmax:
A_{ij} ∝ exp(q_i^T k_j / √d) · exp(-λ_h · |t_i - t_j|)

Physical basis:
ρ(τ) = exp(-λ|τ|)     (autocorrelation of OU process)
K(τ) = (k_B T / k) · exp(-λ|τ|)     (autocovariance)
```

### D. Ornstein-Uhlenbeck Process

```
dx(t) = -(k/γ) x(t) dt + √(2k_BT/γ) dW_t

Explicit solution:
x(t) = x(0) e^{-λt} + σ ∫₀ᵗ e^{-λ(t-s)} dW_s

where λ = k/γ, σ = √(2k_BT/γ)
```

### E. Flexibility Loss Components

```
Absolute RMSF:
RMSF_l = √(1/K Σ_k ||x_l^(k) - x̄_l||² + ε)
L_abs = mean_l [w_l · (RMSF^pred_l - RMSF^GT_l)²]

Global relative (pairwise distance std):
σ_lm = std_k(||x_l^(k) - x_m^(k)||)
L_rel-global = mean_{l,m} [ω_lm · γ_lm · (σ^pred_lm - σ^GT_lm)²]

Distance weights γ_lm: ≤5Å → 4, (5,10] → 2, >10Å → 1
Type weights ω_lm: lig-lig → 10, prot-lig → 5, prot-prot → 1

Local relative (intra-residue):
L_rel-local = mean_{(l,m)∈R} [(σ^pred_lm - σ^GT_lm)²]

Combined: L_flex = 1·L_abs + 4·L_rel-global + 4·L_rel-local
```

### F. Unbinding Metrics

```
Precision(τ) = (1/N) Σ_i I(min_j ||g_i - e_j|| < τ)
Recall(τ)    = (1/M) Σ_j I(min_i ||g_i - e_j|| < τ)
```

---

## Summary of Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base model | Boltz-2 | MIT license, pretrained weights, AF3-equivalent architecture |
| Trunk handling | Frozen + precomputed | Massive training speedup |
| Temporal attention init | Zero-init output projection | Model starts as single-frame predictor, gradually learns temporal |
| λ_h initialization | ALiBi geometric [0.004, 0.7] | Covers fast (side chains) to slow (domain motions) timescales |
| Training paradigm | Noise-as-masking | Unified forecasting + interpolation, no separate encoders |
| Long trajectory strategy | Hierarchical (coarse + fine) | Mitigates error accumulation |
| Unbinding | Causal mask + DD-13M fine-tune | History-dependent metadynamics requires causal attention |
| Memory optimization | Precomputed trunk + gradient checkpointing + dynamic T | Critical for multi-frame training feasibility |
