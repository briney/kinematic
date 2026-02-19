# Boltz-Kinema Technical Debt Remediation Workplan

**Date:** February 19, 2026  
**Context:** Comprehensive technical review findings (14 issues)  
**Objective:** Eliminate correctness blockers first, then reduce architectural and operational debt while increasing test coverage.

---

## 1. Executive Summary

This plan addresses all identified issues in strict severity order and recommended fix sequence:

1. **Runtime correctness blockers** that can crash training/inference or produce wrong samples.
2. **Data/evaluation correctness risks** that silently degrade model quality.
3. **API/config inconsistencies and reproducibility gaps** that create hidden behavior drift.
4. **Test and codebase maintainability debt** that will compound future change cost.

The highest-risk items are:
- atom-window shape crashes for non-divisible atom counts,
- EDM sampler stopping early (never reaching `sigma=0`),
- incompatible `.safetensors` loading paths.

These three items are treated as a stabilization gate before all other work.

---

## 2. Severity and Priority Model

- **Critical:** Can crash common workflows or produce fundamentally incorrect outputs.
- **High:** Silent correctness/reliability issues with material training/inference impact.
- **Medium:** Technical debt causing ambiguity, drift, or long-term fragility.

**Priority order = severity + dependency constraints.**

---

## 3. Ordered Worklist (Severity + Fix Order)

## 3.1 Critical (Fix Order 1-3)

### P1. Atom window divisibility crash in spatial-temporal atom modules
- **Severity:** Critical
- **Files:**
  - `src/boltzkinema/model/spatial_temporal_atom.py:132`
  - `src/boltzkinema/model/spatial_temporal_atom.py:135`
  - `src/boltzkinema/model/spatial_temporal_atom.py:298`
  - `src/boltzkinema/model/spatial_temporal_atom.py:301`
  - `src/boltzkinema/data/collator.py:28`
- **Problem:** Encoder/decoder assume `M % W == 0` and reshape using `NW = M // W`, causing runtime failures for common atom counts.
- **Proposed fix:**
  1. Add robust in-module padding/trimming in atom encoder and decoder:
     - compute `NW = ceil(M / W)`,
     - pad `q/c/mask` to `NW * W`,
     - run windowed ops,
     - trim outputs back to original `M`.
  2. Keep this logic in model code (not only collator) so inference paths are protected too.
  3. Ensure padded atoms are fully masked from attention/loss effects.
- **Implementation steps:**
  1. Add helper in `spatial_temporal_atom.py` for padded window shaping and unshaping.
  2. Apply helper in both encoder and decoder before window ops.
  3. Audit all `view(...)` calls in these modules for divisibility assumptions.
- **Validation:**
  - Unit tests with `M=33`, `M=65`, and mixed-batch atom sizes.
  - End-to-end forward smoke test with non-divisible `M`.

### P2. Sampler off-by-one: denoising never reaches zero noise
- **Severity:** Critical
- **Files:**
  - `src/boltzkinema/inference/sampler.py:86`
  - `src/boltzkinema/inference/sampler.py:205`
- **Problem:** Loop runs `range(self.n_steps - 1)` while schedule has `n_steps + 1` entries ending at `0`, so last update to zero is skipped.
- **Proposed fix:**
  1. Iterate `for k in range(self.n_steps)` so final step transitions `sigma_min -> 0`.
  2. Ensure update remains numerically safe (`sigma_k > 0` during division).
  3. Add explicit regression test for final-step behavior.
- **Implementation steps:**
  1. Change loop bounds.
  2. Add asserts in tests that final sigma is zero-targeted.
- **Validation:**
  - Unit test: schedule endpoints and number of model evaluations.
  - Regression test: output differs from previous bugged path and shows lower residual noise.

### P3. `.safetensors` discovered but loaded with `torch.load`
- **Severity:** Critical
- **Files:**
  - `src/boltzkinema/training/trainer.py:356`
  - `src/boltzkinema/training/trainer.py:362`
  - `scripts/generate.py:158`
  - `scripts/generate.py:164`
- **Problem:** Paths detect `model*.safetensors` but loading uses `torch.load`, which is incompatible.
- **Proposed fix:**
  1. Add shared checkpoint-loading utility that dispatches by suffix:
     - `.safetensors` -> `safetensors.torch.load_file`,
     - others -> `torch.load`.
  2. Use same loader for both training resume and inference.
  3. Add graceful error messaging if `safetensors` dependency is missing.
- **Implementation steps:**
  1. Create `src/boltzkinema/model/checkpoint_io.py`.
  2. Refactor `trainer.py` and `scripts/generate.py` to use it.
  3. Add dependency note in docs.
- **Validation:**
  - Unit tests with mocked `.bin` and `.safetensors` selection.
  - Resume/inference smoke tests on both formats.

---

## 3.2 High (Fix Order 4-8)

### P4. `observed_atom_mask` loading bug
- **Severity:** High
- **Files:** `src/boltzkinema/data/dataset.py:83`
- **Problem:** Gate checks wrong source (`ref`) before reading coords mask, causing fallback to all-ones in many cases.
- **Proposed fix:**
  1. Load `observed_atom_mask` directly from coords NPZ if present.
  2. Cache coords metadata to avoid repeat disk reads.
  3. Fallback to all-ones only when key truly absent.
- **Validation:**
  - Unit test with coords NPZ containing partial observed mask.
  - Assert mask propagates to loss masking.

### P5. MISATO coordinate unit heuristic is unsafe
- **Severity:** High
- **Files:** `scripts/preprocess_misato.py:105`
- **Problem:** Heuristic `coords.max() > 100` can misclassify units, yielding 10x scale errors.
- **Proposed fix:**
  1. Add explicit `--coords-unit {auto,nm,angstrom}` CLI argument (default `auto`).
  2. In auto mode, use stronger checks:
     - HDF metadata if available,
     - robust percentile-based spatial extent checks,
     - optional bond-length sanity checks.
  3. Emit warning with inferred unit and confidence.
- **Validation:**
  - Unit tests for nm and Angstrom fixtures.
  - Roundtrip sanity checks on expected bond length ranges.

### P6. Bond loss dilution from padded fake bonds
- **Severity:** High
- **Files:**
  - `src/boltzkinema/data/collator.py:59`
  - `src/boltzkinema/training/losses.py:286`
- **Problem:** Denominator includes padded bond slots, weakening bond supervision when batches mix bonded/unbonded samples.
- **Proposed fix:**
  1. Collator: add `bond_mask` (`True` for real bonds).
  2. Loss: apply bond mask in numerator and denominator.
  3. Keep behavior zero-safe when no valid bonds exist.
- **Validation:**
  - Tests for mixed bond/no-bond batches.
  - Regression test ensuring equal loss scale irrespective of padding.

### P7. Metrics fail on GPU tensors due to direct `.numpy()`
- **Severity:** High
- **Files:**
  - `src/boltzkinema/evaluation/metrics.py:104`
  - `src/boltzkinema/evaluation/metrics.py:143`
  - `src/boltzkinema/evaluation/metrics.py:194`
  - `src/boltzkinema/evaluation/metrics.py:258`
  - `src/boltzkinema/evaluation/metrics.py:315`
- **Problem:** `.numpy()` on CUDA tensors raises runtime errors.
- **Proposed fix:**
  1. Add helper `_to_numpy_cpu(t)` using `t.detach().cpu().numpy()`.
  2. Replace all direct `.numpy()` conversions.
  3. Keep outputs identical for CPU inputs.
- **Validation:**
  - CPU tests unchanged.
  - CUDA-conditional tests (skip when CUDA unavailable).

### P8. Inference YAML/schema mismatch silently ignores user config
- **Severity:** High
- **Files:**
  - `configs/inference.yaml:40`
  - `configs/inference.yaml:42`
  - `scripts/generate.py:78`
  - `scripts/generate.py:79`
  - `scripts/generate.py:80`
  - `scripts/generate.py:81`
- **Problem:** YAML keys (`coarse_n_frames`, `fine_interp_factor`) do not map to dataclass fields (`fine_dt_ns`, `generation_window`, `history_window`), and unknown keys are dropped.
- **Proposed fix:**
  1. Align YAML keys with `InferenceConfig` exactly.
  2. Add strict config validation:
     - fail on unknown keys,
     - fail on missing required keys when strict mode enabled.
  3. Optionally support backward-compatible aliases with deprecation warnings.
- **Validation:**
  - Config parse tests for valid and invalid key sets.
  - Integration test confirming hierarchical params affect runtime behavior.

---

## 3.3 Medium (Fix Order 9-14)

### P9. Dataset constructor args are unused/ambiguous
- **Severity:** Medium
- **Files:** `src/boltzkinema/data/dataset.py:199`
- **Problem:** `trunk_cache_dir` and `coords_dir` are accepted but not used; manifest paths dominate.
- **Proposed fix:**
  1. Define clear precedence:
     - explicit manifest paths,
     - optional override root dirs from ctor.
  2. Implement override logic and document it.
  3. Fail fast if resolved paths do not exist.
- **Validation:**
  - Tests for path override behavior.
  - Error-path tests for missing files.

### P10. Dataset map-style semantics and frame index bias
- **Severity:** Medium
- **Files:**
  - `src/boltzkinema/data/dataset.py:254`
  - `src/boltzkinema/data/dataset.py:256`
  - `src/boltzkinema/data/dataset.py:266`
- **Problem:** `idx` is ignored; frame-start formula undercounts valid starts (`system.n_frames - self.n_frames * dt_frames` should account for last usable index).
- **Proposed fix:**
  1. Decide model:
     - either true map-style (idx-driven deterministic sampling),
     - or convert to iterable dataset semantics.
  2. Fix max-start formula to include full valid range:
     - `max_start = system.n_frames - 1 - (self.n_frames - 1) * dt_frames`.
  3. Improve reproducibility by deterministic per-worker RNG seeding strategy.
- **Validation:**
  - Tests for index range correctness across short/long trajectories.
  - Reproducibility tests across workers.

### P11. `noise_scale` and `step_scale` are dead config knobs
- **Severity:** Medium
- **Files:**
  - `src/boltzkinema/inference/sampler.py:59`
  - `src/boltzkinema/inference/sampler.py:60`
- **Problem:** Exposed parameters are never used in sampling updates.
- **Proposed fix (preferred):**
  1. Implement Karras churn/noise logic using these parameters.
  2. Document exact semantics and valid ranges.
- **Fallback fix:** remove knobs if stochastic sampler behavior is intentionally disabled.
- **Validation:**
  - Unit tests proving parameter effect on trajectory outputs/statistics.

### P12. Missing tests in core risk areas (empty test modules)
- **Severity:** Medium
- **Files:**
  - `tests/test_shapes.py:1`
  - `tests/test_temporal_attention.py:1`
  - `tests/test_single_frame_equivalence.py:1`
  - `tests/test_noise_masking.py:1`
- **Problem:** Critical invariants are untested despite files implying coverage.
- **Proposed fix:** implement full suites:
  1. shape tests for non-divisible atoms/tokens and mixed padding,
  2. temporal attention tests (causal mask, decay bias monotonicity, zero-init identity behavior),
  3. single-frame equivalence tests vs Boltz-2 behavior for temporal-disabled edge case,
  4. noise-as-masking tests for forecast/interpolate masks and sigma rules.
- **Validation:**
  - New tests required in CI gate.

### P13. High duplication across preprocessing scripts
- **Severity:** Medium
- **Files:**
  - `scripts/preprocess_atlas.py:76`
  - `scripts/preprocess_cath.py:100`
  - `scripts/preprocess_dd13m.py:82`
  - `scripts/preprocess_mdposit.py:83`
  - `scripts/preprocess_megasim.py:201`
  - `scripts/preprocess_misato.py:69`
  - `scripts/preprocess_octapeptides.py:84`
- **Problem:** Nearly identical preprocess pipeline repeated across scripts; bug fixes will diverge.
- **Proposed fix:**
  1. Create shared `scripts/preprocess_common.py` for core pipeline steps and manifest entry creation.
  2. Keep dataset-specific discovery and quirks in thin wrappers.
  3. Consolidate duplicated CLI/output behavior.
- **Validation:**
  - Golden manifest-entry tests for each dataset script before/after refactor.

### P14. Placeholder paths cause avoidable runtime failures
- **Severity:** Medium
- **Files:**
  - `configs/train_mutant_enrichment.yaml:31`
  - `configs/train_unbinding.yaml:30`
  - `configs/inference.yaml:45`
- **Problem:** Defaults include unresolved `step_XXXXX` placeholders.
- **Proposed fix:**
  1. Add explicit preflight validation in training/inference entrypoints.
  2. Fail fast with actionable error messages listing resolution options.
  3. Optionally auto-resolve latest checkpoint when directory parent exists.
- **Validation:**
  - CLI tests for placeholder detection and auto-resolution behavior.

---

## 4. Phase Plan and Dependency Graph

## Phase 0: Stabilization Gate (P1-P3)
- **Goal:** Remove hard crashes and fundamentally incorrect denoising/checkpoint behavior.
- **Exit criteria:**
  - non-divisible atom batch forward pass succeeds,
  - sampler reaches zero-noise endpoint,
  - both `.bin` and `.safetensors` checkpoints load for train/infer.

## Phase 1: Correctness Hardening (P4-P8)
- **Goal:** Eliminate silent data/evaluation/config errors.
- **Exit criteria:**
  - observed mask and bond-mask behavior verified,
  - MISATO unit handling deterministic and logged,
  - metrics GPU-safe,
  - inference config strictness enabled.

## Phase 2: API/Reproducibility Cleanup (P9-P11)
- **Goal:** Align interfaces with behavior and remove ambiguous controls.
- **Exit criteria:**
  - dataset path precedence documented and tested,
  - sampling/index logic deterministic policy chosen,
  - `noise_scale/step_scale` either implemented or removed.

## Phase 3: Test & Maintainability Debt Reduction (P12-P14)
- **Goal:** Improve change safety and reduce duplication.
- **Exit criteria:**
  - missing core tests implemented,
  - preprocessing shared core extracted,
  - placeholder preflight checks in place.

---

## 5. Detailed Test Strategy

- **Unit tests**
  - Atom module shape/padding invariants.
  - Sampler step-count and endpoint invariants.
  - Checkpoint IO dispatch per file suffix.
  - Loss correctness with bond masks.
  - Metrics CPU/CUDA conversion parity.
  - Config strict parse and unknown-key rejection.

- **Integration tests**
  - Minimal forward path (train mode) with odd atom counts.
  - Inference generation using hierarchical mode with verified parameter ingestion.
  - Resume-from-checkpoint paths for both optimizer and model-only resume.

- **Regression tests**
  - Prevent reintroduction of off-by-one sampler bug.
  - Prevent reintroduction of `.safetensors` loading bug.
  - Prevent accidental dropping of observed atom masks.

---

## 6. Tooling and Quality Gates

- Add/enable CI checks:
  1. `ruff check src scripts tests`
  2. `pytest -q`
  3. Optional: focused smoke test for inference sampler.

- Enforce no placeholder checkpoints in production config runs via preflight validators.

---

## 7. Estimated Effort (Rough)

- **Phase 0:** 1-2 days
- **Phase 1:** 2-3 days
- **Phase 2:** 1-2 days
- **Phase 3:** 2-4 days (depends on refactor depth for preprocessing scripts)

**Total:** ~6-11 engineering days.

---

## 8. Deliverables Checklist

- [x] P1 atom window padding/trimming fix + tests
- [x] P2 sampler loop fix + tests
- [x] P3 unified checkpoint IO with `.safetensors` support + tests
- [x] P4 observed mask fix + tests
- [x] P5 MISATO unit handling hardening + tests
- [x] P6 bond mask in collator/loss + tests
- [x] P7 metrics CPU conversion helper + CUDA-safe tests
- [x] P8 strict inference config schema + updated YAML
- [x] P9 dataset path precedence implementation + docs/tests
- [x] P10 dataset sampling/index correctness and reproducibility updates + tests
- [x] P11 implement or remove dead sampler knobs + tests/docs
- [ ] P12 implement missing test modules
- [ ] P13 preprocessing common-core refactor
- [ ] P14 placeholder path preflight checks

---

## 9. Change Control Notes

- Prioritize minimal, behavior-preserving fixes in Phase 0/1.
- Keep refactors (Phase 3) separated from correctness fixes to simplify review.
- For each priority item, submit as independent PR-sized change with:
  - before/after behavior summary,
  - explicit test coverage delta,
  - rollback notes if needed.
