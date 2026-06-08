# Project progress — pipeline + methodology buildout

> ⚠️ **PARTIALLY STALE (2026-06-05).** The "Empirical findings" section below —
> in particular the Phase 5b intrinsic-dimension table (TwoNN 9.4–27.9) and the
> curvature ratios (0.13–0.24) — was produced by estimator code that the
> 2026-06-05 audit found **biased / confounded** (see `AUDIT.md` §2 #1–#2 and
> `results/_STALE_pre_fix_20260605/`). Those numbers must be **regenerated** on
> the fixed estimators before use. For current pipeline *status* and the open
> action list, `AUDIT.md` (2026-06-05) supersedes this file; for the file/output
> inventory see `INVENTORY.md`. Base-model verification and chain-quality
> findings below are unaffected (they use no geometry estimator).
>
> 🔄 **STATUS UPDATE 2026-06-06.** Two further things are now stale: **(1) Phase 4
> full extraction is DONE** — `data/activations/R1-1.5B/` holds the full
> per-behaviour activations (N = 5 027 / 5 829 / 10 267 / 16 728, *not* the 51–145
> smoke cohort), so every "[PENDING on full corpus]" / "awaits Phase 4 full" marker
> below is superseded. The full Phase 5/5b/5c geometry **did run** (28-May) but is
> **quarantined** in `results/_STALE_pre_fix_20260605/` pending regeneration on the
> fixed estimators (Gate 0); the steering vectors were **rebuilt fresh** on the
> fixed PCA. **(2)** The smoke-table caveat below that "`adding-knowledge` (51) is
> too small" is **smoke-specific** — full adding-knowledge N is 5 027. For the
> authoritative current status + the confound/remediation plan,
> **`CONFOUNDS_AND_REMEDIATION.md` (2026-06-06) now supersedes both this file and
> `AUDIT.md`.**

Last updated 2026-05-27. Supersedes the 2026-05-24 version; covers the full
chain-corpus generation, smoke-test pass through every new phase, and the
non-reasoning baseline pipeline that is now being wired in.

## Current pipeline state

```
Phase 1 (task_gen)           [DONE]  data/tasks_final.json — 1000 tasks, 100/cat
Phase 2 (chain_gen, R1-1.5B) [DONE]  data/chains_R1-1.5B.json — 1000 chains
   |                                 quality concern: 50.2% hit max_tokens
Phase 2b (chain_gen, base)   [READY] 02b_generate_baseline_chains.py wired; awaits cluster run
Phase 3 (annotation)         [DONE]  data/annotated_R1-1.5B.json
Phase 4 (extraction)         [DONE on full corpus]  data/activations/R1-1.5B/ N=5k–16k/beh (smoke also done)
   |
   +--> Phase 5  (PCA)              [smoke DONE; awaits Phase 4 full]
   +--> Phase 5b (geometric + null) [smoke DONE]
   +--> Phase 5c (cross-layer)      [smoke DONE]
   |
Phase 6 (steering construction)     [smoke DONE]
   |
   +--> Phase 6.5 (predict_saturation) [smoke DONE]
   +--> Phase 6b  (composition)        [smoke DONE]
   |
Phase 7 (steering eval)              [PENDING] awaits Phase 4 full + steering vectors
   +--> Phase 7b (activation patch)   [PENDING]
```

`R1-1.5B-smoke` is the symlink `data/annotated_R1-1.5B-smoke.json ->
annotated_R1-1.5B.json`, so the "smoke" results above are computed against the
real annotated corpus, not a toy subset. Outputs sit under
`results/<phase>/R1-1.5B-smoke/`.

## Modules in `src/` (new since rebuild)

| File | Lines | Purpose | Pre-registered in |
|------|-------|---------|--------------------|
| `curvature.py` | 309 | Three curvature diagnostics (local-vs-global PCA dim ratio, geodesic/Euclidean ratio, tangent-space variation) with k-sweep + bootstrap CIs | Companion §2.5 |
| `intrinsic_dim.py` | 250 | TwoNN, Levina-Bickel, correlation dim — adaptive to small N | Companion §2.5 |
| `nulls.py` | 295 | Three-tier null hierarchy: chain-stratified perm (primary) + cross-chain perm (secondary) + MP isotropic (finite-sample diagnostic) | Companion §2.5 |
| `activation_patching.py` | 278 | Causal-layer localisation primitives: donor-pair construction, residual hooking, behaviour-marker logprob scoring | Companion §4.2 (Paper 2 main) |

## Top-level runners (new since rebuild)

| File | Phase | Purpose | Status |
|------|-------|---------|--------|
| `verify_base_model.py` | pre-Extension 1 | Weight-comparison check of R1-Distill-Qwen-1.5B's base identity | [DONE] base = Qwen-2.5-Math-1.5B (NOT Instruct) |
| `power_analysis_curvature.py` | pre-Phase 5b | Monte Carlo detection-power simulation for curvature diagnostics across (N, kappa) grid | [BLOCKED] see "Open issues" below |
| `02b_generate_baseline_chains.py` | Phase 2b | Q/A-scaffold generation against Qwen-Math-1.5B for differential analysis | [READY] uncommitted; awaits cluster run |
| `04_cleanup_tasks.py` | Phase 1 | Regenerates lateral_thinking with classic-puzzle blocklist; tops up all categories to 100; final dedup | [DONE] → tasks_final.json |
| `check_chain_quality.py` | post-Phase 2 | Markdown + JSON quality report (integrity, distribution, truncation, anomalies) | [DONE] → results/quality_reports/ |
| `05b_geometric_diagnostics.py` | Phase 5b | Curvature + intrinsic dim + null hierarchy on Phase 4 activations | [smoke DONE] |
| `05c_cross_layer_probing.py` | Phase 5c | Layer-wise probe accuracy + non-adjacent subspace angles | [smoke DONE] |
| `06b_steering_composition.py` | Phase 6b | Pairwise composition diagnostics (cos(v_sum,v_proj), off-manifold ratio) | [smoke DONE] |
| `07b_activation_patching.py` | Phase 7b | Donor-pair patching, per-layer effect curves | [READY] awaits Phase 4 full |
| `predict_saturation.py` | Phase 6.5 | Predicted alpha* from Phase 5b curvature + Phase 6 vectors | [smoke DONE] |

## Configuration

`configs/config.yaml` now has both `models.primary` (R1-1.5B) and a new
`models.baseline` section (Qwen-2.5-Math-1.5B, `is_thinking: false`,
`steering_layer: 27` for direct comparability). `MODELS` dicts in
`04_extract_activations.py`, `05_pca_analysis.py`, and `06_build_steering.py`
were extended to recognise `qwen-math-1.5b` / `QwenMath-1.5B` as a first-class
model short-name. The baseline addition is currently uncommitted.

## Empirical findings collected so far

### Base-model verification — `results/base_model_verification/verdict.md`

| Candidate | embed_tokens cos | lm_head cos | aggregate delta |
|-----------|------------------|-------------|------------------|
| **Qwen-2.5-Math-1.5B** (winner) | **0.994** | **0.973** | 0.098 |
| Qwen-2.5-Math-1.5B-Instruct | 0.839 | 0.846 | 0.192 |
| Qwen-2.5-1.5B (no math) | 0.232 | 0.219 | 1.127 |

Margin over runner-up is 0.094. The Instruct variant is conclusively *not* the
base — R1 distillation began from the math-pretraining checkpoint, before the
instruction-tuning shift. Pre-experiment base-model verification by weight
comparison is itself a small methodological contribution.

### Chain-corpus quality — `results/quality_reports/chains_R1-1.5B.md`

1000 chains generated, 100 per category, zero structural defects. But:

- Mean tokens = 5052, **median = 8192** (the max cap)
- **50.2% of chains hit max_tokens; 49.9% lack closing `</think>` (truncated mid-thinking)**
- Worst categories: lateral_thinking (95% at max), spatial_reasoning (71%), pattern_recognition (66%), probabilistic_thinking (59%), mathematical_logic (55%)
- Cleanest: systems_thinking (4%), causal_reasoning (22%), scientific_reasoning (24%)

This is the dominant open question right now: do we (a) raise the cap and
re-generate the truncated half, (b) accept truncation as a confound and stratify
analyses by complete-vs-truncated, or (c) drop the worst categories. See "Open
issues" below.

### Smoke-test results on annotated corpus (layer 27)

**Phase 5b — intrinsic dim + curvature** (`results/geometric/R1-1.5B-smoke/`):

| Behaviour | N | TwoNN dim | Levina-Bickel | Local/global k=10 | Geodesic/Eucl k=10 |
|-----------|---|-----------|---------------|--------------------|----------------------|
| backtracking | 97 | 9.4 | 4.9 | 0.138 | 1.146 |
| uncertainty-estimation | 145 | 16.8 | 8.6 | 0.147 | 1.307 |
| example-testing | 145 | 13.7 | 7.3 | 0.130 | 1.228 |
| adding-knowledge | 51 | 27.9 | 11.4 | 0.236 | 0.857 |

Low local/global ratios (0.13-0.24) and geodesic/Euclidean > 1 for three of the
four behaviours are consistent with the curved-manifold picture, though N for
`adding-knowledge` (51) is too small to lean on **— NB: this N=51 is the SMOKE
cohort; the full run has adding-knowledge N=5 027 (see the 2026-06-06 status note
at the top of this file). The "too small" caveat does NOT apply to the full run.**

**Phase 5c — cross-layer probing** (`results/cross_layer/R1-1.5B-smoke/`):

Probe accuracy 0.83-0.93 across L14/L21/L27. Non-adjacent subspace angles up to
72° (L14 ↔ L27) — substantial cross-layer rotation rather than a static
direction, supporting the "manifold per behaviour" rather than "single vector
per behaviour" framing.

**Phase 6b — steering composition** (`results/composition/R1-1.5B-smoke/`):

cos(v_sum, v_proj) ranges 0.92-0.99 across the six pairs; off-manifold ratio
0.15-0.40. Off-manifold deviations > 0.1 in five of six pairs are first
evidence against the flat-subspace picture.

**Phase 6.5 — saturation predictions** (`results/saturation_predictions/R1-1.5B-smoke/`):

Predicted alpha* per behaviour: backtracking 1.36, uncertainty 1.15,
example-testing 1.11, adding-knowledge 2.31. These are the falsifiable
quantities for Phase 7.

## Bug + issue history (since 2026-05-24)

1. **Phase 1 batch-diversity bug** (commit `cc9bca4`) — `task_gen.py` was not
   passing prior batches as context, so the model recycled classic-puzzle
   variants across batches; lateral_thinking had 61/67 duplicates. Fixed by
   adding `_USER_WITH_CONTEXT` template + per-category context injection, then
   regenerated the whole category from scratch with an explicit blocklist via
   `04_cleanup_tasks.py`. Final corpus: 1000 deduped tasks.

2. **Phase 2 max_new_tokens mismatch** (commit `251e6c1`) — the pilot used 8192
   but `02_generate_chains.py` defaulted to 2048; producing chains incompatible
   with the pilot distribution. Default lifted to 8192. Note that the symptom
   has now shifted (see truncation finding above): 8192 is *still* too low for
   some categories.

3. **Phase 4 label-convention drift** (commit `89d70fd`) — different modules used
   different spellings for behaviour labels (`adding-knowledge` vs `knowledge_augmentation`
   etc). Unified across `annotation.py`, `pca.py`, `steering.py`, `hooks.py`,
   `evaluation.py`. This commit also adds 5b, 5c, 6b, 7b, 6.5 + `src/nulls.py`,
   `src/activation_patching.py`.

4. **Smoke-test bug sweep** (commit `dbb93bc`) — 5 follow-on bugs caught by
   running the whole pipeline end-to-end on smoke data; touches 5b, 6b, 7,
   6.5, `src/annotation.py`, `src/steered_inference.py`.

## Open issues / next actions

1. **Chain truncation (high priority)**: half the R1-1.5B corpus is cut off
   mid-thinking. Decision needed before scaling Phase 4. Options on the table:
   raise the cap (cost: more cluster time), stratify (cost: power), drop worst
   categories (cost: generality). The annotation pipeline already ran on the
   truncated chains, so re-running 02 also forces re-running 03.

2. **Power analysis still failing** (`results/power_analysis/`): the second
   re-run after the sklearn fix wrote an all-NaN `power_table.csv` (every cell
   for every (N, kappa, diagnostic) combination). `summary.md` reports "> max
   tested" everywhere, but that conclusion is downstream of the NaN values, not
   a real null result. Needs another look — likely a remaining import/silent-
   failure path in `power_analysis_curvature.py` despite the earlier patch to
   stop swallowing exceptions.

3. **Phase 4 on full corpus**: pending the truncation decision. Cluster code
   already updated to support both `R1-1.5B` and `QwenMath-1.5B` short names.

4. **Phase 2b baseline run**: `02b_generate_baseline_chains.py` is ready; the
   config and downstream MODELS dicts know about the baseline. Estimated
   3-6 hr on shared cluster GPU for 1000 prompts. Currently uncommitted.

5. **Phase 7 steering eval + Phase 7b activation patching**: blocked on Phase 4
   full corpus + steering-vector build.
