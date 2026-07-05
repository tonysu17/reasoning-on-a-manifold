# Methodology — The Geometry of Machine Reasoning (R1-1.5B)

> **Living spec. Canonical source for HOW we do things.** Every session: READ this before
> touching geometry/steering; UPDATE it the moment a method changes. Pairs with
> [`RESULTS_LEDGER.md`](RESULTS_LEDGER.md) (what we found) and
> [`CONFOUNDS_AND_REMEDIATION.md`](CONFOUNDS_AND_REMEDIATION.md) (what is wrong / owed).
> **Last updated: 2026-06-20.**

Tags used below: **[CURRENT]** = implemented and run as described · **[CODED, UNRUN]** =
implemented but not yet executed · **[PROPOSED]** = not yet built (design only).

---

## 0. Model & corpus
- **Model R1-1.5B** = `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (base verified = Qwen2.5-Math-1.5B). 28 layers, hidden dim **1536**.
- ~1000 tasks × 10 reasoning categories → chain-of-thought (greedy, `max_new_tokens=8192`).
- **Annotation** (sentence-level behaviour spans, Venhoff arXiv:2506.18167 Appendix-A prompt) via the AWS Bedrock / Claude proxy: **Sonnet 4.5 (primary)**, **Qwen3-235B**, **Nova-Pro** (3-way robustness). **Not gpt-4o** (that is Phase-1 task gen only).
- **4 target behaviours:** backtracking, uncertainty-estimation, example-testing, adding-knowledge.

## 1. Activation extraction  `[CURRENT]`  `04_extract_activations.py`, `src/activation_extraction.py`
- Per behaviour span: `n_preceding_tokens=1`, `n_execution_tokens=10`, **pooling = mean** over the first ~10 tokens of the span (pooling sweep also stores `last`).
- **Occurrence-aware** span location (CF-13 fix) + exact-duplicate row dedup; `row_index.json` provenance sidecar (CF-14). Post-fix duplicate fraction ~1% (was 35–56%).
- Output: `data/activations/R1-1.5B/{behaviour}_layer{L}.npy`, all 28 layers.
- ✅ **Pooling RESOLVED (2026-06-21):** the `[onset−1 : +10]` mean-pool **is exactly the Venhoff
  (arXiv:2506.18167) recipe** — verified in their `train_vectors.py`
  (`layer_outputs[:, start-1:min(end-1, start+10)].mean(dim=1)`) — the field standard for
  span-localized reasoning behaviours, NOT a flaw. CF-6 downgraded from "design-open" to "resolved
  (keep mean)". Huang (arXiv:2505.22411, our manifold source) uses *final-token* diff-of-means +
  PCA k=10 (>70% var) @ L27. **Optional robustness** (gated on a small GPU re-extraction): a
  **position sweep** — mean / onset-token / pre-onset−k / last — since Arditi (2406.11717) treats
  read-position as a validated hyperparameter and 2507.12638 finds the backtracking signal sits
  *pre*-onset (negative offset).

## 2. Geometry metrics — KEEP vs DROP  `[CURRENT]`
**KEEP (citable):**
- **Correlation dimension** — the stable intrinsic-dim estimator. Always report with the chain-stratified control.
- **Participation ratio (PR)** — scale-robust effective dimensionality; used for layer concentration.
- **Chain-stratified variance-ratio null** — the behaviour-specificity test (see §6).
- **local↔global dimension ratio** (curvature) — KEEP only as the diagnostic that produced the **negative** (it is well-powered; see RESULTS_LEDGER §A).

**DROP / do not cite:**
- **TwoNN** intrinsic dim — duplicate/subsample-unstable; disagreed with Levina–Bickel on the sign of the behaviour-vs-chain effect.
- **PCA d_eff at ≥80% variance** — saturated at the top-100-component cap (floors, not estimates). Use d_eff_50 / d_eff_70 or PR instead.
- **tangent-space variation** curvature (~67–70° across all behaviours/layers — uninformative).
- **geodesic/Euclidean** ratio as a *positive* curvature claim — needs N≥2000 (underpowered here) and curvature is a negative regardless.

## 3. Steering vector construction (Phase 6)  `[CURRENT]`  `src/steering.py`, `06_build_steering.py`
For target behaviour *b* at layer *L*, with `ON` = *b*'s pooled activations and `OFF` = the **concatenation of the other three behaviours'** activations (the contrast is behaviour-vs-other-behaviours, not vs a neutral corpus):

- **Single direction (Venhoff, diff-of-means)** — `single_direction_vector()`:
  ```
  r = mean(ON) − mean(OFF);   r ← r / ‖r‖      (unit norm)
  ```
- **Manifold-projected (our method)** — `manifold_projected_vector(k)`:
  ```
  V = top-k PCA components of ON      (PCA fit on ON only, svd_solver="full")
  r_proj = (Vᵀ V) r = Σ_{i≤k} (r·v_i) v_i      # orthogonal projection of r onto b's own top-k subspace
  r_proj ← r_proj / ‖r_proj‖                    (unit norm)
  ```
  i.e. **the same diff-of-means direction, restricted to the behaviour's own principal subspace**, then renormalised.
- **auto_k** = smallest k whose top-k PCs explain ≥ **70%** of ON variance (cap 100). Current auto_k: backtracking 58, uncertainty 71, example-testing 60, adding-knowledge 83 (layer-27 build).
- **k values built:** {1, 3, 5, 10, auto}. Files: `{beh}_single.npy`, `{beh}_manifold_k{...}.npy`.
- **Hold-out:** the 50 category-stratified eval tasks (`src.task_gen.stratified_eval_split`) are **excluded** from ON/OFF before fitting (row-provenance hold-out → Phase 7 is out-of-sample on the task dimension; NOT on the layer dimension).
- Two builds: canonical **all-at-L27** (`results/steering_vectors/R1-1.5B/`) and per-behaviour **`-peak`** (`build_phase6.py`, layers from `config.analysis.peak_layers`).

## 4. Single-vs-manifold comparison protocol (Phase 7)  `src/steered_inference.py`, `07_evaluate_steering.py`, `src/evaluation.py`
**Application of a vector during generation** `[CODED, UNRUN]` (Huang Eq. 3, projective — applied at every position):
```
h' = h − α·(rᵀh)·r     (subtract = suppress the behaviour)   ← current mode
h' = h + α·(rᵀh)·r     (add = amplify)
```
`r` is unit-norm, so **α is the full scale knob**. Because the term is `α·(rᵀh)·r` (projection of h onto r), **α=1 removes the entire r-component** (ablation); α>1 over-removes. This is *not* a constant additive shift.

**Arms** `[CODED, UNRUN]`: `vanilla` (one shared unsteered generation per task), `single_direction`, `manifold_projected` (**auto_k only, currently**), `random_direction` (norm-matched control — the floor that licenses causal language).
**Dose:** α ∈ {0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0} (+ 0 = vanilla). Mode = subtract.

**Metrics:**
- **On-target (primary)** `[CURRENT]`: `behaviour_fraction` = fraction of re-annotated sentences labelled *b*; effect = Δ vs vanilla.
- **Off-target damage** `[CURRENT]`: `repetition_rate` (4-gram), `degenerate_rate` (<32 tokens), `mean_n_tokens`. `n_missing`/`n_empty` re-annotations are counted, never scored 0.
- **Off-target leakage** `[PROPOSED]`: other 3 behaviours' fractions from the SAME re-annotation (cheap — labels already cover all behaviours).
- **Task accuracy preservation** `[PROPOSED — docstring claims it, `aggregate_results` does NOT compute it]`: GSM8k (or per-category) correctness vs vanilla.

**Headline comparison** `[PROPOSED — pin before running]`: the claim is manifold steers *more effectively / less destructively* than single. Operationalise as one pre-registered primary, e.g.:
1. **Matched-effect damage:** at equal on-target suppression, manifold has lower off-target damage (repetition/degenerate/accuracy-drop); compare on the suppression↔damage Pareto frontier.
2. **Efficiency:** max suppression reached before `degenerate_rate` / accuracy-drop crosses a threshold.
3. **Saturation:** empirical α* (suppression plateau) vs the **pre-registered** predicted α* (`results/saturation_predictions/`), cross-behaviour Pearson r > 0.5.
Both single and manifold must beat `random_direction`; manifold-vs-single tested **paired over the 50 held-out tasks** with bootstrap CIs.

**Open methodology gaps to close before/at Phase 7:**
- Sweep manifold **k** ∈ {1,3,5,10,auto}, not just auto_k (the granularity question; composition shows diff-of-means is far from PC1 for some behaviours, so intermediate k may matter most).
- Implement task-accuracy + cross-behaviour-leakage metrics.
- `random_direction` is **norm-matched, not energy-matched** (|rᵀh| smaller for a random r) → describe as a generic-perturbation floor, not an energy-matched twin.

## 5. Steering-layer selection
- `[CURRENT]` Canonical build steers all behaviours at **L27** (Huang's published layer for Qwen-1.5B). The `-peak` build uses per-behaviour `config.analysis.peak_layers` = PR-trough (reconciled 2026-06-20 to 16/16/16/12).
- ⚠️ PR-trough is a **descriptive** (concentration) criterion, not a **causal** one. Behaviour-specificity (variance-ratio null) and steering both want a layer where intervening actually changes the behaviour.
- `[CODED, UNRUN]` Choose the steering layer by a **causal** criterion via **attribution patching**
  — implemented 2026-06-21 in `src/attribution_patching.py` + runner `07c_attribution_patching.py`.
  Gradient×(corrupt−clean) per layer; one forward+backward gets all 28 layers. The **CF-10 fix** is
  built in: the behaviour metric is the **projection of the residual onto the behaviour's steering
  geometry** (diff-of-means direction / top-k PCA subspace), not a lexical "wait/actually" proxy,
  with onset-anchored positional alignment. Writes `results/patching/R1-1.5B/{attribution_curves,
  pilot_effect_curves}.json` → fills the slot triangulation expects (previously MISSING → PR-only).
  **This is exactly Venhoff's published layer-selection method.** Run (GPU, gated):
  `python 07c_attribution_patching.py --behaviours all --n-pairs 20 --brute-check 3`. ⚠️ first-order
  approximation — `--brute-check K` runs the exact patch on the top-K layers to confirm.
- `[RAN 2026-06-21 — CONFOUNDED, do not use for layer pick]` The full sweep (4×20 pairs) returned a
  **monotonic early→late ramp peaking at L26–27 for ALL behaviours** — a **read-out-proximity
  artifact** (the metric reads at L27 = the last layer, so patching near L27 inflates), not a
  behaviour-specific causal layer. A real causal layer shows an interior peak; Venhoff's comparable
  method finds mid-layers (15–18). **Decision: fold layer selection into Phase 7** — build vectors at
  both mid (16, PR-trough) and late (27, Huang) and let steering-effectiveness choose. To salvage
  attribution patching, the read-out must be **downstream/output-based** (not a fixed late layer) +
  ignore early embedding-correlated layers (Venhoff) — a re-design, deferred.
- `[CODED 2026-06-21, UNRUN]` That re-design is implemented as **`07d_layer_steering_sweep.py`** +
  `src/layer_sweep.py` (replaces the confounded 07c path). It measures the **actual non-linear
  per-layer steering effect by FORWARD-PASS intervention**, not a gradient: for each layer ℓ, add
  −α·v_ℓ (the per-layer diff-of-means direction, projective Huang Eq.3 form) over the onset prefix
  (≤ t*−1), then read **Score_b(ℓ) = M_baseline − M_steered** where M is the **OUTPUT** teacher-forced
  log-prob of the donor's REAL behaviour-onset token (CF-10a: a downstream, output-based read-out
  using real tokens, not a lexical-marker list). Because the read-out is the fixed OUTPUT (common to
  every ℓ), there is no read-out-proximity term and the curve **can show a genuine interior peak**
  (forward intervention captures amplification — early perturbations grow through the network — which
  the first-order gradient does not). Default sweep skips L0–L1 (Venhoff). Forward-only,
  ~(#layers)×(#donors)×(#behaviours) passes. Stub-tested (`tests/test_layer_sweep.py`, 21 green):
  the critical test plants an interior-peak stub and verifies `argmax` recovers the MIDDLE layer (NOT
  a monotone ramp to the last), and that the old fixed-late-read-out logic ramps to the last layer on
  the same stub. Caveat: the log-prob read-out is still token-anchored. Writes
  `results/patching/<model>/{steering_effect_curves.json, steering_effect_summary.md}` (+ optional
  `pilot_effect_curves.json` with `--write-pilot` to feed triangulation in place of 07c).
  Run: `python 07d_layer_steering_sweep.py --behaviours all --n-donors 20`.

## 6. Nulls & controls  `[CURRENT]`
- **Chain-stratified variance-ratio null:** the honest unit is the **chain**, not the sentence (within-chain sentences are autocorrelated → effective N ≈ #chains, not #sentences). Permutation null shuffles behaviour labels **within chains** (preserving chain identity, composition, and N), recomputes the statistic (top-k PCA variance ratio), and asks whether the real value exceeds the null. Phipson–Smyth-smoothed p; Holm–Bonferroni across behaviour×layer cells; B=2500 on report layers (11/14/17/20/27).
- **One-sentence-per-chain** + chain-block bootstrap at matched N — isolates the chain effect (this is what turned curvature into a negative).
- **norm-matched random_direction** steering arm; **shared vanilla** baseline; **50-task category-stratified hold-out** (`eval_task_ids.json`).

## 7. Collapse / loop-geometry instruments (E9)  `[CURRENT]`  `src/loop_geometry.py`, `18_loop_geometry.py`, `e9_collapse_table.py`, `e9_1_analysis.py`
Full design + hypothesis register (H-A…H-D) + pre-registered predictions: [`COLLAPSE_AND_ENTROPY.md`](COLLAPSE_AND_ENTROPY.md); confound entry CF-19.
- **Collapse endpoint:** per-chain 4-gram `repetition_rate` > 0.8 (`src/evaluation.py`; the distribution is bimodal loop-to-cap so the threshold is uncritical). `degenerate_rate` (<32 tokens) is BLIND to this mode — collapse makes *long* chains. Regenerable per cell via `e9_collapse_table.py`.
- **Loop-tail detector:** word-level periodicity with mismatch tolerance (`detect_loop_tail`), end-anchored (the loop must reach the cap; anchor deliberately looser at 2×tol), onset REFINED forward (the overall-rate criterion bleeds ~tol·L words back into clean prose — synthetic tests pin this), period canonicalised to the smallest divisor with a comparable tail. Gives per-chain onset for precedence tests; doubles as a corpus-hygiene tool (CF-8 kinship).
- **Loop probe (E9.0a):** chain-grouped (GroupKFold, CF-2) logistic in/out-of-loop on token-level residual states; guard band before onset excluded from the out class. Two loop axes are reported and are NOT interchangeable: the **probe direction** (discriminative; the pre-registered contamination-gate instrument, |cos| ≳ 0.3 vs a steering vector ⇒ rebuild) and the **class-meandiff direction** (where loop states sit; the H-D sign-structure read). Analytic cosine null: random unit vectors in R^d have cos with std 1/√d ≈ 0.0255 at d=1536.
- **Precedence (E9.0b):** windowed participation ratio (Gram-trick exact PR — `participation_ratio` docstring) + token uniformity (mean pairwise cosine), stride-64 windows over the generated region; pre-onset window vs early-chain baseline, ALWAYS against the **matched-relative-position contrast in clean chains** — the control that killed the naive PR-precedence claim (clean chains decline MORE; only uniformity@L17 survives it).
- **Batched sampling (E9.1):** `SteeredModel.generate_batch` supports T>0 with **batch-level seeding** (batches grouped by sample index j, seed = base+j); the reproducibility contract is per (batch composition, seed), not per sequence — draws pool per cell downstream. Greedy path unchanged (validated batched == unbatched).

---
### Change log
- 2026-07-05: §7 added (E9 collapse/loop-geometry instruments; E9.0 executed, E9.1 launched — see `COLLAPSE_AND_ENTROPY.md` + `RESULTS_LEDGER.md` §B3).
- 2026-06-20: created. Captured verified construction (§3) + application (§4) from `src/steering.py` / `src/steered_inference.py`; metric keep/drop list (§2); layer-selection + attribution-patching proposal (§5); reconciled peak_layers.
