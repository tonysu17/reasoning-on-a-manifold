# Methodology — The Geometry of Machine Reasoning (R1-1.5B)

> **Living spec. Canonical source for HOW we do things.** Every session: READ this before
> touching geometry/steering; UPDATE it the moment a method changes. Pairs with
> [`RESULTS_LEDGER.md`](RESULTS_LEDGER.md) (what we found) and
> [`CONFOUNDS_AND_REMEDIATION.md`](CONFOUNDS_AND_REMEDIATION.md) (what is wrong / owed).
> **Last updated: 2026-07-06.**

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

## 8. Featurizer interventions / causal basis (E10)  `[E10 COMPLETE: E10.0 PASS; E10.1 RUN (P1 confirmed, P2b run); E10.2 RUN]`  `19_sae_gate.py`, `20_das_backtracking.py`
The causal-abstraction framing (Geiger et al.): every intervention is performed in a coordinate
system; ours has so far been a **fixed, correlationally chosen linear featurizer** (diff-of-means
direction, optionally restricted to the behaviour's top-k PCA subspace; Huang Eq. 3 at α=1 = zero-
ablation of ONE feature coordinate in that frame). The upgrade path is a frame selected **causally**
(interchange interventions / DAS). Thesis: formalism in `sec:bg-mechinterp` (eq:bg-featurizer),
programme in `sec:steering-featurizer-programme`. Key algebra: swap = `h_b + U Uᵀ(h_s − h_b)` for an
orthonormal frame U; a **swap has no α** (injected values are on-distribution by construction) —
directly relevant to CF-19 collapse.
- **E10.0 dictionary gate** `[EXECUTED 2026-07-05 = PASS, local MPS, 60 chains]` — `19_sae_gate.py` →
  `results/sae_gate/R1-1.5B/` (`REPORT.md`, `e10_0_gate.json`). Public-SAE audit (web-verified 2026-07-05): EleutherAI 65k = MLP-out
  only (all 28 layers); Resa (arXiv 2506.09967) = MLP-out L12, reasoning-triggered, HF `Resa-Yi`;
  DGurgurov = the ONLY residual candidate, metadata self-contradictory (card resid_pre 19 / folder
  blocks.1 / cfg resid_post 19) → gate resolved the site **empirically** (FVU probe over 3 sites ×
  {raw, norm-folded}: **blocks.19.resid_post, raw** = the winner; cfg was right, card+folder wrong).
  Metrics: FVU, L0, CE-splice (clean/recon/zero → CE-recovered). Sealed thresholds: PASS FVU≤0.15 ∧
  CE-rec≥0.85; AMBER FVU≤0.40 ∧ CE-rec≥0.60; else FAIL ⇒ self-train residual SAEs at causal-candidate
  layers on the CoT distribution (Resa recipe moved to the residual hookpoint — kills site+distribution
  mismatch at once). **Verdict: PASS** — DGurgurov @ resid_post19 scored **FVU 0.061, CE-recovered 0.958, L0 71**
  on OUR CoT distribution despite LMSYS-chat training (transfers cleanly); L19-MLP reference FVU 0.478 /
  CE-rec 0.669 confirms the wrong-site penalty. ⇒ a usable external residual dictionary EXISTS but only at L19
  (near ex-test mid-peak); self-trained residual SAEs still owed at the other causal layers (bt17/unc15/L27).
  MLP-site SAEs are NOT admissible as the comparator (edits one addend of the stream, not the accumulated
  state → site confounded with featurizer).
- **E10.1 DAS-1D on backtracking** `[INTERCHANGE STAGE RUN 2026-07-06 → P1 supported @ L17, grounded; L11/L27 ungrounded (see RESULTS_LEDGER §B4 + results/das/R1-1.5B/main/ANALYSIS_2026-07-06.md); P2 generation stage + CIs owed]` — learn 1-D orthogonal frame at bt attribution layer
  (17) + de-confounded mid (11), L27 as read-out-proximity control; onset-anchored donor/recipient
  pairs (E9.0 machinery); objective = teacher-forced CE toward donor's real post-onset continuation
  (CF-10a-style output read-out; annotator stays eval-only → loosens circularity).
  ⚠️ **Minival-1 (2026-07-06) fired the illusion control** — single-onset-token labels admit a
  generic "Wait-booster" solution (shuffled-pair dlp +5.87 ≈ learned +6.56) ⇒ objective upgraded to
  **windowed W=12 teacher-forced continuation, symmetric induce+remove** (prereg Amendment 1);
  primary endpoint = sym Δlogprob + **pair_specificity_gap** (learned − shuffled). W=1 kept only as
  ablation. Eval battery unchanged (E8 Δ_floor machinery for the generation stage + collapse
  endpoints). Controls: untrained random rotation (floor), **shuffled-pair training**
  (Makelov-illusion control), misaligned-position swaps (E9.0b-style positional control). Sealed
  predictions: (P1) causal frame ≥ diff-of-means at matched energy; (P2) swap collapses less than
  projective ablation at matched on-target effect (a collapse-account prediction too).
- **E10.2 causal width** `[PROPOSED]` — boundless-DAS-style learned k*; k*≈corr-dim (6–8) certifies
  low-dim causally, k*=1 explains the manifold null at its root. Bonus: add-knowledge **removal** test
  (swap is symmetric; amplification null ≠ removal null).
- **E10.3 cross-behaviour generalization** `[EXECUTED 2026-07-20 — P-cross alternative CONFIRMED]` —
  identical E10.1+E10.2 recipe on **uncertainty-estimation** {15,16,27} and **example-testing**
  {15,19,27}, hyperparams UNCHANGED (M5); grounding gate codified in `e10_pick_grounded_layer.py`.
  **Verdict:** transfer huge everywhere (+0.62…+0.72 vs dm +0.01…+0.03) but UNGROUNDED at every mid
  layer (unc L15/L16 AUC 0.412/0.626; ex 0.579/0.599; ex width skipped by sealed rule); learned
  frames align ACROSS behaviours (unc↔ex mid |cos| 0.946; bt↔both 0.64–0.77; unc_L27↔bt_L27 0.97)
  ⇒ DAS found ONE shared discourse-shift/onset axis per depth-family, not per-behaviour causal
  structure; only backtracking's frame is also state-reading. Grounded causal frame = n=1
  (backtracking-specific); third independent "transfer ≠ feature" demonstration. Full adjudication:
  `E10_DAS_PREREG.md` **ADJUDICATION 5**.

## 9. Predictive-geometry value track & precursors (PG rungs 3–7)  `[PROPOSED — prereg sealed 2026-07-06]`  [`PREDICTIVE_GEOMETRY.md`](PREDICTIVE_GEOMETRY.md) §11
Adopted reframe of the predictive-geometry side project (branch `predictive-geometry-of-reasoning`)
after the LRS deep-read (2606.00726) + the pilot verdict (residual detector weak 0.54–0.59,
**shuffle-invariant = occupancy not dynamics**, dead at causal L17 / best at read-out L27). The
programme pivots from correctness detection to **mechanism around collapse**, composing E8/E9/E10
assets: value head + ∇V-vs-DAS-vs-diff-of-means triangulation (R3) → precursor lead-time race at
loop onsets (R4) → **dose-response validation of precursors on the E9.1b arms** (R5 — intervention-
validated early warning; all neighbour detectors are observational) → precursor-gated ablation
with count+energy-matched random-gate floor (R6, the WHEN-vs-WHERE factorization) → action-
conditioned forward model (R7, stretch). Full sealed predictions/kills per rung in PG §11.3.
**Two rules bind ALL trajectory claims repo-wide** (not just this programme):
- **M3 order-sensitivity null:** any trajectory claim reports its step-shuffle variant;
  shuffle-invariant results are worded as *state-occupancy geometry* — "trajectory/dynamical" is
  reserved for order-sensitive results.
- **M5 LRS anti-inheritance:** sealed hyperparams + val checkpoint (no test-tuned knobs); gated
  interventions get **count-matched AND energy-matched** random-gate floors (extends §4's floor
  discipline); matched-pair recovery tables as standard endpoints.
- **M7 prefix-contamination control** *(added 2026-08-02, PG §11.1 M7)*: latent value/precursor
  claims include a corrupted-prefix arm (PAIR-style, 2605.17877); signals that survive prefix
  corruption unchanged are worded as *prefix-coherence*, not progress.
Interlocks: collapse endpoint + onsets from §7; frames from §3 (behaviour-PCA), §8 (SAE@L19
dense-arm-only, DAS-1D@L17); R4+R5 share ONE extraction pod session (~$1–3); total new spend <$10.

**Proposed trajectory-dynamics apparatus (A1–A5, 2026-07-07)** `[PROPOSED — full P/K/cost at PG §12]`
— extends the sealed ladder along the "trajectory of reasoning in latent space" axis; each anchored to
JEPA or a named mech-interp method, each cheap, each with a kill criterion. Together they upgrade
Movement 1 from "reasoning occupies static low-dimensional subspaces" to "**reasoning is a dynamical
process with identifiable geometric structure**":
- **A1 belief-state trajectory** — logit / tuned lens (Belrose et al.; Shai et al. belief-state
  geometry) read of the model's own answer-belief per step = a label-free value track, the baseline
  R3's trained V must beat, a free R4 precursor, the M4 5th frame; order-sensitive by construction
  (the M3 positive the pilot lacked).
- **A2 switching LDS** — rSLDS (Linderman et al.; `ssm`) → discrete reasoning modes + switch points;
  unsupervised mode↔annotated-behaviour alignment = an annotator-circularity de-confound; loop-mode
  entry = a lead-time precursor.
- **A3 loops-as-attractors** — the JEPA representational-collapse ↔ reasoning-collapse bridge;
  effective-rank / SIGReg sliding-window precursors + a perturbation-return (basin-of-attraction)
  causal test (no detector paper — Sun/PHi/LRS — runs it).
- **A4 circuit attribution** — induction-head (Olsson et al.) attribution + ablation of loop onset =
  the mechanism *under* the geometric precursor; the mechanism-not-benchmark wedge vs LRS.
- **A5 forward-map grounding** — activation-patch the JEPA-predicted next-state into the model
  (E10.1-style interchange) to test whether the learned direction IS the model's own update.

## 10. Entropy-ladder instruments (creativity–entropy programme, R0)  `[R0 EXECUTED 2026-07-12 — gate PASS, see RESULTS_LEDGER §B5]`  `29_r0_entropy_ladder.py`, `r0_runner.sh`, [`R0_ENTROPY_LADDER_PREREG.md`](R0_ENTROPY_LADDER_PREREG.md)
Programme doc: `../creativity_entropy_extension.md` (parent dir) — creativity as entropy
*management*; four-level entropy ladder **E-1 decoding / E-2 state occupancy / E-3 solution-space /
E-4 semantic** with cross-level relations measured, never assumed. R0 = bookkeeping on existing
corpora (E9.0 shards + E9.1 T06 vanilla arm), local MPS, $0. E-4 deferred by declaration (R4's job).
- **E-1 instrument:** teacher-forced next-token predictive entropy (nats), chunked LM head (never
  materialises the (T, vocab) logits tensor), tokenization + window grid identical to §7's
  loop-geometry extract (window 128 / stride 64); **center-grid equality with the E9.0 state shard
  asserted per chain** — mismatches excluded and counted, not kept. Re-scoring caveat: exact for
  greedy corpora; for T>0 samples it measures local uncertainty along the sampled path.
- **E-2:** §7's windowed PR (Gram-trick) + token uniformity, from the existing shards.
- **E-3:** 1 − mean pairwise 4-gram Jaccard across same-task samples (`e9_1_analysis` definition).
- **Primary-region discipline (R0.a):** clean chains = all generated windows; loop chains =
  pre-onset windows only — else the loop tail manufactures a spurious pooled correlation. The
  E9.0b **matched-relative-position clean control is mandatory** for all onset-anchored tests.
- Sealed predictions P-R0.1–P-R0.5 (dissociation gate at |ρ|<0.9; jam = in-loop E-1 depression;
  state-first vs thermostat-failure pre-onset race; mid-range E-1↔E-3 coupling) in the prereg doc.
- **Ops lessons (2026-07-12, in docstrings):** MPS caching allocator accumulates freed blocks
  across variable-length chains — per-chain `torch.mps.empty_cache()`+gc AND a restart-loop runner
  (`r0_runner.sh`, `--limit` chains/process) are both needed; ~1% of 8192-token bf16 forwards go
  all-NaN on MPS ⇒ `--fp32` retry path (extract stages refuse to save all-NaN shards).

## 11. R1 compression-contrast instruments (creativity–entropy rung 1 + RL.a)  `[BUILT — prereg sealed 2026-07-12, awaiting pod]`  `30_r1_compression.py`, `runpod_r1.sh`, [`R1_COMPRESSION_PREREG.md`](R1_COMPRESSION_PREREG.md)
Post-training ladder at 1.5B, all public: qwenmath (base) → r1 (SFT-distill) → **deepscaler**
(GRPO-RLVR on top of r1, verified agentica-org/DeepScaleR-1.5B-Preview) + star1 (safety-SFT
control). **Primary contrast r1↔deepscaler** (tokenizer gate: byte-identical ids; star1 also
matched-ids; **qwenmath matched-TEXT only** — DeepSeek modified the tokenizer, gate codified in
`--stage gate` → `results/r1_compression/gate_tokenizer.json`).
- **R1-rep:** teacher-forced E-1 + E-2@L17 on the R0 200-chain sample, same full_texts every arm;
  r1 reference REUSED from R0/E9.0 shards; matched-ids arms assert grid equality ⇒ paired
  per-chain deltas (Wilcoxon).
- **R1-beh:** own-generation, 50 E9.1 eval tasks × (greedy + 3×T0.6) per arm, E9.1-style
  batch-seeding; annotation-free proxies sealed in the runner (BT_CUE_RE per 1k, boxed,
  repetition_rate>0.8).
- **R1-score:** each arm re-scores its own generations (on-policy E-1/PR; CF-M cross-check).
- Sealed P-R1.1–P-R1.6 + kill criteria in the prereg; confounds CF-M/N/L/O registered there.

---
### Change log
- 2026-07-11: §10 added — entropy-ladder instruments (creativity–entropy R0); prereg sealed before
  computation (`R0_ENTROPY_LADDER_PREREG.md`); programme doc `creativity_entropy_extension.md`
  drafted in parent dir (proposal only — thesis untouched).
- 2026-07-07: §9 extended — PROPOSED trajectory-dynamics apparatus A1–A5 (belief-lens value track ·
  rSLDS modes/switches · loops-as-attractors JEPA-collapse bridge · induction-head circuit attribution ·
  forward-map patching); full per-item design/predictions/kills/cost at `PREDICTIVE_GEOMETRY.md` §12.
  Motivated by this session's LRS read (value/outcome-map vs forward/dynamics-map dissociation) + the
  latent-space-trajectory focus; connects the programme to JEPA + logit-lens / rSLDS / induction-head /
  activation-patching mech-interp.
- 2026-07-06 (later): §9 added — predictive-geometry value-track reframe ADOPTED (Tony) + rungs 3–7 sealed in `PREDICTIVE_GEOMETRY.md` §11 (M-register M1–M6; H5–H8; per-rung predictions + kills; collapse-primary outcome, correctness secondary). Repo-wide binding rules M3 (order-sensitivity null / occupancy-vs-dynamics wording) + M5 (LRS anti-inheritance: sealed knobs, count+energy-matched gate floors, matched-pair recovery tables). PG doc also reconciled: §8 pilot verdict replaces stale pre-label read, old Rung-3 apex superseded (CEM cut; causal loop → R6), LRS added to §5 wedge, §10 world-model framing added.
- 2026-07-06: §8 reconciled to executed state — E10.0 gate verdict = **PASS** (was tagged RUNNING): DGurgurov resid SAE @ blocks.19.resid_post scores FVU 0.061 / CE-recovered 0.958 on our CoT ⇒ usable external residual dictionary at L19 only, self-trained SAEs owed at bt17/unc15/L27. Thesis `sec:steering-featurizer-programme` rung one updated from failure-branch-only to the partial-pass result (steering.tex; ucl_msc build compiles clean, 0 undefined citations in PDF). E10.1 DAS harness (`20_das_backtracking.py`) noted as built + smoke-only (conclusive run still UNRUN).
- 2026-07-05 (later): §8 added — E10 featurizer/causal-basis programme (audit executed, E10.0 gate launched locally, E10.1–2 pre-registered; thesis sections written same day).
- 2026-07-05: §7 added (E9 collapse/loop-geometry instruments; E9.0 executed, E9.1 launched — see `COLLAPSE_AND_ENTROPY.md` + `RESULTS_LEDGER.md` §B3).
- 2026-06-20: created. Captured verified construction (§3) + application (§4) from `src/steering.py` / `src/steered_inference.py`; metric keep/drop list (§2); layer-selection + attribution-patching proposal (§5); reconciled peak_layers.
