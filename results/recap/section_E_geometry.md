## E. Geometry / Gate-0 — Subspace, Curvature, Specificity

This section documents the **Gate-0 trilogy** — the three load-bearing geometric claims of the thesis ("each reasoning behaviour occupies a *curved, low-dimensional, behaviour-specific* manifold in the residual stream") and the machinery built to test each one against the chain confound. The stage spans the `05*` scripts (`05_pca_analysis.py`, `05b_geometric_diagnostics.py`, `05c_cross_layer_probing.py`, `05d_subtype_clustering.py`), the `src/` library (`pca.py`, `intrinsic_dim.py`, `curvature.py`, `nulls.py`), and the three Tier-1 robustness drivers (`tier1_effective_n.py`, `tier1_geometry_nulls.py`, `power_analysis_curvature.py`).

**Bottom line up front (the honest verdict as of the 2026-06-18 Gate-0 regeneration, per `CONFOUNDS_AND_REMEDIATION.md` status banner):**

| Claim | Verdict | Status |
|---|---|---|
| (1) Low-dimensional behaviour subspace | ✅ **SURVIVES** the chain control | CITABLE |
| (2) Curvature ("curved" manifold) | ❌ **CLEAN, WELL-POWERED NEGATIVE** (chain artefact) | CITABLE as a negative |
| (3) Behaviour-specificity | 🟧 **MIXED (2/4)** — backtracking + uncertainty specific at all layers; example-testing only at L27; **adding-knowledge nowhere (p=1.0)** | CITABLE with caveats |

The word "curved" must be struck from the headline. What survives is "low-dimensional, (partly) behaviour-specific subspace". The keystone confound throughout is **CF-2** — sentences within one reasoning chain are autocorrelated, so the raw N of 5k–16k sentences is illusory and the honest denominator is the ~600–900 *chains*.

---

### E.0 What is the input, and why these four behaviours

Every `05*` script consumes `data/activations/<model>/<behaviour>_layer<N>.npy` — a `(N_sentences, hidden_dim)` float matrix, one per (behaviour, layer), produced by Phase 4. For R1-Distill-1.5B `hidden_dim = 1536`, there are 28 layers, and the four `TARGET_BEHAVIOURS` analysed are `backtracking`, `uncertainty-estimation`, `example-testing`, `adding-knowledge`. The **focus / steering layer is hardcoded to 27** for R1-1.5B (`src/config.STEERING_LAYERS = {'R1-1.5B': 27, ...}`), inherited from "Huang's recommended steering layers". The 1536-D ambient dimension and the per-behaviour N expectations both flow directly into the power analysis (`--d 1536`), so the geometry stage and the steering layer are wired to the same single source of truth (`configs/config.yaml` via `src.config`).

The row-provenance discipline is critical and recently hardened. Chain IDs per activation row are resolved sidecar-first via `src.row_provenance.chain_ids_for(...)`, and `require_aligned(...)` **hard-fails** on misalignment. This is not pedantry: the old proxy fallback (one pseudo-chain per behaviour) silently turned the within-chain permutation null into a no-op returning p≈1.0. The current code refuses to run a vacuous null (see E.4).

---

### E.1 Claim 1 — Low-dimensional subspace: PCA, d_eff, participation ratio

`05_pca_analysis.py` is described in the source as "the core empirical contribution". For each (behaviour, layer) it fits an exact PCA and computes two effective-dimensionality summaries: `d_eff(p)` (smallest k explaining ≥ p of variance) and the **participation ratio** PR. The verbatim core, from `src/pca.py::analyse_behaviour`:

```python
# src/pca.py — analyse_behaviour()
pca = PCA(n_components=k, svd_solver="full")
pca.fit(activation_matrix)
cumvar = np.cumsum(pca.explained_variance_ratio_)
eigvals = pca.explained_variance_
d_effs = {}
for label, thresh in [("d_eff_50", 0.50), ("d_eff_70", 0.70), ("d_eff_80", 0.80),
                       ("d_eff_90", 0.90), ("d_eff_95", 0.95)]:
    idx = int(np.searchsorted(cumvar, thresh))
    d_effs[label] = min(idx + 1, k)
pr = float((eigvals.sum() ** 2) / (np.sum(eigvals ** 2) + 1e-12))
```

**Design defences.** `svd_solver="full"` is a deliberate reproducibility choice — the source comment explains that the default `"auto"` would pick the randomized solver for these `N≈50–145, d=1536, small-k` shapes, which is non-deterministic without a fixed seed; full SVD is cheap at this N. The PR `(Σλ)²/Σ(λ²)` is preferred over a single d_eff threshold because it is a "sample-size-robust estimate of effective dimensionality" — it does not depend on an arbitrary variance cutoff and degrades gracefully when N is small. The PCA components are also persisted (`save_pca_results` writes them to `.npy`), feeding Phase 6 steering-vector construction and Phase 5d clustering.

**What was found.** The live deduplicated summary (`results/robustness/R1-1.5B/geometry_robustness_summary.md`) reports PR ≈ 20 / 23 / 25 / 29 (backtracking / uncertainty / adding-knowledge / example-testing) in 1536-D — a genuine compression. But PR alone is not the claim; PR is a *PCA* (linear) quantity. The "low-dimensional" claim is carried by the **intrinsic-dimension** estimators (E.2), and crucially by their *survival under the chain control* (E.4).

`05c_cross_layer_probing.py` complements this with two cheap analyses: (a) **chain-grouped** layer-wise linear probing (behaviour-vs-other), and (b) non-adjacent layer-PCA principal-angle evolution. The probe deliberately uses `GroupShuffleSplit` so sentences from one chain never straddle train/test:

```python
# 05c_cross_layer_probing.py — probe_accuracy_at_layer (docstring)
"""Chain-grouped split: sentences from the same chain NEVER straddle
train/test. The previous plain train_test_split leaked chain context — the
exact failure mode tests/test_cv_leakage.py demonstrates inflates a null
probe from 0.50 to 0.99 — so the historical 83–93% accuracies are upper
bounds. Exact-duplicate rows are removed per class first (CF-13)."""
```

The principal-angle method (E.6) intentionally compares only **non-adjacent** layers (k ∈ {3,7,14}) because the residual identity `x_{L+1} = x_L + f(x_L)` makes adjacent subspaces near-identical by construction — a methodological trap the authors explicitly avoid.

---

### E.2 Intrinsic dimension — three estimators, honest CIs

`src/intrinsic_dim.py` implements TwoNN (Facco 2017), a robustified Levina–Bickel MLE, and Grassberger–Procaccia correlation dimension, each with a bootstrap CI. The "convergent estimates → credible; divergent → estimator-dependent caveat" framing is baked into the module docstring.

The TwoNN fit reveals a subtle, well-documented numerical fix:

```python
# src/intrinsic_dim.py — twoNN_estimate._fit()
# Empirical CDF over the FULL sample as i/(n+1) (so F < 1 everywhere),
# THEN keep the lower `fraction` (the linear regime). Computing
# F = arange(1,cutoff+1)/cutoff on the *truncated* set instead forces
# F=1.0 at the cutoff; -log(1-F) then explodes and that single
# high-leverage point dominates the through-origin slope, inflating the
# estimate ~35-50% (dim 5 -> ~6.7, dim 8 -> ~10.2). See tests/.
F_full = np.arange(1, n_mu + 1) / (n_mu + 1)
cutoff = max(2, int(np.ceil(fraction * n_mu)))
```

The **CI strategy** is a genuine scientific correction (AUDIT.md §5 #16, CF-9). All three estimators use `_subsample_bootstrap`: it subsamples *points* (m = 0.8·N without replacement) and recomputes the estimator end-to-end, rather than resampling derived μ-ratios or pairwise distances. The comment is explicit about why: derived quantities are mutually dependent (pairs share points, the kNN graph is fixed), so resampling them gave absurdly narrow CIs like `[0.575, 0.587]`. Subsampling (not n-out-of-n) also avoids creating duplicate points whose zero-distance neighbours corrupt every kNN estimator.

**Estimator reliability finding (important).** In the live run, **TwoNN is judged unstable** on this data and the **correlation dimension is the reported estimator**: the live summary states "twoNN is duplicate/subsample-unstable here; correlation dimension is the reliable estimator." The superseded June-8 TwoNN value of "0.168" was a zero-distance (duplicate-row) artefact (CF-13). The citable intrinsic dims are corr-dim ≈ **5.9 / 6.2 / 6.0 / 7.7** (back / unc / example / add) in 1536-D — substantially below the PCA PR of 20–29, which is the actual evidence for "low-dimensional".

---

### E.3 Claim 2 — Curvature: the diagnostics, and why it is a NEGATIVE

`src/curvature.py` implements three complementary flat-vs-curved diagnostics, all sweepable over kNN size k (stable-across-k → credible; k-dependent → flagged artefactual):

1. **Local-vs-global PCA dim ratio** — flat ≈ 1, curved < 1.
2. **Geodesic/Euclidean ratio** — flat ≈ 1, curved > 1.
3. **Tangent-space variation** (mean principal angle, degrees) — flat ≈ 0, curved > 0.

Two of these encode hard-won calibration fixes. The local-vs-global ratio uses a **sample-size-matched baseline** (random k-point subsets), not the full cloud:

```python
# src/curvature.py — local_vs_global_dim_ratio (docstring)
"""CRITICAL calibration note: the baseline MUST be sample-size-matched to the
local neighbourhood. The original implementation divided by the PCA dim of
ALL N points, which conflated curvature with the trivial fact that a k-point
neighbourhood can express at most k-1 dimensions while the full cloud can
express many more — so a perfectly FLAT subspace of dimension > k scored
<< 1 (empirically 0.29-0.85 on flat Gaussian data). Matching the sample
size removes that confound. See tests/test_curvature.py::test_flat_*."""
```

The geodesic graph is symmetrised with `W.maximum(W.T)` rather than `(W+W.T)/2`, because averaging would *halve* one-directional kNN edges and push a flat-manifold ratio to ~0.73 when it must be ≥ 1. `05b` projects to the top-50 PCA subspace before curvature ("to avoid ambient-dim noise dominating") and all CIs again use the point-subsample bootstrap.

**Why the result is a negative.** This is the sharpest scientific finding of the stage. On the *full* data the manifolds look curved (geo/Euclidean ≈ 3.4–4.2). But at **one-sentence-per-chain** (the chain-stratified / random-subsample-to-n_chains regime) the geodesic ratio collapses toward ~2.3–2.5 and is statistically indistinguishable from a chain-matched relabelling. From the live summary:

```
| Behaviour | geo full | geo randsub | geo chainstrat |
| backtracking          | 3.73 | 2.42 | 2.35 |
| uncertainty-estimation| 4.04 | 2.42 | 2.43 |
| adding-knowledge      | 3.44 | 2.55 | 2.53 |
| example-testing       | 4.22 | 2.55 | 2.34 |
```

The apparent curvature is **within-chain autocorrelation** — sentences walk along a chain trajectory, not along a behaviour-specific curved manifold. The status banner records this as: *"Per-behaviour curvature is a CLEAN, WELL-POWERED NEGATIVE — curved on full data, ≈1.0 (flat) at one-sentence-per-chain."* "Clean" matters because `power_analysis_curvature.py` (E.5) shows the analysis was powered to detect real curvature if it existed; a null result is therefore informative rather than merely underpowered.

---

### E.4 The null hierarchy — the methodological spine (`src/nulls.py`)

The companion document commits to a three-level null hierarchy, implemented in `src/nulls.py`:

- **Primary: chain-stratified permutation** — shuffle labels *within* each chain, preserving per-chain composition, then recompute the statistic on the target-labelled rows. This controls for chain identity, within-chain drift, ambient covariance, and sample size simultaneously.
- **Secondary: cross-chain (global) permutation** — global label shuffle; isolates behaviour-level vs category-level effects.
- **Tertiary: Marchenko–Pastur isotropic** — matched-(N,d) Gaussian; explicitly *only* a finite-sample inflation diagnostic, "NOT a structural test" (its `real_value` is left as NaN by design).

The chain-stratified null contains the single most important safeguard in this stage — the **no-op guard**:

```python
# src/nulls.py — chain_stratified_permutation_null()
n_mixed = sum(1 for idxs in chain_to_idx.values()
              if np.unique(labels[idxs]).size > 1)
if n_mixed == 0:
    raise ValueError(
        "chain_strat_perm: within-chain permutation is a NO-OP — no chain "
        "contains more than one distinct label. This usually means proxy "
        "chain ids (one pseudo-chain per behaviour). Fix the chain-id "
        "provenance ... instead of running a vacuous null.")
```

P-values use the **Phipson–Smyth (2010) smoothing** `(1+count)/(1+B)`, not the naive `count/B`. The docstring is candid: the unsmoothed version can return p=0, which is impossible for a permutation test (the observed labelling is itself a permutation) and "fakes infinite resolution" — at B=100 an unsmoothed 0 was being read against a Bonferroni threshold of 4.5e-4 it could never legitimately pass. `05_pca_analysis.py` even prints the minimum attainable p and warns that Bonferroni across 4×28 cells needs B ≥ 2239.

**Statistic-coverage gap (CF-3) and its fix.** `05b`'s null runs *only* on the top-10 variance ratio. But the load-bearing numbers are intrinsic dim and curvature. `tier1_geometry_nulls.py` re-runs the *same* `src/nulls.py` machinery with `statistic_fn ∈ {twoNN, levina_bickel, local_vs_global_dim_ratio}` and **`tail='lower'`** (the claims are "dim LOW" / "ratio LOW (curved)", so real must sit *below* the null). The script carries an explicit hazard note that wiring tail incorrectly silently inverts the test for the other two curvature diagnostics.

---

### E.5 Power analysis — what makes the curvature negative "well-powered"

`power_analysis_curvature.py` is a pre-registered power study: it generates ground-truth curved manifolds (hyperspheres `S^{m-1}` of radius 1/κ embedded in R^1536 + isotropic noise) and matched flat controls (uniform disks), runs each diagnostic on both, and reports detection power as the curved-vs-flat AUC. The pre-registration commitment is hard-coded in the docstring: *"If the smallest detectable curvature at a given N exceeds the curvatures plausibly induced by reasoning structure, the per-behaviour analysis is downgraded to a corpus-pooled analysis."*

The result table (`results/power_analysis/summary.md`): the local-vs-global diagnostic detects κ ≥ 0.5 at **N ≥ 500**, and per-behaviour pools are ~600–900 chains. So the null is not an artefact of low power — the experiment *could* have seen curvature and did not. A defensive guard refuses to write an all-NaN power table, treating systematic numerical failure as a bug rather than a "null result":

```python
# power_analysis_curvature.py — main()
if cells and all(not np.isfinite(c.auc) for c in cells):
    raise RuntimeError(
        f"All {len(cells)} power cells produced non-finite AUC. This is a "
        "systematic failure ... NOT a real 'undetectable' result. "
        "Refusing to write an all-NaN power_table.csv. ...")
```

Caveat: the geodesic and tangent diagnostics are weaker (geodesic needs N≥2000 for κ≥0.5; tangent never reaches AUC 0.95 in the tested grid up to N=2000). So "well-powered" rests primarily on the local-vs-global diagnostic.

---

### E.6 Effective-N — quantifying the chain confound (`tier1_effective_n.py`)

This script computes the honest denominator behind every estimator. It fits PCA per behaviour, computes the **one-way ICC(1)** of the PC scores grouped by chain, and reports the design effect `Deff = 1 + (n0−1)·ICC` and `n_eff = N/Deff`:

```python
# tier1_effective_n.py — icc_oneway()
msb = ssb / (G - 1)
msw = ssw / (N - G)
n0 = (N - (counts ** 2).sum() / N) / (G - 1)
denom = msb + (n0 - 1) * msw
icc = (msb - msw) / denom if denom > 0 else float("nan")
```

The finding is stark. PC1 ICC is **0.88–0.93** across all four behaviours — sentences within a chain are massively autocorrelated. The raw N of 5k–16k collapses to n_eff(PC1) ≈ 700–1000, i.e. essentially the number of chains:

| Behaviour | N sent | N chains | ICC PC1 | n_eff(PC1) |
|---|---|---|---|---|
| backtracking | 10267 | 705 | 0.930 | 756 |
| uncertainty-estimation | 16728 | 909 | 0.891 | 1015 |
| example-testing | 5829 | 636 | 0.922 | 686 |
| adding-knowledge | 5027 | 881 | 0.881 | 980 |

This is *why* the curvature collapses to flat at one-sentence-per-chain and *why* the intrinsic-dim claim is only credible because it survives the chain control. The whole Gate-0 epistemology — "compare to a chain-matched null, report n_eff not N" — rests on these ICC numbers.

---

### E.7 Claim 3 — Behaviour-specificity (MIXED, 2/4)

Specificity is the question "is this subspace *specific* to the behaviour, or a generic chain property?" — answered by the chain-stratified variance-ratio null. The status banner records the verdict bluntly: **backtracking + uncertainty-estimation are specific at all layers (B=2500 null, p<.001); example-testing only at L27; adding-knowledge nowhere (p=1.0).** This is the single most threatening result for the thesis spine, because the headline claims behaviours are "behaviour-specific" and **one of the four fails everywhere**, and a second only at the chosen steering layer.

`05d_subtype_clustering.py` is downstream of this: it K-means-clusters each behaviour's activations (silhouette-selected k ∈ [2,8]) at the per-behaviour manifold-peak layer to discover sub-types, producing centroids used for sub-type steering vectors. Note its layer-selection heuristic is `argmin(participation_ratio)`, with a documented bug history:

```python
# 05d_subtype_clustering.py — _resolve_focus_layer (docstring)
"""... We therefore take argmin(participation_ratio).
The earlier argmax(d_eff_70) rule returned layer 0 whenever d_eff saturated at
the PCA component cap, which is why clustering ran at the wrong layer."""
```

---

### E.8 Critique — what is genuinely established vs over-claimed

**Genuinely established (citable):**
- Low intrinsic dimension (corr-dim ≈ 6–8 in 1536-D) **survives** the chain control — the keystone PASS. This is real and the methodology (chain-stratified null + point-subsample CIs + dedup) is rigorous.
- The curvature negative is clean and well-powered — a defensible, honestly-reported null. Striking "curved" from the headline is the right call.

**Over-claimed or fragile:**
1. **"Behaviour-specific" is half-true at best (2/4).** adding-knowledge fails specificity *at every layer* (p=1.0); example-testing is specific *only at L27*. Any thesis sentence asserting all four behaviours are specific is unsupported. example-testing's L27-only specificity is especially uncomfortable because **L27 is the chosen steering layer** — there is a circularity risk that the layer favouring specificity is also the layer steering will run at.
2. **Specificity is single-annotator.** The R2.2 replication (Sonnet/Qwen3/Nova) confirms intrinsic-dim and curvature-as-artefact replicate 3-way despite κ=0.35–0.44, but the **variance-ratio specificity null is still single-annotator**. So behaviour-specificity replicates at the *subspace* level, not yet at the *specificity-test* level. With fair-to-moderate inter-annotator agreement, the 2/4 split could itself be label-noise-dependent.
3. **Estimator-dependence is real and was nearly fatal.** TwoNN is unstable here and gave a duplicate-artefact "0.168" in the superseded June-8 run. The conclusion now rests on the *correlation dimension* specifically — a single estimator choice. The module's own framing ("divergent estimates flag the caveat that intrinsic dimension is an estimand whose value depends on the estimator") cuts against over-confidence.
4. **Duplicate-row hygiene (CF-13) was load-bearing and recent.** 35–56% of pooled activations were exact duplicates (short repeated markers in a fixed token window) before the fix. Everything pre-2026-06-08 is **superseded / do-not-cite** (the `geometry_nulls_layer27.md` file carries an explicit "⚠️ SUPERSEDED — DO NOT CITE" banner). This is a reminder of how sensitive the pipeline is to preprocessing.
5. **PR vs intrinsic dim conflation risk.** PR (20–29) is a *linear* PCA quantity; intrinsic dim (6–8) is the nonlinear claim. They must not be reported interchangeably as "the dimension". The "low-dimensional" headline should cite the chain-controlled intrinsic dim, not PR.
6. **Bonferroni vs B mismatch.** `05_pca_analysis.py` itself warns that proper multiple-comparison control across 112 cells needs B ≥ 2239, while several runs used B=100–200. The specificity claims that *passed* used B=2500, but cross-layer null sweeps at low B are resolution-limited.
7. **n_eff caveat compounds CIs.** With n_eff ≈ n_chains, any CI or null still implicitly built on raw N is overconfident; R1.2 (chain-block bootstrap) is the owed fix, and `tier1_geometry_nulls.py` runs estimators with `n_bootstrap=0` inside the null (no nested CI) — so the geometry-null p-values do not yet carry honest CIs.

---

### E.9 Connection to the steering decision

This stage feeds the imminent steering experiment in three concrete ways:

1. **Layer choice.** The steering layer is hardcoded to **L27** for R1-1.5B via `STEERING_LAYERS`, inherited from Huang. The geometry stage gives weak independent support for L27: it is the *only* layer where example-testing is behaviour-specific, but that is a double-edged observation (circularity — see critique #1). The MEMORY notes the steering layer is "NOT finalised" and the plan is to build mid-layers (11/16/19) + L27 and let Phase 7 decide. The geometry results do **not** strongly endorse L27 over mid-layers; `05c`'s layer-wise probe curves and the per-layer null sweep (`null_pvalues_per_layer.json`) are the relevant evidence and should be consulted before committing GPU spend to a single layer.

2. **Methodology — vectors and sub-types.** Phase 6/7 build steering vectors from the PCA components (`save_pca_results`) and from the 05d cluster centroids. The "manifold (k-component)" steering arms are direct descendants of the PR / d_eff / clustering outputs here. The git status shows the `*_manifold_k{1,3,5,10,auto}.npy` and `*_single.npy` steering vectors are already staged — i.e. the geometry → steering-vector handoff has run.

3. **What the geometry results *constrain* about steering claims.** Because curvature is a negative, steering claims should **not** invoke "moving along a curved manifold" — only "moving within a low-dimensional subspace". Because adding-knowledge fails specificity, a steering null on adding-knowledge is *expected* and should not be read as a steering failure; conversely a specificity-respecting design should weight backtracking/uncertainty (the two clean PASSes) as the primary steering targets. Spending API+GPU budget to steer adding-knowledge as a headline arm is the highest-risk allocation given the geometry says its subspace is not behaviour-specific.

**Run/unrun status:** the geometry pipeline (PCA, 05b/c/d, tier1 effective-N, tier1 geometry-nulls, power analysis) is **RUN** on clean deduplicated data and **CITABLE** (post-2026-06-18). The pre-2026-06-08 `geometry_nulls_layer27` outputs are **QUARANTINED / do-not-cite**. The specificity null's multi-annotator replication is **OWED**. **Phase-7 steering itself is built but UNRUN.**
