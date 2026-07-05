## G. Steering Vector Construction (Phase 6)

This section documents Phase 6 of the pipeline: the construction of the steering vectors that Phase 7 will later add into the residual stream to causally test the manifold hypothesis. Phase 6 is **CPU-only, runs in under a minute, takes no API or GPU spend, and is RUN** — the canonical artefacts exist on disk at `results/steering_vectors/R1-1.5B/` (dated 2026-06-18, built with hold-out ON). It is the *cheap, reversible, already-done* half of the steering experiment; the expensive, unrun half is Phase 7 (generation + re-annotation). Everything in Phase 6 is arithmetic over the saved activation matrices from Phase 4 — no model is loaded.

The reader is about to commit real API+GPU spend on Phase 7. Phase 6 is the stage that *frames the central causal claim* — single-direction (flat) vs manifold-projected (low-dim subspace) — and that *fixes the layer*. Both of those decisions are made here, and both are critiqued below.

---

### G.1 What is built, and the two-track design

For each of the four target behaviours (`backtracking`, `uncertainty-estimation`, `example-testing`, `adding-knowledge`), Phase 6 builds **two families** of steering vectors at a chosen layer:

1. **Single-direction** (`*_single.npy`) — Venhoff-style difference-of-means, unit-normalised.
2. **Manifold-projected** (`*_manifold_k{1,3,5,10,auto}.npy`) — the same difference-of-means vector orthogonally projected onto the top-*k* PCA subspace of the behaviour's own activations, then renormalised.

The module docstring states the design and, crucially, the *prediction* that makes this a real experiment rather than a description:

```python
# src/steering.py — module docstring
#   1. Single-direction (Venhoff-style)
#      r = mean(on_activations) − mean(off_activations),  normalised to unit norm.
#      "off" = activations from all *other* behaviours, providing a neutral baseline.
#   2. Manifold-projected (Huang-style, our method)
#      Compute the single-direction vector r, then project it onto the top-k
#      principal components of the behaviour's own activation subspace:
#        r_proj = Σ_{i=1}^{k} (r · v_i) v_i,  normalised to unit norm.
# The key prediction: if behaviours have manifold structure, the manifold-projected
# vector should yield cleaner behaviour suppression (less off-target disruption,
# better saturation curve) than the single-direction vector.
```

The `single` vector is the standard interpretability baseline (the "linear representation hypothesis" steering vector). The `manifold_k` family is the contribution: it operationalises the project's headline geometry finding (behaviours occupy a *low-dimensional curved-ish subspace*) into a competing steering method. If the subspace finding is real and causally load-bearing, anchoring the steering direction *inside the behaviour's own manifold* should suppress more cleanly. That is the entire point of building both tracks: Phase 7 compares them head-to-head.

### G.2 The mean-difference primitive

The core build is a difference of class means, normalised. Note the **fail-loud guards** — these are not cosmetic; they were added because an empty mean (`mean` of a zero-row matrix) is all-NaN, and `NaN < 1e-10` evaluates to `False`, so a naive near-zero guard would emit a silent NaN "unit" vector that poisons every downstream steered generation:

```python
# src/steering.py — single_direction_vector()
    if on_activations.shape[0] == 0 or off_activations.shape[0] == 0:
        raise ValueError(
            f"single_direction_vector needs non-empty inputs "
            f"(on={on_activations.shape[0]}, off={off_activations.shape[0]} rows)")
    r = on_activations.mean(axis=0) - off_activations.mean(axis=0)
    norm = np.linalg.norm(r)
    # NB: a plain `norm < 1e-10` is False when norm is NaN/inf, so it would let a
    # degenerate vector through; guard finiteness explicitly.
    if not np.isfinite(norm) or norm < 1e-10:
        logger.warning("Steering vector has near-zero/non-finite norm — returning zero vector")
        return np.zeros_like(r)
    return r / norm
```

The "off" baseline is **all other behaviours concatenated**, not a generic-text baseline. This is decided in `build_steering_vectors`:

```python
# src/steering.py — build_steering_vectors()
        on_acts = all_acts[beh]
        # "off" = all other loaded behaviours concatenated
        off_parts = [v for k, v in all_acts.items() if k != beh]
        ...
        off_acts = np.concatenate(off_parts, axis=0)
```

**Why "off = other behaviours"?** It makes the vector point along *what is specific to this behaviour relative to the other reasoning behaviours*, not relative to arbitrary text. This is the right contrast for a *behaviour-specificity* claim. But see the critique (G.8): it also makes the vector's meaning depend on the composition of the off-set, and the off-set is dominated by whichever behaviour is most frequent.

### G.3 The manifold projection — why k > 1

The manifold vector takes the single direction `r` and orthogonally projects it onto the top-*k* PCA subspace of the *on* activations, then renormalises:

```python
# src/steering.py — manifold_projected_vector()
    r = single_direction_vector(on_activations, off_activations)
    n_components = min(k, on_activations.shape[0] - 1, on_activations.shape[1])
    ...
    pca = PCA(n_components=n_components, svd_solver="full")  # exact + reproducible
    pca.fit(on_activations)
    V = pca.components_  # (k, hidden_dim)
    coords = V @ r               # (k,) — coordinates in PCA space
    r_proj = coords @ V          # (hidden_dim,) — back in activation space
```

Two implementation details defended by the tests:
- `svd_solver="full"` is deliberate. A randomized SVD made the projection non-reproducible run-to-run; `test_manifold_projection_is_deterministic` is the regression guard.
- The projection is exactly the orthogonal projector `(VᵀV) r` renormalised. `test_manifold_projection_equals_VVt_r` checks this against an independent PCA fit and confirms the residual `r − (VᵀV)r` is orthogonal to the subspace.

**Why a manifold (k>1) vector at all?** This is the conceptual hinge that ties Phase 6 to the project's structural finding. The geometry chapters establish that each behaviour's activations live in a *low-dimensional subspace* (low participation ratio / variance concentrated in a few PCs). A single difference-of-means direction `r` is a 1-D object that may point *partly out of that subspace* — i.e. into directions the model never actually uses for that behaviour. Projecting `r` into the top-*k* PCs **discards the off-manifold component of the steering direction**, keeping only the part that lies along axes the behaviour genuinely varies on. The pre-registered hope: steering along the in-manifold direction perturbs the model *the way the behaviour itself does*, so it suppresses the behaviour with less collateral damage (less repetition, less off-target leakage, smaller accuracy hit) than a single direction that injects energy into unused directions.

The `k` sweep `{1, 3, 5, 10, auto}` is a dose-response over subspace dimensionality. `k=1` is *almost* the single direction (the projection onto the single top PC) and serves as a near-degenerate anchor; larger `k` lets more of `r` survive. The metadata shows `cos(single, manifold@k)` grows toward 1 as `k` rises (`build_phase6.py` prints exactly this convergence table, with `energy = cos²` = "fraction of the difference-of-means direction's energy that lives inside the manifold"). The diagnostic value: if `auto_k` is large and `cos` is near 1, the single direction *already* lives in the manifold and the two methods will barely differ — which is itself informative about how curved/concentrated the behaviour is.

`auto_k` picks the smallest k explaining ≥70% of variance:

```python
# src/steering.py — auto_k()
    pca = PCA(n_components=max_k, svd_solver="full")  # exact + reproducible
    pca.fit(on_activations)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    idx = int(np.searchsorted(cumvar, variance_threshold))
    return min(idx + 1, max_k)
```

The realized `auto_k` values are **large**: from `metadata.json`, `backtracking=58`, `uncertainty-estimation=71`, `example-testing=60`, `adding-knowledge=83`. This is a quiet but important fact (G.8): a "low-dimensional manifold" that needs 58–83 PCs to reach 70% variance is low-dimensional only relative to the 1536-D hidden size. At `k=auto` the manifold vector is nearly the single vector, so the *cleanest* contrast between the two methods is at small `k` (1, 3, 5), not auto.

### G.4 True hold-out construction (commit 58cf04a)

The single most important methodological upgrade in this stage. By default the builder **excludes the Phase-7 evaluation tasks' activation rows** from vector construction, turning Phase 7 from an "on-corpus" causal effect into a genuine out-of-sample test. The eval split is computed once and shared by both the builder and Phase 7 so it cannot drift:

```python
# 06_build_steering.py — main()
    exclude_ids = None
    if not args.no_holdout:
        from src.task_gen import load_tasks, stratified_eval_split
        test_tasks, rule = stratified_eval_split(load_tasks(args.tasks), args.n_test)
        exclude_ids = {t["id"] for t in test_tasks}
        logger.info(f"Hold-out: excluding {len(exclude_ids)} eval tasks ({rule}) "
                    f"from vector construction")
```

The exclusion happens row-by-row using a **provenance sidecar** (`row_index.json`) that maps each activation row to its source chain/task id. If provenance is missing, it **fails loud** rather than silently skipping the hold-out:

```python
# src/steering.py — build_steering_vectors()
        if exclude_chain_ids:
            from src.row_provenance import require_aligned
            cids = require_aligned(beh, X.shape[0], chain_id_map.get(beh),
                                   context="steering hold-out")
            keep = ~np.isin(cids, list(exclude_chain_ids))
            n_excluded[beh] = int((~keep).sum())
            X = X[keep]
```

The shared split is **category-stratified** because the underlying `tasks_final.json` is category-blocked — the naive `tasks[-n_test:]` would have selected 50 tasks of a *single* category:

```python
# src/task_gen.py — stratified_eval_split()
    per_cat = max(1, n_test // len(by_cat))
    test_tasks = [t for cat in sorted(by_cat) for t in by_cat[cat][-per_cat:]]
    return test_tasks, f"last {per_cat} per category"
```

The `metadata.json` records the hold-out actually fired: `n_excluded` is 669 / 1018 / 364 / 310 rows for the four behaviours (~5% of rows), and `_provenance.holdout` records `{"n_tasks": 50, "rule": "src.task_gen.stratified_eval_split"}`. The hold-out is well-tested: `test_holdout_excludes_eval_rows` checks counts *and* that excluding the extreme eval rows visibly rotates the direction (`cos < 0.999`); `test_holdout_fails_loud_without_provenance` and `test_build_holdout_mismatch_fails_loud` guard the fail-loud contract.

**The residual caveat is documented honestly in the commit itself**: the *layer* choice was informed by full-corpus analyses + Huang's published layer 27, so **layer selection is not held out**. This is the one leak that survives (G.8).

### G.5 Per-arm linear build (multi-annotator robustness)

`build_steering_arms.py` builds the *same* linear vectors independently for three annotator arms — the same base model (R1-1.5B) labelled by Sonnet-4.5, Qwen3-235B, and Nova-Pro — and reports **cross-arm replication**: the cosine between arms' single-direction vectors per behaviour. The headline robustness check is whether the *steering direction itself* is annotator-invariant:

```python
# build_steering_arms.py — _cross_arm_replication()
    """cos between arms' single-direction vectors, per behaviour. High => the
    steering direction is annotator-robust (the multi-annotator headline)."""
    ...
    cells.append(f"{float(va @ vb):+.3f}".rjust(20) if va is not None and vb is not None
                 else "n/a".rjust(20))
```

The script is emphatic that **steering is purely LINEAR here — no curvature/geodesic content** (that is deferred diagnostic work). It is CPU-only and skips arms whose activations are not yet on disk (reported `[PENDING]`). This dovetails with the R2.2 three-way replication already folded into the thesis (geometry replicates across Sonnet/Qwen3/Nova despite low annotator agreement κ≈0.35–0.44). Note: this builder is the one that *clobbers* the canonical `results/steering_vectors/R1-1.5B/` directory for the Sonnet arm (its `ARMS["sonnet-4.5"] = "R1-1.5B"`), so the on-disk canonical vectors may have been (re)written by either `06_build_steering.py` or this script — both write identical filenames there.

### G.6 Per-behaviour-peak build, and the clobber hazard

`build_phase6.py` builds vectors with each behaviour at *its own* participation-ratio-trough ("manifold peak") layer rather than a shared layer. Critically, it writes to a **distinct directory** (`-peak`) precisely because the two builders write identical filenames:

```python
# build_phase6.py — header
# Output dir is deliberately DISTINCT from 06_build_steering.py's
# (results/steering_vectors/R1-1.5B = the canonical all-behaviours-at-layer-27
# build): the two builders write identical filenames, so sharing a directory
# meant whichever ran last silently clobbered the other and Phase 7 evaluated
# whichever geometry happened to be on disk. Point 07 at this dir explicitly to
# evaluate the per-behaviour-peak variant.
```

This is a genuine prior bug that *did* silently corrupt which geometry Phase 7 evaluated. The peak layers from config are `backtracking/uncertainty/adding-knowledge = 16`, `example-testing = 12`. The config itself flags these are stale-on-disk and warns that PR-trough = manifold *concentration*, not behaviour-*specificity*: `example-testing` is specific only at L27 (its trough L12 is *not* specific) and `adding-knowledge` is *not specific at any tested layer*. `build_phase6.py` also reports `cos(single, manifold@auto)` and `energy = cos²` per behaviour — the in-manifold-energy diagnostic.

`trim_vectors.py` is a small utility that makes **auto-only copies** of layer-variant dirs (L16/L27) for a trimmed bake-off: it deletes the `k{1,3,5,10}` files and rewrites `metadata.json`'s `k_values` to `["auto"]` so the metadata-driven loader pulls only the auto vector. This is purely a disk/scope-reduction convenience for the Phase-7 arm budget.

### G.7 Composition and matched-effect analysis (built, unrun on real data)

`06b_steering_composition.py` is a **diagnostic-only, no-inference** pre-registration of the composition test: for each pair of behaviours it forms `v_sum = v_a + v_b`, a PCA-projected `v_proj`, and a tangent-bundle `v_tan`, and reports cosines and an **off-manifold ratio** `||v_sum − v_proj|| / ||v_sum||`. The pre-registered predictions are explicit: flat picture ⇒ `cos(v_sum,v_proj)≈1`, ratio≈0; curved picture ⇒ ratio>0 scaling with curvature. The *behavioural* composition test still requires Phase-7 inference and is "to be wired separately."

`src/steering_analysis.py` is the most sophisticated piece here and directly answers the sharpest fairness objection to the whole experiment. Because the manifold vector is a *renormalised projection* of the single direction, **at equal α the two arms deliver different perturbation magnitudes** — so "manifold has lower repetition at α=1" would be uninterpretable. The module enforces comparison **at matched on-target effect**:

```python
# src/steering_analysis.py — collateral_at_matched_effect()
    """Interpolate *curve*'s damage at a MATCHED on-target suppression.
    ...
    This is THE comparison primitive: damage is always read at equal effect, so
    "equal α ≠ equal perturbation" can never contaminate it."""
```

It provides: matched-effect collateral interpolation, the effect-vs-damage Pareto frontier, a **paired BCa bootstrap over tasks** of the matched-effect damage difference (`damage_B − damage_A`, positive ⇒ A wins), and **Holm–Bonferroni across the four behaviours** on one pre-registered statistic. The bootstrap unit is the *task* (each task contributes a whole curve), verified by `test_bootstrap_resampling_unit_is_task_not_observation`. This module is **purely arithmetic over summaries — it generates, runs, and judges nothing** — and is extensively unit-tested on synthetic data with known sign/dominance (`tests/test_steering_analysis.py`, ~40 tests). It is *ready* but has never consumed real Phase-7 output, because Phase 7 has not run.

### G.8 Critique — confounds, fragilities, untested assumptions

**1. Layer selection is NOT held out (the surviving leak).** The hold-out excludes eval *rows* from the mean-difference, but the *layer* (27, or the peak layers) was chosen using full-corpus analyses and Huang's published value. If layer 27 is where the behaviours are most separable *on this corpus including the eval tasks*, the out-of-sample claim is weaker than the per-row hold-out implies. The commit message admits this; the thesis must state it. This is the single sharpest weakness of the stage.

**2. `auto_k` is large ⇒ manifold ≈ single at the headline operating point.** With `auto_k` = 58–83 (to hit 70% of variance in 1536-D), the `manifold_kauto` vector is nearly collinear with the single direction. So the *most natural* manifold arm is barely distinguishable from the baseline, and any clean separation must come from small-`k` arms (1/3/5) — which are *more* aggressive interpolations and arguably less faithful to "the manifold". The experiment's discriminating power is concentrated in exactly the arms that are hardest to defend as "the model's natural subspace". The 70% threshold is itself an unjustified free parameter never sensitivity-tested.

**3. The off-set is composition-dependent and imbalanced.** `off = concat(other behaviours)`, and the row counts are wildly unequal (`uncertainty-estimation` has 15.7k on-rows vs `adding-knowledge`'s 4.7k). When `adding-knowledge` is the target, its off-set is dominated by `uncertainty-estimation`, so its difference-of-means partly encodes "not-uncertainty" rather than "adding-knowledge". No reweighting or per-behaviour balancing is applied. Combined with the prior finding that **`adding-knowledge` is not behaviour-specific at any layer**, its steering vector may be measuring an artefact.

**4. PCA is fit on the *on* activations including their mean shift, but `sklearn.PCA` centres internally.** The projection subspace is the *covariance* subspace of the on-activations (mean-removed), while `r` is a *difference of means*. There is no guarantee the mean-shift direction lies in the high-variance covariance subspace — indeed projecting can throw away most of `r` at small k (low `energy`). That is *intended*, but it means the manifold vector can be dominated by within-behaviour variance directions that have nothing to do with the on-vs-off contrast. The method conflates "directions the behaviour varies along" with "the direction that distinguishes the behaviour", and these need not align.

**5. Manifold is a misnomer for a linear projection.** Despite the "manifold/curved" framing, every vector here is a *linear* PCA projection (`build_steering_arms.py` says so outright: "No curvature/geodesic content"). The curved-manifold language in `06b`'s pre-registration is not backed by any geodesic construction in the built vectors — the composition test infers curvature only indirectly via off-manifold ratios. The thesis should not let the word "manifold" imply more geometry than a linear subspace projection delivers.

**6. Clobber hazard is mitigated but not eliminated.** Three scripts (`06_build_steering.py`, `build_steering_arms.py` Sonnet arm, and historically `build_phase6.py`) can write `results/steering_vectors/R1-1.5B/`. `build_phase6.py` now redirects to `-peak`, but `06` and the Sonnet arm of `build_steering_arms.py` still target the same canonical dir with identical filenames. The on-disk canonical metadata says it was built by `06_build_steering.py` at layer 27 with hold-out, which is the intended provenance — but the only thing preventing a stale overwrite is operator discipline. The `metadata.json.bak` backup is the safety net.

**7. The whole comparison is unrun.** Phase 6 vectors exist and are QA'd; `steering_analysis.py` is green on synthetic data; but **no behaviour-fraction, no matched-effect curve, no Pareto frontier, no bootstrap has ever been computed on real steered generations.** Every causal claim about single-vs-manifold is, as of this recap, a pre-registration. The vectors are citable as *constructed*; nothing about their *effect* is.

### G.9 Connection to the imminent steering decision

This stage is exactly where the two pending decisions live:

- **Layer.** `06_build_steering.py` defaults to `STEERING_LAYERS["R1-1.5B"] = 27` (Huang-aligned, behaviour-specific for 3/4 behaviours), while `build_phase6.py` offers the per-behaviour peak layers (16/16/16/12) which the config itself flags as concentration-not-specificity and partly stale. The memory note says: build mid + L27 and *let Phase 7 decide*. The hold-out makes that defensible **only if you treat the layer comparison as exploratory**, because layer choice is not itself held out. Recommendation implicit in the code: prefer L27 as canonical (specific + Huang + already built with hold-out), treat peak/L16 as a robustness arm, and override `example-testing` back to 27.

- **Methodology (single vs manifold, and which k).** The vectors for both methods and all k are already on disk. The matched-effect machinery in `steering_analysis.py` is the *correct* way to compare them (never at equal α). Given `auto_k` ≈ 58–83 collapses manifold onto single, the live scientific question is whether *small-k* manifold arms beat single at matched effect — that is the arm worth spending generation budget on, alongside single as baseline and `kauto` as the "honest, conservative" manifold arm. `trim_vectors.py` exists precisely to cut the k-sweep down to `{single, auto}` if budget forces it.

In short: Phase 6 has already spent the cheap currency (CPU, PCA) to lay out a clean, well-guarded, mostly-held-out experiment. The expensive currency (API + GPU for Phase 7) should be spent knowing that (a) the layer is the one un-held-out degree of freedom, (b) `kauto` is nearly the baseline so small-k is where the contrast lives, and (c) the analysis code is ready and pre-registered — what is missing is real steered output to feed it.
