## F. Robustness & Cross-Model Replication (R2.2)

This section documents the robustness layer of the geometry pipeline: the machinery that asks "does the per-behaviour geometric story survive (a) being re-annotated by a *different* LLM judge, and (b) being run against a *different model's* activations?" Two distinct things travel under nearby names in the codebase and it is essential not to conflate them:

- **R2.2 — annotator replication (RUN, citable).** The same R1-Distill-1.5B activations are re-labelled by three different LLM annotators (Sonnet-4.5, Qwen3-235B, Nova-Pro), and the geometry diagnostics (`robustness_geometry.py`) are recomputed per annotator. The claim under test is annotator-robustness of the geometry, *not* a second model.
- **M6 / Extension A — cross-*model* replication (UNRUN, blocked).** `13_baseline_replication.py` + `src/cbs/comparison.py` compare R1-Distill-1.5B against the base model Qwen-2.5-Math-1.5B. This is a separate, unrun arm whose entire `results/cbs/cross_model/` output directory does not exist on disk.

The bootstrap-CI machinery (`src/nulls.py`, `tests/test_bootstrap_ci.py`, `tests/test_cross_model_bootstrap.py`) underpins both. The reader is about to commit API+GPU spend on a steering experiment; the load-bearing relevance of this section is that **R2.2 is what licenses treating the LLM-annotator labels as a trustworthy enough dependent variable to bother steering at all** — and its limits bound how much the steering result can claim.

---

### F.1 What R2.2 actually does

`robustness_geometry.py` is a single CPU script that runs the "Tier 0" geometry suite from already-extracted activation `.npy` files. It is parameterised by `--model-short`, and the *same script* is run three times with three different annotation files swapped in. The model is always R1-1.5B; only the labels (and hence which sentences land in each behaviour's activation matrix) change.

The core loop, per behaviour `b` at its peak layer `L`:

```python
# robustness_geometry.py, main()
for b, L in PEAK.items():
    Xr = np.load(ACT / f"{b}_layer{L}.npy").astype(np.float32)
    cids = require_aligned(b, Xr.shape[0], cidmap.get(b), context="robustness_geometry")
    dup_pct = 100.0 * (1 - len(np.unique(Xr, axis=0)) / Xr.shape[0])
    X, cu = dedup(Xr, cids); Nu = X.shape[0]
    obc = {}
    for i, c in enumerate(cu): obc.setdefault(c, []).append(i)
    uniq = list(obc); nc = len(uniq)
```

The keystone design is the **same-N control triad** — for both intrinsic dimension and curvature, the script compares three quantities of *equal sample size*:

```python
# robustness_geometry.py, main()
cf = cdim(X, nb=B_DIM); rs, st = [], []
gf, lf = geo(X), lgr(X)
for s in range(B_CURV):
    g = np.random.default_rng(SEED + s)
    ridx = g.choice(Nu, nc, replace=False)                       # random subsample, size = n_chains
    sidx = np.array([g.choice(obc[c]) for c in uniq])            # chain-stratified: 1 sentence/chain
    rs.append(cdim(X[ridx], nb=4)); st.append(cdim(X[sidx], nb=4))
    g_rs.append(geo(X[ridx], nb=6, npairs=250)); g_st.append(geo(X[sidx], nb=6, npairs=250))
```

**Why this design.** The deepest confound in the whole geometry programme is **CF-2 (chain autocorrelation / effective-N)**: sentences inside one CoT chain are not independent draws, so the ~5–16k "points" feeding every intrinsic-dim/curvature estimator violate the i.i.d. assumption those estimators are built on. A naive low-dimension result could be an artifact of sampling many near-identical points from a few trajectories. The triad isolates that:

- `full` uses all unique points.
- `random_sub` draws `n_chains` points at random — same N as the stratified version, but *not* one-per-chain.
- `chain_strat` draws exactly one sentence per chain — kills within-chain autocorrelation while holding N fixed.

If `chain_strat ≈ random_sub ≈ full`, the geometry is a property of the behaviour, not of chain repetition. The summary states the reading explicitly:

```python
# robustness_geometry.py, markdown summary
"- **cdim chainstrat ≈ cdim randsub ≈ cdim full** → low intrinsic dimension is "
"behaviour-intrinsic, NOT a chain confound (keystone PASS).",
"- **geo randsub vs geo chainstrat** at equal N isolates real chain-trajectory "
"curvature from sparse-graph effects.",
```

The curvature axis is deliberately the opposite: `geo full` is expected to be *higher* than `geo chain_strat`, because geodesic curvature is partly a within-chain-trajectory property. A drop from full to stratified is the "curvature is a chain artefact" signature — which is exactly the negative result the project decided to own (CF-2/CF-7), not a positive curvature claim.

Estimator choice is also defended in code: correlation dimension is PRIMARY ("stable"), twoNN is computed but flagged as duplicate/subsample-unstable. This matters because the earlier (quarantined) twoNN-based numbers (e.g. "dim 0.168") were artefacts of exact-duplicate rows (CF-13).

---

### F.2 The 3-way result (RUN, citable)

All three annotator runs exist on disk under `results/robustness/{R1-1.5B, R1-1.5B__nova-pro, R1-1.5B__qwen3-235b}/`, each with `geometry_robustness.json`, `_summary.md`, and `provenance.json` (seed=42, input SHA stamped; note `git_commit: null` — these were produced on the no-git cluster copy).

**Intrinsic dimension (cdim full) replicates with the same ordering across annotators:**

| Behaviour | Sonnet cdim | Qwen3 cdim | Nova cdim |
|---|---|---|---|
| backtracking | 5.85 | 6.67 | 6.64 |
| uncertainty-estimation | 6.21 | 7.00 | 7.15 |
| adding-knowledge | 7.71 | 8.36 | 8.70 |
| example-testing | 6.04 | 6.56 | 5.96 |

The numbers are not identical, but they sit in a tight ~1.0-dim band and preserve ordering (adding-knowledge highest, backtracking/example-testing lowest). Critically, the **keystone PASSES in all three** annotators: `chain_strat` tracks `full` (e.g. Sonnet backtracking full 5.85 vs chainstrat 6.36; Nova 6.64 vs 6.75; Qwen3 6.67 vs 6.72). The low-dimensionality of the per-behaviour subspaces is therefore not an annotator artefact and not a chain artefact.

**Curvature-as-chain-artefact also replicates.** In every annotator, `geo full` (≈3.0–4.2) collapses toward `geo chain_strat` (≈2.3–2.5) — e.g. Sonnet backtracking 3.73 → 2.35; Nova 2.98 → 2.33; Qwen3 2.96 → 2.34. The geodesic "curvature" is largely a within-chain trajectory effect in all three. This is consistent with the project's decision (CONFOUNDS §CF-2) to report curvature as a clean *negative* (chain artefact) rather than a manifold-curvature claim.

**Inter-annotator agreement is low** (`results/robustness/cross_annotator_comparison.md`), which is what makes the replication load-bearing rather than redundant:

| Pair | κ (6-label) | κ (target vs other) | span-F1 (IoU≥0.5) |
|---|---|---|---|
| Sonnet vs Qwen3 | 0.436 | 0.352 | 0.306 |
| Sonnet vs Nova | 0.350 | 0.305 | 0.257 |
| Qwen3 vs Nova | 0.345 | 0.263 | 0.307 |

Label *distributions* diverge substantially: uncertainty-estimation is 21.6% of Sonnet spans but only 7.0% of Qwen3; Nova's deduction is 54.1% (plausibly inflated by the CF-18 unknown→deduction coercion on Nova's ~290/1000 partial-parse failures). The headline claim is therefore strong: **despite κ in the fair-to-moderate range and very different label priors, the geometry lands in the same place.** That is the cleanest external-validity statement the geometry programme has.

---

### F.3 Bootstrap-CI and cross-model machinery

`src/nulls.py` is the null-hypothesis hierarchy used by the *geometric* per-layer runs (the `results/geometric/.../diagnostics_layer*.json` files), not by `robustness_geometry.py` directly, but it is the engine behind the variance-ratio specificity claim that R2.2 is being measured against. Its primary null is the chain-stratified within-chain label permutation, with a hard guard against the vacuous no-op:

```python
# src/nulls.py, chain_stratified_permutation_null()
n_mixed = sum(1 for idxs in chain_to_idx.values()
              if np.unique(labels[idxs]).size > 1)
if n_mixed == 0:
    raise ValueError(
        "chain_strat_perm: within-chain permutation is a NO-OP — no chain "
        "contains more than one distinct label. This usually means proxy "
        "chain ids (one pseudo-chain per behaviour). Fix the chain-id "
        "provenance ...")
```

p-values use Phipson–Smyth smoothing `(1+count)/(1+B)` so that `p=0` is impossible and a non-finite real statistic returns NaN rather than the maximally-significant floor. This is genuinely defensible practice (an unsmoothed 0 at B=100 cannot legitimately clear a Bonferroni threshold of 4.5e-4).

The Nova-Pro per-layer null hierarchy (`results/geometric/R1-1.5B__nova-pro/summary_layer27.md`) shows the specificity test is genuinely *mixed* even when re-annotated: chain-strat p for the top-10 variance ratio is significant for backtracking (0.0050) and example-testing (0.0050) but p=1.0000 for uncertainty-estimation and adding-knowledge. This mirrors the original single-annotator finding (specificity 2/4, add-knowledge fails everywhere) — so the *subspace exists and replicates*, but the *behaviour-specificity test does not pass for all behaviours* under any annotator.

The genuine two-sample bootstrap lives in `src/cbs/comparison.py::cross_model_compare` and is exercised by `tests/test_cross_model_bootstrap.py`. The important fix it encodes (AUDIT §5 #15, CF-12) is that the old "bootstrap" was a Gaussian reconstructed from CI width; the real version resamples each model's persisted effect-size distribution independently:

```python
# src/cbs/comparison.py, cross_model_compare()
if boots_r1 and boots_bs:
    a = np.asarray(boots_r1, dtype=float); b = np.asarray(boots_bs, dtype=float)
    ia = rng.integers(0, a.shape[0], size=n_bootstrap)
    ib = rng.integers(0, b.shape[0], size=n_bootstrap)
    diff = a[ia] - b[ib]
    p = float(min(1.0, 2.0 * min((diff <= 0).mean(), (diff >= 0).mean())))
    method = "two_sample_bootstrap"
elif all(np.isfinite(x) for x in ci_r1 + ci_bs) and delta != 0:
    ...
    method = "normal_approx_from_ci"
```

The tests lock in that the bootstrap path is taken when both arrays are present (`test_bootstrap_used_and_separated_gives_small_p`), that it falls back to the labelled normal approximation otherwise (`test_falls_back_to_normal_approx_without_boots`), and that it is deterministic given a seed. `tests/test_bootstrap_ci.py` separately locks in the point-subsample-bootstrap fix (AUDIT #16, CF-9): CIs are recomputed by resampling *points* (subsample without replacement, to avoid duplicate-point kNN degeneracy), not derived per-pair quantities — fixing the absurdly tight `[0.575, 0.587]`-style bands.

These are good tests, but note what they test: **the plumbing, on synthetic `np.linspace` inputs.** They prove the bootstrap math is correct and deterministic. They do not (cannot) prove any real cross-model finding, because no real `effect_size_boots` arrays for a baseline model have ever been produced.

---

### F.4 The cross-*model* arm is UNRUN

`13_baseline_replication.py` is fully written but its prerequisite chain — Extension A on Qwen-2.5-Math-1.5B (Phase 2b/3/4 + M1–M4 reruns) — has never run. The script is explicitly build-safe: it emits a `blocked` JSON and returns 0 rather than failing.

```python
# 13_baseline_replication.py, main()
blockers = _check_prerequisites(args)
if blockers:
    out = {"status": "blocked", "blockers": blockers, "synthesis_reference": "§M6.2", ...}
    (args.out_dir / "cross_model_blocked.json").write_text(json.dumps(out, indent=2))
    return 0
```

I confirmed `results/cbs/cross_model/` **does not exist on disk** — not even the blocked stub has been written. The distillation-vs-reveals test ("does distillation *create* the geometry or *reveal* a pre-existing one") that this script exists to run is entirely outstanding. Anything in the thesis framed as cross-model replication is, at the time of writing, the *annotator* replication (R2.2) only.

---

### F.5 Critique — where this weakens a thesis claim

**1. Annotator-replication is not the same as geometry-replication of an independent ground truth.** All three annotators are LLMs labelling the *same* fixed CoT text, and the geometry is computed on the *same* fixed R1-1.5B activations. If all three judges share a systematic bias — e.g. all of them over-segment on the same surface cue tokens ("wait", "actually", "let me check") — then three "independent" annotators can converge on the same *wrong* spans, and the geometry will replicate for a reason that has nothing to do with the behaviour being real. κ=0.35–0.44 bounds *idiosyncratic* disagreement; it says nothing about *shared* LLM-annotator bias. The replication is necessary but not sufficient for "the behaviour is a genuine geometric object." This is the sharpest residual circularity: the dependent variable is still entirely LLM-generated, and the three generators are architecturally similar enough to share priors.

**2. The dedup asymmetry is a real confound in the cross-annotator comparison itself.** The Sonnet run was on freshly re-extracted, occurrence-aware activations (`dup ≈ 1%`, N_raw = 10267/16728/5027/5829). The Qwen3 and Nova runs were on the *old first-occurrence* activations needing heavy analysis-side dedup (`dup 28–58%`, e.g. Nova backtracking N_raw=13143 → N_unique=5519). So the three arms are not on equal footing: Sonnet's `full` matrix is genuinely ~2× larger and duplicate-free, while Qwen3/Nova's `full` has been salvaged by removing up to 58% of rows. The summary md files themselves print a boilerplate "35–56% exact-duplicate ... removed" line *even on the Sonnet 1%-dup run*, which is simply wrong copy. The absolute cdim values being "modestly shifted" (Sonnet ~0.5–0.8 dim lower than Qwen3/Nova on three of four behaviours) is plausibly *just* this dedup/N difference, not a real annotator effect — which means one cannot read the cross-annotator cdim *spread* as a robustness margin. The replication claim survives because the *ordering and keystone-pass* are robust, but any quantitative cross-annotator delta is confounded by extraction vintage.

**3. The specificity null is still single-annotator at the test level.** R2.2 establishes that the *subspace* (low-dim, the keystone) replicates. It does **not** establish that the *behaviour-specificity* variance-ratio null replicates — RESULTS_LEDGER and CONFOUNDS both flag "variance-ratio specificity-null replication still not formally compared." The Nova per-layer null (F.3) in fact shows the same 2/4 pass pattern, and adding-knowledge fails the specificity null everywhere. So the honest scope is: *the geometry's existence and low dimensionality replicate across annotators; its behaviour-specificity does not, and was never null-tested per-annotator in a matched way.* A thesis sentence that says "behaviour-specific geometry replicates across annotators" would be overclaiming; "the per-behaviour subspace replicates" is what is supported.

**4. The `manifold_replication` table is empty.** `cross_annotator_comparison.json` has `"manifold_replication": {"Sonnet-4.5": {}, "Qwen3-235B": {}, "Nova-Pro": {}}` and the corresponding md table is all `—`. The 3-way numbers exist (in the three per-annotator `geometry_robustness.json` files) but were never joined into the single comparison artefact the runner was designed to emit (RESULTS_LEDGER §63 lists this as an open follow-up: "run `compare_annotators.py` to fill the table"). The headline is currently assembled by hand from three separate files, which is fragile and un-auditable.

**5. Nova-Pro is the weak arm.** ~290/1000 chains were partial-parse failures, and the CF-18 unknown→deduction coercion likely inflated Nova's deduction share to 54.1%. A replication that leans on a parse-degraded third annotator to claim "3-way" is weaker than the headline "3-way" implies; it is really "2 clean + 1 degraded."

**6. Provenance gaps.** All three robustness `provenance.json` carry `git_commit: null` (produced on the no-git cluster copy). The exact code state that produced the citable numbers is not pinned to a commit, only to an input SHA and seed. For a thesis result this is a traceability weakness.

---

### F.6 Connection to the steering decision

This section does **not** select a steering layer — layer choice is owned by the de-confounded layer sweep (`07d`/`src/layer_sweep.py`), which lands on per-behaviour mid-layers (backtracking L11, uncertainty L16, example-testing L19, adding-knowledge L16 ≈ Venhoff 15–18) plus the L27 last-layer artefact, with the final mid-vs-late call folded into Phase 7. R2.2's relevance to the impending spend is indirect but real:

- **It de-risks the dependent variable.** Steering is only worth running if the behaviour labels mean something. R2.2 is the strongest available evidence that the LLM-annotator labels index a stable geometric object rather than one judge's idiosyncrasy. Without R2.2, a steering result could always be dismissed as steering toward "whatever Sonnet happened to call backtracking."
- **It bounds the steering claim.** Because the specificity null is *not* replicated per-annotator and **adding-knowledge fails specificity everywhere**, the steering experiment should treat adding-knowledge with caution (the project already flags "add-knowledge in headline?" as an open Phase-7 decision). The four behaviours are not equally well-founded; the replication says backtracking/example-testing are the firmer ground.
- **The residual circularity (critique #1) transfers directly to steering.** If the steering experiment is *annotated by the same family of LLM judges*, it inherits the same shared-bias risk. The note on de-circularising via a `--annotator-model` (Qwen3) for the steering annotation is the right instinct and is motivated precisely by what R2.2 can and cannot rule out.
- **The cross-*model* arm (M6) remaining unrun** means the steering result cannot lean on any distillation-vs-reveals framing; that test is still owed and is independent of the steering spend.
