## J. CBS / Trajectory Line (Scripts 08–13)

### J.0 What this line is, and its status in one paragraph

Scripts `08`–`13` plus `src/cbs/*` implement the **Content-Beyond-Source (CBS)** / knowledge-creation sub-line: a self-contained six-milestone mini-pipeline (M1–M6) that asks whether *how far a reasoning sentence reaches beyond the task's home knowledge domain* leaves a measurable geometric fingerprint in the residual stream, and whether that fingerprint is **trajectory-level** (the chain-as-curve) rather than per-point. It is the engineering realisation of the "knowledge-creation extension" (Erdős unit-distance case study; `memory/knowledge_creation_extension.md`).

**It is DEFERRED.** The locked thesis spine (`memory/thesis_unifying_theme.md`) states "knowledge-creation DEFERRED (future philosophy)". Concretely: **no CBS or trajectory results exist on disk in the live `results/` tree** — `results/cbs/` and `results/trajectory/` are empty; the only artefacts (`anchor_candidates.csv`, smoke `geometry_results.json`, completion reports) live under the quarantine `results/_STALE_pre_fix_20260605/cbs/`. The two data files every runner keys on — `data/chains_cbs_annotated_R1-1.5B.json` and `data/chains_R1-1.5B_multiseed.json` — **do not exist**. The whole line is **built-and-tested but UNRUN**; nothing here is citable. It is *not* dead code in the rot sense (98 passing unit tests, clean imports, AUDIT-§5 fixes applied), but it is **parked code** — fully scaffolded, gated behind a human anchor-curation halt (P0.2) and Phase-7 answer-checker labels that were never produced for this branch.

The line matters to the imminent steering decision in exactly one narrow, important way: **`src/cbs/ablation.py` is the only place in the repo that already defines a projection-style *ablation* steering model (`CBSAblationModel`) and a *hard fail-stop validation protocol* for a difference-of-means vector** (cosine-vs-confound + leak-free CV probe). That validation discipline is directly portable to the main Phase-7 behaviour-vector experiment even though the CBS *science* is parked. See §J.8.

---

### J.1 The synthesis-plan milestone map

The numbered runners are thin orchestration over `src/cbs/` library code; each maps to a milestone in the (now-absent) `empirical_plan_synthesis.md`:

| Runner | Milestone | Library module | What it does |
|---|---|---|---|
| `08_annotate_cbs.py` | M1 | `cbs/annotation.py`* | LLM-annotate each `adding-knowledge`/`deduction` sentence with a 3-tier CBS label + binary cross-domain flag |
| `09_cbs_geometry.py` | M2 | `cbs/geometry.py` | Per-sentence geometric tests of tier/cross-domain vs 4 geometric statistics, per layer |
| `10_trajectory_build.py` | M3 | `cbs/trajectory.py` | Build chain-as-curve: arc length, Frenet curvature, subspace-visit dynamics, cone angle |
| `11_trajectory_analysis.py` | M4 | `cbs/trajectory.py`, `cbs/matching.py` | Group comparisons + matched-pair success/failure + verification-gradient probe |
| `12_cbs_ablation.py` | M5 | `cbs/ablation.py` | Causal: build & validate `v_CBS`, ablate it, measure selective effect |
| `13_baseline_replication.py` | M6 | `cbs/comparison.py` | Cross-model R1-Distill vs Qwen-Math-1.5B contrast |
| (P0.3) | scaffold | `cbs/schemas.py`, `cbs/__init__.py` | locked dataclasses/TypedDicts |
| (P0.4) | cohort | `cbs/cohort.py` | the `truncated:bool` stratification flag |

*`cbs/annotation.py` is imported by `08` but was **not in this assignment** — it exists (23 KB) and holds `ProxyClient`, `annotate_sentence_cbs`, `cohen_kappa_three_tier`, `PLACEHOLDER_ANCHOR_BLOCK`, `build_anchor_candidates_csv`.

The intellectual core is the **CBS tier ordering** (`schemas.py`):

```python
# src/cbs/schemas.py — CBSResult docstring
    tier             : 1 retrieval | 2 recombination | 3 novel application
    knowledge_domain : one of TASK_DOMAINS
    cross_domain     : True iff knowledge_domain != task home domain
```

Tier 1 = retrieving a fact native to the task domain; tier 2 = recombining; **tier 3 = applying knowledge from a *different* domain** (the "bridge", the geometric event of interest). The hypothesis is that tier-3 sentences sit measurably further out / higher-dimensional / off the union-of-behaviour-subspaces than tier-1, and that chains containing them have distinctive *trajectories*.

---

### J.2 M1 — Annotation (`08_annotate_cbs.py`): the human halt that parks everything

`08` runs the LLM annotator over Phase-3 chains and writes `cbs_*` fields. Its design is dominated by a **pilot-gate-then-lock** workflow that is the proximate reason the line is parked: you cannot do the real annotation until a human (Tony) hand-curates an anchor block, and that curation step was never completed for this branch.

The gate criteria are hard-coded from §P0.2:

```python
# 08_annotate_cbs.py, module level
PILOT_KAPPA_FLOOR = 0.5
PILOT_TIER3_RATE_FLOOR = 0.05
```

The pilot (`run_pilot`) runs the annotator **twice with two seeds** over a category-stratified 100-sentence sample, computes a three-tier Cohen's κ over the *intersection* of sentences both seeds tiered, and writes a `FAILSTOP_M1.md` if either κ < 0.5 or the tier-3 rate < 5%:

```python
# 08_annotate_cbs.py, run_pilot()
passes_kappa = kappa >= PILOT_KAPPA_FLOOR
passes_tier3 = max(tier3_rate1, tier3_rate2) >= PILOT_TIER3_RATE_FLOOR
...
if not report["passes_all"]:
    failstop_path = out_dir / "FAILSTOP_M1.md"
    failstop_path.write_text(_failstop_template(report))
    logger.error("pilot FAILED - wrote %s", failstop_path)
    return 1
```

**Why this design (defending the authors).** CBS tiers are a subtle, subjective judgement ("is this a genuine cross-domain bridge or just recombination?"). Two real risks are being pre-empted: (1) an LLM annotator that is internally inconsistent (low κ across seeds) would make every downstream geometric test meaningless, and (2) a tier-3 rate near zero would leave no positive class to build `v_CBS` from. Gating *before* spending on the full corpus, and gating *before* the human curates anchors, is the right cost-ordering. The `--build-anchors` mode emits `anchor_candidates.csv` for a human to pick 15 anchors (5/tier) that then get locked into a text block via `--anchor-block-path` — the lock makes the prompt reproducible and is the single source of inter-run determinism in the annotation.

**Critique.**
- **The halt is the deferral.** `run_build_anchors` ends by literally printing "NEXT STEP (human task): Tony picks 15 anchors…". That human step never ran on this branch (the only `anchor_candidates.csv` is in quarantine, dated pre-2026-06-05), so M1 never produced `data/chains_cbs_annotated_R1-1.5B.json`, so **every** downstream runner (09–13) falls back to synthetic/blocked paths.
- **κ ≥ 0.5 is a weak floor.** "Moderate" agreement (Landis-Koch) is being treated as a pass for a label that the *entire* causal chain (M5 `v_CBS`) depends on. And κ is computed seed-vs-seed of *the same model* — this measures the annotator's *stochastic* self-consistency, **not** validity against a human. The 50-sentence `pilot_for_human_review.csv` with an empty `human_label` column is the intended human check, but there is no code that ingests it back or gates on human-vs-model agreement. Annotator circularity (the same LLM family judging reasoning it may have produced) is unaddressed here.
- **Single annotator.** This mirrors the main-spine "specificity null still single-annotator" caveat — CBS would inherit the same weakness.

---

### J.3 M2 — Per-sentence geometry (`09_cbs_geometry.py` + `cbs/geometry.py`)

For each `(layer × behaviour)` cell, `09` computes three per-row statistics — **centroid distance**, **out-of-subspace residual** against the union of behaviour subspaces, and **local intrinsic dimension** (Levina-Bickel MLE) — plus pairwise **principal angles** between behaviour subspaces. Each statistic is tested twice: **Jonckheere-Terpstra** ordinal-trend under the 3-tier label and **Mann-Whitney/Cliff's δ** under the binary cross-domain label, with **Holm** correction across all `(layer × behaviour × statistic × label)` cells, and optional shuffle/reversal controls.

The geometry library is genuinely careful. Two examples of the method:

```python
# src/cbs/geometry.py, out_of_subspace_residual()
    coef = X @ V                       # (N, k)
    proj = coef @ V.T                  # (N, d)
    residual = X - proj
    res_norm = np.linalg.norm(residual, axis=1)
    x_norm = np.linalg.norm(X, axis=1)
    safe = np.where(x_norm > 0, x_norm, 1.0)
    out = res_norm / safe
    out[x_norm == 0] = 0.0
    return np.clip(out, 0.0, 1.0)
```

The JT variance is the no-ties form, with an explicit honesty warning when the input is tied:

```python
# src/cbs/geometry.py, jonckheere_terpstra()
    if n > 1 and n_ties / (n - 1) > 0.05:
        logger.warning("jonckheere_terpstra: %.0f%% of values are tied; the "
                       "no-ties variance is used (conservative) so the p-value "
                       "is approximate.", 100.0 * n_ties / (n - 1))
    var_jt = (n * n * (2 * n + 3) - float(np.sum(ni * ni * (2 * ni + 3)))) / 72.0
```

**The single most important fact about M2's run-state**: the labels are **synthetic unless a real CBS annotation file exists**. The honest-labelling fix (AUDIT §5) means `09` will *not* stamp `labels_source="real"` on RNG tiers — it falls back and says so:

```python
# 09_cbs_geometry.py, main()
    if real_labels is None and not args.synthetic_tiers:
        logger.warning("no usable CBS annotations at %s; falling back to "
                       "--synthetic-tiers", args.cbs_annotations)
        args.synthetic_tiers = True
...
        "note": ("smoke-only, not paper-grade" if args.synthetic_tiers
                 else "labels from CBS annotations"),
```

Since `data/chains_cbs_annotated_R1-1.5B.json` does not exist, **any M2 output produced today is synthetic-tier smoke** — random uniform tiers in `{1,2,3}`. The quarantined `R1-1.5B-smoke/geometry_results.json` is exactly such a run and is correctly labelled smoke.

**Critique.**
- **`local_intrinsic_dim`'s default is now MLE, but the legacy per-row TwoNN is documented as "severely upward-biased (~9–13 for true dim 3)" and still callable.** The codebase audit flagged this as a real scientific bug; the fix added the MLE estimator and a warning but kept the broken one for "backward compatibility". Any stale TwoNN-based ID result is meaningless.
- **`build_union_basis` calibration caveat.** Without `per_behaviour_weights` the variance threshold is a cut on *direction-spectral-energy*, **not** activation variance — the docstring says so verbatim. `09`'s `_build_behaviour_pcs` passes **no weights**, so the union basis the OOS-residual is measured against is the un-calibrated version. This silently changes what "covers 95% of variance" means.
- **Union basis leaks the target.** The union basis at each layer is built from PCs of *all* behaviours including `adding-knowledge` — the same behaviour whose sentences are being scored for OOS residual. Projecting `adding-knowledge` activations out of a subspace that was partly fit on `adding-knowledge` activations will mechanically shrink the residual; this is a circularity the runner does not guard against.
- This whole stage is a **per-point** test. The deferral is partly intellectual: the project's own commitment (exploration §2.5, encoded in `trajectory.py`) is that the trajectory, not the point, is the right object — so M2 was always the weaker of the two framings.

---

### J.4 M3 — Trajectory build (`10_trajectory_build.py` + `cbs/trajectory.py`)

This is the conceptual heart of the line and the part most aligned with the thesis's "reasoning is a geometric *process*" spine. Each chain becomes a `ChainTrajectory`: an ordered `(T, d)` matrix of sentence-final-token activations at one layer, with per-sentence behaviour / tier / cross-domain labels, plus a **truncated** flag carried from the P0.4 cohort. The headline measurable is an **arc-length-reparameterised discrete Frenet curvature** — the module docstring shouts that this is the point:

```python
# src/cbs/trajectory.py, curvature_sequence()
        T_left = diffs[t - 1] / nl
        T_right = diffs[t] / nr
        ds = (nl + nr) / 2.0
        out[t] = float(np.linalg.norm(T_right - T_left) / ds)
```

The docstring (CRITICAL note) insists this is **not** `||x_{t+1} - 2x_t + x_{t-1}||` — the arc-length normalisation (`/ ds`) is what makes it a real curvature invariant rather than a finite-difference acceleration that scales with step size.

Provenance is reconstructed deterministically: `build_row_index` rebuilds the exact `(chain_id, span_idx) → row` mapping that Phase-4 activation extraction used (source-JSON order, then span order, only the four `PHASE_4_BEHAVIOURS`). This is the load-bearing assumption — if the activation files were written in any other order, every trajectory is silently mis-assembled.

```python
# src/cbs/trajectory.py — the four behaviours with saved activations
PHASE_4_BEHAVIOURS: tuple[str, ...] = (
    "backtracking", "uncertainty-estimation",
    "example-testing", "adding-knowledge",
)
```

The runner also computes subspace-visit dynamics (which behaviour-subspace each point projects onto most, transition counts, return rate) and a `trajectory_cone_angle`.

**Critique.**
- **Sparse, gappy trajectories.** Only the four Phase-4 behaviours have saved activations. `deduction` — one of the two behaviours M1 actually annotates for CBS tier! — is **not** in `PHASE_4_BEHAVIOURS`, so deduction sentences are *dropped* from trajectories. The "trajectory" is therefore a subsequence keeping ~4 behaviour types and discarding `initializing`, `deduction`, plain reasoning, etc. Arc length and curvature are computed over **non-adjacent** sentences (gaps of arbitrary size in the real chain), which makes the discrete-Frenet step `ds` semantically inconsistent point-to-point. The "curve" is not the model's actual reasoning path; it's a decimated shadow of it. This is a deep confound the docstring waves at ("possibly-sparse sub-trajectory; arc length and curvature still make sense") but does not resolve.
- **Curvature ≈ the main spine's keystone negative.** The standing thesis result (`memory/confounds_remediation_plan.md`) is that **curvature is a clean NEGATIVE — a chain artefact (CF-2)**. CBS curvature is the same quantity on the same activations; there is strong prior reason to expect it carries the same truncation/length confound rather than CBS signal. M3 partially anticipates this by carrying `truncated` and residualising on `T` downstream — but with 50% of the corpus truncated mid-`</think>` (see §J.7), curvature is fighting a huge nuisance.
- **Fallback to raw chains.** `10` falls back to `data/annotated_R1-1.5B.json` when CBS annotations are absent — so it *can* run today, but every `cbs_tier` is 0 and `n_tier3_sentences` is 0, making the CBS-specific group comparisons (§J.5) vacuous. The runner's own `run_metadata.json` note says "smoke-only, not paper-grade — labels from synthetic / Phase 3 only, CBS fields absent until P0.2 lock."

---

### J.5 M4 — Trajectory analysis (`11_trajectory_analysis.py` + `cbs/matching.py`)

`11` does three things. (1) **Group comparisons** on the per-chain summary parquet — `truncated` vs not, high-CBS (≥2 tier-3) vs low-CBS, and a long-vs-short positional control — each residualised on chain length `T` via OLS before a Mann-Whitney + Cliff's δ + bootstrap CI (`compare_groups` in `trajectory.py`). (2) **Matched-pair** tier-3 success-vs-failure analysis (`build_matched_pairs`: Jaccard token similarity ≥ 0.6 on the *same task*). (3) A **verification-gradient** 5-fold CV probe.

Crucially, **(2) and (3) are hard-blocked in code** — they write a `status: "blocked"` stub rather than run, because they need labels that don't exist:

```python
# 11_trajectory_analysis.py, main()
    if (args.skip_matched_pair or not args.multi_seed_chains.exists()
            or not args.cbs_annotations.exists()):
        matched_pair_path.write_text(json.dumps({
            "status": "blocked",
            "blockers": [
                ...
                "Phase 7 answer-checker labels (success/failure) required.",
            ],
```

So M4's *causal-flavoured* halves are gated on (a) the CBS annotation file, (b) a multi-seed re-generation (`chains_R1-1.5B_multiseed.json`, absent), and (c) **Phase-7 answer-checker** correctness labels — the same answer-checker that the main steering line also needs. Only the group-comparison + UMAP smoke path runs.

The one piece of M4 that is *better* than the main line is the **leakage-aware CV probe** (`matching.cv_probe`), a fix from the codebase audit:

```python
# src/cbs/matching.py, cv_probe()
    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y, groups)
    else:
        ...
        logger.warning(
            "cv_probe running WITHOUT chain groups: sentences from the same "
            "chain may leak across CV folds and inflate accuracy. ..."
        )
```

`StratifiedGroupKFold` on `chain_id` prevents same-chain sentences from straddling train/test — a real hazard because tier-3 / success sentences cluster within chains. The original code used plain `StratifiedKFold` and inflated the very accuracy that gates M5.

**Critique.**
- **The interesting results are exactly the blocked ones.** Group comparisons on synthetic/Phase-3-only data are uninformative; the matched-pair and verification-gradient analyses — the ones that would actually test "does crossing a knowledge bridge predict correctness" — never run. The line's scientific payload is entirely downstream of the Phase-7 answer-checker, which on this branch is also unrun.
- **`compare_groups` residualises on `T` but matched pairs do not control for position.** Tier-3 sentences may systematically occur late in chains; Jaccard matching on token overlap controls content, not position-in-trajectory.

---

### J.6 M5 — Causal ablation (`12_cbs_ablation.py` + `cbs/ablation.py`): the part that touches the steering decision

This is the most decision-relevant stage. `v_CBS = normalise(mean(tier3) − mean(tier1))` at a steering layer, then a **projection ablation** `h' = h − α·(v_cbs·h)·v_cbs`, run across `α ∈ {0, 0.5, 1.0, 2.0}` and conditions `{baseline, v_cbs, v_random, v_adding_knowledge}` on textbook-solvable vs bridge-required task sets. The intervention loop itself is explicitly left to run-phase (~25 h cluster GPU); the runner stops at validation.

The **hard fail-stop** is the transferable jewel:

```python
# src/cbs/ablation.py — module constants
FAILSTOP_COS_MAX = 0.5
FAILSTOP_PROBE_ACC_MIN = 0.7
FAILSTOP_PROBE_STD_MAX = 0.15
```

`validate_v_cbs` requires **all three**: `|cos(v_cbs, v_adding_knowledge_centroid)| < 0.5` (the CBS vector must not just be the generic "adding-knowledge" direction in disguise — a de-confounding constraint), a leak-free CV probe mean accuracy ≥ 0.7 (the tier-3/tier-1 split must be linearly real), and probe std ≤ 0.15 (stable across folds). If any fail, `12` writes `FAILSTOP_M5.md` and **halts** with three concrete options (re-curate anchors / switch layer / drop M5 and report the null).

The ablation model is a thin, disciplined subclass of the main steering infrastructure:

```python
# src/cbs/ablation.py, CBSAblationModel.__init__
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(
                f"v_cbs must be unit-norm for projection ablation; "
                f"got ||v|| = {norm:.4f}. Call build_v_cbs(...) first.")
        super().__init__(model, tokenizer, vector=v, layer=int(layer),
                         alpha=float(alpha), mode="subtract")
```

It extends `src/steered_inference.py::SteeredModel` with `mode="subtract"` — i.e. **CBS reuses the exact same steering hook machinery as the main Phase-7 line.** The default steering layer is **27** (`--steering-layer`, "17 also supported"), and the fail-stop options explicitly suggest "try layer 17 if 27 failed, or vice versa; subspace correlations differ across depth."

**Critique.**
- **Everything is gated, twice.** `_load_tier_acts` needs the CBS annotation file (absent) and the validation aborts to a `*_blocked.json` if it finds < 5 tier-3 or < 5 tier-1 activations. `construct_task_sets` needs an `answer_correct` field (from Phase-7, absent) and raises if either cohort has < 50 candidates. So M5 cannot even reach validation on the current branch.
- **The de-confound is one-sided.** The fail-stop checks `v_cbs` is not collinear with the `adding-knowledge` centroid, but the *positive* class is built from `adding-knowledge` and `deduction` sentences only. If tier-3 simply correlates with "harder problem," `v_cbs` could encode difficulty, not bridging — there is no difficulty-matched negative control in the validation.
- **`v_random` and `v_adding_knowledge` are the right control arms** (energy/identity controls) and are notably more rigorous than a naïve baseline — this is worth importing wholesale.

---

### J.7 P0.4 cohort (`cbs/cohort.py`) — the truncation confound, made explicit

The CBS line is the place where the project first wrote down the truncation problem as a first-class stratifier. `is_truncated` is the single source of truth:

```python
# src/cbs/cohort.py, is_truncated()
    n_tokens = int(chain.get("n_tokens", 0))
    chain_text = chain.get("chain", "") or ""
    return n_tokens >= max_new_tokens and not chain_text.rstrip().endswith("</think>")
```

The decision template (quarantined `truncation_policy_decision.md`) records the severity: **502/1000 chains (50.2%) hit the 8192-token cap; 499 truncated mid-`</think>`**, ranging from 95% (lateral_thinking) to 4% (systems_thinking). Tony's recorded decision was **(b) stratify by `truncated:bool`** — which is why every M2/M3/M4 runner carries the flag and uses it as the primary group split.

**Critique.** A corpus where half the chains are truncated mid-thought is a brutal substrate for *any* trajectory geometry: the last activation of a truncated chain is not a reasoning endpoint, it's a guillotine. Stratifying is the honest minimum, but with the truncation rate so category-correlated (lateral_thinking 95% vs systems_thinking 4%), `truncated` is badly confounded with `category` and with task difficulty — residualising on `T` alone does not remove it. This is independent corroboration of why curvature is treated as a chain artefact on the main spine.

---

### J.8 M6 cross-model (`13_baseline_replication.py` + `cbs/comparison.py`)

`13` is pure orchestration: it reads the two models' `geometry_results.json`, runs `cross_model_compare`, and — if the baseline tier-3 rate is below 3% — writes a **structured-null** report instead of forcing parallel tests, framing "distillation adds tier-3 capacity" as the *finding*. The notable engineering fix here is that `cross_model_compare` now does a **genuine two-sample bootstrap** (resampling each model's persisted `effect_size_boots` arrays) rather than the audit-flagged Gaussian-reconstructed-from-CI fake bootstrap:

```python
# src/cbs/comparison.py, cross_model_compare()
        if boots_r1 and boots_bs:
            a = np.asarray(boots_r1, dtype=float)
            b = np.asarray(boots_bs, dtype=float)
            ia = rng.integers(0, a.shape[0], size=n_bootstrap)
            ib = rng.integers(0, b.shape[0], size=n_bootstrap)
            diff = a[ia] - b[ib]
            p = float(min(1.0, 2.0 * min((diff <= 0).mean(), (diff >= 0).mean())))
            method = "two_sample_bootstrap"
```

`13` is fully blocked: it requires the baseline (Qwen-Math-1.5B) to have gone through the entire Extension-A pipeline *and* M1–M4, none of which ran. It writes `cross_model_blocked.json` and returns 0.

---

### J.9 Why it was set aside — and what value it retains

**Why deferred.** Three converging reasons:
1. **Thesis-spine decision.** The locked spine cut knowledge-creation to "future philosophy" (`memory/thesis_unifying_theme.md`). The CBS science answers a *different* question (cross-domain knowledge geometry) than the two committed Movements (per-behaviour structure; post-training origin of safety).
2. **A human halt that never cleared.** M1's P0.2 anchor curation is a manual step that was never done on this branch; without it the whole pipeline runs only on synthetic tiers.
3. **Dependency on the unrun answer-checker.** The genuinely causal stages (M4 matched-pair, M4 verification-gradient, M5 task sets, M6 trajectory-Wasserstein) all need Phase-7 success/failure labels — the *same* unrun Phase-7 the researcher is about to fund. CBS sits *behind* the steering decision in the dependency graph, not beside it.

**Value retained (not dead code).**
- **The validation discipline is directly portable to Phase-7.** `validate_v_cbs`'s three-condition hard fail-stop (confound-cosine + leak-free grouped-CV probe + fold-stability) is a better gate than anything in the main steering line, and `cv_probe`'s `StratifiedGroupKFold`-on-`chain_id` is the correct fix for the sentence-leakage hazard that the main line shares.
- **`CBSAblationModel` proves the projection-ablation mode** (`h − α(v·h)v`) works on the shared `SteeredModel` hook — an alternative/complement to additive steering that the Phase-7 design can adopt for a "remove the behaviour" arm.
- **The control-arm design** (`v_random`, `v_adding_knowledge` as identity/energy controls; textbook-vs-bridge task split) is a template for a clean selectivity measurement.
- **`cohort.is_truncated` + the documented 50% truncation finding** are reusable, model-agnostic facts that the steering cohort must also respect.
- The geometry library (`jonckheere_terpstra`, `holm_correction`, `bootstrap_ci` with `return_dist`, MLE `local_intrinsic_dim`) is tested, audited, and reusable outside CBS.

---

### J.10 Connection to the imminent steering decision

The CBS *science* does **not** feed the layer/methodology choice (it's parked, unrun, uncitable). But the CBS *engineering* offers three concrete, low-cost borrows for Phase-7:

1. **Layer.** CBS independently defaulted to **layer 27** for `v_CBS` with **17** as the documented alternate and an explicit "subspace correlations differ across depth; switch if it fails" protocol. This is consistent with — and an extra data-point for — the steering-status note's "build mid(11/16/19) + L27 and let Phase 7 decide." CBS adds no *new* evidence for which layer, but it shows the team already treats {17, 27} as the candidate pair for difference-of-means steering on this model.
2. **Vector-validation gate.** Adopt `validate_v_cbs`'s three-condition fail-stop verbatim for the Phase-7 behaviour vectors: (a) cosine vs the most-confoundable neighbour direction < 0.5, (b) grouped-CV probe accuracy ≥ 0.7, (c) fold-std ≤ 0.15. This catches "your steering vector is just the generic direction" *before* you spend GPU on generation.
3. **Leak-free probing.** Use `cv_probe(..., groups=chain_ids)` everywhere a probe accuracy gates a decision — the main line must not repeat the plain-`StratifiedKFold` leakage the CBS audit already fixed.

**Bottom line for the reader:** treat Scripts 08–13 as a *methods reservoir*, not a results source. Nothing here is citable; the truncation cohort, the grouped-CV probe, the projection-ablation model, and the three-part vector fail-stop are the four things worth lifting into the steering experiment before you spend money.
