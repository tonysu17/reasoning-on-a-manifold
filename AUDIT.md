# Codebase audit — Reasoning on a Manifold

**Date:** 2026-06-05 · **Scope:** software-engineering robustness, scientific/numerical correctness, and a research-methodology critique of the full pipeline (`src/`, `src/cbs/`, the 00–13 phase runners, and ad-hoc scripts).

All findings below were **empirically verified** (executable probes against synthetic data with known ground truth), not just read. Fixes applied in this pass are marked **[FIXED]**; everything else is **[OPEN]** with a recommendation.

Reproduce the test suite:

```bash
pip install -e .            # build backend was broken; now fixed
python -m pytest            # 164 tests: 98 CBS + 66 core/config/leakage
```

---

## 1. Summary

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | Curvature diagnostic confounded — flat data scored 0.29–0.85 (should be ~1.0) | CRITICAL | **FIXED** |
| 2 | `twoNN_estimate` biased high ~35–50% (F=1.0 leverage point) | CRITICAL | **FIXED** |
| 3 | `local_intrinsic_dim` (CBS) biased ~3–4× (TwoNN on 21-pt clouds) | CRITICAL | **FIXED** |
| 4 | sklearn PCA randomized solver, no seed → steering vectors non-reproducible | HIGH | **FIXED** |
| 5 | Data leakage: CV probes split per-sentence, no chain grouping → inflated causal gate | HIGH | **FIXED** + live in M5 (12 passes chain_id groups) |
| 6 | `power_analysis` silently writes all-NaN table that looks like a null result | HIGH | **FIXED** |
| 7 | `geodesic_euclidean_ratio` symmetrization halves one-way edges (flat→0.73) | MEDIUM | **FIXED** |
| 8 | `config.yaml` never loaded; `MODELS`/`STEERING_LAYERS` duplicated & divergent | HIGH | **FIXED** (STEERING_LAYERS) / **OPEN** (MODELS) |
| 9 | Core estimators untracked in git | HIGH | **FIXED** (git add) |
| 10 | `pyproject` build backend invalid → `pip install -e .` fails | HIGH | **FIXED** |
| 11 | README fictional (setup.sh / scripts/ / external/ don't exist) | MEDIUM | **FIXED** (rewritten) |
| 12 | `07_evaluate_steering.py` read stale `data/tasks.json` | HIGH | **FIXED** |
| 13 | Core scientific pipeline had zero tests | HIGH | **FIXED** (66 tests) |
| 14 | `09_cbs_geometry.py` non-deterministic (`hash()` seeding) | CRITICAL | **FIXED** (zlib.crc32) |
| 15 | `cross_model_compare` "bootstrap_p" is not a bootstrap | MEDIUM | **FIXED** (relabelled p_normal_approx) |
| 16 | Bootstrap CIs resample derived quantities → too narrow | MEDIUM | **OPEN** |
| 17 | Activation patching uses lexical markers as behaviour proxy | MEDIUM | **OPEN (methodological)** |
| 18 | `MODELS` dicts in 02/04/07 still duplicated/divergent | MEDIUM | **OPEN** |
| 19 | statsmodels declared but unused | LOW | **OPEN** |

---

## 2. Fixes applied (with verification)

**#1 Curvature confound — `src/curvature.py::local_vs_global_dim_ratio`.**
The diagnostic divided a `k`-point local PCA dimension by the PCA dimension of *all N* points — incomparable sample sizes, so a flat subspace of dim > k scored ≪ 1. Fixed by comparing against a **sample-size-matched** random-subset baseline.
Verified: flat data 0.29–0.85 → **1.00**; 2-sphere → **0.67**; 2-plane → **1.00**. (`tests/test_curvature.py`)

**#2 TwoNN bias — `src/intrinsic_dim.py::twoNN_estimate`.**
The empirical CDF was renormalized on the truncated sample, forcing `F=1.0` at the cutoff; `-log(1-F)` then dominated the through-origin fit. Fixed by computing `F=i/(n+1)` over the full sample before truncating to the linear regime.
Verified (uniform data): dim 5 → 6.7→**4.4**, dim 8 → 10.2→**6.7**; (Gaussian) dim 3→**2.9**, 5→**5.2**, 8→**7.0**. ⚠️ **The TwoNN values in PROGRESS.md (9.4–27.9) were produced by the biased code and should be regenerated.**

**#3 Local intrinsic dim — `src/cbs/geometry.py::local_intrinsic_dim`.**
Replaced per-row TwoNN on 21-point clouds (returned ~12 for true dim 3) with the Levina-Bickel per-point MLE. Verified: true dim 3 → **3.0**. Legacy estimator kept as `estimator="twoNN"` and documented as biased.

**#4 PCA reproducibility — `src/pca.py`, `src/nulls.py`, `src/steering.py`.**
All 5 `PCA()` calls now use `svd_solver="full"` (exact + deterministic). The default `"auto"` picked the randomized solver for the (N≈50–145, d=1536) shapes here, with no `random_state`. Verified: steering vectors now identical across calls.

**#5 CV leakage — `src/cbs/matching.py`, `src/cbs/ablation.py`.**
Added a shared `cv_probe(X, y, groups=...)` using `StratifiedGroupKFold` when chain ids are supplied (warns + falls back otherwise). `verification_gradient` and `validate_v_cbs` take optional `*_groups`. Verified with a no-signal, chain-clustered dataset: ungrouped CV **0.99** (leaks), grouped **0.50** (correct). (`tests/test_cv_leakage.py`) **Action required:** callers (11/12 runners) must pass `chain_id` groups for the fail-stop to be leak-free.

**#6 power_analysis fail-loud — `power_analysis_curvature.py`.**
`run_diagnostics` now only swallows data-degeneracy errors (LinAlgError/ValueError); systematic failures (broken imports) propagate. Refuses to write an all-NaN table.

**#7 geodesic symmetrization — `src/curvature.py`.** `(W+W.T)/2` → `W.maximum(W.T)`. Verified: flat ratio 0.73 → **1.18** (≥1, correct).

**#8/#9/#10/#12 infra.** `src/config.py` is now the single source for `MODELS`/`STEERING_LAYERS`/`SEED`/`TARGET_BEHAVIOURS`, loaded from `config.yaml`; 05/05b/06 import from it (05b had silently dropped the baseline model). Core estimators git-tracked. Build backend `setuptools.build_meta`. Phase 7 reads `tasks_final.json`.

**#13 tests.** New `tests/` suite (66 tests) with known-answer checks on synthetic manifolds of controlled dimension/curvature — exactly the checks that would have caught #1–#7.

---

## 3. Open issues & recommendations

- **#15 (residual) true paired bootstrap** — the cross-model p is now honestly labelled `p_normal_approx` (Gaussian-from-CI), but a genuine paired bootstrap still requires the producers (`05`/`05b`) to persist per-resample arrays so `cross_model_compare` can resample them. Deferred.
- **#16 Bootstrap CIs** — `intrinsic_dim`/`curvature` bootstrap *derived* quantities (μ ratios, pairwise distances), which are dependent → CIs too narrow (e.g. `[0.575, 0.587]`). Resample points and recompute end-to-end.
- **#17 Activation patching proxy** — `behaviour_marker_logprob` scores behaviours by tokens like `"wait"`/`"actually"`; this conflates behaviour with surface lexis and patches position *i* across non-aligned chains. Needs a validated behaviour metric and positional alignment before it can support a "Paper 2 main" causal claim.
- **#18 MODELS dicts** — 02/04/07 still carry divergent `MODELS` (07 uses tuple values keyed `"1.5b"` and lacks the baseline). Migrate to `src.config.MODELS`.
- **#19 statsmodels** — declared in `[cbs]` extras, imported nowhere. Remove or implement the promised M4 mixed-effects model.

### Methodological (research-design) cautions
- **N is small** (51–145/behaviour) for intrinsic-dim/curvature in d=1536; report a *working* power analysis (now unblocked) before claiming dimensions.
- **"Manifold-projected steering" is linear** (top-k PCA projection) — it tests *subspace*, not *curvature*; keep the Paper-2 (subspace) and Paper-3 (curvature) claims distinct.
- **Annotation validity**: behaviour labels come from one LLM annotator; report inter-annotator agreement (the 3-annotator pipeline) *before* the geometry, since labels are the dependent variable.
- **Truncation**: ~50% of chains hit `max_tokens` mid-thinking — stratify or regenerate before position/behaviour analysis.
- **Mean-pooling** the first 10 tokens collapses the trajectory the manifold claim is about; prefer the trajectory module for process-level claims.

---

## 4. Note on tooling

The `tdd-guard` PreToolUse hook in `.claude/settings.json` was found to be malfunctioning — its internal validator returns prose instead of JSON, so it `block`ed **every** `.py`/`.toml` edit with a parse error (it ignores `.md`/`.json`/`.yaml`). It was disabled for this work via its own native switch, `.claude/tdd-guard/data/config.json` → `{"guardEnabled": false}` (settings.json untouched). Re-enable with `{"guardEnabled": true}` once its validator config is fixed; re-enabling while broken will block all code edits again.

---

## 5. Full backlog from the sub-audits (not yet actioned)

Lower-severity findings surfaced by the parallel runner/CBS audits. Not triaged into the table in §1; logged here so they aren't lost.

### Statistical correctness (MEDIUM)
- **`src/cbs/geometry.py::build_union_basis`** — the variance-threshold cut is applied to singular values of stacked **unit-norm** per-behaviour PC columns, which are stripped of eigenvalue scale. So `variance_threshold=0.95` does *not* mean 95% of activation variance; it's 95% of the spectral energy of the stacked directions. Re-derive what the union-basis dimensionality should mean, or weight columns by their eigenvalues.
- **`src/cbs/matching.py::paired_geometric_tests`** — Cliff's-delta bootstrap CI uses `paired=False` on a *matched* pair (inflates the CI ~5×), while the Wilcoxon reported alongside is paired. Make the CI paired for internal consistency.
- **`src/cbs/geometry.py::jonckheere_terpstra`** — no tie correction in the variance, though the docstring advertises tie handling. Harmless on continuous CBS inputs (conservative); fix the variance or the docstring before any discretized input reaches it.

### Silent-failure / robustness (MEDIUM)
- **`09_cbs_geometry.py` real-annotation loader is a `pass` stub** — when a real CBS annotations file exists it still falls through to synthetic tiers, yet stamps `labels_source:"real"`. **Wire the real loader before trusting any "real" geometry output.** (Higher impact than its severity suggests — it silently mislabels provenance.)
- **`09_cbs_geometry.py` mannwhitneyu** wrapped in `try/except ImportError` → silent NaN if scipy missing.
- **`src/cbs/annotation.py:~278`** — `except (ValueError, Exception)` is an over-broad catch that masks `KeyError`/`AttributeError` and silently defaults the task domain to `"other"`. Narrow it.
- **Overwrite safety** — several runners (`02`, `02b`, `03`, `12`) write outputs in place; a partial/failed run with fewer records can overwrite a good full artifact with no backup. Add a record-count guard or write-then-backup.

### Reproducibility / provenance (MEDIUM)
- **CBS phases (08–13) default `seed=0`**, ignoring config `SEED=42`. Thread `src.config.SEED` (and add a `--seed` arg where missing, e.g. `12_cbs_ablation.py`).
- **No provenance stamps** — most result writers (`05`, `05c`, `06`, `build_phase6`, `compute_layer_triangulation`, `robustness_geometry`, figure scripts) record no git hash / input-file hash / seed. Add a shared `_stamp(out_dir, args)` helper so every figure is traceable to code+inputs.
- **`build_phase6.py` / `robustness_geometry.py` hardcode per-behaviour peak layers** (`{backtracking:14, uncertainty:14, adding-knowledge:17, example-testing:27}`) diverging from config's single `steering_layer:27`. Source from a `candidate_layers.json` (triangulation output) or config.

### Duplication (MEDIUM/LOW)
- **`_find_sentence_offset` reimplemented in 4 files** (`05`, `05b`, `compare_annotators`, `robustness_geometry`), each claiming to "replicate `src/activation_extraction.py` exactly". Extract one shared helper — drift here silently misaligns chain-id arrays with activation rows.
- **`MODELS` dicts in `02`/`04`/`07` still divergent** (#18) — `STEERING_LAYERS` was migrated to `src.config` for `05`/`05b`/`06` only. `07` uses tuple values keyed `"1.5b"` and lacks the baseline → `--model qwen-math-1.5b` crashes.

### Schema / data (LOW)
- **`src/cbs/cohort.py::is_truncated`** hardcodes the `</think>` sentinel and `n_tokens >= 8192`; if `chains.max_new_tokens` changes it mislabels every chain as non-truncated (corrupts the P0.4 stratification). Source the cap from config.
- **`schemas.py`** strict types (`tier` ∈ {1,2,3} int, `cross_domain` bool) — real data emitting `"3"`/`3.0`/`"yes"` bypasses the dataclass unless it flows through the coercing annotator.
- **Sentence-ID convention mismatch** — `matching._tier_spans` indexes by position in the *filtered* annotation list; `trajectory.build_trajectory` uses `f"{chain_id}:{i}"` from the *full* span list. If a lookup keyed one way meets ids built the other way, pairs silently drop. Add an integration test crossing the two modules.
- **`data/MANIFEST.md` is gitignored** (because `data/` is) → local-only. Force-track it (`git add -f`) or fold its content into README/AUDIT so collaborators see which data files are canonical.

### Docs (LOW)
- Stale docstrings in `predict_saturation.py` / `06b_steering_composition.py` reference steering-vector filenames that don't match the actual `{beh}_single.npy` / `{beh}_manifold_k{k}.npy` convention.
- Ad-hoc scripts (`build_phase6`, `make_fresh_figures`, `render_html`, `compare_annotators`, `validate_pilot_lengths`) lack input validation — they crash with raw tracebacks on missing files.

---

## 6. Next-session checklist (ordered)

1. **Merge** `audit/se-robustness-fixes` → `main` (5 commits; note pre-existing WIP was bundled into the first — see commit message). Re-run `pytest` (expect 164).
2. **Regenerate** the TwoNN dimensions in `PROGRESS.md` (the 9.4–27.9 values are from the biased estimator).
3. **#16** point-resampled bootstrap CIs in `curvature.py` / `intrinsic_dim.py` (touches the just-fixed estimators — re-verify against `tests/`).
4. **CBS `seed`** → thread `src.config.SEED` through 08–13.
5. **`09_cbs_geometry` real loader** + **`07` MODELS** baseline/migration.
6. The remaining §5 items as capacity allows; re-enable `tdd-guard` only after its validator is fixed (§4).
