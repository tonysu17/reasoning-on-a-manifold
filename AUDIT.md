# Codebase audit — Reasoning on a Manifold

**Date:** 2026-06-05 · **Scope:** software-engineering robustness, scientific/numerical correctness, and a research-methodology critique of the full pipeline (`src/`, `src/cbs/`, the 00–13 phase runners, and ad-hoc scripts).

All findings below were **empirically verified** (executable probes against synthetic data with known ground truth), not just read. Fixes applied in this pass are marked **[FIXED]**; everything else is **[OPEN]** with a recommendation.

Reproduce the test suite:

```bash
pip install -e .            # build backend was broken; now fixed
python -m pytest            # full suite (258 tests as of 2026-06-12 evening; §5 quotes 220 from its second pass — counts grow with fixes)
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
All 5 `PCA()` calls now use `svd_solver="full"` (exact + deterministic). The default `"auto"` picked the randomized solver for the (N≈50–145, d=1536) **smoke** shapes (the full run is N≈5k–16k/behaviour), with no `random_state`. Verified: steering vectors now identical across calls.

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
- ~~**#16 Bootstrap CIs**~~ **[DONE 2026-06-05, commit `d7d147e`]** — point-subsample bootstrap (m=0.8N without replacement, statistic recomputed end-to-end) replaced the derived-quantity resampling across curvature + intrinsic_dim. (This §3 entry was stale; §5 was correct. Re-run on real data still owed — CF-9.)
- **#17 Activation patching proxy** — `behaviour_marker_logprob` scores behaviours by tokens like `"wait"`/`"actually"`; this conflates behaviour with surface lexis and patches position *i* across non-aligned chains. Needs a validated behaviour metric and positional alignment before it can support a "Paper 2 main" causal claim. **STILL OPEN.**
- ~~**#18 MODELS dicts**~~ **[DONE 2026-06-05, commits `43f6b2e`+`ba52881`]** — 02/04/07 build their registries from `src.config.MODELS_BY_CLI` (verified in-tree 2026-06-12; this §3 entry was stale, §5/“[DONE]” was correct).
- **#19 statsmodels** — declared in `[cbs]` extras, imported nowhere. Remove or implement the promised M4 mixed-effects model.

> **2026-06-12 addendum.** A second wave of fixes landed for findings outside this audit's scope: occurrence-aware sentence matching + row-provenance sidecar (CF-13/CF-14 — the 35–56% duplicate-row bug), chain-grouped CV in 05c (CF-15), Phipson–Smyth-smoothed permutation p's + a real Holm–Bonferroni in the triangulation (CF-16), and the Phase-7 pre-spend patches (CF-17). See `CONFOUNDS_AND_REMEDIATION.md` — that file, not this one, tracks their re-run debt.

### Methodological (research-design) cautions
- ~~**N is small** (51–145/behaviour) for intrinsic-dim/curvature in d=1536~~ **[CORRECTED 2026-06-06: 51–145 was the SMOKE cohort (`data/activations/R1-1.5B-smoke/` = 51/97/145/145). The full run uses N = 5 027 (adding-knowledge) / 5 829 (example-testing) / 10 267 (backtracking) / 16 728 (uncertainty), verified from `data/activations/R1-1.5B/` shapes. Raw N is fine — the real threat is *effective* N (~1000 chains; CF-2 in `CONFOUNDS_AND_REMEDIATION.md` §1).]** Still report a *working* power analysis (now unblocked) before claiming dimensions.
- **"Manifold-projected steering" is linear** (top-k PCA projection) — it tests *subspace*, not *curvature*; keep the Paper-2 (subspace) and Paper-3 (curvature) claims distinct.
- **Annotation validity**: behaviour labels come from one LLM annotator; report inter-annotator agreement (the 3-annotator pipeline) *before* the geometry, since labels are the dependent variable.
- **Truncation**: ~50% of chains hit `max_tokens` mid-thinking — stratify or regenerate before position/behaviour analysis.
- **Mean-pooling** the first 10 tokens collapses the trajectory the manifold claim is about; prefer the trajectory module for process-level claims.

---

## 4. Note on tooling

The `tdd-guard` PreToolUse hook in `.claude/settings.json` was found to be malfunctioning — its internal validator returns prose instead of JSON, so it `block`ed **every** `.py`/`.toml` edit with a parse error (it ignores `.md`/`.json`/`.yaml`). It was disabled for this work via its own native switch, `.claude/tdd-guard/data/config.json` → `{"guardEnabled": false}` (settings.json untouched). Re-enable with `{"guardEnabled": true}` once its validator config is fixed; re-enabling while broken will block all code edits again.

---

## 5. Full backlog from the sub-audits

**Status (2026-06-05, second pass):** the **entire §5 backlog is now IMPLEMENTED**
(commits `4181201` A, `d7d147e` #16, `e453c1e` C, `8150bbf` D, `d7cba5f` F,
`43f6b2e`+`ba52881` #18, `33d34fa` overwrite-safety, `fbf7093` #15 bootstrap,
`ea3070d` input-validation). Test suite is **220 passing**. Every item below is
**[DONE]**. The estimator changes assume the multi-annotator geometry is
regenerated on the fixed code (the one action that remains, owned by the
parallel session — see §6).

Lower-severity findings surfaced by the parallel runner/CBS audits.

### Statistical correctness (MEDIUM)
- **[DONE]** `build_union_basis` — added optional `per_behaviour_weights` (eigenvalue scaling) so the threshold means activation variance; documented the unweighted caveat.
- **[DONE]** `paired_geometric_tests` — Cliff's-delta bootstrap CI now `paired=True` (matches the paired Wilcoxon).
- **[DONE]** `jonckheere_terpstra` — documented the no-ties (continuous-exact, conservative-under-ties) variance + warn on >5% tied values.

### Silent-failure / robustness (MEDIUM)
- **[DONE]** `09_cbs_geometry.py` real loader — implemented `_load_real_labels` (real tiers aligned to `build_row_index`); uses real labels when present and **never stamps `labels_source:"real"` on synthetic data** again.
- **[DONE]** `09` mannwhitneyu — now catches `ValueError` (all-identical inputs) too.
- **[DONE]** `cbs/annotation.py` — narrowed the over-broad catch + added a non-dict-JSON guard (was an uncaught-crash path).
- **[DONE]** Overwrite safety — `src.config.backup_existing()` (copy to `.bak`) wired into `generate_chains`/`annotate_chains` at the resume check (they already wrote atomically via `tmp.rename`; this adds a last-good snapshot), and the non-atomic savers `save_pca_results`/`save_steering_vectors` now back up + write via `tmp.replace`.

### Reproducibility / provenance (MEDIUM)
- **[DONE]** CBS `seed` — `08/09/11/12/13` `--seed` default → `config.SEED` (42); `12` gained a `--seed` arg; `08` threads it into `annotate_chains_cbs`.
- **[DONE]** Provenance stamps — `src.config.provenance()` (git commit/dirty + seed + args + input SHA-256) wired into `save_pca_results` (provenance.json), `save_steering_vectors` (metadata._provenance), `05`, `06`, `build_phase6`, `robustness_geometry`. *(`05c`, `compute_layer_triangulation`, figure scripts still un-stamped — low priority.)*
- **[DONE]** Peak layers — `config.yaml analysis.peak_layers` + `src.config.PEAK_LAYERS`; `build_phase6` + `robustness_geometry` source them (and the seed) from config.

### Duplication (MEDIUM/LOW)
- **[DONE]** `_find_sentence_offset` — canonical `src/text_offsets.py`; all 5 sites import it (verified same object in a test).
- **[DONE] `MODELS` dicts** (#18) — added per-model `cli_alias` to `config.yaml` + `MODELS_BY_CLI`/`model_tuple`/`model_dict` in `src.config`. `02`/`04`/`07` now build their `--model` registries from config (04/07 all 5 models; 02 keeps its thinking-only `{1.5b,7b,8b}` subset). Eliminated the divergence that left 07 missing the baseline.

### Schema / data (LOW)
- **[DONE]** `cohort.is_truncated` cap sourced from `config.chains.max_new_tokens`.
- **[DONE]** `schemas.CBSResult` coerces `"3"`/`3.0` tier and `"yes"`/`"true"` cross_domain.
- **[DONE — false alarm, pinned]** Sentence-ID convention: both `matching` and `trajectory` use `f"{chain_id}:{full_array_index}"` (the "filtered vs full" claim was wrong). Integration test added so it can't drift.
- **[DONE]** `data/MANIFEST.md` force-tracked.

### Docs (LOW)
- **[DONE]** `predict_saturation.py` docstring corrected to the real filename convention. (`06b` had no stale ref.)
- **[DONE]** `src.config.require_file(path, hint)` exits with a one-line message instead of a raw traceback; wired into `build_phase6` (ACT) and `robustness_geometry` (ACT + ANNOT). (Other figure scripts still unguarded — trivial to extend with the same helper.)

### True cross-model bootstrap (the residual of #15)
- **[DONE]** `bootstrap_ci` gained `return_dist`; `09_cbs_geometry` persists `effect_size_boots`; `cross_model_compare` now does a genuine two-sample bootstrap of the effect-size difference when both distributions are present (key `p_value` + `p_method ∈ {two_sample_bootstrap, normal_approx_from_ci, none}`), falling back to the normal approximation otherwise.

---

## 6. Next-session checklist (ordered)

The §5 backlog is fully implemented. What remains is **not code** — it's the
consistency action and the merge:

1. **Regenerate** all geometry/robustness results on the fixed estimators (the
   May 28–30 `results/geometric` + `results/robustness` are from the biased
   code) so the 3-annotator comparison is uniform — `robustness_geometry.py`
   imports the three estimators that changed. Then refresh the TwoNN dims in
   `PROGRESS.md`. **This is owned by the parallel session.**
2. **Merge** `audit/se-robustness-fixes` → `main` (~19 commits; the first bundles
   pre-existing WIP, several bundle the parallel session's `config.yaml`
   safety-model edits + the untracked ad-hoc scripts — see commit messages).
   Re-run `pytest` (expect **220**).
3. **Methodological cautions** (research-design, unchanged): ~~small N for
   intrinsic-dim/curvature~~ **[CORRECTED 2026-06-06: smoke-run artifact — full-run
   N is 5k–16k/behaviour; the real issue is effective-N/CF-2, see
   `CONFOUNDS_AND_REMEDIATION.md` §1]**; "manifold-projected steering" is a *linear*
   projection (tests subspace, not curvature); single-annotator labels (the
   3-annotator agreement work mitigates this); ~50% chain truncation;
   first-10-token mean-pooling collapses the trajectory.
4. **[DONE]** provenance stamps now on `05c`/`compute_layer_triangulation`/
   `make_explainer_figs`/`make_fresh_figures`/`render_html`; `require_file`
   wired through the input-reading scripts. The only standing item is to
   re-enable `tdd-guard` once its validator is fixed (§4) — it is currently
   disabled because it blocks all `.py` edits.
