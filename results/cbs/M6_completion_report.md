# M6 Completion Report — Baseline-model replication orchestration

**Branch**: `cbs/m6-baseline`
**Date**: 2026-05-27
**Synthesis-plan reference**: §M6.1–§M6.5
**Status**: COMPLETE (orchestration code + cross-model helpers; no smoke run)

## What was implemented

### `src/cbs/comparison.py`

* `cross_model_compare(r1_results, base_results, *, n_bootstrap=500, seed=0)`
  — joins per (layer, behaviour, statistic, label_scheme); emits
  `{value_r1, value_base, delta, bootstrap_p, ci95_r1, ci95_base}`
  records. Skips shuffle / reversal control records. Flags missing
  counterparts with a `missing: "r1" | "base"` field.

* `trajectory_wasserstein(r1_success, r1_failure, base_success,
  base_failure)` — 2-Wasserstein via POT if installed; falls back to a
  1-D sort-based exact computation averaged across feature dimensions.
  Returns the four pairwise W₂ distances + the backend used.

* `cross_model_classifier(r1_train, r1_labels, base_test, base_labels,
  *, seed=0)` — trains a logistic regression on R1 success/failure
  features and reports transfer accuracy on the baseline. Returns NaN
  when sklearn is unavailable.

### `13_baseline_replication.py`

Orchestrator runner. Reads `geometry_results.json` from R1 and Baseline
directories; emits cross-model comparison artefacts into
`results/cbs/cross_model/`.

Build-now path (any prerequisite missing): writes
`results/cbs/cross_model/cross_model_blocked.json` with named blockers
and the exact Extension A commands needed:

```
Phase 2b: 02b_generate_baseline_chains.py
Phase 3:  03_annotate_chains.py --model-short QwenMath-1.5B
Phase 4:  04_extract_activations.py --model qwen-math-1.5b
M1:       08_annotate_cbs.py --model-suffix QwenMath-1.5B
M2:       09_cbs_geometry.py --model-suffix QwenMath-1.5B
M3:       10_trajectory_build.py --model-suffix QwenMath-1.5B
M4:       11_trajectory_analysis.py --model-suffix QwenMath-1.5B
```

Run-phase path (geometry results present on both sides):

1. `cross_model_geometry.json` — full per (layer, behaviour, statistic,
   label_scheme) delta table.
2. `structured_null_baseline.json` — emitted when baseline tier-3 rate
   < 3% per synthesis §M6.4 ("distillation adds tier-3 capacity"
   interpretation).
3. `trajectory_wasserstein_pending.json` and
   `cross_model_classifier_pending.json` — stubbed until Phase 7
   answer-checker provides (success, failure) labels.

## Validation

### Unit tests

```
$ python -m pytest src/cbs/tests/test_comparison.py -q
7 passed
```

Coverage:

* `cross_model_compare`: matches paired records, flags missing,
  skips shuffle / reversal controls, output is JSON-serialisable.
* `trajectory_wasserstein`: runs on multi-dim synthetic data, picks
  POT or sort-based backend.
* `cross_model_classifier`: well-separated transfer ≈ 0.85+.

Full suite (all 8 modules): `98 passed`.

### Build-time artefact

`results/cbs/cross_model/cross_model_blocked.json` — `status: blocked`
with the Extension A prerequisite chain spelled out for the operator.

## Deviations from synthesis

* No smoke run for M6 per synthesis §9 step 10 ("M6 orchestration code
  (no smoke run)").
* The runner emits a `blocked` JSON instead of failing hard when the
  baseline pipeline has not produced its CBS / geometry outputs yet.
  Matches the convention used by M4 and M5 runners.
* `trajectory_wasserstein` and `cross_model_classifier` write `pending`
  artefacts because both depend on (success, failure) labels from Phase
  7's answer-checker, which is gated separately.

## Next milestone

Final — write `results/cbs/BUILD_COMPLETE.md` sentinel and halt.
