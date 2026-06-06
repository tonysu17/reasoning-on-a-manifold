# M5 Completion Report — CBS steering ablation (causal experiment)

**Branch**: `cbs/m5-ablation`
**Date**: 2026-05-27
**Synthesis-plan reference**: §M5.1–§M5.4
**Status**: COMPLETE (build only — intervention run gated on Phase 4 + M1 + M2)
**Scope decision**: Tony approved "Build M5 now" at the M5 gate.

## What was implemented

### `src/cbs/ablation.py`

* `build_v_cbs(tier3_activations, tier1_activations)` — unit-normalised
  mean-difference direction. Raises on empty input or coincident means.
* `validate_v_cbs(v_cbs, v_adding_knowledge_centroid, tier3_acts, tier1_acts,
  *, cv_folds=5, seed=0)` — returns the three diagnostic numbers + a
  `passes: bool` and a `failures: list[str]`. Pass criteria
  (synthesis §M5.3, all three required):

  ```
  |cos(v_cbs, v_adding_knowledge_centroid)| < FAILSTOP_COS_MAX        (0.5)
  cv_probe_accuracy_mean                  >= FAILSTOP_PROBE_ACC_MIN  (0.7)
  cv_probe_accuracy_std                   <= FAILSTOP_PROBE_STD_MAX  (0.15)
  ```

* `CBSAblationModel(SteeredModel)` — projection ablation
  `h' = h - alpha * (v_cbs^T h) * v_cbs`. Constructor rejects non-unit
  vectors with a clear error message.
* `construct_task_sets(annotated_chains, target_per_set=100, floor=50,
  correct_field="answer_correct")` — builds A (textbook-solvable) and
  B (bridge-required) candidate sets. Raises `RuntimeError` if either
  set has fewer than `floor` candidates, telling the caller to widen
  the corpus per synthesis §M5.2.
* `selectivity_ratio(delta_tier3, delta_tier1)` — guarded ratio; NaN
  when |Δ tier-1| is below 1e-6.

### `12_cbs_ablation.py`

CLI mirrors synthesis §M5.2:

```
--cbs-annotations       (default data/chains_cbs_annotated_R1-1.5B.json)
--activations-dir       (default data/activations/R1-1.5B)
--steering-layer        (default 27, also supports 17)
--v-cbs-source / --validation-output  (optional pre-saved artefacts)
--ablation-strengths    (default "0,0.5,1.0,2.0")
--conditions            (default "baseline,v_cbs,v_random,v_adding_knowledge")
--seeds-per-task        (default 5)
--dry-run-validate-only (build-now path: validation only, no intervention)
```

Runner flow:

1. Load tier-3 / tier-1 activations at the steering layer from the
   per-behaviour matrices via the M3 row-index reconstruction.
2. If absent or sparse, emit
   `results/cbs/{model}/v_cbs_construction_blocked.json` with the
   blockers (CBS annotations missing, P0.2 anchor lock + 08_annotate_cbs
   full run required). Build-now exits 0 here so the build phase
   continues; the run-phase resumes once unblocked.
3. Build v_CBS and save to `results/cbs/{model}/v_cbs_layer{N}.npy`.
4. Load the adding-knowledge centroid (mean of all adding-knowledge
   rows at the chosen layer).
5. Validate v_CBS. **HARD FAIL-STOP** on any of the three conditions:
   writes `results/cbs/{model}/FAILSTOP_M5.md` with the violating
   numbers and three options for the user (re-curate anchors / switch
   steering layer / declare v_CBS unsuited and scale up to 7B or drop M5).
6. `--dry-run-validate-only` returns here. The intervention loop
   (model.generate × condition × alpha × seed × task, then re-annotation
   + selectivity ratio) stays in run-phase (~25h cluster GPU + ~$50 API
   per synthesis §M5.4).

## Validation

### Unit tests

```
$ python -m pytest src/cbs/tests/test_ablation.py -q
14 passed
```

Coverage:

* `build_v_cbs`: unit norm; direction matches mean-difference; raises on
  coincident means; raises on empty.
* `validate_v_cbs`:
  - Well-separated tier-3 vs tier-1 with orthogonal centroid → `passes=True`.
  - v_CBS parallel to adding-knowledge centroid → `passes=False`,
    cos-failure recorded.
  - Indistinguishable tier-3 vs tier-1 → `passes=False`, probe-accuracy
    failure recorded.
* `CBSAblationModel`: rejects non-unit vectors; accepts unit vectors and
  inherits SteeredModel's `mode=subtract` discipline.
* `construct_task_sets`: floor-violation raises `RuntimeError`; correct
  pass-through and counts when above floor.
* `selectivity_ratio`: positive case; zero numerator; NaN on zero
  denominator.

Full suite: `92 passed, 1 skipped` (the one skip is M6's stub test).

### Smoke artefact

`results/cbs/R1-1.5B-smoke/v_cbs_construction_blocked.json` — `status:
blocked` with the named blockers. This is the correct artefact for
build-now: real v_CBS construction needs CBS annotations, which are
gated on P0.2.

## Deviations from synthesis

* The runner emits a `blocked` JSON instead of error-exit when CBS
  annotations / activations are missing. Build-now completes without
  crashing the chain; run-phase resumes once unblocked.
* Synthesis §M5.2 task-set construction includes "removing that
  sentence breaks correctness (spot-checked on 20)" — that spot-check
  requires actual generation under sentence masking and lives in the
  run-phase intervention loop, not in `construct_task_sets`.

## What's blocked + what unblocks it

| Step | Unblocked by |
|---|---|
| v_CBS construction | (a) P0.2 anchors locked → (b) `08_annotate_cbs.py` full run → produces `data/chains_cbs_annotated_R1-1.5B.json`; (c) Phase 4 full activations |
| Validation pass / fail decision | step above + an `adding-knowledge_layer{N}.npy` from full Phase 4 |
| Intervention loop | validation passes + cluster GPU + (optional) Phase 7 answer-checker for the textbook/bridge task-set construction |

## Next milestone

M6 — baseline replication orchestration (no smoke run; orchestration only).
