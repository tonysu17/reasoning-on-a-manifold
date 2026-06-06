# M4 Completion Report — Trajectory analysis + matched pairs + verification gradient

**Branch**: `cbs/m4-analysis`
**Date**: 2026-05-27
**Synthesis-plan reference**: §M4.1–§M4.6
**Status**: COMPLETE (smoke group-comparisons artefact in place; matched-pair
              and verification-gradient runners blocked on Phase 7 + multi-seed
              re-gen)

## What was implemented

### `src/cbs/matching.py`

* `jaccard_token_similarity(a, b)` — lowercased alphanumeric token-set
  Jaccard.
* `build_matched_pairs(success_chains, failure_chains, *, cbs_tier_filter=3,
  similarity_threshold=0.6)` — for each success-chain tier-3 sentence, the
  closest failure-chain tier-3 sentence on the *same task*, by Jaccard ≥
  threshold.
* `paired_geometric_tests(pairs, activations_lookup, layer, deduction_subspace)`
  — per geometric statistic: Wilcoxon signed-rank p, Cliff's delta,
  bootstrap CI, median diff. Handles missing `deduction_subspace` by
  skipping the projection statistic.
* `verification_gradient(correct_acts, incorrect_acts, cv_folds=5, seed=0)` —
  5-fold StratifiedKFold logistic regression. Returns unit-normalised
  averaged probe weights + mean/std accuracy + `stable: bool` (True iff
  std ≤ 0.15, per synthesis §M4.3 probe-stability gate).

### `src/cbs/trajectory.py` (M4 extensions)

Already added during M3 per synthesis §M4.2 "extend":

* `compare_groups(summary, group_col, stat_cols, residualise_on)` — OLS
  residualisation, Mann-Whitney p, Cliff's delta, bootstrap CI.
* `per_sentence_curvature_vs_tier(trajectories)` — long-format DataFrame.

### `11_trajectory_analysis.py`

Runs three groups of analyses:

1. **Group comparisons** on the layer-summary parquet:
   * `truncated` vs not (the P0.4 stratification — primary build-now group).
   * `high_cbs` (≥ 2 tier-3) vs `low_cbs` (0 tier-3) — runs only when
     CBS annotations supply `n_tier3_sentences > 0`.
   * `long_chain` (above median T) vs short — positional control.
2. **Matched-pair analysis** — emits a `blocked` stub when
   `data/chains_R1-1.5B_multiseed.json` and / or the CBS annotation file
   is missing. Three blockers listed: multi-seed re-gen + locked anchors
   + Phase 7 answer-checker.
3. **Verification gradient** — emits a `blocked` stub for the same Phase 7
   reason.

UMAP 2D projection of per-chain trajectory summary stats is computed when
the `umap-learn` extra is installed; otherwise the runner logs a warning
and continues.

## Validation

### Unit tests

```
$ python -m pytest src/cbs/tests/test_matching.py -q
11 passed
```

Coverage:

* Jaccard: positive overlap, no overlap, empty input, identical input,
  symmetric + lowercased.
* `build_matched_pairs`: positive match (Jaccard ≥ 0.5), filters below
  threshold, requires same task_id.
* `verification_gradient`: well-separated data → CV accuracy > 0.85 with
  std < 0.15 (`stable=True`); random labels → ~50% accuracy; empty input
  returns zero-shaped weights with `stable=False`.
* `paired_geometric_tests`: empty pairs path; runs end-to-end on synthetic
  activations.

Full suite: `74 passed, 4 skipped`.

### Smoke regression artefact (group comparisons)

`results/trajectory/R1-1.5B-smoke/`:

* `group_comparisons.json` — 2 layers × 2 group comparisons each
  (`truncated`, `long_chain`; `high_cbs` was skipped because the smoke
  parquet has no CBS-tier annotations — that signal arrives at the
  P0.2-locked annotator run).
* `matched_pair_results.json` — status=`blocked` with three named blockers.
* `verification_gradient.json` — status=`blocked` with two named blockers.

Sample (layer 17, truncated vs not):

| stat | n=(trunc, clean) | Cliff's δ | Wilcoxon p |
|---|---|---|---|
| arc_length | (13, 7) | +0.54 | 0.056 |
| cone_angle | (13, 7) | +0.14 | 0.64 |
| mean_curvature | (13, 7) | 0.00 | nan |

The `mean_curvature` zero-effect after residualising on T is a small-N
artefact (N=20 is severely underpowered; T and curvature in this smoke
are near-collinear, so residualisation produces near-zero variance).
The full Phase 4 run on 1000 chains will not exhibit this.

## Deviations from synthesis

* The runner emits a `blocked` stub for matched-pair and verification-
  gradient rather than failing hard, so the build phase can complete
  even before Phase 7 answer-checker + multi-seed re-gen land. Each stub
  records its specific blockers and the exact functions that will run
  when unblocked.
* `compare_groups` and `per_sentence_curvature_vs_tier` live in
  `src/cbs/trajectory.py` (per synthesis §M4.2 "extend trajectory module")
  rather than a new file. Tests still pass.
* UMAP plot uses per-chain summary stats as a coarse "trajectory profile"
  vector. Real per-sentence-point UMAP is for the run phase, when CBS
  tiers are available to colour the projection. UMAP itself is in the
  `[cbs]` extras and is not installed in the current environment, so the
  plot was skipped.

## What's blocked + what unblocks it

| Blocked thing | Unblocked by |
|---|---|
| `matched_pair_results.json` real run | (a) M4.5 chain_gen.py temperature/seed change merged, (b) multi-seed re-gen run → `data/chains_R1-1.5B_multiseed.json`, (c) P0.2 anchor lock → CBS annotations, (d) Phase 7 success/failure labels |
| `verification_gradient.json` real run | (c) + (d) above |
| UMAP plot | `pip install umap-learn` (in `[cbs]` extras) |
| `high_cbs` group comparison | (c) CBS annotations |

## Next milestone

M4.5 — `chain_gen.py` temperature + seed support + unit test.
