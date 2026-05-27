# M3 Completion Report — Trajectory module

**Branch**: `cbs/m3-trajectory`
**Date**: 2026-05-27
**Synthesis-plan reference**: §M3.1–§M3.4
**Status**: COMPLETE (smoke regression artefact in place)

## What was implemented

### `src/cbs/cohort.py`

* `DEFAULT_MAX_NEW_TOKENS = 8192` (Phase 2 cap).
* `is_truncated(chain)` — single source of truth for the P0.4
  truncation-cohort flag. Used by `build_trajectory` and downstream M4
  analyses.

### `src/cbs/trajectory.py`

* `PHASE_4_BEHAVIOURS` — locked tuple of behaviours that Phase 4 saved
  activations for: `backtracking`, `uncertainty-estimation`,
  `example-testing`, `adding-knowledge`. `initializing` and `deduction`
  spans are skipped during trajectory construction because no activations
  exist for them (Phase 4 was scoped to the four thinking-distinct
  behaviours).
* `build_row_index(chains, target_behaviours)` — reconstructs the
  `{behaviour: [(chain_id, span_idx), ...]}` mapping deterministically,
  matching `activation_extraction.py`'s iteration order.
* `load_layer_activations(activations_dir, layer)` — loads every
  `{behaviour}_layer{layer}.npy` in one pass.
* `build_trajectory(chain, ..., row_index, activations)` — assembles a
  `ChainTrajectory` at one layer; carries `truncated` from
  `is_truncated`. Sentences without saved activations are silently
  skipped, producing a possibly-sparse sub-trajectory.
* `arc_length_sequence(traj)` / `total_arc_length(traj)`.
* **`curvature_sequence(traj)`** — locked formula
  (arc-length-reparameterised discrete Frenet) per synthesis §M3.2:

  ```
  T_left  = (x_t     - x_{t-1}) / ||x_t     - x_{t-1}||
  T_right = (x_{t+1} - x_t)     / ||x_{t+1} - x_t||
  ds      = (||x_t - x_{t-1}|| + ||x_{t+1} - x_t||) / 2
  kappa_t = ||T_right - T_left|| / ds
  ```

  NaN at boundaries (t = 0, T−1) and on duplicate consecutive activations.

* `subspace_visit_sequence(traj, subspaces)` — per-step argmax behaviour
  by projection magnitude.
* `cross_subspace_returns(traj, subspaces)` — visit sequence, transition
  count, return rate, transition matrix.
* `trajectory_cone_angle(traj)` — max angular deviation of any
  unit-normalised activation from the trajectory's mean unit direction.
* `compare_groups(...)` and `per_sentence_curvature_vs_tier(...)` —
  scaffolded for M4 (synthesis §M4.2 "extend").

### `10_trajectory_build.py`

CLI per synthesis §M3.2 + smoke extensions. Falls back to
`data/annotated_R1-1.5B.json` when CBS annotations are absent.

Output: per-layer subdir with one JSON per chain + a
`layer{N}_summary.parquet` aggregate. Also writes `run_metadata.json`
with the provenance of the run (activations dir, chains source, layer
list, row-index sizes, truncation policy).

## Validation

### Unit tests

```
$ python -m pytest src/cbs/tests/test_trajectory.py -q
16 passed
```

Coverage:

* Module import smoke.
* `is_truncated`: truncated / clean / boundary-at-max-but-closed /
  short-unclosed.
* `arc_length_sequence`: zero at start, empty trajectory, straight line
  matches Euclidean.
* `total_arc_length`: straight-line Euclidean equivalence.
* **`curvature_sequence` — straight line**: zero curvature on the
  interior, NaN at boundaries.
* **`curvature_sequence` — synthetic helix**: 200-sample helix r(t) =
  (cos t, sin t, 0.3 t) embedded in R^1536 by zero-padding. Recovered
  curvature mean = 0.917, target R / (R^2 + c^2) = 0.917, relative
  error < 5%, interior std < 0.05.
* `curvature_sequence` — degenerate cases T = 1, 2 → all NaN; duplicate
  consecutive points → NaN curvature at that index, valid at others.
* Subspace dynamics: argmax assignment, visit sequence, transitions,
  return rate on a hand-built ABABB example (return_rate = 2/4).
* Trajectory cone angle: aligned points → 0, orthogonal directions →
  large angle.
* `build_row_index`: orders by chain then span; respects
  `target_behaviours` filter.
* `build_trajectory`: uses lookup correctly; truncated flag propagates
  from `is_truncated`.

### Smoke regression artefact

`results/trajectory/R1-1.5B-smoke/` contains:

* `layer17/` and `layer27/` per-chain JSONs (20 chains each — the
  first-20 cohort that the smoke activations were extracted from).
* `layer17_summary.parquet` and `layer27_summary.parquet` (20 rows each).
* `run_metadata.json` with row-index sizes that match the smoke
  `metadata.json::n_extracted` exactly:

  ```
  {backtracking: 97, uncertainty-estimation: 145,
   example-testing: 145, adding-knowledge: 51}
  ```

Sample summary statistics on layer 17 (20 chains, smoke):

| stat | min | median | max |
|---|---|---|---|
| T (sentence count) | 2 | 14 | 97 |
| arc_length | 83 | 919 | 6579 |
| mean_curvature | 0.021 | 0.025 | 0.039 |
| cone_angle (rad) | 0.43 | 0.77 | 0.98 |
| truncated | 7 / 20 chains (35%) | | |

Mean curvature is roughly stable across chains (~0.025) — consistent
with a coarse "geodesic-like" picture of these reasoning manifolds. The
paper-grade interpretation comes from the full Phase 4 run.

## Deviations from synthesis

* Synthesis §M3.2 spec says `build_trajectory(chain, activations_dir,
  layer)` — implementation adds optional `row_index` and `activations`
  parameters so the row index can be precomputed once and reused across
  layers in the runner. The single-chain convenience path (no row index)
  also works for unit tests.
* Sentences whose Phase-3 behaviour is `initializing` or `deduction` are
  excluded because Phase 4 only extracted the four thinking-distinct
  behaviours. The resulting trajectories are sparser than chain length
  (e.g. a 56-sentence chain may produce a 14-point trajectory because
  only those 14 sentences are in the saved behaviours). Flagged for
  Phase 4 re-extraction if M3/M4 paper-grade analyses need denser
  trajectories — but per the hard constraint we do not re-extract.

## Smoke vs full-corpus split

| What | Smoke (now) | Full corpus (later) |
|---|---|---|
| Chains | First 20 of `annotated_R1-1.5B.json` | All 1000 |
| Activations | `data/activations/R1-1.5B-smoke/` | `data/activations/R1-1.5B/` |
| Truncation | `(b) stratify` (P0.4) | same |
| Output | `results/trajectory/R1-1.5B-smoke/` | `results/trajectory/R1-1.5B/` |
| Paper-grade? | No — smoke-only, not paper-grade | Yes |

## Next milestone

M4 — group comparisons + matched-pair scaffold + verification gradient.
