# M2 Completion Report — Per-sentence geometry + bridge angles

**Branch**: `cbs/m2-geometry`
**Date**: 2026-05-27
**Synthesis-plan reference**: §M2.1–§M2.5
**Status**: COMPLETE (smoke regression artefact in place)

## What was implemented

### `src/cbs/geometry.py`

* `centroid_distance(X, centroid)` — per-row L2 distance.
* `out_of_subspace_residual(X, union_basis)` — per-row
  `||(I - V V^T) x|| / ||x||`, clipped to `[0, 1]`, zero-norm rows return 0.
* `principal_angles(V_a, V_b, top_k=10)` — SVD-based, sorted ascending.
* `build_union_basis(per_behaviour_pcs, variance_threshold=0.95)` —
  SVD-based; returns orthonormal basis covering the joint variance threshold.
* `cliffs_delta(a, b)` — vectorised `(gt - lt) / (na * nb)`.
* `jonckheere_terpstra(values, tiers)` — closed-form mean / variance + normal
  approximation Z and two-sided p; trend-direction sign. Handles
  single-tier-group degenerate case.
* `local_intrinsic_dim(X, k=20, estimator="twoNN")` — per-row TwoNN over
  local k-NN cloud; NaN when too few points or degenerate.
* `bootstrap_ci(fn, *args, paired=True|False)` — percentile CI; paired
  resampling for same-length args, separate resampling for two-group
  statistics (e.g. Cliff's delta on differently-sized groups).
* `holm_correction(p_values)` — step-down adjustment with running maximum.

### `09_cbs_geometry.py`

CLI matches synthesis §M2.2 defaults:

```
--activations-dir       data/activations/R1-1.5B
--cbs-annotations       data/chains_cbs_annotated_R1-1.5B.json
--out-dir               results/cbs/{model-suffix}
--model-suffix          R1-1.5B  (smoke runs: R1-1.5B-smoke)
--layers                3,7,10,14,17,21,24,27
--behaviours            adding-knowledge,deduction
--n-bootstrap           1000
--variance-thresholds   0.90,0.95,0.99
```

Plus smoke-extension flags:

```
--synthetic-tiers       assign per-row tiers via rng
--shuffle-control       additionally emit shuffled-tier sanity rerun
--reversal-control      additionally emit reversed-tier sanity rerun
```

Falls back to `--synthetic-tiers` automatically when the CBS annotations
file is missing (build-now path).

Output: `geometry_results.json` (synthesis §M2.3 schema), 32 plots
(`effect_size_vs_layer__*` per (behaviour, statistic, label_scheme) and
`principal_angle_heatmap_layer{N}.png` per layer).

## Validation

### Unit tests

```
$ python -m pytest src/cbs/tests/test_geometry.py -q
28 passed
```

Coverage:

* Centroid distance: zero-at-centroid, L2 correctness, shape-mismatch raise.
* OOS residual: zero inside subspace, ~1 orthogonal to subspace, zero rows.
* Principal angles: identical subspaces -> 0, orthogonal -> π/2, sorted.
* Union basis: respects variance threshold, lower threshold -> smaller basis,
  empty input raises.
* Cliff's delta: positive case, perfect separation ±1, shuffle to ~0.
* Jonckheere-Terpstra: monotonic positive, reversal flips direction,
  no-trend → high p, single-tier degenerate.
* Local intrinsic dim: per-row shape, small N → NaN.
* Bootstrap CI: paired constant returns constant, separate for two groups.
* Holm correction: monotone, all-ones capped.
* **Shuffle sanity**: random tiers + shuffle -> JT p > 0.05 and
  |Cliff's delta| < 0.20.
* **Reversal sanity**: forward tiers trend = +1, reversed tiers trend = -1.

### Smoke regression artefact

`results/cbs/R1-1.5B-smoke/geometry_results.json` exists with:

* `labels_source`: `synthetic` (smoke-only, **not paper-grade**).
* `note`: `"smoke-only, not paper-grade"`.
* `n_records_total`: 576 (192 main + 192 shuffle_control + 192 reversal_control).
* `n_records_main`: 192 (4 behaviours × 8 layers × 3 statistics × 2 label
  schemes).
* `principal_angles`: 48 pairs (8 layers × 6 behaviour-pair combinations
  from {adding-knowledge, backtracking, uncertainty-estimation,
  example-testing}).
* Each record carries `statistic`, `label_scheme`, `test_statistic`,
  `p_raw`, `p_holm`, `effect_size`, `effect_size_ci95`, `n_total`,
  `n_per_tier` (CBS) / `n_cross_domain` (binary), `trend_direction`,
  `layer`, `behaviour`, `labels_source`.

Mean |Cliff's δ| for centroid_distance across synthetic-tier records:
~0.14 (small — consistent with no real tier signal).

### Plots

32 PNGs in `results/cbs/R1-1.5B-smoke/plots/`:

* 24 effect-size-vs-layer plots (4 behaviours × 3 statistics × 2 label
  schemes).
* 8 principal-angle heatmaps (one per layer).

## Deviations from synthesis

* `--synthetic-tiers` mode added (not in synthesis spec). Necessary for
  build-now smoke since real CBS annotations are blocked on P0.2 anchor
  curation.
* Category-stratified rerun (synthesis §M2.4) deferred: needs
  row-to-`(chain_id, category)` provenance which the current
  `metadata.json` does not record. Flagged for M3 / run-phase.

## Smoke vs full-corpus split

| What | Smoke (now) | Full corpus (later) |
|---|---|---|
| Activations | `data/activations/R1-1.5B-smoke/` (~50 per behaviour) | `data/activations/R1-1.5B/` (full Phase 4) |
| Tiers | `--synthetic-tiers` | from `data/chains_cbs_annotated_R1-1.5B.json` |
| Cross-domain | synthetic 30/70 | from CBS annotations |
| Output | `results/cbs/R1-1.5B-smoke/` | `results/cbs/R1-1.5B/` |
| Paper-grade? | No — code-correctness only | Yes |

## Next milestone

M3 — trajectory module + synthetic-helix unit test + smoke run.
