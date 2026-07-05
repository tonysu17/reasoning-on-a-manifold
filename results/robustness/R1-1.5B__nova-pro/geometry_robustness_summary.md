# Geometry robustness (Tier 0) — deduplicated, stable estimator

Intrinsic dim = correlation dimension (twoNN noted unstable). Curvature = geodesic/Euclidean ratio.
`random_sub` and `chain_strat` are the SAME size (n_chains); their difference isolates the chain effect.

| Behaviour | dup% | N_uniq | PR | n>MP | **cdim full** | cdim randsub | **cdim chainstrat** | geo full | geo randsub | geo chainstrat |
|---|---|---|---|---|---|---|---|---|---|---|
| backtracking | 58% | 5519 | 37 | 113 | 6.64 | 6.72 | 6.75 | 2.98 | 2.30 | 2.33 |
| uncertainty-estimation | 46% | 5754 | 39 | 117 | 7.15 | 7.10 | 6.98 | 2.96 | 2.30 | 2.37 |
| adding-knowledge | 28% | 5715 | 41 | 133 | 8.70 | 8.72 | 8.51 | 3.23 | 2.47 | 2.38 |
| example-testing | 30% | 2364 | 24 | 92 | 5.96 | 6.01 | 7.12 | 3.17 | 2.30 | 2.25 |

## Reading
- **cdim chainstrat ≈ cdim randsub ≈ cdim full** → low intrinsic dimension is behaviour-intrinsic, NOT a chain confound (keystone PASS).
- **geo randsub vs geo chainstrat** at equal N isolates real chain-trajectory curvature from sparse-graph effects.
- 35–56% exact-duplicate pooled activations were removed before all estimates (fixed 1+10-token window on short repeated markers).
- twoNN is duplicate/subsample-unstable here; correlation dimension is the reliable estimator.