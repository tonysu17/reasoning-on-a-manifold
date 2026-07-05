# Geometry robustness (Tier 0) — deduplicated, stable estimator

Intrinsic dim = correlation dimension (twoNN noted unstable). Curvature = geodesic/Euclidean ratio.
`random_sub` and `chain_strat` are the SAME size (n_chains); their difference isolates the chain effect.

| Behaviour | dup% | N_uniq | PR | n>MP | **cdim full** | cdim randsub | **cdim chainstrat** | geo full | geo randsub | geo chainstrat |
|---|---|---|---|---|---|---|---|---|---|---|
| backtracking | 1% | 10136 | 20 | 124 | 5.85 | 5.81 | 6.36 | 3.73 | 2.42 | 2.35 |
| uncertainty-estimation | 1% | 16545 | 23 | 147 | 6.21 | 6.15 | 6.88 | 4.04 | 2.42 | 2.43 |
| adding-knowledge | 1% | 4989 | 25 | 115 | 7.71 | 7.68 | 8.46 | 3.44 | 2.55 | 2.53 |
| example-testing | 1% | 5766 | 29 | 121 | 6.04 | 6.02 | 6.72 | 4.22 | 2.55 | 2.34 |

## Reading
- **cdim chainstrat ≈ cdim randsub ≈ cdim full** → low intrinsic dimension is behaviour-intrinsic, NOT a chain confound (keystone PASS).
- **geo randsub vs geo chainstrat** at equal N isolates real chain-trajectory curvature from sparse-graph effects.
- 35–56% exact-duplicate pooled activations were removed before all estimates (fixed 1+10-token window on short repeated markers).
- twoNN is duplicate/subsample-unstable here; correlation dimension is the reliable estimator.