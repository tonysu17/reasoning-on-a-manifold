# Geometry robustness (Tier 0) — deduplicated, stable estimator

Intrinsic dim = correlation dimension (twoNN noted unstable). Curvature = geodesic/Euclidean ratio.
`random_sub` and `chain_strat` are the SAME size (n_chains); their difference isolates the chain effect.

| Behaviour | dup% | N_uniq | PR | n>MP | **cdim full** | cdim randsub | **cdim chainstrat** | geo full | geo randsub | geo chainstrat |
|---|---|---|---|---|---|---|---|---|---|---|
| backtracking | 52% | 4712 | 37 | 104 | 6.67 | 6.72 | 6.72 | 2.96 | 2.29 | 2.34 |
| uncertainty-estimation | 43% | 3601 | 37 | 96 | 7.00 | 6.92 | 7.06 | 2.92 | 2.35 | 2.40 |
| adding-knowledge | 33% | 5422 | 46 | 131 | 8.36 | 8.24 | 8.63 | 3.00 | 2.44 | 2.39 |
| example-testing | 34% | 7242 | 31 | 134 | 6.56 | 6.57 | 6.98 | 3.35 | 2.40 | 2.40 |

## Reading
- **cdim chainstrat ≈ cdim randsub ≈ cdim full** → low intrinsic dimension is behaviour-intrinsic, NOT a chain confound (keystone PASS).
- **geo randsub vs geo chainstrat** at equal N isolates real chain-trajectory curvature from sparse-graph effects.
- 35–56% exact-duplicate pooled activations were removed before all estimates (fixed 1+10-token window on short repeated markers).
- twoNN is duplicate/subsample-unstable here; correlation dimension is the reliable estimator.