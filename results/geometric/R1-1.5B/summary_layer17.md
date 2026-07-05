# Phase 5b diagnostics - R1-1.5B layer 17

k sweep: [10, 30, 100]
PCA topk for curvature: 50
Null hierarchy resamples: 200

## Intrinsic dimension estimates

| Behaviour | N | TwoNN | Levina-Bickel | Correlation dim |
|-----------|---|-------|---------------|------------------|
| backtracking | 10136 | 3.9 [3.3, 3.4] | 3.3 [3.2, 3.3] | 5.4 [5.3, 5.4] |
| uncertainty-estimation | 16545 | 4.2 [3.5, 3.6] | 3.1 [2.9, 3.0] | 5.7 [5.6, 5.8] |
| example-testing | 5766 | 4.1 [3.5, 3.7] | 5.6 [5.3, 5.6] | 6.5 [6.4, 6.6] |
| adding-knowledge | 4989 | 4.3 [3.7, 4.0] | 7.9 [7.6, 8.1] | 7.7 [7.6, 7.8] |

## Curvature diagnostics (post-PCA projection)

Local-vs-global PCA dim ratio (close to 1 = flat, lower = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 0.554 [0.552, 0.638] | 0.514 [0.537, 0.596] | 0.570 [0.589, 0.652] |
| uncertainty-estimation | 0.606 [0.537, 0.631] | 0.567 [0.519, 0.610] | 0.600 [0.565, 0.636] |
| example-testing | 0.643 [0.616, 0.696] | 0.616 [0.594, 0.673] | 0.641 [0.636, 0.696] |
| adding-knowledge | 0.687 [0.670, 0.742] | 0.669 [0.650, 0.714] | 0.687 [0.671, 0.731] |

Geodesic / Euclidean ratio (close to 1 = flat, higher = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 3.234 [3.032, 3.189] | 2.285 [2.124, 2.193] | 1.731 [1.664, 1.704] |
| uncertainty-estimation | 3.220 [3.081, 3.240] | 2.285 [2.188, 2.269] | 1.786 [1.719, 1.766] |
| example-testing | 3.144 [2.923, 3.081] | 2.175 [2.069, 2.148] | 1.695 [1.632, 1.674] |
| adding-knowledge | 2.850 [2.703, 2.822] | 2.079 [2.006, 2.069] | 1.660 [1.617, 1.657] |

Tangent-space variation (degrees, close to 0 = flat, higher = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 68.890 [66.786, 69.718] | 67.984 [67.531, 69.643] | 67.777 [65.842, 68.298] |
| uncertainty-estimation | 67.919 [66.940, 70.387] | 68.625 [67.921, 70.187] | 67.882 [66.815, 69.201] |
| example-testing | 68.786 [68.264, 70.994] | 69.185 [68.200, 70.252] | 69.020 [65.633, 68.506] |
| adding-knowledge | 70.819 [69.341, 71.773] | 70.537 [68.942, 70.630] | 67.978 [65.651, 68.417] |

## Null hypothesis hierarchy (top-10 variance ratio)

Primary: chain-stratified permutation. Secondary: cross-chain permutation. Tertiary: MP isotropic (finite-sample inflation diagnostic only).

| Behaviour | real | chain-strat null (mean, 95% CI), p | cross-chain null (mean, p) | MP null mean |
|-----------|------|-------------------------------------|----------------------------|--------------|
| backtracking | 0.5016 | 0.4718 [0.4705, 0.4735], p=0.0050 | 0.4693, p=0.0050 | 0.0123 |
| uncertainty-estimation | 0.4851 | 0.4752 [0.4743, 0.4761], p=0.0050 | 0.4690, p=0.0050 | 0.0109 |
| example-testing | 0.4799 | 0.4777 [0.4761, 0.4794], p=0.0100 | 0.4698, p=0.0050 | 0.0146 |
| adding-knowledge | 0.4605 | 0.4658 [0.4640, 0.4677], p=1.0000 | 0.4701, p=1.0000 | 0.0154 |