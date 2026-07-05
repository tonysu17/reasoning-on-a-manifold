# Phase 5b diagnostics - R1-1.5B layer 11

k sweep: [10, 30, 100]
PCA topk for curvature: 50
Null hierarchy resamples: 200

## Intrinsic dimension estimates

| Behaviour | N | TwoNN | Levina-Bickel | Correlation dim |
|-----------|---|-------|---------------|------------------|
| backtracking | 10136 | 3.9 [3.3, 3.4] | 3.1 [3.0, 3.1] | 6.0 [5.9, 6.0] |
| uncertainty-estimation | 16545 | 4.2 [3.6, 3.6] | 2.9 [2.7, 2.8] | 6.1 [6.0, 6.1] |
| example-testing | 5766 | 4.0 [3.5, 3.7] | 5.2 [4.9, 5.2] | 5.8 [5.7, 5.9] |
| adding-knowledge | 4989 | 4.5 [3.8, 4.1] | 7.5 [7.1, 7.6] | 7.6 [7.5, 7.7] |

## Curvature diagnostics (post-PCA projection)

Local-vs-global PCA dim ratio (close to 1 = flat, lower = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 0.533 [0.536, 0.617] | 0.492 [0.505, 0.564] | 0.527 [0.547, 0.610] |
| uncertainty-estimation | 0.584 [0.513, 0.597] | 0.531 [0.487, 0.572] | 0.553 [0.525, 0.590] |
| example-testing | 0.641 [0.613, 0.693] | 0.601 [0.575, 0.660] | 0.607 [0.605, 0.667] |
| adding-knowledge | 0.683 [0.659, 0.731] | 0.651 [0.643, 0.704] | 0.673 [0.659, 0.721] |

Geodesic / Euclidean ratio (close to 1 = flat, higher = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 3.285 [3.111, 3.246] | 2.362 [2.225, 2.312] | 1.801 [1.711, 1.758] |
| uncertainty-estimation | 3.250 [3.142, 3.327] | 2.368 [2.262, 2.346] | 1.794 [1.745, 1.797] |
| example-testing | 2.980 [2.862, 3.032] | 2.167 [2.071, 2.149] | 1.690 [1.622, 1.663] |
| adding-knowledge | 2.894 [2.709, 2.840] | 2.077 [1.991, 2.050] | 1.639 [1.592, 1.631] |

Tangent-space variation (degrees, close to 0 = flat, higher = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 65.834 [63.697, 68.289] | 66.330 [65.077, 67.968] | 65.896 [64.555, 66.889] |
| uncertainty-estimation | 65.298 [63.609, 68.818] | 65.964 [65.591, 69.005] | 66.356 [65.794, 68.032] |
| example-testing | 66.170 [66.093, 70.240] | 68.342 [67.064, 69.689] | 67.867 [65.304, 68.454] |
| adding-knowledge | 70.519 [68.506, 71.729] | 70.194 [68.491, 70.562] | 67.519 [65.688, 68.068] |

## Null hypothesis hierarchy (top-10 variance ratio)

Primary: chain-stratified permutation. Secondary: cross-chain permutation. Tertiary: MP isotropic (finite-sample inflation diagnostic only).

| Behaviour | real | chain-strat null (mean, 95% CI), p | cross-chain null (mean, p) | MP null mean |
|-----------|------|-------------------------------------|----------------------------|--------------|
| backtracking | 0.4985 | 0.4608 [0.4594, 0.4626], p=0.0050 | 0.4588, p=0.0050 | 0.0123 |
| uncertainty-estimation | 0.4821 | 0.4628 [0.4619, 0.4639], p=0.0050 | 0.4585, p=0.0050 | 0.0109 |
| example-testing | 0.4744 | 0.4745 [0.4727, 0.4762], p=0.5224 | 0.4594, p=0.0050 | 0.0146 |
| adding-knowledge | 0.4324 | 0.4526 [0.4502, 0.4547], p=1.0000 | 0.4598, p=1.0000 | 0.0154 |