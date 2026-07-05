# Phase 5b diagnostics - R1-1.5B layer 27

k sweep: [10, 30, 100]
PCA topk for curvature: 50
Null hierarchy resamples: 200

## Intrinsic dimension estimates

| Behaviour | N | TwoNN | Levina-Bickel | Correlation dim |
|-----------|---|-------|---------------|------------------|
| backtracking | 10136 | 3.7 [3.1, 3.2] | 2.9 [2.8, 2.9] | 6.8 [6.7, 6.9] |
| uncertainty-estimation | 16545 | 4.0 [3.3, 3.4] | 2.8 [2.7, 2.7] | 6.4 [6.3, 6.5] |
| example-testing | 5766 | 3.8 [3.3, 3.5] | 4.4 [4.3, 4.5] | 6.0 [5.9, 6.1] |
| adding-knowledge | 4989 | 4.2 [3.6, 3.8] | 6.9 [6.6, 7.1] | 8.0 [7.9, 8.1] |

## Curvature diagnostics (post-PCA projection)

Local-vs-global PCA dim ratio (close to 1 = flat, lower = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 0.476 [0.482, 0.561] | 0.425 [0.435, 0.489] | 0.457 [0.478, 0.529] |
| uncertainty-estimation | 0.543 [0.479, 0.555] | 0.469 [0.422, 0.504] | 0.488 [0.470, 0.526] |
| example-testing | 0.595 [0.570, 0.634] | 0.517 [0.492, 0.565] | 0.519 [0.515, 0.554] |
| adding-knowledge | 0.643 [0.622, 0.686] | 0.567 [0.563, 0.613] | 0.570 [0.571, 0.611] |

Geodesic / Euclidean ratio (close to 1 = flat, higher = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 3.442 [3.256, 3.412] | 2.413 [2.248, 2.322] | 1.801 [1.706, 1.748] |
| uncertainty-estimation | 3.550 [3.321, 3.497] | 2.462 [2.299, 2.382] | 1.813 [1.744, 1.785] |
| example-testing | 3.204 [2.994, 3.161] | 2.289 [2.129, 2.204] | 1.723 [1.632, 1.676] |
| adding-knowledge | 2.950 [2.799, 2.949] | 2.157 [2.044, 2.119] | 1.647 [1.575, 1.619] |

Tangent-space variation (degrees, close to 0 = flat, higher = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 67.764 [66.410, 69.227] | 68.162 [66.588, 69.446] | 68.263 [66.628, 69.302] |
| uncertainty-estimation | 68.001 [66.496, 69.660] | 68.195 [67.092, 69.471] | 68.483 [66.997, 69.723] |
| example-testing | 67.743 [67.549, 69.943] | 67.608 [67.990, 70.095] | 69.407 [66.947, 69.638] |
| adding-knowledge | 68.933 [68.372, 70.423] | 68.281 [67.589, 69.362] | 67.654 [65.563, 68.130] |

## Null hypothesis hierarchy (top-10 variance ratio)

Primary: chain-stratified permutation. Secondary: cross-chain permutation. Tertiary: MP isotropic (finite-sample inflation diagnostic only).

| Behaviour | real | chain-strat null (mean, 95% CI), p | cross-chain null (mean, p) | MP null mean |
|-----------|------|-------------------------------------|----------------------------|--------------|
| backtracking | 0.4155 | 0.3893 [0.3877, 0.3910], p=0.0050 | 0.3849, p=0.0050 | 0.0123 |
| uncertainty-estimation | 0.3912 | 0.3812 [0.3802, 0.3823], p=0.0050 | 0.3845, p=0.0050 | 0.0109 |
| example-testing | 0.4261 | 0.4155 [0.4133, 0.4175], p=0.0050 | 0.3854, p=0.0050 | 0.0146 |
| adding-knowledge | 0.3624 | 0.3874 [0.3851, 0.3894], p=1.0000 | 0.3857, p=1.0000 | 0.0154 |