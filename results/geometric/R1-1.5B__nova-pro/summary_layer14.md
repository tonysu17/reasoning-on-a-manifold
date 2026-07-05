# Phase 5b diagnostics - R1-1.5B__nova-pro layer 14

k sweep: [10, 30, 100]
PCA topk for curvature: 50
Null hierarchy resamples: 200

## Intrinsic dimension estimates

| Behaviour | N | TwoNN | Levina-Bickel | Correlation dim |
|-----------|---|-------|---------------|------------------|
| backtracking | 5519 | 5.5 [5.5, 5.9] | 11.2 [11.3, 11.6] | 6.6 [6.5, 6.8] |
| uncertainty-estimation | 5754 | 6.7 [6.8, 7.3] | 12.4 [12.4, 12.7] | 7.2 [7.1, 7.3] |
| example-testing | 2364 | 5.9 [5.5, 6.2] | 10.4 [10.4, 10.8] | 7.4 [7.3, 7.5] |
| adding-knowledge | 5715 | 7.9 [7.9, 8.5] | 14.6 [14.6, 14.9] | 8.6 [8.5, 8.7] |

## Curvature diagnostics (post-PCA projection)

Local-vs-global PCA dim ratio (close to 1 = flat, lower = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 0.879 [0.873, 0.906] | 0.829 [0.829, 0.855] | 0.803 [0.805, 0.829] |
| uncertainty-estimation | 0.895 [0.885, 0.915] | 0.832 [0.839, 0.863] | 0.809 [0.813, 0.833] |
| example-testing | 0.839 [0.826, 0.868] | 0.794 [0.792, 0.824] | 0.781 [0.788, 0.813] |
| adding-knowledge | 0.916 [0.894, 0.924] | 0.857 [0.838, 0.873] | 0.825 [0.813, 0.838] |

Geodesic / Euclidean ratio (close to 1 = flat, higher = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 2.605 [2.514, 2.587] | 1.976 [1.922, 1.969] | 1.613 [1.565, 1.601] |
| uncertainty-estimation | 2.582 [2.484, 2.568] | 1.968 [1.910, 1.966] | 1.617 [1.564, 1.599] |
| example-testing | 2.566 [2.420, 2.509] | 1.919 [1.824, 1.883] | 1.562 [1.498, 1.540] |
| adding-knowledge | 2.688 [2.589, 2.668] | 2.038 [1.979, 2.029] | 1.669 [1.611, 1.649] |

Tangent-space variation (degrees, close to 0 = flat, higher = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 70.696 [69.734, 71.035] | 68.218 [67.139, 68.910] | 64.171 [62.572, 64.750] |
| uncertainty-estimation | 70.384 [70.158, 71.488] | 67.220 [67.374, 69.222] | 63.961 [62.736, 65.524] |
| example-testing | 70.503 [69.814, 71.467] | 68.995 [67.633, 70.003] | 65.127 [62.663, 66.219] |
| adding-knowledge | 71.261 [70.760, 72.043] | 69.195 [68.698, 69.974] | 65.929 [64.217, 66.348] |

## Null hypothesis hierarchy (top-10 variance ratio)

Primary: chain-stratified permutation. Secondary: cross-chain permutation. Tertiary: MP isotropic (finite-sample inflation diagnostic only).

| Behaviour | real | chain-strat null (mean, 95% CI), p | cross-chain null (mean, p) | MP null mean |
|-----------|------|-------------------------------------|----------------------------|--------------|
| backtracking | 0.4215 | 0.3985 [0.3966, 0.4006], p=0.0050 | 0.3953, p=0.0050 | 0.0150 |
| uncertainty-estimation | 0.4008 | 0.3942 [0.3918, 0.3963], p=0.0050 | 0.3957, p=0.0050 | 0.0151 |
| example-testing | 0.4105 | 0.4133 [0.4104, 0.4165], p=0.9701 | 0.3976, p=0.0050 | 0.0212 |
| adding-knowledge | 0.3710 | 0.3905 [0.3886, 0.3923], p=1.0000 | 0.3954, p=1.0000 | 0.0147 |