# Phase 5b diagnostics - R1-1.5B__nova-pro layer 27

k sweep: [10, 30, 100]
PCA topk for curvature: 50
Null hierarchy resamples: 200

## Intrinsic dimension estimates

| Behaviour | N | TwoNN | Levina-Bickel | Correlation dim |
|-----------|---|-------|---------------|------------------|
| backtracking | 5519 | 4.7 [4.6, 4.9] | 8.4 [8.5, 8.7] | 7.3 [7.1, 7.4] |
| uncertainty-estimation | 5754 | 5.7 [5.6, 6.1] | 10.3 [10.4, 10.6] | 7.6 [7.5, 7.7] |
| example-testing | 2364 | 5.6 [5.3, 5.9] | 8.2 [8.2, 8.4] | 6.0 [5.9, 6.1] |
| adding-knowledge | 5715 | 6.9 [6.9, 7.3] | 11.1 [11.1, 11.3] | 7.6 [7.5, 7.7] |

## Curvature diagnostics (post-PCA projection)

Local-vs-global PCA dim ratio (close to 1 = flat, lower = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 0.800 [0.785, 0.819] | 0.723 [0.720, 0.746] | 0.702 [0.706, 0.731] |
| uncertainty-estimation | 0.841 [0.839, 0.870] | 0.764 [0.776, 0.801] | 0.743 [0.753, 0.774] |
| example-testing | 0.812 [0.800, 0.838] | 0.709 [0.713, 0.742] | 0.660 [0.674, 0.699] |
| adding-knowledge | 0.862 [0.848, 0.883] | 0.762 [0.751, 0.783] | 0.704 [0.696, 0.723] |

Geodesic / Euclidean ratio (close to 1 = flat, higher = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 2.688 [2.589, 2.675] | 2.000 [1.933, 1.982] | 1.611 [1.560, 1.600] |
| uncertainty-estimation | 2.619 [2.531, 2.603] | 1.991 [1.917, 1.965] | 1.607 [1.564, 1.598] |
| example-testing | 2.582 [2.403, 2.509] | 1.875 [1.785, 1.839] | 1.520 [1.446, 1.491] |
| adding-knowledge | 2.709 [2.573, 2.664] | 2.016 [1.933, 1.987] | 1.621 [1.566, 1.604] |

Tangent-space variation (degrees, close to 0 = flat, higher = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 69.640 [68.726, 70.532] | 68.650 [67.478, 69.339] | 66.398 [64.613, 67.136] |
| uncertainty-estimation | 70.333 [69.626, 71.167] | 68.192 [67.663, 69.560] | 65.583 [64.390, 66.858] |
| example-testing | 68.900 [69.106, 71.048] | 68.533 [67.439, 70.213] | 66.791 [63.798, 67.650] |
| adding-knowledge | 70.668 [69.837, 71.531] | 69.161 [67.541, 69.742] | 65.965 [64.532, 67.773] |

## Null hypothesis hierarchy (top-10 variance ratio)

Primary: chain-stratified permutation. Secondary: cross-chain permutation. Tertiary: MP isotropic (finite-sample inflation diagnostic only).

| Behaviour | real | chain-strat null (mean, 95% CI), p | cross-chain null (mean, p) | MP null mean |
|-----------|------|-------------------------------------|----------------------------|--------------|
| backtracking | 0.3819 | 0.3777 [0.3755, 0.3803], p=0.0050 | 0.3820, p=0.4925 | 0.0150 |
| uncertainty-estimation | 0.3737 | 0.3782 [0.3758, 0.3804], p=1.0000 | 0.3825, p=1.0000 | 0.0151 |
| example-testing | 0.4225 | 0.4164 [0.4135, 0.4198], p=0.0050 | 0.3839, p=0.0050 | 0.0212 |
| adding-knowledge | 0.3646 | 0.3844 [0.3822, 0.3864], p=1.0000 | 0.3821, p=1.0000 | 0.0147 |