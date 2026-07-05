# Phase 5b diagnostics - R1-1.5B__nova-pro layer 11

k sweep: [10, 30, 100]
PCA topk for curvature: 50
Null hierarchy resamples: 200

## Intrinsic dimension estimates

| Behaviour | N | TwoNN | Levina-Bickel | Correlation dim |
|-----------|---|-------|---------------|------------------|
| backtracking | 5519 | 5.2 [5.1, 5.5] | 10.5 [10.6, 10.9] | 7.0 [6.9, 7.1] |
| uncertainty-estimation | 5754 | 6.3 [6.4, 6.9] | 11.7 [11.7, 12.0] | 7.3 [7.3, 7.4] |
| example-testing | 2364 | 5.9 [5.4, 6.0] | 9.8 [9.7, 10.1] | 7.2 [7.1, 7.3] |
| adding-knowledge | 5715 | 7.3 [7.3, 7.8] | 13.8 [13.8, 14.1] | 8.8 [8.7, 8.9] |

## Curvature diagnostics (post-PCA projection)

Local-vs-global PCA dim ratio (close to 1 = flat, lower = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 0.852 [0.852, 0.883] | 0.807 [0.800, 0.829] | 0.774 [0.776, 0.801] |
| uncertainty-estimation | 0.868 [0.866, 0.897] | 0.801 [0.810, 0.833] | 0.777 [0.786, 0.804] |
| example-testing | 0.809 [0.803, 0.850] | 0.758 [0.762, 0.794] | 0.737 [0.750, 0.772] |
| adding-knowledge | 0.890 [0.873, 0.911] | 0.837 [0.808, 0.847] | 0.789 [0.778, 0.807] |

Geodesic / Euclidean ratio (close to 1 = flat, higher = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 2.657 [2.566, 2.646] | 2.017 [1.951, 2.007] | 1.635 [1.580, 1.624] |
| uncertainty-estimation | 2.617 [2.534, 2.617] | 1.994 [1.933, 1.984] | 1.622 [1.567, 1.608] |
| example-testing | 2.634 [2.450, 2.556] | 1.925 [1.821, 1.875] | 1.558 [1.489, 1.535] |
| adding-knowledge | 2.710 [2.615, 2.705] | 2.063 [2.001, 2.055] | 1.678 [1.622, 1.662] |

Tangent-space variation (degrees, close to 0 = flat, higher = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 70.621 [69.537, 71.155] | 68.134 [67.245, 69.026] | 64.488 [62.911, 65.176] |
| uncertainty-estimation | 70.515 [70.200, 71.561] | 68.158 [67.565, 69.421] | 64.538 [63.353, 65.910] |
| example-testing | 70.227 [70.125, 71.859] | 69.235 [68.336, 70.876] | 66.095 [63.823, 67.344] |
| adding-knowledge | 71.318 [71.198, 72.362] | 69.792 [68.913, 70.671] | 66.423 [65.185, 67.529] |

## Null hypothesis hierarchy (top-10 variance ratio)

Primary: chain-stratified permutation. Secondary: cross-chain permutation. Tertiary: MP isotropic (finite-sample inflation diagnostic only).

| Behaviour | real | chain-strat null (mean, 95% CI), p | cross-chain null (mean, p) | MP null mean |
|-----------|------|-------------------------------------|----------------------------|--------------|
| backtracking | 0.4191 | 0.3867 [0.3849, 0.3887], p=0.0050 | 0.3799, p=0.0050 | 0.0150 |
| uncertainty-estimation | 0.3919 | 0.3794 [0.3772, 0.3813], p=0.0050 | 0.3803, p=0.0050 | 0.0151 |
| example-testing | 0.3943 | 0.3968 [0.3942, 0.4003], p=0.9602 | 0.3823, p=0.0050 | 0.0212 |
| adding-knowledge | 0.3498 | 0.3735 [0.3716, 0.3753], p=1.0000 | 0.3799, p=1.0000 | 0.0147 |