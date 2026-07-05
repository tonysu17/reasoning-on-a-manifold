# Phase 5b diagnostics - R1-1.5B__nova-pro layer 17

k sweep: [10, 30, 100]
PCA topk for curvature: 50
Null hierarchy resamples: 200

## Intrinsic dimension estimates

| Behaviour | N | TwoNN | Levina-Bickel | Correlation dim |
|-----------|---|-------|---------------|------------------|
| backtracking | 5519 | 5.4 [5.3, 5.7] | 10.8 [10.9, 11.1] | 6.8 [6.7, 6.9] |
| uncertainty-estimation | 5754 | 6.6 [6.6, 7.1] | 12.4 [12.4, 12.7] | 7.4 [7.3, 7.5] |
| example-testing | 2364 | 5.8 [5.5, 6.1] | 10.2 [10.2, 10.5] | 7.6 [7.6, 7.7] |
| adding-knowledge | 5715 | 8.0 [7.9, 8.5] | 13.9 [14.0, 14.3] | 8.7 [8.6, 8.8] |

## Curvature diagnostics (post-PCA projection)

Local-vs-global PCA dim ratio (close to 1 = flat, lower = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 0.867 [0.857, 0.892] | 0.821 [0.823, 0.846] | 0.801 [0.807, 0.831] |
| uncertainty-estimation | 0.897 [0.882, 0.911] | 0.823 [0.834, 0.854] | 0.805 [0.811, 0.830] |
| example-testing | 0.848 [0.835, 0.873] | 0.798 [0.792, 0.821] | 0.780 [0.786, 0.808] |
| adding-knowledge | 0.911 [0.893, 0.925] | 0.853 [0.835, 0.868] | 0.816 [0.807, 0.831] |

Geodesic / Euclidean ratio (close to 1 = flat, higher = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 2.594 [2.479, 2.551] | 1.958 [1.895, 1.947] | 1.604 [1.555, 1.593] |
| uncertainty-estimation | 2.533 [2.455, 2.534] | 1.960 [1.891, 1.941] | 1.596 [1.553, 1.589] |
| example-testing | 2.566 [2.414, 2.500] | 1.927 [1.834, 1.888] | 1.570 [1.503, 1.545] |
| adding-knowledge | 2.697 [2.597, 2.684] | 2.059 [1.988, 2.037] | 1.683 [1.615, 1.655] |

Tangent-space variation (degrees, close to 0 = flat, higher = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 70.641 [69.833, 71.171] | 68.209 [67.412, 69.071] | 64.007 [62.344, 64.711] |
| uncertainty-estimation | 70.525 [70.269, 71.823] | 68.082 [67.828, 69.651] | 63.881 [63.102, 65.682] |
| example-testing | 70.190 [69.589, 71.074] | 69.077 [67.270, 69.773] | 64.572 [62.195, 65.529] |
| adding-knowledge | 71.362 [70.447, 71.770] | 69.420 [68.253, 69.830] | 66.080 [63.907, 66.082] |

## Null hypothesis hierarchy (top-10 variance ratio)

Primary: chain-stratified permutation. Secondary: cross-chain permutation. Tertiary: MP isotropic (finite-sample inflation diagnostic only).

| Behaviour | real | chain-strat null (mean, 95% CI), p | cross-chain null (mean, p) | MP null mean |
|-----------|------|-------------------------------------|----------------------------|--------------|
| backtracking | 0.4134 | 0.3938 [0.3918, 0.3960], p=0.0050 | 0.3940, p=0.0050 | 0.0150 |
| uncertainty-estimation | 0.3883 | 0.3896 [0.3875, 0.3919], p=0.8856 | 0.3944, p=1.0000 | 0.0151 |
| example-testing | 0.4129 | 0.4126 [0.4102, 0.4154], p=0.4229 | 0.3963, p=0.0050 | 0.0212 |
| adding-knowledge | 0.3873 | 0.3976 [0.3960, 0.3993], p=1.0000 | 0.3941, p=1.0000 | 0.0147 |