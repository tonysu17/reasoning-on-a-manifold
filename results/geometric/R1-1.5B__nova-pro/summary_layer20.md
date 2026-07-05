# Phase 5b diagnostics - R1-1.5B__nova-pro layer 20

k sweep: [10, 30, 100]
PCA topk for curvature: 50
Null hierarchy resamples: 200

## Intrinsic dimension estimates

| Behaviour | N | TwoNN | Levina-Bickel | Correlation dim |
|-----------|---|-------|---------------|------------------|
| backtracking | 5519 | 5.1 [5.1, 5.4] | 10.0 [10.1, 10.3] | 7.4 [7.2, 7.5] |
| uncertainty-estimation | 5754 | 6.3 [6.3, 6.7] | 11.7 [11.7, 12.0] | 7.9 [7.8, 8.0] |
| example-testing | 2364 | 5.7 [5.5, 6.0] | 9.7 [9.7, 10.0] | 8.2 [8.2, 8.3] |
| adding-knowledge | 5715 | 7.8 [7.8, 8.3] | 13.3 [13.4, 13.6] | 9.3 [9.2, 9.4] |

## Curvature diagnostics (post-PCA projection)

Local-vs-global PCA dim ratio (close to 1 = flat, lower = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 0.850 [0.832, 0.868] | 0.784 [0.783, 0.808] | 0.773 [0.773, 0.796] |
| uncertainty-estimation | 0.876 [0.866, 0.895] | 0.796 [0.808, 0.830] | 0.781 [0.785, 0.806] |
| example-testing | 0.833 [0.826, 0.862] | 0.778 [0.772, 0.798] | 0.753 [0.760, 0.782] |
| adding-knowledge | 0.900 [0.877, 0.911] | 0.834 [0.811, 0.840] | 0.786 [0.774, 0.795] |

Geodesic / Euclidean ratio (close to 1 = flat, higher = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 2.622 [2.528, 2.597] | 1.983 [1.911, 1.963] | 1.618 [1.562, 1.601] |
| uncertainty-estimation | 2.568 [2.476, 2.550] | 1.970 [1.906, 1.960] | 1.611 [1.559, 1.597] |
| example-testing | 2.584 [2.445, 2.543] | 1.950 [1.844, 1.904] | 1.581 [1.503, 1.549] |
| adding-knowledge | 2.721 [2.641, 2.733] | 2.084 [2.001, 2.059] | 1.686 [1.622, 1.664] |

Tangent-space variation (degrees, close to 0 = flat, higher = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 71.064 [70.143, 71.540] | 69.115 [68.122, 69.785] | 65.812 [64.300, 66.601] |
| uncertainty-estimation | 70.787 [70.784, 71.906] | 68.720 [68.427, 70.259] | 65.586 [64.504, 66.851] |
| example-testing | 70.733 [70.117, 71.534] | 69.484 [68.289, 70.382] | 66.016 [63.794, 66.994] |
| adding-knowledge | 71.453 [70.713, 71.900] | 69.891 [68.881, 70.364] | 66.738 [65.338, 67.260] |

## Null hypothesis hierarchy (top-10 variance ratio)

Primary: chain-stratified permutation. Secondary: cross-chain permutation. Tertiary: MP isotropic (finite-sample inflation diagnostic only).

| Behaviour | real | chain-strat null (mean, 95% CI), p | cross-chain null (mean, p) | MP null mean |
|-----------|------|-------------------------------------|----------------------------|--------------|
| backtracking | 0.3904 | 0.3753 [0.3735, 0.3772], p=0.0050 | 0.3743, p=0.0050 | 0.0150 |
| uncertainty-estimation | 0.3702 | 0.3707 [0.3686, 0.3727], p=0.7065 | 0.3747, p=1.0000 | 0.0151 |
| example-testing | 0.3977 | 0.3936 [0.3907, 0.3965], p=0.0100 | 0.3765, p=0.0050 | 0.0212 |
| adding-knowledge | 0.3633 | 0.3751 [0.3735, 0.3767], p=1.0000 | 0.3744, p=1.0000 | 0.0147 |