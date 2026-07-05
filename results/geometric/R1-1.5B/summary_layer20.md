# Phase 5b diagnostics - R1-1.5B layer 20

k sweep: [10, 30, 100]
PCA topk for curvature: 50
Null hierarchy resamples: 200

## Intrinsic dimension estimates

| Behaviour | N | TwoNN | Levina-Bickel | Correlation dim |
|-----------|---|-------|---------------|------------------|
| backtracking | 10136 | 3.8 [3.2, 3.3] | 3.2 [3.1, 3.2] | 5.9 [5.8, 6.0] |
| uncertainty-estimation | 16545 | 4.1 [3.4, 3.5] | 3.0 [2.9, 3.0] | 5.9 [5.8, 6.0] |
| example-testing | 5766 | 4.0 [3.4, 3.6] | 5.4 [5.1, 5.4] | 7.2 [7.1, 7.3] |
| adding-knowledge | 4989 | 4.2 [3.6, 3.9] | 7.5 [7.3, 7.7] | 8.3 [8.2, 8.4] |

## Curvature diagnostics (post-PCA projection)

Local-vs-global PCA dim ratio (close to 1 = flat, lower = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 0.512 [0.523, 0.605] | 0.485 [0.503, 0.556] | 0.530 [0.553, 0.611] |
| uncertainty-estimation | 0.590 [0.523, 0.608] | 0.531 [0.490, 0.577] | 0.564 [0.533, 0.603] |
| example-testing | 0.629 [0.595, 0.671] | 0.585 [0.565, 0.639] | 0.609 [0.609, 0.662] |
| adding-knowledge | 0.661 [0.645, 0.716] | 0.624 [0.616, 0.678] | 0.646 [0.635, 0.688] |

Geodesic / Euclidean ratio (close to 1 = flat, higher = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 3.312 [3.080, 3.256] | 2.280 [2.152, 2.223] | 1.749 [1.676, 1.717] |
| uncertainty-estimation | 3.309 [3.125, 3.311] | 2.332 [2.206, 2.278] | 1.795 [1.721, 1.773] |
| example-testing | 3.194 [2.992, 3.146] | 2.198 [2.093, 2.161] | 1.719 [1.652, 1.693] |
| adding-knowledge | 2.967 [2.776, 2.928] | 2.126 [2.034, 2.094] | 1.673 [1.618, 1.665] |

Tangent-space variation (degrees, close to 0 = flat, higher = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 68.865 [67.274, 70.460] | 69.233 [68.138, 70.414] | 68.809 [66.973, 69.711] |
| uncertainty-estimation | 68.712 [67.483, 70.945] | 68.895 [68.592, 70.841] | 68.703 [68.054, 70.088] |
| example-testing | 68.733 [68.630, 71.178] | 69.665 [68.585, 70.840] | 69.521 [67.110, 69.421] |
| adding-knowledge | 71.233 [69.651, 71.845] | 70.881 [69.276, 70.795] | 68.826 [66.297, 69.061] |

## Null hypothesis hierarchy (top-10 variance ratio)

Primary: chain-stratified permutation. Secondary: cross-chain permutation. Tertiary: MP isotropic (finite-sample inflation diagnostic only).

| Behaviour | real | chain-strat null (mean, 95% CI), p | cross-chain null (mean, p) | MP null mean |
|-----------|------|-------------------------------------|----------------------------|--------------|
| backtracking | 0.4587 | 0.4315 [0.4303, 0.4329], p=0.0050 | 0.4293, p=0.0050 | 0.0123 |
| uncertainty-estimation | 0.4455 | 0.4333 [0.4323, 0.4341], p=0.0050 | 0.4289, p=0.0050 | 0.0109 |
| example-testing | 0.4441 | 0.4420 [0.4402, 0.4436], p=0.0100 | 0.4299, p=0.0050 | 0.0146 |
| adding-knowledge | 0.4163 | 0.4269 [0.4252, 0.4288], p=1.0000 | 0.4303, p=1.0000 | 0.0154 |