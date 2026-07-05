# Phase 5b diagnostics - R1-1.5B layer 14

k sweep: [10, 30, 100]
PCA topk for curvature: 50
Null hierarchy resamples: 200

## Intrinsic dimension estimates

| Behaviour | N | TwoNN | Levina-Bickel | Correlation dim |
|-----------|---|-------|---------------|------------------|
| backtracking | 10136 | 3.9 [3.3, 3.4] | 3.3 [3.1, 3.3] | 5.9 [5.8, 5.9] |
| uncertainty-estimation | 16545 | 4.3 [3.6, 3.7] | 3.0 [2.9, 3.0] | 6.2 [6.1, 6.3] |
| example-testing | 5766 | 4.1 [3.6, 3.8] | 5.6 [5.2, 5.6] | 6.5 [6.4, 6.5] |
| adding-knowledge | 4989 | 4.5 [3.8, 4.1] | 7.9 [7.6, 8.1] | 8.1 [8.0, 8.2] |

## Curvature diagnostics (post-PCA projection)

Local-vs-global PCA dim ratio (close to 1 = flat, lower = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 0.551 [0.555, 0.637] | 0.517 [0.532, 0.594] | 0.559 [0.576, 0.639] |
| uncertainty-estimation | 0.599 [0.531, 0.621] | 0.557 [0.510, 0.599] | 0.591 [0.555, 0.627] |
| example-testing | 0.640 [0.625, 0.704] | 0.616 [0.594, 0.675] | 0.637 [0.631, 0.691] |
| adding-knowledge | 0.693 [0.668, 0.743] | 0.666 [0.658, 0.710] | 0.688 [0.671, 0.731] |

Geodesic / Euclidean ratio (close to 1 = flat, higher = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 3.277 [3.032, 3.169] | 2.307 [2.165, 2.236] | 1.765 [1.688, 1.730] |
| uncertainty-estimation | 3.221 [3.089, 3.253] | 2.333 [2.241, 2.328] | 1.791 [1.742, 1.788] |
| example-testing | 3.029 [2.859, 2.999] | 2.154 [2.065, 2.140] | 1.688 [1.625, 1.660] |
| adding-knowledge | 2.900 [2.724, 2.869] | 2.059 [1.992, 2.054] | 1.637 [1.597, 1.643] |

Tangent-space variation (degrees, close to 0 = flat, higher = curved):

| Behaviour | k=10 | k=30 | k=100 |
|-----------|------|------|--------|
| backtracking | 66.779 [65.081, 68.821] | 67.242 [65.891, 68.816] | 66.627 [65.149, 67.713] |
| uncertainty-estimation | 66.255 [65.048, 69.519] | 66.767 [66.727, 69.359] | 67.357 [66.556, 68.634] |
| example-testing | 67.354 [66.962, 70.716] | 68.675 [67.698, 69.931] | 68.560 [65.425, 68.567] |
| adding-knowledge | 70.614 [69.002, 71.650] | 70.394 [68.734, 70.489] | 67.981 [65.313, 68.118] |

## Null hypothesis hierarchy (top-10 variance ratio)

Primary: chain-stratified permutation. Secondary: cross-chain permutation. Tertiary: MP isotropic (finite-sample inflation diagnostic only).

| Behaviour | real | chain-strat null (mean, 95% CI), p | cross-chain null (mean, p) | MP null mean |
|-----------|------|-------------------------------------|----------------------------|--------------|
| backtracking | 0.4993 | 0.4663 [0.4648, 0.4677], p=0.0050 | 0.4634, p=0.0050 | 0.0123 |
| uncertainty-estimation | 0.4835 | 0.4692 [0.4684, 0.4701], p=0.0050 | 0.4632, p=0.0050 | 0.0109 |
| example-testing | 0.4724 | 0.4729 [0.4711, 0.4746], p=0.6816 | 0.4640, p=0.0050 | 0.0146 |
| adding-knowledge | 0.4450 | 0.4578 [0.4559, 0.4596], p=1.0000 | 0.4644, p=1.0000 | 0.0154 |