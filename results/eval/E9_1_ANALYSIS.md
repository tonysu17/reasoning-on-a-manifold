# E9.1 analysis (annotation-free endpoints)

Records: 6250 (E8 greedy α=1 cells merged as the middle dose)

## Per-cell endpoints (decode | behaviour | arm | α)

| cell | n | collapse | mean rep | mean tok | boxed | boxed(struct) |
|---|--:|--:|--:|--:|--:|--:|
| T06|backtracking|energy_matched_random|0.5 | 150 | 0.06 | 0.273 | 4954 | 0.193 | 0.433 |
| T06|backtracking|energy_matched_random|1.0 | 150 | 0.053 | 0.273 | 4945 | 0.167 | 0.333 |
| T06|backtracking|energy_matched_random|1.5 | 150 | 0.073 | 0.289 | 4836 | 0.18 | 0.4 |
| T06|backtracking|manifold_k5|0.5 | 150 | 0.067 | 0.255 | 4483 | 0.22 | 0.5 |
| T06|backtracking|manifold_k5|1.0 | 150 | 0.06 | 0.221 | 3867 | 0.253 | 0.6 |
| T06|backtracking|manifold_k5|1.5 | 150 | 0.027 | 0.221 | 3380 | 0.347 | 0.7 |
| T06|backtracking|random_subspace_k5|0.5 | 300 | 0.053 | 0.265 | 4910 | 0.187 | 0.317 |
| T06|backtracking|random_subspace_k5|1.0 | 300 | 0.047 | 0.259 | 4809 | 0.197 | 0.367 |
| T06|backtracking|random_subspace_k5|1.5 | 300 | 0.053 | 0.271 | 4908 | 0.173 | 0.35 |
| T06|backtracking|single_direction|0.5 | 150 | 0.047 | 0.255 | 4521 | 0.24 | 0.567 |
| T06|backtracking|single_direction|1.0 | 150 | 0.053 | 0.241 | 4249 | 0.293 | 0.533 |
| T06|backtracking|single_direction|1.5 | 150 | 0.04 | 0.247 | 3867 | 0.293 | 0.667 |
| T06|example-testing|energy_matched_random|0.5 | 150 | 0.067 | 0.289 | 5023 | 0.207 | 0.433 |
| T06|example-testing|energy_matched_random|1.0 | 150 | 0.08 | 0.266 | 4746 | 0.173 | 0.367 |
| T06|example-testing|energy_matched_random|1.5 | 150 | 0.147 | 0.367 | 5507 | 0.14 | 0.3 |
| T06|example-testing|manifold_k5|0.5 | 150 | 0.06 | 0.293 | 5284 | 0.173 | 0.4 |
| T06|example-testing|manifold_k5|1.0 | 150 | 0.12 | 0.327 | 5680 | 0.18 | 0.4 |
| T06|example-testing|manifold_k5|1.5 | 150 | 0.213 | 0.497 | 6923 | 0.12 | 0.367 |
| T06|example-testing|random_subspace_k5|0.5 | 300 | 0.027 | 0.254 | 4714 | 0.177 | 0.35 |
| T06|example-testing|random_subspace_k5|1.0 | 300 | 0.053 | 0.265 | 4828 | 0.197 | 0.417 |
| T06|example-testing|random_subspace_k5|1.5 | 300 | 0.047 | 0.262 | 4659 | 0.233 | 0.483 |
| T06|example-testing|single_direction|0.5 | 150 | 0.047 | 0.297 | 5290 | 0.16 | 0.333 |
| T06|example-testing|single_direction|1.0 | 150 | 0.1 | 0.344 | 5681 | 0.18 | 0.4 |
| T06|example-testing|single_direction|1.5 | 150 | 0.133 | 0.427 | 6387 | 0.113 | 0.2 |
| T06|shared|vanilla|0.0 | 150 | 0.06 | 0.271 | 4905 | 0.16 | 0.367 |
| greedy|backtracking|energy_matched_random|0.5 | 50 | 0.42 | 0.548 | 5902 | 0.06 | 0.2 |
| greedy|backtracking|energy_matched_random|1.0 | 50 | 0.28 | 0.489 | 5582 | 0.12 | 0.2 |
| greedy|backtracking|energy_matched_random|1.5 | 50 | 0.3 | 0.449 | 5108 | 0.08 | 0.2 |
| greedy|backtracking|manifold_k5|0.5 | 50 | 0.3 | 0.434 | 4636 | 0.22 | 0.5 |
| greedy|backtracking|manifold_k5|1.0 | 50 | 0.24 | 0.438 | 4504 | 0.24 | 0.4 |
| greedy|backtracking|manifold_k5|1.5 | 50 | 0.38 | 0.484 | 4761 | 0.2 | 0.4 |
| greedy|backtracking|random_subspace_k5|0.5 | 100 | 0.28 | 0.484 | 5452 | 0.13 | 0.25 |
| greedy|backtracking|random_subspace_k5|1.0 | 100 | 0.37 | 0.538 | 5810 | 0.14 | 0.2 |
| greedy|backtracking|random_subspace_k5|1.5 | 100 | 0.31 | 0.498 | 5580 | 0.09 | 0.25 |
| greedy|backtracking|single_direction|0.5 | 50 | 0.28 | 0.449 | 5070 | 0.14 | 0.4 |
| greedy|backtracking|single_direction|1.0 | 50 | 0.36 | 0.47 | 5051 | 0.2 | 0.3 |
| greedy|backtracking|single_direction|1.5 | 50 | 0.38 | 0.464 | 4849 | 0.18 | 0.5 |
| greedy|example-testing|energy_matched_random|0.5 | 50 | 0.38 | 0.546 | 5990 | 0.08 | 0.3 |
| greedy|example-testing|energy_matched_random|1.0 | 50 | 0.38 | 0.561 | 5971 | 0.08 | 0.2 |
| greedy|example-testing|energy_matched_random|1.5 | 50 | 0.52 | 0.637 | 6462 | 0.04 | 0.2 |
| greedy|example-testing|manifold_k5|0.5 | 50 | 0.42 | 0.626 | 6684 | 0.06 | 0.2 |
| greedy|example-testing|manifold_k5|1.0 | 50 | 0.64 | 0.737 | 7240 | 0.04 | 0.1 |
| greedy|example-testing|manifold_k5|1.5 | 50 | 0.76 | 0.8 | 7521 | 0.0 | 0.0 |
| greedy|example-testing|random_subspace_k5|0.5 | 100 | 0.33 | 0.527 | 5582 | 0.09 | 0.15 |
| greedy|example-testing|random_subspace_k5|1.0 | 100 | 0.36 | 0.534 | 5721 | 0.13 | 0.3 |
| greedy|example-testing|random_subspace_k5|1.5 | 100 | 0.3 | 0.49 | 5491 | 0.11 | 0.2 |
| greedy|example-testing|single_direction|0.5 | 50 | 0.36 | 0.544 | 5991 | 0.1 | 0.2 |
| greedy|example-testing|single_direction|1.0 | 50 | 0.48 | 0.643 | 6610 | 0.04 | 0.0 |
| greedy|example-testing|single_direction|1.5 | 50 | 0.7 | 0.754 | 7283 | 0.0 | 0.0 |
| greedy|shared|vanilla|0.0 | 100 | 0.35 | 0.535 | 5944 | 0.06 | 0.15 |

## Cross-sample diversity at T=0.6 (1 − mean pairwise 4-gram Jaccard)

| behaviour | arm | α | n tasks | diversity | P10 |
|---|---|--:|--:|--:|--:|
| backtracking | energy_matched_random | 0.5 | 50 | 0.96 | 0.929 |
| backtracking | energy_matched_random | 1.0 | 50 | 0.961 | 0.927 |
| backtracking | energy_matched_random | 1.5 | 50 | 0.96 | 0.933 |
| backtracking | manifold_k5 | 0.5 | 50 | 0.961 | 0.935 |
| backtracking | manifold_k5 | 1.0 | 50 | 0.959 | 0.922 |
| backtracking | manifold_k5 | 1.5 | 50 | 0.966 | 0.939 |
| backtracking | random_subspace_k5 | 0.5 | 50 | 0.953 | 0.922 |
| backtracking | random_subspace_k5 | 1.0 | 50 | 0.956 | 0.93 |
| backtracking | random_subspace_k5 | 1.5 | 50 | 0.957 | 0.931 |
| backtracking | single_direction | 0.5 | 50 | 0.961 | 0.927 |
| backtracking | single_direction | 1.0 | 50 | 0.963 | 0.936 |
| backtracking | single_direction | 1.5 | 50 | 0.962 | 0.937 |
| example-testing | energy_matched_random | 0.5 | 50 | 0.96 | 0.921 |
| example-testing | energy_matched_random | 1.0 | 50 | 0.96 | 0.927 |
| example-testing | energy_matched_random | 1.5 | 50 | 0.957 | 0.924 |
| example-testing | manifold_k5 | 0.5 | 50 | 0.96 | 0.929 |
| example-testing | manifold_k5 | 1.0 | 50 | 0.957 | 0.921 |
| example-testing | manifold_k5 | 1.5 | 50 | 0.948 | 0.916 |
| example-testing | random_subspace_k5 | 0.5 | 50 | 0.956 | 0.927 |
| example-testing | random_subspace_k5 | 1.0 | 50 | 0.958 | 0.929 |
| example-testing | random_subspace_k5 | 1.5 | 50 | 0.958 | 0.927 |
| example-testing | single_direction | 0.5 | 50 | 0.958 | 0.924 |
| example-testing | single_direction | 1.0 | 50 | 0.954 | 0.927 |
| example-testing | single_direction | 1.5 | 50 | 0.948 | 0.914 |
| shared | vanilla | 0.0 | 50 | 0.962 | 0.931 |

## P2 — rescue of greedy-α=1 collapses at T=0.6

| behaviour | arm | collapsed (greedy) | scored | rescued | rate |
|---|---|--:|--:|--:|--:|
| example-testing | single_direction | 24 | 24 | 20 | 0.833 |
| example-testing | manifold_k5 | 32 | 32 | 26 | 0.812 |
| example-testing | random_subspace_k5 | 18 | 18 | 17 | 0.944 |
| example-testing | energy_matched_random | 19 | 19 | 16 | 0.842 |
| backtracking | single_direction | 18 | 18 | 17 | 0.944 |
| backtracking | manifold_k5 | 12 | 12 | 9 | 0.75 |
| backtracking | random_subspace_k5 | 15 | 15 | 13 | 0.867 |
| backtracking | energy_matched_random | 14 | 14 | 14 | 1.0 |
