# A1 — belief-state trajectory (logit lens, sparse arm)

Collapse classes over corpus: loop 415 / clean 469 / ambiguous 116 (ambiguous dropped).

Sparse arm: lens applied to mean-pooled span vectors (smeared belief; M4 — suggestive, dense per-token arm confirmatory). Entropy in nats.

## layer 14  (chains: 984 pre-onset / 986 full)
### Collapse (primary, leak-guarded)
- chains: 868 (pos 410 / neg 458)
| feature set | AUC (oof) | fold mean±std | stable | n_pos/n |
|---|---|---|---|---|
| belief_all | 0.749 | 0.751±0.029 | True | 410/868 |
| belief_trajectory | 0.681 | 0.685±0.018 | True | 410/868 |
| belief_occupancy | 0.754 | 0.755±0.030 | True | 410/868 |
| length_gap_trunc | 0.986 | 0.989±0.007 | True | 410/868 |
| persistence_step | 0.856 | 0.857±0.007 | True | 410/868 |
| rung0_curvature | 0.886 | 0.887±0.013 | True | 410/868 |
| length_PLUS_belief_traj | 0.987 | 0.990±0.007 | True | 410/868 |
| length_PLUS_belief_all | 0.991 | 0.993±0.006 | True | 410/868 |

- **M3 order test** (trajectory features): real 0.681 vs step-shuffle null 0.664 (p97.5 0.689), p=0.085 — order-bearing if real ≫ null.

### Correctness (secondary, full trajectory)
- chains: 183 (pos 84 / neg 99)
| feature set | AUC (oof) | fold mean±std | stable | n_pos/n |
|---|---|---|---|---|
| belief_all | 0.655 | 0.688±0.063 | True | 84/183 |
| belief_trajectory | 0.655 | 0.680±0.044 | True | 84/183 |
| belief_occupancy | 0.666 | 0.697±0.052 | True | 84/183 |
| length_gap_trunc | 0.442 | 0.491±0.093 | True | 84/183 |
| persistence_step | 0.549 | 0.569±0.067 | True | 84/183 |
| rung0_curvature | 0.529 | 0.550±0.017 | True | 84/183 |
| length_PLUS_belief_traj | 0.650 | 0.676±0.054 | True | 84/183 |
| length_PLUS_belief_all | 0.651 | 0.692±0.072 | True | 84/183 |

- **M3 order test** (trajectory features): real 0.655 vs step-shuffle null 0.641 (p97.5 0.697), p=0.338 — order-bearing if real ≫ null.

## layer 17  (chains: 984 pre-onset / 986 full)
### Collapse (primary, leak-guarded)
- chains: 868 (pos 410 / neg 458)
| feature set | AUC (oof) | fold mean±std | stable | n_pos/n |
|---|---|---|---|---|
| belief_all | 0.743 | 0.747±0.038 | True | 410/868 |
| belief_trajectory | 0.637 | 0.642±0.032 | True | 410/868 |
| belief_occupancy | 0.749 | 0.752±0.031 | True | 410/868 |
| length_gap_trunc | 0.986 | 0.989±0.007 | True | 410/868 |
| persistence_step | 0.866 | 0.868±0.011 | True | 410/868 |
| rung0_curvature | 0.891 | 0.892±0.007 | True | 410/868 |
| length_PLUS_belief_traj | 0.988 | 0.990±0.007 | True | 410/868 |
| length_PLUS_belief_all | 0.988 | 0.991±0.006 | True | 410/868 |

- **M3 order test** (trajectory features): real 0.637 vs step-shuffle null 0.615 (p97.5 0.645), p=0.090 — order-bearing if real ≫ null.

### Correctness (secondary, full trajectory)
- chains: 183 (pos 84 / neg 99)
| feature set | AUC (oof) | fold mean±std | stable | n_pos/n |
|---|---|---|---|---|
| belief_all | 0.625 | 0.639±0.082 | True | 84/183 |
| belief_trajectory | 0.597 | 0.625±0.076 | True | 84/183 |
| belief_occupancy | 0.641 | 0.660±0.029 | True | 84/183 |
| length_gap_trunc | 0.442 | 0.491±0.093 | True | 84/183 |
| persistence_step | 0.506 | 0.531±0.036 | True | 84/183 |
| rung0_curvature | 0.497 | 0.512±0.033 | True | 84/183 |
| length_PLUS_belief_traj | 0.583 | 0.612±0.080 | True | 84/183 |
| length_PLUS_belief_all | 0.613 | 0.628±0.083 | True | 84/183 |

- **M3 order test** (trajectory features): real 0.597 vs step-shuffle null 0.548 (p97.5 0.631), p=0.154 — order-bearing if real ≫ null.

## layer 27  (chains: 984 pre-onset / 986 full)
### Collapse (primary, leak-guarded)
- chains: 868 (pos 410 / neg 458)
| feature set | AUC (oof) | fold mean±std | stable | n_pos/n |
|---|---|---|---|---|
| belief_all | 0.755 | 0.756±0.028 | True | 410/868 |
| belief_trajectory | 0.609 | 0.616±0.033 | True | 410/868 |
| belief_occupancy | 0.759 | 0.761±0.028 | True | 410/868 |
| length_gap_trunc | 0.986 | 0.989±0.007 | True | 410/868 |
| persistence_step | 0.851 | 0.853±0.007 | True | 410/868 |
| rung0_curvature | 0.884 | 0.884±0.009 | True | 410/868 |
| length_PLUS_belief_traj | 0.989 | 0.991±0.006 | True | 410/868 |
| length_PLUS_belief_all | 0.988 | 0.990±0.005 | True | 410/868 |

- **M3 order test** (trajectory features): real 0.609 vs step-shuffle null 0.604 (p97.5 0.632), p=0.358 — order-bearing if real ≫ null.

### Correctness (secondary, full trajectory)
- chains: 183 (pos 84 / neg 99)
| feature set | AUC (oof) | fold mean±std | stable | n_pos/n |
|---|---|---|---|---|
| belief_all | 0.698 | 0.704±0.070 | True | 84/183 |
| belief_trajectory | 0.628 | 0.655±0.073 | True | 84/183 |
| belief_occupancy | 0.712 | 0.725±0.068 | True | 84/183 |
| length_gap_trunc | 0.442 | 0.491±0.093 | True | 84/183 |
| persistence_step | 0.526 | 0.557±0.051 | True | 84/183 |
| rung0_curvature | 0.543 | 0.560±0.027 | True | 84/183 |
| length_PLUS_belief_traj | 0.622 | 0.646±0.075 | True | 84/183 |
| length_PLUS_belief_all | 0.699 | 0.715±0.070 | True | 84/183 |

- **M3 order test** (trajectory features): real 0.628 vs step-shuffle null 0.670 (p97.5 0.727), p=0.925 — order-bearing if real ≫ null.
