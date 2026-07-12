# A1 — belief-state trajectory (logit lens, sparse arm)

**SMOKE — random lens stats; every AUC should sit near 0.5.**

Collapse classes over corpus: loop 155 / clean 98 / ambiguous 47 (ambiguous dropped).

Sparse arm: lens applied to mean-pooled span vectors (smeared belief; M4 — suggestive, dense per-token arm confirmatory). Entropy in nats.

## layer 14  (chains: 298 pre-onset / 298 full)
### Collapse (primary, leak-guarded)
- chains: 251 (pos 154 / neg 97)
| feature set | AUC (oof) | fold mean±std | stable | n_pos/n |
|---|---|---|---|---|
| belief_all | 0.683 | 0.695±0.053 | True | 154/251 |
| belief_trajectory | 0.480 | 0.507±0.036 | True | 154/251 |
| belief_occupancy | 0.700 | 0.708±0.053 | True | 154/251 |
| length_gap_trunc | 0.954 | 0.959±0.032 | True | 154/251 |
| persistence_step | 0.816 | 0.827±0.062 | True | 154/251 |
| rung0_curvature | 0.852 | 0.861±0.049 | True | 154/251 |
| length_PLUS_belief_traj | 0.948 | 0.942±0.047 | True | 154/251 |
| length_PLUS_belief_all | 0.962 | 0.958±0.041 | True | 154/251 |

- **M3 order test** (trajectory features): real 0.480 vs step-shuffle null 0.459 (p97.5 0.547), p=0.361 — order-bearing if real ≫ null.

## layer 17  (chains: 298 pre-onset / 298 full)
### Collapse (primary, leak-guarded)
- chains: 251 (pos 154 / neg 97)
| feature set | AUC (oof) | fold mean±std | stable | n_pos/n |
|---|---|---|---|---|
| belief_all | 0.683 | 0.695±0.053 | True | 154/251 |
| belief_trajectory | 0.480 | 0.507±0.036 | True | 154/251 |
| belief_occupancy | 0.700 | 0.708±0.053 | True | 154/251 |
| length_gap_trunc | 0.954 | 0.959±0.032 | True | 154/251 |
| persistence_step | 0.844 | 0.849±0.066 | True | 154/251 |
| rung0_curvature | 0.865 | 0.871±0.049 | True | 154/251 |
| length_PLUS_belief_traj | 0.948 | 0.942±0.047 | True | 154/251 |
| length_PLUS_belief_all | 0.962 | 0.958±0.041 | True | 154/251 |

- **M3 order test** (trajectory features): real 0.480 vs step-shuffle null 0.459 (p97.5 0.547), p=0.361 — order-bearing if real ≫ null.

## layer 27  (chains: 298 pre-onset / 298 full)
### Collapse (primary, leak-guarded)
- chains: 251 (pos 154 / neg 97)
| feature set | AUC (oof) | fold mean±std | stable | n_pos/n |
|---|---|---|---|---|
| belief_all | 0.683 | 0.695±0.053 | True | 154/251 |
| belief_trajectory | 0.480 | 0.507±0.036 | True | 154/251 |
| belief_occupancy | 0.700 | 0.708±0.053 | True | 154/251 |
| length_gap_trunc | 0.954 | 0.959±0.032 | True | 154/251 |
| persistence_step | 0.838 | 0.850±0.056 | True | 154/251 |
| rung0_curvature | 0.861 | 0.869±0.054 | True | 154/251 |
| length_PLUS_belief_traj | 0.948 | 0.942±0.047 | True | 154/251 |
| length_PLUS_belief_all | 0.962 | 0.958±0.041 | True | 154/251 |

- **M3 order test** (trajectory features): real 0.480 vs step-shuffle null 0.459 (p97.5 0.547), p=0.361 — order-bearing if real ≫ null.
