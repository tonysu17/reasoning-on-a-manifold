# R3 — trained value head vs the free belief lens

**SMOKE — random labels; every AUROC ~0.5.**

Bars to beat: correctness = A1 belief_occupancy **0.71**; collapse = length **0.986** (both from 24_belief_lens). Collapse classes: loop 155 / clean 98 / ambiguous 47 (dropped).

## layer 17
### collapse
- chains 251 (pos 127 / neg 124)
| model | AUROC (oof) |
|---|---|
| value_gru | 0.442 |
| value_gru_stepshuffled | 0.518 |
| pooled_logistic | 0.529 |
| length_gap_trunc | 0.431 |
| persistence_step | 0.357 |
| rung0_curvature | 0.439 |

- value beats best free baseline (incl. belief floor 0.986) by **-0.544** → **value_track_closed (V <= best free baseline)**
- order gain vs step-shuffled GRU: **-0.076** → occupancy (order adds <0.05 even with a sequence model)

### correctness
- chains 119 (pos 59 / neg 60)
| model | AUROC (oof) |
|---|---|
| value_gru | 0.538 |
| value_gru_stepshuffled | 0.487 |
| pooled_logistic | 0.536 |
| length_gap_trunc | 0.477 |
| persistence_step | 0.431 |
| rung0_curvature | 0.323 |

- value beats best free baseline (incl. belief floor 0.710) by **-0.172** → **value_track_closed (V <= best free baseline)**
- order gain vs step-shuffled GRU: **+0.052** → order_bearing
