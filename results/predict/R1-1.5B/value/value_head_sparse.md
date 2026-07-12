# R3 — trained value head vs the free belief lens

Bars to beat: correctness = A1 belief_occupancy **0.71**; collapse = length **0.986** (both from 24_belief_lens). Collapse classes: loop 415 / clean 469 / ambiguous 116 (dropped).

## layer 14
### collapse
- chains 868 (pos 410 / neg 458)
| model | AUROC (oof) |
|---|---|
| value_gru | 0.960 |
| value_gru_stepshuffled | 0.975 |
| pooled_logistic | 0.976 |
| length_gap_trunc | 0.986 |
| persistence_step | 0.856 |
| rung0_curvature | 0.886 |

- value beats best free baseline (incl. belief floor 0.986) by **-0.026** → **value_track_closed (V <= best free baseline)**
- order gain vs step-shuffled GRU: **-0.015** → occupancy (order adds <0.05 even with a sequence model)

### correctness
- chains 183 (pos 84 / neg 99)
| model | AUROC (oof) |
|---|---|
| value_gru | 0.740 |
| value_gru_stepshuffled | 0.751 |
| pooled_logistic | 0.701 |
| length_gap_trunc | 0.442 |
| persistence_step | 0.549 |
| rung0_curvature | 0.529 |

- value beats best free baseline (incl. belief floor 0.710) by **+0.030** → **value_track_closed (V <= best free baseline)**
- order gain vs step-shuffled GRU: **-0.011** → occupancy (order adds <0.05 even with a sequence model)

## layer 17
### collapse
- chains 868 (pos 410 / neg 458)
| model | AUROC (oof) |
|---|---|
| value_gru | 0.971 |
| value_gru_stepshuffled | 0.974 |
| pooled_logistic | 0.974 |
| length_gap_trunc | 0.986 |
| persistence_step | 0.866 |
| rung0_curvature | 0.891 |

- value beats best free baseline (incl. belief floor 0.986) by **-0.015** → **value_track_closed (V <= best free baseline)**
- order gain vs step-shuffled GRU: **-0.003** → occupancy (order adds <0.05 even with a sequence model)

### correctness
- chains 183 (pos 84 / neg 99)
| model | AUROC (oof) |
|---|---|
| value_gru | 0.746 |
| value_gru_stepshuffled | 0.772 |
| pooled_logistic | 0.726 |
| length_gap_trunc | 0.442 |
| persistence_step | 0.506 |
| rung0_curvature | 0.497 |

- value beats best free baseline (incl. belief floor 0.726) by **+0.020** → **value_track_closed (V <= best free baseline)**
- order gain vs step-shuffled GRU: **-0.026** → occupancy (order adds <0.05 even with a sequence model)

## layer 27
### collapse
- chains 868 (pos 410 / neg 458)
| model | AUROC (oof) |
|---|---|
| value_gru | 0.965 |
| value_gru_stepshuffled | 0.949 |
| pooled_logistic | 0.967 |
| length_gap_trunc | 0.986 |
| persistence_step | 0.851 |
| rung0_curvature | 0.884 |

- value beats best free baseline (incl. belief floor 0.986) by **-0.021** → **value_track_closed (V <= best free baseline)**
- order gain vs step-shuffled GRU: **+0.016** → occupancy (order adds <0.05 even with a sequence model)

### correctness
- chains 183 (pos 84 / neg 99)
| model | AUROC (oof) |
|---|---|
| value_gru | 0.729 |
| value_gru_stepshuffled | 0.754 |
| pooled_logistic | 0.751 |
| length_gap_trunc | 0.442 |
| persistence_step | 0.526 |
| rung0_curvature | 0.543 |

- value beats best free baseline (incl. belief floor 0.751) by **-0.022** → **value_track_closed (V <= best free baseline)**
- order gain vs step-shuffled GRU: **-0.025** → occupancy (order adds <0.05 even with a sequence model)
