# Predictive-geometry gate

## layer 14
- labelled chains: 183 (correct 84 / incorrect 99)
- residual/persistence ratio: 0.876 (>1 ⇒ learned predictor worse than repeating the step)

| feature set | AUC (oof) | fold mean±std | stable | n_pos/n |
|---|---|---|---|---|
| rung1_residual | 0.542 | 0.569±0.054 | True | 84/183 |
| rung0_curvature | 0.529 | 0.550±0.017 | True | 84/183 |
| persistence_step | 0.549 | 0.569±0.067 | True | 84/183 |
| length_gap_trunc | 0.442 | 0.491±0.093 | True | 84/183 |

- label-permutation null: real 0.542 vs null 0.471, p=0.128
- step-shuffle null: real 0.542 vs null 0.554, p=0.721
- **verdict: no_rung1_advantage**

## layer 17
- labelled chains: 183 (correct 84 / incorrect 99)
- residual/persistence ratio: 0.876 (>1 ⇒ learned predictor worse than repeating the step)

| feature set | AUC (oof) | fold mean±std | stable | n_pos/n |
|---|---|---|---|---|
| rung1_residual | 0.486 | 0.521±0.073 | True | 84/183 |
| rung0_curvature | 0.497 | 0.512±0.033 | True | 84/183 |
| persistence_step | 0.506 | 0.531±0.036 | True | 84/183 |
| length_gap_trunc | 0.442 | 0.491±0.093 | True | 84/183 |

- label-permutation null: real 0.486 vs null 0.469, p=0.421
- step-shuffle null: real 0.486 vs null 0.509, p=0.770
- **verdict: no_rung1_advantage**
