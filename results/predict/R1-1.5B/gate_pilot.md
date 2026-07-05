> ⚠️ SUPERSEDED — DO NOT CITE (see RESULTS_LEDGER.md §E): this is the FLAWED/INFLATED gate — the next-step predictor was trained on the labelled subset only (~183 chains), so its residuals encode chain idiosyncrasies that correlate with correctness, inflating AUC to ≈0.61 and the residual/persistence ratio above 1. Cite the corrected train-on-all-986-chains run instead: `results/predict/R1-1.5B/corrected/` (honest AUC 0.54–0.59; see `corrected/SUMMARY.md`). This file is retained only for the before/after comparison referenced there.

# Predictive-geometry gate

## layer 11
- labelled chains: 183 (correct 84 / incorrect 99)
- residual/persistence ratio: 1.541 (>1 ⇒ learned predictor worse than repeating the step)

| feature set | AUC (oof) | fold mean±std | stable | n_pos/n |
|---|---|---|---|---|
| rung1_residual | 0.596 | 0.616±0.056 | True | 84/183 |
| rung0_curvature | 0.530 | 0.547±0.020 | True | 84/183 |
| persistence_step | 0.545 | 0.576±0.072 | True | 84/183 |
| length_gap_trunc | 0.442 | 0.491±0.093 | True | 84/183 |

- label-permutation null: real 0.596 vs null 0.474, p=0.018
- step-shuffle null: real 0.596 vs null 0.589, p=0.436
- **verdict: no_rung1_advantage**

## layer 14
- labelled chains: 183 (correct 84 / incorrect 99)
- residual/persistence ratio: 1.562 (>1 ⇒ learned predictor worse than repeating the step)

| feature set | AUC (oof) | fold mean±std | stable | n_pos/n |
|---|---|---|---|---|
| rung1_residual | 0.614 | 0.646±0.076 | True | 84/183 |
| rung0_curvature | 0.529 | 0.550±0.017 | True | 84/183 |
| persistence_step | 0.549 | 0.569±0.067 | True | 84/183 |
| length_gap_trunc | 0.442 | 0.491±0.093 | True | 84/183 |

- label-permutation null: real 0.614 vs null 0.476, p=0.008
- step-shuffle null: real 0.614 vs null 0.576, p=0.139
- **verdict: no_rung1_advantage**

## layer 17
- labelled chains: 183 (correct 84 / incorrect 99)
- residual/persistence ratio: 1.559 (>1 ⇒ learned predictor worse than repeating the step)

| feature set | AUC (oof) | fold mean±std | stable | n_pos/n |
|---|---|---|---|---|
| rung1_residual | 0.599 | 0.587±0.039 | True | 84/183 |
| rung0_curvature | 0.497 | 0.512±0.033 | True | 84/183 |
| persistence_step | 0.506 | 0.531±0.036 | True | 84/183 |
| length_gap_trunc | 0.442 | 0.491±0.093 | True | 84/183 |

- label-permutation null: real 0.599 vs null 0.478, p=0.024
- step-shuffle null: real 0.599 vs null 0.560, p=0.158
- **verdict: no_rung1_advantage**

## layer 20
- labelled chains: 183 (correct 84 / incorrect 99)
- residual/persistence ratio: 1.557 (>1 ⇒ learned predictor worse than repeating the step)

| feature set | AUC (oof) | fold mean±std | stable | n_pos/n |
|---|---|---|---|---|
| rung1_residual | 0.528 | 0.564±0.051 | True | 84/183 |
| rung0_curvature | 0.535 | 0.557±0.055 | True | 84/183 |
| persistence_step | 0.544 | 0.569±0.054 | True | 84/183 |
| length_gap_trunc | 0.442 | 0.491±0.093 | True | 84/183 |

- label-permutation null: real 0.528 vs null 0.475, p=0.202
- step-shuffle null: real 0.528 vs null 0.553, p=0.743
- **verdict: no_rung1_advantage**

## layer 27
- labelled chains: 183 (correct 84 / incorrect 99)
- residual/persistence ratio: 1.590 (>1 ⇒ learned predictor worse than repeating the step)

| feature set | AUC (oof) | fold mean±std | stable | n_pos/n |
|---|---|---|---|---|
| rung1_residual | 0.618 | 0.643±0.049 | True | 84/183 |
| rung0_curvature | 0.543 | 0.560±0.027 | True | 84/183 |
| persistence_step | 0.526 | 0.557±0.051 | True | 84/183 |
| length_gap_trunc | 0.442 | 0.491±0.093 | True | 84/183 |

- label-permutation null: real 0.618 vs null 0.472, p=0.006
- step-shuffle null: real 0.618 vs null 0.604, p=0.267
- **verdict: no_rung1_advantage**
