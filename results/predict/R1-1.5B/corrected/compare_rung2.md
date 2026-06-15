# Rung-1 (ridge) vs Rung-2 (JEPA) — correctness AUC

## layer 14
- labelled: 183 (correct 84/incorrect 99)

| predictor | AUC (oof) | fold mean±std | resid/persist | label-perm p |
|---|---|---|---|---|
| ridge | 0.542 | 0.569±0.054 | 0.876 | 0.128 |
| jepa | 0.582 | 0.612±0.057 | 1.139 | 0.026 |
| _baseline_ curvature | 0.529 | | | |
| _baseline_ persistence | 0.549 | | | |
| _baseline_ length | 0.442 | | | |

- **Rung-1 beats baselines+null: False; Rung-2 beats Rung-1+baselines+null: True**

## layer 17
- labelled: 183 (correct 84/incorrect 99)

| predictor | AUC (oof) | fold mean±std | resid/persist | label-perm p |
|---|---|---|---|---|
| ridge | 0.486 | 0.521±0.073 | 0.876 | 0.421 |
| jepa | 0.541 | 0.548±0.052 | 1.013 | 0.146 |
| _baseline_ curvature | 0.497 | | | |
| _baseline_ persistence | 0.506 | | | |
| _baseline_ length | 0.442 | | | |

- **Rung-1 beats baselines+null: False; Rung-2 beats Rung-1+baselines+null: False**

## layer 27
- labelled: 183 (correct 84/incorrect 99)

| predictor | AUC (oof) | fold mean±std | resid/persist | label-perm p |
|---|---|---|---|---|
| ridge | 0.578 | 0.598±0.042 | 0.921 | 0.034 |
| jepa | 0.591 | 0.605±0.061 | 0.920 | 0.024 |
| _baseline_ curvature | 0.543 | | | |
| _baseline_ persistence | 0.526 | | | |
| _baseline_ length | 0.442 | | | |

- **Rung-1 beats baselines+null: True; Rung-2 beats Rung-1+baselines+null: True**
