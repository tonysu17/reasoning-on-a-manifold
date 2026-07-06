# E9.1b parity report (amplify vs subtract, greedy)

Vanilla collapse: amplify-run 0.36 / subtract-runs 0.36

| behaviour | arm | α | subtract | amplify | parity | amp arm-only/van-only (p) |
|---|---|--:|--:|--:|---|---|
| backtracking | energy_matched_random | 0.5 | 0.42 | 0.42 | even/flat | 9/6 (p=0.607) |
| backtracking | energy_matched_random | 1.0 | 0.28 | 0.40 | ODD | 8/6 (p=0.791) |
| backtracking | manifold_k5 | 0.5 | 0.30 | 0.58 | ODD | 14/3 (p=0.013) |
| backtracking | manifold_k5 | 1.0 | 0.24 | 0.58 | ODD | 15/4 (p=0.019) |
| backtracking | random_subspace_k5 | 0.5 | 0.28 | 0.37 | ODD | 1/7 (p=0.070) |
| backtracking | random_subspace_k5 | 1.0 | 0.37 | 0.33 | ODD | 1/9 (p=0.021) |
| backtracking | single_direction | 0.5 | 0.28 | 0.46 | ODD | 11/6 (p=0.332) |
| backtracking | single_direction | 1.0 | 0.36 | 0.64 | even/flat | 17/3 (p=0.003) |
| example-testing | energy_matched_random | 0.5 | 0.38 | 0.48 | even/flat | 10/4 (p=0.180) |
| example-testing | energy_matched_random | 1.0 | 0.38 | 0.46 | even/flat | 11/6 (p=0.332) |
| example-testing | manifold_k5 | 0.5 | 0.42 | 0.30 | ODD | 5/8 (p=0.581) |
| example-testing | manifold_k5 | 1.0 | 0.64 | 0.42 | even/flat | 7/4 (p=0.549) |
| example-testing | random_subspace_k5 | 0.5 | 0.33 | 0.37 | ODD | 4/9 (p=0.267) |
| example-testing | random_subspace_k5 | 1.0 | 0.36 | 0.36 | even/flat | 3/9 (p=0.146) |
| example-testing | single_direction | 0.5 | 0.36 | 0.36 | even/flat | 6/6 (p=1.000) |
| example-testing | single_direction | 1.0 | 0.48 | 0.34 | ODD | 6/7 (p=1.000) |
