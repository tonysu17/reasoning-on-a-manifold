# E10.1 P2 — swap vs ablation: REPORT

Date: 2026-07-07 · collapse = 4-gram repetition > 0.8

| arm | n | collapse | mean rep | mean tokens | boxed | displacement |
|---|---|---|---|---|---|---|
| das_add_a0.5 | 50 | 0.50 | 0.631 | 6540 | 0.06 | 4.018 |
| das_add_a1.0 | 50 | 0.56 | 0.697 | 7139 | 0.02 | 7.318 |
| das_clamp_off | 50 | 0.66 | 0.688 | 7016 | 0.08 | 12.370 |
| das_clamp_on | 50 | 0.44 | 0.451 | 8192 | 0.00 | 20.609 |
| das_sub_a1.0 | 50 | 0.30 | 0.457 | 5043 | 0.20 | 9.818 |
| dm_add_a0.5 | 50 | 0.42 | 0.620 | 6708 | 0.02 | 7.612 |
| dm_add_a1.0 | 50 | 0.62 | 0.720 | 7386 | 0.02 | 16.398 |
| dm_clamp_off | 50 | 0.44 | 0.602 | 6082 | 0.12 | 14.168 |
| dm_clamp_on | 50 | 0.86 | 0.845 | 7672 | 0.00 | 24.285 |
| dm_sub_a1.0 | 50 | 0.32 | 0.451 | 4677 | 0.16 | 16.005 |
| vanilla | 50 | 0.32 | 0.523 | 5727 | 0.06 | 0.000 |

## Primary (sealed P2): clamp collapse vs add-curve at matched displacement
- **das**: observed clamp_on collapse 0.44 vs predicted 0.56 at matched displacement 20.609; diff -0.12 CI95 [-0.32, 0.08000000000000002]; McNemar vs add(1.0) p=0.3269; P2 supported: False
- **dm**: observed clamp_on collapse 0.86 vs predicted 0.62 at matched displacement 24.285; diff +0.24 CI95 [0.09999999999999998, 0.38]; McNemar vs add(1.0) p=0.0042; P2 supported: False