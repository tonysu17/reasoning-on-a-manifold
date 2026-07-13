# R3 pilot gate — strategy-entropy substrate check

48 chains / 16 tasks. Prereg: R3_PILOT_PREREG.md

## VERDICT: **ALL GATES PASS — full R3 as designed**

| gate | value | bar | pass |
|---|--:|--:|:--|
| G_R3_1_answerable | 0.896 | 0.7 | ✅ |
| G_R3_2_headroom | 0.896 | [0.2, 0.95] | ✅ |
| G_R3_3_coverage | 1.0 | 0.6 | ✅ |
| G_R3_4_variation | 4/16 | 0.25 | ✅ |

## Per template

| tpl | n | parse | acc | cap-hit | mean tok | strategies seen |
|---|--:|--:|--:|--:|--:|---|
| T1 | 6 | 0.67 | 0.67 | 0.33 | 5121 | pattern, recursion |
| T2 | 6 | 0.67 | 0.67 | 0.33 | 4607 | formula, telescoping |
| T3 | 6 | 0.83 | 0.83 | 0.17 | 4208 | formula |
| T4 | 6 | 1.0 | 1.0 | 0.0 | 1408 | substitution |
| T5 | 6 | 1.0 | 1.0 | 0.0 | 2563 | formula |
| T6 | 6 | 1.0 | 1.0 | 0.0 | 1836 | pattern |
| T7 | 6 | 1.0 | 1.0 | 0.0 | 1729 | casework, complement |
| T8 | 6 | 1.0 | 1.0 | 0.0 | 1648 | explicit_roots, vieta |
