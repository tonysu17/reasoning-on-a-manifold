# R1 — strata-differential compression (creativity–entropy rung 1 + RL.a)

Prereg: R1_COMPRESSION_PREREG.md. Primary contrast r1↔deepscaler (matched-ids).

## R1-rep: paired matched-text deltas vs r1 (per-chain, R0 sample)

| arm | n | Δent median (p) | ΔPR median (p) | Δunif median (p) |
|---|--:|--:|--:|--:|
| deepscaler | 200 | -0.017 (1.2e-13) | -0.017 (0.051) | -0.005 (1.4e-33) |
| star1 | 200 | -0.012 (1.9e-27) | -0.217 (1.5e-34) | +0.007 (1.4e-34) |
| qwenmath | 200 | +0.213 (7.6e-20) | -1.559 (4.8e-28) | +0.027 (1.1e-20) |

**P_R1_1_rlvr_compression:** NOT confirmed

**P_R1_2_base_vs_distill:** SFT-ENTROPY-SEEKING (r1 rank >= base)

**P_R1_6_safety_control:** PASS (star1 ~ r1)

## R1-beh: own-generation ladder (greedy + 3×T0.6, 50 tasks)

| arm | diversity (T06) | math-only | len | bt/1k | boxed | collapse | on-policy ent | on-policy PR |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| r1 | 0.9628680120119577 | 0.941763247035641 | 5026 | 11.99 | 0.18 | 0.11 | 0.7033984995260835 | 28.38580383183351 |
| deepscaler | 0.9464037515743448 | 0.9172417836378457 | 4446 | 8.51 | 0.32 | 0.07 | 0.6693969570472836 | 28.164611452627685 |
| star1 | 0.9605523609106743 | 0.9469620787006743 | 3798 | 7.52 | 0.22 | 0.07 | 0.7477295644581318 | 28.42652853240962 |
| qwenmath | 0.9495179275790613 | 0.9512636529503151 | 1200 | 0.14 | 0.36 | 0.07 | 0.4628050022305333 | 25.448404852918387 |

**P-R1.3 orderings:** {"qwenmath>r1": {"mean_a": 0.9512636529503151, "mean_b": 0.941763247035641, "mw_p_one_sided": 0.27380952380952384, "math_only": true}, "r1>deepscaler": {"mean_a": 0.9628680120119577, "mean_b": 0.9464037515743448, "mw_p_one_sided": 0.0011687452455493535, "math_only": false}}
