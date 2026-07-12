# R0 — entropy-ladder bookkeeping (creativity–entropy rung 0)

Chains: 200 (100 loop; 0 excluded on grid/gen-token mismatch or missing shard). Prereg: R0_ENTROPY_LADDER_PREREG.md

## R0.a cross-level Spearman (primary regions)

| pair | clean ρ (p) | loop pre-onset ρ (p) | pooled ρ (p) |
|---|--:|--:|--:|
| ent~pr17 | +0.670 (2.4e-14) | +0.705 (3.8e-16) | +0.748 (7.2e-37) |
| ent~unif17 | -0.383 (8.4e-05) | -0.618 (9.8e-12) | -0.584 (1.5e-19) |
| pr17~unif17 | -0.461 (1.4e-06) | -0.507 (8.4e-08) | -0.680 (2.2e-28) |
| ent~n_gen | -0.503 (9.5e-08) | -0.112 (0.27) | -0.379 (3.3e-08) |
| pr17~n_gen | -0.352 (0.00033) | +0.139 (0.17) | -0.480 (6.9e-13) |
| unif17~n_gen | +0.691 (1.7e-15) | -0.077 (0.45) | +0.780 (6.1e-42) |

**P-R0.1 gate:** PASS — levels dissociate (gate open) — max |ρ| among ladder pairs 0.705 (bar 0.9)

## R0.b loop entropy profile

**P-R0.2 (jam):** in-loop E-1 0.393 nats (n=100) vs matched clean 0.855 (n=100), one-sided MW p=1.2e-25 → **CONFIRMED (in-loop entropy lower)**

**P-R0.3 (pre-onset):** Δ(pre−base) loop -0.177 (n=70) vs clean -0.049 (n=7), two-sided MW p=0.45 → **STATE-FIRST (no loop-specific pre-onset E-1 differential)**

## R0.c linkage to E-3 (T06 vanilla; re-scoring caveat applies)

| metric | ρ vs diversity | p |
|---|--:|--:|
| mean_ent | +0.419 | 0.0024 |
| mean_pr | +0.483 | 0.00038 |
| mean_unif | -0.655 | 2.4e-07 |

**P-R0.4:** CONFIRMED (mid-range coupling) (n=50 tasks)

