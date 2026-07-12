# A2 — switching linear dynamics over reasoning steps

**SMOKE — first-N chains, random collapse labels; AMI≈0, no enrichment.**

Collapse classes: loop 155 / clean 98 / ambiguous 47.

## layer 17
- 11732 transitions, PCA d'=32 (65% var), K=6 modes
- **mode ↔ behaviour AMI = 0.032** (null -0.000, z=212.4) — the annotation-validation read
- shuffled-ORDER control AMI = 0.005 ⇒ **ORDER-SENSITIVE (dynamics)**
- expanding modes (|λ|>1, loop-attractor candidates): 3/6

| mode | size | |λ|max | loop frac | loop enrich |
|---|---|---|---|---|
| 0 | 16% | 1.07 | 0.44 | 1.02× |
| 1 | 24% | 0.98 | 0.44 | 1.01× |
| 2 | 15% | 1.09 | 0.45 | 1.04× |
| 3 | 15% | 0.93 | 0.45 | 1.06× |
| 4 | 14% | 0.97 | 0.43 | 1.00× |
| 5 | 16% | 1.01 | 0.37 | 0.86× |

- base loop fraction 0.43 (enrichment > 1 ⇒ mode over-represented in collapse-bound chains)
