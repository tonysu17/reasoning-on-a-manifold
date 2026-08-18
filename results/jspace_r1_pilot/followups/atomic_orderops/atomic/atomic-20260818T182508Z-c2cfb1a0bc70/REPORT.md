# A0 — 1.5B atomic-hop behavioural audit

Exploratory follow-up under the sealed protocol. Independent unit: source item.

| Estimand | Estimate | 95% paired-bootstrap interval | Holm p |
|---|---:|---:|---:|
| Delta K | 0.2593 | [0.1605, 0.3580] | 0.000020 |
| Delta X | -0.1481 | [-0.2716, -0.0367] | 0.027390 |

## Behavioural scoring sensitivities

| Cell | Arm | Exact | Whole-label | Substring |
|---|---|---:|---:|---:|
| base | hop1 | 0.5556 | 0.6173 | 0.6420 |
| base | hop2 | 0.6296 | 0.6667 | 0.7284 |
| base | composite | 0.1605 | 0.1728 | 0.1975 |
| base | source_prompt_continuity | 0.0123 | 0.0494 | 0.1111 |
| distill | hop1 | 0.1975 | 0.3210 | 0.3210 |
| distill | hop2 | 0.1728 | 0.4691 | 0.4815 |
| distill | composite | 0.0494 | 0.0988 | 0.1111 |
| distill | source_prompt_continuity | 0.0370 | 0.0741 | 0.0864 |

Registered pattern: **indeterminate**.

This operational composition contrast is not causal mediation.
