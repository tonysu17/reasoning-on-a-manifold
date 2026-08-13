# Arm-blinded P5 generic pilot QA and clustered sizing

Status: **design resource only; no arm-labelled means, differences, directions, or scientific findings**.

- Internal SHA-256: `f8e7df25da19fe514581edf3b4f94a3684ed97a1ba07c79e673021a18638374f`
- File SHA-256: `1368c2a7db408862e6a268eec794605ba05b12247841734baacebb1dfb8b336f`
- Independent unit: task/prompt (20); checkpoint rows are nested repeated observations.
- Parsed annotations: 78/80 (97.5%); frozen gate: 98% — **not passed**.
- Length-cap hits: 34/80 (42.5%); the 5% trigger is exceeded.
- Complete task pairs: 18/20 in one anonymous contrast family and 20/20 in the other.

## Descriptive and sizing interpretation

Endpoint distributions are reported pooled across checkpoints and as task-cluster means/within-task variance. Pooled row counts are descriptive only; N is never 78. Paired sizing uses unsigned squared within-task distances as a conservative variance proxy, so no signed or arm-labelled pilot effect was computed.

Conservative eight-test, 80%-power category-balanced task ranges:

- absolute effect 0.020: 110–1840 tasks
- absolute effect 0.030: 50–820 tasks
- absolute effect 0.050: 20–300 tasks
- absolute effect 0.075: 10–140 tasks
- absolute effect 0.100: 10–80 tasks

These are planning ranges, not a frozen N. The 20-task pilot is small, one contrast loses two complete pairs, and 42.5% of rows hit the 4,096-token cap. The current variance evidence transfers directly only to a 4,096-generated-token analytic-prefix estimand.

## QA disposition

Share with noted caveats for design only. Before powered P5: bind the Phase-2 shared vanilla, choose/hash the analytic prefix, choose effect/power/alpha, determine whether 100 tasks suffice, and prospectively validate generic annotation at at least 98% parse with an exact response/accounting contract.
