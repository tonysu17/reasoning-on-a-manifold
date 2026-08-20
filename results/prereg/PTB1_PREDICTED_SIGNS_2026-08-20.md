# P0.4 — Sealed sign predictions for PT-B1 (committed before any PT-B1 output was read)

**Date:** 2026-08-20. **Inputs:** RQ3-era stored activations only (six adapter arms at
L12/16, seeds 42/43/44; base R1-1.5B behaviour files). **No PT-B1 battery, annotation,
analysis, or report file was opened before this commit** (`no_ptb1_output_read: true` in the
JSON; the executing session's transcript is the witness).

**Heuristic (declared):** project the environment-matched within-seed safety-minus-control
mean-activation difference onto each behaviour's base-frame axis (mean of behaviour rows minus
mean of other-3 rows, unit norm). Positive projection -> predict PT-B1's D_b (safety-minus-
control prevalence) > 0. A prediction is issued only where all three seeds agree at BOTH
layers. Linear-readout heuristic; auxiliary; cannot modify PT-B1's sealed endpoints.

| Behaviour | Predicted sign of D_b | cosine range (L12; L16, seeds 42/43/44) |
|---|---|---|
| backtracking | **positive** | +0.084..+0.109; +0.149..+0.169 |
| uncertainty-estimation | **positive** | +0.049..+0.064; +0.068..+0.109 |
| example-testing | **negative** | -0.050..-0.061; -0.160..-0.187 |
| adding-knowledge | **negative** | -0.119..-0.154; -0.088..-0.138 |

Numbers: `results/prereg/ptb1_predicted_signs.json`; generator `ptb1_predicted_signs.py`.
