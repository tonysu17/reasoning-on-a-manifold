# DAS-1D example-testing (E10.1 recipe) — REPORT

Date: 2026-07-19 · device mps · 8 pairs · window W=12

Windowed interchange transfer: does swapping the 1-D component make the base
context produce the SOURCE's real W-token continuation (induce) and the source
context produce the BASE's (remove)? Metric = mean per-token Δlogprob, patched
vs clean. The W-token window defeats the generic-onset-token degeneracy the
single-token objective admitted (minival-1): a 'Wait-booster' direction can
raise the first token but not the donor's specific continuation.

| layer | dir | sym Δlogprob | induce | remove |
|---|---|---|---|---|
| 15 | learned | +0.003 | +0.003 | +0.002 |
| 15 | warm | +0.179 | +0.286 | +0.071 |
| 15 | diff_of_means | +0.004 | +0.008 | -0.001 |
| 15 | random_rotation | +0.002 | +0.001 | +0.003 |
| 15 | shuffled_pair | +0.002 | +0.002 | +0.001 |
| 15 | learned_positional | +0.017 | +0.029 | +0.005 |
| 15 | pair_specificity_gap (learned − shuffled) | +0.001 | | |
| 15 | cos(learned,diffmeans) | 0.0014268644154071808 | cos(warm,diffmeans)=0.5866835117340088 | |

**P1 (sealed):** learned (causal) direction sym Δlogprob >= diff_of_means.
**Controls that must be near zero:** random_rotation and learned_positional.
**Pair-specificity:** the learned direction must beat shuffled_pair by a clear
margin (the gap row) — a direction that survives pair shuffling is a generic
behaviour-token promoter, not a pair-aligned causal frame (Makelov illusion).

Follow-on (P2, separate stage): free-generation collapse rate, swap vs projective
ablation at matched on-target effect — requires generation, not run here.