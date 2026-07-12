# E10.1 DAS-1D backtracking — REPORT

Date: 2026-07-06 · device mps · 32 pairs · window W=12

Windowed interchange transfer: does swapping the 1-D component make the base
context produce the SOURCE's real W-token continuation (induce) and the source
context produce the BASE's (remove)? Metric = mean per-token Δlogprob, patched
vs clean. The W-token window defeats the generic-onset-token degeneracy the
single-token objective admitted (minival-1): a 'Wait-booster' direction can
raise the first token but not the donor's specific continuation.

| layer | dir | sym Δlogprob | induce | remove |
|---|---|---|---|---|
| 17 | learned | +0.130 | +0.107 | +0.154 |
| 17 | diff_of_means | +0.043 | +0.031 | +0.055 |
| 17 | random_rotation | +0.001 | +0.001 | +0.001 |
| 17 | shuffled_pair | +0.152 | +0.102 | +0.201 |
| 17 | learned_positional | +0.040 | +0.039 | +0.041 |
| 17 | pair_specificity_gap (learned − shuffled) | -0.021 | | |
| 17 | cos(learned,diffmeans) | 0.23323898017406464 | cos(warm,diffmeans)=None | |

**P1 (sealed):** learned (causal) direction sym Δlogprob >= diff_of_means.
**Controls that must be near zero:** random_rotation and learned_positional.
**Pair-specificity:** the learned direction must beat shuffled_pair by a clear
margin (the gap row) — a direction that survives pair shuffling is a generic
behaviour-token promoter, not a pair-aligned causal frame (Makelov illusion).

Follow-on (P2, separate stage): free-generation collapse rate, swap vs projective
ablation at matched on-target effect — requires generation, not run here.