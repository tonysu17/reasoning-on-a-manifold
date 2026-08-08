# DAS-1D example-testing (E10.1 recipe) — REPORT

Date: 2026-07-20 · device cuda · 400 pairs · window W=12

Windowed interchange transfer: does swapping the 1-D component make the base
context produce the SOURCE's real W-token continuation (induce) and the source
context produce the BASE's (remove)? Metric = mean per-token Δlogprob, patched
vs clean. The W-token window defeats the generic-onset-token degeneracy the
single-token objective admitted (minival-1): a 'Wait-booster' direction can
raise the first token but not the donor's specific continuation.

| layer | dir | sym Δlogprob | induce | remove |
|---|---|---|---|---|
| 15 | learned | +0.665 | +0.635 | +0.696 |
| 15 | warm | +0.679 | +0.645 | +0.713 |
| 15 | diff_of_means | +0.011 | +0.013 | +0.009 |
| 15 | random_rotation | +0.000 | -0.000 | +0.000 |
| 15 | shuffled_pair | +0.688 | +0.706 | +0.671 |
| 15 | learned_positional | +0.077 | +0.056 | +0.099 |
| 15 | pair_specificity_gap (learned − shuffled) | -0.023 | | |
| 15 | cos(learned,diffmeans) | 0.11194013804197311 | cos(warm,diffmeans)=0.15448260307312012 | |
| 19 | learned | +0.661 | +0.695 | +0.626 |
| 19 | warm | +0.659 | +0.684 | +0.635 |
| 19 | diff_of_means | +0.020 | +0.026 | +0.013 |
| 19 | random_rotation | +0.000 | +0.001 | -0.000 |
| 19 | shuffled_pair | +0.646 | +0.666 | +0.627 |
| 19 | learned_positional | +0.057 | +0.078 | +0.035 |
| 19 | pair_specificity_gap (learned − shuffled) | +0.015 | | |
| 19 | cos(learned,diffmeans) | 0.15094080567359924 | cos(warm,diffmeans)=0.13842318952083588 | |
| 27 | learned | +0.706 | +0.723 | +0.688 |
| 27 | warm | +0.705 | +0.723 | +0.687 |
| 27 | diff_of_means | +0.007 | +0.010 | +0.004 |
| 27 | random_rotation | +0.000 | +0.000 | +0.000 |
| 27 | shuffled_pair | +0.697 | +0.717 | +0.677 |
| 27 | learned_positional | +0.086 | +0.076 | +0.096 |
| 27 | pair_specificity_gap (learned − shuffled) | +0.008 | | |
| 27 | cos(learned,diffmeans) | -0.09137269109487534 | cos(warm,diffmeans)=0.09138549864292145 | |

**P1 (sealed):** learned (causal) direction sym Δlogprob >= diff_of_means.
**Controls that must be near zero:** random_rotation and learned_positional.
**Pair-specificity:** the learned direction must beat shuffled_pair by a clear
margin (the gap row) — a direction that survives pair shuffling is a generic
behaviour-token promoter, not a pair-aligned causal frame (Makelov illusion).

Follow-on (P2, separate stage): free-generation collapse rate, swap vs projective
ablation at matched on-target effect — requires generation, not run here.