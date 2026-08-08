# DAS-1D uncertainty-estimation (E10.1 recipe) — REPORT

Date: 2026-07-19 · device cuda · 400 pairs · window W=12

Windowed interchange transfer: does swapping the 1-D component make the base
context produce the SOURCE's real W-token continuation (induce) and the source
context produce the BASE's (remove)? Metric = mean per-token Δlogprob, patched
vs clean. The W-token window defeats the generic-onset-token degeneracy the
single-token objective admitted (minival-1): a 'Wait-booster' direction can
raise the first token but not the donor's specific continuation.

| layer | dir | sym Δlogprob | induce | remove |
|---|---|---|---|---|
| 15 | learned | +0.631 | +0.601 | +0.661 |
| 15 | warm | +0.630 | +0.599 | +0.661 |
| 15 | diff_of_means | +0.023 | +0.024 | +0.022 |
| 15 | random_rotation | +0.000 | -0.000 | +0.000 |
| 15 | shuffled_pair | +0.618 | +0.589 | +0.646 |
| 15 | learned_positional | +0.073 | +0.077 | +0.069 |
| 15 | pair_specificity_gap (learned − shuffled) | +0.013 | | |
| 15 | cos(learned,diffmeans) | -0.10209005326032639 | cos(warm,diffmeans)=0.10290946066379547 | |
| 16 | learned | +0.616 | +0.580 | +0.653 |
| 16 | warm | +0.621 | +0.577 | +0.665 |
| 16 | diff_of_means | +0.034 | +0.035 | +0.033 |
| 16 | random_rotation | +0.001 | +0.001 | +0.000 |
| 16 | shuffled_pair | +0.604 | +0.586 | +0.621 |
| 16 | learned_positional | +0.032 | +0.032 | +0.032 |
| 16 | pair_specificity_gap (learned − shuffled) | +0.013 | | |
| 16 | cos(learned,diffmeans) | 0.1744922399520874 | cos(warm,diffmeans)=0.15734317898750305 | |
| 27 | learned | +0.715 | +0.724 | +0.707 |
| 27 | warm | +0.715 | +0.724 | +0.706 |
| 27 | diff_of_means | +0.014 | +0.023 | +0.005 |
| 27 | random_rotation | +0.002 | +0.001 | +0.002 |
| 27 | shuffled_pair | +0.706 | +0.718 | +0.694 |
| 27 | learned_positional | +0.085 | +0.130 | +0.041 |
| 27 | pair_specificity_gap (learned − shuffled) | +0.009 | | |
| 27 | cos(learned,diffmeans) | 0.05773536488413811 | cos(warm,diffmeans)=0.05729314684867859 | |

**P1 (sealed):** learned (causal) direction sym Δlogprob >= diff_of_means.
**Controls that must be near zero:** random_rotation and learned_positional.
**Pair-specificity:** the learned direction must beat shuffled_pair by a clear
margin (the gap row) — a direction that survives pair shuffling is a generic
behaviour-token promoter, not a pair-aligned causal frame (Makelov illusion).

Follow-on (P2, separate stage): free-generation collapse rate, swap vs projective
ablation at matched on-target effect — requires generation, not run here.