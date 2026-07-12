# E10.1 DAS-1D backtracking — REPORT

Date: 2026-07-06 · device cuda · 400 pairs · window W=12

Windowed interchange transfer: does swapping the 1-D component make the base
context produce the SOURCE's real W-token continuation (induce) and the source
context produce the BASE's (remove)? Metric = mean per-token Δlogprob, patched
vs clean. The W-token window defeats the generic-onset-token degeneracy the
single-token objective admitted (minival-1): a 'Wait-booster' direction can
raise the first token but not the donor's specific continuation.

| layer | dir | sym Δlogprob | induce | remove |
|---|---|---|---|---|
| 17 | learned | +0.649 | +0.609 | +0.690 |
| 17 | warm | +0.649 | +0.608 | +0.690 |
| 17 | diff_of_means | +0.048 | +0.048 | +0.048 |
| 17 | random_rotation | +0.000 | +0.000 | +0.000 |
| 17 | shuffled_pair | +0.627 | +0.591 | +0.662 |
| 17 | learned_positional | +0.029 | +0.040 | +0.019 |
| 17 | pair_specificity_gap (learned − shuffled) | +0.023 | | |
| 17 | cos(learned,diffmeans) | 0.19587983191013336 | cos(warm,diffmeans)=0.19442321360111237 | |
| 11 | learned | +0.618 | +0.577 | +0.659 |
| 11 | warm | +0.617 | +0.577 | +0.658 |
| 11 | diff_of_means | +0.010 | +0.013 | +0.007 |
| 11 | random_rotation | -0.000 | -0.000 | -0.000 |
| 11 | shuffled_pair | +0.608 | +0.530 | +0.687 |
| 11 | learned_positional | +0.063 | +0.079 | +0.047 |
| 11 | pair_specificity_gap (learned − shuffled) | +0.010 | | |
| 11 | cos(learned,diffmeans) | -0.1463913917541504 | cos(warm,diffmeans)=0.1389463245868683 | |
| 27 | learned | +0.742 | +0.761 | +0.723 |
| 27 | warm | +0.742 | +0.762 | +0.722 |
| 27 | diff_of_means | +0.016 | +0.032 | +0.001 |
| 27 | random_rotation | +0.001 | +0.001 | +0.001 |
| 27 | shuffled_pair | +0.732 | +0.758 | +0.706 |
| 27 | learned_positional | +0.086 | +0.152 | +0.020 |
| 27 | pair_specificity_gap (learned − shuffled) | +0.011 | | |
| 27 | cos(learned,diffmeans) | -0.10442668199539185 | cos(warm,diffmeans)=0.10395821928977966 | |

**P1 (sealed):** learned (causal) direction sym Δlogprob >= diff_of_means.
**Controls that must be near zero:** random_rotation and learned_positional.
**Pair-specificity:** the learned direction must beat shuffled_pair by a clear
margin (the gap row) — a direction that survives pair shuffling is a generic
behaviour-token promoter, not a pair-aligned causal frame (Makelov illusion).

Follow-on (P2, separate stage): free-generation collapse rate, swap vs projective
ablation at matched on-target effect — requires generation, not run here.