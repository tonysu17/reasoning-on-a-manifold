# E10.1 DAS-1D backtracking — REPORT

Date: 2026-07-06 · device mps · 48 pairs

Interchange induction: does swapping the 1-D component transfer the backtracking
onset into a non-backtracking context? Metric = induction accuracy (argmax == source
onset token) and Δlogprob of the source token, base patched vs clean.

| layer | dir | Δlogprob | induction_acc | base_CE |
|---|---|---|---|---|
| 17 | learned | +6.562 | 0.354 | 2.903 |
| 17 | warm | +6.688 | 0.354 | 2.777 |
| 17 | diff_of_means | +0.274 | 0.125 | 9.192 |
| 17 | random_rotation | +0.004 | 0.104 | 9.461 |
| 17 | shuffled_pair | +5.869 | 0.271 | 3.597 |
| 17 | learned_positional | +0.329 | 0.104 | 9.136 |
| 17 | cos(learned,diffmeans) | 0.11011256277561188 | cos(warm,diffmeans)=0.08860321342945099 | |

**P1 (sealed):** learned (causal) direction induction_acc >= diff_of_means.
**Controls that must fail:** shuffled_pair and random_rotation near chance;
learned_positional (misaligned inject) near chance. A learned direction that beats
its shuffled-pair and positional controls is not a Makelov illusion.

Follow-on (P2, separate stage): free-generation collapse rate, swap vs projective
ablation at matched on-target effect — requires generation, not run here.