# E9.2 structured state-entropy injection — REPORT

Date: 2026-07-08 · site hs[17] · greedy + noise seeds [0, 1, 2] · collapse = rep4 > 0.8

| arm | chains | collapse | boxed | tokens | x-seed diversity | distinct4 | ||eps|| |
|---|---|---|---|---|---|---|---|
| causal_f0.01 | 150 | 0.36 | 0.09 | 5816 | 0.907 | 0.791 | 1.21 |
| causal_f0.03 | 150 | 0.36 | 0.11 | 5679 | 0.954 | 0.859 | 3.62 |
| iso_f0.01 | 150 | 0.35 | 0.11 | 5902 | 0.901 | 0.782 | 1.36 |
| iso_f0.03 | 150 | 0.37 | 0.11 | 5646 | 0.949 | 0.851 | 4.09 |
| pca_f0.01 | 150 | 0.33 | 0.12 | 5682 | 0.912 | 0.796 | 1.21 |
| pca_f0.03 | 150 | 0.38 | 0.10 | 5891 | 0.961 | 0.872 | 3.62 |
| random_f0.01 | 150 | 0.35 | 0.10 | 5544 | 0.902 | 0.790 | 1.21 |
| random_f0.03 | 150 | 0.30 | 0.11 | 5637 | 0.945 | 0.848 | 3.62 |
| vanilla | 50 | 0.32 | 0.06 | 5727 | n/a | n/a | 0.00 |

## Sealed P1 (causal Pareto-dominates iso at matched energy)
- P1_f0.01: causal div 0.9070714842684253 vs iso 0.9009761486919242; boxed drop -0.03 vs -0.05 => P1 not supported
- P1_f0.03: causal div 0.9538422052027542 vs iso 0.9487425529002388; boxed drop -0.05 vs -0.05 => P1 not supported