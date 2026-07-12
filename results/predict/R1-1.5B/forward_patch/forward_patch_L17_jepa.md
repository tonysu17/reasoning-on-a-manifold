# A5 — forward-map causal fidelity (layer 17, sparse arm)

Patch site = onset token of step t+1, at the OUTPUT of block 17 (`layer17.npy` space). KL(patched‖clean) of the next-token dist, nats.

- resolved patch sites: 220 / 223 attempted, over 66 chains

| candidate | mean KL | median KL |
|---|---|---|
| real_pooled | 14.9813 | 15.7685 |
| pred_normed | 14.5570 | 14.8419 |
| pred | 14.5678 | 15.0776 |
| persist | 15.7705 | 16.1063 |
| random | 16.5819 | 17.0078 |

- **pred_normed vs persist** (want KL lower): pred_normed lower in 65% of sites, median Δ(persist−pred)=1.3087, Wilcoxon p=7.67e-06 (n=220)
- pred_normed vs random (norm-matched DIRECTION test): pred_normed lower in 70% (p=9.23e-11)
- pooling ceiling (real ≤ pred_normed): real lower in 45%
- raw ridge ẑ (norm-shrunk) vs persist: pred lower in 65%
- recovery vs pooling ceiling (median): 0.330 (1.0 = pred_normed as faithful as the real pooled state; 0 = no better than persistence)
- **A5-P ordering real≤pred_normed<persist,random holds: False**
