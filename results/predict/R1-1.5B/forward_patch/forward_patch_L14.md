# A5 — forward-map causal fidelity (layer 14, sparse arm)

Patch site = onset token of step t+1, at the OUTPUT of block 14 (`layer14.npy` space). KL(patched‖clean) of the next-token dist, nats.

- resolved patch sites: 220 / 223 attempted, over 66 chains

| candidate | mean KL | median KL |
|---|---|---|
| real_pooled | 14.2032 | 15.2518 |
| pred_normed | 13.8277 | 14.4829 |
| pred | 13.7240 | 14.4971 |
| persist | 14.8909 | 15.4600 |
| random | 16.1584 | 16.8661 |

- **pred_normed vs persist** (want KL lower): pred_normed lower in 62% of sites, median Δ(persist−pred)=1.1821, Wilcoxon p=0.000847 (n=220)
- pred_normed vs random (norm-matched DIRECTION test): pred_normed lower in 72% (p=3.03e-11)
- pooling ceiling (real ≤ pred_normed): real lower in 44%
- raw ridge ẑ (norm-shrunk) vs persist: pred lower in 60%
- recovery vs pooling ceiling (median): 0.547 (1.0 = pred_normed as faithful as the real pooled state; 0 = no better than persistence)
- **A5-P ordering real≤pred_normed<persist,random holds: False**
