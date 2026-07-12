# A5 — forward-map causal fidelity (layer 27, sparse arm)

Patch site = onset token of step t+1, at the OUTPUT of block 27 (`layer27.npy` space). KL(patched‖clean) of the next-token dist, nats.

- resolved patch sites: 220 / 223 attempted, over 66 chains

| candidate | mean KL | median KL |
|---|---|---|
| real_pooled | 15.4057 | 16.0517 |
| pred_normed | 15.0804 | 15.5366 |
| pred | 15.0814 | 15.5358 |
| persist | 16.8449 | 16.8234 |
| random | 16.8417 | 16.5184 |

- **pred_normed vs persist** (want KL lower): pred_normed lower in 70% of sites, median Δ(persist−pred)=1.7928, Wilcoxon p=1.59e-09 (n=220)
- pred_normed vs random (norm-matched DIRECTION test): pred_normed lower in 69% (p=5.18e-10)
- pooling ceiling (real ≤ pred_normed): real lower in 43%
- raw ridge ẑ (norm-shrunk) vs persist: pred lower in 69%
- recovery vs pooling ceiling (median): 0.443 (1.0 = pred_normed as faithful as the real pooled state; 0 = no better than persistence)
- **A5-P ordering real≤pred_normed<persist,random holds: False**
