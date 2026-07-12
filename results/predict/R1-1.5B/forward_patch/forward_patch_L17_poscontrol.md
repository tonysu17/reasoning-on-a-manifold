# A5 — forward-map causal fidelity (layer 17, sparse arm)

Patch site = onset token of step t+1, at the OUTPUT of block 17 (`layer17.npy` space). KL(patched‖clean) of the next-token dist, nats.

- resolved patch sites: 220 / 223 attempted, over 66 chains

| candidate | mean KL | median KL |
|---|---|---|
| real_pooled | 14.7263 | 14.2620 |
| pred_normed | 13.6295 | 13.1160 |
| pred | 13.4724 | 13.1018 |
| persist | 14.5795 | 14.2687 |
| random | 14.0154 | 13.6346 |

- **pred_normed vs persist** (want KL lower): pred_normed lower in 60% of sites, median Δ(persist−pred)=0.9070, Wilcoxon p=6.98e-05 (n=220)
- pred_normed vs random (norm-matched DIRECTION test): pred_normed lower in 52% (p=0.154)
- pooling ceiling (real ≤ pred_normed): real lower in 38%
- raw ridge ẑ (norm-shrunk) vs persist: pred lower in 60%
- recovery vs pooling ceiling (median): 0.352 (1.0 = pred_normed as faithful as the real pooled state; 0 = no better than persistence)
- **A5-P ordering real≤pred_normed<persist,random holds: False**
