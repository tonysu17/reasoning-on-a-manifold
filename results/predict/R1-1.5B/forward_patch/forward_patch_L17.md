# A5 — forward-map causal fidelity (layer 17, sparse arm)

Patch site = onset token of step t+1, at the OUTPUT of block 17 (`layer17.npy` space). KL(patched‖clean) of the next-token dist, nats.

- resolved patch sites: 220 / 223 attempted, over 66 chains

| candidate | mean KL | median KL |
|---|---|---|
| real_pooled | 14.9813 | 15.7685 |
| pred_normed | 14.3159 | 14.6936 |
| pred | 14.2247 | 14.7978 |
| persist | 15.7705 | 16.1063 |
| random | 16.5819 | 17.0078 |

- **pred_normed vs persist** (want KL lower): pred_normed lower in 67% of sites, median Δ(persist−pred)=1.4382, Wilcoxon p=1.41e-07 (n=220)
- pred_normed vs random (norm-matched DIRECTION test): pred_normed lower in 69% (p=1.9e-10)
- pooling ceiling (real ≤ pred_normed): real lower in 47%
- raw ridge ẑ (norm-shrunk) vs persist: pred lower in 67%
- recovery vs pooling ceiling (median): 0.393 (1.0 = pred_normed as faithful as the real pooled state; 0 = no better than persistence)
- **A5-P ordering real≤pred_normed<persist,random holds: False**
