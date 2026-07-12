# A5 DENSE — onset-token forward-map fidelity (layer 17)

Patch = block-17 output at each step's ONSET token (single real state, not a span average). KL(patched‖clean), nats. 1884 sites over 291 chains.

| candidate | mean KL | median KL |
|---|---|---|
| real_onset | 0.000 | 0.000 |
| pred_normed | 12.418 | 13.700 |
| pred | 12.692 | 13.938 |
| persist | 13.978 | 16.578 |
| random | 17.194 | 17.892 |

- **real_onset a valid ceiling? True** (mean KL 0.000 — must be ≈0)
- **pred_normed vs random** (onset-specific direction test): pred lower in 65% (p=6.9e-68, n=1884)
- pred_normed vs persist: pred lower in 57% (p=1.92e-11)
- recovery vs ceiling (median): 0.033 (1 = pred as faithful as the true state; 0 = = persistence)
