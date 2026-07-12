# A5 DENSE — onset-token forward-map fidelity (layer 27)

Patch = block-27 output at each step's ONSET token (single real state, not a span average). KL(patched‖clean), nats. 1884 sites over 291 chains.

| candidate | mean KL | median KL |
|---|---|---|
| real_onset | 0.822 | 0.483 |
| pred_normed | 15.050 | 16.360 |
| pred | 15.049 | 16.366 |
| persist | 15.267 | 17.304 |
| random | 18.896 | 19.039 |

- **real_onset a valid ceiling? False** (mean KL 0.822 — must be ≈0)
- **pred_normed vs random** (onset-specific direction test): pred lower in 62% (p=5.3e-53, n=1884)
- pred_normed vs persist: pred lower in 55% (p=0.00302)
- recovery vs ceiling (median): 0.073 (1 = pred as faithful as the true state; 0 = = persistence)
