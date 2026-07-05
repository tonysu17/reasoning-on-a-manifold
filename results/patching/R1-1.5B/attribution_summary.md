# Attribution patching — R1-1.5B

Read-out layer (metric): L27. Layers swept: 0..27. Pairs/behaviour: 20. Onset window: ±2.

Metric = projection of the residual onto the behaviour's unit diff-of-means steering direction (CF-10 fix; not lexical markers); positions aligned at the behaviour onset across chains (CF-10b).

⚠️ First-order estimate — confirm the peak layer with `--brute-check`.

## Per-layer mean attribution (higher = more causal influence)

| Behaviour | argmax L | L0 | L3 | L7 | L10 | L14 | L17 | L21 | L24 | L27 |
|---|---|---|---|---|---|---|---|---|---|---|
| backtracking | **27** | 9.94 | 12.8 | 22.6 | 29.2 | 26.6 | 27.7 | 41.7 | 53.4 | 75.3 |
| uncertainty-estimation | **27** | 10.4 | 17.1 | 23.1 | 25.2 | 31.2 | 42.9 | 93.1 | 107 | 133 |
| example-testing | **27** | 12.3 | 20.9 | 32.2 | 35 | 44.7 | 53.2 | 91.8 | 94.3 | 98 |
| adding-knowledge | **26** | 7.66 | 11.5 | 15.9 | 19.7 | 22.3 | 38.9 | 76.5 | 86.5 | 87.9 |