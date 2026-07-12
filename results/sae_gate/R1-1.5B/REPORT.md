# E10.0 SAE dictionary gate — REPORT

Date: 2026-07-05 · 60 chains · cap 1024 tok · device mps

## VERDICT (residual candidate DGurgurov/DeepSeek-R1-Distill-Qwen-1.5B-sae): **PASS**

Site resolved empirically: **blocks.19.resid_post** (raw); repo metadata was self-contradictory (card: blocks.19.resid_pre; folder: blocks.1.resid_pre; cfg: blocks.19.resid_post).

Pre-pass site probe (mean FVU over 5 chains): blocks.1.resid_pre raw = 3.213; blocks.1.resid_pre norm = 2.561; blocks.19.resid_post raw = 0.059; blocks.19.resid_post norm = 0.374; blocks.19.resid_pre raw = 0.076; blocks.19.resid_pre norm = 0.315

| metric | resid candidate (resolved site) | MLP reference (layers.19.mlp, own site) |
|---|---|---|
| FVU | 0.0608 | 0.4784 |
| L0 | 70.9 | 32.0 |
| CE clean / recon / zero | 0.601 / 1.287 / 16.937 | 0.601 / 0.635 / 0.705 |
| CE recovered | 0.958 | 0.669 |

Gate (pre-registered): PASS iff FVU<=0.15 and CE-recovered>=0.85; AMBER iff FVU<=0.4 and CE-recovered>=0.6; else FAIL.

FAIL/AMBER consequence: self-train residual-stream SAEs on the CoT distribution
at the causal-candidate layers (Resa-style recipe, residual hookpoint) before any
SAE comparator arm. The DAS arm is unaffected either way (no dictionary needed).