# E9.3 fork-locator gate — REPORT

Date: 2026-07-07 · 60 chains · 61053 tokens · device mps

## VERDICT: **NO-GO (entropy fails bar)**

Fork base rates (how rare): backtracking-onset 0.11% of tokens, branching-onset 0.49%. Entropy concentration (Gini) 0.436.

| locator → target | AUROC pooled | AUROC chain-grouped | lift@top10% | lift@top20% |
|---|---|---|---|---|
| entropy__backtracking_onset | 0.587 | 0.642 | 0.15 | 0.46 |
| entropy__branching_onset | 0.600 | 0.624 | 0.44 | 0.77 |

Sealed bar: AUROC(chain-grouped) ≥ 0.6 AND lift@top20% ≥ 1.5 on branching-onset.

GO ⇒ E9.3 runs with token-entropy fork detection (free, online). NO-GO ⇒ forks are not identifiable above chance; fork-localized ≈ uniform, do not spend a pod as spec'd.

_Residual-spike locator (Rung-1) is the secondary candidate; the pilot's step-shuffle null (p>0.7) already predicts it fails to localize branches — not re-run here since token entropy is what E9.3 uses online._