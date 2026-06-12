# Cross-annotator comparison

Common chains: 1000

## Label distribution (% of spans, common chains)

| Label | Sonnet-4.5 | Qwen3-235B | Nova-Pro |
|---|---|---|---|
| adding-knowledge | 6.5% | 9.1% | 8.8% |
| backtracking | 13.2% | 10.8% | 14.8% |
| deduction | 44.9% | 46.0% | 54.1% |
| example-testing | 7.5% | 12.5% | 3.9% |
| initializing | 6.3% | 14.6% | 7.0% |
| uncertainty-estimation | 21.6% | 7.0% | 11.5% |

## Inter-annotator agreement (character-level)

| Pair | kappa (6-label) | kappa (target vs other) | agree (labeled) | agree (overall) |
|---|---|---|---|---|
| Sonnet-4.5 vs Qwen3-235B | 0.436 | 0.352 | 0.408 | 0.579 |
| Sonnet-4.5 vs Nova-Pro | 0.35 | 0.305 | 0.354 | 0.519 |
| Qwen3-235B vs Nova-Pro | 0.345 | 0.263 | 0.322 | 0.52 |

## Manifold replication (intrinsic dim / curvature at matched trough layers)

| Behaviour | Sonnet-4.5 cdim | Qwen3-235B cdim | Nova-Pro cdim | Sonnet-4.5 geo | Qwen3-235B geo | Nova-Pro geo |
|---|---|---|---|---|---|---|
| backtracking | — | — | — | — | — | — |
| uncertainty-estimation | — | — | — | — | — | — |
| example-testing | — | — | — | — | — | — |
| adding-knowledge | — | — | — | — | — | — |

**Read:** if cdim and geo are similar across annotators despite differing label distributions,
the manifold result is annotator-robust (the headline external-robustness claim).