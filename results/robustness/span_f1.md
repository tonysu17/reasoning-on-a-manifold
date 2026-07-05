# Span-level F1 between annotators

Common chains: 1000. A span = one labelled annotation, located
occurrence-aware. F1 = 2*TP/(|A|+|B|) over greedy same-label matches; symmetric.
Complements the character-level Cohen's kappa in `cross_annotator_comparison.md`.

## Span counts (labelled spans on the common chains)

| Sonnet-4.5 | Qwen3-235B | Nova-Pro |
|---|---|---|
| 77562 | 88836 | 87632 |

## Span-F1 by matching criterion

| Pair | exact boundary | IoU ≥ 0.5 | any overlap |
|---|---|---|---|
| Sonnet-4.5 vs Qwen3-235B | 0.211 | 0.306 | 0.413 |
| Sonnet-4.5 vs Nova-Pro | 0.171 | 0.257 | 0.375 |
| Qwen3-235B vs Nova-Pro | 0.247 | 0.307 | 0.373 |

## Per-label span-F1 (IoU ≥ 0.5)

| Pair | backtracking | uncertainty-estimation | example-testing | adding-knowledge |
|---|---|---|---|---|
| Sonnet-4.5 vs Qwen3-235B | 0.297 | 0.279 | 0.245 | 0.208 |
| Sonnet-4.5 vs Nova-Pro | 0.233 | 0.259 | 0.174 | 0.149 |
| Qwen3-235B vs Nova-Pro | 0.265 | 0.26 | 0.182 | 0.181 |

**Read:** span-F1 at IoU≥0.5 should sit well above the character-level kappa,
since it forgives boundary jitter and scores the agreed-upon spans; the gap between
the *exact* and *overlap* columns is the size of the boundary-disagreement effect that
makes the character metric harsh. Per-label F1 localises where the annotators diverge.