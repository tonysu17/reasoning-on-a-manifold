# D3 — gemma-3-1b

Run `d3-gemma-3-1b-5bf6d1b42fd0`; sealed protocol §5; seal 752dee7 + amendment 1. Diagnostic-only, non-gating.

Model `google/gemma-3-1b-pt` @ `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`; lens `gemma-3-1b/jlens/Salesforce-wikitext/gemma-3-1b-pt_jacobian_lens.pt` (n_prompts=467, layers 0–24); scored vocab 262144 of head 262144.

| Suite | n items | union pass@25 | perm p95 | p | qualifies | logit-lens union |
|---|---:|---:|---:|---:|:--:|---:|
| lens-eval-association | 101 | 0.0000 | 0.0099 | 1.0000 | no | 0.0000 |
| lens-eval-typo | 96 | 0.5417 | 0.0521 | 0.0010 | YES | 0.4583 |
| lens-eval-multihop | 81 | 0.3704 | 0.0494 | 0.0010 | YES | 0.5432 |

**Verdict:** {"n_qualifying_suites": 2, "qualifying_suites": ["lens-eval-typo", "lens-eval-multihop"], "note": "descriptive; interpretation conditional on D2 pass. Question: does this ~1-2B model clear the instrument-level bar on association or multihop?"}

