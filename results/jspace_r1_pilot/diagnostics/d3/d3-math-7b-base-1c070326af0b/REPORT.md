# D3 — math-7b-base

Run `d3-math-7b-base-1c070326af0b`; sealed protocol §5; seal 752dee7 + amendment 1. Diagnostic-only, non-gating.

Model `Qwen/Qwen2.5-Math-7B` @ `b101308fe89651ea5ce025f25317fea6fc07e96e`; lens `/root/lenses/math-7b-base_wikitext100.pt` (n_prompts=100, layers 0–26); scored vocab 151665 of head 152064.

| Suite | n items | union pass@25 | perm p95 | p | qualifies | logit-lens union |
|---|---:|---:|---:|---:|:--:|---:|
| lens-eval-association | 98 | 0.0102 | 0.0102 | 0.1808 | no | 0.0306 |
| lens-eval-typo | 96 | 0.7812 | 0.0521 | 0.0010 | YES | 0.6771 |
| lens-eval-multihop | 81 | 0.6420 | 0.1296 | 0.0010 | YES | 0.5679 |

**Verdict:** {"n_qualifying_suites": 2, "qualifying_suites": ["lens-eval-typo", "lens-eval-multihop"], "note": "descriptive; interpretation conditional on D2 pass. Question: does this ~1-2B model clear the instrument-level bar on association or multihop?"}

