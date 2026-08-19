# D3 — r1-distill-7b

Run `d3-r1-distill-7b-429801a55af0`; sealed protocol §5; seal 752dee7 + amendment 1. Diagnostic-only, non-gating.

Model `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` @ `916b56a44061fd5cd7d6a8fb632557ed4f724f60`; lens `/root/lenses/r1-distill-7b_wikitext100.pt` (n_prompts=100, layers 0–26); scored vocab 151665 of head 152064.

| Suite | n items | union pass@25 | perm p95 | p | qualifies | logit-lens union |
|---|---:|---:|---:|---:|:--:|---:|
| lens-eval-association | 98 | 0.0000 | 0.0000 | 1.0000 | no | 0.0000 |
| lens-eval-typo | 96 | 0.8125 | 0.0417 | 0.0010 | YES | 0.6458 |
| lens-eval-multihop | 81 | 0.0988 | 0.0247 | 0.0010 | YES | 0.0864 |

**Verdict:** {"n_qualifying_suites": 2, "qualifying_suites": ["lens-eval-typo", "lens-eval-multihop"], "note": "descriptive; interpretation conditional on D2 pass. Question: does this ~1-2B model clear the instrument-level bar on association or multihop?"}

