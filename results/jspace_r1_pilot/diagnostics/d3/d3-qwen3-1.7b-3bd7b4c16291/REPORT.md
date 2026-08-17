# D3 — qwen3-1.7b

Run `d3-qwen3-1.7b-3bd7b4c16291`; sealed protocol §5; seal 752dee7 + amendment 1. Diagnostic-only, non-gating.

Model `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; lens `qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt` (n_prompts=466, layers 0–26); scored vocab 151669 of head 151936.

| Suite | n items | union pass@25 | perm p95 | p | qualifies | logit-lens union |
|---|---:|---:|---:|---:|:--:|---:|
| lens-eval-association | 98 | 0.0306 | 0.0204 | 0.0180 | YES | 0.0000 |
| lens-eval-typo | 96 | 0.5938 | 0.0417 | 0.0010 | YES | 0.6146 |
| lens-eval-multihop | 81 | 0.5432 | 0.0988 | 0.0010 | YES | 0.5062 |

**Verdict:** {"n_qualifying_suites": 3, "qualifying_suites": ["lens-eval-association", "lens-eval-typo", "lens-eval-multihop"], "note": "descriptive; interpretation conditional on D2 pass. Question: does this ~1-2B model clear the instrument-level bar on association or multihop?"}

