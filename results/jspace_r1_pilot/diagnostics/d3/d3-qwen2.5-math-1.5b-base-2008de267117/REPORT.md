# D3 — qwen2.5-math-1.5b-base

Run `d3-qwen2.5-math-1.5b-base-2008de267117`; sealed protocol §5; seal 752dee7 + amendment 1. Diagnostic-only, non-gating.

Model `Qwen/Qwen2.5-Math-1.5B` @ `4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`; lens `data/jlens_local/qwen2.5-math-1.5b_wikitext100.pt` (n_prompts=100, layers 0–26); scored vocab 151665 of head 151936.

| Suite | n items | union pass@25 | perm p95 | p | qualifies | logit-lens union |
|---|---:|---:|---:|---:|:--:|---:|
| lens-eval-association | 98 | 0.0102 | 0.0306 | 0.8372 | no | 0.0000 |
| lens-eval-typo | 96 | 0.8750 | 0.0625 | 0.0010 | YES | 0.6354 |
| lens-eval-multihop | 81 | 0.4074 | 0.0988 | 0.0010 | YES | 0.4630 |

**Verdict:** {"n_qualifying_suites": 2, "qualifying_suites": ["lens-eval-typo", "lens-eval-multihop"], "note": "descriptive; interpretation conditional on D2 pass. Question: does this ~1-2B model clear the instrument-level bar on association or multihop?"}

