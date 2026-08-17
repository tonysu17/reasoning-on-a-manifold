# D2 — qwen2.5-7b-it

Run `d2-qwen2.5-7b-it-f1b6cdfdff03`; sealed protocol §4; seal 752dee7 + amendment 1. Diagnostic-only, non-gating.

Model `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28`; lens `qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt` (n_prompts=485, layers 0–26); scored vocab 151665 of head 152064.

| Suite | n items | union pass@25 | perm p95 | p | qualifies | logit-lens union |
|---|---:|---:|---:|---:|:--:|---:|
| lens-eval-association | 98 | 0.0612 | 0.0102 | 0.0010 | YES | 0.0306 |
| lens-eval-typo | 96 | 0.4583 | 0.0312 | 0.0010 | YES | 0.4062 |
| lens-eval-multihop | 81 | 0.6173 | 0.0926 | 0.0010 | YES | 0.5247 |

**Verdict:** {"n_qualifying_suites": 3, "qualifying_suites": ["lens-eval-association", "lens-eval-typo", "lens-eval-multihop"], "scorer_exonerated": true, "sealed_pass_rule": ">=2 of 3 suites: union > permutation p95 and p <= 0.05", "consequence": "account (a) scorer/recipe defect REJECTED; D3 interpretable"}

