# P5 safety-prompt train-overlap audit

**Date:** 8 August 2026  
**Status:** completed, zero-cost planning audit; not a benchmark result.

The current `data/grpo_refusal_prompts.json` has 500 unique normalised prompts: 250 harmful STAR-1 prompts and 250 benign `tasks_final` prompts. Exact matching uses Unicode NFKC, case-folding, and collapsed whitespace. The conservative near-duplicate screen uses word 5-gram Jaccard similarity at a threshold of 0.80 after the same normalisation.

| Evaluated checkpoint role | Training manifest audited | Exact overlaps | Near-only overlaps | Consequence |
|---|---|---:|---:|---|
| Public STAR1 full SFT | `data/safety_star1_sft.json` (1,000) | 250/500, all harmful | 0 | Current harmful half is training data and is inadmissible as held-out evaluation. |
| Owned full-FT safety seed 42 | `data/safety_star1_sft.json` (1,000) | 250/500, all harmful | 0 | Same exclusion. |
| Owned matched full-FT non-safety seed 42 | `data/control_offpolicy_sft.json` (1,000) | 0/500 | 0 | No overlap found under these rules; final benchmark still uses one union-disjoint manifest across all arms. |
| R1 base | No experiment-specific training manifest available | not auditable | not auditable | Do not call this a pretraining-contamination audit. |

Input hashes and all record-level exact matches are stored in `P5_TRAIN_OVERLAP_AUDIT_2026-08-08.json`. The audit proves that the existing harmful prompts cannot be used for held-out safety claims about either STAR1 safety-trained checkpoint. The final safety benchmark therefore requires a new harmful/benign-lookalike manifest screened against the union of every admitted post-training manifest. The old pool may be used only for a clearly labelled non-claim pipeline smoke test.

