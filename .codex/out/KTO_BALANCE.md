# KTO label balance — pt18 construction only

**Status:** labels built; no training run. **The strict file is an audit subset, not training-ready**, because the available correctness pilot leaves too few matched high-backtracking chains.

## Leakage and split order

Problem-level split was created first with seed 20260802, stratified by the ten task categories: train/discovery/eval = 600/200/200.
Pairwise problem-id intersections: `{'discovery&eval': 0, 'discovery&train': 0, 'eval&train': 0}`. **Zero overlap: True**.
Backtracking quartiles and all matching decisions were computed only after this check, and only inside the train split.

## Label construction

Among 597 annotation-complete train chains, exact rank quartiles selected 149 bottom and 149 top candidates. Bottom selected max = 0.000000 and top selected min = 2.197266 backtracking-labelled sentences per 1,000 generated tokens.
Ties were ordered by a seeded random key; this matters because the linear 25th percentile is 0.000000 and many chains have zero backtracking labels.
Strict exact matching retained 2 chains per class across (observed correctness x fixed length-band) strata.

Correctness coverage is limited to the existing 200-chain pilot, whose provenance says `allow_truncated=false`. Missingness is therefore not plausibly random with respect to length: unknown correctness was not treated as a value and was excluded from the strict file. The larger candidate file is audit-only until correctness is completed under a sealed rule that covers cap-hit chains.

## Strict matched balance

| class | n | correctness counts | length-band counts | difficulty proxy | mean tokens | mean lexical token H (bits) | mean bt/1k |
|---|---:|---|---|---|---:|---:|---:|
| desirable | 2 | `{'False': 1, 'True': 1}` | `{'2048_4095': 1, 'lt_2048': 1}` | `{'moderate': 2}` | 2225.0 | 6.727 | 2.434 |
| undesirable | 2 | `{'False': 1, 'True': 1}` | `{'2048_4095': 1, 'lt_2048': 1}` | `{'hard': 1, 'moderate': 1}` | 1829.5 | 6.844 | 0.000 |

Correctness and length-band counts are equal by construction. Difficulty and lexical token entropy are reported diagnostics, not additional matching variables.

`lexical_token_entropy_bits` is Shannon entropy over lower-cased lexical token types in the generated chain. It is a deterministic text-diversity proxy, **not** next-token predictive entropy from the model; predictive-entropy coverage is not available for the full label set.

## Shuffled control

The control contains 4 chains. Labels were permuted within the same correctness x length strata, so class totals and those two matching margins are preserved while label assignment is randomized subject to the fixed seed. With only four strict rows this control is an implementation audit, not an inferential comparator; a permutation may retain some original labels.

## Files

- `kto_labels_splits.json`: frozen problem IDs and leakage audit.
- `kto_labels_candidates.json`: all top/bottom train-quartile candidates; includes missing correctness and is not training-authorised.
- `kto_labels_train.json`: strict correctness-observed, exactly matched **audit subset**; not training-ready at the current coverage.
- `kto_labels_shuffled_control.json`: within-stratum label-shuffled control.
