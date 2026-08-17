# AMENDMENT 2 — D4 item selection under the skip-16 rule

**Date:** 2026-08-17, recorded **before D4 executes.** **Amends:** §6 of
`JSPACE_R1_DIAGNOSTIC_PROTOCOL_2026-08-16.md` (seal `752dee7`).
**Status: SEALED** by Tony in-session 2026-08-17 ("seal amendment 2"), option 1 below.
D4 runs with `--selection amended`.

## Problem (discovered before execution, from prompt lengths only)

Sealed §6 selects "the 8 eligible multihop items with lowest SHA-256(ASCII item name)" and fits each a
single-prompt Jacobian "with the released skip-16 rule". The released
`jlens.fitting.valid_position_mask` requires `seq_len > skip_first + 1`, i.e. **at least 18 R1 tokens**,
and raises otherwise. Prompt lengths were read from the sealed eligibility manifest (`prompt_token_ids`);
no model was run and no rank was inspected.

| # | sha256 prefix | item | R1 tokens | valid positions | fits? |
|---|---|---|---:|---:|:--:|
| 1 | 013383ee | pred-valentines-prevmonth | 15 | 0 | no |
| 2 | 036280cf | dual-photosynthesis-opposite | 17 | 0 | no |
| 3 | 0ab37d1c | rhyme-shoe-doubled | 14 | 0 | no |
| 4 | 0aca9bf9 | nhop-guitar-planet | 37 | 20 | yes |
| 5 | 0fbe027e | planet-3-moons | 18 | 1 | yes (n=1 position) |
| 6 | 0fcb7b5d | greatwall-ocean | 17 | 0 | no |
| 7 | 101f0d39 | nhop-primary-planet | 36 | 19 | yes |
| 8 | 1041de98 | violin-strings | 17 | 0 | no |

**5 of 8 cannot yield a single-prompt Jacobian at all**, and a sixth rests on one position. Across the
whole eligible multihop set, 48 of 81 items are ≤17 tokens. Sealed D4 would therefore compare 3 items
(effectively 2 well-supported), which is thin evidence for account (d).

## Proposed change

Replace the §6 selection sentence with:

> Items: the 8 eligible multihop items **whose prompts contain at least 18 R1 tokens** with lowest
> SHA-256(ASCII item name), ascending hex — fixed before any computation.

Nothing else changes: layers {17, 25}, `dim_batch=8`, `skip_first=16`, `max_seq_len=128`, BF16 model,
eager attention, censored-rank endpoint, descriptive-only status, no threshold.

**Why this is legitimate rather than gate-shopping.** The added criterion is a property of *instrument
applicability* (whether the estimator is defined for the prompt), decided from token counts alone, with
no rank, logit, or model output inspected — the same class of criterion as the existing single-token
label eligibility already sealed in Phase 1. It is declared before D4 runs.

**Residual bias to state in the report either way.** Length-eligible multihop items skew toward the
longer, multi-clause ("nhop-…") constructions; D4's conclusion about averaging is therefore bounded to
prompts long enough for a single-prompt Jacobian, and does not generalise to the 48 short items.

## Options presented to Tony (option 1 adopted)

1. **Seal this amendment** → D4 runs 8 length-eligible items (`--selection amended`). **ADOPTED.**
2. Decline → D4 runs the sealed 8, reporting 3 fitted and 5 `insufficient_valid_positions`.
3. Drop D4 → account (d) untested.

## Resulting selection (fixed at seal time, before any computation)

| # | sha256 prefix | item | R1 tokens | valid positions |
|---|---|---|---:|---:|
| 1 | 0aca9bf9 | nhop-guitar-planet | 37 | 20 |
| 2 | 0fbe027e | planet-3-moons | 18 | 1 |
| 3 | 101f0d39 | nhop-primary-planet | 36 | 19 |
| 4 | 1df04539 | nhop-volleyball-planet | 39 | 22 |
| 5 | 244d4be9 | func-filters-count | 18 | 1 |
| 6 | 320281ea | chem-organic-Z | 18 | 1 |
| 7 | 36e49b5b | func-pumps-chambers | 20 | 3 |
| 8 | 38ecc63a | etym-janus-monthnum | 19 | 2 |

**Additional caveat to carry into the report:** four of the eight rest on 1–3 valid positions, so their
"prompt-specific" Jacobian averages over very few token positions and is correspondingly noisy. The
per-item `n_valid_positions` is recorded by the runner and must be shown beside every rank; conclusions
about account (d) lean on the three 19–22-position items.
