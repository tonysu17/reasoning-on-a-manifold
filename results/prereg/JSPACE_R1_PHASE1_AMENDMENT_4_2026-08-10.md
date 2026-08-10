# J-space R1 pilot — Phase-1 post-failure, pre-ranking amendment 4

**Protocol marker:** amended

**Replacement execution status:** prospective/unrun at resealing

**Failed-run status:** current execution resource; no scientific gate result

**Sealed UTC:** 2026-08-10T22:43:38Z

**Parent amendment:** `results/prereg/JSPACE_R1_PHASE1_AMENDMENT_3_2026-08-10.md` (`d5aa9c31c66b650c4027a66acc2f0b0d1071dd67bf0ac306002aa7d88c4534a5`)

**Correction source commit:** `54a6538212091961d906b738f2ccc83c2f58f818`

**Failed run UUID:** `jspace-p1-20260810T221321Z-73a20baaafd1`

**Replacement run UUID:** `jspace-p1-20260810T224338Z-996077031021`

## 1. Exact stopping point of the failed run

The failed run completed fit A and fit B with 50 successful prompts each, the in-memory merge, six lens serializations, and the numerical/serialization gate. It then entered the first held-out row and the first position chunk (positions 16--47), computed the fit-A transported residual and raw unembedding logits, and stopped on the raw tensor's final dimension before calling the top-25 routine.

The terminal error was `readout vocab size 151936 != 151665`. The failed run's `FAILED.json` has SHA-256 `cf7145c7b54c9f0074139a401ded2baba42e32c1d75e2205f4913c80df38b015`; its `phase1_report.json` has SHA-256 `4d82bab67f2eca7d4722d3c05b6eedad96b157b412e6142e85f0fb84121b9def`; and its `numerical_validation.json` has SHA-256 `7497d8430855a752f420ed5cc4afb81f776adc197d9ed2923c514be0a3aaaaea`.

No top-25 token IDs, held-out or external ranks, Jaccards, permutation nulls, scores, positive-control results, or scientific gate values were computed or persisted. The failed output inventory contains only `RUN_ENVIRONMENT.txt`, `phase1_runner.log`, `phase1_report.json`, `numerical_validation.json`, and the six lens files. No held-out or external values were inspected while deriving this correction. The failure is an execution error, not a failed scientific validity gate.

## 2. Static source of the mismatch

The pinned model config and LM-head weight both define a padded raw output domain of 151,936 rows. The pinned tokenizer defines exactly 151,665 unique, contiguous token IDs, 0 through 151,664. Rows 151,665 through 151,935 are 271 output-head padding rows with no tokenizer token. This distinction is established from the already-pinned model and tokenizer files, independently of held-out or external readouts.

Amendment 1 explicitly registered the stability bijection over 151,665 vocabulary IDs. The primary ranking domain therefore remains the tokenizer-addressable domain rather than expanding post hoc to the padded head.

## 3. Frozen correction

For every fit-A, fit-B, merged, and vanilla-logit readout, the replacement runner must:

1. require the raw unembedding dimension to equal 151,936;
2. require every value across the complete 151,936-row head, including the padding tail, to be finite;
3. independently require the tokenizer IDs to be unique and exactly contiguous from 0 through 151,664;
4. restrict candidate logits to those 151,665 tokenizer IDs; and
5. apply the already-registered deterministic top-25 and all subsequent statistics on that restricted domain.

The stability-null bijection remains 151,665. Every stored primary top-25 ID must be below 151,665, as independently checked by the validator. The run report and validator must record and agree on `raw_head_domain=151936`, `primary_ranking_domain=151665`, `valid_token_id_min=0`, `valid_token_id_max=151664`, `stability_null_domain=151665`, and `excluded_head_padding_rows=271`.

The released J-lens adapter returns the complete model head, and its generic visualizer applies top-k to that complete width. The repaired primary readout is therefore a model-specific tokenizer-domain ranking, not bit-for-bit the generic full-head visualizer when padding rows exist. A padded-full-head comparison would be descriptive and non-gating; it is not added to this replacement execution and remains prospective.

## 4. Frozen executable hashes

| File | SHA-256 |
|---|---|
| `jspace_phase1_scoring.py` | `6f610e61dc13fb2adceb6d86151467cdf8c5cd4366522da39f9f3de1bc45d426` |
| `jspace_phase1_run.py` | `b7f5a27a291f66c623132dcdbc7bf4c557e1eded31b505f74702b26722db67fd` |
| `jspace_phase1_validate.py` | `289c391eefd914052bcb79563066070c7d80a539ba734cbda4a41ea8e2b6458f` |
| `tests/test_jspace_phase1_scoring.py` | `0119faedd3e59b015d8f706c16f9595e96ad61f677b6165a398cf7a0897b8797` |

Twenty synthetic/unit tests pass under both local Python environments. The added production-width test makes all 271 padding rows dominate the raw head and verifies that none can enter the primary top 25 while valid ID 151,664 is preserved. An independent code and methods review returned GO.

## 5. Replacement execution and non-reuse

The replacement uses fresh runtime, output, and scratch paths and `resume=False`. It reruns both original 50-prompt fits from scratch under the unchanged corpus, tokenizer, model, J-lens commit, estimator, seeds, gates, and validation rules. It does not load, continue from, or otherwise reuse the failed run's lenses or checkpoints. The failed run remains unchanged as a superseded execution resource and is not treated as Phase-1 validity evidence.
