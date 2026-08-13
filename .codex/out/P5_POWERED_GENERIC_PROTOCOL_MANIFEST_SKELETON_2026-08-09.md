# Draft powered P5 generic protocol/manifest skeleton

Status: **non-executable**. No generation, scoring, model, proxy, or pod call is authorized.

- Internal SHA-256: `639a1b7c48bb77ffa800ecc670178449753bdf0be2db387927e85bed4f61cb5d`
- File SHA-256: `abffad6d6452b3bc135e625aab2a6ea07415f5b76d62986ed9ba386514e0c78b`
- Phase-2 canonical manifest SHA: unset/blocking
- Primary analytic prefix: unset/blocking (recommended: first 4,096 generated token IDs or earlier EOS)
- Selected effect, power, alpha strategy, and task N: unset/blocking

The independent unit is the task. Checkpoint outputs are repeated observations nested within task. Primary inference uses task-paired contrasts, 10-category-stratified bootstrap resampling, 10,000 draws, seed 20260808, complete-pair missingness, and Holm correction across four endpoints by two primary contrasts.

The shared Phase-2 vanilla artefact is read-only and must be bound by manifest, internal-ID, row, and shard hashes. P5 may not regenerate missing shared rows. P5-owned roles require a separate generation manifest and authorization using identical task bytes, tokenizer/template, and decoding.

## Owner decisions

- Choose the primary analytic prefix: recommended 4,096 generated tokens; alternatives require new variance evidence.
- Choose the smallest effect from the frozen 0.02/0.03/0.05/0.075/0.10 grid and 80% or 90% power.
- Choose nominal versus conservative eight-test alpha for planning; Holm remains the analysis correction.
- Confirm whether the 100-task Phase-2 manifest is a hard maximum or may be prospectively expanded.
- Confirm whether DeepScaleR is secondary observational or part of the primary contrast family.

Execution remains blocked until every required hash and validation field in the JSON skeleton is populated and a new exact spend/scope authorization is issued.
