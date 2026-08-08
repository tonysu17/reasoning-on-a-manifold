# P5 pilot and pre-spend QA plan

**Date:** 8 August 2026  
**Status:** zero-cost design and local audits complete; model generation, retraining, and paid scoring unrun.  
**Execution gate:** Tony's explicit pilot spend authorisation. The received handoff confirms that Phase-2 completion and `analyse.done` are not pilot prerequisites; the final generic study separately waits for the Phase-2 manifest SHA.

## Completed without spend

- Confirmed the two owned full-FT seed-42 checkpoints exist and computed full weight hashes.
- Confirmed the public/base immutable checkpoint revisions and full cached weight hashes. All four core checkpoint hashes pass the executable inventory gate.
- Received definitive local-disk and full-volume search disposition: the pt02 safety-SFT LoRA adapters do not survive. LoRA evaluation remains provenance-clean retrain-conditional; surviving RL adapters are outside the four-checkpoint pilot.
- Audited all 500 `grpo_refusal_prompts.json` records against the STAR-1 safety and MetaMathQA control training manifests. All 250 harmful prompts exactly overlap STAR-1 training; none overlap the matched control training manifest.
- Recomputed all twelve raw steering table cells from the current read-only source arrays and verified the thesis rounding.
- Compiled both local audit scripts successfully. Neither script writes outside `.codex/out`.
- Implemented `p5_preflight.py` and a 48-test suite covering prompt schemas, paired safety manifests, metric denominators, missingness, generation, six-label behaviour-annotation, and safety-score schemas, provenance fields, checkpoint/tokenizer identity, deterministic expected input-ID manifests, prompt-paired category-stratified bootstrap intervals, Holm adjustment, idempotent resume, frozen-scorer integrity/fail-closed behaviour, and spend refusal. All 48 tests pass. Empty sentence annotations remain unresolved, every scored/annotated row is bound to its generation-record hash, and complete-pair/missingness counts travel with each bootstrap estimate. The safety rubric is frozen as `p5-safety-rubric-1` (sha256 `b91f4df77c9287f8e04d5ac9ce7ddcefed0af7bab4cf75bcb88a8fd1fa41f806`) with six parser/endpoint fixtures (sha256 `d871c6a7c0cabb4299a57115425dd4bb78e78a4d0fd54d58108d0808336a7585`). The 24-prompt safety pilot manifest is frozen and passes the union training-overlap audit. The offline common-base-tokenizer gate passes for all 44 generic/safety pilot prompts across all four checkpoint roles with zero hash mismatches (`P5_INPUT_ID_GATE_2026-08-08.json`, sha256 `7c585f15410bf5db8bf05a59d6f372bbc18cbbeeb53efc67627f677ff9c0aaef`). With the received handoff, pilot readiness now waits only for Tony's spend authorisation; final-study readiness separately remains blocked on the Phase-2 manifest SHA and final safety manifest/audit.

## Stage A — pre-spend tests

These tests must pass before any model is loaded for the P5 pilot.

| ID | Test | Pass condition | Failure action |
|---|---|---|---|
| A1 | Checkpoint inventory | Every admitted arm has an immutable ID/path, weight/revision hash, config hash, role, and training-manifest status. | Exclude the arm; do not infer identity from an activation-directory name. |
| A2 | LoRA admission | Adapter plus merged hashes and immutable train manifest exist. | Keep LoRA extension conditional; do not retrain without separate spend approval. |
| A3 | Final-manifest separation | Pilot and final prompt IDs/hashes are disjoint. | Redraw/freeze before generation. |
| A4 | Train-overlap audit | Final safety prompts have zero exact/near matches against the union of admitted training manifests under the frozen rule. | Exclude matches for every checkpoint and replace before freeze. |
| A5 | Prompt schema | Unique IDs; required stratum/category/pair/text fields; no empty prompts; stable whole-file hash. | Refuse execution. |
| A6 | Common tokenizer/template | Every checkpoint produces byte-identical input IDs for every pilot prompt under the base alias. | Refuse the failing arm; no checkpoint-specific tokenization exception. |
| A7 | Output schema fixtures | Synthetic complete, truncated, repeated, empty, errored, and resumed records validate; missing values remain null/unresolved. | Fix parser/runner before GPU use. |
| A8 | Metric fixtures | Hand-worked sentence fractions, bt/1k, exact correctness, length, four-gram repetition, cap-hit, refusal, and compliance cases match expected values. | Fix metric implementation before generation. |
| A9 | Scorer dry run | Rubric prompt and structured-output parser accept fixed local fixtures with no API call. | Revise rubric/schema, version it, and repeat. |
| A10 | Provenance dry run | A mock record contains all input/checkpoint/tokenizer/code/environment hashes and attempt history. | Refuse execution until complete. |
| A11 | Spend refusal | Generation/annotation commands refuse without an explicit authorisation flag and manifest hash. | Treat as a blocking defect. |
| A12 | Resume/idempotence | Re-running a completed mock stage neither duplicates rows nor changes hashes; an interrupted row preserves the prior attempt. | Fix before execution. |

## Stage B — spend-gated pilot design

The pilot is for pipeline reliability and base-rate/variance planning only. It is never combined with the full study and never cited as a result.

### Sample

- **Generic:** the existing 20-task category-balanced `data/tasks_pilot.json`, or another frozen 20-task manifest, excluded from the final P5 manifest.
- **Safety:** 24 newly frozen held-out pilot prompts: 12 harmful and 12 benign look-alikes, balanced across the intended primary categories and screened against all admitted training manifests.
- **Checkpoints:** base R1, public STAR1, owned safety full FT seed 42, and owned matched non-safety full FT seed 42.
- **Generation volume:** 44 prompts × 4 checkpoints = 176 greedy generations, before any identical-setting retries for infrastructure failures.
- **Scoring:** all 176 receive annotation-free metrics; behavioural annotation applies to generic outputs; safety-rubric scoring applies to safety outputs. Independently double-score a frozen 20% subset, balanced by stratum/checkpoint.

### Exact pilot authorisation line

- **Generation:** 44 prompts × 4 checkpoints = **176 greedy generations**. Scaling the sealed Phase-2 4090 envelope and allowing checkpoint-load overhead gives an estimated `$1–3`; hard authorisation ceiling **`$3`**.
- **Scoring:** 176 primary annotations/safety scores plus a frozen 20% independently double-scored subset (`ceil(176 × 0.20) = 36`) = **212 proxy scoring calls**. The linear planning estimate from the sealed Phase-2 Nova-Pro envelope is approximately `$6–14`; hard authorisation ceiling **`$15`**.
- **Requested total ceiling:** **`$18`**. Execution must enforce the fixed 176-generation and 212-score call counts; retries do not expand either count or the budget envelope without a new authorisation. Because the proxy does not expose a guaranteed dollar pre-check, the dollar figures are conservative planning ceilings derived from the sealed Phase-2 envelope, while call counts are the mechanically enforceable limit.

Pilot execution requires Tony's explicit approval of this `$3 generation + $15 scoring` line. LoRA training/evaluation is not part of this pilot unless separately admitted.

### Pilot pass conditions

1. All 176 planned row keys are present or have an explicit terminal error; no silent omissions or duplicates.
2. Input-ID hashes are identical across all four checkpoints for every prompt.
3. At least 99% of successful raw generations pass the output schema; every failure is retained with its reason.
4. Behaviour and safety scorer structured outputs parse for at least 98% of submitted rows after identical retry policy; unresolved rows remain missing, never zero.
5. Non-builder double scoring reaches at least 90% agreement on the binary safety decisions and no primary class has Cohen's kappa below 0.60. If prevalence makes kappa unstable, report the contingency table and require rubric adjudication rather than waiving the gate.
6. No checkpoint has more than 5% empty/degenerate outputs.
7. If more than 5% of otherwise valid outputs hit 4,096 tokens, freeze 6,144 for every checkpoint in the full study. If the common 6,144 cap is infeasible, stop and redesign rather than use arm-specific caps.
8. Exact/reference-answer coverage and scorer uncertainty are sufficient to define the correctness estimand; unavailable references remain a reported coverage gap.
9. Harmful-refusal and benign-compliance pilot prevalences are not structurally all-zero/all-one solely because of prompt or rubric failure. Real model ceiling/floor is recorded and handled in power planning, not “fixed” by selecting prompts against desired model outputs.
10. The generated provenance report reproduces every input and checkpoint hash and records complete missingness/attempt history.

## Stage C — powered-size freeze

After the pilot passes, use only blinded aggregate variance/base-rate information to simulate paired-power or interval-width curves across candidate prompt counts. Freeze:

- the final generic and safety sample sizes;
- smallest effect/interval width of interest;
- the proposed 5-point safety/usefulness and damage margins after supervisor confirmation;
- endpoint families, Holm correction, 10,000 paired category-stratified bootstrap draws, and resampling seed;
- checkpoint set and every prompt-manifest hash;
- scorer/rubric version, double-score fraction, generation cap, and missing-data rules.

Do not inspect arm-labelled pilot differences when choosing sample size or prompts. The powered run requires a new explicit spend/scope authorisation.

## Stage D — later execution and integration gates

1. Run the powered benchmark once under the frozen manifest and settings.
2. Finalise machine-readable result/provenance objects before narrative inspection.
3. Update analysis ledger/methodology, then refresh the thesis evidence snapshot.
4. Integrate P5 into the thesis page-neutrally only after the evidence chain passes.
5. Run the steering replication last, after the behavioural work, as specified by the integrated plan.

## Explicitly not authorised or completed

- no pilot or powered model generation;
- no API/non-builder scoring;
- no LoRA recovery or retraining;
- no edits to `ph2_*`, `results/`, sealed preregistrations, pod state, `RESULTS_LEDGER.md`, or `METHODOLOGY.md`;
- no thesis claim promotion and no removal of existing provenance caveats.
