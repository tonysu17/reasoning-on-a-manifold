# Integrated thesis refactor and post-training results plan

**Date:** 2026-08-08  
**Purpose:** durable context handoff for future sessions  
**Status:** integrated operating plan; not itself a preregistration, result artefact, thesis-scope amendment, or authority to spend  
**Repositories:** `reasoning-on-manifold/` is the analysis repository; `reasoning-on-manifold/thesis/` is a separate nested thesis repository

## 0. How to use this document

This document reconciles two workstreams:

1. the thesis refactor begun in the Codex task **“Refocus thesis on steering”**; and
2. the pre-refactor post-training transport programme recorded in
   `SESSION_HANDOFF_2026-08-08.md` and
   `../post_training_geometry_unified_plan_2026-08-02.md`.

It is an orientation and sequencing document. When it conflicts with a sealed preregistration,
machine-readable result, provenance record, thesis charter, or claim ledger, those sources
control. In particular:

- Phase-2 execution is governed by
  `results/prereg/PHASE2_TRANSPORT_PREREG_2026-08-08.md`.
- Thesis claim status is governed by the thesis `AGENTS.md`,
  `_planning/THESIS_REVISION_CHARTER_2026-07-26.md`, and
  `_planning/THESIS_CLAIM_ARTIFACT_LEDGER_2026-07-26.md` until a valid scope amendment
  supersedes them.
- Quantitative claims must be checked against machine-readable artefacts and provenance, not
  against this document or another prose summary.

All status words retain their precise meanings. `Amended` is a protocol marker, not an evidence
status. `Unrun`, `failed`, `provisional`, `exploratory`, and `current non-confirmatory` are not
interchangeable. Negative results remain local to the tested model, layer, statistic,
representation, control, and achieved sensitivity.

## 1. Executive decision

The two plans should be merged rather than treated as alternatives. The thesis's narrative spine
should be **intervention on machine reasoning at two timescales**:

1. **Inference-time intervention:** steering selected residual-stream coordinates while model
   weights stay fixed.
2. **Training-time intervention:** post-training changes model parameters and produces persistent
   representational and potentially behavioural effects.
3. **Bridge:** causal transport asks whether a steering handle remains accessible and causally
   coupled after post-training.
4. **Role of geometry:** geometry constructs and measures these interventions and constrains their
   interpretation; it is not assumed to be a universal reasoning manifold or a complete mechanism.

The strongest unifying question is:

> When post-training changes a model's representations or behaviour, what happens to the tested
> backtracking coordinate: is it retained, rescaled, rotated, decoupled, or inaccessible under the
> sealed intervention family?

The word `disabled` may be used only under the exact Phase-2 decision rule and adequate achieved
sensitivity. It never means that backtracking is absent everywhere. The existing E10 frame is the
sealed R1-1.5B L17 width-2 DAS frame; its thesis status remains bounded by the original design and
the new target-model gates. Do not promote it into an unqualified universal “grounded coordinate.”

## 2. The reconciled scientific structure

| Evidence plane | Question | Current status | Next result needed |
|---|---|---|---|
| Behaviour-indexed representation | What measurable structure is available for intervention? | Existing bounded and mixed geometry evidence | No broad new geometry programme |
| Inference-time behaviour | Do tested residual-stream operators change generated reasoning behaviour? | Amended, provisional backtracking result; other behaviours mixed or underpowered | Independent rescoring/accuracy guard is useful but second priority |
| Fixed-input post-training | What changes in activations when the same sequences are replayed through post-trained checkpoints? | Translation clearer than strong reshaping; recipe-direction attribution exploratory | Preserve as the descriptive post-training leg |
| On-policy post-training behaviour | What do post-trained models naturally generate differently? | Missing from the authoritative thesis result | Free-generation behavioural evaluation |
| Causal transport | Does the same backtracking handle remain functional after post-training? | Causal ledger empty across post-training boundaries | Sealed Phase 2 |
| Behaviour-targeted training | Does training explicitly toward/away from backtracking reuse the same coordinate? | Prospective | Phase 3 KTO; follow-on paper, not thesis-minimum |

This structure resolves the apparent tension between the two earlier plans:

- The **refactor plan** correctly identified the missing behavioural consequence of post-training.
- The **transport plan** supplies the mechanistic bridge that makes steering and post-training one
  scientific programme rather than two engineering case studies.
- Neither study substitutes for the other. Behavioural evaluation measures what the post-trained
  model does on policy; Phase 2 measures whether a particular causal intervention transfers.

## 3. What the current evidence can and cannot support

### 3.1 Steering

The existing steering study directly measures generated behaviour, but it is amended and
provisional. Backtracking is the strongest local result under the executed comparators. The study
does not establish general steerability, accuracy-preserving control, a universal manifold
advantage, or a complete mechanism.

### 3.2 Post-training representations

The existing safety/full-checkpoint and matched-adapter comparisons measure fixed-input,
teacher-forced representation change. The supported thesis-level description is that translation
is clearer than strong reshaping under the tested conditions. Rotation is bounded and mixed;
recipe-associated directionality remains exploratory. These analyses do not measure refusal,
benign compliance, task accuracy, or natural reasoning-behaviour prevalence.

### 3.3 Method-plane and RLVR results

The wider analysis repository contains descriptive SFT/DPO/GRPO/RLVR results, but they must not
become the thesis spine:

- the DPO safety-versus-control contrast still owes a dose-matched control;
- the GRPO arms are near-no-op and do not provide positive RLVR evidence;
- the DeepScaleR result is one public-checkpoint case study, not identification of a general RLVR
  effect;
- broad “SFT sharpens / DPO broadens / GRPO no-op” language collapses distinct designs, doses,
  provenance states, and evidence levels.

These results may motivate the checkpoint choice and future work, but should remain contextual or
exploratory unless separately audited and authorised.

## 4. Time-sensitive snapshot at creation

**This subsection is a snapshot, not a durable fact. Recheck it at the start of the next session.**

### 4.1 Analysis repository

- Branch: `codex/phase0-support`.
- The Phase-0 support wave and other analysis changes are uncommitted.
- Phase-0 pod s0 is complete and routed F5 to `FALLBACK_65PAIR`: the 500-pair route projected
  36.42 GPU-hours, beyond the 12-hour threshold.
- F5 fallback training is a manual gate and had not run at this snapshot.
- STAR1 deduction/initializing extraction completed into the pod's
  `data/activations/STAR1-1.5B/`; the wrapper's expected short-name path was wrong, producing the
  cosmetic `FAILED:s2` status. The extracted data must still be staged and checked locally.
- Phase-0 s3 had completed R1 extraction (200/200, zero failures) and was extracting DeepScaleR.
- `PH0_DONE.marker` and local `s3_analysis.json` were absent at the snapshot.
- `results/prereg/phase2_task_manifest.json` was absent.
- The Phase-2 design is sealed, but a complete Phase-2 execution runner has not yet been built.
- Local full-FT safety/control checkpoints exist under `checkpoints/pod_fullft/`.
- Local LoRA activation artefacts exist, but no local `adapter_model.safetensors` files were found;
  LoRA behavioural evaluation therefore requires exact checkpoint recovery or a provenance-clean
  retrain.

### 4.2 Thesis repository

- Branch: `main`.
- The intervention-first refactor is uncommitted and was still undergoing its second-pass audit.
- The rendered `ucl_msc.pdf` was 62 pages at this snapshot. Page count is live and may change as
  the concurrent audit finishes.
- The current draft uses the new title and three-RQ intervention framing, but the 26 July charter
  still freezes the old title and four RQs pending a joint dated author-supervisor decision.

## 5. Priority-ordered execution plan

### Priority 0 — finish and sign Phase 0

Do not launch Phase 2 until Phase 0 is closed under the sealed launch condition.

After `PH0_DONE.marker` appears:

1. Pull the normal Phase-0 outputs with `POD=runpod ./runpod_phase0.sh pull`.
2. Pull the STAR1 inert activations into a staging directory, never into the local canonical July
   `data/activations/STAR1-1.5B/` directory.
3. Verify `s3_analysis.json`:
   - aggregation gate against the authoritative DeepScaleR paired median dPR;
   - singular-value family check;
   - calibrated mapping only where the family rule permits it.
4. Run the local deduction/initializing inert-control battery with row-index/token-start parity,
   the preregistered layers, and the preregistered Holm family.
5. Decide the F5 fallback while the pod is live:
   - recommended if operationally cheap: run the approximately 20-minute fallback solely to close
     the owed repository control;
   - retain the data-repetition confound;
   - do not make it load-bearing for the refocused thesis.
   - if it is not run, record `prospective/unrun` or the exact closure disposition; do not call it
     failed.
6. Append the s2, s3, inert-control, and F5 dispositions to the Phase-0 freeze and results ledger.
7. Commit the analysis-repository closure as one logical pass with exact provenance.
8. Terminate the pod from the RunPod console only after verified pulls.

### Priority 1 — obtain a valid scope/resource amendment

The current three-RQ/title refactor supersedes the 26 July charter. Before treating it as the final
submission scope or committing new result spend, obtain a joint dated written author-supervisor
decision that records:

- the new title and subtitle;
- the three-RQ intervention framing;
- the changed evidence basis;
- Phase 2 as thesis-relevant new evidence;
- whether the behavioural benchmark is thesis scope;
- compute, annotation, and schedule authority;
- the treatment of the post-charter exploratory appendix.

A suitable prospective RQ3 is:

> **RQ3:** How does post-training change generated reasoning behaviour and generic-reasoning
> representations, and does the tested backtracking coordinate remain causally coupled across the
> selected post-training boundaries?

Do not write the causal-transport outcome into this RQ before the result exists.

### Priority 2 — add a bounded behavioural adjunct before Phase-2 generation

Phase 2 already includes a vanilla arm for R1, STAR1, and DeepScaleR. Before any Phase-2
generation, append a dated preregistration amendment if the following observational endpoints are
to be used:

- prevalence of all four annotated reasoning behaviours;
- backtracking count per 1,000 generated tokens;
- boxed exact-match/task accuracy;
- response length;
- repetition and truncation.

Rules:

- This is a separate observational secondary/exploratory family, not the Phase-2 causal primary.
- Freeze the annotation prompt, unit, aggregation, exclusions, multiplicity family, and missing-row
  handling before generation.
- Reuse the same 100 task-paired vanilla generations where possible; do not imply that this makes
  annotation free.
- Do not treat STAR1 versus DeepScaleR as an objective-level SFT-versus-RLVR causal comparison;
  they differ in data, recipe, dose, selection, and provenance.

This adjunct supplies a generic-reasoning on-policy result at low additional generation cost. It
does not supply harmful-refusal or benign-compliance evidence.

### Priority 3 — implement and test the Phase-2 executor

The sealed design still requires execution code. Build the smallest auditable pipeline that
implements the preregistration exactly:

1. deterministic task-manifest drawing, exclusion checks, stratification, hashing, and commit
   verification;
2. teacher-forced target discovery extraction on byte-identical R1-tokenizer input IDs;
3. transported raw, scalar-norm, and diagonal-whitened frames;
4. target sham/random-orthogonal gates and L16/L18 neighbour controls;
5. sealed target frame re-fit at L17 with width in `{1,2}` and no target-specific tuning;
6. resumable generation for the exact causal battery, signs, doses, floors, and vanilla arm;
7. pre-outcome base-model injection-recovery;
8. non-builder annotation with missing rows unresolved rather than coerced to zero;
9. paired task-level analysis, Holm correction, damage gates, and the five-outcome hierarchy;
10. complete provenance payload: code/config hashes, commit and dirty state, checkpoint revisions,
    task hash, tokenizer identity, frame source, counts, seeds, annotator record, and achieved MDE.

Required pre-spend checks:

- unit tests for pooling and floor pairing;
- identity/reload and byte-identical-input gates;
- synthetic or tiny plumbing dry run without scientific interpretation;
- resume/idempotency test;
- failure-path test for missing annotations and gate failure;
- output-schema and provenance validation.

### Priority 4 — launch sealed Phase 2

Launch only after all three standing preconditions are satisfied:

1. Phase-0 closure is signed;
2. the 100-task manifest is drawn, hashed, and committed;
3. Tony explicitly approves the approximately `$20–40` pod cost and `$100–250` Nova-Pro
   annotation envelope.

Execution order matters:

1. resolve provenance, input alignment, discovery extraction, and target gates;
2. run the base-model injection-recovery gate;
3. evaluate achieved sensitivity before target outcomes are inspected;
4. if the gate fails, follow the sealed `150 → 200 → 300` enlargement path with new spend
   approval at each step;
5. if no approved battery passes, remove `retained` from the vocabulary as preregistered;
6. run target outcomes, damage gates, paired inference, and the outcome hierarchy without layer,
   width, dose, or prompt search toward a positive result.

The desired product is not necessarily a positive transfer. A bounded retained, rescaled,
rotated, decoupled, inaccessible, or sensitivity-limited result all answer the study if reported
under the sealed rules.

### Priority 5 — controlled post-training behavioural benchmark

This study fills the specific gap identified by the thesis refactor: whether the post-training
conditions change generated behaviour, not only teacher-forced activations.

#### Initial checkpoint set

Use a regime-matched hierarchy rather than one undifferentiated model list:

1. R1-1.5B base;
2. public STAR1 full SFT as an externally produced checkpoint case study;
3. locally available owned full-FT safety seed 42;
4. locally available owned matched full-FT non-safety seed 42;
5. optional LoRA safety/control seeds 42–44 only after exact weights are recovered or retrained
   under immutable run manifests.

DeepScaleR belongs in the generic-reasoning/transport comparison, not in the safety-versus-matched
non-safety attribution contrast.

#### Prompt strata

Keep two strata separate:

- **Generic reasoning:** task-paired free generations measuring the four behaviours, boxed
  correctness, length, repetition, and truncation.
- **Safety:** held-out harmful requests and benign look-alikes measuring harmful-request refusal
  and benign compliance, plus degeneration guards.

The existing `data/grpo_refusal_prompts.json` is partly derived from STAR-1 material. It requires a
record-level train-overlap audit and cannot automatically be treated as held out. The safety
evaluation set must exclude prompts or near-duplicates used to train the evaluated safety model.

#### Statistical and provenance rules

- prompt/problem is the scientific unit;
- use paired prompts across checkpoints;
- training seed is a higher-level unit when multiple seeds exist;
- conduct a pipeline/base-rate pilot before freezing the powered sample size, but do not turn the
  pilot into a claim;
- freeze primary endpoints and multiplicity before the full run;
- separate safety-recipe attribution from public-checkpoint differences;
- use a non-builder scorer where feasible;
- keep free-generation behaviour distinct from teacher-forced representation transport;
- record task hashes, checkpoint hashes/revisions, generation settings, seeds, scorer version,
  missing rows, and complete input lineage.

If only the one-seed full-FT safety/control pair is available, report it as a bounded checkpoint
comparison with the seed limitation. Do not present row or prompt resampling as training-recipe
replication.

### Priority 6 — integrate results into the thesis through the evidence chain

Do not edit the thesis directly from a notebook, chat summary, or analysis-repository prose report.
For each new result:

1. finalise the machine-readable artefact and provenance record;
2. update `RESULTS_LEDGER.md` and `METHODOLOGY.md` in the analysis repository;
3. refresh the thesis evidence snapshot using `_planning/review/refresh_evidence.sh` from an
   environment that can see both repositories;
4. verify source commit and sha256 entries in the evidence manifest;
5. add new claim-ledger rows with model, unit, N, representation, layer rule, script, artefact,
   protocol marker, evidence status, provenance status, and thesis disposition;
6. only then update Methods, post-training Results, abstract, introduction, and conclusion;
7. rebuild and report the page count in the commit message.

## 6. Thesis integration decision rules

| What lands | Permitted thesis synthesis |
|---|---|
| Phase 2 + controlled behavioural benchmark | Relate fixed-input representation change, natural generated behaviour, and causal coupling, while keeping their estimands distinct |
| Phase 2 + vanilla behavioural adjunct only | Report generic-reasoning behaviour and coordinate transport for R1/STAR1/DeepScaleR; no controlled safety-recipe behavioural attribution |
| Phase 2 only | Report the bounded transport outcome; do not claim safety improvement or natural behavioural change |
| Behavioural benchmark only | Report output changes under tested checkpoints/recipes; do not claim coordinate reuse or causal transport |
| Neither new study lands | Retain the current translation-dominant fixed-input post-training result and the explicit behavioural gap |

Outcome wording must remain local:

- **Retained:** only within achieved detectable attenuation and tested frame/battery.
- **Rescaled:** only where the predeclared correction recovers effect without excess damage.
- **Rotated:** requires target re-fit recovery plus null-calibrated misalignment.
- **Decoupled:** representational signal remains while the causal effect weakens under adequate
  sensitivity.
- **Disabled/inaccessible:** only under the sealed hierarchy, search family, and adequate
  sensitivity; never global absence.
- **Inconclusive:** the correct verdict when sensitivity, provenance, execution, or controls are
  inadequate.

## 7. Page-budget rule

The rendered thesis was **62 pages at this document's snapshot**, with the page count still live
because the refactor audit was in progress. The contractual ceiling is 63 pages. New results must
be integrated by replacement and compression, not simple addition.

Preferred page-neutral moves:

1. replace the prospective post-training behavioural paragraph with the executed result;
2. replace the existing post-training figure with a combined
   representation/behaviour/transport figure rather than adding another float;
3. compress adapter decomposition, seed, and sensitivity detail into one table/caption while
   preserving the seed-level and provenance limits;
4. shorten the candidate-parent context to the minimum needed for checkpoint lineage;
5. replace conclusion future-work prose with the bounded new verdict;
6. if more space is still required, obtain explicit approval before cutting or relocating the
   post-charter exploratory appendix.

Every thesis pass that changes length must rebuild `ucl_msc.tex`, report the new page count, and
state the compensating cut.

## 8. Explicitly deferred work

Until Phase 2 and the controlled behavioural benchmark are complete, defer:

- the α-dial calibration organism;
- pt21 gpt-oss-safeguard versus gpt-oss;
- broad RLVR-family sweeps;
- Phase 3 behaviour-targeted KTO;
- agentic/tool-use transport and replay;
- sequential order effects;
- multi-agent geometry;
- geometry-aware training;
- categorical method-plane claims.

Phase 3 KTO is the strongest follow-on paper after the thesis-minimum programme because it directly
tests whether post-training toward or away from backtracking reuses, rotates, or bypasses the same
coordinate used by activation steering. It should not delay the thesis-minimum result.

## 9. Start-of-session checklist

A future session should begin by checking, in order:

1. Read this file, `SESSION_HANDOFF_2026-08-08.md`, the Phase-0 freeze, and the sealed Phase-2
   preregistration.
2. Recheck `PH0_DONE.marker`, `PH0_STATUS`, pod process state, and whether `s3_analysis.json` has
   landed.
3. Check both Git repositories separately: branch, status, current commit, and uncommitted files.
4. Check whether the “Refocus thesis on steering” audit finished and note its final page count.
5. Check whether a joint dated thesis-scope/resource amendment now exists.
6. Check whether `phase2_task_manifest.json` and a tested Phase-2 runner now exist.
7. Check exact checkpoint availability, especially LoRA seed weights versus activation-only
   artefacts.
8. Do not launch paid generation or annotation without satisfying the sealed preconditions.
9. Do not integrate a number into the thesis before it enters the evidence snapshot and claim
   ledger with provenance.

## 10. Canonical document chain

Read these sources for authority and detail:

1. `SESSION_HANDOFF_2026-08-08.md` — operational Phase-0/Phase-2 handoff.
2. `../post_training_geometry_unified_plan_2026-08-02.md` — full staged programme and red-team.
3. `results/prereg/PHASE0_TRANSPORT_FREEZE_2026-08-02.md` — Phase-0 rules and executed
   dispositions.
4. `results/prereg/PHASE2_TRANSPORT_PREREG_2026-08-08.md` — sealed Phase-2 execution authority.
5. `RESULTS_LEDGER.md` and `METHODOLOGY.md` — canonical living result/method trackers.
6. `thesis/AGENTS.md` — thesis evidence, hedge, file-authority, and build contract.
7. `thesis/_planning/THESIS_REVISION_CHARTER_2026-07-26.md` — current frozen scope authority.
8. `thesis/_planning/THESIS_CLAIM_ARTIFACT_LEDGER_2026-07-26.md` — current claim-to-artefact
   mapping.
9. `thesis/chapters/v2/safety.tex` — current post-training chapter and explicit behavioural gap.
10. `thesis/chapters/v2/conclusion.tex` — current intervention synthesis and priority follow-ups.

## Bottom line

The thesis should not become a catalogue of steering methods, post-training recipes, and geometric
statistics. Its coherent contribution is a bounded intervention study:

> steering provides the inference-time manipulation; post-training provides the persistent model
> change; fixed-input geometry describes what moved; free generation measures what behaviour
> changed; and causal transport tests whether the same internal handle remains functional.

The immediate critical path is therefore:

> **Phase-0 closure → scope/resource amendment → behavioural adjunct seal → Phase-2 runner and
> manifest → Phase-2 causal transport → controlled behavioural benchmark → evidence-snapshot and
> page-neutral thesis integration.**
