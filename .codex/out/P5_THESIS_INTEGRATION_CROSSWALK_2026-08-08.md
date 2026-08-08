# P5 thesis-integration crosswalk

**Date:** 8 August 2026  
**Current rendered thesis:** 62 pages; one page below the 63-page ceiling.  
**Current evidence state:** P5 prospective/unrun; no P5 result is admissible yet.

## Narrative consistency now

The authoritative chapters already maintain the correct evidence boundary:

- the abstract says corresponding post-training behavioural effects require separate evaluation;
- the introduction frames persistent post-training effects without claiming behavioural safety improvement;
- Methods explicitly says the teacher-forced comparison does not estimate generated behaviour;
- the safety chapter separates fixed-input representation change from refusal, benign compliance, task accuracy, and behaviour frequency;
- the conclusion identifies the controlled behavioural benchmark as the first follow-up.

One future-work mismatch was corrected in `chapters/v2/conclusion.tex`: the owned seed-42 full-parameter safety/matched-control pair is now the primary planned attribution contrast; LoRA arms are recovery/retrain-conditional; and safety prompts must be train-disjoint at record level. This is a plan correction, not a result claim.

## Result-to-chapter map after the evidence chain passes

| P5 object | Thesis destination | Required wording boundary |
|---|---|---|
| Checkpoint/prompt/generation design | Methods, training-time intervention subsection | State prompt as unit, seed limitation, paired generation, train-overlap exclusion, common tokenizer/input-ID gate, and missing-data rule. |
| Generic four-behaviour estimates and guards | Safety chapter, new free-generation subsection | Report behavioural sentence fractions separately from bt/1k, correctness, length, repetition, and truncation. |
| Owned full-FT safety versus control | Safety chapter, controlled behavioural subsection | A one-seed bounded checkpoint comparison, not recipe-level replication or general safety improvement. |
| Public STAR1 versus base | Safety chapter, checkpoint case study | Externally produced checkpoint contrast; do not merge with the owned attribution estimand. |
| Representation-behaviour association | Safety discussion/conclusion | Exploratory arm-level association; no mediation, circuit, or causal-mechanism claim. |
| P5 headline | Abstract/introduction/conclusion | Include only after source result, ledger/methodology, evidence snapshot, hashes, and claim-ledger row are complete. |

## Page-neutral integration budget

P5 will need roughly 1.5–2 pages for design, the primary owned-pair result, public case study, guards, and a compact synthesis figure/table. With only one free page, integration must remove at least 0.5–1 page elsewhere. Preferred cuts, made only when P5 lands, are:

1. replace the current repeated prospective-behaviour gap across Methods and the safety opening/answer with one Methods definition and one cross-reference;
2. compress the safety candidate-parent and repeated translation-versus-rotation qualification without removing any hedge or provenance disclosure;
3. replace prose repetitions of checkpoint/adapter point estimates with one compact result table;
4. keep detailed category, scorer-agreement, sensitivity, and per-seed tables in the authoritative appendix rather than the main chapter.

Do not cut the raw steering table, the one robustness paragraph, unresolved-provenance disclosures, failed/unrun controls, sample-unit statements, or estimand distinctions to make room.

## Admission gate

The thesis remains unchanged on P5 outcomes until all of the following exist: final machine-readable result; complete provenance; analysis ledger and methodology entries; refreshed evidence snapshot and manifest hashes; claim-ledger row; and a clean 63-page-or-fewer PDF rebuild. If P5 does not land, retain the current explicit behavioural gap.

