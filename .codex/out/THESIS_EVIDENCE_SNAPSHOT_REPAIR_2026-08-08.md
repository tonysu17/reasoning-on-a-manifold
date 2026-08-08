# Thesis evidence-snapshot repair specification

**Date:** 8 August 2026  
**Status:** repair specification and read-only arithmetic audit; no evidence snapshot, result artefact, ledger, methodology file, preregistration, or thesis claim was changed.

## Outcome of the immediate audit

The current thesis wording is correctly conservative. The raw steering table can be recomputed from the analysis repository's present generation and annotation arrays, and all twelve cells reproduce the values printed in the thesis after rounding. It cannot yet be recomputed from the frozen thesis evidence snapshot because neither the paired annotation array nor a compact raw-effect summary is present there.

The audit output is `STEERING_RAW_RECOMPUTATION_CANDIDATE_2026-08-08.json`. It is deliberately labelled **non-authoritative**. It establishes arithmetic consistency with the files currently on disk; it does not repair their historical execution lineage.

The source run record still has `git_commit: null` and no input hashes. Therefore the thesis's unresolved-provenance boundary must remain even after a compact summary is added.

## Missing evidence-chain objects

| Claim area | Source object | Bytes | sha256 | Required treatment |
|---|---|---:|---|---|
| Raw steering generations | `results/eval/R1-1.5B__E1/steering_results.json` | 38,946,681 | `b9f4eedfa535f2979d40361051016dd697a2f1971716a075479b3523a8f108d2` | Record by hash/size; do not copy if snapshot size is a concern. |
| Raw steering annotations | `results/eval/R1-1.5B__E1/annotated_steered.json` | 72,137,651 | `cfcc458db3ea376f33be7daaf8eda811916696a7dc1de79470d2d48b0214d392` | Record by hash/size and retain externally; a hash alone does not make the table recomputable. |
| Direction decomposition | `results/safety_posttrain/pt04b_decomposition.json` | 3,169 | `35c130afc9907b0d572e014fc43bfd14292d71acd70cb1c5540a4ecbfec2942b` | Add to the next evidence snapshot. |
| Per-seed directions | `results/safety_posttrain/pt06_perseed.json` | 1,724 | `8181cbc068aeaa422b2acde1b2f17bf8f3f99fc602a5e0cda58dce1a863733e1` | Add to the next evidence snapshot. |
| Sensitivity analyses | `results/safety_posttrain/pt07_sensitivity.json` | 7,841 | `36ea56c123cbde31797a37a460370dffd48a6acdd8bca96eda01aab77cc9101e` | Add to the next evidence snapshot. |

The three safety JSON files contain the values cited by the thesis but do not themselves contain complete execution/input provenance or exact checkpoint-weight hashes. Snapshot inclusion repairs auditability of the arithmetic, not that provenance defect.

## Required compact steering artefact

The analysis repository should create a ledger-tracked machine-readable artefact from the two large steering arrays. It should contain, for every target-behaviour/operator cell at `alpha=1`:

- behaviour, operator, dose, representation/layer rule, and pairing key;
- generated, annotated, non-empty paired, missing, and empty counts;
- paired vanilla mean, arm mean, steered-minus-vanilla percentage-point change, and relative change;
- the exact estimand: mean over paired tasks of the target-labelled sentence fraction in the arm minus the paired vanilla value;
- generation-array, annotation-array, task-manifest, scorer/annotator, and relevant configuration hashes;
- producing script path and hash, source commit, dirty flag, software environment, and output hash;
- an explicit historical-lineage field that remains unresolved if it cannot be recovered.

The candidate audit demonstrates the expected values and schema, but must not simply be renamed into `results/`. The analysis-repository owner must generate/finalise the artefact under the normal result, `RESULTS_LEDGER.md`, and `METHODOLOGY.md` controls.

## Safe execution sequence

1. Wait for the current analysis work and pod/Phase-0 closure to finish. The analysis worktree is currently dirty and contains in-flight result changes, so refreshing now would create a mixed-time snapshot. The received handoff defines the release signal precisely: a commit on `codex/phase0-support` whose message begins `Phase-0 closure`.
2. In the analysis repository, finalise the compact steering artefact and provenance record without rewriting either large source array.
3. Add the compact steering artefact and the three small safety files to the analysis ledger and methodology, preserving all unresolved-provenance markers.
4. From a clean, explicitly recorded source state, update/run `thesis/_planning/review/refresh_evidence.sh` so the manifest records one source commit, dirty status, sizes, and sha256 values. Record the two large steering arrays in the excluded-by-size table if they are not copied.
5. Verify that the compact artefact alone reproduces the thesis raw table and that the copied safety files reproduce every cited sensitivity/decomposition number.
6. Only after those gates pass, replace the thesis sentence “presently unverifiable within this repository” with a narrower statement that the values are snapshot-recomputable but retain unresolved execution lineage. Do not remove the provenance caveat.

## Current disposition

- **Arithmetic:** reproduced from current source files.
- **Snapshot completeness:** unresolved pending the clean refresh above.
- **Historical execution provenance:** unresolved; not repairable by copying or hashing alone.
- **Thesis claims:** no promotion licensed at this stage.
- **Received handoff:** raw steering and safety artefact maps are now recorded, but the compact steering summary and ledger/methodology registrations remain queued behind the Phase-0 closure commit.
