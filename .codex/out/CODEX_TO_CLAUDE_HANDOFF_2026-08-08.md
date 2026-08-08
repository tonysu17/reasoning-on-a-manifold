# Codex → Claude handoff

**Date:** 8 August 2026  
**Scope:** the three requested friction-removal deliverables plus the reaffirmed steering-output boundary.

## 1. Phase-2 MDE primary pooling confirmation

Confirmed. The primary sentence-fraction inputs in `ph2_mde_sim.py` are arm-correct (`single_direction` against `energy_matched_random`), pool replicates within base task, and are exactly equivalent to `src.delta_floor.per_task_fraction` across all 49 paired tasks. Maximum arm/floor difference is zero and the paired delta is exactly `0.05378923565449617`. No `_rates`-style extraction enters the primary curves.

Qualification: the simulation reimplements rather than imports the authoritative extractor. Its per-1k candidate shares the pooling/floor rule but has no corresponding authoritative E8 equality check.

Full audit: `PH2_MDE_POOLING_CONFIRMATION_2026-08-08.md`  
SHA-256: `4ca2a8ccd7ca3cd919bc933b47191cab7c47b8286363d4e2610f81484418893b`

## 2. P5 shared Phase-2 vanilla contract

P5 requires and accepts the sealed E8 settings: base-R1 tokenizer/template alias, 8,192 new-token cap, greedy decoding (`do_sample=false`, semantic temperature zero), seed `20260808`, EOS/pad ID `151643`, no custom stop, and no stop at `</think>`. The canonical logical artefact is 300 rows: 100 tasks for each of R1, STAR1, and DeepScaleR.

P5 consumes that artefact read-only by artefact and row hash. It will never regenerate an overlapping Phase-2 vanilla row. Missing or incompatible source rows block the dependent analysis or remain provenance-unresolved. P5-owned safety/control generic additions are separate and must not overwrite or call themselves Phase-2 vanilla.

Full contract: `P5_PHASE2_SHARED_VANILLA_CONTRACT_2026-08-08.md`  
SHA-256: `391461ca66ef60015d4790a4226ffdb839edd9fdc2ba14fa8be2947acf1184a2`

## 3. Evidence-manifest and claim-field specification

The exact refresh-manifest, result-provenance, and eight-column claim-ledger fields are separated in `EVIDENCE_MANIFEST_AND_CLAIM_FIELD_SPEC_2026-08-08.md`.

Important compatibility finding: current `ph2_executor.py` uses `manifest_sha256` for the manifest's `ids_sha256`, not its file-byte hash. Preserve that compatibility value, while additionally emitting distinct `manifest_ids_sha256` and `manifest_file_sha256`. Likewise retain the current short `prereg_sha256` compatibility field and add `prereg_sha256_full`.

Full specification SHA-256: `e944454aca17006e382853f0bb343175bfef4ba81126c5fed8ed03d54c45027e`

## Boundary reaffirmation

Future Codex steering rescoring writes only beneath `.codex/out/` or the thesis repository. It will not write into `results/` or `results/eval/R1-1.5B__E1/` unless a later, explicit promotion proceeds through the analysis ledger and methodology controls.
