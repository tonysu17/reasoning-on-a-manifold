# Phase-2 Amendment A2 — observational behavioural adjunct (vanilla arms)

**Status: SEALED 2026-08-08** — Tony in chat ("seal A2 and approve the $0.50 manifest
generation"); recorded in session memory at decision time; file edit applied post-restart the
same day. Precedes any Phase-2 generation (integrated plan Priority 2). Attaches to
`PHASE2_TRANSPORT_PREREG_2026-08-08.md` as a **separate observational secondary/exploratory
family**; it does not touch the causal primary, the battery, the gates, or the MDE rules.

## Purpose

Fill the thesis's on-policy behavioural gap for post-training at near-zero generation cost:
what do R1-1.5B, STAR1, and DeepScaleR *naturally generate differently* on the same tasks?
Uses only the three **vanilla arms** of the already-sealed battery (100 paired tasks × 3
models — no new generation).

## Endpoints (frozen; all six, no additions after outcome inspection)

Per task, per model:
1. **Prevalence of each of the four annotated behaviours** (backtracking,
   uncertainty-estimation, example-testing, adding-knowledge): fraction of annotated sentences
   carrying the label (`src/evaluation.py` sentence-fraction convention).
2. **Backtracking per 1,000 generated tokens.**
3. **Boxed exact-match accuracy.**
4. **Response length** (generated tokens).
5. **Repetition**: chain-level loop flag (rep4 > 0.8, E9 convention).
6. **Truncation** (hit max-token cap).

## Analysis rules (frozen)

- Unit = task; all contrasts **paired across checkpoints** on the shared manifest.
- **Estimation only**: paired differences (each target vs R1) with cluster-bootstrap 95% CIs
  (by task, B = 10,000, seed 20260808). **No significance verdicts, no causal language, no
  objective-level attribution** — STAR1 vs DeepScaleR is never an SFT-vs-RLVR comparison
  (data, recipe, dose, selection, and provenance all differ).
- Annotation: the same single Nova-Pro pass as the main battery (the house schema labels all
  behaviours in one pass, so the adjunct's incremental annotation cost is ≈ 0 beyond the
  already-budgeted envelope — this does not make annotation itself free).
- Missing/unparseable annotation rows are **unresolved** (never coerced to zero); pairwise
  deletion with per-endpoint missing counts reported.
- Multiplicity: none claimed (estimation family); if any reader-requested test is later run it
  requires a further dated amendment.
- Reporting: one table (2 contrasts × 6 endpoint groups) in the exploratory/secondary section;
  wording per the integrated plan's locality rules.

## What this adjunct cannot show

Harmful-request refusal, benign compliance, safety-recipe attribution, or anything causal —
those belong to the controlled behavioural benchmark (integrated plan Priority 5) and to the
Phase-2 causal family respectively.

## Seal line

Sealed by: Tony (in chat: "seal A2 and approve the $0.50 manifest generation") · Date: 2026-08-08.
