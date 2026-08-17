# SEALED SHEET — base-model control cell (Qwen2.5-Math-1.5B J-lens readout)

**Date:** 2026-08-17. **Status: SEALED** — authorized by Tony in-session ("go with option D plus the
base-control run"). Recorded before any fitting or scoring of this cell. Descriptive, non-gating,
cannot reclassify Phase 1 or any D1–D5 verdict. Extends the D3 ladder of
`JSPACE_R1_DIAGNOSTIC_PROTOCOL_2026-08-16.md` (seal `752dee7`) by one cell; all D3 scoring machinery
and its sealed estimands apply unchanged.

## Question

D3 showed non-distilled ~1–2B models (qwen3-1.7b 3/3, gemma-3-1b 2/3) qualifying at instrument level
where R1-Distill-1.5B failed. The missing discriminating cell is R1-1.5B's **own base**: does the
deficit predate distillation (math-narrow base) or did distillation suppress a readout the base had?

## Fixed inputs

| Role | Identity | Pin |
|---|---|---|
| Model | `Qwen/Qwen2.5-Math-1.5B` (R1-Distill-1.5B's base) | rev `4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2` (local snapshot), BF16, eager |
| Fit prompts | the 100 raw texts with `split` ∈ {fit_a, fit_b} from `results/prereg/jspace_r1_fit_manifest.json` | manifest sha256 `59c8695f4d8bc37e8a6cfb592bba09fc72afaec7f5eca6b415f443cb90defa6e`; texts re-encoded by the Math tokenizer (token IDs may differ from the R1 manifest; raw bytes identical) |
| Fit recipe | released `jlens.fit`: one 100-prompt lens, all source layers 0–26, target 27, `dim_batch=8`, `max_seq_len=128`, `skip_first=16` | mirrors Phase-1 estimator settings; single merged fit (no A/B split — descriptive cell, no stability gate) |
| Scoring | `jspace_d2d3_readout.py --stage d3 --lens-path …` (commit `2220eeb` machinery) | instrument-level any-layer union pass@25 + layer profile + 1,000 label permutations, seed 20260817; per-model tokenizer eligibility re-derivation as D2/D3 |
| Venue | local M4 (MPS), $0; resume-safe checkpointing (`checkpoint_every=1`) | overnight run |

## Registered interpretation matrix (descriptive)

- **Base qualifies on multihop** → the base had silent-composition readout; the R1 distill lost it ⇒
  the deficit is **distillation-recipe-specific** (sharpens D1's externalization account).
- **Base fails multihop** → the deficit **predates distillation** (math-narrow pretraining);
  distillation-externalization remains supported behaviourally (D1) but the readout absence cannot be
  attributed to it.
- Association: expected to fail (math-narrow base; gemma-3-1b general base already at 0.0000);
  qualification would be surprising and reported as such.
- Typo: expected to qualify (all cells so far do); a failure would flag a fit problem, not a model fact.

## Boundaries

Same licensed wording class as D3. Cross-tokenizer caveats as sealed. No thesis claim licensed;
feeds only the decision memo's recipe-contrast paragraph and the ledger entry.
