# SEALED SHEET — 7B matched-pair scale replication (Qwen2.5-Math-7B vs R1-Distill-Qwen-7B)

**Date:** 2026-08-17. **Status: SEALED** — authorized by Tony in-session ("seal the 7B pair").
Recorded before any 7B fitting or scoring. Descriptive, instrument-level, non-gating; extends the D3
ladder and the 1.5B base-control cell (`JSPACE_BASE_CONTROL_SHEET_2026-08-17.md`) by one matched pair
at 7B. Cannot reclassify Phase 1, any D-diagnostic, or the 1.5B pair.

## Question

At 1.5B, CoT distillation roughly halved late-layer silent bridge resolution (base union 0.407 →
distill 0.185), selectively (knowledge-type bridges lost, symbol/letter/calendar bridges retained),
with both checkpoints at exactly 0 in the mid-stack band and the base's signal in proto-output form.
Does this pattern replicate one scale up, on the architecture-matched pair whose general-instruct
sibling (Qwen2.5-7B-Instruct, hosted lens) is already measured?

## Fixed inputs

| Role | Identity | Pin |
|---|---|---|
| Base | `Qwen/Qwen2.5-Math-7B` | rev `b101308fe89651ea5ce025f25317fea6fc07e96e`, BF16, eager |
| Distill | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | rev `916b56a44061fd5cd7d6a8fb632557ed4f724f60`, BF16, eager |
| Architecture check | both: 28 layers, d_model 3584, head vocab 152,064 | verified from configs 2026-08-17; asserted again at load |
| Fit prompts | the 100 raw texts, `split` ∈ {fit_a, fit_b}, from `results/prereg/jspace_r1_fit_manifest.json` | sha256 `59c8695f…f443cb90defa6e`; re-encoded per model tokenizer |
| Fit recipe | released `jlens.fit`: one 100-prompt lens per model, sources 0–26 → target 27, `max_seq_len=128`, `skip_first=16`, `dim_batch=8` with a **registered OOM fallback ladder 8→4→2** (mathematically identical estimator; attempted value recorded) | mirrors 1.5B pair recipe |
| Scoring | `jspace_d2d3_readout.py --stage d3 --lens-path …` machinery, unmodified | instrument-level union pass@25 + layer profile + 1,000 label permutations, seed 20260817; per-model eligibility re-derivation |
| Bands (28-layer models, zero-based) | mid-stack = L10–L18; late = L19–L26 | fixed for E2 before any 7B number exists |
| Registered evaluator | `jspace_pair_report.py` (committed with this sheet) | computes E1–E5 verbatim from run bundles |
| Venue & envelope | RunPod, one 48 GB GPU (A6000/L40S class), **ephemeral pod with ≥100 GB container disk, no network volume**; hard stop 10 h / **$10** | artifacts (fp16 lenses, npz, reports) pulled back before pod stop |

## Registered expectations (directions, not thresholds; all descriptive)

- **E1 — replication of the reduction.** Distill-7B multihop union **<** Math-7B multihop union.
  Supports: the distillation reduction is scale-robust. Refuted (distill ≥ base): the 1.5B reduction is
  scale-specific; report as such.
- **E2 — location.** Both 7B cells' multihop signal confined to the late band (mid-stack max ≈ 0),
  as in every Qwen-family cell measured so far (including 7B-Instruct, mid-stack 0.000). A genuine
  mid-stack band in either cell would be a surprising positive, reported prominently as such.
- **E3 — format (the discriminating cell for the gradient).** Two live accounts for the J-minus-logit
  multihop delta at Math-7B: *scale account* predicts positive, near 7B-Instruct's +0.093; *math-
  narrowing account* predicts reduced/negative, like Math-1.5B's −0.056. Whichever obtains, the
  distill-7B delta is reported beside it. This cell separates parameter count from pretraining breadth
  in the format gradient.
- **E4 — world-model narrowing.** Math-7B association union at or below general 7B-Instruct's 0.061;
  nearer chance ⇒ math-continued-pretraining caps broad-knowledge abstraction even at 7B; ≈0.06 ⇒
  breadth survives math CPT and the 1.5B association floor is scale-dominated.
- **E5 — selectivity.** Among Math-7B's multihop hit-items, the distill's retained/gained items skew
  toward letter/symbol/calendar bridges and its lost items toward factual-knowledge bridges, as at
  1.5B. Item-level, qualitative, post-hoc by construction — no inferential weight.
- **Internal positive control.** Typo qualifies in both cells; a typo failure flags a fit problem and
  suspends interpretation of that cell pending diagnosis.

## Boundaries

Instrument-level readout, one checkpoint pair per scale, no training-seed replication, no causal
intervention, no workspace claim; cross-scale statements are qualitative. Thesis use: the replication
slot in `sec:safety-distill-substrate` and the appendix table row, only after results are committed
with hash-bound manifests. Licensed wording class as D3.
