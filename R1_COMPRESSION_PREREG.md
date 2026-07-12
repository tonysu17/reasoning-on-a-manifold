# R1 — Strata-differential compression across post-training (creativity–entropy rung 1 + RL.a) — SEALED PRE-REGISTRATION

**Sealed:** 2026-07-12, before any extraction or generation. Programme doc:
`../creativity_entropy_extension.md` §6 (R1) + §6b (RL.a). Runner: `30_r1_compression.py`.
Output: `results/r1_compression/`. Gate R0 passed 2026-07-12 (ladder dissociates — levels are
named per claim below).

## Arms (the post-training ladder at 1.5B, all public)

| arm | model | post-training delta | comparison tier |
|---|---|---|---|
| `r1` (reference) | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | SFT-distill from RLVR teacher | — |
| `deepscaler` | agentica-org/DeepScaleR-1.5B-Preview | **GRPO-RLVR applied on top of `r1`** (verified 2026-07-12: Berkeley/Agentica, AIME 43.1 vs 28.8) | **matched-ids** (tokenizer gate: vocab + corpus ids byte-identical to r1) |
| `star1` | UCSC-VLAA/STAR1-R1-Distill-1.5B | safety SFT on top of `r1` | matched-ids (gate: identical) |
| `qwenmath` | Qwen/Qwen2.5-Math-1.5B | NONE — the base `r1` was distilled from | **matched-text only** (gate FAILED: DeepSeek modified the tokenizer; ids differ even on chain bodies) |

**Primary contrast = `r1` ↔ `deepscaler`**: same lineage, same tokenizer, RLVR is the only delta.
**Secondary = `r1` ↔ `star1`** (SFT-control; spillover work predicts ≈no change).
**Descriptive anchor = `qwenmath`** (base; matched-text tier + prompt-format difference ⇒ its rows
support weaker claims, restricted to the MATH category for anything quantitative).

## Design

- **R1-rep (matched-text/ids, teacher-forced):** the R0 200-chain sample (sample.json reused
  verbatim: 100 loop / 100 clean), each arm teacher-forced on the SAME full_texts. Measures:
  E-1 windowed entropy + E-2 windowed PR/uniformity @ L17 (E9.0 grid). `r1`'s values already exist
  (R0 ent shards + E9.0 state shards) and are reused, not recomputed. For matched-ids arms the
  window grids coincide with r1's ⇒ paired per-chain deltas; for `qwenmath` the grid is its own
  tokenization of the same text ⇒ per-chain summary comparison only.
- **R1-beh (own generation):** the 50 E9.1 eval tasks per arm, greedy + 3 samples @ T=0.6,
  max_new_tokens 8192, batched with per-(batch,seed) contract. Deepseek-family arms use
  `format_prompt` (identical template); `qwenmath` uses its own chat template with step-by-step
  instruction (declared CF-N). Measures: E-3 per-task diversity (1 − mean pairwise 4-gram Jaccard,
  T06), length, backtracking-cue rate per 1k tokens (annotation-free regex proxy, documented in
  the runner), boxed rate, collapse rate (4-gram repetition > 0.8), per-category accuracy proxy.
- **R1-score (own-generation re-scoring):** each arm's generations re-scored by the generating
  model for E-1 + PR@17 (the on-policy complement to R1-rep's off-policy caveat).

## Sealed predictions

- **P-R1.1 (the RLVR compression claim, representational):** on matched text, `deepscaler` shows
  LOWER chain-level PR@17 than `r1` (paired per-chain Wilcoxon p<.05, median Δ<0). The in-house
  test of "RLVR is compression-seeking" (li2025tracing) at the representation level.
- **P-R1.2 (base-vs-distill race, sealed as two readings):** *SFT-entropy-seeking* (li2025tracing:
  distillation raises rank ⇒ r1 ≥ qwenmath) vs *inherited-compression* (distill mimics compressed
  teacher ⇒ r1 < qwenmath). **We predict SFT-entropy-seeking.** Matched-text tier: direction only,
  no effect-size claim.
- **P-R1.3 (solution-space, behavioural):** T06 strategy-agnostic diversity declines monotonically
  along the ladder `qwenmath > r1 > deepscaler` (MATH category primary for the base arm; pairwise
  Wilcoxon, Holm). The core in-house Yue-style result if it holds.
- **P-R1.4 (own-gen decoding entropy):** mean on-policy E-1 orders `qwenmath > r1 > deepscaler`.
- **P-R1.5 (strata-differential, H1b/H-RL cross-section):** control-stratum stats move OPPOSITE to
  object-stratum: backtracking-cue rate and length `r1 ≫ qwenmath` (distillation installs
  deliberation) while `deepscaler < r1` on length (efficiency phase / CF-L schedule) with diversity
  NOT recovering (`deepscaler ≤ r1`). Rise-then-fall across the ladder.
- **P-R1.6 (safety-SFT negative control):** `star1` ≈ `r1` on ALL ladder metrics (paired |Δ| small,
  CIs overlapping zero) — consistent with the spillover translation-not-rotation null.

## Kill criteria

- `deepscaler ≥ r1` on BOTH diversity and rank ⇒ the "RLVR spends entropy" premise fails in-house
  at this scale — plank (i) of the programme must be reworked (this is a publishable negative and
  the primary reason to run R1 before anything expensive).
- `qwenmath` produces degenerate/empty output on >50% of MATH tasks ⇒ base arm uninterpretable;
  drop to a footnote, do not substitute claims.

## Confounds (extend §9/§6b registers)

- **CF-M (off-policy re-scoring):** teacher-forcing non-generator arms on r1's chains measures
  representation of off-distribution text; R1-score is the on-policy cross-check; claims cite both.
- **CF-N (prompt-format, base arm):** qwenmath cannot receive the deepseek template; its arm
  carries a prompt-format difference by necessity. Matched-ids arms unaffected.
- **CF-L (DeepScaleR schedule):** its context-length curriculum compresses length BY DESIGN —
  length findings for deepscaler are schedule-driven; diversity/rank findings are the cleaner read.
- **CF-O (capability matching):** compression claims for deepscaler-vs-r1 are conditioned on
  matched-or-better task performance (its AIME gain + our boxed/accuracy proxy); if deepscaler
  underperforms r1 on our tasks, P-R1.1/P-R1.3 verdicts carry that caveat explicitly.
- Intermediate DeepScaleR checkpoints NOT released (only the final Preview) ⇒ R1 is an endpoint
  contrast; trajectory claims wait for RL.b (own GRPO run).

## Guards

Sample and tasks fixed before running (R0 sample.json; E9.1 eval_task_ids.json). All bars fixed
here; seeds fixed (generation seeds 0/1/2 for T06 samples, batch-seeding contract as E9.1).
Fail-soft with exclusions counted; all-NaN guard + `--fp32` retry inherited from R0. M3 wording
discipline: all R1-rep results are OCCUPANCY statements; M5: no test-set-tuned knobs.
