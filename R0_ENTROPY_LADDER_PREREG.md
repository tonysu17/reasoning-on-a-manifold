# R0 — Entropy-ladder bookkeeping (creativity–entropy programme, rung 0) — SEALED PRE-REGISTRATION

**Sealed:** 2026-07-11, before any entropy computation. Programme doc:
`../creativity_entropy_extension.md` §6 (parent dir). Runner: `29_r0_entropy_ladder.py`.
Output: `results/r0_entropy_ladder/R1-1.5B/`. Compute: local MPS only, $0.

## Purpose

The creativity–entropy programme rests on a four-level entropy ladder (E-1 decoding, E-2 state
occupancy, E-3 solution-space, E-4 semantic). R0 measures the **cross-level relations** on corpora
already on disk, plus the **loop entropy profile** that the pump-and-jam picture predicts. E-4 is
**deferred by declaration** (no domain annotation exists on disk; it is R4's job). Everything below
re-uses E9.0/E9.1 artefacts and instruments unchanged; the only new computation is token-level
predictive entropy (E-1) via teacher-forced re-scoring.

## Data (fixed before running)

- **Base corpus:** the 884 E9.0 loop-geometry shards (`results/loop_geometry/R1-1.5B/shards/`).
  Stratified subsample, rng seed 0: **100 loop chains** (is_loop=1 ∧ onset_tok_gen>0) + **100 clean
  chains**. E-2 (windowed PR + uniformity at L15/16/17, window 128 / stride 64) comes from the
  shards as-is; E-1 is computed fresh with the *identical* tokenization (add_special_tokens=False,
  truncation 10240) and window grid — **center-grid equality with the shard is asserted per chain;
  mismatches are excluded and reported**, not silently kept.
- **T06 corpus (E-3):** E9.1 T=0.6 **vanilla** arm (`results/eval/R1-1.5B__E9_1_T06/`), 50 eval
  tasks × 3 samples. Prompts reconstructed from the corpus chains file via `base_task_id`
  (formatted prompt stored there verbatim). Per-task diversity = 1 − mean pairwise 4-gram Jaccard
  across the 3 samples (`e9_1_analysis._ngrams` definition, unchanged). Steered arms are OUT of
  R0's primary scope (declared; vanilla only).

## Measures

- **E-1:** next-token predictive entropy (nats) of the model's own distribution at each generated
  position, teacher-forced over the stored text (R1-Distill-Qwen-1.5B, fp16, `use_cache=False`,
  entropy in float32, chunked LM head). Caveat stated up front: for T>0 samples this re-scores the
  sampled path — it measures the model's local uncertainty along that path, not the sampler's
  temperature. Base corpus is greedy, so there re-scoring = generation-time distribution exactly.
- **E-2:** windowed participation ratio (`pr_17` primary; 15/16 secondary) and token uniformity
  (`unif_17`) from the E9.0 shards; for T06 chains, freshly extracted L17 windows (same code path,
  `ActivationCache` + `windowed_state_metrics`).
- **E-3:** per-task cross-sample diversity as above.

## Analyses and sealed predictions

**R0.a — Chain-level cross-level correlations (the ladder-dissociation gate).**
Per-chain E-1 summary = mean windowed entropy; E-2 summaries = mean windowed PR / uniformity over
the **same windows**. Primary regions (sealed to avoid the loop tail driving a spurious pooled
correlation): clean chains → all generated-region windows; loop chains → pre-onset windows only
(centers < onset). Statistic: Spearman ρ with p-values; pooled and per-class reported; chain length
(n_gen_tokens) correlations reported alongside as the visible nuisance.

- **P-R0.1 (dissociation, the programme gate):** no {E-1, E-2} pair reaches |ρ| ≥ 0.9 in the
  primary regions. **Gate:** if ALL pairwise |ρ| ≥ 0.9 (levels redundant), the ladder collapses to
  one number and the programme simplifies accordingly — documented either way.

**R0.b — Loop entropy profile (pump-and-jam).**
Aligned curves (bins of window-center − onset, −2048…+1024 step 128) for E-1, PR, uniformity over
the 100 loop chains; E9.0b's matched-relative-position clean control is MANDATORY for all tests
(the control that killed naive PR-precedence).

- **P-R0.2 (jam = predictable):** in-loop windowed E-1 (centers ≥ onset) is LOWER than
  matched-relative-position windows in clean chains (per-chain means, one-sided Mann-Whitney,
  α=.01). Expected large — a verbatim loop should be near-zero-entropy text. This doubles as the
  manipulation check for the E-1 instrument.
- **P-R0.3 (state-first vs thermostat-failure, the informative one).** Two rival readings sealed:
  *thermostat-failure* = loop chains show an extra pre-onset E-1 decline beyond the clean
  positional trend (entropy starvation precedes the jam); *state-first* = no loop-specific
  pre-onset E-1 differential (the fork is a state event; text-level entropy moves only at/after
  onset — consistent with E9.0b uniformity-only precedence and E9.3 fork entropy-depletion).
  Test: Δ(pre-onset-512 mean − early-baseline [128,640) mean) for loop vs matched clean, two-sided
  Mann-Whitney, α=.05. **We predict state-first (null differential).** A significant loop-specific
  decline would instead SUPPORT thermostat-failure and strengthen the "entropy starvation causes
  collapse" reading — either outcome feeds the programme; the prediction is which one nature picks.

**R0.c — Linkage to E-3 (n=50 tasks, vanilla T06).**
Per-task mean E-1 (3 samples, whole generated region) and mean windowed PR@17 vs per-task
diversity.

- **P-R0.4:** E-1 ↔ diversity Spearman ρ > 0, p < .05, with magnitude in the mid-range
  (0.2 ≤ ρ < 0.9). ρ ≥ 0.9 would collapse E-1 and E-3 into one level; ρ ≈ 0 would dissociate them
  entirely (both outcomes update the ladder, the mid-range is what "distinct but coupled" predicts).
- **P-R0.5 (exploratory, no pass/fail bar):** PR@17 ↔ diversity reported. E9.2's dose-not-geometry
  result motivates the expectation that the E-2 → E-3 link is weaker than the E-1 → E-3 link.

## Guards

Matched-position control mandatory (E9.0b lesson). No threshold is tuned after seeing data — all
bars fixed here. Seeds fixed (subsample seed 0). Truncation and tokenization identical to E9.0
extract. Fail-soft per chain with exclusions counted in the report. E-1 layer-free by construction;
no per-layer selection is available to tune. The T06 re-scoring caveat is stated in every table
that uses it.
