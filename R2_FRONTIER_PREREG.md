# R2 — The entropy–value frontier (creativity–entropy rung 2) — SEALED PRE-REGISTRATION

**Sealed:** 2026-07-12, before the temperature-sweep generation. Programme doc:
`../creativity_entropy_extension.md` §6 (R2). Runner: `31_r2_frontier.py`. Output:
`results/r2_frontier/`. Gates: R0 passed (ladder dissociates); R1 running (post-training contrast).

## The question

Trace **value vs solution-diversity frontiers** for R1-Distill-1.5B under two entropy knobs and ask
which knob buys diversity more cheaply — i.e. reaches a given solution-diversity at less loss of
task completion:
- **Knob T (thermostat / exogenous decoder entropy):** vanilla sampling, temperature swept.
- **Knob A (pump / endogenous directed entropy):** backtracking `single_direction` steering,
  amplification α swept, at fixed T=0.6 (so diversity is measurable).

## Value axis (decision + caveat)

The 50 eval tasks have **no gold answers** (`data/tasks_final.json`: id/prompt/difficulty/category
only); `correctness_R1-1.5B_pilot.json` covers just 7/50. So R2's value axis is the
**annotation-free `boxed` completion rate** (fraction of chains containing `\boxed`) — the same
signal as E9.1b's overthinking-tax result. **Caveat carried in every table:** this measures
*completion / reaching a final answer*, NOT correctness; a confident wrong box counts as completion.
Collapse rate (4-gram repetition > 0.8) is reported as the inverse-completion companion. A true
correctness axis (LLM-judge on a subset, or gold-answer MATH subset) is a **declared future
robustness pass**, budget-gated, out of R2 scope.

## Metrics (per cell = (knob, level))

- **Diversity (E-3):** 1 − mean pairwise 4-gram Jaccard across the 3 T>0 samples per task, averaged
  over tasks (`r0._ngrams` definition, identical to E9.1/R1). Only T>0 cells have diversity.
- **Value:** `boxed` rate (mean over all rows in the cell).
- **Collapse:** mean(`repetition_rate` > 0.8) (`src.evaluation.repetition_rate`).
- **Length:** mean `n_tokens` (context for the overthinking read).
- **Decoding entropy (E-1):** mean per-row where available (R2-sweep rows can be scored; E9.1 rows
  are not re-scored here — declared).

## Data inventory (what is REUSED vs NEW)

**Knob A (pump) — FULLY REUSED, no new generation:** E9.1 T06 (`results/eval/R1-1.5B__E9_1_T06/`)
supplies `backtracking/single_direction` at α∈{0.5,1.0,1.5} @ T=0.6 (150 rows = 50 tasks × 3
samples each) + `shared/vanilla` α=0 @ T=0.6 (150). ⇒ a 4-point frontier (α=0,0.5,1.0,1.5) with
diversity + boxed already on disk.
**Knob T (thermostat) — PARTIALLY REUSED:** `shared/vanilla` exists at T=0 (greedy, boxed only, no
diversity) and T=0.6 (150, diversity + boxed). **NEW generation = vanilla r1 at T∈{0.3, 0.9, 1.2}**
× 3 samples × 50 tasks = 450 chains (~1 GPU-h). Fills the temperature frontier so it spans a
diversity range comparable to knob A.
**Controls (reused, optional points):** state-noise (E9.2, the pre-refuted geometry-blind knob) and
example-testing steering arms are available as extra frontier points but are NOT part of the
primary P-R2 comparison.

## Sealed predictions

- **P-R2.1 (pump beats thermostat — the headline):** at **matched solution-diversity**, knob A
  (bt-amplification) retains a HIGHER boxed rate than knob T (temperature). Test: interpolate each
  knob's boxed-vs-diversity curve onto a common diversity grid over the overlapping diversity range;
  compare boxed at matched diversity (paired over grid points, sign test + mean gap with bootstrap
  CI). The directed endogenous knob is the more value-efficient source of diversity.
- **P-R2.2 (both knobs trade value for diversity):** boxed rate declines monotonically as diversity
  rises along BOTH knobs (Spearman ρ<0 each). Establishes there IS a frontier (no free diversity).
- **P-R2.3 (collapse asymmetry):** knob A reaches high diversity partly via loop-collapse (collapse
  rate rises with α — E9.1b), whereas knob T raises diversity with less collapse. I.e. the pump's
  "diversity" is partly degenerate. Reported honestly even though it complicates P-R2.1: the
  clean read is boxed-at-matched-diversity EXCLUDING collapsed chains (a sealed secondary endpoint).

## Kill / null criteria

- If knob T dominates knob A at matched diversity (temperature retains more boxed), **P-R2.1 is
  refuted** — the deliberation loop is *only* an overthinking liability and creative diversity is
  cheapest at the decoder. Equally publishable; it would refocus the programme on decoder-entropy
  as the creativity handle and demote the pump.
- If the two knobs' diversity ranges do not overlap (no matched-diversity region), the comparison is
  undefined ⇒ widen the temperature or α grid before claiming anything.

## Confounds

- **CF-P (completion≠correctness):** the value axis is boxed completion; carried in every table.
- **CF-Q (collapse inflates diversity):** loop-collapsed chains are maximally "diverse" by 4-gram
  Jaccard vs each other only if their loops differ; within a task the 3 samples' loops are often
  similar ⇒ collapse can DEPRESS diversity too. Net effect measured, not assumed; P-R2.3 +
  collapse-excluded endpoint address it.
- **CF-R (steering-vector provenance):** knob A uses the within-annotator bt vector (E1-pooled,
  builder=Sonnet) — inherits the steering chapter's within-annotator hedge; the frontier *shape*
  claim is annotation-free but the "bt direction" label is not.
- **CF-S (min-p deferred):** the third decoder knob (min-p) is declared and not run; P-R2 is a
  two-knob comparison only.

## Guards

Value axis + all bars fixed here pre-sweep. Diversity/boxed/collapse definitions inherited verbatim
from E9.1/R1 (no re-definition). New generation reuses fixed eval_task_ids.json + the E9.1
batch-seeding contract (seeds 0/1/2). Analyse runs on existing data first (preliminary pump frontier
before the thermostat sweep) — the sweep only ADDS knob-T points, it does not change any sealed bar.

---

## PREP FINDING + REFRAME (2026-07-12 — before any sweep; the sweep is CANCELLED)

Running `analyse` on existing E9.1 T06 data (no pod) invalidated the frontier's x-axis and **saved
the temperature-sweep spend**:

1. **4-gram diversity (E-3 as defined) is SATURATED.** Per-task distribution at T=0.6: min 0.87,
   median 0.966, max 0.99, **0% of tasks < 0.9**. Long sampled reasoning chains always differ at the
   surface 4-gram level, so 1−Jaccard ≈ 0.96 regardless of knob. It measures token variation, not
   solution diversity, and has no dynamic range. The pump knob confirms this: diversity is flat
   (0.961–0.963) across α∈{0,0.5,1,1.5} — bt-amplification does not move surface diversity at all.
2. **Answer-level diversity has range but is SPARSE + confounded.** Distinct-`\boxed` / n-answered
   rises with α, but only **30–46%** of the 50 general-reasoning tasks produce ANY boxed answer, so
   the metric rests on 6–14 tasks/cell and conflates "produces an answer" (completion) with
   "produces a different answer" (diversity).
3. ⇒ **The value-vs-diversity FRONTIER is not measurable on the 50 general-reasoning eval tasks.**
   No usable diversity x-axis exists on this substrate. **The temperature sweep is cancelled** — it
   would trace a saturated axis. **P-R2.1 (pump-vs-thermostat) is UNTESTABLE here** and moves to R3.

**What existing data DOES support cleanly (no pod, the salvaged R2 result):** backtracking
amplification at T=0.6 is a monotone **completion/efficiency gain at constant surface-diversity and
no collapse cost** — boxed 0.16→0.24→0.29→0.29, answered-frac 0.30→0.46, mean length 4905→3867 tok,
collapse ~flat 0.06→0.04, **plateauing at α≈1.0**. This corroborates + extends E9.1b's overthinking
tax at T=0.6: mild directed bt-amplification is a near-free completion win, not a diversity source.

**Reframe (adopted):** R2's *frontier* question requires **R3's dedicated multi-solution,
answer-checkable task family** — so R2 folds into R3 rather than running its own sweep. R2's
standalone deliverable is the completion/efficiency result above (existing data) + this measurement
negative (surface-diversity saturation), which is itself a load-bearing methodological finding: it
is the empirical justification for R3's strategy-level (not token-level) diversity metric. The
`31_r2_frontier.py generate` stage is retained but **shelved** (only runs if a future task set makes
a temperature-diversity frontier meaningful).
