# R3 — Strategy-entropy pilot gate (creativity–entropy rung 3, prep) — SEALED PRE-REGISTRATION

**Sealed:** 2026-07-12, before any pilot generation. Programme doc:
`../creativity_entropy_extension.md` §6 (R3), which R2 now folds into (see
`R2_FRONTIER_PREREG.md` addendum — the empirical case for a STRATEGY-level metric).
Runner: `32_r3_strategy.py`. Output: `results/r3_strategy/`. Pilot compute: local MPS, $0.

## Why a pilot gate

R2 prep proved the eval-50 general-reasoning set is not a solution-diversity substrate
(4-gram diversity saturated; `\boxed` sparse). R3's centrepiece claims (strategy entropy as the
real E-3; state-space excursions at strategy switches; the value×diversity plane) all presuppose
three things nobody has verified at 1.5B: that the model **answers** a dedicated multi-solution
task family, that its answers are **checkable**, and that its sampled solutions actually **vary in
strategy**. The pilot buys those three facts for $0 before any pod or annotation spend — the same
prep discipline that just cancelled R2's sweep.

## Task family (built, not scraped — gold answers computable by construction)

`stage tasks` generates a parametric family, 8 templates × 2 instances = **16 pilot tasks**, every
answer a computable INTEGER, every template with a **declared strategy space** (the classifier only
chooses within it):

| id | template | gold | declared strategy space |
|---|---|---|---|
| T1 | domino tilings of 2×n | Fib(n+1) | recursion · pattern/known-sequence · casework |
| T2 | Σ k(k+1), k=1..n | n(n+1)(n+2)/3 | formula-split · telescoping · induction/pattern |
| T3 | lattice paths m×n grid | C(m+n, m) | binomial/formula · recursion/DP · casework |
| T4 | 2×2 integer linear system | x·y at solution | substitution · elimination · matrix/Cramer |
| T5 | Σ k², k=1..n | n(n+1)(2n+1)/6 | formula · induction · pattern |
| T6 | last digit of a^b | cycle lookup | pattern/cycle · modular arithmetic |
| T7 | #sequences of n die rolls with ≥1 six | 6^n − 5^n | complement · direct casework |
| T8 | roots r,s of x²−px+q: r²+s² | p²−2q | Vieta/identity · explicit-roots |

Prompt = the DeepSeek family template (`format_prompt`) with "...reason step by step, and put your
final answer within \boxed{}." Answer scoring = last brace-balanced `\boxed{...}`, normalised
(strip space/commas/trailing dot), exact match to the integer gold. **Value axis is therefore TRUE
CORRECTNESS here** — the eval-50's completion-only caveat (CF-P) does not apply to R3.

## Strategy classifier (annotation-free first pass; circularity-safe)

Lexical keyword families per strategy (defined in the runner, fixed pre-pilot); a chain's
**primary strategy** = the family with most hits *within the template's declared space*
(ties → declared order; zero hits → `unclassified`). This is deliberately crude and cheap:
it cannot inherit LLM-annotator circularity (CF-B), and the upgrade path — LLM-judge labels with
the R2.2 multi-annotator κ protocol — is DECLARED for the full R3, gated on this pilot showing
there is strategy variation to label at all.
**Instrument validation (allowed pre-pilot, like R2's analyse-on-existing):** the classifier is
smoke-tested on existing eval-50 chains only for firing-rate sanity (not zero, not degenerate);
no gate reads those numbers.

## Pilot design

16 tasks × 3 samples @ T=0.6 (seeds 0/1/2, E9.1 batch-seeding contract) = **48 chains**,
`max_new_tokens` 3072 (easy tasks; cap-hit fraction recorded as its own diagnostic), batch 4,
local MPS (bf16), fail-soft, checkpointed. No steering arms in the pilot (pump arms enter full R3).

## Sealed gates (all four must pass for full R3 as designed)

- **G-R3.1 (answerable):** ≥ 70% of chains produce a parseable `\boxed` integer.
  *(eval-50 managed 30–46% — this family must be qualitatively better or the substrate is wrong.)*
- **G-R3.2 (value axis has headroom):** overall accuracy in **[20%, 95%]** — the model can do some
  of it, and it isn't saturated (a saturated value axis cannot trade against diversity).
- **G-R3.3 (classifier coverage):** ≥ 60% of answer-producing chains get a primary strategy label.
- **G-R3.4 (strategy variation exists — THE gate):** ≥ 25% of tasks show ≥ 2 distinct primary
  strategies across their 3 samples. This is the R2-saturation lesson applied preemptively: if
  sampled solutions at 1.5B don't vary in strategy, strategy entropy has no range and the full R3
  must be redesigned (options, declared now: prompted-strategy conditioning arm; higher T;
  larger/open model; per-template instance diversification) rather than run blind.

## What the full R3 adds if the gates pass (not run in the pilot)

Scale to ~50–80 tasks × more samples on a pod; the value×strategy-diversity plane (the reframed
P-R2.1 pump-vs-thermostat comparison lives HERE, with a working diversity axis); LLM-judge strategy
labels under the multi-annotator protocol vs the lexical proxy; R3(ii) excursion signature at
within-chain strategy switches using the R0 E-2 instruments (windowed PR/uniformity on the E9.0
grid) with matched-position controls.

## Confounds (register)

- **CF-T (lexical classifier validity):** keyword hits ≠ strategy use (a chain can *mention*
  recursion while doing casework). Pilot treats the classifier as a RANGE-FINDER only; no
  scientific claim about which strategy is used rests on it. Full R3 upgrades to judged labels.
- **CF-U (memorised answers):** these templates are classic; the model may retrieve answers with
  degenerate reasoning. Accuracy-without-consistent-method is partially visible via
  strategy-vs-correctness cross-tabs; carried as a caveat, not resolved at pilot scale.
- **CF-V (difficulty ceiling):** instances are sized for a 1.5B model (small parameters); the
  family under-represents the hard end where strategy choice matters most. Declared; full R3 can
  stratify difficulty.

## Guards

Gates, bars, task parameters, seeds, and classifier keyword lists fixed before generation. Task
file written to `data/r3_tasks_pilot.json` and committed with the code. Fail-soft with exclusions
counted. M3 wording discipline inherited; no test-set tuning (the pilot IS the test of the
instrument, not of the science).

---

## Amendment 1 (2026-07-12, after batch 1 of 12; instrument fix, gates UNCHANGED)

First batch (4 chains): ALL hit the 3072 cap; 3/4 no `\boxed`; rep4 = 0.08–0.14 (no looping); two
chains had already stated the CORRECT gold answer mid-chain ("the number of ways is 233",
"gives me 1360") and were still re-verifying / trying alternative methods when truncated — the
programme's own overthinking tax biting the instrument. This is a pure truncation artifact:
`max_new_tokens` 3072 under-budgeted R1-Distill's verbosity even on easy tasks (corpus mean ~4.9k).
**Fix: `max_new_tokens` 3072 → 6144; the 4 capped rows deleted so all 48 chains regenerate under
the uniform new cap; rows now record their `max_new` so the gate's cap-hit diagnostic is
self-describing. All four gate bars unchanged** — this amendment is made before any gate statistic
was computed. Precedent: E10 Amendment 1 (instrument upgrade pre-analysis).

---

## Amendment 2 (2026-07-13, FULL-run design; registered before any full-run generation)

The pilot passed all four gates (2026-07-13, `results/r3_strategy/PILOT_GATE.md`), but its verdict
(RESULTS_LEDGER §B5) imposes three full-run requirements this document's §"What the full R3 adds"
did not spell out. They are registered here, pre-generation, together with the minimal choices the
prereg left silent (full list in the `32_r3_strategy.py` docstring):

1. **Difficulty stratified upward (CF-V):** pilot accuracy 0.896 was near ceiling — every parsed
   answer was correct; the only headroom came from T1/T2 truncation. Full task set = same 8
   templates × 8 instances (**4 easy + 4 hard**, hard = larger parameters), 64 tasks total
   (inside the "~50–80"), parameters **disjoint from the 16 pilot instances**, golds still
   computable by construction, declared strategy spaces and the sealed lexical classifier
   **unchanged** → `data/r3_tasks_full.json`.
2. **k ≥ 5 samples:** G-R3.4 passed exactly at the 0.25 bar at k=3 — too thin for an entropy
   axis. Full run: **k=8** (seeds 0–7), per-(batch, seed) E9.1 contract.
3. **Temperature arm:** strategy entropy must be readable against decoding entropy. Cells =
   the R2 prereg's shelved P-R2.1 sweep transplanted verbatim (R2 folded into R3): thermostat
   vanilla **T ∈ {0.3, 0.6, 0.9, 1.2}**; pump **bt `single_direction` (E1-pooled, L17,
   mode=subtract — the sign E9.1 T06 actually ran) α ∈ {0.5, 1.0, 1.5} @ T=0.6**; vanilla T=0.6
   doubles as the pump's α=0 anchor. 7 cells × 64 × 8 = 3,584 chains, `max_new` 6144 (Amendment
   1's cap kept — requirement 4).

Analysis (sealed with this amendment): strategy entropy = Shannon entropy (bits) of
primary-strategy labels over a task's k chains (unclassified excluded; <2 labelled → undefined;
unclassified-included sensitivity reported); value = TRUE correctness; the value×strategy-entropy
plane tests **P-R2.1/P-R2.2/P-R2.3 (reframed)** with the R2 prereg's own matched-level
interpolation + bootstrap machinery; difficulty strata (CF-V) and strategy×correctness cross-tabs
(CF-U) reported. LLM-judge labels (R2.2 κ protocol) and R3(ii) excursion signatures remain
declared follow-ons consuming `full_gen.json`, not part of this run. Runner stages:
`full-tasks` / `full-generate` / `full-analyse`; pod runner `pod_r3_full.sh`.
