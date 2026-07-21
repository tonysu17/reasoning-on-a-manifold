# H1 result — gpt-oss-20b deliberative-safety object (run 2026-07-21, sealed prereg)

**VERDICT: H1 SUPPORTED — a separable safety-reasoning object exists in
gpt-oss-20b's residual stream and survives the capability control. The
"low-dimensional" qualifier is STRUCK per the sealed leg-(a) rule.**

All numbers at the primary layer (L11 = fraction 0.50 of 24; sealed) on the
primary topic-controlled contrast (benign-arm DSR vs benign-arm generic;
n = 52 vs 3,241 rows) unless stated. Instruments: F2 engine (chain-grouped
held-out d, row-permutation null, stimulus bootstrap), seed 0.

## Leg (b) separation — PASS
- Chain-held-out Cohen's d **5.02** (5/5 folds positive: 3.40–7.48), AUROC **0.979**
- Permutation null p = **0.001** (floor at n=1000 — the N≪d guard)
- Bootstrap CI [4.36, 5.59]
- Sensitivity layers agree: d 3.69 (f25) / 5.07 (f75), both p = .001
- Per-label (benign, held-out): spec_citation **7.14**, decision **5.74**,
  harm_recognition **4.59** — each citable label separates on its own
- Secondary all-arms d 7.17 (p .001) — larger, as expected with the arm/topic
  confound in it; reported, not verdict-bearing

## Leg (c) capability control — PASS
- |cos(safety axis, capability axis)| = **0.192** (threshold ≤ 0.50)
- Retention after projecting capability out: **0.995** (threshold ≥ 0.50);
  d 3.53 → 3.51
- The object is essentially orthogonal to the hard-vs-easy capability axis
  (fit on 590 vs 450 generic rows of the graded capability chains). The
  Ponkshe collapse-into-capability account does not explain this separation.

## Leg (a) low dimension — qualifier STRUCK
- TwoNN ID(DSR) = **3.5** [3.0, 5.5]; matched-N generic null mean 4.7,
  5th pct = 2.8 → 3.5 is not below the null's 5th percentile
- PCA thresholds (post-seal descriptive addition, Tony request 2026-07-21;
  n = 52 caps rank at 51, hence the matched-N null alongside):
  **PCA-90 = 17** (generic matched-N 18.8 [17, 21]), **PCA-95 = 25** (24.0
  [21, 27]), **PCA-99 = 39** (32.6 [29, 37]) — at 90/95 the DSR object sits
  inside the generic band; at 99 it sits *above* the null's p95, i.e. its
  variance tail is if anything longer than generic reasoning's
- Reading: the DSR object is about as intrinsically low-dimensional as generic
  reasoning at matched N — separable and capability-orthogonal, but not
  *specially compressed*. The thesis claim drops "low-dimensional".

## Deviations from prereg
None.

## Standing caveats (from the prereg, all apply)
v2 labels are LLM-consensus — the 50-sentence human re-anchor (H1V2) was
UNLABELLED at analysis time, so this verdict is **provisional until the
re-anchor lands**; harmful↔benign difficulty matching dropped by decision
2026-07-21 (grades exist only on the capability arm) — capability-axis
projection and collinearity legs unaffected; benign DSR rows partly reflect
XSTest over-refusal deliberation (in-scope by definition); mixed chunking
protocol in annotation; single model, single reasoning-effort level; N = 52
primary DSR rows.

## What this unlocks (prereg order)
H1 holding opens H2–H4 and the forged-policy provenance probe (P3: effort
sweep with F6 length/label-composition controls, recipe fingerprint, forgery
battery) — new GPU + API spend, separate decision.

Data: `h1_results.json` (this dir); activations `results/safety/p2_activations/`;
code `rom-safety-worktree/p2_h1_analysis.py`; spec `H1_PREREG.md` (sealed).
