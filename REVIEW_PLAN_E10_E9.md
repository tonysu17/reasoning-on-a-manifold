# Coherence review — E10 + E9 arc (prepared 2026-07-07, for launch at the credit-reset window)

> Ready-to-launch spec for the multi-agent thesis coherence review. Launch = `/code-review`-style
> multi-agent pass OR a Workflow fan-out over the sections below. Treats the E9.2 rung-three
> paragraph as a PREDICTION (it is, until E9.2 lands) and flags the collapse throughline as
> E9.2-outcome-contingent.

## Scope (what changed since the last 74-agent pass, 2026-07-02)
Canonical build: `thesis/ucl_msc.tex` (74pp, compiles). Review the **v2 chapters** with emphasis on
the material added this session:
1. `chapters/v2/background.tex` — the **featurizer/causal-abstraction** subsection in
   `sec:bg-mechinterp` (eq:bg-featurizer; IIT/DAS/SAE-as-featurizer; our method located as a fixed
   correlational featurizer; Makelov illusion caveat). Cross-ref clause at :381.
2. `chapters/v2/steering.tex` — `sec:steering-featurizer-programme` (ALL THREE E10 rungs now written
   as executed findings): dictionary gate (E10.0 PASS), interchange (E10.1 P1 confirmed + grounding
   controls splitting L17 vs L11/L27), generation stage (P2b clamp-gentler-only-where-grounded),
   causal width (E10.2 ~2-D). PLUS the pre-existing `sec:steering-collapse` +
   `sec:steering-collapse-programme` (E9.0/E9.1/E9.1b executed; rung-three = E9.2, prospective).
3. Bib: 7 new entries (geiger2022inducing, geiger2024alignments, wu2023boundless,
   geiger2025causalabstraction, makelov2023illusion, wang2025resa) — verify keys/venues/no dupes.

## What each reviewer dimension checks
- **Factual accuracy vs results** — every number in the featurizer section against
  `results/das/R1-1.5B/main/cis.json` (P1 +0.601 CI[0.538,0.666]; +0.649 vs +0.048),
  `results/das/R1-1.5B/width/report.json` (width AUC 0.73→0.89, transfer 0.62→1.68),
  `results/eval/R1-1.5B__E10_P2b/p2b_analysis.json` (clamp −0.31 CI[−0.44,−0.19] das; +0.18 dm),
  `results/sae_gate/R1-1.5B/` (FVU 0.061 / CE-rec 0.958). Flag any drift.
- **Prediction vs result discipline** — the rung-three "injection in representation space" paragraph
  must read as a SEALED PREDICTION, not a finding (E9.2 not landed at review time). Any sentence
  asserting a structured-vs-iso outcome is a bug.
- **Cross-chapter coherence** — does the featurizer section's "grounding decides" throughline
  (P2b: bounded intervention gentle only on the grounded axis) connect cleanly to the collapse
  chapter and the conclusion's RSI/interpolation paragraph? Flag contradictions with the geometry
  chapter's low-dimensionality claim (E10.2 width ~2 is BELOW corr-dim 6–8 — this mismatch must be
  stated, not smoothed).
- **Hedging / caveat integrity** — every on-target claim inherits the within-annotator caveat;
  the E10.2 add-knowledge-ungrounded and P2b boxed-0.00 caveats must be present.
- **No forbidden moves** (guards from COLLAPSE_AND_ENTROPY.md §6): manifold-null stays geometric,
  no blanket "steering degrades", no curvature resurrection, no dose/diversity claim beyond what ran.

## E9.2-CONTINGENT FLAG (for the post-landing targeted pass, NOT this review)
The collapse throughline leans on E9.2 confirming that **causal/grounded-frame noise
Pareto-dominates isotropic** (P1). If E9.2 REFUTES P1 (geometry doesn't matter), the
"structured entropy is special / grounding decides" framing in `sec:steering-collapse-programme`
and any conclusion sentence echoing it must soften to "entropy helps but geometry is not the lever".
Phase-2 (after E9.2 lands + integrates) re-examines ONLY: rung-three paragraph, the collapse
throughline sentences, and the conclusion RSI paragraph.

## Launch options (pick at fire time)
- **A. `/code-review`-style multi-agent** over the file list above (correctness of claims-vs-data).
- **B. Workflow fan-out**: one agent per dimension (accuracy / prediction-discipline / cross-chapter
  coherence / hedging / guards), each returns findings, synthesize + apply verified fixes.
Either way: report findings ranked, apply only verified factual/coherence fixes, recompile both
builds, do not touch the dilution-artefact analysis or the geometry-chapter curvature negative.
