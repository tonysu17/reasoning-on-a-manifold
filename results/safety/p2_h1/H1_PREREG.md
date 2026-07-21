# H1 analysis prereg — sealed 2026-07-21, BEFORE any statistic was computed

**Question (thesis §safety H1):** does a separable, low-dimensional deliberative-
safety-reasoning object exist in gpt-oss-20b's residual stream, and does it
survive a capability control?

**Data (fixed):** `results/safety/p2_activations/` — per-label span-mean residual
activations, 24 layers, from the 100 v2-annotated P0 chains (P2 extraction,
2026-07-21). Labels enter as sealed by the κ gates: `harm_recognition`,
`spec_citation`, `decision` (citable) form the **DSR object** (union, deduped by
(chain_id, sentence text) since multi-label sentences repeat identical vectors);
`adjudication` is DROPPED (κ 0.226, no-third-iteration rule); `__generic__` =
consensus-unlabelled sentences is the complement class.

**Layers (F2 rule, no argmax):** fractional depths {0.25, 0.50, 0.75} of the 24
decoder layers via `layer_at_fraction` → {6, 12, 18}. **Primary = 0.50 (L12).**
0.25/0.75 are sensitivity reads; they cannot change the verdict.

**Contrasts:**
- **PRIMARY (topic-controlled):** benign-arm rows only — DSR-on-benign vs
  generic-on-benign. Rationale: harmful chains are near-wall-to-wall labelled
  (4 generic rows), so the all-arms contrast confounds DSR-ness with topic/arm;
  within-benign, deliberative safety reasoning (XSTest over-refusal deliberation)
  and content planning occur in the same chains.
- Secondary: all-arms DSR vs generic (arm composition declared as confound).
- Descriptive only: per-label separations; full-depth profile.

**Leg (b) separation — PASS iff** permutation p < .05 AND chain-grouped held-out
d > 0 in ≥ 4/5 usable folds, at the primary layer on the primary contrast.
Instruments (F2, sealed): `separation_heldout` (5 chain-grouped folds; d + AUROC),
`separation_permutation_null` (n=1000; the N≪d guard), `bootstrap_separation_ci`
(n=1000). Seeds: 0.

**Leg (a) low dimension — comparative, reported with (b)/(c) but verdict-bearing
only as stated:** TwoNN + Levina–Bickel ID of the DSR rows at the primary layer
vs a null of 100 matched-N subsamples of benign generic rows. "Low-dimensional"
claim iff DSR TwoNN point estimate < 5th percentile of the matched-N generic
null. PCA-90% dimension reported descriptively. If N(DSR, benign) < 50 the leg
is reported as under-powered, not passed/failed.

**Leg (c) capability control — PASS iff BOTH** (sealed thresholds from
`capability.py`): retention ≥ 0.50 of the primary separation after projecting
out the capability axis, AND |cos(safety direction, capability axis)| ≤ 0.50.
Capability axis = `capability_direction`(hard, easy) over generic rows of the
20 capability chains, hard = authored grade 5 (9 chains), easy = grade 4 (11
chains). **Difficulty matching of harmful↔benign strata is DROPPED by Tony's
decision 2026-07-21** ("consistent across all questions" — grades exist only on
the capability arm): declared limitation, partially mitigated by the
benign-only primary contrast (topic constant); the capability-axis projection
and collinearity legs are unaffected by the drop.

**H1 verdict rule:** SUPPORTED iff legs (b) AND (c) pass on the primary
contrast at the primary layer. Leg (a) qualifies the claim ("low-dimensional"
kept or struck) but cannot flip SUPPORTED/NOT. Anything else = NOT SUPPORTED,
reported with which leg failed and how.

**Declared caveats (apply to any outcome):** v2 labels are LLM-consensus —
human re-anchor (H1V2, 50 sentences) still unlabelled at seal time; mixed
chunking protocol in annotation (recorded per record); benign DSR rows partly
reflect XSTest over-refusal deliberation (in-scope: that IS deliberative safety
reasoning); N≈tens–hundreds ≪ d=2880 — hence the held-out/permutation
discipline; single model, single effort level (medium).

Analysis code: `p2_h1_analysis.py` (worktree). Outputs: `h1_results.json` +
`H1_REPORT.md` here. Seed 0 everywhere. Sealed before first run; any deviation
gets logged in the report's deviations section.
