> ⚠️ SUPERSEDED — DO NOT CITE (see RESULTS_LEDGER.md §E): pre-dedup 2026-06-08 run; the TwoNN "0.168" intrinsic-dim is a zero-distance (duplicate-row, CF-13) artefact and the p-values are unsmoothed count/B. Live replacement: `results/robustness/R1-1.5B/geometry_robustness_summary.md`. (Re-run owed: `tier1_geometry_nulls.py` after re-extraction.)

# Tier-1 R1.1 - Geometry nulls (R1-1.5B, layer 27)

CF-3 fix: the chain-stratified null applied to the *intrinsic-dim* and
*curvature* numbers (not just the variance ratio). tail='lower' — the
claim holds only if the real value sits **below** the null and p is small.
A p-value near the null mean (real ~= null) means the statistic is a
property of the chains, not the behaviour.

n_resamples=200, max_points=0

| Behaviour | Statistic | real | chain-strat null mean | p (chain-strat) | p (cross-chain) |
|-----------|-----------|-----:|----------------------:|----------------:|----------------:|
| backtracking | twoNN_intrinsic_dim | 0.168 | 0.151 | 0.9950 | 0.9050 |
| backtracking | levina_bickel_intrinsic_dim | 3.348 | 4.600 | 0.0000 | 0.0000 |
| uncertainty-estimation | twoNN_intrinsic_dim | 3.460 | 0.235 | 1.0000 | 0.9750 |
| uncertainty-estimation | levina_bickel_intrinsic_dim | 5.069 | 5.618 | 0.0000 | 0.0100 |
| example-testing | twoNN_intrinsic_dim | 4.817 | 0.415 | 1.0000 | 1.0000 |
| example-testing | levina_bickel_intrinsic_dim | 7.418 | 8.101 | 0.0000 | 0.2550 |
| adding-knowledge | twoNN_intrinsic_dim | 6.626 | 4.771 | 1.0000 | 1.0000 |
| adding-knowledge | levina_bickel_intrinsic_dim | 10.268 | 11.548 | 0.0000 | 1.0000 |

**Interpretation key.** chain-strat is the primary test (controls chain
identity + composition + sample size). If real < null with small p, the
behaviour carries genuinely lower-dimensional / more-curved structure than
a chain-matched random relabelling. If p is large, the headline number was
the chain confound (CF-2) wearing a behaviour label.