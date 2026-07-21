# pt13 — contraction-instrument MDE (ledger §F7)

**Result: the "translation without contraction" null is a CALIBRATED BOUND, not
evidence-of-absence-below-detection.** The participation-ratio dPR instrument is
extremely sensitive to contraction; the observed near-zero dPRs correspond to
removing ≤ ~1.7% of the representation's variance.

## Method
The analogue of pt05's rotation MDE. Inject a known contraction into each base
(R1-1.5B) behaviour matrix by scaling the singular values beyond rank k=5 by a
factor c (tail-variance removal — the operation that lowers effective dimension
and drives dPR negative, matching the observed sign), and record the dPR the
instrument (`spillover.d_eff`) reads. dPR depends only on the singular values, so
the curve is exact and deterministic. Observed dPRs are from
`spillover_star1_full.json` (the §B2 STAR1 arm).

## Sensitivity curve (representative; full grid in pt13_contraction_mde.json)
| injected contraction | variance removed | dPR read |
|---|---|---|
| c = 0.9 (tail ×0.81 variance) | ~11–13% | **−3.6 to −5.5** |
| c = 0.5 | ~45–50% | **−12 to −19** |

## Observed → bound
| cell | observed dPR | ⇒ variance removed |
|---|---|---|
| backtracking L12 | −0.273 | 0.8% |
| backtracking L16 | −0.223 | 0.7% |
| uncertainty L12 / L16 | −0.241 / −0.317 | 0.6% / 0.9% |
| example-testing L12 / L16 | −0.045 / −0.310 | 0.1% / 0.9% |
| adding-knowledge L12 / L16 | −0.164 / −0.682 | 0.4% / **1.7%** (worst cell) |

## Reading
A modest 11% tail contraction would read dPR ≈ −4, an order of magnitude larger
than any observed value. So the observed dPRs are **not** "a contraction hiding
below detection" — the instrument would have shouted. They bound post-training's
effect on the behaviour representations to ≤ ~1.7% variance removal. "STAR-1
translates the residual stream without contracting the behaviour subspaces" is a
defensible bounded claim.

## Scope
The curve is computed on the R1-1.5B base matrices, which are the reference for
BOTH the §B2 (STAR1) and §B5 (RLVR) dPR comparisons, so the instrument-sensitivity
result covers both. The §B2 observed values are mapped directly above; the §B5
RLVR arm's own dPR values should be mapped onto this same curve when consolidating
that claim (same instrument, same base family — the sensitivity conclusion carries;
only the per-cell bound needs the RLVR dPRs substituted). This removes the "two
nulls from one uncalibrated instrument" objection: the instrument is now shown
sensitive to <15% contraction at dPR ≈ −4.
