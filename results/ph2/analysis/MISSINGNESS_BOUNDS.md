# Phase-2 arm-differential missingness bounds

Spec: `results/prereg/PH2_MISSINGNESS_BOUNDING_SPEC_2026-08-19.md` (imputation rules fixed before running).
Complete-case anchors reproduced exactly (8 A2 cells + 3 steered cells).

## A2 vanilla prevalence contrasts (target − base)

| Role | Endpoint | Complete-case (n) | Manski [lo, hi] | m* minuend / subtrahend | q50 / q75 / q90 | Verdict |
|---|---|---|---|---|---|---|
| star1 | prev_backtracking | -0.0287 (90) | [-0.0809, +0.0291] | 0.580 / out | -0.0335 / -0.0319 / -0.0317 | sign-robust (behaviour-dense) |
| star1 | prev_uncertainty-estimation | -0.0520 (90) | [-0.1057, +0.0043] | out / out | -0.0589 / -0.0552 / -0.0562 | sign-robust (tipping) |
| star1 | prev_example-testing | -0.0271 (90) | [-0.0719, +0.0381] | 0.437 / out | -0.0249 / -0.0227 / -0.0197 | sign-robust (behaviour-dense) |
| star1 | prev_adding-knowledge | +0.0255 (90) | [-0.0265, +0.0835] | out / 0.593 | +0.0253 / +0.0270 / +0.0270 | sign-robust (behaviour-dense) |
| deepscaler | prev_backtracking | +0.0205 (89) | [-0.0308, +0.0892] | out / 0.523 | +0.0216 / +0.0236 / +0.0227 | sign-robust (behaviour-dense) |
| deepscaler | prev_uncertainty-estimation | -0.0257 (89) | [-0.0735, +0.0465] | 0.449 / out | -0.0217 / -0.0195 / -0.0210 | sign-robust (behaviour-dense) |
| deepscaler | prev_example-testing | -0.0242 (89) | [-0.0723, +0.0477] | 0.381 / out | -0.0233 / -0.0221 / -0.0184 | sign-robust (behaviour-dense) |
| deepscaler | prev_adding-knowledge | -0.0105 (89) | [-0.0622, +0.0578] | 0.229 / out | -0.0112 / -0.0117 / -0.0111 | sign-robust (behaviour-dense) |

## Primary steered cells (energy floor − transported arm, prev_backtracking)

| Role | Complete-case (n) | Manski [lo, hi] | m* minuend / subtrahend | q50 / q75 / q90 | Verdict |
|---|---|---|---|---|---|
| base | -0.0132 (74) | [-0.1890, +0.0810] | 0.195 / 0.038 | -0.0048 / -0.0129 / -0.0305 | sign-robust (behaviour-dense) |
| star1 | -0.0219 (85) | [-0.1370, +0.0130] | 0.765 / out | -0.0082 / -0.0165 / -0.0362 | sign-robust (behaviour-dense) |
| deepscaler | -0.0587 (73) | [-0.2633, +0.0167] | out / out | -0.0566 / -0.0885 / -0.1116 | sign-robust (tipping) |

Citation rule (from the spec): cite complete-case values with per-arm unresolved counts adjacent plus the robustness label; no imputed value is a result. Full-chain endpoints are unaffected by construction.
