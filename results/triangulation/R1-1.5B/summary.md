# Layer triangulation summary

Geometry signal = **participation ratio** (argmin: lower PR = stronger low-dimensional manifold). Probe accuracy and patching effect use argmax.

## Inputs

- Phase 5 (PR curves): `results/pca/R1-1.5B/layer_profiles.json` (found)
- Phase 5 (null p-values): `results/pca/R1-1.5B/null_pvalues_per_layer.json` (found)
- Phase 5c (probe accuracy): `results/cross_layer/R1-1.5B/probe_accuracy.json` (found)
- Phase 7b-pilot (patching): `results/patching/R1-1.5B/pilot_effect_curves.json` (MISSING)

## Per-behaviour layer peaks

| Behaviour | PR trough | Probe peak | Patching peak | Candidate set | Agreement |
|---|---|---|---|---|---|
| backtracking | 16 | (flat) | (missing) | 16 | single-signal (PR only; others flat/missing) |
| uncertainty-estimation | 16 | (flat) | (missing) | 16 | single-signal (PR only; others flat/missing) |
| example-testing | 12 | (flat) | (missing) | 12 | single-signal (PR only; others flat/missing) |
| adding-knowledge | 16 | (flat) | (missing) | 16 | single-signal (PR only; others flat/missing) |

## Null-test significance (Holm–Bonferroni, family α=0.05, m=20 behaviour×layer cells)

11/20 cells significant; 0 cells resolution-limited (min attainable p = 1/(B+1) exceeds their Holm threshold — non-significance there is a resolution statement, not evidence of absence).

| Behaviour | Layer | p (smoothed) | Holm threshold | Significant | Resolution-limited |
|---|---|---|---|---|---|
| backtracking | 11 | 0.0003998 | 0.0025 | **yes** | no |
| backtracking | 14 | 0.0003998 | 0.00263 | **yes** | no |
| backtracking | 17 | 0.0003998 | 0.00278 | **yes** | no |
| backtracking | 20 | 0.0003998 | 0.00294 | **yes** | no |
| backtracking | 27 | 0.0003998 | 0.00313 | **yes** | no |
| uncertainty-estimation | 11 | 0.0003998 | 0.00333 | **yes** | no |
| uncertainty-estimation | 14 | 0.0003998 | 0.00357 | **yes** | no |
| uncertainty-estimation | 17 | 0.0003998 | 0.00385 | **yes** | no |
| uncertainty-estimation | 20 | 0.0003998 | 0.00417 | **yes** | no |
| uncertainty-estimation | 27 | 0.0003998 | 0.00455 | **yes** | no |
| example-testing | 27 | 0.0003998 | 0.005 | **yes** | no |
| example-testing | 17 | 0.01919 | 0.00556 | no | no |
| example-testing | 20 | 0.01919 | 0.00625 | no | no |
| example-testing | 11 | 0.6074 | 0.00714 | no | no |
| example-testing | 14 | 0.7533 | 0.00833 | no | no |
| adding-knowledge | 11 | 1 | 0.01 | no | no |
| adding-knowledge | 14 | 1 | 0.0125 | no | no |
| adding-knowledge | 17 | 1 | 0.0167 | no | no |
| adding-knowledge | 20 | 1 | 0.025 | no | no |
| adding-knowledge | 27 | 1 | 0.05 | no | no |

## Methodological commitments (pre-registered)

1. 3-point moving average applied to each curve before peak detection.
2. Geometry peak = argmin of smoothed participation-ratio curve.
   Probe/patching peak = argmax of their smoothed curves. Plateau = within 1 SD.
3. Fallback to {L18, L27} if all curves are flat (CV < 0.03).
4. Candidate set capped at 4 layers per behaviour.
5. Holm–Bonferroni across all behaviour×layer null cells (implemented above; p-values are Phipson–Smyth smoothed).