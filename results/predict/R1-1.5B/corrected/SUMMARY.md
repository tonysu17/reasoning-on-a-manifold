# Predictive Geometry — pilot verdict (corrected, train-on-all)

**Setup.** Pilot judge: 200 chains → **183 usable** (84 correct / 99 incorrect; 11 uncertain
excluded). Predictor (label-agnostic) trained on **all 986 chains**, chain-grouped OOF;
correctness AUC evaluated on the labelled subset only. Layers 14, 17, 27.

## Headline
Residual geometry of a next-step predictor carries a **modest but significant**
correctness signal, **concentrated at later layers**, driven by predictability
**magnitude** (not step order); a **nonlinear (JEPA) predictor extracts it slightly
better** than a linear one.

| layer | ridge AUC (lp p) | JEPA AUC (lp p) | best baseline | step-shuffle null |
|---|---|---|---|---|
| 14 | 0.542 (0.128) | **0.582 (0.026)** | 0.549 | n.s. (p=0.72) |
| 17 | 0.486 (0.421) | 0.541 (0.146) | 0.506 | n.s. (p=0.77) |
| 27 | **0.578 (0.034)** | **0.591 (0.024)** | 0.543 | — |

(lp = label-permutation null, within difficulty strata; "n.s." = not significant.)

## Reading
- **The predictor works.** residual/persistence ≈ 0.88 (ridge) and displacement R² ≈ 0.3
  (sweep) — step-to-step reasoning is partially predictable in latent space and
  generalises across chains.
- **Signal is real but small.** AUC ~0.58–0.59 at L27 (both predictors) and L14 (JEPA),
  beating all baselines and surviving the label-permutation null. L17 is a dead spot.
- **Order does not matter** (step-shuffle null never rejects) → the correctness signal is
  in residual *magnitude* (overall predictability of the chain), not trajectory *order*.
  Supports **H1**; does **not** support **H3** (order/branch geometry). Matches PHi
  (2503.13431): predictor unpredictability ↔ correctness.
- **Rung-2 > Rung-1** at every layer tested → the JEPA's nonlinearity adds a little.
  **Convergence check (60 epochs, `../corrected_ep60/`):** JEPA does NOT improve with more
  training — L14 ratio *worsens* 1.139→1.573 (overfits the displacement), L27 AUC slips
  0.591→0.580. So 25 epochs ≈ optimal and the JEPA edge over ridge is small and **does not
  scale** — i.e. the correctness signal is largely **linear-accessible**; extra nonlinear
  capacity buys little.

## Integrity note (important)
The first gate trained the predictor on the **labelled subset only** (~183 chains) and
reported **inflated** AUC ≈ 0.61 (p<0.01) — an artifact of a small-sample predictor whose
residuals encoded chain idiosyncrasies that correlate with correctness. Training on all
chains (the correct design) gives the honest 0.54–0.59. **Report the corrected numbers**;
the flawed run is preserved at `../gate_pilot.md` for comparison.

## Recommendation
Scale to the **full ~1000 labels** to tighten the AUC estimates, map the layer profile
properly, and pin the ridge-vs-JEPA gap — the L27 signal is real but the 183-chain
estimate is noisy. (User decision; it is 5× the pilot spend.)
