# Pre-spend de-risk results — 2026-06-27

Two $0–$15 de-risks the recap flagged, both run on cluster `spark-06aa`, both complete.

---

## 1. Covariance-matched-null re-run of the 07d layer sweep (GPU, $0)

**Question:** the original 07d de-confounded sweep used an **isotropic** random null. Critic N argued the behaviour vectors align with high-variance/near-logit directions, so an isotropic null *structurally cannot* subtract the late-layer read-out inflation — making L27's argmax a possible artefact. The test: redo the sweep with a null drawn from the **ambient residual covariance Σ_ℓ** (shares the stream's anisotropy). Pre-registered reading: *if uncertainty's L27 collapses toward the interior, or the adding-knowledge null's L27 peak collapses → L27 was an anisotropy artefact; if L27 still dominates → it survives.*

**Result: L27 SURVIVES. The anisotropy-artefact hypothesis is falsified.**

The covariance null is genuinely stronger than the isotropic one (it subtracts 1.3–2× more at every layer — it did its job), yet L27 still dominates:

| Behaviour @ L27 | raw behaviour effect | iso null | **iso de-confounded** | cov null | **cov de-confounded** |
|---|---|---|---|---|---|
| uncertainty-estimation | 0.0978 | 0.0140 | 0.0838 | 0.0246 | **0.0705** |
| adding-knowledge *(NULL)* | 0.0998 | 0.0172 | 0.0827 | 0.0224 | **0.0772** |
| backtracking | 0.0453 | 0.0159 | 0.0294 | 0.0164 | **0.0310** |
| example-testing | 0.0709 | 0.0110 | 0.0600 | 0.0150 | **0.0500** |

The behaviour direction at L27 moves the output **~4× more than an anisotropy-matched random direction of equal norm** (0.098 vs 0.025 for uncertainty), and the de-confounded L27 effect (0.0705) still **~5.5× the interior** (~0.013 at L11/L16). Argmax = L27 for all four under both nulls. So steering at L27 reliably moves the output, and **not** because of generic high-variance leverage.

**BUT — the "tell" persists and is now sharper.** `adding-knowledge` is a pre-registered specificity **null** (p=1.0 everywhere), yet its covariance-de-confounded L27 effect (0.0772) is the **largest of all four** and its raw L27 leverage (0.0998) is essentially identical to uncertainty's (0.0978). A behaviour with no specificity signal cannot be "causally localised" at L27 — so the L27 effect, while real and robust, is **behaviour-non-specific**: it reflects read-out proximity (any behaviour-correlated direction near the logits moves the next-token distribution a lot), not behaviour-specific causation.

**Net for the decision (this UPDATES the recap's lean):**
- The recap's worry that "L27 will collapse under a fair null" is **resolved — it does not.** This **removes** critic N's anisotropy-artefact objection. L27 is defensible as the build/steer layer on published precedent (Huang) **plus** robust, anisotropy-controlled output leverage.
- The covariance null also **weakens the mid-layer hedge**: under it the interior effects shrink (example-testing goes *negative* at L11/L16; backtracking loses its [11,18] shortlist → just [27]). So 07d gives *less* support for a mid arm than the isotropic run suggested. The PR-trough still points mid *descriptively*, but the causal-intervention evidence (even de-confounded) points late, not mid.
- The **irreducible caveat** (unchanged): 07d cannot show L27 is the behaviour-*specific* locus — the null behaviour peaks there too. Keep CF-10/CF-17 disclosed: L27 is the steering layer by precedent + output leverage, **not** a demonstrated behaviour-specific causal site.

Files: `results/patching/R1-1.5B/steering_effect_{summary,curves}_covmatched.{md,json}` (original isotropic files untouched).

---

## 2. Annotation-noise band — vanilla re-annotation (API, ~$5)

**Question:** the vanilla behaviour-fraction swung **0.181→0.153** across two pilot annotations of byte-identical chains (~25% of the −0.114 headline effect). Put an error bar on it: re-annotate the 10 vanilla chains K=8× with the same annotator (Sonnet 4.5, temp 0) + the 2 historical annotations = 10 runs, same metric as the pilot.

**Result (Sonnet self-consistency band, n=10 runs):**

| Behaviour | vanilla mean frac | **band (std)** | min..max | pilot steering effect | effect / band |
|---|---|---|---|---|---|
| **uncertainty-estimation** | 0.167 | **±0.014** | 0.153..0.196 | **−0.114** | **~8×** ✓ |
| backtracking | 0.053 | ±0.010 | 0.039..0.071 | ~−0.006 … −0.021 | ~0.6–2× ✗ |
| example-testing | 0.069 | ±0.012 | 0.052..0.093 | wrong-sign in pilot | — |
| adding-knowledge *(null)* | 0.113 | ±0.012 | 0.091..0.137 | pre-registered null | — |

**Reading:**
- **uncertainty-estimation** clears its own annotator self-noise by ~8× (single-arm band 0.014; the band on a vanilla−steered *difference* is ≈√2× = ~0.020, still ~5.7×). This is the one behaviour whose effect is comfortably above annotator noise.
- **backtracking** is **within** the difference-band (~0.014) — its tiny pilot effect is not distinguishable from re-annotation noise. Confirms critic O: backtracking is under-powered as a confirmatory arm.
- The historical 0.181/0.153 swing sits inside the measured uncertainty range (0.153–0.196), so it was indeed annotator nondeterminism, now bounded.

**Caveat (important):** this is the **Sonnet-vs-Sonnet self-consistency** band — a *lower bound*. The METHODOLOGY_REFINEMENT §2.6 *acceptance* band is **cross-annotator** (Sonnet vs a non-builder, Qwen3/Nova), which is larger and still requires the non-builder endpoint. uncertainty's ~8× margin is wide enough that a 2–3× larger cross-annotator band would likely still be cleared, but that gate is not yet run.

Files: `results/eval/noise_band/annotation_noise_band.{md,json}`, raw `noise_band_runs.jsonl`.

---

## Combined bottom line

1. **Layer:** build/steer at **L27** — it survives an anisotropy-matched null and has ~4× the leverage of a matched random direction; the mid hedge is *less* supported by causal evidence than the recap assumed. Disclose that L27 is non-behaviour-specific (read-out proximity; the null behaviour peaks there too).
2. **Behaviour:** **uncertainty-estimation is the only defensible headline** — its effect clears the annotator self-noise band by ~8×. Backtracking is within noise; treat as exploratory at most.
3. **Still owed before the real spend:** the **cross-annotator** band (needs the non-builder annotator id) and the **build-A/score-B circularity** fix — neither is touched by these two de-risks.
