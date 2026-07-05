# Per-layer steering-effect sweep — R1-1.5B

Method: FORWARD-PASS intervention, de-confounded (CF-10 replacement for 07c attribution patching). Layers 5..27 (start-layer skips the embedding-correlated early band); per-layer δ-frac=0.1 (delta norm = δ-frac·median‖h‖); suppress (−α); **primary read-out = kl**; R=2 norm-matched random directions; donors/behaviour=12; bootstrap n=1000; each donor cropped to a local window of W=1024 pre-onset tokens (+span tail) — the post-onset tail is dropped exactly (it never enters the teacher-forced onset read-out) and the pre-onset crop is the standard local-context approximation, a no-op for chains whose onset ≤ W.

**De-confounded effect(ℓ) = Score_b(ℓ) − mean_r Score_random(ℓ)**, where Score is the KL(p_steered‖p_baseline) at onset−1 (primary; captures redistribution to synonym onsets) — the secondary onset-token Δlog-prob is also recorded. The random-direction null subtracts any *global* sensitivity of a layer to ANY perturbation (e.g. late layers near the logits), isolating the behaviour-specific causal effect. The read-out is the fixed OUTPUT, so there is no read-out-proximity term either.

**The SHORTLIST is a PRE-FILTER for Phase 7 to CONFIRM, not a single load-bearing argmax.** Layers are shortlisted if they win ≥ 15% of bootstrap resamples OR sit within 1 SEM of the max de-confounded effect (SEM = std(ddof=1)/√n).

⚠️ Caveats: the secondary log-prob read-out is token-anchored; KL is unsigned (magnitude of redistribution). α/δ-frac and the suppression window are choices.

## Per-behaviour SHORTLIST (pre-filter for Phase 7)

| Behaviour | SHORTLIST (candidate ℓ) | point argmax | n donors |
|---|---|---|---|
| backtracking | **27**, **11**, **18** | 27 | 12 |
| uncertainty-estimation | **27** | 27 | 12 |
| example-testing | **27** | 27 | 12 |
| adding-knowledge | **27** | 27 | 12 |

## De-confounded effect curve (mean ± SEM)

| Behaviour | L5 | L8 | L11 | L13 | L16 | L19 | L21 | L24 | L27 |
|---|---|---|---|---|---|---|---|---|---|
| backtracking | 0.00294±0.00065 | 0.0102±0.0037 | 0.0191±0.0045 | 0.0123±0.0043 | 0.0084±0.0039 | 0.00974±0.0032 | 0.00635±0.0033 | 0.0131±0.0061 | 0.0294±0.013 |
| uncertainty-estimation | 0.00601±0.0018 | 0.0124±0.0019 | 0.0186±0.0056 | 0.0145±0.0055 | 0.0197±0.0061 | 0.0141±0.0038 | 0.0107±0.0029 | 0.0113±0.0024 | 0.0838±0.014 |
| example-testing | 0.000615±0.0034 | 0.00872±0.0029 | 0.00368±0.0026 | 0.00397±0.0026 | 0.00903±0.0037 | 0.0132±0.0036 | 0.0105±0.0018 | 0.0102±0.0023 | 0.06±0.028 |
| adding-knowledge | 0.0116±0.0042 | 0.0147±0.0035 | 0.015±0.0047 | 0.0325±0.014 | 0.0437±0.014 | 0.0318±0.0077 | 0.0203±0.0046 | 0.0216±0.0066 | 0.0827±0.03 |

Per-layer {behaviour_effect, random_null, deconfounded_effect, kl_effect, logprob_effect, sem, n} and the bootstrap frequencies are in `steering_effect_curves.json`; per-donor breakdowns under each behaviour's `trials`.