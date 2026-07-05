## I. The Layer-Choice & Steering-Methodology Decision (CURRENT STEP)

This is the decision the project is parked on, right before committing real RunPod GPU + Bedrock-annotation spend: **at which layer (and with what steering methodology) do we build and apply the per-behaviour steering vectors for the Phase-7 causal headline?** Everything upstream — the geometry (low-dimensional behaviour-specific *subspace*, citable; curvature a clean negative) — is settled. Phase 7 is the project's *one causal result* and the differentiator vs LRS (arXiv:2606.00726). The layer choice gates every single-layer claim in that chapter, so its evidentiary basis has to be stated without overclaim. This section lays out the candidate layers, the four pieces of code that bear on the choice (cross-layer probing, triangulation, 07b activation patching, 07c attribution patching, 07d steering sweep), exactly *why* 07c is quarantined and what 07d does differently, and the open methodological forks that remain.

### I.1 The candidate layers

The model is `DeepSeek-R1-Distill-Qwen-1.5B`: **28 decoder layers (index 0–27), hidden dim 1536**. The live candidates are:

- **L27 (late / final block).** Huang's (arXiv:2505.22411) *published* steering layer for this exact model. The canonical vector build steers all four behaviours here (`results/steering_vectors/R1-1.5B/` are all-at-L27). The 07d de-confounded forward sweep argmaxes here for all four behaviours. The pilot bake-off confirmed it (clean suppression of uncertainty, gentle damage).
- **~L16 (mid).** The participation-ratio (PR) trough — the *descriptive* concentration layer. `METHODOLOGY §5` records the reconciled per-behaviour PR-trough as **16 / 16 / 16 / 12** (back / unc / ex / add). Venhoff (arXiv:2506.18167), whose diff-of-means recipe we use, steers **mid (layers 15–18)**. So mid is a genuine live alternative grounded in the closest methodological sibling, not a strawman.
- **L11, L18 (backtracking only).** The 07d bootstrap shortlist for backtracking is `[27, 11, 18]` — i.e. backtracking has interior mass the others lack, which matters if we ever commit to a per-behaviour rather than a single global layer.

The fundamental tension: **the published/late evidence (Huang L27, 07d argmax) all reads out *near* L27, and the descriptive/mid evidence (PR-trough L16, Venhoff 15–18) all sits mid.** The two families of evidence disagree by ~11 layers, and — crucially — *the layer is held out on the task axis only, never on the layer axis* (confound CF-17). No piece of code below removes that.

### I.2 Evidence piece 1 — cross-layer probing (Phase 5c): a NULL for layer choice

`results/cross_layer/R1-1.5B/probe_accuracy.json` (consumed by `compute_layer_triangulation.load_phase5c_curves`) trains a linear probe per behaviour at every layer 0–27. Read verbatim from `summary.md`, the probe accuracy is **essentially flat across all 28 layers**:

- backtracking: 0.70–0.76 (no depth trend)
- uncertainty-estimation: 0.72–0.78 (a faint monotone rise to L27=0.78)
- example-testing: 0.78–0.84 (flat, no clear peak)
- adding-knowledge: 0.79–0.84 (flat)

This is *why the triangulation logic classifies every probe curve as `flat`* (the `is_curve_flat` CV<0.03 test in `compute_layer_triangulation.py`). The probe **provides no layer signal at all** — a behaviour is roughly equally linearly decodable everywhere. This is itself an honest finding (and note CF-15: probe leakage was deflated from 0.83–0.93 to 0.70–0.84 and is *flat across depth*, consistent with this). The takeaway for the decision: **linear decodability cannot arbitrate the layer.** Decodable-everywhere ≠ causally-effective-anywhere; the layer question is causal, not probe-based.

### I.3 Evidence piece 2 — multi-criteria triangulation: PR-only, points mid

`compute_layer_triangulation.py` was designed to union three signals — geometry (PR), probe accuracy, attribution-patching effect — into a per-behaviour candidate set. The pre-registered rules are explicit in the docstring (3-point smoothing; geometry peak = **argmin** of PR because "lower PR = stronger low-dimensional manifold"; probe/patching = argmax; flat→fallback {L18, L27}; cap 4). The geometry-signal choice is defended in-file:

```python
# compute_layer_triangulation.py, module docstring
# GEOMETRY SIGNAL = participation ratio (PR), NOT d_eff_70.
#   The manifold hypothesis predicts a *low*-dimensional curved manifold, so the
#   layer where structure is strongest is the one with the LOWEST PR (variance most
#   concentrated). We therefore take argmin(PR). The earlier design used
#   argmax(d_eff_70), but d_eff_70 saturated at the PCA component cap, producing a
#   flat curve that always triggered the fallback. PR is sample-size-robust...
```

But the **on-disk triangulation output (`results/triangulation/R1-1.5B/summary.md`, dated Jun 18) is effectively single-signal**: the patching input is recorded as `MISSING` (it predates the 07d pilot write), the probe is `flat`, so the candidate sets reduce to the PR trough alone:

| Behaviour | PR trough | Probe peak | Patching peak | Candidate set | Agreement |
|---|---|---|---|---|---|
| backtracking | 16 | (flat) | (missing) | **16** | single-signal (PR only) |
| uncertainty-estimation | 16 | (flat) | (missing) | **16** | single-signal (PR only) |
| example-testing | 12 | (flat) | (missing) | **12** | single-signal (PR only) |
| adding-knowledge | 16 | (flat) | (missing) | **16** | single-signal (PR only) |

Two cautions on reading this as evidence *for* mid: (1) the **PR plateaus are enormous** — backtracking's PR-trough plateau is layers `[10..20]`, i.e. "within 1 SD of the minimum" spans half the network, so the argmin=16 is barely distinguished from the late layers; (2) PR is a *descriptive concentration* statistic, not causal. `METHODOLOGY §5` flags this directly: "PR-trough is a **descriptive** (concentration) criterion, not a **causal** one." The triangulation's separate Holm–Bonferroni null table (the geometry *specificity* test, not a layer test) is also informative as colour: 11/20 cells significant, with **adding-knowledge p=1.0 at every layer** and example-testing significant only at L27 — i.e. the two "soft" behaviours have no clean specificity signal at *any* layer.

### I.4 Evidence piece 3 — activation patching (07b): the brute-force reference, confounded metric

`07b_activation_patching.py` + `src/activation_patching.py` are the original causal-localisation attempt: for a behaviour-positive donor and a DEDUCTION-labelled negative, patch the positive's residual at layer L with the negative's residual at the matched position, and measure the shift in **behaviour-marker next-token logprob**. The metric is the lexical-marker proxy that the whole later redesign exists to escape:

```python
# src/activation_patching.py — BEHAVIOUR_MARKER_TOKENS
"backtracking":           ["wait", "actually", "no", "hmm", "alternatively"],
"uncertainty-estimation": ["maybe", "perhaps", "possibly", "might", "unsure", "guess"],
"example-testing":        ["test", "example", "try", "consider", "case", "instance"],
"adding-knowledge":       ["recall", "know", "formula", "fact", "definition", "theorem"],
```

This is **CF-10a** (the metric conflates the behaviour with its surface lexis — a chain can backtrack without "wait", or say "wait" without backtracking; the marker lists differ in size/base-rate so cross-behaviour scores are not commensurable) and **CF-10b** (it patches "position i across chains," but position i is a different point in two non-aligned chains). The runner also does `tpos = T_min - 1` — *always patch the boundary/last common token* — which is the crude positional rule the later modules replace. The 07b pilot files on disk (`effect_curves_*_pilot.json`, dated **May 28**) are stale and were never folded into the triangulation. **07b is superseded; do not use its numbers for the layer pick.** Its lasting value is `brute_force_patch_effect` in `src/attribution_patching.py` as the exact check on the first-order estimator.

### I.5 Evidence piece 4 — attribution patching (07c): CONFOUNDED, do-not-use

`07c_attribution_patching.py` + `src/attribution_patching.py` were the principled fix to 07b: keep the donor-pair causal design, but (a) replace the lexical metric with a **geometry-based** one — projection of the residual onto the behaviour's own unit diff-of-means steering direction (the CF-10a fix; "scoring the same object we steer") — and (b) align positions at the behaviour-onset token ± a window (CF-10b). It then uses **attribution patching** (Syed 2023; Nanda 2023), the first-order Taylor approximation:

```python
# src/attribution_patching.py — attribution_patching(), the estimator
for tc, tk in alignment.pairs:
    delta = a_corr[0, tk] - a_clean[0, tc]      # (d,)
    eff = float(torch.dot(g[0, tc], delta).item())
    pos_map[int(tc)] = eff
# effect(L,t) ≈ grad_clean[L,t] · (act_corrupt[L,t'] − act_clean[L,t])
```

It was **RAN on 2026-06-21 (20 pairs × 4 behaviours, all 28 layers)** and came back confounded. The on-disk `attribution_summary.md` is unambiguous — every behaviour ramps *monotonically* to its argmax at L26/27, with no interior peak:

| Behaviour | argmax L | L0 | L10 | L17 | L24 | L27 |
|---|---|---|---|---|---|---|
| backtracking | **27** | 9.94 | 29.2 | 27.7 | 53.4 | 75.3 |
| uncertainty-estimation | **27** | 10.4 | 25.2 | 42.9 | 107 | 133 |
| example-testing | **27** | 12.3 | 35.0 | 53.2 | 94.3 | 98 |
| adding-knowledge | **26** | 7.66 | 19.7 | 38.9 | 86.5 | 87.9 |

**Why it is confounded (read-out proximity, CF-10):** the metric reads the residual at a *fixed late layer* (`--read-layer 27`, the build layer). Attribution then estimates ∂(L27 metric)/∂(act at ℓ); that gradient is mechanically larger the closer ℓ sits to the read-out, because there is less non-linear stack between them. A first-order gradient at a fixed late read-out **structurally has no interior peak to find** — the L27 argmax is an artefact of *where you read*, not *where the behaviour is decided*. The 07c file even anticipates this in its own docstring ("a layer downstream of the read-out has zero effect to first order"), and `METHODOLOGY §5` records the verdict: `[RAN — CONFOUNDED, do not use for layer pick]`. **07c is quarantined.** Its surviving contributions are the reusable, GPU-free `BehaviourMetric` (geometry projection, differentiable, unit-testable) and the brute-force reference.

### I.6 Evidence piece 5 — the de-confounded steering sweep (07d): the only code that informs the pick (still a proxy)

`07d_layer_steering_sweep.py` + `src/layer_sweep.py` are the redesign that replaces 07c. The conceptual fix is to stop reading at a fixed late layer and stop linearising. Instead it **actually intervenes** — adds (or subtracts) a per-layer direction during a real forward pass — and reads out at the **OUTPUT** (common to every ℓ), so there is no fixed-read-out proximity term. Five de-confounding controls are built in (verbatim from the docstring): per-layer diff-of-means direction `v_ℓ` built *at each layer* from the all-layer activations; a **norm-matched random-direction null** subtracted per layer; **per-layer α-normalisation** (inject a delta of fixed norm-fraction `δ-frac=0.1` of the median ‖h‖ at that depth, so "same α" is the same strength across depth); a **bootstrap shortlist** rather than a single argmax; and a `--start-layer 5` skip of the embedding-correlated early band (Venhoff). The de-confounded statistic:

```
De-confounded effect(ℓ) = Score_b(ℓ) − mean_r Score_random(ℓ)
   Score = KL(p_steered ‖ p_baseline) at onset−1   (primary; captures redistribution to synonyms)
```

It was **RAN (2026-06-22; n_donors=12, R=2 random, KL read-out, layers 5–27)**. The on-disk `steering_effect_summary.md` shortlists:

| Behaviour | SHORTLIST (candidate ℓ) | point argmax |
|---|---|---|
| backtracking | **27, 11, 18** | 27 |
| uncertainty-estimation | **27** | 27 |
| example-testing | **27** | 27 |
| adding-knowledge | **27** | 27 |

So 07d, *even after de-confounding*, lands on **L27 for all four** (only backtracking has interior candidates 11/18). De-confounded effect at L27 dominates the curve (e.g. uncertainty L27=0.084 vs ~0.012–0.020 mid; adding-knowledge L27=0.083 vs a mid bump ~0.044 at L16). **This is the single strongest positive for L27 that is not just "Huang said so."** But three caveats keep it a *proxy*, not the headline:

1. **Residual late-layer proximity.** The output read-out kills the *fixed-read-out* proximity term, but the KL-at-output is still mechanically more sensitive to perturbations injected *near the logits*. The random-direction null is meant to subtract exactly this "global late-layer sensitivity," but with only **R=2** random directions and **n=12** donors, the null is thin — the residual L27 spike `METHODOLOGY` flags is plausibly under-subtracted. CF-10 stays open ("07d a proxy with a residual L27 spike").
2. **Token-anchored.** The secondary read-out is the onset-token Δlog-prob; even the KL primary is anchored at the onset-1 position. It measures "does steering move the next-token distribution at the behaviour boundary," which is a proxy for "does steering suppress the behaviour over a generated chain" — the actual Phase-7 endpoint.
3. **It is not the floored Δ_floor.** 07d uses a random-direction null, *not* the `energy_matched_random` floor the headline will use. It is a layer arbiter, not a causal-effect measurement.

### I.7 The honest reconciliation: why the pick still lands on L27

Three converging positives (`REVIEW_PHASE7 §1.4`): (1) Huang published L27 for this exact model; (2) 07d de-confounded argmax = L27 ×4; (3) the pre-registered pilot rule confirmed L27 (it suppressed uncertainty −63% relative AND stayed clean — 7/8 cells repetition ≤ vanilla, whereas L16 *inflated* repetition in 7/8 cells and ran chains 500–1120 tokens longer, the "steering breaking the model" signature). The **caveats that must appear on every single-layer headline**: the 07d proxy reads near L27 (late-layer proximity bias even after the null); Venhoff steers mid and the PR-trough is mid; **CF-17 — the layer is not held out on the layer axis**; and the cheap pilot arbiter is *vanilla-relative*, not the de-confounded floor. There is a documentation inconsistency to reconcile before any thesis citation: `RESULTS_LEDGER:73` still cites "Venhoff mid-peaks 11/16/19/16" as *the* de-confounded finding, which contradicts the on-disk 07d **argmax L27** (`REVIEW_PHASE7 §5`).

### I.8 The open methodological decisions (the real forks)

Beyond the layer, `METHODOLOGY_REFINEMENT §2` and `§7` enumerate the methodology choices still owed to the PI:

- **Layer rule (pre-registered, `§2.9`):** annotate L27 first with the non-builder annotator, compute `vanilla_fraction − steered_fraction` (single_direction, α=1), confirm L27 iff it suppresses AND stays clean (repetition ≤ vanilla + margin = "surgical not weak"); fall back to L16 *only if L27 fails*. If per-behaviour orderings disagree (07d backtracking 11/18 vs others 27), do **not** majority-vote — drop to a per-behaviour mid/late build and flag "needs the full 50-task run." **The layer pick on existing chains is $0-GPU and vanilla-relative (coarse, L27-proximity-confounded); the floored Δ_floor headline needs a fresh generation run.**
- **Coefficient / dose:** sealed per-behaviour α\* from `predictions_layer27.json` (back 0.994 / unc 0.969 / ex 0.964 / add 1.056), used as a *fixed dose*, NOT tuned on the eval grid — and explicitly **bypass `effect_quantile=0.8`** in `compare_across_behaviours` (that reads the matched target off the eval curves = tune-on-eval leak). Note α\* is **L27-only**: if any behaviour lands on L16, its dose is unsealed.
- **Pooling:** SETTLED — mean over `[onset−1 : +10]`, verified identical to Venhoff's published code. (Minor caveat: 15.8% of sentences are shorter than the window.)
- **Single vs manifold vector:** both are headline arms, but the floors differ — `single_direction` is **energy-floored** (vs `energy_matched_random`), the **manifold arms are only dimension-matched** (vs `random_subspace_k`), NOT energy-matched. Do not claim the manifold headline is energy-floored unless `energy_matched_random_k{k}` is built. Decide *before* running whether both are in the frozen Holm family.
- **Control arms:** vanilla (shared reference, not the headline subtrahend), single_direction, manifold_k*, random_subspace_k (dimension floor), energy_matched_random (real floor, single only), orthogonal_complement, random_direction (~19× under-energy sanity floor only — never the baseline). Gate-vs-gradient arms (B2) demoted to conditional follow-up.

### I.9 Decision table

| Layer | Evidence FOR | Evidence AGAINST / risk | Held-out on layer axis? | Status |
|---|---|---|---|---|
| **L27 (late)** | Huang published (same model); 07d de-confounded argmax ×4; pilot confirmed (clean −63% uncertainty, gentle damage); α\* sealed here | residual late-layer/read-out proximity even after 07d null (R=2, n=12 thin); next-token-dominated final block; 07c L27 was a pure artefact (cautionary) | **No (CF-17)** | Leaning choice; pilot-confirmed but headline UNRUN |
| **~L16 (mid)** | PR-trough (16/16/16/12); Venhoff steers 15–18; 07d backtracking shortlist includes 18; mid bump in 07d add-knowledge curve | descriptive (concentration) not causal; pilot showed L16 *inflates* repetition + lengthens chains (dirty); PR plateau spans [10..20] so argmin barely distinguished; α\* not sealed at L16 | No | Live fallback; fails pilot cleanliness bar |
| **Per-behaviour mid/late** | honest if 07d orderings genuinely split (backtracking interior) | 10-task pilot under-powered for a 4-way split; complicates the Holm family; unsealed doses off L27 | No | Deferred to full 50-task run if orderings disagree |

### I.10 How this connects to the spend decision

The layer choice *is* the cheap, first decision and it conditions everything downstream. Two facts make it tractable and two make it fragile. Tractable: (a) the layer pick on the **existing bake-off chains** is $0-GPU (vanilla-relative annotated suppression at L27); (b) the pilot already confirmed L27 by the pre-registered clean-guard. Fragile: (a) the de-confounded *headline* (Δ_floor vs energy-matched-random at sealed α\*) requires a **fresh generation run** — the bake-off chains lack the floor arms and the α\* dose, so they cannot be reused; (b) **no non-builder annotator id is located** (Qwen3-235B / Nova-Pro live endpoint), so without it any headline is "preliminary, band-ungated" — `src/annotation.py` knows only the Sonnet id, and Sonnet built the labels *and* would score the steered outputs (circular, CF-7). The recommended scope is **Tier-C (~2,400 chains, ≈$130–180 + RunPod)**, with the explicit expectation that the clean causal win may rest on **uncertainty-estimation (± backtracking)** — adding-knowledge is a pre-registered null (p=1.0 everywhere) and example-testing nudged the *wrong* direction at L27 in the pilot. The single sharpest thing to internalise before spending: **L27 is defensible but its strongest non-Huang support (07d) reads out near L27, so the "surgical late-layer" story can never fully shed the late-layer-proximity confound, and the layer is not held out on the layer axis — disclose CF-17 verbatim on every single-layer headline.**
