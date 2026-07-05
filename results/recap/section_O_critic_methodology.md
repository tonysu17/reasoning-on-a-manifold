## O. Adversarial Review — The Steering Methodology

*Scope: not "is L27 the right layer" (largely settled) but "will the Phase-7 steering experiment, as currently designed and budgeted, produce a result a thesis examiner cannot dismiss?" Read against `07_evaluate_steering.py`, `src/steered_inference.py`, `src/evaluation.py`, `06_build_steering.py`, `build_steering_arms.py`, the two bake-off scripts, `REVIEW_PHASE7_2026-06-25.md`, `METHODOLOGY_REFINEMENT_2026-06-25.md`, `CONFOUNDS_AND_REMEDIATION.md` (CF-5/7/10/17), and `GPU_GUIDE.md`. The recap's own §G/§H were read but treated as the thing under audit, not as ground truth.*

The headline verdict up front: **the engineering is unusually disciplined and the confound register is honest, but the experiment as scoped has one fatal flaw and two structural under-powerings that the trackers under-rate.** The fatal flaw is not the layer and not the energy floor — it is that **the primary outcome is scored by the same model that defined the construct, and the proposed fix does not exist.** Everything else is a question of how small a defensible result can be made.

---

### O.1 Is "behaviour-fraction under steering" a valid causal outcome? — Partly. The metric is fine; the *judge* makes it circular, and the circularity is worse than a same-family annotator.

`behaviour_fraction` (`src/evaluation.py:28`) is sentence-fraction of spans the annotator labels as the target behaviour. As a *causal* outcome it is structurally sound in one respect the trackers correctly defend: the shared-vanilla baseline (`steered_inference.py:627`), the missing/empty re-annotation skip-not-zero accounting (`evaluation.py:135-145`), and the row-provenance task hold-out are all genuinely de-confounding moves that most steering papers skip. That part is real.

But the circularity here is **not** the generic "same model family judges" worry — it is sharper and the docs soft-pedal it:

1. **The annotator did not merely score the construct, it *defined* it.** The four behaviours, their operational boundaries, and every training label for the steering vectors come from Sonnet's Venhoff-prompt annotations. The steering vector is literally `mean(Sonnet-ON) − mean(Sonnet-OFF)`. Then Sonnet re-scores the steered output. This is not "an LLM judging an LLM" — it is **the same labelling function appearing on both the X and the Y axis of a causal claim.** If Sonnet has an idiosyncratic, lexically-anchored notion of "uncertainty-estimation" (CF-18 shows it does — uncertainty spans are 21.6% for Sonnet vs 7.0% for Nova), then a vector that suppresses *Sonnet's lexical markers of uncertainty* will, tautologically, lower *Sonnet's measured uncertainty fraction*. You will have shown that subtracting the Sonnet-uncertainty direction lowers the Sonnet-uncertainty count. That is closer to a consistency check than a causal discovery.

2. **A non-builder annotator only half-fixes it.** `--annotator-model` (wired at `07_evaluate_steering.py:118,244`) lets Qwen3 *score* the steered chains. Good — it breaks the score-side circularity. But the *vector is still built from Sonnet labels.* So even with Qwen3 scoring, you are testing "does the Sonnet-direction suppress the Qwen3-construct?" — which is a *better* test (the two must agree for the effect to be real-not-lexical), but the gold-standard de-circularised design is **build the vector from annotator A's labels, score with annotator B, and confirm it survives swapping A↔B.** The infrastructure to do this exists (`build_steering_arms.py` already builds per-annotator vector arms `R1-1.5B__qwen3-235b`, `R1-1.5B__nova-pro`), but the proposed Tier-C run does *not* cross them. **Recommendation: the single cheapest credibility upgrade is to run the headline on the Qwen3-built vectors scored by Sonnet AND the Sonnet-built vectors scored by Qwen3, and report only the effect that survives both.** This is nearly free on the build side (vectors are CPU, <1 min, already partly on disk) and converts a tautology-risk into a genuine cross-construct causal claim.

3. **The "non-builder annotator does not exist" blocker is real and is on the critical path.** Both review docs flag it; I am elevating it. `src/annotation.py` knows only the Sonnet id; the 3-way replication used *static* `data/annotated_*.json` files, not a live endpoint. **Until a live Qwen3/Nova proxy id is registered, the steering headline cannot be anything but "preliminary, single-annotator, band-ungated" — and a single-annotator steering result on a construct that annotator defined is, for an adversarial examiner, nearly worthless as causal evidence.** This one item gates whether Tier C is worth funding at all.

**Verdict:** the metric is valid; the outcome is causal *only* if scored by a non-builder annotator, and is *robustly* causal only if it survives the A-builds/B-scores swap. The current default (Sonnet builds, Sonnet scores) produces a number that should never appear in the thesis as a causal claim.

---

### O.2 Are the control/energy-matched arms sufficient to rule out generic norm-increase / fluency degradation? — For `single_direction`, yes. For the manifold (headline) arm, **no** — and this is the asymmetry the trackers correctly name but the *experiment* does not fund away.

The energy-matched machinery is genuinely good. `energy_matched_scale` (`steered_inference.py:188`) equalises injected energy `α·|vᵀh|` measured on the *same unperturbed stream* (`mode="measure"`, `:286`) — this is the right reference and is better than most published floors. The `repetition_rate` / `degenerate_rate` / `mean_n_tokens` damage axes (`evaluation.py:117-119`) catch the "steering reduced the behaviour by wrecking the model" failure cheaply, pre-annotation. The length-conditioned read (B7) catches length-driven suppression. Collectively these *do* rule out a generic norm-increase or fluency-collapse explanation **for the single direction.**

The hole, stated precisely:

- `energy_matched_random` is calibrated against **`single_direction` only** (`steered_inference.py:680-694`: `e_beh = measure_mean_abs_proj(..., vecs["single_direction"], ...)`). The manifold arms are matched only on **dimension** (`random_subspace_k`), not energy.
- So the **manifold-vs-single Pareto — explicitly named in `REVIEW_PHASE7_2026-06-25.md §5` as "the thesis's one causal result and the differentiator vs LRS" — is NOT energy-floored.** If `manifold_auto` beats `single_direction` on suppression-per-damage, an examiner can ask: is that because the PCA subspace is special, or because the projection-then-renormalise changed the injected energy? The `random_subspace_k` control answers "is the subspace special vs a random subspace" but does *not* answer "is this energy or geometry." The refinement doc flags this as open decision #3 and proposes `energy_matched_random_k{k}` — but marks it *if-funded*, and the Tier-C scope does not include it.

**This is a genuine design gap, not a documentation nit.** The differentiating result of the whole chapter rests on an arm that is not energy-floored. Two ways out:

- **(a) Build `energy_matched_random_k{k}` for the candidate k (auto only).** Cost is one extra arm × candidate behaviours; the calibration code already generalises (`measure_mean_abs_proj` takes any vector). This is the principled fix and I recommend it for the 1–2 candidate behaviours at minimum.
- **(b) Demote the manifold-vs-single Pareto to "descriptive, dimension-matched-not-energy-matched" and make the energy-floored single-direction Δ_floor the sole headline.** Cheaper, honest, but it surrenders the LRS differentiator. Given the manifold arm is the *novel* contribution, (b) guts the chapter. **Prefer (a).**

One subtle additional risk the docs miss: at `auto_k`, `cos(single, manifold) ≈ 0.95–0.97` (stated in `_build_arms` docstring). **If the manifold and single vectors are nearly collinear at the operating k, the manifold-vs-single comparison has almost no signal to detect — the two arms ARE nearly the same arm.** The k-sweep (`manifold_k1` very different, `manifold_k10`≈single) is where any separation lives, and `k=1` is the only arm that genuinely tests "does collapsing to the single dominant PC help." **The honest headline contrast is single vs `manifold_k1`, not single vs `manifold_auto`** — and that should be pre-registered, because at auto the experiment is under-powered *by construction*.

---

### O.3 Is single-vector vs manifold-k fairly powered? — No. Three compounding power problems, all under-stated.

1. **N is tiny and the resample unit makes it tinier.** 50 hold-out tasks, BCa over *tasks* as the resample unit (correct choice — samples pool within cell, `METHODOLOGY_REFINEMENT §2.6`). Effective N = 50, dropping to **~10–15 in the bake-off subset** and to whatever the candidate-behaviour restriction leaves. The pilot effects are small: uncertainty −0.114 (the *strong* one), backtracking −0.006 to −0.021. A paired BCa CI on a −0.02 effect at N=50 with Holm correction across an (arm × behaviour) family will **very plausibly straddle zero for everything except uncertainty.** The trackers admit "the clean win may rest on uncertainty-estimation" — I'd put it more bluntly: **on current effect sizes, backtracking is likely to come back inconclusive, and the manifold-vs-single delta (an effect-of-an-effect) is almost certainly under-powered at N=50.**

2. **The manifold advantage is a second-order effect being tested at first-order N.** You are asking "is (manifold suppression − floor) > (single suppression − floor)" — a *difference of differences*. The variance of a difference-of-differences is larger than either component. To detect a manifold-vs-single Pareto gap of plausibly ~0.02–0.05 fraction units, paired at the task level, you would want N in the low hundreds, not 50. **The n=3 samples at T=0.7 help variance but do not increase effective N (they pool within cell).** This is the single biggest reason the headline may land "directionally suggestive, not significant."

3. **adding-knowledge is a pre-registered null and example-testing nudges the *wrong way* at L27 (+0.011).** So of four behaviours, the experiment realistically has *one* clean positive (uncertainty), *one* maybe (backtracking, pending α-sweep), and *two* nulls/wrong-sign. A 1/4 or 2/4 causal result is publishable *if framed as specificity* (the nulls are the point — "the direction is behaviour-specific, it does not move the others"), but it is **not** a "steering works" headline, and the manifold-vs-single comparison effectively has n=1–2 behaviours to demonstrate itself on.

**Fair-powering recommendation:** do not spread the budget across four behaviours and eight α. Concentrate. Run the **two candidate behaviours (uncertainty, backtracking) at the sealed α\*, with the single vs `manifold_k1` contrast, energy-floored on both arms, n=3 T=0.7, on all 50 hold-out tasks.** Pre-register uncertainty as the confirmatory test and backtracking as exploratory. Report adding-knowledge + example-testing as generation-only specificity nulls (cheap, no annotation). This is the most power per dollar.

---

### O.4 Minimal defensible design + what to CUT

**The single result that would falsify the hypothesis** (state this in the pre-registration, before unblinding): *For uncertainty-estimation, the energy-matched Δ_floor — suppression by `single_direction` minus suppression by `energy_matched_random` at equal injected energy and equal injection schedule, scored by a NON-builder annotator — has a paired-BCa 95% CI that includes 0 (or fails to exceed the annotator noise band `band_b`).* If that CI includes zero, the central causal claim ("a named behaviour's direction causally suppresses that behaviour beyond a generic energy-matched perturbation") is **falsified for the strongest behaviour**, and the chapter must retreat to geometry-only. Uncertainty is the right falsification anchor because it is the *only* behaviour with a pilot effect large enough (−0.114) that a null would be genuinely informative rather than just under-powered.

**Trimmed arm list (per candidate behaviour, at sealed α\*, fixed L27):**

| Arm | Keep? | Why |
|---|---|---|
| `vanilla` (shared) | **KEEP** | reference for both suppressions |
| `single_direction` | **KEEP** | energy-floored headline arm |
| `manifold_k1` | **KEEP** | the *only* manifold arm meaningfully different from single (auto≈single, cos 0.96) |
| `manifold_auto`, `manifold_k{3,5,10}` | **CUT** from headline (gen-only if curiosity) | redundant with single at the operating point; burns annotation $ on near-duplicates |
| `energy_matched_random` | **KEEP** | the real floor for single_direction |
| `energy_matched_random_k1` | **BUILD + KEEP** | the missing energy floor for the manifold arm (O.2 fix) |
| `random_subspace_k1` | **KEEP (1 rep, not 3)** | dimension floor; 3 reps is over-spend at N=50 — average of 1–2 suffices for a floor |
| `orthogonal_complement` | **CUT** from headline | mechanism probe, not a causal-claim arm; run gen-only if Δ_floor>0 |
| `random_direction` | **CUT** | sanity floor only, ~19× under-energy; the energy floor subsumes it |
| `gated_*` (B2) | **CUT** (already conditional) | second-order; build only after a positive |

**Sample/behaviour scope:** 2 behaviours annotated (uncertainty + backtracking), 2 generation-only (adding-knowledge, example-testing as specificity nulls), n=3 / T=0.7, 50 tasks, α\* single dose (NOT the 8-point grid for the headline; generate the grid only for the descriptive Pareto if budget remains).

**Cost estimate (sanity-checked against `GPU_GUIDE` $0.055/chain annotation, ~2–3 min/gen RunPod 4090 @ $0.44/hr):**
- Annotated chains ≈ 2 behaviours × 4 headline arms × 50 tasks × 3 samples ≈ **1,200 chains** + shared vanilla (50×3=150) ≈ **~1,350 annotated chains** → **≈ $75–95 annotation.**
- Generation: ~1,350 + (2 gen-only behaviours × 4 arms × 50) ≈ ~1,750 gens × ~2.5 min ≈ **~73 GPU-hours ≈ $32** on a 4090 (parallelism shrinks wall-clock).
- **Total ≈ $110–130** — *below* the review's Tier-C $130–180, because cutting to k1+the redundant-manifold arms and dropping the 8-α grid removes the bulk. Add the A↔B cross-annotator swap (O.1) and it roughly doubles annotation to **~$160–180** — which I argue is the *correct* place to spend, not the manifold k-sweep.

**What to cut, ranked by $ saved with least credibility lost:** (1) the 8-point α grid for the headline (use sealed α\* — saves ~7×), (2) the redundant manifold arms k3/k5/k10/auto (saves ~4 arms), (3) random_subspace replicates 3→1, (4) orthogonal_complement and random_direction from the annotated set, (5) the gated arms (already deferred). **What NOT to cut even though it costs the most:** the non-builder annotator and the A↔B construct swap — these are the difference between a causal claim and a tautology.

---

### O.5 Prior-art inheritance — what to borrow and what to refuse

**Inherit:**
- **Energy-matched floor at equal injected energy (de-confound LRS itself lacks).** Already built; this is the strongest methodological card and must be the headline subtrahend. Keep.
- **Matched-pair transitions + McNemar exact (B3).** Cheap, label-free, runs on the annotated chains, and McNemar's exact (not χ²) is correct for N≈50. This is the right order-free corroborator of the BCa CI at small N. **Inherit.**
- **p_last/p_mean magnitude-vs-order decomposition** (predictive-geometry lane): correctly positions *away* from a raw-AUROC race (0.58 vs Sun et al. 0.87) onto the order-vs-magnitude question a static probe cannot produce. **Inherit the framing**, not the AUROC competition.
- **Suppression as primary polarity, amplification as descriptive sign-check only.** Correct; amplification has no clean floor here.

**Refuse:**
- **Gate-vs-gradient ablation as a headline arm (B2).** Correctly demoted to conditional. It is a second-order refinement of an effect not yet shown to exist; building it before a positive Δ_floor is premature. **Do not inherit into the core run.**
- **The LRS Transformer reward head / compute-matched best-of-N for a suppression endpoint.** Best-of-N has no coherent selection criterion for a *suppression* (non-accuracy) outcome — correctly dropped. **Refuse.**
- **Reward-gradient steering (LTO/Du, RISER, CREST/Zhang) as a method.** Keep the predicted direction strictly as a *probe* fenced in `src/predict/`, never in `_build_arms`. A behaviour-agnostic reward-gradient cannot answer the named-behaviour specificity question and would invite the AUROC race. **Refuse as a deployed method; allow only as a null-scored probe.**
- **`effect_quantile=0.8` matched-target-effect read off the eval curves.** This is a tune-on-eval leak (`steering_analysis.py:613`); the refinement correctly bypasses it. **Refuse — match post-hoc on the realized behaviour-arm effect.**

---

### O.6 Residual risks the trackers under-weight (the blind spots)

1. **CF-17's "fixed" status is generous.** The hold-out is fixed on the task axis, but the *layer* (CF-17 residual + CF-10 read-out proximity) is chosen with full-corpus info and a proxy (07d) that spikes near its own read-out layer. Every single-layer headline must carry this verbatim. The recap does flag it; the *ledger's* "FIXED" label could mislead a skim-reader into thinking the layer is held out. It is not.
2. **`auto`-collinearity (cos 0.96) silently neuters the manifold-vs-single test** — covered in O.2/O.3 but worth isolating: this is a *power* bug hiding in a *design* choice, and no tracker names it as the reason the comparison may come back null.
3. **15.8% of sentences are shorter than the pooling window** (pools into the next sentence; `clip_window` off by default). On a behaviour like backtracking whose markers are short, this could systematically blur the very spans being measured. Minor, but it touches the weakest candidate behaviour.
4. **Truncation (CF-8, 50.2% of corpus chains hit the 8192 cap)** flows into the steered chains. A steered chain that suppresses a behaviour *and* runs longer could hit the cap differently than vanilla, confounding length. The `mean_n_tokens` axis catches gross cases; a length-matched read (B7) is the right control and must actually be run, not just specified.
5. **The cluster is 94% contended at ~8.5 min/gen** (memory note + scripts hardcode `/home/tony`). The whole budget math assumes RunPod 4090. Confirm the run is on RunPod, not the cluster, or the wall-clock estimate is multi-week and the "fits in ~1 day" framing breaks.

---

### O.7 Bottom line

The apparatus is better than most published steering work: real energy floor, shared vanilla, skip-not-zero accounting, task hold-out, sealed dose, pre-registered nulls. The chapter fails or succeeds on three things the budget currently under-funds: **(1) a non-builder annotator that actually exists as a live endpoint — without it there is no causal claim, only a consistency check; (2) energy-flooring the manifold arm and contrasting single vs `manifold_k1` (not auto), or the novel result is both un-floored and under-powered by collinearity; (3) concentrating all N on uncertainty (confirmatory) + backtracking (exploratory) rather than spreading thin across four behaviours and eight α.** Do those three and a ~$110–180 run yields a defensible, appropriately-narrow causal result: *the uncertainty-estimation direction causally and specifically suppresses uncertainty beyond an energy-matched floor, scored by a non-builder annotator* — plus a behaviour-specificity story from the nulls. That is a real thesis chapter. The full 16-arm × 8-α grid is not; it spends ~$900 to dilute the one effect that survives.
