## N. Adversarial Review — The Layer Choice

*Scope: the single decision the project is parked on before ~$260 of RunPod + Bedrock spend — which layer(s) to build and steer at, and whether the evidence that points to L27 is trustworthy or circular. Everything below was verified against the on-disk result files and the code, not the digest. Where I disagree with `results/recap/section_I_layer_decision.md`, I say so.*

---

### N.0 Verdict in one paragraph

The leaning choice (L27) is **defensible but not de-risked**, and the strongest non-Huang evidence for it (07d) is *more* confounded than the digest admits — not less. The two confounds that matter are (1) **07d's late-layer KL inflation is only partially subtracted by its isotropic random null**, so 07d cannot distinguish "L27 is behaviour-specifically causal" from "perturbing the last block before the unembedding inflates output-KL for anything", and (2) **the pilot's "−63% on uncertainty" is measured against a vanilla baseline that itself moves ~±0.03 under re-annotation** — the same magnitude as smaller effects being celebrated. The cheapest de-risking experiment is NOT a new generation run; it is a **$0 re-analysis of the data already on disk** plus one **~$5–15 annotation-noise floor**. Recommended set: **build at BOTH L27 and a mid layer (L15, the true PR argmin — not L16), carry both as a pre-registered arm**, and let the floored headline arbitrate — do not pre-commit to L27 on 07d alone.

---

### N.1 What the evidence actually says (verified, with corrections to the digest)

**Probe (Phase 5c) — a null for *L27 specifically*, not just "flat".** I recomputed the per-behaviour argmaxes from `results/cross_layer/R1-1.5B/probe_accuracy.json`:

| Behaviour | probe min–max | argmax | L27 |
|---|---|---|---|
| backtracking | 0.700–0.763 | **L18** | 0.718 |
| uncertainty | 0.718–0.776 | **L27** | 0.776 |
| example-testing | 0.776–0.844 | **L18** | 0.812 |
| adding-knowledge | 0.794–0.843 | **L20** | 0.807 |

The triangulation code classifies all four as `flat` (CV < 0.03, `compute_layer_triangulation.py:109`), which is fair — but note that *when forced to argmax, the probe leans L18–L20 mid-late, and only uncertainty peaks at L27*. So the probe is not neutral on the mid-vs-late question; it weakly contradicts L27 for 3/4 behaviours. The digest treats the probe purely as "no signal"; it is in fact a soft mid-late signal that L27 fails for 3/4.

**PR (geometry) — clean MID signal that explicitly excludes L27 (digest is wrong here).** I recomputed from `results/pca/R1-1.5B/layer_profiles.json`:

| Behaviour | PR argmin | PR_min | PR_L16 | PR_L27 | plateau (≤min+1SD) |
|---|---|---|---|---|---|
| backtracking | **L15** | 12.20 | 12.55 | 22.58 | [10..19] |
| uncertainty | **L15** | 13.18 | 13.52 | 23.58 | [10..19] |
| example-testing | **L11** | 13.08 | 13.65 | 17.44 | [7..19] |
| adding-knowledge | **L15** | 14.34 | 15.27 | 25.91 | [7..19] |

Two corrections to `section_I`: (a) the argmin is **L15**, not L16 (a small thing, but the canonical mid candidate should be L15, and α\* is not sealed there either); (b) the digest claims "the PR-trough plateau spans half the network so the argmin is barely distinguished from the late layers." **That is false.** L27's PR (~22–26) is nearly *double* the trough (~12–14) and sits firmly *outside* the plateau ([7/10..19]) for all four behaviours. PR gives a clean, unambiguous mid-network answer and an unambiguous "**not L27**". It is descriptive, not causal — but it is not weak, and it disagrees with the pick. The honest framing is "PR and the probe both lean mid; the only thing that lands on L27 is the *intervention* family (07c, 07d) and Huang" — i.e. the two evidence families split exactly along descriptive-vs-causal, which is suspicious because the causal family is the one with the read-out confound.

**07c attribution patching — genuinely confounded, correctly quarantined.** `attribution_summary.md` shows all four behaviours ramping monotonically to L26/27 (backtracking 9.9→75; uncertainty 10→133) with no interior peak. The mechanism is exactly as documented: the metric is read at a fixed L27, attribution estimates ∂(L27 metric)/∂(act at ℓ), which is mechanically larger as ℓ→27 because less non-linear stack intervenes. A first-order gradient at a fixed late read-out **cannot** produce an interior peak. Quarantine is correct. Do not use.

---

### N.2 Does 07d actually fix 07c, or is it a dressed-up version of the same confound?

**07d removes the *fixed-read-out* term but not the *late-layer-sensitivity* term — and its null is too weak to subtract the residual.** This is the central technical finding of this review, and it is stronger than `section_I §I.6`'s three-caveat hedge.

07d's claim to de-confounding rests on two pillars: (a) the read-out is the OUTPUT, common to every ℓ, so there is no fixed-read-out proximity; (b) a per-layer norm-matched random null subtracts "any global sensitivity of a layer to ANY perturbation." Pillar (a) is real. **Pillar (b) is broken**, and the on-disk numbers show it.

I pulled the raw `behaviour_effect` and `random_null` components from `steering_effect_curves.json`:

| Behaviour | L11 beh / null / deconf | L16–18 beh / null / deconf | **L27 beh / null / deconf** |
|---|---|---|---|
| backtracking | 0.0227 / 0.0036 / 0.0191 | 0.024 / 0.006 / 0.018 | **0.0453 / 0.0159 / 0.0294** |
| uncertainty | 0.0242 / 0.0057 / 0.0186 | 0.031 / 0.007 / 0.024 | **0.0978 / 0.0140 / 0.0838** |
| example-testing | 0.0096 / 0.0059 / 0.0037 | 0.020 / 0.008 / 0.012 | **0.0709 / 0.0110 / 0.0600** |
| adding-knowledge | 0.0212 / 0.0062 / 0.0150 | 0.054 / 0.007 / 0.047 | **0.0998 / 0.0172 / 0.0827** |

Three things are fatal to reading 07d's L27 argmax as causal localisation:

1. **The raw behaviour_effect *itself* jumps 2–4× at L27 for all four behaviours** (uncertainty 0.031→0.098; example-testing 0.020→0.071; adding-knowledge 0.054→0.100). This is the classic signature of perturbing the residual stream immediately before the unembedding: a fixed-norm delta at L27 lands almost directly in logit space, so the output-KL is mechanically inflated *regardless of behaviour*. The "no fixed-read-out proximity term" argument is technically true and practically irrelevant — when ℓ = the last layer, the OUTPUT read-out *is* the proximate read-out.

2. **The random null only roughly doubles at L27 (e.g. uncertainty 0.007→0.014) while the behaviour effect quadruples**, so the de-confounded effect stays huge. Why does the null under-subtract? Because `random_unit_direction` (`src/layer_sweep.py:554`) draws an **isotropic** Gaussian direction. The residual stream is strongly anisotropic; the behaviour diff-of-means vector is aligned with high-variance (and at L27, near-logit) directions, while an isotropic random vector mostly points into low-variance null space. So at the layer where alignment-with-output matters most (L27), the behaviour vector gets the full inflation and the isotropic null does not — the null **structurally cannot** match the late-layer sensitivity it is supposed to subtract. With only **R=2** draws (`steering_effect_summary.md`) the estimate is also noisy. The correct null for this purpose is a **covariance-matched** random direction (or, better, the `energy_matched_random` arm the headline already builds), not isotropic.

3. **The L27 SEM is the largest of any layer for every behaviour** (uncertainty 0.084±0.014; backtracking 0.029±0.013) — the spike is donor-unstable, exactly what you expect from a few donors whose onset token happens to sit near a high-logit direction. n=12 donors, bootstrap notwithstanding.

The tell that 07d is reading inflation, not localisation: **adding-knowledge has a genuine interior peak at L16–18 (deconf 0.044–0.047) that rivals other behaviours' L27 values, yet L27 (0.083) still wins** — because the L27 inflation is *universal*. If 07d were localising behaviour-specific causality, adding-knowledge (a pre-registered specificity null, p=1.0 everywhere in the Holm table) should NOT light up at L27 at all; it lights up *most* of all. That is a contradiction only explicable as a layer artefact.

**Conclusion on 07d:** it is not a fix; it is the same read-out-proximity confound moved from "fixed late metric layer" to "the output is adjacent to L27", with a null too isotropic and too thin (R=2) to subtract it. The digest's "still a proxy with a residual L27 spike" *understates* this. 07d should be downgraded from "the single strongest non-Huang positive for L27" to "uninformative about L27 vs mid; weakly informative that backtracking has interior mass (11/18)." The only layer-discriminating thing 07d says that survives is its *one* non-L27 finding (backtracking shortlist 11/18), and even that is the only behaviour where the interior competes with the artefact.

---

### N.3 Is the pilot's "L27 confirmed" trustworthy?

Partly. I reproduced the pilot from `results/eval/R1-1.5B__{L27,L16}_trim/eval_summary.json` and found three problems the digest glosses.

**(a) The pilot is 10 tasks from TWO categories, not the stratified hold-out.** `eval_task_ids.json` shows the trim used `CAUS_*` ×5 + `CREA_*` ×5 ("first 10 of the hold-out split") — causal_reasoning and creative_problem_solving only. The real hold-out is 5/category × 10 categories. So the layer pick was confirmed on 20% of the category space. Backtracking/adding-knowledge behaviour may be category-dependent; the pilot cannot see that.

**(b) The vanilla baseline is not stable under re-annotation — and the "−63%" is measured against it.** The vanilla chains are byte-identical across the L16 and L27 trims (I verified: 10/10 identical). Yet the re-annotated vanilla *fraction* differs: uncertainty vanilla = **0.181** in the L27 trim vs **0.153** in the L16 trim — a 0.028 swing on identical text, purely from annotation stochasticity. The celebrated effect is uncertainty single-direction suppression = 0.114 (L27). So the annotation noise on the *baseline alone* is ~25% of the headline effect, and there is no noise band on the pilot at all (`annotator_model: null`, single Sonnet pass, `skip_annotation: true` at generation then a separate scoring pass). The pilot "confirms" L27 by a margin that is only ~4× its own un-characterised annotation noise, on n=10, single annotator (the builder). This is suggestive, not confirmatory.

**(c) The cleanliness guard does hold, and it is the pilot's real contribution.** Recomputed repetition rates: L27 keeps repetition ≤ vanilla (0.28) in most cells (uncertainty 0.20, adding-knowledge 0.08, backtracking 0.21) and shortens chains; **L16 inflates repetition badly** (uncertainty 0.50–0.59, example-testing 0.41–0.51, adding-knowledge 0.40–0.52) and *lengthens* chains by 300–1100 tokens. This is robust and not annotation-dependent (repetition is computed on the text). So the defensible pilot statement is narrow: **"L16 visibly damages the model at α=1; L27 does not"** — a damage result, not an on-target-efficacy result. It rules out *naive* L16, but it does not establish L27 is the causally correct layer; it establishes L27 is the *cleaner* layer at this dose. Those are different claims, and the chapter must not conflate them.

---

### N.4 Where the $260 gets wasted

1. **Spending it to "confirm L27" when the layer pick is the cheap step.** The floored headline needs new generation regardless; but committing the *whole* budget to a single layer chosen by 07d (a confound) risks discovering post-hoc that mid was right for adding-knowledge/example-testing. Carry both layers as an arm or you may buy a single-layer null you cannot re-point.
2. **The manifold arms are dimension-matched, not energy-matched** (`include_energy_matched` calibrates `single_direction` only). If the manifold-vs-single Pareto is a headline and any manifold arm lands on a different layer, its "floor" is not energy-comparable — money spent generating manifold floor chains that cannot support an energy-floored claim.
3. **α\* is L27-only and unvalidated.** `predictions_layer27.json` has `empirical: {}` — the sealed α\* (0.96–1.06) is a κ-based theoretical prediction never checked against realised effect. If any behaviour lands on mid, its dose is unsealed and re-prediction (or an α-sweep) is needed — generation at the wrong dose is wasted.
4. **Single-annotator headline.** Without the non-builder annotator id (still not located; `src/annotation.py` knows only Sonnet), every steered chain you generate is scored circularly. You can generate first and annotate later, but if you *annotate* with Sonnet during the run, that spend buys a "preliminary, band-ungated" result only — re-annotation later is a second spend.
5. **Generating example-testing / adding-knowledge at full n=3×floor-arms.** Both are pre-registered nulls/wrong-direction in the pilot. Generation-only (no multi-sample, no floor arms) for these two, as the refinement already says, or the budget evaporates on cells that cannot produce a positive.

---

### N.5 The cheapest experiment that de-risks the layer BEFORE the big spend

Two $0–$15 steps, in order, both runnable on what is already on disk:

**Step 1 — $0, re-run 07d's null correctly (no GPU, no generation).** The killer ambiguity is "is L27 behaviour-specific or universal output-KL inflation?" You can answer it *from the data already saved*. The per-donor `behaviour_effect`, `random_null`, `kl_effect` are all in `steering_effect_curves.json`. Re-derive the de-confounded curve with a **covariance-matched** null instead of the isotropic one: either (a) rescale the existing random null per layer so its mean |rᵀh| equals the behaviour arm's (the `measure_mean_abs_proj` / `energy_matched_scale` logic in `src/steered_inference.py` already does exactly this — port it), or (b) at minimum, report the **behaviour/null ratio per layer** (already computed in N.2: it is 2.85–6.98 at L27, indistinguishable from interior layers' 4–8 for some behaviours). If, after an energy-matched null, the L27 deconf effect collapses toward the interior, 07d's L27 argmax was an artefact — and you have proven it for free, before spending. This single re-analysis is worth more than any new generation.

**Step 2 — ~$5–15, an annotation-noise floor on the existing pilot chains.** Re-annotate the *existing* L27 trim chains a second time with Sonnet (and, if any non-builder id surfaces, with it) to measure `band_b` = fraction-RMS of |frac_passA − frac_passB| on identical text. The vanilla 0.181-vs-0.153 swing already tells you this band is ~0.03; pin it. Any layer "confirmation" whose margin does not clear this band is noise. This is the missing denominator on every pilot claim and costs a handful of dollars on chains you already generated.

Only after Steps 1–2 should generation dollars flow. If Step 1 collapses L27 and PR/probe both say mid, the honest pick may be **mid for the soft behaviours, L27 for uncertainty** — exactly the per-behaviour split the methodology fears but the data may force.

---

### N.6 Recommended layer set + coefficient strategy

**Layer set (pre-registered, do not collapse to one before the floor arbitrates):**
- **Primary build at L27** (Huang, same model; clean at α=1; uncertainty's probe argmax) AND **mid at L15** (true PR argmin; probe argmax for back/ex; Venhoff 15–18). L15, not L16 — fix the canonical mid candidate to the actual argmin.
- Carry **both** as a frozen arm in the headline for the two candidate behaviours (uncertainty, backtracking). This is +1 layer of generation on 2 behaviours, not 4 — affordable, and it is the only way to make the mid-vs-late choice *empirical at the floored endpoint* rather than inherited from a confounded proxy.
- Backtracking additionally gets L11/L18 in its shortlist if Step-1 re-analysis confirms interior mass survives the energy-matched null.
- example-testing / adding-knowledge: generation-only, L27, no floor arms, pre-registered nulls.

**Coefficient strategy:**
- Use sealed α\* (back 0.994 / unc 0.969 / ex 0.964 / add 1.056) **as a fixed dose at L27 only**, exactly as specified, and bypass `effect_quantile=0.8` (tune-on-eval leak, `steering_analysis.py:613`).
- For the **L15 arm, α\* is NOT sealed** — `predictions_layer27.json` is L27-only. Either re-run the saturation prediction at L15 or (cheaper, honest) run a *3-point* α∈{0.7, 1.0, 1.5} mini-sweep at L15 for the 2 candidate behaviours and report it as exploratory/unsealed. Do not silently reuse the L27 α\* at L15 — the residual norm at L15 (~13 PR concentration) differs from L27, so equal nominal α is not equal strength (07d's own `delta_frac` machinery exists precisely because of this).
- Match arms on **realised behaviour-arm effect**, not equal α, and never read the matched target off the eval curves.

---

### N.7 Go / No-Go checklist (must all be green before the Tier-C generation spend)

1. **[BLOCKER] Step-1 energy-matched-null re-analysis of 07d run, $0.** If L27 survives an energy-matched (not isotropic) null as a behaviour-specific peak → L27 is genuinely supported. If it collapses → DO NOT pre-commit to L27; carry mid as co-primary. *This gate flips the whole layer story and costs nothing.*
2. **[BLOCKER] Annotation-noise band `band_b` measured, ~$5–15.** No layer "confirmation" or headline Δ is reportable until its margin is compared to this band. The pilot's vanilla instability (0.181 vs 0.153) makes this non-optional.
3. **[BLOCKER] Non-builder annotator id located** (Qwen3-235B / Nova-Pro live endpoint) OR explicit acceptance that the headline is "preliminary, band-ungated." Without it the steered-chain scoring is circular (CF-7); `src/annotation.py:44` knows only Sonnet.
4. **Layer set frozen as {L27, L15} arm for ≤2 candidate behaviours**, generation-only at L27 for the 2 nulls. Holm family fixed in the pre-registration before any delta is unblinded.
5. **Energy-match scope disclosed:** single_direction energy-floored; manifold arms dimension-matched only. Do not claim manifold energy-floored unless `energy_matched_random_k{k}` is built.
6. **α\* sealed only at L27; L15 dose flagged unsealed/exploratory** (or re-predicted at L15).
7. **CF-17 disclosed verbatim on every single-layer headline:** the layer is held out on the task axis only, never the layer axis. The pilot arbiter reads at L27's own output-proximate read-out and cannot shed the late-layer-proximity confound.
8. **Scope = Tier-C (~2,400 chains, ~$130–180 + RunPod), tell-me-first.** Reject the full grid.

Bottom line: **L27 is the right *default* (Huang, clean, uncertainty-supported) but the data on disk does not yet justify treating it as the causally-localised layer — the one experiment that would justify it (an energy-matched re-null of 07d) is free and has not been run. Run that first; carry L15 as insurance; and put the $0.03 annotation noise band under every number before any of it counts.**
