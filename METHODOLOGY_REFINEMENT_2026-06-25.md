<!-- Generated 2026-06-25 from the LRS (arXiv:2606.00726) review via a 12-agent ground->draft->critique->synthesize workflow. -->
<!-- Non-destructive proposal: supersedes the METHODOLOGY.md §4-§5 [PROPOSED] placeholders once accepted. All file:line refs + sealed alpha* values verified against the repo. -->

# Refined Methodology — Phase-7 Steering + Predictive Geometry

*Lead methodologist's definitive specification. Supersedes the §4–§5 "Headline comparison [PROPOSED]" placeholders in `METHODOLOGY.md`. Pre-register against this file before any annotation spend.*

---

## 1. Refinement in one paragraph

We are asking a **different question, not chasing a different outcome**. The standing niche — *does a NAMED reasoning behaviour occupy its own residual subspace, separable from a generic energy-matched perturbation?* — is unchanged, and the standing geometry results (low-dim subspace survives controls; curvature a clean well-powered NEGATIVE / chain-artefact; 3-way Sonnet/Qwen3/Nova replication; behaviour-specificity MIXED 2/4 with adding-knowledge a pre-registered null) stand independently of any steering result and are **not** on the line here. What the LRS review (arXiv:2606.00726) changes is the *scoring discipline*, not the apparatus: the headline on-target statistic moves from **arm-vs-vanilla** to **behaviour-arm-minus-energy-matched-random at equal injected energy AND equal injection schedule** (B1) — the de-confound LRS itself lacks — and the dose moves from an eval-tuned grid to the **sealed per-behaviour α\*** already on disk (B4). Everything needed for this already exists in `src/steered_inference.py` and `src/steering_analysis.py` (verified: `method_curve` already accepts `vanilla_method`; the `energy_matched_random` arm + `measure_mean_abs_proj` calibration; BCa + Holm; the multi-sample path), so the core is *one comparator function and an invocation discipline*, not new code. We keep fixed per-behaviour directions as **probes** (reward-gradient steering stays in the predictive-geometry lane, Rung-3, never `_build_arms`), keep suppression as the primary polarity, and keep every existing control (matched-effect-not-equal-α, CF-2 chain-grouped nulls, missing/empty re-annotation skipped-not-zeroed, row-provenance hold-out). Applying the adversarial critique, we **drop** the over-scoped second tier the feasibility lens flagged (the LRS Transformer reward head, compute-matched best-of-N for a suppression endpoint, n=5 across all four behaviours), **demote** the gate-vs-gradient ablation (B2) from "mandatory headline arm" to a conditional follow-up, **reconcile** the layer-pick statistic (it is honestly *vanilla-relative* on the existing chains — the floored version requires a small new generation, disclosed), and **fix** the noise-band scale error (fraction-RMS, never κ-derived) and the over-conjunctive acceptance rule (one primary gate; the rest are reported-not-vetoing).

---

## 2. The refined Phase-7 protocol (end to end)

**Niche header (governs everything):** the endpoint is *behaviour specificity against a floor*, not *task performance*. Suppression is the primary polarity. Fixed per-behaviour directions are probes.

### 2.1 Primary endpoint
`behaviour_fraction(chain, b)` = sentence-fraction of re-annotated spans labelled *b* (`src/evaluation.py:28`). Unchanged definition; token/sentence-fraction, never wall-clock. Missing/empty re-annotations remain `n_missing`/`n_empty` (skipped, never scored 0.0) — this is the substrate for the "Unresolved" transition class (§2.7) and must not regress.

### 2.2 Headline comparator (the core change, B1)
For each behaviour *b*, arm *m* ∈ {`single_direction`, `manifold_auto`}, at the sealed α\* and fixed layer:

```
suppression_X = vanilla_fraction − steered_fraction_X         (X measured at SAME α*, SAME layer, SAME schedule)
Δ_floor(b, m) = suppression_behaviour_arm(m) − suppression_energy_matched_random
```

Vanilla is retained **only** as the shared reference both suppressions are measured against — it is *not* the headline subtrahend. The arm counts iff `Δ_floor > 0`, paired-BCa CI excludes 0 after Holm, **and** Δ_floor exceeds the annotator-noise band (§2.6).

**Energy-match scope correction (critique blocker).** The `energy_matched_random` calibration (`measure_mean_abs_proj`, scale = E_single/E_random) matches mean |vᵀh| against **single_direction only**. Therefore:
- `single_direction` is the **energy-floored** headline arm (Δ_floor as above).
- `manifold_auto` / `manifold_k{k}` are **dimension-matched** against `random_subspace_k{k}` (Haar k-dim projector, renormalised identically), **not** energy-matched. Report the manifold arms against `random_subspace_k` with the stated limitation "dimension-matched, not energy-matched." Do **not** claim the manifold headline is energy-floored. *(Optional, if-funded: build `energy_matched_random_k{k}` calibrated so mean |Pᵀh| over the k-dim projector equals the manifold arm's; only then is the manifold headline energy-floored.)*

**Equal-schedule clause (pinned, not assumed).** All non-gated arms steer **every position incl. prefill** (current `_hook_fn` behaviour). Assert this and record it in `provenance.json` so the behaviour arm and its floor are byte-identical in schedule. This makes "equal energy AND equal schedule" a verifiable property of the run.

### 2.3 Arm taxonomy

| Arm | Role | Status |
|---|---|---|
| `vanilla` (shared) | reference fraction | exists |
| `single_direction` | Venhoff diff-of-means PROBE → **energy-floored headline** | exists |
| `manifold_auto` (+ `manifold_k{1,3,5,10}`) | subspace PROBE → **dimension-floored headline** | exists |
| `energy_matched_random` | **energy floor (single_direction only)** | exists, re-pointed |
| `random_subspace_k{k}` | **dimension floor (manifold arms)** | exists |
| `orthogonal_complement` | "discarded component = pure collateral?" | exists |
| `random_direction` | SANITY floor (~19× under-energy) — NEVER the causal baseline | exists, demoted |
| `gated_energy_matched_random` | **B2 gate floor** | **conditional follow-up** |
| `confidence_gate_only` | **B2 zero-injection** | **conditional follow-up** |

**Gate-vs-gradient ablation (B2) — demoted to conditional, with mechanics fixed.** The critique (de-confounding + feasibility lenses) is decisive: these arms (a) require **new generation** at ~8.5 min/gen and (b) are a second-order refinement of an effect not yet shown to exist. **Build B2 only after the energy-matched headline shows Δ_floor > 0 on ≥1 behaviour, and only if generation budget remains.** When built, fix the gate mechanics: the gate is a **frozen position schedule** computed once on the vanilla stream and applied byte-identically to *both* gated arms — NOT a per-step confidence read on the steered stream (which diverges ~0.92 from vanilla, so a vanilla-derived per-step gate would not correspond to steered decoding states). By construction `confidence_gate_only` then equals vanilla (the intended floor); `gated_energy_matched_random` is the "any gated nudge" floor the behaviour arm must beat. Implementation: optional `gate_mask` (frozen index set) in `_hook_fn` (`steered_inference.py:272`), two arms behind `include_gated` / `--no-gated` in `07_evaluate_steering.py`.

### 2.4 Polarity
**Suppression (`mode="subtract"`) is the pre-registered PRIMARY.** Amplification (`mode="add"`) is a **secondary descriptive sign-consistency check only** — same direction, same sealed α\*, report behaviours only, labelled "axis-reality check, not a controllability claim." It does NOT enter any headline or Holm family. If reported, either floor it against an energy-matched add-mode arm or carry the explicit un-floored caveat.

### 2.5 Dose: sealed α\* (B4)
Read per-behaviour α\* verbatim from `results/saturation_predictions/R1-1.5B/predictions_layer27.json` (backtracking 0.994 / uncertainty 0.969 / example-testing 0.964 / adding-knowledge 1.056) as the **single operating point**. Do **not** tune on the eval grid. Critically, **bypass the `effect_quantile=0.8` step** in `compare_across_behaviours` (`steering_analysis.py:613`, verified) — that reads the matched target effect off the eval curves' overlap, the exact tune-on-eval leak. Per the feasibility lens, do **not** derive a "sealed" matched target-effect from the unvalidated saturation curve (`empirical:{}` is empty, verified): use α\* as a fixed *dose*, and match on the **realized behaviour-arm effect at that fixed α\*** (symmetric across arms, read post-hoc), disclosed as "effect matched at the realized behaviour-arm effect." The grid is generated only for the descriptive Pareto curve. *Caveat: α\* is L27-only — if the layer lands on L16 for any behaviour, that behaviour's dose is unsealed and must be flagged (or re-predicted at L16).* The saturation r>0.5 cross-behaviour test stays SECONDARY (α\* clustered 0.96–1.06 ⇒ low power).

### 2.6 Variance + annotator de-circularisation (B5, B6)
- **Seeds + CIs:** run the headline at `--n-samples 3 --temperature 0.7 --sample-seed-base <fixed>` (critique: 3 not 5 — 3 still yields a BCa CI at this N; 5× is over-engineering against the budget). Per-cell BCa over **tasks** as the resample unit (samples pooled within cell; effective-N = task count). Vanilla and floors sampled the same N (N-vs-N; the multi-sample guard rejects N>1 at greedy as variance-fraud). Restrict the multi-sample headline to the **two candidate behaviours** (backtracking, uncertainty); run example-testing / adding-knowledge generation-only. Write the gen-hour + dollar estimate into the pre-registration as a tell-me-first gate.
- **Non-builder annotator (HARD prerequisite, not a footnote):** the headline `behaviour_fraction` must be scored by a **non-builder** annotator via `--annotator-model` (wired end-to-end, `07_evaluate_steering.py:118,244-255`). **Blocker, elevated:** no concrete Qwen3-235B / Nova-Pro proxy id exists in any live-call path (verified: `src/annotation.py:44` knows only the Sonnet id; the 3-way files are static `data/annotated_*.json`). **Locating + registering that id is on the critical path for any *pass* verdict.** A single-annotator (Sonnet) run may be reported only as **"preliminary, band-ungated"** — never as "passing." Add the id to `configs/config.yaml` as `annotation_model_alt`.
- **Noise-band acceptance gate (scale fixed):** define `band_b` as the **per-behaviour fraction-scale RMS of |frac_sonnet − frac_nonbuilder|** computed on the *same re-annotated steered chains*. **Never derive the band from κ** (κ is chance-corrected span-label agreement — wrong scale; cite κ only as *why* a band is needed). A behaviour passes iff `Δ_floor` clears `band_b` AND the Holm-corrected paired-BCa CI excludes 0. Pre-register the band before unblinding deltas. *Cheap runnable fallback while the non-builder id is missing: derive a PROXY band from the existing static 3-way corpus files (Sonnet-vs-Qwen3 per-chain fraction disagreement), stated explicitly as a corpus-chain lower bound, not a steered-chain band.*

### 2.7 Matched-pair + McNemar (B3)
Add `matched_pair_transitions()` beside `paired_bootstrap_matched_effect` (`steering_analysis.py:343`), reusing the `{task_id → curve}` pairing in `per_task_curves` (`:440`). Per task, classify (behaviour-arm vs its floor) into **{Improved, Degraded, Preserved, Unresolved}** on the on-target fraction (Unresolved = either side missing/empty, reusing `n_missing`/`n_empty`). Then **McNemar's exact test** on the discordant Improved/Degraded cells (exact, not χ², for the ~10–50-task hold-out) + a **sign test** as the order-free corroborator of the BCa CI. Label-free, runs on the bake-off chains the moment they are annotated.

### 2.8 Statistical contract
- **Matched-EFFECT, never equal-α** (inherited verbatim; `collateral_at_matched_effect`, paired BCa, task = resample unit).
- **One primary gate** (critique fix for over-conjunction): `Δ_floor` vs `energy_matched_random`, Holm-corrected across the family, must exceed `band_b`. The gated floor (B2) and any length-conditioned read are **reported, not independently vetoing** — an under-powered secondary failing is "inconclusive," never an overturn of the primary. This prevents manufacturing a null on backtracking/uncertainty (the 2/4 specificity survivors) by stacking AND-gates against wide CIs.
- **Holm family frozen in the pre-registration:** the (arm × behaviour) grid of the ONE primary statistic. Decide *before* running whether both `single_direction` and `manifold_auto` are headline arms; do not choose family size post-hoc on "eligibility."
- **Length-conditioned read (B7, descriptive):** a suppression driven by *shorter/degenerate* chains is a length artefact. Report `Δ_floor` at matched `mean_n_tokens` between arm and floor (or regress fraction-drop on token-count, report residual). Token-fraction controls per-chain density; this catches the cross-arm length difference. **Drop the compute-matched best-of-N vanilla floor** (S6) from the suppression headline — best-of-N has no coherent selection criterion for a suppression (not accuracy) endpoint, and the energy floor already controls "effort." It returns only if an amplification/recovery accuracy claim is ever made.
- **Selectivity = token-fraction, never wall-clock** (the cluster is contended ~8.5 min/gen; wall-clock confounds with load).

### 2.9 Layer-decision rule (L16 vs L27) — reconciled, honest
The layer is the *first, cheapest* decision and conditions every single-layer claim, so its statistic must be stated without overclaim.

- **Free signals lean L27 but cannot settle it:** the 07d de-confounded sweep reads out near L27 (late-layer proximity); "low repetition / long chains at L27" cannot distinguish *surgical* from *weak*; the $0 trigram-divergence check (both layers ≈0.92 from vanilla) confirms L27 is a *large clean* perturbation (kills "L27 is a no-op") but **saturates at 0.92 and cannot confirm on-target**.
- **The arbiter statistic is VANILLA-relative, disclosed (critique blocker reconciled).** The existing bake-off chains contain **only** `vanilla`/`single_direction`/`manifold_auto` at α∈{0,1} (verified) — **no `energy_matched_random` arm**. So the cheap, no-new-GPU layer pick is **vanilla-relative annotated suppression**, explicitly flagged as a *coarser, non-floored, L27-proximity-confounded* arbiter. It does **not** claim to be the de-confounded Δ_floor. Two axes of the draft contradicted each other on this; the reconciled story is: *layer pick = vanilla-relative (coarse); floored Δ_floor = full run only.*
- **Decision rule (pre-registered):** annotate the **leaning layer (L27) first** with the non-builder annotator (feasibility: do NOT annotate both layers up front on a 10-task pilot). Compute `vanilla_fraction − steered_fraction` (single_direction, α=1) per behaviour. **L27 is confirmed iff** it suppresses the named behaviour AND stays clean (repetition ≤ vanilla + a pre-registered margin) — operationalising "surgical, not weak." Annotate **L16 only if L27 fails** that guard; otherwise tie-break to L27 (Huang, published, same model) without spending on L16.
- **If per-behaviour orderings disagree** (plausible given 07d mid-peaks back-L11/unc-L16/ex-L19/add-L16), do **not** majority-vote a global layer; fall back to the per-behaviour mid/late build (`--vectors-dir`, `STEERING_LAYERS`) and flag "needs the full 50-task run to settle" — the 10-task pilot is under-powered for a 4-way split.
- **Standing caveat (CF-17):** the layer is held out on the *task* axis only (`eval_task_ids.json`), never on the *layer* axis. The annotated arbiter makes the mid-vs-late choice empirical, but a vanilla-relative arbiter at L27's own read-out layer does **not** remove the L27-proximity confound — disclose this on every single-layer headline.

### 2.10 Reporting contract (one table, per behaviour)
`suppression_behaviour`, `suppression_floor`, `Δ_floor` with paired-BCa CI (Holm across the frozen family), pass/fail vs `band_b` (or "preliminary, band-ungated" if single-annotator), the matched-pair transition table + McNemar/sign p, the length-conditioned read, and the secondary amplification sign-check. Pre-register **adding-knowledge as a NEGATIVE**; report under-powered cells as inconclusive, never widened into positives.

---

## 3. The refined predictive-geometry protocol

**Mechanism-first framing (P0).** The load-bearing result is the **step-shuffle null never rejects** (p≈0.72–0.77) ⇒ the correctness signal lives in residual **magnitude** (whole-chain predictability), not **order**. This is the niche and the **pre-registered PRIMARY read-out**; AUROC is secondary/descriptive. Positioning:
- **vs Sun et al. 2604.05655** (static probe, AUROC 0.87): they answer *can correctness be detected?* We answer *which component carries it, and is it order-dependent?* **Do NOT race on raw AUROC** (0.58 vs 0.87 — we lose); we win on the order-vs-magnitude decomposition a static probe cannot produce.
- **vs Du LTO 2509.26314** (latent reward model that steers): we keep the predicted direction as a **probe** (Rung-3, out of core Phase-7); any head is scored only under nulls, never deployed as a performance method.

**P1 — p_last / p_mean diagnostic (the in-budget upgrade, B8).** Today `RESIDUAL_FEATURE_NAMES` is all whole-chain aggregates, so the magnitude finding cannot be localised. Add `resid_last` (norm at the final retained step) and `resid_last_minus_mean` to `chain_residual_features` (`src/predict/predictor.py:153`), extending the tuple to 7. Because these flow into `grouped_auc` + both nulls unchanged, run three per-layer probes — **p_mean-only**, **p_last-only**, **full** — each under `label_permutation_null` and `step_shuffle_null`. This localises whether the magnitude signal is *end-of-trace* (the "does the conclusion land" intuition) or *chain-wide*, and sharpens the magnitude-not-order claim at **zero generation cost**. *p_last definition: residual norm at the final retained step is free now (inherits the sparse-subsequence + CF-6 mean-pooled caveat); the true-last-token version needs the un-pooled GPU re-extraction.*

**LRS-style reward head rung (B8) — DEFERRED to future / if-funded.** Per the feasibility and niche lenses: the head (LayerNorm-on-raw + sinusoidal PE + 2 blocks + sigmoid, BCE) is **L effort**, its validating **token-NLL floor needs an unrun GPU logit re-extraction** (CF-6), and a BCE-on-judge-label head is a *correctness-detection* object that invites the 0.58-vs-0.87 race P0 forbids. The methodology **specifies** it (if built: chain-grouped GroupKFold OOF-pooled AUROC via `grouped_auc`, scored under **both** nulls, with the step-shuffle null pre-registered **two-sided** — survival = corroboration, collapse = a real partial contradiction reported as such, never pre-framed as confirmatory; must beat curvature/persistence/length **and** the NLL floor) but marks it **out of scope for this thesis pass**.

**Reward-gradient as Rung-3 (B9) — out of core Phase-7.** The predicted next-step direction `f(x_t)` can be injected via `SteeredModel` exactly like a behaviour vector, but it lives **in `src/predict/`** and **must not touch `_build_arms`** (`steered_inference.py:400`). Framed as confirmation, not novelty (H4); gated on Rungs 1–2 + a real correctness hold-out. Behaviour-agnostic ⇒ it cannot answer the named-behaviour specificity question and must never be presented as if it does. Specified-but-unbuilt; likely future work.

**De-circularising the judge (B6, lane-local, cheap).** Add `--annotator-model` to `14_label_correctness.py` (library `generate_correctness_labels(model=)` already supports it; only the CLI flag is missing) so a non-builder judge can break judge↔head circularity. Pre-register that any AUROC delta must exceed a **judge-disagreement band** (re-judge the pilot with the non-builder, fraction/agreement scale). Until then the 0.58–0.59 AUROC is **single-judge, provisional**.

**Carry-overs (do-not-regress):** residual-only geometry read-out (anti-collapse stays in the LOSS); chain-grouped GroupKFold everywhere; step-shuffle null reused verbatim as the keystone; uncertain (`None`) verdicts excluded not coerced; train-on-all-986 vs train-on-labelled-183 integrity (the inflated 0.61 pilot stays DO-NOT-CITE); 183-chain pilot p in the fragile 0.024–0.034 band reported honestly; mean-pooled (CF-6) caveat on every result.

---

## 4. Change table (merged, de-duped, ranked by payoff/effort)

| # | Item | Current | Refined | Why (LRS) | Attaches at | Effort |
|---|---|---|---|---|---|---|
| 1 | Headline on-target statistic | `on_target_effect = vanilla − steered` (arm-vs-vanilla) | `Δ_floor = suppression_arm − suppression_floor` (single_direction vs `energy_matched_random`; manifold vs `random_subspace_k`); vanilla = shared reference only | B1: the de-confound LRS lacks | `src/steering_analysis.py:55,115` (`vanilla_method` already present), `compare_across_behaviours:569`; arm `steered_inference.py:483` | **S** |
| 2 | Dose | `--alpha-values` grid; `effect_quantile=0.8` reads target off eval | Sealed α\* fixed dose; match on realized behaviour-arm effect (not from the unvalidated saturation curve); bypass `effect_quantile` | B4: never tune on eval | `predictions_layer27.json` → `07_evaluate_steering.py:67`; `steering_analysis.py:613` | **S** |
| 3 | Equal-schedule clause | implicit "every position"; not asserted/recorded | All non-gated arms steer every position incl. prefill; asserted + logged to `provenance.json` | B1: equal energy AND schedule | `steered_inference.py:_hook_fn:272` + provenance write | **S** |
| 4 | p_last / p_mean diagnostic | all-whole-chain residual features | add `resid_last`, `resid_last_minus_mean`; 3 per-layer probes under both nulls | B8: localise magnitude signal; free | `src/predict/predictor.py:48,153`; `15_predict_gate.py:67` | **S–M** |
| 5 | `--annotator-model` on judge | CLI pins Sonnet | add flag, thread through; pre-register judge-disagreement band | B6: break judge circularity | `14_label_correctness.py:44`; `labels.py:220` | **S** |
| 6 | Variance | n=1 greedy | `--n-samples 3 --temperature 0.7`, candidate behaviours only; per-cell BCa | B5: seeds + CIs | `07_evaluate_steering.py:96-111`; `steering_analysis.py:305` | **S** |
| 7 | Non-builder annotator + noise band | builder (Sonnet) default; no band; **no non-builder id exists** | non-builder = headline annotator (HARD prereq for a *pass*); `band_b` = fraction-RMS of \|frac_A−frac_B\|, **never κ-derived** | B6 | `07_evaluate_steering.py:118,244`; `configs/config.yaml:195` (`annotation_model_alt`); `cross_annotator_comparison.md` | **M** |
| 8 | Matched-pair + McNemar | paired BCa of damage only | `{Improved/Degraded/Preserved/Unresolved}` + McNemar exact + sign test (arm vs floor) | B3: paired transitions, label-free | new fn in `steering_analysis.py` beside `:343`/`:440` | **M** |
| 9 | Layer rule | unrun; vanilla-relative leans L27 (confounded) | annotate L27 first (non-builder), vanilla-relative + "surgical-not-weak" guard, L16 only if L27 fails; disclosed coarse/non-floored | B6 + niche | `results/eval/R1-1.5B__{L16,L27}_trim` → `evaluation.py:aggregate_results`; `06_build_steering.py:STEERING_LAYERS` | **M** |
| 10 | Length-conditioned read | damage axes exist, not conditioned | report Δ_floor at matched `mean_n_tokens` (or regress-out token count) | B7: catch length-driven suppression | `evaluation.py:185`; `steering_analysis.py:48` | **S** |
| 11 | Holm family + one-primary-gate | Holm over 4 behaviours of one stat | freeze (arm×behaviour) family pre-reg; **one primary gate**, B2/length reported-not-vetoing | LRS hygiene + power honesty | `steering_analysis.py:523,569` | **S** |
| 12 | Gate-vs-gradient arms (B2) | absent | **conditional follow-up**: build only after Δ_floor>0 on ≥1 behaviour + budget; gate = frozen vanilla-derived schedule shared across both gated arms | B2: gate-not-gradient ablation | `steered_inference.py:_build_arms`, `_hook_fn:272` (`gate_mask`); `07_evaluate_steering.py` | **M (deferred)** |
| 13 | LRS reward head (B8) | absent | **out of scope this pass**; spec'd: GroupKFold OOF AUROC, both nulls (step-shuffle two-sided), beats NLL floor | B8 | `src/predict/jepa.py:136` (drop-in) | **L (deferred)** |
| 14 | Reward-gradient steering (Rung-3, B9) | doc commitment only | structurally fenced into `src/predict/`; **never** `_build_arms` | B9: probe-not-method | `src/predict/__init__.py:11`; `steered_inference.py:232` | **M (future)** |
| 15 | Compute-matched best-of-N (S6) | proposed for suppression headline | **dropped** from suppression endpoint (no coherent selection); returns only for accuracy/recovery claims | B7 (correctly scoped) | n/a | **—** |

---

## 5. What explicitly does NOT change

- **The niche:** "does a NAMED behaviour occupy its own residual subspace vs a noise floor?" — NOT LRS's accuracy question. This is the literal pre-registration header.
- **Standing geometry results:** low-dim subspace survives controls; curvature a clean well-powered NEGATIVE / chain-artefact; 3-way Sonnet/Qwen3/Nova replication; specificity MIXED 2/4 with adding-knowledge a pre-registered NEGATIVE (p=1.0). Independent of any steering outcome; not scooped by LRS.
- **Existing discipline, inherited verbatim:** matched-EFFECT (never equal-α) comparison; CF-2 chain-grouped / chain-stratified nulls + Phipson–Smyth smoothing as the keystone; BCa (task = resample unit) + Holm; missing/empty re-annotation skipped-not-zeroed (`evaluation.py:64-92`); row-provenance task hold-out (`stratified_eval_split` + `eval_task_ids.json`); `random_direction` as labelled sanity floor (never the causal baseline); fixed per-behaviour directions as probes; token-fraction never wall-clock.

---

## 6. Cheapest-sufficient execution path

**Tier A — $0, code-only, no new generation, no annotation (do now):**
1. Re-point the headline comparator (change #1) + bypass `effect_quantile`, wire sealed α\* (change #2) + assert/record equal schedule (change #3). *One comparator function + config; verified `vanilla_method` and `effect_quantile` already in code.*
2. Add `matched_pair_transitions()` + McNemar/sign (change #8) and the length-conditioned read (change #10) — pure analysis, run when chains are annotated.
3. Predictive geometry: p_last/p_mean split (change #4) + `--annotator-model` on `14_` (change #5). Re-run scripts 15/16 on the **existing 183-chain pilot** — free, sharpens the magnitude-not-order claim.
4. Write `results/eval/<run>/preregistration.json`: primary endpoint, sealed α\*, frozen Holm family, one-primary-gate acceptance rule, equal-schedule clause, under-power note. Fill `empirical:{}` only *after* the run.

**Tier B — creds/$-gated annotation (tell-me-first; no new GPU):**
5. **Locate + register the non-builder proxy id** (Qwen3-235B) in `config.yaml` — *critical path for any pass verdict.* Without it, only "preliminary, band-ungated" is reportable.
6. Annotate the **existing L27 bake-off chains** with the non-builder annotator → vanilla-relative layer arbiter + "surgical-not-weak" guard. Annotate L16 only if L27 fails.

**Tier C — $-gated, new generation (tell-me-first, costed in pre-reg):**
7. Headline run: 2 candidate behaviours × {single_direction, energy_matched_random / random_subspace_k, vanilla} × n=3 × T=0.7, at sealed α\*, fixed layer. *This requires new generation — the bake-off chains have no floor arm and no α\* dose (verified).* Re-annotate (non-builder), compute `band_b`, gate, report.
8. (If Δ_floor>0 on ≥1 behaviour and budget remains) B2 gated arms.

**The one decision still owed to the PI before Tier C:** whether to fund the **new generation run** that the floored headline requires (the layer pick in Tier B is genuinely $0-GPU; the headline is not — this distinction was blurred in the draft and must be surfaced to the budget owner).

---

## 7. Open decisions for the PI (genuine forks)

1. **Non-builder annotator id (go/no-go, critical path).** Is a Qwen3-235B (or Nova-Pro) id reachable on the lab Bedrock proxy for *live* calls on new steered chains? If no, the headline can only be "preliminary, band-ungated" — accept that, or block the pass verdict until the id is found?
2. **Fund the floored headline's new generation?** The layer pick is $0-GPU and vanilla-relative; the de-confounded Δ_floor headline needs a fresh run with the `energy_matched_random` arm + sealed-α\* dose. Fund it, or ship the layer pick + the geometry results and defer the steering headline?
3. **Manifold-arm floor.** Accept the manifold headline as *dimension-matched only* (against `random_subspace_k`, cheaper, honestly caveated), or fund `energy_matched_random_k{k}` so the manifold arm is *energy-floored* like single_direction?
4. **L16 dose if the layer splits.** If the arbiter selects L16 for any behaviour, re-predict α\* at L16, or report that behaviour's L16 dose as exploratory/unsealed?
5. **Per-behaviour vs single global layer.** If the 10-task pilot shows disagreeing orderings (07d mid-peaks vs Huang L27), commit to a per-behaviour mid/late build now, or defer to the full 50-task run?
6. **Predictive-geometry reward head + Rung-3.** Confirm these stay **out of scope** for this thesis pass (recommended: yes — keep P1 p_last/p_mean + the magnitude-not-order framing as the in-budget contribution), or fund the GPU logit re-extraction that the NLL floor and a defensible reward head require?
