# PROJECT RECAP — The Geometry of Machine Reasoning

**Compiled:** 2026-06-27 · **Purpose:** a faithful, stage-by-stage refresher of the whole project, written specifically to inform the *next* decision: **which layer to steer at, and the steering methodology, before committing ~$130–260 of API + GPU spend on Phase 7.**

This document was produced by a multi-agent pass: 13 readers each deep-read one pipeline stage and wrote a section (A–M) with real code snippets and a critique; two adversarial reviewers then stress-tested the layer choice (N) and the steering methodology (O). The sections below are those agents' output, lightly framed. **Read the executive synthesis here first; sections A–O are the evidence.**

---

## 0. Executive synthesis

### 0.1 What the project is

A computational-interpretability thesis arguing **reasoning is a geometric process**: that distinct reasoning *behaviours* in a chain-of-thought occupy a **low-dimensional, behaviour-specific subspace** of a reasoning model's residual stream, and that this structure is *causal* (you can steer the behaviour by moving along the subspace). The model under study is **DeepSeek-R1-Distill-Qwen-1.5B** (28 layers, hidden dim 1536). Four behaviours are tracked: **backtracking, uncertainty-estimation, example-testing, adding-knowledge**. The thesis has two "movements" — **Structure** (the geometry, Movement 1) and **Origin/Safety** (post-training as the hinge, Movement 2). Several extensions branch off (predictive geometry, safety-spillover, CBS/trajectory).

### 0.2 The pipeline in stages (and what is RUN vs UNRUN)

| Stage | Scripts | What | Status |
|---|---|---|---|
| **A** Theory & taxonomy | THEORY.md | The geometric claim, the four behaviours, the rung ladder | Settled framing |
| **B** Data generation | 00, 01, 02, 02b | Tasks → CoT chains (R1-1.5B) + baseline/control chains | RUN |
| **C** Annotation | 03, src/annotation | Span-level behaviour labels via Bedrock proxy (Sonnet) | RUN (κ≈0.35–0.44) |
| **D** Activations & pooling | 04, src/activation_extraction | Residual-stream capture, span pooling = **mean** (Venhoff) | RUN |
| **E** Geometry / Gate-0 | 05, 05b/c/d | Low-dim subspace ✔ · curvature ✘ (chain artefact) · specificity 2/4 | RUN — **citable** |
| **F** Replication R2.2 | robustness_geometry, 13 | Geometry replicates across Sonnet/Qwen3/Nova | RUN — **citable** |
| **G** Steering build (Phase 6) | 06, 06b, build_steering_arms | Diff-of-means vectors: `single` vs `manifold_k{1..10,auto}`, hold-out | BUILT |
| **H** Steering eval (Phase 7) | 07, src/steered_inference | Inject vector → generate → annotate → behaviour-fraction; 14 arms | **BUILT, UNRUN** (only $0 smoke) |
| **I** Layer & method decision | 07b/c/d, triangulation | The current fork — which layer, which arms | **DECISION PENDING** |
| **J** CBS / trajectory | 08–13 | Cohort scoring + trajectory geometry | **Parked/deferred** |
| **K** Predictive geometry | 14–17 (current branch) | JEPA next-step predictor, residual geometry vs correctness | Pilot RUN (weak) |
| **L** Safety post-training | pt01–03 | LoRA safety SFT → measure reasoning-geometry spillover | BUILT, UNRUN |
| **M** Confounds & ledger | CONFOUNDS, RESULTS_LEDGER | What is citable / quarantined / needs-rebuild | Governance |

**Bottom line on status:** Movement 1 (the geometry, E + F) is the finished, citable result. **Phase 7 steering is the project's one *causal* result and the differentiator vs LRS (arXiv:2606.00726) — and it has not been run.** The layer decision gates it.

### 0.3 The decision — reconciled recommendation (this is the part that matters)

Section **I** (the project's own working position) leans **L27 (late)**, on three converging positives: Huang (arXiv:2505.22411) published L27 for this exact model; the de-confounded 07d sweep argmaxes at L27 for all four behaviours; and the cheap pilot confirmed L27 (suppressed uncertainty −63% relative, stayed "clean" while L16 inflated repetition).

**The two adversarial critics materially weaken that lean. Do not treat L27 as settled.** The most important findings:

**Critic N (layer) — 07d does *not* de-confound, so L27's strongest non-Huang support may be an artefact:**
- The 07d random null is `random_unit_direction` — **isotropic**. Behaviour vectors align with high-variance / near-logit directions, so the isotropic null *structurally cannot subtract* the late-layer read-out inflation. The raw effect roughly quadruples at L27 for **all four** behaviours; the null only doubles. R=2, n=12 — thin.
- **The tell:** `adding-knowledge` — a pre-registered *specificity null* (p=1.0 everywhere) — lights up **most** at L27. If 07d were localising behaviour-specific causality, a null behaviour could not peak. This is explicable only as a layer/read-out artefact.
- The descriptive geometry says **mid, unambiguously**: PR argmin is ~L15, and L27's participation ratio is roughly double the trough — **outside** the plateau. The probe weakly leans L18–L20.
- Pilot caveats glossed over: only **10 tasks, 2 categories**; vanilla behaviour-fraction swings **0.181→0.153 on byte-identical chains** across re-annotation (~25% of the celebrated 0.114 effect) with **no noise band**. The pilot's *robust* finding is narrow: L16 **damages** the model (repetition 0.40–0.59); L27 doesn't. That's a *damage* finding, not on-target efficacy.

**Critic O (methodology) — the deeper risk is circularity and power, not the layer:**
- The steering vector is **built from Sonnet labels and scored by Sonnet** — the same labelling function on both the X and Y axes of a causal claim (CF-7). A non-builder annotator half-fixes it; the gold standard is a **build-with-A / score-with-B swap** (cheap — per-annotator vectors already exist via `build_steering_arms.py`).
- `manifold_auto` ≈ `single_direction` (**cos 0.96**) — at the operating k the two "competing" headline arms are nearly identical, so the **manifold-vs-single test is under-powered by construction**. The honest contrast is `single` vs `manifold_k1`.
- The energy floor (`energy_matched_random`) exists for `single_direction` only; the **manifold arm is merely dimension-matched** — build `energy_matched_random_k1` or the manifold headline is un-floored.
- Power: N=50 (≈10–15 per cell in the bake-off), effects tiny except **uncertainty (−0.114, strong)**; backtracking (−0.006/−0.021) likely inconclusive after multiple-comparison correction. n=3 samples don't raise effective N.

**Synthesized recommendation:**

1. **Before any spend, run the two $0–$15 de-risks** (both critics independently demand them):
   - **[$0]** Re-run the 07d sweep curve with an **energy/covariance-matched null** (the `measure_mean_abs_proj` / `energy_matched_scale` logic already exists) on data already on disk. **If L27 collapses toward the interior, its argmax was an artefact — proven before spending a cent.**
   - **[$5–15]** Compute an **annotation-noise band** by re-annotating existing vanilla pilot chains (quantify the 0.181→0.153 swing) so every effect has an error bar.
2. **Hedge the layer.** Build and freeze **both {L27, ~L15/L16}** for the two viable behaviours (**uncertainty-estimation** = confirmatory, **backtracking** = exploratory). Do **not** sink the whole budget into a single, possibly-confounded layer. α\* is sealed only at L27 (`predictions_layer27.json`); flag any mid-layer dose as unsealed.
3. **Fix the circularity where the money should go.** Spend the marginal dollars on the **A↔B annotator swap** (build-from-one-annotator, score-with-another), *not* on the manifold k-sweep. Until a non-builder annotator endpoint exists, the headline is a *consistency check*, not a causal claim. (Memory flags the live blocker: proxy creds `CLAUDE_PROXY_URL/KEY` not set; the non-builder Qwen3-235B / Nova-Pro id is not located.)
4. **Trim the arms** (Critic O's design): **2 annotated behaviours + 2 generation-only nulls**; arms = `vanilla / single / manifold_k1 / energy_matched_random / energy_matched_random_k1 / random_subspace_k1` (1 rep); **sealed α\***, n=3, T=0.7, 50 tasks. **Cut:** the 8-point α grid, manifold k3/5/10/auto, `orthogonal_complement`, `random_direction`, and all gated/gradient arms. Est. **~$110–130** (or ~$160–180 with the annotator swap — the *correct* place to spend).
5. **Pre-register the falsifier:** uncertainty-estimation's **energy-matched Δ_floor, non-builder-scored, paired BCa CI includes 0 ⇒ the central causal claim is falsified**, retreat to geometry-only. Uncertainty is the only effect large enough that a null is informative rather than merely under-powered.

### 0.4 Go / no-go checklist before spending

- [ ] **$0 energy-matched-null re-analysis of 07d** — does L27 survive, or collapse to the interior? (de-risks the layer)
- [ ] **Annotation-noise band** on existing vanilla chains — every effect gets an error bar
- [ ] **Non-builder annotator endpoint located + credentialed** (Qwen3-235B / Nova-Pro; proxy creds set) — without it the headline is "preliminary, band-ungated"
- [ ] **`energy_matched_random_k1` built** — so the manifold arm has a real floor, not just dimension-matching
- [ ] **Doc inconsistency resolved** — `RESULTS_LEDGER` still cites "Venhoff mid-peaks 11/16/19/16" while on-disk 07d argmaxes L27 (`REVIEW_PHASE7 §5`)
- [ ] **α\* reconciled** — sealed only at L27; if a behaviour lands mid, its dose is unsealed and must be flagged
- [ ] **Arms + sample counts frozen** and the Holm family fixed *before* generation (no tune-on-eval; bypass `effect_quantile=0.8`)
- [ ] **CF-17 disclosed verbatim** on every single-layer headline — the layer is held out on the task axis only, never the layer axis

### 0.5 How to read the rest

Sections **A–M** are the stage-by-stage recap (each: what it does · why · code snippets · critique). Section **I** is the project's existing layer-decision writeup (leans L27). Sections **N** and **O** are the adversarial reviews that should override that lean toward the hedged, de-risked plan in §0.3 above.

---
