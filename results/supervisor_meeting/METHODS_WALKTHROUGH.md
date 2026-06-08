# Reasoning on a Manifold — Methods Walkthrough

*A complete step-by-step explanation of the project: the research question, every pipeline phase in detail, the geometric measures, the robustness program, the extensions, and how it all confirms the hypothesis. Written as a reference / thesis methods-chapter skeleton.*

**Model under study:** DeepSeek-R1-Distill-Qwen-1.5B (28 layers, hidden dim 1536, ≤8192 reasoning tokens, greedy decoding).
**Last updated:** 5 June 2026.

---

# PART I — The research question

## The idea in one sentence
We test whether each *individual reasoning behaviour* a reasoning model performs — backtracking, expressing uncertainty, testing examples, recalling knowledge — lives on its own **curved, low-dimensional manifold** in the model's internal activations.

## Why this is a real question (the "wedge")
Two recent papers stake out opposite positions and neither tested the middle ground:

- **Venhoff et al. (2025)** showed each reasoning behaviour can be captured by a **single linear direction** in the residual stream — effective dimension = 1 per behaviour. Add a vector, the behaviour increases. Clean, but it assumes the behaviour is one-dimensional.
- **Huang et al.** showed the *aggregate* "overthinking" phenomenon lives on a **manifold** (a curved, multi-dimensional surface), but never separated it into individual behaviours.

Our project is the **wedge** between them: *do individual behaviours each have their own multi-dimensional curved manifold?* If yes, Venhoff's single-direction picture is an oversimplification and Huang's geometric framing applies at a finer grain. This is falsifiable: if a behaviour's effective dimension is ≫ 1, its intrinsic dimension is far below its ambient dimension (the signature of curvature), and the structure survives a statistical control, then the manifold view wins.

## The model under study
DeepSeek-R1-Distill-Qwen-1.5B is a 1.5B-parameter *distilled* reasoning model: it learned chain-of-thought by imitating the larger DeepSeek-R1. We decode greedily (deterministic) for reproducibility.

## The three pre-registered outcomes ("stories")
- **Story A — curved low-dimensional manifold** (the win): low intrinsic dimension, strong curvature, survives the null. Supports the wedge.
- **Story B — flat high-dimensional subspace** (partial): high dimension but *flat* — interesting but not a "manifold" geometrically.
- **Story C — chain confound** (negative/fatal): the apparent structure is just sentences clustering by which chain they came from, not the behaviour. Kills the project.

The entire methodology is designed to distinguish A from B from C.

---

# PART II — The pipeline, phase by phase

## The taxonomy (the labels)
Venhoff's 6-label scheme for reasoning sentences: `initializing`, `deduction`, `adding-knowledge`, `example-testing`, `uncertainty-estimation`, `backtracking`. We analyse the **4 "target" behaviours** distinctive to *reasoning* models: **backtracking, uncertainty-estimation, example-testing, adding-knowledge**. (`deduction` and `initializing` are the "ordinary" modes.)

## Phase 1 — Task generation
**Purpose:** a diverse set of reasoning problems that provoke rich chains-of-thought.
**How:** generated **1,000 tasks across 10 reasoning categories** (causal reasoning, creative problem-solving, lateral thinking, mathematical logic, pattern recognition, probabilistic thinking, scientific reasoning, spatial reasoning, systems thinking, verbal logic) — 100 each. Deliberately broader/harder than Venhoff's mostly-math benchmarks.
**Output:** `data/tasks.json`.

## Phase 2 — Chain generation
**Purpose:** make the model reason and record its full thinking trace.
**How:** each task through R1-Distill-1.5B, greedy decoding, ≤8192 tokens, capturing the `<think>…</think>` chain.
**Result:** **1,000 chains, mean ~5,052 tokens.** Caveat tracked: ~50% hit the token cap (truncated mid-thought), heavily category-dependent (lateral_thinking 95% truncated, systems_thinking 4%). Doesn't hurt per-sentence geometry; matters for whole-chain analysis.
**Output:** `data/chains_R1-1.5B.json`.

## Phase 3 — Annotation (sentence-level behaviour labels)
**Purpose:** label every sentence with its behaviour — this is what lets us group activations by behaviour.
**How:** Venhoff's *verbatim* prompt (the model splits a chain into `["label"]text["end-section"]` segments). Originally **Claude Sonnet 4.5** via the AWS Bedrock proxy. Engineering detail: the API-Gateway has a **29-second hard timeout**, so long chains are **chunked** (~1000-token chunks on paragraph boundaries, with overlap de-duplicated at merge, plus a "continuation" prefix so the model doesn't mislabel each chunk's first sentence as `initializing`).
**Result:** **993/1000 chains fully annotated, 77,183 labelled spans.** First finding: ~3× more backtracking and ~2.4× more uncertainty than Venhoff — because our tasks are harder/more varied.
**Output:** `data/annotated_R1-1.5B.json`.

> **Session extension (multi-annotator).** To answer the supervisor's robustness demand, the annotator was generalised. The proxy returns Anthropic responses as a *list* of content blocks but other models (Qwen, Nova) as a plain *string*, so a universal text-extractor was added and a model parameter threaded through. After probing the proxy's 54-model whitelist and testing format-following, two more full annotation passes were run: **Qwen3-235B (Alibaba)** and **Nova-Pro (Amazon)** — independent model families — giving three independent labelings of the same 1,000 chains.

## Phase 4 — Activation extraction (turning text into geometry)
**Purpose:** for each labelled sentence, capture the model's internal state when it produced that sentence.
**How — in detail (the token choice matters):**
1. Reconstruct the exact `prompt + chain` text; run **one forward pass** with hooks on the residual stream at every layer.
2. For each sentence, locate it and find its **onset token** (first token of the sentence).
3. Take a fixed window: **1 token *before* onset + the first 10 tokens *of* the sentence** (~11 tokens), following Venhoff's "preceding + execution tokens".
4. **Mean-pool** the residual-stream activations over those positions → one 1536-d vector per (sentence, layer).
5. Accumulate into per-behaviour, per-layer matrices.

**Result:** **37,851 vectors × 28 layers** (~6.5 GB). Per behaviour: backtracking 10,267, uncertainty 16,728, example-testing 5,829, adding-knowledge 5,027.
**Output:** `data/activations/R1-1.5B/{behaviour}_layer{N}.npy`.

> **Why subtle (source of a bug found this session):** the window is fixed-length, anchored at onset — it ignores the sentence's true end. Short repeated markers ("Wait.", "Hmm.") produce **near-identical pooled vectors**, creating many duplicate rows that later distorted an estimator (Part IV).
> **Session fix:** earlier extraction read the wrong JSON fields and used wrong (uppercase) labels, silently breaking downstream phases; corrected to `ann["label"]`/`ann["text"]` with hyphen-lowercase labels, and fixed the residual hook to handle both tuple and tensor layer outputs.

## Phase 5 — PCA dimensionality (first geometric measurement)
**Purpose:** "how many dimensions does this behaviour's activation cloud occupy?"
**How:** fit PCA per behaviour per layer; compute **d_eff(70%)** (components to reach 70% variance) and the **participation ratio** (how many directions dominate).
**Result:** d_eff(70%) ranges **45–98** — decisively **≫ 1**, falsifying Venhoff's single-direction prediction at every layer. Participation ratio troughs at **middle layers L14–L17** — each behaviour's "manifold-peak" layer.
**Output:** `results/pca/R1-1.5B/` (+ `layer_profiles.json`).

> **Session fixes:** (1) PCA was capped at 50 components, so d_eff **saturated at exactly 50** for 3/4 behaviours — raised cap to 100, revealing true values (50→57/54/61). (2) Added a missing `numpy` import crashing the null path. (3) Added the chain-stratified null at *every* layer (was 3 layers).

## Phase 5b — Geometric diagnostics (the heart of the project)
**Purpose:** PCA gives the *ambient* dimension; Phase 5b answers the three questions distinguishing A/B/C — **is it low-dimensional, curved, real?** — via three "batteries":
- **Battery A — intrinsic dimension** (TwoNN, Levina-Bickel, correlation dimension): true free parameters, independent of the embedding.
- **Battery B — curvature** (local-vs-global PCA ratio, geodesic/Euclidean ratio, tangent-space variation): is the manifold bent?
- **Battery C — null hierarchy** (chain-stratified permutation, cross-chain, Marchenko-Pastur): is the structure behaviour-specific?

**Result (original):** intrinsic dim ~10–13 vs PCA ~47–61 (~5× compression — the curvature signature); all three curvature diagnostics agree it's strongly curved; chain-stratified null rejects for the strong behaviours. **Story A.**
**Output:** `results/geometric/R1-1.5B/`. *(Deepened substantially this session — Part IV.)*

## Phase 5c — Cross-layer probing
**Purpose:** *where in the network is each behaviour linearly readable?*
**How:** train a linear probe ("behaviour X vs not") from activations, at each of the 28 layers.
**Result:** every behaviour is decodable **85–92% at ALL layers**, and the curve is *flat*. Clean, citable distinction: **linear decodability ≠ manifold geometry** — a behaviour can be read off a single hyperplane everywhere, yet its *geometric* organisation only crystallises at the middle layers.
**Output:** `results/cross_layer/R1-1.5B/`.

## Phase 5d — Sub-type clustering
**Purpose:** is each behaviour one thing or a mixture of sub-types (e.g. backtracking = arithmetic-recheck vs strategy-pivot)?
**How:** k-means in the top-PCA subspace at the manifold-peak layer; k chosen by silhouette.
**Result:** all four select **k=2 with weak silhouettes (0.11–0.18)** → *no clean discrete sub-types*. Behaviours are **continuous curved manifolds**, not unions of blobs. (Refuted the earlier hypothesis that adding-knowledge splits into 4–6 sub-types.)
**Output:** `results/clustering/R1-1.5B/`.

> **Session fix:** focus-layer selector used `argmax(d_eff)`, returning layer 0 when d_eff saturated — changed to `argmin(participation_ratio)` (the true manifold-peak).

## Layer triangulation
**Purpose:** choose, per behaviour, the layer to build steering vectors at, by combining signals.
**How:** peak of three curves — geometry (PR trough), probe accuracy, attribution-patching effect — forms a candidate layer set.
**Result:** bt **L13–14**, unc **L13–14**, adding-knowledge **L17–18**, example-testing **L27**.

> **Session fixes:** switched geometry signal from `d_eff argmax` (unreliable once saturated → kept defaulting) to **participation-ratio argmin**; relaxed an over-strict flatness gate; fixed boundary smoothing so example-testing's L27 edge-trough survives.

## Phase 6 — Steering vector construction
**Purpose:** build the two competing intervention vectors per behaviour.
**How:**
- **Single-direction (Venhoff):** `r = mean(on) − mean(off)`, unit-normalised — difference-of-means between the behaviour and everything else.
- **Manifold-projected (ours, adapting Huang):** project that direction onto the top-k PCA subspace of the behaviour's manifold: `r_proj = Σᵢ(r·vᵢ)vᵢ`.

**Result (done this session):** built both at each behaviour's manifold-peak layer. Free interim finding: the difference-of-means direction lies **88–94% inside** each behaviour's own manifold subspace, but is **nearly orthogonal to the top principal component** for example-testing (cos 0.005) and adding-knowledge (0.098). So the behaviour-defining direction is a *distributed, mid-spectrum* manifold feature, not the dominant variance axis — independent corroboration of the manifold picture.
**Output:** `results/steering_vectors/R1-1.5B/`.

## Phase 7 / 7b — Steering evaluation & activation patching (the causal half)
**Purpose:** move from *correlational* geometry to *causal* evidence.
- **Phase 7 (steering eval):** apply each vector during generation across strengths α, re-annotate outputs, measure the behaviour-fraction shift. *Sufficiency:* add the vector → behaviour rises. *Necessity:* ablate the manifold subspace → behaviour drops. Compare single-direction vs manifold-projected.
- **Phase 7b (attribution patching):** patch a behaviour's activations donor→recipient and measure the causal change, localising the behaviour to layers.

**Status:** **pending** — Phase 7 needs API credits (to re-annotate steered outputs) and a free GPU. Vectors built and staged.

---

# PART III — The geometric measures, in detail

All share one foundation: **on a d-dimensional manifold, a small ball of radius r contains a number of points growing like rᵈ.** Measure that scaling → recover d, regardless of curvature.

## Dimensionality measures (Phase 5)
- **PCA d_eff(70%):** components for 70% of variance. *Tail-sensitive*. The **ambient** linear dimension.
- **Participation ratio** PR = (Σλ)²/Σ(λ²), λ = eigenvalues. Number of *dominant* directions; *head-dominated*, sample-size robust. We use **argmin(PR) across layers** for the manifold-peak layer.

High d_eff + low PR = "a few dominant directions plus a long tail" — exactly what a *curved* low-dim manifold looks like under linear PCA (many short segments to trace a circle).

## Intrinsic-dimension estimators (Phase 5b, Battery A)
- **TwoNN:** μ = r₂/r₁ (2nd-nearest ÷ 1st-nearest distance) follows a Pareto law whose shape parameter *is* the dimension. Most local, normally most robust — **but** destroyed by duplicate points (Part IV).
- **Levina-Bickel:** maximum-likelihood over the first k neighbours. Lower variance, reads slightly high.
- **Correlation dimension:** log-log slope of "fraction of point-pairs within radius r" vs r. Global, very stable, biased slightly low for curved manifolds. **The reliable one here.**

Agreement of the three (within ~2×) → dimension well-defined. Their *ordering* (corr-dim < TwoNN < Levina-Bickel) is itself the fingerprint of mild curvature.

## Curvature diagnostics (Phase 5b, Battery B)
- **Local-vs-global PCA dim ratio:** PCA on a small neighbourhood vs the whole cloud. Flat → 1; curved → ≪ 1 (locally you see only the tangent plane). Ours ≈ **0.02–0.09**.
- **Geodesic/Euclidean ratio:** distance *along* the manifold (graph shortest-path) ÷ straight line. Flat → 1; curved → >1. Ours ≈ **1.6–1.8**.
- **Tangent-space variation:** angle between locally-fit tangent planes at nearby points. Flat → 0°; curved → large. Ours ≈ **57–68°**.

Three independent formalisms, all agreeing → curvature is real.

## The null hierarchy (Phase 5b, Battery C) — the rigour test
The danger (Story C): two backtracking sentences from the *same* chain look alike *because of the chain*, not the behaviour.
- **Chain-stratified permutation (primary):** shuffle behaviour labels *within each chain* and recompute. Beating this null = structure is behaviour-specific, beyond chain identity. The strongest control, rare in this literature.
- **Cross-chain permutation (secondary):** shuffle labels globally — weaker.
- **Marchenko-Pastur (tertiary):** compare the eigenvalue spectrum to random-matrix noise — checks high dimension isn't finite-sample noise.

**Result:** chain-stratified null rejects at p<0.01 at all 28 layers for backtracking & uncertainty, 19/28 for example-testing, L17–19 for adding-knowledge.

---

# PART IV — The robustness program (this session's methodological deepening)

The supervisor's "make it more rigorous and causal" demand, turned onto the geometry itself. We treated our own results adversarially.

## The discovery: duplicate activations
**35–56% of the pooled activation vectors are *exact* duplicates.** Cause: the fixed 1+10-token mean-pool window on short repeated markers collapses many sentences to identical vectors. A genuine methodology issue nobody would catch without stress-testing.

## The consequence: twoNN is unreliable here
TwoNN was stable at full N (7.2) but **collapsed to <1 under subsampling** — degenerate, because duplicates wreck the nearest-neighbour ratio distribution. **Fix:** lead with **correlation dimension**, rock-stable (6.6–6.7 across every sample size) and barely changed by deduplication.

## The keystone test (the single most important robustness result)
*Is the low intrinsic dimension a property of the behaviour, or of which chain the sentence came from (Story C)?* Compute intrinsic dimension three ways on deduplicated data:
1. **full** sample;
2. **random subsample** (same size as #3 — controls for sample size);
3. **chain-stratified** — *one sentence per chain*, removing all within-chain correlation.

| Behaviour | full | random-sub | chain-stratified |
|---|---|---|---|
| backtracking | 6.85 | 6.87 | 6.86 |
| uncertainty-estimation | 6.91 | 7.13 | 7.22 |
| adding-knowledge | 8.83 | 8.66 | 8.76 |
| example-testing | 6.28 | 6.30 | 6.79 |

One sentence per chain leaves the intrinsic dimension unchanged → **the low-dimensional manifold is behaviour-intrinsic, NOT a chain confound. Story C refuted on the geometry itself.**

## The curvature density control (the loose end, now closed)
The geodesic ratio *dropped* under chain-stratification (1.65→1.30), which could mean curvature was chain-driven. But the same-N **random subsample** also dropped to ~1.30 → the drop is a **sparse-sampling effect, not a chain effect** (fewer points → sparser graph → geodesics can't hug the manifold as tightly). Curvature stays >1 throughout → also behaviour-intrinsic.

## Scale-robustness
PCA d_eff roughly *doubles* under z-scoring (a few "massive-activation" feature dimensions dominate the raw covariance), so the specific number "~50" is preprocessing-dependent — we now lead with the participation ratio. But the *qualitative* conclusion (d_eff ≫ 1, intrinsic ≪ ambient) holds under every preprocessing choice, and Marchenko-Pastur confirms 100+ eigenvalues above the noise floor (real signal, not finite-sample noise).

**Net effect:** two real methodology flaws found (duplicates, twoNN), fixed, and the central claim came out *stronger*. Rigour that converts "suggestive" into "defensible."

---

# PART V — The extensions

## Multi-annotator (running now)
**Why:** the geometry depends on *labels*; one annotator (Sonnet) is a single point of failure. **Tests:** does the manifold replicate when a *different, independent* model labels the same chains? **How:** full re-annotation by Qwen3-235B and Nova-Pro, then the entire geometry pipeline per annotator, then a 3-way comparison. The annotators genuinely disagree on label *frequencies* (uncertainty: Sonnet 22% vs Qwen3 7% vs Nova 12%) — so the real test is whether each annotator's labels still yield the same low-dim curved manifold. That comparison (`cross_annotator_comparison.md`) is the headline deliverable being computed now.

## Baseline control (Qwen-Math-1.5B)
**Why:** to claim the *reasoning distillation* introduced these behaviours, we need a non-reasoning control. **What:** 1,000 chains from `Qwen2.5-Math-1.5B` — generated, **annotation pending credits.**

## Scale extension (7B)
**Why:** does the finding hold at larger scale (Venhoff scaled too)? **What:** 500 balanced chains (50 × 10 categories) from R1-Distill-Qwen-7B, generating now. Phase 4/5/5b on 7B need no API.

---

# PART VI — How it all confirms the hypothesis

| Test | Result | Rules out |
|---|---|---|
| d_eff ≫ 1 | 45–98 at every layer | Venhoff's single-direction |
| intrinsic ≪ ambient | ~7 (corr-dim) vs 47–98 (PCA), ~5–7× | Story B (flat subspace) |
| 3 curvature diagnostics | all agree: strongly curved | Story B |
| chain-stratified null | rejects; **keystone passes** | **Story C (chain confound)** |
| robust to dedup / subsampling / scaling | yes | artifacts |

We land squarely in **Story A — a curved, low-dimensional, behaviour-intrinsic manifold.** Backtracking and uncertainty are the flagship cases (significant at all 28 layers); example-testing is clear but layer-localised; adding-knowledge is the honest outlier (highest dimension, narrowest layer window). The wedge between Venhoff and Huang is empirically supported for *individual* reasoning behaviours.

---

# PART VII — Current state

- **Done & verified:** Phases 1–6 + all geometry diagnostics + the full Tier-0 robustness program + multi-annotator annotation (3 families).
- **Running autonomously on the cluster:** per-annotator extraction → Phase 5/5b/5c/5d → 3-way manifold-replication comparison (~14h), plus 7B generation.
- **Pending (needs credits/GPU):** Phase 7 steering evaluation (the causal half), baseline annotation, 7B downstream.

**The one genuine open question** is the causal half: the geometry is now airtight, but "intervening on the manifold changes behaviour" (Phase 7) is the claim still to establish — where the remaining credits and GPU time go.

---

## Companion documents (`results/supervisor_meeting/`)
- `PROJECT_LOG.md` — consolidated project journal with figures.
- `ROBUSTNESS_PLAN.md` — the claim → threat → test robustness agenda.
- `PHASE_5B_DEEP_DIVE.md` — full methodology + Goodfire comparison.
- `SUMMARY.md` / `CHECKPOINT.md` — status snapshots.
- **`METHODS_WALKTHROUGH.md` (this file)** — the step-by-step explanation.
