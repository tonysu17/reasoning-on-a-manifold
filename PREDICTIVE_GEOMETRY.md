# The Predictive Geometry of Reasoning

*A side project of "The Geometry of Reasoning" thesis. Branch: `predictive-geometry-of-reasoning`.*

## 1. One sentence

Train a model to predict the **next reasoning step of a chain-of-thought in latent (embedding) space**, and test whether the **geometry of where that predictor succeeds and fails** is a signature of correct reasoning — moving the thesis's claim that "reasoning is a geometric process" from a *descriptive* statement about static representations to a *dynamical, predictive* one about how reasoning unfolds in time.

## 2. Why this, why now

The thesis (Movement 1, *Structure*) shows that per-behaviour reasoning representations occupy low-dimensional, structured subspaces. But it deliberately leaves a door open:

- **ch07** tests per-behaviour *curvature*, finds it confounded by within-chain autocorrelation, and **relocates it "as an open question, to the trajectory"** — "if curvature lives anywhere in this data it lives in that trajectory… left to future work."
- **ch02** commits to **"locus-in-process"**: "it is the unfolding *trajectory* and not any static activation point that bears the reasoning… the natural future experiments are trajectory-based."

JEPA (Joint-Embedding Predictive Architecture — LeCun's program; I-JEPA, V-JEPA 2, LeJEPA) is precisely a machine for **predicting future states in a learned latent space while discarding unpredictable detail**. It is therefore the natural apparatus for the trajectory question the thesis promised but did not build. This project is that apparatus.

## 3. The core question

> **Is reasoning a *predictable* geometric process, and does its *predictability structure* distinguish good reasoning from bad?**

Concretely: given the embedding of reasoning steps `x_1 … x_t`, predict `x_{t+1}`. Study the **residual** `r_t = x_{t+1} − f(x_≤t)` — the part of the next step the predictor could *not* anticipate. The bet is that where the predictor breaks (sharp residuals) localizes the *branch points / backtracks* of reasoning, and that the structure of those breaks separates correct from incorrect chains.

## 4. Hypotheses

| # | Hypothesis | Primary metric | Must beat |
|---|---|---|---|
| **H1** (keystone, functional) | Trajectory **predictability** (residual magnitude / growth / direction-churn) predicts chain-level **correctness** | chain-grouped ROC-AUC | token-NLL, raw-curvature, length+difficulty; and both nulls |
| **H2** (representational) | A *learned* predictor recovers correctness/behaviour structure that raw hidden states and a random projection do not | probe AUC / decoding | raw `x`, random projection, untrained predictor |
| **H3** (geometric) | Residual spikes localize geometrically special points (high curvature / subspace transitions = *backtracking*); correct vs incorrect trajectories differ here after difficulty control | curvature@residual-peaks; matched-pair Δ | within-chain shuffle null |
| **H4** (causal, apex — *confirmation, not novelty*) | Steering along the **predicted next-step direction** changes reasoning more than a random direction | behaviour/correctness shift under steering | random-direction & mean-direction controls |

**Null discipline (non-negotiable).** Every positive claim must survive (a) **chain-stratified label permutation** (does the signal exceed chance given chain composition?) and (b) **within-chain step-order shuffle** (does it need the genuine temporal order, or is it a static-cloud artifact?). A clean negative that beats neither is itself a reportable result.

## 5. What is genuinely novel (and what is not)

- **Novel (our atom), NARROWED (2026-06-15 lit pass):** a forward predictor over the **behaviour-segmented reasoning _steps_** whose **residual _geometry over time_** — *where* it spikes — localizes branch/backtrack points and classifies chains under chain-grouped nulls. We do **NOT** claim to be first to relate predictor unpredictability to correctness — that claim is retired: **PHi (arXiv:2503.13431)** already does this at the _per-token_ residual-stream level, and **SSP (2604.18464)** trains a next-_step_ predictor but reads _smoothness_ (AUC≈0.5). The defensible wedge: step/behaviour segmentation + residual _geometry_ (not a per-token scalar) + explicit branch-point localization, all under the chain-stratified + step-shuffle nulls.
- **Already taken — do not claim:** predictor-unpredictability↔correctness at the per-token level (PHi 2503.13431); "low-dimensional curved reasoning manifold" (arXiv:2510.09782 ≈ our thesis title; SSP 2604.18464; phase-space 2410.04415); "predict next step in latent space" (COCONUT 2412.06769, generative; LLM-JEPA 2509.14252, paired-view); "steer along a predicted/ideal next-state direction" (ASM, ICLR 2026).
- **Our wedge against the nearest neighbours:** SSP (2604.18464) reports that trajectory *smoothness* does **not** encode correctness (AUC≈0.5) and explicitly declines to build a latent-prediction feedback loop; 2604.05655 gets AUC 0.87 but from a *static* activation-difference probe. Our quantity — a *learned predictor's residual geometry* — is a **third thing**, and it directly targets the open question SSP left negative.

## 6. Method — the escalation ladder

Each rung is gated: it is pursued only if it beats the previous rung **and** both nulls.

- **Rung 0 — predictor-free.** Raw-trajectory Frenet curvature (existing `src/cbs/trajectory.py`) on difficulty-matched correct/incorrect chains. Cheap; tells us if *any* trajectory geometry separates correctness before we build anything.
- **Rung 1 — linear, no anti-collapse.** Ridge predictor of the **displacement** `x_{t+1}−x_t` (a residual/skip connection, so the residual measures the *unpredictable* part and the predictor can only improve on persistence). Residual geometry has **no** anti-collapse regulariser, so reading geometry off it is not circular. *(Built and tested.)*
- **Rung 2 — small JEPA.** A compact learned predictor (causal MLP/transformer over step embeddings, or the LLM-JEPA `[PRED]`-token trick) with **SIGReg** (LeJEPA) or **Barlow-Twins** anti-collapse. Pursued only to *beat* Rung 1; only *relative/functional* claims (never absolute intrinsic-dim/isotropy of the regularised latent).
- **Rung 3 — apex.** Goal-conditioned latent rollout + CEM planning, and **causal steering** along the predicted direction via `src/steered_inference.SteeredModel`.

## 7. Confound-aware design (inherits the thesis register)

- **CF-2 (chain confound):** every split is **chain-grouped** — a chain never trains and tests together; effective-N is the chain count, not the step count.
- **Anti-collapse circularity:** geometry is read off the **residual only**, never off a regularised latent; Rung 1 (no regulariser) is run first.
- **CF-6 (mean-pooling / sparsity):** trajectories are the sparse 4-behaviour subsequence; every pair carries its **step-gap**, with a `max_gap=1` adjacency control. Mean-pooled local data is *suggestive, not confirmatory* — a last-token / un-pooled re-extraction (GPU) is the confirmatory arm.
- **CF-8 (truncation):** 50% of chains hit the token cap; truncation correlates with category. Correctness labels carry a `truncated` flag and selection prefers complete chains; the contrast must beat a length/gap/truncation baseline.
- **CF-7 (label noise):** correctness comes from an LLM judge (open-ended proofs have no exact-match grader); `confidence` is recorded for a confidence floor, and "uncertain" verdicts are excluded, never coerced.

## 8. Data & current status

- **Model/data:** R1-Distill-Qwen-1.5B chains over 1000 tasks × 10 reasoning categories; behaviour-annotated into sentence spans; residual-stream embeddings for 4 behaviours (backtracking, uncertainty-estimation, example-testing, adding-knowledge), mean-pooled, all 28 layers.
- **Built & green:** `src/predict/` (dataset, predictor, evaluation, nulls, labels), 9/9 unit tests, real-data plumbing verified (986/1000 chains usable at layer 14, median 20 steps), label-generation + gate runners validated on no-API paths.
- **Early read (pre-labels):** a linear **cross-chain** next-step map is *worse than persistence* (residual/step ≈ 1.4–1.5). Cross-chain CV likely kills a chain-specific linear map ⇒ **Rung 1 may be a null**, which would motivate Rung 2 and sharpen the SSP-targeting story.

## 9. What counts as success

- **Positive:** Rung-1 (or Rung-2) residual AUC beats Rung-0 curvature, persistence, and length/difficulty, on difficulty-matched chains, with **both nulls rejecting**. → "predictive geometry of reasoning is a correctness signal."
- **Informative null:** Rung-1 ≈ baselines but Rung-2 (nonlinear) clears the bar → "the signal is nonlinear; capacity matters."
- **Hard null:** neither rung beats baselines → a clean, publishable negative that *confirms and extends* SSP (smoothness, static probes, **and** learned-residual geometry all fail) — still a thesis-relevant result about the limits of trajectory geometry.

## 10. Thesis placement

Movement 1 (**Structure**), as the trajectory-level realization of the ch07 "curvature-is-a-trajectory-property" hook and the ch02 "locus-in-process" commitment. The safety angle is **cut** (scope + capability/difficulty confound).
