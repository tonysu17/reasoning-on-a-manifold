# The Predictive Geometry of Reasoning — Theory & Walk-through

*Companion to [PREDICTIVE_GEOMETRY.md](PREDICTIVE_GEOMETRY.md). Built up from first
principles: the conceptual bet, the apparatus (JEPA), the geometry, the data, the
prediction setup, the statistics that make it trustworthy, the method as executed,
and how the theory maps onto the pilot results.*

---

## Part I — The central bet

The thesis claims **reasoning is a geometric process**: when a model reasons, each step of
its chain-of-thought corresponds to a point in a high-dimensional activation space (the
residual stream, here 1536-D), and those points have structure — they lie near a
low-dimensional surface and move in patterned ways.

The thesis so far is **descriptive**: it measures the *static* geometry of clouds of these
points (effective dimension, curvature, anisotropy). This project adds the **dynamical /
predictive** complement:

> If reasoning is a geometric *process*, the *motion* through this space should be partly
> **predictable**, and the **failures of prediction** — moments the model does something you
> couldn't anticipate — should be **diagnostic** of branching, confusion, or error.

That sentence is the whole project. It also fills a hook the thesis already names: ch07
relocates per-behaviour curvature "as an open question, to the trajectory," and ch02 commits
to "locus-in-process" (the unfolding trajectory, not the static point, bears the reasoning).

---

## Part II — The apparatus: predictive latent models (JEPA)

### The manifold hypothesis
High-dimensional data concentrates near a much lower-dimensional curved surface — a
*manifold* — in the ambient space. A 1536-D activation does not roam freely; it lives near a
surface of intrinsic dimension maybe 4–40. Geometry is the right language *because* the data
is manifold-structured.

### Why predict in *latent* space (the JEPA thesis)
Two ways to model "what comes next":

1. **Generative / reconstructive** (LLMs, pixel video models): predict the next thing in
   *input space* (tokens, pixels). Problem: most detail in a high-dim signal is
   **unpredictable noise**; spending capacity to predict it is wasteful, and when many
   futures are valid, minimizing reconstruction error yields **blurry, mode-averaged**
   output (the average of all valid next-frames is a blur).
2. **Joint-Embedding Predictive (JEPA):** predict in a *learned latent space* `s = Enc(x)`.
   The encoder may **discard** unpredictable detail before prediction; you predict
   `Enc(future)` from `Enc(past)` and measure error *in embedding space*.

The defining move: **the prediction target is a learned embedding, not the raw input** —
what separates JEPA from both generative modelling and contrastive learning.

### The energy-based framing
JEPA is an **energy-based model**: instead of a normalized probability `p(y|x)` (which needs
the intractable partition function `Z`), learn a scalar **energy** `F(x,y)`, low when `y` is
a compatible continuation and high otherwise — no normalization. Here
`F = ||Predict(Enc(past)) − Enc(next)||²`: low energy = the next step was predictable.

### The collapse problem (deeply geometric)
If the only objective is "make the prediction match the target embedding," the encoder can
**cheat** by mapping everything to a constant — perfect prediction, useless representation.
This is **representational collapse**:
- **Full collapse:** everything → one point (zero variance).
- **Dimensional collapse:** embeddings lose rank, occupying a thin subspace; the covariance
  spectrum has most eigenvalues ≈ 0.

Both are statements about the **spectrum/geometry** of the embedding distribution — which is
exactly why intrinsic-dimension and rank tooling is the right diagnostic.

### Anti-collapse mechanisms (matters for Rung 2)
- **Contrastive (SimCLR/InfoNCE):** push negative pairs apart; needs many negatives.
- **Distillation / asymmetry (BYOL, SimSiam, I-JEPA):** stop-gradient on the target + EMA
  target encoder; the predictor chases a slow target it can't collapse.
- **Regularization:**
  - **VICReg** = Variance (per-dim variance floor) + Invariance (match target) + Covariance
    (off-diagonal covariance → 0, decorrelating dimensions).
  - **Barlow Twins** (our Rung-2 default): cross-correlation matrix of predicted vs target →
    **identity** (diagonal → 1, off-diagonal → 0); decorrelated dims can't collapse.
- **Distributional (LeJEPA / SIGReg, 2025):** prove the ideal embedding distribution and
  regularize toward it — no EMA/stop-gradient/predictor heuristics.

### The LeJEPA theorem (the deepest piece)
LeJEPA proves the **isotropic Gaussian** is the *unique* embedding distribution minimizing
worst-case downstream prediction risk (isotropy = "no prior on which directions matter,"
optimal under task uncertainty). It enforces this with **SIGReg**: by the **Cramér–Wold
theorem** a distribution is determined by all its 1-D projections, so it penalizes the
deviation of many random 1-D projections from a standard Gaussian (Epps–Pulley /
characteristic-function test). Our Rung-2 includes a CPU-light moment-matching approximation.
This is a *geometric thesis about embeddings stated as a theorem* — and it creates the
circularity trap of Part VI.

---

## Part III — The geometry, formally

### Intrinsic dimension (ID)
- **Correlation dimension** (Grassberger–Procaccia; the project's chosen estimator): for
  data on a `d`-manifold the count of point-pairs within radius `r` grows as `C(r) ∝ r^d`, so
  `d = d log C / d log r`. A robust global scaling exponent.
- **TwoNN** (Facco): ratio of 2nd-to-1st nearest-neighbour distances pins `d`; fast but
  unstable under near-duplicate points.
- **MLE (Levina–Bickel):** maximum likelihood on nearest-neighbour distances.
- **Participation ratio** `PR = (Σλ_i)² / Σλ_i²` (λ = covariance eigenvalues): a *spectral*
  effective dimension, scale-robust; small when variance concentrates, large when spread.

The step-**displacement** cloud has correlation dimension ≈ **3.7–3.9** — reasoning steps
move on a ~4-D surface.

### Curvature of a trajectory
A chain is an ordered sequence `x_0…x_T` — a discrete curve. The right measure is the
**arc-length-reparameterized discrete Frenet curvature**:

    T_left  = (x_t − x_{t−1}) / ||x_t − x_{t−1}||        (incoming unit tangent)
    T_right = (x_{t+1} − x_t) / ||x_{t+1} − x_t||        (outgoing unit tangent)
    ds      = (||x_t − x_{t−1}|| + ||x_{t+1} − x_t||) / 2 (local step size)
    κ_t     = ||T_right − T_left|| / ds

The **arc-length normalization** (`/ds`) is what makes it a real curvature (turning per unit
distance) rather than just a second difference; without it, "bends sharply" and "takes big
steps" get conflated. (Exactly why ch07's curvature claim was confounded.)

---

## Part IV — Turning reasoning into a measurable object

1. **Tasks → chains:** 1,000 prompts × 10 categories; R1-Distill-Qwen-1.5B generates the CoT.
2. **Segmentation:** an LLM annotator splits each chain into sentence-level **spans**, each
   labelled with a behaviour. A "reasoning step" = one labelled span.
3. **Embedding:** pool the **residual-stream** activations of each span into one 1536-D
   vector at a chosen layer. The residual stream is the transformer's running working memory
   — where the reasoning state lives.
4. Each chain → an ordered sequence `x_1…x_T ∈ ℝ^1536` — a trajectory.

**Two baked-in confounds the design must respect:**
- **Sparsity (CF-6):** activations saved for only 4 of 6 behaviours → the trajectory is a
  *sparse subsequence*; we track each step's original index and the **gap** between retained
  steps.
- **Pooling:** mean-pooling each step can wash out within-step structure — relevant to
  whether "order" effects are visible (Part VIII).

---

## Part V — The heart: next-step prediction and the residual

### Setup
Train `f` to predict the next step. Two formulations:
- **Absolute:** predict `x_{t+1}`.
- **Displacement / "delta" (used):** predict `x_{t+1} − x_t`, reconstruct
  `x_{t+1} = x_t + f(x_t)`.

**Why delta** (a residual/skip connection): the model's job becomes "predict how the state
*changes*," so a model that learns nothing falls back to **persistence** (`x_{t+1} ≈ x_t`),
and the residual measures *only the motion not explained by "keep going as you were."* Same
inductive bias as ResNets and world-models.

### The residual is the object of study
    r_t = x_{t+1} − f(x_≤t)        ("surprise" at step t)
Everything is read off `r_t`: magnitude, growth over the trajectory, direction churn.

### Reference baselines that give the residual meaning
- **Persistence:** `f(x_t) = x_t` (residual = raw displacement). The "do-nothing" predictor.
- **Variance explained:** `R² = 1 − Var(residual)/Var(displacement)`; R² > 0 ⇒ the learned
  predictor genuinely beats persistence. We measured **R² ≈ 0.3** — ~30% of step-to-step
  motion is linearly predictable, cross-chain.

---

## Part VI — Why the results are trustworthy (statistical theory)

### Out-of-fold prediction
To claim a predictor works it must be tested on unseen data. We compute **out-of-fold (OOF)**
residuals: every residual comes from a model that never saw that point.

### The chain confound & chain-grouped CV (the keystone)
Steps within a chain are highly **autocorrelated**. Random splits put near-identical steps in
both train and test → **leakage** → inflated scores. Fix: **GroupKFold by chain** (whole
chains go entirely to train or test). The honest **effective sample size** is the number of
*chains* (~1,000), not steps (~37,000). This is CF-2, the keystone confound; enforced
everywhere (a regression test shows ungrouped CV faking 0.99 vs grouped 0.50).

### ROC-AUC: the correctness metric
A logistic probe maps per-chain residual features → correct/incorrect (chain-grouped), scored
by **ROC-AUC** = P(a random correct chain ranks above a random incorrect one). **0.5 =
chance**, 1.0 = perfect; threshold-free and imbalance-robust. Our 0.58 = weak but
better-than-chance.

### Permutation null tests (the soul of the rigor)
A number is meaningless without knowing what *chance* looks like *for this exact setup*. A
**permutation test** scrambles the thing under test, recomputes the statistic hundreds of
times, and locates the real value in that null distribution. The **p-value** = fraction of
scrambles ≥ reality, **Phipson–Smyth smoothed** `(1 + #beats)/(1 + #scrambles)` (the +1
forbids a fake p = 0). Two nulls, isolating different things:
1. **Label-permutation:** shuffle correct/incorrect (within difficulty strata). *Does it beat
   chance given chain composition?* — our signal **passes** (p≈0.03 at the best layer).
2. **Step-shuffle:** shuffle step *order* within each chain. *Does the signal need the
   temporal order, or is it a static cloud property?* — our signal **fails** (p>0.7), so the
   correctness signal is **overall surprise magnitude, not trajectory shape** (refutes H3).

### The circularity trap (why we read the residual, not the latent)
A JEPA's anti-collapse term *manufactures* geometry (VICReg pins variance and decorrelates →
inflates rank and isotropy). Measuring the regularized latent's intrinsic dimension/isotropy
and calling it a finding would measure the regularizer, not reasoning. Defenses: read geometry
**only off the residual**; run **Rung 1 (linear, no regularizer) first**; make only
**relative/functional** claims.

---

## Part VII — The method as executed (the ladder)

A **gated escalation ladder** — build the cheap thing first, escalate only on evidence:

1. **Rung 0 — predictor-free curvature:** raw-trajectory Frenet curvature vs correctness.
2. **Rung 1 — linear ridge predictor:** ridge `x_t → (x_{t+1}−x_t)`, chain-grouped OOF, no
   anti-collapse (L2 penalty `α||w||²` controls variance in 1536-D). The **scientifically
   central, uncontaminated** rung.
3. **Rung 2 — small JEPA:** 1-hidden-layer MLP + Barlow-Twins / SIGReg anti-collapse, trained
   per fold on CPU; justified only if it beats Rung 1. Same `ResidualResult` interface, so all
   downstream geometry/nulls work unchanged.
4. **Rung 3 — causal steering (not built):** the apex. Theory = **Model-Predictive Control**:
   imagine latent rollouts, score against a goal embedding (energy), pick the best via the
   **Cross-Entropy Method** — exactly how V-JEPA 2 plans robot actions; the reasoning analog
   steers along the predicted next-step direction.

**The label problem:** open-ended proofs have no exact-match grader, so correctness is a
*latent variable* estimated by an **LLM judge** — inherently noisy (cf. behaviour-annotation
κ 0.35–0.44). We record judge **confidence** and exclude "uncertain" (never coerce).

---

## Part VIII — Reading the results through the theory

- **Reasoning is predictable & low-dimensional** (R²≈0.3; displacement ID≈4): the manifold
  hypothesis holds for *dynamics*, not just static clouds — the cleanest, label-free result.
- **The residual is higher-D than the displacement** (~5–6 vs ~4): the predictor strips the
  low-D predictable component, leaving higher-D, more isotropic noise — the structure is in
  what's removed.
- **Correctness signal is real but modest** (AUC ~0.58, p≈0.03) and **magnitude-not-order**
  (fails step-shuffle): chains the model finds *globally harder to anticipate* are more often
  wrong — an *unpredictability ↔ error* link (the per-token version is PHi 2503.13431), not a
  trajectory-shape effect. H1 partly supported; H3 refuted.
- **The signal is largely linear** (JEPA barely beats ridge; doesn't improve with training):
  the correctness information is ~linear in residual magnitude; more model capacity won't
  help — more data or a different signal might.
- **Caveat on the H3 negative:** steps are mean-pooled, which may wash out order structure, so
  "order doesn't matter" is "not found *in this representation*"; a last-token / un-pooled
  re-extraction is the principled way to retest.

---

## In one paragraph

We turn each chain-of-thought into a trajectory of activation vectors, train a model to
predict the next step *in latent space* (the JEPA idea), and study the **residual** — the
unpredictable part of each step — as a geometric signal. JEPA theory (predict-in-latent,
energy-based, collapse/anti-collapse) says *what* to build; manifold theory (intrinsic
dimension, curvature) says *what to measure*; resampling statistics (chain-grouped CV, AUC,
permutation nulls) say *whether to believe it*. Verdict: reasoning trajectories are genuinely
predictable and low-dimensional, and how unpredictable a chain is weakly but significantly
tracks correctness — a magnitude effect, largely linear, extending prior per-token work to
the reasoning-step level under unusually strict null discipline.
