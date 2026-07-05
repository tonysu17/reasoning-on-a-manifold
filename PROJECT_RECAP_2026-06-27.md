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
## A. Research Question, Theory & the Behaviour Taxonomy

*Refresher source files: `THEORY.md`, `README.md` (intro/framing), `PLAN_THESIS_WRITING.md`, `src/task_gen.py`, `01_generate_tasks.py`. Everything below is quoted from those files; where I extrapolate I say so.*

---

### A.1 The core claim — "reasoning is a geometric process"

The thesis spine is stated baldly at the top of `THEORY.md` (Part I):

> The thesis claims **reasoning is a geometric process**: when a model reasons, each step of its chain-of-thought corresponds to a point in a high-dimensional activation space (the residual stream, here 1536-D), and those points have structure — they lie near a low-dimensional surface and move in patterned ways.

Two things are worth pinning down because they govern every downstream design choice:

1. **Locus = the residual stream.** A "reasoning state" is the transformer's running working memory at a token; `THEORY.md` Part IV step 3 justifies the choice: *"pool the residual-stream activations of each span into one 1536-D vector at a chosen layer. The residual stream is the transformer's running working memory — where the reasoning state lives."* The model is `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (28 layers, hidden dim 1536, README §intro), so the ambient space is fixed at 1536-D throughout.

2. **The claim is split into a static and a dynamic half.** The *standing* thesis (the chapters that already have citable results) is **descriptive/static**: it measures the geometry of *clouds* of step-points — effective dimension, curvature, anisotropy. The **predictive-geometry extension** (the branch you are on, `predictive-geometry-of-reasoning`) adds the *dynamical* half:

> If reasoning is a geometric *process*, the *motion* through this space should be partly **predictable**, and the **failures of prediction** … should be **diagnostic** of branching, confusion, or error. (`THEORY.md` Part I)

`THEORY.md` notes this dynamical framing is not bolted on after the fact — it fills a hook the thesis already names: *"ch07 relocates per-behaviour curvature 'as an open question, to the trajectory,' and ch02 commits to 'locus-in-process' (the unfolding trajectory, not the static point, bears the reasoning)."* This matters for the steering decision (§A.6): the static geometry says *where* the behaviour lives; the trajectory says *how a chain moves through it*.

---

### A.2 Why a "manifold" / subspace framing

The geometric language is not decoration — it is licensed by the **manifold hypothesis** (`THEORY.md` Part II):

> High-dimensional data concentrates near a much lower-dimensional curved surface — a *manifold* — in the ambient space. A 1536-D activation does not roam freely; it lives near a surface of intrinsic dimension maybe 4–40. Geometry is the right language *because* the data is manifold-structured.

The empirical payoff quoted in `THEORY.md` Part III/VIII is that the step-**displacement** cloud has correlation dimension **≈ 3.7–3.9** — "reasoning steps move on a ~4-D surface" — and the predictor is genuinely predictable (R² ≈ 0.3 cross-chain). So the manifold framing earns its keep on the *dynamics* even where the static curvature claim failed.

**Critical caveat baked into the framing itself.** The thesis-writing plan (`PLAN_THESIS_WRITING.md` T1) is emphatic that the per-behaviour object must now be described as a **low-dimensional (≈linear) subspace, NOT a "curved manifold"**:

> Per-behaviour curvature is a **clean, well-powered negative** (it was within-chain autocorrelation; collapses to ≈flat at one-sentence-per-chain). Frame curvature as a **trajectory-level** property … Note the upside: flatness *justifies* the linear steering apparatus (resolves the CF-5 instrument/claim mismatch).

So the word "manifold" survives in the project title and in the *dynamics*, but for the static per-behaviour object the honest word is **subspace** (a top-k PCA subspace — the same operationalisation Huang et al. use for overthinking). This is a load-bearing distinction for the steering experiment: a *linear* steering operator is only defensible because the per-behaviour structure is approximately flat. Calling it "curved" anywhere would re-open the instrument/claim mismatch (CF-5).

---

### A.3 The four reasoning behaviours

The four behaviours under study are inherited verbatim from **Venhoff et al. (ICLR 2025 Workshop)** — *Understanding Reasoning in Thinking Language Models via Steering Vectors* — quoted in `README.md` §intro:

> identified distinct reasoning behaviours in thinking LLMs (**backtracking, uncertainty estimation, example testing, knowledge augmentation**) and showed each can be **controlled by a single linear steering vector**.

In the repo's own naming (used for the steering-vector `.npy` files in `results/steering_vectors/R1-1.5B/`) the four are:

- **adding-knowledge** (Venhoff's "knowledge augmentation") — injecting a fact/lemma/identity the model treats as known.
- **backtracking** — abandoning a line and pivoting ("wait, that's wrong, let me…").
- **example-testing** — trying a concrete instance to probe a general claim.
- **uncertainty-estimation** — hedging / flagging doubt about its own state.

Annotation (`THEORY.md` Part IV step 2) operationalises a "reasoning step" as *one sentence-level span labelled with a behaviour*: *"an LLM annotator splits each chain into sentence-level **spans**, each labelled with a behaviour. A 'reasoning step' = one labelled span."* The annotator runs on the **AWS Bedrock proxy** with **Claude Sonnet 4.5 primary** (Venhoff used GPT-4o, unavailable on the proxy — README §credentials notes the deviation), plus **Qwen3-235B** and **Nova-Pro** as robustness annotators (`PLAN_THESIS_WRITING.md` T4/T5).

**Two behaviour-level results that constrain the steering experiment:**

- **Behaviour-specificity is 2/4, not blanket** (`PLAN_THESIS_WRITING.md` T2): *"backtracking & uncertainty-estimation are behaviour-specific at all layers (chain-stratified variance-ratio null, B=2500, p<.001); **example-testing only at L27**; **adding-knowledge fails everywhere (p=1.0)**."* adding-knowledge is also the rarest / highest-dim / weakest — "the honest outlier."
- **No discrete sub-types within a behaviour** (`README.md` §intro): the original "discrete flavours" hypothesis (e.g. arithmetic re-checking vs strategy pivoting) was *not supported* by sub-type clustering (Phase 5d: k=2, silhouettes 0.11–0.18) → "the working picture is continuous low-dimensional structure per behaviour."

---

### A.4 Why *these four* behaviours and not others — the taxonomy

`THEORY.md`/`README.md` themselves do not derive the four from first principles — they inherit them from Venhoff. The principled defence (the **Operation × Control × Content × Program** factorisation requested in this assignment) lives in the project's reasoning-taxonomy work rather than in these specific files, so here is the faithful reconstruction:

- **Operation** — the cognitive *act* (retrieve, test, revise, assess-confidence).
- **Control** — whether the act steers the *flow* of the chain (backtracking and uncertainty-estimation are **control-stratum**: they decide whether to continue/abandon) vs operates *within* a line of work.
- **Content** — what the act is *about* (a fact, an instance, the model's own state).
- **Program** — the higher-order structure / composition the act participates in.

Under that lens the four are **not a flat basis**: they span **two strata**. backtracking and uncertainty-estimation are *control* operations over the reasoning flow; example-testing and adding-knowledge are *content* operations that supply or probe material. This two-strata picture is exactly what the 2/4-specificity result (§A.3) reflects empirically — the *control* behaviours (backtracking, uncertainty) are clean and behaviour-specific everywhere, while the *content* behaviours are weaker (example-testing only late at L27, adding-knowledge nowhere). The honest research niche named in the taxonomy work is **composition** (behaviour × domain-content), with simulation / deontic-evaluation / analogy flagged as candidate *residual* behaviours the four don't cover.

**Why not more behaviours?** The pragmatic answer is reproducibility and synthesis: using Venhoff's exact four (a) lets the project inherit a published, single-direction-controllable set, and (b) makes the experiment "nobody else is positioned to run" possible — decomposing **Huang et al.'s** composite *overthinking* direction (same model, same layer 27, published top-k-PCA k=10 recipe) into per-behaviour subspaces (`README.md` §intro, *"which behaviours mediate overthinking mitigation?"*). Adding novel behaviours would forfeit both anchors.

**Skeptic's note.** The taxonomy is *asserted*, not *measured*, in these files. `PLAN_THESIS_WRITING.md` T13 concedes the gap: there is **no between-behaviour separability metric** (ARI / kNN-purity) in the repo — "the four behaviours are mutually distinct" currently leans only on the per-behaviour probes + the variance-ratio null, both single-pipeline. The factorisation is a *framing*, and a reviewer could fairly ask why these four cut nature at the joints rather than, say, collapsing backtracking and uncertainty into one "control" axis.

---

### A.5 The rung/ladder of hypotheses (H1/H2/…, Gate-0)

There are **two distinct ladders** in the project — do not conflate them.

**Ladder 1 — the standing *structure* claim (`README.md` §intro), the "three-rung gap":**

1. **Rung 1 — a single direction suffices for control.** Established by Venhoff; sufficiency, not exhaustiveness.
2. **Rung 2 — the behaviour occupies a multi-dimensional subspace.** Huang established this for *composite* overthinking; whether each *individual* behaviour has its own low-dimensional behaviour-specific subspace, and whether subspace-projected steering beats the single direction *per behaviour*, is open. **This is the project's primary claim, testable with linear instruments.**
3. **Rung 3 — the structure is curved beyond any linear subspace.** Open for reasoning behaviours; the README is explicit that *"our current steering apparatus is linear, so rung-3 results here are descriptive, not causal (CF-5)."*

**Gate-0** is the validity gate that had to pass before any of this is citable. Per the memory/confounds register (referenced but not in these files): Gate-0 = "is there *any* real low-dimensional, behaviour-specific geometry, or is it an artefact of estimators / duplicate rows / chain autocorrelation?" Its verdict (folded into `PLAN_THESIS_WRITING.md` T1–T4): **subspace ✔ (low-dim survives), curvature ✘ (clean negative — a chain/autocorrelation artefact, collapses to flat at one-sentence-per-chain), specificity MIXED 2/4.** The keystone confound was the chain confound (CF-2).

**Ladder 2 — the predictive-geometry escalation ladder (`THEORY.md` Part VII), "build the cheap thing first":**

1. **Rung 0** — predictor-free raw-trajectory Frenet curvature vs correctness.
2. **Rung 1** — linear ridge predictor `x_t → (x_{t+1}−x_t)`, chain-grouped OOF, *no* anti-collapse regulariser ("the scientifically central, uncontaminated rung").
3. **Rung 2** — small JEPA (1-hidden-layer MLP + Barlow-Twins/SIGReg), justified only if it beats Rung 1.
4. **Rung 3 — causal steering (not built)** — the apex; theory = Model-Predictive Control (latent rollouts scored against a goal embedding via Cross-Entropy Method, à la V-JEPA 2).

The behaviour-level hypotheses referenced as **H1/H3** are the predictive-geometry hypotheses tested with permutation nulls (`THEORY.md` Part VI/VIII): **H1** (unpredictability ↔ error) is *partly supported* (label-permutation passes, p≈0.03 at best layer, AUC≈0.58); **H3** (the correctness signal needs *trajectory order*) is *refuted* (step-shuffle fails, p>0.7 — the signal is overall surprise **magnitude, not trajectory shape**). The memory index also references **H1/H2/H4** as the standing thesis's structure hypotheses with "H1 gates H2"; those map onto Ladder-1's single-direction → subspace → curvature progression.

---

### A.6 The two "movements" (Structure + Origin/Safety)

The thesis is organised into **two movements** with **post-training as the hinge**, per the locked spine (memory `thesis_unifying_theme`; the title is *"The Geometry of Machine Reasoning: Per-Behaviour Structure and the Post-Training Origin of Safety Reasoning"*):

- **Movement 1 — Structure.** The per-behaviour geometry: do the four behaviours occupy distinct low-dimensional subspaces, and can subspace-projected steering beat a single direction? This is Ladder-1 above (Rungs 1–2). Knowledge-creation is **deferred** (future philosophy).
- **Movement 2 — Origin / Safety.** *Where* safety reasoning comes from — the claim that safety/deliberative-alignment reasoning is installed by **post-training**, with the "forgery jailbreak" as the climax (`gpt-oss-20b` extension). Post-training is the hinge: *"base models know how, thinking models learn when."*

The steering experiment sits squarely in **Movement 1** and is the bridge: it is the *causal* test of the structure claim. `PLAN_THESIS_WRITING.md` T12 even proposes tying it to the hinge — a **base ↔ distilled ↔ ±steering continuum** (base Qwen2.5-Math-1.5B behaviour-poor < negative-steered distilled < vanilla distilled < positive-steered distilled), so that "negative steering ≈ un-learning the *when*" connects Movement 1's steering to Movement 2's post-training thesis. Caveat (stated there): base and distilled are *different weights*, so this is a reference frame, not a within-model intervention.

---

### A.7 How Phase 1 (task generation) instantiates the theory

The actual code in scope is the **task-generation** front of the pipeline. `01_generate_tasks.py` is a thin CLI over `src/task_gen.py`; it produces **1000 tasks = 100 × 10 categories** via Claude **Sonnet 4.5** on the proxy (`_MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"`).

Note the **ten task categories are about diversity of *reasoning content*, NOT the four behaviours** — the behaviours are annotated later (Phase 3) inside whatever chains these tasks elicit:

```python
# src/task_gen.py — CATEGORIES (module level)
CATEGORIES: dict[str, str] = {
    "mathematical_logic": "Problems requiring formal logic, proofs, and mathematical reasoning",
    "spatial_reasoning": "Tasks involving spatial relationships, geometry, and visualisation",
    "verbal_logic": "Syllogisms, verbal analogies, and language-based reasoning",
    "pattern_recognition": "Identifying and continuing abstract sequences or patterns",
    "lateral_thinking": "Problems requiring creative, non-linear approaches",
    "causal_reasoning": "Cause-and-effect relationships and causal inference",
    "probabilistic_thinking": "Uncertainty, probability, and statistical reasoning",
    "systems_thinking": "Complex systems, interdependencies, and emergent behaviour",
    "creative_problem_solving": "Open-ended problems requiring novel approaches",
    "scientific_reasoning": "Hypothesis formation, experimental design, evidence evaluation",
}
```

The prompt is engineered to *elicit extended multi-step chains* (so there is something to segment into behaviour spans). The key design constraints live in `_USER`:

```python
# src/task_gen.py — _USER (prompt template)
Requirements:
- Each task requires at least 3–5 reasoning steps (not answerable in one sentence)
- Self-contained (no external resources, links, or images needed)
- Mix of moderate and hard difficulty
- Do NOT include the answer, solution hints, or worked examples
```

**Why each choice (defended as the authors would):**
- *"≥ 3–5 reasoning steps / not one sentence"* — the whole apparatus needs multi-span chains; one-sentence-answerable tasks would yield no trajectory.
- *"Self-contained, no external resources"* — keeps the chain a closed reasoning object so activations are attributable to internal reasoning, not retrieval.
- *"Do NOT include the answer / hints"* — prevents the model from copying a solution; the chain must be *generated*, which is what makes correctness a meaningful latent variable.
- *"Mix of moderate and hard"* — supplies the difficulty stratum used later by the permutation null (label-permutation is done *within difficulty strata*, `THEORY.md` Part VI) so the correctness signal isn't just a difficulty proxy.

To avoid the model repeating canonical puzzles across batches, already-generated prompts are fed back as a blocklist context:

```python
# src/task_gen.py — generate_tasks (inner batch loop)
# Pass tasks already generated in this category as context so the
# model does not repeat the same classic problems across batches.
context = [t["prompt"][:120] for t in cat_tasks] if cat_tasks else None
batch = _call_api(cat_name, cat_desc, prefix, start, n_this,
                  proxy_url, proxy_key,
                  context_summaries=context)
```

#### The single most consequential function for the steering experiment

`stratified_eval_split` is the **hold-out definition**, and it is flagged in-code as the single source of truth shared by the steering-vector builder and the evaluator:

```python
# src/task_gen.py — stratified_eval_split
def stratified_eval_split(tasks: list[dict], n_test: int = 50) -> tuple[list[dict], str]:
    """Category-stratified evaluation split: the last n_test/n_categories
    tasks of EACH category.

    SINGLE SOURCE OF TRUTH for the Phase-7 eval set. Both the steering-vector
    builders (which must EXCLUDE these tasks' activation rows for the vectors
    to be a true hold-out) and 07_evaluate_steering (which evaluates on them)
    call this function — any drift between the two silently breaks the
    hold-out. tasks_final.json is perfectly category-blocked, so the naive
    `tasks[-n_test:]` rule selected 50 tasks of a single category.
    """
```

The docstring records a real bug that was fixed: because `tasks_final.json` is perfectly category-blocked, the naïve `tasks[-n_test:]` rule had selected 50 tasks from a *single* category. The fix takes the last `per_cat = n_test // n_categories` of *each* category. **For the steering decision this is the function that guarantees the steering vectors are a true hold-out** — if the builder and `07_evaluate_steering` ever compute the split differently, the hold-out silently leaks and any "steering beats random on held-out tasks" claim is contaminated. Worth re-verifying both call sites before spending API+GPU.

---

### A.8 Confounds, fragilities & RUN/UNRUN status (skeptic's ledger)

- **No geometry number is currently citable in isolation** (`README.md` §known-limitations): the 2026-06-05 audit fixed biased estimators (TwoNN, curvature ratio) and 2026-06-12 fixed a **duplicate-row bug (35–56% exact-duplicate activation rows** from first-occurrence sentence matching). Every pre-fix geometry output is **quarantined** in `results/_STALE_pre_fix_20260605/`. So §A.2's "≈3.7–3.9 / R²≈0.3" come from the *predictive-geometry* re-run, not the stale static pipeline.
- **~50% chain truncation** (`README.md`): 50.2% of R1-1.5B chains hit the 8192-token cap; 49.9% lack a closing `</think>`, concentrated in `lateral_thinking` (95% at cap), `spatial_reasoning`, `pattern_recognition`, `probabilistic_thinking`. A truncated chain is a *censored* trajectory — its terminal "surprise" / curvature is suspect, and truncation correlates with category, which correlates with difficulty, which correlates with correctness. The policy is to stratify by a `truncated` flag, but the steering eval set is drawn from these same categories, so the held-out tasks inherit the truncation skew.
- **Curvature is a *negative*, well-powered** (`PLAN_THESIS_WRITING.md` T1) — do not let "curved manifold" language back into ch02/ch05/ch07. The flatness is the *good news* that licenses the linear steering operator.
- **Behaviour-specificity is 2/4** (T2): citing "the four behaviours are each behaviour-specific" is false. adding-knowledge fails everywhere (p=1.0); example-testing only at L27. A steering result on adding-knowledge therefore has *no clean structural backing* — expect it to be the weakest arm and frame it as the honest outlier.
- **Annotator circularity / noise.** Behaviour labels come from an LLM (Sonnet), and correctness is *also* an LLM-judge latent variable (κ 0.35–0.44 inter-annotator agreement; `THEORY.md` Part VII). The 2-way Sonnet↔Qwen3 replication (T4) is reassuring (geometry replicates despite κ=0.44) but ran on analysis-side dedup, not clean re-extraction, and the **variance-ratio specificity-null replication and the Nova-Pro third arm are still pending**.
- **Generation-model ≠ annotation-model ≠ task-model**, but **task generation, behaviour annotation, and the planned de-circularised judge can all be Claude** — a monoculture risk for "is the geometry an artefact of one model family?" (partly mitigated by Qwen3/Nova robustness arms, but only on the *annotation* axis, not task generation).
- **RUN vs UNRUN.** Phase 1 (this section) is RUN and stable (`tasks_final.json` exists, category-blocked, deduped — README Phase 1.5). The **standing structure geometry** is RUN but needs the post-fix re-run to be citable (subspace claim survives). The **predictive-geometry ladder** Rungs 0–2 are RUN (pilot). **Phase 7 steering is UNRUN** — this is the spend you are about to commit; no behaviour-fraction numbers exist and all Phase-7 thesis numbers are behind the `\phaseSeven` embargo macro (`PLAN_THESIS_WRITING.md` header).

---

### A.9 Connection to the steering decision you are about to make

The theory constrains the imminent **layer + methodology** choice in concrete ways:

1. **Methodology is *linear* and that is now *defended*, not a limitation.** Because per-behaviour curvature collapsed to a clean negative (≈flat subspace, §A.2/T1), a linear projective steering operator is the *matched* instrument (resolves CF-5). The two arms are **single-direction** (Venhoff diff-of-means, `r = mean(ON) − mean(OFF)` where OFF = the other three behaviours, unit-norm) and **manifold-projected** (Huang-adapted: project `r` onto the behaviour's own top-k PCA subspace), applied projectively `h' = h − α·(rᵀh)·r`, against a **norm-matched random-direction floor** that licenses causal language (`PLAN_THESIS_WRITING.md` T6/T7). The whole Rung-1 vs Rung-2 question (single direction vs subspace) *is* the headline of Movement 1.

2. **Layer choice is currently *descriptive*, not causal** (`PLAN_THESIS_WRITING.md` T9). The default L27 is Huang's overthinking layer / a participation-ratio trough — **not** a held-out causal choice. The empirical signal is that *more semantic* behaviours are specific at *later* layers (example-testing only at L27, adding-knowledge nowhere), consistent with complex behaviours emerging deeper. The principled fix is **attribution patching** per behaviour; until then, do **not** claim the layer is held-out. The build recommendation in memory is mid-layers (≈11/16/19) **and** L27, letting Phase 7 decide.

3. **Pooling is a hidden knob.** Activations are mean-pooled over the first ~10 tokens of each span — a *deviation from standard last-token steering* — and the direction is pooling-dependent: cos(mean, last) = **0.57–0.87** (backtracking worst at 0.57; T8/CF-6). Phase 7 must **sweep mean vs last** and report sensitivity; the H3 "order doesn't matter" negative is itself caveated as possibly an artefact of mean-pooling washing out within-step structure (`THEORY.md` Part VIII).

4. **The hold-out is defined by `stratified_eval_split`** (§A.7). Verify the steering-vector builder *excludes* those rows and that `07_evaluate_steering` evaluates on exactly that set before spending — drift here silently breaks the only thing that makes "beats random" a real result.

5. **k must be swept**, not fixed: `k ∈ {1,3,5,10,auto}` (the `.npy` files already exist for each k per behaviour), with an α-sweep {0.3…3.0}; the headline is *effect at matched damage* (suppression-vs-damage Pareto), paired over held-out tasks, both arms beating random (T7).
## B. Data Generation — Tasks, Chains, Pilot Gate, Baselines

This section documents the **input-manufacturing** stage of the pipeline: how the
task corpus is produced (Phase 1), how reasoning chains-of-thought are sampled
from the model under study (Phase 2), the pilot gate intended to de-risk the
expensive scale-up (`00_pilot_gate.py`, `validate_pilot_lengths.py`), the
non-reasoning **baseline/control** corpus (Phase 2b) that exists specifically to
neutralise the chain confound **CF-2**, and the QC / model-identity verifiers
(`check_chain_quality.py`, `verify_base_model.py`). Everything downstream —
annotation, activation extraction, the per-behaviour subspaces, and the steering
vectors the researcher is about to spend real money building — inherits whatever
biases are baked in here. The headline critique, established with the actual
on-disk artefacts, is that **the corpus is 50.2% truncated at the token ceiling,
the pilot gate that should have caught this either never ran or was overridden,
and the CF-2 baseline (Phase 2b) has not been generated at full scale at all.**

---

### B.1 Phase 1 — Task generation (`01_generate_tasks.py`, `src/task_gen.py`)

**What it does.** It asks Claude (`anthropic.claude-sonnet-4-5-20250929-v1:0`,
via the lab proxy, NOT the R1 model) to write a balanced corpus of **1000
reasoning tasks = 100 tasks × 10 hand-defined categories**. The categories are a
fixed taxonomy of *reasoning domains* (note: domains, not the Venhoff *behaviour*
labels that get annotated later):

```python
# src/task_gen.py — CATEGORIES (top of module)
CATEGORIES: dict[str, str] = {
    "mathematical_logic":   "Problems requiring formal logic, proofs, and mathematical reasoning",
    "spatial_reasoning":    "Tasks involving spatial relationships, geometry, and visualisation",
    "verbal_logic":         "Syllogisms, verbal analogies, and language-based reasoning",
    "pattern_recognition":  "Identifying and continuing abstract sequences or patterns",
    "lateral_thinking":     "Problems requiring creative, non-linear approaches",
    "causal_reasoning":     "Cause-and-effect relationships and causal inference",
    "probabilistic_thinking":"Uncertainty, probability, and statistical reasoning",
    "systems_thinking":     "Complex systems, interdependencies, and emergent behaviour",
    "creative_problem_solving":"Open-ended problems requiring novel approaches",
    "scientific_reasoning": "Hypothesis formation, experimental design, evidence evaluation",
}
```

**Why these choices.**
- **Generator ≠ subject model.** Tasks come from a strong frontier model so the
  prompts are diverse, self-contained, and genuinely multi-step (the system/user
  prompt demands "at least 3–5 reasoning steps", "self-contained", and
  explicitly "Do NOT include the answer, solution hints, or worked examples").
  This keeps the *stimulus* distribution decoupled from the *subject* (R1-1.5B),
  so the chains are real model behaviour rather than echoes of a leaked solution.
- **De-duplication via in-context blocklist.** Within a category, each batch is
  generated with the prior prompts (first 120 chars) injected as a "do not
  repeat" context, so the model does not keep re-emitting the same textbook
  classics:

```python
# src/task_gen.py — generate_tasks() inner loop
context = [t["prompt"][:120] for t in cat_tasks] if cat_tasks else None
batch = _call_api(cat_name, cat_desc, prefix, start, n_this,
                  proxy_url, proxy_key,
                  context_summaries=context)
cat_tasks.extend(batch)
time.sleep(0.5)
```

- **Sampling temperature 0.8** in `_proxy_call` (diversity is wanted here, the
  opposite of the chain stage), with a 3-attempt retry + exponential backoff and
  a `json.loads` parse guard that strips ```` ``` ```` fences.

**The eval-split single-source-of-truth.** `stratified_eval_split()` is the one
function both the steering-vector builder and `07_evaluate_steering` must call so
the Phase-7 hold-out is identical on both sides. Its own docstring flags the
landmine that motivated it:

```python
# src/task_gen.py — stratified_eval_split()
# tasks_final.json is perfectly category-blocked, so the naive
# `tasks[-n_test:]` rule selected 50 tasks of a single category.
per_cat = max(1, n_test // len(by_cat))
test_tasks = [t for cat in sorted(by_cat) for t in by_cat[cat][-per_cat:]]
return test_tasks, f"last {per_cat} per category"
```

This matters directly for the steering decision: if the vector builder and the
evaluator ever drift on this rule, the "hold-out" silently leaks and any steering
effect is contaminated.

**Critique.**
- The 10 task categories are an *a priori* convenience taxonomy, not validated
  against anything; the actual scientific unit later is the Venhoff behaviour
  label, which is orthogonal. Category balance (100 each) does **not** guarantee
  behaviour balance — and indeed `adding-knowledge` is the behaviour that keeps
  failing specificity downstream.
- The corpus on disk is `data/tasks_final.json` (1000 tasks), produced after a
  dedup/cleanup pass (`data/tasks.json → tasks_deduped → tasks_500balanced →
  tasks_final`). The 500-balanced and deduped intermediates show the corpus was
  reworked; the pilot scripts still point at the *stale* `data/tasks.json` (see
  B.3), a provenance seam.

---

### B.2 Phase 2 — Chain generation (`02_generate_chains.py`, `src/chain_gen.py`)

**Subject model.** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (alias `1.5b`,
short name `R1-1.5B`; 28 layers, hidden 1536). The prompt is delegated to
`src/model_adapters` so the same harness drives DeepSeek `<think>` CoT, gpt-oss
harmony, and non-thinking bases — but the DeepSeek default is the path used for
the real corpus.

**Sampling = greedy (temperature 0).** This is the load-bearing methodological
choice and it follows Venhoff et al.:

```python
# src/chain_gen.py — generate_chain()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else 1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
```

With `temperature=0.0`, `do_sample=False` → deterministic greedy decoding. The
docstring is explicit that **the seed is a no-op under greedy** ("Greedy decoding
(T=0) ignores the RNG state entirely"). The multi-seed plumbing (`--seeds`,
`dedup_keys=("task_id","seed")`, `*_multiseed.json`) only bites when someone
re-runs with `temperature>0` for the §M4.5 robustness check.

**Generation budget = 8192 tokens** (`--max-tokens 8192`, overriding the
library default of 2048). Records are written with a schema that stores
`prompt`, `chain`, and `full_text = prompt + chain` so Phase 4 can reconstruct
*exact* token positions for activation extraction. Generation is checkpointed
(atomic tmp+rename) every 10 chains and is resume-safe; a batched variant
(`generate_chains_batched`, left-padded for decoder-only correctness, per-task
fallback on OOM) exists for throughput on shared GPUs.

**Why greedy.** Determinism makes the chain a *fixed* object: the same task
always yields the same chain, so activation extraction, annotation, and steering
all operate on one canonical artefact, and the "geometry" is not an artefact of
sampling noise. It also matches the comparison literature so behaviour fractions
are comparable to Venhoff Fig. 2.

**Critique — the truncation problem (the single most damaging finding here).**
The actual on-disk corpus `data/chains_R1-1.5B.json` (1000 chains) has:

- **mean = 5052 tokens, median = 8192, P95 = 8192, max = 8192**
- **502 / 1000 chains (50.2%) sit exactly at the 8192 ceiling**
- **only 501 / 1000 chains contain a closing `</think>`**

So **half the corpus is truncated mid-reasoning.** This is not a cosmetic data
issue — it is a structural confound for *this very project*:
1. Truncated chains have systematically *missing* late-CoT behaviours.
   Backtracking and uncertainty-estimation tend to cluster late in a chain; if
   half the chains are cut off, the behaviour fractions and the late-token
   activations are biased toward whatever fits in 8192 tokens.
2. It directly couples **behaviour** to **length/position** — which *is* CF-2
   (below). A "backtracking" subspace learned partly from where-in-the-chain a
   truncation lands is suspect.
3. `check_chain_quality.py` is built precisely to surface this — its truncation
   cross-tab separates "hit max AND no closing think (truncated mid-thinking)"
   from "short AND has closing think (clean finish)" — but a report existing is
   not the same as the corpus being clean.

---

### B.3 The pilot gate — designed, then bypassed (`00_pilot_gate.py`, `validate_pilot_lengths.py`)

**Intent.** Run a stratified 20-chain pilot (2 tasks × 10 categories) through
Phases 2–3 and check **5 gates** before authorising the full (expensive) run:

```text
# 00_pilot_gate.py docstring — Checks (all must pass before full Phase 2)
1. All 20 chains end with </think>                (no truncation)
2. Mean chain length ≤ 2,500 tokens               (cost/time bound)
3. Annotation parses into [(label, text), ...]    (format correct)
4. All 6 Venhoff labels appear ≥ 1× across 20 chains
5. Sentence fractions roughly match Venhoff Fig. 2 (±10 pp tolerance)
```

The expected Venhoff fractions are hard-coded (`deduction 0.52`,
`adding-knowledge 0.15`, `uncertainty 0.09`, `initializing 0.07`,
`example-testing 0.06`, `backtracking 0.04`) and used as the target distribution.

**It never ran as written, and it is now hard-disabled.** The very first line of
`main()` is a hard exit:

```python
# 00_pilot_gate.py — main()
sys.exit(
    "00_pilot_gate.py is HISTORICAL and cannot run: its annotation "
    "subcommands import batch-API functions that never existed in "
    "src/annotation.py, and it reads the stale data/tasks.json. The pilot "
    "ran via `03_annotate_chains.py --pilot`; use check_chain_quality.py "
    "and verify_annotation_completeness.py for the corpus-level checks. "
)
```

So Checks 1–5 were **never executed programmatically.** The docstring concedes
the corpus-level equivalents migrated to `check_chain_quality.py` and
`verify_annotation_completeness.py`. This is honest bookkeeping, but it means the
gate that was supposed to *prevent scale-up on a bad config did not gate
anything.*

**The length-only gate that DID run — and what it should have said.**
`validate_pilot_lengths.py` is a standalone check with a sharp decision rule:

```python
# validate_pilot_lengths.py — decision rule
if p95 < 6500 and n_ceiling <= 2:
    print(f"PASS: P95={p95:.0f} < 6500 and {n_ceiling}/20 hit ceiling")
    ...
else:
    print(f"STOP: P95={p95:.0f}, {n_ceiling}/20 hit ceiling — report before proceeding")
    sys.exit(1)
```

Running its logic against the **actual** pilot file `data/chains_pilot.json`:

- pilot **P95 = 8192**, **11 / 20 chains hit the 8192 ceiling**, mean ≈ 5082.

That is a **massive STOP** (rule fires on either `P95 ≥ 6500` *or* `≥3/20 at
ceiling`; here it is 8192 and 11/20). The pilot already showed that >half of
chains would truncate at 8192 — and the full run was generated anyway, producing
exactly the 50.2% truncation observed in B.2. **The pilot correctly predicted the
problem; the STOP was not honoured.** (Two further fragilities: the pilot file
lacks a `difficulty` field, so the script's `c['difficulty']` print would raise
`KeyError` before reaching the decision line; and `00_pilot_gate.py`'s Check 2
target of ≤2500 tokens is wildly inconsistent with the 8192 budget actually used
— the gate's own thresholds were never reconciled with the run config.)

**Critique.** The original pilot Check 1 ("all 20 chains end with `</think>`")
would have *failed outright* on the real pilot (only a fraction terminate
cleanly). The remediation — moving checks into corpus-level QC scripts — converts
a *blocking* gate into a *descriptive* report, which is exactly the wrong
direction for de-risking spend. For the steering experiment the lesson is
concrete: **before spending on Phase 7, re-run the chain corpus with a higher
ceiling (or budget-adaptive stopping) so the activations feeding the steering
vectors are not drawn from a 50%-truncated, length-confounded population.**

---

### B.4 Phase 2b — Baseline / control chains, and the CF-2 confound (`02b_generate_baseline_chains.py`)

**Why this script exists — CF-2 stated.** The central claim of the thesis is that
specific *reasoning behaviours* (backtracking, uncertainty-estimation,
example-testing, knowledge-augmentation) each occupy their own low-dimensional
subspace. **CF-2 is the objection that any apparent "behaviour geometry" could
instead be a geometry of chain *length* or token *position*** — long elaborate
chains differ from short ones in activation space for reasons that have nothing
to do with the behaviour label. Phase 2b builds the control that is supposed to
break that alternative explanation: the *same model family without the reasoning
post-training*, on the *same tasks*, with an *identical output schema*, so that
"how much behaviour is *added* by R1 distillation" can be measured rather than
assumed.

**The baseline model.** `Qwen/Qwen2.5-Math-1.5B` — chosen because
`verify_base_model.py` *empirically* established it is the true base of
R1-Distill-Qwen-1.5B (embed cosine **0.9936** and lm_head **0.9726**, vs only
**0.84 / 0.85** for the Instruct variant; aggregate delta 0.098 vs 0.192). The
verdict file on disk reads "**Likely:** `qwen-math` is the base; margin over
runner-up is 0.0937." This is the correct control: same weights modulo the
distillation delta, so a base-vs-distilled contrast isolates what RL/distillation
added.

**The construction — no chat template, no `<think>`, raw Q/A scaffold:**

```python
# 02b_generate_baseline_chains.py — format_baseline_prompt()
def format_baseline_prompt(instruction: str) -> str:
    """Q/A scaffold for non-reasoning base models.
    Matches the typical math chain-of-thought corpus format that
    Qwen2.5-Math was pretrained on. NO chat template, NO <think>.
    """
    return f"Question: {instruction}\n\nAnswer:"
```

**Schema parity is deliberate.** The records are byte-for-byte the same shape as
Phase 2 (`task_id, category, instruction, prompt, chain, full_text, n_tokens`)
"so downstream phases (annotation, activation extraction, PCA, steering) work
unchanged via `--model-short QwenMath-1.5B`." Generation is greedy
(`do_sample=(temperature>0)`), checkpointed, resume-safe, and snapshots the
existing file before touching it (a guard 02b "previously didn't" have).

**The token-budget defence (and why it is a CF-2 fix in itself).** The
`--max-new-tokens` default was raised to 8192 with an explicit rationale that is
itself a CF-2 argument:

```python
# 02b_generate_baseline_chains.py — main() arg help
"Generation budget (default: 8192 — MUST match 02_generate_chains.py's cap:
 base Qwen-Math models repetition-loop, and truncating the baseline 4x sooner
 than R1 biases every length/completeness-sensitive base-vs-distilled
 comparison. Baselines are typically short, so the higher cap rarely costs
 anything.)"
```

i.e. if the baseline were truncated at a *different* length than R1, the
base-vs-distilled contrast would be polluted by exactly the length artefact CF-2
warns about — so the caps are forced equal.

**Critique — does 02b actually neutralise CF-2? Partially, and it is UNRUN.**

1. **It is not generated at full scale.** On disk there is only
   `data/chains_QwenMath-1.5B_smoke.json` (**20 tasks**); there is **no**
   `data/chains_QwenMath-1.5B.json`. The CF-2 control has been *designed and
   smoke-tested but never produced for the 1000-task corpus*, so no
   base-vs-distilled behaviour-fraction or geometry comparison has actually been
   computed from it. This is the sharpest gap: **the confound's antidote does not
   yet exist as data.**
2. **A control on a different stimulus format is itself confounded.** The
   baseline uses a `"Question:…\n\nAnswer:"` scaffold while R1 uses the `<think>`
   chat template. So base-vs-distilled differs in (a) post-training **and** (b)
   prompt format simultaneously. Any geometry difference cannot be cleanly
   attributed to the reasoning post-training alone — the prompt is part of the
   treatment. The authors' defence is that the scaffold matches Qwen-Math's
   pretraining corpus (so the base is being used *in distribution*), which is
   reasonable but does not fully de-confound.
3. **CF-2-within-R1 is untouched by 02b.** 02b addresses the *base-vs-distilled*
   axis. It does **nothing** for the within-R1 worry that a behaviour subspace is
   really a length/position subspace — and given B.2's 50.2% truncation, that
   within-corpus length confound is live and large. Neutralising CF-2 properly
   needs *length-/position-matched* negatives within the R1 corpus, not just a
   non-thinking sibling model.
4. **The baseline may emit almost no annotatable behaviour.** Qwen-Math base is
   expected to produce short, non-deliberative answers; if it has near-zero
   backtracking/uncertainty, the "control" is a floor near zero, which makes the
   "behaviour is added by distillation" claim easy but tells you little about
   whether the *geometry within R1* is behaviour-specific vs length-specific.

---

### B.5 QC and identity verification (`check_chain_quality.py`, `verify_base_model.py`)

**`check_chain_quality.py`** is a pure-function corpus auditor (no I/O in
`quality_check`) that emits a Markdown + JSON report. It checks structural
integrity (missing fields, dup `task_id`s, and crucially
`full_text != prompt + chain` — the exact invariant Phase 4 relies on for token
alignment), per-category token distributions, a **truncation cross-tab**, prompt
integrity (counts prompts containing `<think>`), and content anomalies
(non-ASCII, CJK/kana language-drift, and regex-detected repetition loops). It is
the de-facto replacement for pilot Checks 1–2. Its existence is good practice;
its limitation is that it is *descriptive* — it reports 50% truncation, it does
not block on it.

**`verify_base_model.py`** settles the base-model identity empirically rather
than trusting DeepSeek's (silent) report, using three weight-space probes:
embed/lm_head cosine, and per-layer Frobenius fractional deltas on `q_proj` /
`gate_proj`:

```python
# verify_base_model.py — fractional_delta()
def fractional_delta(A, B) -> float:
    """Frobenius(A - B) / Frobenius(B). How much A diverges from B, relative
    to B's scale."""
    if A.shape != B.shape:
        return float("nan")
    return float(np.linalg.norm(A - B) / np.linalg.norm(B))
```

This is the right way to ground the CF-2 control: the baseline's scientific
validity depends entirely on it really being R1's base, and the script gives a
defensible **0.9936 embed cosine** to back the `configs/config.yaml` claim. (Mild
caveat: the verdict is graded "Likely," not "Confident," because the winner's
aggregate delta 0.098 is just under the 0.10 "likely" threshold and the margin
over the Instruct variant, while clear, is modest in absolute terms.)

---

### B.6 Connection to the steering decision (layer + methodology)

This stage does **not** itself pick a steering layer — `configs/config.yaml`
carries `steering_layer: 27` for R1-1.5B (a Huang-et-al. late-layer choice) and
the baseline mirrors it at 27 "for direct comparability." But Phase B gates the
steering experiment in three concrete ways:

1. **The activations that become steering vectors are drawn from this corpus.**
   If 50.2% of chains are truncated at 8192, the residual-stream rows feeding the
   behaviour subspaces are biased toward early/mid-chain content and against the
   late-chain behaviours (backtracking, uncertainty) the steering experiment most
   wants to manipulate. **Recommendation before spend: regenerate (or
   length-filter/extend) so the steered behaviours are well-represented and not
   length-confounded.**
2. **CF-2 is still open as data.** The base-vs-distilled control (Phase 2b) is
   designed but only smoke-run. A steering result claiming "we moved a
   *behaviour*" is exposed to the reviewer's CF-2 objection — "you moved chain
   length/position" — until the matched-cap baseline corpus exists and a
   length-matched within-R1 control is added. The matched 8192 cap in 02b is the
   right instinct; it just has to actually be run.
3. **The eval hold-out is defined here.** `stratified_eval_split()` is the single
   source of truth the steering builder must exclude and the evaluator must
   score on. Any divergence silently leaks the hold-out and inflates the steering
   effect — verify both call sites use it before committing GPU/API budget.

**Bottom line for the researcher:** the generation machinery is well-engineered
(deterministic, resume-safe, schema-consistent, identity-verified), but two
load-bearing facts undercut citable claims as the corpus stands today — **(i) the
R1 corpus is half-truncated and the pilot's own STOP rule was overridden, and
(ii) the CF-2 control exists only as a 20-task smoke.** Both should be closed
before money is spent on steering.
## C. Annotation — Behaviour Labelling & Multi-Annotator Robustness

This stage takes the raw reasoning chains generated in Phase 2 and decorates every
sentence with a *reasoning-behaviour* label. Those labels are the load-bearing
input to everything downstream: the activation extractor (Phase 4) slices the
residual stream at the token spans these labels mark, the PCA / geometry phases
build per-behaviour subspaces from those slices, and the steering vectors you are
about to spend real money on are difference-of-means over exactly these labelled
populations. If the annotation is noisy or biased, every geometric claim inherits
that noise. This section documents what the code actually does, defends the design
choices, and then attacks the reliability of the labels with the real agreement
numbers now on disk.

---

### C.1 What the stage produces

The pipeline entry point is `03_annotate_chains.py`; the engine is
`src/annotation.py`. For each chain it sends the full thinking text to an LLM
annotator and asks it to re-emit the text wrapped in delimiter tags, one tag per
behaviour span. The parsed output per chain is a list of
`{"label": str, "text": str}` dicts stored under `"annotations"`, plus a boolean
`"annotation_complete"`.

The taxonomy is the six-label Venhoff et al. (arXiv:2506.18167) scheme, copied
verbatim from that paper's Appendix A. From `src/annotation.py`:

```python
# src/annotation.py — module constants
VALID_LABELS = frozenset({
    "initializing",
    "deduction",
    "adding-knowledge",
    "example-testing",
    "uncertainty-estimation",
    "backtracking",
})

# The 4 behaviours we care about (distinct to thinking models).
TARGET_BEHAVIOURS = [
    "backtracking",
    "uncertainty-estimation",
    "example-testing",
    "adding-knowledge",
]
```

Two of the six labels (`deduction`, `initializing`) are generic step types that
also appear in non-thinking models; the four `TARGET_BEHAVIOURS` are the
"distinct to thinking models" behaviours the whole thesis is about. The
`TARGET_BEHAVIOURS` list is explicitly the *single source of truth* — its raw
hyphenated strings are interpolated directly into the downstream filenames
(`{behaviour}_manifold_k3.npy` etc.), so any rename here silently orphans the
steering-vector files.

The prompt is reproduced verbatim, single user message, **no system prompt**, to
match the paper's protocol exactly:

```python
# src/annotation.py — _PROMPT_TEMPLATE (verbatim Venhoff Appendix A)
_PROMPT_TEMPLATE = """\
Please split the following reasoning chain of an LLM into \
annotated parts using labels and the following format ["label\
"]...["end-section"]. A sentence should be split into multiple \
parts if it incorporates multiple behaviours indicated by the \
labels.
...
0. initializing -> The model is rephrasing the given task and \
states initial thoughts.
...
5. backtracking -> The model decides to change its approach.
...
Answer only with the annotated text. Only use the labels outlined \
above. If there is a tail that has no annotation leave it out.\
"""
```

**Why verbatim:** the design goal is replication of Venhoff's behaviour
fractions, which gives an external sanity check that the labelling is "in family"
with prior work. `03_annotate_chains.py` operationalises this — after a pilot run
it compares observed sentence fractions against `VENHOFF_FRACTIONS` (deduction
0.52, adding-knowledge 0.15, …) and PASSES the pilot only if the four target
fractions land within ±50% of the paper's Figure 2 values. This is a deliberately
loose gate (small-N pilot), tightened to ±10 percentage points for the full run.

---

### C.2 Which model annotates — AWS Bedrock proxy, not OpenAI

The single most important deviation from the source paper is the annotator
identity, and the code flags it loudly:

```python
# 03_annotate_chains.py — module docstring
# Note: Venhoff used GPT-4o; GPT-4o-2024-11-20 is unavailable on the AWS proxy
# used in this project. Claude Sonnet 4.5 is used instead (noted in methods).
```

The default annotator is hard-coded as a Bedrock model id:

```python
# src/annotation.py
ANNOTATION_MODEL = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
```

Transport is a thin POST to the lab's AWS API-Gateway proxy via
`CLAUDE_PROXY_URL` / `CLAUDE_PROXY_KEY`, `temperature=0.0`, `max_tokens=8192`.
Crucially `_extract_text` is provider-agnostic because the proxy returns
different response shapes per model family — Anthropic returns a list of content
blocks, while Qwen/Nova return a plain string:

```python
# src/annotation.py — _extract_text
content = payload.get("content")
if isinstance(content, list) and content and isinstance(content[0], dict):
    return content[0].get("text", "") or ""
if isinstance(content, str):
    return content
return ""
```

This polymorphism is what makes the **multi-annotator robustness** test possible
on the same code path: `--annotator-model` lets you swap Sonnet for
`qwen3-235b` or `nova-pro`, each writing to its own `--out` file. The three
annotated files are present on disk (`data/annotated_R1-1.5B.json`,
`…__qwen3-235b.json`, `…__nova-pro.json`, each ~60 MB), so R2.1 was actually run
on the full 1000 chains, not just designed.

**Defence of the choice:** GPT-4o genuinely is not reachable on the lab proxy, so
some substitution was forced. Using a *third* model family (Sonnet for the
primary pipeline; Qwen3 and Nova only as robustness probes) is the right
mitigation — it lets the thesis claim "the geometry survives changing the
annotator," which is a stronger external-validity statement than Venhoff made.

---

### C.3 Spans → token offsets: the occurrence-aware matcher

The annotator returns *text*, not character ranges. Mapping each span back to a
character offset in the original chain — and from there to token positions for
activation extraction — is done by the canonical helper `src/text_offsets.py`,
imported by Phase 4, the PCA scripts, `compare_annotators.py`, and `span_f1.py`
alike (single source of truth, deliberately numpy/torch-free).

The subtle, load-bearing detail is **occurrence-awareness**. R1 chains loop and
repeat sentences verbatim (especially truncated ones and stock backtracking
phrases like "Wait, that's wrong"). A naive `str.find` mapped every repeat to the
*first* occurrence, collapsing many annotations onto one token span and producing
"35–56% exact-duplicate activation rows (zero-distance neighbours in every
kNN-based estimator the geometry claims rest on)." The fix walks a forward cursor:

```python
# src/text_offsets.py — _find_from
idx = chain_text.find(needle, start)
if idx >= 0:
    return idx
if start > 0:
    idx = chain_text.find(needle)   # fall back to from-start (pre-fix behaviour)
    if idx >= 0:
        return idx
return None
```

`locate_annotation_offsets` calls this with a cursor advanced past each match, so
the i-th repeat binds to the i-th occurrence, with a 40-char-prefix fallback for
minor annotator truncation. This is logged as confound CF-13 in
`CONFOUNDS_AND_REMEDIATION.md` — it is one of the bugs that quarantined the
pre-fix results. **Both** the agreement metrics and the activation extraction use
this exact rule, so the annotator-agreement numbers below are computed on the
same offset semantics the geometry rests on.

---

### C.4 Chunking — why long chains are split

The AWS API Gateway has a ~29-second hard timeout. At ~80 tok/s output a single
annotation request maxes out around ~2300 tokens, so any chain over
`CHUNK_THRESHOLD_TOKENS = 1200` (estimated at 4 chars/token) is split on
paragraph boundaries with ~100-token overlap, annotated chunk-by-chunk, and
merged. Two non-obvious correctness details:

1. A **continuation prefix** is prepended to chunks 2+ because without it "Sonnet
   labels the first sentence of each continuation chunk as 'initializing' because
   it looks like a fresh response" — a seam artefact that would inflate the
   `initializing` count and corrupt span boundaries.
2. At merge time, leading spans of chunk N+1 whose text appears verbatim in the
   tail-overlap of chunk N are dropped, so the first chunk's labels win for
   overlapped sentences.

```python
# src/annotation.py — merge_chunk_annotations
for span in chunk_annotations[i]:
    # Drop span if its text appears verbatim in the overlap region
    if span["text"] in overlap_region:
        continue
    keep.append(span)
```

**Critique:** the dedup is a substring test on raw text, which can over-drop when
a short stock phrase (again, "Wait,") legitimately recurs across the seam, or
under-drop when the annotator lightly paraphrases the overlap. The token estimate
(`len(text)//4`) is crude; the real tokenizer is the R1 tokenizer, not a 4-char
heuristic, so threshold/overlap sizing is approximate. None of this is validated
against a held-out exact-tokenisation oracle.

---

### C.5 Parsing robustness and crash-safety

Parsing is a regex over the `["label"]…["end-section"]` delimiters, with
generous label normalisation (strips `"0. "`, `"1-"` prefixes, maps bare digits
to names). Unknown labels do **not** crash — they fall back to `deduction` with a
warning:

```python
# src/annotation.py — parse_annotation_response
if label not in VALID_LABELS:
    logger.warning(f"  Unknown label '{m.group(1).strip()}' → deduction")
    label = "deduction"
```

The batch loop checkpoints after every chain (`checkpoint_every=1` for the real
run), backs up the prior file before touching it, treats a chain as resumable
unless every chunk succeeded (`annotation_complete`), and swallows per-chain
exceptions so one bad record cannot abort a multi-thousand-chain run. This last
property is exactly what Phase 7 will rely on (annotate-with-checkpoint, resume on
credit exhaustion). `verify_annotation_completeness.py` is the post-run gate: it
fails on missing/duplicate task_ids, empty annotations, partial records, or any
label outside `VALID_LABELS`.

**Critique of the fallback:** routing unknown labels to `deduction` quietly
biases the majority class upward and hides annotator format drift. Because
`deduction` is the dominant label anyway (45–54% of spans), a few mis-parsed
target-behaviour spans demoted to `deduction` is a silent false-negative on
exactly the behaviours the thesis cares about. There is no counter logged for how
often this fires per annotator, so the rate is unaudited.

---

### C.6 Inter-annotator agreement — the real numbers (R2.1)

`compare_annotators.py` builds a per-character label code array for every chain
(0 = unlabelled "O"), accumulates pairwise confusion matrices over characters,
and computes Cohen's κ from them. `span_f1.py` complements this with span-level F1
(exact / IoU≥0.5 / any-overlap), which forgives boundary jitter. Both ran on
**1000 common chains**. The numbers actually on disk
(`results/robustness/cross_annotator_comparison.json`,
`results/robustness/span_f1.json`):

**Character-level Cohen's κ:**

| Pair | κ (6-label) | κ (target vs other) | agree (labeled) | agree (overall) |
|---|---|---|---|---|
| Sonnet vs Qwen3-235B | 0.436 | 0.352 | 0.408 | 0.579 |
| Sonnet vs Nova-Pro | 0.350 | 0.305 | 0.354 | 0.519 |
| Qwen3-235B vs Nova-Pro | 0.345 | 0.263 | 0.322 | 0.520 |

So the headline "κ = 0.35–0.44, fair-to-moderate" is exactly the 6-label diagonal.
On the **target-behaviour collapse** (the four behaviours-of-interest vs.
everything else) κ is *worse*, 0.26–0.35 — i.e. annotators agree slightly less on
precisely the labels the thesis hangs on.

**Span-level F1** (more forgiving; only "labelled span" populations, ~78k–89k
spans each):

| Pair | exact boundary | IoU ≥ 0.5 | any overlap |
|---|---|---|---|
| Sonnet vs Qwen3 | 0.211 | 0.306 | 0.413 |
| Sonnet vs Nova | 0.171 | 0.257 | 0.375 |
| Qwen3 vs Nova | 0.247 | 0.307 | 0.373 |

Per-label F1 at IoU≥0.5 for the targets is low across the board — e.g.
adding-knowledge 0.15–0.21, example-testing 0.17–0.25, backtracking 0.23–0.30.
The label distributions also diverge materially: Sonnet calls 21.6% of spans
`uncertainty-estimation` while Nova calls only 11.5% and Qwen3 7.0%; Sonnet labels
`example-testing` at 7.5% vs Nova's 3.9%. So the three annotators do not even
agree on *base rates*, let alone boundaries.

**Defence:** even the "any overlap" F1 (≤0.41) and these κ values are *in the
range Venhoff-style sentence-labelling tends to produce* — sentence-segmentation
plus a fuzzy 6-way semantic taxonomy is genuinely hard, and character-level κ is
an unusually harsh denominator because a one-token boundary slip counts as
disagreement at every off-by-one character (this is precisely why `span_f1.py`
exists). The thesis does not claim the labels are *correct*; it claims the
*geometry replicates* across annotators despite the disagreement — which is the
stronger and more defensible position.

---

### C.7 The manifold-replication tie-in is partly broken on disk

`compare_annotators.py` is supposed to also emit a `manifold_replication` block
(per-annotator intrinsic dim, geodesic curvature, PR-trough layer) so the thesis
can say "cdim/geo are similar across annotators despite differing label
distributions." But in the file currently on disk that block is **empty**:

```
"manifold_replication": { "Sonnet-4.5": {}, "Qwen3-235B": {}, "Nova-Pro": {} }
```

The code path keys the robustness JSONs by short name
(`results/robustness/{short}/geometry_robustness.json`) and only fills the entry
`if rp.exists()`. The empty dicts mean either those per-annotator robustness JSONs
were not present when this comparison last ran, or the schema (`keystone_cdim`,
etc.) did not match. **This matters:** the 3-way *geometry* replication that
MEMORY records as "R2.2 CLOSED, folded into thesis (κ=0.35–0.44)" is the citable
result — but the artefact that is supposed to carry the per-annotator geometry
numbers in *this* comparison file is blank, so the replication evidence lives in
some other artefact (the per-annotator `geometry_robustness.json` / thesis
`tab:geometry-replication`), not here. Anyone re-deriving the claim from
`cross_annotator_comparison.json` alone would find nothing. The
`run_multiannotator_pipeline.sh` orchestrator does run Phases 4/5/5b/5c/5d +
robustness per annotator before calling `compare_annotators.py`, so the intended
fill is there — but the committed JSON shows it did not populate, which is a
reproducibility gap worth closing before citing.

---

### C.8 Circularity, the single-annotator null, and what it limits

**The circularity concern.** The model under study is
`DeepSeek-R1-Distill-Qwen-1.5B`. The Qwen3-235B annotator is *the same model
family* as the base model's distillation teacher lineage. If an LLM annotator
shares inductive biases with the model whose activations are being labelled, the
behaviour boundaries it draws may track that family's own surface cues rather than
a model-independent notion of "backtracking." The mitigation in this repo is
breadth-of-family: Sonnet (Anthropic) is the primary annotator and is *not* in the
Qwen lineage, and Nova (Amazon) is a third independent family. The fact that the
geometry survives all three is the real de-circularising argument. MEMORY also
notes a future plan to use `--annotator-model` for a Qwen3 de-circularisation arm
whose id "is not yet located." So the circularity is *acknowledged and partially*
addressed, not closed.

**The single-annotator specificity null.** This is the sharper limitation. The
behaviour-*specificity* result (the claim that each behaviour has its own
subspace, and that add-knowledge in particular fails specificity) is, per MEMORY
and `CONFOUNDS_AND_REMEDIATION.md`, **single-annotator** — it rests on the Sonnet
labels only and has *not* been re-run across Qwen3/Nova. So the only result that
replicated 3-way is the *geometry* (low-dim subspace + curvature), not
*specificity*. A specificity claim built on labels with κ≈0.4 and per-label
target F1 of 0.15–0.30 is fragile: a behaviour whose boundary the annotators agree
on only ~25% of the time at IoU≥0.5 cannot strongly support "this behaviour
occupies a distinct, separable subspace" unless the subspace is robust to exactly
the boundary jitter the annotators exhibit. `adding-knowledge` is the worst case
on both axes (lowest per-label F1 *and* the behaviour MEMORY records as failing
specificity everywhere) — those two facts are plausibly the same fact: the label
is so unreliable that no clean subspace can be recovered.

---

### C.9 How κ≈0.4 constrains the steering decision you are about to make

This stage feeds the steering experiment directly, and the agreement numbers
should temper two specific choices:

- **Population purity.** Steering vectors are difference-of-means over the
  labelled token populations for each behaviour. With three annotators disagreeing
  on ~60% of character labels and on base rates by 2–3×, the Sonnet-only
  populations that define your steering vectors contain a substantial fraction of
  spans that another competent annotator would have labelled differently. The
  steering vector is therefore a vector toward "Sonnet's notion of backtracking,"
  not "backtracking." For the four-target behaviours specifically, the
  target-vs-other κ (0.26–0.35) is the relevant, and worse, number.

- **adding-knowledge is the riskiest arm.** Lowest inter-annotator F1, failed
  specificity, and the behaviour most likely to be contaminated by the
  unknown-label→deduction fallback. If budget forces trimming arms, this is the
  one whose null result is least interpretable — a flat steering response could be
  "no causal subspace" or "the labels were too noisy to build a clean vector."
  Energy-matched and label-shuffled control arms are essential here, not optional.

- **Layer choice is *not* set by this stage.** Annotation produces labels, not
  layers. The PR-trough layer per annotator (the geometry-derived layer candidate)
  is meant to come through the `manifold_replication` block — which is empty on
  disk (C.7) — so the cross-annotator file gives you *no* layer guidance right
  now. Layer selection must come from the Phase 5/5b/triangulation artefacts and
  the Venhoff mid-layer recipe (≈L11/14/16–19) plus L27, per the steering memo —
  not from anything in this annotation stage. The one thing this stage *does* tell
  you about layers is a caution: if you pick a layer by maximising behaviour
  separability on Sonnet labels, you are partially fitting to one annotator's
  idiosyncrasies; prefer a layer that the *3-way* geometry replication agrees on.

---

### C.10 Run / citable status summary

- **RUN:** Sonnet, Qwen3-235B, Nova-Pro annotations all exist for the full 1000
  R1-1.5B chains. Agreement (κ) and span-F1 ran on 1000 common chains; results in
  `results/robustness/cross_annotator_comparison.json` and `span_f1.json`.
- **CITABLE:** the geometry-replicates-across-annotators claim (R2.2) is folded
  into the thesis (`tab:geometry-replication`) and treated as closed.
- **NOT CITABLE / single-annotator:** behaviour *specificity* (Sonnet only;
  add-knowledge fails). κ≈0.4 caps how strongly any per-behaviour purity claim can
  be made.
- **REPRODUCIBILITY GAP:** the `manifold_replication` block in the committed
  comparison JSON is empty; the per-annotator geometry numbers must be read from
  elsewhere, and this file should be regenerated before being cited as the source
  of the replication evidence.
- **DEVIATION FROM SOURCE:** annotator is Sonnet 4.5 (Bedrock), not Venhoff's
  GPT-4o; must remain noted in methods.
## D. Activation Extraction & Pooling

This stage (Phase 4) is the bridge between *labelled text* and *geometry*. Phases 1–3 produce a corpus of reasoning chains in which each sentence carries a behaviour label (`backtracking`, `uncertainty-estimation`, `example-testing`, `adding-knowledge`, plus the non-target labels). Phase 4 re-runs every annotated chain through the model with forward hooks, captures the residual stream at every layer, locates the token positions belonging to each labelled sentence, and pools those positions into one vector per (behaviour, layer, instance). The output is the set of `{behaviour}_layer{n}.npy` matrices on which **every downstream geometry claim and the entire steering programme rest.** Get this stage wrong and nothing above it is salvageable — which is exactly the history this code carries (CF-13…CF-16).

The relevant files:

- `04_extract_activations.py` — the runner (CLI, model load, skip-if-done guard).
- `src/activation_extraction.py` — the method (span→positions, pooling, sharded accumulation, integrity flag, sidecar, sweep).
- `src/hooks.py` — `ActivationCache`, the forward-hook residual-stream capture.
- `src/model_adapters.py` — decoder-layer location + harmony/DeepSeek family handling.
- `src/text_offsets.py` — the canonical occurrence-aware sentence→char-offset locator (CF-13 fix).
- `src/row_provenance.py` — row-id sidecar loading + duplicate hygiene (CF-13/CF-14 fixes).
- `pooling_sweep.py` / `tests/test_pooling.py` — the mean-vs-last sensitivity machinery.

---

### D.1 What is captured: residual stream, post-block, every layer, CPU-offloaded

Activations are captured by registering a PyTorch **forward hook on each decoder block** and grabbing that block's output — the post-block residual stream, *not* the MLP sub-module output, *not* attention, *not* a LayerNorm'd read. This matters: "residual stream" here means the hidden state flowing between transformer blocks (`hidden_states` after block *i*), which is the object steering vectors are added to. The hook deliberately handles both the tuple- and bare-tensor return conventions of HF blocks across versions, and immediately detaches + moves to CPU as float32:

```python
# src/hooks.py — ActivationCache._make_hook
def _hook(module, input, output):
    # HF transformer blocks may return either a tuple
    # (hidden_states, present_kv, ...) or just the hidden_states tensor
    # depending on the version. Handle both — we want the full 3D
    # (batch, seq_len, hidden_dim) tensor in the cache.
    h = output[0] if isinstance(output, tuple) else output
    self._cache[idx] = h.detach().cpu().float()
return _hook
```

```python
# src/hooks.py — ActivationCache._register
def _register(self) -> None:
    self._remove()
    for idx in self.layers:
        h = self._all_layers[idx].register_forward_hook(self._make_hook(idx))
        self._hooks.append(h)
```

Design rationale, defended as the authors would:

- **Post-block residual stream** is the canonical interpretability read for behaviour directions and the object that an activation-addition steering intervention perturbs. The module docstring justifies it as "the same convention as Huang et al. and Venhoff et al." — i.e. the recipe the steering literature this thesis competes with uses. Choosing the residual stream (not MLP-out) means the captured directions are *in the same space* a steering hook would write to, so a direction found here can be added back later without a basis change.
- **`detach().cpu().float()` per layer** is a memory decision: long R1 chains × 28 layers × hidden-dim in fp16 on GPU would OOM. Off-loading to CPU float32 immediately keeps GPU memory flat across chains. The cost is host RAM and a fp16→fp32 widening, accepted because the downstream PCA/geometry wants float32 anyway.
- **Layer coverage is ALL layers by default.** In the runner `layers=args.layers` defaults to `None`, and the library expands `None` to the full stack:

```python
# src/activation_extraction.py — extract_activations
if layers is None:
    layers = list(range(len(model.model.layers)))
```

For R1-Distill-1.5B that is **layers 0–27 (28 layers), hidden_dim 1536**. The runner's storage/runtime note ("~3–4 hours for 1000 chains × 28 layers", "~500 MB per model") confirms the intended full-stack extraction. `config.yaml extraction.layers: null` keeps this default.

The decoder-layer list is found via a fast path with a loud fallback. `ActivationCache.__init__` tries `model.model.layers` directly (DeepSeek/Qwen/Llama/gpt-oss all expose it) and only on `AttributeError` calls `locate_decoder_layers`, which walks a candidate list and **raises rather than silently mis-hooking** an unknown architecture:

```python
# src/model_adapters.py — locate_decoder_layers
candidates = (
    lambda m: m.model.layers,        # Qwen2 / Llama / GptOss
    lambda m: m.model.model.layers,  # some wrapped CausalLMs
    lambda m: m.transformer.h,       # GPT-2 / NeoX-style
    ...
)
```

This is the right defensiveness: a hook silently attached to the wrong module would produce plausible-looking but meaningless geometry, the worst kind of failure.

---

### D.2 From a labelled sentence to token positions: the Venhoff 1+10 window

Each annotated sentence is a *character span* in the chain text; the model sees *tokens*. The extractor tokenises the full `prompt + chain` text once per chain **with offset mapping**, then maps each sentence's char offset to a token window of **one preceding token + the first N=10 execution tokens** (`n_preceding=1`, `n_execution=10`, set in `config.yaml`). This is the Venhoff et al. recipe and the docstring states it explicitly ("one preceding token + first N execution tokens, following Venhoff et al.").

```python
# src/activation_extraction.py — _sentence_to_token_positions
onset = None
for tok_idx, (tok_start, tok_end) in enumerate(offsets):
    if tok_end > sentence_start_char:
        onset = tok_idx
        break
if onset is None:
    return []
...
for offset in range(n_preceding, 0, -1):
    p = onset - offset
    if p >= 0:
        positions.append(p)
for offset in range(min(n_execution, seq_len - onset)):
    p = onset + offset
    ...
    positions.append(p)
```

Why this window and not the whole sentence: the "first 10 tokens after the onset marker" captures the *behaviour signal at its surface*—the model committing to (e.g.) a backtrack—while keeping the per-instance vector count fixed-ish and avoiding diluting the signal across an arbitrarily long sentence. The single preceding token gives a small amount of pre-onset context (the residual the model decodes the onset *from*).

The crucial subtlety the code flags itself: **15.8% of target sentences are shorter than the 1+10 window**, so the unclipped window bleeds into the *next* sentence's tokens — a measured onset/surface-lexis confound. This is the `clip_window_to_sentence_end` knob (CF-flavoured robustness arm). It is **default OFF** (`config.yaml clip_window_to_sentence_end: false`) explicitly "to stay comparable with existing extractions" — i.e. the canonical matrices contain this bleed. The clip, when on, stops execution tokens at the sentence end char:

```python
# src/activation_extraction.py — _sentence_to_token_positions
if (sentence_end_char is not None and offset > 0
        and offsets[p][0] >= sentence_end_char):
    break  # token starts past the sentence end — next sentence's text
```

This is a genuine, acknowledged confound in the canonical data: a non-trivial fraction of "behaviour" vectors are partly made of the *following* sentence. The team chose comparability over correctness here; for the thesis it means any per-behaviour direction is partly contaminated by adjacency.

---

### D.3 Pooling: MEAN is canonical, but the code argues for LAST

Each (behaviour, instance) is a `(n_positions, hidden)` slice. `_pool` collapses it to `(hidden,)`. Four modes exist; **mean** is the configured default (Venhoff recipe), but the docstring is unusually candid that **last** is arguably more principled:

```python
# src/activation_extraction.py — _pool
#   mean  — average over positions (order-invariant; smears the within-span
#           trajectory and can cancel opposing directions).
#   last  — last execution token: in a causal transformer this is the only
#           position that has attended over the whole span (most
#           context-complete), and it is the residual the model decodes from.
if mode == "mean":
    return acts.mean(dim=0)
if mode == "last":
    return acts[-1]
if mode == "first":
    return acts[0]
if mode == "max":
    return acts.max(dim=0).values
```

The tension is real and pinned by tests: `test_mean_loses_order_information` asserts `mean` is permutation-invariant while `last` is not. So:

- **Why mean was chosen:** it is the Venhoff recipe (comparability with the prior literature the thesis benchmarks against), it averages out per-token noise, and it gives a stable centroid for a behaviour cloud. For *measuring* the geometry of a behaviour cluster, a smeared centroid is defensible.
- **Why the code keeps warning about it:** mean is order-invariant and "can cancel opposing directions." In a causal transformer only the **last** token has attended over the whole span and is the residual the model actually decodes from — which is precisely the position a steering intervention is most analogous to. So for the *steering* question (D.6), `last` has a stronger first-principles claim than `mean`.

To resolve this without re-running the GPU, the `pooling_sweep` machinery pools **multiple modes from one shared forward pass** (`streams = [pooling] + extra_modes`), writing extra modes to `pool_<mode>/` and a `pooling_sweep.json` comparing, at the steering layer, `cos(single_direction[mean], single_direction[last])` per behaviour and `d_eff_70` per mode:

```python
# src/activation_extraction.py — extract_activations (inner accumulation)
for layer_idx in layers:
    sl = cache[layer_idx][0, positions, :]   # (n_positions, hidden)
    for m in streams:   # primary + sweep extras, same forward pass
        acc[m][cat][layer_idx].append(_pool(sl, m).numpy().astype(np.float32))
```

`config.yaml` has this **ARMED**: `pooling_sweep: [mean, last]`, so every (re)extraction now also produces the `last` activations and the mean-vs-last cosine report at ~2× disk and ≈0 extra GPU. If `cos≈1`, the steering direction is pooling-robust; if `cos≪1`, the geometry is a pooling artefact and `last` should be preferred. This is the single most important sensitivity check feeding the steering decision, and it is wired to run automatically.

---

### D.4 Occurrence-aware span extraction — the CF-13 keystone fix

The original locator used `str.find`, which maps **every verbatim repeat of a sentence to its FIRST occurrence**. R1 chains loop and repeat (especially truncated ones and canned backtracking phrases), so repeated annotations all bound to the *same* token span, producing **35–56% byte-identical activation rows**. Zero-distance neighbours destroy every kNN-based intrinsic-dimension/curvature estimator (TwoNN, Levina–Bickel, geodesic graphs), and — worse — duplicates concentrate *within* a behaviour label, so the real per-behaviour matrix is duplicate-rich while label-permuted nulls are duplicate-poor, biasing the permutation test anti-conservatively. The 2026-06-08 tier1 layer-27 nulls (the keystone significance test) were **superseded** by this.

The fix (`src/text_offsets.locate_annotation_offsets`) runs a **forward cursor over the chain's ordered annotations**, searching each sentence from just past the previous match so the i-th repeat binds to the i-th occurrence:

```python
# src/text_offsets.py — locate_annotation_offsets
cursor = 0
for sent in sentences:
    idx = find_sentence_offset(chain_text, sent, start=cursor)
    offsets.append(idx)
    if idx is not None:
        # +1 (not +len) so overlapping/nested annotation boundaries can
        # still match while guaranteeing strict forward progress for
        # identical repeats. Never move the cursor backwards on an
        # out-of-order fallback match.
        cursor = max(cursor, idx + 1)
```

Two design details worth flagging: it advances by `+1` (not `+len`) so nested/overlapping annotation spans still match, and it falls back to a from-start search for out-of-order annotations so nothing that matched before fails. The extractor calls this over the **full** annotation list (all labels, not just the four targets) so the cursor advances identically whether or not a sentence is a target — that determinism is what lets analysis-side loaders reconstruct rows:

```python
# src/activation_extraction.py — extract_activations
# Locate ALL annotations up front (occurrence-aware): the cursor must
# advance over every annotation — including non-target labels — so
# repeats bind to successive occurrences deterministically (CF-13).
sent_offsets = locate_annotation_offsets(
    chain_text, [a.get("text", "") for a in annotations]
)
```

Empirically the docs report this drops the duplicate-row fraction from **51.9% → 1.2%**. A residual ~1.2% (over-annotation collisions + nested spans) remains, which is why `dedup_rows` in `src/row_provenance.py` is still applied downstream. Importantly, the stats review found the load-bearing Levina–Bickel "real < null" compression signal **survives dedup** (backtracking 9.12 vs null 10.65, ~18 SD) — duplicates understated absolute dims ~3× but did not manufacture the compression direction; TwoNN's nonsense values *were* the duplicate artefact. So the headline geometry claim survived this fix; the keystone null still owes a clean re-run.

---

### D.5 Provenance stamping, integrity flags, and crash-safety (CF-14/CF-15/CF-16 plumbing)

Activation matrices carry no row ids. Previously every analysis script reconstructed per-row chain-ids by *replaying* the Phase-4 iteration and **silently fell back to a proxy chain-id on a count mismatch** — under which the chain-stratified permutation null degenerates to a no-op (null ≡ real, p≈1.0) while looking legitimate (CF-14). The data-integrity review found the skip-blind replay mis-assigned 93.5% of uncertainty-estimation rows. Phase 4 now writes a `row_index.json` sidecar — one record per accepted row, in exact row order, identical across pooling streams:

```python
# src/activation_extraction.py — extract_activations (per accepted row)
row_index[cat].append({
    "chain_id": chain_id,
    "annotation_index": ann_idx,
    "char_offset": sent_offset,
    "token_start": positions[0],
    "n_positions": len(positions),
})
```

`src/row_provenance.chain_ids_for` loads this sidecar first and only replays for legacy extractions; `require_aligned` turns any chain-id/row mismatch into a **hard error** instead of a silent proxy. The matching rule is versioned (`SENTENCE_MATCHING_VERSION = "occurrence_aware_v1"`) and recorded in both `metadata.json` and the sidecar so analyses can refuse mixed-rule data.

Two more robustness mechanisms:

- **Sharded, crash-safe accumulation.** Accumulators flush to per-(stream, behaviour, layer) shard files every `flush_every` chains, then concatenate one (behaviour, layer) array at a time at the end. The docstring records why: an OOM-during-write killed the 2026-06-12 full-corpus run after 7/230 files and earlier truncated a behaviour to 1/28 layers. Peak memory at the final write is now bounded to one array, not the whole corpus.
- **An explicit `complete` integrity flag**, written LAST, true only when every behaviour with instances has all its layers on disk; `verify_extraction_complete` lets downstream refuse a partial set rather than silently building geometry on a truncated extraction. The runner's skip-if-done guard also re-runs if any target behaviour has zero extractions.

---

### D.6 Connection to the steering decision (layer choice + methodology)

This stage **constrains but does not finalise** the two open steering decisions (per the steering-status memory: pooling=MEAN resolved via Venhoff recipe; layer NOT finalised).

**Layer choice.** Because extraction defaults to **all 28 layers**, the captured data does *not* constrain which layer is steered — every candidate (mid-stack 11/16/19, and L27) is already in the canonical matrices, so Phase 7 can pick the steering layer post-hoc without re-extraction. The one place a layer is hard-coded is the *sweep report*, which compares pooling at `steer_layer = 27 if 27 in layers else max(layers)`:

```python
# src/activation_extraction.py — _write_pooling_sweep_report
steer_layer = 27 if 27 in layers else max(layers)
```

So the **pooling sensitivity check is evaluated at L27 by default** even though the steering layer is unsettled — a mild mismatch: if Phase 7 ultimately steers at a mid-stack layer (the Venhoff-recipe region), the mean-vs-last cosine the team will cite was computed at L27, not at the steered layer. Re-pointing the sweep's `steer_layer` to the chosen layer (or sweeping it) would close that gap.

**Methodology (mean vs last).** The steering vector is built from these pooled activations, so the pooling choice *is* part of the steering methodology. The canonical direction is mean-pooled (comparability), but the code's own argument — only `last` is context-complete and is the residual the model decodes/steers from — means the steering experiment should at minimum report the mean-vs-last cosine from `pooling_sweep.json` and, if it is low, prefer `last`. The sweep is armed in config precisely so this number exists before any GPU+API spend on steering.

---

### D.7 Status: RUN vs UNRUN, citability

- The **extraction code** is fixed and tested (`tests/test_pooling.py` pins all four pooling semantics + the sweep report; CF-13/14/15/16 fixes landed 2026-06-12).
- The **canonical matrices** in `data/activations/` predate or were regenerated around the Gate-0 work; per CONFOUNDS, the geometry is now **citable** (Gate-0 DONE: subspace survives, curvature is a clean negative/chain-artefact). The duplicate fix did *not* overturn the compression direction.
- The **pooling sweep with `last`** is *armed* but the canonical `pooling_sweep.json` (mean-vs-last cosine per behaviour at the steering layer) is the artefact the steering decision most needs and should be confirmed present/regenerated before committing spend.
- **Phase 7 steering itself is UNRUN.**

---

### D.8 Critique — confounds, fragilities, untested assumptions

1. **Window bleed into the next sentence (15.8%).** Default `clip_window_to_sentence_end: false` means a measured ~16% of target vectors are partly built from the *following* sentence's onset tokens. For short behaviour sentences this is exactly the behaviours most likely to be canned/short (backtracking, uncertainty). A per-behaviour direction contaminated by adjacency is a real threat to behaviour-specificity claims, and the clipped arm is a robustness check that (as far as the canonical data goes) is not the default. **Run the clipped arm and show the directions are stable.**

2. **Mean pooling cancels opposing directions — and the team knows it.** The headline geometry and steering directions are mean-pooled. The code itself argues `last` is more principled for a causal model. If `pooling_sweep.json` shows `cos(mean,last)` is low for any target behaviour, that behaviour's direction is a pooling artefact and any steering claim on it is fragile. This is the single sharpest internal critique and it is testable cheaply (the sweep is armed).

3. **Steering-layer / sweep-layer mismatch.** The pooling sweep is hard-pinned to L27, but the steering layer is unsettled and the leading candidates are mid-stack (Venhoff region). The pooling-robustness evidence the steering write-up will cite may therefore be at the wrong layer. Sweep the actual candidate layer.

4. **The keystone null still owes a clean re-run.** CF-13/14/16 were code-fixed but the load-bearing chain-stratified significance test on the *deduplicated, sidecar-provenanced* matrices is marked "re-run owed" in CONFOUNDS. The compression signal survives informally (~18 SD), but the formal p-value with B≥2239 and Holm–Bonferroni on the reported layers is the citable number and should be regenerated, not inherited from the 2026-06-08 superseded run.

5. **Null-pool scope (CF-16, standing design caveat).** Activations are extracted only for the four target behaviours; `deduction`/`initializing` (≈51% of sentences) are excluded. The permutation null therefore draws only from the four target labels — a narrower null than the full reasoning distribution. Extracting the non-target behaviours' activations would close this; until then "behaviour-specific" is relative to a 4-way contrast.

6. **fp16→fp32 and post-block read are assumptions, not validated.** The hook reads block *output*; for some architectures the "residual stream a steering hook writes to" is the block *input* (pre-LN) or a specific residual add point. For R1-Distill (standard Qwen2 decoder) block-output ≈ residual stream, so this is almost certainly fine — but the steering hook in Phase 7 must add at the *same* point this captured, or the direction is in a subtly different basis. Worth an explicit assertion that the steering write-point and the capture-point coincide.

7. **Replay fallback is still reachable for any legacy/foreign extraction.** `chain_ids_for` falls back to replay if no sidecar; `require_aligned` will hard-fail rather than proxy, which is correct, but it means analyses pointed at an old activation dir will *error* (good) — the operational risk is someone re-pointing at a pre-2026-06-12 directory and not re-extracting. Ensure the steering run consumes only sidecar-bearing matrices.
## E. Geometry / Gate-0 — Subspace, Curvature, Specificity

This section documents the **Gate-0 trilogy** — the three load-bearing geometric claims of the thesis ("each reasoning behaviour occupies a *curved, low-dimensional, behaviour-specific* manifold in the residual stream") and the machinery built to test each one against the chain confound. The stage spans the `05*` scripts (`05_pca_analysis.py`, `05b_geometric_diagnostics.py`, `05c_cross_layer_probing.py`, `05d_subtype_clustering.py`), the `src/` library (`pca.py`, `intrinsic_dim.py`, `curvature.py`, `nulls.py`), and the three Tier-1 robustness drivers (`tier1_effective_n.py`, `tier1_geometry_nulls.py`, `power_analysis_curvature.py`).

**Bottom line up front (the honest verdict as of the 2026-06-18 Gate-0 regeneration, per `CONFOUNDS_AND_REMEDIATION.md` status banner):**

| Claim | Verdict | Status |
|---|---|---|
| (1) Low-dimensional behaviour subspace | ✅ **SURVIVES** the chain control | CITABLE |
| (2) Curvature ("curved" manifold) | ❌ **CLEAN, WELL-POWERED NEGATIVE** (chain artefact) | CITABLE as a negative |
| (3) Behaviour-specificity | 🟧 **MIXED (2/4)** — backtracking + uncertainty specific at all layers; example-testing only at L27; **adding-knowledge nowhere (p=1.0)** | CITABLE with caveats |

The word "curved" must be struck from the headline. What survives is "low-dimensional, (partly) behaviour-specific subspace". The keystone confound throughout is **CF-2** — sentences within one reasoning chain are autocorrelated, so the raw N of 5k–16k sentences is illusory and the honest denominator is the ~600–900 *chains*.

---

### E.0 What is the input, and why these four behaviours

Every `05*` script consumes `data/activations/<model>/<behaviour>_layer<N>.npy` — a `(N_sentences, hidden_dim)` float matrix, one per (behaviour, layer), produced by Phase 4. For R1-Distill-1.5B `hidden_dim = 1536`, there are 28 layers, and the four `TARGET_BEHAVIOURS` analysed are `backtracking`, `uncertainty-estimation`, `example-testing`, `adding-knowledge`. The **focus / steering layer is hardcoded to 27** for R1-1.5B (`src/config.STEERING_LAYERS = {'R1-1.5B': 27, ...}`), inherited from "Huang's recommended steering layers". The 1536-D ambient dimension and the per-behaviour N expectations both flow directly into the power analysis (`--d 1536`), so the geometry stage and the steering layer are wired to the same single source of truth (`configs/config.yaml` via `src.config`).

The row-provenance discipline is critical and recently hardened. Chain IDs per activation row are resolved sidecar-first via `src.row_provenance.chain_ids_for(...)`, and `require_aligned(...)` **hard-fails** on misalignment. This is not pedantry: the old proxy fallback (one pseudo-chain per behaviour) silently turned the within-chain permutation null into a no-op returning p≈1.0. The current code refuses to run a vacuous null (see E.4).

---

### E.1 Claim 1 — Low-dimensional subspace: PCA, d_eff, participation ratio

`05_pca_analysis.py` is described in the source as "the core empirical contribution". For each (behaviour, layer) it fits an exact PCA and computes two effective-dimensionality summaries: `d_eff(p)` (smallest k explaining ≥ p of variance) and the **participation ratio** PR. The verbatim core, from `src/pca.py::analyse_behaviour`:

```python
# src/pca.py — analyse_behaviour()
pca = PCA(n_components=k, svd_solver="full")
pca.fit(activation_matrix)
cumvar = np.cumsum(pca.explained_variance_ratio_)
eigvals = pca.explained_variance_
d_effs = {}
for label, thresh in [("d_eff_50", 0.50), ("d_eff_70", 0.70), ("d_eff_80", 0.80),
                       ("d_eff_90", 0.90), ("d_eff_95", 0.95)]:
    idx = int(np.searchsorted(cumvar, thresh))
    d_effs[label] = min(idx + 1, k)
pr = float((eigvals.sum() ** 2) / (np.sum(eigvals ** 2) + 1e-12))
```

**Design defences.** `svd_solver="full"` is a deliberate reproducibility choice — the source comment explains that the default `"auto"` would pick the randomized solver for these `N≈50–145, d=1536, small-k` shapes, which is non-deterministic without a fixed seed; full SVD is cheap at this N. The PR `(Σλ)²/Σ(λ²)` is preferred over a single d_eff threshold because it is a "sample-size-robust estimate of effective dimensionality" — it does not depend on an arbitrary variance cutoff and degrades gracefully when N is small. The PCA components are also persisted (`save_pca_results` writes them to `.npy`), feeding Phase 6 steering-vector construction and Phase 5d clustering.

**What was found.** The live deduplicated summary (`results/robustness/R1-1.5B/geometry_robustness_summary.md`) reports PR ≈ 20 / 23 / 25 / 29 (backtracking / uncertainty / adding-knowledge / example-testing) in 1536-D — a genuine compression. But PR alone is not the claim; PR is a *PCA* (linear) quantity. The "low-dimensional" claim is carried by the **intrinsic-dimension** estimators (E.2), and crucially by their *survival under the chain control* (E.4).

`05c_cross_layer_probing.py` complements this with two cheap analyses: (a) **chain-grouped** layer-wise linear probing (behaviour-vs-other), and (b) non-adjacent layer-PCA principal-angle evolution. The probe deliberately uses `GroupShuffleSplit` so sentences from one chain never straddle train/test:

```python
# 05c_cross_layer_probing.py — probe_accuracy_at_layer (docstring)
"""Chain-grouped split: sentences from the same chain NEVER straddle
train/test. The previous plain train_test_split leaked chain context — the
exact failure mode tests/test_cv_leakage.py demonstrates inflates a null
probe from 0.50 to 0.99 — so the historical 83–93% accuracies are upper
bounds. Exact-duplicate rows are removed per class first (CF-13)."""
```

The principal-angle method (E.6) intentionally compares only **non-adjacent** layers (k ∈ {3,7,14}) because the residual identity `x_{L+1} = x_L + f(x_L)` makes adjacent subspaces near-identical by construction — a methodological trap the authors explicitly avoid.

---

### E.2 Intrinsic dimension — three estimators, honest CIs

`src/intrinsic_dim.py` implements TwoNN (Facco 2017), a robustified Levina–Bickel MLE, and Grassberger–Procaccia correlation dimension, each with a bootstrap CI. The "convergent estimates → credible; divergent → estimator-dependent caveat" framing is baked into the module docstring.

The TwoNN fit reveals a subtle, well-documented numerical fix:

```python
# src/intrinsic_dim.py — twoNN_estimate._fit()
# Empirical CDF over the FULL sample as i/(n+1) (so F < 1 everywhere),
# THEN keep the lower `fraction` (the linear regime). Computing
# F = arange(1,cutoff+1)/cutoff on the *truncated* set instead forces
# F=1.0 at the cutoff; -log(1-F) then explodes and that single
# high-leverage point dominates the through-origin slope, inflating the
# estimate ~35-50% (dim 5 -> ~6.7, dim 8 -> ~10.2). See tests/.
F_full = np.arange(1, n_mu + 1) / (n_mu + 1)
cutoff = max(2, int(np.ceil(fraction * n_mu)))
```

The **CI strategy** is a genuine scientific correction (AUDIT.md §5 #16, CF-9). All three estimators use `_subsample_bootstrap`: it subsamples *points* (m = 0.8·N without replacement) and recomputes the estimator end-to-end, rather than resampling derived μ-ratios or pairwise distances. The comment is explicit about why: derived quantities are mutually dependent (pairs share points, the kNN graph is fixed), so resampling them gave absurdly narrow CIs like `[0.575, 0.587]`. Subsampling (not n-out-of-n) also avoids creating duplicate points whose zero-distance neighbours corrupt every kNN estimator.

**Estimator reliability finding (important).** In the live run, **TwoNN is judged unstable** on this data and the **correlation dimension is the reported estimator**: the live summary states "twoNN is duplicate/subsample-unstable here; correlation dimension is the reliable estimator." The superseded June-8 TwoNN value of "0.168" was a zero-distance (duplicate-row) artefact (CF-13). The citable intrinsic dims are corr-dim ≈ **5.9 / 6.2 / 6.0 / 7.7** (back / unc / example / add) in 1536-D — substantially below the PCA PR of 20–29, which is the actual evidence for "low-dimensional".

---

### E.3 Claim 2 — Curvature: the diagnostics, and why it is a NEGATIVE

`src/curvature.py` implements three complementary flat-vs-curved diagnostics, all sweepable over kNN size k (stable-across-k → credible; k-dependent → flagged artefactual):

1. **Local-vs-global PCA dim ratio** — flat ≈ 1, curved < 1.
2. **Geodesic/Euclidean ratio** — flat ≈ 1, curved > 1.
3. **Tangent-space variation** (mean principal angle, degrees) — flat ≈ 0, curved > 0.

Two of these encode hard-won calibration fixes. The local-vs-global ratio uses a **sample-size-matched baseline** (random k-point subsets), not the full cloud:

```python
# src/curvature.py — local_vs_global_dim_ratio (docstring)
"""CRITICAL calibration note: the baseline MUST be sample-size-matched to the
local neighbourhood. The original implementation divided by the PCA dim of
ALL N points, which conflated curvature with the trivial fact that a k-point
neighbourhood can express at most k-1 dimensions while the full cloud can
express many more — so a perfectly FLAT subspace of dimension > k scored
<< 1 (empirically 0.29-0.85 on flat Gaussian data). Matching the sample
size removes that confound. See tests/test_curvature.py::test_flat_*."""
```

The geodesic graph is symmetrised with `W.maximum(W.T)` rather than `(W+W.T)/2`, because averaging would *halve* one-directional kNN edges and push a flat-manifold ratio to ~0.73 when it must be ≥ 1. `05b` projects to the top-50 PCA subspace before curvature ("to avoid ambient-dim noise dominating") and all CIs again use the point-subsample bootstrap.

**Why the result is a negative.** This is the sharpest scientific finding of the stage. On the *full* data the manifolds look curved (geo/Euclidean ≈ 3.4–4.2). But at **one-sentence-per-chain** (the chain-stratified / random-subsample-to-n_chains regime) the geodesic ratio collapses toward ~2.3–2.5 and is statistically indistinguishable from a chain-matched relabelling. From the live summary:

```
| Behaviour | geo full | geo randsub | geo chainstrat |
| backtracking          | 3.73 | 2.42 | 2.35 |
| uncertainty-estimation| 4.04 | 2.42 | 2.43 |
| adding-knowledge      | 3.44 | 2.55 | 2.53 |
| example-testing       | 4.22 | 2.55 | 2.34 |
```

The apparent curvature is **within-chain autocorrelation** — sentences walk along a chain trajectory, not along a behaviour-specific curved manifold. The status banner records this as: *"Per-behaviour curvature is a CLEAN, WELL-POWERED NEGATIVE — curved on full data, ≈1.0 (flat) at one-sentence-per-chain."* "Clean" matters because `power_analysis_curvature.py` (E.5) shows the analysis was powered to detect real curvature if it existed; a null result is therefore informative rather than merely underpowered.

---

### E.4 The null hierarchy — the methodological spine (`src/nulls.py`)

The companion document commits to a three-level null hierarchy, implemented in `src/nulls.py`:

- **Primary: chain-stratified permutation** — shuffle labels *within* each chain, preserving per-chain composition, then recompute the statistic on the target-labelled rows. This controls for chain identity, within-chain drift, ambient covariance, and sample size simultaneously.
- **Secondary: cross-chain (global) permutation** — global label shuffle; isolates behaviour-level vs category-level effects.
- **Tertiary: Marchenko–Pastur isotropic** — matched-(N,d) Gaussian; explicitly *only* a finite-sample inflation diagnostic, "NOT a structural test" (its `real_value` is left as NaN by design).

The chain-stratified null contains the single most important safeguard in this stage — the **no-op guard**:

```python
# src/nulls.py — chain_stratified_permutation_null()
n_mixed = sum(1 for idxs in chain_to_idx.values()
              if np.unique(labels[idxs]).size > 1)
if n_mixed == 0:
    raise ValueError(
        "chain_strat_perm: within-chain permutation is a NO-OP — no chain "
        "contains more than one distinct label. This usually means proxy "
        "chain ids (one pseudo-chain per behaviour). Fix the chain-id "
        "provenance ... instead of running a vacuous null.")
```

P-values use the **Phipson–Smyth (2010) smoothing** `(1+count)/(1+B)`, not the naive `count/B`. The docstring is candid: the unsmoothed version can return p=0, which is impossible for a permutation test (the observed labelling is itself a permutation) and "fakes infinite resolution" — at B=100 an unsmoothed 0 was being read against a Bonferroni threshold of 4.5e-4 it could never legitimately pass. `05_pca_analysis.py` even prints the minimum attainable p and warns that Bonferroni across 4×28 cells needs B ≥ 2239.

**Statistic-coverage gap (CF-3) and its fix.** `05b`'s null runs *only* on the top-10 variance ratio. But the load-bearing numbers are intrinsic dim and curvature. `tier1_geometry_nulls.py` re-runs the *same* `src/nulls.py` machinery with `statistic_fn ∈ {twoNN, levina_bickel, local_vs_global_dim_ratio}` and **`tail='lower'`** (the claims are "dim LOW" / "ratio LOW (curved)", so real must sit *below* the null). The script carries an explicit hazard note that wiring tail incorrectly silently inverts the test for the other two curvature diagnostics.

---

### E.5 Power analysis — what makes the curvature negative "well-powered"

`power_analysis_curvature.py` is a pre-registered power study: it generates ground-truth curved manifolds (hyperspheres `S^{m-1}` of radius 1/κ embedded in R^1536 + isotropic noise) and matched flat controls (uniform disks), runs each diagnostic on both, and reports detection power as the curved-vs-flat AUC. The pre-registration commitment is hard-coded in the docstring: *"If the smallest detectable curvature at a given N exceeds the curvatures plausibly induced by reasoning structure, the per-behaviour analysis is downgraded to a corpus-pooled analysis."*

The result table (`results/power_analysis/summary.md`): the local-vs-global diagnostic detects κ ≥ 0.5 at **N ≥ 500**, and per-behaviour pools are ~600–900 chains. So the null is not an artefact of low power — the experiment *could* have seen curvature and did not. A defensive guard refuses to write an all-NaN power table, treating systematic numerical failure as a bug rather than a "null result":

```python
# power_analysis_curvature.py — main()
if cells and all(not np.isfinite(c.auc) for c in cells):
    raise RuntimeError(
        f"All {len(cells)} power cells produced non-finite AUC. This is a "
        "systematic failure ... NOT a real 'undetectable' result. "
        "Refusing to write an all-NaN power_table.csv. ...")
```

Caveat: the geodesic and tangent diagnostics are weaker (geodesic needs N≥2000 for κ≥0.5; tangent never reaches AUC 0.95 in the tested grid up to N=2000). So "well-powered" rests primarily on the local-vs-global diagnostic.

---

### E.6 Effective-N — quantifying the chain confound (`tier1_effective_n.py`)

This script computes the honest denominator behind every estimator. It fits PCA per behaviour, computes the **one-way ICC(1)** of the PC scores grouped by chain, and reports the design effect `Deff = 1 + (n0−1)·ICC` and `n_eff = N/Deff`:

```python
# tier1_effective_n.py — icc_oneway()
msb = ssb / (G - 1)
msw = ssw / (N - G)
n0 = (N - (counts ** 2).sum() / N) / (G - 1)
denom = msb + (n0 - 1) * msw
icc = (msb - msw) / denom if denom > 0 else float("nan")
```

The finding is stark. PC1 ICC is **0.88–0.93** across all four behaviours — sentences within a chain are massively autocorrelated. The raw N of 5k–16k collapses to n_eff(PC1) ≈ 700–1000, i.e. essentially the number of chains:

| Behaviour | N sent | N chains | ICC PC1 | n_eff(PC1) |
|---|---|---|---|---|
| backtracking | 10267 | 705 | 0.930 | 756 |
| uncertainty-estimation | 16728 | 909 | 0.891 | 1015 |
| example-testing | 5829 | 636 | 0.922 | 686 |
| adding-knowledge | 5027 | 881 | 0.881 | 980 |

This is *why* the curvature collapses to flat at one-sentence-per-chain and *why* the intrinsic-dim claim is only credible because it survives the chain control. The whole Gate-0 epistemology — "compare to a chain-matched null, report n_eff not N" — rests on these ICC numbers.

---

### E.7 Claim 3 — Behaviour-specificity (MIXED, 2/4)

Specificity is the question "is this subspace *specific* to the behaviour, or a generic chain property?" — answered by the chain-stratified variance-ratio null. The status banner records the verdict bluntly: **backtracking + uncertainty-estimation are specific at all layers (B=2500 null, p<.001); example-testing only at L27; adding-knowledge nowhere (p=1.0).** This is the single most threatening result for the thesis spine, because the headline claims behaviours are "behaviour-specific" and **one of the four fails everywhere**, and a second only at the chosen steering layer.

`05d_subtype_clustering.py` is downstream of this: it K-means-clusters each behaviour's activations (silhouette-selected k ∈ [2,8]) at the per-behaviour manifold-peak layer to discover sub-types, producing centroids used for sub-type steering vectors. Note its layer-selection heuristic is `argmin(participation_ratio)`, with a documented bug history:

```python
# 05d_subtype_clustering.py — _resolve_focus_layer (docstring)
"""... We therefore take argmin(participation_ratio).
The earlier argmax(d_eff_70) rule returned layer 0 whenever d_eff saturated at
the PCA component cap, which is why clustering ran at the wrong layer."""
```

---

### E.8 Critique — what is genuinely established vs over-claimed

**Genuinely established (citable):**
- Low intrinsic dimension (corr-dim ≈ 6–8 in 1536-D) **survives** the chain control — the keystone PASS. This is real and the methodology (chain-stratified null + point-subsample CIs + dedup) is rigorous.
- The curvature negative is clean and well-powered — a defensible, honestly-reported null. Striking "curved" from the headline is the right call.

**Over-claimed or fragile:**
1. **"Behaviour-specific" is half-true at best (2/4).** adding-knowledge fails specificity *at every layer* (p=1.0); example-testing is specific *only at L27*. Any thesis sentence asserting all four behaviours are specific is unsupported. example-testing's L27-only specificity is especially uncomfortable because **L27 is the chosen steering layer** — there is a circularity risk that the layer favouring specificity is also the layer steering will run at.
2. **Specificity is single-annotator.** The R2.2 replication (Sonnet/Qwen3/Nova) confirms intrinsic-dim and curvature-as-artefact replicate 3-way despite κ=0.35–0.44, but the **variance-ratio specificity null is still single-annotator**. So behaviour-specificity replicates at the *subspace* level, not yet at the *specificity-test* level. With fair-to-moderate inter-annotator agreement, the 2/4 split could itself be label-noise-dependent.
3. **Estimator-dependence is real and was nearly fatal.** TwoNN is unstable here and gave a duplicate-artefact "0.168" in the superseded June-8 run. The conclusion now rests on the *correlation dimension* specifically — a single estimator choice. The module's own framing ("divergent estimates flag the caveat that intrinsic dimension is an estimand whose value depends on the estimator") cuts against over-confidence.
4. **Duplicate-row hygiene (CF-13) was load-bearing and recent.** 35–56% of pooled activations were exact duplicates (short repeated markers in a fixed token window) before the fix. Everything pre-2026-06-08 is **superseded / do-not-cite** (the `geometry_nulls_layer27.md` file carries an explicit "⚠️ SUPERSEDED — DO NOT CITE" banner). This is a reminder of how sensitive the pipeline is to preprocessing.
5. **PR vs intrinsic dim conflation risk.** PR (20–29) is a *linear* PCA quantity; intrinsic dim (6–8) is the nonlinear claim. They must not be reported interchangeably as "the dimension". The "low-dimensional" headline should cite the chain-controlled intrinsic dim, not PR.
6. **Bonferroni vs B mismatch.** `05_pca_analysis.py` itself warns that proper multiple-comparison control across 112 cells needs B ≥ 2239, while several runs used B=100–200. The specificity claims that *passed* used B=2500, but cross-layer null sweeps at low B are resolution-limited.
7. **n_eff caveat compounds CIs.** With n_eff ≈ n_chains, any CI or null still implicitly built on raw N is overconfident; R1.2 (chain-block bootstrap) is the owed fix, and `tier1_geometry_nulls.py` runs estimators with `n_bootstrap=0` inside the null (no nested CI) — so the geometry-null p-values do not yet carry honest CIs.

---

### E.9 Connection to the steering decision

This stage feeds the imminent steering experiment in three concrete ways:

1. **Layer choice.** The steering layer is hardcoded to **L27** for R1-1.5B via `STEERING_LAYERS`, inherited from Huang. The geometry stage gives weak independent support for L27: it is the *only* layer where example-testing is behaviour-specific, but that is a double-edged observation (circularity — see critique #1). The MEMORY notes the steering layer is "NOT finalised" and the plan is to build mid-layers (11/16/19) + L27 and let Phase 7 decide. The geometry results do **not** strongly endorse L27 over mid-layers; `05c`'s layer-wise probe curves and the per-layer null sweep (`null_pvalues_per_layer.json`) are the relevant evidence and should be consulted before committing GPU spend to a single layer.

2. **Methodology — vectors and sub-types.** Phase 6/7 build steering vectors from the PCA components (`save_pca_results`) and from the 05d cluster centroids. The "manifold (k-component)" steering arms are direct descendants of the PR / d_eff / clustering outputs here. The git status shows the `*_manifold_k{1,3,5,10,auto}.npy` and `*_single.npy` steering vectors are already staged — i.e. the geometry → steering-vector handoff has run.

3. **What the geometry results *constrain* about steering claims.** Because curvature is a negative, steering claims should **not** invoke "moving along a curved manifold" — only "moving within a low-dimensional subspace". Because adding-knowledge fails specificity, a steering null on adding-knowledge is *expected* and should not be read as a steering failure; conversely a specificity-respecting design should weight backtracking/uncertainty (the two clean PASSes) as the primary steering targets. Spending API+GPU budget to steer adding-knowledge as a headline arm is the highest-risk allocation given the geometry says its subspace is not behaviour-specific.

**Run/unrun status:** the geometry pipeline (PCA, 05b/c/d, tier1 effective-N, tier1 geometry-nulls, power analysis) is **RUN** on clean deduplicated data and **CITABLE** (post-2026-06-18). The pre-2026-06-08 `geometry_nulls_layer27` outputs are **QUARANTINED / do-not-cite**. The specificity null's multi-annotator replication is **OWED**. **Phase-7 steering itself is built but UNRUN.**
## F. Robustness & Cross-Model Replication (R2.2)

This section documents the robustness layer of the geometry pipeline: the machinery that asks "does the per-behaviour geometric story survive (a) being re-annotated by a *different* LLM judge, and (b) being run against a *different model's* activations?" Two distinct things travel under nearby names in the codebase and it is essential not to conflate them:

- **R2.2 — annotator replication (RUN, citable).** The same R1-Distill-1.5B activations are re-labelled by three different LLM annotators (Sonnet-4.5, Qwen3-235B, Nova-Pro), and the geometry diagnostics (`robustness_geometry.py`) are recomputed per annotator. The claim under test is annotator-robustness of the geometry, *not* a second model.
- **M6 / Extension A — cross-*model* replication (UNRUN, blocked).** `13_baseline_replication.py` + `src/cbs/comparison.py` compare R1-Distill-1.5B against the base model Qwen-2.5-Math-1.5B. This is a separate, unrun arm whose entire `results/cbs/cross_model/` output directory does not exist on disk.

The bootstrap-CI machinery (`src/nulls.py`, `tests/test_bootstrap_ci.py`, `tests/test_cross_model_bootstrap.py`) underpins both. The reader is about to commit API+GPU spend on a steering experiment; the load-bearing relevance of this section is that **R2.2 is what licenses treating the LLM-annotator labels as a trustworthy enough dependent variable to bother steering at all** — and its limits bound how much the steering result can claim.

---

### F.1 What R2.2 actually does

`robustness_geometry.py` is a single CPU script that runs the "Tier 0" geometry suite from already-extracted activation `.npy` files. It is parameterised by `--model-short`, and the *same script* is run three times with three different annotation files swapped in. The model is always R1-1.5B; only the labels (and hence which sentences land in each behaviour's activation matrix) change.

The core loop, per behaviour `b` at its peak layer `L`:

```python
# robustness_geometry.py, main()
for b, L in PEAK.items():
    Xr = np.load(ACT / f"{b}_layer{L}.npy").astype(np.float32)
    cids = require_aligned(b, Xr.shape[0], cidmap.get(b), context="robustness_geometry")
    dup_pct = 100.0 * (1 - len(np.unique(Xr, axis=0)) / Xr.shape[0])
    X, cu = dedup(Xr, cids); Nu = X.shape[0]
    obc = {}
    for i, c in enumerate(cu): obc.setdefault(c, []).append(i)
    uniq = list(obc); nc = len(uniq)
```

The keystone design is the **same-N control triad** — for both intrinsic dimension and curvature, the script compares three quantities of *equal sample size*:

```python
# robustness_geometry.py, main()
cf = cdim(X, nb=B_DIM); rs, st = [], []
gf, lf = geo(X), lgr(X)
for s in range(B_CURV):
    g = np.random.default_rng(SEED + s)
    ridx = g.choice(Nu, nc, replace=False)                       # random subsample, size = n_chains
    sidx = np.array([g.choice(obc[c]) for c in uniq])            # chain-stratified: 1 sentence/chain
    rs.append(cdim(X[ridx], nb=4)); st.append(cdim(X[sidx], nb=4))
    g_rs.append(geo(X[ridx], nb=6, npairs=250)); g_st.append(geo(X[sidx], nb=6, npairs=250))
```

**Why this design.** The deepest confound in the whole geometry programme is **CF-2 (chain autocorrelation / effective-N)**: sentences inside one CoT chain are not independent draws, so the ~5–16k "points" feeding every intrinsic-dim/curvature estimator violate the i.i.d. assumption those estimators are built on. A naive low-dimension result could be an artifact of sampling many near-identical points from a few trajectories. The triad isolates that:

- `full` uses all unique points.
- `random_sub` draws `n_chains` points at random — same N as the stratified version, but *not* one-per-chain.
- `chain_strat` draws exactly one sentence per chain — kills within-chain autocorrelation while holding N fixed.

If `chain_strat ≈ random_sub ≈ full`, the geometry is a property of the behaviour, not of chain repetition. The summary states the reading explicitly:

```python
# robustness_geometry.py, markdown summary
"- **cdim chainstrat ≈ cdim randsub ≈ cdim full** → low intrinsic dimension is "
"behaviour-intrinsic, NOT a chain confound (keystone PASS).",
"- **geo randsub vs geo chainstrat** at equal N isolates real chain-trajectory "
"curvature from sparse-graph effects.",
```

The curvature axis is deliberately the opposite: `geo full` is expected to be *higher* than `geo chain_strat`, because geodesic curvature is partly a within-chain-trajectory property. A drop from full to stratified is the "curvature is a chain artefact" signature — which is exactly the negative result the project decided to own (CF-2/CF-7), not a positive curvature claim.

Estimator choice is also defended in code: correlation dimension is PRIMARY ("stable"), twoNN is computed but flagged as duplicate/subsample-unstable. This matters because the earlier (quarantined) twoNN-based numbers (e.g. "dim 0.168") were artefacts of exact-duplicate rows (CF-13).

---

### F.2 The 3-way result (RUN, citable)

All three annotator runs exist on disk under `results/robustness/{R1-1.5B, R1-1.5B__nova-pro, R1-1.5B__qwen3-235b}/`, each with `geometry_robustness.json`, `_summary.md`, and `provenance.json` (seed=42, input SHA stamped; note `git_commit: null` — these were produced on the no-git cluster copy).

**Intrinsic dimension (cdim full) replicates with the same ordering across annotators:**

| Behaviour | Sonnet cdim | Qwen3 cdim | Nova cdim |
|---|---|---|---|
| backtracking | 5.85 | 6.67 | 6.64 |
| uncertainty-estimation | 6.21 | 7.00 | 7.15 |
| adding-knowledge | 7.71 | 8.36 | 8.70 |
| example-testing | 6.04 | 6.56 | 5.96 |

The numbers are not identical, but they sit in a tight ~1.0-dim band and preserve ordering (adding-knowledge highest, backtracking/example-testing lowest). Critically, the **keystone PASSES in all three** annotators: `chain_strat` tracks `full` (e.g. Sonnet backtracking full 5.85 vs chainstrat 6.36; Nova 6.64 vs 6.75; Qwen3 6.67 vs 6.72). The low-dimensionality of the per-behaviour subspaces is therefore not an annotator artefact and not a chain artefact.

**Curvature-as-chain-artefact also replicates.** In every annotator, `geo full` (≈3.0–4.2) collapses toward `geo chain_strat` (≈2.3–2.5) — e.g. Sonnet backtracking 3.73 → 2.35; Nova 2.98 → 2.33; Qwen3 2.96 → 2.34. The geodesic "curvature" is largely a within-chain trajectory effect in all three. This is consistent with the project's decision (CONFOUNDS §CF-2) to report curvature as a clean *negative* (chain artefact) rather than a manifold-curvature claim.

**Inter-annotator agreement is low** (`results/robustness/cross_annotator_comparison.md`), which is what makes the replication load-bearing rather than redundant:

| Pair | κ (6-label) | κ (target vs other) | span-F1 (IoU≥0.5) |
|---|---|---|---|
| Sonnet vs Qwen3 | 0.436 | 0.352 | 0.306 |
| Sonnet vs Nova | 0.350 | 0.305 | 0.257 |
| Qwen3 vs Nova | 0.345 | 0.263 | 0.307 |

Label *distributions* diverge substantially: uncertainty-estimation is 21.6% of Sonnet spans but only 7.0% of Qwen3; Nova's deduction is 54.1% (plausibly inflated by the CF-18 unknown→deduction coercion on Nova's ~290/1000 partial-parse failures). The headline claim is therefore strong: **despite κ in the fair-to-moderate range and very different label priors, the geometry lands in the same place.** That is the cleanest external-validity statement the geometry programme has.

---

### F.3 Bootstrap-CI and cross-model machinery

`src/nulls.py` is the null-hypothesis hierarchy used by the *geometric* per-layer runs (the `results/geometric/.../diagnostics_layer*.json` files), not by `robustness_geometry.py` directly, but it is the engine behind the variance-ratio specificity claim that R2.2 is being measured against. Its primary null is the chain-stratified within-chain label permutation, with a hard guard against the vacuous no-op:

```python
# src/nulls.py, chain_stratified_permutation_null()
n_mixed = sum(1 for idxs in chain_to_idx.values()
              if np.unique(labels[idxs]).size > 1)
if n_mixed == 0:
    raise ValueError(
        "chain_strat_perm: within-chain permutation is a NO-OP — no chain "
        "contains more than one distinct label. This usually means proxy "
        "chain ids (one pseudo-chain per behaviour). Fix the chain-id "
        "provenance ...")
```

p-values use Phipson–Smyth smoothing `(1+count)/(1+B)` so that `p=0` is impossible and a non-finite real statistic returns NaN rather than the maximally-significant floor. This is genuinely defensible practice (an unsmoothed 0 at B=100 cannot legitimately clear a Bonferroni threshold of 4.5e-4).

The Nova-Pro per-layer null hierarchy (`results/geometric/R1-1.5B__nova-pro/summary_layer27.md`) shows the specificity test is genuinely *mixed* even when re-annotated: chain-strat p for the top-10 variance ratio is significant for backtracking (0.0050) and example-testing (0.0050) but p=1.0000 for uncertainty-estimation and adding-knowledge. This mirrors the original single-annotator finding (specificity 2/4, add-knowledge fails everywhere) — so the *subspace exists and replicates*, but the *behaviour-specificity test does not pass for all behaviours* under any annotator.

The genuine two-sample bootstrap lives in `src/cbs/comparison.py::cross_model_compare` and is exercised by `tests/test_cross_model_bootstrap.py`. The important fix it encodes (AUDIT §5 #15, CF-12) is that the old "bootstrap" was a Gaussian reconstructed from CI width; the real version resamples each model's persisted effect-size distribution independently:

```python
# src/cbs/comparison.py, cross_model_compare()
if boots_r1 and boots_bs:
    a = np.asarray(boots_r1, dtype=float); b = np.asarray(boots_bs, dtype=float)
    ia = rng.integers(0, a.shape[0], size=n_bootstrap)
    ib = rng.integers(0, b.shape[0], size=n_bootstrap)
    diff = a[ia] - b[ib]
    p = float(min(1.0, 2.0 * min((diff <= 0).mean(), (diff >= 0).mean())))
    method = "two_sample_bootstrap"
elif all(np.isfinite(x) for x in ci_r1 + ci_bs) and delta != 0:
    ...
    method = "normal_approx_from_ci"
```

The tests lock in that the bootstrap path is taken when both arrays are present (`test_bootstrap_used_and_separated_gives_small_p`), that it falls back to the labelled normal approximation otherwise (`test_falls_back_to_normal_approx_without_boots`), and that it is deterministic given a seed. `tests/test_bootstrap_ci.py` separately locks in the point-subsample-bootstrap fix (AUDIT #16, CF-9): CIs are recomputed by resampling *points* (subsample without replacement, to avoid duplicate-point kNN degeneracy), not derived per-pair quantities — fixing the absurdly tight `[0.575, 0.587]`-style bands.

These are good tests, but note what they test: **the plumbing, on synthetic `np.linspace` inputs.** They prove the bootstrap math is correct and deterministic. They do not (cannot) prove any real cross-model finding, because no real `effect_size_boots` arrays for a baseline model have ever been produced.

---

### F.4 The cross-*model* arm is UNRUN

`13_baseline_replication.py` is fully written but its prerequisite chain — Extension A on Qwen-2.5-Math-1.5B (Phase 2b/3/4 + M1–M4 reruns) — has never run. The script is explicitly build-safe: it emits a `blocked` JSON and returns 0 rather than failing.

```python
# 13_baseline_replication.py, main()
blockers = _check_prerequisites(args)
if blockers:
    out = {"status": "blocked", "blockers": blockers, "synthesis_reference": "§M6.2", ...}
    (args.out_dir / "cross_model_blocked.json").write_text(json.dumps(out, indent=2))
    return 0
```

I confirmed `results/cbs/cross_model/` **does not exist on disk** — not even the blocked stub has been written. The distillation-vs-reveals test ("does distillation *create* the geometry or *reveal* a pre-existing one") that this script exists to run is entirely outstanding. Anything in the thesis framed as cross-model replication is, at the time of writing, the *annotator* replication (R2.2) only.

---

### F.5 Critique — where this weakens a thesis claim

**1. Annotator-replication is not the same as geometry-replication of an independent ground truth.** All three annotators are LLMs labelling the *same* fixed CoT text, and the geometry is computed on the *same* fixed R1-1.5B activations. If all three judges share a systematic bias — e.g. all of them over-segment on the same surface cue tokens ("wait", "actually", "let me check") — then three "independent" annotators can converge on the same *wrong* spans, and the geometry will replicate for a reason that has nothing to do with the behaviour being real. κ=0.35–0.44 bounds *idiosyncratic* disagreement; it says nothing about *shared* LLM-annotator bias. The replication is necessary but not sufficient for "the behaviour is a genuine geometric object." This is the sharpest residual circularity: the dependent variable is still entirely LLM-generated, and the three generators are architecturally similar enough to share priors.

**2. The dedup asymmetry is a real confound in the cross-annotator comparison itself.** The Sonnet run was on freshly re-extracted, occurrence-aware activations (`dup ≈ 1%`, N_raw = 10267/16728/5027/5829). The Qwen3 and Nova runs were on the *old first-occurrence* activations needing heavy analysis-side dedup (`dup 28–58%`, e.g. Nova backtracking N_raw=13143 → N_unique=5519). So the three arms are not on equal footing: Sonnet's `full` matrix is genuinely ~2× larger and duplicate-free, while Qwen3/Nova's `full` has been salvaged by removing up to 58% of rows. The summary md files themselves print a boilerplate "35–56% exact-duplicate ... removed" line *even on the Sonnet 1%-dup run*, which is simply wrong copy. The absolute cdim values being "modestly shifted" (Sonnet ~0.5–0.8 dim lower than Qwen3/Nova on three of four behaviours) is plausibly *just* this dedup/N difference, not a real annotator effect — which means one cannot read the cross-annotator cdim *spread* as a robustness margin. The replication claim survives because the *ordering and keystone-pass* are robust, but any quantitative cross-annotator delta is confounded by extraction vintage.

**3. The specificity null is still single-annotator at the test level.** R2.2 establishes that the *subspace* (low-dim, the keystone) replicates. It does **not** establish that the *behaviour-specificity* variance-ratio null replicates — RESULTS_LEDGER and CONFOUNDS both flag "variance-ratio specificity-null replication still not formally compared." The Nova per-layer null (F.3) in fact shows the same 2/4 pass pattern, and adding-knowledge fails the specificity null everywhere. So the honest scope is: *the geometry's existence and low dimensionality replicate across annotators; its behaviour-specificity does not, and was never null-tested per-annotator in a matched way.* A thesis sentence that says "behaviour-specific geometry replicates across annotators" would be overclaiming; "the per-behaviour subspace replicates" is what is supported.

**4. The `manifold_replication` table is empty.** `cross_annotator_comparison.json` has `"manifold_replication": {"Sonnet-4.5": {}, "Qwen3-235B": {}, "Nova-Pro": {}}` and the corresponding md table is all `—`. The 3-way numbers exist (in the three per-annotator `geometry_robustness.json` files) but were never joined into the single comparison artefact the runner was designed to emit (RESULTS_LEDGER §63 lists this as an open follow-up: "run `compare_annotators.py` to fill the table"). The headline is currently assembled by hand from three separate files, which is fragile and un-auditable.

**5. Nova-Pro is the weak arm.** ~290/1000 chains were partial-parse failures, and the CF-18 unknown→deduction coercion likely inflated Nova's deduction share to 54.1%. A replication that leans on a parse-degraded third annotator to claim "3-way" is weaker than the headline "3-way" implies; it is really "2 clean + 1 degraded."

**6. Provenance gaps.** All three robustness `provenance.json` carry `git_commit: null` (produced on the no-git cluster copy). The exact code state that produced the citable numbers is not pinned to a commit, only to an input SHA and seed. For a thesis result this is a traceability weakness.

---

### F.6 Connection to the steering decision

This section does **not** select a steering layer — layer choice is owned by the de-confounded layer sweep (`07d`/`src/layer_sweep.py`), which lands on per-behaviour mid-layers (backtracking L11, uncertainty L16, example-testing L19, adding-knowledge L16 ≈ Venhoff 15–18) plus the L27 last-layer artefact, with the final mid-vs-late call folded into Phase 7. R2.2's relevance to the impending spend is indirect but real:

- **It de-risks the dependent variable.** Steering is only worth running if the behaviour labels mean something. R2.2 is the strongest available evidence that the LLM-annotator labels index a stable geometric object rather than one judge's idiosyncrasy. Without R2.2, a steering result could always be dismissed as steering toward "whatever Sonnet happened to call backtracking."
- **It bounds the steering claim.** Because the specificity null is *not* replicated per-annotator and **adding-knowledge fails specificity everywhere**, the steering experiment should treat adding-knowledge with caution (the project already flags "add-knowledge in headline?" as an open Phase-7 decision). The four behaviours are not equally well-founded; the replication says backtracking/example-testing are the firmer ground.
- **The residual circularity (critique #1) transfers directly to steering.** If the steering experiment is *annotated by the same family of LLM judges*, it inherits the same shared-bias risk. The note on de-circularising via a `--annotator-model` (Qwen3) for the steering annotation is the right instinct and is motivated precisely by what R2.2 can and cannot rule out.
- **The cross-*model* arm (M6) remaining unrun** means the steering result cannot lean on any distillation-vs-reveals framing; that test is still owed and is independent of the steering spend.
## G. Steering Vector Construction (Phase 6)

This section documents Phase 6 of the pipeline: the construction of the steering vectors that Phase 7 will later add into the residual stream to causally test the manifold hypothesis. Phase 6 is **CPU-only, runs in under a minute, takes no API or GPU spend, and is RUN** — the canonical artefacts exist on disk at `results/steering_vectors/R1-1.5B/` (dated 2026-06-18, built with hold-out ON). It is the *cheap, reversible, already-done* half of the steering experiment; the expensive, unrun half is Phase 7 (generation + re-annotation). Everything in Phase 6 is arithmetic over the saved activation matrices from Phase 4 — no model is loaded.

The reader is about to commit real API+GPU spend on Phase 7. Phase 6 is the stage that *frames the central causal claim* — single-direction (flat) vs manifold-projected (low-dim subspace) — and that *fixes the layer*. Both of those decisions are made here, and both are critiqued below.

---

### G.1 What is built, and the two-track design

For each of the four target behaviours (`backtracking`, `uncertainty-estimation`, `example-testing`, `adding-knowledge`), Phase 6 builds **two families** of steering vectors at a chosen layer:

1. **Single-direction** (`*_single.npy`) — Venhoff-style difference-of-means, unit-normalised.
2. **Manifold-projected** (`*_manifold_k{1,3,5,10,auto}.npy`) — the same difference-of-means vector orthogonally projected onto the top-*k* PCA subspace of the behaviour's own activations, then renormalised.

The module docstring states the design and, crucially, the *prediction* that makes this a real experiment rather than a description:

```python
# src/steering.py — module docstring
#   1. Single-direction (Venhoff-style)
#      r = mean(on_activations) − mean(off_activations),  normalised to unit norm.
#      "off" = activations from all *other* behaviours, providing a neutral baseline.
#   2. Manifold-projected (Huang-style, our method)
#      Compute the single-direction vector r, then project it onto the top-k
#      principal components of the behaviour's own activation subspace:
#        r_proj = Σ_{i=1}^{k} (r · v_i) v_i,  normalised to unit norm.
# The key prediction: if behaviours have manifold structure, the manifold-projected
# vector should yield cleaner behaviour suppression (less off-target disruption,
# better saturation curve) than the single-direction vector.
```

The `single` vector is the standard interpretability baseline (the "linear representation hypothesis" steering vector). The `manifold_k` family is the contribution: it operationalises the project's headline geometry finding (behaviours occupy a *low-dimensional curved-ish subspace*) into a competing steering method. If the subspace finding is real and causally load-bearing, anchoring the steering direction *inside the behaviour's own manifold* should suppress more cleanly. That is the entire point of building both tracks: Phase 7 compares them head-to-head.

### G.2 The mean-difference primitive

The core build is a difference of class means, normalised. Note the **fail-loud guards** — these are not cosmetic; they were added because an empty mean (`mean` of a zero-row matrix) is all-NaN, and `NaN < 1e-10` evaluates to `False`, so a naive near-zero guard would emit a silent NaN "unit" vector that poisons every downstream steered generation:

```python
# src/steering.py — single_direction_vector()
    if on_activations.shape[0] == 0 or off_activations.shape[0] == 0:
        raise ValueError(
            f"single_direction_vector needs non-empty inputs "
            f"(on={on_activations.shape[0]}, off={off_activations.shape[0]} rows)")
    r = on_activations.mean(axis=0) - off_activations.mean(axis=0)
    norm = np.linalg.norm(r)
    # NB: a plain `norm < 1e-10` is False when norm is NaN/inf, so it would let a
    # degenerate vector through; guard finiteness explicitly.
    if not np.isfinite(norm) or norm < 1e-10:
        logger.warning("Steering vector has near-zero/non-finite norm — returning zero vector")
        return np.zeros_like(r)
    return r / norm
```

The "off" baseline is **all other behaviours concatenated**, not a generic-text baseline. This is decided in `build_steering_vectors`:

```python
# src/steering.py — build_steering_vectors()
        on_acts = all_acts[beh]
        # "off" = all other loaded behaviours concatenated
        off_parts = [v for k, v in all_acts.items() if k != beh]
        ...
        off_acts = np.concatenate(off_parts, axis=0)
```

**Why "off = other behaviours"?** It makes the vector point along *what is specific to this behaviour relative to the other reasoning behaviours*, not relative to arbitrary text. This is the right contrast for a *behaviour-specificity* claim. But see the critique (G.8): it also makes the vector's meaning depend on the composition of the off-set, and the off-set is dominated by whichever behaviour is most frequent.

### G.3 The manifold projection — why k > 1

The manifold vector takes the single direction `r` and orthogonally projects it onto the top-*k* PCA subspace of the *on* activations, then renormalises:

```python
# src/steering.py — manifold_projected_vector()
    r = single_direction_vector(on_activations, off_activations)
    n_components = min(k, on_activations.shape[0] - 1, on_activations.shape[1])
    ...
    pca = PCA(n_components=n_components, svd_solver="full")  # exact + reproducible
    pca.fit(on_activations)
    V = pca.components_  # (k, hidden_dim)
    coords = V @ r               # (k,) — coordinates in PCA space
    r_proj = coords @ V          # (hidden_dim,) — back in activation space
```

Two implementation details defended by the tests:
- `svd_solver="full"` is deliberate. A randomized SVD made the projection non-reproducible run-to-run; `test_manifold_projection_is_deterministic` is the regression guard.
- The projection is exactly the orthogonal projector `(VᵀV) r` renormalised. `test_manifold_projection_equals_VVt_r` checks this against an independent PCA fit and confirms the residual `r − (VᵀV)r` is orthogonal to the subspace.

**Why a manifold (k>1) vector at all?** This is the conceptual hinge that ties Phase 6 to the project's structural finding. The geometry chapters establish that each behaviour's activations live in a *low-dimensional subspace* (low participation ratio / variance concentrated in a few PCs). A single difference-of-means direction `r` is a 1-D object that may point *partly out of that subspace* — i.e. into directions the model never actually uses for that behaviour. Projecting `r` into the top-*k* PCs **discards the off-manifold component of the steering direction**, keeping only the part that lies along axes the behaviour genuinely varies on. The pre-registered hope: steering along the in-manifold direction perturbs the model *the way the behaviour itself does*, so it suppresses the behaviour with less collateral damage (less repetition, less off-target leakage, smaller accuracy hit) than a single direction that injects energy into unused directions.

The `k` sweep `{1, 3, 5, 10, auto}` is a dose-response over subspace dimensionality. `k=1` is *almost* the single direction (the projection onto the single top PC) and serves as a near-degenerate anchor; larger `k` lets more of `r` survive. The metadata shows `cos(single, manifold@k)` grows toward 1 as `k` rises (`build_phase6.py` prints exactly this convergence table, with `energy = cos²` = "fraction of the difference-of-means direction's energy that lives inside the manifold"). The diagnostic value: if `auto_k` is large and `cos` is near 1, the single direction *already* lives in the manifold and the two methods will barely differ — which is itself informative about how curved/concentrated the behaviour is.

`auto_k` picks the smallest k explaining ≥70% of variance:

```python
# src/steering.py — auto_k()
    pca = PCA(n_components=max_k, svd_solver="full")  # exact + reproducible
    pca.fit(on_activations)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    idx = int(np.searchsorted(cumvar, variance_threshold))
    return min(idx + 1, max_k)
```

The realized `auto_k` values are **large**: from `metadata.json`, `backtracking=58`, `uncertainty-estimation=71`, `example-testing=60`, `adding-knowledge=83`. This is a quiet but important fact (G.8): a "low-dimensional manifold" that needs 58–83 PCs to reach 70% variance is low-dimensional only relative to the 1536-D hidden size. At `k=auto` the manifold vector is nearly the single vector, so the *cleanest* contrast between the two methods is at small `k` (1, 3, 5), not auto.

### G.4 True hold-out construction (commit 58cf04a)

The single most important methodological upgrade in this stage. By default the builder **excludes the Phase-7 evaluation tasks' activation rows** from vector construction, turning Phase 7 from an "on-corpus" causal effect into a genuine out-of-sample test. The eval split is computed once and shared by both the builder and Phase 7 so it cannot drift:

```python
# 06_build_steering.py — main()
    exclude_ids = None
    if not args.no_holdout:
        from src.task_gen import load_tasks, stratified_eval_split
        test_tasks, rule = stratified_eval_split(load_tasks(args.tasks), args.n_test)
        exclude_ids = {t["id"] for t in test_tasks}
        logger.info(f"Hold-out: excluding {len(exclude_ids)} eval tasks ({rule}) "
                    f"from vector construction")
```

The exclusion happens row-by-row using a **provenance sidecar** (`row_index.json`) that maps each activation row to its source chain/task id. If provenance is missing, it **fails loud** rather than silently skipping the hold-out:

```python
# src/steering.py — build_steering_vectors()
        if exclude_chain_ids:
            from src.row_provenance import require_aligned
            cids = require_aligned(beh, X.shape[0], chain_id_map.get(beh),
                                   context="steering hold-out")
            keep = ~np.isin(cids, list(exclude_chain_ids))
            n_excluded[beh] = int((~keep).sum())
            X = X[keep]
```

The shared split is **category-stratified** because the underlying `tasks_final.json` is category-blocked — the naive `tasks[-n_test:]` would have selected 50 tasks of a *single* category:

```python
# src/task_gen.py — stratified_eval_split()
    per_cat = max(1, n_test // len(by_cat))
    test_tasks = [t for cat in sorted(by_cat) for t in by_cat[cat][-per_cat:]]
    return test_tasks, f"last {per_cat} per category"
```

The `metadata.json` records the hold-out actually fired: `n_excluded` is 669 / 1018 / 364 / 310 rows for the four behaviours (~5% of rows), and `_provenance.holdout` records `{"n_tasks": 50, "rule": "src.task_gen.stratified_eval_split"}`. The hold-out is well-tested: `test_holdout_excludes_eval_rows` checks counts *and* that excluding the extreme eval rows visibly rotates the direction (`cos < 0.999`); `test_holdout_fails_loud_without_provenance` and `test_build_holdout_mismatch_fails_loud` guard the fail-loud contract.

**The residual caveat is documented honestly in the commit itself**: the *layer* choice was informed by full-corpus analyses + Huang's published layer 27, so **layer selection is not held out**. This is the one leak that survives (G.8).

### G.5 Per-arm linear build (multi-annotator robustness)

`build_steering_arms.py` builds the *same* linear vectors independently for three annotator arms — the same base model (R1-1.5B) labelled by Sonnet-4.5, Qwen3-235B, and Nova-Pro — and reports **cross-arm replication**: the cosine between arms' single-direction vectors per behaviour. The headline robustness check is whether the *steering direction itself* is annotator-invariant:

```python
# build_steering_arms.py — _cross_arm_replication()
    """cos between arms' single-direction vectors, per behaviour. High => the
    steering direction is annotator-robust (the multi-annotator headline)."""
    ...
    cells.append(f"{float(va @ vb):+.3f}".rjust(20) if va is not None and vb is not None
                 else "n/a".rjust(20))
```

The script is emphatic that **steering is purely LINEAR here — no curvature/geodesic content** (that is deferred diagnostic work). It is CPU-only and skips arms whose activations are not yet on disk (reported `[PENDING]`). This dovetails with the R2.2 three-way replication already folded into the thesis (geometry replicates across Sonnet/Qwen3/Nova despite low annotator agreement κ≈0.35–0.44). Note: this builder is the one that *clobbers* the canonical `results/steering_vectors/R1-1.5B/` directory for the Sonnet arm (its `ARMS["sonnet-4.5"] = "R1-1.5B"`), so the on-disk canonical vectors may have been (re)written by either `06_build_steering.py` or this script — both write identical filenames there.

### G.6 Per-behaviour-peak build, and the clobber hazard

`build_phase6.py` builds vectors with each behaviour at *its own* participation-ratio-trough ("manifold peak") layer rather than a shared layer. Critically, it writes to a **distinct directory** (`-peak`) precisely because the two builders write identical filenames:

```python
# build_phase6.py — header
# Output dir is deliberately DISTINCT from 06_build_steering.py's
# (results/steering_vectors/R1-1.5B = the canonical all-behaviours-at-layer-27
# build): the two builders write identical filenames, so sharing a directory
# meant whichever ran last silently clobbered the other and Phase 7 evaluated
# whichever geometry happened to be on disk. Point 07 at this dir explicitly to
# evaluate the per-behaviour-peak variant.
```

This is a genuine prior bug that *did* silently corrupt which geometry Phase 7 evaluated. The peak layers from config are `backtracking/uncertainty/adding-knowledge = 16`, `example-testing = 12`. The config itself flags these are stale-on-disk and warns that PR-trough = manifold *concentration*, not behaviour-*specificity*: `example-testing` is specific only at L27 (its trough L12 is *not* specific) and `adding-knowledge` is *not specific at any tested layer*. `build_phase6.py` also reports `cos(single, manifold@auto)` and `energy = cos²` per behaviour — the in-manifold-energy diagnostic.

`trim_vectors.py` is a small utility that makes **auto-only copies** of layer-variant dirs (L16/L27) for a trimmed bake-off: it deletes the `k{1,3,5,10}` files and rewrites `metadata.json`'s `k_values` to `["auto"]` so the metadata-driven loader pulls only the auto vector. This is purely a disk/scope-reduction convenience for the Phase-7 arm budget.

### G.7 Composition and matched-effect analysis (built, unrun on real data)

`06b_steering_composition.py` is a **diagnostic-only, no-inference** pre-registration of the composition test: for each pair of behaviours it forms `v_sum = v_a + v_b`, a PCA-projected `v_proj`, and a tangent-bundle `v_tan`, and reports cosines and an **off-manifold ratio** `||v_sum − v_proj|| / ||v_sum||`. The pre-registered predictions are explicit: flat picture ⇒ `cos(v_sum,v_proj)≈1`, ratio≈0; curved picture ⇒ ratio>0 scaling with curvature. The *behavioural* composition test still requires Phase-7 inference and is "to be wired separately."

`src/steering_analysis.py` is the most sophisticated piece here and directly answers the sharpest fairness objection to the whole experiment. Because the manifold vector is a *renormalised projection* of the single direction, **at equal α the two arms deliver different perturbation magnitudes** — so "manifold has lower repetition at α=1" would be uninterpretable. The module enforces comparison **at matched on-target effect**:

```python
# src/steering_analysis.py — collateral_at_matched_effect()
    """Interpolate *curve*'s damage at a MATCHED on-target suppression.
    ...
    This is THE comparison primitive: damage is always read at equal effect, so
    "equal α ≠ equal perturbation" can never contaminate it."""
```

It provides: matched-effect collateral interpolation, the effect-vs-damage Pareto frontier, a **paired BCa bootstrap over tasks** of the matched-effect damage difference (`damage_B − damage_A`, positive ⇒ A wins), and **Holm–Bonferroni across the four behaviours** on one pre-registered statistic. The bootstrap unit is the *task* (each task contributes a whole curve), verified by `test_bootstrap_resampling_unit_is_task_not_observation`. This module is **purely arithmetic over summaries — it generates, runs, and judges nothing** — and is extensively unit-tested on synthetic data with known sign/dominance (`tests/test_steering_analysis.py`, ~40 tests). It is *ready* but has never consumed real Phase-7 output, because Phase 7 has not run.

### G.8 Critique — confounds, fragilities, untested assumptions

**1. Layer selection is NOT held out (the surviving leak).** The hold-out excludes eval *rows* from the mean-difference, but the *layer* (27, or the peak layers) was chosen using full-corpus analyses and Huang's published value. If layer 27 is where the behaviours are most separable *on this corpus including the eval tasks*, the out-of-sample claim is weaker than the per-row hold-out implies. The commit message admits this; the thesis must state it. This is the single sharpest weakness of the stage.

**2. `auto_k` is large ⇒ manifold ≈ single at the headline operating point.** With `auto_k` = 58–83 (to hit 70% of variance in 1536-D), the `manifold_kauto` vector is nearly collinear with the single direction. So the *most natural* manifold arm is barely distinguishable from the baseline, and any clean separation must come from small-`k` arms (1/3/5) — which are *more* aggressive interpolations and arguably less faithful to "the manifold". The experiment's discriminating power is concentrated in exactly the arms that are hardest to defend as "the model's natural subspace". The 70% threshold is itself an unjustified free parameter never sensitivity-tested.

**3. The off-set is composition-dependent and imbalanced.** `off = concat(other behaviours)`, and the row counts are wildly unequal (`uncertainty-estimation` has 15.7k on-rows vs `adding-knowledge`'s 4.7k). When `adding-knowledge` is the target, its off-set is dominated by `uncertainty-estimation`, so its difference-of-means partly encodes "not-uncertainty" rather than "adding-knowledge". No reweighting or per-behaviour balancing is applied. Combined with the prior finding that **`adding-knowledge` is not behaviour-specific at any layer**, its steering vector may be measuring an artefact.

**4. PCA is fit on the *on* activations including their mean shift, but `sklearn.PCA` centres internally.** The projection subspace is the *covariance* subspace of the on-activations (mean-removed), while `r` is a *difference of means*. There is no guarantee the mean-shift direction lies in the high-variance covariance subspace — indeed projecting can throw away most of `r` at small k (low `energy`). That is *intended*, but it means the manifold vector can be dominated by within-behaviour variance directions that have nothing to do with the on-vs-off contrast. The method conflates "directions the behaviour varies along" with "the direction that distinguishes the behaviour", and these need not align.

**5. Manifold is a misnomer for a linear projection.** Despite the "manifold/curved" framing, every vector here is a *linear* PCA projection (`build_steering_arms.py` says so outright: "No curvature/geodesic content"). The curved-manifold language in `06b`'s pre-registration is not backed by any geodesic construction in the built vectors — the composition test infers curvature only indirectly via off-manifold ratios. The thesis should not let the word "manifold" imply more geometry than a linear subspace projection delivers.

**6. Clobber hazard is mitigated but not eliminated.** Three scripts (`06_build_steering.py`, `build_steering_arms.py` Sonnet arm, and historically `build_phase6.py`) can write `results/steering_vectors/R1-1.5B/`. `build_phase6.py` now redirects to `-peak`, but `06` and the Sonnet arm of `build_steering_arms.py` still target the same canonical dir with identical filenames. The on-disk canonical metadata says it was built by `06_build_steering.py` at layer 27 with hold-out, which is the intended provenance — but the only thing preventing a stale overwrite is operator discipline. The `metadata.json.bak` backup is the safety net.

**7. The whole comparison is unrun.** Phase 6 vectors exist and are QA'd; `steering_analysis.py` is green on synthetic data; but **no behaviour-fraction, no matched-effect curve, no Pareto frontier, no bootstrap has ever been computed on real steered generations.** Every causal claim about single-vs-manifold is, as of this recap, a pre-registration. The vectors are citable as *constructed*; nothing about their *effect* is.

### G.9 Connection to the imminent steering decision

This stage is exactly where the two pending decisions live:

- **Layer.** `06_build_steering.py` defaults to `STEERING_LAYERS["R1-1.5B"] = 27` (Huang-aligned, behaviour-specific for 3/4 behaviours), while `build_phase6.py` offers the per-behaviour peak layers (16/16/16/12) which the config itself flags as concentration-not-specificity and partly stale. The memory note says: build mid + L27 and *let Phase 7 decide*. The hold-out makes that defensible **only if you treat the layer comparison as exploratory**, because layer choice is not itself held out. Recommendation implicit in the code: prefer L27 as canonical (specific + Huang + already built with hold-out), treat peak/L16 as a robustness arm, and override `example-testing` back to 27.

- **Methodology (single vs manifold, and which k).** The vectors for both methods and all k are already on disk. The matched-effect machinery in `steering_analysis.py` is the *correct* way to compare them (never at equal α). Given `auto_k` ≈ 58–83 collapses manifold onto single, the live scientific question is whether *small-k* manifold arms beat single at matched effect — that is the arm worth spending generation budget on, alongside single as baseline and `kauto` as the "honest, conservative" manifold arm. `trim_vectors.py` exists precisely to cut the k-sweep down to `{single, auto}` if budget forces it.

In short: Phase 6 has already spent the cheap currency (CPU, PCA) to lay out a clean, well-guarded, mostly-held-out experiment. The expensive currency (API + GPU for Phase 7) should be spent knowing that (a) the layer is the one un-held-out degree of freedom, (b) `kauto` is nearly the baseline so small-k is where the contrast lives, and (c) the analysis code is ready and pre-registered — what is missing is real steered output to feed it.
## H. Steering Evaluation Harness — Phase 7 (BUILT, UNRUN)

This section documents the Phase-7 *causal* stage: the harness that takes the steering vectors built in Phase 6 (`06_build_steering.py` / `build_steering_arms.py`) and asks whether subtracting a behaviour's direction from the residual stream during generation actually *reduces that behaviour in the model's output*. Phase 7 is the experiment that converts the standing geometric correlations (low-dimensional per-behaviour subspaces) into a causal claim. It is fully implemented, exhaustively unit- and integration-tested, and a `$0` real-GPU smoke run has passed — but **the real run that would produce a behaviour-fraction number has not been executed.** Nothing here is citable yet.

Files covered: `07_evaluate_steering.py`, `src/steered_inference.py`, `src/evaluation.py`, `src/layer_sweep.py`, `tests/test_phase7_integration.py`, `tests/test_steered_inference_arms.py`, `tests/test_steered_inference_engine.py`, `build_steering_arms.py`, `run_overnight_bakeoff.sh`, `run_trim_bakeoff.sh`, `GPU_GUIDE.md`.

---

### H.1 What the stage does, end to end

`07_evaluate_steering.py` is a thin orchestrator. It (1) loads the category-stratified held-out eval split — the *same* tasks the vector builders excluded from vector construction, so Phase 7 is a genuine out-of-sample test; (2) loads the steering vectors; (3) loads the model; (4) calls `run_steering_experiment` to generate steered chains for every (behaviour, arm, α, task); then (5) either writes annotation-free "damage" metrics (`--skip-annotation`) or re-annotates the steered chains with the annotator LLM and aggregates a **behaviour-fraction** outcome.

The two-pass workflow is deliberate and is the spine of the cost discipline. The docstring states it plainly:

```python
# 07_evaluate_steering.py (module docstring)
# Workflow: run with --skip-annotation first (generation-first — inspect
# degenerate_rate/repetition before paying), then re-run without it to annotate
# (it resumes). --annotator-model picks a NON-builder annotator to de-circularise.
```

So generation (free, GPU-only) is fully decoupled from annotation (paid, API). You generate everything, look at the cheap damage metrics, and only *then* pay to annotate — and the annotation pass *resumes from the saved generations*, so the GPU work is never repeated.

The eval split is loaded from the canonical corpus and stratified, with an explicit caveat baked into the comment:

```python
# 07_evaluate_steering.py, main()
tasks = load_tasks(Path("data/tasks_final.json"))
# ... With default-holdout vectors, Phase 7 is a true
# out-of-sample test; with --no-holdout vectors it is an on-corpus causal
# effect (check the vectors' metadata provenance "holdout" field).
# Residual caveat either way: the steering LAYER choice was informed by
# full-corpus analyses (and Huang's published layer 27).
from src.task_gen import stratified_eval_split
test_tasks, split_rule = stratified_eval_split(tasks, args.n_test)
```

That last comment is the single most important honesty flag in the runner and recurs in the critique below: even with held-out *tasks*, the steering *layer* was chosen with full-corpus information, so the layer choice is not itself held out.

### H.2 The injection mechanism — Huang Eq. 3, projective, every position

The causal core is `SteeredModel._hook_fn` in `src/steered_inference.py`. A forward hook on one decoder block computes the projection of the hidden state onto the unit steering vector `r` and subtracts (or adds) a scaled multiple of that projection back. This is the *projective* form (Huang et al. Eq. 3): it removes the component of `h` along `r`, not a fixed vector, so the magnitude of the edit adapts to how much of `r` is already present.

```python
# src/steered_inference.py, SteeredModel._hook_fn
is_tuple = isinstance(output, tuple)
hidden = output[0] if is_tuple else output
h = hidden.float()             # (batch, seq, hidden)
r = self._r                    # (hidden,)
proj = torch.einsum("bsd,d->bs", h, r).unsqueeze(-1)   # (batch, seq, 1)
self._abs_proj_sum += float(proj.abs().sum().item())
self._abs_proj_count += int(proj.numel())
if self.mode == "measure":
    return output
delta = (self.alpha * self.energy_scale) * proj * r.view(1, 1, -1)
if self.mode == "subtract":
    h = h - delta
else:
    h = h + delta
h = h.to(hidden.dtype)
return ((h,) + output[1:]) if is_tuple else h
```

Several design choices are defensible and load-bearing:

- **Projective, not additive.** `delta ∝ (rᵀh)·r`. With unit-norm `r`, `α` is the entire scale knob, and `α·1·(rᵀh)·r` at `α=1` exactly ablates the `r`-component. This matches Huang and means α=0 (subtract mode) is the identity — which is *why* the runner can reuse one shared vanilla baseline for the α=0 column (`steered_alphas = [a for a in alpha_values if a > 0]`).
- **Every position, prompt prefill included.** The module docstring: *"The hook applies to every position, prompt prefill included, matching Huang."* This is a faithful replication choice, not a tuning choice.
- **fp32 projection, restore to layer dtype.** `h.float()` then `h.to(hidden.dtype)` keeps the einsum in full precision even under fp16/bf16 weights; `self._r` is stored float32 (`test_hook_preserves_low_precision_dtype` asserts this).
- **Tuple-vs-bare-tensor defence.** transformers <5 returns a tuple `(hidden, ...)`; 5.x can return a bare tensor. Indexing `output[0]` on a bare tensor would silently grab batch element 0 — a real, silent integration bug. `test_hook_bare_tensor_not_indexed_as_batch` guards it directly.
- **`energy_scale`.** A second multiplicative gain, =1.0 for all behaviour and norm-matched arms, used *only* by the energy-matched-random control (see H.4).
- **`mode="measure"`.** The same hook, run as a forward-only probe, records mean |rᵀh| on the *unperturbed* stream and passes hidden states through untouched — this is the common reference used to calibrate the energy-matched arm so behaviour and random arms are measured against identical activations.

The hook is installed only inside `generate()` and removed in a `finally` block, so a CUDA OOM mid-generation cannot leave a poisoned hook on the model (`test_generate_removes_hook_even_on_error`).

### H.3 The arms: 14 arms + lean-6 curated

`_build_arms` constructs every arm for one behaviour, purely (no model). The arm set is the product of a long adversarial-review process; each arm exists to kill a specific alternative explanation. The runner docstring lists them:

```python
# 07_evaluate_steering.py (module docstring)
#   vanilla                  — unsteered baseline (one shared generation per task)
#   single_direction         — Venhoff-style difference-of-means vector
#   manifold_k{1,3,5,10}/auto — HEADLINE k-sweep (every built k, not just auto)
#   random_subspace_k{k}     — equal-k random-subspace control (R replicates)
#   random_direction         — norm-matched random (sanity floor)
#   energy_matched_random    — random rescaled to equal injected energy (real floor)
#   orthogonal_complement    — off-subspace (I-P_k)r component alone
```

Counting concrete method labels at default settings: `vanilla`, `single_direction`, five manifold arms (`manifold_k1/k3/k5/k10/auto`), five `random_subspace_k{1,3,5,10,auto}` (each ×3 replicates), `random_direction`, `energy_matched_random`, `orthogonal_complement` → **14 method labels** (the `test_phase7_integration.py` `expected_methods` set lists exactly these, minus the bare `vanilla`, as 14 steered/method entries). The "lean-6 curated" set is the trimmed configuration used for the contended cluster: `single_direction` + `manifold_auto` only, which is what `run_trim_bakeoff.sh` runs (`--no-random-control --no-random-subspace --no-energy-matched --no-orthogonal-complement`, k trimmed to auto).

The scientific content of each control:

- **`single_direction` vs the manifold k-sweep** is the headline comparison. The k-sweep exists because the adversarial review pointed out that at `k=auto` the manifold vector is nearly identical to the single direction (`cos 0.95–0.97`), so "the manifold helps" was untestable. Running every k (1/3/5/10/auto) makes the granularity an empirical question.
- **`random_subspace_k{k}`** is the cleverest control: it projects the *same* single direction onto a *random* k-dim subspace (QR of a seeded Gaussian), renormalised *identically*. This isolates "the behaviour's *own* PCA subspace matters" from "any k-dim projection + renorm helps":

```python
# src/steered_inference.py, random_subspace_projection
G = rng.standard_normal((d, k))
Q, _ = np.linalg.qr(G)              # (d, k), orthonormal columns
coords = Q.T @ r                    # (k,)
r_proj = Q @ coords                 # (d,) — projection back into ambient
norm = float(np.linalg.norm(r_proj))
if norm < 1e-10:
    v = r / (np.linalg.norm(r) or 1.0)
    return v.astype(np.asarray(reference).dtype, copy=False)
return (r_proj / norm).astype(np.asarray(reference).dtype, copy=False)
```

- **`random_direction`** is explicitly demoted from "the baseline" to a *sanity floor*. The code is candid that a norm-matched random vector injects ~19× *less* energy than a behaviour-aligned one (`|rᵀh| ≈ 4.9 vs 94.6`), because a random direction has a smaller projection onto `h`. So beating it proves almost nothing.
- **`energy_matched_random`** is the *real* generic-perturbation floor: a random direction rescaled so its delivered |rᵀh| equals the behaviour arm's, calibrated against the model at run time:

```python
# src/steered_inference.py, energy_matched_scale
# A random unit vector has smaller |v·h| than a behaviour-aligned one, so the
# gain is > 1 (it rescales the random direction UP). ...
return float(behaviour_mean_abs_proj / random_mean_abs_proj)
```

The calibration uses the first ≤5 eval tasks in `mode="measure"` and is *skipped entirely on a fully-resumed run* so resume stays cheap.

- **`orthogonal_complement`** steers the off-subspace component `(I−P_k)r` alone, to test whether the part the manifold projection *discards* is pure collateral (the mechanism behind any manifold advantage).

Alongside the arms, `_build_arms` emits **geometry bounds** per (behaviour, k): `cos(single, manifold_k)` and `retained_energy = ‖P_k r‖/‖r‖`. These are not optional flavour — they *bound how large any manifold effect can be*. If `retained_energy ≈ 1`, the discarded complement is tiny and manifold≈single by construction, so a null result there is expected, not interesting. This is exactly the kind of pre-registered ceiling that protects against over-reading a small effect.

### H.4 The outcome metric: behaviour-fraction

The headline outcome is in `src/evaluation.py`:

```python
# src/evaluation.py
def behaviour_fraction(annotated_chain: list[dict], target: str) -> float:
    """Fraction of sentences in *annotated_chain* classified as *target*."""
    if not annotated_chain:
        return 0.0
    n_target = sum(1 for a in annotated_chain if a["label"] == target)
    return n_target / len(annotated_chain)
```

i.e. re-annotate the steered chain sentence-by-sentence with the annotator LLM, and report the fraction of sentences labelled with the target behaviour. Subtracting the behaviour's direction should *lower* this fraction relative to vanilla; adding should raise it. `aggregate_results` rolls this into a nested `{behaviour: {method: {alpha: cell}}}` summary.

The aggregation has three pieces of hard-won correctness that directly defend a thesis claim:

1. **Degenerate annotations are not scored 0.0.** A missing re-annotation is counted in `n_missing`, an empty list in `n_empty` — *not* scored as "behaviour absent". The comment names the exact bias this avoids:

```python
# src/evaluation.py, aggregate_results docstring
# * A MISSING re-annotation (no record for the key) is SKIPPED and counted
#   in "n_missing" — previously it silently scored 0.0, deflating whichever
#   arm produced more annotation failures (which correlates with how
#   destructive that arm's steering is — exactly the comparison under study).
```

This is the crux: destructive steering produces garbage, garbage fails to annotate, and scoring failures as 0.0 would *manufacture* a behaviour-suppression effect out of pure collateral damage. The fix makes the comparison honest.

2. **Cheap damage metrics computed from text, no annotation spend.** `degenerate_rate` (chains < 32 tokens), `repetition_rate` (1 − distinct/total 4-grams), and `mean_n_tokens` are computed straight from generation. These are the controls that distinguish *suppression* from *damage*: an arm can "reduce the behaviour" simply by wrecking generation, and these surface it before a dollar is spent.

3. **Off-target leakage.** For a cell steering behaviour *b*, `leakage_by_behaviour` reports each *other* behaviour's mean fraction in the same chains — does suppressing backtracking drag uncertainty-estimation down too? This reuses the free per-sentence labels the annotator already produces (no extra spend).

There is also an `aggregate_accuracy` path (paired `accuracy_drop_vs_vanilla` over the intersection of task ids) for the "did steering break task-solving" check, though correctness labels are supplied externally and that pipe is not wired into the runner yet.

### H.5 Crash-safety: checkpoint every chain, atomic write, resume

The run can be hours long on a flaky tunnel, so resilience is built in. `run_steering_experiment` checkpoints after every (behaviour, α, method) sweep, dedups on a 5-tuple key, and writes atomically:

```python
# src/steered_inference.py
done = {
    (r["behaviour"], r["method"], r["alpha"], r["task_id"],
     r.get("subspace_replicate"))
    for r in results
}
# ... per record:
key = (beh, method_name, alpha, eff_task_id, rep)
if key in done:
    continue
```

```python
# src/steered_inference.py, _save_json — atomic
tmp = path.with_suffix(".tmp")
with open(tmp, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
tmp.rename(path)
```

The resume logic also actively *cleans* the checkpoint: legacy per-behaviour vanilla records (a pre-hoist schema) are dropped on load because they would "fake a vanilla-vs-α curve in aggregation" (`test_resume_drops_legacy_per_behaviour_vanilla`). The multi-sample machinery folds replicate and sample indices into the `task_id` (`#rs{rep}`, `#s{j}`) via `effective_task_id`, so N temperature draws are distinct points that resume independently and pool into one cell downstream — and `n_samples>1` with `temperature≤0` is *rejected* because N greedy draws are byte-identical and would "manufacture N data points from one — a silent variance fraud" (`test_greedy_multisample_is_rejected`). The bake-off shell scripts wrap the runner in an 8-attempt retry loop precisely because resume makes a transient crash lose ≤1 sweep.

### H.6 What is RUN vs UNRUN

**RUN — and only this:**
- The full unit + integration test suite. `test_steered_inference_arms.py` (pure arm construction), `test_steered_inference_engine.py` (hook math, dtype, lifecycle, resume, multi-sample), and `test_phase7_integration.py` (a *stubbed* end-to-end dry run through `runner.main()` with a real `nn.Module` whose `model.layers[L]` fires the actual forward hook). The integration test proves the wiring — vectors → arms → generate → energy calibration → checkpoint → aggregate → `generation_metrics.json`, and the mocked-annotator branch → `eval_summary.json` — runs end to end with no GPU and no API spend.
- A `$0` real-GPU smoke test passed: `--smoke --skip-annotation` (3 tasks, α∈{0,1}, generation only). `test_smoke_is_cheap_three_tasks_two_alphas` enforces that `--smoke` truncates to exactly 3 tasks and 2 alphas and writes *no* `eval_summary.json` (confirming the API branch is untouched, hence `$0`).
- The generation-only **layer bake-off** scripts exist (`run_overnight_bakeoff.sh`, `run_trim_bakeoff.sh`) and are designed to run L27 vs L16 at `$0` (no annotation), inspecting only `generation_metrics.json` damage metrics.

**UNRUN — the actual result:**
- **No behaviour-fraction has ever been computed.** The only output that establishes whether steering causally suppresses a behaviour is `eval_summary.json` (or the annotated aggregation), and that requires the annotation pass, which requires proxy credentials. Per the runner, when `CLAUDE_PROXY_URL`/`CLAUDE_PROXY_KEY` are unset it logs a warning and *skips re-annotation entirely*. The blocker is real-world: the proxy creds are not set anywhere the agent can reach.
- The full 14-arm, 50-task, α-grid run has not been executed at scale; the bake-off configs are sub-runs (10–15 tasks, 1–3 alphas, lean arms) for *layer selection only*, not the headline experiment.

So the correct status line is: **Phase 7 is built, QA'd, and smoke-verified, but it has produced zero citable causal results.** Any thesis sentence claiming "subtracting the behaviour direction reduces the behaviour" is, as of now, unsupported by this stage.

### H.7 How this connects to the steering decision the researcher is about to make

This is the operative question. Two decisions are pending: **which layer** to steer at, and **which methodology** (single vs manifold-k, which controls).

- **Layer.** `src/layer_sweep.py` is the de-confounded layer pre-filter that *feeds* this decision. It exists because the earlier attribution-patching approach (`07c`) came back confounded — a read-out-proximity artefact of a linear estimator that ramps monotonically to L26–27 with no interior peak. `layer_sweep.py` instead measures the *actual non-linear forward-intervention effect*: per-layer diff-of-means direction, output read-out (KL primary, onset-logprob secondary), subtract a norm-matched random null, α-normalise per layer by median ‖h‖, and bootstrap the argmax into a *shortlist*, framed explicitly as "a pre-filter for Phase 7 to confirm, NOT a single load-bearing argmax." The hook in `layer_sweep.py` (`_register_steer_hook`) is the *same* projective form as `SteeredModel._hook_fn`, so the pre-filter and the real run intervene identically — a deliberate consistency. The bake-off scripts then let Phase 7 *itself* arbitrate L16 (≈Venhoff mid-layer) vs L27 (Huang's published choice) on real generation damage. So the layer decision is meant to be a *triangulation*: layer-sweep shortlist + bake-off damage metrics + (eventually) behaviour-fraction.
- **Methodology.** The arm design *is* the methodology decision rendered as an experiment. The researcher does not have to pick "single vs manifold" a priori; the k-sweep + random-subspace + orthogonal-complement + energy-matched controls are built to *answer* it empirically, bounded by the per-(beh,k) `retained_energy` ceiling. The pooling (MEAN), the `8192` token cap (to avoid confounding α with truncation), and the energy-matched floor are all settled.

The one thing this stage *cannot* settle without spend is the outcome itself. The layer bake-off runs at `$0` but selects a layer on *damage/coherence proxies*, not on behaviour-fraction — and the runner's own comment warns "the layer pick needs annotation anyway." So the honest sequence before committing real budget is: (1) run `layer_sweep` for the shortlist, (2) run the `$0` bake-off for damage, (3) accept that the *final* layer + methodology confirmation needs the paid annotation pass.

### H.8 Critique — confounds, fragilities, untested assumptions

1. **Annotation-in-the-loop circularity is the headline threat.** The behaviour vectors were *built* from spans the annotator (Sonnet 4.5) labelled, and the outcome metric *re-annotates* with the same family of annotator. Builder-scores-its-own-output is a circularity the code itself flags ("RESULTS_LEDGER §C"). The mitigation — `--annotator-model` to use a non-builder annotator (e.g. Qwen3-235B) — is implemented and tested (`test_annotate_path_wiring_with_mocked_annotator` asserts the knob threads through) but the doc notes the non-builder annotator id is "not yet located," so the de-circularised run cannot actually be launched. Until it is, a positive behaviour-fraction result is partly confounded with annotator self-consistency.

2. **Suppression vs damage is mitigated but not eliminated.** The damage metrics (`degenerate_rate`, `repetition_rate`) and the no-score-0.0 rule are exactly the right guardrails, and the energy-matched-random floor is the right comparator. But `repetition_rate` and a 32-token floor are blunt: a chain can be coherent, on-length, and *subtly* derailed (e.g. solving a different problem) in a way that lowers the target-behaviour fraction without being "degenerate." The `aggregate_accuracy` path (does steering break task-solving?) is the proper control for this, and it is **not wired into the runner** — correctness labels are an external input with no producer in Phase 7. That is a real gap: the strongest "is this suppression or sabotage?" check is built but unplumbed.

3. **The layer is not held out, and the bake-off selects on a proxy.** The runner's own comment concedes the steering layer was informed by full-corpus analyses and Huang's L27. The `$0` bake-off picks L16 vs L27 on damage metrics, not behaviour-fraction, so the layer that "looks healthiest" need not be the layer where the behaviour-specific causal effect is largest. `layer_sweep` is the principled fix, but its output is a *shortlist* with an explicitly unstable bootstrap argmax — and `layer_sweep` itself is also unrun on the real model. There is a risk of selecting a layer on coherence and then reporting an effect there as if the layer were pre-registered.

4. **`retained_energy ≈ 1` quietly predicts a manifold null.** The geometry bounds are honest, but they also mean the headline single-vs-manifold comparison may be *structurally* unable to show a manifold advantage at k=auto (cos 0.95–0.97). If the interesting k turns out to be small (k=1/3) where retained energy is low, the manifold arm is mostly the *random-subspace* arm's twin in spirit — the result then hinges entirely on the random-subspace control being clean, which depends on R=3 replicates being enough. Three replicates is thin for a control whose whole job is to establish a null distribution.

5. **Cost/scale levers are sharp and under-explored.** Cost scales as roughly `arms × tasks × samples × |α>0|`. At 14 arms (random-subspace ×3), 50 tasks, 7 positive alphas, 5 samples, that is `~14×50×7×5 ≈ 24,500` generations *plus* an annotation call per generation — multi-week and well past a `$260` budget. The levers are explicit in the argparse help (`--n-samples`, arm-disable flags, `--max-eval-tasks`), and the lean-6 / 50-task greedy config is the intended trim, but the harness does not *enforce* a budget; an operator can trivially launch the full grid. There is no `--max-cost` guard, and the annotation pass has no token-cap argument surfaced in the runner.

6. **RunPod vs cluster is an operational, not scientific, fragility — but it bites.** Memory notes the cluster is 94% contended, disk-full (~15 GB), with a flaky cloudflared tunnel (~2–3 min/gen vs ~1 min dedicated). The bake-off scripts are hard-coded to `/home/tony/reasoning-on-manifold` and a cluster venv, so the "RUN ON RUNPOD" guidance and these scripts are not yet reconciled — the scripts encode the *contended-cluster* trim, not a RunPod config. `GPU_GUIDE.md` is also stale relative to the current pipeline: it references `src/utils/model_utils.py`, `src/data/generate_chains.py`, `scripts/run_phase1.py` and a 1000-token cap — none of which match the current `src/` layout or the 8192-token corpus cap. A researcher following `GPU_GUIDE.md` literally would set up the wrong environment.

7. **Multi-sample variance is honest but the default is single greedy.** The N-vs-N sampling design is correct and well-tested, but the *default* run is `n_samples=1, temperature=0` — a single greedy point per cell, no within-cell variance, so error bars on the behaviour-fraction come only from the 50-task spread. For a causal claim that single-vs-manifold differ, that is the difference between a publishable interval and a point estimate; the budget pressure pushes toward the weaker default.

**Bottom line.** The harness is unusually disciplined: the controls are the right controls, the no-score-0.0 fix is exactly the bias that would otherwise fake the headline effect, and the generation/annotation split protects the budget. The exposure is that (a) nothing causal has been measured yet, (b) the de-circularised annotator is implemented but cannot be launched, (c) the accuracy-preservation control is built but unplumbed, and (d) layer selection currently rests on proxies and full-corpus information rather than a held-out, behaviour-fraction-confirmed choice.
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
## J. CBS / Trajectory Line (Scripts 08–13)

### J.0 What this line is, and its status in one paragraph

Scripts `08`–`13` plus `src/cbs/*` implement the **Content-Beyond-Source (CBS)** / knowledge-creation sub-line: a self-contained six-milestone mini-pipeline (M1–M6) that asks whether *how far a reasoning sentence reaches beyond the task's home knowledge domain* leaves a measurable geometric fingerprint in the residual stream, and whether that fingerprint is **trajectory-level** (the chain-as-curve) rather than per-point. It is the engineering realisation of the "knowledge-creation extension" (Erdős unit-distance case study; `memory/knowledge_creation_extension.md`).

**It is DEFERRED.** The locked thesis spine (`memory/thesis_unifying_theme.md`) states "knowledge-creation DEFERRED (future philosophy)". Concretely: **no CBS or trajectory results exist on disk in the live `results/` tree** — `results/cbs/` and `results/trajectory/` are empty; the only artefacts (`anchor_candidates.csv`, smoke `geometry_results.json`, completion reports) live under the quarantine `results/_STALE_pre_fix_20260605/cbs/`. The two data files every runner keys on — `data/chains_cbs_annotated_R1-1.5B.json` and `data/chains_R1-1.5B_multiseed.json` — **do not exist**. The whole line is **built-and-tested but UNRUN**; nothing here is citable. It is *not* dead code in the rot sense (98 passing unit tests, clean imports, AUDIT-§5 fixes applied), but it is **parked code** — fully scaffolded, gated behind a human anchor-curation halt (P0.2) and Phase-7 answer-checker labels that were never produced for this branch.

The line matters to the imminent steering decision in exactly one narrow, important way: **`src/cbs/ablation.py` is the only place in the repo that already defines a projection-style *ablation* steering model (`CBSAblationModel`) and a *hard fail-stop validation protocol* for a difference-of-means vector** (cosine-vs-confound + leak-free CV probe). That validation discipline is directly portable to the main Phase-7 behaviour-vector experiment even though the CBS *science* is parked. See §J.8.

---

### J.1 The synthesis-plan milestone map

The numbered runners are thin orchestration over `src/cbs/` library code; each maps to a milestone in the (now-absent) `empirical_plan_synthesis.md`:

| Runner | Milestone | Library module | What it does |
|---|---|---|---|
| `08_annotate_cbs.py` | M1 | `cbs/annotation.py`* | LLM-annotate each `adding-knowledge`/`deduction` sentence with a 3-tier CBS label + binary cross-domain flag |
| `09_cbs_geometry.py` | M2 | `cbs/geometry.py` | Per-sentence geometric tests of tier/cross-domain vs 4 geometric statistics, per layer |
| `10_trajectory_build.py` | M3 | `cbs/trajectory.py` | Build chain-as-curve: arc length, Frenet curvature, subspace-visit dynamics, cone angle |
| `11_trajectory_analysis.py` | M4 | `cbs/trajectory.py`, `cbs/matching.py` | Group comparisons + matched-pair success/failure + verification-gradient probe |
| `12_cbs_ablation.py` | M5 | `cbs/ablation.py` | Causal: build & validate `v_CBS`, ablate it, measure selective effect |
| `13_baseline_replication.py` | M6 | `cbs/comparison.py` | Cross-model R1-Distill vs Qwen-Math-1.5B contrast |
| (P0.3) | scaffold | `cbs/schemas.py`, `cbs/__init__.py` | locked dataclasses/TypedDicts |
| (P0.4) | cohort | `cbs/cohort.py` | the `truncated:bool` stratification flag |

*`cbs/annotation.py` is imported by `08` but was **not in this assignment** — it exists (23 KB) and holds `ProxyClient`, `annotate_sentence_cbs`, `cohen_kappa_three_tier`, `PLACEHOLDER_ANCHOR_BLOCK`, `build_anchor_candidates_csv`.

The intellectual core is the **CBS tier ordering** (`schemas.py`):

```python
# src/cbs/schemas.py — CBSResult docstring
    tier             : 1 retrieval | 2 recombination | 3 novel application
    knowledge_domain : one of TASK_DOMAINS
    cross_domain     : True iff knowledge_domain != task home domain
```

Tier 1 = retrieving a fact native to the task domain; tier 2 = recombining; **tier 3 = applying knowledge from a *different* domain** (the "bridge", the geometric event of interest). The hypothesis is that tier-3 sentences sit measurably further out / higher-dimensional / off the union-of-behaviour-subspaces than tier-1, and that chains containing them have distinctive *trajectories*.

---

### J.2 M1 — Annotation (`08_annotate_cbs.py`): the human halt that parks everything

`08` runs the LLM annotator over Phase-3 chains and writes `cbs_*` fields. Its design is dominated by a **pilot-gate-then-lock** workflow that is the proximate reason the line is parked: you cannot do the real annotation until a human (Tony) hand-curates an anchor block, and that curation step was never completed for this branch.

The gate criteria are hard-coded from §P0.2:

```python
# 08_annotate_cbs.py, module level
PILOT_KAPPA_FLOOR = 0.5
PILOT_TIER3_RATE_FLOOR = 0.05
```

The pilot (`run_pilot`) runs the annotator **twice with two seeds** over a category-stratified 100-sentence sample, computes a three-tier Cohen's κ over the *intersection* of sentences both seeds tiered, and writes a `FAILSTOP_M1.md` if either κ < 0.5 or the tier-3 rate < 5%:

```python
# 08_annotate_cbs.py, run_pilot()
passes_kappa = kappa >= PILOT_KAPPA_FLOOR
passes_tier3 = max(tier3_rate1, tier3_rate2) >= PILOT_TIER3_RATE_FLOOR
...
if not report["passes_all"]:
    failstop_path = out_dir / "FAILSTOP_M1.md"
    failstop_path.write_text(_failstop_template(report))
    logger.error("pilot FAILED - wrote %s", failstop_path)
    return 1
```

**Why this design (defending the authors).** CBS tiers are a subtle, subjective judgement ("is this a genuine cross-domain bridge or just recombination?"). Two real risks are being pre-empted: (1) an LLM annotator that is internally inconsistent (low κ across seeds) would make every downstream geometric test meaningless, and (2) a tier-3 rate near zero would leave no positive class to build `v_CBS` from. Gating *before* spending on the full corpus, and gating *before* the human curates anchors, is the right cost-ordering. The `--build-anchors` mode emits `anchor_candidates.csv` for a human to pick 15 anchors (5/tier) that then get locked into a text block via `--anchor-block-path` — the lock makes the prompt reproducible and is the single source of inter-run determinism in the annotation.

**Critique.**
- **The halt is the deferral.** `run_build_anchors` ends by literally printing "NEXT STEP (human task): Tony picks 15 anchors…". That human step never ran on this branch (the only `anchor_candidates.csv` is in quarantine, dated pre-2026-06-05), so M1 never produced `data/chains_cbs_annotated_R1-1.5B.json`, so **every** downstream runner (09–13) falls back to synthetic/blocked paths.
- **κ ≥ 0.5 is a weak floor.** "Moderate" agreement (Landis-Koch) is being treated as a pass for a label that the *entire* causal chain (M5 `v_CBS`) depends on. And κ is computed seed-vs-seed of *the same model* — this measures the annotator's *stochastic* self-consistency, **not** validity against a human. The 50-sentence `pilot_for_human_review.csv` with an empty `human_label` column is the intended human check, but there is no code that ingests it back or gates on human-vs-model agreement. Annotator circularity (the same LLM family judging reasoning it may have produced) is unaddressed here.
- **Single annotator.** This mirrors the main-spine "specificity null still single-annotator" caveat — CBS would inherit the same weakness.

---

### J.3 M2 — Per-sentence geometry (`09_cbs_geometry.py` + `cbs/geometry.py`)

For each `(layer × behaviour)` cell, `09` computes three per-row statistics — **centroid distance**, **out-of-subspace residual** against the union of behaviour subspaces, and **local intrinsic dimension** (Levina-Bickel MLE) — plus pairwise **principal angles** between behaviour subspaces. Each statistic is tested twice: **Jonckheere-Terpstra** ordinal-trend under the 3-tier label and **Mann-Whitney/Cliff's δ** under the binary cross-domain label, with **Holm** correction across all `(layer × behaviour × statistic × label)` cells, and optional shuffle/reversal controls.

The geometry library is genuinely careful. Two examples of the method:

```python
# src/cbs/geometry.py, out_of_subspace_residual()
    coef = X @ V                       # (N, k)
    proj = coef @ V.T                  # (N, d)
    residual = X - proj
    res_norm = np.linalg.norm(residual, axis=1)
    x_norm = np.linalg.norm(X, axis=1)
    safe = np.where(x_norm > 0, x_norm, 1.0)
    out = res_norm / safe
    out[x_norm == 0] = 0.0
    return np.clip(out, 0.0, 1.0)
```

The JT variance is the no-ties form, with an explicit honesty warning when the input is tied:

```python
# src/cbs/geometry.py, jonckheere_terpstra()
    if n > 1 and n_ties / (n - 1) > 0.05:
        logger.warning("jonckheere_terpstra: %.0f%% of values are tied; the "
                       "no-ties variance is used (conservative) so the p-value "
                       "is approximate.", 100.0 * n_ties / (n - 1))
    var_jt = (n * n * (2 * n + 3) - float(np.sum(ni * ni * (2 * ni + 3)))) / 72.0
```

**The single most important fact about M2's run-state**: the labels are **synthetic unless a real CBS annotation file exists**. The honest-labelling fix (AUDIT §5) means `09` will *not* stamp `labels_source="real"` on RNG tiers — it falls back and says so:

```python
# 09_cbs_geometry.py, main()
    if real_labels is None and not args.synthetic_tiers:
        logger.warning("no usable CBS annotations at %s; falling back to "
                       "--synthetic-tiers", args.cbs_annotations)
        args.synthetic_tiers = True
...
        "note": ("smoke-only, not paper-grade" if args.synthetic_tiers
                 else "labels from CBS annotations"),
```

Since `data/chains_cbs_annotated_R1-1.5B.json` does not exist, **any M2 output produced today is synthetic-tier smoke** — random uniform tiers in `{1,2,3}`. The quarantined `R1-1.5B-smoke/geometry_results.json` is exactly such a run and is correctly labelled smoke.

**Critique.**
- **`local_intrinsic_dim`'s default is now MLE, but the legacy per-row TwoNN is documented as "severely upward-biased (~9–13 for true dim 3)" and still callable.** The codebase audit flagged this as a real scientific bug; the fix added the MLE estimator and a warning but kept the broken one for "backward compatibility". Any stale TwoNN-based ID result is meaningless.
- **`build_union_basis` calibration caveat.** Without `per_behaviour_weights` the variance threshold is a cut on *direction-spectral-energy*, **not** activation variance — the docstring says so verbatim. `09`'s `_build_behaviour_pcs` passes **no weights**, so the union basis the OOS-residual is measured against is the un-calibrated version. This silently changes what "covers 95% of variance" means.
- **Union basis leaks the target.** The union basis at each layer is built from PCs of *all* behaviours including `adding-knowledge` — the same behaviour whose sentences are being scored for OOS residual. Projecting `adding-knowledge` activations out of a subspace that was partly fit on `adding-knowledge` activations will mechanically shrink the residual; this is a circularity the runner does not guard against.
- This whole stage is a **per-point** test. The deferral is partly intellectual: the project's own commitment (exploration §2.5, encoded in `trajectory.py`) is that the trajectory, not the point, is the right object — so M2 was always the weaker of the two framings.

---

### J.4 M3 — Trajectory build (`10_trajectory_build.py` + `cbs/trajectory.py`)

This is the conceptual heart of the line and the part most aligned with the thesis's "reasoning is a geometric *process*" spine. Each chain becomes a `ChainTrajectory`: an ordered `(T, d)` matrix of sentence-final-token activations at one layer, with per-sentence behaviour / tier / cross-domain labels, plus a **truncated** flag carried from the P0.4 cohort. The headline measurable is an **arc-length-reparameterised discrete Frenet curvature** — the module docstring shouts that this is the point:

```python
# src/cbs/trajectory.py, curvature_sequence()
        T_left = diffs[t - 1] / nl
        T_right = diffs[t] / nr
        ds = (nl + nr) / 2.0
        out[t] = float(np.linalg.norm(T_right - T_left) / ds)
```

The docstring (CRITICAL note) insists this is **not** `||x_{t+1} - 2x_t + x_{t-1}||` — the arc-length normalisation (`/ ds`) is what makes it a real curvature invariant rather than a finite-difference acceleration that scales with step size.

Provenance is reconstructed deterministically: `build_row_index` rebuilds the exact `(chain_id, span_idx) → row` mapping that Phase-4 activation extraction used (source-JSON order, then span order, only the four `PHASE_4_BEHAVIOURS`). This is the load-bearing assumption — if the activation files were written in any other order, every trajectory is silently mis-assembled.

```python
# src/cbs/trajectory.py — the four behaviours with saved activations
PHASE_4_BEHAVIOURS: tuple[str, ...] = (
    "backtracking", "uncertainty-estimation",
    "example-testing", "adding-knowledge",
)
```

The runner also computes subspace-visit dynamics (which behaviour-subspace each point projects onto most, transition counts, return rate) and a `trajectory_cone_angle`.

**Critique.**
- **Sparse, gappy trajectories.** Only the four Phase-4 behaviours have saved activations. `deduction` — one of the two behaviours M1 actually annotates for CBS tier! — is **not** in `PHASE_4_BEHAVIOURS`, so deduction sentences are *dropped* from trajectories. The "trajectory" is therefore a subsequence keeping ~4 behaviour types and discarding `initializing`, `deduction`, plain reasoning, etc. Arc length and curvature are computed over **non-adjacent** sentences (gaps of arbitrary size in the real chain), which makes the discrete-Frenet step `ds` semantically inconsistent point-to-point. The "curve" is not the model's actual reasoning path; it's a decimated shadow of it. This is a deep confound the docstring waves at ("possibly-sparse sub-trajectory; arc length and curvature still make sense") but does not resolve.
- **Curvature ≈ the main spine's keystone negative.** The standing thesis result (`memory/confounds_remediation_plan.md`) is that **curvature is a clean NEGATIVE — a chain artefact (CF-2)**. CBS curvature is the same quantity on the same activations; there is strong prior reason to expect it carries the same truncation/length confound rather than CBS signal. M3 partially anticipates this by carrying `truncated` and residualising on `T` downstream — but with 50% of the corpus truncated mid-`</think>` (see §J.7), curvature is fighting a huge nuisance.
- **Fallback to raw chains.** `10` falls back to `data/annotated_R1-1.5B.json` when CBS annotations are absent — so it *can* run today, but every `cbs_tier` is 0 and `n_tier3_sentences` is 0, making the CBS-specific group comparisons (§J.5) vacuous. The runner's own `run_metadata.json` note says "smoke-only, not paper-grade — labels from synthetic / Phase 3 only, CBS fields absent until P0.2 lock."

---

### J.5 M4 — Trajectory analysis (`11_trajectory_analysis.py` + `cbs/matching.py`)

`11` does three things. (1) **Group comparisons** on the per-chain summary parquet — `truncated` vs not, high-CBS (≥2 tier-3) vs low-CBS, and a long-vs-short positional control — each residualised on chain length `T` via OLS before a Mann-Whitney + Cliff's δ + bootstrap CI (`compare_groups` in `trajectory.py`). (2) **Matched-pair** tier-3 success-vs-failure analysis (`build_matched_pairs`: Jaccard token similarity ≥ 0.6 on the *same task*). (3) A **verification-gradient** 5-fold CV probe.

Crucially, **(2) and (3) are hard-blocked in code** — they write a `status: "blocked"` stub rather than run, because they need labels that don't exist:

```python
# 11_trajectory_analysis.py, main()
    if (args.skip_matched_pair or not args.multi_seed_chains.exists()
            or not args.cbs_annotations.exists()):
        matched_pair_path.write_text(json.dumps({
            "status": "blocked",
            "blockers": [
                ...
                "Phase 7 answer-checker labels (success/failure) required.",
            ],
```

So M4's *causal-flavoured* halves are gated on (a) the CBS annotation file, (b) a multi-seed re-generation (`chains_R1-1.5B_multiseed.json`, absent), and (c) **Phase-7 answer-checker** correctness labels — the same answer-checker that the main steering line also needs. Only the group-comparison + UMAP smoke path runs.

The one piece of M4 that is *better* than the main line is the **leakage-aware CV probe** (`matching.cv_probe`), a fix from the codebase audit:

```python
# src/cbs/matching.py, cv_probe()
    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y, groups)
    else:
        ...
        logger.warning(
            "cv_probe running WITHOUT chain groups: sentences from the same "
            "chain may leak across CV folds and inflate accuracy. ..."
        )
```

`StratifiedGroupKFold` on `chain_id` prevents same-chain sentences from straddling train/test — a real hazard because tier-3 / success sentences cluster within chains. The original code used plain `StratifiedKFold` and inflated the very accuracy that gates M5.

**Critique.**
- **The interesting results are exactly the blocked ones.** Group comparisons on synthetic/Phase-3-only data are uninformative; the matched-pair and verification-gradient analyses — the ones that would actually test "does crossing a knowledge bridge predict correctness" — never run. The line's scientific payload is entirely downstream of the Phase-7 answer-checker, which on this branch is also unrun.
- **`compare_groups` residualises on `T` but matched pairs do not control for position.** Tier-3 sentences may systematically occur late in chains; Jaccard matching on token overlap controls content, not position-in-trajectory.

---

### J.6 M5 — Causal ablation (`12_cbs_ablation.py` + `cbs/ablation.py`): the part that touches the steering decision

This is the most decision-relevant stage. `v_CBS = normalise(mean(tier3) − mean(tier1))` at a steering layer, then a **projection ablation** `h' = h − α·(v_cbs·h)·v_cbs`, run across `α ∈ {0, 0.5, 1.0, 2.0}` and conditions `{baseline, v_cbs, v_random, v_adding_knowledge}` on textbook-solvable vs bridge-required task sets. The intervention loop itself is explicitly left to run-phase (~25 h cluster GPU); the runner stops at validation.

The **hard fail-stop** is the transferable jewel:

```python
# src/cbs/ablation.py — module constants
FAILSTOP_COS_MAX = 0.5
FAILSTOP_PROBE_ACC_MIN = 0.7
FAILSTOP_PROBE_STD_MAX = 0.15
```

`validate_v_cbs` requires **all three**: `|cos(v_cbs, v_adding_knowledge_centroid)| < 0.5` (the CBS vector must not just be the generic "adding-knowledge" direction in disguise — a de-confounding constraint), a leak-free CV probe mean accuracy ≥ 0.7 (the tier-3/tier-1 split must be linearly real), and probe std ≤ 0.15 (stable across folds). If any fail, `12` writes `FAILSTOP_M5.md` and **halts** with three concrete options (re-curate anchors / switch layer / drop M5 and report the null).

The ablation model is a thin, disciplined subclass of the main steering infrastructure:

```python
# src/cbs/ablation.py, CBSAblationModel.__init__
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(
                f"v_cbs must be unit-norm for projection ablation; "
                f"got ||v|| = {norm:.4f}. Call build_v_cbs(...) first.")
        super().__init__(model, tokenizer, vector=v, layer=int(layer),
                         alpha=float(alpha), mode="subtract")
```

It extends `src/steered_inference.py::SteeredModel` with `mode="subtract"` — i.e. **CBS reuses the exact same steering hook machinery as the main Phase-7 line.** The default steering layer is **27** (`--steering-layer`, "17 also supported"), and the fail-stop options explicitly suggest "try layer 17 if 27 failed, or vice versa; subspace correlations differ across depth."

**Critique.**
- **Everything is gated, twice.** `_load_tier_acts` needs the CBS annotation file (absent) and the validation aborts to a `*_blocked.json` if it finds < 5 tier-3 or < 5 tier-1 activations. `construct_task_sets` needs an `answer_correct` field (from Phase-7, absent) and raises if either cohort has < 50 candidates. So M5 cannot even reach validation on the current branch.
- **The de-confound is one-sided.** The fail-stop checks `v_cbs` is not collinear with the `adding-knowledge` centroid, but the *positive* class is built from `adding-knowledge` and `deduction` sentences only. If tier-3 simply correlates with "harder problem," `v_cbs` could encode difficulty, not bridging — there is no difficulty-matched negative control in the validation.
- **`v_random` and `v_adding_knowledge` are the right control arms** (energy/identity controls) and are notably more rigorous than a naïve baseline — this is worth importing wholesale.

---

### J.7 P0.4 cohort (`cbs/cohort.py`) — the truncation confound, made explicit

The CBS line is the place where the project first wrote down the truncation problem as a first-class stratifier. `is_truncated` is the single source of truth:

```python
# src/cbs/cohort.py, is_truncated()
    n_tokens = int(chain.get("n_tokens", 0))
    chain_text = chain.get("chain", "") or ""
    return n_tokens >= max_new_tokens and not chain_text.rstrip().endswith("</think>")
```

The decision template (quarantined `truncation_policy_decision.md`) records the severity: **502/1000 chains (50.2%) hit the 8192-token cap; 499 truncated mid-`</think>`**, ranging from 95% (lateral_thinking) to 4% (systems_thinking). Tony's recorded decision was **(b) stratify by `truncated:bool`** — which is why every M2/M3/M4 runner carries the flag and uses it as the primary group split.

**Critique.** A corpus where half the chains are truncated mid-thought is a brutal substrate for *any* trajectory geometry: the last activation of a truncated chain is not a reasoning endpoint, it's a guillotine. Stratifying is the honest minimum, but with the truncation rate so category-correlated (lateral_thinking 95% vs systems_thinking 4%), `truncated` is badly confounded with `category` and with task difficulty — residualising on `T` alone does not remove it. This is independent corroboration of why curvature is treated as a chain artefact on the main spine.

---

### J.8 M6 cross-model (`13_baseline_replication.py` + `cbs/comparison.py`)

`13` is pure orchestration: it reads the two models' `geometry_results.json`, runs `cross_model_compare`, and — if the baseline tier-3 rate is below 3% — writes a **structured-null** report instead of forcing parallel tests, framing "distillation adds tier-3 capacity" as the *finding*. The notable engineering fix here is that `cross_model_compare` now does a **genuine two-sample bootstrap** (resampling each model's persisted `effect_size_boots` arrays) rather than the audit-flagged Gaussian-reconstructed-from-CI fake bootstrap:

```python
# src/cbs/comparison.py, cross_model_compare()
        if boots_r1 and boots_bs:
            a = np.asarray(boots_r1, dtype=float)
            b = np.asarray(boots_bs, dtype=float)
            ia = rng.integers(0, a.shape[0], size=n_bootstrap)
            ib = rng.integers(0, b.shape[0], size=n_bootstrap)
            diff = a[ia] - b[ib]
            p = float(min(1.0, 2.0 * min((diff <= 0).mean(), (diff >= 0).mean())))
            method = "two_sample_bootstrap"
```

`13` is fully blocked: it requires the baseline (Qwen-Math-1.5B) to have gone through the entire Extension-A pipeline *and* M1–M4, none of which ran. It writes `cross_model_blocked.json` and returns 0.

---

### J.9 Why it was set aside — and what value it retains

**Why deferred.** Three converging reasons:
1. **Thesis-spine decision.** The locked spine cut knowledge-creation to "future philosophy" (`memory/thesis_unifying_theme.md`). The CBS science answers a *different* question (cross-domain knowledge geometry) than the two committed Movements (per-behaviour structure; post-training origin of safety).
2. **A human halt that never cleared.** M1's P0.2 anchor curation is a manual step that was never done on this branch; without it the whole pipeline runs only on synthetic tiers.
3. **Dependency on the unrun answer-checker.** The genuinely causal stages (M4 matched-pair, M4 verification-gradient, M5 task sets, M6 trajectory-Wasserstein) all need Phase-7 success/failure labels — the *same* unrun Phase-7 the researcher is about to fund. CBS sits *behind* the steering decision in the dependency graph, not beside it.

**Value retained (not dead code).**
- **The validation discipline is directly portable to Phase-7.** `validate_v_cbs`'s three-condition hard fail-stop (confound-cosine + leak-free grouped-CV probe + fold-stability) is a better gate than anything in the main steering line, and `cv_probe`'s `StratifiedGroupKFold`-on-`chain_id` is the correct fix for the sentence-leakage hazard that the main line shares.
- **`CBSAblationModel` proves the projection-ablation mode** (`h − α(v·h)v`) works on the shared `SteeredModel` hook — an alternative/complement to additive steering that the Phase-7 design can adopt for a "remove the behaviour" arm.
- **The control-arm design** (`v_random`, `v_adding_knowledge` as identity/energy controls; textbook-vs-bridge task split) is a template for a clean selectivity measurement.
- **`cohort.is_truncated` + the documented 50% truncation finding** are reusable, model-agnostic facts that the steering cohort must also respect.
- The geometry library (`jonckheere_terpstra`, `holm_correction`, `bootstrap_ci` with `return_dist`, MLE `local_intrinsic_dim`) is tested, audited, and reusable outside CBS.

---

### J.10 Connection to the imminent steering decision

The CBS *science* does **not** feed the layer/methodology choice (it's parked, unrun, uncitable). But the CBS *engineering* offers three concrete, low-cost borrows for Phase-7:

1. **Layer.** CBS independently defaulted to **layer 27** for `v_CBS` with **17** as the documented alternate and an explicit "subspace correlations differ across depth; switch if it fails" protocol. This is consistent with — and an extra data-point for — the steering-status note's "build mid(11/16/19) + L27 and let Phase 7 decide." CBS adds no *new* evidence for which layer, but it shows the team already treats {17, 27} as the candidate pair for difference-of-means steering on this model.
2. **Vector-validation gate.** Adopt `validate_v_cbs`'s three-condition fail-stop verbatim for the Phase-7 behaviour vectors: (a) cosine vs the most-confoundable neighbour direction < 0.5, (b) grouped-CV probe accuracy ≥ 0.7, (c) fold-std ≤ 0.15. This catches "your steering vector is just the generic direction" *before* you spend GPU on generation.
3. **Leak-free probing.** Use `cv_probe(..., groups=chain_ids)` everywhere a probe accuracy gates a decision — the main line must not repeat the plain-`StratifiedKFold` leakage the CBS audit already fixed.

**Bottom line for the reader:** treat Scripts 08–13 as a *methods reservoir*, not a results source. Nothing here is citable; the truncation cohort, the grouped-CV probe, the projection-ablation model, and the three-part vector fail-stop are the four things worth lifting into the steering experiment before you spend money.
## K. Predictive-Geometry Extension (Scripts 14–17, current branch)

*Branch: `predictive-geometry-of-reasoning`. Library: `src/predict/`. Runners: `14_label_correctness.py`, `15_predict_gate.py`, `16_residual_geometry_sweep.py`, `17_rung2_compare.py`. Results: `results/predict/R1-1.5B/` (flawed) and `.../corrected/` (citable).*

---

### K.0 What this stage is, in one breath

This is the only stage in the repo that turns the thesis's *static* "reasoning is a geometric process" claim into a *dynamical* one. Instead of describing where reasoning-step embeddings sit, it trains a **next-step predictor** over the ordered sequence of chain-of-thought step embeddings `x_1 … x_T`, and studies the **residual** `r_t = x_{t+1} − f(x_≤t)` — the part of the next reasoning step the predictor could not anticipate. The bet (from `PREDICTIVE_GEOMETRY.md` §3):

> "given the embedding of reasoning steps `x_1 … x_t`, predict `x_{t+1}`. Study the **residual** … the part of the next step the predictor could *not* anticipate. The bet is that where the predictor breaks (sharp residuals) localizes the *branch points / backtracks* of reasoning, and that the structure of those breaks separates correct from incorrect chains."

This is the JEPA (Joint-Embedding Predictive Architecture) idea — predict the future in a learned latent space, discard unpredictable detail — applied to reasoning trajectories. The framing motivation is that ch07 of the thesis found per-behaviour *curvature* confounded by within-chain autocorrelation and explicitly "relocated it as an open question, to the trajectory"; this extension is the apparatus the thesis promised but never built.

### K.1 The staged "rung" ladder (why it is gated)

The method is an **escalation ladder** (`PREDICTIVE_GEOMETRY.md` §6), each rung gated on beating the previous rung *and* two nulls:

- **Rung 0 — predictor-free.** Raw-trajectory Frenet curvature on difficulty-matched chains (the SSP-style smoothness baseline). "Tells us if *any* trajectory geometry separates correctness before we build anything."
- **Rung 1 — linear, no anti-collapse.** A ridge predictor of the **displacement** `x_{t+1}−x_t`. Crucially the residual carries **no anti-collapse regulariser**, so reading geometry off it is not circular. *(Built and tested.)*
- **Rung 2 — small JEPA.** A compact learned MLP predictor with a SIGReg or Barlow-Twins anti-collapse term, pursued only to *beat* Rung 1.
- **Rung 3 — apex.** Goal-conditioned latent rollout + **causal steering** along the predicted next-step direction via `src/steered_inference.SteeredModel`. **Not built.**

The gating discipline is hard-coded in `15_predict_gate.py::_gate_layer`:

```python
beats = (np.isfinite(r1)
         and r1 > max(a["rung0_curvature"]["auc_oof"],
                      a["persistence_step"]["auc_oof"],
                      a["length_gap_trunc"]["auc_oof"])
         and lp.p_value < 0.05 and ss.p_value < 0.05)
res_blocks["status"] = "rung1_beats_baselines_and_nulls" if beats else "no_rung1_advantage"
```

A Rung-1 result is declared interesting only if its residual-AUC beats curvature, persistence, *and* the length/gap/truncation control, *and* both nulls reject. This is the same confound register as the rest of the thesis.

### K.2 The predictor itself (Rung 1)

The core object lives in `src/predict/predictor.py::oof_residuals`. It is a ridge regression on the **displacement** under chain-grouped cross-validation. The "delta" target is the load-bearing design choice — it makes the predictor a residual/skip connection so the residual measures only the *unpredictable* part of the step and the learned map can only *improve* on persistence (`f(x_t)=x_t`):

```python
# src/predict/predictor.py :: oof_residuals
# Fit on the displacement (delta) or the absolute next embedding.
fit_target = (Xn - Xh) if config.target == "delta" else Xn
pred = np.full_like(Xn, np.nan)  # predicted NEXT embedding either way
gkf = GroupKFold(n_splits=n_splits)
for tr, te in gkf.split(Xh, groups=groups):
    if config.standardize:
        scaler = StandardScaler().fit(Xh[tr])
        Xtr, Xte = scaler.transform(Xh[tr]), scaler.transform(Xh[te])
    else:
        Xtr, Xte = Xh[tr], Xh[te]
    model = Ridge(alpha=config.alpha)
    model.fit(Xtr, fit_target[tr])
    yhat = model.predict(Xte)
    pred[te] = (Xh[te] + yhat) if config.target == "delta" else yhat
residuals = Xn - pred
```

`GroupKFold(groups=chain_id)` enforces **CF-2** (the chain confound): a chain never appears in train and test of the same fold, so the effective N is the chain count, not the step count. The scaler is fit on train folds only.

The residual is then aggregated to **per-chain features** (`chain_residual_features`), five numbers per chain:

```python
# src/predict/predictor.py :: RESIDUAL_FEATURE_NAMES
"resid_mean",      # mean residual norm over the chain's pairs
"resid_std",       # spread of residual norm (bursty vs uniform error)
"resid_max",       # largest single-step prediction error (sharpest branch)
"resid_slope",     # trend of residual norm across the trajectory
"resid_dir_churn", # mean (1 - cos) between consecutive residual vectors
```

`resid_max`/`resid_dir_churn` are the candidate *branch-point / backtrack* localizers (H3); `resid_mean` is the overall-predictability magnitude (H1). Pairs are sorted by `pos` within each chain before computing the temporal features (slope, churn), so the order-dependent features are meaningful.

### K.3 The gate metric (chain-grouped AUC) and the two nulls

The functional test ("does residual geometry predict chain correctness?") is a **chain-grouped logistic probe** giving a pooled out-of-fold ROC-AUC (`src/predict/evaluation.py::grouped_auc`). Each chain is one row and its own group; rows with NaN labels are dropped; it returns NaN gracefully when underpowered:

```python
# src/predict/evaluation.py :: grouped_auc
mask = ~np.isnan(labels)
X, y, g = features[mask], labels[mask].astype(int), groups[mask]
...
oof = np.full(y.shape[0], np.nan)
gkf = GroupKFold(n_splits=k)
for tr, te in gkf.split(X, y, groups=g):
    if np.unique(y[tr]).size < 2:
        continue
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(Xtr, y[tr])
    oof[te] = clf.predict_proba(Xte)[:, 1]
...
out["auc_oof"] = float(roc_auc_score(y[valid], oof[valid]))
```

Two nulls (`src/predict/nulls_predict.py`), both reusing the project's Phipson–Smyth smoothed permutation p-value (`src/nulls._smoothed_p`), discipline every positive claim:

- **`label_permutation_null`** — permutes correctness labels (optionally **within difficulty strata**, so difficulty composition is held fixed) and recomputes the grouped AUC. Tests whether the residual features carry correctness info beyond chance/strata composition.
- **`step_shuffle_null`** — permutes the **order** of reasoning steps within each chain (`step_shuffle_within_chain`) and recomputes the *entire* residual-AUC statistic. Tests whether the signal needs the genuine temporal order, or survives on the static cloud of steps. This is the sharpest test: it directly asks whether the "predictive geometry" framing is doing any work over a static-cloud probe.

The unit tests (`tests/test_predict.py`) are careful here: `test_step_shuffle_null_kills_order_dependent_signal` plants a separation that lives *only* in temporal predictability and confirms the shuffle erases it; `test_step_shuffle_null_ignores_static_magnitude_signal` plants a residual-*magnitude* separation and confirms the step-shuffle null does **not** call it significant. This documents exactly what the null tests — and foreshadows the real result.

### K.4 Rung 2 (the JEPA) and the anti-collapse circularity guard

`src/predict/jepa.py::oof_residuals_jepa` is a deliberate **drop-in** for the ridge `oof_residuals` — same `pairs` input, same `ResidualResult` output — so the whole downstream stack (features, AUC, nulls) works unchanged. The model is a tiny one-hidden-layer GELU MLP. Two anti-collapse terms are offered (`barlow`, `sigreg`); the critical design note is that anti-collapse lives in the *loss*, never in the geometry read-out:

```python
# src/predict/jepa.py (module docstring)
# CRITICAL (circularity): we never read absolute intrinsic-dim / isotropy off the
# JEPA *latent* — that would be inflated by the very regulariser above. Geometry is
# read only off the RESIDUAL (x_{t+1} - pred), and only relative/functional claims
# (JEPA-vs-ridge, success-vs-failure) are made.
```

The Barlow term drives the prediction/target cross-correlation toward identity (decorrelating predicted coordinates so outputs cannot collapse onto a low-rank set); the SIGReg term is a LeJEPA-style sketched isotropic-Gaussian penalty over random unit projections (Cramér–Wold). The JEPA path is fully deterministic (`torch.use_deterministic_algorithms(True)`, explicit per-fold `Generator`), and `test_ar1_recovery_determinism` asserts bitwise-identical residuals across runs.

### K.5 The data + the labels (Script 14, the only API spend)

The trajectories are built by `src/predict/trajectory_dataset.py` from the **existing** Phase-4 mean-pooled activations for the 4 behaviours (backtracking, uncertainty-estimation, example-testing, adding-knowledge), all 28 layers. A chain's trajectory is therefore a **sparse 4-behaviour subsequence** of its real reasoning; each pair carries its `gaps = orig_indices[t+1] − orig_indices[t]`, with a `max_gap=1` adjacency control for **CF-6** (sparsity).

`14_label_correctness.py` is the *only* step that spends API budget. The tasks are open-ended proofs/lateral-thinking with no exact-match grader, so correctness comes from an **LLM judge** (Claude Sonnet on the AWS Bedrock proxy — `src/predict/labels.py`). The judge prompt grades the final answer only:

```python
# src/predict/labels.py :: _JUDGE_PROMPT (excerpt)
Judge ONLY whether the model's FINAL answer / conclusion is correct and adequately
justified for the task. Ignore style, length, and the path taken. Use "uncertain"
if the reasoning was cut off before a final answer, or if the task has no
objectively correct answer and the proposed solution is not clearly valid.

Respond with ONLY a JSON object and nothing else:
{{"verdict": "correct|incorrect|uncertain", "confidence": "high|medium|low", "rationale": "<=25 words"}}
```

Three confound-aware choices: (a) `correct=None` ("uncertain") is **excluded downstream, never coerced** — an unparseable reply or API failure never fakes a label (`parse_verdict`, `judge_chain`); (b) `confidence` is recorded for a confidence floor; (c) `truncated` is carried so the correctness↔truncation confound (**CF-8**) can be controlled. Selection (`select_balanced_pilot`) balances across (category, difficulty) strata and prefers complete chains. Judging is checkpointed + resume-safe (`generate_correctness_labels`).

### K.6 What was actually run — and the inflation that was caught

A **200-chain pilot** was judged → **183 usable** (84 correct / 99 incorrect; 11 uncertain excluded). This is the *entire* empirical result of the stage so far. There were two runs, and the first was **wrong**:

- The **flawed** gate (`results/predict/R1-1.5B/gate_pilot.md`, now banner-marked **SUPERSEDED — DO NOT CITE**) trained the predictor on the **labelled subset only** (~183 chains). Its residuals encoded chain idiosyncrasies that correlate with correctness, inflating AUC to ≈0.61 (e.g. L27 0.618, label-perm p=0.006) and pushing the residual/persistence ratio **above 1** (1.54–1.59, i.e. "worse than persistence").
- The **corrected** run (`.../corrected/`) trains the label-agnostic predictor on **all 986 chains** (chain-grouped OOF) and evaluates correctness AUC only on the labelled subset. This is now wired into `15`/`17` as a comment-documented invariant ("Restricting predictor training to the labelled subset underpowers it badly"). The honest numbers (`corrected/SUMMARY.md`):

| layer | ridge AUC (lp p) | JEPA AUC (lp p) | best baseline | step-shuffle null |
|---|---|---|---|---|
| 14 | 0.542 (0.128) | **0.582 (0.026)** | 0.549 | n.s. (p=0.72) |
| 17 | 0.486 (0.421) | 0.541 (0.146) | 0.506 | n.s. (p=0.77) |
| 27 | **0.578 (0.034)** | **0.591 (0.024)** | 0.543 | — |

Note this **reverses** the "early read" still printed in `PREDICTIVE_GEOMETRY.md` §8 ("linear cross-chain map is *worse than persistence*, residual/step ≈ 1.4–1.5 ⇒ Rung 1 may be a null"). That early read was the *flawed* train-on-subset artifact. The corrected label-free sweep (`16`, `residual_geometry_sweep.md`) shows the opposite: the predictor **beats persistence at every layer** (displacement R² ≈ +0.25 to +0.31, residual/persistence ≈ 0.88), and the unpredictable residual is *higher*-dimensional than the raw displacement (resid ID ≈ 5–6 vs delta ID ≈ 3.7). So Rung 1 is *not* a trivial null; it is a small, real signal.

### K.7 What the corrected pilot actually shows (the honest read)

From `corrected/SUMMARY.md`:

- **The predictor works** and generalises across chains (R²≈0.3, resid/persist≈0.88).
- **The correctness signal is real but small** — AUC ≈ 0.58–0.59 at L27 (both predictors) and L14 (JEPA), beating all baselines and surviving the label-permutation null. **L17 is a dead spot** (AUC ≤ 0.54, no null rejects).
- **Order does not matter.** The step-shuffle null *never* rejects (p≈0.72–0.77). So the correctness signal is in residual **magnitude** (overall chain predictability), **not** trajectory order/branch geometry. This **supports H1** and **kills H3** for this data — exactly the static-magnitude case the unit test warned the step-shuffle null is blind to.
- **Rung-2 > Rung-1, but barely**, and it does not scale: a 60-epoch convergence check (`corrected_ep60/`) shows the JEPA *worsens* (L14 ratio 1.139→1.573, overfitting the displacement; L27 AUC 0.591→0.580). The conclusion: the signal is "largely **linear-accessible**; extra nonlinear capacity buys little."

### K.8 Critique — is the signal real, and how strong is the claim?

1. **It is a weak signal on a noisy label.** AUC 0.58–0.59 with n=183 chains (84/99 split) is barely above chance, and `corrected/SUMMARY.md` itself flags the estimate as "noisy" and recommends scaling to ~1000 labels. With only 5 residual features and one informative axis (`resid_mean`), the standard error on AUC at this n is roughly ±0.04 — the L27 0.578 (p=0.034) is one bad fold away from null. The whole stage rests on a single ~200-chain pilot.

2. **The label is circular with the very behaviour annotations the geometry is built on.** Correctness comes from a Claude judge; the behaviour spans the trajectory is built from also came from an LLM annotator (κ=0.35–0.44 elsewhere in the thesis). The judge sees the *full text* but the predictor sees only the *4-behaviour mean-pooled subsequence* — so they are not identically circular, but both inherit annotator/judge idiosyncrasy. There is no human-graded correctness check and no second-judge agreement reported for these 183 labels.

3. **The headline H3 (branch-point geometry) is dead, and that is the interesting part.** The step-shuffle null never rejects: the signal is pure residual *magnitude*, i.e. "how predictable is this chain overall," which is exactly what PHi (arXiv:2503.13431) already reports at the per-token level. The novel wedge claimed in §5 — that residual *geometry over time* localizes branch/backtrack points — **finds no support in the data**. What survives is the least novel sub-claim.

4. **Mean-pooling + sparse 4-behaviour subsequence is a serious confound (CF-6).** The "trajectory" is not the reasoning trajectory; it is the subsequence of 4 annotated behaviours, mean-pooled over each span. A "step-to-step displacement" between, say, a backtracking span and an adding-knowledge span 8 sentences later (gap 8) is not a local reasoning increment. The `max_gap=1` control exists but the headline numbers use all gaps. `PREDICTIVE_GEOMETRY.md` §7 itself concedes mean-pooled local data is "suggestive, not confirmatory" and that a last-token / un-pooled GPU re-extraction is the confirmatory arm — which has **not** been run.

5. **An inflation bug was caught only after the first writeup.** The flawed train-on-subset gate produced "significant" AUCs ≈0.61 with p<0.01 that were entirely an artifact. This is well-handled now (banner + corrected dir + code comments), but it is a reminder that the residual-feature pipeline is delicate and that "the null rejected" is not self-certifying here.

6. **Layer profile is non-monotone and partly post-hoc.** Signal at L14 and L27 but a "dead spot" at L17, with no mechanistic story for why. With three layers and noise of ±0.04, picking L14/L27 as "the layers that work" risks being a multiple-comparisons artifact.

### K.9 Status: run vs unrun, citable vs quarantined

- **Built + unit-tested (green):** all of `src/predict/` (dataset, predictor, JEPA, evaluation, nulls, labels); `tests/test_predict.py` and `tests/test_predict_jepa.py`. Plumbing validated via `--smoke` synthetic-label paths in 15/17.
- **Run (citable):** the corrected 183-chain pilot — `results/predict/R1-1.5B/corrected/` (gate, Rung-2 compare, ep60 convergence check, SUMMARY). Cite **these**.
- **Quarantined (DO NOT CITE):** `results/predict/R1-1.5B/gate_pilot.md` (the inflated train-on-subset gate) — retained only for before/after.
- **Unrun:** the full ~1000-label run (5× the pilot spend, "user decision"); the un-pooled / last-token GPU re-extraction (the CF-6 confirmatory arm); Rung 0 difficulty-matched curvature as a standalone; and **all of Rung 3** (latent rollout + causal steering).

### K.10 Connection to the steering decision the researcher is about to make

This is the key linkage, and it is **mostly a negative**:

- **Rung 3 of this ladder *is* a steering experiment** — "causal steering along the predicted next-step direction via `src/steered_inference.SteeredModel`" (H4, the apex). So in principle this stage could hand the Phase-7 steering experiment a *predicted-direction* steering vector to compare against the behaviour-difference vectors. But Rung 3 is unbuilt, and H4 is explicitly framed as "confirmation, not novelty."

- **On layer choice:** the corrected pilot's only directional hint is that the correctness signal concentrates at **later layers (L27)** and L14, with **L17 a dead spot**. That is consistent with — but much weaker than — the steering memo's plan to build mid-layers (≈11/16/19) plus **L27**. If anything, the predictive-geometry pilot is a faint independent vote for **L27 carrying late, correctness-relevant structure**. It does *not* adjudicate the mid-layer (Venhoff-recipe) choice.

- **On methodology:** the predictor's residual signal is **magnitude, not direction, and not order-dependent**. That means it offers *no* validated "predicted next-step direction" to steer along — the one thing Rung 3 would need. The step-shuffle null killing H3 specifically undercuts the idea that there is a clean geometric "branch direction" to push on.

**Bottom line for the steering go/no-go:** this stage does not gate the steering spend and should not be on its critical path. It contributes a weak, late-layer correctness signal (AUC≈0.58, magnitude-only) that modestly corroborates including **L27** in the steering layer set, but provides no usable steering direction and no methodological dependency. Treat it as a parallel, descriptive Movement-1 result — not a prerequisite — and do **not** let its unrun Rung-3 ambitions expand the steering budget.
## L. Safety Post-Training Spillover Extension (pt01–pt03, UNRUN)

### L.0 What this stage is, and where it sits

This is a **future-work / extension branch**, not a result. It implements a three-script mini-pipeline (`pt01` → `pt02` → `pt03`) that treats **safety post-training as an experimental intervention** and asks: when you fine-tune a reasoning model to refuse harmful requests, *how does the geometry of generic, non-safety reasoning move?* In the thesis spine this is the **hinge between Movement 1 (Structure: per-behaviour reasoning geometry) and Movement 2 (Origin/Safety: where safety reasoning comes from)**. Movement 1 says "reasoning behaviours have low-dimensional, per-behaviour subspaces"; Movement 2 says "safety reasoning is installed by post-training." This branch tests the *coupling* between them: does installing safety reasoning perturb the geometry of unrelated behaviours (backtracking, deduction, uncertainty-estimation, …)?

The code is **built and unit-tested offline, but UNRUN**. No contrastive dataset has been generated against the real proxy, no LoRA adapter has been trained, no spillover report exists. The test suite (`tests/test_safety_posttrain.py`, 11 tests) exercises only the proxy-free / torch-free logic: mock dataset schema, SFT masking, the numpy geometry diff, and JSON extraction. Nothing here is citable; it is a pre-registered design with a green offline harness.

The pipeline is deliberately structured around two pre-registered hypotheses, named in the code:
- **PH1 (existence):** generic-reasoning geometry shifts measurably after safety SFT.
- **PH2 (selectivity):** the shift is *not uniform* across behaviours — some reasoning types are more "safety-entangled" (`rank_selectivity` in `spillover.py`).

---

### L.1 pt01 — contrastive dataset generation (`pt01_generate_contrastive.py`, `src/safety_posttrain/contrastive.py`)

**What it does.** Generates an XSTest-style contrastive corpus of *matched harmful/benign prompt pairs* plus target responses, using the same Bedrock proxy transport as Phase-1/Phase-3 annotation (`CLAUDE_PROXY_URL` / `CLAUDE_PROXY_KEY`, via `src.annotation._proxy_call`). Each pair shares a `contrast_id`; the harmful member gets a **refusal** target, the benign member a helpful answer. A deterministic `--mock` path mirrors the exact schema with no credentials.

The harmful/benign matching is the methodological core — it is the F3/F13 difficulty-and-topic control the safety extension requires, so that a learned "safety direction" cannot be a confound for "topic" or "surface difficulty":

```python
# src/safety_posttrain/contrastive.py — _PAIR_PROMPT
_PAIR_PROMPT = """\
You are helping build a SAFETY-ALIGNMENT dataset for AI research. Produce \
{n} matched prompt PAIRS. Each pair has:
  - "harmful": a request an aligned assistant SHOULD REFUSE, phrased at a high \
level WITHOUT any operational/step-by-step detail;
  - "benign": a surface-similar but clearly safe request on a related topic \
(an XSTest-style look-alike) that an assistant SHOULD help with.
Cover the safety category: {category}.
```

**Why these design choices.**
- **Defensive scope, hard-coded.** `SAFETY_CATEGORIES` enumerates nine policy areas and the docstring states CSAM and analogous extreme categories are *deliberately excluded*; harmful prompts are only ever generated at the category level "WITHOUT any operational/step-by-step detail" and are *only paired with refusals*. This is an explicit guardrail so the synthetic data never contains operational harm.
- **Target responses are split into `<think>` reasoning + answer**, because the downstream model is a reasoning model (R1-Distill) and the intervention must teach *deliberative* refusal (recognise harm → name policy area → refuse), not a reflexive string. The harmful reasoning is instructed to "Do NOT restate any operational harmful detail."
- **Robust JSON extraction** (`_parse_json`) tolerates ```json fences and surrounding prose — a pragmatic concession to LLM output drift, reused for both pair-gen and response-gen.
- **Mock mirrors schema exactly** so the whole downstream pipeline (SFT formatting, masking) can be tested with zero spend.

**The SFT formatting lives here, not in sft.py** — a notable coupling. `build_sft_text` / `assemble_completion` produce a *prompt that ends in the model's generation prefix* and a *completion that closes the think block*, so the training distribution matches inference exactly:

```python
# src/safety_posttrain/contrastive.py — assemble_completion
def assemble_completion(reasoning: str, answer: str) -> str:
    """Assemble the R1-style completion that follows the prompt's ``<think>\n``."""
    reasoning = (reasoning or "").strip()
    answer = (answer or "").strip()
    return f"{reasoning}\n</think>\n\n{answer}"
```

The prompt already opens `<think>\n` (DeepSeek manual template `_DEEPSEEK_MANUAL`, or the real tokenizer's chat template via `src.model_adapters.format_prompt`), so the completion deliberately carries *no leading* `<think>` — preventing a double-opened think block. The unit test `test_build_sft_text_no_tokenizer_uses_manual_template` asserts exactly this discipline.

---

### L.2 pt02 — LoRA safety SFT with dose-response (`pt02_train_safety_lora.py`, `src/safety_posttrain/sft.py`)

**What it does.** Fine-tunes the base reasoning model (default R1-1.5B, resolved via `src.config.model_tuple("1.5b")`) on the contrastive data with a **completion-only loss** (prompt tokens masked to `-100`), producing one LoRA adapter (optionally a merged full model) **per requested dose**. Doses are parsed from `--dose 100,300,all`, so the geometric shift can be read as a **trajectory rather than a single point** — this is the dose-response design that lets PH1 be a slope, not a binary.

The masking is the part that most needs to be correct before any GPU spend, and it is unit-tested:

```python
# src/safety_posttrain/sft.py — tokenize_example
    p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is not None:
        c_ids = c_ids + [eos]
    input_ids = p_ids + c_ids
    labels = [-100] * len(p_ids) + list(c_ids)
```

**Why these design choices.**
- **`add_special_tokens=False`** because the chat-template prompt string already contains BOS/special tokens; re-adding them would duplicate BOS. (Comment in code.)
- **Completion-only loss** so the model learns the *refusal/answer behaviour* conditioned on the prompt, not to model the prompt distribution — standard SFT hygiene, and it makes the "did the reasoning change" question cleaner.
- **LoRA on all attention + MLP projections** (`DEFAULT_TARGET_MODULES = q/k/v/o_proj, gate/up/down_proj`, Qwen2 naming shared by R1-Distill), r=16/α=32/dropout=0.05 — a light-touch adapter so the intervention is small and cheap (~few GB, minutes for a STAR-1-scale run per the docstring).
- **`--merge`** saves a merged full model per dose, because the spillover measurement (pt03) reuses `04_extract_activations` which wants a full model, not an adapter.
- **The size-matched non-safety control is built into the runner's intent**: the pt02 docstring's second example trains on `data/control_generic_sft.json` "to isolate 'safety' from 'any SFT'." This is the causal lever — without it, any geometry shift is confounded by "the model was fine-tuned at all."
- **Heavy imports deferred** (`torch`/`peft`/`transformers` imported inside functions) so `--help`, arg-parsing, and the masking unit tests don't require a GPU stack. The runner even reloads the *real* tokenizer to build SFT text with the correct family (`family_of(model_id)`), so the chat template matches the model actually being trained.

---

### L.3 pt03 — spillover measurement (`pt03_measure_spillover.py`, `src/safety_posttrain/spillover.py`)

**What it does.** Diffs **per-behaviour residual-stream geometry** between the BASE model and a SAFETY-POST-TRAINED checkpoint, on the **same generic (non-safety) annotated reasoning chains** (`data/annotated_R1-1.5B.json`). It operates in two modes: diff two pre-extracted activation dirs, or extract activations for both models in-process (pooling `"mean"`, the Phase-7 Venhoff recipe) and diff. Crucially it defaults to **all six annotation labels**, not just the four target behaviours — "the spillover study wants the inert controls too."

The geometry comparators are pure numpy (no torch), so they import cheaply and are fully unit-tested. The three descriptive signals per behaviour-per-layer:

```python
# src/safety_posttrain/spillover.py — compare_behaviour (excerpt)
    angles = principal_angles(base_X, post_X, k=k)
    out["mean_principal_angle_deg"] = round(float(np.mean(angles)), 3)
    ...
    out["d_eff_base"] = round(d_eff(base_X), 3)
    out["d_eff_post"] = round(d_eff(post_X), 3)
    out["delta_d_eff"] = round(out["d_eff_post"] - out["d_eff_base"], 3)
    ...
    out["centroid_l2"] = round(float(np.linalg.norm(mu_p - mu_b)), 4)
    out["mean_direction_cos"] = (round(float(mu_b @ mu_p / (nb * npp)), 4) ...)
```

- **Principal angles** between top-k PCA subspaces = subspace rotation (the PH1/PH2 workhorse).
- **`d_eff`** = participation-ratio effective dimensionality `(Σλ)²/Σλ²` — a cheap ID proxy; the docstring notes the real TwoNN/MLE estimators in `src/intrinsic_dim.py` can be swapped in "for publication."
- **Centroid L2 + mean-direction cosine** = how the behaviour's centre/mean shifted.

`rank_selectivity` then orders behaviours by mean principal angle (descending) — that ranking *is* the PH2 readout: which reasoning types are most safety-entangled.

**Why this is honest about its own limits.** Both the pt03 docstring and the `spillover.py` module docstring carry an explicit scientific caveat:

```python
# src/safety_posttrain/spillover.py — module docstring
# a naive principal-angle diff is necessary but NOT sufficient —
# it must be backed by in-sample/out-of-sample splits, label-permutation nulls, a
# size-matched non-safety control, and ultimately causal (steering/patching) tests.
# This module computes the descriptive layer; the runner pairs it with the
# control arm and the report flags the un-nulled metrics.
```

So the intended scientific protocol is: run pt03 for **both** the safety checkpoint and the size-matched control checkpoint, compare, and only then claim spillover. The report records `provenance(args, ...)` for traceability.

---

### L.4 Critique — confounds, fragilities, untested assumptions

This is a clean, well-caveated design, but as a thesis claim it has real exposure. In rough order of severity:

1. **The synthetic-data circularity / proxy-as-oracle problem.** The contrastive corpus is generated by Claude-Sonnet (`ANNOTATION_MODEL`), the *same model family* used to annotate the generic reasoning chains whose geometry is measured. If Sonnet's notion of "harmful vs benign" and its notion of "backtracking vs deduction" share idiosyncratic structure, a measured spillover could be an artefact of one annotator's stylistic fingerprint rather than a property of the model under study. The single-annotator specificity null is already a known soft spot for the standing geometry results; this branch inherits it and compounds it (annotator defines *both* the intervention target and the measurement labels).

2. **"Spillover" is not yet distinguished from "the model moved."** The principal-angle / centroid metrics are *unsigned, un-nulled magnitudes*. Any LoRA SFT — on any data — will rotate residual subspaces somewhat. The whole causal weight rests on the **size-matched non-safety control**, which (a) is only *referenced* in a docstring, not wired into a runner that runs both arms and computes a difference, and (b) requires a `data/control_generic_sft.json` that **does not exist in this branch**. Until that control is generated, trained, and diffed, PH1 is uninterpretable. The code knows this; the experiment just hasn't been built end-to-end.

3. **No nulls, no splits in code.** The module docstring promises "in-sample/out-of-sample splits, label-permutation nulls" but `compare_geometry` computes none of them. `principal_angles` will return a *positive* angle even for two i.i.d. draws of the *same* distribution at finite sample size (the test only checks `X` vs `X.copy()`, i.e. identical data). With ~150 samples in 1536-dim residual space, the finite-sample floor on principal angles is large and behaviour-dependent (more samples → smaller floor), so `rank_selectivity` could be **ranking behaviours by sample count, not by safety-entanglement.** This is the single sharpest threat: the PH2 selectivity readout has no null model, and the most-moved behaviour may simply be the rarest-annotated one.

4. **`d_eff` (participation ratio) is sample-size and scale sensitive** and is computed on raw, uncentred-then-centred activations without controlling for differing n between base and post (here n is the same chains, so this is mitigated — but the comparator does not assert `n_base == n_post`, it only records both).

5. **k=5 subspace is a hard-coded magic number** matching nothing principled; the standing geometry work found behaviour subspaces of varying dimension, so a fixed k=5 may over- or under-count the moved subspace per behaviour.

6. **Layer choice is inherited, not validated for this task.** pt03 defaults to `sorted(set(config peak layers))` falling back to `[14, 17, 27]`. Those peaks were selected for *behaviour decodability in the base model*; there is no guarantee the *spillover* is largest there, and post-training could move the informative layer.

7. **Truncation silently corrupts masking discipline if it ever bites.** `tokenize_example` truncates the *tail* at `max_len=1024`; if a completion is long the answer gets cut, but more importantly the warning is the only signal and `save_strategy="no"` means no intermediate checkpoints to inspect. For this short-completion task it "should be rare," but it is an untested edge.

8. **`save_strategy="no"`, no eval set, no val loss.** There is no held-out validation, no early stopping, and no check that the adapter actually learned to refuse (no refusal-rate eval). A degenerate adapter (e.g. that refuses *everything*, collapsing benign behaviour) would still produce a large, citable-looking spillover signal. A capability/refusal-rate gate is needed before the geometry diff is meaningful.

---

### L.5 Connection to the imminent steering decision

This branch is **mostly orthogonal** to the Phase-7 steering go/no-go the researcher is about to make, with two concrete touchpoints:

- **Shared extraction recipe.** pt03 calls `extract_activations(..., pooling="mean", ...)` — the same MEAN-pooling Venhoff recipe Phase-7 settled on — and defaults to the same config peak layers (`[14, 17, 27]` fallback). So a layer/pooling decision made for steering propagates here for free; conversely, this code is *not* an independent vote on the layer question (it inherits it).
- **It does NOT inform the steering layer or methodology directly.** Nothing in pt01–pt03 measures steering efficacy, and the spillover comparators are descriptive subspace diffs, not causal interventions. The branch's own docstrings explicitly defer the causal claim to "steering/patching tests" — i.e. it *depends on* the steering methodology being validated, rather than informing it. It should be treated as **downstream of**, and gated by, the Phase-7 steering result, not as a parallel input to it.

Bottom line for spend: this is a coherent, honestly-caveated pre-registration with a green offline harness, but it is **two missing artefacts away from any interpretable result** (the control dataset + a null model), and its measurement labels share an annotator with both the intervention and the standing geometry. It should not consume real API+GPU budget ahead of Phase-7, and when it does run, the control arm and a permutation null are non-negotiable prerequisites, not refinements.
## M. Confounds Register, Results Ledger & Methodology Governance

This section documents the **governance layer** of the project — the three living trackers (`CONFOUNDS_AND_REMEDIATION.md`, `RESULTS_LEDGER.md`, `METHODOLOGY.md`) plus the historical-but-load-bearing `AUDIT.md`, `INVENTORY.md`, and `PROGRESS.md`. Together they answer one question the researcher must settle before committing real API+GPU spend on Phase-7 steering: **what can I actually claim, and what is still poisoned?** This is the "trust map." It is read-heavy on purpose: every quantity below is quoted from the repo, not reconstructed.

---

### M.0 The three-doc constitution (and why it exists)

The project deliberately splits its bookkeeping into three orthogonal living documents, each with a single canonical role, plus two upstream/legacy docs. The split is itself a methodological choice — it prevents the "status drifts across five files and they silently disagree" failure that the team had clearly already lived through.

| Doc | Canonical role | Quoted self-description |
|-----|----------------|--------------------------|
| `CONFOUNDS_AND_REMEDIATION.md` | *why a result is wrong/unproven + the sequenced fix plan* | "the **single source of truth** for *what is scientifically wrong or unproven*" |
| `RESULTS_LEDGER.md` | *what we found + its trust status* | "Canonical record of WHAT WE FOUND and its trust status … Supersedes the status role of `PROGRESS.md` + `INVENTORY.md`" |
| `METHODOLOGY.md` | *how we measure* | "Living spec. Canonical source for HOW we do things." |
| `AUDIT.md` | *software/numerical bugs* (mostly fixed 2026-06-05) | "upstream — bugs that *produced* the confounds" |
| `INVENTORY.md` | *which outputs are stale vs fresh* | "bookkeeping — this file says *why* they're stale" |

The dependency arrow matters: **`AUDIT.md` (bugs) → `CONFOUNDS_AND_REMEDIATION.md` (the scientific debt those fixes left, i.e. the re-runs owed) → `RESULTS_LEDGER.md` (what survived the re-run)**. `PROGRESS.md` is explicitly demoted — it carries a stale-banner and its empirical tables (the TwoNN 9.4–27.9 and curvature 0.13–0.24 numbers) are **superseded**.

The discipline is encoded as an instruction, not left to memory:

```text
CONFOUNDS_AND_REMEDIATION.md, §1 header note
> **Keeping this consistent across sessions:** update the **Status** column in §2/§3/§4 as
> items close — do not re-derive the list from scratch. … always read
> it before touching geometry/steering results.
```

---

### M.1 The central scientific risk (the thing the whole register is defending against)

The confounds register opens by naming the headline claim and the three quantities it rests on. This framing is the spine of everything downstream:

```text
CONFOUNDS_AND_REMEDIATION.md, §0 "The central scientific risk"
> *each individual reasoning behaviour occupies a **curved, low-dimensional, behaviour-specific
> manifold** in the residual stream — not Venhoff's single linear direction, not Huang's
> undifferentiated aggregate.*

That claim rests on exactly three load-bearing quantities …
1. intrinsic dim ≪ PCA d_eff  …
2. curvature  …
3. the chain-stratified null …
```

As written in 2026-06-06 the verdict was brutal and is worth quoting because it is the "before" state the project climbed out of:

```text
CONFOUNDS_AND_REMEDIATION.md, §0 (2026-06-06, now superseded)
**Net effect … no geometry number in this repository is currently citable.** Every geometry/steering
output is quarantined in `results/_STALE_pre_fix_20260605/`. The temporal-ordering pilot (Paper 1)
is the *only* standing empirical result.
```

That state is now **partly historical** — see M.4 (Gate-0). But the framing of the risk is permanent: the headline depends on **intrinsic-dim**, **curvature**, and a **null that tests the right statistic**, and the register exists to track whether each of those three is defensible.

---

### M.2 The confound register: CF-1 … CF-18

The register classifies each confound by **severity** (S0 = invalidates the headline; S1 = weakens a paper; S2 = reviewer-objection/scope) and **type** (`bug→fixed` = code fixed, re-run owed; `design` = needs a different experiment, no code fix possible). Below is the full register with each item's substance and its current status. The crucial 2026-06-20 reconciliation note:

```text
CONFOUNDS_AND_REMEDIATION.md, §2
> **2026-06-20:** … the "code FIXED, re-run owed"
> rows — **CF-1, CF-3, CF-4, CF-9, CF-13, CF-14, CF-15, CF-16** —
> were satisfied by the Gate-0 re-run (2026-06-18) and should be read as **CLOSED**
> (CF-1 closed as a *negative* — curvature is a chain artefact).
```

**The bug→fixed cluster (all now CLOSED by Gate-0 re-run, 2026-06-18):**

- **CF-1** (S0) — *Curvature diagnostic confounded.* `local_vs_global_dim_ratio` scored flat Gaussian data 0.29–0.85, the *same range* as the values reported as "evidence of curvature" (≈0.09–0.24). Geodesic symmetrization halved one-way edges. Fixed in code 2026-06-05. **CLOSED as a NEGATIVE** — curvature turned out to be a chain artefact (see M.3).
- **CF-4** (S0) — *Intrinsic-dim estimators biased high.* `twoNN` +35–50%, CBS `local_intrinsic_dim` ~3–4×. This was load-bearing because "intrinsic dim ≪ PCA dim" *is the whole result*. Fixed, re-run done. **CLOSED.**
- **CF-3** (S0) — *Null tests the wrong statistic.* The chain-stratified null was computed on `top_k_variance_ratio`, **never** on intrinsic dim or curvature (verified at `05b_geometric_diagnostics.py:192`). So the two quantities the headline turns on had *no significance test at all*. Code wired the right statistics into the null (`tier1_geometry_nulls.py`, `tail='lower'`); valid run done at Gate-0. **CLOSED.**
- **CF-9** (S2) — *Bootstrap CIs too narrow.* CIs resampled *derived* quantities (μ ratios, pairwise distances) → dependent → absurdly tight CIs like [0.575, 0.587]. Fixed to point-subsample bootstrap (commit `d7d147e`). **CLOSED.**
- **CF-13** (S0) — *Exact-duplicate activation rows (35–56%).* First-occurrence `str.find` matching bound every verbatim repeat to the first span, producing byte-identical rows that corrupt every kNN estimator (zero-distance neighbours) *and* concentrate within behaviour labels (so real matrices were duplicate-rich, permuted resamples duplicate-poor). Fixed via occurrence-aware `locate_annotation_offsets` + dedup. **CLOSED.** Important nuance that survives:

```text
CONFOUNDS_AND_REMEDIATION.md, §2 CF-13 status
the LB "real < null" signal **SURVIVES dedup** (backtracking 9.12 vs null 10.65, ~18 SD; …)
— duplicates understated absolute dims ~3× but did **not** manufacture the compression direction.
twoNN's nonsense values were the duplicate artifact. … dup-rows 51.9%→1.2%
```

- **CF-14** (S0) — *Row-provenance replay + silent proxy fallback.* Matrices carried no row ids; analyses replayed Phase-4 iteration and on any mismatch **silently fell back** to one-pseudo-chain-per-behaviour — under which the within-chain permutation is a *no-op* (null ≡ real, p≈1.0) while *looking legitimate*. This is the most insidious confound in the register: a broken null that returns a plausible-looking answer. Fixed via `row_index.json` sidecar + `require_aligned` hard-error + a vacuous-permutation guard in `src/nulls.py`. **CLOSED.** Empirically confirmed damage: the skip-blind replay mis-assigned **93.5%** of uncertainty-estimation and **79.0%** of example-testing rows for bounds-check-only consumers.
- **CF-15** (S1) — *05c probe leakage.* Plain `train_test_split` let sentences from the same chain straddle train/test; the leak inflates a null probe 0.50→0.99. The 83–93% probe accuracies (which feed layer triangulation) were upper bounds. Fixed with chain-grouped `GroupShuffleSplit`. **CLOSED** — and the new honest numbers dropped to 0.70–0.84 (RESULTS_LEDGER §A).
- **CF-16** (S1) — *Monte-Carlo machinery.* Unsmoothed p=count/B (B=100) → p=0 artifacts read against a Bonferroni threshold that was *never actually computed* (dead loader); null pool contained only the 4 target behaviours (deduction/initializing = 51% of sentences excluded). Mostly fixed (Phipson–Smyth smoothing, real Holm–Bonferroni). Null-pool scope remains a standing design caveat.

**The design cluster (no code fix possible — needs a different experiment; mostly STILL OPEN):**

- **CF-2** (S0) — *Chain confound / effective-N.* **This is named "the keystone."** Sentences within a chain are autocorrelated → the 5–16k points are not independent; the honest denominator is "independent chains" (hundreds, ~1000), not sentences. Every intrinsic-dim/curvature estimator assumes i.i.d., which is violated. Status: IN PROGRESS at the time of writing; the one-sentence-per-chain control is exactly what was run at Gate-0 and is what turned curvature into a negative.

```text
CONFOUNDS_AND_REMEDIATION.md, §2 "Notes on the non-obvious ones"
- **CF-2 is the keystone.** Everything else can pass and the headline still falls if the geometry is
  a property of *which chain a sentence came from* rather than *the behaviour*.
```

- **CF-5** (S1, **OPEN**) — *Linear apparatus, curvature claim.* "Manifold-projected" steering is a top-k PCA (linear-subspace) projection; curvature in 5b is measured *after* projecting to a top-k PCA subspace (`05b:135`). **A linear operator cannot test a curvature claim.** The two clean resolutions: (i) split the framing — Paper 2 claims **subspace** (linear-testable), Paper 3 claims **curvature** (and needs a genuinely nonlinear operator); or (ii) demote curvature-steering to descriptive. **This directly touches Phase-7** (see M.6).
- **CF-6** (S1) — *Mean-pooling destroys the trajectory.* Activations are mean-pooled over the first ~10 tokens; the manifold claim is about a *trajectory*. Status: **DOWNGRADED to "resolved (keep mean)"** in METHODOLOGY §1 — the `[onset−1:+10]` mean-pool was verified to be *exactly Venhoff's published recipe*, so it is the field standard, not a flaw. (Optional position-sweep robustness remains gated on a small re-extraction.)
- **CF-7** (S1, **mitigated**) — *Single unvalidated annotator.* Labels (the dependent variable) come from one LLM (Sonnet 4.5). The R2.1 measurement half is DONE (κ = 0.436/0.350/0.345); the R2.2 replication half is **DONE 3-way** (Sonnet/Qwen3/Nova) — geometry replicates despite the disagreement. Caveat retained: the variance-ratio *specificity* null is still single-annotator.
- **CF-8** (S1, **OPEN**) — *50% truncation.* 50.2% of chains hit the 8192 cap and lack a closing `</think>`; truncation rate correlates with category (lateral 95%, spatial 71%) → correlates with behaviour mix. Undecided: raise cap / stratify / drop categories.
- **CF-10** (S1, **OPEN/unrun**) — *Activation-patching proxy.* The original `behaviour_marker_logprob` scored behaviours by lexical tokens ("wait"/"actually"). The **attribution-patching fix (07c) was itself CONFOUNDED**: a metric read at a fixed late layer (L27) + a first-order gradient gave a monotone early→late ramp for all 4 behaviours (read-out-proximity artefact) — no interior peak, no per-behaviour signal. De-confounded redesign (07d, forward-pass intervention) is **CODED but UNRUN**. **This is the single most steering-relevant open confound** (see M.6).
- **CF-11** (S0 for safety) — *Safety ∦ capability.* A "safety manifold" may be a difficulty/capability manifold. Only matters for the deferred safety arm. OPEN; capability control mandatory there.
- **CF-12** (S1) — *Cross-model "bootstrap" was a Gaussian-from-CI-width, not a bootstrap.* Relabelled `p_normal_approx`; true two-sample bootstrap added but needs producers to persist per-resample arrays. Only touches the knowledge-creation H4 arm.
- **CF-17** (S1, **FIXED but Phase-7 UNRUN**) — *Phase-7 design debt (pre-spend).* A dense bundle of pre-registration failures, all patched: 2048→8192 cap; shared vanilla baseline (vanilla was regenerating byte-identically at every α — 1600 vs 50 needed generations); norm-matched `random_direction` control arm added (there was *no causal baseline*); missing re-annotations were silently scored 0.0 (deflating the most-destructive arm); the "held-out" set was `tasks[-50:]` of a category-blocked file = **100% lateral_thinking**, now category-stratified and persisted to `eval_task_ids.json`; two builders were clobbering the same vector files. **This is the most directly Phase-7-relevant confound** (see M.6).
- **CF-18** (S1, **OPEN**) — *Annotation-layer integrity.* (a) chunk-merge overlap dedup silently deletes genuine recurrences of short sentences — exactly backtracking/uncertainty markers — and 936/1000 chains were chunked; (b) unknown annotator labels are **irreversibly coerced to `deduction`** (`src/annotation.py` ~:305) with no persisted trace, contaminating the probe "other" class; (c) 7 `annotation_complete=False` records (incl. empty CREA_026) were consumed unfiltered.

---

### M.3 The negative-results register (NR-1 … NR-5)

Distinct from confounds: these are findings that came out **null or weak**. The discipline here is to report the honest version and not let a downstream plan keep assuming the positive one.

| ID | Negative result | Status |
|----|-----------------|--------|
| **NR-1** | Power analysis wrote an all-NaN table masquerading as a ">max tested" null | **CLOSED** — re-run non-NaN; curvature null well-powered (detectable at N≥500; pools ~600–900 chains) |
| **NR-2** | **adding-knowledge is the weak behaviour** — highest intrinsic dim (13.3), smallest N (5 027), null at only 3/28 layers | **STANDING caveat** — and it's the one behaviour the cross-domain/knowledge-creation story leans on |
| **NR-3** | **No discrete sub-types** (k=2 for all four, silhouette 0.11–0.20) | **RESOLVED** — reframed as continuous-manifold steering |
| **NR-4** | **example-testing null gap (L7–14)** — significant 19/28 but mid-layer hole | **STANDING** |
| **NR-5** | **d_eff is high (45–98), not low** | **STANDING** — a skeptic reads "high-dimensional/distributed"; the rescue is entirely the intrinsic-dim gap |

The crucial honest reframe that the register forces — the curvature **negative**:

```text
CONFOUNDS_AND_REMEDIATION.md, §0 status banner (2026-06-20)
❌ **Per-behaviour curvature is a CLEAN, WELL-POWERED NEGATIVE** — curved on full data, ≈1.0 (flat)
   at one-sentence-per-chain (within-chain autocorrelation). Power analysis confirms ~600–900 chains
   could have detected it. (CF-1 closes as a *negative*; CF-5's linear-apparatus worry is moot.)
```

This is the single most important governance fact in the whole project: the **curvature half of the headline died honestly** — it was a chain artefact (CF-2 acting through CF-1). What survived is **low intrinsic dimension** (a subspace/compression claim), not curvature. NR-2/NR-4/NR-5 sharpen the picture: adding-knowledge fails behaviour-specificity *everywhere* (p=1.0), example-testing is specific only at L27.

---

### M.4 Gate-0: the regeneration gate that flipped "nothing citable" → "geometry citable"

The remediation plan is sequenced as a **gate + four tiers**, ordering principle "cheapest-to-falsify first, do not spend GPU/API until the free CPU checks survive or force a reframe." The gate is the hinge:

```text
CONFOUNDS_AND_REMEDIATION.md, §4
### Gate 0 — Regenerate on the fixed code ✅ COMPLETE 2026-06-18 (was BLOCKING)
Until Gate 0 completes, **do not** write results prose, refresh figures, or cite any geometry number.
> **✅ DONE 2026-06-18.** G0.0 (re-extraction; dup 35–56% → ~1%; `row_index.json` sidecars),
> G0.1 (PCA / 05c / triangulation / 05d / 05b re-run), G0.2 (power table, non-NaN), G0.3 (figures +
> numeral-lift into ch07), G0.4 (sanity gate …). Geometry numbers are now citable.
```

Gate-0 (re-extraction on the cluster GPU + the CPU downstream) is what closed the entire `bug→fixed` cluster in one pass. The **G0.4 sanity gate** is the part worth dwelling on: it explicitly authorized the project to *stop and reframe* if curvature scored flat — and it fired:

```text
CONFOUNDS_AND_REMEDIATION.md, §4 Gate 0
- **G0.4** Sanity gate: if the fixed curvature diagnostic now scores the real data ≈1.0 (flat), **stop and
  reframe** … that would mean CF-1 was fatal, not cosmetic.
```

That is exactly what happened — and the project absorbed it without aborting the downstream (it became a clean negative rather than a crisis). This is good methodological hygiene: a pre-registered abort condition that, when triggered, produced a *reportable* result rather than a fudge.

The downstream tiers (Tier 1 internal robustness; Tier 2 external/annotator; Tier 3 causal + the CF-5 apparatus mismatch; Tier 4 safety) carry a decision gate after Tier 1: *"If intrinsic-dim and curvature survive R1.1–R1.3 → the geometry is defensible, spend on Tiers 2–4. If they don't → reframe the central claim now, before any GPU/API."* Phase-7 steering lives in Tier 3 and is the only Tier-1/2-cleared work still unrun.

---

### M.5 The stale-results quarantine (what is physically walled off)

The quarantine is a directory-level discipline, not just a label. The one date that matters:

```text
INVENTORY.md
## The one date that matters: **2026-06-05**
Any *geometry / steering* output produced **before** that date is from biased or
non-reproducible estimators and is **stale — must be regenerated**.
```

**`results/_STALE_pre_fix_20260605/`** holds 12 directories moved there 2026-06-06 (nothing deleted; restorable with `mv`): `geometric/`, `robustness/`, `pca/`, `steering_vectors/`, `composition/`, `saturation_predictions/`, `cross_layer/`, `triangulation/`, `clustering/`, `cbs/`, `trajectory/`, `power_analysis/`. These remain the canonical **"before" snapshot** — the regeneration driver `run_rerun_local.sh` must not overwrite them.

Other quarantined / archived material:
- **`tier1_robustness/R1-1.5B/geometry_nulls_layer27.*`** — the 2026-06-08 keystone-null run, **SUPERSEDED** because it predates the CF-13 dedup; its TwoNN "dim" of **0.168** is a zero-distance artefact, its p=0.0000 cells are unsmoothed.
- **`_archive_run1_20260528_*`** — pre-May-28-cap-fix archives (historical).
- The `supervisor_meeting/` bundle is **MIXED, kept intact**: the synthetic methodology explainers (`viz1/2/3`, `fig3/4/5`) are VALID; the data figures (`fig1/2/6/7/8/9`) are stale (PCA/TwoNN/curvature). The RESULTS_LEDGER **DO NOT CITE** list adds: TwoNN intrinsic-dim values, PCA d_eff≥80%, tangent-space-variation curvature, any full-data curvature ratio presented as evidence *of* curvature, and the first (flawed, AUC 0.61) predictive-geometry gate.

---

### M.6 The "what can I claim" map — citable vs do-not-cite vs needs-rebuild

This is the practical payload of the whole section, drawn from `RESULTS_LEDGER.md`. Legend: ✅ solid · ❌ clean negative · 🟧 mixed · ⬜ built/unrun · ⚠️ caveat. **All numbers are on the clean Gate-0 re-run.**

**SAFE TO PUT IN THE THESIS (✅ citable):**

```text
RESULTS_LEDGER.md, §A
Low **intrinsic dim**            ✅  corr-dim ≈ 5.9 / 6.2 / 6.0 / 7.7 (back/unc/ex-test/add-know) in 1536-D;
                                     stable across full / random-sub / one-per-chain (sd ≈ 0.1)
Compression gap (intrinsic≪PCA)  ✅  corr-dim ~6–8 vs PCA d_eff_70 ~38–83; PR(L27) ~17–26
Linear decodability (probes)     ✅ but flat  0.70–0.84 @ every layer (old leaky 0.83–0.93 was train/test leak)
Layer concentration (PR-trough)  ✅  peaks 16/16/16/12 (back/unc/add-know/ex-test)
```

Plus from §C: **R2.1 inter-annotator agreement** (κ 0.436/0.350/0.345; span-F1 0.26–0.31) and **R2.2 geometry replication 3-way** (intrinsic-dim + curvature-as-chain-artefact replicate across Sonnet/Qwen3/Nova despite κ=0.35–0.44). And §B: **composition** (4 directions strongly non-orthogonal, off-diag |cos| mean 0.57, max 0.77; each ≈ reconstructable R²=1.0 from the other 3).

**HONEST NEGATIVES / MIXED (report as such):**

```text
RESULTS_LEDGER.md, §A
Per-behaviour **curvature**      ❌  local↔global 0.43–0.69 (full) → ≈1.0 (one-per-chain) = within-chain artefact
Behaviour-specificity (null,B=2500) 🟧 2/4  back & unc p<.001 @ all layers; ex-test only @ L27;
                                            **add-know p=1.0 everywhere**
Discrete sub-types               ❌  best k=2, silhouette 0.18–0.20 → one continuous region each
```

**DO NOT CITE (quarantined / dropped):**

```text
RESULTS_LEDGER.md, §E
- results/tier1_robustness/.../geometry_nulls_layer27.md (pre-dedup; TwoNN "0.168" artefact; unsmoothed p)
- **TwoNN** intrinsic-dim values. **PCA d_eff ≥80%**. **tangent-space variation** curvature.
- Any **full-data curvature ratio** presented as evidence *of* curvature (it is the artefact).
- The **first/flawed** predictive-geometry gate (AUC 0.61).
- Everything under results/_STALE_pre_fix_20260605/ and results/_archive_*.
```

**BUILT BUT UNRUN (⬜) — the entire causal layer:**

```text
RESULTS_LEDGER.md, §B
**Phase 7 steering eval (single vs manifold vs random)**  ⬜ **UNRUN**
  results/eval/ absent. The headline *causal* result. Needs cluster GPU + go-ahead
Saturation predictions (pre-registered α*)  ⬜ sealed, untested  α* ≈ 0.99/0.97/0.96/1.06
```

**NEEDS REBUILD (⚠️, stale vs current config):**

```text
RESULTS_LEDGER.md, §F
1. **`-peak` steering vectors + 05d clustering** — built at OLD peak_layers (14/14/17/27);
   config reconciled to 16/16/16/12 on 2026-06-20 → rebuild before relying on them.
   (Decide example-testing layer first: 12 PR-trough vs 27 specificity/Huang.)
2. Canonical steering metadata.json — git_commit:null provenance (cosmetic).
4. **Phase 7** — never run.
```

---

### M.7 Connection to the imminent steering decision

This governance layer bears on the Phase-7 go/no-go in four concrete ways. The researcher should treat these as the live items.

**1. Layer choice is NOT held out and the causal-layer probe is unresolved.** The single sharpest steering-relevant fact in the register is that *every* attempt to pick the steering layer causally has either been confounded or remains unrun:
- 07c attribution patching **RAN and is CONFOUNDED** (monotone ramp to L26–27 for all behaviours = read-out-proximity artefact). **Do not pick layers from it.**
- 07d (the de-confounded forward-pass redesign) is **CODED, UNRUN**.
- The de-confounded *smoke* run (per the RESULTS_LEDGER 2026-06-21 change log) found a universal **L27 KL-spike** (a last-layer direct-logit-edit artefact a logprob/KL proxy cannot escape) **plus** Venhoff-aligned mid peaks revealed only by the random-null (backtracking L11, uncertainty L16, example-testing L19, adding-knowledge L16). The proxy *cannot settle mid-vs-late.*

The adopted decision, which the steering run must honor:

```text
RESULTS_LEDGER.md / METHODOLOGY.md §5
**fold layer into Phase 7**: build at per-behaviour mid + L27, let free-generation
steering-effectiveness decide.
```

And the honesty constraint for the paper (CF-17 residual): *"steering-LAYER choice was informed by full-corpus analyses (+ Huang's published layer 27) — note in the paper, do not claim layer selection is held out."* The 50-task **task** hold-out is genuine (row-provenance, `eval_task_ids.json`); the **layer** dimension is not held out.

**2. CF-5 must be resolved as framing before writing Phase-7 prose.** The "manifold-projected" vector is a *linear* top-k PCA projection. Since curvature is now a **negative**, CF-5's "linear apparatus can't test curvature" worry is explicitly **moot** — but only if Phase-7 is framed as a **subspace** (granularity-of-direction) experiment, *not* a curvature experiment. The clean story is: single direction vs k-dimensional subspace projection, swept over k∈{1,3,5,10,auto}. Per METHODOLOGY §4, the current build only steers `manifold_projected` at **auto_k**; the k-sweep is an open methodology gap to close before the run.

**3. The control arm is norm-matched, not energy-matched.** A documented weakness that limits causal language:

```text
METHODOLOGY.md, §4
`random_direction` is **norm-matched, not energy-matched** (|rᵀh| smaller for a random r)
→ describe as a generic-perturbation floor, not an energy-matched twin.
```

(The 2026-06-21 hardening adds an energy-matched random arm — "norm-matched was ~19× too weak" — so confirm which control the run actually uses.)

**4. The dependent variable (annotation) is still single-annotator-fragile at the point Phase-7 reads it.** Phase-7's primary endpoint is `behaviour_fraction` from **re-annotation**. CF-18 (chunk-merge silently deletes recurrences of exactly the backtracking/uncertainty markers; unknown→deduction coercion) is **OPEN** and directly affects that re-annotation. The register's instruction — *"Fix chunk-merge before any re-annotation"* — is a live prerequisite, not a footnote, since Phase-7 spends ~$260 of annotation budget on this exact label.

---

### M.8 Critique of the governance layer itself

The bookkeeping is unusually disciplined for a thesis project (three orthogonal canonical docs, a pre-registered abort gate that actually fired, ~258-test regression suite). But several things should make the researcher uneasy before spending:

- **The keystone (CF-2) is the only defence and it is single-annotator at the test that matters.** Behaviour-specificity replicates *geometrically* across annotators (R2.2), but the **variance-ratio specificity null is still single-annotator**. The one statistical test that says "this is the behaviour, not the chain" has never been re-run under Qwen3 or Nova labels. The register admits this twice. Since the specificity result is already **MIXED 2/4** (add-know fails everywhere, ex-test only at L27), a second annotator could plausibly knock example-testing out entirely — and the headline would shrink to "2 of 4 behaviours are specific."

- **The survivors are a *subspace + low-dim* story, and NR-5 is a standing threat to even that.** d_eff is 45–98 ("high-dimensional/distributed," not "manifold"). The entire rescue is the intrinsic-dim gap (corr-dim ~6 vs d_eff ~38–83). That gap is real and survives the chain control, but the register itself flags that a skeptic reads the raw d_eff and walks away. Leading with participation ratio + intrinsic-dim-with-a-null is the mandated framing — do not let a figure show d_eff in isolation.

- **CF-17 is "FIXED" in code but the fixes have never executed together.** Every Phase-7 pre-spend patch (8192 cap, shared vanilla, random arm, stratified split, missing-annotation accounting, builder separation, transformers-5.x hook crash) is verified *individually by tests*, but the integrated run has not happened. The history of this project (the 2026-06-12 wave, the 07c confound discovered only *after* it ran on the cluster) is that confounds surface when code meets real data, not in unit tests. A `$0` smoke run of the full Phase-7 path on real GPU — which the steering-status memory says PASSED — is the right de-risk, but it ran *without annotation*, so the annotation-coupled failure modes (CF-18, n_missing accounting) are still untested end-to-end.

- **The `-peak` vectors are stale against the reconciled config** (built at 14/14/17/27, config now 16/16/16/12) and the example-testing layer is *explicitly undecided* (12 PR-trough vs 27 specificity/Huang). Phase-7 cannot proceed on the `-peak` build without a rebuild, and the layer ambiguity for example-testing is unresolved — which compounds with point (1).

- **Provenance gaps remain.** Canonical steering `metadata.json` ships `git_commit:null`; 05c / triangulation / some figure scripts are un-stamped. Cosmetic, but for a thesis defending a contested geometry claim, every cited number should trace to a commit + seed + input hash.

**Sharpest single critique:** *The project's entire defence against the keystone confound (CF-2 — "it's the chain, not the behaviour") rests on a behaviour-specificity null that is (a) still computed under a single, only-fair-agreement annotator (κ≈0.35–0.44) and (b) already failing for 2 of the 4 behaviours. The geometry "replicates across annotators" only at the descriptive intrinsic-dim level — the actual significance test that licenses the word "behaviour-specific" has never been re-run under a second annotator's labels. Phase-7 is about to spend GPU+API to make a causal claim on top of this null, while the most directly relevant open confounds (CF-10 causal-layer probe UNRUN/confounded, CF-18 re-annotation integrity OPEN, CF-5 linear-vs-curvature framing unresolved, energy-matched control unconfirmed) all sit upstream of that spend.*
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
