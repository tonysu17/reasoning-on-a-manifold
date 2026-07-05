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
