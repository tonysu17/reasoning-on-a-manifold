# Reasoning on a Manifold

**What is the per-behaviour geometry of reasoning in thinking LLMs — and does it matter for control?**

This project synthesises two lines of work:

- **Huang et al. (NeurIPS 2025)** — *Mitigating Overthinking in Large Reasoning Models via Manifold Steering*: showed that the composite phenomenon of overthinking lives on a low-dimensional manifold (operationalised as a top-k PCA subspace) in activation space, and that projecting steering vectors onto it dramatically improves intervention effectiveness.

- **Venhoff et al. (ICLR 2025 Workshop)** — *Understanding Reasoning in Thinking Language Models via Steering Vectors*: identified distinct reasoning behaviours in thinking LLMs (backtracking, uncertainty estimation, example testing, knowledge augmentation) and showed each can be **controlled by a single linear steering vector**.

**The gap, as a three-rung ladder.** For *individual* reasoning behaviours:

1. **Rung 1 — a single direction suffices for control.** Established by Venhoff et al. (They showed sufficiency; they did not claim a single direction exhausts the behaviour's structure.)
2. **Rung 2 — the behaviour occupies a multi-dimensional subspace.** Huang et al. established this for the *composite* overthinking signal; whether each individual behaviour has its own low-dimensional, behaviour-specific subspace — and whether subspace-projected steering beats the single direction per behaviour — is open. **This is our primary claim, and it is testable with linear instruments.** (The instrument itself is borrowed — top-k-PCA projection is Huang's method and the baseline in Curveball; the novelty is the per-behaviour question, not the operator.)
3. **Rung 3 — the structure is curved beyond any linear subspace.** Concept-dependent curvature and nonlinear-beats-linear steering have been shown for non-reasoning concepts (persona/safety traits, cyclic features — see Related work); for reasoning behaviours this is open. Curvature claims require the estimator-validated diagnostics in `src/curvature.py` *plus* a nonlinear steering operator; our current steering apparatus is linear, so rung-3 results here are descriptive, not causal (see `CONFOUNDS_AND_REMEDIATION.md` CF-5).

The original motivating hypothesis — that behaviours come in discrete flavours (e.g. "arithmetic re-checking" vs "strategy-level pivoting") — was **not supported** by sub-type clustering (Phase 5d: k=2, silhouettes 0.11–0.18, pre-dedup data); the working picture is continuous low-dimensional structure per behaviour.

**The experiment nobody else is positioned to run:** Huang's composite overthinking direction is exactly reconstructable (same model, same layer 27, published top-k-PCA k=10 recipe — stable across v1→v2). Decomposing it into our per-behaviour subspaces — *which behaviours mediate overthinking mitigation?* — directly synthesises the two anchor papers and requires a per-behaviour annotation corpus like this project's. (Verified unclaimed as of 2026-06-12.)

The primary model is `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (28 layers, hidden dim 1536, steering layer 27). A knowledge-creation extension (the "CBS" / Content-Beyond-Source pipeline, phases 8–13) contrasts the distilled model against its empirically-verified base, `Qwen/Qwen2.5-Math-1.5B`. **Status: deferred.** The binary "does distillation create or reveal the mechanisms?" is substantially answered in the literature (base models already contain them — arXiv:2510.07364, arXiv:2507.12638); what survives as novel is the *geometric refinement* — does distillation sharpen, rotate, or expand the pre-existing per-behaviour subspaces? — which is future work until the core per-behaviour results ship.

## Repository layout

The codebase uses a **flat layout**: numbered phase runners live at the repository root and import shared logic from a flat `src/` package.

```
reasoning-on-manifold/
├── 00_pilot_gate.py … 13_baseline_replication.py   # phase runners (see below)
├── src/                                             # shared library modules
│   ├── task_gen.py            # Phase 1: task generation via Claude proxy
│   ├── chain_gen.py           # Phase 2: R1-Distill chain generation (greedy)
│   ├── hooks.py               # residual-stream forward hooks
│   ├── annotation.py          # Phase 3: sentence-level behaviour annotation
│   ├── activation_extraction.py # Phase 4: per-behaviour activation matrices
│   ├── pca.py                 # Phase 5: PCA manifold analysis
│   ├── curvature.py           # curvature diagnostics
│   ├── intrinsic_dim.py       # intrinsic-dimension estimators (TwoNN, etc.)
│   ├── nulls.py               # null-hypothesis hierarchy
│   ├── steering.py            # Phase 6: steering-vector construction
│   ├── steered_inference.py   # Phase 7: steered generation
│   ├── evaluation.py          # Phase 7: evaluation metrics
│   ├── activation_patching.py # Phase 7b: causal layer localisation
│   └── cbs/                   # knowledge-creation extension (phases 8–13)
│       ├── schemas.py  cohort.py  annotation.py  geometry.py
│       ├── matching.py  trajectory.py  ablation.py  comparison.py
│       └── tests/             # CBS pytest suite (full repo suite: `python -m pytest`)
├── configs/config.yaml        # single config source — loaded by src/config.py
├── data/                      # generated data (gitignored); see data/MANIFEST.md
├── results/                   # outputs (mostly gitignored)
├── PROGRESS.md                # detailed build/run status and empirical findings
├── GPU_GUIDE.md               # cluster / GPU notes
└── pyproject.toml             # packaging + dependencies
```

`src/` is a flat module collection (not the older `data/extraction/analysis/steering/...`
sub-package hierarchy that earlier versions of this README described).

## Installation

There is no `setup.sh` and no conda environment file. Install the package in
editable mode against `pyproject.toml`:

```bash
pip install -e .              # core (Python >= 3.10): numpy, scipy, scikit-learn,
                              # matplotlib, seaborn, pandas, tqdm, pyyaml, openai
```

Optional extras:

```bash
pip install -e ".[gpu]"       # torch, transformers, accelerate, huggingface_hub
                              # — needed for Phases 2, 2b, 4, 7 (GPU)
pip install -e ".[cbs]"       # statsmodels, umap-learn, POT, pyarrow, pytest
                              # — needed for the CBS extension (phases 8–13) + tests
```

### Credentials

Task generation (Phase 1) and behaviour annotation (Phase 3) call an LLM through
a proxy rather than a local model. Set:

```bash
export CLAUDE_PROXY_URL=https://...
export CLAUDE_PROXY_KEY=rp_...
```

(See `.env.example`. Phase 3 uses Claude Sonnet 4.5 via the proxy because the
GPT-4o build Venhoff et al. used is unavailable there; this deviation is noted
in the methods.)

## Pipeline

Phases run top to bottom. **Local** phases need only CPU + the proxy; **GPU**
phases load the model and should run on the cluster (see `GPU_GUIDE.md`).
Status and quantitative findings live in `PROGRESS.md`.

| Phase | Runner | What it does | Where |
|------|--------|--------------|-------|
| Pilot gate | `00_pilot_gate.py` | HISTORICAL — checks 1–5 import a batch-annotation API that was never implemented; the pilot actually ran through `03 --pilot`. Kept as a record, guarded against accidental runs | — |
| 1 | `01_generate_tasks.py` | Generate 1000 reasoning tasks (100 × 10 categories) via the proxy | Local |
| 1.5 | `04_cleanup_tasks.py` | Regenerate `lateral_thinking` with a classic-puzzle blocklist, top every category to 100, final dedup → `tasks_final.json` | Local |
| 2 | `02_generate_chains.py` | Run R1-Distill on the task corpus to produce reasoning chains (greedy, max 8192 tokens) | GPU |
| 2b | `02b_generate_baseline_chains.py` | Non-reasoning baseline: one Q/A-scaffold response per task from Qwen-2.5-Math-1.5B (no `<think>`), identical output schema | GPU |
| 3 | `03_annotate_chains.py` | Sentence-level behaviour annotation (verbatim Venhoff prompt); checkpointed/resumable | Local |
| 4 | `04_extract_activations.py` | Re-run annotated chains with residual hooks → per-behaviour activation matrices at every layer | GPU |
| 4 (multi) | `04b_extract_annotator.py` | Same extraction for a specific annotator's labels (manifold-replication / multi-annotator test) | GPU |
| 5 | `05_pca_analysis.py` | PCA per behaviour × layer; effective-dimensionality metrics | Local |
| 5b | `05b_geometric_diagnostics.py` | Intrinsic dimension + curvature + null-hypothesis hierarchy | Local |
| 5c | `05c_cross_layer_probing.py` | Layer-wise linear probing + non-adjacent subspace principal angles | Local |
| 5d | `05d_subtype_clustering.py` | K-means sub-type discovery per behaviour in PCA space (sub-type steering vectors) | Local |
| 6 | `06_build_steering.py` | Build single-direction (Venhoff) and manifold-projected steering vectors | Local |
| 6b | `06b_steering_composition.py` | Pairwise composition diagnostics — cos(v_sum, v_proj), off-manifold ratio | Local |
| 7 | `07_evaluate_steering.py` | Apply steering vectors to a category-stratified eval split (shared vanilla baseline + single-direction + manifold-projected + norm-matched random-direction control); behavioural shift, saturation | GPU |
| 7b | `07b_activation_patching.py` | Donor-pair activation patching → per-layer causal effect curves | GPU |
| 8 (M1) | `08_annotate_cbs.py` | CBS-tier + cross-domain second-pass annotation of `adding-knowledge`/`deduction` sentences | Local |
| 9 (M2) | `09_cbs_geometry.py` | Per-sentence geometric tests by CBS tier and cross-domain flag (Holm-corrected) | Local |
| 10 (M3) | `10_trajectory_build.py` | Per-chain trajectory construction (arc length, curvature, subspace dynamics) | Local |
| 11 (M4) | `11_trajectory_analysis.py` | Group comparisons + matched-pair tier-3 analysis + verification-gradient probe | Local |
| 12 (M5) | `12_cbs_ablation.py` | CBS steering ablation (causal experiment) with fail-stop validation | GPU (generation) |
| 13 (M6) | `13_baseline_replication.py` | Replicate M1–M4 on the baseline model; cross-model comparison tables | Local (orchestration) |

Several stand-alone helpers also live at the root, e.g. `check_chain_quality.py`
(post-Phase 2 quality report), `verify_base_model.py` (weight-comparison check
of the base-model identity), `predict_saturation.py`, `power_analysis_curvature.py`,
`compute_layer_triangulation.py`, `compare_annotators.py`, `robustness_geometry.py`,
and the figure/HTML builders (`make_explainer_figs.py`, `make_fresh_figures.py`,
`render_html.py`). Orchestration shell scripts: `run_multiannotator_pipeline.sh`,
`run_remaining_phases.sh`, `run_rerun_local.sh`.

Most phases accept `--smoke` (or equivalent) for a fast subset run; see each
runner's `--help`.

## Tests

Full suite (core + CBS; 255 tests as of 2026-06-12):

```bash
pip install -e ".[cbs]"
python -m pytest
```

## Known limitations

- **No geometry number is currently citable.** The 2026-06-05 audit fixed
  biased estimators (TwoNN, curvature ratio) and 2026-06-12 fixed the
  duplicate-row/alignment bug (35–56% exact-duplicate activation rows from
  first-occurrence sentence matching); every pre-fix geometry output is
  quarantined in `results/_STALE_pre_fix_20260605/` pending regeneration on the
  fixed code (re-extraction first — the activation matrices themselves predate
  the occurrence-aware matcher). See `CONFOUNDS_AND_REMEDIATION.md` (the
  authoritative confound register) and `INVENTORY.md`.

- **~50% chain truncation.** Per `PROGRESS.md`, 50.2% of the R1-1.5B chains hit
  the 8192-token cap and 49.9% lack a closing `</think>` (truncated mid-thinking),
  concentrated in `lateral_thinking` (95% at cap), `spatial_reasoning`,
  `pattern_recognition`, and `probabilistic_thinking`. Downstream analyses must
  account for this (the chosen policy is to stratify by a `truncated` flag; see
  `src/cbs/cohort.py`). Re-generating chains forces re-annotation.

- **The Phase-7 eval split is stratified but not a true hold-out.** Activation
  extraction (and therefore steering-vector construction) runs over the whole
  corpus, including the eval tasks. Phase-7 numbers are on-corpus causal
  effects, not out-of-sample generalisation, until extraction excludes the
  eval split.

- **Single LLM annotator.** All behaviour labels come from one annotator
  (Claude Sonnet 4.5 via the lab proxy — not GPT-4o, despite Venhoff's
  original setup); the 3-annotator robustness arm (Qwen3-235B, Nova-Pro) is in
  flight. Labels are the dependent variable for everything downstream.

(Resolved former entries: `configs/config.yaml` is now loaded by `src/config.py`
as the single config source; Phase 7 reads the canonical `data/tasks_final.json`;
the `pyproject.toml` build backend is fixed — `pip install -e .` works.)

## Related work

- **Huang et al., NeurIPS 2025** (arXiv:2505.22411) — composite overthinking on a
  low-dim manifold (top-k PCA subspace), manifold-projected steering. *We ask the
  per-behaviour version, with the same model and steering layer.*
- **Venhoff et al., ICLR 2025 Workshop** (arXiv:2506.18167) — one linear direction
  per behaviour suffices for control. *We measure the structure a single
  direction may not exhaust; rung 1 is theirs, not contested.*
- **Curveball Steering** (arXiv:2603.09313, Mar 2026) — concept-dependent
  curvature; nonlinear steering beats linear on persona/safety traits. *Different
  concept class; reasoning behaviours remain open.*
- **Goodfire Manifold Steering** (arXiv:2605.05115, May 2026) — linear steering
  drifts off-manifold on cyclic concepts. *Same moral, different concepts.*
- **REMA** (arXiv:2509.22518) — occupies the term "reasoning manifold" (also
  TwoNN-based). *We therefore headline "per-behaviour activation geometry"
  rather than "reasoning manifold".*
- **arXiv:2510.07364** (Venhoff et al., *Base Models Know How to Reason,
  Thinking Models Learn When*) — base models already contain reasoning
  mechanisms; a hybrid model recovers up to 91% of the base→thinking gap while
  steering only ~12% of tokens. *Their base suite includes our base,
  Qwen2.5-Math-1.5B; the 91% headline is the Qwen2.5-32B/QwQ-32B pair on
  MATH500, not the 1.5B. Pre-answers CBS's binary question; motivates the
  geometric-refinement reframing and the CBS deferral.*
- **arXiv:2507.12638** (Ward et al.) — a *latent* backtracking direction
  pre-exists in base-model activations (repurposed by fine-tuning; it does not
  itself induce backtracking in the base) — shown on Llama-8B. *Same
  implication for CBS.*
- **CREST** (arXiv:2512.24574, Dec 2025) — calibrates attention heads
  correlated with distinct cognitive behaviours (verification, backtracking)
  and steers them at test-time by rotating hidden states. *Nearest neighbour
  to rungs 1–2, but head-localization + rotation, not per-behaviour subspace
  geometry: no per-behaviour intrinsic dimension, curvature, chain-stratified
  nulls, or subspace-vs-single-direction test, and it does not decompose
  Huang's overthinking direction.*

## References

- Huang et al., "Mitigating Overthinking in Large Reasoning Models via Manifold Steering," NeurIPS 2025 (arXiv:2505.22411).
- Venhoff et al., "Understanding Reasoning in Thinking Language Models via Steering Vectors," ICLR 2025 Workshop (arXiv:2506.18167).
