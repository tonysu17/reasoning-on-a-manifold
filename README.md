# Reasoning on a Manifold

**Do individual reasoning behaviours in thinking LLMs have manifold structure?**

This project synthesises two lines of work:

- **Huang et al. (NeurIPS 2025)** — *Mitigating Overthinking in Large Reasoning Models via Manifold Steering*: showed that the composite phenomenon of overthinking lives on a low-dimensional manifold in activation space, and that projecting steering vectors onto this manifold dramatically improves intervention effectiveness.

- **Venhoff et al. (ICLR 2025 Workshop)** — *Understanding Reasoning in Thinking Language Models via Steering Vectors*: identified several distinct reasoning behaviours in thinking LLMs (backtracking, uncertainty estimation, example testing, knowledge augmentation) and showed each can be controlled via a single linear steering vector.

**The gap:** Venhoff assumes each behaviour is captured by a single direction. Huang only studies the composite overthinking phenomenon. Nobody has checked whether *individual* reasoning behaviours have richer geometric structure (multi-dimensional manifolds rather than single directions).

**The hypothesis:** Behaviours like backtracking come in multiple flavours (e.g., "arithmetic re-checking" vs "strategy-level pivoting"). If so, they should occupy a multi-dimensional subspace, and manifold-projected steering per behaviour should enable finer-grained control than a single vector.

The primary model is `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (28 layers, hidden dim 1536, steering layer 27). A knowledge-creation extension (the "CBS" / Content-Beyond-Source pipeline, phases 8–13) additionally contrasts the distilled model against its empirically-verified base, `Qwen/Qwen2.5-Math-1.5B`, to test whether the geometry is *created* by distillation or merely *revealed* by it.

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
│       └── tests/             # pytest suite (98 tests)
├── configs/config.yaml        # documentary config — NOT loaded by code (see Known limitations)
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
| Pilot gate | `00_pilot_gate.py` | Mandatory 20-chain stratified pilot through Phases 2–3 + 5 validation checks before scale-up | GPU (chains) + local |
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
| 7 | `07_evaluate_steering.py` | Apply steering vectors to held-out tasks; behavioural shift, saturation, generalisation | GPU |
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

The CBS extension has a pytest suite (98 tests):

```bash
pip install -e ".[cbs]"
pytest src/cbs/tests/
```

## Known limitations

- **`configs/config.yaml` is documentary only — no code currently loads it.**
  All parameters (model IDs, layers, token caps, alpha grids, paths) are
  hardcoded in the individual runners and `src/` modules. The YAML is a useful
  reference for intended settings but editing it changes nothing at runtime.
  Note in particular that its `paths.tasks` points at the stale `data/tasks.json`
  (see next point and `data/MANIFEST.md`).

- **~50% chain truncation.** Per `PROGRESS.md`, 50.2% of the R1-1.5B chains hit
  the 8192-token cap and 49.9% lack a closing `</think>` (truncated mid-thinking),
  concentrated in `lateral_thinking` (95% at cap), `spatial_reasoning`,
  `pattern_recognition`, and `probabilistic_thinking`. Downstream analyses must
  account for this (the chosen policy is to stratify by a `truncated` flag; see
  `src/cbs/cohort.py`). Re-generating chains forces re-annotation.

- **Phase 7 reads the wrong task file.** `07_evaluate_steering.py` loads
  `data/tasks.json` (the older, stale corpus) rather than the canonical
  `data/tasks_final.json`. This is a known bug, documented in `data/MANIFEST.md`.

- **`pyproject.toml` build backend.** The `build-backend` line is currently
  being fixed separately; `pip install -e .` is the intended installation path.

## References

- Huang et al., "Mitigating Overthinking in Large Reasoning Models via Manifold Steering," NeurIPS 2025.
- Venhoff et al., "Understanding Reasoning in Thinking Language Models via Steering Vectors," ICLR 2025 Workshop.
