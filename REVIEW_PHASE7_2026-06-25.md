# Phase 7 Steering — Methodology & Results Review (2026-06-25)

> Prepared for the fund/no-fund decision on the full RunPod steering run. Synthesised from a
> 5-agent independent review of `METHODOLOGY.md`, `METHODOLOGY_REFINEMENT_2026-06-25.md`,
> `CONFOUNDS_AND_REMEDIATION.md`, `RESULTS_LEDGER.md`, `PLAN_EXPERIMENTS.md`, the code
> (`src/steering.py`, `src/steered_inference.py`, `06`/`07`), and on-disk results. Every number
> verified against the result files, not just the trackers.

---

## 0. Executive summary + the decision

**The geometry foundation is solid and citable; the steering layer (L27) is defensibly chosen; the
causal steering *headline* has NOT been run.** The pilot bought us the layer decision, not the result.

**Two things gate a credible full run — resolve before funding:**
1. **The de-circularised annotator is not located.** Sonnet built the behaviour labels *and* would score
   the steered outputs → circular. The runner supports `--annotator-model` but no non-builder proxy id
   (Qwen3-235B / Nova-Pro live endpoint) exists. **Without it the headline can only be "preliminary,
   band-ungated."** This is the single biggest blocker.
2. **On current pilot signal, the clean steering effect may reduce to 1–2 behaviours.** L27 cleanly
   suppresses **uncertainty-estimation** (−63% relative); backtracking is weak (needs the α-sweep to
   confirm), adding-knowledge is a *pre-registered null*, example-testing nudges the wrong way at L27.

**Recommended scope if we proceed:** the **Tier-C headline** (~2,400 chains, **≈$130–180** annotation at
the measured $0.055/chain + RunPod GPU time), **not** the full default grid (~16,850 chains ≈ **$900–1000**).
A fresh generation run is required either way — the existing bake-off chains lack the floor arms and the α\* dose.

---

## 1. Methodology

### 1.1 Model, corpus, behaviours
- **Model:** `DeepSeek-R1-Distill-Qwen-1.5B` — 28 layers, hidden 1536, fp16. Base verified = `Qwen2.5-Math-1.5B` (embed cos 0.994).
- **Corpus:** 1000 tasks, 100 × 10 categories (`tasks_final.json`; the May-21 `tasks.json` is stale, not used). Categories: mathematical_logic, spatial_reasoning, verbal_logic, pattern_recognition, lateral_thinking, causal_reasoning, probabilistic_thinking, systems_thinking, creative_problem_solving, scientific_reasoning.
- **Four target behaviours** (Venhoff taxonomy subset): **backtracking** (abandon/pivot), **uncertainty-estimation** (express doubt/confidence), **example-testing** (try a concrete case), **adding-knowledge** (inject external facts).

### 1.2 Pipeline (7 phases)
1 task-gen → 2 chain-gen (greedy, 8192 cap) → 3 annotation (Sonnet, Venhoff prompt) → 4 activation extraction (all 28 layers, mean-pooled) → 5 PCA/geometry → 6 steering-vector build → 7 steering eval (generate + re-annotate). Hold-out = 50 tasks, category-stratified (5/cat), shared by 06 & 07 via `stratified_eval_split`. **Held out on the task axis only — NOT the layer axis.**

### 1.3 Steering-vector construction
- **`single_direction`** = diff-of-means `mean(ON) − mean(OFF)`, unit-normalised, where **OFF = the other three behaviours** (not a neutral corpus). Venhoff (arXiv:2506.18167) recipe.
- **`manifold_k{1,3,5,10,auto}`** = the same `r` orthogonally projected onto the top-k PCA subspace of the ON activations, renormalised: `r_proj = Σ_{i≤k}(r·v_i)v_i`. The top-k-PCA operator is Huang's (arXiv:2505.22411); the per-behaviour question is ours.
- **`auto_k`** = smallest k explaining ≥70% ON variance (back 58 / unc 71 / ex 60 / add 83). Huang's literal k=10 ⇒ `manifold_k10` is the closest-to-published arm.
- **Pooling = mean over `[onset−1 : +10]` — SETTLED** (verified identical to Venhoff's published code). Minor caveat: 15.8% of sentences are shorter than the window (pools into the next sentence; `clip_window` exists, default off).

### 1.4 Why **layer 27** (the central justification)
Three converging positive signals:
1. **Huang (published)** uses L27 for this *exact* model.
2. **De-confounded `07d` forward-pass sweep** — argmax = **L27 for all four** behaviours (read-out at the *output*, random-direction null subtracted; shortlists [27]×3, backtracking [27,11,18]).
3. **Pre-registered pilot rule** (`METHODOLOGY_REFINEMENT §2.9`): "annotate L27 first; confirm iff it suppresses AND stays clean (repetition ≤ vanilla + margin); fall back to L16 only if L27 fails." The pilot **confirmed** this (§3.2).

**Honest caveats (must appear on every single-layer headline):**
- The layer-pick *proxy* reads out near L27 → **structural late-layer proximity bias** even after the null. The cheap pilot arbiter is vanilla-relative, not the de-confounded floor.
- **Venhoff steers mid (15–18)** — a genuine live alternative; the PR-trough is also mid (16/16/16/12).
- **CF-17: the layer is NOT held out on the layer axis.** Choice informed by full-corpus analyses + Huang. *Do not claim layer selection is held out.*
- `07c` attribution patching is **confounded (read-out proximity), do-not-use**; only the de-confounded `07d` informs the pick — and it is still a token-anchored proxy.

### 1.5 Arms / controls
| Arm | Controls for |
|---|---|
| `vanilla` (shared) | reference fraction |
| `single_direction` | the Venhoff probe → **energy-floored** headline arm |
| `manifold_k*` | the Huang subspace probe → **dimension-floored** headline (k-sweep) |
| `random_subspace_k*` (×3) | "the behaviour's *PCA* subspace matters" vs any k-dim projection |
| `random_direction` | **sanity floor only** — norm-matched, injects ~19× less energy; never the baseline |
| `energy_matched_random` | **the real floor** — random rescaled to equal injected energy |
| `orthogonal_complement` | is the discarded off-subspace component pure collateral? |

⚠️ **Asymmetry to disclose:** `energy_matched_random` calibrates the **single_direction** arm only. The manifold arms are **dimension-matched (random_subspace_k), not energy-matched.** Do not claim the manifold headline is energy-floored unless `energy_matched_random_k{k}` is built.

### 1.6 Alpha
Grid {0,0.3,0.5,0.7,1,1.5,2,3}; subtract (suppression) primary; α=1 ≈ full ablation of the r-component. **Sealed per-behaviour α\*** (single fixed dose, L27): back 0.994 / unc 0.969 / ex 0.964 / add 1.056 — clustered tight ⇒ the saturation-prediction test is low-power/secondary. Discipline: do **not** read matched effect off the eval curves (`effect_quantile=0.8` = tune-on-eval leak); use α\* as a fixed dose, match post-hoc on realised effect.

---

## 2. Results — geometry foundation (Movement 1 / Gate-0)

| Claim | Status | Key numbers |
|---|---|---|
| **Low-dimensional subspace** | ✅ **CITABLE** | correlation-dim ~6–8 (back 5.85 / unc 6.21 / ex 6.04 / add 7.71) ≪ linear d_eff; survives one-sentence-per-chain control; **replicates 3-way** |
| **Curvature** | ❌ **clean, well-powered NEGATIVE** | geodesic ratio 3.4–4.2 on pooled data collapses to ~2.3–2.5 (flat ≈1.0) at one-per-chain → **within-chain artefact, not a curved manifold.** Drop "curved." |
| **Behaviour-specificity** | 🟧 **MIXED 2/4** | backtracking + uncertainty p=0.0004 all layers; example-testing only L27; **adding-knowledge p=1.0 nowhere.** Single-annotator. |
| **3-way annotator replication** | ✅ subspace replicates | Sonnet/Qwen3/Nova, κ = 0.436 / 0.350 / 0.345; geometry ordering preserved. **Specificity null NOT re-tested 3-way.** |

**Do-not-cite:** TwoNN intrinsic-dim (0.168 = duplicate artefact), d_eff≥80% (saturates at 100), tangent-space-variation, any full-data curvature ratio, pre-dedup `tier1_robustness`. **Headline must read "low-dimensional, behaviour-specific *subspace*" — never "curved."**

---

## 3. Results — the pilot (layer bake-off + annotation)

Pilot config: 10 hold-out tasks, greedy, α=1, arms = vanilla + single + manifold_auto, **Sonnet (builder) annotator**, cap 4096. Cost $9.83. n=10, 0 missing.

### 3.1 Damage (repetition vs vanilla 0.28; lower = gentler)
**L27 decisively gentler — 7/8 cells ≤ vanilla; L16 inflates repetition in 7/8 cells (up to +0.31) and runs chains 500–1120 tokens longer** (the "steering is breaking the model" signature). Degenerate rate 0 everywhere.

### 3.2 On-target effectiveness (Δ vs vanilla, α=1)
| Behaviour | L27 single | L27 manifold | L16 single | note |
|---|---|---|---|---|
| **uncertainty-est.** | **−0.114** | **−0.108** | −0.011 | L27 clean dominance (~63% rel. cut) |
| backtracking | −0.006 | −0.021 | −0.036 | L27 clean-but-weak; L16 harder but dirty |
| adding-knowledge | −0.005 | −0.013 | −0.007 | ~null both (pre-registered null behaviour) |
| example-testing | +0.011 | +0.008 | +0.090 | wrong direction; L16 worse + a damage artefact |

**Layer verdict = L27**, by the pre-registered clean-guard: L16 fails the cleanliness bar for all four behaviours; L27 passes + cleanly confirmed on uncertainty. L27's clean-but-weak backtracking is an α-dosing question the headline sweep tests — *not* a reason to retreat to the destructive layer.

---

## 4. Confound register (the defensibility backbone)

18 confounds tracked. Resolved keystones: **CF-1/CF-2 chain confound** (→ curvature negative), **CF-13 duplicate rows** (52%→1%), **CF-14 provenance/vacuous-null**, **CF-15 probe leakage** (0.83–0.93 → 0.70–0.84, flat across depth), **CF-9 bootstrap CIs**.

**Still open / standing — and they touch the Phase-7 headline:**
- **CF-7 annotator circularity** — mitigated for geometry (3-way), **UNREMEDIATED for steering** until a non-builder annotator scores the run (Qwen3 id not located).
- **CF-17 layer-not-held-out** — standing caveat, disclose on every headline.
- **CF-10 read-out proximity** — 07c do-not-use; 07d a proxy with a residual L27 spike.
- **Control asymmetry** — manifold arms not energy-matched (only dimension-matched).
- **CF-5** — the steering operator is linear (top-k PCA); honest given curvature is a negative.
- **CF-8 truncation** (50.2% hit 8192 cap) and **CF-18 annotation integrity** flow through from the corpus.

**Documented negatives (honest spine):** curvature; adding-knowledge specificity (p=1.0); no discrete sub-types (silhouette 0.18–0.20); d_eff high (rescued only by intrinsic-dim gap); probe flat across depth; predictive-geometry gate AUROC 0.54–0.59 (the earlier 0.61 was inflated).

---

## 5. The full-grid decision

### What it costs
| Scope | Generations / chains | Annotation $ (@~$0.055/chain) | Notes |
|---|---|---|---|
| **Tier-C headline (recommended)** | ~2,400 | **≈$130–180** | 2 behaviours × floor arms × n=3 samples × 50 tasks @ sealed α\*; + gen-only for the other two |
| Full default grid | ~16,850 | ≈$900–1000 | all arms × 8 α × 50 tasks; over-scoped, methodology rejects it |

Generation is RunPod GPU-time (~2–3 min/gen vs the cluster's ~8.5). **The existing bake-off chains cannot be reused** for the headline — they lack `energy_matched_random` / `random_subspace_k` and the α\* dose. A fresh generation run is required.

### What it buys
The de-confounded causal headline: **for a named behaviour, its direction suppresses that behaviour beyond an energy/dimension-matched random floor at equal injected energy** — and the **manifold-vs-single suppression-vs-damage Pareto** (matched-effect, paired BCa bootstrap, Holm). This is the thesis's one causal result and the differentiator vs LRS (2606.00726).

### Risks to weigh
1. **No non-builder annotator → preliminary-only headline** (the critical blocker).
2. **Clean effect may be 1–2 behaviours** (uncertainty solid; backtracking TBD via α-sweep; example-testing/adding-knowledge likely null). A 1–2-behaviour causal result + the manifold-vs-single Pareto is still publishable, but go in knowing it.
3. McNemar/length-conditioned analyses are specified-not-built; `git_commit:null` provenance gaps.

### Consistency findings (doc hygiene, none change conclusions)
- `METHODOLOGY.md §4` arm list + "Δ vs vanilla" headline are **stale** — superseded by `METHODOLOGY_REFINEMENT` (full arms + Δ_floor).
- `RESULTS_LEDGER:73` still cites "Venhoff mid-peaks 11/16/19/16" as the de-confounded finding vs the on-disk **argmax L27** — reconcile before thesis citation.
- `results/eval/` exists (ledger says absent) but is smoke/scaffolding — the *scientific* "headline unrun" claim holds.
- Dead config key `evaluation.annotation_model`; stale `-peak` vectors (built at old 14/14/17/27); task-gen model labelled both gpt-4o and Claude-proxy.

---

## 6. Recommendation
1. **Before funding:** locate the Qwen3-235B / Nova-Pro live proxy id (or decide a preliminary Sonnet-scored headline + a static-corpus noise-band is acceptable).
2. **If proceeding:** run **Tier-C** (~$130–180 + RunPod GPU), pre-register the single primary metric (Δ_floor, Holm-corrected, must clear the annotator noise-band), and disclose the layer-not-held-out + manifold-not-energy-matched caveats verbatim.
3. **Set expectations:** the clean causal win may rest on uncertainty-estimation (± backtracking). That, plus the manifold-vs-single Pareto and the citable geometry foundation, is a coherent thesis chapter.
