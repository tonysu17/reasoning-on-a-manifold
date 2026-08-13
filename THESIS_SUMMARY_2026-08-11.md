# Thesis Summary — *Intervening on Machine Reasoning*

**Detailed chapter-by-chapter summary of the MSc dissertation.**
Compiled 11 August 2026 from the working-tree sources in `thesis/chapters/v2/` (which contain
uncommitted edits ahead of thesis-repo `HEAD` = `b50d321`).

---

## 1. Identity and physical description

| Field | Value |
|---|---|
| **Title** | Intervening on Machine Reasoning |
| **Subtitle** | Residual-Stream Steering, Post-Training, and Behaviour-Indexed Representation Change |
| **Degree** | MSc Computational Finance, Dept. of Computer Science, UCL |
| **Author** | Tony Su |
| **Academic supervisor** | Dr Paolo Barucca (UCL) |
| **Industrial supervisor** | Zekun Wu (Research Scientist, Holistic AI) |
| **Submission date on title page** | 8 August 2026 |
| **Length** | 63 pages |
| **Structure** | 7 chapters + 1 appendix + bibliography |
| **Figures / tables** | 5 figures, 13 tables |
| **References** | 67 cited (of 146 entries in `references.bib`) |
| **Canonical build** | `tectonic ucl_msc.tex` (root: `thesis/ucl_msc.tex`; shared chapters in `chapters/v2/`) |
| **Model studied** | DeepSeek-R1-Distill-Qwen-1.5B ("R1-1.5B"); 28 layers, width 1536, 12 heads/layer |

### Page map

| Ch. | Title | Pages |
|---|---|---|
| — | Declaration, Abstract, Acknowledgements | i–iii |
| 1 | Introduction | 1–4 |
| 2 | Background and Literature Review | 5–13 |
| 3 | Data and Methods | 14–22 |
| 4 | Representational Foundations for Intervention | 23–27 |
| 5 | Inference-Time Intervention: Residual-Stream Steering | 28–33 |
| 6 | Training-Time Intervention: Safety Reasoning and Post-Training | 34–39 |
| 7 | Discussion and Conclusion | 40–44 |
| A | Exploratory and Provisional Analyses | 45–50 |
| — | Bibliography | 51– |

---

## 2. The spine in one paragraph

The thesis takes **intervention — not manifold discovery — as its narrative spine**. Reasoning models
emit long, annotatable chains; two interventions can alter them at different scales. *At inference
time*, activation steering writes to the residual stream with weights frozen. *At training time*,
post-training changes the weights and can persist. The thesis measures what each intervention
changes, how specific the change is, and which claims survive the relevant comparison. Geometry is
demoted to an **instrument**: it constructs, measures, and constrains the intervention claims, but is
explicitly *not* offered as a causal mechanism or as a universal "reasoning manifold". The dominant
stylistic feature of the document is its **discipline about evidential status**: passes, failures,
amendments, unrun analyses, and unresolved execution provenance are all reported in-line rather than
suppressed.

### Research questions

| RQ | Question | Answered in | Headline outcome |
|---|---|---|---|
| **RQ1** | What measurable structure do annotation-indexed reasoning activations exhibit, and which properties are behaviour-specific under chain-aware controls? | Ch. 4 | Low occupancy recurs; **direct specificity fails**; composite robustness fails; curvature mixed |
| **RQ2** | How do estimated residual-stream directions change target behaviour vs paired unsteered generation, and how specific/robust is that change? | Ch. 5 | Clear descriptive change for **backtracking**; provisional after matched-perturbation check |
| **RQ3** | Which aspects of generic-reasoning representations change under safety vs matched non-safety post-training? | Ch. 6 | **Translation clearer than reshaping**; magnitude not safety-specific; direction exploratory |

---

## 3. Chapter 1 — Introduction (pp. 1–4)

**Function.** Sets the intervention spine, states the claim boundary before any result, and poses the
three RQs.

**§1.1 Intervening on reasoning behaviour.** Chain-of-thought prompting improves multi-step
performance (Wei et al.) and reasoning-model training makes extended traces a default part of the
trajectory (DeepSeek R1). Those traces contain recognisable moves — framing, recalling, testing a
case, expressing doubt, abandoning an approach — that can be annotated and aligned to activations.
Two interventions act on them: residual-stream steering (weights fixed) and post-training (weights
changed).

Three disclaimers are installed immediately, and they govern the wording of every later result:

1. **Observability is not transparency.** A written chain may incompletely report what the answer
   depends on (Lanham; Chen et al.).
2. **Decodability is not use.** A decodable direction does not show the model uses it (Hewitt &
   Liang; Elazar et al.).
3. **A label is an index, not a latent process.** Each annotated sentence provides "an operational
   index into a defined activation cloud" — nothing more.

The relationship to Venhoff et al. (2025) is stated frankly: the thesis **adopts and extends their
operational framework and is not designed as a direct replication**. It keeps their behaviour
vocabulary, annotation prompt, pooling, direction construction, and layer-selection logic, but
applies them to an independently generated corpus, adds geometric and post-training questions, and is
narrower in scope (one 1.5B checkpoint vs their multiple architectures/sizes).

**§1.2 Representation-level summaries and their limits.** Huang et al. (2025) motivate the second
comparator: single-direction steering on *overthinking* helps only to a threshold, then degrades;
projecting the vector into a low-dimensional PCA subspace sustains the effect ("manifold steering").
The chapter immediately separates *direction* (1-D) from *subspace* (k-D) and notes that neither
shows linear coordinates exhaust the geometry, and that a high variance ratio does not establish a
nonlinear manifold.

**§1.3 RQs and evidential gates.** The key methodological move: the word "manifold" collapses
estimands that must be kept apart — correlation dimension (intrinsic-dimensional occupancy), PCA
variance ratio (linear concentration), curvature (departure from local linearity), and steering
(does an operator move an observable endpoint?). **The chain, not the sentence, is the independent
scientific unit.** The section then pre-announces that "the evidence is uneven across these
questions" and names the failures before the reader reaches them.

**§1.4 Contributions.** Three claimed: (i) a bounded empirical study of intervention at two scales in
R1-1.5B; (ii) a chain-aware, representation-level protocol keeping five estimands separate, with
amendments and failed criteria reported alongside positives; (iii) an audited annotation and
activation corpus with a provenance and confound record, bounded to what could be recovered. The
three model-generated annotation sets are explicitly **not** independent human validation.

**§1.5 Organisation.** Roadmap: Ch. 2 literature → Ch. 3 corpus/pipeline/confounds → Ch. 4 RQ1 →
Ch. 5 RQ2 → Ch. 6 RQ3 → Ch. 7 comparison at supported inference levels.

---

## 4. Chapter 2 — Background and Literature Review (pp. 5–13)

**Function.** Locates both intervention scales in five literatures and derives the claim boundary
used throughout. Structured *narrative* review, not a meta-analysis (the literature supplies no
poolable effects); updated through 8 August 2026 across arXiv, ACL Anthology, OpenReview, PMLR,
NeurIPS, publisher pages.

**§2.2 Reasoning behaviour as an intervention target.** Faithfulness work establishes that trace
dependence varies by task and model; only architectures that hand a translated program to a
deterministic solver are faithful by construction. Consequence: the scientific object is an
**annotation-indexed activation cloud**, whose validity depends on annotation rule, extraction
position, pooling rule, model, and chain population. A second consequence: for a *reasoning*
behaviour the intervention repeats during generation and the endpoint is scored from a stochastic
trace, so a frequency change may reflect control, generic disruption, or chain length — hence counts
and length-adjusted rates are needed alongside fractions. Venhoff's six-label taxonomy and
behaviour-specific layers (≈L17 backtracking, L18 adding-knowledge/uncertainty, L15 example-testing)
are introduced. Their behaviour fractions used a 1,000-token cap, so their frequencies describe only
chain openings and are not comparable to this thesis's 8,192-token regime.

**§2.3 Residual-stream essentials.** Notation: `h^(ℓ)_t ∈ ℝ^1536`, pre-norm transformer, 28 layers,
`h^ℓ = h^(ℓ−1) + Attn + MLP`. Additivity is what makes both point-cloud measurement and the steering
write `h + v` well defined.

**§2.4 Inference-time intervention.** Covers the linear representation hypothesis and its limits
(day-of-week ring as a nonlinear counterexample), the site hierarchy (embeddings → residual stream →
heads → MLPs), polysemantic neurons as the reason to intervene on *directions*, and the
ablation/patching/steering family. Notes that the layer selector used later is Venhoff's first-order
*attribution* approximation — it does **not** transplant a stored activation, so full activation-patch
confirmation remains outstanding. Also surveys recent threats: kernel-PCA nonlinear steering
(Raval), source-context and readout-location sensitivity (Ye), inverted-control results (Torop),
CoT-faithfulness steering transfer (Nguyen). Featurizer/DAS framing (Geiger et al.) is introduced,
with the warning that optimisation can carve effective-looking subspaces from noise (Makelov) —
which is why the thesis's own fitted-frame exercise is quarantined to the appendix.

**§2.5 Training-time intervention.** R1-1.5B is an SFT distill of Qwen2.5-Math-1.5B on 800k examples
(~600k reasoning + 200k non-reasoning); **no RL stage and no dedicated safety stage are reported for
the distill**. Venhoff et al. (base-vs-thinking) suggest SFT distillation installs new mechanisms
rather than merely retiming them. The central methodological demand of this chapter: separate
**translation** (movement of a representation's centre) from **reshaping** (covariance, orientation,
dimension, curvature); a mean shift is not evidence that internal geometry was rebuilt. Parameter-space
findings (safety basin) must not be read as activation-space findings. Safety-geometry work qualifies
the rank-one refusal-direction account (cones; dominant + orthogonal directions). Model diffing
(crosscoders, LoRA-vs-full-FT, template-induced shifts) motivates the **matched non-safety
comparator** in RQ3.

**§2.6 Geometry as a diagnostic framework.** Three quantities are separated: PCA effective rank
(global, coordinate-dependent), intrinsic dimension (local/scaling; can be low for a curved set
needing many PCs), curvature (departure from a flat reference). Huang et al.'s subspace concentration
and projected operator are adopted; the *manifold interpretation* is rejected.
**`fig_flat_vs_curved.pdf`** makes the argument visually: both a flat and a curved cloud can be
low-rank, and both admit well-defined `v_b` and `P_k v_b` — so the projected "manifold" operator can be
built without the set being a manifold. Two sampling caveats are registered: ~5k–17k labelled
sentences come from only ~1,000 chains (effective units in the *hundreds*), and finite-sample
covariance spectra are dispersed even for isotropic Gaussians (Marchenko–Pastur as the noise
reference). Grassberger–Procaccia correlation dimension is the primary ID estimator; TwoNN/MLE are
cross-checks only.

**§2.7 The gap.** Table `tab:bg-positioning` maps six literature strands to object → what it
establishes → remaining gap, with the thesis as the final row. Three gaps follow, one per RQ.

**§2.8 Construct validity and claim boundary.** The strictest statement in the thesis: the label is
an annotation construct, the cloud is an extraction construct, every statistic is an estimator on
that construct; estimator agreement cannot turn a label into a natural kind; **a failed null does not
show absence of structure** — only that the specified statistic did not distinguish it under that
comparison and resolution.

---

## 5. Chapter 3 — Data and Methods (pp. 14–22)

**Function.** Builds the shared corpus, defines both intervention designs and the geometric
diagnostics, and records the controls, confound register, and provenance boundary.
Table `tab:methods-rq-map` fixes object → decisive analysis → results chapter for each RQ.

### 5.1 Corpus (§3.2.1)

- **1,000 reasoning tasks**, 100 in each of ten categories (mathematical logic, spatial, verbal
  logic, pattern recognition, lateral thinking, causal, probabilistic, systems thinking, creative
  problem solving, scientific). Explicitly "coverage, not a principled taxonomy."
- Generated via the lab proxy with **Claude Sonnet 4.5, temperature 0.8, batches of five**;
  immutable request/response logs were **not located**, so settings are *reconstructed, not verified*.
- Chains: one greedy-decoded answer per task from R1-1.5B under an **8,192-token cap** (vs Venhoff's
  1,000) — chosen to admit the extended-deliberation regime the overthinking literature studies.
- **Truncation is the chapter's most consequential corpus fact** (registered as CF-8):

| | Mean tokens | At cap |
|---|---|---|
| Lateral thinking | 7,856 | 95.0% |
| Spatial reasoning | 6,906 | 71.0% |
| Probabilistic | 6,320 | 59.0% |
| Pattern recognition | 6,252 | 66.0% |
| Mathematical logic | 5,685 | 55.0% |
| Verbal logic | 5,630 | 62.0% |
| Creative problem solving | 4,397 | 44.0% |
| Scientific reasoning | 3,008 | 24.0% |
| Causal reasoning | 2,837 | 22.0% |
| Systems thinking | 1,630 | 4.0% |
| **Overall** | **5,052** | **50.2%** |

  499 chains (49.9%) lack a closing `</think>`. Because category predicts both truncation and
  behaviour mix, observed behaviour fractions depend on where the budget was exhausted.

### 5.2 Behavioural annotation (§3.2.2)

- Venhoff's verbatim prompt and six-way taxonomy: `initializing`, `deduction`, `adding-knowledge`,
  `example-testing`, `uncertainty-estimation`, `backtracking`. **Four are analysed**; deduction
  (~half of all sentences, similarly frequent in non-thinking models) and head-of-chain initialising
  are kept for segmentation only.
- Annotator: **Claude Sonnet 4.5 at temperature 0** via Bedrock proxy (Venhoff's GPT-4o checkpoint
  being unavailable). Long chains went through a chunked path whose exact window manifest is **not
  recoverable** — so cap, chunk thresholds, overlap, and split-chain count are not treated as facts.
- **Row counts (these are not interchangeable — each belongs to a different preprocessing contract):**

| Contract | Total rows | backtracking | uncertainty | example-testing | adding-knowledge |
|---|---|---|---|---|---|
| Raw extraction (993 chains) | 37,851 | 10,267 | 16,728 | 5,829 | 5,027 |
| Earlier within-label rule | 37,436 | 10,136 | 16,545 | 5,766 | 4,989 |
| Amended audit (symmetric cross-label collision removal) | 37,324 | — | — | — | — |
| Primary L27 samples | — | 10,094 | 16,503 | 5,750 | 4,977 |
| Eligible chains (L27) | — | 705 | 909 | 635 | 880 |

- **Three annotation sets** (Sonnet 4.5, Qwen3-235B, Nova-Pro) address single-model uncalibration
  (CF-7). Character-level Cohen's κ: **0.436** (Sonnet–Qwen3), **0.350** (Sonnet–Nova), **0.345**
  (Qwen3–Nova); lower after target/other collapse (0.352 / 0.305 / 0.263). Scoring is deliberately
  harsh — every boundary disagreement counts in full — and disagreement concentrates on the softer
  behaviours. These are **not independent annotators**; they test robustness to annotation
  *pipelines* only.

### 5.3 Representation extraction (§3.2.3)

Read hooks replay each chain once, copy the residual stream unmodified, cache **all 28 layers**.
Span window follows Venhoff: **one context token before sentence onset + up to ten execution tokens
after (≤11 positions), mean-pooled**. The window is *not clipped to the sentence end*, so the
operational object is an onset-centred token window rather than the sentence text.

Layer rules (all zero-based) are kept deliberately separate:

| Analysis | Layers |
|---|---|
| Registered primary correlation dimension | common **L27** (aligned with Huang et al.; avoids outcome-informed selection) |
| Older one-sentence-per-chain sensitivity | L14 / L14 / L27 / L17 (bt / unc / ex / ak) |
| Registered five-depth cdim + variance-concentration family | L11, L14, L17, L20, L27 |
| Cross-annotation exploratory PCA | L12, L16 |
| Registered curvature diagnostic | L16 |
| Executed steering (gradient attribution) | L17 / L15 / L15 / L17 |
| Post-training adapter extraction | L12, L16 |

A `row_index.json` provenance sidecar records one entry per accepted row in order; the loader
**hard-fails on misalignment** and drops exact duplicates before estimation. Mean-pooling discards
within-span trajectory (CF-6).

### 5.4 Intervention designs (§3.3)

**Steering.** Directions built from pooled spans *after excluding the fifty evaluation tasks*.
Operators: target-vs-other-five-labels direction, plus rank-3 and rank-5 PCA projections. Primary
comparison pairs each intervention against the shared unsteered generation per task, reading
behaviour fraction, count, and count per token. Matched random directions/subspaces are a *separate
robustness comparison*. Amendments stated up front: one full-ablation dose, gradient-attribution
layer selection, two ranks; dose frontier, activation-patch confirmation, and **task-accuracy guard
remain unrun**. Construction and outcome labels share an annotator family. Run record has
`git_commit: null` and no input hashes.

**Post-training.** Two comparisons.
*Observational:* byte-identical sequences through R1-1.5B and STAR1-R1-Distill-1.5B, every accepted
row paired.
*Controlled:* safety adapters on 100/300/1,000 STAR-1 examples vs a non-safety adapter on 1,000
shuffled generic-reasoning chains, approximately quantile-matched on completion word length,
truncated at sentence boundaries with a fixed neutral closing — "a matched fine-tuning control, not a
token-identical or semantic control."
Recipe: completion-only bf16 SFT, 5 epochs, cosine LR 1e-5, warm-up 0.03, max length 4,096, batch 4 ×
32 accumulation (effective 128, single device), **LoRA r=16, α=32, dropout 0.05** on attention + MLP
projections; adapters merged before extraction at L12/L16. Seed 42 for every dose and the initial
pair; seeds 43/44 for the 1,000-example pair. Gated behaviour–layer summaries draw 1,500 paired spans
per cell; the calibrated rotation statistic alone uses a fixed rank-5 PCA subspace.
Explicitly flagged: teacher-forced comparisons **do not** estimate post-training effects on generated
behaviour; the full-checkpoint record reports a dirty tree; several JSONs omit lineage; checkpoint
weight hashes are absent.

### 5.5 Geometric diagnostics (§3.4)

- **Linear spectral:** exact PCA; `d_eff(p)` (smallest number of components reaching cumulative
  variance p, with `d_eff(0.70)` retained for comparability to Huang); participation ratio
  `PR = (Σλ)² / Σλ²`; **fixed top-ten variance share, with k=10 fixed before the permutation analysis**.
  None of these is an intrinsic-dimension estimate.
- **Correlation dimension:** Euclidean `pdist`; subsample 2,000 rows (seed 42) if larger; 20 log-spaced
  radii between the 2nd and 98th percentile of non-zero finite distances; `C(r)=#{d_ij<r}/#{d_ij}`;
  OLS slope of `log C` on `log r` over the middle 20–80% of valid radii (fallback: all valid radii if
  fewer than three). No internal bootstrap — chain-level stability is built separately.
- **Registered primary (H1/H2):** exact-deduplicate the L27 pool, then select **at most one
  target-labelled occurrence per eligible chain** by deterministic digest. *H1* (descriptive) requires
  all four cdim estimates in (0, 10]. *H2* (specificity) permutes the complete label multiset
  **within each chain**, reselects one occurrence per eligible chain, compares against B=2,500 valid
  lower-tail draws (≤2,750 attempts, identity assignments retained), smoothed Monte-Carlo p, one Holm
  correction across four behaviours. A full diagnostic-seed repetition checks verdict stability.
- **H3 (composite robustness):** 500 deterministic draws retaining 80% of eligible chains (bands are
  *chain-stability bands, not confidence intervals*) **and** a truncation/category leg — if the largest
  complete/truncated category share differs by >10 pp, fixed category-matched chain lists are
  mandatory. H3 is an **exhaustive cross-product**, not an average over partial successes.
- **H4 (pooling/window):** requires matched cdim tests over mean/first/last pooling × clipped/unclipped
  windows. **Inputs were never extracted → unrun.**
- **Curvature (registered, non-confirmatory, L16):** 500 replicates; each parent seed spawns a chain
  draw and an equal-size sentence-weighted control draw; local-to-global dimension ratio with k=10,
  variance threshold 0.90, ≤150 anchors; pair retained only if both ratios finite and positive.
  Bounded-negative criterion = chain ratio band contains 1 **and** `D_chain ≤ D_control`. It produces
  **no p-value and no global-flatness conclusion**.
- **Exploratory only:** smoothed PR-trough layer rule (3-point moving average, argmin), per-layer
  ℓ2-penalised logistic probes (class-balanced, single 70/30 `GroupShuffleSplit` grouped by chain —
  *one split, not cross-validation*; run record commit `06f07fec` but dirty tree), principal angles
  between top-ten PCA spans at gaps 3/7/14, and a k-means silhouette sweep k=2..8 at PR-trough layers
  (L16/L16/L12/L16) with `d_eff(0.70)` = 38/44/49/53.

### 5.6 Controls, validation, provenance (§3.5–3.6)

Chain-stratified permutations shuffle labels **within** each chain, preserving chain identity,
within-chain drift, ambient covariance, and the within-chain label multiset; they hard-fail if the
mixed-label-chain gate is unmet. Global label shuffling is a secondary reference only. The chain
confound (CF-2) is the register's central concern: thousands of rows nested in 635–909 chains; an
earlier ICC/design-effect calculation is absent from the frozen snapshot and therefore **not used**.
Permutation p-values are smoothed `(1+#extreme)/(1+B)`, floored at `1/(B+1)` — raw floor
`1/2501 = 0.00039984` — with Holm–Bonferroni over declared families. The cross-annotation check is
explicitly *separate and exploratory*: B=500, seed 42, unadjusted α=.05, **not** a Holm-adjusted
replication.

§3.6 states the reproducibility boundary plainly: analysis and thesis are separate repositories; the
thesis carries a frozen evidence snapshot with source repo, commit, and file hashes; **null commit
fields, dirty-run records, missing input hashes, and absent checkpoint revisions are disclosed rather
than inferred from file times**; activation tensors and model-derived corpora are not publicly
deposited at this freeze, so no DOI or access URL is claimed. Ethics: model-generated text only, no
human participants, harmful prompts used only for controlled training/evaluation and not reproduced.

---

## 6. Chapter 4 — Representational Foundations for Intervention (pp. 23–27) — **RQ1**

**Function.** Establishes what the representation-level substrate does and does not support. This is
the chapter where the thesis's own preregistered criteria mostly **fail**, and it says so.

### 6.1 H1 passes, H2 fails (§4.1)

Common-L27 equal-chain correlation dimension:

| Behaviour | Chains | d_corr | p_raw | p_Holm | H1 | H2 |
|---|---|---|---|---|---|---|
| backtracking | 705 | **7.208** | .842 | 1.000 | pass | **fail** |
| uncertainty estimation | 909 | **7.364** | 1.000 | 1.000 | pass | **fail** |
| example testing | 635 | **7.040** | .895 | 1.000 | pass | **fail** |
| adding knowledge | 880 | **8.252** | 1.000 | 1.000 | pass | **fail** |

All four estimates are finite, positive, and ≤ 10, so **H1 passes**: low estimated intrinsic-dimensional
occupancy under one representation. But against 2,500 within-chain label permutations, *every* raw
lower-tail p is ≥ .842 and every Holm-adjusted p is 1.000, and a full diagnostic-seed repetition
reproduces all four failures. **H2 fails**: correlation dimension does not distinguish observed labels
from the within-chain permuted reference. The chapter states that the PCA analysis below "is not
evidence against this decision."

`fig_geometry_direct_null.pdf` panel **A** shows the estimates against the null's mean and central 95%;
panel **B** shows observed-minus-null top-ten PCA variance share in percentage points, with the eleven
Holm-surviving cells outlined.

### 6.2 H3 fails on all four behaviours (§4.2)

- *Chain leg:* bands contain the full-sample cdim and PR values for backtracking, uncertainty, and
  example testing. **Adding knowledge fails** — its cdim 8.252 sits just above its band endpoint 8.220
  (PR remains inside). Chain-only result: 3 pass, 1 fail.
- *Truncation leg:* all four complete-vs-truncated category-share imbalances exceed the 10 pp trigger,
  so category matching is mandatory. Matched counts 197/197, 254/254, 199/199, 249/249. Correlation-dimension
  differences stay within 25% everywhere — **but every behaviour fails because complete and truncated
  PR stability bands do not overlap** (backtracking 30.3% complete-vs-combined PR departure; matched
  uncertainty 26.1%; example testing 29.2%).
- Under the registered cross-product rule, **all four H3 decisions fail**.

| Behaviour | cdim [band] | PR [band] | Chain leg | Truncation leg | H3 |
|---|---|---|---|---|---|
| backtracking | 7.208 [6.931, 7.657] | 39.13 [36.79, 39.79] | pass | fail | **fail** |
| uncertainty | 7.364 [6.945, 7.586] | 37.30 [34.91, 38.12] | pass | fail | **fail** |
| example testing | 7.040 [6.397, 7.054] | 28.66 [26.97, 30.14] | pass | fail | **fail** |
| adding knowledge | 8.252 [7.534, 8.220] | 50.24 [46.52, 50.56] | **fail** | fail | **fail** |

**H4 is unrun, not failed** — the six aligned representation matrices do not exist, and no GPU
re-extraction or reduced-grid substitute was used. Therefore *no* claim of cdim robustness across
pooling/window is permitted.

### 6.3 Secondary specificity families also fail (§4.3)

No cell survives Holm in either registered secondary cdim family: the smallest raw value in the
20-cell Sonnet five-depth family is **p = 0.00920 → p_Holm = .18393**; every adjusted value in the
24-cell two-depth × three-annotation family is 1. Because the deterministic occurrence selector
includes family and layer coordinates, these repeat the frozen procedure rather than reuse
byte-identical L27 samples.

### 6.4 Amended variance-concentration specificity (§4.4)

A *different* quantity — fixed-top-ten PCA variance concentration — does show structure. Sonnet
annotation, L11/14/17/20/27, B=2,500, Holm over the 4×5 family:

| Behaviour | L11 | L14 | L17 | L20 | L27 |
|---|---|---|---|---|---|
| Backtracking | .4984/.4608 | .4993/.4662 | .5015/.4718 | .4587/.4315 | .4155/.3894 |
| Uncertainty estimation | .4820/.4628 | .4835/.4692 | .4851/.4752 | .4455/.4333 | .3912/.3812 |
| Example testing | .4744/.4747 | .4724/.4730 | .4798/.4779 | .4441/.4421 | .4260/.4155 |
| Adding knowledge | .4325/.4526 | .4451/.4578 | .4606/.4658 | .4162/.4270 | .3622/.3874 |

*(observed / null-mean)*

**Eleven cells tie at the raw smoothing floor** (0.00039984 → Holm ≈ 0.0080): all five backtracking
cells, all five uncertainty cells, and example-testing L27. Example testing L17 and L20 (raw
p = 0.0191923) do **not** survive correction (Holm ≈ 0.1727). **Adding knowledge is unsupported at all
five depths** (raw and Holm p = 1) — and the chapter offers no semantic explanation for that.

### 6.5 Curvature is mixed (§4.5)

| Behaviour | Median | 95% band | D_chain / D_control | Decision |
|---|---|---|---|---|
| Backtracking | 1.0339 | [1.0141, 1.0527] | .0339 / .0111 | **Nonconforming** |
| Uncertainty estimation | 1.0153 | [.9982, 1.0339] | .0153 / .0196 | Negative |
| Example testing | 1.0000 | [.9835, 1.0183] | .0061 / .0093 | Negative |
| Adding knowledge | .9649 | [.9465, .9836] | .0351 / .0166 | **Nonconforming** |

Uncertainty and example testing pass the bounded-negative criterion (this operator resolves no
curvature there). Backtracking's band sits **entirely above** 1 and adding knowledge's **entirely
below** 1, with chain departures exceeding row-control departures in both. Family decision:
**mixed** — neither a four-behaviour negative nor evidence of global flatness. All cells used 500/500
valid paired draws.

### 6.6 Bounded answer to RQ1 (§4.6)

Low common-L27 mean-pooled correlation-dimension estimates (7.040–8.252) are supported; **behaviour-specific
correlation-dimensional occupancy is not**; no linear-subspace claim follows. PCA concentration is
behaviour- and depth-dependent (all depths for bt/unc, L27 only for ex, nowhere for ak); at L12/L16 the
first two recur across all three annotation sets under a *separate, exploratory, unadjusted* analysis
that "cannot rescue the failed direct correlation-dimension families." Curvature is mixed. The
frozen core-hardening provenance record reports **`code_commit: null` and `dirty: true`**, so these
numbers retain unresolved execution lineage. The chapter closes by noting that Ch. 5 and 6 must rely
on their *own* controls, because nothing here establishes an intervention effect or a post-training
origin.

---

## 7. Chapter 5 — Inference-Time Intervention: Residual-Stream Steering (pp. 28–33) — **RQ2**

**Function.** Tests whether an explicitly constructed residual-stream operator changes annotated
behaviour on held-out tasks. Opens by stating that **neither tested operator is validated by the
geometric diagnostics** of Ch. 4.

### 7.1 Protocol (§5.1)

**Direction.** For behaviour b at a layer: *on* set = b's pooled span activations; *off* set =
concatenated pooled activations of the other five labels (the three remaining targets + initialising +
deduction). `d_b = μ_on − μ_off`, unit-normalised → the **single direction** (coincides with Venhoff's
behaviour-vs-overall direction after normalisation). The **projected operator** (Huang's "manifold
steering") is the same `d_b` projected onto `P_k`, the top-k principal subspace of the pooled span
activations at that layer, then renormalised. `fig_manifold_steering.pdf` contrasts them, with the
reminder that `P_k` is a flat linear subspace.

**Application.** Projective, following Huang:

```
h^(ℓ)  ↦  h^(ℓ) − α (vᵀ h^(ℓ)) v
```

applied at every token, sign reversed to amplify. At α=1 the whole v-component is removed; α>1
over-removes. **Registered** sweep: α ∈ {0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0}, k ∈ {1,3,5,10,auto}.
**Executed:** α = 1 only, k ∈ {3,5} only.

**Not a replication.** Venhoff et al. use three distill models on an independent 500-task corpus,
1,000-token chains, GPT-4o annotation, and *additive* positive/negative steering; this thesis uses one
model, a separate 1,000-task corpus, an 8,192-token cap, Sonnet 4.5 annotation, and one
*projective-ablation* dose with two projected variants. Exact reproduction of their effect sizes is
outside the executed estimand.

### 7.2 Metrics (§5.2)

Primary readout = paired change in target behaviour fraction, `Δ_{b,A} = mean_i (f^A_ib − f^V_ib)`,
task as unit, spans as nested measurements; relative change divides by the paired unsteered mean.
Because fractions are length-sensitive, absolute count and **count per thousand generated tokens** are
also recorded, plus repetition rate, degenerate-output rate, mean token count, and all-four-behaviour
scoring for off-target change. **The registered task-accuracy guard was not computed**, so accuracy
preservation is not established. Nominal n = 50 category-stratified held-out tasks; annotation
attrition leaves **47–49 paired tasks**.

### 7.3 What the test cannot decide (§5.3)

Both operators are prespecified linear comparators — the curvature diagnostic neither licenses nor
excludes nonlinear alternatives. The hold-out is genuine (the 50 evaluation tasks are excluded before
any mean or PC is computed), **but construction and evaluation share an annotator family**. Layer
selection used first-order gradient attribution, not an activation transplant, so a null at one
selected layer cannot distinguish "no effect" from "unsuitable site."

### 7.4 Results (§5.4)

**Amendments defining the executed estimand:** gradient-attribution layers L17 (bt), L15 (unc), L15
(ex), L17 (ak) — note example testing is steered at L15, *not* at L27 where Ch. 4 found its only
supported PCA cell; α collapsed to 1.0 with greedy decoding and one sample per task; k restricted to
{3,5}. Annotation reached **1,629 of 1,650** generated chains (98.7%). Run provenance:
`git_commit: null`, no input hashes; **the frozen snapshot lacks the paired annotation artefact, so
the raw estimates below are presently unverifiable within the repository.**

Paired steered-minus-unsteered change in annotated target-behaviour fraction (percentage points;
relative change in parentheses):

| Target behaviour | Single direction | Projected k=3 | Projected k=5 |
|---|---|---|---|
| **backtracking** | −5.52 (**−50.7%**) | −3.81 (**−35.0%**) | −5.74 (**−52.1%**) |
| uncertainty estimation | −2.49 (−11.6%) | +0.01 (+0.04%) | **+2.05 (+9.6%)** |
| example testing | −4.45 (−47.3%) | −1.01 (−10.8%) | −1.35 (−14.3%) |
| adding knowledge | −0.10 (−1.6%) | −0.89 (−14.2%) | **+1.27 (+19.8%)** |

Paired task counts (single / k=3 / k=5): 49/49/48 (bt), 49/47/49 (unc), 49/49/49 (ex), 48/49/48 (ak).

**Backtracking falls under all three operators (35–52%)**; example testing falls in every comparison
but its 47.3% single-direction reduction shrinks to 10.8–14.3% under projection; uncertainty and
adding knowledge **change sign** across operators.

**Length adjustment.** Single-direction reductions per thousand generated tokens: bt **1.093**, unc
0.320, ex 0.731, ak 0.023 — the ordering survives.

**Cross-behaviour specificity** (`fig_steering_specificity.pdf`). With
`s_tBb = r^V_tb − r^A_tBb` and `g^abs_tB = s_tBB − ⅓Σ_{b≠B} s_tBb` (and a base-rate-controlled
relative analogue `g^rel`), the backtracking vector reduces its own target by 1.093 per thousand but
also reduces **uncertainty by 0.881** (then 0.240 and −0.166). Its own-over-other **absolute** margin
is +0.774 (95% CI [0.093, 1.966], p = 0.0618 — *unresolved*); its **relative** margin is +0.7348
(CI [0.3498, 1.3636], **p = 0.0012** — resolved; note this is a 73.5-*percentage-point* difference
between relative suppression proportions, not a 73.5% increase). The uncertainty vector reduces
*backtracking* by 0.523, more than its own 0.320. The example-testing vector has a +64.4 pp relative
margin (p = 0.0056). **The off-target change precludes a claim of isolated target selectivity.**

**Projection comparison.** No consistent benefit: projection *weakens* the example-testing change,
fails to stabilise uncertainty or adding knowledge, and at k=5 reproduces the single-direction
backtracking result — unsurprising, since **cos(single, k=5) = 0.965** for backtracking, so projection
removes little of the original direction. Huang et al. report advantage *especially as strength
increases*; a single dose cannot trace that frontier.

**Robustness check (the decisive narrowing).** Each learned intervention was compared with a matched
random perturbation through the same rule (energy-matched random direction for the single operator;
same-rank random subspaces for projected operators — a weaker match, since rank rather than removed
energy is held fixed). **Only backtracking single-direction and k=5 remained supported on both the
absolute-count and per-thousand-token endpoints (Holm-adjusted p = 0.007 and 0.002).** Example
testing's apparent fraction reduction did not persist on those endpoints; adding knowledge did not
exceed its matched perturbation; uncertainty remained underpowered.

### 7.5 Answer to RQ2 (§5.5)

All three backtracking operators reduce the annotated endpoint and the single direction also produces
a substantial example-testing reduction; uncertainty and adding knowledge vary in sign. Under the
registered decision rule, **the stronger supported result is limited to the backtracking single
direction and k=5 projection**. For backtracking's single direction the relative own-over-other margin
is resolved but the absolute margin is not, and uncertainty also falls. No projected-vs-single
advantage is detected. Verdict wording: "a bounded change in the tested backtracking endpoint, not
general or accuracy-preserving control of reasoning."

---

## 8. Chapter 6 — Training-Time Intervention: Safety Reasoning and Post-Training (pp. 34–39) — **RQ3**

**Function.** Asks which aspects of *generic-reasoning* representations change under safety vs matched
non-safety post-training. Safety reasoning is the **training target**; generic reasoning is the
**held-fixed representation family on which spillover is measured**. The four behaviours are not
treated as safety concepts.

### 8.1 Framing and statistics (§6.1)

Deliberative alignment (train the model to reason over an explicit safety spec before complying) is the
substantive training-time intervention whose spillover is measurable. The question is explicitly *not*
whether safety reasoning forms a manifold or whether the checkpoint is safer.

For base activation `h_i`, post-training `h'_i`, `d_i = h'_i − h_i`:

```
M = mean_i‖d_i‖ / mean_i‖h_i‖          (mean row-displacement magnitude)
C = ‖mean_i d_i‖ / mean_i‖d_i‖          (directional coherence)
MC = centroid displacement relative to mean base activation norm
```

Rotation = excess principal angle under a chain-partition calibration whose null expectation is zero.
Chain sign flips, disjoint partitions, chain bootstraps, and training seeds address **different**
uncertainties and are not interchangeable.

### 8.2 Candidate-parent boundary (§6.2)

A bounded direct-weight audit is *consistent with* DeepSeek's identification of Qwen2.5-Math-1.5B as
base (token-embedding cosine **0.994**, output-head **0.973**), but the frozen similarity table has no
execution-provenance block, so it is not used to rank candidate parents, does not exclude an unobserved
intermediate stage, and does not make any later contrast an origin experiment.

### 8.3 Design scope (§6.3)

Both comparisons reuse the same 993 annotated generic-reasoning chains and behaviour-indexed spans as
RQ1. The **checkpoint** comparison includes *every* difference between two specified models; the
**adapter** comparison approximates a *recipe* contrast at low rank. Similar magnitudes under safety
and non-safety adapters = evidence of generic post-training movement, **not** safety specificity.
Byte-identical teacher forcing makes rows pairable but measures no refusal, benign compliance, task
accuracy, or free-generation behaviour frequency. Provenance boundary: `git_dirty: true` on the public
spillover record; gated/rotation/attribution JSONs lack complete lineage; **no exact checkpoint weight
hashes**; the frozen snapshot lacks the decomposition, per-seed, and depth/rank-sweep artefacts.

### 8.4 Results (§6.4)

**Checkpoint rotation.** All 993 chains teacher-forced through both models after explicit token-ID
equality checks → **37,851 row-paired span activations per model**. The rotation statistic partitions
chains into disjoint halves and subtracts the within-model angle from the cross-model angle on the same
partition (exactly zero for identical matrices). Rotation excess is **positive in all eight
behaviour × layer cells at L12 and L16, +0.26° to +0.45°**, but the partition interval excludes zero
**only for uncertainty estimation, at both layers**. A synthetic 5° single-plane rotation yields only
+0.45°–0.72° on this statistic — the estimator *attenuates* a known rotation, which does not justify
inverting the calibration to recover a latent angle. Sign is stable for k ∈ {2,3,5,8,10}.

**Checkpoint translation — the clearer effect.**

| Quantity | Value |
|---|---|
| Mean paired row displacement **M** | **4.7–6.4%** of mean activation norm |
| Coherence **C** | **0.70–0.78** (vs chain sign-flip null 0.0075–0.0142) |
| Normalised centroid displacement **MC** | **3.5–4.6%** |
| Cross-behaviour direction cosine | **0.95–0.99** |
| Depth profile | ~3–6% displacement at every 4th layer L2→L26; rotation excess positive and sub-degree throughout |

**Matched-adapter rotation and magnitude.** Four LoRA adapters (safety at 100/300/1,000 STAR-1;
non-safety at 1,000 length-matched generic chains). Largest rotation excess across **32 cells is
+0.012°** and every interval includes zero — **no adapter rotation is resolved**. Magnitude also fails
to separate: **≈0.005 of activation norm for safety vs 0.005–0.006 for control**.

**Direction attribution — where the recipe does show up.** Cosine of the adapter translation direction
to the public safety checkpoint's own translation direction:

| Arm | L12 | L16 |
|---|---|---|
| Safety (1,000) | **0.58** [0.57, 0.59] | **0.57** [0.56, 0.58] |
| Size-matched non-safety control | **0.14** [0.13, 0.15] | **0.17** [0.16, 0.19] |
| Paired contrast | **+0.44** [0.43, 0.45], p < 5×10⁻⁴ | **+0.39** [0.38, 0.41], p < 5×10⁻⁴ |
| Random-direction floor | ≈0.02 | ≈0.02 |

Chain-level bootstrap, B=2,000. Safety-arm direction is similar across dose (pairwise cosine 0.92–0.98);
a disjoint-half sensitivity gives the same ordering. A **shared-component decomposition tempers this**:
roughly a third of the alignment rides on a component along the base mean activation shared with the
full fine-tune; after removal the contrast is **0.50 vs 0.18** (L12) and **0.49 vs 0.24** (L16). These
bootstrap quantities condition on the two fitted adapters and quantify *row-sampling* uncertainty —
they do not test recipe replication.

**Seeds and sensitivity.** Across three seeds per recipe: within-recipe cosines **0.989–0.994**
(including control); cross-recipe **0.16–0.29** over nine pairs (0.26–0.29 at L12, 0.16–0.18 at L16);
alignment ranges to the full fine-tune do not overlap (0.57–0.59 vs 0.14–0.18, three-against-three).
**The exact permutation value is one-sided p = 0.05, two-sided p = 0.10 — not confirmatory.** Also:
per-span surprisal weakly anti-correlates with displacement (Spearman −0.22, −0.20 over 37,851 spans);
under **Nova-Pro span boundaries** the two translation directions have **cosine 0.999** at both layers
with coherence 0.75/0.71 clearing its sign-flip null (≈0.035, permutation p = 0.002) — robustness to
one alternative segmentation only. Registered behaviour-selectivity prediction: **not supported**.

`fig_posttraining_geometry.pdf` carries the three panels (A: M and MC per behaviour–layer cell,
1,500 matched activations each; B: disjoint-half rotation excess with only the two uncertainty
intervals excluding zero; C: adapter-to-checkpoint alignment with bootstrap whiskers and three seed
points).

### 8.5 Answer to RQ3 (§6.5)

**Translation rather than strong reshaping.** Full-checkpoint M = 4.7–6.4%, MC = 3.5–4.6%, direction
cosines 0.95–0.99; rotation excess positive but +0.26°–0.45° and resolved above split noise only for
uncertainty estimation — so **no categorical no-rotation claim**, but translation is better supported.
The matched adapters produce similar magnitudes with no resolved rotation, so **displacement magnitude
is not safety-specific under the control**; their *directions* differ, with safety seeds aligning more
closely to the public checkpoint, but with two-sided p = 0.10 that **direction attribution is
exploratory, not confirmatory**. Nothing here establishes that safety post-training created the
geometry, a universal safety axis, or any change in refusal, compliance, task performance, or
generated reasoning behaviour.

---

## 9. Chapter 7 — Discussion and Conclusion (pp. 40–44)

**§7.1 Answers, restated at supported strength.** RQ1: low occupancy (7.040–8.252) over 37,851 raw rows
in 993 chains, but H2 fails, H3 fails, H4 unrun, PCA behaviour/depth-dependent, curvature mixed, and
the record shows a null commit and dirty run. RQ2: conditional on a materially amended protocol
(execution-selected layers, α=1.0, k ∈ {3,5}, small cells, no accuracy guard) — backtracking falls
35–52%, example testing 11–47%, uncertainty and adding knowledge inconsistent; stronger conclusion
limited to backtracking single + k=5; snapshot lacks the paired annotation summary. RQ3: translation
4.7–6.4% / centroid 3.5–4.6% / cosines 0.95–0.99, rotation +0.26°–0.45° resolved only for uncertainty,
matched adapters similar in magnitude, seed-level two-sided p = 0.10 → exploratory recipe attribution.

**§7.2 Scientific interpretation.** The two scales give **complementary but asymmetric** evidence, and
the asymmetry is itself presented as a result: steering has a *behavioural* endpoint with limited
intervention coverage, while post-training has *persistent representation-level* evidence with
behavioural evaluation still outstanding. The steering result is qualitatively comparable with Venhoff
but is not a replication; relative to Huang's manifold steering, no projected-vs-single advantage
appears at one dose. The training-time result is "consistent with an approximately affine account" of
the tested change, compatible with dominant/secondary safety-direction reports and with systematic
cross-stage geometry change — but three seeds keep attribution exploratory. On geometry: low
descriptive correlation dimension may reflect mean-pooling, the sampled chain population, or estimator
behaviour at this scale, and **the current design does not distinguish these explanations**. The
behaviour-dependent PCA cells do not contradict the non-confirmatory cdim family because *linear
variance concentration and pair-count scaling are different estimands*.

**§7.3 Methodological contribution.** A chain-aware measurement and falsification protocol with an
audited corpus, keeping five estimands distinct, recording amendments before interpreting results, and
distinguishing **current / exploratory / provisional / prospective / superseded** evidence. Key line:
"the PCA null does not substitute for the failed direct correlation-dimension null."

**§7.4 Limitations and priorities.** Four limitations: (1) one 1.5B checkpoint, three *model*
annotation sets with differing extraction vintages — since Venhoff also evaluate Qwen-1.5B, model size
cannot explain differences; the missing thing is their cross-architecture/cross-scale evidence.
(2) Representation dependence remains — H3 fails on PR bands, H4 unrun, and the frozen occurrence
selector includes family/layer coordinates so families test their own deterministic selections rather
than identical point samples. (3) Steering inherits its amendment: layer/dose/k differ from the
registered design, scoring is within-annotator, cells are small, and without an accuracy guard an
endpoint change cannot establish useful behaviour-specific control; the design also cannot determine
whether discrepancies with Venhoff arise from task distribution, generation length, annotation
pipeline, operator, sampling, or their interaction. (4) The safety comparison has limited causal
identification and, being teacher-forced, shows nothing about refusal, compliance, accuracy, or
generated behaviour.

**Two named priorities:**
1. **Behavioural evaluation of the post-trained models** on held-out safety and generic-reasoning
   benchmarks — prompts paired across base, public safety checkpoint, and the owned seed-42
   full-parameter safety/non-safety pair; low-rank arms admitted only if exact weights are recovered or
   retrained under immutable manifests; safety prompt manifest train-disjoint at record level; measure
   harmful-request refusal, benign compliance, task correctness, response length, repetitive decoding,
   and the four behaviour prevalences — kept **separate** from the teacher-forced comparison, with
   independent scoring where feasible.
2. **Independent rescoring and replication of the steering result** with an accuracy guard and held-out
   design, preserving a prespecified direction-specificity test, to see whether the provisional
   backtracking contrast survives independent measurement without trading away task performance.

**§7.5 Conclusion.** "Reasoning-trace annotations index activation space, but the measured clouds do
not support a uniform manifold account." What stands instead is a bounded two-scale intervention
picture: clear descriptive inference-time changes with the stronger provisional result restricted to
backtracking, and post-training conditions that shift generic-reasoning representations more clearly by
translation than by reshaping, with behavioural consequences unevaluated. Geometry constrains the
interpretation rather than supplying a mechanism.

---

## 10. Appendix A — Exploratory and Provisional Analyses (pp. 45–50)

### A.1 Implementation audit and supersession history

The quarantine record. Faults corrected before the retained analyses:

| ID | Fault | Remedy |
|---|---|---|
| CF-1 | Curvature compared unequal local/global sample sizes (flat Gaussian data scored in the band first read as curvature) | Sample-size-matched operator + known-answer checks: flat Gaussian and two-plane now score **1.00**, two-sphere **0.67** |
| CF-3 | Initial null tested PCA variance, not the primary geometric statistic | Direct equal-chain cdim family + paired curvature test |
| CF-4 | Intrinsic-dimension implementations biased upward (TwoNN renormalised on a truncated sample) | Corrected estimators; alternatives retained descriptively only |
| CF-13 | First-occurrence string matching created exact duplicate rows | Occurrence-aware re-extraction + duplicate hard checks |
| CF-14 | Silent provenance fallback reduced each behaviour to one pseudo-chain, making within-chain permutation vacuous | Ordered sidecar + hard failure on missing/misaligned provenance |
| CF-15 | Probe cross-validation leaked rows across chains | Chain-grouped split + complete rerun |
| CF-16 | Monte-Carlo p-values unsmoothed, multiplicity not run | Smoothed values + executed Holm correction |

Design limits that repair does **not** remove: **CF-2** (within-chain dependence → chain-level
inference), **CF-5** (linear PCA cannot establish curvature), **CF-6** (mean-pooling collapses each
trajectory to one point).

### A.2 Exploratory representational diagnostics

- **Earlier chronology.** Behaviour-specific cells (L14/L14/L27/L17) gave cdim **5.85, 6.21, 6.04,
  7.71**; a 10-seed one-random-row-per-chain sensitivity gave means **6.36, 6.88, 6.72, 8.46** (population
  SDs .1128/.0992/.1638/.0879) — no inferential coverage. Variance-threshold dimension is
  threshold-sensitive: 70% ≈ 40–80 components; 90% and 95% hit the 100-component cap and are only lower
  bounds.
- **Earlier curvature.** Local-to-global ratio 0.55–0.69 on full rows, **0.98–1.01 after
  one-sentence-per-chain sampling**, 0.75–0.86 for an equal-size chain-blind row sample — a vivid
  demonstration that the chain confound drove the original signal. Synthetic calibration (B=50,
  N ∈ {100, 500, 2000}) reached AUC 1 in two cells at the lower curvature setting but **AUC 0.5 at the
  higher setting for every sample size** — explicitly flagged as counterintuitive and not evidence of
  power.
- **Pooling sensitivity at L27:**

| Behaviour | cos(last, mean) | d_eff mean | d_eff last | d_eff first |
|---|---|---|---|---|
| backtracking | 0.57 | 57 | **100 (cap)** | 18 |
| uncertainty | 0.83 | 71 | **100 (cap)** | 18 |
| example testing | 0.87 | 52 | **100 (cap)** | 27 |
| adding knowledge | 0.81 | 76 | **100 (cap)** | 38 |

  Spectral summary and direction are representation-sensitive; this is **not** the unrun H4 family.
- **Cross-layer.** PR-trough layer rule: L16, L16, **L12** (example testing), L16. Chain-grouped probe
  accuracy **0.70–0.84** vs 0.50 baseline with little variation across 28 layers — replacing an earlier
  **chain-leaky 0.83–0.93**. Mean principal angles between top-ten PCA spans grow from **34–37° at gap
  3 to 61–64° at gap 14**.
- **Clustering.** Silhouette sweep peaks at k=2 with only **0.18–0.20** — a limited negative that does
  not establish universal absence of subtypes.
- **Cross-annotation recurrence.** cdim recurs at ≈6–9 across all three annotation sets:

| Behaviour | Sonnet full / 1-per-chain | Qwen3 full / 1-per-chain | Nova full / 1-per-chain |
|---|---|---|---|
| Backtracking | 5.9 / 6.4 | 6.7 / 6.7 | 6.6 / 6.8 |
| Uncertainty | 6.2 / 6.9 | 7.0 / 7.1 | 7.2 / 7.0 |
| Adding knowledge | 7.7 / 8.5 | 8.4 / 8.6 | 8.7 / 8.5 |
| Example testing | 6.0 / 6.7 | 6.6 / 7.0 | 6.0 / 7.1 |

  But secondary pools carry **¼–⅗ raw duplication vs ~1% in the primary re-extraction**, and extraction
  vintages differ, so cross-annotation numerical comparison is prohibited.
  The separate exploratory L12/L16 PCA family (rows 37,380 / 34,282 / 34,406; B=500, seed 42,
  **unadjusted** α=.05): backtracking and uncertainty exceed their permuted references at both depths
  under all three annotations, each at the smoothing floor 1/501 = .001996; example testing is
  unsupported (Sonnet p=.958 at L12, .0599 at L16; both others p=1) as is adding knowledge (all p=1).

### A.3 Execution-record corrections outside geometry

- **Steering:** one shared unsteered annotation (`SCIE_103`) is missing from the authoritative file, so
  the paired analysis uses only doubly-annotated tasks (47–49 across the twelve cells); **no incomplete
  recovery annotation was substituted**; the accuracy guard is reported as unexecuted.
- **Checkpoints:** a tokeniser mismatch in an early extraction was found in validation; the retained
  extraction was regenerated and gated on exact token-ID equality. An early principal-angle calibration
  compared *overlapping* rows in the observed term against *disjoint* rows in its within-model floor,
  inducing a negative expectation under no change; the retained statistic partitions chains into
  disjoint halves for both terms and is exactly zero for identical matrices. **Only the corrected
  calibration appears in Ch. 6.**

### A.4 Steering follow-ups

- **Repetitive decoding collapse** (chain classified as collapsed when 4-gram repetition > 0.8):
  vanilla greedy **0.34** of tasks; full ablation raises it to **0.64** (example testing, k=5) and
  **0.68** (uncertainty, k=3); the **backtracking k=5 operator instead falls to 0.24**. A small
  dose-by-decoding check is consistent with an attractor description — example testing k=5 rises
  0.42 → 0.64 → 0.76 over α ∈ {0.5, 1.0, 1.5} under greedy decoding, while at T=0.6 between 75% and
  100% of greedy full-ablation cases no longer cross the threshold. (The dose-by-decoding artefact is
  absent from the snapshot; `collapse_table.json` verifies the headline rates but has no provenance
  block.)
- **Fitted-frame (DAS) diagnostics:** the exercise reused the **same 400 interchange pairs for
  optimisation and evaluation**, so its transfer/coordinate/width summaries are resubstitution
  diagnostics, not held-out evidence. A fresh task- and pair-held-out evaluation is required before
  they can enter the main results.

### A.5 Unexecuted deliberative-safety programme

The gpt-oss-20b deliberative-safety label schema was gated on a reliability pilot. First pass: the
**decision** label reached Fleiss **κ = 0.223** across three LM judges, below the preregistered 0.4
threshold. Audit found *decision* conflated a policy verdict with the action it licenses, and
*adjudication* did not separate safety weighing from content planning. Under a prespecified
**no-third-iteration rule** the schema was revised once, reaching **0.798** (decision), **0.885**
(specification citation), **0.839** (harm recognition); **adjudication stayed at 0.226 and was
excluded**. The frozen human-comparison artefact is absent, so no human-agreement estimate is used.
These results concern a planned object and are **not part of RQ3**; the programme does not license the
planned recipe fingerprint, jailbreak localisation, process divergence, or forged-vs-genuine provenance
analyses.

### A.6 Provisional DS-H1 pilot

52 deliberative-safety rows from 12 chains vs 3,241 generic rows from benign chains. At the stored
primary layer: chain-held-out **Cohen's d = 5.02** (positive in all five folds), **AUROC 0.979**,
permutation **p = 0.001**, bootstrap [4.36, 5.59]; quarter- and three-quarter-depth reads also positive
(d = 3.69 and 5.07, both p = 0.001). Capability control: the candidate safety direction has
**|cos| = 0.192** with the hard-vs-easy capability axis and retains **0.995** of its separation after
projecting that axis out.

**The low-dimensional qualifier is struck**: TwoNN gave 3.5, *above* the 5th percentile of the
matched-N generic null (2.8); PCA dimensions at 90/95/99% variance were 17/25/39 vs generic 18.8/24.0/32.6.
Any future DS-H1 claim must omit a low-dimensional qualifier.

Two unresolved provenance conflicts are documented rather than papered over: the stored `h1_results.json`
uses zero-based `f25=6, f50=11, f75=17` (primary read = L11) while the preregistration prose records
L6/L12/L18 and calls L12 primary; and the preregistration and `H1_REPORT.md` state **seed 0** whereas
the serialized provenance states **seed 42**. Together with LM-consensus labels pending a 50-sentence
human re-anchor, unavailable difficulty matching, and coverage of one model at one reasoning-effort
level, these keep DS-H1 provisional and leave DS-H2–DS-H4 unlicensed.

---

## 11. Cross-cutting: the evidence-status ledger

The thesis maintains five explicit tiers. Reading the document without them will misrepresent it.

| Status | Instances |
|---|---|
| **Passing / supported** | H1 descriptive occupancy (all four ≤ 10); fixed-top-ten PCA concentration (11/20 cells: bt ×5, unc ×5, ex L27); curvature bounded-negative for unc and ex; checkpoint translation (M, C, MC, direction cosines); backtracking steering vs matched perturbation (Holm p = .007 / .002) |
| **Failed** | H2 direct specificity (4/4); H3 composite (4/4, driven by non-overlapping PR bands under truncation matching); both registered secondary cdim families; adapter *magnitude* specificity; adapter rotation; registered behaviour-selectivity prediction |
| **Mixed** | Registered L16 curvature (2 negative, 2 nonconforming, in opposite directions) |
| **Provisional / exploratory** | Whole steering result (amended protocol, within-annotator, no accuracy guard); adapter *direction* attribution (two-sided p = 0.10); cross-annotation PCA recurrence (unadjusted); parent-weight check; DS-H1 pilot; repetition-collapse follow-ups; DAS fitted frame |
| **Unrun** | H4 pooling/window family (inputs absent); dose frontier; full activation-patch confirmation; task-accuracy guard; free-generation behavioural safety evaluation; DS-H2–DS-H4 |

### Provenance gaps disclosed in the text (unusual, and deliberate)

- Corpus generation: immutable request/response logs not located → settings reconstructed.
- Annotation chunking: no executed window manifest → cap/threshold/overlap not treated as facts.
- Core-hardening geometry record: **`code_commit: null`, `dirty: true`**.
- Cross-layer probe record: commit `06f07fec` but **dirty tree**.
- Steering run: **`git_commit: null`**, no input hashes; **paired annotation artefact absent from the
  snapshot → raw percentages presently unverifiable in-repo**.
- Post-training: **`git_dirty: true`**; gated/rotation/attribution JSONs lack complete lineage; **no
  exact checkpoint weight hashes**; decomposition and per-seed artefacts absent → those intervals and
  the exact seed-level value unverifiable in-repo.
- `collapse_table.json`: no provenance block; dose-by-decoding artefact absent.
- DS-H1: `git_dirty: true`, plus the layer-index and seed conflicts above.
- ICC/design-effect calculation for CF-2 absent from the snapshot → not used.
- No public deposition of activation tensors or model-derived corpora at this freeze → no DOI claimed.

---

## 12. Quick numerical reference

| Quantity | Value |
|---|---|
| Tasks / chains / raw target rows | 1,000 / 993 analysed / 37,851 |
| Token cap / mean chain length / at cap | 8,192 / 5,052 / 50.2% |
| Annotation κ (6-label) | 0.436, 0.350, 0.345 |
| Common-L27 cdim (bt/unc/ex/ak) | 7.208 / 7.364 / 7.040 / 8.252 |
| H2 Holm p | 1.000 across all four |
| PCA specificity survivors | 11 of 20 cells, Holm ≈ 0.0080 |
| Curvature medians | 1.0339 / 1.0153 / 1.0000 / 0.9649 |
| Steering: backtracking fraction change | −50.7% (single), −35.0% (k=3), −52.1% (k=5) |
| Steering: robustness survivors | backtracking single + k=5 only (Holm p = .007, .002) |
| cos(single, k=5) for backtracking | 0.965 |
| Checkpoint M / C / MC | 4.7–6.4% / 0.70–0.78 / 3.5–4.6% |
| Checkpoint rotation excess | +0.26° to +0.45° (resolved only for uncertainty) |
| Adapter rotation (max, 32 cells) | +0.012°, all intervals include zero |
| Adapter direction alignment (L12) | safety 0.58 vs control 0.14; contrast +0.44 |
| Seed test | one-sided p = 0.05, two-sided p = 0.10 |
