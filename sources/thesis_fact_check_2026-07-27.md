# Thesis fact-check record — 27 July 2026

This record supports the thesis-wide scientific-writing revision. It records only
claims that were checked against a primary paper, official project repository, or a
controlling local analysis artefact. It is not a substitute for the bibliography.

## External sources

### DeepSeek-R1 and the 1.5B distill

- Primary sources:
  - <https://github.com/deepseek-ai/DeepSeek-R1>
  - <https://arxiv.org/abs/2501.12948>
- Verified:
  - DeepSeek identifies Qwen2.5-Math-1.5B as the base model for
    DeepSeek-R1-Distill-Qwen-1.5B.
  - The Qwen distills were fine-tuned on 800,000 samples curated using DeepSeek-R1.
  - The distill is a supervised fine-tune rather than a model that itself underwent
    the teacher's reinforcement-learning procedure.
- Wording control:
  - State that the public documentation identifies Qwen2.5-Math-1.5B as the base.
    The local weight comparison independently ranks it as the closest of the three
    tested candidate checkpoints; neither observation establishes every detail of
    lineage.
  - Do not say that the source proves the absence of all safety-related examples.
    Say that it reports no dedicated safety-training stage for this distill.

### Venhoff et al. — reasoning-behaviour steering

- Primary source: <https://arxiv.org/html/2506.18167>
- Verified:
  - 500 tasks across ten categories were generated for the extraction study.
  - The paper defines six sentence-level categories and studies four steering
    targets: uncertainty estimation, example testing, backtracking, and adding
    knowledge.
  - For DeepSeek-R1-Distill-Qwen-1.5B, the reported selected layers are 18, 15, 17,
    and 18 respectively.
  - Layer selection uses a first-order gradient approximation to an additive
    steering intervention and is described by the authors as attribution patching.
- Wording control:
  - In this thesis, use “first-order gradient attribution” when referring to the
    executed selector. It did not perform a stored-activation transplant. The
    background may note that the source places this approximation in the
    attribution-patching family.
  - Do not claim that the source used matched random-direction or magnitude floors;
    those are controls added by this thesis.

### Huang et al. — manifold steering

- Primary source: <https://arxiv.org/html/2505.22411>
- Verified:
  - The top ten principal components captured more than 70% of variance in the
    studied overthinking activation set.
  - The paper reports that a single-direction intervention plateaus and can
    deteriorate at larger strengths.
  - Its proposed operator projects the steering direction into the PCA subspace.
  - Theorem 4.2 concerns propagation of an activation shift attributed to the
    orthogonal interference component; it is not direct evidence for any theorem
    about the present thesis's behavioural categories.
- Wording control:
  - Describe the paper's empirical result and operator directly. Do not use
    Theorem 4.2 to imply that the present projected operator must outperform a
    single direction.
  - Treat “manifold” as the source's terminology. PCA concentration alone does not
    establish curvature or a nonlinear manifold.

### STAR-1

- Primary source: <https://arxiv.org/html/2504.01903>
- Verified:
  - STAR-1 contains 1,000 policy-grounded deliberative-safety examples.
  - The reported checkpoint training uses full-parameter supervised fine-tuning,
    five epochs, learning rate \(10^{-5}\), batch size 128, and an 8,192-token
    sequence limit.
  - The paper reports an average 1.1% decrease across five reasoning evaluations,
    alongside improved safety.
- Wording control:
  - Describe the public checkpoint as reported full-parameter fine-tuning on the
    STAR-1 dataset. Avoid the stronger, unverified phrase “and nothing else”.
  - If mentioning the reasoning cost, attribute it to the STAR-1 evaluation rather
    than presenting it as an outcome of this thesis.

### Pan et al. — safety residual space

- Primary source: <https://arxiv.org/html/2502.09674>
- Verified:
  - The paper defines a safety residual space from representation shifts induced by
    safety fine-tuning.
  - It fits an affine approximation to those shifts and analyses the leading
    directions of the linear term.
  - It reports a dominant refusal-related direction and smaller directions
    associated with features such as hypothetical narrative and role-playing.
- Wording control:
  - This source can motivate separating translation, linear transformation, and
    direction structure. It does not establish that every safety fine-tune is a
    low-rank affine map, nor does it validate the present checkpoint comparison.

### Nakamura — template-controlled activation differences

- Primary source: <https://arxiv.org/abs/2605.24583>
- Verified:
  - The paper shows that unmatched chat templates can confound aligned-minus-base
    activation differences.
  - It proposes four contrasts, including a difference-in-differences estimator,
    and reports that template control changes effective-rank and refusal-direction
    estimates.
- Wording control:
  - Cite this source specifically for template/tokenisation control. The present
    design instead uses byte-identical token IDs across each checkpoint pair; it
    does not implement Nakamura's four-arm DiD protocol.

### Soligo et al. — convergent misalignment directions

- Primary source: <https://arxiv.org/abs/2506.11618>
- Verified:
  - In an emergent-misalignment model organism, different fine-tunes converge
    towards similar activation directions.
  - A direction extracted from one fine-tune can ablate misaligned behaviour in
    other fine-tunes.
- Wording control:
  - This is an analogy for recipe-associated directions, not evidence about safety
    post-training or the present models. Use only as contextual comparison.

### Shuttleworth et al. — LoRA versus full fine-tuning

- Primary source: <https://arxiv.org/abs/2410.21228>
- Verified:
  - LoRA and full fine-tuning can have different weight-spectrum structure even
    when downstream performance is similar.
  - The paper's principal object is parameter-space spectral structure and
    generalisation, not the residual-stream translation measured here.
- Wording control:
  - Cite only to motivate caution when comparing LoRA adapters with a
    full-parameter checkpoint. Do not present it as external support for this
    thesis's activation-space result.

### Venhoff et al. — base versus thinking models

- Primary source: <https://arxiv.org/abs/2510.07364>
- Verified from the current v4 abstract:
  - The paper distinguishes training regimes. Its reported hybrid reconstruction
    recovers substantially more of the base-to-thinking gap for RL-trained pairs
    than for SFT-distilled pairs.
  - The authors interpret RL as primarily learning deployment heuristics and
    SFT-distillation as more often installing new mechanisms.
- Wording control:
  - Do not generalise “base models already contain the mechanism” without
    conditioning on training regime.
  - For the SFT-distilled model used in this thesis, cite this work as evidence that
    the base/post-training distinction is empirical and regime-dependent, not as
    proof that the candidate parent already contains every studied behaviour
    mechanism.

## Controlling local artefacts

### RQ1 and RQ2

- Preregistration and amendments:
  `results/prereg/THESIS_CORE_HARDENING_PREREG_2026-07-26.md`.
- Primary common-layer result:
  `results/robustness/core_hardening/R1-1.5B/primary_L27_cdim_null.json`.
- Depth-family and annotator-family sealed JSON files under
  `results/robustness/core_hardening/R1-1.5B/`.
- Curvature:
  `results/robustness/core_hardening/R1-1.5B/curvature_L16.json`.
- Verified controlling conclusions:
  - H1 passes: all four chain-level correlation-dimension estimates lie in the
    preregistered occupancy interval.
  - H2 fails at the common layer, across five depths, and across three annotators
    after familywise correction.
  - H3 fails as a family; chain stability is mixed and truncation sensitivity fails.
  - Curvature is mixed across behaviours.
  - H4 is unrun because the preregistered aligned inputs are absent.

### Steering

- Primary evaluation:
  `results/eval/R1-1.5B__E1/delta_floor_report.json`.
- Strengthened inference:
  `results/eval/R1-1.5B__E1/strengthen_report.json`.
- Verified controlling conclusion:
  - Backtracking single-direction and \(k=5\) projected steering exceed the matched
    random-direction floor and increase both count and length-normalised rate.
  - The \(k=3\) projected operator does not clear the floor.
  - Example testing has a positive fraction contrast but fails the degeneration
    controls.
  - Uncertainty estimation and adding knowledge do not clear the declared floor.
  - Cross-behaviour specificity is mixed: the relative contrast is resolved, but
    the absolute margin is not.

### Post-training and safety

- Full checkpoint:
  `results/safety_posttrain/spillover_gated_full.json`.
- Direction inference:
  `results/safety_posttrain/pt04_perarm_null.json` and
  `results/safety_posttrain/pt04b_decomposition.json`.
- Rotation half-split:
  `results/safety_posttrain/pt05_rotation_halfsplit.json`.
- Seed replication:
  `results/safety_posttrain/pt06_perseed.json`.
- Depth and rank sensitivity:
  `results/safety_posttrain/pt07_sensitivity.json`.
- Annotation swap:
  `results/safety_posttrain/pt04c_annotator_swap.json`.
- Candidate reference:
  `results/base_model_verification/similarity_table.json` and
  `results/base_model_verification/verdict.md`.
- Verified controlling conclusion:
  - The public full-parameter safety checkpoint differs from the reference by a
    mean paired row displacement of 4.7--6.4% of mean activation norm.
    Directional coherence is 0.70--0.78, so the corresponding centroid
    displacement is 3.5--4.6% of mean activation norm. The earlier shorthand
    that called 4.7--6.4% the cloud-mean shift was corrected during the
    consistency pass.
  - Behaviour-specific mean-displacement directions are highly aligned
    (pairwise cosine 0.948--0.991).
  - The half-split rotation estimate is small (approximately
    \(0.26^\circ\)--\(0.45^\circ\)); most intervals include zero, so “unresolved
    rotation” is the appropriate general wording.
  - LoRA safety and size-matched non-safety arms have similar translation
    magnitudes, whereas the safety arm is more aligned with the public checkpoint
    direction at layers 12 and 16.
  - The direction separation replicates across three seeds, but the exact
    three-versus-three permutation resolution is one-sided \(p=0.05\)
    (two-sided \(p=0.10\)).
  - The local weight comparison ranks Qwen2.5-Math-1.5B as the closest tested
    candidate reference. It does not by itself prove full lineage.
