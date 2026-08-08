# Literature-review search log

Search date: 2026-07-27  
Project: *Measuring the Geometry of Machine Reasoning*  
Purpose: examiner-facing revision of the integrated Background and Literature Review

## Scope

The review asks what prior work establishes about:

1. chain-of-thought traces as observations of reasoning;
2. linear and nonlinear representations in language-model activations;
3. activation probing, patching, steering, and the strength of causal claims;
4. low-rank concentration, intrinsic dimension, curvature, and finite-sample estimation;
5. geometry across reasoning-model training and safety/non-safety post-training.

The search covers foundational work and work available by 2026-07-27. It is a
structured, comprehensive narrative review, not a PRISMA systematic review or a
meta-analysis: the literatures use heterogeneous objects, models, interventions, and
outcomes that do not support a common pooled effect.

## Sources searched

- arXiv (broad preprint and technical-report coverage)
- ACL Anthology (computational-linguistics proceedings and TACL)
- OpenReview (ICLR, workshops, and open reviewing records)
- PMLR (ICML, COLT, CLeaR, and related proceedings)
- NeurIPS proceedings
- Publisher pages for Nature and journal articles

Search-engine discovery was restricted to these scholarly or primary-source domains.
Metadata and substantive claims were checked against the paper, proceedings page, or
official technical report rather than against secondary summaries.

## Query families

The following database-targeted queries were run, with title/author citation chaining
from the most relevant hits:

### Reasoning traces and faithfulness

- `chain-of-thought faithfulness language models reasoning models`
- `reasoning model activations geometry trajectories manifold`
- `reasoning behaviours activation steering language models backtracking uncertainty`
- `internal activations predict reasoning correctness large language models`
- `chain of thought prompting large language models Wei 2022`
- `process supervision language models reasoning`
- `DeepSeek R1 reasoning reinforcement learning`
- `test time scaling reasoning models overthinking`

### Representation and intervention

- `activation steering language models representation engineering`
- `linear representation hypothesis language models geometry`
- `causal intervention activation patching subspace language models`
- `difference in means activation addition steering vectors language models`
- `linear probes language models diagnostic classifiers limitations control tasks`
- `amnesic probing causal probing language representations`
- `causal abstraction neural networks interchange intervention`
- `best practices activation patching language models`

### Representation geometry and estimators

- `intrinsic dimension neural network representations language models`
- `intrinsic dimension transformer hidden representations`
- `correlation dimension estimator neural representations finite sample`
- `geometry neural representations curvature manifold language models`

### Post-training and safety

- `safety fine-tuning representation geometry language models refusal direction`
- `post-training activation geometry language models fine tuning representations`
- `safety alignment reasoning models representation subspace`
- `model diffing fine tuning crosscoder representation language models`
- `deliberative alignment reasoning enables safer language models`
- `safety alignment makes reasoning models less reasonable safety tax`
- `geometry of refusal concept cones representational independence`
- `hidden dimensions LLM alignment orthogonal safety directions`

## Inclusion and exclusion rules

Included:

- primary empirical or methodological work directly concerning at least one review
  theme;
- foundational estimator papers needed to define or qualify the thesis's measurements;
- peer-reviewed proceedings/journal articles, clearly identified preprints, and
  first-party technical reports when the model or training recipe is documented only
  there;
- work that supplies a competing explanation, negative result, or methodological
  limitation, not only work supportive of the thesis.

Excluded from the integrated review:

- generic benchmark papers with no representational, intervention, or estimator
  relevance;
- application papers that merely use the word “manifold” without measuring an
  activation-space object relevant to the thesis;
- secondary news, blog summaries, unsourced leaderboards, and social-media claims;
- anonymous or unaccepted manuscripts when a published or author-identified source
  supports the same point;
- planned but unexecuted safety experiments that do not bear on RQ4.

## Thematic evidence map

### Chain-of-thought is useful but not automatically faithful

- Wei et al. (2022), *Chain-of-Thought Prompting Elicits Reasoning in Large Language
  Models* — establishes the performance role of elicited intermediate reasoning.
- Lanham et al. (2023), *Measuring Faithfulness in Chain-of-Thought Reasoning* —
  intervention-based evidence that reliance on the written chain varies by task and
  model.
- Lyu et al. (2023), *Faithful Chain-of-Thought Reasoning* — separates natural-language
  translation from deterministic solving to guarantee faithfulness in the constructed
  pipeline.
- Elazar et al. (2021), *Amnesic Probing* — decodability alone does not establish that a
  representation is used.
- Chen et al. (2025), *Reasoning Models Don't Always Say What They Think* — extends the
  faithfulness concern to reasoning models.
- Venhoff et al. (2025), *Understanding Reasoning in Thinking Language Models via
  Steering Vectors* — supplies the behaviour vocabulary and direct intervention target
  used in this thesis.

### Linear access, distributed representations, and intervention

- Park et al. (2024), *The Linear Representation Hypothesis and the Geometry of Large
  Language Models* — formal connection between counterfactual representations, probes,
  steering, and the choice of inner product.
- Engels et al. (2024), *Not All Language Model Features Are Linear* — empirical
  counterexample to universal one-direction accounts.
- Hewitt and Liang (2019), *Designing and Interpreting Probes with Control Tasks* —
  probe selectivity and memorisation controls.
- Geiger et al. (2021, 2022) — causal abstraction and interchange interventions.
- Meng et al. (2022), Zhang and Nanda (2024), and Makelov et al. (2024) — activation
  patching, its design choices, and subspace-patching illusions.
- Turner et al. (2023) and Rimsky et al. (2024) — activation addition and contrastive
  activation addition.
- Arditi et al. (2024) and Pan et al. (2025) — one-dimensional and multi-dimensional
  accounts of safety-related activation control.

### Geometry and the meaning of “manifold”

- Mamou et al. (2020), *Emergence of Separable Manifolds in Deep Language
  Representations* — manifold capacity, radius, dimension, and centre correlation in
  contextual representations.
- Ansuini et al. (2019), *Intrinsic Dimension of Data Representations in Deep Neural
  Networks* — nonlinear intrinsic dimension differs from PCA rank.
- Facco et al. (2017) and Levina and Bickel (2004) — nearest-neighbour intrinsic-dimension
  estimators.
- Grassberger and Procaccia (1983) — correlation dimension.
- Cheng et al. (2023), *Bridging Information-Theoretic and Geometric Compression in
  Language Models* — estimator choice matters on linguistic data.
- Marchenko and Pastur (1967) — sample-dependent covariance noise floor.
- Huang et al. (2025), *Mitigating Overthinking ... via Manifold Steering* — low-rank
  PCA projection improves a steering operator, while not by itself identifying
  curvature.
- Zhou et al. (2026), Sun et al. (2026), and Li et al. (2025) — trajectory-level
  geometry of reasoning, distinct from the annotation-indexed point clouds studied
  here.

### Post-training and safety geometry

- DeepSeek-AI (2025) and Yang et al. (2024) — model and candidate-parent training
  context.
- Li et al. (2025), *Tracing the Representation Geometry ... from Pretraining to
  Post-training* — representation change across training stages.
- Guan et al. (2024), *Deliberative Alignment* — safety reasoning as an explicit
  post-training target.
- Arditi et al. (2024), Wollschläger et al. (2025), and Pan et al. (2025) — competing
  one-direction, cone, and multi-direction descriptions of refusal geometry.
- Minder et al. (2025) and Lindsey et al. (2024) — crosscoder model diffing and its
  artefacts.
- Nakamura (2026) — template-controlled difference-in-differences for activation shifts.
- Shuttleworth et al. (2024) — LoRA and full fine-tuning need not produce equivalent
  representations.

## Verified anchor records

The following records were checked directly against official abstract/proceedings pages:

- arXiv:2506.18167 — Venhoff et al., reasoning-behaviour steering
- arXiv:2505.22411 and NeurIPS 2025 proceedings — Huang et al., manifold steering
- arXiv:2510.09782 — Zhou et al., reasoning flows
- arXiv:2604.05655 — Sun et al., reasoning trajectories (ACL 2026)
- arXiv:2509.23024 — Li et al., pretraining-to-post-training geometry
- arXiv:2510.07364 — Venhoff et al., base models and reasoning deployment
- arXiv:2605.24583 — Nakamura, template-controlled activation shifts
- arXiv:2603.12277 — Ye et al., role confusion
- ACL 2024 long paper 828 — Rimsky et al., contrastive activation addition
- ACL/TACL 2021 paper 1.10 — Elazar et al., amnesic probing
- EMNLP 2019 paper D19-1275 — Hewitt and Liang, probe controls
- EMNLP 2023 main paper 762 — Cheng et al., geometric compression
- ICML 2020 PMLR 119:6713–6723 — Mamou et al., language manifolds
- NeurIPS 2019 paper cfcce062... — Ansuini et al., intrinsic dimension
- ICML 2024 PMLR 235:39643–39666 — Park et al., linear representations
- NeurIPS 2024 paper f545448... — Arditi et al., refusal direction
- ICML 2025 PMLR 267:47697–47716 — Pan et al., orthogonal safety directions
- Nature 645:633–638 — DeepSeek-R1

## Review boundary

“Exhaustive” in the thesis means exhaustive across the defined conceptual boundary and
the principal primary-source venues above, not an assertion that every paper containing
one of the broad keywords has been enumerated. The final text reports the search date,
scope, inclusion criteria, and publication status so an examiner can evaluate that
boundary.
