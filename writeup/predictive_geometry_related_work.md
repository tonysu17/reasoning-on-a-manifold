# Predictive Geometry of Reasoning — Related Work & Novelty Verdict

*Literature-verification pass, 2026-06-15. Every claim below was checked against a
PRIMARY source (arXiv abstract/HTML/PDF or OpenReview). Quotes are <=15 words.
Items that could not be verified against a primary source are listed in §6.*

**Our novelty atom (the thing we must protect):** *the **residual of a LEARNED forward
predictor over reasoning STEPS**, read out as a **correctness / branch-point** signal —
i.e. where the predictor breaks localizes backtracking, and the geometry of those breaks
separates correct from incorrect chains, under chain-grouped nulls.*

---

## 1. NOVELTY VERDICT (verified)

**The atom survives, but it MUST be narrowed.** The framing in `PREDICTIVE_GEOMETRY.md` §5 —
*"No prior work trains a next-step predictor over reasoning steps and reads its error as the
diagnostic"* — is **too strong as written** and should be retired. One paper (PHi,
2503.13431) already inserts a learned hidden-state predictor and reports that its
prediction-error magnitude correlates with chain correctness; and SSP/STP (2604.18464)
already trains a learned next-*step* predictor over reasoning trajectories (though it reads
*smoothness*, not the residual-as-classifier, and finds smoothness does **not** predict
correctness). So "first to read a learned predictor's error against correctness" is **not**
defensible.

**What IS still defensible (the narrowed atom):** the diagnostic unit and reading are novel
in combination. Specifically: (i) a forward predictor over the **sparse, discrete sequence
of behaviour-segmented reasoning STEPS** (not per-token residual-stream prediction as in
PHi), whose (ii) **residual GEOMETRY over time** — magnitude/growth/direction-churn and the
**location of residual spikes** — is used to **localize branch-points/backtracking and
classify chains**, (iii) under explicit **chain-grouped label-permutation and within-chain
step-shuffle nulls**. No verified paper does this combination: PHi reads an aggregate
per-token *complexity* scalar (and finds high unpredictability → *correct*, the opposite
polarity to a naive "breaks = bad" story, which we must engage); SSP/STP reads smoothness and
explicitly *declines* to build the feedback loop and finds AUC ~= 0.5; 2604.05655 hits AUC
0.87 but from a **static activation-difference probe**, not a predictor residual; ASM
computes a predict-correct error but **spends it on steering, not diagnosis**.

**Net:** keep the project; rewrite the novelty sentence to claim *step-level residual-geometry
as a branch-point / correctness signal under chain-grouped nulls*, and explicitly position
against PHi (polarity + granularity), SSP/STP (smoothness vs residual-structure; their
negative is our motivation), 2604.05655 (static probe vs learned residual), and ASM (control
vs readout). H4 (steer-along-prediction) remains *confirmation, not novelty* — ASM owns it.

---

## 2. LOCKED RELATED-WORK TABLE

`arXiv id | one-line | how WE differ`

### 2a. Nearest neighbours to the atom (cite + distinguish explicitly)

| id | one-line (verified) | how we differ |
|---|---|---|
| **2503.13431** (PHi) | Per-token "prediction of hidden states" bottleneck in the residual stream; its unpredictability measures in-context computation complexity and correlates with reasoning-chain correctness. | PHi predicts the **next token's** hidden state and reads an **aggregate complexity scalar**; we predict the **next discrete reasoning-STEP** embedding and read **residual GEOMETRY (spike location) as branch-points + chain classifier** under chain-grouped nulls. PHi finds high unpredictability => **correct** (effort proxy); we must test residual-*structure*, not aggregate magnitude, and engage that polarity. |
| **2604.18464** (SSP / Semantic Step Prediction) | Trains a (linear + 3-layer-MLP) next-step latent predictor over reasoning steps; finds shaped trajectories are smooth, and that **smoothness != correctness** (binary AUC ~= 0.5); declines to build a latent-reasoning feedback loop. | Same apparatus (learned step predictor), **different functional**: they read *predictability/smoothness*; we read **residual structure as a correctness/branch signal**. Their negative result is **exactly the gap we target**; their declined feedback loop is our Rung 3. |
| **2604.05655** (LLM Reasoning as Trajectories) | Logistic-regression on **late-step activation-difference / transition features** predicts final-answer correctness, ROC-AUC up to 0.87 (layer 29). | **Static** activation-difference probe with **no forward predictor and no residual**; we use a *learned predictor's residual geometry*. This is the AUC number our atom must beat / complement, with a different (dynamical) quantity. |
| **p17En1bhCY** (ASM, ICLR 2026 OpenReview) | Activation State Machine: a control-theoretic predict-correct loop that predicts an ideal hidden state, observes the activation, and **corrects** it to steer reasoning. | ASM computes a prediction error but **spends it on causal steering (control)**; we read the residual as a **diagnostic**. This is our H4 ("confirmation, not novelty") — ASM owns predicted-direction steering. |

### 2b. Trajectory-correctness neighbours (different mechanism)

| id | one-line (verified) | how we differ |
|---|---|---|
| **2510.10494** (Tracing the Traces) | **Hand-crafted** latent-trajectory features (start-to-end change, accumulated change, progress-to-final) predict solution accuracy for test-time selection. | No learned predictor / no residual; features are descriptive geometry. We learn a forward model and read **its error**, and target branch-point localization, not answer selection. |
| **2511.14773** (Temporal Predictors of Outcome) | **Linear probes** on frozen hidden states after t tokens show correctness is decodable very early; later-step drop is a length selection artifact. | Static probe of hidden states, not a forward-predictor residual; their length/selection-artifact finding is a **confound warning we must honor** (CF-2/CF-8). |
| **2509.22518** (REMA) | Failure severity = **kNN distance** of an erroneous representation to a manifold of **correct** representations (static geometric deviation). | Static manifold-distance probe (manifold framing we do not claim); we use a learned predictor's temporal residual, not distance-to-correct-cloud. |
| **2601.02170** (Streaming Hallucination Detection) | Treats step-level hallucination judgments as local observations + a cumulative prefix-level latent state for real-time detection. | Uses judged step-level labels + cumulative signal, **not** a learned next-step predictor residual. |
| **2601.17467** (ARS) | Hallucination detection via **answer-agreement representation shaping** with counterfactual latent interventions. | Counterfactual-intervention representation shaping, not predictor residual geometry. |

### 2c. Method anchors (apparatus we build on — not competitors)

| id | one-line (verified) | role for us |
|---|---|---|
| **2511.08544** (LeJEPA) | Introduces **SIGReg = Sketched Isotropic Gaussian Regularization** to drive JEPA embeddings to an isotropic Gaussian; provable, heuristic-free SSL (Balestriero & LeCun). | Rung-2 anti-collapse option; only *relative* claims off the regularised latent. |
| **2509.14252** (LLM-JEPA) | JEPA for LLMs; a special **[PRED] token** appended to the input yields the predicted-view embedding, reusing the LLM's own weights. | Rung-2 predictor trick ([PRED]-token over step views). |
| **2603.02765** (NE-Dreamer) | Decoder-free MBRL: a temporal transformer forecasts next encoder embeddings, aligned to a **stop-gradient target with a Barlow-Twins loss**. | Confirms "Barlow-Twins next-embedding" anchor; template for Rung-2 anti-collapse. |
| **2506.09985** (V-JEPA 2 / V-JEPA 2-AC) | Self-supervised video JEPA; the action-conditioned variant plans via **CEM** in a receding-horizon (MPC) loop over latent goal states. | Rung-3 apex: latent rollout + CEM/MPC planning template. |
| **2412.06769** (COCONUT) | Reasons in a **continuous latent space**, feeding the last hidden state back as the next input embedding (generative latent reasoning). | The "predict next step in latent space (generative)" line we explicitly do NOT claim. |
| **2510.09782** (Geometry of Reasoning: Flowing Logics) | Models reasoning as **flows** (position/velocity/curvature) in representation space; descriptive geometric framework. | The manifold/flow framing (~= our thesis title) we do NOT claim; descriptive, no predictor residual. |
| **2512.19171** (JEPA-Reasoner) | Decouples latent reasoning (JEPA engine) from a "Talker" decoder; predictor **is** the reasoning engine, L2-normed residual *connections*. | Generative latent-reasoning architecture (COCONUT-class); predictor-as-engine, not residual-as-diagnostic. |
| **2604.08065** (Pearl) | JEPA-inspired **multimodal tool-use**: SmoothL1 predictive-embedding alignment (stop-gradient target) to drop tool calls; **inference is standard decoding**, predictor error never read out. | See §3 — does NOT pre-empt the atom. |

---

## 3. PEARL (arXiv:2604.08065) — DEFINITIVE adjudication

**Verdict: Pearl does NOT pre-empt our atom.** Verified against the full PDF text.

- It is a **multimodal VLM tool-use method**, not a reasoning-correctness diagnostic:
  Pearl = "Predictive Embedding Alignment for Reasoning in Latent space", a JEPA-inspired
  framework that learns from tool-use trajectories to *eliminate explicit tool calls*.
- **The predictor is a training-time objective only.** Loss (Eq. 1) is `D(h_R_hat, sg[h_R])`
  with **SmoothL1** distance to a **stop-gradient** target; an auxiliary next-latent term
  (Eq. 2) is the same SmoothL1-to-stop-gradient. The only collapse guard is stop-gradient
  (no SIGReg/VICReg/Barlow).
- **At inference the residual is never read out:** the overhead "applies only during
  training; inference remains identical to standard VLM decoding."
- **Zero diagnostic use of error.** Full-text search found no use of residual / prediction
  error / correctness / confidence / uncertainty / branch / backtrack as a *signal* (the
  only hits are "object detection" the tool and tool-call "errors").
- Pearl itself flags step-by-step latent reasoning and interpreting the learned latent as
  **future work** ("remains an open question").

How we differ, in one line: Pearl uses a predictor to *replace tool actions in a VLM*; we use
a predictor's *residual geometry over reasoning steps as a correctness/branch-point readout*.
Different modality, different role (training objective vs inference diagnostic), different
quantity (aligned embedding vs residual structure).

> Note: Pearl's next-latent term cites **Teoh et al. 2025** ("latent dynamics... belief
> states"). Unverified here (see §6); flagged as a possible deeper neighbour for the
> belief-state reading of step embeddings, but it is a representation-learning regularizer,
> not a correctness diagnostic.

---

## 4. POSITIONING STATEMENT (drop-in for the thesis intro)

> A recent line of work reads the *geometry of reasoning trajectories* for correctness: static
> activation-difference probes reach ROC-AUC ~0.87 mid-chain (Sun et al., 2026), hand-crafted
> trajectory features guide test-time selection (Vilas et al., 2025), early linear probes show
> correctness is decodable after a few tokens (David, 2025), and distance to a "correct
> manifold" grades failures (Li et al., 2025). A parallel line *learns* predictors over latent
> reasoning — generative continuous-thought decoders (Hao et al., 2024; Liu et al., 2025),
> JEPA-style alignment objectives (Huang et al., 2025; Adhikari & Lapata, 2026), and a
> per-token hidden-state bottleneck whose unpredictability tracks in-context computation
> *complexity* and correlates with chain correctness (Behrouz/Schmidhuber-style PHi, 2025).
> Yet the two lines stay apart: the predictability work that touches *reasoning steps* reads
> *smoothness* and finds it does **not** encode correctness (AUC ~= 0.5) while explicitly
> declining to close the feedback loop (Yuan, 2026), and the correctness work reads *static*
> or *hand-crafted* geometry rather than a forward model's error. We occupy the gap between
> them: we train a forward predictor over the sparse, behaviour-segmented sequence of
> reasoning steps and read the **geometry of its residual over time** — where the predictor
> breaks — as a *branch-point and correctness* signal, validated against chain-grouped
> label-permutation and within-chain step-shuffle nulls. Steering along the predicted
> direction (Li et al., ICLR 2026) is a downstream *confirmation*, not our claim; our atom is
> the residual as a **diagnostic of where reasoning forks**.

*(Author-year strings above must be reconciled with the BibTeX keys in §5; two are flagged
unverified in §6 — PHi author identity and Teoh et al.)*

---

## 5. BibTeX (verified authors/year/title unless flagged in §6)

```bibtex
@misc{adhikari2026pearl,
  title        = {Multimodal Latent Reasoning via Predictive Embeddings},
  author       = {Adhikari, Ashutosh and Lapata, Mirella},
  year         = {2026},
  eprint       = {2604.08065},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  note         = {Preprint, under review},
  url          = {https://arxiv.org/abs/2604.08065}
}

@misc{binhammer2025phi,
  title        = {Measuring In-Context Computation Complexity via Hidden State Prediction},
  author       = {Behrouz, Vincent and others},
  year         = {2025},
  eprint       = {2503.13431},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  note         = {AUTHOR STRING UNVERIFIED -- see section 6; confirm against PDF before citing},
  url          = {https://arxiv.org/abs/2503.13431}
}

@misc{yuan2026ssp,
  title        = {Semantic Step Prediction: Multi-Step Latent Forecasting in {LLM} Reasoning Trajectories via Step Sampling},
  author       = {Yuan, Yidi},
  year         = {2026},
  eprint       = {2604.18464},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2604.18464}
}

@misc{sun2026trajectories,
  title        = {{LLM} Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals},
  author       = {Sun, Lihao and Dong, Hang and Qiao, Bo and Lin, Qingwei and Zhang, Dongmei and Rajmohan, Saravan},
  year         = {2026},
  eprint       = {2604.05655},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2604.05655}
}

@inproceedings{li2026asm,
  title        = {Steering {LLMs}' Reasoning With Activation State Machines},
  author       = {Li, Ian and Chen, Philip and Huang, Max and D'Antoni, Loris and Yu, Rose},
  booktitle    = {International Conference on Learning Representations (ICLR)},
  year         = {2026},
  note         = {OpenReview id p17En1bhCY},
  url          = {https://openreview.net/forum?id=p17En1bhCY}
}

@misc{vilas2025tracing,
  title        = {Tracing the Traces: Latent Temporal Signals for Efficient and Accurate Reasoning},
  author       = {Vilas, Martina G. and Yousefi, Safoora and Nushi, Besmira and Horvitz, Eric and Balachandran, Vidhisha},
  year         = {2025},
  eprint       = {2510.10494},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2510.10494}
}

@misc{david2025temporal,
  title        = {Temporal Predictors of Outcome in Reasoning Language Models},
  author       = {David, Joey},
  year         = {2025},
  eprint       = {2511.14773},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2511.14773}
}

@misc{li2025rema,
  title        = {{REMA}: A Unified Reasoning Manifold Framework for Interpreting Large Language Models},
  author       = {Li, Bo and Deng, Guanzhi and Chen, Ronghao and Yue, Junrong and Zhang, Shuo and Zhao, Qinghua and Song, Linqi and Wen, Lijie},
  year         = {2025},
  eprint       = {2509.22518},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2509.22518}
}

@misc{lu2026streaming,
  title        = {Streaming Hallucination Detection in Long Chain-of-Thought Reasoning},
  author       = {Lu, Haolang and Pan, Minghui and Li, Ripeng and Nan, Guoshun and Zhuang, Jialin and Zhao, Zijie and Sun, Zhongxiang and Wang, Kun and Liu, Yang},
  year         = {2026},
  eprint       = {2601.02170},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2601.02170}
}

@misc{zhang2026ars,
  title        = {Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping},
  author       = {Zhang, Jianxiong and Guo, Bing and Jiang, Yuming and Wang, Haobo and An, Bo and Du, Sean},
  year         = {2026},
  eprint       = {2601.17467},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2601.17467}
}

@misc{balestriero2025lejepa,
  title        = {{LeJEPA}: Provable and Scalable Self-Supervised Learning Without the Heuristics},
  author       = {Balestriero, Randall and LeCun, Yann},
  year         = {2025},
  eprint       = {2511.08544},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2511.08544}
}

@misc{huang2025llmjepa,
  title        = {{LLM-JEPA}: Large Language Models Meet Joint Embedding Predictive Architectures},
  author       = {Huang, Hai and LeCun, Yann and Balestriero, Randall},
  year         = {2025},
  eprint       = {2509.14252},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2509.14252}
}

@misc{bredis2026nedreamer,
  title        = {Next Embedding Prediction Makes World Models Stronger},
  author       = {Bredis, George and Balagansky, Nikita and Gavrilov, Daniil and Rakhimov, Ruslan},
  year         = {2026},
  eprint       = {2603.02765},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2603.02765}
}

@article{assran2025vjepa2,
  title        = {{V-JEPA 2}: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author       = {Assran, Mido and Bardes, Adrien and Fan, David and Garrido, Quentin and Howes, Russell and others and Rabbat, Michael and Ballas, Nicolas},
  journal      = {arXiv preprint arXiv:2506.09985},
  year         = {2025},
  url          = {https://arxiv.org/abs/2506.09985}
}

@misc{zhou2025flowinglogics,
  title        = {The Geometry of Reasoning: Flowing Logics in Representation Space},
  author       = {Zhou, Yufa and Wang, Yixiao and Yin, Xunjian and Zhou, Shuyan and Zhang, Anru R.},
  year         = {2025},
  eprint       = {2510.09782},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2510.09782}
}

@article{hao2024coconut,
  title        = {Training Large Language Models to Reason in a Continuous Latent Space},
  author       = {Hao, Shibo and Sukhbaatar, Sainbayar and Su, DiJia and Li, Xian and Hu, Zhiting and Weston, Jason and Tian, Yuandong},
  journal      = {arXiv preprint arXiv:2412.06769},
  year         = {2024},
  url          = {https://arxiv.org/abs/2412.06769}
}

@misc{liu2026jepareasoner,
  title        = {{JEPA-Reasoner}: Decoupling Latent Reasoning from Token Generation},
  author       = {Liu, Bingyang and Chen, Ziyu and Woodruff, David P.},
  year         = {2026},
  eprint       = {2512.19171},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  note         = {arXiv id 2512.x indicates Dec 2025 submission; cite year per arXiv listing},
  url          = {https://arxiv.org/abs/2512.19171}
}
```

---

## 6. CLAIMS I COULD NOT VERIFY AGAINST A PRIMARY SOURCE

1. **PHi (2503.13431) author identity.** Verified: title, abstract, the residual-stream
   per-token bottleneck architecture, and the "high PHi loss => more likely correct" finding
   (all from the arXiv HTML). **NOT verified:** the author list — the BibTeX key/author
   string `Behrouz, Vincent and others` is a PLACEHOLDER and must be corrected from the PDF
   author block before citing. (Search heuristics suggested a Schmidhuber-lab/KAUST lineage
   but I did not confirm names against the primary source.)

2. **PHi correctness numbers.** The quoted "Answers with high PHi loss are clearly more
   likely to be correct" and the MATH partial-correlation r=0.079 came from the arXiv HTML
   body via WebFetch summarization, not a direct table read; treat the exact r as
   approximate pending a PDF check.

3. **Teoh et al. 2025** (cited inside Pearl's Eq. 2 as the next-latent / belief-state
   inspiration) was **not** independently located or read. Flagged as a possible deeper
   neighbour for the belief-state reading; verify before relying on it.

4. **V-JEPA 2 (2506.09985) full author list** — truncated with "and others"; the CEM/MPC
   planning claim is verified (direct quote: minimize via "the Cross-Entropy Method"), but
   complete the author list from the PDF.

5. **JEPA-Reasoner (2512.19171) year** — arXiv id prefix `2512` => December 2025 submission;
   I have written 2026 conditionally. Confirm the canonical year on the arXiv listing.

6. **2604.05655 mechanism quotes** ("logistic regression on concatenated late-step features",
   peak AUC 0.87 @ layer 29) came from the arXiv HTML via WebFetch; the headline claim
   (static activation-difference / step-transition features, NO forward predictor, AUC up to
   0.87) is solid, but exact layer/feature-construction details should be re-read from the PDF
   if quoted in the thesis.

All other entries (Pearl loss/inference, SSP "Smoothness != Correctness" + AUC~=0.5 + declined
feedback loop, ASM predict-correct-for-steering, LeJEPA SIGReg expansion, LLM-JEPA [PRED]
token, NE-Dreamer Barlow-Twins+stop-gradient, COCONUT continuous-thought, Flowing-Logics
descriptive framing, REMA kNN-to-correct-manifold) were verified against primary text and are
safe to cite as stated.
