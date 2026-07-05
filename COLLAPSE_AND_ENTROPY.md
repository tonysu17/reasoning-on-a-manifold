# Collapse & Entropy — steering-induced degeneration as representational collapse, and entropy injection as the remedy

> **Status: PROPOSAL + secondary analysis of the executed E8 run (2026-07-04).** No new
> generations were run for this document; §2 is a re-analysis of
> `results/eval/R1-1.5B__E1/steering_results.json` (scripts in session scratchpad, numbers
> reconciled against `generation_metrics.json` to 3 dp). Companion docs:
> [`METHODOLOGY.md`](METHODOLOGY.md) (how we measure), [`RESULTS_LEDGER.md`](RESULTS_LEDGER.md)
> (what stands), [`CONFOUNDS_AND_REMEDIATION.md`](CONFOUNDS_AND_REMEDIATION.md) (CF-19 registered
> from this analysis), [`PREDICTIVE_GEOMETRY.md`](PREDICTIVE_GEOMETRY.md) +
> [`THEORY.md`](THEORY.md) (the trajectory/JEPA apparatus §E9.3 builds on).

---

## 1. One paragraph

In the E8 steering run, ablating a behaviour's direction (`h' = h − α(rᵀh)r`, every token,
mid-layer) sometimes tipped generation into verbatim loop-to-cap repetition. The collapse is
**binary** (chains either stay clean or loop to the 8192 cap — the repetition distribution is
U-shaped), **arm-induced** (equal-k random-subspace ablations add ≈0 repetition; the behaviour's
own PCA-subspace direction roughly doubles the collapsed fraction), **behaviour-dependent**
(example-testing and uncertainty-estimation collapse-inflate; backtracking and adding-knowledge
ablation *reduce* repetition), and **task-gated** (only open-ended prompts collapse; vanilla
greedy already collapses on 34%). We propose to interpret this as **representational (geometric
mode) collapse** — inference-time confinement of the hidden-state trajectory after removal of a
high-variance, load-bearing direction — *not* as "model collapse" simpliciter (a training-time
term with 8 conflicting definitions), and to test an **entropy-injection** remedy at three
scales: decoding entropy (temperature/min-p), state-space entropy (structured residual-stream
noise), and trajectory entropy (perturbation localized at branch points found by the
predictive-geometry residual). The one clean steering result (backtracking) is the arm that does
**not** collapse, so the headline survives; the collapse phenomenon is a *new, separately
publishable object*, and it supplies the mechanistic reading of *why* the two null arms were
null (their "suppression" was partly generation damage — already de-confounded away in
`strengthen_report.json`).

---

## 2. What actually happened in E8 (secondary analysis, 2026-07-04)

Mechanics recap (from `src/steered_inference.py:272-297`): every steered arm **subtracts** the
projection of the hidden state onto a **unit** vector at one mid layer (bt L17 / unc L15 /
ex L15 / ak L17), at **every position including prefill**, α=1, greedy, `max_new_tokens` 8192.
"Manifold_k" = the same diff-of-means direction first projected onto the behaviour's top-k PCA
subspace and renormalized — same mechanism, purified direction.

### 2.1 Collapse is bimodal and equals loop-to-cap

Per-chain 4-gram repetition (`src/evaluation.py:36-48`) is **U-shaped**: e.g. example-testing
manifold_k5 histogram (0.1 bins) = `4 3 0 0 1 1 3 6 15 17`. Repetition and length are
near-collinear (r ≈ 0.87–0.94 in every arm); non-cap chains are uniformly clean (rep 0.09–0.30).
So the operative event is **"entered a loop and ran to the cap"**, and `degenerate_rate`
(<32 tokens) is blind to it — collapse produces *long* chains, never short ones (why it reads
0.0 everywhere).

| behaviour · arm | mean rep | collapsed (rep>0.8) | cap-hit |
|---|--:|--:|--:|
| ex-test · **manifold_k5** | **0.737** | **64%** | 86% |
| ex-test · single | 0.643 | 48% | 74% |
| ex-test · random_subspace_k5 | 0.534 | 36% | 61% |
| ex-test · energy_matched_random | 0.561 | 38% | 66% |
| uncertainty · **manifold_k3** | **0.747** | **68%** | 84% |
| uncertainty · random_subspace_k3 | 0.516 | 35% | 61% |
| backtracking · **manifold_k5** | **0.438** | 24% | 40% |
| adding-knowledge · single | 0.403 | 20% | 34% |
| **vanilla (shared)** | 0.551 | 34% | 68% |

### 2.2 Arm-induced, subspace-specific, directional

Paired per-task Δrep vs vanilla: ex-test manifold_k5 **+0.186** (its rank-matched floor:
−0.017); uncertainty manifold_k3 **+0.195** (floor −0.035); backtracking manifold_k5 **−0.113**;
adding-knowledge single **−0.148**. McNemar-style collapse transitions vs vanilla: ex-test k5
arm-only 17 vs vanilla-only 2; unc k3 arm-only 20 vs vanilla-only 3. Two task populations:
task-hard (LATE_*, CREA_098 — collapse under everything incl. vanilla) and **arm-induced**
(SCIE_103/104/105, VERB_103 — vanilla 0.05–0.39 → 0.93+ under the manifold arm). The equal-k
**random**-subspace floors do not inflate repetition at all ⇒ the excess is specific to the
behaviour's own subspace, not to "any k-dim ablation" and not to injected energy
(energy-matched floor stays low).

### 2.3 The loop itself, and the backtracking inversion

Collapsed chains — steered *and* vanilla — oscillate between two paraphrases of one
hypothesis-revision step ("*Wait, maybe the problem is… / Wait, perhaps the problem is…*",
"*…but that's probably not feasible. Alternatively, maybe…*") to the cap; 0–1 of ~33 collapsed
chains produce `\boxed{}`. The loop's shape is **identical in vanilla and steered chains**:
steering changes *how often* the model falls in, not the shape of the pit. And the loop is
lexically a **backtracking/uncertainty cycle** — which makes the inversion coherent rather than
paradoxical: **ablating backtracking removes the loop's re-entry move** (chains get shorter,
cleaner, terminate in answers), while **ablating example-testing/uncertainty removes the
resolve-and-exit machinery** (test the hypothesis, commit a confidence judgement) that lets an
open-ended deliberation terminate.

### 2.4 The variance clue

`steering_geometry.json` energy multipliers (how much a random unit direction must be scaled to
match the behaviour direction's mean |rᵀh|): **ex-test 7.73, uncertainty 6.12** vs backtracking
2.06, adding-knowledge 2.37. The two collapse-prone behaviours are exactly the two whose
directions carry ~3× more projection energy — i.e. whose ablation removes the most
**state-space variance**. And the manifold arms (direction purified into the behaviour subspace)
collapse *more* than the raw single direction. Both facts point the same way: **the more
variance you remove along a load-bearing behavioural direction, the likelier the trajectory
falls into the repetition attractor.**

### 2.5 What this does NOT change

The E8 headline is untouched — it was already de-confounded. `strengthen_report.json` re-read
every effect on collapse-immune endpoints (absolute count, per-1k-tokens, non-degenerate
subset): **backtracking survives all three (and collapses *less* than its floor — the confound
acts against it); example-testing and uncertainty do not survive** (their apparent
"suppression" was partly this very collapse diluting behaviour sentences). This document is the
*mechanistic story of the artefact*, not a revision of the verdict. Standing caveats inherit:
within-annotator, α=1 only, greedy only, single sample, n=50.

---

## 3. Framing: what to call it (and what not to)

**The claim we can defend.** The intervention performs a *surgical support contraction in
activation space* — it deletes, at every token, the component of the state along one direction
the model actively uses. The generation then exhibits the canonical collapse signature:
diversity → 0, convergence to a short limit cycle. The **mechanism** shared with the collapse
literatures is *variance/support/tail contraction plus self-reinforcing feedback*:

- Training-time **model collapse**: early collapse = tail loss, late = low-variance point mass
  (Shumailov et al., Nature 631:755, 2024 — whose own repetition ablation shows loops as the
  symptom); collapse as tail-truncation rewriting scaling laws (Dohmatob et al., ICML 2024,
  arXiv:2402.07043); MAD — sampling bias onto a high-density subset trades diversity for
  quality (Alemohammad et al., ICLR 2024, arXiv:2307.01850); collapse as
  generalization→memorization under declining entropy (Shi et al., NeurIPS 2025,
  arXiv:2509.16499).
- **Inference-time, representational versions already exist** — we do not need the analogy to
  carry causal weight: generation mode collapse as *"trajectory confined to a low-dimensional
  region of representation space"*, fixed by low-rank geometric damping (Du & Tanaka-Ishii,
  ICML 2026, arXiv:2605.00435 — the paper to build on); **"state collapse" preceding textual
  repetition in R1-style reasoners** (Duan et al. 2026, arXiv:2601.05693 — our model class);
  the loop regime is linearly decodable from hidden states (Xie et al., EMNLP 2025,
  arXiv:2511.00536); repetition as self-reinforcing feedback (Xu et al., NeurIPS 2022,
  arXiv:2206.02369); rank collapse / token uniformity as the structural substrate (Dong et
  al., ICML 2021, arXiv:2103.03404).
- **The model was already diversity-narrowed before we touched it**: RLVR sharpens but narrows
  support (Yue et al., NeurIPS 2025, arXiv:2504.13837); policy entropy collapses during
  reasoning-RL and entropy-preserving fixes reverse it (Cui et al. 2025, arXiv:2505.22617);
  R1-distill's vendor documents endless-repetition under greedy and recommends T 0.5–0.7
  (arXiv:2501.12948).

**The three-sinks (entropy budget) picture.** Three independent entropy sinks stack in E8:
(1) RLVR post-training already contracted the policy's support; (2) greedy decoding removes
*all* sampling entropy — one bad argmax locks the loop (inconsistency of greedy decoding:
Welleck et al., EMNLP 2020, arXiv:2002.02492); (3) our ablation removes state-space variance
along a load-bearing direction. Collapse happens when the joint budget crosses a threshold —
which is why it is task-gated (open-ended prompts sit nearer the attractor basin) and why
vanilla alone already collapses at 34%. This is directly testable (§5, E9.1: iso-collapse
contours in the α × decoding-entropy plane).

**Terminology discipline.** Do NOT headline the phrase "model collapse": it names a
training-time recursive-retraining phenomenon, has 8 conflicting definitions in the wild
(Schaeffer et al., arXiv:2503.03150), and invites the objection that our causal engine (a
single fixed intervention) has no resampling loop. Say **"representational collapse" /
"geometric mode collapse"** (Du & Tanaka-Ishii) or **"steering-induced degeneration"**, and
cite training-time model collapse explicitly *as the mechanism-level analogy* (variance
contraction + positive feedback), via Bertrand et al. (ICLR 2024, arXiv:2310.00429) if
importing attractor/contraction formalism.

**Relation to the user-framing "stuck in / projected onto a manifold."** Two corrections that
make the intuition *stronger*, not weaker: (i) the E8 mechanism is **subtractive** — the
trajectory is not being pushed *onto* the behaviour manifold, it is having a load-bearing
direction *deleted*, which confines the reachable state distribution to a lower-variance slice;
confinement is the right concept, the manifold in question is the *state distribution's
support*, not the behaviour subspace itself; (ii) the collapse destination (the two-phrase limit
cycle) is the model's own pre-existing attractor, not a property of the steering subspace — the
intervention moves the **basin boundary**, the attractor was always there. Framed this way the
observation and the geometry chapters compose: reasoning lives on low-dimensional structure
(thesis Movement 1), and *forcibly shrinking the live dimensionality below what the computation
needs* produces degenerate dynamics — low-dimensionality is a description of healthy reasoning,
not a safe operating regime to clamp into.

---

## 4. Competing explanations (all must be addressed before publishing)

| # | Hypothesis | Prediction that separates it | Test |
|---|---|---|---|
| H-A | **Variance-removal / confinement** (our lead): ablating a high-variance behavioural direction contracts state support; trajectory falls into the pre-existing loop attractor | Collapse propensity scales with ablated variance (energy multiplier: ex 7.7 / unc 6.1 vs bt 2.1 / ak 2.4 — already consistent); dose-response in α; rescued by restoring entropy anywhere in the stack | E9.1 (α × T factorial), E9.2 (noise injection) |
| H-B | **Vector contamination**: the steering vectors have incidental projection on a "repetition feature/direction" (SAE repetition features: Yao et al., ACL-Findings 2025, arXiv:2504.14218; repetition neurons: arXiv:2410.13497), so looping is an off-target artefact of imperfect vectors | cos(steering vector, loop direction) predicts per-arm collapse; projecting the loop direction OUT of the vector removes the excess collapse without removing on-target effect | E9.0a (loop-probe direction from our own corpus; cosine table; cleaned-vector re-run on a 10-task subset) |
| H-C | **Off-distribution fallback**: any sufficiently disruptive intervention degrades to the model's most primitive fallback = repetition (Ivgi et al., arXiv:2407.06071); nothing specific to behaviour subspaces | Matched-disruption controls should collapse equally — **already partially refuted**: equal-k random subspaces and energy-matched random stay at floor | Strengthen with KL-matched (not just energy-matched) floor in E9.1 |
| H-D | **Functional-role story** (rides on H-A): the loop is a backtracking/uncertainty cycle; ablating exit-machinery (ex-test/unc) traps it, ablating re-entry (backtracking) drains it | Loop-span activations should load on backtracking/uncertainty subspaces; per-behaviour sign of Δcollapse follows the behaviour's role in the loop | E9.0a/b (loop-span geometry vs behaviour subspaces) |

CF-8 kinship note: the corpus itself has 50% cap-hit chains — the same attractor contaminates
the *observational* data upstream. The loop probe from E9.0 doubles as a corpus-hygiene tool
(flag/loop-trim chains before geometry).

---

## 4b. Terminology discipline: "localising" vs "inducing" (2026-07-05)

Two verbs were doing too much work in discussion; fixed senses, binding for all write-ups.

**Localising** — three distinct claims, never conflated:
1. **In depth** (which layer): concentration (PR-trough 16/16/16/12 bt/unc/ak/ex) ≠
   specificity (bt/unc all layers; ex L27 only; ak nowhere) ≠ causal purchase (steered at
   bt17/unc15/ex15/ak17). Example-testing splits three ways (12 / 27 / 15) — always name
   which sense is meant.
2. **In direction** (which subspace): diff-of-means vector / top-k PCA; adjudicated by the
   chain-stratified specificity null + the steering floor.
3. **In time** (where along the trajectory): loop onset (median 25% into chain, detector);
   forks/residual spikes (predictive geometry; E9.3 — NOT yet run).

**Inducing** — the crucial asymmetry: **no behaviour has ever been induced.** All executed
causal cells are subtract-mode (suppression/ablation); the amplify (+) arm is protocol-defined
with zero cells run. What HAS been induced causally is **collapse** (damage induction via
ex-test/unc subspace ablation, beyond matched floors) — never say "we can control behaviour X"
(control implies both signs); planned inductions: entropy (E9.2/E9.3).

**Per-behaviour verdict ledger** (all numbers in §2 / RESULTS_LEDGER §B/B3):

| Behaviour | Localised? | Suppression (subtract) | Collapse role when ablated | Loop-region loading (meandiff) |
|---|---|---|---|---|
| backtracking | ✅ direction+depth (spec. all layers; L17) | ✅ Δ_floor +0.054/+0.070 (Holm .002), de-confound-robust, −49/−64% rel., d_z .46/.64; entangled w/ unc (0.881 vs 1.093 per-1k; abs p=.06); k3 arm fails floor | REDUCES (k5 24%, Δrep −0.11; k3 anomaly 50% — worse direction estimate, matches its floor fail) | **+0.34** (loop partly made of it → ablation drains re-entry) |
| uncertainty-est. | ✅ direction+depth (spec. all layers; L15) | ❓ UNDETERMINED (sd 1.7×, power .10–.13 @Δ=.05); vector not own-specific (bt 0.523 > own 0.320) | WORST INFLATOR (k3 68%, +0.195, 20 net new) | **+0.47** (largest) |
| example-testing | 🟧 partial (spec. L27 only; conc. 12; steered 15) | ❌ ARTEFACT (fraction +0.060 = dilution; count p=.27, per-1k p=.10) | INFLATES at every arm (48/54/64%); dose-response persists at T=0.6 (E9.1 prelim) | **−0.25** (loops starved of it → ablation removes the testing exit) |
| adding-knowledge | ❌ (spec. p=1.0 everywhere) | ❌ NULL as pre-registered; vector barely moves own behaviour (0.023 per-1k) | REDUCES (single 20%, −0.148) — but NOT evidence about the behaviour (direction isn't behaviour-specific) | **−0.75** (strongest negative) |

One-liners: localised = bt, unc; partially = ex; not = ak. Suppressed = bt only;
undetermined = unc; artefact = ex; null = ak. Collapse-inducing ablations = ex, unc;
collapse-reducing = bt, ak-vector. Induced (amplify) = nothing yet (open cell).

---

## 5. Proposed experiments — the E9 ladder (gated, cheap-first)

**E9.0 — Loop geometry on existing data — ✅ EXECUTED 2026-07-04/05** (runner
`18_loop_geometry.py` + `src/loop_geometry.py`, 20 tests green; RunPod 4090 ~10 min after the
`use_cache=False` + Gram-trick-PR fixes; results `results/loop_geometry/R1-1.5B/REPORT.md`).
Corpus detector: **415 loop / 469 clean / 116 ambiguous** of 1000 chains (periodic-tail
detector, median period 117 words, onset at median 25% of chain). **(a) Probe + gate: H-B
REJECTED** — chain-grouped in/out-of-loop probe reaches token OOF AUC **0.994** (chain-level
1.0; loop regime linearly separable — Xie 2511.00536 replicated on R1-distill), and
|cos(probe direction, steering vector)| ≤ **0.13** for all 12 vectors, under the
pre-registered 0.3 gate ⇒ **no cleaned-vector arm needed in E9.1**. The class-MEANDIFF loop
axis shows the H-D sign structure: loop states displaced TOWARD backtracking (+0.34) and
uncertainty (+0.47), AWAY from ex-test (−0.25) and add-know (−0.75) (own-layer, averaged over
arms) — activation-level support for the degenerate-deliberation reading (correlational).
**(b) Precedence: METRIC-DEPENDENT** — windowed-PR contraction pre-onset is large (−3.0,
Wilcoxon p≈1e-36) but SMALLER than the matched-position decline in clean chains (−4.6;
one-sided control p≈1.0) ⇒ PR "precedence" is generic positional decline, not a loop
signature (the matched-position control most papers don't run kills it); token-uniformity
precedence DOES clear the control at L17 (+0.037 vs +0.018, MW p=8e-4; weak L16 p=0.043,
absent L15) ⇒ Duan-style state-before-text survives only as uniformity-at-L17. Folded into
thesis `steering.tex` rung-one paragraph 2026-07-05.

**E9.1 — Dose-response × decoding-entropy factorial — 🚀 LAUNCHED 2026-07-05** (generation
running on RunPod; declared deltas from the spec below: (i) **min-p arm deferred** (owed;
greedy + T=0.6 only), (ii) **greedy α=1.0 cells reused from E8** — same model, vectors dir,
eval split, arms, decoding, so the merge is exact; new greedy generation covers α∈{0.5,1.5}
only, (iii) rs replicates = 2 not 3, (iv) batched sampling added to `steered_inference`
(batch-level seeding per sample-index; per-chain T>0 generation was the wall-clock
bottleneck — see `_generate_batch_impl` docstring; E9.0's gate result means no
cleaned-vector arm). Analysis: `e9_1_analysis.py` → `results/eval/E9_1_ANALYSIS.md`.
Original spec: 2 behaviours only (ex-test = collapser, backtracking = anti-collapser) × arms
{vanilla, single, manifold_k5, random_subspace_k5, energy_matched} × α ∈ {0.5, 1.0, 1.5} ×
decoding ∈ {greedy, T=0.6 (vendor-recommended), min-p} × 50 tasks × 3 samples at T>0.
Annotation-free endpoints: collapse rate (loop-to-cap), per-chain repetition, length,
task-accuracy guard (`\boxed` grading on structured tasks — closes the owed guard from
`steering.tex:253-255`), **cross-sample diversity** (distinct-n / self-BLEU across the 3
samples — the entropy endpoint the run never had). Annotated endpoint (budget-gated, non-builder
annotator if the band is funded): does backtracking Δ_floor survive at T=0.6? Pre-registered
predictions: (P1) collapse rises with α for ex-test manifold, flat for floors (H-A
dose-response); (P2) T=0.6 rescues ≥ half of steering-induced collapses (entropy injection at
the output compensates variance removal in the state); (P3) backtracking suppression persists
at T>0 (the effect is not a greedy artefact) — P3 failing would *demote the E8 headline*, so
this doubles as the strongest robustness check available for the thesis; (P4) iso-collapse
contours: α needed to collapse falls as decoding entropy falls (the budget picture). Also
delivers the α-sweep + Pareto frontier already owed (`PLAN_EXPERIMENTS.md` E6/E7,
`steering.tex:494-513`).

**E9.1b — Amplify arm: the sign/parity test — 📋 PRE-REGISTERED 2026-07-05 (before any
amplify cell has ever been generated; Tony-approved).** Everything causal so far is
SUPPRESSION (subtract mode, §4b); E9.1b runs the SAME arms in **add mode**
(h' = h + α(rᵀh)r; engine support existed, plumbing + `--steer-mode add` added 2026-07-05,
mode recorded per row). Design: {backtracking, example-testing} × arms {single, manifold_k5,
random_subspace_k5 ×2, energy_matched} × α ∈ {0.5, 1.0} × greedy × 50 tasks × 1 sample
(~1,050 chains incl. fresh vanilla; the greedy regime is where the attractor has headroom in
BOTH directions: vanilla 34%). **The headline is response PARITY in the intervention sign,
per behaviour** — the single cleanest discriminator between the functional story (H-D) and
generic-damage stories: H-D predicts an ODD response (backtracking: subtract ↓collapse /
add ↑collapse — amplifying the re-entry move feeds the cycle; ex-test: subtract ↑ / add ↓ —
amplifying the exit machinery resolves deliberation), while any
perturbation-is-perturbation account predicts an EVEN response (both signs damage).
Pre-registered predictions: **(P5)** amplifying backtracking RAISES collapse vs vanilla and
vs the add-mode floors (falsifier: unchanged/decreased); **(P6)** amplifying ex-test LOWERS
collapse below vanilla's 34% (falsifier: increase ⇒ even parity ⇒ damage account);
**(P7)** add-mode floors stay ≈ vanilla (as their subtract twins did); **(P8)** |Δcollapse|
grows with α on behaviour arms only. Caveats sealed with it: amplification inflates
activation norms (Householder critique), which the energy-matched add-floor controls at
matched added energy; the behaviour-frequency (on-target induction) half needs annotation
and is budget-gated exactly like P3; α capped at 1.0 (at α=1 add doubles the component —
larger doses risk trivial norm blow-up, reserved for a later sweep). Endpoints + analysis:
same annotation-free battery via `e9_1_analysis.py` extension. Runs gated on the E9.1
greedy leg finishing (same pod, queued launcher).

**E9.2 — Structured state-entropy injection (same harness, new hook mode; ~$10).** Replace
subtract with **add-noise**: ε ~ N(0, σ²) per token, three geometries at matched energy —
isotropic (full 1536-D), within-behaviour-subspace (P_k ε), orthogonal-complement — plus a
front-loaded schedule (σ decaying along the chain, per RSP arXiv:2605.11936). Question: which
noise geometry restores diversity (cross-sample distinct-n ↑, collapse ↓) without coherence
cost (accuracy guard flat)? The world-model-geometry repo's finding (isotropizing destroys kNN
structure while linear probes stay flat) predicts **naive isotropic injection damages
structure**; the interesting arm is noise *within* the low-dim reasoning-relevant subspaces.
Either outcome is a result: "entropy must be structured" or "any entropy works" both close the
naive-remedy question the collapse literature leaves open (uniform entropy bonuses are blunt —
SIREN arXiv:2509.25133, AER arXiv:2510.10959 make the same point at training time).

**E9.3 — Trajectory-localized entropy injection at branch points (the creativity arm; gated on
E9.0 + Rung-1 residual).** The predictive-geometry pilot already localizes *unpredictable*
steps (residual spikes of the Rung-1 ridge predictor; `PREDICTIVE_GEOMETRY.md`). Inject
entropy **only there** — (i) burst-temperature for the next sentence, (ii) state-noise burst
(E9.2 hook, gated), (iii) textual "Wait" injection as the s1-style baseline
(arXiv:2501.19393) — vs uniform injection at **matched total entropy** and vs
self-consistency (parallel sampling, arXiv:2203.11171) at matched compute. Endpoints: strategy
diversity across samples (LLM-judged distinct solution approaches — non-builder judge),
pass@k on gradeable tasks, collapse rate. Hypothesis: **fork-localized injection dominates
uniform injection at matched entropy budget** (inference-time analogue of the 80/20
high-entropy-fork result, arXiv:2506.01939; fork detection cross-checkable against token
entropy — Bigelow et al., ICLR 2025, arXiv:2412.07961). This is the piece that operationalizes
"perturb the path to induce creativity": deterministic reasoning + targeted trajectory
perturbation, with the perturbation sites *chosen by the geometry* rather than by decoding
heuristics. It is also H4 of `PREDICTIVE_GEOMETRY.md` grown into a full experiment — the
residual isn't just a diagnostic, it is the *address* for intervention.

**Publishability assessment.** (1) E9.0b alone = a tight replication-with-mechanism note
("state collapse precedes textual repetition in R1-distill; loop region is linearly separable;
behaviour-vector geometry predicts basin shifts"). (2) E9.0+E9.1 = the workshop/short-paper
package: *"Ablating behavioural directions tips reasoning models into their repetition
attractor: collapse is subspace-specific, dose-dependent, and rescued by decoding entropy."*
Novel vs Du & Tanaka-Ishii (they regulate generic mode collapse; we show *which* semantic
directions are load-bearing and that collapse is behaviour-sign-dependent — the backtracking
inversion is the memorable finding). (3) E9.3 = the main-track swing: geometry-guided,
trajectory-localized entropy injection for reasoning diversity, benchmarked against
self-consistency at matched compute. Complementary, not redundant, with the thesis: Movement 1
says reasoning is low-dimensional; this says *what happens when you squeeze it below its
operating dimensionality, and how controlled re-expansion buys diversity*.

---

## 6. Thesis integration (v2 build; exact slots, with guards)

Integration points (ranked; see the 2026-07-04 mapping session for line refs):

1. **`conclusion.tex:110-119`** (RSI/interpolation future-work ¶): one connective move — the
   steering-induced collapse is the *empirically observed face* of "sharpening within the
   installed subspaces"; entropy injection is the constructive counterpart ("opening variance
   off" them). This is also where the **Funes/compression segue** lives if wanted: pretraining
   memorizes because it can; bounded agents abstract because they must (Borges' Funes as the
   limit case — cite as essay-tradition, not literature: Tirado 2020, Buckman & Gelada 2022;
   the rigorous neighbours are prediction≡compression, Delétang et al. ICLR 2024
   arXiv:2309.10668, and JEPA's discard-to-predict, LeCun 2022 OpenReview BZ5a1r-kVsf; the
   collapse bridge is Shi et al.'s generalization→memorization-under-declining-entropy,
   arXiv:2509.16499).
2. **`steering.tex` §"What this test cannot yet decide" / end of status §**: a future-work ¶
   for E9.1 (the α × decoding-entropy factorial subsumes the owed α-sweep), plus ONE
   connective sentence in the degeneration ¶ (`:344-360`) naming the loop-to-cap bimodality
   and pointing at the collapse framing. Do not rewrite the dilution-artefact analysis — it is
   correct and load-bearing.
3. **`METHODOLOGY.md` §4 metrics** (or its 06-25 refinement doc): add collapse-aware metrics —
   collapse rate (loop-to-cap), windowed state-rank series, cross-sample distinct-n — and note
   `degenerate_rate`'s blindness (catches short chains; collapse makes long ones).
4. **CONFOUNDS_AND_REMEDIATION.md**: CF-19 registered (2026-07-04) — see register.

**Guards (must not be violated by any new prose):**
- The **manifold-null explanation stays geometric** (single↔k5 cos 0.965 + single dose) — do
  not re-explain it as collapse.
- **No blanket "steering induces degeneration"** — the clean behaviour degenerates *less* than
  its floor; the collapse claim is arm- and behaviour-specific.
- **No dose-response claim yet** (α=1 only), **no cross-sample diversity claim yet** (greedy,
  1 sample) — both are E9.1 deliverables, currently future work.
- **No curvature resurrection** via collapse language (subspace-relative wording only), and
  every new result inherits the within-annotator hedge.

---

## 7. Verified citation shortlist (load-bearing ten)

1. Du & Tanaka-Ishii, *Escaping Mode Collapse in LLM Generation via Geometric Regulation*, ICML 2026, arXiv:2605.00435 — inference-time geometric collapse + low-rank remedy.
2. Shumailov et al., *AI models collapse when trained on recursively generated data*, Nature 631:755 (2024) — tail-loss→variance-collapse definition; repetition ablation.
3. Duan et al., *Circular Reasoning: Self-Reinforcing Loops in LRMs*, arXiv:2601.05693 — state collapse precedes textual repetition, R1-class.
4. Xu et al., *Learning to Break the Loop*, NeurIPS 2022, arXiv:2206.02369 — repetition self-reinforcement.
5. Dong et al., *Attention is Not All You Need*, ICML 2021, arXiv:2103.03404 — rank collapse / token uniformity.
6. Cui et al., *The Entropy Mechanism of RL for Reasoning LMs*, arXiv:2505.22617 (preprint) — RL entropy collapse + preservation fixes.
7. Yue et al., *Does RLVR Really Incentivize…*, NeurIPS 2025, arXiv:2504.13837 — RLVR narrows support (contested; hedge).
8. Ziwen Xu et al., *Why Steering Works*, ACL 2026, arXiv:2602.02343 — coherence loss = off-valid-manifold interventions.
9. Xie et al., *Word Salad Chopper*, EMNLP 2025, arXiv:2511.00536 — loop regime linearly decodable (basis of E9.0a).
10. Jain et al., *NEFTune*, ICLR 2024, arXiv:2310.05914 (+ RSP arXiv:2605.11936) — representation-noise injection helps generation; front-loaded schedule.

Flags: "model collapse" definitions contested (Schaeffer arXiv:2503.03150); Tan et al.
arXiv:2407.12404 supports steering *unreliability*, not fluency breakage (use Braun
arXiv:2505.22637 / Da Silva arXiv:2504.04635 for degradation); Golden Gate Claude = Transformer
Circuits web pub, no arXiv ID; LeCun JEPA = OpenReview BZ5a1r-kVsf, not arXiv; Funes-in-ML
peer-reviewed citations unverified — treat as essay tradition. Unrefereed 2026 illustrative
only: Atkinson arXiv:2602.17691, Chen arXiv:2512.14879, arXiv:2602.02195.
