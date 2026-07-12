# Post-training as entropy reduction — primer and RL experiment menu

> Companion to `METHODOLOGY_SAFETY_SPILLOVER_2026-07-03.md`. Written 2026-07-11, after the
> spillover Rung-0/Rung-1 results (RESULTS_LEDGER §B2) and the E9 collapse ladder (§B3).
> Purpose: extend the post-training-origin programme beyond LoRA SFT to preference- and
> RL-based methods, under one unifying frame — **post-training spends entropy to buy
> compliance** — with geometric predictions our existing instruments can falsify.

## 1. The frame

**A pre-trained LM is a maximum-coverage object.** Trained by maximum likelihood on a broad
corpus, the base policy π₀ must put probability mass on everything the corpus contains: it
is deliberately high-entropy, per prompt, over continuations. Post-training of every flavour
reshapes π₀ toward preferred outputs, and the near-universal objective is

    max_π  E_π[r(x, y)] − β · KL(π ‖ π₀)

whose closed-form optimum is an **exponential tilting** of the base:

    π*(y|x) ∝ π₀(y|x) · exp(r(x, y) / β).

Three consequences carry the whole intuition:

1. **Support can only shrink.** π* is absolutely continuous w.r.t. π₀ — a KL-anchored
   method cannot create modes the base lacks; it reweights and prunes the ones that exist.
   This is the formal core of the RLVR "sharpening" result (yue2025rlvr): pass@1 rises,
   pass@k at large k falls — the distribution narrows onto what the reward likes.
2. **The KL term is a budget.** Reward is purchased in bits of distributional change.
   "Dose" across methods is therefore commensurable only as KL(π‖π₀) (measured on a frozen
   corpus), never as example count — 1,000 SFT examples and 1,000 GRPO updates are not the
   same dose.
3. **Entropy is the resource being spent.** For near-binary rewards, tilting concentrates
   mass on the reward-satisfying subset, so conditional entropy falls as reward rises.
   cui2025entropy report exactly this empirical exchange law for RL-for-reasoning (policy
   entropy collapses as performance climbs), and wang2025highentropy locate the spend: RLVR
   updates concentrate on the minority of high-entropy "fork" tokens.

**Methods differ in *how* they spend, not *whether*.**

| Method | Mechanism | Where the entropy goes (hypothesis) |
|---|---|---|
| SFT | forward KL to a fixed external demo set (off-policy imitation) | diffuse: the whole conditional distribution is pulled toward the demo style; at low dose the cheapest fit is a **constant reweighting** |
| DPO | tilting via implicit preference reward (off-policy, contrastive) | intermediate: pulls toward chosen *and* pushes off rejected (the push can even raise entropy off-target — a known DPO pathology) |
| RLVR / GRPO | on-policy tilting toward verifier-approved trajectories | concentrated: prunes the model's own modes, mostly at fork tokens; distribution **sharpens** rather than moves |

## 2. The geometric reading (what our instruments should see)

Our executed spillover result gives SFT's signature: **translation without rotation** — a
single recipe-specific mean-shift direction of the residual stream (5–6% of norm for full
SFT), shared across behaviours, with every behaviour subspace left unrotated. The
entropy frame explains why that is the *expected* signature of low-dose off-policy
imitation: a constant residual-stream shift v adds W_U·v to every logit vector — a fixed,
input-independent reweighting of output preferences. That is exponential tilting at dose
one: the *cheapest possible entropy-reduction move*, changing no feature geometry, only the
standing weights of existing features. (Prior art for the SFT case: the "safety residual
space" affine map of arXiv 2502.09674 — our translation is its bias term **b**, measured on
generic reasoning with a matched non-safety control.)

On-policy RL should not look like this. Selection among the model's own modes operates on
the *variance* of the state distribution, not (primarily) its mean. Pre-registrable
predictions:

- **P-RL1 (contraction vs translation).** At matched KL(π‖π₀), RLVR/GRPO arms contract the
  activation distribution — participation ratio down, top-k variance share up,
  within-behaviour dispersion down (row-paired base→post) — **more** than SFT at the same
  KL, while translating **less** along any single direction.
- **P-RL2 (where the entropy falls).** Token-level predictive-entropy drop ΔH(t) on the
  frozen corpus is diffuse across positions for SFT, but concentrated at high-entropy /
  branch-onset tokens for RLVR (connects to the E9 rung-4 locator finding that token
  entropy is *depleted* at branch onsets in the RL-distilled base — RLVR distillation had
  already spent those bits).
- **P-RL3 (basin deepening).** The E9 loop attractor is an entropy sink. RL arms deepen it:
  loop-probe basin occupancy under matched sampling rises relative to SFT arms at the same
  KL. This ties Movement 1's collapse account to Movement 2's origin question — the
  steering chapter already names RL entropy contraction (cui2025entropy, yue2025rlvr) as
  "a standing hypothesis this thesis does not test"; these arms test it.
- **Frame falsifier.** If RLVR at matched KL shows the *same* translation-dominant,
  contraction-free signature as SFT, then translation is not an off-policy/imitation
  signature but a generic property of low-dose adaptation, and the geometric reading of the
  entropy frame is wrong. That is a clean, publishable negative.

## 3. Experiment menu (testbed: R1-1.5B, GB10 Spark, TRL + LoRA, pt03 harness)

All measurement on the frozen 986-chain corpus, teacher-forced (no generation at eval), so
every arm is row-paired with the existing base activations and directly comparable to the
executed SFT arms.

**New instruments (one script, reused by every rung):**
- *Entropy battery*: per-token predictive entropy H(p_t) and per-token KL(post‖base) —
  the dose meter — teacher-forced over the corpus; aggregated overall, on-span vs off-span,
  and at annotated branch onsets vs matched non-onsets.
- *Contraction battery*: row-paired per-behaviour PR / top-k variance share / mean pairwise
  dispersion, base→post, with the chain-level bootstrap from pt04 for CIs.
- Existing pt03 battery: rotation (excess principal angle over matched-size null) +
  translation (magnitude, direction cosines to the SFT/full-FT directions).

**Rungs, cheapest first, each gated on the last:**

| Rung | What | Compute | Yield |
|---|---|---|---|
| R1 | Entropy + contraction batteries on the **existing** arms (safety100/300/1000, control1000, STAR1 full) | Spark, hours (logits-only forward passes) — or Mac MPS overnight | The KL dose meter retro-fitted to all executed arms; SFT baseline curves for P-RL1/P-RL2; no training |
| R2 | Seed replication ×2 of the four SFT arms (the owed one-seed hardening) | Spark, overnight | per-seed direction variance → per-seed null for the cos-0.57 claim (pairs with pt04) |
| R3 | **DPO** safety arm + DPO non-safety control, 2 KL doses × 2 seeds. Safety pairs: STAR-1 prompts, chosen = STAR-1 deliberative response, rejected = base model's own response (generated once). Control pairs: math tasks, chosen = correct chain, rejected = model's own wrong chain (verifiable, no judge) | Spark, 1–2 nights | first off-policy-contrastive point on the translation↔contraction plane |
| R4 | **GRPO / RLVR-lite**: (i) safety arm with rule-based verifiable reward (refusal-format on harmful prompts, compliance on benign — crude but judge-free); (ii) control arm = math RLVR, boxed-answer correctness on our task pool (the canonical RLVR). Group size 8, ~300 prompts, LoRA | Spark, 2–3 nights | the on-policy point; decides P-RL1/P-RL2/P-RL3 vs the frame falsifier |

**Design rules (inherited + new):**
- Dose axis = measured KL on the frozen corpus; checkpoint each run at ~2 KL levels so
  methods are compared at *matched KL*, not matched steps (fixes the incommensurable-dose
  problem; also the DiD-style per-seed protocol of arXiv 2605.24583 applies).
- ≥2 seeds per arm from R2 onward; per-seed direction variance reported.
- LoRA-vs-full-FT is a live confound for direction comparisons (intruder dimensions,
  arXiv 2410.21228): keep all *new* arms LoRA so they are internally comparable, and treat
  cosines to the full-SFT STAR1 direction as cross-regime, caveated.
- Every arm keeps the size/token-budget-matched non-safety control (the attribution
  design's core) — for RL that is the math-RLVR arm.
- Behavioural side-panel per arm: refusal rate on a small harmful/benign set + math
  accuracy delta (the safety tax), so geometric signatures anchor to behaviour.

## 4. Disk/ops notes (Spark)

Base model already in the Spark HF cache; `peft`/`trl` need `pip install --no-cache-dir`.
Disk has ~5.5 GB free: train + extract one arm at a time, rsync the ~900 MB activation set
back to the Mac, delete on the pod-side before the next arm (pattern proven in the E9/E10
pods). On-policy generation for R4 fits GB10 unified memory comfortably at 1.5B.
