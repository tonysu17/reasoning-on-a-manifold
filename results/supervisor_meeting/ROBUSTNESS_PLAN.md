# Robustness & Causality Plan

*Consolidates the two robustness threads — (a) steering / labels / datasets / contrasts and (b) PCA / intrinsic-dimension geometry — into one sequenced agenda. Written in response to the supervisor's mandate: "consistent behaviour across different settings" and "more causal." Nothing here is executed yet.*

**Last updated:** 28 May 2026.

---

## 0. Organising principle (the iteration)

Three reframings over the earlier ad-hoc lists:

1. **Every check defends a specific claim.** A robustness task that isn't tied to a load-bearing claim is over-engineering — we drop it.
2. **Cheapest-to-falsify first.** Order the agenda by information-value ÷ cost. The tests most likely to *break* our own result, using the *cheapest* resources, come first. (Counter-intuitively, that means the single most important task is also free.)
3. **A decision gate.** Internal robustness (same data, perturb the method) is cheap and comes first. Only after the result survives it do we spend GPU/credits on external replication and causal tests. If an internal check reveals a confound, we **reframe before spending**, not after.

Internal robustness = "is our existing result an artifact of our choices?" · External replication = "does it reproduce under a new annotator / dataset / model?" · Causal = "does intervening on the manifold change behaviour?"

---

## 1. The claims, their biggest threat, and the test that defends each

| # | Claim | Current status | Biggest threat | Defending test |
|---|---|---|---|---|
| C1 | d_eff ≫ 1 (single-direction falsified) | **robust** (47–160 under any preprocessing) | none material | — |
| C2 | intrinsic dim ≈ 7–13 (low) | suggestive | chain confound (i.i.d. violation); distance concentration in 1536-D | chain-stratified 5b; intrinsic dim on PCA-reduced inputs |
| C3 | the manifold is *curved* (not a flat subspace) | suggestive | noise/density mimics curvature; low-rank + structured noise | synthetic null model; chain-stratified curvature |
| C4 | the structure is **behaviour-specific** (not chain/content) | **untested for 5b** | chain confound | chain-stratified 5b (keystone) |
| C5 | geometry peaks at middle layers (L14–17) | suggestive | d_eff is scale-dependent (2× under z-scoring); PR is more stable | standardization + pooling sweep; lead with PR |
| C6 | the steering direction is a manifold property | partial | contrast set is narrow (no neutral/deduction baseline) | multi-contrast + orthogonalised vectors |
| C7 | intervening on the manifold changes behaviour | **not yet made** | — | steering (sufficiency) + ablation/patching (necessity) |

**Read this table as the spine of the plan.** C4 is the keystone: if the low-dimensional curvature is a property of *which chain a sentence came from* rather than *the behaviour*, the headline weakens — and we have not yet tested it, because the chain-stratified null was only ever applied to the top-10 variance ratio, never to intrinsic dim or curvature.

---

## 2. Tier 0 — Internal robustness (free / CPU, existing data, do FIRST)

These need no new data, no API, no GPU (except where noted). They are the cheapest and the most likely to falsify or reframe the central claim.

- **0.1 Chain-stratified 5b — the keystone.** Recompute TwoNN / Levina-Bickel / correlation-dim and all three curvature diagnostics on **one sentence per chain** (and via a chain-block bootstrap). *Defends C2, C3, C4.* If intrinsic dim stays ~10 and curvature stays high → the manifold is behaviour-specific. If intrinsic dim jumps → much of it was the chain confound, and we reframe. *Cost: CPU, hours.*
- **0.2 Lead with the participation ratio; report d_eff with a noise floor.** PR is scale-robust; raw d_eff doubles under z-scoring. Report PR as the primary scalar, and d_eff *after* the Marchenko-Pastur edge (count only above-noise eigenvalues), raw **and** standardised. *Defends C1, C5.* *Cost: CPU.*
- **0.3 Preprocessing sweep.** Intrinsic dim + curvature under {raw, z-scored, massive-activation dims removed} and on {raw 1536-D, PCA-50 inputs}. Confirms the result isn't an artifact of scaling or high-D distance concentration. *Defends C2, C3, C5.* *Cost: CPU.*
- **0.4 Synthetic null model.** Simulate (i) low-rank + matched structured noise and (ii) a curved-manifold ground truth at our N, d, variance; run the *same* d_eff / TwoNN / curvature pipeline. Real data must show **more** curvature than the flat-subspace simulation. *Defends C3.* *Cost: CPU.*
- **0.5 Common-N subsampling + bootstrap CIs.** Re-estimate all behaviours at a shared N (≈5,000) with CIs, so cross-behaviour comparisons aren't N-confounded. *Defends cross-behaviour claims.* *Cost: CPU.*
- **0.6 Estimator hyperparameter sweep.** TwoNN discard fraction, LB k-range, corr-dim fit window, kNN k for curvature/geodesic. Report stability. *Defends C2, C3.* *Cost: CPU.*

> **Decision gate.** Run 0.1 first. If C2/C3/C4 survive Tier 0, invest in Tiers 1–3. If not, reframe the central claim *before* spending on external replication.

---

## 3. Tier 1 — Contrast / ablation & layer causality (GPU, no API)

- **1.1 Neutral baseline + multi-contrast vectors.** Re-extract activations for **all 6 labels** (adds `deduction`, `initializing`). Rebuild every steering vector under several contrasts — vs deduction (Venhoff-style D⁻), vs all-other-labels, pairwise / leave-one-out — plus a **Gram-Schmidt orthogonalised** vector (the behaviour component not explained by the others). A robust direction is stable across contrasts. *Defends C6.* *Cost: GPU re-extraction, no API.*
- **1.2 Activation patching (necessity).** Patch a behaviour's activations from donor → recipient at the candidate layer; measure the causal change. *Toward C7.* *Cost: GPU.*
- **1.3 Token-window re-extraction.** Re-extract under {onset-only, first-10, full-segment} pooling and confirm geometry holds (addresses the fixed-window caveat). *Defends C2, C5.* *Cost: GPU.*

---

## 4. Tier 2 — Annotator robustness (modest API)

- **2.1 Second annotator + agreement.** Re-annotate a stratified subset (~100–150 chains) with a *different* model (GPT-4o / Gemini / Claude-Opus); compute span-F1 and Cohen's κ; anchor both to a small human-labelled gold set. *Defends label validity.*
- **2.2 Re-derive the manifold under the second annotator's labels.** The headline external-robustness test: does the geometry replicate when the labels come from a different annotator? *Defends every geometry claim, externally.* *Cost: API (subset) + CPU.*

---

## 5. Tier 3 — Dataset diversity & full causal (GPU + API)

- **3.1 Second dataset, end-to-end.** Run generation → annotation → Phase 4/5/5b on a different distribution — one math-heavy (MATH / GSM8K, à la Venhoff) and/or one deliberately distinct. ~300 chains is enough for a replication check (no need for a full 1,000). The **7B run already supplies scale diversity**; this adds *input-distribution* diversity. *Defends all claims, externally.*
- **3.2 Steering: sufficiency + necessity.** Add the vector (behaviour ↑ = sufficiency) and ablate the manifold subspace (behaviour ↓ = necessity); compare single-direction vs manifold-projected vs subspace-ablation. This is where §8's finding bites (the difference-of-means is nearly orthogonal to PC1 for example-testing / adding-knowledge, so projection should matter most at intermediate k). *Establishes C7.* *Cost: GPU + API.*
- **3.3 Vector transfer.** Build the vector on dataset/annotator A, test it steers on B — causal generalisation. *Cost: GPU + API.*

---

## 6. Minimal viable robustness (if resources are tight)

The smallest set that makes the geometry half defensible to a skeptical reviewer:

- **0.1** chain-stratified 5b (keystone),
- **0.2** PR-primary + MP noise floor,
- **0.4** synthetic null model,
- **1.1** neutral-baseline / orthogonalised contrast,
- **2.1** annotator agreement on a subset.

Everything else strengthens the paper but is not load-bearing. Doing only these five answers "is it an artifact?" (0.1, 0.2, 0.4), "is the vector well-defined?" (1.1), and "are the labels reliable?" (2.1).

---

## 7. Effort / cost / dependency summary

| Tier | Items | Resource | New data? | Rough effort |
|---|---|---|---|---|
| 0 — internal | 0.1–0.6 | CPU (laptop) | no | ~1 day |
| 1 — contrast/causal-layer | 1.1–1.3 | GPU | re-extraction only | ~1 day GPU |
| 2 — annotator | 2.1–2.2 | API (subset) + CPU | re-annotate subset | modest credits |
| 3 — dataset/causal | 3.1–3.3 | GPU + API | new corpus | largest spend |

**Dependencies:** 1.1 unlocks the neutral-baseline contrast; 2.1 must precede 2.2; 3.2 needs Phase-6 vectors (✅ already built) + a free GPU + credits.

---

## 8. Recommended first move

Run **Tier 0 in full** — one day of laptop CPU, no credits, no new data — starting with **0.1 (chain-stratified 5b)**. It is the cheapest task on the board and the one most able to either bulletproof or reframe the central manifold claim. Reporting "intrinsic dimension and curvature survive chain-stratification, hold under standardization and PCA-reduction, and exceed a low-rank-plus-noise simulation" would convert the geometry from *suggestive* to *defensible* before a single credit is spent — and it directly answers the supervisor on both robustness and the chain confound.

---

## 9. What this buys with the supervisor

A single, defensible sentence: *every headline claim survives perturbation of the annotator, dataset, contrast set, preprocessing, and the i.i.d. assumption, and is backed by a causal (steering + ablation) test.* That is exactly "consistent behaviour across different settings" plus "more causal," made concrete and sequenced by cost.
