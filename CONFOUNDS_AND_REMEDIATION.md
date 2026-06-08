# Confounds & Negative Results — register and remediation plan

**Created:** 2026-06-06 · **Last verified against repo state:** 2026-06-06 (branch `main`, HEAD `18e2aaf`).

This is the **single source of truth** for *what is scientifically wrong or unproven* in
the project and *how, in what order, we fix it*. It unifies three pre-existing docs that
each cover only one slice:

| Doc | Covers | Relationship to this file |
|-----|--------|---------------------------|
| [`AUDIT.md`](AUDIT.md) | software/numerical **bugs** (mostly fixed 2026-06-05) | upstream — bugs that *produced* the confounds; this file tracks what their fixes still owe (re-runs) |
| [`results/supervisor_meeting/ROBUSTNESS_PLAN.md`](results/supervisor_meeting/ROBUSTNESS_PLAN.md) | the **menu of robustness tests** (written 2026-05-28, *pre-audit*) | downstream — its Tier 0–3 tests become Tiers 1–4 here, re-sequenced behind the regeneration gate |
| [`INVENTORY.md`](INVENTORY.md) | which **outputs are stale vs fresh** | bookkeeping — this file says *why* they're stale and what re-running buys |

> **Keeping this consistent across sessions:** update the **Status** column in §2/§3/§4 as
> items close — do not re-derive the list from scratch. The memory pointer
> `[[confounds-remediation-plan]]` makes a fresh session aware this file exists; always read
> it before touching geometry/steering results. When a number is regenerated, also clear the
> matching row in `INVENTORY.md` and the stale banner in `PROGRESS.md`. Bump the
> "Last verified" date above when you re-check repo state.

---

## 0. The central scientific risk (read this first)

The headline claim of Papers 2–3 is:

> *each individual reasoning behaviour occupies a **curved, low-dimensional, behaviour-specific
> manifold** in the residual stream — not Venhoff's single linear direction, not Huang's
> undifferentiated aggregate.*

That claim rests on exactly three load-bearing quantities, and **as of today all three are in
limbo:**

1. **intrinsic dim ≪ PCA d_eff** — the intrinsic-dim estimators (`twoNN`, CBS `local_intrinsic_dim`)
   were **biased high** (35–50%, and ~3–4× in CBS). Fixed in code 2026-06-05; **not re-run**.
2. **curvature** — the primary diagnostic (`local_vs_global_dim_ratio`) was **confounded by
   sample-size-vs-k**, scoring flat Gaussian data 0.29–0.85 — *the same range as the values
   reported as "evidence of curvature" (≈0.09–0.24)*. The geodesic ratio also halved one-way
   edges. Both fixed in code 2026-06-05; **not re-run**.
3. **the chain-stratified null** — verified in `05b_geometric_diagnostics.py:192`, the null is
   computed with `statistic_fn=top_k_variance_ratio`. **It tests a PCA variance ratio, never the
   intrinsic-dim or curvature numbers.** So the two quantities the whole claim turns on have **no
   significance test at all** against the obvious confound (the chain).

On top of that there is **no working power analysis** (`power_analysis_curvature.py` last wrote an
all-NaN table; fixed to fail-loud but **not re-run**), so we cannot even say whether our effective
sample size *could* detect the curvature we claim.

**Net effect: no geometry number in this repository is currently citable.** Every geometry/steering
output is quarantined in `results/_STALE_pre_fix_20260605/`. The temporal-ordering pilot (Paper 1)
is the *only* standing empirical result. This is not a crisis — the code is fixed; what remains is
to **re-run, and to test the right statistics.** This document sequences that.

---

## 1. Verified current state (2026-06-06)

What I confirmed directly against the repo (not from memory):

- **Geometry NOT regenerated.** No fresh `results/geometric|pca|cross_layer|triangulation|clustering`;
  all 12 dirs sit in `results/_STALE_pre_fix_20260605/`. The regeneration gate (§4 Gate 0) is **open**.
- **Steering vectors ARE fresh** (`results/steering_vectors/R1-1.5B/`, 2026-06-06, built on the fixed
  `svd_solver="full"` PCA). But they are **linear** (diff-of-means projected onto a top-k PCA subspace)
  and built on **mean-pooled** activations — see CF-5, CF-6.
- **Power analysis NOT re-run** — no fresh `results/power_analysis/`; the stale one is the all-NaN table.
- **`results/robustness/` exists but is empty** — the robustness pipeline has produced nothing yet.
- **The N story (resolved — the audit's "N=51–145" was a smoke-run artifact):**
  - The *full* geometry run (28-May, now stale) used the **full** per-behaviour counts:
    backtracking **10 267**, uncertainty **16 728**, example-testing **5 829**, adding-knowledge **5 027**
    sentences (confirmed from activation shapes and steering metadata `n_on`).
  - The *smoke* run used N=51–145 (a capped cohort; `R1-1.5B-smoke` symlinks the real corpus). The
    audit memo's "small N (51–145)" caution was citing the **smoke** numbers. **Raw N is not the
    problem for the full run.** The real problem is **effective N** (CF-2): those thousands of
    sentences come from only ~1 000 chains, so within-chain autocorrelation collapses the number of
    *independent* units to roughly the chain count.

---

## 2. Confound register

Severity: **S0** = invalidates the headline claim if unaddressed · **S1** = weakens a paper ·
**S2** = reviewer-objection / scope caveat. Type: **bug→fixed** (code fixed, re-run owed) vs
**design** (research-design choice, no code fix possible — needs a different experiment).

| ID | Confound | Claim threatened | Sev | Type | Status |
|----|----------|------------------|-----|------|--------|
| **CF-1** | Curvature diagnostic confounded (flat data scored in the same range as "evidence"); geodesic symmetrization halved edges | "manifold is **curved**" (Paper 2/3 core) | S0 | bug→fixed | code FIXED 2026-06-05; **re-run owed** |
| **CF-2** | **Chain confound / effective-N.** Sentences within a chain are autocorrelated → ~5–16k points are not independent; the i.i.d. assumption behind every intrinsic-dim/curvature estimator is violated | "behaviour-specific, low-dim, curved" (all geometry) | S0 | design | **OPEN — the keystone** |
| **CF-3** | **Null tests the wrong statistic.** Chain-stratified null is on `top_k_variance_ratio`, not on intrinsic dim or curvature (verified `05b:192`) → the load-bearing numbers have no significance test | "the structure is real, not the chain" | S0 | bug→design | **OPEN** (needs the null wired onto the right statistics) |
| **CF-4** | Intrinsic-dim estimators biased high (`twoNN` +35–50%, CBS `local_intrinsic_dim` ~3–4×) | "intrinsic dim ≪ PCA dim" (the compression gap = the whole result) | S0 | bug→fixed | code FIXED 2026-06-05; **re-run owed** |
| **CF-5** | **Linear apparatus, curvature claim.** "Manifold-projected" steering is a top-k PCA (linear-subspace) projection; curvature in 5b is measured *after* projecting to a top-k PCA subspace (`05b:135`). A linear operator cannot test a curvature claim | Paper 3 (curvature → steering) | S1 | design | **OPEN** (reframe or build a nonlinear operator) |
| **CF-6** | **Mean-pooling destroys the trajectory.** Activations are mean-pooled over the first ~10 tokens of each span; the manifold claim is fundamentally about a *trajectory* (a curve through the chain) | process/trajectory claims; "curve not point" | S1 | design | **OPEN** |
| **CF-7** | **Single unvalidated annotator.** Labels (the dependent variable) come from one LLM (Sonnet 4.5). 3-annotator robustness (Qwen3-235B, Nova-Pro) is in flight but Nova-Pro had ~290/1000 partial-parse failures (weak arm); inter-annotator κ not yet reported | every geometry claim (labels gate everything) | S1 | design (mitigation in progress) | **IN PROGRESS** |
| **CF-8** | **50% truncation.** 50.2% of chains hit the 8192 cap and lack a closing `</think>`; truncation rate correlates with category (lateral 95%, spatial 71%) → correlates with behaviour mix | position/behaviour analyses; corpus validity | S1 | design | **OPEN** (raise cap / stratify / drop categories — undecided) |
| **CF-9** | Bootstrap CIs too narrow — `intrinsic_dim`/`curvature` resample *derived* quantities (μ ratios, pairwise distances), which are dependent → CIs like [0.575, 0.587] (AUDIT #16) | any CI-backed geometry comparison | S2 | bug | **OPEN** (resample points end-to-end) |
| **CF-10** | Activation-patching proxy — `behaviour_marker_logprob` scores behaviours by tokens like "wait"/"actually" and patches position *i* across non-aligned chains → conflates behaviour with surface lexis | causal "Paper 2 main" claim | S1 | design | **OPEN** (validated metric + positional alignment) |
| **CF-11** | **Safety ∦ capability** ([2505.14185]). A "safety manifold" may be a difficulty/capability manifold; linear separability of safety from capability is not given | the proposed safety flagship (Part II) | S0 (for safety arm) | design | **OPEN** (capability control is mandatory, not optional) |
| **CF-12** | Cross-model "bootstrap" was a Gaussian reconstructed from CI width, not a bootstrap — the load-bearing distillation-vs-reveals test for the knowledge-creation arm | knowledge-creation / cross-model H4 | S1 | bug→partial | relabelled `p_normal_approx`; true two-sample bootstrap added but needs producers to persist per-resample arrays — **verify end-to-end** |

### Notes on the non-obvious ones

- **CF-2 is the keystone.** Everything else can pass and the headline still falls if the geometry is
  a property of *which chain a sentence came from* rather than *the behaviour*. Raw N (5–16k) hides
  this — the honest denominator is "independent chains," which is in the hundreds and is **not yet
  quantified** (task R1.0 below). Every curvature/intrinsic-dim estimate should be recomputed under
  (a) one-sentence-per-chain and (b) a chain-block bootstrap.
- **CF-3 turns the headline from "significant" to "unsupported."** The pretty p<.01 "28/28 layers"
  result in the project log is the *variance-ratio* null — a real result, but about a *different*
  quantity than the one we headline. The intrinsic-dim and curvature numbers currently have **no null
  and over-narrow CIs (CF-9)**, i.e. no error bars worth the name.
- **CF-1 + CF-4 mean the numbers will move on re-run.** Direction is genuinely unknown: the
  bias-correction lowers intrinsic dim (good for the compression story), but the confound-correction
  to curvature could push the ratio toward 1.0 (flat) on real data — which would *weaken* the
  curvature claim. We will not know until Gate 0 runs. Do not pre-commit prose either way.
- **CF-5 is a claim/instrument mismatch, not a bug.** Two clean resolutions: (i) split the framing —
  Paper 2 claims **subspace** (rank ≫ 1, honest, linear-projection-testable) and Paper 3 claims
  **curvature** (and then needs a genuinely nonlinear steering operator: geodesic / local-PCA-on-the-fly);
  or (ii) drop the curvature-steering link and keep curvature as a descriptive geometry result only.

---

## 3. Negative-results register

These are findings that came out null/weak. Several are *honest* results worth reporting; the danger
is letting a downstream plan keep assuming the positive version.

| ID | Negative result | What it kills / weakens | Honest reframe | Status |
|----|-----------------|-------------------------|----------------|--------|
| **NR-1** | Power analysis wrote an **all-NaN** table that masqueraded as a ">max tested" null | we have **no** detection-power result for curvature at our (effective) N | not a null — a **silent failure**; must re-run the fixed fail-loud version | **OPEN** (re-run) |
| **NR-2** | **adding-knowledge is the weak behaviour** — null at only L17–19 (3/28), highest intrinsic dim (13.3), smallest N (5 027), no sub-types | it is *also* the one behaviour shared by every arm (knowledge-invocation thread) and the cross-domain story leans on it | report as the honest outlier; do not let the knowledge-creation arm over-rely on it | **STANDING** (carry as caveat) |
| **NR-3** | **No discrete sub-types** (5d: k=2 for all four, silhouette 0.11–0.18) | kills the planned "sub-type steering" study; the 4–6 sub-types hypothesis for adding-knowledge | reframed as **continuous-manifold steering** — graceful, already adopted | **RESOLVED (reframed)** |
| **NR-4** | **example-testing null gap (L7–14)** — significant 19/28 but with a mid-layer hole, an early-and-late profile | complicates the "geometry peaks at middle layers" story for this behaviour | report per-behaviour rather than averaging; the variance-ratio null (re-run) will confirm or move this | **STANDING** |
| **NR-5** | **d_eff is high (45–98), not low** | a skeptic reads "high-dimensional/distributed," not "manifold"; the rescue is entirely the intrinsic-dim gap — which is exactly what CF-2/CF-3/CF-4 put in doubt | lead with **participation ratio** (scale-robust) + intrinsic-dim *with* a null and honest CIs | **STANDING** |

---

## 4. Sequenced remediation plan

Ordering principle (inherited from ROBUSTNESS_PLAN): **cheapest-to-falsify first**, and **do not spend
GPU/API until the free CPU checks either survive or force a reframe.** Each task names the confound(s)
it closes.

### Gate 0 — Regenerate on the fixed code (BLOCKING; nothing downstream is valid until this is done)

Until Gate 0 completes, **do not** write results prose, refresh figures, or cite any geometry number.

- **G0.1** Run `run_rerun_local.sh` (Phase 5 → 5c → triangulation → 5d → 5b on the fixed
  estimators; CPU, ~1 hr). Closes the *re-run* half of **CF-1, CF-4**. ⚠️ It archives the current
  `results/{pca,...}` — make sure the `_STALE_` copies remain the canonical "before" snapshot.
- **G0.2** Fix and re-run `power_analysis_curvature.py`; confirm a non-NaN `power_table.csv`. Closes **NR-1**.
- **G0.3** Refresh the TwoNN dims in `PROGRESS.md`, regenerate `supervisor_meeting` figs (`make_fresh_figures.py`),
  re-render HTML, and clear the matching rows in `INVENTORY.md`. Bookkeeping for CF-1/CF-4.
- **G0.4** Sanity gate: if the fixed curvature diagnostic now scores the real data ≈1.0 (flat), **stop and
  reframe** before spending anything on Tiers 1–4 — that would mean CF-1 was fatal, not cosmetic.

### Tier 1 — Internal robustness (CPU, existing data; the cheapest and most likely to break the result)

- **R1.0** *(new — do first)* **Quantify effective N**: count distinct chains contributing to each
  behaviour, and sentences-per-chain. This is the honest denominator for **CF-2** and tells us how much
  the i.i.d. violation costs.
- **R1.1** **Keystone — wire the null onto the right statistics.** Re-run the chain-stratified null with
  `statistic_fn` = intrinsic dim **and** each curvature diagnostic (not just `top_k_variance_ratio`).
  Closes **CF-3**. Also run one-sentence-per-chain + chain-block bootstrap (CF-2).
- **R1.2** **Honest CIs**: replace the derived-quantity bootstrap with a point-resampling end-to-end
  bootstrap for intrinsic dim and curvature. Closes **CF-9**.
- **R1.3** **Synthetic null model**: simulate (i) low-rank + matched structured noise and (ii) a curved
  ground-truth manifold at our N, d, variance; run the *same* pipeline. Real data must beat the flat
  simulation on curvature. Hardens **CF-1** beyond "the estimator is unbiased on a sphere."
- **R1.4** **Preprocessing + estimator sweeps**: {raw, z-scored, massive-activation-dims removed} ×
  {1536-D, PCA-50}; TwoNN discard fraction, LB k-range, curvature kNN k. Report stability (CF-2, CF-9).
- **R1.5** **PR-primary reporting** + Marchenko–Pastur noise floor on d_eff. Addresses **NR-5**.

> **Decision gate after Tier 1.** If intrinsic-dim and curvature survive R1.1–R1.3 → the geometry is
> *defensible*, spend on Tiers 2–4. If they don't → reframe the central claim **now**, before any GPU/API.

### Tier 2 — External robustness (modest API + CPU)

- **R2.1** **Annotator agreement first.** Finish the 3-annotator pipeline; report span-F1 + Cohen's κ,
  anchored to a small human-labelled gold set; down-weight Nova-Pro per its partial-parse rate. Closes
  the *measurement* half of **CF-7**.
- **R2.2** **Re-derive geometry under each annotator's labels** — the headline external-robustness test
  (does the manifold replicate when the labels change?). Closes the *replication* half of **CF-7**.
- **R2.3** **Truncation decision + stratification.** Choose raise-cap / stratify / drop-categories; at
  minimum stratify every geometry result by complete-vs-truncated. Closes **CF-8**.

### Tier 3 — Causal & the apparatus/claim mismatch (GPU + API)

- **R3.1** **Resolve CF-5 explicitly.** Either (a) split Paper 2 = subspace / Paper 3 = curvature with a
  genuinely nonlinear steering operator (geodesic step / local-tangent projection), or (b) demote
  curvature-steering to descriptive. Pick before building Phase-7 prose.
- **R3.2** **Trajectory (un-pooled) re-extraction** under {onset-only, first-10, full-segment} pooling;
  confirm geometry holds and run the process-level analysis on the *un-pooled* trajectory. Closes **CF-6**.
- **R3.3** **Fix activation patching**: a validated behaviour metric (not lexical markers) + positional
  alignment across chains before any necessity claim. Closes **CF-10**.
- **R3.4** **Steering sufficiency + necessity**: single vs manifold-projected vs subspace-ablation, across
  α; this is where §8's near-orthogonality (diff-of-means ⟂ PC1 for example-testing/adding-knowledge)
  predicts projection matters most at intermediate k.

### Tier 4 — Safety extension prerequisites (only after Part I geometry is defensible)

- **R4.1** **Capability control is mandatory** — a difficulty/capability-matched contrast for every safety
  geometry claim. Closes **CF-11**. Without it the safety flagship is not defensible.
- **R4.2** **Real cross-model bootstrap** — verify the two-sample bootstrap in `cross_model_compare` is
  wired end-to-end (producers persist per-resample arrays) before any H4 distillation-vs-reveals claim.
  Closes **CF-12**.

---

## 5. Definition of done (per headline claim)

A claim is citable only when its row is fully checked.

| Claim | Done when |
|-------|-----------|
| "behaviours are **low-dimensional**" | intrinsic dim (fixed estimator) ≪ PCA d_eff **with** a chain-stratified null on the intrinsic-dim statistic (R1.1) and point-resampled CIs (R1.2), surviving the preprocessing sweep (R1.4) |
| "the manifold is **curved**" | fixed curvature diagnostics (Gate 0) beat the flat-subspace synthetic null (R1.3) **and** carry a chain-stratified null + honest CIs (R1.1–R1.2); reframed cleanly vs the linear steering apparatus (R3.1) |
| "structure is **behaviour-specific** (not the chain)" | survives one-sentence-per-chain + chain-block bootstrap (R1.1/CF-2) **and** replicates under a second annotator (R2.2) |
| "geometry **peaks at middle layers**" | holds for PR-primary + MP noise floor (R1.5) and per-behaviour, not just on-average (NR-4) |
| "intervening on the manifold **changes behaviour**" | steering sufficiency + a *fixed* activation-patching necessity test (R3.3–R3.4), un-pooled (R3.2) |
| any **safety** geometry claim | passes a capability-matched control (R4.1) on the fixed estimators |

---

## 6. What is NOT broken (so we don't over-correct)

- **Base-model verification** (R1 distilled from Qwen-2.5-Math-1.5B, not Instruct) — no estimator, valid.
- **Chain-quality / truncation stats** — descriptive, valid (and they *surfaced* CF-8).
- **Temporal-ordering pilot (Paper 1)** — the one standing empirical result.
- **The core idea** — Huang-aggregate-manifold vs Venhoff-per-behaviour-direction is a genuine,
  novel wedge. The problems above are about *proving* it rigorously, not about whether it's worth proving.
- **The software bugs themselves** — almost all fixed and now regression-tested (220+ tests). This file
  is about the *scientific* debt their fixes left behind (the re-runs) plus the design-level confounds no
  code change can fix.
