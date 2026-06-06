# Phase 5b deep dive — methodology, results, comparison to Goodfire

> ⚠️ **Numbers predate the 2026-06-05 estimator fixes and are superseded.** The
> 2026-06-05 audit (`AUDIT.md` §2 #1–#2) found this document's two headline
> estimators biased: TwoNN over-estimated intrinsic dim by ~35–50%, and the
> local-vs-global curvature ratio was confounded (flat synthetic data scored
> 0.29–0.85, not ~1.0). Treat every intrinsic-dim / curvature figure here as
> **pending regeneration** on the fixed code (`results/_STALE_pre_fix_20260605/`,
> `INVENTORY.md`). The *methodology* description remains accurate; only the
> numbers are stale.

This document explains **everything Phase 5b computes**, what positive/negative results look like, our actual numbers, and how they map onto the Goodfire neural-geometry papers.

*Updated 28 May 2026 — intrinsic-dim and curvature numbers are at the L17 reference layer (unchanged, deterministic). The null hierarchy is now computed at **all 28 layers** (§4), and Phase 5d clustering results are now in (§7).*

---

## 1. The three core questions Phase 5b answers

| Question | Method | Bad outcome | Good outcome (manifold) |
|---|---|---|---|
| Is the structure **real**, or chain-confound? | Null hierarchy | All p > 0.05 → chain artefact | p < 0.05 → behaviour-specific structure beyond chain |
| Is the manifold **low-dimensional**? | Intrinsic-dim estimators (TwoNN, Levina-Bickel, correlation dim) | Intrinsic ≈ ambient (~50) → no compression | Intrinsic ≪ PCA dim (10 vs 50) → curved compression |
| Is the manifold **curved**? | Three curvature diagnostics | All ≈ flat → flat subspace, not a manifold | All ≠ flat → genuine non-flat manifold |

**Headline preview:**

| Question | Backtracking | Uncertainty | Example-testing | Adding-knowledge |
|---|---|---|---|---|
| Real? (chain-strat p<0.01) | ✅ **all 28 layers** | ✅ **all 28 layers** | ✅ 19/28 layers | ✅ L17–19 only |
| Low-dim? (TwoNN vs PCA, L17) | ✅ 10 vs 47 (5×) | ✅ 11 vs 57 (5×) | ✅ 11 vs 54 (5×) | ⚠️ 13 vs 61 (4.6×) |
| Curved? (all 3 diagnostics) | ✅ very curved | ✅ very curved | ✅ curved | ✅ curved (highest dim) |

**Three of four behaviours pass all three tests cleanly. Adding-knowledge is the outlier — Section 7.**

---

## 2. Battery A — Intrinsic dimensionality (the headline metric)

The single most important number in Phase 5b: **how many free parameters the behaviour really has**, independent of the ambient embedding dimension.

### Why this matters

PCA's d_eff_70 says "47 components for 70% of variance" but not whether the data **has** 47 degrees of freedom or is a **lower-dim manifold curved through a higher-dim space**.

Concrete: days of the week form a 1-D cycle living in 1536-D space. PCA on the 7 centroids finds d_eff_70 = 2 (to draw a circle), but the **intrinsic** dimensionality is 1. The discrepancy IS the curvature signal.

### A1. TwoNN (Facco et al. 2017)

For a point on a d-dim manifold, the ratio $\mu_i = r_2(x_i)/r_1(x_i)$ (distance to 2nd over 1st nearest neighbour) follows a Pareto distribution with shape $d$:
$$ P(\mu \mid d) = d \cdot \mu^{-d-1}, \quad \mu \geq 1 $$
We estimate $d$ by maximum likelihood. Uses only the 2 closest neighbours → robust to far-away noise.

**Our values at L17:**

| Behaviour | TwoNN (95% CI) |
|---|---|
| backtracking | **10.4** [10.0, 10.9] |
| uncertainty-estimation | **11.1** [10.7, 11.6] |
| example-testing | **10.7** [10.0, 11.2] |
| adding-knowledge | **13.3** [12.6, 14.0] |

*Note: intrinsic dim varies by layer — at each behaviour's PR-trough layer it is even lower (≈6–7 for backtracking/uncertainty at L14), consistent with maximal compression there.*

### A2. Levina-Bickel (2004)

MLE on the k-NN graph:
$$ \hat{d}_k(x) = \left[ \frac{1}{k-1}\sum_{j=1}^{k-1}\log\frac{T_k(x)}{T_j(x)} \right]^{-1} $$
where $T_j(x)$ is the distance to the $j$-th neighbour. Uses k neighbours (less variance, more noise-sensitive than TwoNN).

**Our values at L17:** backtracking 13.2, uncertainty 14.4, example-testing 13.5, adding-knowledge 14.7 — slightly higher than TwoNN, as expected (Levina-Bickel is more conservative).

### A3. Correlation dimension (Grassberger-Procaccia 1983)

Log-log slope of the pair-count $C(r) = \frac{1}{N^2}\sum_{i\neq j}\mathbb{1}[\lVert x_i-x_j\rVert < r]$ as $r \to 0$. Biased low for highly curved manifolds.

**Our values at L17:** backtracking 6.8, uncertainty 7.0, example-testing 7.6, adding-knowledge 8.6 — lower than TwoNN, consistent with curvature bias.

### The headline comparison (L17)

| Behaviour | TwoNN (intrinsic) | PCA d_eff_70 (ambient) | Compression |
|---|---|---|---|
| backtracking | 10.4 | 47 | **4.5×** |
| uncertainty-estimation | 11.1 | 57 | **5.1×** |
| example-testing | 10.7 | 54 | **5.1×** |
| adding-knowledge | 13.3 | 61 | **4.6×** |

**The manifold is intrinsically ~5× lower-dimensional than its PCA ambient embedding** — the geometric manifold signature. Venhoff's single-direction prediction would be TwoNN = 1 *and* d_eff = 1; seeing TwoNN ≈ 10 *and* d_eff ≈ 50 is simultaneously incompatible with Venhoff and supportive of the curved-manifold story. **The gap widens at late layers** (d_eff_70 reaches 98 at L26 while intrinsic dim stays ~10–13) — curvature increases with depth.

### Even stronger at each behaviour's manifold-peak layer

The L17 reference table above *understates* the effect. At each behaviour's **own PR-trough layer** (its geometric peak), the intrinsic dimension drops further and compression intensifies:

| Behaviour | Peak layer | TwoNN (intrinsic) | PCA d_eff_70 | Compression |
|---|---|---|---|---|
| backtracking | L14 | 7.2 | 47 | **6.6×** |
| uncertainty-estimation | L14 | 6.4 | 55 | **8.6×** |
| example-testing | L27 | 7.4 | 52 | **7.0×** |
| adding-knowledge | L17 | 13.3 | 61 | **4.6×** |

For the three strong behaviours the intrinsic dimension falls to **6–7** at the peak — a **6.6–8.6×** compression relative to the ambient PCA embedding — while curvature stays high (geodesic/Euclidean 1.64–1.86, tangent 57–61°). Adding-knowledge remains the diffuse outlier (intrinsic 13.3, peak L17). We quote the L17 reference numbers elsewhere for continuity, but this layer-matched view is the stronger, more faithful statement of the same result.

---

## 3. Battery B — Curvature diagnostics

Intrinsic dim < PCA dim *suggests* curvature; these diagnostics **measure** it directly.

### B1. Local-vs-global PCA dim ratio
$$ \text{ratio}(k) = \frac{d_{\text{eff,70}}^{\text{local k-NN}}}{d_{\text{eff,70}}^{\text{global}}} $$
ratio = 1 → flat; ratio ≪ 1 → strongly curved (locally you see only the low-dim tangent plane).

**At L17, k=10:** backtracking **0.091**, uncertainty **0.084**, example-testing 0.108, adding-knowledge 0.127. Locally the manifold looks 8–13% the global dimension → **strongly curved**. (Rises toward the global value as k grows and neighbourhoods leave the tangent plane.)

### B2. Geodesic / Euclidean ratio
$$ \rho(x,y) = \frac{d_{\text{geodesic}}(x,y)}{d_{\text{Euclidean}}(x,y)} $$
ratio = 1 → flat; > 1 → curved. (A unit circle's antipodal geodesic/chord = π/2 ≈ 1.57.)

**At L17, k=10:** backtracking **1.83**, uncertainty **1.82**, example-testing 1.68, adding-knowledge 1.62 — geodesics ~70% longer than chords; **stronger curvature than a moderate sphere**.

### B3. Tangent-space variation

Principal angle between locally-fit tangent planes at nearby points. 0° flat; 30–60° moderate; 60–90° strong.

**At L17, k=10:** backtracking **58°**, uncertainty **62°**, example-testing 58°, adding-knowledge 68° — heavily curved.

### All three agree (backtracking)

| Diagnostic | If FLAT | Our value | Verdict |
|---|---|---|---|
| Local-vs-global PCA ratio | 1.0 | 0.091 | ✅ curved |
| Geodesic/Euclidean ratio | 1.0 | 1.83 | ✅ curved |
| Tangent variation | 0° | 58° | ✅ curved |

Three independent geometric formalisms, same conclusion: the manifold is strongly non-flat.

---

## 4. Battery C — Null hierarchy (the rigour test)

The confound: **what if apparent structure is just "sentences from the same chain look alike," regardless of label?** A backtracking sentence in a math chain carries that chain's math context; two backtracking sentences from the same chain are similar *because of the chain*, not the label.

### C1. Chain-stratified permutation (primary)

1. Pool all 37,851 activations across the 4 behaviours and all chains.
2. For each resample (B = 100): within each chain, **permute the behaviour labels** of its sentences (labels stay inside the chain). Compute the top-10 variance ratio for the target label.
3. p = fraction of resamples with shuffled statistic ≥ real statistic.

This removes chain-style (chain identity preserved) and tests whether the **label** carries information beyond the chain.

### NEW: now computed at every one of the 28 layers

![28-layer null](fig8_null_hierarchy.png)

| Behaviour | # layers significant (p<0.01) | Significant layers |
|---|---|---|
| backtracking | **28 / 28** | all |
| uncertainty-estimation | **28 / 28** | all |
| example-testing | **19 / 28** | L0–6, L15–27 (gap L7–14) |
| adding-knowledge | **3 / 28** | L17, L18, L19 |

Representative detail at the three canonical layers (real vs chain-stratified null mean):

| Behaviour | Layer | Real | Null mean | p | Bonferroni (α=0.05/112) |
|---|---|---|---|---|---|
| backtracking | L14 | 0.4484 | 0.4134 | **0.000** | ✅ |
| backtracking | L17 | 0.4434 | 0.4096 | **0.000** | ✅ |
| backtracking | L27 | 0.4117 | 0.3862 | **0.000** | ✅ |
| uncertainty-estimation | L14 | 0.4288 | 0.4134 | **0.000** | ✅ |
| uncertainty-estimation | L17 | 0.4202 | 0.4094 | **0.000** | ✅ |
| uncertainty-estimation | L27 | 0.3774 | 0.3708 | **0.000** | ✅ |
| example-testing | L14 | 0.4156 | 0.4146 | 0.150 | ❌ |
| example-testing | L17 | 0.4225 | 0.4160 | **0.000** | ✅ |
| example-testing | L27 | 0.4410 | 0.4210 | **0.000** | ✅ |
| adding-knowledge | L14 | 0.3938 | 0.4035 | 1.000 | ❌ |
| adding-knowledge | L17 | 0.4129 | 0.4090 | **0.000** | ✅ |
| adding-knowledge | L27 | 0.3650 | 0.3845 | 1.000 | ❌ |

**Bonferroni across 4 behaviours × 28 layers = 112 tests; corrected α ≈ 0.00045.** Backtracking and uncertainty clear it at every layer; example-testing at the 19 layers above; adding-knowledge precisely at L17–19. The 28-layer view is far more informative than the earlier 3-layer snapshot: it shows the two strong behaviours are significant *everywhere*, while adding-knowledge's manifold is a **tight 3-layer window**.

### C2/C3 (secondary/tertiary)
Cross-chain permutation (weaker null, lets the chain confound contribute) and a Marchenko-Pastur isotropic null (diagnostic for finite-sample inflation). Both used as comparison points; the chain-stratified null is primary.

---

## 5. What positive vs negative results would have looked like

| Scenario | Intrinsic dim | Curvature | Null | Conclusion |
|---|---|---|---|---|
| Venhoff right (single direction) | ≈ 1 | ≈ flat | n/a | **No project** |
| Flat high-dim subspace | ≈ PCA dim (50) | flat (ratios=1) | rejects | "Rich subspace" — interesting, not a classical manifold |
| Chain confound (negative) | high | high | **fails to reject** | Apparent structure is chain artefact — kills the project |
| **Our actual result** | **10–13** | **strongly curved** | **rejects (esp. 28/28 for 2 behaviours)** | **Curved low-dim manifold — strong positive** |

**We landed in the strong-positive cell.** All three batteries point the same way.

---

## 6. Relation to Goodfire's findings

### Goodfire paper 1: Manifold Steering (Wurgaft et al. 2026, arXiv 2605.05115)

- Llama 3.1 8B; cyclic/sequential concepts (days, months, ages, letters); last-token activations; 64-dim PCA + cubic-spline fit through centroids.
- Days-of-week form a clean 1-D circle (intrinsic 1, PCA 2); isometry score r = 0.99 (weekdays); manifold steering follows the curve where linear steering "teleports."

### How we compare

| Aspect | Goodfire | Us |
|---|---|---|
| Model | Llama 3.1 8B | R1-Distill-Qwen-1.5B |
| Concept domain | **Parametric cyclic concepts** | **Categorical reasoning behaviours** |
| Intrinsic dim | **1** (1-D circle) | **10–13** (moderately-curved multi-D manifold) |
| PCA d_eff (ambient) | 2 | 47–98 (layer-dependent) |
| Curvature evidence | Visualisation / spline only | **Three independent curvature diagnostics** |
| Null hypothesis test | None | **Chain-stratified null at every layer** |
| Sub-types | Each value is a single point | **k-means: no clean discrete sub-types (k=2, weak silhouette)** — continuous manifold |

### The key conceptual difference

Goodfire's concepts are **parametric** (Monday=0,…,Sunday=6) → a clean 1-D spline. Our behaviours are **categorical** with no ground-truth ordering. We originally expected each behaviour to be a *mixture of discrete sub-types* (backtracking = arithmetic-recheck vs strategy-pivot…), predicting intrinsic dim ≈ k_subtypes × dim_within. **Phase 5d did not find clean discrete sub-types** (Section 7), so the better description is a **single continuous curved manifold of intrinsic dim ~10**, not a union of sub-type clusters.

### What this means for the paper

Both works demonstrate the same fundamental claim — **neural representations form curved low-dimensional manifolds** — in complementary regimes: Goodfire's clean 1-D parametric circles, and our ~10-D moderately-curved manifolds for non-parametric reasoning behaviours. Our **methodological contribution** is the rigorous null hierarchy (per layer) and multi-estimator intrinsic-dim triangulation, which Goodfire (descriptive + interventional) does not have. If Phase 7 lands we add interventional evidence on top.

---

## 7. The adding-knowledge anomaly + the sub-type result (5d)

Adding-knowledge has the weakest signal: highest intrinsic dim (13.3), null rejects **only at L17–19**, highest PR.

**Phase 5d sub-type clustering (now complete):** at each behaviour's PR-trough layer, k-means with silhouette selection chooses **k = 2 for all four behaviours**, with **weak silhouettes (0.11–0.18)**:

| Behaviour | Cluster layer | best_k | silhouette | sizes |
|---|---|---|---|---|
| backtracking | L14 | 2 | 0.124 | 5536 / 4731 |
| uncertainty-estimation | L14 | 2 | 0.115 | 9692 / 7036 |
| example-testing | L27 | 2 | 0.180 | 2688 / 3141 |
| adding-knowledge | L17 | 2 | 0.123 | 2194 / 2833 |

Silhouettes this low (≪ 0.25) indicate **no substantial discrete cluster structure**. So:
- The earlier hypothesis that **adding-knowledge fractures into 4–6 sub-types is not supported.** It is diffuse (high intrinsic dim, weak null) but *continuous*, not a clean mixture.
- The behaviours are best modelled as **continuous curved manifolds**, which is fully consistent with the intrinsic-dim + curvature story (a smooth ~10-D manifold, not k discrete blobs).

**Implication for Phase 7:** the "sub-type steering" case study is reframed as **continuous manifold steering** (steer along the curved manifold vs the single linear direction), which is the more defensible framing given 5d.

**Not a project killer:** three of four behaviours pass the null cleanly (two at all 28 layers), and adding-knowledge still passes at its geometric peak (L17–19).

---

## 8. Why this is the central finding

1. **d_eff_70 ≫ 1 (45–98)** → Venhoff falsified.
2. **Intrinsic dim ≪ PCA dim (~5×, widening with depth)** → curved manifold, not high-dim flat subspace.
3. **Three independent curvature diagnostics agree** → curvature is real.
4. **Chain-stratified null rejects at every layer for 2 behaviours, 19/28 for a third, L17–19 for the fourth** → behaviour-specific structure, not chain confound.
5. **Geometry peaks at L14–17 while decodability is flat (85–92% everywhere)** → geometric structure is computationally localised even where the behaviour is linearly readable throughout.

The wedge between Venhoff (single direction) and Huang (aggregate manifold) is now empirically supported for **individual** reasoning behaviours. Phases 6/7 (steering) build on this once API credits are available.
