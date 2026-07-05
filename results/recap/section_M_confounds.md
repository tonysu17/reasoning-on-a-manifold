## M. Confounds Register, Results Ledger & Methodology Governance

This section documents the **governance layer** of the project — the three living trackers (`CONFOUNDS_AND_REMEDIATION.md`, `RESULTS_LEDGER.md`, `METHODOLOGY.md`) plus the historical-but-load-bearing `AUDIT.md`, `INVENTORY.md`, and `PROGRESS.md`. Together they answer one question the researcher must settle before committing real API+GPU spend on Phase-7 steering: **what can I actually claim, and what is still poisoned?** This is the "trust map." It is read-heavy on purpose: every quantity below is quoted from the repo, not reconstructed.

---

### M.0 The three-doc constitution (and why it exists)

The project deliberately splits its bookkeeping into three orthogonal living documents, each with a single canonical role, plus two upstream/legacy docs. The split is itself a methodological choice — it prevents the "status drifts across five files and they silently disagree" failure that the team had clearly already lived through.

| Doc | Canonical role | Quoted self-description |
|-----|----------------|--------------------------|
| `CONFOUNDS_AND_REMEDIATION.md` | *why a result is wrong/unproven + the sequenced fix plan* | "the **single source of truth** for *what is scientifically wrong or unproven*" |
| `RESULTS_LEDGER.md` | *what we found + its trust status* | "Canonical record of WHAT WE FOUND and its trust status … Supersedes the status role of `PROGRESS.md` + `INVENTORY.md`" |
| `METHODOLOGY.md` | *how we measure* | "Living spec. Canonical source for HOW we do things." |
| `AUDIT.md` | *software/numerical bugs* (mostly fixed 2026-06-05) | "upstream — bugs that *produced* the confounds" |
| `INVENTORY.md` | *which outputs are stale vs fresh* | "bookkeeping — this file says *why* they're stale" |

The dependency arrow matters: **`AUDIT.md` (bugs) → `CONFOUNDS_AND_REMEDIATION.md` (the scientific debt those fixes left, i.e. the re-runs owed) → `RESULTS_LEDGER.md` (what survived the re-run)**. `PROGRESS.md` is explicitly demoted — it carries a stale-banner and its empirical tables (the TwoNN 9.4–27.9 and curvature 0.13–0.24 numbers) are **superseded**.

The discipline is encoded as an instruction, not left to memory:

```text
CONFOUNDS_AND_REMEDIATION.md, §1 header note
> **Keeping this consistent across sessions:** update the **Status** column in §2/§3/§4 as
> items close — do not re-derive the list from scratch. … always read
> it before touching geometry/steering results.
```

---

### M.1 The central scientific risk (the thing the whole register is defending against)

The confounds register opens by naming the headline claim and the three quantities it rests on. This framing is the spine of everything downstream:

```text
CONFOUNDS_AND_REMEDIATION.md, §0 "The central scientific risk"
> *each individual reasoning behaviour occupies a **curved, low-dimensional, behaviour-specific
> manifold** in the residual stream — not Venhoff's single linear direction, not Huang's
> undifferentiated aggregate.*

That claim rests on exactly three load-bearing quantities …
1. intrinsic dim ≪ PCA d_eff  …
2. curvature  …
3. the chain-stratified null …
```

As written in 2026-06-06 the verdict was brutal and is worth quoting because it is the "before" state the project climbed out of:

```text
CONFOUNDS_AND_REMEDIATION.md, §0 (2026-06-06, now superseded)
**Net effect … no geometry number in this repository is currently citable.** Every geometry/steering
output is quarantined in `results/_STALE_pre_fix_20260605/`. The temporal-ordering pilot (Paper 1)
is the *only* standing empirical result.
```

That state is now **partly historical** — see M.4 (Gate-0). But the framing of the risk is permanent: the headline depends on **intrinsic-dim**, **curvature**, and a **null that tests the right statistic**, and the register exists to track whether each of those three is defensible.

---

### M.2 The confound register: CF-1 … CF-18

The register classifies each confound by **severity** (S0 = invalidates the headline; S1 = weakens a paper; S2 = reviewer-objection/scope) and **type** (`bug→fixed` = code fixed, re-run owed; `design` = needs a different experiment, no code fix possible). Below is the full register with each item's substance and its current status. The crucial 2026-06-20 reconciliation note:

```text
CONFOUNDS_AND_REMEDIATION.md, §2
> **2026-06-20:** … the "code FIXED, re-run owed"
> rows — **CF-1, CF-3, CF-4, CF-9, CF-13, CF-14, CF-15, CF-16** —
> were satisfied by the Gate-0 re-run (2026-06-18) and should be read as **CLOSED**
> (CF-1 closed as a *negative* — curvature is a chain artefact).
```

**The bug→fixed cluster (all now CLOSED by Gate-0 re-run, 2026-06-18):**

- **CF-1** (S0) — *Curvature diagnostic confounded.* `local_vs_global_dim_ratio` scored flat Gaussian data 0.29–0.85, the *same range* as the values reported as "evidence of curvature" (≈0.09–0.24). Geodesic symmetrization halved one-way edges. Fixed in code 2026-06-05. **CLOSED as a NEGATIVE** — curvature turned out to be a chain artefact (see M.3).
- **CF-4** (S0) — *Intrinsic-dim estimators biased high.* `twoNN` +35–50%, CBS `local_intrinsic_dim` ~3–4×. This was load-bearing because "intrinsic dim ≪ PCA dim" *is the whole result*. Fixed, re-run done. **CLOSED.**
- **CF-3** (S0) — *Null tests the wrong statistic.* The chain-stratified null was computed on `top_k_variance_ratio`, **never** on intrinsic dim or curvature (verified at `05b_geometric_diagnostics.py:192`). So the two quantities the headline turns on had *no significance test at all*. Code wired the right statistics into the null (`tier1_geometry_nulls.py`, `tail='lower'`); valid run done at Gate-0. **CLOSED.**
- **CF-9** (S2) — *Bootstrap CIs too narrow.* CIs resampled *derived* quantities (μ ratios, pairwise distances) → dependent → absurdly tight CIs like [0.575, 0.587]. Fixed to point-subsample bootstrap (commit `d7d147e`). **CLOSED.**
- **CF-13** (S0) — *Exact-duplicate activation rows (35–56%).* First-occurrence `str.find` matching bound every verbatim repeat to the first span, producing byte-identical rows that corrupt every kNN estimator (zero-distance neighbours) *and* concentrate within behaviour labels (so real matrices were duplicate-rich, permuted resamples duplicate-poor). Fixed via occurrence-aware `locate_annotation_offsets` + dedup. **CLOSED.** Important nuance that survives:

```text
CONFOUNDS_AND_REMEDIATION.md, §2 CF-13 status
the LB "real < null" signal **SURVIVES dedup** (backtracking 9.12 vs null 10.65, ~18 SD; …)
— duplicates understated absolute dims ~3× but did **not** manufacture the compression direction.
twoNN's nonsense values were the duplicate artifact. … dup-rows 51.9%→1.2%
```

- **CF-14** (S0) — *Row-provenance replay + silent proxy fallback.* Matrices carried no row ids; analyses replayed Phase-4 iteration and on any mismatch **silently fell back** to one-pseudo-chain-per-behaviour — under which the within-chain permutation is a *no-op* (null ≡ real, p≈1.0) while *looking legitimate*. This is the most insidious confound in the register: a broken null that returns a plausible-looking answer. Fixed via `row_index.json` sidecar + `require_aligned` hard-error + a vacuous-permutation guard in `src/nulls.py`. **CLOSED.** Empirically confirmed damage: the skip-blind replay mis-assigned **93.5%** of uncertainty-estimation and **79.0%** of example-testing rows for bounds-check-only consumers.
- **CF-15** (S1) — *05c probe leakage.* Plain `train_test_split` let sentences from the same chain straddle train/test; the leak inflates a null probe 0.50→0.99. The 83–93% probe accuracies (which feed layer triangulation) were upper bounds. Fixed with chain-grouped `GroupShuffleSplit`. **CLOSED** — and the new honest numbers dropped to 0.70–0.84 (RESULTS_LEDGER §A).
- **CF-16** (S1) — *Monte-Carlo machinery.* Unsmoothed p=count/B (B=100) → p=0 artifacts read against a Bonferroni threshold that was *never actually computed* (dead loader); null pool contained only the 4 target behaviours (deduction/initializing = 51% of sentences excluded). Mostly fixed (Phipson–Smyth smoothing, real Holm–Bonferroni). Null-pool scope remains a standing design caveat.

**The design cluster (no code fix possible — needs a different experiment; mostly STILL OPEN):**

- **CF-2** (S0) — *Chain confound / effective-N.* **This is named "the keystone."** Sentences within a chain are autocorrelated → the 5–16k points are not independent; the honest denominator is "independent chains" (hundreds, ~1000), not sentences. Every intrinsic-dim/curvature estimator assumes i.i.d., which is violated. Status: IN PROGRESS at the time of writing; the one-sentence-per-chain control is exactly what was run at Gate-0 and is what turned curvature into a negative.

```text
CONFOUNDS_AND_REMEDIATION.md, §2 "Notes on the non-obvious ones"
- **CF-2 is the keystone.** Everything else can pass and the headline still falls if the geometry is
  a property of *which chain a sentence came from* rather than *the behaviour*.
```

- **CF-5** (S1, **OPEN**) — *Linear apparatus, curvature claim.* "Manifold-projected" steering is a top-k PCA (linear-subspace) projection; curvature in 5b is measured *after* projecting to a top-k PCA subspace (`05b:135`). **A linear operator cannot test a curvature claim.** The two clean resolutions: (i) split the framing — Paper 2 claims **subspace** (linear-testable), Paper 3 claims **curvature** (and needs a genuinely nonlinear operator); or (ii) demote curvature-steering to descriptive. **This directly touches Phase-7** (see M.6).
- **CF-6** (S1) — *Mean-pooling destroys the trajectory.* Activations are mean-pooled over the first ~10 tokens; the manifold claim is about a *trajectory*. Status: **DOWNGRADED to "resolved (keep mean)"** in METHODOLOGY §1 — the `[onset−1:+10]` mean-pool was verified to be *exactly Venhoff's published recipe*, so it is the field standard, not a flaw. (Optional position-sweep robustness remains gated on a small re-extraction.)
- **CF-7** (S1, **mitigated**) — *Single unvalidated annotator.* Labels (the dependent variable) come from one LLM (Sonnet 4.5). The R2.1 measurement half is DONE (κ = 0.436/0.350/0.345); the R2.2 replication half is **DONE 3-way** (Sonnet/Qwen3/Nova) — geometry replicates despite the disagreement. Caveat retained: the variance-ratio *specificity* null is still single-annotator.
- **CF-8** (S1, **OPEN**) — *50% truncation.* 50.2% of chains hit the 8192 cap and lack a closing `</think>`; truncation rate correlates with category (lateral 95%, spatial 71%) → correlates with behaviour mix. Undecided: raise cap / stratify / drop categories.
- **CF-10** (S1, **OPEN/unrun**) — *Activation-patching proxy.* The original `behaviour_marker_logprob` scored behaviours by lexical tokens ("wait"/"actually"). The **attribution-patching fix (07c) was itself CONFOUNDED**: a metric read at a fixed late layer (L27) + a first-order gradient gave a monotone early→late ramp for all 4 behaviours (read-out-proximity artefact) — no interior peak, no per-behaviour signal. De-confounded redesign (07d, forward-pass intervention) is **CODED but UNRUN**. **This is the single most steering-relevant open confound** (see M.6).
- **CF-11** (S0 for safety) — *Safety ∦ capability.* A "safety manifold" may be a difficulty/capability manifold. Only matters for the deferred safety arm. OPEN; capability control mandatory there.
- **CF-12** (S1) — *Cross-model "bootstrap" was a Gaussian-from-CI-width, not a bootstrap.* Relabelled `p_normal_approx`; true two-sample bootstrap added but needs producers to persist per-resample arrays. Only touches the knowledge-creation H4 arm.
- **CF-17** (S1, **FIXED but Phase-7 UNRUN**) — *Phase-7 design debt (pre-spend).* A dense bundle of pre-registration failures, all patched: 2048→8192 cap; shared vanilla baseline (vanilla was regenerating byte-identically at every α — 1600 vs 50 needed generations); norm-matched `random_direction` control arm added (there was *no causal baseline*); missing re-annotations were silently scored 0.0 (deflating the most-destructive arm); the "held-out" set was `tasks[-50:]` of a category-blocked file = **100% lateral_thinking**, now category-stratified and persisted to `eval_task_ids.json`; two builders were clobbering the same vector files. **This is the most directly Phase-7-relevant confound** (see M.6).
- **CF-18** (S1, **OPEN**) — *Annotation-layer integrity.* (a) chunk-merge overlap dedup silently deletes genuine recurrences of short sentences — exactly backtracking/uncertainty markers — and 936/1000 chains were chunked; (b) unknown annotator labels are **irreversibly coerced to `deduction`** (`src/annotation.py` ~:305) with no persisted trace, contaminating the probe "other" class; (c) 7 `annotation_complete=False` records (incl. empty CREA_026) were consumed unfiltered.

---

### M.3 The negative-results register (NR-1 … NR-5)

Distinct from confounds: these are findings that came out **null or weak**. The discipline here is to report the honest version and not let a downstream plan keep assuming the positive one.

| ID | Negative result | Status |
|----|-----------------|--------|
| **NR-1** | Power analysis wrote an all-NaN table masquerading as a ">max tested" null | **CLOSED** — re-run non-NaN; curvature null well-powered (detectable at N≥500; pools ~600–900 chains) |
| **NR-2** | **adding-knowledge is the weak behaviour** — highest intrinsic dim (13.3), smallest N (5 027), null at only 3/28 layers | **STANDING caveat** — and it's the one behaviour the cross-domain/knowledge-creation story leans on |
| **NR-3** | **No discrete sub-types** (k=2 for all four, silhouette 0.11–0.20) | **RESOLVED** — reframed as continuous-manifold steering |
| **NR-4** | **example-testing null gap (L7–14)** — significant 19/28 but mid-layer hole | **STANDING** |
| **NR-5** | **d_eff is high (45–98), not low** | **STANDING** — a skeptic reads "high-dimensional/distributed"; the rescue is entirely the intrinsic-dim gap |

The crucial honest reframe that the register forces — the curvature **negative**:

```text
CONFOUNDS_AND_REMEDIATION.md, §0 status banner (2026-06-20)
❌ **Per-behaviour curvature is a CLEAN, WELL-POWERED NEGATIVE** — curved on full data, ≈1.0 (flat)
   at one-sentence-per-chain (within-chain autocorrelation). Power analysis confirms ~600–900 chains
   could have detected it. (CF-1 closes as a *negative*; CF-5's linear-apparatus worry is moot.)
```

This is the single most important governance fact in the whole project: the **curvature half of the headline died honestly** — it was a chain artefact (CF-2 acting through CF-1). What survived is **low intrinsic dimension** (a subspace/compression claim), not curvature. NR-2/NR-4/NR-5 sharpen the picture: adding-knowledge fails behaviour-specificity *everywhere* (p=1.0), example-testing is specific only at L27.

---

### M.4 Gate-0: the regeneration gate that flipped "nothing citable" → "geometry citable"

The remediation plan is sequenced as a **gate + four tiers**, ordering principle "cheapest-to-falsify first, do not spend GPU/API until the free CPU checks survive or force a reframe." The gate is the hinge:

```text
CONFOUNDS_AND_REMEDIATION.md, §4
### Gate 0 — Regenerate on the fixed code ✅ COMPLETE 2026-06-18 (was BLOCKING)
Until Gate 0 completes, **do not** write results prose, refresh figures, or cite any geometry number.
> **✅ DONE 2026-06-18.** G0.0 (re-extraction; dup 35–56% → ~1%; `row_index.json` sidecars),
> G0.1 (PCA / 05c / triangulation / 05d / 05b re-run), G0.2 (power table, non-NaN), G0.3 (figures +
> numeral-lift into ch07), G0.4 (sanity gate …). Geometry numbers are now citable.
```

Gate-0 (re-extraction on the cluster GPU + the CPU downstream) is what closed the entire `bug→fixed` cluster in one pass. The **G0.4 sanity gate** is the part worth dwelling on: it explicitly authorized the project to *stop and reframe* if curvature scored flat — and it fired:

```text
CONFOUNDS_AND_REMEDIATION.md, §4 Gate 0
- **G0.4** Sanity gate: if the fixed curvature diagnostic now scores the real data ≈1.0 (flat), **stop and
  reframe** … that would mean CF-1 was fatal, not cosmetic.
```

That is exactly what happened — and the project absorbed it without aborting the downstream (it became a clean negative rather than a crisis). This is good methodological hygiene: a pre-registered abort condition that, when triggered, produced a *reportable* result rather than a fudge.

The downstream tiers (Tier 1 internal robustness; Tier 2 external/annotator; Tier 3 causal + the CF-5 apparatus mismatch; Tier 4 safety) carry a decision gate after Tier 1: *"If intrinsic-dim and curvature survive R1.1–R1.3 → the geometry is defensible, spend on Tiers 2–4. If they don't → reframe the central claim now, before any GPU/API."* Phase-7 steering lives in Tier 3 and is the only Tier-1/2-cleared work still unrun.

---

### M.5 The stale-results quarantine (what is physically walled off)

The quarantine is a directory-level discipline, not just a label. The one date that matters:

```text
INVENTORY.md
## The one date that matters: **2026-06-05**
Any *geometry / steering* output produced **before** that date is from biased or
non-reproducible estimators and is **stale — must be regenerated**.
```

**`results/_STALE_pre_fix_20260605/`** holds 12 directories moved there 2026-06-06 (nothing deleted; restorable with `mv`): `geometric/`, `robustness/`, `pca/`, `steering_vectors/`, `composition/`, `saturation_predictions/`, `cross_layer/`, `triangulation/`, `clustering/`, `cbs/`, `trajectory/`, `power_analysis/`. These remain the canonical **"before" snapshot** — the regeneration driver `run_rerun_local.sh` must not overwrite them.

Other quarantined / archived material:
- **`tier1_robustness/R1-1.5B/geometry_nulls_layer27.*`** — the 2026-06-08 keystone-null run, **SUPERSEDED** because it predates the CF-13 dedup; its TwoNN "dim" of **0.168** is a zero-distance artefact, its p=0.0000 cells are unsmoothed.
- **`_archive_run1_20260528_*`** — pre-May-28-cap-fix archives (historical).
- The `supervisor_meeting/` bundle is **MIXED, kept intact**: the synthetic methodology explainers (`viz1/2/3`, `fig3/4/5`) are VALID; the data figures (`fig1/2/6/7/8/9`) are stale (PCA/TwoNN/curvature). The RESULTS_LEDGER **DO NOT CITE** list adds: TwoNN intrinsic-dim values, PCA d_eff≥80%, tangent-space-variation curvature, any full-data curvature ratio presented as evidence *of* curvature, and the first (flawed, AUC 0.61) predictive-geometry gate.

---

### M.6 The "what can I claim" map — citable vs do-not-cite vs needs-rebuild

This is the practical payload of the whole section, drawn from `RESULTS_LEDGER.md`. Legend: ✅ solid · ❌ clean negative · 🟧 mixed · ⬜ built/unrun · ⚠️ caveat. **All numbers are on the clean Gate-0 re-run.**

**SAFE TO PUT IN THE THESIS (✅ citable):**

```text
RESULTS_LEDGER.md, §A
Low **intrinsic dim**            ✅  corr-dim ≈ 5.9 / 6.2 / 6.0 / 7.7 (back/unc/ex-test/add-know) in 1536-D;
                                     stable across full / random-sub / one-per-chain (sd ≈ 0.1)
Compression gap (intrinsic≪PCA)  ✅  corr-dim ~6–8 vs PCA d_eff_70 ~38–83; PR(L27) ~17–26
Linear decodability (probes)     ✅ but flat  0.70–0.84 @ every layer (old leaky 0.83–0.93 was train/test leak)
Layer concentration (PR-trough)  ✅  peaks 16/16/16/12 (back/unc/add-know/ex-test)
```

Plus from §C: **R2.1 inter-annotator agreement** (κ 0.436/0.350/0.345; span-F1 0.26–0.31) and **R2.2 geometry replication 3-way** (intrinsic-dim + curvature-as-chain-artefact replicate across Sonnet/Qwen3/Nova despite κ=0.35–0.44). And §B: **composition** (4 directions strongly non-orthogonal, off-diag |cos| mean 0.57, max 0.77; each ≈ reconstructable R²=1.0 from the other 3).

**HONEST NEGATIVES / MIXED (report as such):**

```text
RESULTS_LEDGER.md, §A
Per-behaviour **curvature**      ❌  local↔global 0.43–0.69 (full) → ≈1.0 (one-per-chain) = within-chain artefact
Behaviour-specificity (null,B=2500) 🟧 2/4  back & unc p<.001 @ all layers; ex-test only @ L27;
                                            **add-know p=1.0 everywhere**
Discrete sub-types               ❌  best k=2, silhouette 0.18–0.20 → one continuous region each
```

**DO NOT CITE (quarantined / dropped):**

```text
RESULTS_LEDGER.md, §E
- results/tier1_robustness/.../geometry_nulls_layer27.md (pre-dedup; TwoNN "0.168" artefact; unsmoothed p)
- **TwoNN** intrinsic-dim values. **PCA d_eff ≥80%**. **tangent-space variation** curvature.
- Any **full-data curvature ratio** presented as evidence *of* curvature (it is the artefact).
- The **first/flawed** predictive-geometry gate (AUC 0.61).
- Everything under results/_STALE_pre_fix_20260605/ and results/_archive_*.
```

**BUILT BUT UNRUN (⬜) — the entire causal layer:**

```text
RESULTS_LEDGER.md, §B
**Phase 7 steering eval (single vs manifold vs random)**  ⬜ **UNRUN**
  results/eval/ absent. The headline *causal* result. Needs cluster GPU + go-ahead
Saturation predictions (pre-registered α*)  ⬜ sealed, untested  α* ≈ 0.99/0.97/0.96/1.06
```

**NEEDS REBUILD (⚠️, stale vs current config):**

```text
RESULTS_LEDGER.md, §F
1. **`-peak` steering vectors + 05d clustering** — built at OLD peak_layers (14/14/17/27);
   config reconciled to 16/16/16/12 on 2026-06-20 → rebuild before relying on them.
   (Decide example-testing layer first: 12 PR-trough vs 27 specificity/Huang.)
2. Canonical steering metadata.json — git_commit:null provenance (cosmetic).
4. **Phase 7** — never run.
```

---

### M.7 Connection to the imminent steering decision

This governance layer bears on the Phase-7 go/no-go in four concrete ways. The researcher should treat these as the live items.

**1. Layer choice is NOT held out and the causal-layer probe is unresolved.** The single sharpest steering-relevant fact in the register is that *every* attempt to pick the steering layer causally has either been confounded or remains unrun:
- 07c attribution patching **RAN and is CONFOUNDED** (monotone ramp to L26–27 for all behaviours = read-out-proximity artefact). **Do not pick layers from it.**
- 07d (the de-confounded forward-pass redesign) is **CODED, UNRUN**.
- The de-confounded *smoke* run (per the RESULTS_LEDGER 2026-06-21 change log) found a universal **L27 KL-spike** (a last-layer direct-logit-edit artefact a logprob/KL proxy cannot escape) **plus** Venhoff-aligned mid peaks revealed only by the random-null (backtracking L11, uncertainty L16, example-testing L19, adding-knowledge L16). The proxy *cannot settle mid-vs-late.*

The adopted decision, which the steering run must honor:

```text
RESULTS_LEDGER.md / METHODOLOGY.md §5
**fold layer into Phase 7**: build at per-behaviour mid + L27, let free-generation
steering-effectiveness decide.
```

And the honesty constraint for the paper (CF-17 residual): *"steering-LAYER choice was informed by full-corpus analyses (+ Huang's published layer 27) — note in the paper, do not claim layer selection is held out."* The 50-task **task** hold-out is genuine (row-provenance, `eval_task_ids.json`); the **layer** dimension is not held out.

**2. CF-5 must be resolved as framing before writing Phase-7 prose.** The "manifold-projected" vector is a *linear* top-k PCA projection. Since curvature is now a **negative**, CF-5's "linear apparatus can't test curvature" worry is explicitly **moot** — but only if Phase-7 is framed as a **subspace** (granularity-of-direction) experiment, *not* a curvature experiment. The clean story is: single direction vs k-dimensional subspace projection, swept over k∈{1,3,5,10,auto}. Per METHODOLOGY §4, the current build only steers `manifold_projected` at **auto_k**; the k-sweep is an open methodology gap to close before the run.

**3. The control arm is norm-matched, not energy-matched.** A documented weakness that limits causal language:

```text
METHODOLOGY.md, §4
`random_direction` is **norm-matched, not energy-matched** (|rᵀh| smaller for a random r)
→ describe as a generic-perturbation floor, not an energy-matched twin.
```

(The 2026-06-21 hardening adds an energy-matched random arm — "norm-matched was ~19× too weak" — so confirm which control the run actually uses.)

**4. The dependent variable (annotation) is still single-annotator-fragile at the point Phase-7 reads it.** Phase-7's primary endpoint is `behaviour_fraction` from **re-annotation**. CF-18 (chunk-merge silently deletes recurrences of exactly the backtracking/uncertainty markers; unknown→deduction coercion) is **OPEN** and directly affects that re-annotation. The register's instruction — *"Fix chunk-merge before any re-annotation"* — is a live prerequisite, not a footnote, since Phase-7 spends ~$260 of annotation budget on this exact label.

---

### M.8 Critique of the governance layer itself

The bookkeeping is unusually disciplined for a thesis project (three orthogonal canonical docs, a pre-registered abort gate that actually fired, ~258-test regression suite). But several things should make the researcher uneasy before spending:

- **The keystone (CF-2) is the only defence and it is single-annotator at the test that matters.** Behaviour-specificity replicates *geometrically* across annotators (R2.2), but the **variance-ratio specificity null is still single-annotator**. The one statistical test that says "this is the behaviour, not the chain" has never been re-run under Qwen3 or Nova labels. The register admits this twice. Since the specificity result is already **MIXED 2/4** (add-know fails everywhere, ex-test only at L27), a second annotator could plausibly knock example-testing out entirely — and the headline would shrink to "2 of 4 behaviours are specific."

- **The survivors are a *subspace + low-dim* story, and NR-5 is a standing threat to even that.** d_eff is 45–98 ("high-dimensional/distributed," not "manifold"). The entire rescue is the intrinsic-dim gap (corr-dim ~6 vs d_eff ~38–83). That gap is real and survives the chain control, but the register itself flags that a skeptic reads the raw d_eff and walks away. Leading with participation ratio + intrinsic-dim-with-a-null is the mandated framing — do not let a figure show d_eff in isolation.

- **CF-17 is "FIXED" in code but the fixes have never executed together.** Every Phase-7 pre-spend patch (8192 cap, shared vanilla, random arm, stratified split, missing-annotation accounting, builder separation, transformers-5.x hook crash) is verified *individually by tests*, but the integrated run has not happened. The history of this project (the 2026-06-12 wave, the 07c confound discovered only *after* it ran on the cluster) is that confounds surface when code meets real data, not in unit tests. A `$0` smoke run of the full Phase-7 path on real GPU — which the steering-status memory says PASSED — is the right de-risk, but it ran *without annotation*, so the annotation-coupled failure modes (CF-18, n_missing accounting) are still untested end-to-end.

- **The `-peak` vectors are stale against the reconciled config** (built at 14/14/17/27, config now 16/16/16/12) and the example-testing layer is *explicitly undecided* (12 PR-trough vs 27 specificity/Huang). Phase-7 cannot proceed on the `-peak` build without a rebuild, and the layer ambiguity for example-testing is unresolved — which compounds with point (1).

- **Provenance gaps remain.** Canonical steering `metadata.json` ships `git_commit:null`; 05c / triangulation / some figure scripts are un-stamped. Cosmetic, but for a thesis defending a contested geometry claim, every cited number should trace to a commit + seed + input hash.

**Sharpest single critique:** *The project's entire defence against the keystone confound (CF-2 — "it's the chain, not the behaviour") rests on a behaviour-specificity null that is (a) still computed under a single, only-fair-agreement annotator (κ≈0.35–0.44) and (b) already failing for 2 of the 4 behaviours. The geometry "replicates across annotators" only at the descriptive intrinsic-dim level — the actual significance test that licenses the word "behaviour-specific" has never been re-run under a second annotator's labels. Phase-7 is about to spend GPU+API to make a causal claim on top of this null, while the most directly relevant open confounds (CF-10 causal-layer probe UNRUN/confounded, CF-18 re-annotation integrity OPEN, CF-5 linear-vs-curvature framing unresolved, energy-matched control unconfirmed) all sit upstream of that spend.*
