# Plan — Thesis Writing (handoff for the thesis-refinement session)

> **Handoff from the experimental/repo side, 2026-06-21.** The thesis lives in its own private
> repo; this is a queue of WRITING changes for the session doing thesis refinement. The repo
> side will NOT edit `thesis/`. Each item says **what / why / where / what numbers to use**.
> Numbers come from [`RESULTS_LEDGER.md`](RESULTS_LEDGER.md); methods from
> [`METHODOLOGY.md`](METHODOLOGY.md). **Do not invent numbers — pull from those.** The
> number-embargo discipline in `thesis/_planning/STYLE.md` (`\gateZero` / `\phaseSeven`) still
> applies; steering / Phase-7 numbers stay embargoed until that experiment runs.

Priority tags: **P0** correctness/honesty · **P1** clarity · **P2** additions/polish.
"Verify" = check the current draft first; it may already be done.

---

## P0 — correctness (must match the cleaned Gate-0 re-run)

**T1. Curvature reframe wording — `ch02`, `ch07`.** State the per-behaviour object as a
**low-dimensional (≈linear) subspace**, NOT a "curved manifold." Per-behaviour curvature is a
**clean, well-powered negative** (it was within-chain autocorrelation; collapses to ≈flat at
one-sentence-per-chain). Frame curvature as a **trajectory-level** property (a feature of the
*path a chain traces over time*, not the static cloud) — which is where the predictive-geometry
extension and the "flowing logics" literature put it. Note the upside: flatness *justifies* the
linear steering apparatus (resolves the CF-5 instrument/claim mismatch).

**T2. Behaviour-specificity is 2/4, not blanket — `ch07`.** Say plainly: backtracking &
uncertainty-estimation are behaviour-specific at all layers (chain-stratified variance-ratio
null, B=2500, p<.001); **example-testing only at L27**; **adding-knowledge fails everywhere
(p=1.0)**. adding-knowledge is also the rarest / highest-dim / weakest — present as the honest
outlier, not buried.

**T3. Drop poor metrics from the results chapter — `ch07`.** Remove or relegate to an appendix:
**TwoNN** intrinsic dim (duplicate/subsample-unstable; disagreed with Levina–Bickel on sign),
**PCA d_eff at ≥80% variance** (saturated at the top-100-component cap — floors, not estimates),
and **tangent-space-variation** curvature (~67–70° across everything, uninformative). Lead
intrinsic dim with **correlation dimension** + **participation ratio**.

**T4. Annotator-robustness UPDATE (NEW result) — `ch05`/`ch07`.** The 2-way replication landed:
**intrinsic dim + curvature REPLICATE Sonnet ↔ Qwen3-235B** (cdim within ~0.6, same ordering,
keystone PASS in both; curvature-artefact holds) **despite only κ=0.44 label agreement** — i.e.
the geometry is not an artefact of one labeller. Caveats to state: the Qwen3 arm ran on
analysis-side dedup (dup 33–52%) not clean re-extraction (Sonnet 1%), so absolute values are
modestly shifted; the **variance-ratio specificity-null** replication and the **Nova-Pro** third
arm are still pending. (Pull exact numbers from RESULTS_LEDGER §C.)

**T5. Annotator naming — `ch04` (verify).** Confirm the corpus/methods name the annotators as
**Claude Sonnet 4.5 (primary), Qwen3-235B, Nova-Pro — via the AWS Bedrock proxy, not gpt-4o**.
(`ch04:200–206` already says Sonnet; `brief_methods.md` already flags the config `gpt-4o` string
as stale. Just confirm the 3-way robustness set is named where the κ is reported.)

---

## P1 — methodology clarity

**T6. Steering-vectors primer (NEW subsection) — `ch08`.** A self-contained primer:
- **Contrastive diff-of-means** (Venhoff-style): `r = mean(ON) − mean(OFF)`, unit-norm, where
  `OFF` = the *other three behaviours* (behaviour-vs-other contrast).
- **Manifold-projected (our method, Huang-adapted):** the same `r` orthogonally projected onto
  the behaviour's own top-k PCA subspace, renormalised.
- **Application is projective** (Huang Eq. 3): `h' = h − α·(rᵀh)·r` (subtract = suppress);
  α=1 removes the whole r-component (ablation), α>1 over-removes. `r` unit-norm ⇒ α is the scale.
- Position it against **CAA (Rimsky), ActAdd (Turner), Panickssery, Huang** and explain the
  norm-matched **random_direction** control (the floor that licenses causal language).
  Source: METHODOLOGY §3–§4.

**T7. Steering methodology section — `ch08`.** Align with METHODOLOGY §4: arms
(vanilla/single/manifold/random); **sweep k ∈ {1,3,5,10,auto}** (not just auto_k); α-sweep
{0.3…3.0}; metrics = on-target Δ behaviour-fraction **plus** off-target damage (repetition /
degenerate / mean-tokens) **plus** task-accuracy preservation **plus** cross-behaviour leakage;
headline = *effect at matched damage* (suppression-vs-damage Pareto), paired over the held-out
tasks, both arms beating random.

**T8. Pooling caveat in methods — `ch04`/`ch05`.** State that activations are **mean-pooled over
the first ~10 tokens** of each behaviour span, that this **deviates from standard last-token
steering**, and that the direction is **pooling-dependent**: cos(mean, last) = **0.57–0.87**
(backtracking worst, 0.57). Note Phase 7 sweeps **mean vs last** and reports sensitivity (CF-6).

**T9. Layer-selection honesty — `ch08`.** Be explicit that the steering layer is currently a
**descriptive** choice (Huang's L27 / participation-ratio trough), not causal. The principled
fix is **attribution patching** to pick each behaviour's causal layer. Note the empirical signal
that **more semantic behaviours are specific at later layers** (example-testing only at L27;
adding-knowledge nowhere) — consistent with complex behaviours emerging deeper. Don't claim the
layer is held-out.

**T10. MI primer framing — `ch03` (verify/expand).** `ch03` already has MI methods (SAEs,
crosscoders, activation patching) and `§sec:bg-faithfulness`. Consider a short **self-contained
primer up front**: what MI is, *why* (unreliable reasoning traces), and the main methods
(SAEs, activation/attribution patching) — so a non-MI reader gets the motivation before the
faithfulness argument.

**T11. Trace-vs-mech framing — `ch01`/`ch03`/`ch08`.** Make the epistemic point explicit: the
steering result is a **behavioural-causal** result (we measure an observable surface behaviour in
the trace, which is robust to *faithfulness* worries), and the **mechanistic layer is what makes
the direction meaningful** and guards against "we just injected a lexical tic" (CF-10). Spell out
the loop: annotations(trace) → labels → geometry/direction → steer → trace changes; closed by a
trace-independent mech readout (logit shift / internal subspace movement / task accuracy).

---

## P2 — additions / figures

**T12. Base ↔ distilled ↔ ±steering continuum (figure) — `ch08`.** Propose a figure placing
steering between the two natural endpoints: base model (Qwen2.5-Math-1.5B, behaviour-poor) <
negative-steered distilled < vanilla distilled < positive-steered distilled. If negative steering
pushes the distilled model's behaviour fraction toward the base level, that ties steering to the
**post-training-is-the-hinge** thesis ("base models know *how*, thinking models learn *when*";
negative steering ≈ un-learning the *when*). Caveat: base/distilled are different weights → a
reference frame, not a within-model intervention.

**T13. Inter-behaviour separability gap — `ch07`.** Note that the clustering result is
**within-behaviour** sub-typing only (no discrete sub-types: silhouette 0.16–0.18). There is **no
between-behaviour** separability metric (ARI / kNN-purity) in the repo, so "the four behaviours
are mutually distinct" should lean on the probes + variance-ratio null, or add such a metric.

---

## Cross-checks for the writer
- Every geometry number traces to `RESULTS_LEDGER.md` §A (Sonnet) / §C (replication). Steering /
  Phase-7 numbers do not exist yet → keep behind the embargo macros.
- Keep the curvature claim consistent across ch02 ↔ ch05 ↔ ch07 (subspace ✔ / per-behaviour
  curvature ✘ chain-artefact / curvature → trajectory level).
- The "DO NOT CITE" list in `RESULTS_LEDGER.md` §E must not appear as evidence anywhere.
