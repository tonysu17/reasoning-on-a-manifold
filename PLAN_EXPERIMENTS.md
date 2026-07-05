# Plan — Experiments (repo side)

> **Action queue for the experimental side, 2026-06-21.** Everything we've decided to verify,
> build, run, or rebuild — sequenced toward the headline causal result (**Phase 7: single vs
> manifold steering**). Methods spec = [`METHODOLOGY.md`](METHODOLOGY.md); status/results =
> [`RESULTS_LEDGER.md`](RESULTS_LEDGER.md); confounds = [`CONFOUNDS_AND_REMEDIATION.md`](CONFOUNDS_AND_REMEDIATION.md).
> Update those three as items land. **Tell-Tony-first before any GPU/$ run.**

Tags: **P0** blocks Phase 7 · **P1** important · **P2** hardening/cleanup.
Compute: **[code]** pure CPU/no model · **[GPU]** needs the model · **[$]** API re-annotation spend.

---

## Progress log — 2026-06-21 (eve)

**Done this session (code / CPU / research; 41 tests green):**
- ✅ **E1** — off-target leakage folded into `aggregate_results` + new `aggregate_accuracy()` in `src/evaluation.py`; 10 new tests.
- ✅ **E2 — pooling question RESOLVED by deep research.** Our `[onset−1 : +10]` mean-pool **is exactly the Venhoff (arXiv:2506.18167) recipe** — verified in their code `layer_outputs[:, start-1:min(end-1,start+10)].mean()` — the published method for the *same* model family + behaviours. So **keep mean-pooling; it is the field standard, not a flaw.** (Huang 2505.22411 = *final-token* diff-of-means + PCA k=10 @ L27.) Optional robustness add: a **position sweep** (mean / onset / pre-onset−k / last) — Arditi treats position as a validated hyperparameter; 2507.12638 finds backtracking signal sits *pre*-onset. Sweep needs a small GPU re-extraction → gated.
- ✅ **E3 (code; run gated)** — attribution patching implemented: `src/attribution_patching.py` + runner `07c_attribution_patching.py`, with a **CF-10-fixing behaviour metric** (projection onto the steering geometry, not lexical markers) + positional alignment; 24 synthetic tests, math validated vs brute force. Writes `results/patching/R1-1.5B/{attribution_curves,pilot_effect_curves}.json` → fills triangulation's missing 3rd signal. **This is exactly Venhoff's layer-selection method.**
- ✅ **E10** — both DO-NOT-CITE files bannered in place (referenced, so not moved); `_STALE_` README updated.
- ✅ **E11** — sub-type clustering rebuilt at reconciled layers (16/16/16/12); silhouette 0.18–0.20, best_k=2 → "no discrete sub-types" **still holds**; old run archived to `R1-1.5B__prerebuild_jun18/`.

**⚠️ E3 DONE but result CONFOUNDED (2026-06-21 eve).** The full attribution sweep ran clean on the
cluster (4 behaviours × 20 pairs × 28 layers, bf16, ~60 min; `results/patching/R1-1.5B/`). **But all
four behaviours peak at L26–27 with a monotonic early→late ramp** = a **read-out-proximity artifact**
(metric reads at L27 = the last layer, so patching near L27 inflates), NOT a per-behaviour causal
layer. A real causal layer would show an interior peak; Venhoff's comparable method finds mid-layers
(15–18). So this CANNOT select per-behaviour layers as-is (not a bug — math validated 1e-8; it's the
read-out choice). **Decision: do NOT pick layers from this; fold layer selection INTO Phase 7** —
build E4 vectors at BOTH the mid PR-trough (**16**) and late/Huang (**27**) and let the actual
steering-effectiveness comparison choose empirically (the gold standard; Huang's downstream sweep also
landed on L27 for R1-1.5B). Optional later: redesign the attribution read-out (downstream/output,
Venhoff-style) + re-run.

**Gated on Tony's go-ahead (GPU / $ / cluster):** E2 position-sweep re-extraction · E4 final vector
build (causal layer × pooling) · E6/E7 Phase 7 + analysis ($ re-annotation) · E8 base↔distilled ·
E9 Nova-Pro finish + optional clean re-extract.

---

## Critical path to Phase 7 (the one causal experiment)

```
E1 metrics ─┐
E2 pooling ─┼─► E4 build vectors (chosen layer × pooling × k) ─► E6 Phase-7 run [GPU+$] ─► E7 analysis
E3 layer ───┘                                                          ▲
                                                                E5 eval-set size
```
Phase 7 should not run until **E1–E4** are done (verify-before-spend). E3 (layer) and E2
(pooling) are the two "is the apparatus even right?" gates.

---

## TRACK A — Pre-Phase-7 verification & build (do first)

**E1. Implement the two missing Phase-7 metrics — `[code]` P0.**
- **Task-accuracy preservation** (GSM8k / per-category correctness vs vanilla) — the real
  "less destructive" measure; `src/evaluation.py` docstring claims it but `aggregate_results`
  doesn't compute it.
- **Off-target leakage** — the other 3 behaviours' fractions from the *same* re-annotation
  (free; the annotator already labels every sentence). Steering A must not inflate B.
- *Where:* `src/evaluation.py` (`aggregate_results`) + a correctness checker.
- *Done when:* both appear per (behaviour, method, α) cell; unit tests added.

**E2. Pooling verification — mean vs last — `[code]` then `[GPU+$]` at E6. P0 (gate).**
- The direction is pooling-dependent: cos(mean,last) = 0.57 / 0.83 / 0.87 / 0.81
  (back / unc / ex-test / add-know); under `last`, d_eff_70 is censored at 100 (much
  higher-dim). Span-mean-over-10 deviates from standard last-token steering.
- *Action:* the extraction already stores mean **and** last activations — **build both vector
  sets** `[code]` and, at E6, **run both as arms** so the single-vs-manifold result is reported
  under each. Default expectation: `last` (standard). Also verify what Venhoff actually pooled
  (their appendix) and record it.
- *Done when:* mean- and last-pooled vector sets exist; Phase-7 includes a pooling factor.

**E3. Steering-layer validation via attribution patching — `[GPU]` P0.**
- Current layer is descriptive only (Huang L27 / PR-trough); `results/patching/` is **absent**
  (pilot never produced output), and L27 is borrowed from a different setup. Behaviours differ
  and the semantic ones are specific later (example-testing only L27; add-know nowhere).
- *Action:* (a) **validate the behaviour-effect metric first** — CF-10: replace the lexical
  "wait/actually" proxy + add positional alignment across chains (`src/activation_patching.py`,
  `07b_activation_patching.py`); (b) run **attribution patching** (grad·(corrupt−clean), ~1
  backward pass) per layer per behaviour over all 28 layers; (c) pick each behaviour's **causal**
  steering layer; (d) feed the result into triangulation's empty 3rd slot.
- ⚠️ Don't assume "later is better" blindly — final layers are next-token-dominated; let the
  patching curve decide.
- *Done when:* per-behaviour causal layer chosen with evidence; example-testing 12-vs-27
  question resolved on causal grounds (supersedes the config note).

**E4. Rebuild steering vectors at the chosen config — `[code]` P0.**
- Build `single` + `manifold k∈{1,3,5,10,auto}` at: **the E3 causal layers** × **{mean, last}
  pooling** (E2), on the true 50-task hold-out.
- Replaces the stale `-peak` build (currently at the old 14/14/17/27 layers) and the
  pooling-unswept canonical build.
- *Done when:* vector sets + metadata stamped with git commit (fixes the `git_commit:null`
  provenance gap); old sets archived.

**E5. (Optional) enlarge the eval hold-out — `[code]` P1.**
- 50 tasks is workable (paired design) but tight; generation is cheap, re-annotation is the $.
  Consider 100–150, category-stratified, re-persisted to `eval_task_ids.json`. Vectors must
  exclude the enlarged set (re-run E4). Decide vs budget.

---

## TRACK B — Phase 7: the causal experiment `[GPU + $]`

**E6. Run the steering experiment — P0.**
- Arms: vanilla (shared) · single_direction · manifold (sweep **k**, not just auto_k) ·
  norm-matched random_direction. Factors: **α ∈ {0.3,0.5,0.7,1.0,1.5,2.0,3.0}** × **pooling
  {mean,last}** × per-behaviour layer. Mode = subtract (primary); add a small **amplify (add)**
  arm for bidirectional evidence (P1). Greedy, 8192-token cap.
- Re-annotate steered outputs with **Sonnet via the proxy** (NOT gpt-4o). Track n_missing/n_empty.
- *Cost:* GPU ~hours + the re-annotation $ (the only API spend). **Tell-Tony-first.**

**E7. Analysis & headline — P0.**
- Primary (pre-register before E6): **manifold vs single at matched on-target suppression →
  lower off-target damage / accuracy-drop** (suppression-vs-damage Pareto); both arms beat
  random; paired over hold-out tasks with bootstrap CIs.
- Secondary: saturation — empirical α* vs the pre-registered predicted α* (r>0.5; note only 4
  points, tightly clustered 0.96–1.06 → low power, report honestly); cross-behaviour leakage;
  pooling sensitivity; k-granularity curve.
- Write results into `RESULTS_LEDGER.md` §B and unblock thesis ch08.

---

## TRACK C — Supporting experiments

**E8. Base ↔ distilled ↔ ±steering trace continuum — `[GPU]` P1.**
- Compare behaviour fractions across: base Qwen2.5-Math-1.5B (`02b_generate_baseline_chains.py`,
  may already exist) → negative-steered distilled → vanilla distilled → positive-steered.
  Tests whether negative steering moves the distilled model toward the base level (ties steering
  to the post-training hinge). Reference frame (different weights), matched tasks.

**E9. Finish R2.2 (annotator-robustness replication) — `[cluster]` P1.**
- 2-way Sonnet↔Qwen3 geometry **done & replicates** (pulled 2026-06-21). Remaining: (a) Nova-Pro
  `05b` finishes (~1 day, per-layer ×5) → **3-way compare** (`compare_annotators.py`); (b) also
  replicate the **variance-ratio specificity null** per annotator (only cdim+curvature compared
  so far); (c) optional hardening: **re-extract Qwen3/Nova clean** (they ran on lossy
  analysis-dedup, dup 33–52%, vs Sonnet's clean 1%). rsync outputs local when done.

---

## TRACK D — Cleanup / hygiene (parallel, low-risk) `[code]` P2

**E10. Quarantine DO-NOT-CITE outputs.** Move to `results/_STALE_`: the pre-dedup
`tier1_robustness/.../geometry_nulls_layer27.md`, the flawed predictive gate (AUC 0.61), and add
explicit "superseded" notes. Keep TwoNN / d_eff≥80% / tangent-variation out of any figure.

**E11. Rebuild `05d` sub-type clustering** at the reconciled layers (it ran at the old 14/14/17/27).

**E12. Keep the trackers current.** Update `METHODOLOGY.md` / `RESULTS_LEDGER.md` /
`CONFOUNDS_AND_REMEDIATION.md` as each item lands (this is the standing per-session protocol).

---

## Decisions needed from Tony
1. **example-testing steering layer:** resolve on E3 (causal) grounds — interim 12 (PR-trough)
   vs 27 (specificity + Huang).
2. **Eval-set size:** 50 vs 100–150 (E5) — vs re-annotation budget.
3. **Green-light Phase 7** (E6) after E1–E4 pass — GPU + the re-annotation $.
4. **Re-extract the Qwen3/Nova annotator arms clean** (E9c) or accept the analysis-dedup caveat.
