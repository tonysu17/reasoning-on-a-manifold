# E8 — Phase-7 Steering Grid: Launch Plan

**Status: DRAFTED, launch-gated.** This is the headline experiment — does *manifold* steering
suppress a named reasoning behaviour beyond an energy/dimension-matched random floor, and does it
sit on a better suppression-vs-damage Pareto than the *single* direction? Generation is turnkey;
a *pass*-grade verdict is blocked on the items in §7.

Drafted 2026-06-28 from a 4-agent map of `07_evaluate_steering.py`, `src/steered_inference.py`,
`src/steering_analysis.py`, `predict_saturation.py`, and the canonical specs
(`METHODOLOGY_REFINEMENT_2026-06-25.md`, `REVIEW_PHASE7_2026-06-25.md`). Where docs conflict, the
two 2026-06-25 docs win (they supersede `METHODOLOGY.md §4-5`).

---

## 1. Primary hypothesis & the ONE gate

**Endpoint = behaviour specificity against a floor, NOT task performance.** Suppression is primary.

```
suppression_X      = vanilla_fraction − steered_fraction_X      (same α*, same layer, same schedule)
Δ_floor(b, single) = suppression_single        − suppression_energy_matched_random
Δ_floor(b, mani_k) = suppression_manifold_k    − suppression_random_subspace_k
```
`behaviour_fraction` = sentence-fraction of re-annotated spans labelled *b* (`src/evaluation.py:28`;
token/sentence fraction, NEVER wall-clock). Vanilla is the shared reference both suppressions
subtract — it is NOT the headline subtrahend.

**PASS (all three must hold, per behaviour):**
1. `Δ_floor > 0` with paired-BCa CI excluding 0 **after Holm** across the frozen (arm×behaviour) family;
2. `Δ_floor` exceeds the annotator noise band `band_b` (§5);
3. matched on **realized behaviour-arm effect** (not equal-α).

**One-primary-gate discipline:** the gated-floor (B2) and length-conditioned reads are *reported, not
vetoing*. An under-powered secondary failing = "inconclusive," never an overturn. Corroborators
(McNemar exact + sign test on per-task {Improved/Degraded/Preserved/Unresolved}) support, don't gate.

**Single-annotator only ⇒ report "preliminary, band-ungated" — never "passing."**

---

## 2. Arms (all exist in `_build_arms`, `src/steered_inference.py:400-498`)

| Arm | Role | Floor it uses |
|---|---|---|
| `vanilla` (one shared gen/task) | reference fraction | — |
| `single_direction` | Venhoff diff-of-means probe → **headline** | `energy_matched_random` |
| `manifold_k{3,5}` (± k1/k10/auto) | Huang top-k-PCA probe → **headline** | `random_subspace_k{k}` |
| `energy_matched_random` | random rescaled to equal injected energy | floor (calibrates **single only**) |
| `random_subspace_k{k}` (×N reps) | Haar random k-projector, renormalized | floor (manifold arms) |
| `orthogonal_complement` | `(I−P_k)r_single` — is the off-subspace pure collateral? | control |
| `random_direction` | norm-matched (~19× under-energy) | **sanity floor only — demote** |

**Critical asymmetry to disclose (verified `steered_inference.py:689`):** `energy_matched_random`
calibrates `single_direction` ONLY. Manifold arms are **dimension-matched** (`random_subspace_k`),
NOT energy-matched. Do not call the manifold headline "energy-floored" unless an
`energy_matched_random_k{k}` arm is added.

**k choice (settled by E4, 2026-06-28):** use **k = 3–5**. The pooled manifold's leading directions
are highly stable (split-half ≤ 8–11°) and capture cos ≈ 0.77–0.90 of `r`; **k=10 reaches the
unstable spectral tail** (PC8–10 rotate 24–66°) and is NOT recommended for the headline.

**DO NOT** add a reward-gradient / predicted-direction arm to `_build_arms` (fenced to `src/predict/`,
Rung-3 only). B2 gate-vs-gradient arms are a *conditional follow-up*, built only after a pass.

---

## 3. Layer (decision rule — pending E1/E2 overnight)

No `--layer` flag — layer is read per-behaviour from the vector dir's metadata, so **layer = choice of
`--vectors-dir`.** Candidates and their vector dirs:

| Hypothesis | Source | Vector dir | α* ready? |
|---|---|---|---|
| **L27** | Huang-published + 07d forward argmax | `results/steering_vectors/R1-1.5B` | ✅ `predictions_layer27.json` |
| **Venhoff mid** 15/17/18 | published + E1 attribution | `…__venhoff` (+ `…__huang` for pooled) | ❌ need 05b @ 15,18 |
| **E1-argmax** | tonight's attribution | `…__E1` (auto-built by e1vec) | ❌ need 05b + α* at those L |

**Tonight's E1 (attribution) so far:** backtracking 17 ✅=pub, example-testing 15 ✅=pub,
uncertainty **15** (pub 18), adding-knowledge pending. **E2 (forward sweep) pending** — previously
argmaxed L27. So forward→L27 vs attribution→mid(15-17) is the live tension.

**Decision rule (from `METHODOLOGY_REFINEMENT §2.9`):** generate+annotate **L27 first** (the leaning
layer); fall to mid only if L27 fails the coarse vanilla-relative arbiter. If per-behaviour orderings
disagree, **do NOT majority-vote** — build per-behaviour at its own layer and flag. Disclose on every
single-layer headline that the hold-out is on the **task axis only, not the layer axis** (CF-17).

---

## 4. Dose — sealed α*

Per-behaviour fixed operating point, read verbatim (`results/saturation_predictions/R1-1.5B/predictions_layer27.json`):

| behaviour | α* (L27) |
|---|---|
| backtracking | 0.9940 |
| uncertainty-estimation | 0.9688 |
| example-testing | 0.9637 |
| adding-knowledge | 1.0561 |

All cluster at ≈1.0 (≈ one-component ablation, since `r` is unit-norm). **α* is LAYER-SPECIFIC.** For a
non-27 layer: `05b_geometric_diagnostics.py --layers 15 18` (κ source; CPU, ~10–30 min, $0 — only
`diagnostics_layer{15,18}.json` are missing) → `predict_saturation.py --layer {15,17,18}`.

**07 sweeps a global `--alpha-values` grid, not per-behaviour α*.** Two options:
- **(A, simplest)** run a single dose `α=1.0` (within 4% of every α*) + `α=0` vanilla. The
  realized-effect matching absorbs the small offset. **Recommended for Stage 1.**
- **(B, exact)** add a small per-behaviour α* path to 07 (~15 lines). Defer unless reviewers want it.
- For the descriptive Pareto only, a grid `0 0.5 0.7 1.0 1.3` brackets all α*.

**Do NOT** read the matched operating point off the eval curves — bypass `effect_quantile=0.8`
(`steering_analysis.py:613`, the tune-on-eval leak). Match post-hoc on realized effect at fixed α*.

---

## 5. Annotator + noise band (HARD prerequisite)

Sonnet built the behaviour labels (the DV) → scoring with Sonnet is circular (CF-7). The headline
must be scored by a **non-builder** model via `--annotator-model` (wired end-to-end,
`07_evaluate_steering.py:118`). **BLOCKER:** the live Qwen3-235B/Nova-Pro proxy id string is *nowhere
in the repo* (the static 3-way label files carry no model-id), and proxy creds
(`CLAUDE_PROXY_URL`/`CLAUDE_PROXY_KEY`) are unset. Tony holds both. With the id + creds, it runs with
zero code change.

`band_b` = per-behaviour **fraction-scale RMS** of `|frac_sonnet − frac_nonbuilder|` on the *same*
re-annotated steered chains. **NEVER κ-derived** (κ is the wrong scale; cite only as motivation). The
implemented `annotation_noise_band.py` is Sonnet-vs-Sonnet self-noise = a *lower bound* only; the
cross-annotator acceptance band is unbuilt (blocked on the non-builder id). Cheap fallback: a proxy
band from the static 3-way corpus (Sonnet-vs-Qwen3 per-chain fraction disagreement), labelled a
corpus-chain lower bound.

---

## 6. Cost & staging (~$260 envelope; tell-Tony-first)

Annotation ≈ **$0.055/chain**. Generation on RunPod 4090 ≈ 2–3 min/chain. Levers (biggest first):
**n_samples, n behaviours, α-grid size, random_subspace reps, n manifold-k.** Existing bake-off chains
CANNOT be reused (no floor arms, no α* dose) — a fresh generation run is required.

**Stage 1 — lean greedy preliminary** (`n_samples=1, temperature=0, α∈{0,1.0}`, drop
`random_direction`+`orthogonal_complement`, `--n-random-subspaces 2`, **2 headline behaviours**
backtracking+uncertainty, manifold k∈{3,5}):
≈ **900–1000 chains** → ~40 GPU-hr (~1.5 days) + ~$50 annotation. Fast read of "is there any signal."

**Stage 2 — CI'd headline** (`n_samples=3, temperature=0.7`, same lean arms, surviving behaviours):
≈ **1,800–2,400 chains** → ~$130–180 annotation + GPU. This is the *pass*-grade run.

Full 8-α × all-arms × 4-behaviour grid (~16,850 chains, ~$900) is **REJECTED** as over-scoped.

To hit "2 behaviours × k∈{3,5}" precisely, either build a trimmed vector dir (2 behaviours, k3/k5
only) or add `--behaviours` / `--k-values` flags to 07. `--max-eval-tasks` already trims tasks.

---

## 7. Blockers & decisions owed (before any *pass*-grade launch)

1. **Non-builder annotator id + proxy creds** (Tony) — hard blocker for a non-circular headline.
2. **Layer decision** — needs E2 (forward) + adding-knowledge E1 argmax (overnight) → then α* recompute
   if not L27 ($0 CPU step).
3. **Budget approval** — Stage 1 ~$75, Stage 2 ~$150 (tell-Tony-first rule).
4. **Δ_floor analysis layer is UNBUILT** (see §8) — needed to compute the headline (but post-generation,
   so it can be built in parallel with Stage 1).
5. **Pre-registration** — write `results/eval/<run>/preregistration.json` (primary endpoint, sealed α*,
   frozen Holm family, one-gate rule, equal-schedule clause, adding-knowledge=pre-registered-negative,
   under-power note) BEFORE unblinding deltas.

---

## 8. Analysis code to build (greenfield — specified, not implemented)

`src/steering_analysis.py` today is the OLD vanilla-relative / damage-at-matched-effect design. To
compute the §1 headline we must add (pure-Python, no GPU, buildable + testable locally now):
- **`delta_floor()`** — suppression_arm − matched floor (single↔energy_matched; manifold_k↔random_subspace_k).
- **realized-effect matching at sealed α*** — call `paired_bootstrap_matched_effect(...target=<realized>)`
  directly; do NOT use `compare_across_behaviours` with default `effect_quantile=0.8`.
- **`matched_pair_transitions()` + McNemar exact + sign test** (beside `paired_bootstrap_matched_effect`).
- **`band_b`** cross-annotator fraction-RMS gate (needs the non-builder annotation).
- **single-axis `damage_fn`** for the headline (not the summed `damage_scalar`).
- **an analysis runner** (`08_steering_analysis.py`?) — none exists; `steering_analysis.py` is imported
  only by its test. It must load `steering_results.json` + `annotated_steered.json` + `eval_summary.json`,
  emit the §1 table + `preregistration.json`'s `empirical:{}`.

---

## 9. Turnkey commands (see `runpod_e8.sh`)

```bash
# PREREQ (if layer ≠ 27): on the pod, CPU, $0
python -u 05b_geometric_diagnostics.py --model-short R1-1.5B --layers 15 18
python -u predict_saturation.py --layer 15 ; python -u predict_saturation.py --layer 18

# STAGE 1 — generation only (GPU, no API/creds needed):
python -u 07_evaluate_steering.py --model 1.5b \
  --vectors-dir results/steering_vectors/R1-1.5B \           # L27 first
  --out-dir results/eval/R1-1.5B__L27 \
  --alpha-values 0 1.0 --n-samples 1 --temperature 0 \
  --no-random-control --no-orthogonal-complement --n-random-subspaces 2 \
  --skip-annotation

# STAGE 2 — re-annotate (needs non-builder id + proxy creds):
export CLAUDE_PROXY_URL=... CLAUDE_PROXY_KEY=...
python -u 07_evaluate_steering.py --model 1.5b \
  --vectors-dir results/steering_vectors/R1-1.5B --out-dir results/eval/R1-1.5B__L27 \
  --alpha-values 0 1.0 --annotator-model "<NON_BUILDER_PROXY_ID>"   # resumes from saved gens
```
