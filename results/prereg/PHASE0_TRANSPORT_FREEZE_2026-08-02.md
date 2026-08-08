# Phase 0 — Audit, Closure, and Preregistration Freeze

**Date:** 2026-08-02 · **Authorisation:** Tony approved Phase 0 of
`../post_training_geometry_unified_plan_2026-08-02.md` ("approve phase 0", this date).
**Scope:** Phase 0 items only. This document licenses **no Phase 1+ hypothesis tests, no thesis
edits, and no new training** beyond the F5 control already owed in the ledger. Decision 8
(α-dial calibration organism) remains OPEN → plan item 0.8 is NOT run.
**Discipline:** the rules in §1 were written before the corresponding computations were run
(same session). Phase-0 computations are instrument calibrations and reproductions, not
hypothesis tests; Phase 1/2 outcomes get their own sealed prereg before launch.

---

## 1. Pre-committed rules

### 1.1 Contraction-calibration fix (plan item 0.3) — `pt13b_l17_curve.py`

**Problem being fixed:** pt13's injection curve lives on span-matrix `d_eff` at layers 12/16;
the DeepScaleR/STAR1 spectrum deltas in `results/r1_compression/report.json` are per-chain
**windowed-PR** deltas at **layer 17** (E9.0 grid). Different estimands; no mapping is valid
until the curve is re-derived on the measurement's own instrument.

**Instrument (mirrored exactly):** per-task windowed PR at L17 computed by
`src.loop_geometry.windowed_state_metrics` on the E9.0 window/stride grid; per-task summary =
`nanmean` over windows; arm effect = paired per-task delta vs r1; headline = **median** over
tasks (exactly `_rep_summaries` + `stage_analyse` in `30_r1_compression.py`).

**Substrate constraint discovered pre-run (grid survey, this session):** the local E9.0 shards
store full-sequence `pr_17` values (grid confirmed W = 128 / S = 64 by `len(pr) =
floor((T−128)/64)+1` across sampled shards) but only a **192-token state excerpt** per task
(`Xout_17` = 192×1536 fp16 → exactly 2 grid windows). Full r1 sequences are not on local disk.
The curve is therefore derived on the excerpts, with the caveat declared (excerpts are
onset-locked → loop-adjacent material over-represented), and the pod batch gains a full-sequence
validation stage (§2 s3).

**Injection family (mirrors pt13, applied to the excerpt):** center the excerpt matrix
(192×1536, float64), SVD, scale singular values beyond rank **k = 5** by
c ∈ {1.0, 0.9, 0.75, 0.5, 0.25, 0.0}, reconstruct, un-center, re-run
`windowed_state_metrics(·, 128, 64)`; per-task value = nanmean over its 2 windows; curve point =
median over tasks of (injected − uninjected). Sensitivity: k ∈ {2, 10}. Secondary layer: L16
(same rule); L17 primary.

**Self-gates (all must pass before any mapping is reported):**
- **Gate A (aggregation identity):** recompute `rep_paired_vs_r1` pr medians for deepscaler and
  star1 from the stored ent_shards vs the loop-shard `pr_17`; must equal `report.json` to 1e-9.
- **Gate B1 (grid identity):** `len(pr_17) = floor((T−128)/64)+1` for ≥ 95% of the 200-task set.
- **Gate B2 (excerpt sanity):** median over tasks of excerpt-recomputed PR within ±20% of the
  median stored full-sequence per-task PR (distribution-level check only; exact per-window
  gating impossible from excerpts — that is what pod s3 confirms on full sequences).
- **Gate C (null identity):** the c = 1.0 SVD round-trip must reproduce per-task deltas
  |Δ| < 1e-6.

**Mapping set (pre-committed):** matched-ids arms only — **deepscaler** (median dPR −0.0167)
and **star1** (−0.2170). **qwenmath is excluded** from interpolation (its grid differs;
`report.json` marks that tier direction-only).

**Labelling rule:** every mapped number carries "calibrated bound under the isotropic
tail-shrink family; family membership unverified" until the empirical singular-value profile
check runs on pod-extracted DeepScaleR states (pod batch s3). Out-of-family profiles ⇒ report
the raw delta as uncalibrated; no variance-removed claim.

### 1.2 F5 dose-matched DPO control route rule (plan item 0.1)

Run the fixed throughput probe first (pod batch s0; `/usr/bin/time`→`bash SECONDS` fix already
in place). **Pre-committed threshold:** if the probe projects ≥ 500 control pairs at
≤ **12 GPU-hours** on the rented GPU (batched generation permitted, e.g. vLLM), run the full
control at optimizer-steps AND realised-KL matched to `dpo-safety-v2` (KL 0.000587). If it
projects > 12 GPU-hours, run the pre-declared fallback: the existing 65 pairs trained to
matched steps/KL (~23 epochs), with the data-repetition confound declared in the ledger row.
No third option; no threshold adjustment after the probe.

### 1.3 Inert-control completion (plan item 0.2)

R1-1.5B side **already exists on disk**:
`data/activations/_volume_R1-1.5B_6label_archive/` (deduction 34,848 × 1536 and initializing
4,863 × 1536 at all 28 layers, with `row_index.json` + `metadata.json`). STAR1 side = pod batch
s2: extract deduction + initializing spans at layers {12, 16} on **byte-identical input_ids**
(`--tokenizer-alias 1.5b`, the C1 fix). Then run the standard gated battery (within-model
disjoint-subsample rotation null; sign-flip translation null; matched-n m = 1500,
n_perm = 250; d_eff) on both inert behaviours. These two behaviours are **negative controls**:
the pre-registered expectation is rotation null AND translation coherence comparable to the
four target behaviours (the translation is global). Multiplicity: Holm within
{behaviour × layer ∈ {12,16}}, family = the 2 inert behaviours (the 4 target behaviours were
already tested and are not re-tested).

### 1.4 Transport-MDE first cut (plan items 0.5/0.6) — `p0_mde_firstcut.py`

Purpose: size the Phase 2 intervention battery. Method (analytic first cut, conservative):
task-cluster bootstrap (B = 10,000, seed 20260802) of the E8 backtracking Δ_floor
(single-direction and manifold-k5 arms, n = 50 tasks) → SE(Δ̂); detectable attenuation
fraction under an unpaired two-battery comparison:
`f* = (z_0.95 + z_0.80) · √2 · SE(Δ̂) / Δ̂`.
This deliberately assumes no cross-model task pairing benefit (Phase 2 will pair tasks, so the
real MDE is at least this good). **Consequence rule:** the Phase 2 prereg must set its task
count so that f* ≤ 0.5 (a 50% attenuation must be detectable) — if n = 50 gives f* > 0.5, the
Phase 2 battery is enlarged (n scales as 1/f*²) or the verdict vocabulary drops "retained" in
favour of "not disabled". The full simulation-based MDE (sham-frame mixing) is built in the
Phase 2 prereg itself; this first cut only sizes it.

### 1.5 Units, uncertainty, and stop rules (plan item 0.5, standing for all Phase ≥ 1 work)

- Independent unit: chain/task (spillover, steering); episode (agentic, later). Never tokens,
  windows, or layers.
- Cluster bootstrap by unit, B ≥ 2,000; hierarchical over training seeds where seeds exist.
- Multiplicity: Holm within pre-declared families; the family is declared per experiment
  before its run (Phase-1/2 preregs), never post hoc.
- Regime matching: STAR1 (full SFT) comparisons use the full-FT control
  (`R1-1.5B-fullft-control-s42`); LoRA controls compare against LoRA arms only.
- Stop rule: any gate failing twice after one good-faith fix ⇒ stop that line and report the
  bound; do not iterate toward a positive result.
- Wording constraints inherited: G3 (no "behaviour-specific six-dimensional linear subspace"),
  M3 (order-sensitivity nulls on trajectory claims), M5 (sealed hyperparams, matched floors).

### 1.6 Provenance requirement (plan item 0.4)

`configs/analysis/checkpoint_provenance.yaml` + human-readable table. A checkpoint may enter
Phase 1 only when its row has: HF id, declared parent, architecture check (hidden/layers/
max-pos/bos), tokenizer-file sha256 (local) or "not cached", chat-template status, precision,
license, training-stage description, adapters-merged status, verification date. Local hashes
are computed for the four cached checkpoints today; remote-only rows carry HF-page
verification dates from the 2026-08-02 sweeps.

### 1.7 M7 registration (plan item 0.7)

The PAIR-style prefix-contamination control is registered as **M7** in
`PREDICTIVE_GEOMETRY.md` §11.1 (additive amendment, this date — M6 was already taken by
predictor-only JEPA) and pointed to from `METHODOLOGY.md` §9: any latent value/precursor claim
must show the signal tracks task progress and not coherence-with-a-corrupted-prefix
(corrupted-prefix probe arm required).

---

## 2. Pod batch specification (spend items; ~$10–20 total, one pod session)

| Stage | What | Est. |
|---|---|---|
| s0 | F5 throughput probe (fixed), then route per §1.2 | minutes |
| s1 | F5 control per routed option (full or fallback) + extraction L12/16 + gated battery | fallback ~1 GPU-h; full ≤ 12 GPU-h |
| s2 | STAR1 inert-control extraction (deduction+initializing, L12/16, byte-identical ids) + battery; doubles as the plan's one-checkpoint smoke test (throughput/storage/cost measured) | ~1–2 GPU-h |
| s3 | DeepScaleR **and r1** L17 full-sequence state extraction on the 200-task matched set → (a) empirical SV-profile domain check for §1.1, (b) full-sequence re-derivation of the injection curve to validate the excerpt-derived curve | ~1–2 GPU-h |

Runbook: `runpod_phase0.sh` (house pattern: push/launch/watch/status/pull; Mac-side watcher;
no on-pod self-kill; termination from the Mac after a verified pull).

## 3. Explicitly NOT in Phase 0

Phase 1 screen runs, frame transport, KTO, any agentic work, α-dial (decision 8 open),
pt21 (decision 7 open), thesis edits, commits/pushes.

## 4. Executed results (appended same-day, after §1 was written)

- **pt13b (item 3)** — Gates A (aggregation identity to 1e-9: deepscaler −0.016709, star1
  −0.216980, n = 200), B1 (grid W128/S64 confirmed), C (null identity ≤ 1e-13) **PASS**;
  **Gate B2 FAIL, root cause substantive**: the stored `Xout_*` are class-selected
  non-contiguous 192-token subsamples (out-of-loop tokens; `18_loop_geometry.py
  select_class_token_indices`), not sequence excerpts — the windowed instrument is undefined
  on them, and no local full-sequence source exists (main + `_pod_archive` checked). Per the
  §1.1 labelling rule: **mapping withheld; deferred to pod s3** (`ph0_s3_curve.py`).
  Substrate-invalid curve archived for the audit record only:
  `results/safety_posttrain/{pt13b_l17_curve.json, PT13B_L17_CURVE.md}`.
- **MDE first cut (items 5/6)** — E8 estimand **recovered exactly** (sentence-fraction,
  suppression-oriented floor−arm; rel-err 0.0 on backtracking|single_direction). Bootstrap
  (B = 10k, seed 20260802): SE 0.0164 paired / 0.0198 unpaired; **f\* ≈ 1.29 at n = 49**
  (manifold_k5: estimand 28% off → SE-ratio fallback, f\* ≈ 1.38); **n ≈ 330 tasks for
  f\* = 0.5**. **Consequence rule §1.4 FIRES:** the Phase 2 prereg must enlarge the battery
  (~6× under these conservative assumptions; cross-model pairing and variance-reduced
  estimands may lower that — to be established by the Phase 2 simulation MDE) or restrict
  verdicts to "not disabled".
  `results/safety_posttrain/{ph0_mde_firstcut.json, PH0_MDE_FIRSTCUT.md}`.
- **Provenance (item 4)** — 4 core rows hashed locally + 15 remote-verified rows:
  `configs/analysis/checkpoint_provenance.yaml` +
  `results/prereg/CHECKPOINT_PROVENANCE_2026-08-02.md`. Finding: tokenizer.json is NOT
  byte-identical across the family (R1 `88145e3c` vs STAR1/DeepScaleR `e20ddafc`; config bos
  151643 vs 151646) — the C1 byte-identical-ids rule is now provenance-evidenced for every arm.
- **M7 (item 7)** — registered: `PREDICTIVE_GEOMETRY.md` §11.1 additive amendment +
  `METHODOLOGY.md` §9 pointer (corrected from "M6": that slot was already predictor-only JEPA).
- **Inert controls (item 2)** — audit: the R1 side already exists on disk
  (`data/activations/_volume_R1-1.5B_6label_archive/`: deduction 34,848 / initializing 4,863
  rows, all 28 layers, row_index present). STAR1 side = pod s2 (`ph0_s2_extract_inert.py`,
  fresh `STAR1-1.5B-6label` short-name so the existing 4-behaviour metadata is untouched).
- **Pod batch (items 1/2/3/6)** — staged and CLI-verified: `runpod_phase0.sh` +
  `ph0_pod_job.sh` (s0 probe → routed s1 per §1.2, training step an explicit manual gate;
  s2 doubles as the smoke test; s3 full-sequence curve + SV family check). ~$10–20;
  **needs a pod — no RunPod key on this Mac, so launch is Tony's step.**
- **Not run** — α-dial (decision 8 open), pt21 (decision 7 open), all Phase 1+ work.

### §4b — Pod-batch dispositions (2026-08-08; run complete 16:37; CLOSURE)

Fresh pod (root@213.173.111.9, 4090; fixes re-applied: rsync/tmux, transformers==4.49.0, plus a
new compat fix — `src/chain_gen.py` `dtype=`→`torch_dtype=`). All artifacts live on the euro-2
volume; **local pull deferred to the next pod session**; pod terminated on Tony's order after
verification (self-term via PID-1 key, pod yifdnc2lujeb29, unreachable confirmed).

- **s0 / F5 route:** probe 1,049 s / 8 tasks, yield 0.5 pairs/task ⇒ **36.42 projected GPU-h**
  for 500 pairs > the pre-committed 12 h threshold ⇒ **route = FALLBACK_65PAIR**
  (`results/ph0_f5_route.json`, volume). F5 training did NOT run on this pod — the 65-pair
  file's only copy is local and local access was blocked at the time; recorded
  **prospective/unrun**, first item of the next pod session (recipe in
  `SESSION_HANDOFF_2026-08-08.md` §3 / memory).
- **s2:** COMPLETE — deduction 34,848 + initializing 4,863 rows @L12/16 on byte-identical ids;
  wrote to pod `data/activations/STAR1-1.5B/` (`--short-name` is inert for registry models; the
  dir was empty beforehand ⇒ nothing clobbered). The job's `FAILED:s2` status was a cosmetic
  wrapper self-check path bug (fixed same day). Local staging pull (→ `STAR1-1.5B-6label`,
  never the local canonical dir) + the §1.3 inert battery = next pod session.
- **s3:** COMPLETE, 200/200 both arms, zero failures. **Gate S3-A FAIL** — the fresh
  full-sequence re-extraction gives paired median dPR (deepscaler − r1) **+0.139 vs the
  report's −0.0167** (sign flip; both ≈ 0 against baseline PR ≈ 30) ⇒ the July micro-delta is
  **not re-extraction-stable**; the qualitative no-contraction reading survives; per the §1.1
  labelling rule the calibrated mapping is **NOT LICENSED** (the s3 script emitted mapping
  fields regardless — script defect noted; values retained only as unlicensed bounds:
  deepscaler ≈ 0.08%, star1 ≈ 1.0% variance, star1 family-unverified as sealed).
  **Family check PASS:** deepscaler/r1 singular-value ratio profile flat at 0.984–0.986 across
  all ranks (RMS residual 0.0024) ⇒ DeepScaleR's spectral change is a **uniform ≈1.5% global
  shrink, not a tail/low-rank contraction**. Artifacts (volume):
  `results/safety_posttrain/ph0_s3/{r1_fullseq.npz, deepscaler_fullseq.npz, s3_analysis.json}`.
- **Closure status:** every §1 rule is discharged or dispositioned; two mechanical items
  (volume→local artifact pull; inert-control battery) are the opening steps of the next pod
  session. The closure commit accompanying this entry is the analysis-repo stable point — the
  go-signal for the thesis evidence-snapshot refresh.
