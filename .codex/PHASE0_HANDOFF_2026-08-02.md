# Codex handoff — Phase 0, post-training transport programme (2026-08-02)

Self-contained context + parallel task assignments. Written at session end by the Claude session
that executed Phase 0's local half and staged the pod batch.

---

## 0. Constraints (read first)

- Branch: create `codex/phase0-support` off `predictive-geometry-of-reasoning`. NEW FILES ONLY
  except where a task explicitly grants an edit. Do not commit to existing branches; do not push.
- **DO NOT touch:** the pod (ssh `runpod`), `runpod_phase0.sh`, `ph0_pod_job.sh`,
  `ph0_s2_extract_inert.py`, `ph0_s3_curve.py`, `pt13b_l17_curve.py`, `ph0_mde_firstcut.py`,
  `RESULTS_LEDGER.md`, `METHODOLOGY.md`, `PREDICTIVE_GEOMETRY.md`, `E10_DAS_PREREG.md`,
  anything under `results/` or `results/prereg/`, `thesis/` (separate private git repo).
  Exception: T3 may append fields inside `configs/analysis/checkpoint_provenance.yaml`.
- No training, no GPU spend, no API/annotation spend, no result regeneration, no thesis edits.
- Wording rules binding any text you draft: **G3** ("low estimated dimension within
  annotation-indexed clouds", never "behaviour-specific six-dimensional linear subspace");
  **M3** (trajectory claims report step-shuffle nulls or are worded as state-occupancy);
  **M5** (sealed hyperparams, count+energy-matched floors); **M7** (latent value/precursor
  claims need a corrupted-prefix control — PG §11.1).
- Write your outputs under `.codex/out/`, reviews under `.codex/reviews/`, patches under
  `.codex/patches/` (create dirs as needed).

## 1. Project in one paragraph

MSc thesis "The Geometry of Machine Reasoning": per-behaviour activation geometry
(backtracking / uncertainty-estimation / example-testing / adding-knowledge, sentence-level
annotations on 986 self-generated math chains) in `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`,
plus the post-training origin of safety reasoning (gpt-oss-20b DSR object; STAR1 spillover).
Movement I established descriptive geometry + one causally grounded coordinate (E10 DAS
backtracking frame, L17, width 2 — n=1, does not generalize to other behaviours). Movement II
(post-training) established: fine-tuning leaves behaviour subspaces essentially unrotated
(bounded sub-degree) but writes a coherent global translation whose DIRECTION encodes the
recipe (safety-SFT cos 0.57 to full-SFT STAR-1 vs control 0.15, floor 0.026). The current
programme ("causal transport"): what happens to the grounded coordinate under post-training —
retained / rescaled / rotated / disabled / decoupled.

## 2. Canonical documents (read in this order)

1. `../post_training_geometry_unified_plan_2026-08-02.md` — the operating plan (Phases 0–5,
   gates, five causal outcomes; amended same day, §9 amendment log).
2. `results/prereg/PHASE0_TRANSPORT_FREEZE_2026-08-02.md` — Phase-0 pre-committed rules (§1)
   and executed results (§4). THE reference for what today's numbers mean.
3. `../post_training_geometry_brainstorm_2026-08-02.md` — wider experiment menu (pt15–pt23,
   AG1–AG5) + literature positioning with verified arXiv ids.
4. `RESULTS_LEDGER.md` (2026-08-02 change-log entry) + `METHODOLOGY.md` — canonical trackers,
   READ-ONLY for you.

## 3. Today's executed results (you build on these)

- **pt13b** (`pt13b_l17_curve.py` → `results/safety_posttrain/{pt13b_l17_curve.json,
  PT13B_L17_CURVE.md}`): re-derivation of the contraction injection curve on the R1-compression
  instrument (windowed PR @L17, W=128/S=64). Gates A (aggregation identity to 1e-9: deepscaler
  −0.016709, star1 −0.216980, n=200), B1 (grid), C (null) PASS; **Gate B2 FAIL — root cause:
  the locally stored `Xout_*` shards are CLASS-SELECTED NON-CONTIGUOUS 192-token subsamples
  (out-of-loop tokens; `18_loop_geometry.py::select_class_token_indices`), not sequence
  excerpts**; no local full-sequence source exists. Mapping withheld; deferred to pod stage s3.
- **MDE first cut** (`ph0_mde_firstcut.py` → `results/safety_posttrain/{ph0_mde_firstcut.json,
  PH0_MDE_FIRSTCUT.md}`): E8 estimand recovered EXACTLY — per-task backtracking
  sentence-fraction, suppression-oriented (delta_floor = mean(floor) − mean(arm)); rel-err 0.0
  on backtracking|single_direction (reported +0.05379). Bootstrap B=10k seed 20260802:
  SE 0.0164 paired / 0.0198 unpaired. **f\* = (z.95+z.80)·√2·SE/Δ ≈ 1.29 at n=49 ⇒ n≈330
  tasks needed for f\*=0.5.** Consequence rule (freeze §1.4) FIRES: Phase-2 battery grows ~6×
  or verdicts downgrade to "not disabled". Unresolved: manifold_k5 recomputes to +0.0500 vs
  reported +0.0698 (28% off) while single_direction matches to machine precision → T1.
- **Provenance** (`configs/analysis/checkpoint_provenance.yaml` +
  `results/prereg/CHECKPOINT_PROVENANCE_2026-08-02.md`): 4 cached checkpoints sha256-hashed,
  15 remote rows HF-verified. Finding: **tokenizer.json NOT byte-identical across the family**
  (R1 `88145e3c` vs STAR1/DeepScaleR `e20ddafc`; config bos 151643 vs 151646) ⇒ the C1
  byte-identical-input-ids rule (`--tokenizer-alias 1.5b`) applies to every arm.
- **M7 registered** (PG §11.1 additive amendment + METHODOLOGY §9 pointer).
- **Inert controls half-done:** R1 side on disk
  (`data/activations/_volume_R1-1.5B_6label_archive/`: deduction 34,848 rows, initializing
  4,863, all 28 layers, row_index.json). STAR1 side = pod s2.

## 4. Pod state (context only — you never touch the pod)

- ssh alias `runpod` → root@213.173.110.81:16284 (4090 24GB). Volume cleaned 66G→43G
  (deleted: 13G stale gpt-oss cache copy, 3.4G duplicate R1 cache, dataset caches, five synced
  job dirs, rom-rl minus checkpoints; KEPT as unique/needed: `rl_adapters_keep` 14G,
  `rom-rl/checkpoints` 7G, `venv-r1` 12G [TRL env for the F5 training step],
  `keep_lora_control_ckpt` 3.4G).
- Environment fixes applied this session: rsync + tmux installed (minimal image);
  rsync flags → `-rltz` (mfs volume forbids chown); **transformers pinned 5.14.1 → 4.49.0**
  (image torch is 2.4.1; transformers 5.x needs DTensor/torch≥2.6) — see T7.
- Job = `ph0_pod_job.sh`: s0 F5 throughput probe → pre-committed 12 GPU-h route rule
  (`results/ph0_f5_route.json`) → s1 control-pair generation (FULL route only; **DPO training
  itself is a MANUAL GATE** — matched steps + realised KL vs dpo-safety-v2 KL 0.000587, ledger
  §F5) → s2 STAR1 inert extraction (= smoke test) → s3 full-sequence L17/L16 states for
  r1+deepscaler → injection curve + SV family check (`ph0_s3_curve.py --analyse` runs pod-side).
  Job script sets HF_HOME=/workspace/hf, expandable_segments, and a du-based 70G disk gate
  (df is meaningless on the mfs volume — it reports cluster-wide space).
- **Status at handoff: first launch failed on the transformers import (now fixed); RELAUNCH
  PENDING.** From the repo root on the Mac:
  `POD=runpod ./runpod_phase0.sh launch && POD=runpod ./runpod_phase0.sh watch`
  (`status` to tail; `pull` to sync back; watcher notifies and never kills the pod;
  termination = Tony, from the RunPod console, after a verified pull.)

## 5. Tasks (priority order; all parallel-safe)

### T1 — Red-team the Phase-0 artifacts (HIGH)
Adversarially verify: `pt13b_l17_curve.py` (gate logic; the aggregation mirror vs
`30_r1_compression.py::_rep_summaries`; SVD tail-shrink injection; var-removed formula),
`ph0_mde_firstcut.py` (bootstrap validity; the f\* formula's √2 and one-sided z's; conservative
unpaired assumption; estimand-recovery circularity risk), and freeze §1 internal consistency.
**Specific puzzle to resolve:** why does manifold_k5 recompute to +0.0500 vs reported +0.0698
while single_direction matches exactly? (Candidate causes: different task subset — cell says
n_tasks=49, my pairing found 48; a per-1k estimand for that cell; alpha handling. Find the
authoritative delta_floor computation — try `src/delta_floor.py`, `07_evaluate_steering.py`,
or the E8 analysis path — and adjudicate.) Deliverable:
`.codex/reviews/PHASE0_REDTEAM.md`, findings marked CONFIRMED/REFUTED with runnable repro
snippets. Do not edit the reviewed files.

### T2 — `ph2_mde_sim.py`: simulation MDE for Phase 2 (HIGH)
Replace the analytic first cut. Inputs: `results/eval/R1-1.5B__E1/{annotated_steered.json,
delta_floor_report.json}`. Simulate two-battery transport comparisons: task counts
{50,100,150,200,300,400} × attenuation {0.1…1.0} × estimand {sentence-fraction, per-1k} ×
design {unpaired, task-paired across models (hierarchical bootstrap: shared task effect + arm
noise)}; α=.05 one-sided, power .80. Output f\*(n) curves + n\* for f\*≤0.5 per design →
`.codex/out/ph2_mde_sim.{json,md}` (promotion into `results/` is not yours). Acceptance:
reproduces the first-cut f\*≈1.29 within ±10% under its exact assumptions (n=49, unpaired,
fraction estimand); fixed seed; honest README section in the md.

### T3 — Provenance completion (MED)
`.codex/tools/hash_remote_checkpoints.py`: via `huggingface_hub`, download ONLY
tokenizer.json / tokenizer_config.json / config.json (KB-scale; NO weights) for the 15
`remote_verified` rows in `configs/analysis/checkpoint_provenance.yaml` (incl. both CoRT
variants); add sha256_16 + arch fields to those rows (the one permitted edit). CoRT deep-check:
tokenizer hash vs R1's `88145e3c…`/`e20ddafc…`, chat-template diff, config diff; weight-delta
check = write the spec only. Report: `.codex/out/PROVENANCE_COMPLETION.md`.

### T4 — `pt19_alpha_dial.py` implementation only (MED)
Weight interpolation θ_α = θ_R1 + α(θ_STAR1 − θ_R1), α∈{0.25,0.5,0.75,1.0}; safetensors
streaming merge to an output dir; `--dry-run` validates 3 tensors. Unit tests with tiny random
state dicts (`tests/test_pt19_alpha_dial.py`): α=0/1 reproduce endpoints exactly; linearity on
a probe tensor. NO execution against real checkpoints (decision 8 is still open). Acceptance:
pytest green.

### T5 — pt18 KTO label builder (MED)
`pt18_build_kto_labels.py` per the unified plan's §3.1.E discipline: read
`data/annotated_R1-1.5B.json`; chain-level backtracking rate per 1k tokens; PROBLEM-level
train/discovery/eval split (60/20/20, seed arg) BEFORE any thresholding; desirable/undesirable
= top/bottom quartiles within the train split, matched on correctness + length bands; emit a
label-shuffled control file and a balance report (correctness, length, difficulty proxy, token
entropy per class). Outputs: `.codex/out/kto_labels_*.json` + `.codex/out/KTO_BALANCE.md`.
Acceptance: an explicit leakage-check function proving zero problem overlap across splits.
No training.

### T6 — Phase-2 prereg skeleton (MED)
Draft `.codex/out/PHASE2_TRANSPORT_PREREG_DRAFT.md` implementing unified-plan Phase 2 + its
amendments: five outcomes (incl. rescaled), target-model grounding-gate nulls, MDE consequence
(battery size from T2's n\*, or verdict-vocabulary downgrade), sham + random-orthogonal +
optional on-manifold floors (arXiv 2602.02315), cross-model task pairing, multiplicity family,
stop rules. Mark every number that awaits T2 or pod-s3. Header must say DRAFT — Tony seals.

### T7 — Dependency pinning hygiene (LOW, ~15 min)
`.codex/patches/pyproject_transformers_pin.diff`: propose `transformers>=4.40,<5` in the
relevant extras (today's pod failure: unbounded `>=4.40` pulled 5.14.1 against torch 2.4.1)
plus a `smoke_imports.py` for pod setup. Do NOT apply the patch.

## 6. Standing numbers (context — do not recompute)

| Result | Value |
|---|---|
| Rotation under fine-tuning | bounded sub-degree (+0.26…+0.45°), every arm incl. control |
| Translation | 4.7–6.4% of norm; direction recipe-specific (cos 0.57 vs control 0.15, floor 0.026); seed-replicated; annotator-swap 0.999 |
| pt12 method plane | SFT dH −0.0104 vs DPO +0.0145; GRPO arms no-op; "DPO safety-specific" RETRACTED → F5 owed |
| R1-compression dissociation | RLVR: behavioural compression w/o representational (ΔPR med −0.017, p=.051); distillation raises rank (−1.56 base→distill, p≈5e-28) |
| pt13 calibration | 11–13% variance contraction reads dPR −3.6…−5.5 (span-matrix L12/16 instrument) |
| E8 steering | backtracking only clean: Δ_floor +0.054 single / +0.070 k5, Holm .002 |
| E9.1b parity | ablate 24% / vanilla 36% / amplify 58–64% (bidirectional handle) |
| E10 frame | grounded @L17 width 2, ~13×; E10.3: n=1, other behaviours transfer +0.62…+0.72 but ungrounded |
| DSR (gpt-oss-20b) | d 5.02 / AUROC .979; capability control |cos| .19, retention .995 (H1 provisional) |

## 7. Open decisions (Tony's, not yours)

Decision 7: pt21 (gpt-oss-safeguard-20b vs gpt-oss-20b) thesis- vs paper-scope.
Decision 8: run the α-dial calibration organism (~$10–15).
F5 training step: manual gate even after the pod batch generates pairs.
