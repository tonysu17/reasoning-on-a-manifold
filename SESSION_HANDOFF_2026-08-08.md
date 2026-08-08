# SESSION HANDOFF — post-training transport programme (2026-08-08)

Continuation context for a new session. Written while the Phase-0 pod batch was mid-run.
Canonical trackers stay `METHODOLOGY.md` + `RESULTS_LEDGER.md` (change-log entries 2026-08-02
and 2026-08-08 cover this programme); this file is the operational picture.

## 0. The programme in three sentences

Post-training is treated as an intervention on a fixed geometric substrate; the flagship
question is **causal transport**: what happens to the one grounded causal coordinate (the E10
backtracking DAS frame, L17, width 2) across post-training boundaries — retained / rescaled /
rotated / decoupled / disabled. Descriptive Movement-II results already stand (rotation bounded
sub-degree; recipe-specific global translation cos 0.57 vs control 0.15; SFT-sharpens/DPO-broadens/
GRPO-no-op; RLVR = behavioural compression without representational compression). The causal
ledger is empty until Phase 2 runs.

## 1. Document chain (read in this order if new)

1. `../post_training_geometry_unified_plan_2026-08-02.md` — operating plan, Phases 0–5, five
   outcomes, §9 amendment log.
2. `results/prereg/PHASE0_TRANSPORT_FREEZE_2026-08-02.md` — Phase-0 rules (§1) + executed
   results (§4; the pod-batch closure gets appended there when s3 lands).
3. `results/prereg/PHASE2_TRANSPORT_PREREG_2026-08-08.md` — **SEALED** Phase-2 design (see §4
   below).
4. `.codex/PHASE0_HANDOFF_2026-08-02.md`, `.codex/reviews/PHASE0_REDTEAM.md`,
   `.codex/out/ph2_mde_sim.md` — codex support wave (all delivered).
5. `../post_training_geometry_brainstorm_2026-08-02.md` — the wider menu (pt15–pt23, AG1–AG5).

## 2. Live state at handoff (2026-08-08, ~15:40 pod time)

**Pod:** ssh alias `runpod` → root@213.173.111.9:10981 (4090; euro-2 volume ~50G / 70G gate).
**Batch `ph0_pod_job.sh` status:**

| stage | state | notes |
|---|---|---|
| s0 F5 probe | ✅ DONE | 0.5 pairs/task → 36.4 projected GPU-h for 500 pairs > 12 h threshold ⇒ **route = FALLBACK_65PAIR** (`results/ph0_f5_route.json` pod-side). F5 = train existing 65 pairs to matched steps/realised-KL (dpo-safety-v2, KL 0.000587), data-repetition confound declared. |
| s1 F5 training | ⏸ MANUAL GATE (Tony) | ~20 min in `/workspace/venv-r1` (TRL env, kept for this). |
| s2 STAR1 inert extraction | ✅ DONE (status string says FAILED:s2 — **cosmetic**) | deduction 34,848 + initializing 4,863 rows @L12/16, byte-identical ids. Files are in pod `data/activations/STAR1-1.5B/` (NOT …-6label): `--short-name` is ignored for registry models. That pod dir was EMPTY before — nothing clobbered. Wrapper self-check fixed locally post-hoc. |
| s3 full-seq curve + family check | 🟢 RUNNING | r1 arm 200/200 **zero failures** (`r1_fullseq.npz` written); DeepScaleR arm extracting since 15:35; then `--analyse` writes `results/safety_posttrain/ph0_s3/s3_analysis.json`. ETA ~1 h from 15:35. |

Because s2's cosmetic failure is recorded in PH0_STATUS, the job will finish with
`PH0_STATUS=FAILED:s2` **and** `PH0_DONE.marker` — key any watcher on the **marker**, not the
status string. A monitor may or may not still be running from the previous session; verify.

## 3. On-completion checklist (next session, in order)

1. **Pull:** `POD=runpod ./runpod_phase0.sh pull` (gets ph0_s3/, route/probe files, log), PLUS
   the manual inert pull — **into a staging dir, never into the local canonical STAR1 dir**:
   `rsync -rltz runpod:/workspace/reasoning-on-manifold/data/activations/STAR1-1.5B/ data/activations/STAR1-1.5B-6label/`
   (local `data/activations/STAR1-1.5B/` holds the July 4-behaviour canon — do not merge).
2. **Verify s3:** `s3_analysis.json` → gate_s3a (paired median dPR vs report −0.0167, 20% rel
   tolerance), family_check (head/tail ratio step-fit, RMS ≤ 0.05 = tail-shrink family
   membership), mapping (deepscaler family-verified; star1 mapped but family-unverified).
3. **Close Phase 0:** append the s3 + s2 + F5-route dispositions to
   `PHASE0_TRANSPORT_FREEZE_2026-08-02.md` §4; ledger change-log entry; then the freeze's
   Phase-0 closure is signed.
4. **Run the inert-control battery** (local/cheap): pt03b-style gated rotation + translation +
   d_eff on deduction/initializing, R1 side from
   `data/activations/_volume_R1-1.5B_6label_archive/`, STAR1 side from the staging dir;
   parity via row_index/token_start. Prereg expectation (freeze §1.3): rotation null AND
   translation coherence comparable to target behaviours; Holm over {2 behaviours × 2 layers}.
5. **Pod:** after verified pulls, terminate from the RunPod console (never on-pod). If the F5
   fallback training is wanted first, run it in `venv-r1` before terminating (manual gate).

## 4. Phase 2 — SEALED, three launch preconditions

`results/prereg/PHASE2_TRANSPORT_PREREG_2026-08-08.md`. Key seals: STAR1 + DeepScaleR at
pinned snapshots; **n = 100 task-paired battery** (sim: paired f\*(100) ≈ 0.296; unpaired
fallback n = 300); discovery = teacher-forced corpus spans through each target (re-fit needs no
new annotation); transported-raw / scalar-norm / diag-whitened arms + sealed E10 re-fit (L17,
width {1,2}, M5) + sham/rand-orth/count/energy floors; target-null gates = 20 sham + 20
random-orthogonal per target at p95; **pre-outcome injection-recovery** f ∈ {0.25, 0.5, 0.75},
pass = ≥80% power at f = 0.5, else battery grid 150→200→300, else "retained" leaves the
vocabulary; Nova-Pro (non-builder) verdicts; Holm over the 2 checkpoints; damage gates sealed;
on-manifold floor EXCLUDED this launch. **Launch requires:** (1) Phase-0 closure (§3 above),
(2) task manifest drawn/hashed per prereg §3 (seed 20260808, n=100, stratified from
`data/tasks_final.json` excluding annotation-corpus problems and E8 eval ids →
`results/prereg/phase2_task_manifest.json`), (3) Tony's spend sign-off (~$20–40 pod +
~$100–250 Nova annotation — the dominant line).

## 5. Valid vs invalid numbers (red-team dispositions)

- VALID: single_direction MDE inputs (Δ=+0.05379, n=49, exact estimand recovery); the T2
  simulation (`.codex/out/ph2_mde_sim.json`); pt13b Gates A/B1/C and its core math.
- INVALID, do not use: the Phase-0 first-cut **manifold** numbers (f\*=1.38 / n\*=366) — wrong
  floor (contract: manifold_k5 ↔ random_subspace_k5; authoritative Δ=+0.06982/n=49), unpooled
  replicates, failed-gate headline. Correction block lives in
  `results/safety_posttrain/PH0_MDE_FIRSTCUT.md`.
- LABEL: the archived single_direction f\*=1.29 is the extra-conservative variant; the literal
  frozen rule gives 1.071 (n\*=225). Sizing authority = simulation + in-run injection-recovery.
- pt13b's local mapping stays WITHHELD (Gate B2, structural: `Xout_*` shards are class-selected
  non-contiguous subsamples) — s3 is the fix.

## 6. Infrastructure gotchas (all bit us this week)

- Fresh pod containers need, every time: `apt-get install -y rsync tmux`;
  `pip install -e ".[gpu]" "transformers==4.49.0"` (image torch 2.4.1; transformers 5.x needs
  torch ≥ 2.6 — pyproject's `>=4.40` is unbounded; codex patch `.codex/patches/` proposes `<5`).
- `src/chain_gen.py::load_model` now passes `torch_dtype=` (was 5.x-only `dtype=`) — **working
  tree only, not committed**; do not lose this on checkout.
- rsync to the mfs volume: use `-rltz` (chown forbidden → `-az` exits 23); Mac's Apple rsync
  lacks `--info=stats1`.
- `df` on the euro-2 volume reports cluster-wide space; gate on `du -s /workspace` (70G rule).
- `04_extract_activations.py --short-name` only applies with `--model-path`, not registry
  `--model`.
- Kill discipline: no on-pod self-kill; Mac watcher pulls; Tony terminates from console.
- Volume keeps (do NOT delete): `rl_adapters_keep/` (14G, RL arms not local),
  `rom-rl/checkpoints/` (7G), `venv-r1/` (12G, F5 training env), `keep_lora_control_ckpt/`.

## 7. Repo hygiene + open decisions

- Branch `codex/phase0-support`, everything from 08-02 and 08-08 UNCOMMITTED (Phase-0 scripts +
  results, prereg files, codex outputs, chain_gen fix, ledger/METHODOLOGY/PG edits). A commit
  pass is overdue once Phase 0 closes. `thesis/` is its own private repo — untouched by this
  programme so far; the Phase-2 result is the planned thesis extension (unified plan §6
  "thesis-minimum": Phase 0 + Phase 2).
- Open decisions (Tony): **7** pt21 gpt-oss-safeguard vs gpt-oss (thesis- vs paper-scope);
  **8** α-dial calibration organism (~$10–15; implementation + tests ready, could share the
  Phase-2 pod session); F5 fallback training gate; Phase-2 spend sign-off.
