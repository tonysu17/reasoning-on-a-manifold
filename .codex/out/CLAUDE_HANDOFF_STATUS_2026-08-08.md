# Claude → codex handoff status (2026-08-08, post-closure-commit)

Mirrors and updates the in-chat handoff. Authoritative sources: the sealed preregs, the
Phase-0 freeze §4/§4b, and RESULTS_LEDGER.md (2026-08-08 entries).

## 1. Phase 2 — NOT executed; corrected expectations

- **A2 adjunct amendment: SEALED** (file: `results/prereg/PHASE2_ADJUNCT_AMENDMENT_2026-08-08.md`).
- **Manifest: not yet generated.** Tooling ready + tested; generation is approved ($0.50) and
  waits only on `CLAUDE_PROXY_URL/KEY` in the environment. Fresh ids will start at **_117**
  (A1 same-day correction: corpus ids reach _116 — do not assume _100/_102 anywhere).
- **Execution/analysis: not launched.** Gates, in order: P1 author–supervisor scope amendment
  (Tony) → tested runner completion (Claude) → Tony's spend sign-off. `results/ph2/markers/*`
  will appear only from the real run — never wait on them for your current work.
- **Executor pre-spend checks: 16/16 green** (`tests/test_ph2_executor.py`).

## 2. Manifest handoff

- Path (fixed): `results/prereg/phase2_task_manifest.json` — does not exist yet; the
  `ids_sha256` lands with generation and will be relayed immediately.
- **P5 reuse of the manifest's vanilla generations: CONFIRMED and prescribed** (sealed A2 +
  integrated plan): one shared vanilla generation set (E8 sealed decoding, seed 20260808),
  generic stratum only; P5 may analyze the observational endpoints freely (separate family)
  but must not inspect Phase-2 causal-family outcomes before the injection-recovery gate.
- **Before the one-and-only generation run, Claude needs your P5 generation-config
  requirements** (max-token cap, stop conditions, template/parsing assumptions) to check
  compatibility with the E8 sealed settings.

## 3. Evidence chain

- Raw steering (ledger §B): `results/eval/R1-1.5B__E1/` — `annotated_steered.json` (1,629
  rows), `steering_results.json` (1,650 generations), `delta_floor_report.json` (floor
  contract: `src/delta_floor.py`), `eval_task_ids.json`, `generation_metrics.json`,
  `steering_geometry.json`, `collapse_table.json`, `strengthen_report.json`, `provenance.json`.
- Safety decomposition / per-seed / depth (ledger §B2): `results/safety_posttrain/` —
  `spillover_gated_full.json`, `pt04*`, `pt04c_annotator_swap.json`, `pt06_perseed.json`,
  pt07 outputs, `ANALYSIS_2026-07-04.md`.
- **Snapshot refresh: GO once the closure commit is on `codex/phase0-support`** (message
  begins "Phase-0 closure"). Note two artifact classes are volume-side until the next pod
  session (s3 fullseq npz/json; the STAR1 inert extraction) — neither is needed for your
  evidence-repair lane; their dispositions are fully recorded in freeze §4b.
- Claude still needs from you: (a) confirmation `ph2_mde_sim`'s PRIMARY curves used the
  authoritative `src/delta_floor.py` pooling (your own red-team's requirement), (b) the
  evidence-manifest field spec consumed by `refresh_evidence.sh`, so Phase-2/closure artifacts
  emit matching provenance natively.

## 4. LoRA disposition — DEFINITIVE

- pt02 LoRA-SFT arms (safety-100/300/1000, control-1000, seeds 42–44): **not found anywhere**
  (local disk + full volume search). **Retrain-conditional**: `pt02_train_safety_lora.py`,
  ~$1–3 pod, immutable run manifests required.
- Surviving on the volume (adapters intact, hashes on request): RL arms —
  `rl_adapters_keep/checkpoints/{dpo_safety,dpo_control}/ckpt_frac_{0.5,1}`,
  `{grpo_math,grpo_refusal}/final_adapter`, v3 variants in `rom-rl/checkpoints/`, plus the
  pt08 merged lora-control (`keep_lora_control_ckpt/merged/`).
- The owned **full-FT safety/control s42 pair is local** (`checkpoints/pod_fullft/`) — your
  strongest attribution contrast is available today.

## 5. Pilot authorization

The 176-generation/scoring pilot is **Tony's** authorization (spend + sequencing relative to
Phase 2), not Claude's. Put the exact cost line to him directly; spec and dry-run plumbing are
unblocked now.

## 6. New facts from the executed Phase-0 batch (context for your protocol)

- F5 routed to FALLBACK_65PAIR by pre-committed rule (500-pair generation = 36.4 GPU-h);
  fallback itself prospective/unrun until the next pod session.
- s3: the July DeepScaleR paired dPR is **not re-extraction-stable** (sign flip at ≈0
  magnitude) — treat any tiny spectral delta as unstable; and the SV-ratio profile is **flat
  (uniform ≈1.5% shrink, no tail contraction)** — spectrum-level support for the
  translation/rescaling account. Wording stays bounded per freeze §4b (mapping unlicensed).
