# SESSION HANDOFF — post-training transport programme, 2026-08-09

Supersedes `SESSION_HANDOFF_2026-08-08_EVENING.md`. Fresh-session read order:
`INTEGRATED_THESIS_POSTTRAIN_PLAN_2026-08-08.md` → this file → prereg chain
(`results/prereg/PHASE2_TRANSPORT_PREREG_2026-08-08.md` + A1/A1-corr/A2/A3) →
`RESULTS_LEDGER.md` 2026-08-09 entry → `.codex/out/CODEX_RESPONSE_TO_CLAUDE_HANDOFF_2026-08-08.md`
→ `.codex/out/P1_AUTHOR_SUPERVISOR_SCOPE_AMENDMENT_DRAFT_2026-08-09.md`.
Memory (`post_training_reasoning_spillover.md`) mirrors everything.

## 1. What changed this session (commits `f0f5998`, `7e92b92`; NOT pushed)

- **Phase-2 executor BUILT COMPLETE — all 8 stages implemented, 47/47 pre-spend
  tests, repo suite 761 green, MPS smoke on the real sealed checkpoints passed**
  (identity gate byte-identical 2/2; clamp displaces and diverges; pair-states
  extract; class-mean separation visible in raw coords). New `src/ph2_stages.py`
  + stage bodies in `ph2_executor.py`; `src/annotation.py` gained backward-
  compatible `max_tokens` + 504-shrink params (corpus defaults untouched).
  Details: ledger 2026-08-09 entry. Nothing launched; no spend.
- **Codex correction consumed: E8 cap is 8,192 (not 2,048)** — shared-vanilla
  contract stands, no Phase-2 amendment needed. All three owed codex items
  delivered (MDE pooling confirmation, shared-vanilla contract, evidence-manifest
  field spec — the last is now implemented in `build_provenance`).
- All three sealed checkpoint revisions verified present in the local HF cache
  at the exact pinned snapshots (no download needed for local/pod-prep work).

## 2. Live state

- **P5 pilot LIVE** (codex's workstream; Tony resumed ~11:00 local): 137/176 at
  handoff time, ~2 min/row, screen `p5pilot` + caffeinate. Its `generations.jsonl`
  is intentionally uncommitted while churning. Codex rebuilt the scorer
  Sonnet-only (v2/v2.1 + dry runs, no calls made): **old 212-call/$15 envelope
  is NOT provably sufficient (350 initial requests; worst-case 1,400)** — a new
  Tony authorization with explicit ceilings is required before any scoring.
- **Pod terminated** (`yifdnc2lujeb29`); euro-2 volume persists s3 artifacts,
  STAR1 inert extraction, `venv-r1`, RL adapters (see evening handoff §1 for
  paths — still accurate).
- Thesis repo: 11 modified uncommitted files (F09 κ + A2-snapshot fix pass in
  flight at `b50d321`) — not touched this session.

## 2b. Post-handoff same-day updates (2026-08-09 afternoon)

- **P1 joint scope seal APPROVED in chat** (signatures + resource checkboxes
  still open). **Prereg A4 sealed**: ~3,000-token annotation window (Sonnet
  bill $582→$275); generation cap untouched. **Scorer v2.1 protocol + gates
  approved** (ceilings still owed).
- **⚠️ CAP FORK (new launch blocker):** codex holds Tony's cost instruction as
  a 2,048–4,096 GENERATION cap constraint (+ a pause on the Phase-2 battery);
  A4 holds it as an annotation window with generation sealed at 8,192. Tony
  must pick Option A (recommended: generation stays 8,192; A4 carries the
  saving; P5 recaps post-hoc) or Option B (freeze a generation cap ⇒ amendment
  A5 + contract re-cut + boxed-correctness loss on ~43–49% of rows). Full
  reconciliation: `.codex/out/CLAUDE_TO_CODEX_2026-08-09.md` §2 + ledger.

## 3. Decisions Tony owns (nothing else blocks)

1. **P1 joint seal** — codex's draft is ready for signatures (new title, 3 RQs,
   Phase-2 + P5 as prospective scope; spend explicitly separate). Blocks all
   Phase-2/P5-powered spend.
2. **Phase-2 spend envelope** — ~$20–40 pod + extraction $5–10 + **Sonnet
   annotation re-costing** (A3 replaced Nova; the $100–250 line must be re-signed).
3. **P5 scorer authorization** — new ceiling for the v2.1 Sonnet-only scorer
   (see §2; pilot generation ceiling unchanged).
4. **Key rotation** — the proxy key touched a chat transcript; rotate after the
   next push (standing item).
5. (Pre-launch eyeball, not a seal) executor operationalizations, marked in code:
   base-role aliasing; floors run suppression-oriented; energy-floor displacement
   calibration on 5 tasks; `ATTENUATION_MARGIN = 0.5` tied to the injection gate.

## 4. Next-session queue

1. **Next pod session** (any 4090; fresh containers: `apt install rsync tmux`,
   `pip install -e ".[gpu]" "transformers==4.49.0"`):
   a. pull s3 artifacts + STAR1 inert → LOCAL STAGING `STAR1-1.5B-6label`;
   b. inert-control battery (freeze §1.3; Holm {2 behaviours × 2 layers});
   c. **F5 fallback** (65-pairs json is LOCAL — push it + `merge_adapter.py` +
      `pt08_surprisal_entropy.py`; train `pt10 --epochs 23` in `venv-r1` to
      KL 0.000587; merge; extract L12/16);
   d. **Phase-2 pod stages** (no spend flag needed): `ph2_executor.py --stage
      discovery_extract` → `target_gates` → `refit`. Wall-clock driver = 41 sham
      trainings (~few min each at the sealed budget).
2. **Phase-2 launch** once P1 + spend land: `generate_battery --authorised` →
   `injection_recovery --authorised` → `annotate --authorised` → `analyse`.
   Sequencing/gates live in the prereg; the executor enforces refusals.
3. **Watch codex**: P5 pilot completion + scorer v2.1 authorization request +
   the powered-design freeze (which may want the early stand-alone vanilla slice).

## 5. Infra gotchas (inherited; see evening handoff §4 for the full list)

- 29-s proxy ceiling on every call; ≤2 concurrent; chunking handles it.
- Layer-index trap: prereg sites are hs-indices; hs[K] = machinery layer K−1
  (`src/ph2_stages.machinery_layer`, locked by test — do not hand-translate).
- macOS TCC can revoke Documents access mid-session; fix = quit + reopen app.
- Watchers key on markers, never status strings; terminate pods only after
  verified pulls.
