# SESSION HANDOFF (EVENING) — post-training transport programme, 2026-08-08

Supersedes `SESSION_HANDOFF_2026-08-08.md` (morning; pre-batch). A fresh session should read,
in order: `INTEGRATED_THESIS_POSTTRAIN_PLAN_2026-08-08.md` → this file →
`results/prereg/PHASE0_TRANSPORT_FREEZE_2026-08-02.md` (§4/§4b) →
`results/prereg/PHASE2_TRANSPORT_PREREG_2026-08-08.md` (+ A1, A1-correction, A2 file, A3) →
`RESULTS_LEDGER.md` 2026-08-08 entries → `.codex/out/CLAUDE_HANDOFF_STATUS_2026-08-08.md`.
Persistent memory (`post_training_reasoning_spillover.md`) mirrors everything below.

## 1. Where things stand (all committed on `codex/phase0-support`; NOT pushed)

- **Phase 0 CLOSED** — commit `b44d392`; dispositions in freeze §4b. Batch science:
  s0 → **F5 route = FALLBACK_65PAIR** (36.4 GPU-h projected > 12 h rule; F5 itself
  prospective/unrun); s2 → STAR1 inert extraction complete (deduction 34,848 / initializing
  4,863 @L12/16; `FAILED:s2` was cosmetic); s3 → **Gate S3-A FAIL: the July DeepScaleR paired
  dPR is not re-extraction-stable** (+0.139 vs −0.0167, sign flip, both ≈0 vs PR≈30 ⇒ mapping
  NOT LICENSED, unlicensed bounds only: deepscaler ≈0.08%, star1 ≈1.0%); **family check PASS:
  SV-ratio flat 0.984–0.986 all ranks ⇒ uniform ≈1.5% global shrink, NOT tail contraction**.
- **Phase 2 SEALED + tooled** — prereg + A1 (fresh-task manifest; corrected: dynamic id start
  = `_117`) + **A2 sealed** (observational vanilla-arm adjunct) + **A3 sealed** (owner
  decision: **Sonnet-only annotation** everywhere; builder-annotator caveat declared;
  Nova clause superseded). **Manifest COMMITTED** (`4482161`):
  `results/prereg/phase2_task_manifest.json`, n=100, 10/category, ids `_117+`,
  **ids_sha256 `c7fefd59557f95a35f621787c2ab2c19f149ff96847e504c8295ee0f182c96d6`**.
- **Executor**: `ph2_executor.py` core implemented + **16/16 pre-spend tests**
  (`tests/test_ph2_executor.py`); model-touching stages are spec'd stubs (docstrings = build
  contracts). `ph2_manifest.py --verify` re-checks the manifest any time.
- **Proxy creds CONFIGURED** in `~/.zshrc` (`CLAUDE_PROXY_URL`/`CLAUDE_PROXY_KEY`; endpoint
  `https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke`; transport:
  X-Api-Key header, `content[0].text`, log `usage.cost` + `remaining_quota`).
  **HARD RULE: 29-second AWS API-Gateway timeout on every proxy call** — chunk, modest
  max_tokens (≤~1,000), backoff-retry on 504, ≤2 concurrent. **Key touched a chat transcript —
  ROTATE after the current push (Tony).**
- **Pod terminated** (`yifdnc2lujeb29`). The euro-2 volume persists: s3 artifacts
  (`results/safety_posttrain/ph0_s3/{r1,deepscaler}_fullseq.npz`, `s3_analysis.json`), the
  STAR1 inert extraction (pod-side `data/activations/STAR1-1.5B/` — pull to LOCAL STAGING
  `STAR1-1.5B-6label`, never the local canonical dir), `venv-r1` (TRL env for F5), RL adapters,
  `keep_lora_control_ckpt`. Local: full-FT s42 pair at `checkpoints/pod_fullft/`.

## 2. Decisions log (today)

- **Cap collision RESOLVED (Tony): the shared 100-task vanilla set STAYS SEALED at E8
  generation settings.** P5 adds its own long-cap generic arm if its pilot demands it.
- Sonnet-only annotation (A3). F5 = fallback route, prospective/unrun. pt02 LoRA arms:
  definitively absent everywhere ⇒ retrain-conditional (~$1–3, manifests required).
- Codex's P5 pilot is Tony-authorized and running (106/176 generations at last report);
  scoring (212 Sonnet calls) unblocked via env creds.
- **STILL OPEN (Tony):** P1 author–supervisor scope amendment (blocks Phase-2 + P5-powered
  spend); Phase-2 spend envelope (~$20–40 pod + ~$100–250 annotation); early stand-alone
  vanilla slice (~300 gens, ~$3–5) IF codex's powered freeze needs the shared artifact before
  Phase-2 launch; key rotation. **Deferred by plan §8:** α-dial, pt21, RLVR sweep, KTO
  (follow-on paper), agentic, order effects.

## 3. Next-session queue (Claude)

1. **Build the executor model stages** against their stub specs: discovery extraction
   (teacher-forced corpus spans, byte-identical ids, L16/17/18), target gates (20 sham + 20
   rand-orth, p95), sealed re-fit glue, resumable battery runner, Sonnet annotation glue
   (29-s chunking), analysis stage (authoritative `src/delta_floor` pooling, Holm,
   `decide_outcome`). Then the remaining pre-spend checks (identity/reload gate, plumbing
   dry-run, resume + failure-path tests on real stages).
2. **Next pod session opener** (any 4090; fresh containers need: `apt install rsync tmux`,
   `pip install -e ".[gpu]" "transformers==4.49.0"`): pull s3 artifacts + STAR1 inert staging
   → run the inert-control battery locally (freeze §1.3: gated rotation/translation/d_eff,
   Holm over {2 behaviours × 2 layers}) → **F5 fallback** (push the 65-pairs json — find it in
   local `results/safety_posttrain/rl/` or `_pod_backup` — plus `merge_adapter.py` +
   `pt08_surprisal_entropy.py`; train `pt10_train_dpo.py --epochs 23` in `venv-r1` to matched
   steps/KL 0.000587; merge; extract L12/16).
3. **Phase-2 launch** once P1 + spend land: provenance gates → discovery → target gates/re-fit
   → injection-recovery (pre-outcome, f∈{0.25,0.5,0.75}, pass=80%@f=0.5) → battery → Sonnet
   annotation → five-outcome verdicts + A2 adjunct table. Sequencing rules live in the prereg.
4. **Watch for codex's three owed items**: `ph2_mde_sim` authoritative-pooling confirmation;
   P5 generation-config requirements (parser/scorer compatibility for the shared set);
   evidence-manifest field spec (shape executor provenance to match).

## 4. Infra gotchas (bitten this week — do not relearn)

- Fresh pod containers lose everything container-side: rsync, tmux, the transformers pin
  (`==4.49.0` for torch-2.4.1 images), every time. `src/chain_gen.py` now uses `torch_dtype=`
  (committed). rsync to the mfs volume: `-rltz` (no chown). `df` on the volume shows
  cluster-wide space — gate on `du -s /workspace` (70 G rule). `--short-name` in
  `04_extract_activations.py` only works with `--model-path`.
- macOS TCC can revoke Documents access mid-session (Edit/python EPERM while ssh still works):
  fix = quit + reopen the app; grant Full Disk Access to prevent recurrence.
- Kill discipline: watchers key on `PH0_DONE.marker`-style markers, never status strings;
  no on-pod self-kill mid-job; terminate after verified pulls (or rely on the volume).
- The annotation/scoring proxy is shared with codex — coordinate, stay ≤2 concurrent.

## 5. Start-of-session checklist

1. `git -C . status` + `git log --oneline -3` (expect `4482161`, clean-ish tree) and
   `git -C thesis status` separately (own repo).
2. Read the §0 document chain; recheck open decisions (§2) with Tony.
3. Do not launch paid generation/annotation without the sealed preconditions (prereg §12).
4. Verify `zsh -ic 'env | grep -c CLAUDE_PROXY'` = 2 before any proxy work; respect 29 s.
