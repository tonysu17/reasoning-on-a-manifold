# Session handoff — 2026-08-17 evening (J-space arc complete; 7B pair IN FLIGHT)

**Read first:** `RESULTS_LEDGER.md` §B6 (canonical status) and
`results/jspace_r1_pilot/diagnostics/DECISION_MEMO.md` (+ addendum + re-analysis). Memory file
`jspace_jacobian_lens_programme.md` mirrors this. Repo HEAD at handoff: `2100e0a` (pushed).

## LIVE: 7B matched pair running on an ephemeral RunPod L40S

- **Pod:** `ssh root@202.181.159.233 -p 14182 -i ~/.ssh/id_ed25519` — L40S 48GB, 100GB container
  disk, NO volume (deliberate; quota-proof). **Billing until stopped.**
- **Sealed:** `results/prereg/JSPACE_7B_PAIR_SHEET_2026-08-17.md` — E1–E5 registered before any 7B
  number; envelope $10/10h. Cells: `Qwen/Qwen2.5-Math-7B@b101308f` vs
  `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B@916b56a4` (28L, d=3584, both ungated).
- **Job:** `runpod_jspace_7b_pair.sh` under nohup; per model: fit (`jspace_lens_fit.py`, OOM ladder
  8→4→2) → score (`jspace_d2d3_readout.py --stage d3`, labels `math-7b-base` / `r1-distill-7b`) →
  cache wipe. Log `/root/pair7b.log`; stage markers `=== FIT|SCORE <label> START|DONE|FAILED ===`;
  completion `ALL 7B PAIR STAGES COMPLETE`; tarball `/root/jspace_7b_pair_results.tar.gz`.
- **At handoff (~21:15 UTC):** git clone still transferring (slow DC pipe; ~1GB repo); script chains
  automatically after. Expect ~5–6h from first fit ⇒ done late night / by morning.
- **Post-run checklist:** (1) scp tarball to scratchpad; (2) extract `results/jspace_r1_pilot/
  diagnostics/d3/*` bundles + `/root/lenses` fit metas into repo; (3) `.venv-jlens/bin/python
  jspace_pair_report.py` → E1–E5 verdict at `diagnostics/pair7b/`; (4) commit evidence + report;
  (5) **STOP THE POD**; (6) update `DECISION_MEMO.md`, ledger §B6, and the thesis replication slot
  (last sentence of `sec:safety-distill-substrate` + a 7B row in `tab:appendix-jspace-cells`).
- Watchers/monitors from the old session die with it — check the pod directly or re-arm.

## What this arc established (compressed; full numbers in memo/ledger)

Phase-1 lens gate FAIL (registered, unchanged) → D1–D5 closed all six accounts: scorer exonerated
(hosted 7B lens 3/3), precision + averaging rejected (exact local Jacobian equally blind at L17),
association outside competence (0.061 w/ CoT), multihop only-via-generated-chain (0.617 vs 0.099,
bridge verbalized 0.716). Base-control: base 0.407 vs distill 0.185 multihop, **both 0.000 mid-stack**
⇒ mid-stack absence predates distillation; loss selective (knowledge bridges lost, symbol kept);
base late signal proto-output (logit 0.463 > J 0.407). **Do-not-write rule:** no "workspace
relocation" claims — supported claim is selective halving of late-layer proto-output silent
resolution. Cross-model: qwen3-1.7b 3/3, gemma-3-1b 2/3 (only real mid-stack band), 7B-it 3/3 but
mid-stack 0.000; association gradient 0.000@1B→0.061@7B; J−logit format gradient −0.056→+0.037→+0.093.

## Thesis state (SEPARATE repo `thesis/`, `git -C thesis status` FIRST)

Tony's reframe edits are UNCOMMITTED in 5 files (his bundle to commit). My additions (also
uncommitted, append-only): `chapters/v2/safety.tex` new §`sec:safety-distill-substrate` (after RQ3
answer), `chapters/v2/appendix_exploratory.tex` new §`app:jspace-pilot` (+ cells table),
`references.bib` + `gurnee2026verbalizable`. Builds clean: `cd thesis && tectonic ucl_msc.tex` →
68pp; 8 BibTeX warnings are PRE-EXISTING (stash-tested). Verify content via pypdf (fi-ligature
gotcha when grepping extracted text).

## Owed next (priority order)

1. Finish 7B pair post-run checklist above (incl. POD STOP).
2. **Ph2 arm-differential missingness sensitivity analysis** — still the biggest real thesis blocker
   (see memory `ph2_annotation_coverage_saga`); do not let J-space displace it again.
3. Tony to commit the thesis bundle (his reframe + the two J-space sections).
4. Optional parked: gpt-oss-20b DSR × hosted lens (Path C); note its lens identity-distance 0.94.

## Ops lessons this session (also in memory)

Volume quota killed the first D2/D3 attempt (old caches on persistent volume) ⇒ 7B job is
ephemeral/container-only. venv on MooseFS volume = painfully slow smallfile writes ⇒ venv on
container disk. `pgrep -f <script>` false-positives on the launcher's own cmdline. caffeinate does
not survive lid-close (clamshell) — but jlens fits are checkpoint-per-prompt resume-safe.
