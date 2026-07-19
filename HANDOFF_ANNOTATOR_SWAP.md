# HANDOFF: annotator-swap gate run (last declared gate in thesis ch:safety)

**Task**: re-run the spillover span extraction with NOVA-PRO's annotations instead of
Sonnet's, then re-run the directional analysis. If the recipe-direction result
reproduces under a second annotator's span boundaries, the safety chapter's last
caveat ("single-annotator") becomes a passed gate. Budget: ~2-3h GPU (~$2).

## Context (read these for full picture)
- `RESULTS_LEDGER.md` §B2 — what the spillover/attribution results are.
- Thesis claim being hardened: `thesis/chapters/v2/safety.tex` results section
  (cos(safety_arm, STAR1_dir) ≈ 0.58 vs control ≈ 0.14; final caveat sentence says
  "annotator-swap gate is un-run").
- Everything needed is in this repo; the models are on HF (cached on pod volumes).

## Steps (on the new pod)
1. Push code+data (rsync needs `--no-owner --no-group --no-perms` for /workspace;
   `apt-get install -y rsync` on the pod first — RunPod images lack it):
   `04_extract_activations.py`, `src/`, `configs/`, and
   `data/annotated_R1-1.5B__nova-pro.json`, `data/chains_R1-1.5B.json`.
2. On pod (export HF_HOME=/workspace/hf if a cache volume is attached):
   - Symlink/copy `annotated_R1-1.5B__nova-pro.json` to the name the extractor
     expects per its `--annotated` or short-name convention (READ the script's CLI
     first; pattern from previous runs:)
   - Extract BASE:  `python3 04_extract_activations.py --model 1.5b \
       --annotated data/annotated_R1-1.5B__nova-pro.json \
       --short-name R1-1.5B-novaspans --tokenizer-alias 1.5b --layers 12 16`
   - Extract STAR1: same but `--model star1-1.5b --short-name STAR1-1.5B-novaspans`
   - CHECK the script's actual flags with --help; if `--annotated` doesn't exist,
     the annotated file path may come from config/convention — inspect how
     runpod_safety_runA.sh did it (it symlinked data/annotated_STAR1-1.5B.json).
   - GPU dtype note: if pod torch is cu124 + Blackwell GPU → upgrade torch to cu128
     (`pip install -U torch --index-url https://download.pytorch.org/whl/cu128`)
     and `pip uninstall -y torchvision` (stale torchvision breaks transformers).
3. Pull both activation dirs back to `data/activations/` locally (~1-6G each).
4. Local analysis: rerun the direction comparison on nova-span activations —
   adapt `pt04_perarm_directional_null.py` (change BASE/ARMS dirs to the
   `-novaspans` pair; the STAR1 direction vs base, coherence, and — key —
   cos(direction_novaspans, direction_sonnetspans) at L12/16).
   PASS = translation direction reproduces (cos to Sonnet-span direction high,
   ~>0.8) and coherence clears its chain-level null (~0.05 floor).
5. Record in `RESULTS_LEDGER.md` §B2 (new row, pattern of existing rows) and update
   the thesis caveat sentence in `thesis/chapters/v2/safety.tex` (the sentence
   "One caveat remains: the annotator-swap gate is un-run...") to report the gate's
   outcome honestly (pass OR fail). Rebuild: `cd thesis && tectonic ucl_msc.tex`
   (must stay 0 errors; ~68pp).

## Coordination notes
- ANOTHER pod is currently running the R3/R4 DPO/GRPO pipeline (ssh alias `runpod`
  = 213.173.107.78:19557, euro-2 volume, finishes ~late morning). DO NOT touch it.
  Its Mac-side `rl_puller.sh` has SYNCED_OK deliberately deferred (this swap task
  was going to ride its grace window). Once THIS handoff task is done instead, tell
  the main session OR simply run:
  `ssh runpod "touch /workspace/rom-rl/SYNCED_OK.marker"` AFTER the R3/R4 pipeline's
  ALL_DONE marker exists and the puller log shows the final sweep completed —
  that releases the pod to self-terminate promptly.
- Do NOT git-commit anything; thesis/ is gitignored; leave working tree as-is.
- Pod self-kill pattern if you want it: see `pod_runner3.sh` kill_pod() (pod-scoped
  GraphQL terminate via /etc/rp_environment).

# PART 2: gpt-oss H1 prep (begin AFTER the swap gate, same session OK)

**Goal**: prepare (not run) the gpt-oss-20b H1 pilot — the fingerprint campaign's
gate. Approved tier: reliability pilot only (~$30-60 API), pending Tony's final go.

1. READ: thesis/chapters/v2/safety.tex (H1 operationalisation: capability control,
   three legs decided separately), and the design docs
   `../safety_reasoning_extension.md` (F1-F15, esp. F3 capability control, F5 power)
   in the PARENT directory.
2. Branch `safety/gpt-oss-extraction` has the built modules (CoT-span extraction,
   capability control, forgery builder, de-confounded fingerprint; suite was 303
   tests green on 2026-06-13). Check it out in a WORKTREE (don't disturb the main
   branch), run the test suite, report drift/breakage.
3. Draft the pilot execution plan into `GPT_OSS_H1_PILOT_PLAN.md`: (a) generate
   ~100 gpt-oss-20b chains (harmful+benign+capability-control prompts, one effort
   level) on a pod — gpt-oss-20b runs in MXFP4 on 24GB, safer on 48GB; (b) DSR
   4-label annotation of the pilot via the lab Bedrock proxy (Sonnet), κ reliability
   gate BEFORE any geometry; (c) cost table per tier. DO NOT spend API credits
   without Tony's explicit go in that session.
4. Log status updates in RESULTS_LEDGER.md and memory conventions per repo docs.
