# Session handoff — 2026-08-17 evening (J-space arc COMPLETE; 7B pair PREPARED, NOT YET RUN)

**Read first:** `RESULTS_LEDGER.md` §B6 (canonical status) and
`results/jspace_r1_pilot/diagnostics/DECISION_MEMO.md` (verdicts + addendum + re-analysis).
Memory file `jspace_jacobian_lens_programme.md` mirrors this. Repo HEAD: `cb8b84a` (pushed).

## STATUS: nothing is running. No pod is live. Nothing is billing.

The 7B pair was **prepared and sealed but never executed** — three launch attempts on RunPod pod
`202.181.159.233:14182` (L40S) all failed for infrastructure reasons, then the pod went away
(SSH refused). **Zero results were produced and zero data was lost**; the job never reached a fit.
All local state is committed and clean.

## Next session: run the 7B pair (everything is ready)

1. Start a pod: **48 GB GPU (L40S/A6000), PyTorch template, ≥100 GB CONTAINER disk, NO volume.**
   If setup (pip/torch) crawls, that datacenter's link is bad — kill it and take a different region;
   waiting cost most of 2026-08-17's evening.
2. `scripts/launch_7b_pair.sh --host <IP> --port <PORT>` — preflights the pod, pushes an 840 KB
   payload (NO repo clone), launches fully detached, verifies it started. ~2 min to first fit.
3. (Optional, for unattended finish) store the API key **yourself**, zsh-safe:
   `bash -c 'read -rs -p "RunPod API key: " k && printf "RUNPOD_API_KEY=%s\n" "$k" > ~/.runpod_env && chmod 600 ~/.runpod_env && echo stored'`
4. `scripts/pod_sync_and_terminate.sh --host <IP> --port <PORT>` — waits for the completion marker,
   syncs every artefact, verifies sha256 + structure, installs into the repo, and ONLY THEN terminates
   the pod. Without the key file it syncs/verifies and leaves the pod up for a manual stop.
5. `.venv-jlens/bin/python jspace_pair_report.py` → E1–E5 verdict in `diagnostics/pair7b/`.
6. Commit evidence; update `DECISION_MEMO.md`, ledger §B6, and the thesis replication slot (last
   sentence of `sec:safety-distill-substrate` + a 7B row in `tab:appendix-jspace-cells`).

Expected: ~5–6 h wall (two ~2.5 h fits), ~$4–6.

## Hard-won ops lessons from the failed attempts (all now encoded in the scripts)

1. **Never clone the repo on a pod** — ~1 GB of committed results stalled twice. Push a tar payload
   over ssh instead (`launch_7b_pair.sh` does this).
2. **Always launch with `setsid nohup … & disown`** — a non-detached launch was SIGHUP-killed when the
   ssh session closed, and git removed the partial clone, so *nothing ran for 25 min while appearing
   healthy*.
3. **`pgrep -f <pattern>` false-positives on your own ssh command** (the pattern text is in the remote
   cmdline). Same for `pkill -f` — it killed my own session once. Match on `ps -eo pid,args` with a
   bracket-class regex, or kill by PID.
4. Volume quotas killed an earlier D2/D3 attempt ⇒ 7B job is ephemeral/container-disk-only by design.
5. `read -rs -p` is a **bash** idiom; the user's shell is **zsh** (where `-p` means coprocess). Wrap
   credential prompts in `bash -c '…'`.

## What the J-space arc established (complete, committed, thesis-integrated)

Phase-1 R1-1.5B lens gate **FAIL** (registered; consequence verbatim "the fitted lens did not pass the
prespecified validity gate"; Phase-2 decomposition never ran, shelved by option D). D1–D5 closed all
six failure accounts: scorer **exonerated** (hosted 7B lens 3/3 through our scorer), precision and
corpus-averaging **rejected** (exact position-local Jacobian equally blind at L17, 0/81), association
**outside competence** (0.061 with free CoT), multihop within competence **only via the generated
chain** (0.617 CoT vs 0.099 silent; bridge verbalised 0.716).

Base-control (Qwen2.5-Math-1.5B, matched): base multihop 0.407 vs distill 0.185, **both exactly 0.000
mid-stack** ⇒ the mid-stack absence **predates distillation**; loss is **selective** (knowledge bridges
lost, symbol/letter/calendar kept); base's late signal is proto-output (logit 0.463 > J 0.407).
**DO-NOT-WRITE rule:** never "distillation relocated composition out of the silent workspace" — no
mid-stack workspace existed to relocate from.

Cross-model: qwen3-1.7b 3/3, gemma-3-1b 2/3 (only genuine mid-stack band), Qwen2.5-7B-it 3/3 but
mid-stack 0.000. Association scale gradient 0.000@1B → 0.061@7B. J−logit format gradient (Qwen family)
−0.056 → +0.037 → +0.093.

## Thesis state (SEPARATE repo — `git -C thesis status` FIRST)

Tony's reframe edits are UNCOMMITTED in 5 files (his bundle). My additions, also uncommitted and
append-only: `chapters/v2/safety.tex` new §`sec:safety-distill-substrate` (main text, after the RQ3
answer), `chapters/v2/appendix_exploratory.tex` new §`app:jspace-pilot` + cells table, `references.bib`
+ `gurnee2026verbalizable`. Builds clean: `cd thesis && tectonic ucl_msc.tex` → 68 pp. The 8 BibTeX
warnings PRE-DATE these edits (stash-tested). When grepping extracted PDF text, normalise the `fi`
ligature or you'll get false negatives.

## Priorities

1. **Ph2 arm-differential missingness sensitivity analysis** — still the largest genuine thesis
   blocker (memory `ph2_annotation_coverage_saga`). Do not let J-space displace it again.
2. 7B pair run (above) — one pod-evening, <$10, upgrades a main-chapter claim to two scales.
3. Tony to commit the thesis bundle.
4. Parked: gpt-oss-20b DSR × hosted lens (Path C); check that lens's identity-distance 0.94 first.
