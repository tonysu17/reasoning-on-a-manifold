# Session handoff — 2026-07-13 (~22:30) — post-training geometry + creativity strands

Read this to pick up a live, multi-pod research session. Repo:
`/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold`. Canonical
trackers: `RESULTS_LEDGER.md` (what/status), `METHODOLOGY.md` (how). Thesis (gitignored):
`thesis/ucl_msc.tex`, build `cd thesis && tectonic ucl_msc.tex` (68pp, 0 undefined).

## THE BIG PICTURE
Two thesis movements running in parallel:
- **Safety / post-training geometry** (Movement 2): does post-training leave a geometric
  signature on generic reasoning? Spillover chapter is DONE + hardened; now extending to
  RL methods (entropy frame) + gpt-oss H1 fingerprint pilot.
- **Creativity / entropy** (separate strand, `../creativity_entropy_extension.md`): R0–R3
  ladder. R3 strategy-entropy full run executing now.
Unifying result emerging: **post-training spends entropy in the DECODING distribution, not
representational geometry** — SFT translates (no contraction), RLVR compresses behaviour
not representation. pt12 + R1-compression converge on this.

## LIVE PODS (all on RunPod; ssh aliases in ~/.ssh/config as runpod2/3/4; API key DEAD so
## Tony must provision pods; each runner self-terminates via deadman; euro ROM volume =
## "rom" in Storage tab; region-locked to euro, NOT eu-se-1)

| alias | pod | job | state @22:32 | ETA |
|---|---|---|---|---|
| runpod2 | Pod B (32GB Blackwell, ROM vol) | **R3 full run** (creativity) | 2160/3584 chains | ~06:00 |
| runpod3 | Pod 3 (32GB Blackwell, ROM vol) | **RL re-dose v3** (safety entropy frame) | GRPO-refusal 323/400 | ~06:00-07:30 |
| runpod4 | A40 46GB (eu-se-1, fresh vol NOT ROM) | **gpt-oss P0** smoke3 running | smoke3 in progress | smoke mins; full P0 after |

## AUTONOMOUS MACHINERY (Mac-side background jobs + monitors)
- `rl_puller_v3.sh` → pulls RL v3 arms when V3_ALL_DONE, drops V3_READY_FOR_ANALYSIS.
- `r3_puller.sh` → pulls R3 when R3_ALL_DONE, drops R3_FULL_READY.
- `p0p1_runner.sh` (NOT yet launched — launch when full P0 starts) → waits P0_DONE → pulls
  chains → auto-runs P1 DSR annotation via proxy → drops P1_READY_FOR_REVIEW.
- Completion monitors are armed for each; they wake the session to run analysis.
- Deadman pattern in every pod_runner*.sh: podTerminate GraphQL via /etc/rp_environment
  after grace; global watchdog. Results persist on volume regardless of laptop.

## COMPLETED RESULTS THIS SESSION (all in RESULTS_LEDGER §B2, numbers there)
1. **Spillover chapter fully hardened** — every pre-registered gate run + passed:
   rotation RECALIBRATED (pt05: was miscalibrated null → now bounded sub-degree rotation),
   per-arm bootstrap null (pt04), DC decomposition + chain coherence null (pt04b),
   seed replication 3/recipe (pt06: recipe-specificity holds, within-recipe cos 0.989-0.994),
   k-sweep + depth profile (pt07), full-FT + off-policy control (consolidated run: LoRA-regime
   + off-policy caveats discharged), surprisal gate C4 (passed), **annotator-swap gate**
   (nova-spans cos 0.999 — done by a PARALLEL session). Thesis 68pp clean.
2. **pt12 method plane (partial)** — SFT = translate-without-contract CONFIRMED across dose
   ladder; RL side under-dosed in v1 (512-tok cap truncated rollouts). **DPO-v2 landed the
   key new result**: DPO trained on SAME data as SFT moves along a DIFFERENT axis (cos 0.16-0.20
   to STAR1 vs SFT 0.58), RAISES entropy (+0.014), slightly expands spectrum ⇒ installed
   direction is METHOD-dependent not just data-dependent. (`pt12_entropy_frame_verdict.py`)
3. **Creativity R0/R1/R2/R3-pilot** (RESULTS_LEDGER §B5): R0 entropy-ladder dissociates;
   R1-compression = behavioural-not-representational compression; R2 measurement-negative
   (token diversity saturates ⇒ motivated R3); R3 pilot passed all 4 gates.
4. **Lit review** (`LIT_POSITIONING_2026-07-13.md`): nearest threats = 2606.23740 (weight-space
   SFT-vs-DPO, 3wk old — our activation/KL/entropy deltas survive), SPREAD 2511.08305 (R3),
   Ponkshe 2505.14185 (H1 null to beat). **H1-on-gpt-oss UNCLAIMED as of today; runway weeks-months.**

## RUNNING EXPERIMENTS (what completes overnight → auto-analysis)
- **RL v3 (Pod 3)**: GRPO-refusal + GRPO-math (group4/microbatch2/gradckpt after 4 OOM fixes)
  + DPO-control re-dose 6ep + extract + pt08. Completes pt12 method plane (GRPO point +
  controlled DPO). On V3_READY: rerun `pt12_entropy_frame_verdict.py` incl -v3 arms.
- **R3 full (Pod B)**: 3584 chains, thermostat (T 0.3-1.2) + pump (bt-amp) arms, then on-pod
  full-analyse → strategy-entropy × value plane. On R3_FULL_READY: read FULL_REPORT.md.
- **gpt-oss P0 (A40)**: ~100 chains (40 harmful StrongREJECT / 40 benign XSTest / 20 capability).
  Smoke3 confirming analysis-channel capture NOW.

## CONSENTS GIVEN BY TONY (do not re-ask)
1. **P1 auto-runs after P0** (no hold-for-review) — P0 → pull → P1 DSR annotation via Bedrock
   proxy (3 judges: Sonnet/Qwen3/Nova) → per-label κ + sealed gates. Envelope $30-60. Kill
   criterion: if `decision` label κ<0.4, schema broken, stop. (`p0p1_runner.sh` does this.)
2. **Tony is the human gold anchor** — prepare a ≥100-sentence gold-annotation file (CSV or
   HTML — he hasn't picked; ASK) for him to label; fold human-vs-LLM κ in after.
3. **Auto-fold pt12/R3 verdicts into thesis prose** + rebuild + coherence check, show diff
   post-hoc. Ledger updates already autonomous.

## PENDING / NEEDS TONY
- Balance watch: ~$14-16/night dual-pod burn (now 3 pods). API key dead → he provisions pods.
- His gold-set labelling (~1-1.5h) when P1 sentences ready — non-blocking.
- gpt-oss P2 (H1 geometry) is a SEPARATE decision after P1 κ gate — NOT approved yet.

## KEY OPS LESSONS (bit us this session — avoid re-learning)
- RunPod community pods get RECLAIMED mid-run; endpoints remap between pods. Verify volume
  contents before acting (`ls /workspace` expect rom-rl/rom-r3/hf).
- Pod images LACK rsync → `apt-get install -y rsync` first; rsync to /workspace needs
  `--no-owner --no-group --no-perms`.
- torch cu124 has NO Blackwell/sm_120 kernels AND no `torch.accelerator` (gpt-oss MXFP4 needs
  it) → upgrade `pip install -U torch --index-url .../cu128`; then `pip uninstall torchvision`
  (stale torchvision breaks transformers import).
- **gpt-oss-20b load**: `device_map` offloads MoE experts to CPU → garbage generation. FIX
  (patched in worktree `src/chain_gen.py`): force `.to('cuda:0')` when any param off-GPU.
  MXFP4 dequants to bf16 (~40GB, fits 46GB A40; NOT 32GB). Run OFFLINE (HF_HUB_OFFLINE=1) —
  `hf download <repo>` fetches 18 redundant files that STALL; we only need the 3 MXFP4 shards.
- GRPO OOM chain: group8→OOM, group4→gen-OOM (grad-accum), still OOM→gradient_checkpointing,
  still→micro-batch<group. Final working: group4 + micro-batch2 + grad-accum2 + grad-ckpt.
- pgrep -f self-matches your own kill command; use bracket `[p]attern` or separate connections.

## WORKTREE
gpt-oss code on branch `safety/gpt-oss-extraction`, checked out at `../rom-safety-worktree`
(HEAD ~05a75e3, suite 205/205 green). P0 = `p0_generate_gptoss_chains.py`; P1 = `14b_annotate_dsr.py`.
Plan: `GPT_OSS_H1_PILOT_PLAN.md`. Stimuli built: `data/gptoss_stimuli.json` (gitignored).

## IMMEDIATE NEXT ACTIONS (in order)
1. Watch smoke3 verdict → if analysis-channel captured, launch FULL P0
   (`pod_p0_gptoss.sh` on runpod4, offline env) + launch `p0p1_runner.sh`.
2. On V3_READY / R3_FULL_READY (overnight): run pt12 (incl v3) + read R3 FULL_REPORT →
   auto-fold verdicts into thesis + rebuild + show Tony diff.
3. On P1_READY: report κ gates; build Tony's gold-annotation file; STOP for P2 decision.
