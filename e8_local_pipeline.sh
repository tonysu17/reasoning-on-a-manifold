#!/bin/bash
# ============================================================================
# E8 — ONE unattended laptop command: sync → annotate → Δ_floor analysis.
#
# Start this on the laptop BEFORE you step away, ideally so it survives closing
# the Terminal window:
#
#   cd "/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold"
#   nohup bash e8_local_pipeline.sh > e8_local.out 2>&1 &
#
# Then you can close Claude, close the Terminal, and walk away. Keep the laptop
# OPEN and PLUGGED IN (caffeinate stops it sleeping, but a dead battery will).
#
# What it does, in order, with no further input:
#   1. SYNC  — pulls the pod's generation output until it is complete
#              (waits for E8_gen_DONE.marker). Needs the pod UP for this step.
#   2. ANNOTATE — once generation is fully synced it NO LONGER needs the pod, so
#                 you can stop the pod remotely any time after "GEN DONE". Runs
#                 the sharded, self-healing Sonnet annotation on the laptop.
#   3. ANALYSIS — Δ_floor headline -> results/eval/R1-1.5B__E1/delta_floor_report.json
#
# Fully resumable: if anything interrupts it, just run the same command again.
# Progress: tail -f e8_local.out   (or results/eval/R1-1.5B__E1/local_annotate.log)
# ============================================================================
set -u
cd "/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold" || exit 1

# keep the laptop awake for the WHOLE pipeline (sync + annotate, ~7h)
caffeinate -dimsu -w $$ &

echo "================================================================"
echo "$(date '+%F %T') | E8 LOCAL PIPELINE — sync → annotate → analyze"
echo "================================================================"

echo "$(date '+%F %T') | [1/3] waiting for generation to finish + sync…"
if ! bash keep_syncing_until_done.sh; then    # exit!=0 ⇒ gen stalled / incomplete
  echo ""
  echo "$(date '+%F %T') | ABORT: generation did not complete cleanly — NOT annotating a"
  echo "  partial run (would waste API credits + give wrong fractions). Investigate the"
  echo "  pod, finish generation, then re-run this script (it resumes)."
  exit 1
fi

echo ""
echo "$(date '+%F %T') | [2/3]+[3/3] generation synced — annotating + analyzing locally."
echo "  (The pod is no longer needed — safe to stop it remotely now to save \$.)"
bash run_local_annotation.sh       # sharded annotation + self-heal + merge + Δ_floor

echo ""
echo "$(date '+%F %T') | E8 LOCAL PIPELINE FINISHED."
echo "  Result: results/eval/R1-1.5B__E1/delta_floor_report.json"
