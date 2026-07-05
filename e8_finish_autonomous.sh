#!/bin/bash
# ============================================================================
# E8 — FULLY AUTONOMOUS finish (Tony away; explicit autonomy granted 2026-06-29).
# Runs unattended on the laptop, start to finish:
#
#   1. SYNC      pull the pod's generation until ALL 1650 chains are local
#                (aborts safely if generation stalls — no terminate, no annotate)
#   2. VERIFY    confirm 1650 chains are local before doing anything destructive
#   3. TERMINATE terminate the RunPod pod via the RunPod API (self-scoped key) so
#                it costs nothing more AND is never used for annotation. Falls back
#                to podStop, then to an SSH kill of the pod's annotation, and flags
#                loudly if it cannot stop the pod at all.
#   4. ANNOTATE  Sonnet annotation locally (sharded, self-healing) — no GPU
#   5. ANALYSE   Δ_floor headline -> results/eval/R1-1.5B__E1/delta_floor_report.json
#
# Start it so it survives the Terminal/Claude closing:
#   nohup bash e8_finish_autonomous.sh > e8_autonomous.out 2>&1 &
# Keep the laptop OPEN + PLUGGED IN. Fully resumable: re-run if interrupted.
# ============================================================================
set -u
cd "/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold" || exit 1

caffeinate -dimsu -w $$ &            # keep the laptop awake for the whole run (~7h)

OUT="results/eval/R1-1.5B__E1"
EXPECTED=1650
LOG="$OUT/e8_autonomous.log"
mkdir -p "$OUT"
log(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

if [ ! -f "$HOME/.rom_runpod_env" ]; then log "FATAL: ~/.rom_runpod_env missing"; exit 1; fi
# shellcheck disable=SC1091
source "$HOME/.rom_runpod_env"       # RUNPOD_KEY, RUNPOD_POD_ID
GQL="https://api.runpod.io/graphql?api_key=$RUNPOD_KEY"

log "=== E8 AUTONOMOUS FINISH: sync → verify → terminate pod → annotate → analyze ==="

# ── 1. sync until generation is complete ───────────────────────────────────
log "[1/5] syncing generation until all $EXPECTED chains are local…"
if ! bash keep_syncing_until_done.sh; then
  log "ABORT: generation did not complete (stalled/incomplete). NOT terminating the pod,"
  log "       NOT annotating. Investigate the pod (gen.log), then re-run this script."
  exit 1
fi

# ── 2. verify local completeness BEFORE anything destructive ───────────────
N=$(python3 -c "import json;print(len(json.load(open('$OUT/steering_results.json'))))" 2>/dev/null || echo 0)
log "[2/5] verified $N / $EXPECTED chains local"
if [ "${N:-0}" -lt "$EXPECTED" ]; then
  log "ABORT: only $N/$EXPECTED chains synced locally — refusing to terminate the pod"
  log "       (its volume persists, but terminating now risks the un-synced tail). Re-run."
  exit 1
fi

# ── 3. terminate the pod (API; self-scoped key) ────────────────────────────
gql(){ curl -s --max-time 30 -X POST "$GQL" -H "Content-Type: application/json" -d "$1"; }
# return 0 (running) unless we get a CLEAR signal it is gone/stopped; an empty or
# ambiguous read counts as "still running" so we keep trying (never a false-clear).
pod_running(){
  local s; s=$(gql "{\"query\":\"query { pod(input:{podId:\\\"$RUNPOD_POD_ID\\\"}){desiredStatus}}\"}")
  echo "$s" | grep -q '"desiredStatus":"RUNNING"' && return 0
  echo "$s" | grep -q '"pod":null'        && return 1
  echo "$s" | grep -q 'POD_NOT_FOUND'     && return 1
  echo "$s" | grep -q '"desiredStatus"'   && return 1   # some other (non-RUNNING) status
  return 0
}
log "[3/5] terminating pod $RUNPOD_POD_ID via RunPod API…"
TERMINATED=0
for attempt in 1 2 3; do
  r=$(gql "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}")
  log "  podTerminate #$attempt: ${r:0:160}"
  sleep 8
  if ! pod_running; then TERMINATED=1; log "  ✓ pod terminated (no longer RUNNING)"; break; fi
done
if [ "$TERMINATED" = 0 ]; then
  log "  terminate unconfirmed — trying podStop (halts GPU billing)…"
  r=$(gql "{\"query\":\"mutation { podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}){id desiredStatus} }\"}")
  log "  podStop: ${r:0:160}"; sleep 8
  if ! pod_running; then TERMINATED=1; log "  ✓ pod STOPPED (GPU billing halted; remove it manually to fully delete)"; fi
fi
if [ "$TERMINATED" = 0 ]; then
  log "  ‼ COULD NOT STOP POD VIA API — SSH fallback: killing the pod's annotation only"
  ssh -o ConnectTimeout=20 runpod 'tmux kill-session -t e8full 2>/dev/null; pkill -f annotate_steered.py 2>/dev/null; pkill -f 08_steering_analysis.py 2>/dev/null' 2>/dev/null
  log "  ‼ POD MAY STILL BE RUNNING — TERMINATE IT MANUALLY in the RunPod console."
fi

# ── 4 + 5. annotate locally + Δ_floor analysis ─────────────────────────────
log "[4/5]+[5/5] local annotation + Δ_floor analysis (no GPU needed)…"
bash run_local_annotation.sh

log "=== E8 AUTONOMOUS FINISH COMPLETE — report: $OUT/delta_floor_report.json ==="
