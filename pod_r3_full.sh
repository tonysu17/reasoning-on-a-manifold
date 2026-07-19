#!/bin/bash
# POD-SIDE runner for the FULL R3 strategy-entropy run (R3_PILOT_PREREG.md
# Amendment 2; runner 32_r3_strategy.py --stage full-*).
#   full_tasks    (CPU, seconds)  64 stratified tasks → data/r3_tasks_full.json
#   full_generate (GPU)           3,584 chains = 64 tasks × k8 × 7 cells
#                                 (thermostat T{0.3,0.6,0.9,1.2} + bt-pump
#                                 α{0.5,1.0,1.5}@T0.6), max_new 6144, batch 8,
#                                 resume-safe (rerun this script after any
#                                 crash — done rows are skipped)
#   full_analyse  (CPU, minutes)  strategy-entropy report → FULL_REPORT.md
# Assumes the repo is synced to /workspace/rom-r3 INCLUDING
# results/steering_vectors/R1-1.5B__E1_pooled/ (pump vector; preflight checks).
# Generation estimate on a 4090 @ batch 8: ~6–8 h (R2 calibration ≈ 2.2M
# tok/GPU-h; 3,584 chains × ~3.5k tok ≈ 12.5M tok) ⇒ GLOBAL_H = 12 (est + 50%).
# Deadman: after R3_ALL_DONE, waits GRACE_H for the Mac to pull results and
# touch R3_SYNCED.marker, then podTerminates either way. A background watchdog
# enforces GLOBAL_H even mid-generation. /workspace is a persistent volume:
# results survive pod death.
# Launch: setsid nohup bash pod_r3_full.sh > pod_r3_full.log 2>&1 < /dev/null &
set -u
cd /workspace/rom-r3
export HF_HOME=/workspace/hf
GRACE_H=4; GLOBAL_H=12; START=$(date +%s)
mkdir -p results/r3_strategy data
log(){ echo "$(date -u '+%F %T') | $*"; }
kill_pod(){ log "KILL"; source /etc/rp_environment
  curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; sleep 60; }
gc(){ [ $(( $(date +%s) - START )) -gt $(( GLOBAL_H * 3600 )) ] && { log "WATCHDOG"; kill_pod; }; }
stage(){ local n="$1"; shift
  [ -f "R3_DONE_${n}.marker" ] && { log "stage ${n}: cached"; return 0; }
  log "stage ${n}: start"
  if "$@" >> "r3stage_${n}.log" 2>&1; then touch "R3_DONE_${n}.marker"; log "stage ${n}: DONE"
  else touch "R3_FAILED_${n}.marker"; log "stage ${n}: FAILED"; return 1; fi; }

log "=== R3 full run start ($(grep -oE 'RUNPOD_POD_ID=\S+' /etc/rp_environment)) ==="

# GLOBAL watchdog must bite even inside the long generate stage
( while true; do gc; sleep 300; done ) & WATCHDOG_PID=$!

# 0. preflight: pump vector + generation deps present (fail fast, not 6h in)
stage preflight python3 -c "
import numpy as np, torch, transformers
v = np.load('results/steering_vectors/R1-1.5B__E1_pooled/backtracking_single.npy')
assert v.ndim == 1, v.shape
print('vector OK', v.shape, '| cuda:', torch.cuda.is_available())
" || { log "preflight FAILED — refusing to burn GPU hours"; kill_pod; }

# 1–3. the run (each stage cached by marker; full-generate resumes internally)
stage full_tasks    python3 32_r3_strategy.py --stage full-tasks
stage full_generate python3 32_r3_strategy.py --stage full-generate --device auto --batch 8
stage full_analyse  python3 32_r3_strategy.py --stage full-analyse

log "stages: $(ls R3_DONE_*.marker 2>/dev/null | wc -l) DONE, $(ls R3_FAILED_*.marker 2>/dev/null | wc -l) FAILED"
touch R3_ALL_DONE.marker

# deadman: give the Mac GRACE_H to pull results/r3_strategy + data/r3_tasks_full.json
DEADLINE=$(( $(date +%s) + GRACE_H * 3600 ))
until [ -f R3_SYNCED.marker ]; do
  [ $(date +%s) -gt $DEADLINE ] && { log "deadman elapsed"; break; }
  gc; sleep 120
done
kill -9 "$WATCHDOG_PID" 2>/dev/null
kill_pod
