#!/bin/bash
set -u
cd "$(dirname "$0")"
LOG=results/safety/p0_puller.log
mkdir -p results/safety data
log(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
log "=== p0 puller start ==="
until ssh -o BatchMode=yes -o ConnectTimeout=20 runpod4 "[ -f /workspace/rom-p0/P0_DONE.marker ] || [ -f /workspace/rom-p0/P0_FAILED.marker ]" 2>/dev/null; do
  log "waiting for P0 completion"; sleep 300
done
log "P0 finished — pulling chains + verification"
rsync -az runpod4:/workspace/rom-p0/data/chains_gpt-oss-20b_p0.json data/ 2>>"$LOG" || log "chains missing"
rsync -az runpod4:/workspace/rom-p0/results/safety/ results/safety/ 2>>"$LOG" || true
rsync -az runpod4:/workspace/rom-p0/p0_full.log results/safety/ 2>>"$LOG" || true
touch results/safety/P0_READY_FOR_REVIEW.marker
log "=== P0_READY_FOR_REVIEW (stopping for Tony per consent) ==="
