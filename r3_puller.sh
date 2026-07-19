#!/bin/bash
set -u
cd "$(dirname "$0")"
LOG=results/r3_strategy/r3_puller.log
log(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
mkdir -p results/r3_strategy
log "=== r3 puller start ==="
until ssh -o BatchMode=yes -o ConnectTimeout=20 runpod2 "[ -f /workspace/rom-r3/R3_ALL_DONE.marker ]" 2>/dev/null; do
  st=$(ssh -o BatchMode=yes -o ConnectTimeout=20 runpod2 "grep -cE 'DONE' /workspace/rom-r3/pod_r3_full.log 2>/dev/null" 2>/dev/null)
  log "waiting (log DONE-lines: ${st:-unreachable})"; sleep 900
done
log "R3_ALL_DONE — pulling"
rsync -az runpod2:/workspace/rom-r3/results/r3_strategy/ results/r3_strategy/ 2>>"$LOG"
rsync -az runpod2:/workspace/rom-r3/pod_r3_full.log "runpod2:/workspace/rom-r3/r3stage_*.log" results/r3_strategy/ 2>>"$LOG" || true
ssh -o BatchMode=yes runpod2 "touch /workspace/rom-r3/R3_SYNCED.marker" 2>/dev/null && log "R3_SYNCED — pod continues to P0 or self-terminates"
touch results/r3_strategy/R3_FULL_READY.marker
log "=== r3 puller complete ==="
