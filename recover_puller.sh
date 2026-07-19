#!/bin/bash
set -u
cd "$(dirname "$0")"
HOST=runpod; REMOTE=/workspace/rom-rl
LOG=results/safety_posttrain/recover_puller.log
log(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
mkdir -p results/safety_posttrain/rl data/activations
log "=== recover puller start ==="
while :; do
  for a in R1-1.5B-dpo-safety R1-1.5B-dpo-control R1-1.5B-grpo-refusal R1-1.5B-grpo-math; do
    [ -d "data/activations/$a" ] || ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" "[ -f $REMOTE/R_DONE_extract_$a.marker ]" 2>/dev/null && \
      { [ -d "data/activations/$a" ] || { rsync -az "$HOST:$REMOTE/data/activations/$a" data/activations/ 2>>"$LOG" && log "pulled $a"; }; }
    [ -f "results/safety_posttrain/rl/pt08_$a.json" ] || ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" "[ -f $REMOTE/R_DONE_pt08_$a.marker ]" 2>/dev/null && \
      { [ -f "results/safety_posttrain/rl/pt08_$a.json" ] || { rsync -az "$HOST:$REMOTE/results/pt08_$a.json" results/safety_posttrain/rl/ 2>>"$LOG" && log "pulled pt08 $a"; }; }
  done
  if ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" "[ -f $REMOTE/RECOVER_DONE.marker ]" 2>/dev/null; then
    log "RECOVER_DONE — final sweep"
    rsync -az "$HOST:$REMOTE/results/" results/safety_posttrain/rl/ 2>>"$LOG"
    rsync -az "$HOST:$REMOTE/pod_recover.log" "$HOST:$REMOTE/rstage_"'*'".log" results/safety_posttrain/rl/ 2>>"$LOG" || true
    ssh -o BatchMode=yes "$HOST" "touch $REMOTE/RECOVER_SYNCED.marker" 2>/dev/null && log "RECOVER_SYNCED — pod self-terminating"
    touch results/safety_posttrain/rl/READY_FOR_ANALYSIS.marker
    log "=== recover puller complete ==="; break
  fi
  sleep 300
done
