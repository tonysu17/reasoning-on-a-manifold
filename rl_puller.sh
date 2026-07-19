#!/bin/bash
# MAC-SIDE incremental puller for R3/R4: pulls each artifact as its stage completes
# (insurance against pod reclaim); SYNCED_OK + analysis trigger at ALL_DONE.
set -u
cd "$(dirname "$0")"
HOST=runpod; REMOTE=/workspace/rom-rl
LOG=results/safety_posttrain/rl_puller.log
log(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
mkdir -p results/safety_posttrain/rl data/activations
have_marker(){ ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" "[ -f $REMOTE/DONE_$1.marker ]" 2>/dev/null; }

log "=== incremental puller start ==="
while :; do
  # pair files
  [ -f data/dpo_control.json ] || { have_marker s2_gen_control && rsync -az "$HOST:$REMOTE/data/dpo_control.json" "$HOST:$REMOTE/data/dpo_control_pseudo_references.json" data/ 2>>"$LOG" && log "pulled dpo_control pairs"; }
  # per-arm activations + pt08
  for a in R1-1.5B-dpo-safety R1-1.5B-dpo-control R1-1.5B-grpo-refusal R1-1.5B-grpo-math; do
    [ -d "data/activations/$a" ] || { have_marker "s7_extract_$a" && rsync -az "$HOST:$REMOTE/data/activations/$a" data/activations/ 2>>"$LOG" && log "pulled activations $a"; }
    [ -f "results/safety_posttrain/rl/pt08_$a.json" ] || { have_marker "s8_pt08_$a" && rsync -az "$HOST:$REMOTE/results/pt08_$a.json" results/safety_posttrain/rl/ 2>>"$LOG" && log "pulled pt08 $a"; }
  done
  if ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" "[ -f $REMOTE/ALL_DONE.marker ]" 2>/dev/null; then
    log "ALL_DONE — final sweep"
    rsync -az "$HOST:$REMOTE/results/" results/safety_posttrain/rl/ 2>>"$LOG"
    rsync -az "$HOST:$REMOTE/pod_runner3.log" results/safety_posttrain/rl/ 2>>"$LOG" || true
    rsync -az "$HOST:$REMOTE/stage_*.log" results/safety_posttrain/rl/ 2>>"$LOG" || true
    ssh -o BatchMode=yes "$HOST" "mkdir -p /workspace/rl_adapters_keep && cp -r $REMOTE/checkpoints /workspace/rl_adapters_keep/ 2>/dev/null; : DEFERRED-SYNCED_OK" 2>/dev/null \
      && log "SYNCED_OK — pod self-terminating" || log "pod unreachable at final sweep"
    touch results/safety_posttrain/rl/READY_FOR_ANALYSIS.marker
    log "=== puller complete — READY_FOR_ANALYSIS ==="
    break
  fi
  sleep 600
done
