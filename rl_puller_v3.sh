#!/bin/bash
set -u
cd "$(dirname "$0")"
LOG=results/safety_posttrain/rl_puller_v3.log
log(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
mkdir -p results/safety_posttrain/rl data/activations
log "=== v2 puller start ==="
until ssh -o BatchMode=yes -o ConnectTimeout=20 runpod3 "[ -f /workspace/rom-rl/V3_ALL_DONE.marker ]" 2>/dev/null; do
  st=$(ssh -o BatchMode=yes -o ConnectTimeout=20 runpod3 "cd /workspace/rom-rl 2>/dev/null && ls V3_DONE_*.marker V3_FAILED_*.marker 2>/dev/null | tr '\n' ' '" 2>/dev/null)
  log "waiting (${st:-none})"; sleep 900
done
log "V3_ALL_DONE — pulling"
rsync -az "runpod3:/workspace/rom-rl/results/pt08_*-v3.json" results/safety_posttrain/rl/ 2>>"$LOG"
rsync -az "runpod3:/workspace/rom-rl/pod_runner5.log" "runpod3:/workspace/rom-rl/v2stage_*.log" results/safety_posttrain/rl/ 2>>"$LOG" || true
for a in R1-1.5B-grpo-refusal-v3 R1-1.5B-grpo-math-v3 R1-1.5B-dpo-control-v3; do
  rsync -az "runpod3:/workspace/rom-rl/data/activations/$a" data/activations/ 2>>"$LOG" && log "pulled $a" || log "$a missing"
done
ssh -o BatchMode=yes runpod3 "touch /workspace/rom-rl/V3_SYNCED.marker" 2>/dev/null && log "V3_SYNCED — pod self-terminating"
touch results/safety_posttrain/rl/V3_READY_FOR_ANALYSIS.marker
log "=== v2 puller complete ==="
