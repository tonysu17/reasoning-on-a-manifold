#!/bin/bash
set -u
cd "$(dirname "$0")"
LOG=results/safety_posttrain/rl_puller_v2.log
log(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
mkdir -p results/safety_posttrain/rl data/activations
log "=== v2 puller start ==="
until ssh -o BatchMode=yes -o ConnectTimeout=20 runpod "[ -f /workspace/rom-rl/V2_ALL_DONE.marker ]" 2>/dev/null; do
  st=$(ssh -o BatchMode=yes -o ConnectTimeout=20 runpod "cd /workspace/rom-rl 2>/dev/null && ls V2_DONE_*.marker V2_FAILED_*.marker 2>/dev/null | tr '\n' ' '" 2>/dev/null)
  log "waiting (${st:-none})"; sleep 900
done
log "V2_ALL_DONE — pulling"
rsync -az "runpod:/workspace/rom-rl/results/pt08_*-v2.json" results/safety_posttrain/rl/ 2>>"$LOG"
rsync -az "runpod:/workspace/rom-rl/pod_runner4.log" "runpod:/workspace/rom-rl/v2stage_*.log" results/safety_posttrain/rl/ 2>>"$LOG" || true
for a in R1-1.5B-dpo-safety-v2 R1-1.5B-dpo-control-v2 R1-1.5B-grpo-refusal-v2 R1-1.5B-grpo-math-v2; do
  rsync -az "runpod:/workspace/rom-rl/data/activations/$a" data/activations/ 2>>"$LOG" && log "pulled $a" || log "$a missing"
done
ssh -o BatchMode=yes runpod "touch /workspace/rom-rl/V2_SYNCED.marker" 2>/dev/null && log "V2_SYNCED — pod self-terminating"
touch results/safety_posttrain/rl/V2_READY_FOR_ANALYSIS.marker
log "=== v2 puller complete ==="
