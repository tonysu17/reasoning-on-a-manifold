#!/bin/bash
# POD-SIDE recovery of the R3/R4 final stages (merge GRPO adapters, extract all 4
# arms @ L12/16, pt08 per arm). DPO arms already have merged/; GRPO need merging.
# Trainings + s0 cell survived on the volume — this is the cheap tail only.
# Launch: setsid nohup bash pod_recover.sh > pod_recover.log 2>&1 < /dev/null &
set -u
cd /workspace/rom-rl
export HF_HOME=/workspace/hf
KEEP=/workspace/rl_adapters_keep/checkpoints
GRACE_H=4; GLOBAL_H=8; START=$(date +%s)
mkdir -p results checkpoints data/activations
log(){ echo "$(date -u '+%F %T') | $*"; }
kill_pod(){ log "KILL"; source /etc/rp_environment
  curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; sleep 60; }
gc(){ [ $(( $(date +%s) - START )) -gt $(( GLOBAL_H * 3600 )) ] && { log "WATCHDOG"; kill_pod; }; }
stage(){ local n="$1"; shift
  [ -f "R_DONE_${n}.marker" ] && { log "stage ${n}: cached"; return 0; }
  log "stage ${n}: start"
  if "$@" >> "rstage_${n}.log" 2>&1; then touch "R_DONE_${n}.marker"; log "stage ${n}: DONE"
  else touch "R_FAILED_${n}.marker"; log "stage ${n}: FAILED"; return 1; fi; }

log "=== recovery start ($(grep -oE 'RUNPOD_POD_ID=\S+' /etc/rp_environment)) ==="

# 1. merge the two GRPO adapters (DPO already merged)
stage merge_grpo_refusal python3 merge_adapter.py "$KEEP/grpo_refusal/final_adapter" checkpoints/grpo_refusal_merged
stage merge_grpo_math    python3 merge_adapter.py "$KEEP/grpo_math/final_adapter"    checkpoints/grpo_math_merged
gc

# 2. resolve each arm's merged-model path
declare -A MDL
MDL[R1-1.5B-dpo-safety]="$KEEP/dpo_safety/ckpt_frac_1/merged"
MDL[R1-1.5B-dpo-control]="$KEEP/dpo_control/ckpt_frac_1/merged"
MDL[R1-1.5B-grpo-refusal]="checkpoints/grpo_refusal_merged"
MDL[R1-1.5B-grpo-math]="checkpoints/grpo_math_merged"

# 3. extract activations + pt08 per arm
for arm in R1-1.5B-dpo-safety R1-1.5B-dpo-control R1-1.5B-grpo-refusal R1-1.5B-grpo-math; do
  m="${MDL[$arm]}"
  [ -d "$m" ] || { log "skip $arm (no merged model at $m)"; continue; }
  stage "extract_$arm" python3 04_extract_activations.py --model 1.5b \
    --model-path "$m" --short-name "$arm" --tokenizer-alias 1.5b --layers 12 16
  stage "pt08_$arm" python3 pt08_surprisal_entropy.py --base 1.5b --post "$m" \
    --tokenizer-alias 1.5b --device auto --batch-size 1024 --out "results/pt08_$arm.json"
  gc
done

log "recovery stages: $(ls R_DONE_*.marker 2>/dev/null | wc -l) DONE, $(ls R_FAILED_*.marker 2>/dev/null | wc -l) FAILED"
touch RECOVER_DONE.marker

DEADLINE=$(( $(date +%s) + GRACE_H * 3600 ))
until [ -f RECOVER_SYNCED.marker ]; do
  [ $(date +%s) -gt $DEADLINE ] && { log "deadman elapsed"; break; }
  gc; sleep 120
done
kill_pod
