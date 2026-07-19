#!/bin/bash
# POD-SIDE RL RE-DOSE run (fixes the under-dose found by pt12):
#   - completion cap 512 -> 4096 (rollouts must be able to finish a thought)
#   - LoRA lr -> 1e-5, epochs: DPO 3, GRPO 2 (n 200/150 to fit the night)
#   - pair data REUSED from the first run (no regeneration)
# Then merge -> extract L12/16 -> pt08, per arm, -v2 names. Deadman self-kill.
# Launch: setsid nohup bash pod_runner4.sh > pod_runner4.log 2>&1 < /dev/null &
set -u
cd /workspace/rom-rl
export HF_HOME=/workspace/hf
GRACE_H=6; GLOBAL_H=30; START=$(date +%s)
mkdir -p results checkpoints data/activations
log(){ echo "$(date -u '+%F %T') | $*"; }
kill_pod(){ log "KILL"; source /etc/rp_environment
  curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; sleep 60; }
gc(){ [ $(( $(date +%s) - START )) -gt $(( GLOBAL_H * 3600 )) ] && { log "WATCHDOG"; kill_pod; }; }
stage(){ local n="$1"; shift
  [ -f "V2_DONE_${n}.marker" ] && { log "stage ${n}: cached"; return 0; }
  log "stage ${n}: start"
  if "$@" >> "v2stage_${n}.log" 2>&1; then touch "V2_DONE_${n}.marker"; log "stage ${n}: DONE"
  else touch "V2_FAILED_${n}.marker"; log "stage ${n}: FAILED"; return 1; fi; }

log "=== RL re-dose runner start ==="
stage dpo_safety_v2 python3 pt10_train_dpo.py --data data/dpo_safety.json --beta 0.1 \
  --lr 1e-5 --epochs 3 --seed 42 --merge --out-dir checkpoints/dpo_safety_v2
gc
stage dpo_control_v2 python3 pt10_train_dpo.py --data data/dpo_control.json --beta 0.1 \
  --lr 1e-5 --epochs 3 --seed 42 --merge --out-dir checkpoints/dpo_control_v2
gc
stage grpo_refusal_v2 python3 pt11_train_grpo.py --reward refusal-format \
  --prompts data/grpo_refusal_prompts.json --group-size 8 --n 200 --epochs 2 \
  --lr 1e-5 --max-completion-len 4096 --seed 42 --out-dir checkpoints/grpo_refusal_v2
gc
stage grpo_math_v2 python3 pt11_train_grpo.py --reward math --prompts data/tasks_final.json \
  --reference-answers data/dpo_control_pseudo_references.json --group-size 8 --n 150 \
  --epochs 2 --lr 1e-5 --max-completion-len 4096 --seed 42 --out-dir checkpoints/grpo_math_v2
gc
stage merge_grpo_refusal_v2 python3 merge_adapter.py checkpoints/grpo_refusal_v2/final_adapter checkpoints/grpo_refusal_v2_merged
stage merge_grpo_math_v2    python3 merge_adapter.py checkpoints/grpo_math_v2/final_adapter    checkpoints/grpo_math_v2_merged
gc
declare -A MDL
MDL[R1-1.5B-dpo-safety-v2]="checkpoints/dpo_safety_v2/ckpt_frac_1/merged"
MDL[R1-1.5B-dpo-control-v2]="checkpoints/dpo_control_v2/ckpt_frac_1/merged"
MDL[R1-1.5B-grpo-refusal-v2]="checkpoints/grpo_refusal_v2_merged"
MDL[R1-1.5B-grpo-math-v2]="checkpoints/grpo_math_v2_merged"
for arm in R1-1.5B-dpo-safety-v2 R1-1.5B-dpo-control-v2 R1-1.5B-grpo-refusal-v2 R1-1.5B-grpo-math-v2; do
  m="${MDL[$arm]}"
  [ -d "$m" ] || { log "skip $arm (no model at $m)"; continue; }
  stage "extract_$arm" python3 04_extract_activations.py --model 1.5b \
    --model-path "$m" --short-name "$arm" --tokenizer-alias 1.5b --layers 12 16
  stage "pt08_$arm" python3 pt08_surprisal_entropy.py --base 1.5b --post "$m" \
    --tokenizer-alias 1.5b --device auto --batch-size 1024 --out "results/pt08_$arm.json"
  gc
done
log "v2 stages: $(ls V2_DONE_*.marker 2>/dev/null | wc -l) DONE, $(ls V2_FAILED_*.marker 2>/dev/null | wc -l) FAILED"
touch V2_ALL_DONE.marker
DEADLINE=$(( $(date +%s) + GRACE_H * 3600 ))
until [ -f V2_SYNCED.marker ]; do
  [ $(date +%s) -gt $DEADLINE ] && { log "deadman elapsed"; break; }
  gc; sleep 120
done
kill_pod
