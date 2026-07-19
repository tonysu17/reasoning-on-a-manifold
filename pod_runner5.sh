#!/bin/bash
# GRPO retry (group 4, cap 4096) + DPO-control re-dose. v3 markers. Deadman kill.
set -u
cd /workspace/rom-rl
export HF_HOME=/workspace/hf
GRACE_H=6; GLOBAL_H=24; START=$(date +%s)
log(){ echo "$(date -u '+%F %T') | $*"; }
kill_pod(){ log KILL; source /etc/rp_environment
  curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; sleep 60; }
gc(){ [ $(( $(date +%s) - START )) -gt $(( GLOBAL_H * 3600 )) ] && { log WATCHDOG; kill_pod; }; }
stage(){ local n="$1"; shift
  [ -f "V3_DONE_${n}.marker" ] && { log "stage ${n}: cached"; return 0; }
  log "stage ${n}: start"
  if "$@" >> "v3stage_${n}.log" 2>&1; then touch "V3_DONE_${n}.marker"; log "stage ${n}: DONE"
  else touch "V3_FAILED_${n}.marker"; log "stage ${n}: FAILED"; return 1; fi; }
log "=== v3 retry runner start ==="
stage grpo_refusal_v3 python3 pt11_train_grpo.py --reward refusal-format \
  --prompts data/grpo_refusal_prompts.json --group-size 4 --micro-batch 2 --grad-accum 2 --n 200 --epochs 2 \
  --lr 1e-5 --max-completion-len 4096 --seed 42 --out-dir checkpoints/grpo_refusal_v3
gc
stage grpo_math_v3 python3 pt11_train_grpo.py --reward math --prompts data/tasks_final.json \
  --reference-answers data/dpo_control_pseudo_references.json --group-size 4 --micro-batch 2 --grad-accum 2 --n 150 \
  --epochs 2 --lr 1e-5 --max-completion-len 4096 --seed 42 --out-dir checkpoints/grpo_math_v3
gc
stage dpo_control_v3 python3 pt10_train_dpo.py --data data/dpo_control.json --beta 0.1 \
  --lr 1e-5 --epochs 6 --seed 42 --merge --out-dir checkpoints/dpo_control_v3
gc
stage merge_refusal_v3 python3 merge_adapter.py checkpoints/grpo_refusal_v3/final_adapter checkpoints/grpo_refusal_v3_merged
stage merge_math_v3    python3 merge_adapter.py checkpoints/grpo_math_v3/final_adapter    checkpoints/grpo_math_v3_merged
gc
declare -A MDL
MDL[R1-1.5B-grpo-refusal-v3]="checkpoints/grpo_refusal_v3_merged"
MDL[R1-1.5B-grpo-math-v3]="checkpoints/grpo_math_v3_merged"
MDL[R1-1.5B-dpo-control-v3]="checkpoints/dpo_control_v3/ckpt_frac_1/merged"
for arm in R1-1.5B-grpo-refusal-v3 R1-1.5B-grpo-math-v3 R1-1.5B-dpo-control-v3; do
  m="${MDL[$arm]}"; [ -d "$m" ] || { log "skip $arm"; continue; }
  stage "extract_$arm" python3 04_extract_activations.py --model 1.5b \
    --model-path "$m" --short-name "$arm" --tokenizer-alias 1.5b --layers 12 16
  stage "pt08_$arm" python3 pt08_surprisal_entropy.py --base 1.5b --post "$m" \
    --tokenizer-alias 1.5b --device auto --batch-size 1024 --out "results/pt08_$arm.json"
  gc
done
log "v3: $(ls V3_DONE_*.marker 2>/dev/null | wc -l) DONE, $(ls V3_FAILED_*.marker 2>/dev/null | wc -l) FAILED"
touch V3_ALL_DONE.marker
DEADLINE=$(( $(date +%s) + GRACE_H * 3600 ))
until [ -f V3_SYNCED.marker ]; do
  [ $(date +%s) -gt $DEADLINE ] && { log "deadman elapsed"; break; }; gc; sleep 120
done
kill_pod
