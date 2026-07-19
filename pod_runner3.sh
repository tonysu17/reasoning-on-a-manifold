#!/bin/bash
# POD-SIDE R3/R4 entropy-frame run (DPO + GRPO, 4 arms) + the missing pt08 LoRA-control
# cell. Generation-dominated (~20-30h). Self-kills via deadman. Launch:
#   setsid nohup bash pod_runner3.sh > pod_runner3.log 2>&1 < /dev/null &
set -u
cd /workspace/rom-rl
export HF_HOME=/workspace/hf
GRACE_H=6; GLOBAL_H=36; START=$(date +%s)
mkdir -p results checkpoints data/activations
log(){ echo "$(date -u '+%F %T') | $*"; }
kill_pod(){ log "KILL"; source /etc/rp_environment
  curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; sleep 60; }
gc(){ [ $(( $(date +%s) - START )) -gt $(( GLOBAL_H * 3600 )) ] && { log "WATCHDOG"; kill_pod; }; }
stage(){ local n="$1"; shift
  [ -f "DONE_${n}.marker" ] && { log "stage ${n}: cached"; return 0; }
  log "stage ${n}: start"
  if "$@" >> "stage_${n}.log" 2>&1; then touch "DONE_${n}.marker"; log "stage ${n}: DONE"
  else touch "FAILED_${n}.marker"; log "stage ${n}: FAILED"; return 1; fi; }

log "=== R3/R4 runner start ==="
stage s0_pt08_lora_control python3 pt08_surprisal_entropy.py --base 1.5b \
  --post /workspace/keep_lora_control_ckpt/merged --tokenizer-alias 1.5b \
  --device auto --batch-size 1024 --out results/pt08_lora_control.json
gc
stage s1_gen_safety python3 pt09_build_dpo_pairs.py --arm safety --generate --n 500 \
  --max-new-tokens 1024 --temperature 0.8 --seed 42 --out data/dpo_safety.json
gc
stage s2_gen_control python3 pt09_build_dpo_pairs.py --arm control --generate --n 150 --k 6 \
  --max-new-tokens 2048 --temperature 0.8 --seed 42 --out data/dpo_control.json
gc
[ -f DONE_s1_gen_safety.marker ] && stage s3_dpo_safety python3 pt10_train_dpo.py \
  --data data/dpo_safety.json --beta 0.1 --seed 42 --merge --out-dir checkpoints/dpo_safety
gc
[ -f DONE_s2_gen_control.marker ] && stage s4_dpo_control python3 pt10_train_dpo.py \
  --data data/dpo_control.json --beta 0.1 --seed 42 --merge --out-dir checkpoints/dpo_control
gc
stage s5_grpo_refusal python3 pt11_train_grpo.py --reward refusal-format \
  --prompts data/grpo_refusal_prompts.json --group-size 8 --n 250 --seed 42 \
  --out-dir checkpoints/grpo_refusal
gc
[ -f checkpoints/grpo_refusal/adapter_config.json ] || true
[ -f DONE_s5_grpo_refusal.marker ] && stage s5b_merge_refusal python3 merge_adapter.py \
  checkpoints/grpo_refusal checkpoints/grpo_refusal_merged
gc
[ -f DONE_s2_gen_control.marker ] && stage s6_grpo_math python3 pt11_train_grpo.py \
  --reward math --prompts data/tasks_final.json \
  --reference-answers data/dpo_control_pseudo_references.json \
  --group-size 8 --n 150 --seed 42 --out-dir checkpoints/grpo_math
gc
[ -f DONE_s6_grpo_math.marker ] && stage s6b_merge_math python3 merge_adapter.py \
  checkpoints/grpo_math checkpoints/grpo_math_merged
gc
mdl(){ local d="$1"; [ -d "$d/merged" ] && echo "$d/merged" || echo "$d"; }
for spec in "dpo_safety R1-1.5B-dpo-safety" "dpo_control R1-1.5B-dpo-control" \
            "grpo_refusal_merged R1-1.5B-grpo-refusal" "grpo_math_merged R1-1.5B-grpo-math"; do
  set -- $spec
  [ -d "checkpoints/$1" ] || { log "skip extract/$2 (no checkpoint)"; continue; }
  stage "s7_extract_$2" python3 04_extract_activations.py --model 1.5b \
    --model-path "$(mdl checkpoints/$1)" --short-name "$2" --tokenizer-alias 1.5b --layers 12 16
  stage "s8_pt08_$2" python3 pt08_surprisal_entropy.py --base 1.5b \
    --post "$(mdl checkpoints/$1)" --tokenizer-alias 1.5b --device auto --batch-size 1024 \
    --out "results/pt08_$2.json"
  gc
done
log "stages: $(ls DONE_*.marker 2>/dev/null | wc -l) DONE, $(ls FAILED_*.marker 2>/dev/null | wc -l) FAILED"
touch ALL_DONE.marker
DEADLINE=$(( $(date +%s) + GRACE_H * 3600 ))
until [ -f SYNCED_OK.marker ]; do
  [ $(date +%s) -gt $DEADLINE ] && { log "deadman elapsed"; break; }
  gc; sleep 120
done
kill_pod
