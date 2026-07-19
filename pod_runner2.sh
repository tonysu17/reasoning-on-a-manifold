#!/bin/bash
# POD-SIDE consolidated run (2026-07-12, RTX 4090): four deliverables, then self-kill.
#   S1 off-policy control data (pt01c, MetaMathQA)         → data/control_offpolicy_sft.json
#   S2 surprisal control + entropy battery, base vs STAR1  → results/pt08_star1.json
#   S3 full-FT safety arm (STAR-1 recipe, seed 42) + extraction @ L12/16
#   S4 full-FT off-policy-control arm + extraction @ L12/16
#   S5 LoRA re-trains (safety1000/control1000, seed 42, merged) — pt08 inputs only
#   S6 pt08 for the four owned arms vs base (KL dose meter, LoRA-vs-full-FT)
# Persistent volume holds everything; deadman kills the pod with or without the Mac.
# Launch: setsid nohup bash pod_runner2.sh > pod_runner2.log 2>&1 < /dev/null &
set -u
cd /workspace/rom-consol
export HF_HOME=/workspace/hf
GRACE_H=6
GLOBAL_H=20
START=$(date +%s)
mkdir -p results checkpoints
log(){ echo "$(date -u '+%F %T') | $*"; }

kill_pod(){
  log "KILL: terminating pod"
  source /etc/rp_environment
  curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"
  sleep 60; log "KILL: still alive 60s after terminate (unexpected)"
}
global_check(){
  [ $(( $(date +%s) - START )) -gt $(( GLOBAL_H * 3600 )) ] && { log "WATCHDOG ${GLOBAL_H}h"; kill_pod; }
}
stage(){  # $1 name, rest = command; marks DONE/FAILED, never aborts the runner
  local name="$1"; shift
  if [ -f "DONE_${name}.marker" ]; then log "stage ${name}: already done"; return 0; fi
  log "stage ${name}: start"
  if "$@" >> "stage_${name}.log" 2>&1; then
    touch "DONE_${name}.marker"; log "stage ${name}: DONE"
  else
    touch "FAILED_${name}.marker"; log "stage ${name}: FAILED (see stage_${name}.log)"; return 1
  fi
}

log "=== consolidated runner start ($(grep -oE 'RUNPOD_POD_ID=\S+' /etc/rp_environment)) ==="

stage s1_offpolicy_data python3 pt01c_build_offpolicy_control.py --n 1000 --seed 42 \
  --out data/control_offpolicy_sft.json
global_check

stage s2_pt08_star1 python3 pt08_surprisal_entropy.py --base 1.5b --post star1-1.5b \
  --tokenizer-alias 1.5b --device auto --batch-size 1024 --out results/pt08_star1.json
global_check

stage s3a_fullft_safety python3 pt02c_train_full_ft.py --data data/safety_star1_sft.json \
  --seed 42 --out-dir checkpoints/fullft_safety_s42
global_check
[ -f DONE_s3a_fullft_safety.marker ] && stage s3b_extract_fullft_safety python3 04_extract_activations.py \
  --model 1.5b --model-path checkpoints/fullft_safety_s42 \
  --short-name R1-1.5B-fullft-safety-s42 --tokenizer-alias 1.5b --layers 12 16
global_check

if [ -f DONE_s1_offpolicy_data.marker ]; then
  stage s4a_fullft_control python3 pt02c_train_full_ft.py --data data/control_offpolicy_sft.json \
    --seed 42 --out-dir checkpoints/fullft_control_s42
  global_check
  [ -f DONE_s4a_fullft_control.marker ] && stage s4b_extract_fullft_control python3 04_extract_activations.py \
    --model 1.5b --model-path checkpoints/fullft_control_s42 \
    --short-name R1-1.5B-fullft-control-s42 --tokenizer-alias 1.5b --layers 12 16
fi
global_check

stage s5a_lora_safety python3 pt02_train_safety_lora.py --data data/safety_star1_sft.json \
  --dose all --merge --epochs 5 --lr 1e-5 --batch-size 4 --grad-accum 32 --max-len 4096 \
  --seed 42 --out-dir checkpoints/lora_safety_s42
stage s5b_lora_control python3 pt02_train_safety_lora.py --data data/control_generic_sft.json \
  --dose all --merge --epochs 5 --lr 1e-5 --batch-size 4 --grad-accum 32 --max-len 4096 \
  --seed 42 --out-dir checkpoints/lora_control_s42
global_check

pt08_arm(){  # $1 ckpt-dir  $2 out-name
  local d="$1"; [ -d "$d/dose_all/merged" ] && d="$d/dose_all/merged"
  [ -d "$d/dose_1000/merged" ] && d="$1/dose_1000/merged"
  python3 pt08_surprisal_entropy.py --base 1.5b --post "$d" --tokenizer-alias 1.5b \
    --device auto --batch-size 1024 --out "results/pt08_$2.json"
}
[ -f DONE_s3a_fullft_safety.marker ]  && stage s6a_pt08_fullft_safety  pt08_arm checkpoints/fullft_safety_s42  fullft_safety
[ -f DONE_s4a_fullft_control.marker ] && stage s6b_pt08_fullft_control pt08_arm checkpoints/fullft_control_s42 fullft_control
[ -f DONE_s5a_lora_safety.marker ]    && stage s6c_pt08_lora_safety    pt08_arm checkpoints/lora_safety_s42    lora_safety
[ -f DONE_s5b_lora_control.marker ]   && stage s6d_pt08_lora_control   pt08_arm checkpoints/lora_control_s42   lora_control

rm -rf checkpoints/lora_safety_s42 checkpoints/lora_control_s42
log "stages done: $(ls DONE_*.marker 2>/dev/null | wc -l) DONE, $(ls FAILED_*.marker 2>/dev/null | wc -l) FAILED"
touch ALL_DONE.marker

log "deadman: up to ${GRACE_H}h for SYNCED_OK.marker"
DEADLINE=$(( $(date +%s) + GRACE_H * 3600 ))
until [ -f SYNCED_OK.marker ]; do
  [ $(date +%s) -gt $DEADLINE ] && { log "deadman: grace elapsed (results persist on volume)"; break; }
  global_check; sleep 120
done
kill_pod
