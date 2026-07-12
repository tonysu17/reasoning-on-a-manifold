#!/bin/bash
# Seed-replication driver, MAC-SIDE (statistician review, GPU item 1): trains
# {safety1000, control1000} x seeds {43,44} on the DGX Spark and pulls the
# activations back arm-by-arm (disk on the Spark allows only one arm in flight).
# With the existing seed-42 arms this gives 3 seeds per recipe — the per-seed
# distribution behind the recipe-direction claim.
#
#   ./spark_seedrep_driver.sh          # full run (push deps, 4 arms, pull each)
#
# Output: data/activations/R1-1.5B-lora-{safety,control}1000-s{43,44}/ locally,
# log in results/safety_posttrain/seedrep_driver.log
set -u
cd "$(dirname "$0")"
HOST="${HOST:-spark-06aa}"
REMOTE="${REMOTE:-\$HOME/rom-seedrep}"
PYBIN="${PYBIN:-\$HOME/venv/bin/python}"
LOG=results/safety_posttrain/seedrep_driver.log
mkdir -p results/safety_posttrain
log(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

log "=== seed replication driver start ==="

# 0. push code + data + deps
ssh "$HOST" "mkdir -p $REMOTE/data $REMOTE/configs $REMOTE/checkpoints" || { log "FATAL: ssh push mkdir"; exit 1; }
rsync -az --no-owner --no-group --no-perms \
  pt02_train_safety_lora.py 04_extract_activations.py spark_seedrep_remote.sh "$HOST:$REMOTE/"
rsync -az --no-owner --no-group --no-perms configs/ "$HOST:$REMOTE/configs/"
rsync -az --no-owner --no-group --no-perms src/ "$HOST:$REMOTE/src/"
rsync -az --no-owner --no-group --no-perms \
  data/safety_star1_sft.json data/control_generic_sft.json \
  data/annotated_R1-1.5B.json data/chains_R1-1.5B.json "$HOST:$REMOTE/data/"
ssh "$HOST" "${PIPBIN:-\$HOME/venv/bin/pip} install --no-cache-dir -q peft 2>&1 | tail -1" | tee -a "$LOG"
ssh "$HOST" "ls $REMOTE/pt02_train_safety_lora.py $REMOTE/spark_seedrep_remote.sh $REMOTE/data/safety_star1_sft.json $REMOTE/data/annotated_R1-1.5B.json >/dev/null" \
  || { log "FATAL: push verification failed — files missing on remote"; exit 1; }
log "push verified + deps done"

run_arm(){  # $1 data-json  $2 seed  $3 short-name
  local data="$1" seed="$2" name="$3"
  # resume-aware: skip launch if this arm is already done or in flight
  st=$(ssh "$HOST" "cd $REMOTE 2>/dev/null && ls DONE_${name}.marker 2>/dev/null; pgrep -f '[s]park_seedrep_remote.sh .*${name}' >/dev/null && echo RUNNING" 2>&1)
  case "$st" in
    *DONE_${name}.marker*) log "${name} already DONE on pod — pulling" ;;
    *RUNNING*) log "${name} already in flight — polling" ;;
    *) log "--- launching ${name} (seed ${seed})"
       ssh "$HOST" "cd $REMOTE && PYBIN=$PYBIN nohup bash spark_seedrep_remote.sh data/$(basename "$data") $seed $name >/dev/null 2>&1 & echo launched" ;;
  esac
  # poll: 5 min cadence, 8 h ceiling per arm (GB10 measured ~4.7 h train + extract)
  for i in $(seq 1 96); do
    st=$(ssh -o ConnectTimeout=20 "$HOST" "cd $REMOTE 2>/dev/null && ls DONE_${name}.marker FAILED_${name}.marker 2>/dev/null; tail -1 arm_${name}.log 2>/dev/null | tr '\r' '\n' | tail -1" 2>&1)
    echo "$(date '+%F %T') | poll ${name}: ${st}" >> "$LOG"
    case "$st" in
      *DONE_${name}.marker*) log "${name} DONE on pod"; break ;;
      *FAILED_${name}.marker*) log "ABORT: ${name} FAILED on pod — see arm_${name}.log remotely"; return 1 ;;
    esac
    [ "$i" = 96 ] && { log "ABORT: ${name} timed out after 8 h"; return 1; }
    sleep 300
  done
  mkdir -p data/activations
  rsync -az "$HOST:$REMOTE/data/activations/${name}" data/activations/ || { log "ABORT: pull failed for ${name}"; return 1; }
  ssh "$HOST" "rm -rf $REMOTE/data/activations/${name}"
  log "${name} pulled ($(du -sh "data/activations/${name}" | cut -f1)) and remote-cleaned"
}

run_arm data/safety_star1_sft.json  43 R1-1.5B-lora-safety1000-s43  || exit 1
run_arm data/control_generic_sft.json 43 R1-1.5B-lora-control1000-s43 || exit 1
run_arm data/safety_star1_sft.json  44 R1-1.5B-lora-safety1000-s44  || exit 1
run_arm data/control_generic_sft.json 44 R1-1.5B-lora-control1000-s44 || exit 1

log "=== ALL 4 ARM-SEEDS COMPLETE — next: pt04 rerun over seeds (per-seed direction distribution) ==="
