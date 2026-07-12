#!/bin/bash
# Seed-replication, SPARK-SIDE: one arm-seed = train LoRA (dose all) -> merge ->
# extract activations at layers {12,16} -> free disk. Launched via nohup by the
# Mac-side driver (spark_seedrep_driver.sh). Recipe matches runpod_runB_remote.sh
# exactly except --seed. Disk-constrained host: checkpoints are deleted the moment
# extraction finishes.
#
# usage: bash spark_seedrep_remote.sh <data.json> <seed> <short-name>
set -u
cd "$(dirname "$0")"
PY="${PYBIN:-$HOME/venv/bin/python}"
DATA="$1"; SEED="$2"; NAME="$3"
CKPT="checkpoints/seedrep_${NAME}"
LOG="arm_${NAME}.log"

echo "=== ${NAME} start: $(date) ===" | tee -a "$LOG"
rm -f "DONE_${NAME}.marker" "FAILED_${NAME}.marker"

FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt 5 ]; then
  echo "ABORT: only ${FREE_GB}G free" | tee -a "$LOG"; touch "FAILED_${NAME}.marker"; exit 1
fi

$PY -u pt02_train_safety_lora.py --data "$DATA" --dose all --merge \
    --epochs 5 --lr 1e-5 --batch-size 4 --grad-accum 32 --max-len 4096 \
    --seed "$SEED" --out-dir "$CKPT" >> "$LOG" 2>&1 \
  || { echo "ABORT: training failed" | tee -a "$LOG"; touch "FAILED_${NAME}.marker"; exit 1; }

MERGED="$CKPT/dose_all/merged"; [ -d "$MERGED" ] || MERGED="$CKPT/dose_1000/merged"
[ -d "$MERGED" ] || { echo "ABORT: merged checkpoint missing under $CKPT" | tee -a "$LOG"; touch "FAILED_${NAME}.marker"; exit 1; }

$PY -u 04_extract_activations.py --model 1.5b --model-path "$MERGED" \
    --short-name "$NAME" --tokenizer-alias 1.5b --layers 12 16 >> "$LOG" 2>&1 \
  || { echo "ABORT: extraction failed" | tee -a "$LOG"; touch "FAILED_${NAME}.marker"; exit 1; }

rm -rf "$CKPT"
echo "=== ${NAME} COMPLETE: $(date) ===" | tee -a "$LOG"
touch "DONE_${NAME}.marker"
