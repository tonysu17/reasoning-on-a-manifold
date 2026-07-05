#!/bin/bash
# Run B, POD-SIDE (RUNPOD_SAFETY_PROPOSAL.md): LoRA dose-response + control,
# then activation extraction of every merged checkpoint at the pre-committed
# layers {12, 16}. Launched in tmux by the Mac-side driver. ~1.5-2.5 GPU-h.
#
# Recipe notes: published STAR-1 = FULL SFT, 5 epochs, eff. batch 128, LR 1e-5
# cosine, seq 8192, loss on completion. We match epochs/LR/eff-batch with LoRA
# (declared approximation) and cap seq at 4096 (dataset p99 << 4096).
set -u
cd /workspace/reasoning-on-manifold
echo "=== Run B start: $(date) ===" | tee runB.log

train () {  # $1 data  $2 doses  $3 outdir
  python3 -u pt02_train_safety_lora.py --data "$1" --dose "$2" --merge \
    --epochs 5 --lr 1e-5 --batch-size 4 --grad-accum 32 --max-len 4096 \
    --seed 42 --out-dir "$3" >> runB.log 2>&1
}

echo "--- train safety doses: $(date)" | tee -a runB.log
train data/safety_star1_sft.json 100,300,all checkpoints/r1_1.5b_safety \
  || { echo "ABORT: safety training failed" | tee -a runB.log; exit 1; }
echo "--- train control (all): $(date)" | tee -a runB.log
train data/control_generic_sft.json all checkpoints/r1_1.5b_control \
  || { echo "ABORT: control training failed" | tee -a runB.log; exit 1; }

extract () {  # $1 merged-dir  $2 short-name
  python3 -u 04_extract_activations.py --model 1.5b --model-path "$1" \
    --short-name "$2" --tokenizer-alias 1.5b --layers 12 16 \
    >> runB.log 2>&1
}

for spec in \
  "checkpoints/r1_1.5b_safety/dose_100/merged R1-1.5B-lora-safety100" \
  "checkpoints/r1_1.5b_safety/dose_300/merged R1-1.5B-lora-safety300" \
  "checkpoints/r1_1.5b_safety/dose_1000/merged R1-1.5B-lora-safety1000" \
  "checkpoints/r1_1.5b_control/dose_1000/merged R1-1.5B-lora-control1000" ; do
  set -- $spec
  # tolerate dose_all naming if pt02 labels the full dose differently
  d="$1"; [ -d "$d" ] || d="${1/dose_1000/dose_all}"
  [ -d "$d" ] || { echo "ABORT: merged checkpoint missing: $1" | tee -a runB.log; exit 1; }
  echo "--- extract $2 from $d: $(date)" | tee -a runB.log
  extract "$d" "$2" || { echo "ABORT: extraction failed for $2" | tee -a runB.log; exit 1; }
done

touch RUNB_DONE.marker
echo "=== Run B COMPLETE: $(date) ===" | tee -a runB.log
