#!/bin/bash
# Phase-2 pod session job (pod side) — SESSION_HANDOFF_2026-08-09 §4 queue.
# Launched by runpod_ph2.sh in tmux 'ph2'. Stages (each resumable; a re-run
# skips completed work via the executor's markers / existing artifacts):
#
#   j1  F5 fallback  : DPO control on the 65 local pairs (route FALLBACK_65PAIR,
#                      ledger §F5) in /workspace/venv-r1 -> merge -> extract
#                      L12/16 -> pt08 surprisal/entropy vs base
#   j2  ph2 discovery: pair-state extraction, all 3 sealed checkpoints
#   j3  ph2 gates    : 20 sham + 20 rand-orth per target (WALL-CLOCK DRIVER,
#                      ~10-17 GPU-h) + base sham0
#   j4  ph2 refit    : widths {1,2} per target + alignment
#
# No generation/annotation here — those stages spend and refuse without
# --authorised (launched separately after Tony's sign-offs).
# House rules: status file PH2_STATUS, done marker PH2_DONE.marker; NO on-pod
# self-kill; the Mac watcher pulls on DONE/FAILED; terminate from console only
# after a verified pull.
set -uo pipefail
cd "$(dirname "$0")"
LOG=ph2.log
exec >>"$LOG" 2>&1

# Persistent-volume HF cache (fresh containers otherwise re-download ~10 GB of
# checkpoints to container disk every launch) + fragmentation-tolerant alloc.
export HF_HOME=/workspace/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VENV_R1=/workspace/venv-r1/bin/python   # persistent TRL env (torch 2.11 + trl 1.8)
F5_OUT=results/safety_posttrain/rl/dpo_control_f5
F5_MERGED=$F5_OUT/merged

status() { echo "$1" > PH2_STATUS; echo "[ph2 $(date -u +%H:%M:%S)] status=$1"; }
die()    { status "FAILED:$1"; touch PH2_DONE.marker; exit 1; }

status "RUNNING:j0-sanity"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" || die j0-cuda
python -c "import transformers as t; v=t.__version__; assert v.startswith('4.'), f'transformers {v} != 4.x pin'" || die j0-pin
python -c "import sklearn" || pip install -q scikit-learn || die j0-sklearn
USED_G=$(du -s /workspace 2>/dev/null | awk '{print int($1/1048576)}')
[ "${USED_G:-0}" -lt 70 ] || die j0-disk70G
[ -f results/prereg/phase2_task_manifest.json ] || die j0-manifest
[ -f results/das/R1-1.5B/main/pairs.json ] || die j0-pairs
[ -f results/das/R1-1.5B/width/frame_k2.npy ] || die j0-frame

# ── j1: F5 fallback (skipped cleanly if already merged+extracted) ────────────
if [ ! -d "data/activations/R1-1.5B-dpo-control-f5" ]; then
  status "RUNNING:j1-f5"
  [ -x "$VENV_R1" ] || die j1-venv-r1-missing
  [ -f data/dpo_control.json ] || die j1-pairs-json
  if [ ! -d "$F5_MERGED" ]; then
    "$VENV_R1" pt10_train_dpo.py --data data/dpo_control.json --epochs 23 \
      --merge --out-dir "$F5_OUT" --seed 42 || die j1-train
  fi
  python 04_extract_activations.py --model-path "$F5_MERGED" \
    --short-name R1-1.5B-dpo-control-f5 --tokenizer-alias 1.5b \
    --layers 12 16 || die j1-extract
  python pt08_surprisal_entropy.py --base 1.5b --post "$F5_MERGED" \
    --tokenizer-alias 1.5b \
    --out results/safety_posttrain/rl/pt08_R1-1.5B-dpo-control-f5.json \
    || die j1-pt08
else
  echo "[ph2] j1-f5 already extracted — skip"
fi

# ── j2-j4: Phase-2 pod stages (executor markers make each resumable) ─────────
status "RUNNING:j2-discovery"
python ph2_executor.py --stage discovery_extract || die j2-discovery
status "RUNNING:j3-gates"
python ph2_executor.py --stage target_gates || die j3-gates
status "RUNNING:j4-refit"
python ph2_executor.py --stage refit || die j4-refit

status "DONE"
touch PH2_DONE.marker
echo "[ph2] all stages complete — Mac watcher will pull; do not self-kill"
