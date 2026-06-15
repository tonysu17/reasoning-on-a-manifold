#!/usr/bin/env bash
# Overnight v2: the CORRECTED correctness runs (predictor trained on ALL chains,
# chain-grouped OOF; correctness AUC evaluated on the labelled subset only).
# Local CPU only — no proxy creds needed. Outputs to results/predict/R1-1.5B/corrected/.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
OUT="results/predict/R1-1.5B/corrected"

echo "[v2] === Rung-1 (ridge) vs Rung-2 (JEPA), train-on-all @ $(date) ==="
python3 17_rung2_compare.py --labels data/correctness_R1-1.5B_pilot.json \
  --layers 14,17,27 --jepa-epochs 25 --label-resamples 500 --out "$OUT"
echo "[v2] 17_ exit=$? @ $(date)"

echo "[v2] === ridge gate + step-shuffle null, train-on-all @ $(date) ==="
python3 15_predict_gate.py --labels data/correctness_R1-1.5B_pilot.json \
  --layers 14,17 --label-resamples 500 --shuffle-resamples 60 --out "$OUT"
echo "[v2] 15_ exit=$? @ $(date)"
echo "[v2] === done @ $(date) ==="
