#!/usr/bin/env bash
# Overnight: judge pilot correctness labels, then run the Rung-0/Rung-1 gate.
# Proxy creds are read from the environment (CLAUDE_PROXY_URL / CLAUDE_PROXY_KEY);
# this script does NOT contain the key. Launched in the background by Claude.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

echo "[overnight] === judge pilot (200) @ $(date) ==="
python3 14_label_correctness.py --pilot 200 --out data/correctness_R1-1.5B_pilot.json
JUDGE_RC=$?
echo "[overnight] judge exit=$JUDGE_RC @ $(date)"

if [ -s data/correctness_R1-1.5B_pilot.json ]; then
  echo "[overnight] === gate (layers 11,14,17,20,27) @ $(date) ==="
  python3 15_predict_gate.py \
    --labels data/correctness_R1-1.5B_pilot.json \
    --layers 11,14,17,20,27 \
    --label-resamples 500 --shuffle-resamples 100
  echo "[overnight] gate exit=$? @ $(date)"
else
  echo "[overnight] ERROR: no labels file produced; skipping gate"
fi
echo "[overnight] === predict pipeline done @ $(date) ==="
