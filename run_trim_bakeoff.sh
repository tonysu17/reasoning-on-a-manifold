#!/bin/bash
# TRIMMED layer bake-off for the contended cluster — minimal config to pick L16
# vs L27 fast: single + manifold_auto (2 arms), alpha=1.0 only, 10 hold-out
# tasks, token cap 4096 (defensible for layer-selection; NOT the final grid).
# Generation-only ($0), L27 then L16, retry+resume, detached in tmux.
set -u
cd /home/tony/reasoning-on-manifold
PY=/home/tony/venv/bin/python
mkdir -p results/eval
MAIN=results/eval/trim_orchestration.log
echo "=== [$(date)] START TRIM bake-off: single+manifold_auto, a=1.0, 10 tasks, cap4096, L27 then L16 (\$0) ===" | tee -a "$MAIN"

# --- one-time setup (idempotent): auto-only vector copies + trim to k=auto ---
for L in L16 L27; do
  if [ ! -f results/steering_vectors/R1-1.5B__${L}_auto/metadata.json ]; then
    rm -rf results/steering_vectors/R1-1.5B__${L}_auto
    cp -r results/steering_vectors/R1-1.5B__${L} results/steering_vectors/R1-1.5B__${L}_auto
  fi
done
echo "=== [$(date)] trimming vectors to single+manifold_auto ===" | tee -a "$MAIN"
$PY trim_vectors.py 2>&1 | tee -a "$MAIN"

run_layer () {
  local vdir=$1 odir=$2 log=$3
  for attempt in $(seq 1 8); do
    echo "=== [$(date)] $odir attempt $attempt ===" | tee -a "$log"
    $PY 07_evaluate_steering.py --model 1.5b --skip-annotation \
      --vectors-dir "$vdir" --out-dir "$odir" \
      --max-eval-tasks 10 --alpha-values 1.0 --max-new-tokens 4096 \
      --no-random-control --no-random-subspace --no-energy-matched --no-orthogonal-complement \
      >> "$log" 2>&1 && { echo "=== [$(date)] $odir DONE ===" | tee -a "$log"; return 0; }
    echo "=== [$(date)] $odir attempt $attempt exited non-zero; resume in 30s ===" | tee -a "$log"
    sleep 30
  done
  echo "=== [$(date)] $odir GAVE UP after 8 attempts ===" | tee -a "$log"; return 1
}

run_layer results/steering_vectors/R1-1.5B__L27_auto results/eval/R1-1.5B__L27_trim results/eval/trim_L27.log
run_layer results/steering_vectors/R1-1.5B__L16_auto results/eval/R1-1.5B__L16_trim results/eval/trim_L16.log

echo "=== [$(date)] ALL GENERATION DONE — damage-metric summary ===" | tee -a "$MAIN"
for tag in L27 L16; do
  echo "----- $tag generation_metrics.json -----" | tee -a "$MAIN"
  $PY -c "import json;print(json.dumps(json.load(open('results/eval/R1-1.5B__${tag}_trim/generation_metrics.json')),indent=2))" >> "$MAIN" 2>&1 \
    || echo "(no $tag metrics yet)" | tee -a "$MAIN"
done
echo "=== [$(date)] TRIM BAKE-OFF COMPLETE ===" | tee -a "$MAIN"
date > results/eval/trim_bakeoff_DONE.marker
