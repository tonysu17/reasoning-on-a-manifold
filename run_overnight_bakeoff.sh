#!/bin/bash
# Overnight LAYER BAKE-OFF — L27 then L16, generation-only ($0), fully detached.
# Launched in a cluster tmux session so it survives the cloudflared tunnel / a
# closed laptop. Each layer runs under a retry loop; 07 resumes from its
# checkpoint, so a transient crash/disk-hiccup loses <=1 sweep, never the run.
# Annotation is SKIPPED (no creds, and the layer pick needs annotation anyway)
# -> ZERO dollars spent. Inspect generation_metrics.json (damage/coherence) in
# the morning; annotate later once proxy creds + a $ cap are provided.
set -u
cd /home/tony/reasoning-on-manifold
PY=/home/tony/venv/bin/python
mkdir -p results/eval
MAIN=results/eval/overnight_orchestration.log
echo "=== [$(date)] START overnight bake-off (L27 then L16, generation-only, \$0) ===" | tee -a "$MAIN"

run_layer () {
  local vdir=$1 odir=$2 log=$3
  for attempt in $(seq 1 8); do
    echo "=== [$(date)] $odir attempt $attempt ===" | tee -a "$log"
    $PY 07_evaluate_steering.py --model 1.5b --skip-annotation \
      --vectors-dir "$vdir" --out-dir "$odir" \
      --max-eval-tasks 15 --alpha-values 0.5 1.0 2.0 \
      --no-random-control --no-random-subspace --no-energy-matched --no-orthogonal-complement \
      >> "$log" 2>&1 && { echo "=== [$(date)] $odir DONE ===" | tee -a "$log"; return 0; }
    echo "=== [$(date)] $odir attempt $attempt exited non-zero; resume in 30s ===" | tee -a "$log"
    sleep 30
  done
  echo "=== [$(date)] $odir GAVE UP after 8 attempts ===" | tee -a "$log"
  return 1
}

run_layer results/steering_vectors/R1-1.5B__L27 results/eval/R1-1.5B__L27_bakeoff results/eval/bakeoff_L27.log
run_layer results/steering_vectors/R1-1.5B__L16 results/eval/R1-1.5B__L16_bakeoff results/eval/bakeoff_L16.log

echo "=== [$(date)] ALL GENERATION DONE — damage-metric summary ===" | tee -a "$MAIN"
for tag in L27 L16; do
  echo "----- $tag generation_metrics.json -----" | tee -a "$MAIN"
  $PY -c "import json,sys;d=json.load(open('results/eval/R1-1.5B__${tag}_bakeoff/generation_metrics.json'));print(json.dumps(d,indent=2))" >> "$MAIN" 2>&1 \
    || echo "(no $tag metrics yet)" | tee -a "$MAIN"
done
echo "=== [$(date)] BAKE-OFF COMPLETE ===" | tee -a "$MAIN"
date > results/eval/bakeoff_DONE.marker
