#!/bin/bash
# Auto-pipeline (RunPod): waits for E0 (faithful 6-label clipped extraction),
# then runs the high-fidelity program on the clean activations:
#   E5  build steering vectors at Venhoff layers 17/18/15/18   (CPU, fast)
#   E3+E4  Huang pooled-PCA manifold + robustness battery       (CPU, fast)
#   E1  faithful Venhoff attribution  n=500, ctx=1024           (GPU, ~overnight)
#   E2  de-confounded forward-sweep (07d) cross-check           (GPU, ~1-2h)
# Quick CPU steps run first so manifold/vector results land in minutes; the long
# GPU attribution runs after. Each step logs separately; failures don't abort the rest.
set -u
cd /workspace/reasoning-on-manifold
export HF_HOME=/workspace/hf
# 48-core pod: cap BLAS threads or full-SVD PCA thrashes (>90s/SVD -> ~2s/SVD).
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
PY="python -u"            # unbuffered: logs flush live (avoids the buffered-table issue)
mkdir -p results
LOG=results/pipeline.log

echo "=== [$(date)] pipeline armed — waiting for E0 (extract_DONE.marker) ===" | tee -a "$LOG"
while [ ! -f results/extract_DONE.marker ]; do sleep 60; done
echo "=== [$(date)] E0 complete. activation label files: ===" | tee -a "$LOG"
ls data/activations/R1-1.5B/*_layer17.npy 2>/dev/null | sed 's#.*/##' | tee -a "$LOG"

step () {  # step NAME LOGFILE CMD...
  local name="$1" log="$2"; shift 2
  echo "=== [$(date)] $name: START ===" | tee -a "$LOG"
  "$@" > "results/$log" 2>&1
  echo "=== [$(date)] $name: DONE rc=$? (-> results/$log) ===" | tee -a "$LOG"
}

# --- quick CPU steps first ---
step "E5 build-venhoff-vectors" E5_vectors.log         $PY build_venhoff_vectors.py
step "E3E4 huang-manifold+battery" E3E4_huang.log      $PY build_huang_manifold.py
grep -hE "cos_pool|E_POOL|PASS|FAIL|argmax" results/E3E4_huang.log 2>/dev/null | tail -20 | tee -a "$LOG"

# --- long GPU steps ---
step "E1 venhoff-attribution n500 ctx1024" E1_venhoff_attribution.log \
     $PY 07e_venhoff_layer_attribution.py --behaviours all --n-examples 500 --context-window 1024
echo "--- E1 argmax vs published 17/18/15/18 ---" | tee -a "$LOG"
grep "argmax layer" results/E1_venhoff_attribution.log 2>/dev/null | tee -a "$LOG"

step "E2 deconfounded-sweep 07d" E2_layer_sweep.log \
     $PY 07d_layer_steering_sweep.py --behaviours all --n-donors 20

echo "=== [$(date)] PIPELINE COMPLETE ===" | tee -a "$LOG"
date > results/pipeline_DONE.marker
