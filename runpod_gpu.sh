#!/bin/bash
# GPU track: E1 (faithful Venhoff attribution) then E2 (forward-sweep cross-check).
# Runs in parallel with the CPU robustness battery (E4) — E1/E2 are GPU, battery is CPU,
# so they don't contend. Decoupled because E4's battery was slow and needn't block E1.
set -u
cd /workspace/reasoning-on-manifold
export HF_HOME=/workspace/hf
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
PY="python -u"

echo "=== [$(date)] E1: venhoff attribution n=500 ctx=1024 ==="
$PY 07e_venhoff_layer_attribution.py --behaviours all --n-examples 500 --context-window 1024 \
    > results/E1_venhoff_attribution.log 2>&1
echo "=== [$(date)] E1 DONE rc=$? ==="
grep "argmax layer" results/E1_venhoff_attribution.log 2>/dev/null
date > results/E1_DONE.marker

echo "=== [$(date)] E2: de-confounded forward-sweep (07d) ==="
$PY 07d_layer_steering_sweep.py --behaviours all --n-donors 20 \
    > results/E2_layer_sweep.log 2>&1
echo "=== [$(date)] E2 DONE rc=$? ==="
date > results/gpu_track_DONE.marker
