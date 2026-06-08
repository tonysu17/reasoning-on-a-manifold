#!/bin/bash
# Master script: run Phase 5c, 5d, 7b-pilot, triangulation, Phase 6 in sequence.
# Each step logs to logs/<phase>.log; the script logs progress to logs/master.log

set -e
cd ~/reasoning-on-manifold
mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a logs/master.log
}

PY=/home/tony/venv/bin/python3

log "=== START — running remaining Phase 5+6 phases ==="

# Phase 5c: probes at all 28 layers (CPU only)
log "Phase 5c: probe accuracy at all 28 layers..."
$PY 05c_cross_layer_probing.py --model-short R1-1.5B > logs/phase5c.log 2>&1
log "  done: $(grep -c 'OK' logs/phase5c.log || true) probes"

# Phase 5d: sub-type clustering (needs Phase 5 outputs; auto-picks d_eff peak per behaviour)
log "Phase 5d: sub-type clustering..."
$PY 05d_subtype_clustering.py --model-short R1-1.5B > logs/phase5d.log 2>&1
log "  done"

# Phase 7b-pilot: patching at all 28 layers, 5 donors each (GPU)
log "Phase 7b-pilot: patching all 28 layers, 5 donors each per behaviour..."
$PY 07b_activation_patching.py --model-short R1-1.5B --pilot > logs/phase7b_pilot.log 2>&1
log "  done"

# Compute layer triangulation
log "Compute layer triangulation..."
$PY compute_layer_triangulation.py --model-short R1-1.5B > logs/triangulation.log 2>&1
log "  done"

# Phase 6: steering vector construction at candidate layers
log "Phase 6: steering vectors..."
$PY 06_build_steering.py --model-short R1-1.5B > logs/phase6.log 2>&1
log "  done"

log "=== ALL DONE ==="
