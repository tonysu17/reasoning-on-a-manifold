#!/bin/bash
# Fresh master pipeline re-run (R1-1.5B) on the LOCAL laptop CPU, using the
# corrected code (Fixes 2-4). Archives the old geometry results first so the new
# run is not contaminated by the previous (d_eff-saturated) outputs, then runs:
#   Phase 5 (--with-nulls) -> 5c -> triangulation -> 5d -> 5b
# Phase 7b patching is intentionally skipped: it is GPU-bound and the cluster GPU
# is occupied by the 7B chain-gen for ~11 days. Triangulation handles a missing
# patching curve (status "missing") and uses PR + probe instead.

set -u
ROOT="/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold"
cd "$ROOT" || exit 1
PY="/Users/tonysu/.pyenv/versions/3.13.12/bin/python3"
M="R1-1.5B"
STAMP="$(date '+%Y%m%d_%H%M%S')"
ARCH="results/_archive_run1_${STAMP}"
mkdir -p logs "$ARCH"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/master_rerun.log; }

log "=== FRESH RE-RUN START (model=$M, archive=$ARCH, py=$PY) ==="

# 0) Archive old geometry-pipeline outputs so the new run is uncontaminated.
for d in pca cross_layer triangulation clustering geometric; do
  if [ -d "results/$d/$M" ]; then
    mkdir -p "$ARCH/$d"
    mv "results/$d/$M" "$ARCH/$d/$M"
    log "archived results/$d/$M -> $ARCH/$d/$M"
  fi
done

# 1) Phase 5: PCA across all layers + chain-stratified nulls. cap is now 100,
#    so d_eff_70 / PR no longer saturate at 50.
log "Phase 5: PCA + per-layer nulls (all layers, cap=100)..."
"$PY" 05_pca_analysis.py --model-short "$M" --with-nulls --null-resamples 100 \
    > logs/rerun_phase5.log 2>&1
log "  Phase 5 exit=$?"

# 2) Phase 5c: linear probes at all 28 layers.
log "Phase 5c: cross-layer probes (all 28 layers)..."
"$PY" 05c_cross_layer_probing.py --model-short "$M" > logs/rerun_phase5c.log 2>&1
log "  Phase 5c exit=$?"

# 3) Triangulation: PR(argmin) + probe peaks -> candidate layers.
log "Triangulation: PR(argmin) + probe..."
"$PY" compute_layer_triangulation.py --model-short "$M" > logs/rerun_triangulation.log 2>&1
log "  triangulation exit=$?"
if [ -f "results/triangulation/$M/candidate_layers.json" ]; then
  log "  candidate_layers.json written"
else
  log "  WARN: candidate_layers.json missing"
fi

# 4) Phase 5d: sub-type clustering at the PR-trough layer per behaviour (auto).
log "Phase 5d: sub-type clustering (auto PR-trough layer)..."
"$PY" 05d_subtype_clustering.py --model-short "$M" > logs/rerun_phase5d.log 2>&1
log "  Phase 5d exit=$?"

# 5) Phase 5b: geometric deep-dive (intrinsic dim + curvature) across the
#    manifold region. Fixed profile 11/14/17/20/27 spans the expected L14-17 peak
#    plus Huang's reference layer 27. Slowest step, so it runs last.
log "Phase 5b: geometric diagnostics at layers 11 14 17 20 27 (n_resamples=300)..."
"$PY" 05b_geometric_diagnostics.py --model-short "$M" --layers 11 14 17 20 27 \
    --n-resamples 300 > logs/rerun_phase5b.log 2>&1
log "  Phase 5b exit=$?"

log "=== FRESH RE-RUN COMPLETE ==="
