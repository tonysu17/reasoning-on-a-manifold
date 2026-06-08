#!/bin/bash
# Autonomous multi-annotator pipeline (cluster). Runs Phase 4 extraction + all
# downstream geometry phases for Qwen3-235B and Nova-Pro annotations, then the
# 3-way cross-annotator agreement + manifold-replication comparison.
# Resumable (skip-if-done), disk-guarded, detached-friendly. Shares the GPU with 7B.
set -u
cd ~/reasoning-on-manifold || exit 1
PY=/home/tony/venv/bin/python3
mkdir -p logs
M=logs/multiannotator.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$M"; }
free_gb(){ echo $(( $(df -P . | awk 'NR==2{print $4}') / 1024 / 1024 )); }

log "=== MULTI-ANNOTATOR PIPELINE START (free $(free_gb)GB) ==="
TAGS="qwen3-235b nova-pro"

for tag in $TAGS; do
  SHORT="R1-1.5B__${tag}"
  ANNOT="data/annotated_${SHORT}.json"
  ACTDIR="data/activations/${SHORT}"
  log "----- annotator: ${tag} (${SHORT}) -----"
  if [ ! -f "$ANNOT" ]; then log "  MISSING ${ANNOT}; skipping"; continue; fi

  FG=$(free_gb)
  if [ "$FG" -lt 15 ]; then log "  ABORT: only ${FG}GB free (<15GB) — not extracting ${tag}"; continue; fi

  # Phase 4 — extraction (GPU, shared with 7B)
  if [ -f "${ACTDIR}/metadata.json" ]; then
    log "  [4] extraction already present — skip"
  else
    log "  [4] extraction -> ${ACTDIR} (free ${FG}GB)"
    $PY 04b_extract_annotator.py --annotated "$ANNOT" --save-dir "$ACTDIR" > "logs/ma_${tag}_p4.log" 2>&1
    log "  [4] exit=$?"
  fi

  # Phase 5 — PCA + per-layer nulls
  if [ -f "results/pca/${SHORT}/layer_profiles.json" ]; then log "  [5] skip"; else
    log "  [5] PCA + nulls"
    $PY 05_pca_analysis.py --model-short "$SHORT" --with-nulls --null-resamples 100 --annotated "$ANNOT" > "logs/ma_${tag}_p5.log" 2>&1
    log "  [5] exit=$?"
  fi

  # Phase 5c — cross-layer probing
  if [ -f "results/cross_layer/${SHORT}/probe_accuracy.json" ]; then log "  [5c] skip"; else
    log "  [5c] probing"
    $PY 05c_cross_layer_probing.py --model-short "$SHORT" > "logs/ma_${tag}_p5c.log" 2>&1
    log "  [5c] exit=$?"
  fi

  # Phase 5b — geometric diagnostics at matched layers
  if [ -f "results/geometric/${SHORT}/diagnostics_layer17.json" ]; then log "  [5b] skip"; else
    log "  [5b] geometric diagnostics (L 11 14 17 20 27)"
    $PY 05b_geometric_diagnostics.py --model-short "$SHORT" --layers 11 14 17 20 27 --n-resamples 200 > "logs/ma_${tag}_p5b.log" 2>&1
    log "  [5b] exit=$?"
  fi

  # Phase 5d — clustering
  if [ -d "results/clustering/${SHORT}" ]; then log "  [5d] skip"; else
    log "  [5d] clustering"
    $PY 05d_subtype_clustering.py --model-short "$SHORT" > "logs/ma_${tag}_p5d.log" 2>&1
    log "  [5d] exit=$?"
  fi

  # Triangulation
  if [ -f "results/triangulation/${SHORT}/candidate_layers.json" ]; then log "  [tri] skip"; else
    log "  [tri] triangulation"
    $PY compute_layer_triangulation.py --model-short "$SHORT" > "logs/ma_${tag}_tri.log" 2>&1
    log "  [tri] exit=$?"
  fi

  # Robustness (dedup + keystones)
  if [ -f "results/robustness/${SHORT}/geometry_robustness.json" ]; then log "  [rob] skip"; else
    log "  [rob] robustness"
    $PY robustness_geometry.py --model-short "$SHORT" --annotated "$ANNOT" > "logs/ma_${tag}_rob.log" 2>&1
    log "  [rob] exit=$?"
  fi
  log "  ----- ${tag} done -----"
done

# Sonnet robustness (for the 3-way comparison) — fast, data already present
if [ ! -f "results/robustness/R1-1.5B/geometry_robustness.json" ]; then
  log "[rob] sonnet baseline robustness"
  $PY robustness_geometry.py --model-short R1-1.5B --annotated data/annotated_R1-1.5B.json > logs/ma_sonnet_rob.log 2>&1
  log "[rob] sonnet exit=$?"
fi

# 3-way cross-annotator comparison
log "[cmp] cross-annotator agreement + manifold replication"
$PY compare_annotators.py > logs/ma_compare.log 2>&1
log "[cmp] exit=$?"

log "=== MULTI-ANNOTATOR PIPELINE COMPLETE (free $(free_gb)GB) ==="
