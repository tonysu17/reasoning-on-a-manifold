#!/bin/bash
# E8 — Phase-7 steering grid (RunPod). GENERATION-FIRST: Stage 1 (this script) runs the
# steered generation with NO API/creds needed; Stage 2 (re-annotation, bottom of file) is
# gated on the non-builder annotator id + proxy creds. See E8_LAUNCH_PLAN.md for the design.
#
# TELL TONY FIRST before running — this is a GPU/$ run.
set -u
cd /workspace/reasoning-on-manifold
export HF_HOME=/workspace/hf
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
PY="python -u"

# ── PARAMETERS — edit before launch ───────────────────────────────────────────────────
# Layer hypothesis = choice of vectors dir (07 reads per-behaviour layer from its metadata).
#   L27 first (the leaning layer); switch to __venhoff / __huang / __E1 if L27 fails.
VECTORS_DIR="results/steering_vectors/R1-1.5B"          # L27 | …__venhoff | …__huang | …__E1
OUT_DIR="results/eval/R1-1.5B__L27"
ALPHAS="0 1.0"           # Stage 1: vanilla + single dose ≈ α* (all α*∈0.96–1.06). Pareto: 0 0.5 0.7 1.0 1.3
N_SAMPLES=1             # Stage 1 greedy=1; Stage 2 headline=3 (then TEMPERATURE must be >0)
TEMPERATURE=0          # >0 required if N_SAMPLES>1
# Lean arm set: drop the ~19×-weak norm-matched sanity floor + the orthogonal control,
# halve random-subspace reps. KEEP energy_matched_random (single floor) + random_subspace (mani floor).
LEAN_FLAGS="--no-random-control --no-orthogonal-complement --n-random-subspaces 2"
# ──────────────────────────────────────────────────────────────────────────────────────

mkdir -p "$OUT_DIR"

# ── PREREQ: α* for a non-27 layer (CPU, $0). Skip if VECTORS_DIR is the L27 build. ──────
case "$VECTORS_DIR" in
  *R1-1.5B) echo "=== [$(date)] L27 build — α* already sealed (predictions_layer27.json) ===" ;;
  *)
    echo "=== [$(date)] non-L27 build — ensure α* is computed for its layers (15/17/18). ==="
    echo "    If diagnostics_layer{15,18}.json are missing, run FIRST (CPU, ~10-30 min, \$0):"
    echo "      $PY 05b_geometric_diagnostics.py --model-short R1-1.5B --layers 15 18"
    echo "      $PY predict_saturation.py --layer 15 ; $PY predict_saturation.py --layer 18 ; $PY predict_saturation.py --layer 17"
    ;;
esac

# ── STAGE 1: steered GENERATION only (GPU; writes generation_metrics.json, no annotation) ─
echo "=== [$(date)] E8 Stage 1: generation  vectors=$VECTORS_DIR  alphas=[$ALPHAS]  n=$N_SAMPLES ==="
$PY 07_evaluate_steering.py --model 1.5b \
    --vectors-dir "$VECTORS_DIR" --out-dir "$OUT_DIR" \
    --alpha-values $ALPHAS --n-samples $N_SAMPLES --temperature $TEMPERATURE \
    $LEAN_FLAGS \
    --skip-annotation \
    > "$OUT_DIR/E8_gen.log" 2>&1
echo "=== [$(date)] E8 Stage 1 DONE rc=$? (-> $OUT_DIR/E8_gen.log) ==="
$PY - <<'PYEOF' 2>/dev/null || true
import json, glob, os
d=sorted(glob.glob("results/eval/*/generation_metrics.json"))[-1] if glob.glob("results/eval/*/generation_metrics.json") else None
print("damage preview:", d)
PYEOF
date > "$OUT_DIR/E8_stage1_DONE.marker"

# ── STAGE 2: re-annotate with a NON-BUILDER annotator (resumes from saved generations). ──
# GATED: requires the non-builder proxy id (NOT in repo — Tony) + creds. Uncomment to run.
# export CLAUDE_PROXY_URL="..." CLAUDE_PROXY_KEY="..."
# $PY 07_evaluate_steering.py --model 1.5b \
#     --vectors-dir "$VECTORS_DIR" --out-dir "$OUT_DIR" \
#     --alpha-values $ALPHAS --n-samples $N_SAMPLES --temperature $TEMPERATURE \
#     $LEAN_FLAGS \
#     --annotator-model "<NON_BUILDER_PROXY_ID>" \
#     > "$OUT_DIR/E8_annotate.log" 2>&1
# echo "=== [$(date)] E8 Stage 2 (annotation) DONE rc=$? ==="
# date > "$OUT_DIR/E8_stage2_DONE.marker"
#
# Then the Δ_floor analysis (build per E8_LAUNCH_PLAN.md §8 — not yet implemented):
#   $PY 08_steering_analysis.py --eval-dir "$OUT_DIR"   # <- to be written
