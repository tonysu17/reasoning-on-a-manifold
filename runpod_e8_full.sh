#!/bin/bash
# E8 FULL run — the Phase-7 steering headline. Steering at the E1-ARGMAX
# per-behaviour layers (backtracking 17, uncertainty 15, example-testing 15,
# adding-knowledge 17 — OUR faithful Venhoff-attribution result, NOT L27),
# k∈{3,5} (E4 sweet spot), 50-task hold-out, greedy n=1, α∈{0, 1.0}. Pipeline:
#   1. generate (GPU, full-length, resumable per sweep)
#   2. annotate with Sonnet (per Tony 2026-06-28; saves EVERY chain)
#   3. Δ_floor analysis (band-UNGATED → "preliminary" verdicts)
# NOTE: Sonnet is the BUILDER of the original labels, so this is NOT the
# de-circularised headline; Δ_floor (arm vs floor, same annotator) is a
# within-annotator relative contrast. No second annotator ⇒ no noise band ⇒
# preliminary, never PASS. Add a non-builder band (e.g. qwen.qwen3-vl-235b-a22b)
# later for a pass-grade verdict.
# Every step is resumable — re-run this script to continue from the last saved
# chain after a credit-out / disconnect. TELL TONY before launching ($ run).
set -u
cd /workspace/reasoning-on-manifold
source ~/.rom_proxy_env
export HF_HOME=/workspace/hf
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # anti-fragmentation OOM guard
PY="python -u"

VEC=results/steering_vectors/R1-1.5B__E1_pooled
OUT=results/eval/R1-1.5B__E1
ANNOTATOR=eu.anthropic.claude-sonnet-4-5-20250929-v1:0   # Sonnet (per Tony) — NB: =builder
LEAN="--no-random-control --no-orthogonal-complement --n-random-subspaces 2"
BATCH=32                                                  # ~14GB worst-case of 24GB (~10GB margin) — OOM-proof for unattended overnight; self-healing fallback halves on OOM anyway
mkdir -p "$OUT"

echo "=== [$(date)] E8 1/3 GENERATION (GPU, batch=$BATCH; checkpoint per batch) ==="
$PY 07_evaluate_steering.py --model 1.5b --vectors-dir "$VEC" --out-dir "$OUT" \
    --alpha-values 0 1.0 $LEAN --batch-size "$BATCH" --skip-annotation > "$OUT/gen.log" 2>&1
echo "=== [$(date)] gen rc=$? chains=$($PY -c "import json;print(len(json.load(open('$OUT/steering_results.json'))))" 2>/dev/null) ==="
date > "$OUT/E8_gen_DONE.marker"

echo "=== [$(date)] E8 2/3 ANNOTATE = $ANNOTATOR (save EVERY chain) ==="
$PY annotate_steered.py --eval-dir "$OUT" --annotator "$ANNOTATOR" \
    --out annotated_steered.json > "$OUT/annotate.log" 2>&1
echo "=== [$(date)] annotate rc=$? ==="
date > "$OUT/E8_annotate_DONE.marker"

echo "=== [$(date)] E8 3/3 Δ_floor ANALYSIS (band-ungated → preliminary) ==="
$PY 08_steering_analysis.py --eval-dir "$OUT" \
    --alpha-star "backtracking=1.0,uncertainty-estimation=1.0,example-testing=1.0,adding-knowledge=1.0" \
    --arms single_direction manifold_k3 manifold_k5 \
    --n-resamples 10000 > "$OUT/analysis.log" 2>&1
echo "=== [$(date)] analysis rc=$? ==="
grep -A40 "Δ_floor HEADLINE" "$OUT/analysis.log" | tail -20
date > "$OUT/E8_DONE.marker"
echo "=== [$(date)] E8 FULL COMPLETE ==="
