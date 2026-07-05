#!/bin/bash
# E8 SMOKE — validate the whole pipeline end-to-end FAST + CHEAP before the real run:
#   generate (1 task, lean arms, 256 tok) → annotate with the NON-builder Nova-Pro
#   (saving after EVERY chain) → Δ_floor analysis. Confirms creds, the annotator id,
#   annotation FORMAT-following, per-chain checkpointing, and 08 analysis.
set -u
cd /workspace/reasoning-on-manifold
source ~/.rom_proxy_env
export HF_HOME=/workspace/hf
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY="python -u"

VEC=results/steering_vectors/R1-1.5B__venhoff
OUT=results/eval/E8_smoke
ANNOTATOR=amazon.nova-pro-v1:0
LEAN="--no-random-control --no-orthogonal-complement --n-random-subspaces 1"
COMMON="--model 1.5b --vectors-dir $VEC --out-dir $OUT --alpha-values 0 1.0 --max-eval-tasks 1 --max-new-tokens 256 $LEAN"
mkdir -p "$OUT"

echo "=== [$(date)] SMOKE 1/3 generation ==="
$PY 07_evaluate_steering.py $COMMON --skip-annotation > "$OUT/gen.log" 2>&1
echo "gen rc=$?"
$PY -c "import json;d=json.load(open('$OUT/steering_results.json'));print('generated chains:',len(d))" 2>&1

echo "=== [$(date)] SMOKE 2/3 annotate with $ANNOTATOR (checkpoint EVERY chain) ==="
$PY 07_evaluate_steering.py $COMMON --annotator-model "$ANNOTATOR" > "$OUT/annotate.log" 2>&1
echo "annotate rc=$?"
$PY -c "
import json
d=json.load(open('$OUT/annotated_steered.json'))
ne=sum(1 for r in d if r.get('annotations'))
comp=sum(1 for r in d if r.get('annotation_complete'))
print(f'annotated records: {len(d)} | non-empty annotations: {ne} | complete: {comp}')
ex=[r for r in d if r.get('annotations')]
if ex: print('sample labels:', [a.get('label') for a in ex[0]['annotations'][:8]])
" 2>&1

echo "=== [$(date)] SMOKE 3/3 Δ_floor analysis ==="
$PY 08_steering_analysis.py --eval-dir "$OUT" \
    --alpha-star "backtracking=1.0,uncertainty-estimation=1.0,example-testing=1.0,adding-knowledge=1.0" \
    --arms single_direction manifold_k3 manifold_k5 --n-resamples 2000 > "$OUT/analysis.log" 2>&1
echo "analysis rc=$?"
tail -20 "$OUT/analysis.log"
date > "$OUT/SMOKE_DONE.marker"
echo "=== [$(date)] SMOKE COMPLETE ==="
