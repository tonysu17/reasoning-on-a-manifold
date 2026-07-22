#!/bin/bash
# Mac-side: extract R1-1.5B activations pooled per QWEN3 annotator spans (F3 3-way).
# Small model (R1-1.5B) on a 24GB card; no MXFP4. Push code + qwen3 annotation,
# run 04b_extract_annotator.py, pull the qwenspans dir.
#   POD_HOST=root@IP POD_PORT=39822 ./qwenspans_runner.sh
set -u
cd "$(dirname "$0")"
MAIN="$(pwd)"; LOG=results/robustness/qwenspans_runner.log
POD_HOST="${POD_HOST:?}"; POD_PORT="${POD_PORT:?}"; POD_KEY="${POD_KEY:-$HOME/.ssh/id_ed25519}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -p "$POD_PORT" -i "$POD_KEY" "$POD_HOST")
RS="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $POD_PORT -i $POD_KEY"
log(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

log "=== qwenspans extraction start ($POD_HOST) ==="
"${SSH[@]}" "command -v rsync >/dev/null || (apt-get update -q && apt-get install -y -q rsync)" >>"$LOG" 2>&1 || { log "no rsync"; exit 1; }
"${SSH[@]}" "mkdir -p /workspace/rom-qs/data /workspace/rom-qs/src"
# code (light) + the qwen3 annotated file + config
rsync -az --no-owner --no-group -e "$RS" --exclude '.git' --exclude '__pycache__' --exclude 'results' \
  --exclude 'data' --exclude 'thesis' --exclude '*.pdf' \
  "$MAIN/src" "$MAIN/04b_extract_annotator.py" "$POD_HOST:/workspace/rom-qs/" 2>>"$LOG" \
  || { log "code push FAILED"; exit 1; }
rsync -az --no-owner --no-group -e "$RS" "$MAIN/data/annotated_R1-1.5B__qwen3-235b.json" \
  "$POD_HOST:/workspace/rom-qs/data/" 2>>"$LOG" || { log "annotation push FAILED"; exit 1; }
"${SSH[@]}" "[ -s /workspace/rom-qs/data/annotated_R1-1.5B__qwen3-235b.json ]" || { log "verify FAILED"; exit 1; }

log "launching 04b_extract_annotator (layers 12 16, 4 behaviours)"
"${SSH[@]}" "cd /workspace/rom-qs && rm -f QS_DONE.marker QS_FAILED.marker && setsid nohup bash -c '
  export HF_HOME=/workspace/hf
  pip install -q -U \"transformers>=4.44,<5\" accelerate safetensors 2>/dev/null
  python3 04b_extract_annotator.py --annotated data/annotated_R1-1.5B__qwen3-235b.json \
    --save-dir data/activations/R1-1.5B-qwenspans \
    --behaviours backtracking uncertainty-estimation example-testing adding-knowledge \
    --layers 12 16 --cache-dir /workspace/hf > qs.log 2>&1 \
  && touch QS_DONE.marker || touch QS_FAILED.marker
' > /dev/null 2>&1 < /dev/null & echo launched"

log "waiting on QS marker (poll 60s)"
until "${SSH[@]}" "[ -f /workspace/rom-qs/QS_DONE.marker ] || [ -f /workspace/rom-qs/QS_FAILED.marker ]" 2>/dev/null; do sleep 60; done
if "${SSH[@]}" "[ -f /workspace/rom-qs/QS_FAILED.marker ]" 2>/dev/null; then
  rsync -az --no-owner --no-group -e "$RS" "$POD_HOST:/workspace/rom-qs/qs.log" results/robustness/ 2>>"$LOG" || true
  log "QS FAILED — qs.log pulled; pod left up"; exit 1
fi
log "pulling qwenspans"
rsync -az --no-owner --no-group -e "$RS" "$POD_HOST:/workspace/rom-qs/data/activations/R1-1.5B-qwenspans/" \
  data/activations/R1-1.5B-qwenspans/ 2>>"$LOG"
python3 -c "
import numpy as np, json
from pathlib import Path
d=Path('data/activations/R1-1.5B-qwenspans')
n=len(list(d.glob('*_layer*.npy')))
assert n>=8, f'only {n} npy'
for f in d.glob('*_layer*.npy'): assert np.isfinite(np.load(f)).all(), f.name
print('QWENSPANS OK:', n, 'matrices')
" >> "$LOG" 2>&1 && { touch results/robustness/QWENSPANS_READY.marker; log "=== qwenspans COMPLETE + verified ==="; } || { log "local verify FAILED"; exit 1; }
