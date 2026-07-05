#!/bin/bash
# Recover the annotation stragglers SEQUENTIALLY (low load → no 503s), then re-run
# the Δ_floor analysis on the now-fuller annotated set. Resumable / re-runnable.
set -u
cd "/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold" || exit 1
caffeinate -dimsu -w $$ &
source "$HOME/.rom_proxy_env"
OUT=results/eval/R1-1.5B__E1
LOG="$OUT/recover.log"
log(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

log "=== straggler recovery (sequential) ==="
prev=99999
for pass in 1 2 3 4 5; do
  log "recovery pass $pass…"
  python3 recover_stragglers.py 2>&1 | grep -ivE "HTTP Request|it/s\]|Attempt [0-9]|Annotation failed|Chunk [0-9]" | tee -a "$LOG"
  rem=$(tail -1 "$OUT/_recover_remaining" 2>/dev/null || echo "")
  # derive remaining straggler count directly
  rem=$(python3 -c "
import json
D='$OUT'
ann=json.load(open(f'{D}/annotated_steered.json')); allc=json.load(open(f'{D}/steering_results.json'))
K=('task_id','behaviour','method','alpha'); ck={tuple(r.get(k) for k in K) for r in ann}
print(sum(1 for c in allc if tuple(c.get(k) for k in K) not in ck))")
  log "after pass $pass: $rem stragglers remain"
  [ "$rem" -eq 0 ] && break
  [ "$rem" -ge "$prev" ] && { log "no further progress ($rem) — stopping retries"; break; }
  prev="$rem"
  sleep 20
done

log "=== re-running Δ_floor analysis ==="
python3 08_steering_analysis.py --eval-dir "$OUT" \
  --alpha-star "backtracking=1.0,uncertainty-estimation=1.0,example-testing=1.0,adding-knowledge=1.0" \
  --arms single_direction manifold_k3 manifold_k5 \
  --n-resamples 10000 2>&1 | tee -a "$LOG"
log "=== recovery + analysis COMPLETE — report: $OUT/delta_floor_report.json ==="
