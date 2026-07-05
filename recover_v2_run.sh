#!/bin/bash
# Patient straggler recovery (paced for Bedrock throttling) + final Δ_floor analysis.
set -u
cd "/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold" || exit 1
caffeinate -dimsu -w $$ &
source "$HOME/.rom_proxy_env"
OUT=results/eval/R1-1.5B__E1
LOG="$OUT/recover_v2.log"
log(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
rem_count(){ python3 -c "
import json
D='$OUT'; ann=json.load(open(f'{D}/annotated_steered.json')); allc=json.load(open(f'{D}/steering_results.json'))
K=('task_id','behaviour','method','alpha'); ck={tuple(r.get(k) for k in K) for r in ann}
print(sum(1 for c in allc if tuple(c.get(k) for k in K) not in ck))"; }

prev=99999
for pass in 1 2 3 4 5; do
  log "=== patient recovery pass $pass ==="
  python3 recover_stragglers_v2.py 2>&1 \
    | grep -ivE "HTTP Request|it/s\]|Attempt [0-9]|Annotation failed|Chunk [0-9]|parsed 0 spans" | tee -a "$LOG"
  rem=$(rem_count); log "after pass $pass: $rem remain"
  [ "$rem" -eq 0 ] && break
  [ "$rem" -ge "$prev" ] && { log "no progress ($rem) — proxy still throttled; stopping retries"; break; }
  prev="$rem"
  log "cooldown 180s (let Bedrock rate window recover)…"
  sleep 180
done

log "=== final Δ_floor analysis ($(rem_count) chains still unrecovered) ==="
python3 08_steering_analysis.py --eval-dir "$OUT" \
  --alpha-star "backtracking=1.0,uncertainty-estimation=1.0,example-testing=1.0,adding-knowledge=1.0" \
  --arms single_direction manifold_k3 manifold_k5 --n-resamples 10000 2>&1 | tee -a "$LOG"
log "=== DONE — report: $OUT/delta_floor_report.json ==="
