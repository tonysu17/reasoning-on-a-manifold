#!/bin/bash
# MAC-SIDE puller (optional for correctness — the pod self-terminates either way).
# Polls the pod for per-arm DONE markers, pulls each activation set as it appears,
# and writes SYNCED_OK.marker on the pod once all four are local, which lets the
# pod's deadman kill it early. Idempotent and sleep-tolerant: ssh failures and
# laptop sleep just delay the next poll; re-running it any time is safe.
set -u
cd "$(dirname "$0")"
HOST=runpod
REMOTE=/workspace/rom-seedrep
LOG=results/safety_posttrain/seedrep_puller.log
ARMS="R1-1.5B-lora-safety1000-s43 R1-1.5B-lora-control1000-s43 R1-1.5B-lora-safety1000-s44 R1-1.5B-lora-control1000-s44"
log(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

log "=== puller start ==="
while :; do
  all=1
  for name in $ARMS; do
    [ -f "data/activations/${name}/row_index.json" ] && continue
    all=0
    if ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" "[ -f $REMOTE/DONE_${name}.marker ]" 2>/dev/null; then
      log "pulling ${name}…"
      mkdir -p data/activations
      if rsync -az "$HOST:$REMOTE/data/activations/${name}" data/activations/ 2>>"$LOG" \
         && [ -f "data/activations/${name}/row_index.json" ]; then
        log "pulled ${name} ($(du -sh "data/activations/${name}" | cut -f1))"
      else
        log "pull of ${name} failed; will retry"
        rm -rf "data/activations/${name}"
      fi
    fi
  done
  if [ "$all" = 1 ]; then
    log "all four arms local — sending SYNCED_OK"
    ssh -o BatchMode=yes "$HOST" "touch $REMOTE/SYNCED_OK.marker" 2>/dev/null \
      && log "SYNCED_OK written; pod will self-terminate shortly" \
      || log "could not write SYNCED_OK (pod already dead?) — results are local, done either way"
    break
  fi
  sleep 180
done
log "=== puller complete ==="
