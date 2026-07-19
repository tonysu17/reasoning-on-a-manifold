#!/bin/bash
# MAC-SIDE puller for the consolidated run (optional — pod self-terminates anyway).
# Waits for ALL_DONE, pulls results JSONs + the two full-FT activation dirs + logs,
# then writes SYNCED_OK so the pod's deadman kills it early. Idempotent, sleep-tolerant.
set -u
cd "$(dirname "$0")"
HOST=runpod
REMOTE=/workspace/rom-consol
LOG=results/safety_posttrain/consol_puller.log
log(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
mkdir -p results/safety_posttrain data/activations

log "=== consol puller start ==="
until ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" "[ -f $REMOTE/ALL_DONE.marker ]" 2>/dev/null; do
  st=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" "cd $REMOTE 2>/dev/null && ls DONE_*.marker FAILED_*.marker 2>/dev/null | tr '\n' ' '" 2>/dev/null)
  log "waiting (markers: ${st:-none/unreachable})"
  sleep 300
done
log "ALL_DONE seen — pulling"

rsync -az "$HOST:$REMOTE/results/" results/safety_posttrain/consol/ 2>>"$LOG" && log "results pulled"
rsync -az "$HOST:$REMOTE/pod_runner2.log" "$HOST:$REMOTE/stage_*.log" results/safety_posttrain/consol/ 2>>"$LOG" || true
for a in R1-1.5B-fullft-safety-s42 R1-1.5B-fullft-control-s42; do
  if ssh -o BatchMode=yes "$HOST" "[ -d $REMOTE/data/activations/$a ]" 2>/dev/null; then
    rsync -az "$HOST:$REMOTE/data/activations/$a" data/activations/ 2>>"$LOG" \
      && log "pulled $a ($(du -sh "data/activations/$a" | cut -f1))"
  else
    log "activations $a missing on pod (stage failed?)"
  fi
done
rsync -az "$HOST:$REMOTE/data/control_offpolicy_sft.json" data/ 2>>"$LOG" || log "control_offpolicy_sft.json missing"

n_json=$(ls results/safety_posttrain/consol/pt08_*.json 2>/dev/null | wc -l | tr -d ' ')
log "pt08 JSONs local: ${n_json}/5"
ssh -o BatchMode=yes "$HOST" "touch $REMOTE/SYNCED_OK.marker" 2>/dev/null \
  && log "SYNCED_OK written — pod will self-terminate" \
  || log "could not write SYNCED_OK (pod already dead?)"
log "=== consol puller complete ==="
