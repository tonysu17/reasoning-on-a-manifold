#!/bin/bash
# POD-SIDE autonomous runner: finishes the seed-replication experiment and then
# KILLS THIS POD, regardless of whether the laptop is awake.
#
#   - waits for the in-flight arm 1 (safety1000-s43), then runs arms 2-4
#   - touches ALL_DONE.marker when every arm has DONE
#   - deadman: waits up to GRACE_H hours for the Mac to write SYNCED_OK.marker
#     (it does so after pulling all four activation sets), then terminates the
#     pod via the pod-scoped GraphQL API either way
#   - /workspace is a persistent network volume: results survive pod death,
#     so the worst case (laptop closed past the grace window) loses nothing
#   - GLOBAL_H is the absolute watchdog: pod dies at that age no matter what
#
# Launched once via: setsid nohup bash pod_runner.sh > pod_runner.log 2>&1 &
set -u
cd /workspace/rom-seedrep
GRACE_H=6
GLOBAL_H=14
START=$(date +%s)
log(){ echo "$(date -u '+%F %T') | $*"; }

kill_pod(){
  log "KILL: terminating pod at $(date -u)"
  source /etc/rp_environment
  curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"
  sleep 60; log "KILL: still alive 60s after terminate call (unexpected)"
}

global_check(){
  if [ $(( $(date +%s) - START )) -gt $(( GLOBAL_H * 3600 )) ]; then
    log "WATCHDOG: ${GLOBAL_H}h global limit hit"; kill_pod; fi
}

wait_arm(){  # $1 name — waits for DONE/FAILED of an arm someone else launched
  local name="$1"
  until [ -f "DONE_${name}.marker" ]; do
    [ -f "FAILED_${name}.marker" ] && { log "arm ${name} FAILED — continuing to next"; return 1; }
    global_check; sleep 60
  done
  log "arm ${name} DONE"
}

run_arm(){  # $1 data  $2 seed  $3 name
  local data="$1" seed="$2" name="$3"
  [ -f "DONE_${name}.marker" ] && { log "arm ${name} already done"; return 0; }
  log "launching arm ${name} (seed ${seed})"
  PYBIN=python3 bash spark_seedrep_remote.sh "$data" "$seed" "$name"
  wait_arm "$name"
}

log "=== pod runner start (pod $(grep -oE 'RUNPOD_POD_ID=\S+' /etc/rp_environment)) ==="
wait_arm R1-1.5B-lora-safety1000-s43 || true
run_arm data/control_generic_sft.json 43 R1-1.5B-lora-control1000-s43 || true
run_arm data/safety_star1_sft.json  44 R1-1.5B-lora-safety1000-s44  || true
run_arm data/control_generic_sft.json 44 R1-1.5B-lora-control1000-s44 || true

n_done=$(ls DONE_*.marker 2>/dev/null | wc -l)
log "all arms attempted: ${n_done}/4 DONE"
touch ALL_DONE.marker

log "deadman: waiting up to ${GRACE_H}h for SYNCED_OK.marker from the Mac"
DEADLINE=$(( $(date +%s) + GRACE_H * 3600 ))
until [ -f SYNCED_OK.marker ]; do
  [ $(date +%s) -gt $DEADLINE ] && { log "deadman: grace elapsed, results remain on the persistent volume"; break; }
  global_check; sleep 120
done
[ -f SYNCED_OK.marker ] && log "SYNCED_OK received"
kill_pod
