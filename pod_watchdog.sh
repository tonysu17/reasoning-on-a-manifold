#!/bin/bash
# Autonomous pod watchdog — terminate RunPod pods ONLY when their run is COMPLETED
# (or has definitively CRASHED). Authorized by Tony 2026-07-13:
#   "only kill the runs when they are completed" + "make sure the 12h watchdog
#    doesn't fire again and kill the other runs".
# The pods' own time-watchdogs have been neutralized (R3 subshell killed; runpod3
# API key overridden), so this daemon is the SOLE terminator. It NEVER kills a pod
# whose driver script is still alive with no completion marker — i.e. never kills a
# run that is merely slow or paused between stages. Terminates via saved creds
# (~/.pod_creds) through the RunPod API from the Mac, so it works even when a pod's
# self-kill is disabled or the pod is unreachable. Data-safe: safety-pull first.
set -u
cd "$(dirname "$0")"
LOG=results/pod_watchdog.log
mkdir -p results
log(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
SSH(){ ssh -o BatchMode=yes -o ConnectTimeout=20 "$@"; }
CREDS="$HOME/.pod_creds"

SLEEP=540         # ~9 min/check
CONFIRM=2         # consecutive strikes to confirm crash / unreachable (avoid transient blips)

# per-pod config: workdir | pod-side completion marker | driver pgrep-pattern (bracketed
# so pgrep can't self-match the probe shell) | local READY marker
cfg(){ case "$1" in
  runpod2) echo "/workspace/rom-r3|R3_ALL_DONE.marker|[p]od_r3_full.sh|results/r3_strategy/R3_FULL_READY.marker";;
  runpod3) echo "/workspace/rom-rl|V3_ALL_DONE.marker|[p]od_runner5.sh|results/safety_posttrain/rl/V3_READY_FOR_ANALYSIS.marker";;
  runpod4) echo "/workspace/rom-p0|P0_COMPLETE.marker|[p]od_p0_gptoss.sh|results/safety/P0_READY_FOR_REVIEW.marker";;
esac; }

safety_pull(){ case "$1" in
  runpod2)
    rsync -az --timeout=300 runpod2:/workspace/rom-r3/results/r3_strategy/ results/r3_strategy/ 2>>"$LOG" || true
    rsync -az --timeout=180 runpod2:/workspace/rom-r3/pod_r3_full.log results/r3_strategy/ 2>>"$LOG" || true ;;
  runpod3)
    rsync -az --timeout=300 "runpod3:/workspace/rom-rl/results/pt08_*-v3.json" results/safety_posttrain/rl/ 2>>"$LOG" || true
    rsync -az --timeout=180 runpod3:/workspace/rom-rl/pod_runner5.log results/safety_posttrain/rl/ 2>>"$LOG" || true ;;
  runpod4)
    rsync -az --timeout=180 runpod4:/workspace/rom-p0/data/chains_gpt-oss-20b_p0.json data/ 2>>"$LOG" || true
    rsync -az --timeout=180 runpod4:/workspace/rom-p0/results/safety/ results/safety/ 2>>"$LOG" || true ;;
esac; }

terminate_pod(){  # $1 alias — via saved creds (Mac-side RunPod API)
  local line pid key
  line=$(grep "^$1 " "$CREDS" 2>/dev/null); pid=$(echo "$line" | awk '{print $2}'); key=$(echo "$line" | awk '{print $3}')
  if [ -z "$pid" ] || [ -z "$key" ]; then log "$1: NO saved creds — cannot terminate"; return 1; fi
  log "$1: sending podTerminate (pod=$pid)"
  curl -s "https://api.runpod.io/graphql?api_key=${key}" -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${pid}\\\"})}\"}" >>"$LOG" 2>&1 \
    && log "$1: TERMINATE mutation sent" || log "$1: terminate call FAILED"
}

log "=== pod watchdog start (COMPLETION-ONLY policy; watching: runpod2 runpod3 runpod4) ==="
LIVE="runpod2 runpod3 runpod4"
crash_runpod2=0; crash_runpod3=0; crash_runpod4=0
unr_runpod2=0;   unr_runpod3=0;   unr_runpod4=0
while [ -n "$LIVE" ]; do
  NEW=""
  for p in $LIVE; do
    IFS='|' read -r wd done_marker runner local_ready <<<"$(cfg "$p")"
    # 1) data already pulled locally => completed
    if [ -f "$local_ready" ]; then log "$p: COMPLETED (data pulled) — terminating"; terminate_pod "$p"; continue; fi
    # 2) probe pod: completion marker + driver alive (single ssh)
    probe=$(SSH "$p" "[ -f $wd/$done_marker ] && echo DONE; pgrep -f $runner >/dev/null 2>&1 && echo ALIVE" 2>/dev/null)
    if [ -z "$probe" ] && ! SSH "$p" true 2>/dev/null; then
      eval "unr_${p}=\$((unr_${p}+1))"; u=$(eval echo "\$unr_${p}")
      if [ "$u" -ge "$CONFIRM" ]; then log "$p: UNREACHABLE x$u — gone/self-terminated; dropping"; continue; fi
      log "$p: unreachable (strike $u/$CONFIRM)"; NEW="$NEW $p"; continue
    fi
    eval "unr_${p}=0"
    if echo "$probe" | grep -q DONE; then
      # Pipeline complete on the pod, but the AUTHORITATIVE puller (which pulls everything —
      # incl. large activations for runpod3 — and drops the local READY marker) may not be done.
      # NEVER terminate here: a pull can fail (e.g. pod lacks rsync) and terminating would strand
      # the data on the volume. Best-effort safety_pull, then WAIT for the puller's READY marker
      # (path 1 at loop top) to terminate. Lesson: runpod2/R3 was stranded this way 2026-07-14.
      log "$p: pipeline-complete marker present — best-effort pull; awaiting puller READY to terminate"
      safety_pull "$p"
      eval "crash_${p}=0"; NEW="$NEW $p"; continue
    fi
    if echo "$probe" | grep -q ALIVE; then
      eval "crash_${p}=0"; NEW="$NEW $p"; continue      # driver alive, not done => RUNNING (never kill)
    fi
    # driver NOT alive and NO completion marker => suspected crash (confirm over CONFIRM strikes)
    eval "crash_${p}=\$((crash_${p}+1))"; c=$(eval echo "\$crash_${p}")
    if [ "$c" -ge "$CONFIRM" ]; then
      log "$p: driver died with NO completion marker x$c — CRASHED; safety-pull + terminate"; safety_pull "$p"; terminate_pod "$p"; continue
    fi
    log "$p: driver not found (strike $c/$CONFIRM)"; NEW="$NEW $p"
  done
  LIVE=$(echo $NEW | xargs)
  [ -z "$LIVE" ] && break
  sleep "$SLEEP"
done
log "=== pod watchdog complete — all watched pods terminated/gone ==="
