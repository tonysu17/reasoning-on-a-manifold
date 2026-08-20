#!/bin/bash
# PT-B1 mac-side turnkey: push | launch | status | pull  (RunPod 4090-class pod)
#
#   bash runpod_ptb1.sh push         --host H --port P [--key ~/.ssh/id_ed25519]
#   bash runpod_ptb1.sh launch       --host H --port P
#   bash runpod_ptb1.sh status       --host H --port P
#   bash runpod_ptb1.sh pull-partial --host H --port P   (mid-run insurance)
#   bash runpod_ptb1.sh pull         --host H --port P
#
# Runs on CONTAINER/EPHEMERAL DISK — no network volume (the rom volume sat at
# ~60/70 GB and quota exhaustion killed the first J-space D2/D3 attempt; this
# job churns ~20 GB writing and deleting six merged checkpoints). Ephemeral
# disk means no persistence across a pod death, which is what pull-partial is
# for. Size the container disk at >= 40 GB (peak need ~12 GB).
#
# House rules baked in (jspace ops lessons): tar payload, never a repo clone;
# setsid-detached launch verified PPID=1; pull only on PTB1_ALL_DONE.marker,
# hash-verified against the pod's remote_manifest before anything lands in
# results/. Termination is a separate, human-triggered step AFTER a verified
# pull (scripts/pod_sync_and_terminate.sh pattern) — this script never stops
# a pod.
set -euo pipefail
cd "$(dirname "$0")"

CMD="${1:?push|pull-partial|launch|status|pull}"; shift
HOST=""; PORT=""; KEY="$HOME/.ssh/id_ed25519"
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --key)  KEY="$2";  shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ -n "$HOST" ] && [ -n "$PORT" ] || { echo "--host and --port required" >&2; exit 2; }
SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new root@$HOST"
REMOTE=/root/ptb1

PAYLOAD=(
  ptb1_executor.py ptb1_pod_arm.py ptb1_pod_job.sh
  pt02_train_safety_lora.py ph2_executor.py ph2_manifest.py
  ph2_missingness_bounds.py
  data/safety_star1_sft.json data/control_generic_sft.json
  results/prereg/phase2_task_manifest.json
  results/prereg/PTB1_SAFETY_CONTROL_BEHAVIOUR_PREREG_2026-08-20.md
  results/prereg/PTB1_AMENDMENT_1_2026-08-20.md
  results/ph2/battery/base.json
  results/ptb1/identity_gate/reference.npz
  results/ptb1/identity_gate/reference_keys.json
  results/ptb1/identity_gate/verification_chains.json
  results/ptb1/identity_gate/reference_meta.json
)

case "$CMD" in
  push)
    for f in "${PAYLOAD[@]}"; do [ -e "$f" ] || { echo "missing payload file: $f" >&2; exit 1; }; done
    git rev-parse HEAD > PTB1_SOURCE_COMMIT.txt
    TAR=/tmp/ptb1_payload.tar.gz
    tar -czf "$TAR" PTB1_SOURCE_COMMIT.txt "${PAYLOAD[@]}" src configs
    echo "payload: $(du -h "$TAR" | cut -f1)"
    $SSH "mkdir -p $REMOTE"
    scp -P "$PORT" -i "$KEY" "$TAR" "root@$HOST:$REMOTE/payload.tar.gz"
    $SSH "cd $REMOTE && tar -xzf payload.tar.gz && ls ptb1_pod_job.sh ptb1_pod_arm.py results/ptb1/identity_gate/reference.npz >/dev/null && echo PUSH-VERIFIED"
    ;;
  launch)
    $SSH "cd $REMOTE && setsid nohup bash ptb1_pod_job.sh >/dev/null 2>&1 & echo \$! > ptb1.pid; sleep 2; pid=\$(cat ptb1.pid); ps -o pid,ppid,etime,args -p \$pid && [ \"\$(ps -o ppid= -p \$pid | tr -d ' ')\" = 1 ] && echo DETACHED-OK"
    echo "Launched. Watch:  bash runpod_ptb1.sh status --host $HOST --port $PORT"
    ;;
  status)
    $SSH "cd $REMOTE && ls PTB1_JOB_RUNNING PTB1_ALL_DONE.marker PTB1_JOB_FAILED.marker 2>/dev/null; ls results/ptb1/status/ 2>/dev/null; tail -8 ptb1_job.log 2>/dev/null"
    ;;
  pull-partial)
    # Ephemeral-disk insurance: mirror whatever has completed so far into a
    # staging tree WITHOUT touching results/ptb1/. Safe to run repeatedly
    # mid-run; a pod death then costs at most the in-flight arm. Never
    # hash-verified (the run is still writing), so this tree is NEVER the
    # artefact of record — only `pull` promotes files into results/ptb1/.
    STAGING="results/ptb1/.partial_$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$STAGING"
    rsync -az -e "ssh -p $PORT -i $KEY" \
      --exclude 'identity_gate/extract_*' --exclude 'identity_gate/reference.npz' \
      "root@$HOST:$REMOTE/results/ptb1/" "$STAGING/" || true
    rsync -az -e "ssh -p $PORT -i $KEY" "root@$HOST:$REMOTE/ptb1_job.log" \
      "$STAGING/ptb1_job.log" || true
    echo "partial snapshot -> $STAGING (NOT verified, NOT of record)"
    ls "$STAGING/battery" 2>/dev/null || true
    ;;
  pull)
    $SSH "cd $REMOTE && ls PTB1_ALL_DONE.marker" >/dev/null \
      || { echo "refusing pull: PTB1_ALL_DONE.marker absent" >&2; exit 1; }
    STAGING="results/ptb1/.pull_staging"
    rm -rf "$STAGING"; mkdir -p "$STAGING"
    rsync -az -e "ssh -p $PORT -i $KEY" "root@$HOST:$REMOTE/results/ptb1/" "$STAGING/"
    rsync -az -e "ssh -p $PORT -i $KEY" "root@$HOST:$REMOTE/ptb1_job.log" "$STAGING/provenance/ptb1_job.log"
    ( cd "$STAGING" && while read -r want path; do
        got=$(shasum -a 256 "$path" | cut -d' ' -f1)
        [ "$got" = "$want" ] || { echo "HASH MISMATCH: $path" >&2; exit 1; }
      done < remote_manifest.txt && echo "PULL-VERIFIED $(wc -l < remote_manifest.txt | tr -d ' ') files" )
    rsync -a "$STAGING/" results/ptb1/
    rm -rf "$STAGING"
    echo "Pulled into results/ptb1/. Terminate the pod only after eyeballing this."
    ;;
  *) echo "unknown command $CMD" >&2; exit 2 ;;
esac
