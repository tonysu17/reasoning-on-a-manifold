#!/bin/bash
# E9.0 loop-geometry gate (COLLAPSE_AND_ENTROPY.md §5) on RunPod.
# GPU work = one forward pass per selected chain (~884 loop+clean chains at
# default); ~15-25 min on a 4090, well under $1 of pod time. The detect and
# analyse stages are CPU-only, so we run detect+analyse LOCALLY and only the
# forward-pass extraction on the pod.
#
# Prereqs (same as runpod_safety_runA.sh): a running pod with an ssh config
# host (default "runpod"); on the pod, deps installed via runpod_setup.sh.
#
# Usage from the Mac:
#   ./runpod_e9_loopgeom.sh detect            # local: label loops (CPU, ~1 min)
#   POD=runpod ./runpod_e9_loopgeom.sh push   # sync code + data + loop_labels
#   POD=runpod ./runpod_e9_loopgeom.sh setup  # one-time deps on a fresh pod
#   POD=runpod ./runpod_e9_loopgeom.sh launch # forward-pass extraction in tmux 'e9'
#   POD=runpod ./runpod_e9_loopgeom.sh status # tail progress
#   POD=runpod ./runpod_e9_loopgeom.sh pull   # rsync shards back
#   ./runpod_e9_loopgeom.sh analyse           # local: probe + cosines + precedence
#   # then STOP THE POD.
#
# One-shot convenience (assumes pod already set up):
#   POD=runpod ./runpod_e9_loopgeom.sh detect && \
#     POD=runpod ./runpod_e9_loopgeom.sh push && \
#     POD=runpod ./runpod_e9_loopgeom.sh launch
set -euo pipefail
cd "$(dirname "$0")"
POD="${POD:-runpod}"
REMOTE=/workspace/reasoning-on-manifold
OUT=results/loop_geometry/R1-1.5B
LAYERS="${LAYERS:-15 16 17}"

case "${1:-}" in
detect)
  python3 18_loop_geometry.py --stage detect --layers $LAYERS
  echo "labels: $OUT/loop_labels.json  (push next)"
  ;;

setup)
  ssh "$POD" "cd $REMOTE && bash runpod_setup.sh"
  ;;

push)
  test -f "$OUT/loop_labels.json" || { echo "run './runpod_e9_loopgeom.sh detect' first"; exit 1; }
  ssh "$POD" "mkdir -p $REMOTE/data $REMOTE/src $REMOTE/configs $REMOTE/$OUT $REMOTE/results/steering_vectors"
  rsync -az --no-owner --no-group --no-perms \
    18_loop_geometry.py runpod_setup.sh pyproject.toml "$POD:$REMOTE/"
  rsync -az --no-owner --no-group --no-perms src/ "$POD:$REMOTE/src/"
  rsync -az --no-owner --no-group --no-perms configs/ "$POD:$REMOTE/configs/"
  rsync -az --no-owner --no-group --no-perms data/chains_R1-1.5B.json "$POD:$REMOTE/data/"
  rsync -az --no-owner --no-group --no-perms "$OUT/loop_labels.json" "$POD:$REMOTE/$OUT/"
  rsync -az --no-owner --no-group --no-perms \
    results/steering_vectors/R1-1.5B__E1_pooled "$POD:$REMOTE/results/steering_vectors/"
  echo "pushed (code + chains + loop_labels + E1-pooled vectors)."
  ;;

launch)
  ssh "$POD" "cd $REMOTE && tmux new-session -d -s e9 \
    'python3 -u 18_loop_geometry.py --stage extract --layers $LAYERS \
      > e9_extract.log 2>&1; touch E9_EXTRACT_DONE.marker'"
  echo "launched extraction in tmux 'e9' (ETA ~15-25 min on a 4090)."
  ;;

status)
  ssh "$POD" "cd $REMOTE && tail -3 e9_extract.log | tr '\r' '\n' | tail -3; \
    echo \"shards: \$(ls $OUT/shards 2>/dev/null | wc -l)\"; \
    ls E9_EXTRACT_DONE.marker 2>/dev/null && echo DONE"
  ;;

pull)
  mkdir -p "$OUT/shards"
  # NB: no --info=progress2 — the macOS system rsync (2.6.9) does not know it
  rsync -azv --no-owner --no-group --no-perms "$POD:$REMOTE/$OUT/shards/" "$OUT/shards/"
  echo "pulled $(ls "$OUT/shards" | wc -l) shards. Now: ./runpod_e9_loopgeom.sh analyse (and STOP THE POD)."
  ;;

analyse)
  python3 18_loop_geometry.py --stage analyse --layers $LAYERS
  echo "report: $OUT/report.json + $OUT/REPORT.md + $OUT/directions.npz"
  ;;

*)
  echo "usage: [POD=host] $0 {detect|setup|push|launch|status|pull|analyse}"
  exit 1
  ;;
esac
