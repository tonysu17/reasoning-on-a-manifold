#!/bin/bash
# E10.1 DAS-1D on backtracking (E10_DAS_PREREG.md) on RunPod.
# GPU work = DAS training (3 dirs × 3 layers × 60 epochs, short forwards) + one
# interchange-eval pass. ~1-3 GPU-hours on a 4090, ~$1-5 of pod time. No API credits.
#
# Prereqs: a running pod with an ssh config host (default "runpod"); on the pod,
# deps installed via runpod_setup.sh. The pairs stage is CPU and runs on the pod
# inside 'launch' (it needs only the tokenizer), so nothing to pre-stage.
#
# Usage from the Mac:
#   POD=runpod ./runpod_e10_das.sh setup    # one-time deps on a fresh pod
#   POD=runpod ./runpod_e10_das.sh push     # sync code + annotated data + E1 vector
#   POD=runpod ./runpod_e10_das.sh launch   # DAS train+eval in tmux 'e10'
#   POD=runpod ./runpod_e10_das.sh status   # tail progress
#   POD=runpod ./runpod_e10_das.sh pull     # rsync results/das back
#   # then STOP THE POD.
#
# One-shot (assumes pod already set up):
#   POD=runpod ./runpod_e10_das.sh push && POD=runpod ./runpod_e10_das.sh launch
set -euo pipefail
cd "$(dirname "$0")"
POD="${POD:-runpod}"
REMOTE=/workspace/reasoning-on-manifold
OUT=results/das/R1-1.5B
LAYERS="${LAYERS:-17 11 27}"
NPAIRS="${NPAIRS:-400}"

case "${1:-}" in
setup)
  ssh "$POD" "cd $REMOTE && bash runpod_setup.sh"
  ;;

push)
  ssh "$POD" "mkdir -p $REMOTE/data $REMOTE/src $REMOTE/configs $REMOTE/$OUT \
    $REMOTE/results/steering_vectors/R1-1.5B__E1_pooled"
  rsync -az --no-owner --no-group --no-perms \
    20_das_backtracking.py runpod_setup.sh pyproject.toml "$POD:$REMOTE/"
  rsync -az --no-owner --no-group --no-perms src/ "$POD:$REMOTE/src/"
  rsync -az --no-owner --no-group --no-perms configs/ "$POD:$REMOTE/configs/" 2>/dev/null || true
  rsync -az --no-owner --no-group --no-perms data/annotated_R1-1.5B.json "$POD:$REMOTE/data/"
  rsync -az --no-owner --no-group --no-perms \
    results/steering_vectors/R1-1.5B__E1_pooled/backtracking_single.npy \
    results/steering_vectors/R1-1.5B__E1_pooled/metadata.json \
    "$POD:$REMOTE/results/steering_vectors/R1-1.5B__E1_pooled/"
  echo "pushed (code + annotated chains + E1 backtracking vector)."
  ;;

launch)
  ssh "$POD" "cd $REMOTE && tmux new-session -d -s e10 \
    'python3 -u 20_das_backtracking.py --stage all --layers $LAYERS --n-pairs $NPAIRS \
      > e10_das.log 2>&1 && \
     python3 -u 20_das_backtracking.py --stage controls --layers $LAYERS --n-pairs $NPAIRS \
      >> e10_das.log 2>&1; touch E10_DAS_DONE.marker'"
  echo "launched DAS train+eval+controls in tmux 'e10' (layers $LAYERS, $NPAIRS pairs; ETA ~1-3 h on a 4090)."
  ;;

status)
  ssh "$POD" "cd $REMOTE && tail -6 e10_das.log 2>/dev/null | tr '\r' '\n' | tail -6; \
    ls E10_DAS_DONE.marker 2>/dev/null && echo DONE || echo RUNNING"
  ;;

pull)
  mkdir -p "$OUT"
  rsync -azv --no-owner --no-group --no-perms "$POD:$REMOTE/$OUT/" "$OUT/"
  echo "pulled DAS results -> $OUT/ . Now inspect $OUT/main/REPORT.md (and STOP THE POD)."
  ;;

*)
  echo "usage: [POD=host] [LAYERS='17 11 27'] [NPAIRS=400] $0 {setup|push|launch|status|pull}"
  exit 1
  ;;
esac
