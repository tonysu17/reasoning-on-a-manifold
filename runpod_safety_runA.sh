#!/bin/bash
# Run A (RUNPOD_SAFETY_PROPOSAL.md): full-corpus STAR-1 extraction + gated diff.
# Cost ~ $1 of pod time on an RTX 4090. Usage, from the Mac, once Tony has a pod
# and an ssh config host for it (e.g. "runpod"):
#
#   POD=runpod ./runpod_safety_runA.sh push     # sync code + data to the pod
#   POD=runpod ./runpod_safety_runA.sh launch   # start extraction in tmux 'runA'
#   POD=runpod ./runpod_safety_runA.sh status   # tail progress
#   POD=runpod ./runpod_safety_runA.sh pull     # rsync activations back when done
#   ./runpod_safety_runA.sh analyse             # local: pt03 + pt03b gated report
#
# Then STOP THE POD. The extraction writes data/activations/STAR1-1.5B on the
# pod (~6 GB, 28 layers); pull copies it next to the local full-corpus
# R1-1.5B activations, and the analysis runs entirely locally.
set -euo pipefail
cd "$(dirname "$0")"
POD="${POD:-runpod}"
REMOTE=/workspace/reasoning-on-manifold

case "${1:-}" in
push)
  ssh "$POD" "mkdir -p $REMOTE/data $REMOTE/src/safety_posttrain $REMOTE/configs"
  rsync -az --no-owner --no-group --no-perms \
    04_extract_activations.py configs/config.yaml "$POD:$REMOTE/" \
    --rsync-path="mkdir -p $REMOTE/configs && rsync" || true
  rsync -az --no-owner --no-group --no-perms configs/ "$POD:$REMOTE/configs/"
  rsync -az --no-owner --no-group --no-perms src/ "$POD:$REMOTE/src/"
  rsync -az --no-owner --no-group --no-perms \
    data/annotated_R1-1.5B.json data/chains_R1-1.5B.json "$POD:$REMOTE/data/"
  ssh "$POD" "cd $REMOTE/data && ln -sf annotated_R1-1.5B.json annotated_STAR1-1.5B.json"
  echo "pushed. (deps: the pod needs pip install -e .[gpu] or torch+transformers+pyyaml)"
  ;;
launch)
  ssh "$POD" "cd $REMOTE && tmux new-session -d -s runA \
    'python3 -u 04_extract_activations.py --model star1-1.5b --tokenizer-alias 1.5b \
     > star1_full.log 2>&1; touch RUNA_DONE.marker'"
  echo "launched in tmux runA (ETA ~45-90 min on a 4090)"
  ;;
status)
  ssh "$POD" "cd $REMOTE && tail -2 star1_full.log | tr '\r' '\n' | tail -2; ls RUNA_DONE.marker 2>/dev/null && echo DONE"
  ;;
pull)
  mkdir -p data/activations
  rsync -az --info=progress2 "$POD:$REMOTE/data/activations/STAR1-1.5B" data/activations/
  echo "pulled. Now: ./runpod_safety_runA.sh analyse   (and STOP THE POD)"
  ;;
analyse)
  mkdir -p results/safety_posttrain
  python3 pt03_measure_spillover.py \
    --base-acts data/activations/R1-1.5B \
    --post-acts data/activations/STAR1-1.5B \
    --annotated data/annotated_R1-1.5B.json \
    --out results/safety_posttrain/spillover_star1_full.json
  python3 pt03b_spillover_nulls.py \
    --base-acts data/activations/R1-1.5B \
    --post-acts data/activations/STAR1-1.5B \
    --out results/safety_posttrain/spillover_gated_full.json
  echo "reports: results/safety_posttrain/spillover_star1_full.json + spillover_gated_full.json"
  ;;
*)
  echo "usage: POD=<sshhost> $0 push|launch|status|pull|analyse"; exit 1 ;;
esac
