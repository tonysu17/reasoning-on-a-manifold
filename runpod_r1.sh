#!/bin/bash
# R1 compression contrast (R1_COMPRESSION_PREREG.md) on RunPod.
# GPU work: teacher-forced extract 3 arms × 200 chains (~1h) + own-generation
# 4 arms × 200 chains @ ≤8192 new tokens (~2-4h batched) + score-gen (~1h).
# Single 4090 ≈ 4-6 GPU-hours ≈ $3-5. No API credits. Analyse runs on the pod
# (CPU, needs the pushed r1 reference shards) and is pulled back with results.
#
# Usage from the Mac:
#   POD=runpod ./runpod_r1.sh setup     # one-time deps on a fresh pod
#   POD=runpod ./runpod_r1.sh push      # code + chains + 200-chain shard subset
#   POD=runpod ./runpod_r1.sh launch    # full run in tmux 'r1'
#   POD=runpod ./runpod_r1.sh status    # tail progress
#   POD=runpod ./runpod_r1.sh pull      # rsync results/r1_compression back
#   # then STOP THE POD (verify pull first — E10 playbook).
set -euo pipefail
cd "$(dirname "$0")"
POD="${POD:-runpod}"
REMOTE=/workspace/reasoning-on-manifold
MODELS="${MODELS:-deepscaler star1 qwenmath r1}"
BATCH="${BATCH:-8}"

case "${1:-}" in
setup)
  ssh "$POD" "cd $REMOTE && bash runpod_setup.sh"
  ;;

push)
  ssh "$POD" "mkdir -p $REMOTE/data $REMOTE/src $REMOTE/results/loop_geometry/R1-1.5B/shards \
    $REMOTE/results/r0_entropy_ladder/R1-1.5B/ent_shards $REMOTE/results/eval/R1-1.5B__E9_1_greedy \
    $REMOTE/results/r1_compression"
  rsync -az --no-owner --no-group --no-perms \
    30_r1_compression.py 29_r0_entropy_ladder.py runpod_setup.sh pyproject.toml "$POD:$REMOTE/"
  rsync -az --no-owner --no-group --no-perms src/ "$POD:$REMOTE/src/"
  rsync -az --no-owner --no-group --no-perms data/chains_R1-1.5B.json "$POD:$REMOTE/data/"
  rsync -az --no-owner --no-group --no-perms \
    results/eval/R1-1.5B__E9_1_greedy/eval_task_ids.json "$POD:$REMOTE/results/eval/R1-1.5B__E9_1_greedy/"
  # R0 sample + its ent shards (r1 reference arm) + ONLY the 200 sampled E9.0
  # state shards (grid reference; full shard dir is ~3 GB, the subset ~700 MB)
  rsync -az --no-owner --no-group --no-perms \
    results/r0_entropy_ladder/R1-1.5B/sample.json "$POD:$REMOTE/results/r0_entropy_ladder/R1-1.5B/"
  rsync -az --no-owner --no-group --no-perms \
    results/r0_entropy_ladder/R1-1.5B/ent_shards/ "$POD:$REMOTE/results/r0_entropy_ladder/R1-1.5B/ent_shards/"
  python3 - <<'EOF' > /tmp/r1_shard_list.txt
import json
s = json.load(open('results/r0_entropy_ladder/R1-1.5B/sample.json'))
print("\n".join(f"{t}.npz" for t in s["loop"] + s["clean"]))
EOF
  rsync -az --no-owner --no-group --no-perms --files-from=/tmp/r1_shard_list.txt \
    results/loop_geometry/R1-1.5B/shards/ "$POD:$REMOTE/results/loop_geometry/R1-1.5B/shards/"
  ;;

launch)
  ssh "$POD" "cd $REMOTE && tmux new-session -d -s r1 \
    'python3 -u 30_r1_compression.py --stage all --models $MODELS --batch $BATCH \
       2>&1 | tee r1_run.log; echo R1_ALL_DONE >> r1_run.log'"
  echo "launched tmux 'r1' on $POD — ./runpod_r1.sh status to follow"
  ;;

status)
  ssh "$POD" "tail -25 $REMOTE/r1_run.log 2>/dev/null || echo 'no log yet'"
  ;;

pull)
  rsync -az --no-owner --no-group --no-perms \
    "$POD:$REMOTE/results/r1_compression/" results/r1_compression/
  rsync -az --no-owner --no-group --no-perms "$POD:$REMOTE/r1_run.log" results/r1_compression/ || true
  echo "pulled — verify results/r1_compression/REPORT.md before stopping the pod"
  ;;

*)
  echo "usage: POD=<host> $0 {setup|push|launch|status|pull}"; exit 1 ;;
esac
