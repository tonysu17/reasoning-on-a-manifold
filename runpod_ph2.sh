#!/bin/bash
# Phase-2 pod session (Mac side) — SESSION_HANDOFF_2026-08-09 §4.
# Pod job (ph2_pod_job.sh): F5 fallback -> ph2 discovery -> target gates -> refit.
# Gates are the wall-clock driver (~10-17 GPU-h ≈ overnight, ~$6-12 on a 4090).
# Battery/annotation are NOT here (spend stages; separate --authorised launch).
# House kill discipline: NO on-pod self-kill; watcher pulls on marker; terminate
# from the RunPod console only after `pull` is verified.
#
# Usage (pod = ssh alias, e.g. ~/.ssh/config Host runpod):
#   POD=runpod ./runpod_ph2.sh setup    # one-time deps (fresh containers every time:
#                                       #   apt rsync+tmux, pip -e .[gpu], pin transformers)
#   POD=runpod ./runpod_ph2.sh push     # code + sealed inputs + F5 pairs
#   POD=runpod ./runpod_ph2.sh launch   # run ph2_pod_job.sh in tmux 'ph2'
#   POD=runpod ./runpod_ph2.sh status   # PH2_STATUS + log tail
#   POD=runpod ./runpod_ph2.sh watch    # background watcher: pull + notify on DONE/FAILED
#   POD=runpod ./runpod_ph2.sh pull     # results + s3 backlog + STAR1 inert staging
set -euo pipefail
cd "$(dirname "$0")"
POD="${POD:-runpod}"
REMOTE=/workspace/reasoning-on-manifold

case "${1:-}" in
setup)
  ssh "$POD" "command -v rsync >/dev/null && command -v tmux >/dev/null || { apt-get update -q && apt-get install -y -q rsync tmux; }"
  ssh "$POD" "mkdir -p $REMOTE"
  rsync -rltz runpod_setup.sh pyproject.toml "$POD:$REMOTE/"
  ssh "$POD" "cd $REMOTE && bash runpod_setup.sh && pip install -q 'transformers==4.49.0'"
  ;;
push)
  ssh "$POD" "mkdir -p $REMOTE/data $REMOTE/results/prereg \
    $REMOTE/results/das/R1-1.5B/main $REMOTE/results/das/R1-1.5B/width \
    $REMOTE/results/eval/R1-1.5B__E1 $REMOTE/results/safety_posttrain/rl"
  rsync -rltz \
    ph2_pod_job.sh ph2_executor.py ph2_manifest.py \
    20_das_backtracking.py 21_das_width.py 04_extract_activations.py \
    pt10_train_dpo.py merge_adapter.py pt08_surprisal_entropy.py \
    pyproject.toml runpod_setup.sh \
    src configs "$POD:$REMOTE/"
  rsync -rltz data/annotated_R1-1.5B.json data/tasks_final.json \
    data/dpo_control.json "$POD:$REMOTE/data/"
  # STAR1 symlinks to the same corpus (teacher-forced design) — materialise it
  rsync -rltzL data/annotated_STAR1-1.5B.json "$POD:$REMOTE/data/" 2>/dev/null || true
  rsync -rltz results/prereg/ "$POD:$REMOTE/results/prereg/"
  rsync -rltz results/das/R1-1.5B/main/pairs.json "$POD:$REMOTE/results/das/R1-1.5B/main/"
  rsync -rltz results/das/R1-1.5B/width/frame_k2.npy "$POD:$REMOTE/results/das/R1-1.5B/width/"
  rsync -rltz results/eval/R1-1.5B__E1/eval_task_ids.json \
    "$POD:$REMOTE/results/eval/R1-1.5B__E1/" 2>/dev/null || true
  echo "pushed. Next: POD=$POD $0 launch"
  ;;
launch)
  ssh "$POD" "cd $REMOTE && rm -f PH2_DONE.marker PH2_STATUS && chmod +x ph2_pod_job.sh && tmux new-session -d -s ph2 'bash ph2_pod_job.sh'"
  echo "launched tmux 'ph2' on $POD — use: POD=$POD $0 watch"
  ;;
status)
  ssh "$POD" "cd $REMOTE && echo STATUS=\$(cat PH2_STATUS 2>/dev/null || echo none) && tail -20 ph2.log 2>/dev/null"
  ;;
watch)
  ( while true; do
      ST=$(ssh "$POD" "cat $REMOTE/PH2_STATUS 2>/dev/null" || echo ssh-fail)
      case "$ST" in
        DONE|FAILED*) break ;;
      esac
      sleep 120
    done
    "$0" pull || true
    MSG="ph2 pod job: ${ST}"
    command -v osascript >/dev/null && osascript -e "display notification \"$MSG\" with title \"runpod_ph2\"" || echo "$MSG"
  ) >/dev/null 2>&1 &
  echo "watcher started (pid $!) — polls PH2_STATUS every 2 min (marker-keyed, never a status string match on logs); pulls + notifies on DONE/FAILED; never kills the pod"
  ;;
pull)
  mkdir -p results/ph2 results/safety_posttrain/ph0_s3 data/activations \
           results/safety_posttrain/rl
  # Phase-2 pod-stage outputs (discovery states, frames, gates, refit, provenance)
  rsync -rltz "$POD:$REMOTE/results/ph2/" results/ph2/ || true
  # Phase-0 backlog: s3 artifacts off the volume
  rsync -rltz "$POD:/workspace/reasoning-on-manifold/results/safety_posttrain/ph0_s3/" \
    results/safety_posttrain/ph0_s3/ 2>/dev/null || true
  # STAR1 inert extraction -> LOCAL STAGING NAME (never the canonical dir):
  # the s2 run wrote to pod data/activations/STAR1-1.5B/ (--short-name ignored
  # for registry models) — map it to STAR1-1.5B-6label here on the Mac side.
  rsync -rltz "$POD:$REMOTE/data/activations/STAR1-1.5B/" \
    data/activations/STAR1-1.5B-6label/ 2>/dev/null || true
  # F5 fallback outputs (adapter/summaries/pt08; merged weights stay on volume)
  rsync -rltz "$POD:$REMOTE/data/activations/R1-1.5B-dpo-control-f5" \
    data/activations/ 2>/dev/null || true
  rsync -rltz --exclude merged "$POD:$REMOTE/results/safety_posttrain/rl/dpo_control_f5" \
    "$POD:$REMOTE/results/safety_posttrain/rl/pt08_R1-1.5B-dpo-control-f5.json" \
    results/safety_posttrain/rl/ 2>/dev/null || true
  rsync -rltz "$POD:$REMOTE/ph2.log" "$POD:$REMOTE/PH2_STATUS" results/ph2/ 2>/dev/null || true
  echo "pulled. Verify results/ph2/{discovery,gates,frames,refit} + staging dirs,"
  echo "then terminate the pod from the RunPod console (never on-pod)."
  echo "Local follow-ons: inert-control battery (freeze §1.3) + s3 closure notes."
  ;;
*)
  grep '^#' "$0" | head -17
  ;;
esac
