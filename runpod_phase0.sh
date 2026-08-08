#!/bin/bash
# Phase 0 pod batch (Mac side) — freeze: results/prereg/PHASE0_TRANSPORT_FREEZE_2026-08-02.md §2.
# Stages on pod (ph0_pod_job.sh): s0 F5 throughput probe -> pre-committed route rule (12 GPU-h)
#   s1 F5 control generation (FULL route only; TRAINING = manual gate per ledger §F5)
#   s2 STAR1 inert-control extraction (deduction+initializing, L12/16) = smoke test
#   s3 full-sequence L17/L16 states (r1 + deepscaler) -> injection curve + SV family check
# Cost ~$10-20 on a 4090/A100. Needs a pod (ssh alias in $POD; no RunPod key on this Mac).
# House kill discipline: NO on-pod self-kill; watcher pulls on done/crash; terminate from Mac.
#
# Usage:
#   POD=runpod ./runpod_phase0.sh setup    # one-time deps (runpod_setup.sh)
#   POD=runpod ./runpod_phase0.sh push     # code + data + reference jsons
#   POD=runpod ./runpod_phase0.sh launch   # run ph0_pod_job.sh in tmux 'ph0'
#   POD=runpod ./runpod_phase0.sh status   # tail log + status
#   POD=runpod ./runpod_phase0.sh watch    # background watcher: pull + notify on done/fail
#   POD=runpod ./runpod_phase0.sh pull     # rsync results back
set -euo pipefail
cd "$(dirname "$0")"
POD="${POD:-runpod}"
REMOTE=/workspace/reasoning-on-manifold

case "${1:-}" in
setup)
  ssh "$POD" "cd $REMOTE && bash runpod_setup.sh"
  ;;
push)
  ssh "$POD" "mkdir -p $REMOTE/data $REMOTE/results/r0_entropy_ladder/R1-1.5B $REMOTE/results/r1_compression"
  rsync -rltz \
    ph0_pod_job.sh ph0_s2_extract_inert.py ph0_s3_curve.py \
    pt09_build_dpo_pairs.py pt10_train_dpo.py 04_extract_activations.py \
    29_r0_entropy_ladder.py 30_r1_compression.py \
    pyproject.toml runpod_setup.sh \
    src configs "$POD:$REMOTE/"
  rsync -rltz data/annotated_R1-1.5B.json data/annotated_STAR1-1.5B.json \
    data/chains_R1-1.5B.json data/tasks_final.json "$POD:$REMOTE/data/"
  rsync -rltz results/r0_entropy_ladder/R1-1.5B/sample.json \
    "$POD:$REMOTE/results/r0_entropy_ladder/R1-1.5B/"
  rsync -rltz results/r1_compression/report.json results/r1_compression/gate_tokenizer.json \
    "$POD:$REMOTE/results/r1_compression/"
  ;;
launch)
  ssh "$POD" "cd $REMOTE && rm -f PH0_DONE.marker PH0_STATUS && tmux new-session -d -s ph0 'bash ph0_pod_job.sh'"
  echo "launched tmux 'ph0' on $POD — use ./runpod_phase0.sh watch"
  ;;
status)
  ssh "$POD" "cd $REMOTE && echo STATUS=\$(cat PH0_STATUS 2>/dev/null || echo none) && tail -20 ph0.log 2>/dev/null"
  ;;
watch)
  ( while true; do
      ST=$(ssh "$POD" "cat $REMOTE/PH0_STATUS 2>/dev/null" || echo ssh-fail)
      case "$ST" in
        DONE|FAILED*) break ;;
      esac
      sleep 120
    done
    "$0" pull || true
    MSG="phase0 pod job: ${ST}"
    command -v osascript >/dev/null && osascript -e "display notification \"$MSG\" with title \"runpod_phase0\"" || echo "$MSG"
  ) >/dev/null 2>&1 &
  echo "watcher started (pid $!) — polls every 2 min, pulls + notifies on DONE/FAILED; never kills the pod"
  ;;
pull)
  mkdir -p results/safety_posttrain/ph0_s3 data/activations
  rsync -rltz "$POD:$REMOTE/results/safety_posttrain/ph0_s3/" results/safety_posttrain/ph0_s3/ || true
  rsync -rltz "$POD:$REMOTE/results/dpo_control_probe.json" "$POD:$REMOTE/results/probe_wall_seconds.txt" \
    "$POD:$REMOTE/results/ph0_f5_route.json" results/ 2>/dev/null || true
  rsync -rltz "$POD:$REMOTE/results/dpo_control_500.json" results/ 2>/dev/null || true
  rsync -rltz "$POD:$REMOTE/data/activations/STAR1-1.5B-6label" data/activations/ 2>/dev/null || true
  rsync -rltz "$POD:$REMOTE/ph0.log" "$POD:$REMOTE/PH0_STATUS" results/safety_posttrain/ph0_s3/ 2>/dev/null || true
  echo "pulled. Verify, then terminate the pod from the RunPod console (never on-pod)."
  ;;
*)
  grep '^#' "$0" | head -18
  ;;
esac
