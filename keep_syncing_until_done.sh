#!/bin/bash
# STANDALONE sync — run this in YOUR OWN Terminal (independent of Claude Code, so
# it keeps running after Claude is closed). It pulls the E8 GENERATION output from
# the pod every few minutes and tells you exactly when generation is finished and
# fully synced, so you can STOP THE POD and run annotation locally (no GPU needed).
#
#   cd "/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold"
#   bash keep_syncing_until_done.sh
#
# When it prints "GEN DONE", stop the pod and run:  bash run_local_annotation.sh
#
# Exit status: 0 = generation complete + fully synced (clean).
#              1 = generation stalled / never completed (do NOT annotate yet —
#                  the pod likely crashed or was stopped early; investigate).
cd "/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold" || exit 1
REMOTE="runpod:/workspace/reasoning-on-manifold/results/eval/R1-1.5B__E1/"
LOCAL="results/eval/R1-1.5B__E1/"
EXPECTED=1650        # 50 vanilla + 4 behaviours × 8 arm-variants × 50 tasks
STALL_LIMIT=9        # consecutive syncs (×5min ≈ 45min) with no new chains ⇒ stalled
MAX_ITERS=200        # hard cap (~16h) so this can never hang forever
mkdir -p "$LOCAL"
count () { python3 -c "import json,os;p='$LOCAL/$1';print(len(json.load(open(p))) if os.path.exists(p) else 0)" 2>/dev/null; }

do_sync () {
  rsync -az --no-owner --no-group --no-perms \
    -e "ssh -o ConnectTimeout=30 -o ServerAliveInterval=8" "$REMOTE" "$LOCAL" 2>/dev/null
}

prev=-1; stall=0
for ((i=1; i<=MAX_ITERS; i++)); do
  do_sync
  g=$(count steering_results.json); g=${g:-0}
  echo "[$(date '+%H:%M')] synced — generated=$g / $EXPECTED"

  # ── clean completion: ALL chains local (count >= EXPECTED). The gen marker is
  #    only an early hint — it can sync before the final chains, so we require the
  #    full count (which itself proves generation produced everything). ──
  if [ "$g" -ge "$EXPECTED" ]; then
    do_sync   # one last pull to be certain the final chains + marker are local
    g=$(count steering_results.json)
    echo ""
    echo "=================================================================="
    echo "✅ GEN DONE + FULLY SYNCED ($g chains). SAFE TO STOP THE POD NOW."
    echo "   Then run annotation locally (laptop only, no GPU):"
    echo "       bash run_local_annotation.sh"
    echo "=================================================================="
    exit 0
  fi

  # ── stall detection: no new chains for STALL_LIMIT syncs ⇒ pod crashed/stopped ──
  if [ "$g" -le "$prev" ]; then stall=$((stall + 1)); else stall=0; fi
  prev="$g"
  if [ "$stall" -ge "$STALL_LIMIT" ]; then
    echo ""
    echo "⚠️  GENERATION STALLED — no new chains for ~$((STALL_LIMIT * 5)) min ($g/$EXPECTED)."
    echo "    The pod likely crashed or was stopped early. Do NOT annotate a partial run."
    echo "    Check the pod (gen.log) / restart generation, then re-run this script."
    exit 1
  fi
  sleep 300
done

echo "⚠️  hit the ${MAX_ITERS}-iteration cap without generation completing — investigate the pod."
exit 1
