#!/bin/bash
# E10.3 — DAS-1D cross-behaviour generalization (E10_DAS_PREREG.md Amendment 4).
# uncertainty-estimation + example-testing: main (3 layers, train+eval+controls+cis)
# then a width probe ONLY at the grounded layer picked by the SEALED rule
# (e10_pick_grounded_layer.py); no grounded layer => width skipped = sealed outcome.
#
# ~10-11 GPU-hours on a 4090 (~$4-6). No API credits. Locally built + inspected
# pairs.json files are pushed VERBATIM (the pod does not rebuild pairs — prereg).
#
# Kill discipline (Tony's 2026-07-13 authorization): NO on-pod self-kill. The Mac-side
# 'watch' subcommand safety-pulls on completion/crash and notifies; termination is done
# from the Mac AFTER a verified pull.
#
# Usage from the Mac:
#   POD=runpod ./runpod_e10_3.sh setup     # one-time deps on a fresh pod
#   POD=runpod ./runpod_e10_3.sh push      # code + data + E1 vectors + INSPECTED pairs
#   POD=runpod ./runpod_e10_3.sh launch    # full chain in tmux 'e10x'
#   POD=runpod ./runpod_e10_3.sh watch     # background watcher (pull+notify on done/crash)
#   POD=runpod ./runpod_e10_3.sh status    # tail progress
#   POD=runpod ./runpod_e10_3.sh pull      # rsync results/das back
set -euo pipefail
cd "$(dirname "$0")"
POD="${POD:-runpod}"
REMOTE=/workspace/reasoning-on-manifold
OUT=results/das/R1-1.5B
NPAIRS="${NPAIRS:-400}"
MARKER=E10_3_DONE.marker
LOG=e10_3.log

case "${1:-}" in
setup)
  ssh "$POD" "cd $REMOTE && bash runpod_setup.sh"
  ;;

push)
  ssh "$POD" "mkdir -p $REMOTE/data $REMOTE/src $REMOTE/configs \
    $REMOTE/$OUT/unc_main $REMOTE/$OUT/ex_main \
    $REMOTE/results/steering_vectors/R1-1.5B__E1_pooled"
  rsync -az --no-owner --no-group --no-perms \
    20_das_backtracking.py 21_das_width.py e10_pick_grounded_layer.py e10_3_chain.sh \
    runpod_setup.sh pyproject.toml "$POD:$REMOTE/"
  rsync -az --no-owner --no-group --no-perms src/ "$POD:$REMOTE/src/"
  rsync -az --no-owner --no-group --no-perms configs/ "$POD:$REMOTE/configs/" 2>/dev/null || true
  rsync -az --no-owner --no-group --no-perms data/annotated_R1-1.5B.json "$POD:$REMOTE/data/"
  rsync -az --no-owner --no-group --no-perms \
    results/steering_vectors/R1-1.5B__E1_pooled/uncertainty-estimation_single.npy \
    results/steering_vectors/R1-1.5B__E1_pooled/example-testing_single.npy \
    results/steering_vectors/R1-1.5B__E1_pooled/metadata.json \
    "$POD:$REMOTE/results/steering_vectors/R1-1.5B__E1_pooled/"
  # the locally built + inspected pairs, pushed verbatim (prereg Amendment 4)
  rsync -az --no-owner --no-group --no-perms \
    "$OUT/unc_main/pairs.json" "$POD:$REMOTE/$OUT/unc_main/"
  rsync -az --no-owner --no-group --no-perms \
    "$OUT/ex_main/pairs.json" "$POD:$REMOTE/$OUT/ex_main/"
  echo "pushed (code + annotated chains + unc/ex E1 vectors + inspected pairs)."
  ;;

launch)
  # e10_3_chain.sh reuses the pushed pairs.json (--stage train only rebuilds when missing).
  ssh "$POD" "cd $REMOTE && chmod +x e10_3_chain.sh && \
    tmux new-session -d -s e10x \"NPAIRS=$NPAIRS bash e10_3_chain.sh > $LOG 2>&1\""
  echo "launched E10.3 chain in tmux 'e10x' (ETA ~10-11 h on a 4090). Start the watcher:"
  echo "  POD=$POD ./runpod_e10_3.sh watch"
  ;;

watch)
  # Background watcher: success + error + process-death, safety-pull, notify.
  # On DONE: pull -> VERIFY artifacts -> TERMINATE the pod (creds in ~/.pod_creds,
  # line "runpod-e103 <podid> <apikey>") — kill-on-completion authorized by Tony
  # 2026-07-19 ("make sure the pod will be killed once the job is finished"); this
  # respects the 2026-07-13 rule (only completed runs are killed; a CRASHED pod is
  # left alive for diagnosis and only pulled + notified).
  # Start AFTER 'launch' (a not-yet-started tmux session would read as DEAD).
  W="$OUT/e10_3_watcher.sh"
  {
    echo '#!/bin/bash'
    echo "cd \"$(pwd)\""
    echo "POD=$POD; REMOTE=$REMOTE; OUT=$OUT; MARKER=$MARKER; LOG=$LOG"
    cat <<'EOS'
WLOG="$OUT/e10_3_watch.log"
verify_pull() {  # both mains parsed + 3 layers each + controls/cis present
  python3 - <<'PY'
import json, sys
for s in ("unc", "ex"):
    d = f"results/das/R1-1.5B/{s}_main"
    r = json.load(open(f"{d}/report.json"))
    assert len(r["layers"]) == 3, (s, "layers", list(r["layers"]))
    json.load(open(f"{d}/controls.json")); json.load(open(f"{d}/cis.json"))
print("verified")
PY
}
terminate_pod() {
  line=$(grep "^runpod-e103 " "$HOME/.pod_creds" 2>/dev/null)
  pid=$(echo "$line" | awk '{print $2}'); key=$(echo "$line" | awk '{print $3}')
  [ -n "$pid" ] && [ -n "$key" ] || { echo "$(date) NO CREDS - cannot terminate" >> "$WLOG"; return 1; }
  R=$(curl -s --max-time 30 "https://api.runpod.io/graphql?api_key=${key}" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${pid}\\\"})}\"}")
  echo "$(date) podTerminate -> $R" >> "$WLOG"
  echo "$R" | grep -q '"podTerminate"' && ! echo "$R" | grep -qi error
}
while true; do
  ST=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$POD" "cd $REMOTE 2>/dev/null && \
    if [ -f $MARKER ]; then echo DONE; \
    elif tmux has-session -t e10x 2>/dev/null; then echo RUNNING; \
    else echo DEAD; fi" 2>/dev/null || echo UNREACHABLE)
  case "$ST" in
    DONE)
      rsync -az --timeout=300 "$POD:$REMOTE/$OUT/" "$OUT/" || true
      rsync -az --timeout=120 "$POD:$REMOTE/$LOG" "$OUT/e10_3_pod.log" || true
      if verify_pull >> "$WLOG" 2>&1; then
        if terminate_pod; then
          osascript -e 'display notification "E10.3 complete - results pulled, verified, POD TERMINATED." with title "E10.3 DONE"' 2>/dev/null
          echo "$(date) DONE - pulled, verified, pod terminated" >> "$WLOG"
        else
          osascript -e 'display notification "E10.3 pulled+verified but TERMINATE FAILED - kill pod manually!" with title "E10.3 DONE (pod alive)"' 2>/dev/null
          echo "$(date) DONE - pulled+verified, TERMINATE FAILED" >> "$WLOG"
        fi
      else
        osascript -e 'display notification "E10.3 marker present but pull FAILED VERIFICATION - pod left running." with title "E10.3 VERIFY FAILED"' 2>/dev/null
        echo "$(date) DONE but verification failed - pod left running" >> "$WLOG"
      fi
      break ;;
    DEAD)
      rsync -az --timeout=300 "$POD:$REMOTE/$OUT/" "$OUT/" || true
      rsync -az --timeout=120 "$POD:$REMOTE/$LOG" "$OUT/e10_3_pod.log" || true
      osascript -e 'display notification "E10.3 tmux died WITHOUT marker - partials pulled, POD LEFT ALIVE for diagnosis." with title "E10.3 CRASHED"' 2>/dev/null
      echo "$(date) DEAD - partials pulled, pod left alive" >> "$WLOG"; break ;;
    *) echo "$(date) $ST" >> "$WLOG" ;;
  esac
  sleep 540
done
EOS
  } > "$W"
  chmod +x "$W"
  nohup "$W" > /dev/null 2>&1 &
  echo "watcher started (pid $!): ~9 min checks; on DONE pulls, verifies, TERMINATES pod;"
  echo "on CRASH pulls partials and leaves the pod alive. Log: $OUT/e10_3_watch.log"
  ;;

status)
  ssh "$POD" "cd $REMOTE && tail -8 $LOG 2>/dev/null | tr '\r' '\n' | tail -8; \
    ls $MARKER 2>/dev/null && echo DONE || echo RUNNING"
  ;;

pull)
  mkdir -p "$OUT"
  rsync -azv --no-owner --no-group --no-perms "$POD:$REMOTE/$OUT/" "$OUT/"
  echo "pulled -> $OUT/ . Inspect {unc,ex}_main/REPORT.md + controls.json + cis.json"
  echo "and {unc,ex}_width/REPORT.md (if grounded). Then terminate the pod from the Mac."
  ;;

*)
  echo "usage: [POD=host] [NPAIRS=400] $0 {setup|push|launch|watch|status|pull}"
  exit 1
  ;;
esac
