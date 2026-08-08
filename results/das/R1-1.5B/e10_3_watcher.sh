#!/bin/bash
cd "/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold"
POD=runpod-e103; REMOTE=/workspace/reasoning-on-manifold; OUT=results/das/R1-1.5B; MARKER=E10_3_DONE.marker; LOG=e10_3.log
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
