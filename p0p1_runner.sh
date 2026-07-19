#!/bin/bash
# Mac-side autonomous chain: wait for P0 chains -> pull -> run P1 DSR annotation
# (3-judge panel via proxy) -> drop P1_READY. Tony's gold labels fold in later.
set -u
cd "$(dirname "$0")"
MAIN="$(pwd)"
WT="$MAIN/../rom-safety-worktree"
LOG=results/safety/p0p1_runner.log
mkdir -p results/safety data
log(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

log "=== P0->P1 runner start (waiting for P0 completion on runpod4) ==="
until ssh -o BatchMode=yes -o ConnectTimeout=20 runpod4 "[ -f /workspace/rom-p0/P0_DONE.marker ] || [ -f /workspace/rom-p0/P0_FAILED.marker ]" 2>/dev/null; do
  sleep 300
done
if ssh -o BatchMode=yes runpod4 "[ -f /workspace/rom-p0/P0_FAILED.marker ]" 2>/dev/null; then
  log "P0 FAILED on pod — see p0_full.log; NOT running P1"; touch results/safety/P0_FAILED.marker; exit 1
fi

log "P0 done — pulling chains + verification"
rsync -az runpod4:/workspace/rom-p0/data/chains_gpt-oss-20b_p0.json data/ 2>>"$LOG"
rsync -az runpod4:/workspace/rom-p0/results/safety/ results/safety/ 2>>"$LOG" || true
rsync -az runpod4:/workspace/rom-p0/p0_full.log results/safety/ 2>>"$LOG" || true
CHAINS="$MAIN/data/chains_gpt-oss-20b_p0.json"
[ -s "$CHAINS" ] || { log "chains file empty/missing — abort P1"; exit 1; }
touch results/safety/P0_READY_FOR_REVIEW.marker
log "P0 chains local ($(python3 -c "import json;print(len(json.load(open('$CHAINS'))))" 2>/dev/null) chains)"

# best-effort: free the GPU pod (P1 is API-only, no GPU needed)
ssh -o BatchMode=yes runpod4 "touch /workspace/rom-p0/P0_SYNCED.marker" 2>/dev/null && log "signalled A40 to self-terminate"

log "=== P1: DSR 3-judge annotation via proxy ==="
source ~/.rom_proxy_env 2>/dev/null || { log "NO PROXY CREDS — P1 blocked"; exit 1; }
cd "$WT"
python3 14b_annotate_dsr.py --chains "$CHAINS" \
  --out "$MAIN/results/safety/dsr_annotated.json" \
  --agreement-out "$MAIN/results/safety/dsr_agreement.json" >> "$MAIN/$LOG" 2>&1
rc=$?
cd "$MAIN"
if [ $rc -eq 0 ]; then
  touch results/safety/P1_READY_FOR_REVIEW.marker
  log "=== P1 DONE — dsr_agreement.json ready (LLM-LLM kappa + gates); human anchor pending Tony ==="
else
  log "=== P1 FAILED (rc=$rc) — see log ==="; touch results/safety/P1_FAILED.marker
fi
