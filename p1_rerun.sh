#!/bin/bash
# Re-run P1 DSR 3-judge annotation with CORRECTED proxy model IDs (the pod is already
# terminated, so this runs the annotation step standalone on the locally-pulled chains).
set -u
MAIN="/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold"
WT="/Users/tonysu/Documents/Reasoning on a Manifold/rom-safety-worktree"
CHAINS="$MAIN/data/chains_gpt-oss-20b_p0.json"
LOG="$MAIN/results/safety/p1_rerun.log"
log(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

log "=== P1 re-run: DSR 3-judge annotation (corrected IDs) on $(python3 -c "import json;print(len(json.load(open('$CHAINS'))))" 2>/dev/null) chains ==="
source ~/.rom_proxy_env 2>/dev/null || { log "NO PROXY CREDS"; touch "$MAIN/results/safety/P1_FAILED.marker"; exit 1; }
cd "$WT" || { log "worktree missing"; touch "$MAIN/results/safety/P1_FAILED.marker"; exit 1; }
python3 14b_annotate_dsr.py --chains "$CHAINS" \
  --judges "Sonnet-4.5:eu.anthropic.claude-sonnet-4-5-20250929-v1:0" \
           "Qwen3-235B:qwen.qwen3-235b-a22b-2507-v1:0" \
           "Nova-Pro:amazon.nova-pro-v1:0" \
  --out "$MAIN/results/safety/dsr_annotated.json" \
  --agreement-out "$MAIN/results/safety/dsr_agreement.json" >> "$LOG" 2>&1
rc=$?
if [ $rc -eq 0 ]; then touch "$MAIN/results/safety/P1_READY_FOR_REVIEW.marker"; log "=== P1 DONE — dsr_agreement.json ready ==="
else touch "$MAIN/results/safety/P1_FAILED.marker"; log "=== P1 FAILED (rc=$rc) ==="; fi
