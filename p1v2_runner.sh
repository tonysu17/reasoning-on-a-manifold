#!/bin/bash
# P1v2 — re-run the DSR 3-judge annotation under SCHEMA v2 (post-human-anchor
# revision, results/safety/DSR_SCHEMA_V2_DRAFT.md). Chains already local from P0;
# API-only via the lab proxy (~$10-35, envelope $60, Tony-approved 2026-07-20).
# Fresh output paths: the annotator resumes by --out, so v1 artifacts are never
# touched and a mid-run death resumes from the v2 checkpoint without re-billing
# completed records.
set -u
cd "$(dirname "$0")"
MAIN="$(pwd)"
WT="$MAIN/../rom-safety-worktree"
LOG=results/safety/p1v2_runner.log
CHAINS="$MAIN/data/chains_gpt-oss-20b_p0.json"
log(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

log "=== P1v2 runner start (schema v2) ==="
[ -s "$CHAINS" ] || { log "chains file empty/missing — abort"; exit 1; }
source ~/.rom_proxy_env 2>/dev/null || { log "NO PROXY CREDS — blocked"; exit 1; }

# guard: the worktree prompt must actually be v2
SCHEMA=$(cd "$WT" && python3 -c "from src.safety.deliberation import DSR_SCHEMA_VERSION as v; print(v)") || exit 1
[ "$SCHEMA" = "v2" ] || { log "worktree schema is '$SCHEMA', not v2 — abort"; exit 1; }

cd "$WT"
python3 14b_annotate_dsr.py --chains "$CHAINS" \
  --out "$MAIN/results/safety/dsr_annotated_v2.json" \
  --agreement-out "$MAIN/results/safety/dsr_agreement_v2.json" >> "$MAIN/$LOG" 2>&1
rc=$?
cd "$MAIN"
if [ $rc -eq 0 ]; then
  touch results/safety/P1V2_READY_FOR_REVIEW.marker
  log "=== P1v2 DONE — dsr_agreement_v2.json ready (sealed gates apply per label) ==="
else
  touch results/safety/P1V2_FAILED.marker
  log "=== P1v2 FAILED (rc=$rc) — see log ==="
fi
exit $rc
