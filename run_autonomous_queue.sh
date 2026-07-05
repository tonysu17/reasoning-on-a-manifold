#!/bin/bash
# =====================================================================
# Autonomous pipeline queue (2026-06-14).
# Waits for the running Gate-0.1 Phase 5 (05_pca) to finish, then runs the
# remaining Gate-0.1 downstream stages and writes a status digest + a
# "ready for numeral-lift" marker for the ch07 thesis update.
#
# CLASH-SAFE vs a concurrent session:
#   - skips a stage whose output dir already exists (someone did it);
#   - if a stage's script is already running elsewhere, WAITS for it rather
#     than starting a second copy.
# It does NOT modify the thesis (the numeral-lift needs human judgment) and
# does NOT run cluster/GPU work (R2.2 per-annotator extraction, Phase 7).
#
# Monitor:  tail -f logs/autonomous_queue.log   |   cat results/autonomous_queue_status.md
# Stop:     kill "$(cat results/.autonomous_queue.lock)"   (then rm that file)
# =====================================================================
set -u
ROOT="/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold"
cd "$ROOT" || exit 1
PY="/Users/tonysu/.pyenv/versions/3.13.12/bin/python3"
M="R1-1.5B"
LOG="logs/autonomous_queue.log"
STATUS="results/autonomous_queue_status.md"
LOCK="results/.autonomous_queue.lock"
MAXWAIT=28800          # 8 h cap on any single wait loop
mkdir -p logs results

# ---- single-instance lock ----
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[queue] another instance is running (PID $(cat "$LOCK")); exiting"; exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
status(){ printf '%s\n' "$*" >> "$STATUS"; }

wait_proc_gone(){    # $1 pgrep-pattern  $2 label
  local waited=0
  while pgrep -f "$1" >/dev/null 2>&1; do
    log "  ... waiting for $2 (running elsewhere)"; sleep 90; waited=$((waited+90))
    [ "$waited" -ge "$MAXWAIT" ] && { log "  TIMEOUT waiting for $2"; return 1; }
  done
  return 0
}

run_stage(){         # $1 label  $2 output-dir  $3 pgrep-pattern  $4... command
  local label="$1" outdir="$2" pat="$3"; shift 3
  if [ -d "$outdir" ] && [ -n "$(ls -A "$outdir" 2>/dev/null)" ]; then
    log "SKIP $label — $outdir already present"; status "- SKIP **$label** (output already present)"; return 0
  fi
  if pgrep -f "$pat" >/dev/null 2>&1; then
    log "$label already running in another session — waiting"; wait_proc_gone "$pat" "$label"
    if [ -d "$outdir" ] && [ -n "$(ls -A "$outdir" 2>/dev/null)" ]; then
      log "SKIP $label — produced by other session"; status "- SKIP **$label** (produced by other session)"; return 0
    fi
  fi
  log "RUN  $label"
  "$@" >> "logs/queue_${label}.log" 2>&1
  local rc=$?
  log "DONE $label (exit $rc)"
  status "- RAN **$label** (exit $rc) -> $outdir"
  return $rc
}

# ---- run ----
log "=== autonomous queue START (PID $$) ==="
: > "$STATUS"
status "# Autonomous queue status"
status ""
status "Started \`$(date)\`. Driver PID \`$$\`. Live log: \`$LOG\`."
status "Stop with: \`kill $$\` then \`rm $LOCK\`."
status ""
status "## Plan"
status "Wait for Gate-0.1 Phase 5 (05_pca) -> run downstream (05c, triangulation, 05d, 05b) -> write numeral-lift digest. Clash-safe; thesis untouched."
status ""
status "## Progress"

# Phase A — wait for the running Phase-5 null to finish (it may be restarted; keep waiting)
log "Phase A: waiting for 05_pca (Gate-0.1 Phase 5, null) to finish"
status "- WAIT for 05_pca (Phase 5) ..."
wait_proc_gone "05_pca_analysis" "05_pca (Phase 5)"
log "Phase A done: 05_pca no longer running"
status "- Phase 5 (05_pca) finished or absent — proceeding"

# Phase B — Gate-0.1 downstream, guarded
log "Phase B: Gate-0.1 downstream stages"
run_stage 05c_probes      "results/cross_layer/$M"   "05c_cross_layer_probing"     "$PY" 05c_cross_layer_probing.py --model-short "$M"
run_stage triangulation   "results/triangulation/$M" "compute_layer_triangulation" "$PY" compute_layer_triangulation.py --model-short "$M"
run_stage 05d_clustering  "results/clustering/$M"    "05d_subtype_clustering"      "$PY" 05d_subtype_clustering.py --model-short "$M"
run_stage 05b_geometry    "results/geometric/$M"     "05b_geometric_diagnostics"   "$PY" 05b_geometric_diagnostics.py --model-short "$M" --layers 11 14 17 20 27 --n-resamples 500

# Phase C — finalise + numeral-lift digest (thesis untouched)
log "Phase C: status digest"
status ""
status "## Gate-0.1 outputs present"
for d in pca cross_layer triangulation clustering geometric; do
  if [ -d "results/$d/$M" ] && [ -n "$(ls -A "results/$d/$M" 2>/dev/null)" ]; then
    status "- \`results/$d/$M\` — present"
  else
    status "- \`results/$d/$M\` — MISSING (stage may have failed; see logs/queue_*.log)"
  fi
done
status ""
status "## READY FOR NUMERAL-LIFT"
status "Gate-0.1 is complete. The ch07 \`\\gateZero\` placeholders + the \`\\todoT\` numeral-lift can now be filled from:"
status "- \`results/robustness/$M/geometry_robustness_summary.md\` (Tier-0 intrinsic dim + curvature, already final)"
status "- \`results/geometric/$M/\` (05b per-layer battery), \`results/triangulation/$M/\` (layer candidates), \`results/cross_layer/$M/\` (probe accuracies)"
status "Do this with human review — the embargo lift is a deliberate step (STYLE.md §5)."
status ""
status "## NOT done by this queue (need cluster / your go)"
status "- R2.2: per-annotator geometry replication (needs per-annotator activation extraction, GPU)."
status "- Phase 7: steering evaluation (cluster + re-annotation)."
status "- 05b nulls were run at 500 resamples; bump (e.g. 2500, matching the Phase-5 null) for final significance if wanted."

log "=== autonomous queue COMPLETE ==="
status ""
status "**Queue complete \`$(date)\`.**"
