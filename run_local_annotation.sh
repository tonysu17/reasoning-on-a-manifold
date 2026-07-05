#!/bin/bash
# ============================================================================
# E8 LOCAL annotation + Δ_floor analysis — runs entirely on the laptop (API
# only, no GPU). Use this AFTER the pod has finished GENERATION and you have
# stopped the pod.
#
#   cd "/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold"
#   bash run_local_annotation.sh
#
# What it does, unattended:
#   1. caffeinate — keeps the laptop awake for the whole run (lid can stay open)
#   2. annotates all generated chains with Sonnet, SHARDED in parallel (fast)
#      — each shard reuses the PROVEN annotate_chains, saves after EVERY chain
#   3. self-healing: re-runs until every chain is complete (transient API
#      failures get retried automatically; resumable if you Ctrl-C and re-run)
#   4. merges shards -> annotated_steered.json (complete records only)
#   5. runs the Δ_floor analysis -> delta_floor_report.json
#
# Fully resumable: if interrupted (laptop sleeps, credits blip, Ctrl-C), just
# run it again — it skips everything already done and finishes the rest.
# ============================================================================
set -u
cd "/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold" || exit 1

# ── config (override on the command line, e.g.  SHARDS=4 bash run_local_annotation.sh) ──
OUT="${OUT:-results/eval/R1-1.5B__E1}"
ANNOTATOR="${ANNOTATOR:-eu.anthropic.claude-sonnet-4-5-20250929-v1:0}"   # Sonnet (per Tony)
SHARDS="${SHARDS:-6}"            # parallel annotators; ~1650 chains / 6 ≈ 4–5 h
MAX_PASSES="${MAX_PASSES:-8}"    # self-healing retry passes for stragglers
PY=python3
LOG="$OUT/local_annotate.log"

mkdir -p "$OUT"
echo "$(date '+%F %T') | E8 local annotation starting" | tee -a "$LOG"

# ── proxy creds ──
if [ ! -f "$HOME/.rom_proxy_env" ]; then
  echo "ERROR: ~/.rom_proxy_env not found (proxy URL + key). Cannot annotate." | tee -a "$LOG"; exit 1
fi
# shellcheck disable=SC1091
source "$HOME/.rom_proxy_env"

# ── preflight: generated chains present? ──
if [ ! -f "$OUT/steering_results.json" ]; then
  echo "ERROR: $OUT/steering_results.json missing — sync the pod's generation first." | tee -a "$LOG"; exit 1
fi
NCHAINS=$($PY -c "import json;print(len(json.load(open('$OUT/steering_results.json'))))" 2>/dev/null)
echo "  generated chains to annotate: $NCHAINS  (shards=$SHARDS, annotator=$ANNOTATOR)" | tee -a "$LOG"
if [ ! -f "$OUT/E8_gen_DONE.marker" ]; then
  echo "  NOTE: E8_gen_DONE.marker not present — generation may still be in flight." | tee -a "$LOG"
  echo "        It is safe to start now and re-run later to pick up any late chains," | tee -a "$LOG"
  echo "        but ideally wait until the pod has written E8_gen_DONE.marker." | tee -a "$LOG"
fi

# ── keep the laptop awake for the duration of THIS script ──
caffeinate -dimsu -w $$ &
echo "  caffeinate engaged (laptop will not sleep until this finishes)" | tee -a "$LOG"

run_all_shards () {
  local pids=()
  for i in $(seq 0 $((SHARDS - 1))); do
    $PY annotate_local.py --eval-dir "$OUT" --annotator "$ANNOTATOR" \
        --shards "$SHARDS" --shard-id "$i" >> "$OUT/local_shard${i}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
}

# ── annotate with self-healing retry until every chain is complete ──
for pass in $(seq 1 "$MAX_PASSES"); do
  echo "$(date '+%F %T') | annotation pass $pass/$MAX_PASSES ($SHARDS parallel shards)…" | tee -a "$LOG"
  run_all_shards
  $PY annotate_local.py --eval-dir "$OUT" --shards "$SHARDS" --merge | tee -a "$LOG"
  INC=$(cat "$OUT/_incomplete_count" 2>/dev/null || echo 99999)
  if [ "$INC" -eq 0 ]; then
    echo "$(date '+%F %T') | ✅ all chains annotated complete" | tee -a "$LOG"
    break
  fi
  echo "$(date '+%F %T') | $INC chains still incomplete — cooldown 30s then retry" | tee -a "$LOG"
  sleep 30
done

INC=$(cat "$OUT/_incomplete_count" 2>/dev/null || echo 99999)
if [ "$INC" -ne 0 ]; then
  echo "$(date '+%F %T') | WARNING: $INC chains never completed after $MAX_PASSES passes." | tee -a "$LOG"
  echo "  Analysis will run on the complete ones (those tasks drop from the affected pairs)." | tee -a "$LOG"
  echo "  Re-run this script later to try the stragglers again before trusting the numbers." | tee -a "$LOG"
fi
date > "$OUT/E8_annotate_DONE.marker"

# ── Δ_floor analysis (same config as the pod pipeline) ──
echo "$(date '+%F %T') | Δ_floor analysis…" | tee -a "$LOG"
$PY 08_steering_analysis.py --eval-dir "$OUT" \
    --alpha-star "backtracking=1.0,uncertainty-estimation=1.0,example-testing=1.0,adding-knowledge=1.0" \
    --arms single_direction manifold_k3 manifold_k5 \
    --n-resamples 10000 2>&1 | tee -a "$LOG"
date > "$OUT/E8_DONE.marker"

echo "" | tee -a "$LOG"
echo "==================================================================" | tee -a "$LOG"
echo "✅ E8 LOCAL ANNOTATION + ANALYSIS COMPLETE" | tee -a "$LOG"
echo "   report : $OUT/delta_floor_report.json" | tee -a "$LOG"
echo "   log    : $LOG" | tee -a "$LOG"
echo "==================================================================" | tee -a "$LOG"
