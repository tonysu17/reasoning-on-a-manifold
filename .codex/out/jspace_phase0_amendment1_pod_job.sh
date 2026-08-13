#!/usr/bin/env bash
# Corrective Phase-0 rerun sealed by JSPACE_R1_BENCHMARK_AMENDMENT_1_2026-08-10.md.
# It records the missing per-prompt resource fields and never overwrites run 1.
set -uo pipefail

ROOT=/workspace/jspace-phase0
PYTHON="$ROOT/.venv/bin/python"
CACHE=/workspace/hf/hub
MANIFEST="$ROOT/results/prereg/JSPACE_R1_BENCHMARK_MANIFEST_2026-08-10_AMENDMENT1.json"
LOG="$ROOT/JSPACE_PHASE0_A1.log"
STATUS="$ROOT/JSPACE_PHASE0_A1_STATUS"
PID_FILE="$ROOT/JSPACE_PHASE0_A1_DRIVER_PID"
DONE="$ROOT/JSPACE_PHASE0_A1_DONE.marker"
FAILED="$ROOT/JSPACE_PHASE0_A1_FAILED.marker"
OUTPUT="$ROOT/results/jspace_r1_pilot/benchmark_amendment1"

log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"
}

finish_failed() {
  local stage="$1"
  local code="${2:-1}"
  printf 'FAILED:%s:rc=%s\n' "$stage" "$code" > "$STATUS"
  touch "$FAILED"
  log "terminal failure: stage=$stage rc=$code"
  exit "$code"
}

cd "$ROOT" || exit 90
test ! -e "$OUTPUT" || finish_failed preexisting_output 91
test ! -e "$DONE" || finish_failed stale_done_marker 92
test ! -e "$FAILED" || finish_failed stale_failed_marker 93
printf '%s\n' "$$" > "$PID_FILE"
: > "$LOG"

export HF_HOME=/workspace/hf
export HF_HUB_CACHE="$CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME="$ROOT/.runtime-cache"

printf 'RUNNING:amended_preflight\n' > "$STATUS"
log "validating amended manifest and fixed inputs"
"$PYTHON" "$ROOT/jspace_pilot_preflight.py" \
  --manifest "$MANIFEST" \
  --jlens-checkout "$ROOT/_external/jacobian-lens" \
  > "$ROOT/JSPACE_PHASE0_A1_PREFLIGHT.json" 2>> "$LOG" \
  || finish_failed amended_preflight "$?"

{
  printf 'UTC_START=%s\n' "$(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  "$PYTHON" --version
  "$PYTHON" -c 'import torch, transformers, huggingface_hub; print("torch=" + torch.__version__); print("transformers=" + transformers.__version__); print("huggingface_hub=" + huggingface_hub.__version__)'
  git -C "$ROOT/_external/jacobian-lens" rev-parse HEAD
  sha256sum \
    "$ROOT/jspace_pilot_preflight.py" \
    "$ROOT/jspace_pilot_benchmark.py" \
    "$MANIFEST" \
    "$ROOT/results/prereg/JSPACE_R1_BENCHMARK_AMENDMENT_1_2026-08-10.md"
} > "$ROOT/JSPACE_PHASE0_A1_RUN_ENVIRONMENT.txt" 2>&1 \
  || finish_failed environment_record "$?"

printf 'RUNNING:amended_benchmark\n' > "$STATUS"
log "starting authorised amended five-prompt benchmark"
"$PYTHON" "$ROOT/jspace_pilot_benchmark.py" \
  --authorised \
  --manifest "$MANIFEST" \
  --jlens-checkout "$ROOT/_external/jacobian-lens" \
  --cache-dir "$CACHE" \
  --local-files-only \
  --device cuda:0 \
  >> "$LOG" 2>&1
benchmark_rc=$?

if test -d "$OUTPUT"; then
  (
    cd "$ROOT" || exit 1
    find results/jspace_r1_pilot/benchmark_amendment1 -type f -print0 \
      | sort -z \
      | xargs -0 sha256sum
  ) > "$ROOT/PHASE0_A1_SHA256SUMS" 2>> "$LOG" || finish_failed checksums "$?"
fi

if test "$benchmark_rc" -ne 0; then
  finish_failed amended_benchmark "$benchmark_rc"
fi

"$PYTHON" - <<'PY' >> "$LOG" 2>&1 || finish_failed final_gate "$?"
import json
from pathlib import Path

path = Path("results/jspace_r1_pilot/benchmark_amendment1/benchmark_report.json")
report = json.loads(path.read_text())
checks = report.get("gate", {}).get("checks", {})
if report.get("status") != "complete" or report.get("gate", {}).get("pass") is not True:
    raise SystemExit("amended controller report is not a completed passing gate")
if checks.get("per_prompt_resource_metrics_recorded") is not True:
    raise SystemExit("amended resource-field gate is absent or false")
print("validated passing amended controller report")
PY

printf 'DONE\n' > "$STATUS"
touch "$DONE"
log "terminal success"

