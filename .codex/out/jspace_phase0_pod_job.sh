#!/usr/bin/env bash
# Network-volume-contained launcher for the authorised J-space Phase-0 benchmark.
# It never terminates the pod and never resumes or overwrites an output tree.
set -uo pipefail

ROOT=/workspace/jspace-phase0
PYTHON="$ROOT/.venv/bin/python"
CACHE=/workspace/hf/hub
LOG="$ROOT/JSPACE_PHASE0.log"
STATUS="$ROOT/JSPACE_PHASE0_STATUS"
PID_FILE="$ROOT/JSPACE_PHASE0_DRIVER_PID"
DONE="$ROOT/JSPACE_PHASE0_DONE.marker"
FAILED="$ROOT/JSPACE_PHASE0_FAILED.marker"
OUTPUT="$ROOT/results/jspace_r1_pilot/benchmark"

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

printf 'RUNNING:verification\n' > "$STATUS"
log "verifying complete network-volume bundle"
"$PYTHON" "$ROOT/jspace_phase0_bundle_verify.py" \
  --root "$ROOT" \
  --cache-dir "$CACHE" \
  --output "$ROOT/BUNDLE_VERIFICATION_LAUNCH.json" \
  >> "$LOG" 2>&1 || finish_failed verification "$?"

{
  printf 'UTC_START=%s\n' "$(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  "$PYTHON" --version
  "$PYTHON" -c 'import torch, transformers, huggingface_hub; print("torch=" + torch.__version__); print("transformers=" + transformers.__version__); print("huggingface_hub=" + huggingface_hub.__version__)'
  git -C "$ROOT/_external/jacobian-lens" rev-parse HEAD
  sha256sum \
    "$ROOT/jspace_pilot_preflight.py" \
    "$ROOT/jspace_pilot_benchmark.py" \
    "$ROOT/results/prereg/JSPACE_R1_BENCHMARK_MANIFEST_2026-08-10.json" \
    "$ROOT/results/prereg/JSPACE_R1_STEERING_PILOT_PREREG_2026-08-10.md"
} > "$ROOT/RUN_ENVIRONMENT.txt" 2>&1 || finish_failed environment_record "$?"

printf 'RUNNING:benchmark\n' > "$STATUS"
log "starting authorised five-prompt benchmark"
"$PYTHON" "$ROOT/jspace_pilot_benchmark.py" \
  --authorised \
  --jlens-checkout "$ROOT/_external/jacobian-lens" \
  --cache-dir "$CACHE" \
  --local-files-only \
  --device cuda:0 \
  >> "$LOG" 2>&1
benchmark_rc=$?

if test -d "$OUTPUT"; then
  (
    cd "$ROOT" || exit 1
    find results/jspace_r1_pilot/benchmark -type f -print0 \
      | sort -z \
      | xargs -0 sha256sum
  ) > "$ROOT/PHASE0_SHA256SUMS" 2>> "$LOG" || finish_failed checksums "$?"
fi

if test "$benchmark_rc" -ne 0; then
  finish_failed benchmark "$benchmark_rc"
fi

"$PYTHON" - <<'PY' >> "$LOG" 2>&1 || finish_failed final_gate "$?"
import json
from pathlib import Path

path = Path("results/jspace_r1_pilot/benchmark/benchmark_report.json")
report = json.loads(path.read_text())
if report.get("status") != "complete" or report.get("gate", {}).get("pass") is not True:
    raise SystemExit("controller report is not a completed passing gate")
print("validated passing controller report")
PY

printf 'DONE\n' > "$STATUS"
touch "$DONE"
log "terminal success"

