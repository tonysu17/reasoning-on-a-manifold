#!/usr/bin/env bash
# P5 WildGuard 8-row smoke — pod job wrapper, v1.1 (hash-bound into the
# manifest). DO NOT RUN until Tony authorizes the exact
# (runner_sha256, manifest sha256) pair AND an authorised manifest revision is
# issued — the runner refuses spend stages otherwise, whoever invokes this.
#
# v1.1: ONE cumulative job clock. JOB_START is written BEFORE apt/pip and every
# stage + every remaining `timeout` derives from it, so setup, staging,
# inference, and verification share the single 2,700 s budget. This bounds JOB
# runtime; the billed pod lifetime still ends only at console termination after
# the Mac watcher's verified pull (operator boundary — house rule: no on-pod
# self-kill).
#
# Installs use ONLY the hash-locked wheels in p5wg_requirements.lock (bound
# into the manifest). Usage (on pod, repo at /workspace/reasoning-on-manifold):
#   bash runpod_p5wg_smoke.sh <MANIFEST_SHA256>
set -euo pipefail

MANIFEST_SHA="${1:?usage: runpod_p5wg_smoke.sh <authorised manifest sha256>}"
REPO="${REPO:-/workspace/reasoning-on-manifold}"
MANIFEST="$REPO/results/p5_wildguard/P5WG_SMOKE_MANIFEST_2026-08-09.json"
OUT="$REPO/results/p5_wildguard/smoke"
MAX_S=2700   # mirrors manifest guards.max_job_duration_s

cd "$REPO"
mkdir -p "$OUT"
status() { echo "$(date -u +%FT%TZ) $1" | tee -a P5WG_STATUS; }

# ── single job clock: BEFORE any setup work ──────────────────────────────────
JOB_START=$(date +%s)
echo "$JOB_START" > "$OUT/JOB_START"
remain() {
  local left=$(( MAX_S - ( $(date +%s) - JOB_START ) ))
  if [ "$left" -le 0 ]; then
    status "FAILED: job budget exhausted"; touch "$OUT/SMOKE_FAILED.marker"; exit 3
  fi
  echo "$left"
}

status "SETUP under job clock: apt + locked wheels"
timeout "$(remain)" bash -c "apt-get update -qq && apt-get install -y -qq rsync tmux" > /dev/null
timeout "$(remain)" pip install -q --no-deps --require-hashes -r "$REPO/p5wg_requirements.lock"

status "PREFLIGHT (offline, zero-spend)"
timeout "$(remain)" python3 p5wg_smoke_runner.py --stage preflight \
  --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" --out-dir "$OUT"

status "STAGE_WEIGHTS (network; pinned revision, verified snapshot)"
timeout "$(remain)" python3 p5wg_smoke_runner.py --stage stage_weights --authorised \
  --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" --out-dir "$OUT"

status "RUN (offline; local snapshot, BF16-at-construction, batch 1)"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
timeout "$(remain)" python3 p5wg_smoke_runner.py --stage run --authorised \
  --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" --out-dir "$OUT"

status "VERIFY (chain + rows + diag linkage; no aggregation)"
timeout "$(remain)" python3 p5wg_smoke_runner.py --stage verify \
  --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" --out-dir "$OUT"

status "DONE — waiting for Mac watcher verified pull; terminate from console"
