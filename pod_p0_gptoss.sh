#!/bin/bash
# POD-SIDE P0 — gpt-oss-20b chain generation for the H1 pilot (plan: ~100 chains,
# ~40 harmful / ~40 benign / ~20 capability-control, ONE effort level = medium).
# Stage/marker/deadman skeleton copied from pod_recover.sh (this repo's ops
# playbook: watchdog + per-stage markers + self-terminate).
#
# Launch:
#   setsid nohup bash pod_p0_gptoss.sh > pod_p0.log 2>&1 < /dev/null &
#
# The operator MUST drop the real red-team prompt JSON on the secured volume and
# point STIMULI_JSON at it (StrongREJECT harmful + XSTest benign, schema per
# src/safety/stimuli.py). Real harmful content is NEVER bundled in the repo, so a
# run without STIMULI_JSON fails loudly rather than silently using placeholders.
set -u

WORKDIR=/workspace/rom-p0
export HF_HOME=/workspace/hf
STIMULI_JSON="${STIMULI_JSON:-}"          # REQUIRED: harmful+benign prompt JSON
CAPABILITY_JSON="${CAPABILITY_JSON:-builtin}"   # optional external difficulty anchor
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-16000}"   # gpt-oss-20b MXFP4 ~13-16 GB
GRACE_H=4; GLOBAL_H=6; START=$(date +%s)

cd "$WORKDIR" || { echo "FATAL: $WORKDIR missing"; exit 1; }
mkdir -p results/safety data
log(){ echo "$(date -u '+%F %T') | $*"; }

kill_pod(){ log "KILL"; source /etc/rp_environment 2>/dev/null || true
  if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H "Content-Type: application/json" \
      -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}"; sleep 60
  else log "no RunPod creds — cannot self-terminate; stop the pod manually"; fi; }

gc(){ [ $(( $(date +%s) - START )) -gt $(( GLOBAL_H * 3600 )) ] && { log "WATCHDOG"; kill_pod; }; }

stage(){ local n="$1"; shift
  [ -f "P0_DONE_${n}.marker" ] && { log "stage ${n}: cached"; return 0; }
  log "stage ${n}: start"
  if "$@" >> "p0stage_${n}.log" 2>&1; then touch "P0_DONE_${n}.marker"; log "stage ${n}: DONE"
  else touch "P0_FAILED_${n}.marker"; log "stage ${n}: FAILED"; return 1; fi; }

log "=== P0 start ($(grep -oE 'RUNPOD_POD_ID=\S+' /etc/rp_environment 2>/dev/null || echo local)) ==="

# ── 0. Assert free VRAM up front (fail loudly < MIN_FREE_VRAM_MIB) ────────────
if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "FATAL: nvidia-smi not found — no GPU. gpt-oss-20b needs a CUDA GPU (>=${MIN_FREE_VRAM_MIB} MiB free)."
  kill_pod; exit 20
fi
FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | tail -1)
log "free VRAM: ${FREE_VRAM} MiB (require >= ${MIN_FREE_VRAM_MIB} MiB)"
if [ -z "$FREE_VRAM" ] || [ "$FREE_VRAM" -lt "$MIN_FREE_VRAM_MIB" ]; then
  log "FATAL: only ${FREE_VRAM} MiB free VRAM; gpt-oss-20b needs ~13-16 GB in MXFP4 "
  log "       (>= ${MIN_FREE_VRAM_MIB} MiB required). Provision an A6000/A40 (48 GB safe tier). Aborting."
  kill_pod; exit 21
fi

# ── 0b. Require the real stimuli JSON (no silent placeholder science) ─────────
if [ -z "$STIMULI_JSON" ] || [ ! -f "$STIMULI_JSON" ]; then
  log "FATAL: STIMULI_JSON unset or missing ('$STIMULI_JSON'). Place the real "
  log "       StrongREJECT+XSTest prompt JSON on the volume and export STIMULI_JSON=/path.json"
  kill_pod; exit 22
fi

# ── 1. Deps (gpt-oss needs transformers>=4.55.1 + MXFP4 kernels) ──────────────
stage deps bash -c '
  pip install -q -U "transformers>=4.55.1" accelerate "torch>=2.4" safetensors && \
  pip install -q -U kernels triton 2>/dev/null || true'   # MXFP4 kernels: best-effort
gc

# ── 2. Generate + verify (script self-verifies and exits nonzero on hard fail) ─
stage generate python3 p0_generate_gptoss_chains.py \
  --stimuli "$STIMULI_JSON" --capability "$CAPABILITY_JSON" \
  --reasoning-effort "$REASONING_EFFORT" --max-new-tokens "$MAX_NEW_TOKENS" \
  --chains-out data/chains_gpt-oss-20b_p0.json \
  --provenance-out results/safety/p0_provenance.json \
  --verification-out results/safety/p0_verification.json
gc

if [ -f P0_DONE_generate.marker ]; then log "P0 generate+verify: DONE"
else log "P0 generate+verify: FAILED — inspect p0stage_generate.log"; fi

touch P0_COMPLETE.marker
log "P0 stages: $(ls P0_DONE_*.marker 2>/dev/null | wc -l) DONE, $(ls P0_FAILED_*.marker 2>/dev/null | wc -l) FAILED"
log "PULL THESE BEFORE THE POD DIES:"
log "  data/chains_gpt-oss-20b_p0.json   results/safety/p0_verification.json   results/safety/p0_provenance.json"

# ── 3. Deadman: wait for the puller to signal, then self-terminate ────────────
DEADLINE=$(( $(date +%s) + GRACE_H * 3600 ))
until [ -f P0_SYNCED.marker ]; do
  [ $(date +%s) -gt $DEADLINE ] && { log "deadman elapsed"; break; }
  gc; sleep 120
done
kill_pod
