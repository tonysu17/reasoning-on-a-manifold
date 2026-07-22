#!/bin/bash
# POD-SIDE: (A) Qwen3-span extraction  (B) DPO-control deps+smoke+throughput probe.
#
# MEMORY / DISK DISCIPLINE (root overlay is only 20 GB; GPU is 24 GB):
#   * HF_HOME + ALL outputs live on /workspace  (root stays empty)
#   * every stage is a SEPARATE process -> GPU fully freed between stages
#   * expandable_segments allocator (no fragmentation creep)
#   * pre-stage free-VRAM gate + per-stage peak-VRAM log
#   * conservative batch/token caps
# SELF-KILL: on DONE or FAILED, wait for SYNCED (Mac pulls) up to GRACE_H, then
#   terminate the pod via the RunPod API. Never idles indefinitely.
set -u

WORKDIR=/workspace/rom-jobs
export HF_HOME=/workspace/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
MIN_FREE_VRAM_MIB=${MIN_FREE_VRAM_MIB:-18000}   # R1-1.5B jobs need ~12; gate well above
GRACE_H=${GRACE_H:-8}
PROBE_TASKS=${PROBE_TASKS:-20}                  # throughput probe size
PROBE_K=${PROBE_K:-8}
PROBE_MAXTOK=${PROBE_MAXTOK:-2048}

cd "$WORKDIR" || { echo "FATAL: $WORKDIR missing"; exit 1; }
mkdir -p data results logs
log(){ echo "$(date -u '+%F %T') | $*"; }

kill_pod(){
  log "self-terminating pod"
  source /etc/rp_environment 2>/dev/null || true
  if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
      -H 'Content-Type: application/json' \
      -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}" || true
  else
    log "NO RunPod creds — stop the pod manually"
  fi; }

finish(){   # $1 = DONE|FAILED
  touch "JOBS_$1.marker"; log "=== JOBS $1 ==="
  log "waiting up to ${GRACE_H}h for SYNCED marker, then self-kill"
  local waited=0
  while [ ! -f SYNCED.marker ] && [ $waited -lt $((GRACE_H*3600)) ]; do
    sleep 60; waited=$((waited+60))
  done
  [ -f SYNCED.marker ] && log "SYNCED seen" || log "grace expired"
  kill_pod; exit 0; }

vram_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

stage(){ local n="$1"; shift
  [ -f "S_${n}.marker" ] && { log "stage $n: cached"; return 0; }
  local free; free=$(vram_free)
  log "stage $n: start (free VRAM ${free} MiB)"
  if [ "$free" -lt "$MIN_FREE_VRAM_MIB" ]; then
    log "stage $n: REFUSING — only ${free} MiB free (< ${MIN_FREE_VRAM_MIB})"; finish FAILED
  fi
  if "$@" >> "logs/${n}.log" 2>&1; then
    touch "S_${n}.marker"; log "stage $n: DONE"
  else
    log "stage $n: FAILED (see logs/${n}.log)"; finish FAILED
  fi; }

deps(){
  # torch 2.4 + sm_89: pin transformers <5 (5.x needs a newer torch DTensor API)
  # and a trl that still exposes DPOTrainer(processing_class=...) as pt10 expects.
  pip install -q -U "transformers>=4.44,<5" "trl>=0.12,<0.14" "peft>=0.12,<0.15" \
      "datasets<4" "accelerate<2" safetensors 2>&1 | tail -3
  python3 - <<'PY'
import inspect, torch, transformers
from trl import DPOConfig, DPOTrainer
assert "processing_class" in inspect.signature(DPOTrainer.__init__).parameters, \
    "trl too old: no processing_class (pt10 needs it)"
DPOConfig(output_dir="/tmp/_cfg", beta=0.1, max_length=128)   # validates arg names
import peft, datasets
print("OK torch", torch.__version__, "transformers", transformers.__version__,
      "trl", __import__("trl").__version__, "peft", peft.__version__)
PY
}

qwen_extract(){
  python3 04b_extract_annotator.py \
    --annotated data/annotated_R1-1.5B__qwen3-235b.json \
    --save-dir /workspace/rom-jobs/data/activations/R1-1.5B-qwenspans \
    --behaviours backtracking uncertainty-estimation example-testing adding-knowledge \
    --layers 12 16 --cache-dir /workspace/hf
  python3 - <<'PY'
import numpy as np, glob
fs = glob.glob("/workspace/rom-jobs/data/activations/R1-1.5B-qwenspans/*_layer*.npy")
assert len(fs) >= 8, f"only {len(fs)} matrices"
for f in fs: assert np.isfinite(np.load(f)).all(), f
print("qwenspans OK:", len(fs), "matrices")
PY
}

# Throughput probe: measure real s/task and pair-yield for control generation, so
# the full run can be costed BEFORE committing GPU-days. Generation in pt09 is
# sequential (k calls/task, max_new_tokens 2048) — this measures that honestly.
gen_probe(){
  # NB: minimal pod images have no /usr/bin/time — use SECONDS (bash builtin).
  local t0=$SECONDS
  python3 pt09_build_dpo_pairs.py \
    --arm control --generate --n "$PROBE_TASKS" --k "$PROBE_K" \
    --temperature 0.8 --max-new-tokens "$PROBE_MAXTOK" \
    --tasks data/tasks_final.json --out results/dpo_control_probe.json || return 1
  echo "PROBE_WALL_SECONDS=$((SECONDS - t0))" | tee results/probe_wall_seconds.txt
  python3 - <<PY
import json
d = json.load(open("results/dpo_control_probe.json"))
pairs = d if isinstance(d, list) else d.get("records", [])
n_tasks = $PROBE_TASKS; k = $PROBE_K
yield_rate = len(pairs)/n_tasks if n_tasks else 0
print(json.dumps({"probe_tasks": n_tasks, "k": k, "pairs": len(pairs),
                  "yield_per_task": round(yield_rate,3),
                  "tasks_needed_for_500": None if yield_rate==0 else round(500/yield_rate)},
                 indent=1))
PY
}

log "=== jobs start (GPU $(nvidia-smi --query-gpu=name --format=csv,noheader)) ==="
df -h /workspace / | tail -2
stage deps          deps
stage qwen          qwen_extract
stage gen_probe     gen_probe
finish DONE
