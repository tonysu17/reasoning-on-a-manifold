#!/bin/bash
# POD-SIDE P2 — gpt-oss-20b DSR span-activation extraction (H1 input).
# Stage/marker/watchdog skeleton per pod_p0_gptoss.sh (this repo's ops playbook).
#
# HARD REQUIREMENT: ~48 GB VRAM. The verified-correct gpt-oss loader dequantizes
# MXFP4 experts to bf16 (~42 GB weights; Ada/Ampere have no native MXFP4 kernel
# and expert offload produces garbage — see src/chain_gen.py). A 24 GB card
# CANNOT run this; the VRAM stage below refuses loudly instead of OOMing slowly.
#
# Launch:
#   setsid nohup bash pod_p2_extract.sh > pod_p2.log 2>&1 < /dev/null &
set -u

WORKDIR=/workspace/rom-p2
export HF_HOME=/workspace/hf          # root disk is ~20 GB — cache must not live there
MIN_TOTAL_VRAM_MIB="${MIN_TOTAL_VRAM_MIB:-44000}"
GLOBAL_H=6; START=$(date +%s)

cd "$WORKDIR" || { echo "FATAL: $WORKDIR missing"; exit 1; }
mkdir -p results/safety data
log(){ echo "$(date -u '+%F %T') | $*"; }

gc(){ [ $(( $(date +%s) - START )) -gt $(( GLOBAL_H * 3600 )) ] && { log "WATCHDOG: ${GLOBAL_H}h exceeded"; touch P2_FAILED.marker; exit 1; }; }

stage(){ local n="$1"; shift
  [ -f "P2_DONE_${n}.marker" ] && { log "stage ${n}: cached"; return 0; }
  log "stage ${n}: start"; gc
  if "$@" >> "p2stage_${n}.log" 2>&1; then touch "P2_DONE_${n}.marker"; log "stage ${n}: DONE"
  else log "stage ${n}: FAILED (see p2stage_${n}.log)"; touch P2_FAILED.marker; exit 1; fi; }

vram_check(){
  local total
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  log "GPU total VRAM: ${total} MiB (need >= ${MIN_TOTAL_VRAM_MIB})"
  [ "$total" -ge "$MIN_TOTAL_VRAM_MIB" ] || { log "REFUSING: card too small for dequantized gpt-oss-20b (~42 GB weights)"; return 1; }; }

cuda_smoke(){ python3 - <<'EOF'
import torch
assert torch.cuda.is_available(), "no CUDA device"
x = torch.randn(64, 64, device="cuda")
(x @ x).sum().item()
print("torch", torch.__version__, "on", torch.cuda.get_device_name(0), "OK")
EOF
}

deps(){
  # Blackwell (sm_120) pods ship with torch builds that predate the arch —
  # the first kernel launch dies with "no kernel image". Smoke-test, upgrade
  # to a cu128 build only if needed, then smoke again (hard fail if still bad).
  if ! cuda_smoke; then
    log "torch cannot drive this GPU — upgrading to cu128 build"
    pip install -q -U torch --index-url https://download.pytorch.org/whl/cu128
    cuda_smoke || return 1
  fi
  # transformers pinned <5: the pipeline (Mxfp4Config dequantize path, P0/P1)
  # was verified on the 4.55 line; unpinned resolves to 5.x today. sklearn is
  # pulled in transitively by src.safety's package imports.
  pip install -q -U "transformers>=4.55.1,<5" accelerate safetensors scikit-learn && \
  pip uninstall -y -q torchvision torchaudio 2>/dev/null; \
  python3 -c "import transformers, sklearn; import transformers.models.gpt_oss.modeling_gpt_oss; print('transformers', transformers.__version__, '+ gpt-oss modeling OK')"
}

verify(){ python3 - <<'EOF'
import json, sys
import numpy as np
from pathlib import Path
out = Path("results/safety/p2_activations")
rep = json.loads((out / "P2_EXTRACTION_REPORT.json").read_text())
rows = rep["rows_per_label"]
ok = True
for lab in ("harm_recognition", "spec_citation", "decision", "__generic__"):
    if rows.get(lab, 0) < 10:
        print(f"FAIL: {lab} has only {rows.get(lab, 0)} rows"); ok = False
for f in out.glob("*_layer*.npy"):
    m = np.load(f)
    if not np.isfinite(m).all():
        print(f"FAIL: non-finite values in {f.name}"); ok = False
vram = json.loads((out / "vram_log.json").read_text())
peak = max((v["peak_vram_mib"] for v in vram), default=0)
print(f"peak VRAM across chains: {peak} MiB over {len(vram)} forwards")
print(json.dumps(rows))
sys.exit(0 if ok else 1)
EOF
}

log "=== P2 extraction start ==="
stage vram    vram_check
stage deps    deps
stage extract python3 p2_extract_dsr.py --chains data/dsr_annotated_v2.json \
                --out-root results/safety/p2_activations --cache-dir /workspace/hf
stage assemble python3 p2_extract_dsr.py --chains data/dsr_annotated_v2.json \
                --out-root results/safety/p2_activations --assemble
stage verify  verify
touch P2_DONE.marker
log "=== P2 COMPLETE — results/safety/p2_activations ready for pull ==="
