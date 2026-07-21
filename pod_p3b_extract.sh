#!/bin/bash
# POD-SIDE P3b — gpt-oss-20b forgery injection-span extraction (S3 probe input).
# Same skeleton + memory discipline as pod_p2_extract.sh. Needs ~48 GB VRAM.
# Launch: setsid nohup bash pod_p3b_extract.sh > pod_p3b.log 2>&1 < /dev/null &
set -u

WORKDIR=/workspace/rom-p3b
export HF_HOME=/workspace/hf
MIN_TOTAL_VRAM_MIB="${MIN_TOTAL_VRAM_MIB:-44000}"
GLOBAL_H=6; START=$(date +%s)

cd "$WORKDIR" || { echo "FATAL: $WORKDIR missing"; exit 1; }
mkdir -p results/safety/p3_forgery
log(){ echo "$(date -u '+%F %T') | $*"; }
gc(){ [ $(( $(date +%s) - START )) -gt $(( GLOBAL_H * 3600 )) ] && { log "WATCHDOG"; touch P3B_FAILED.marker; exit 1; }; }

stage(){ local n="$1"; shift
  [ -f "P3B_DONE_${n}.marker" ] && { log "stage ${n}: cached"; return 0; }
  log "stage ${n}: start"; gc
  if "$@" >> "p3bstage_${n}.log" 2>&1; then touch "P3B_DONE_${n}.marker"; log "stage ${n}: DONE"
  else log "stage ${n}: FAILED (see p3bstage_${n}.log)"; touch P3B_FAILED.marker; exit 1; fi; }

vram_check(){
  local total; total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  log "GPU total VRAM: ${total} MiB (need >= ${MIN_TOTAL_VRAM_MIB})"
  [ "$total" -ge "$MIN_TOTAL_VRAM_MIB" ] || { log "REFUSING: card too small"; return 1; }; }

cuda_smoke(){ python3 - <<'PYEOF'
import torch
assert torch.cuda.is_available()
x = torch.randn(64, 64, device="cuda"); (x @ x).sum().item()
print("torch", torch.__version__, "on", torch.cuda.get_device_name(0), "OK")
PYEOF
}

deps(){
  if ! cuda_smoke; then
    log "torch cannot drive this GPU — upgrading to cu128"
    pip install -q -U torch --index-url https://download.pytorch.org/whl/cu128
    cuda_smoke || return 1
  fi
  pip install -q -U "transformers>=4.55.1,<5" accelerate safetensors scikit-learn && \
  pip uninstall -y -q torchvision torchaudio 2>/dev/null; \
  python3 -c "import transformers, sklearn; import transformers.models.gpt_oss.modeling_gpt_oss; print('transformers', transformers.__version__, '+ gpt-oss OK')"; }

verify(){ python3 - <<'PYEOF'
import json, sys
import numpy as np
from pathlib import Path
out = Path("results/safety/p3_forgery/activations")
rep = json.loads((out / "P3B_EXTRACTION_REPORT.json").read_text())
ok = rep["n_rows"] >= 300
for f in out.glob("acts_layer*.npy"):
    if not np.isfinite(np.load(f)).all():
        print("FAIL: non-finite in", f.name); ok = False
print(json.dumps(rep["by_variant_style"]))
sys.exit(0 if ok else 1)
PYEOF
}

log "=== P3b extraction start ==="
stage vram    vram_check
stage deps    deps
stage extract python3 p3b_extract_forgery.py --root results/safety/p3_forgery --layers 6 11 12 18 --cache-dir /workspace/hf
stage assemble python3 p3b_extract_forgery.py --root results/safety/p3_forgery --assemble
stage verify  verify
touch P3B_DONE.marker
log "=== P3b COMPLETE — results/safety/p3_forgery/activations ready ==="
