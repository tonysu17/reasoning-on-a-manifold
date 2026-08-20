#!/bin/bash
# PT-B1 pod-side job: env -> preflight -> six arms (train -> gate -> generate).
# Launched detached by runpod_ptb1.sh. Markers:
#   PTB1_JOB_RUNNING / PTB1_ALL_DONE.marker / PTB1_JOB_FAILED.marker
# Fail-closed per arm; a failed arm stops the job (sealed stop conditions).
set -u
cd "$(dirname "$0")"
LOG="ptb1_job.log"
exec >> "$LOG" 2>&1

echo "=== PT-B1 job start: $(date -u +%FT%TZ) ==="
touch PTB1_JOB_RUNNING
rm -f PTB1_ALL_DONE.marker PTB1_JOB_FAILED.marker

fail(){ echo "JOB FAILED: $1 ($(date -u +%FT%TZ))"; touch PTB1_JOB_FAILED.marker; rm -f PTB1_JOB_RUNNING; exit 1; }

PY="${PYBIN:-python3}"
# Pin the FULL torch family, not just transformers. RunPod PyTorch images ship
# torch 2.4.1, on which transformers 5.15.0 dies (its tensor-parallel module
# dereferences torch behind a version guard). Upgrading torch alone then leaves
# torchvision/torchaudio built against the old ABI, whose C++ ops fail to
# resolve and surface as a MISLEADING "Could not import BloomPreTrainedModel"
# from transformers' lazy loader. These versions reproduce the committed 7B pod
# environment (pair7b/provenance/PIP_FREEZE.txt).
$PY -m pip install --no-cache-dir -q \
    "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124 || fail "pip install torch family"
$PY -m pip install --no-cache-dir -q \
    "transformers==5.15.0" "peft==0.20.0" "accelerate==1.14.0" \
    sentencepiece protobuf || fail "pip install"
$PY - <<'EOF' || exit 1
import torch, transformers, peft
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__, "peft", peft.__version__)
EOF
$PY -m pip freeze > results/ptb1/provenance/PIP_FREEZE.txt 2>/dev/null || { mkdir -p results/ptb1/provenance && $PY -m pip freeze > results/ptb1/provenance/PIP_FREEZE.txt; }
nvidia-smi > results/ptb1/provenance/NVIDIA_SMI.txt 2>&1 || true

echo "--- preflight ---"
$PY -u ptb1_pod_arm.py preflight || fail "preflight"

for ARM in safety1000-s42 safety1000-s43 safety1000-s44 \
           control1000-s42 control1000-s43 control1000-s44; do
  echo "--- arm ${ARM}: $(date -u +%FT%TZ) ---"
  $PY -u ptb1_pod_arm.py arm "$ARM" || fail "arm ${ARM}"
done

# Result manifest (hashes of everything the pull must verify).
( cd results/ptb1 && find . -type f ! -name remote_manifest.txt -print0 \
    | sort -z | xargs -0 sha256sum > remote_manifest.txt )
echo "=== PT-B1 job COMPLETE: $(date -u +%FT%TZ) ==="
touch PTB1_ALL_DONE.marker
rm -f PTB1_JOB_RUNNING
