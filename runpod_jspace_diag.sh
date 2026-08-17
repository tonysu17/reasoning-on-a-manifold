#!/usr/bin/env bash
# J-space diagnostics D2-D5 on a RunPod pod.
# Sealed protocol JSPACE_R1_DIAGNOSTIC_PROTOCOL_2026-08-16.md (752dee7) + AMENDMENT 1.
#
# PREREQUISITES on the pod (see runbook block at the bottom of this file):
#   /workspace/artifacts/merged.fp32.pt              (255 MB, from the validated A5 bundle)
#   /workspace/artifacts/external_readout_arrays.npz (1.2 MB, same bundle)
#   HF_TOKEN exported (gemma-3-1b-pt is a gated repo)
#
# Sealed execution order: D2 -> D3 -> D4 -> D5. D2's outcome gates how D3 may
# be READ (not whether it runs); the runner records the dependency either way.
set -uo pipefail

REPO_URL="${REPO_URL:-https://github.com/tonysu17/reasoning-on-manifold.git}"
BRANCH="${BRANCH:-codex/phase0-support}"
WORK=/workspace
REPO=$WORK/reasoning-on-manifold
ART=$WORK/artifacts
export HF_HOME=$WORK/hf                    # house rule: HF cache on the volume
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
LOG=$WORK/jspace_diag.log

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---------- 0. environment ----------
log "=== setup ==="
apt-get update -qq >/dev/null 2>&1 || true   # house rule: fresh containers need this
apt-get install -y -qq git curl >/dev/null 2>&1 || true
mkdir -p "$ART" "$HF_HOME"

if [ ! -d "$REPO" ]; then
  git clone --branch "$BRANCH" --depth 50 "$REPO_URL" "$REPO" || {
    log "FATAL: clone failed — push the branch or scp the repo to $REPO"; exit 1; }
fi
cd "$REPO"
log "repo at $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"

python -m venv "$WORK/venv-jlens"
source "$WORK/venv-jlens/bin/activate"
pip install -q --upgrade pip
# house rule: container torchvision/torchaudio poison the venv's transformers import
pip uninstall -y -q torchvision torchaudio 2>/dev/null || true
pip install -q torch --index-url https://download.pytorch.org/whl/cu124
pip install -q "transformers>=4.44" accelerate safetensors huggingface_hub numpy
pip install -q "git+https://github.com/anthropics/jacobian-lens.git@581d398613e5602a5af361e1c34d3a92ea82ba8e"
python - <<'EOF'
import torch, transformers, jlens
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "| transformers", transformers.__version__, "| jlens OK")
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0),
          round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), "GB")
EOF

# ---------- artifact preflight ----------
for f in merged.fp32.pt external_readout_arrays.npz; do
  [ -f "$ART/$f" ] || { log "FATAL: missing $ART/$f — scp it before running (see runbook)"; exit 1; }
done
python - <<EOF
import hashlib
want = {"merged.fp32.pt": "6b5f1043b3c3fa3fcd8d4d69919772a2d763c145f996477bf715fe4b9b89f672",
        "external_readout_arrays.npz": "02ee47f173b034db744758ac915460d1bd7ed5c347bd4022ca97dd95760c6289"}
for name, expect in want.items():
    h = hashlib.sha256()
    with open("$ART/" + name, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    assert got == expect, f"{name} hash mismatch: {got}"
    print("artifact OK:", name)
EOF

run_stage() {   # run_stage <name> <command...>
  local name="$1"; shift
  log "=== $name START ==="
  local t0=$SECONDS
  if "$@" 2>&1 | tee -a "$LOG"; then
    log "=== $name DONE in $((SECONDS - t0))s ==="
    return 0
  else
    log "=== $name FAILED (exit ${PIPESTATUS[0]}) after $((SECONDS - t0))s — continuing ==="
    return 1
  fi
}

# ---------- D2: scorer positive control (gates how D3 is read) ----------
run_stage D2 python jspace_d2d3_readout.py --stage d2 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --lens-repo neuronpedia/jacobian-lens \
  --lens-file qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt \
  --label qwen2.5-7b-it

# ---------- D3: small-end scale ladder ----------
run_stage D3-qwen3-1.7b python jspace_d2d3_readout.py --stage d3 \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --lens-repo neuronpedia/jacobian-lens \
  --lens-file qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt \
  --label qwen3-1.7b

run_stage D3-gemma-3-1b python jspace_d2d3_readout.py --stage d3 \
  --model google/gemma-3-1b-pt \
  --revision fcf18a2a879aab110ca39f8bffbccd5d49d8eb29 \
  --lens-repo neuronpedia/jacobian-lens \
  --lens-file gemma-3-1b/jlens/Salesforce-wikitext/gemma-3-1b-pt_jacobian_lens.pt \
  --label gemma-3-1b

# ---------- D4: prompt-specific vs averaged Jacobian (AMENDMENT 2: length-eligible) ----------
run_stage D4 python jspace_d4_prompt_specific.py --merged-lens "$ART/merged.fp32.pt" \
  --selection amended

# ---------- D5: FP32 vs BF16 readout margins ----------
run_stage D5 python jspace_d5_precision.py --merged-lens "$ART/merged.fp32.pt" \
  --stored-npz "$ART/external_readout_arrays.npz"

# ---------- collect ----------
log "=== packaging ==="
cd "$REPO"
tar czf "$WORK/jspace_diag_results.tar.gz" results/jspace_r1_pilot/diagnostics "$LOG" 2>/dev/null || \
  tar czf "$WORK/jspace_diag_results.tar.gz" results/jspace_r1_pilot/diagnostics
ls -la "$WORK/jspace_diag_results.tar.gz"
log "ALL STAGES COMPLETE — pull $WORK/jspace_diag_results.tar.gz then STOP THE POD"
grep -E "^\[.*\] === .* (DONE|FAILED)" "$LOG" | tail -20

: <<'RUNBOOK'
# ---- on the Mac, before starting the job ----
# 1. push the branch so the pod can clone it:
git push origin codex/phase0-support
# 2. start pod (RTX 4090 community, PyTorch template, >=60 GB volume), then:
scp -P <PORT> -i ~/.ssh/id_ed25519 \
  "<scratchpad>/podxfer/merged.fp32.pt" \
  "<scratchpad>/podxfer/external_readout_arrays.npz" \
  root@<HOST>:/workspace/artifacts/
# 3. on the pod:
export HF_TOKEN=hf_xxx            # required: gemma-3-1b-pt is gated
mkdir -p /workspace/artifacts
bash runpod_jspace_diag.sh 2>&1 | tee /workspace/run.log
# 4. back on the Mac:
scp -P <PORT> root@<HOST>:/workspace/jspace_diag_results.tar.gz .
# 5. STOP THE POD.
RUNBOOK
