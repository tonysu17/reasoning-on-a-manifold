#!/usr/bin/env bash
# 7B matched-pair scale replication (sealed JSPACE_7B_PAIR_SHEET_2026-08-17.md).
#
# POD SPEC (ephemeral — no network volume, avoids all quota drama):
#   one 48 GB GPU (A6000 / L40S class), >=100 GB CONTAINER disk, PyTorch template.
# Everything lives under /root; artifacts are tarred for scp before pod stop.
#
# Per model, sequentially: fit (jspace_lens_fit.py, OOM ladder 8->4->2)
# -> score (jspace_d2d3_readout.py --stage d3) -> wipe model weights cache.
set -uo pipefail

REPO_URL="${REPO_URL:-https://github.com/tonysu17/reasoning-on-a-manifold.git}"
BRANCH="${BRANCH:-codex/phase0-support}"
WORK=/root
REPO=$WORK/reasoning-on-manifold
export HF_HOME=$WORK/hf
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
LOG=$WORK/pair7b.log

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== setup ==="
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq git curl >/dev/null 2>&1 || true
free_gb=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc '0-9')
log "container disk free: ${free_gb}G"
[ "$free_gb" -ge 60 ] || { log "FATAL: need >=60G free on container disk (pod spec says 100G)"; exit 1; }

if [ ! -d "$REPO" ]; then
  git clone --branch "$BRANCH" --depth 50 "$REPO_URL" "$REPO" || { log "FATAL: clone failed"; exit 1; }
fi
cd "$REPO"; log "repo at $(git rev-parse --short HEAD)"

python -m venv "$WORK/venv-jlens"
source "$WORK/venv-jlens/bin/activate"
pip install -q --upgrade pip
pip uninstall -y -q torchvision torchaudio 2>/dev/null || true
pip install -q torch --index-url https://download.pytorch.org/whl/cu124
pip install -q "transformers>=4.44" accelerate safetensors huggingface_hub numpy
pip install -q "git+https://github.com/anthropics/jacobian-lens.git@581d398613e5602a5af361e1c34d3a92ea82ba8e"
python - <<'EOF'
import torch, transformers, jlens
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| transformers", transformers.__version__, "| jlens OK")
print("gpu:", torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/1e9,1), "GB")
EOF

run_cell() {  # run_cell <label> <hf_model> <revision>
  local label="$1" model="$2" rev="$3"
  local t0=$SECONDS
  log "=== FIT $label START ==="
  if python jspace_lens_fit.py --model "$model" --revision "$rev" \
       --out-dir "$WORK/lenses" --out-name "$label" 2>&1 | tee -a "$LOG"; then
    log "=== FIT $label DONE in $((SECONDS-t0))s ==="
  else
    log "=== FIT $label FAILED ==="; return 1
  fi
  t0=$SECONDS
  log "=== SCORE $label START ==="
  if python jspace_d2d3_readout.py --stage d3 --model "$model" --revision "$rev" \
       --lens-path "$WORK/lenses/${label}_wikitext100.pt" --label "$label" 2>&1 | tee -a "$LOG"; then
    log "=== SCORE $label DONE in $((SECONDS-t0))s ==="
  else
    log "=== SCORE $label FAILED ==="; return 1
  fi
  # wipe the 15 GB weights cache before the next model (lenses + results kept)
  rm -rf "$HF_HOME/hub/models--${model//\//--}"
  log "cache wiped for $model; disk free now $(df -BG --output=avail $WORK | tail -1 | tr -dc '0-9')G"
}

run_cell math-7b-base       Qwen/Qwen2.5-Math-7B                     b101308fe89651ea5ce025f25317fea6fc07e96e || true
run_cell r1-distill-7b      deepseek-ai/DeepSeek-R1-Distill-Qwen-7B  916b56a44061fd5cd7d6a8fb632557ed4f724f60 || true

log "=== packaging ==="
cd "$REPO"
tar czf "$WORK/jspace_7b_pair_results.tar.gz" \
    results/jspace_r1_pilot/diagnostics/d3 "$WORK/lenses" "$LOG" 2>/dev/null || \
tar czf "$WORK/jspace_7b_pair_results.tar.gz" results/jspace_r1_pilot/diagnostics/d3 "$WORK/lenses"
ls -la "$WORK/jspace_7b_pair_results.tar.gz"
log "ALL 7B PAIR STAGES COMPLETE — pull the tarball then STOP THE POD"
grep -E "=== .* (DONE|FAILED)" "$LOG" | tail -8

: <<'RUNBOOK'
# Mac side:
# 1. git push origin codex/phase0-support   (must include the sealed sheet + this script)
# 2. Start pod: 48GB GPU (A6000/L40S), PyTorch template, >=100GB CONTAINER disk, no volume.
# 3. ssh in, then:  bash <(curl -sL https://raw.githubusercontent.com/tonysu17/reasoning-on-a-manifold/codex/phase0-support/runpod_jspace_7b_pair.sh)
#    (or clone first and run the script from the repo)
# 4. Pull:  scp -P <PORT> root@<HOST>:/root/jspace_7b_pair_results.tar.gz .
# 5. STOP THE POD.  Then locally: python jspace_pair_report.py
RUNBOOK
