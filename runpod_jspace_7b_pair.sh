#!/usr/bin/env bash
# 7B matched-pair scale replication (sealed JSPACE_7B_PAIR_SHEET_2026-08-17.md).
#
# POD SPEC (ephemeral — no network volume, avoids all quota drama):
#   one 48 GB GPU (A6000 / L40S class), >=100 GB CONTAINER disk, PyTorch template.
# Everything lives under /root; artifacts are tarred for scp before pod stop.
#
# Per model, sequentially: fit (jspace_lens_fit.py, OOM ladder 8->4->2)
# -> score (jspace_d2d3_readout.py --stage d3) -> wipe model weights cache.
set -Eeuo pipefail

WORK=/root
REPO=$WORK/reasoning-on-manifold
export HF_HOME=$WORK/hf
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
LOG=$WORK/pair7b.log
STATUS=$WORK/pair7b.status
RUN_ID="${ROM_RUN_ID:?launcher must set ROM_RUN_ID}"
SOURCE_COMMIT="${ROM_GIT_COMMIT:?launcher must set ROM_GIT_COMMIT}"
PIP_VERSION=26.2.1

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
write_status() {
  printf '{"run_id":"%s","source_commit":"%s","status":"%s","utc":"%s"}\n' \
    "$RUN_ID" "$SOURCE_COMMIT" "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
}
package_results() {
  mkdir -p "$REPO/results/jspace_r1_pilot/diagnostics/d3" "$WORK/lenses"
  cp "$STATUS" "$WORK/lenses/RUN_STATUS.json"
  cp "$WORK/bootstrap.log" "$WORK/lenses/bootstrap.log" 2>/dev/null || true
  tar czf "$WORK/jspace_7b_pair_results.tar.gz" -C "$WORK" \
      reasoning-on-manifold/results/jspace_r1_pilot/diagnostics/d3 \
      lenses pair7b.log pair7b.status bootstrap.log
  sha256sum "$WORK/jspace_7b_pair_results.tar.gz" > "$WORK/jspace_7b_pair_results.tar.gz.sha256"
}
on_error() {
  rc=$?
  trap - ERR
  write_status FAILED
  log "PAIR FAILED rc=$rc run_id=$RUN_ID"
  package_results || true
  exit "$rc"
}
trap on_error ERR

log "=== setup ==="
write_status RUNNING
[ "$(cat "$REPO/.source_commit")" = "$SOURCE_COMMIT" ]
[ "$(cat "$REPO/.run_id")" = "$RUN_ID" ]
cd "$REPO"
sha256sum -c PAYLOAD_MANIFEST.sha256
mkdir -p "$WORK/lenses"
cp .source_commit "$WORK/lenses/SOURCE_COMMIT.txt"
cp .run_id "$WORK/lenses/RUN_ID.txt"
cp PAYLOAD_MANIFEST.sha256 "$WORK/lenses/PAYLOAD_MANIFEST.sha256"

apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq git curl rsync >/dev/null 2>&1
command -v rsync sha256sum tar python >/dev/null
free_gb=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc '0-9')
log "container disk free: ${free_gb}G"
[ "$free_gb" -ge 60 ] || { log "FATAL: need >=60G free on container disk (pod spec says 100G)"; exit 1; }

log "payload source commit $SOURCE_COMMIT; run_id=$RUN_ID"

python -m venv "$WORK/venv-jlens"
source "$WORK/venv-jlens/bin/activate"
python -m pip install -q "pip==$PIP_VERSION"
pip uninstall -y -q torchvision torchaudio 2>/dev/null || true
pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -q -r results/prereg/jspace_7b_requirements_lock.txt
pip install -q --no-deps --no-build-isolation \
  "git+https://github.com/anthropics/jacobian-lens.git@581d398613e5602a5af361e1c34d3a92ea82ba8e"
pip check
pip freeze | sort > "$WORK/lenses/PIP_FREEZE.txt"
nvidia-smi -q > "$WORK/lenses/NVIDIA_SMI.txt"
df -BG > "$WORK/lenses/DISK_LAYOUT.txt"
python - <<'EOF'
import json, sys
from importlib.metadata import distribution

import jlens, numpy, pip, torch, transformers

assert sys.version_info[:2] == (3, 11), sys.version
assert pip.__version__ == "26.2.1", pip.__version__
assert torch.__version__ == "2.6.0+cu124", torch.__version__
assert transformers.__version__ == "5.15.0", transformers.__version__
assert numpy.__version__ == "2.2.6", numpy.__version__
direct_url = json.loads(distribution("jlens").read_text("direct_url.json"))
assert direct_url["vcs_info"]["commit_id"] == "581d398613e5602a5af361e1c34d3a92ea82ba8e", direct_url
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| transformers", transformers.__version__, "| jlens OK")
print("gpu:", torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/1e9,1), "GB")
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
assert torch.cuda.get_device_properties(0).total_memory >= 45_000 * 1024 * 1024
assert torch.cuda.is_bf16_supported()
EOF

run_cell() {  # run_cell <label> <hf_model> <revision>
  local label="$1" model="$2" rev="$3"
  local t0=$SECONDS
  log "=== FIT $label START ==="
  if python jspace_lens_fit.py --model "$model" --revision "$rev" \
       --out-dir "$WORK/lenses" --out-name "$label" \
       --expected-layers 28 --expected-d-model 3584 \
       --expected-head-vocab 152064 2>&1 | tee -a "$LOG"; then
    log "=== FIT $label DONE in $((SECONDS-t0))s ==="
  else
    log "=== FIT $label FAILED ==="; return 1
  fi
  t0=$SECONDS
  log "=== SCORE $label START ==="
  if python jspace_d2d3_readout.py --stage d3 --model "$model" --revision "$rev" \
       --lens-path "$WORK/lenses/${label}_wikitext100.pt" --label "$label" \
       --expected-layers 28 --expected-d-model 3584 \
       --expected-head-vocab 152064 2>&1 | tee -a "$LOG"; then
    log "=== SCORE $label DONE in $((SECONDS-t0))s ==="
  else
    log "=== SCORE $label FAILED ==="; return 1
  fi
  if ! test -s "$WORK/lenses/${label}_wikitext100.pt" ||
     ! test -s "$WORK/lenses/${label}_fit_meta.json" ||
     ! compgen -G "$REPO/results/jspace_r1_pilot/diagnostics/d3/*${label}*/report.json" >/dev/null ||
     ! compgen -G "$REPO/results/jspace_r1_pilot/diagnostics/d3/*${label}*/eligibility.json" >/dev/null; then
    log "=== OUTPUT VALIDATION $label FAILED ==="; return 1
  fi
  # wipe the 15 GB weights cache before the next model (lenses + results kept)
  rm -rf "$HF_HOME/hub/models--${model//\//--}"
  log "cache wiped for $model; disk free now $(df -BG --output=avail $WORK | tail -1 | tr -dc '0-9')G"
}

failures=0
run_cell math-7b-base       Qwen/Qwen2.5-Math-7B                     b101308fe89651ea5ce025f25317fea6fc07e96e || failures=$((failures+1))
run_cell r1-distill-7b      deepseek-ai/DeepSeek-R1-Distill-Qwen-7B  916b56a44061fd5cd7d6a8fb632557ed4f724f60 || failures=$((failures+1))
[ "$failures" -eq 0 ] || { write_status FAILED; log "PAIR FAILED: $failures cell(s) failed"; package_results; exit 1; }

log "=== packaging ==="
write_status SUCCEEDED
package_results
ls -la "$WORK/jspace_7b_pair_results.tar.gz"
log "ALL 7B PAIR STAGES COMPLETE run_id=$RUN_ID — pull the tarball then STOP THE POD"
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
