#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/jspace-phase1
PYTHON=/workspace/jspace-phase0/.venv/bin/python
MODEL_SNAPSHOT=/workspace/hf/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562
STATUS="$ROOT/JSPACE_PHASE1_MANIFEST_STATUS"
LOG="$ROOT/JSPACE_PHASE1_MANIFEST.log"

fail() {
  code=$?
  printf 'FAILED:%s\n' "$code" > "$STATUS"
  exit "$code"
}
trap fail ERR

if [[ -e "$ROOT/results/prereg/jspace_r1_fit_manifest.json" || \
      -e "$ROOT/results/prereg/jspace_r1_eval_eligibility_manifest.json" ]]; then
  printf 'REFUSED:OUTPUT_EXISTS\n' > "$STATUS"
  exit 2
fi

printf 'RUNNING\n' > "$STATUS"
export HF_HOME=/workspace/hf
export HF_DATASETS_CACHE=/workspace/hf/datasets
export TOKENIZERS_PARALLELISM=true

"$PYTHON" "$ROOT/jspace_phase1_manifest.py" \
  --root "$ROOT" \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --jlens-checkout "$ROOT/_external/jacobian-lens" \
  --dataset-cache /workspace/hf/datasets \
  --corpus-output "$ROOT/results/prereg/jspace_r1_fit_manifest.json" \
  --eligibility-output "$ROOT/results/prereg/jspace_r1_eval_eligibility_manifest.json" \
  --batch-size 2048 >> "$LOG" 2>&1

cd "$ROOT"
sha256sum \
  results/prereg/jspace_r1_fit_manifest.json \
  results/prereg/jspace_r1_eval_eligibility_manifest.json \
  > JSPACE_PHASE1_MANIFEST_SHA256SUMS
printf 'DONE\n' > "$STATUS"
