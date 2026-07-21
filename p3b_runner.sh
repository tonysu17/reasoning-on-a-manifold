#!/bin/bash
# Mac-side P2 chain: push worktree code + v2 annotations to the pod, launch the
# staged extraction (pod_p3b_extract.sh), wait for the marker, pull, verify.
#
# Usage:
#   POD_HOST=root@IP POD_PORT=22591 [POD_KEY=~/.ssh/id_ed25519] ./p3b_runner.sh
#
# NEEDS A ~48 GB CARD (A40/A6000): the gpt-oss loader dequantizes MXFP4 to bf16
# (~42 GB weights). The pod script refuses smaller cards at stage `vram`.
set -u
cd "$(dirname "$0")"
MAIN="$(pwd)"
WT="$MAIN/../rom-safety-worktree"
LOG=results/safety/p3b_runner.log
POD_HOST="${POD_HOST:?set POD_HOST=root@IP}"
POD_PORT="${POD_PORT:?set POD_PORT}"
POD_KEY="${POD_KEY:-$HOME/.ssh/id_ed25519}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -p "$POD_PORT" -i "$POD_KEY" "$POD_HOST")
RS="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $POD_PORT -i $POD_KEY"
log(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

log "=== P2 runner start ($POD_HOST:$POD_PORT) ==="

"${SSH[@]}" "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" | tee -a "$LOG" || { log "ssh failed"; exit 1; }

log "pushing code + data"
"${SSH[@]}" "command -v rsync >/dev/null || (apt-get update -q && apt-get install -y -q rsync)" >>"$LOG" 2>&1 \
  || { log "cannot provision rsync on pod — abort"; exit 1; }
"${SSH[@]}" "mkdir -p /workspace/rom-p3b/data /workspace/rom-p3b/results/safety"
rsync -az --no-owner --no-group -e "$RS" --exclude '.git' --exclude '__pycache__' --exclude 'results' \
  --exclude 'data' --exclude 'doc' --exclude 'tests' \
  "$WT/" "$POD_HOST:/workspace/rom-p3b/" 2>>"$LOG" \
  || { log "code push FAILED — abort before launch"; exit 1; }
"${SSH[@]}" "mkdir -p /workspace/rom-p3b/results/safety/p3_forgery"
rsync -az --no-owner --no-group -e "$RS" "$MAIN/results/safety/p3_forgery/" \
  "$POD_HOST:/workspace/rom-p3b/results/safety/p3_forgery/" 2>>"$LOG" \
  || { log "data push FAILED — abort before launch"; exit 1; }
"${SSH[@]}" "[ -s /workspace/rom-p3b/pod_p3b_extract.sh ] && [ -s /workspace/rom-p3b/results/safety/p3_forgery/manifest_attacker.jsonl ]" \
  || { log "push verification FAILED — files missing on pod"; exit 1; }

log "launching pod_p3b_extract.sh (staged, watchdog 6h)"
"${SSH[@]}" "cd /workspace/rom-p3b && rm -f P3B_FAILED.marker && setsid nohup bash pod_p3b_extract.sh > pod_p3b.log 2>&1 < /dev/null & echo launched"

log "waiting on P3B_DONE / P3B_FAILED marker (poll 120s)"
until "${SSH[@]}" "[ -f /workspace/rom-p3b/P3B_DONE.marker ] || [ -f /workspace/rom-p3b/P3B_FAILED.marker ]" 2>/dev/null; do
  sleep 120
done
if "${SSH[@]}" "[ -f /workspace/rom-p3b/P3B_FAILED.marker ]" 2>/dev/null; then
  rsync -az --no-owner --no-group -e "$RS" "$POD_HOST:/workspace/rom-p3b/pod_p3b.log" results/safety/ 2>>"$LOG" || true
  rsync -az --no-owner --no-group -e "$RS" --include 'p3bstage_*.log' --exclude '*' "$POD_HOST:/workspace/rom-p3b/" results/safety/ 2>>"$LOG" || true
  log "P2 FAILED on pod — logs pulled to results/safety/; pod left RUNNING for diagnosis"
  exit 1
fi

log "P2 done — pulling activations"
rsync -az --no-owner --no-group -e "$RS" "$POD_HOST:/workspace/rom-p3b/results/safety/p3_forgery/activations/" \
  results/safety/p3_forgery/activations/ 2>>"$LOG"
rsync -az --no-owner --no-group -e "$RS" "$POD_HOST:/workspace/rom-p3b/pod_p3b.log" results/safety/ 2>>"$LOG" || true

python3 - <<'EOF' >> "$LOG" 2>&1
import json
import numpy as np
from pathlib import Path
out = Path("results/safety/p3_forgery/activations")
rep = json.loads((out / "P3B_EXTRACTION_REPORT.json").read_text())
assert rep["n_rows"] >= 300, f"only {rep['n_rows']} rows"
for f in out.glob("acts_layer*.npy"):
    assert np.isfinite(np.load(f)).all(), f"non-finite {f.name}"
print("LOCAL VERIFY OK:", json.dumps(rep["by_variant_style"]))
EOF
if [ $? -eq 0 ]; then
  touch results/safety/P3B_READY_FOR_REVIEW.marker
  log "=== P2 COMPLETE + verified local — safe to stop the pod ==="
else
  log "=== local verification FAILED — pod left RUNNING, do not delete remote data ==="
  exit 1
fi
