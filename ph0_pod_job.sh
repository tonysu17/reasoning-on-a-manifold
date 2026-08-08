#!/bin/bash
# Phase 0 pod job — runs ON the pod under tmux (launched by runpod_phase0.sh).
# Freeze: results/prereg/PHASE0_TRANSPORT_FREEZE_2026-08-02.md §2 (stages s0–s3).
# Kill discipline: NO on-pod self-kill — Mac watcher pulls, Tony terminates.
set -uo pipefail
cd /workspace/reasoning-on-manifold
export HF_HOME=/workspace/hf                            # network volume — root overlay is 20G
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # house VRAM lesson (pod_jobs.sh)
export TOKENIZERS_PARALLELISM=false
LOG=ph0.log; : > "$LOG"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail(){ log "STAGE FAILED: $1"; echo "FAILED:$1" > PH0_STATUS; }
# NB: df on the mfs network volume reports CLUSTER free space — gate on OUR usage instead.
disk_gate(){ local used; used=$(du -s --block-size=1G /workspace 2>/dev/null | cut -f1)
  if [ "${used:-999}" -gt 70 ]; then fail "disk_gate:${used}G>70G"; return 1; fi
  log "disk_gate ok (/workspace ${used}G used, gate 70G)"; }
vram_log(){ log "VRAM: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null)"; }
echo RUNNING > PH0_STATUS
log "=== phase0 job start (GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)) ==="
df -h /workspace | tail -1 | tee -a "$LOG"

# ── s0: F5 throughput probe (freeze §1.2; bash-SECONDS fix, no /usr/bin/time) ──
log "=== s0: F5 throughput probe ==="
disk_gate || true; vram_log
t0=$SECONDS
if python3 pt09_build_dpo_pairs.py --arm control --generate --n 8 --k 6 \
     --temperature 0.8 --max-new-tokens 2048 \
     --tasks data/tasks_final.json --out results/dpo_control_probe.json >>"$LOG" 2>&1; then
  echo "PROBE_WALL_SECONDS=$((SECONDS-t0))" | tee results/probe_wall_seconds.txt >>"$LOG"
  python3 - <<'PY' 2>&1 | tee -a "$LOG"
import json
wall = int(open("results/probe_wall_seconds.txt").read().strip().split("=")[1])
d = json.load(open("results/dpo_control_probe.json"))
pairs = d if isinstance(d, list) else d.get("records", [])
per_task_s, yield_pt = wall / 8, (len(pairs) / 8) or 1e-9
hours = (500 / yield_pt) * per_task_s / 3600
route = "FULL" if hours <= 12 else "FALLBACK_65PAIR"   # pre-committed 12 GPU-h threshold
json.dump({"probe_wall_s": wall, "yield_per_task": yield_pt, "tasks_needed_500": round(500/yield_pt),
           "projected_hours_500": round(hours, 2), "route": route},
          open("results/ph0_f5_route.json", "w"), indent=1)
print(f"F5 ROUTE: {route} ({hours:.1f} h projected for 500 pairs)")
PY
else fail s0; fi

# ── s1: F5 control generation (route FULL only; training = MANUAL GATE per ledger §F5) ──
log "=== s1: F5 control ==="
if [ -f results/ph0_f5_route.json ] && grep -q '"route": "FULL"' results/ph0_f5_route.json; then
  N=$(python3 -c "import json;print(json.load(open('results/ph0_f5_route.json'))['tasks_needed_500'])")
  log "route FULL -> generating control pairs (n=$N tasks)"
  python3 pt09_build_dpo_pairs.py --arm control --generate --n "$N" --k 6 \
      --temperature 0.8 --max-new-tokens 2048 \
      --tasks data/tasks_final.json --out results/dpo_control_500.json >>"$LOG" 2>&1 \
    || fail s1-gen
else
  log "route FALLBACK_65PAIR (or s0 failed): no generation. Fallback trains the existing 65"
  log "pairs to matched steps/KL with the data-repetition confound declared."
fi
log "s1 TRAINING is a MANUAL GATE either route: pt10_train_dpo.py at optimizer-steps AND"
log "realised-KL matched to dpo-safety-v2 (KL 0.000587) — ledger §F5 spec; verify before run."

# ── s2: STAR1 inert-control extraction (= the one-checkpoint smoke test, freeze §1.3) ──
log "=== s2: STAR1 inert extraction (deduction+initializing, L12/16, byte-identical ids) ==="
disk_gate || true; vram_log
t0=$SECONDS
if python3 ph0_s2_extract_inert.py >>"$LOG" 2>&1; then
  log "S2_WALL_SECONDS=$((SECONDS-t0)) S2_DISK=$(du -sh data/activations/STAR1-1.5B-6label 2>/dev/null | cut -f1)"
else fail s2; fi

# ── s3: full-sequence curve + SV family check (pt13b Gate-B2 fix, freeze §1.1/§2-s3) ──
log "=== s3: full-sequence states r1 + deepscaler -> curve + family check ==="
disk_gate || true; vram_log
python3 ph0_s3_curve.py --extract --arm r1          >>"$LOG" 2>&1 || fail s3-r1
python3 ph0_s3_curve.py --extract --arm deepscaler  >>"$LOG" 2>&1 || fail s3-deepscaler
python3 ph0_s3_curve.py --analyse                   >>"$LOG" 2>&1 || fail s3-analyse

grep -q FAILED PH0_STATUS 2>/dev/null || echo DONE > PH0_STATUS
touch PH0_DONE.marker
log "=== phase0 job end: $(cat PH0_STATUS) ==="
