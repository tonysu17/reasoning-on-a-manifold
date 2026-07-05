#!/bin/bash
# Local Gate-0.1 orchestration (cluster is too loaded for CPU work).
# Usage: ./run_gate01_local.sh <running_null_pid>
# Waits for the B=200 variance-ratio null to finish, then runs the full
# regeneration + rebuild + prediction sequence on the clean focus-layer data,
# stopping BEFORE Phase 7 (the $290 GPU step, which needs Tony's go-ahead).
#
# Every step is continue-on-error: a failure is logged and the chain proceeds,
# so one slow/degenerate cell cannot abort the overnight run. A final summary
# records what produced output.

set -u
ROOT="/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold"
cd "$ROOT" || exit 1
PY=python3
RLAYERS="11 14 17 20 27"
SUM=results/gate01_local_summary.md
mkdir -p logs results
echo "# Gate-0.1 local orchestration — started $(date)" > "$SUM"

step () {  # step "name" command...
  local name="$1"; shift
  echo ""
  echo "=================================================================="
  echo "### STEP: $name   ($(date))"
  echo "=================================================================="
  "$@"
  local rc=$?
  echo "- **$name** — exit $rc ($(date))" >> "$SUM"
  return $rc
}

# ---- 0. Wait for the running B=200 null -------------------------------------
NULLPID="${1:-}"
if [ -n "$NULLPID" ]; then
  echo "Waiting for variance-ratio null pid $NULLPID ..."
  while kill -0 "$NULLPID" 2>/dev/null; do sleep 60; done
fi
if [ ! -f results/pca/R1-1.5B/null_pvalues_per_layer.json ]; then
  echo "ABORT: B=200 null produced no null_pvalues_per_layer.json" | tee -a "$SUM"
  exit 1
fi
cp results/pca/R1-1.5B/null_pvalues_per_layer.json results/pca/R1-1.5B/null_pvalues_B200.json
echo "B=200 variance-ratio null complete; snapshot saved as null_pvalues_B200.json" | tee -a "$SUM"

# ---- 1. Behaviour-specificity verdict + conditional high-res null -----------
VERDICT=$($PY - <<'EOF'
import json
d = json.load(open("results/pca/R1-1.5B/null_pvalues_per_layer.json"))
report = {11,14,17,20,27}
hits = total = 0
detail = []
for b, layers in d.items():
    for L, c in layers.items():
        if int(L) in report:
            total += 1
            rv, nm, p = c.get("real_value"), c.get("null_mean"), c.get("p_value")
            ok = (rv is not None and nm is not None and rv > nm and (p if p is not None else 1) < 0.05)
            hits += int(ok)
            detail.append(f"{b} L{L}: real={rv:.3f} null={nm:.3f} p={p:.4g} {'HIT' if ok else '-'}")
print("PROMISING" if hits >= max(1, total//3) else "FLAT")
import sys
print(f"behaviour-specificity: {hits}/{total} report cells with real>null & p<0.05", file=sys.stderr)
for line in detail: print("  " + line, file=sys.stderr)
EOF
)
echo "Behaviour-specificity verdict: $VERDICT" | tee -a "$SUM"
if [ "$VERDICT" = "PROMISING" ]; then
  step "high-res null B=2500 (report layers)" \
    $PY 05_pca_analysis.py --with-nulls --null-resamples 2500 --layers $RLAYERS --model-short R1-1.5B
else
  echo "- high-res null SKIPPED (B=200 verdict FLAT; behaviour-specificity handed to the steering result)" >> "$SUM"
fi

# ---- 2. Geometry battery at the report layers (ch06 numerals) ---------------
step "05b geometry battery (report layers)" \
  $PY 05b_geometric_diagnostics.py --model-short R1-1.5B --layers $RLAYERS --n-resamples 200

# ---- 3. Full-sweep steps, ONLY if all 28 layers are clean locally ----------
ALL28_CLEAN=$($PY - <<'EOF'
import json, numpy as np
from pathlib import Path
ACT = Path("data/activations/R1-1.5B")
# sample two non-focus layers; clean if exact-duplicate fraction is low
bad = False
for L in (0, 5, 23):
    p = ACT / f"backtracking_layer{L}.npy"
    if not p.exists():
        bad = True; break
    X = np.load(p)
    if (1 - len(np.unique(X, axis=0)) / X.shape[0]) > 0.05:
        bad = True; break
print("NO" if bad else "YES")
EOF
)
if [ "$ALL28_CLEAN" = "YES" ]; then
  echo "All 28 layers clean locally — running full-sweep steps." | tee -a "$SUM"
  step "05 layer profiles (all layers)"      $PY 05_pca_analysis.py --model-short R1-1.5B
  step "05c cross-layer probing (grouped CV)" $PY 05c_cross_layer_probing.py --model-short R1-1.5B
  step "layer triangulation"                  $PY compute_layer_triangulation.py --model-short R1-1.5B
else
  echo "- Full-sweep steps (05 profiles / 05c / triangulation) SKIPPED — all-28 clean pull not finished; rerun after rsync_all28 completes" >> "$SUM"
fi

# ---- 4. Steering-vector rebuild on clean data (true hold-out) --------------
step "rebuild vectors @ layer 27 (hold-out)"  $PY 06_build_steering.py --model-short R1-1.5B
step "rebuild vectors @ peak layers (hold-out)" $PY build_phase6.py

# ---- 5. Pre-registered saturation predictions ------------------------------
step "saturation predictions (pre-register alpha*)" $PY predict_saturation.py

echo "" | tee -a "$SUM"
echo "## Gate-0.1 local orchestration finished $(date)" | tee -a "$SUM"
echo "STOPPED before Phase 7 (smoke + \$290 spend need the GPU and Tony's go-ahead)." | tee -a "$SUM"
