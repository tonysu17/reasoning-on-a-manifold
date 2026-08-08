#!/bin/bash
# E10.3 on-pod chain (run by runpod_e10_3.sh launch inside tmux 'e10x').
# Per behaviour: main (train+eval, reusing the PUSHED inspected pairs.json) ->
# controls -> cis -> width ONLY at the grounded layer per the sealed rule.
# A missing grounded layer skips width — that is the sealed outcome (Amendment 4).
# No self-kill anywhere (2026-07-13 kill discipline: Mac-side watcher terminates).
set -uo pipefail
cd "$(dirname "$0")"
OUT=results/das/R1-1.5B
NPAIRS="${NPAIRS:-400}"

for B in uncertainty-estimation example-testing; do
  case "$B" in
    uncertainty-estimation) S=unc ;;
    example-testing)        S=ex ;;
  esac
  echo "[e10.3-chain] ===== $B ====="
  python3 -u 20_das_backtracking.py --stage train    --behaviour "$B" --n-pairs "$NPAIRS" || exit 1
  python3 -u 20_das_backtracking.py --stage controls --behaviour "$B" --n-pairs "$NPAIRS" || exit 1
  python3 -u 20_das_backtracking.py --stage cis      --behaviour "$B" --n-pairs "$NPAIRS" || exit 1
  if GL=$(python3 e10_pick_grounded_layer.py "$OUT/${S}_main/controls.json"); then
    echo "[e10.3-chain] $B grounded layer = $GL -> width probe"
    python3 -u 21_das_width.py --behaviour "$B" --layer "$GL" --skip-ak --n-pairs "$NPAIRS" || exit 1
  else
    echo "[e10.3-chain] $B: NO grounded layer - width SKIPPED (sealed outcome)"
  fi
done
touch E10_3_DONE.marker
echo "[e10.3-chain] ALL DONE"
