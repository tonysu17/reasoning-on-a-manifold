#!/bin/bash
# PILOT ANNOTATION of the trimmed bake-off — turns generated chains into the
# on-target behaviour-fraction that decides L16 vs L27. Re-runs 07 WITHOUT
# --skip-annotation, so it RESUMES the saved generations (no regeneration) and
# only annotates. Sonnet 4.5 via the lab proxy. Staged L27 -> L16 so we can
# inspect after layer 1. Checkpoints every chain (crash-safe + resumable).
# Scope is fixed at 90 chains/layer = 180 total -> the $-cap is enforced by this
# bounded scope (est ~$5-11; <$20 even at a pessimistic $0.10/chain). No live $
# meter exists on the proxy, so the chain ceiling IS the cap.
set -u
cd /home/tony/reasoning-on-manifold
if [ ! -f "$HOME/.rom_proxy_env" ]; then echo "MISSING ~/.rom_proxy_env (creds)"; exit 1; fi
source "$HOME/.rom_proxy_env"   # exports CLAUDE_PROXY_URL + CLAUDE_PROXY_KEY (chmod 600)
PY=/home/tony/venv/bin/python
MAIN=results/eval/pilot_annotation.log
echo "=== [$(date)] START pilot annotation: L27 then L16, 90 chains each, Sonnet 4.5 ===" | tee -a "$MAIN"

annotate () {
  local vdir=$1 odir=$2 log=$3
  echo "=== [$(date)] annotating $odir ===" | tee -a "$log"
  $PY 07_evaluate_steering.py --model 1.5b \
    --vectors-dir "$vdir" --out-dir "$odir" \
    --max-eval-tasks 10 --alpha-values 1.0 --max-new-tokens 4096 \
    --no-random-control --no-random-subspace --no-energy-matched --no-orthogonal-complement \
    >> "$log" 2>&1
  echo "=== [$(date)] $odir annotation exit rc=$? ===" | tee -a "$log"
}

annotate results/steering_vectors/R1-1.5B__L27_auto results/eval/R1-1.5B__L27_trim results/eval/anno_L27.log
echo "=== [$(date)] L27 done; proceeding to L16 ===" | tee -a "$MAIN"
annotate results/steering_vectors/R1-1.5B__L16_auto results/eval/R1-1.5B__L16_trim results/eval/anno_L16.log
echo "=== [$(date)] PILOT ANNOTATION COMPLETE ===" | tee -a "$MAIN"
date > results/eval/pilot_annotation_DONE.marker
