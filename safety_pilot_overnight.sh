#!/bin/bash
# Post-training spillover Rung-0 PILOT (safety chapter): STAR-1 vs R1-1.5B
# geometry diff on the SAME first-100 annotated generic-reasoning chains.
# Runs unattended on the Mac (MPS). ~10h total at ~3 min/chain.
#
# Steps:
#   0. wait for the in-flight 20-chain STAR-1 smoke to finish, then park its
#      output out of the way (it wrote to the un-suffixed dir).
#   1. extract STAR1-1.5B  on chains[:100]  -> data/activations/STAR1-1.5B-pilot100
#   1b. pt03 self-diff smoke-check on the fresh dir (format validation, ~min)
#   2. extract R1-1.5B     on chains[:100]  -> data/activations/R1-1.5B-pilot100
#   3. pt03 diff -> results/safety_posttrain/spillover_star1_pilot100.json
#
# NOTE (design): same chains through both checkpoints (teacher-forced), so the
# diff reads as representational spillover on fixed text; the 100-chain subset
# is first-N (pilot-grade, not stratified) — the full-corpus run supersedes it.
set -u
cd "$(dirname "$0")"
LOG=safety_pilot_overnight.log
exec >>"$LOG" 2>&1
echo "=== safety pilot orchestrator start: $(date) ==="

# (v2 2026-07-03 01:0x: relaunched after the adversarial methodology review
# caught a REAL confound — STAR1's tokenizer prepends an extra BOS vs base,
# shifting every activation. Fix: --tokenizer-alias 1.5b forces byte-identical
# input_ids across checkpoints. Contaminated first-attempt outputs deleted.)

# 1. STAR-1 pilot extraction (base tokenizer => identical input_ids)
echo "--- step 1: STAR1 pilot100 extraction (shared tokenizer): $(date)"
python3 -u 04_extract_activations.py --model star1-1.5b --max-chains 100 \
  --tokenizer-alias 1.5b \
  > star1_pilot100.log 2>&1
if ! grep -q "Done\. Activations saved" star1_pilot100.log; then
  echo "ABORT: STAR1 pilot extraction failed (see star1_pilot100.log)"; exit 1
fi

# 1b. pt03 self-diff format check (angles ~0 expected; validates I/O only)
echo "--- step 1b: pt03 self-diff smoke-check: $(date)"
mkdir -p results/safety_posttrain
python3 -u pt03_measure_spillover.py \
  --base-acts data/activations/STAR1-1.5B-pilot100 \
  --post-acts data/activations/STAR1-1.5B-pilot100 \
  --annotated data/annotated_R1-1.5B.json \
  --out results/safety_posttrain/selfdiff_check.json \
  > pt03_selfdiff.log 2>&1 \
  || { echo "ABORT: pt03 self-diff failed (see pt03_selfdiff.log)"; exit 1; }

# 2. R1-1.5B pilot extraction (same first-100 chains)
echo "--- step 2: R1-1.5B pilot100 extraction: $(date)"
python3 -u 04_extract_activations.py --model 1.5b --max-chains 100 \
  > r1_pilot100.log 2>&1
if ! grep -q "Done\. Activations saved" r1_pilot100.log; then
  echo "ABORT: R1 pilot extraction failed (see r1_pilot100.log)"; exit 1
fi

# 3. the real diff
echo "--- step 3: pt03 spillover diff: $(date)"
python3 -u pt03_measure_spillover.py \
  --base-acts data/activations/R1-1.5B-pilot100 \
  --post-acts data/activations/STAR1-1.5B-pilot100 \
  --annotated data/annotated_R1-1.5B.json \
  --out results/safety_posttrain/spillover_star1_pilot100.json \
  > pt03_pilot100.log 2>&1 \
  || { echo "ABORT: pt03 diff failed (see pt03_pilot100.log)"; exit 1; }

# 4. the GATED analysis (pre-registered nulls; the only report to cite from)
echo "--- step 4: pt03b gated nulls: $(date)"
python3 -u pt03b_spillover_nulls.py \
  --base-acts data/activations/R1-1.5B-pilot100 \
  --post-acts data/activations/STAR1-1.5B-pilot100 \
  --out results/safety_posttrain/spillover_gated_pilot100.json \
  > pt03b_pilot100.log 2>&1 \
  || { echo "ABORT: pt03b gated analysis failed (see pt03b_pilot100.log)"; exit 1; }

touch results/safety_posttrain/PILOT_DONE.marker
echo "=== safety pilot COMPLETE: $(date) ==="
