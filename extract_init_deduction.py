#!/usr/bin/env python3
"""Re-extract the TWO missing Venhoff labels — initializing + deduction — so the
attribution's "overall" mean becomes a true 6-label mean (was 4-label).

Disk-safe subset: the cluster disk is ~4 GB free, and deduction has ~35k spans
(a full all-layer extraction would be ~6 GB). But "overall" is a MEAN, which
converges in a few thousand rows, so a subset of chains gives a faithful overall
at a fraction of the disk. We extract MEAN-pooling only (no last-pool sweep →
~half the bytes) and UNCLIPPED (matching the existing 4-target files, so the
6-label overall is internally consistent). Writes into the existing
data/activations/R1-1.5B/ alongside the 4 targets.
"""
import sys
sys.path.insert(0, ".")
import json
import logging
from pathlib import Path

from src.chain_gen import load_model
from src.activation_extraction import extract_activations
from src.config import model_tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

MODEL_ID, SHORT, DTYPE = model_tuple("1.5b")
SAVE = Path(f"data/activations/{SHORT}")
ANNOT = Path(f"data/annotated_{SHORT}.json")
MAX_CHAINS = 150   # disk-safe; ~5,250 deduction + ~735 init rows -> stable mean

raw = json.load(open(ANNOT))
chains = raw if isinstance(raw, list) else raw.get("chains", raw.get("annotated", raw))
print(f"loaded {len(chains)} annotated chains; extracting first {MAX_CHAINS}")

model, tok = load_model(MODEL_ID, dtype=DTYPE)
extract_activations(
    model, tok, chains, layers=None, save_dir=SAVE,
    behaviours=["initializing", "deduction"],
    max_chains=MAX_CHAINS,
    pooling="mean", sweep_modes=[],      # MEAN only — no last-pool sweep
    clip_to_sentence_end=False,          # match the existing 4-target extraction
    keep_in_memory=False,
)
print("DONE: initializing + deduction extracted (mean, "
      f"{MAX_CHAINS} chains) -> {SAVE}")
