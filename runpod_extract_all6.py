#!/usr/bin/env python3
"""FAITHFUL full activation extraction (RunPod, clean disk).

Unlike the cluster (4 labels, unclipped, disk-starved), this extracts the FULL
Venhoff setup in one clean run:
  - all 6 labels: backtracking, uncertainty-estimation, example-testing,
    adding-knowledge, initializing, deduction  (so 'overall' is a true 6-label mean)
  - clip_window_to_sentence_end=True            (Venhoff's exact span pooling)
  - all 1000 annotated chains                   (no subset — ample disk on RunPod)
  - mean pooling                                (the resolved Venhoff/field default)

Produces a faithful 6-label activation set under data/activations/R1-1.5B/ that
the Venhoff attribution (true 6-label overall) and the vector builds (clipped
windows) both consume. ~3-4 h on an RTX 4090.
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
SAVE.mkdir(parents=True, exist_ok=True)

raw = json.load(open(f"data/annotated_{SHORT}.json"))
chains = raw if isinstance(raw, list) else raw.get("chains", raw.get("annotated", raw))
SIX = ["backtracking", "uncertainty-estimation", "example-testing",
       "adding-knowledge", "initializing", "deduction"]
print(f"{len(chains)} chains; extracting all 6 labels, clipped, mean -> {SAVE}")

model, tok = load_model(MODEL_ID, dtype=DTYPE)
extract_activations(
    model, tok, chains, layers=None, save_dir=SAVE,
    behaviours=SIX,
    pooling="mean", sweep_modes=[],
    clip_to_sentence_end=True,          # Venhoff-faithful span clipping
    keep_in_memory=False,
)
print("DONE: faithful 6-label clipped extraction ->", SAVE)
