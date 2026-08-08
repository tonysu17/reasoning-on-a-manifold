#!/usr/bin/env python3
"""ph0 s2 — STAR1 inert-control extraction wrapper (pod stage).

Freeze §1.3: extract deduction + initializing spans at layers {12,16} on byte-identical
input_ids. The R1-1.5B side already exists (data/activations/_volume_R1-1.5B_6label_archive/).

04_extract_activations.py extracts src.annotation.TARGET_BEHAVIOURS; this wrapper patches the
behaviour set to the two inert controls and writes to a FRESH short-name dir
(STAR1-1.5B-6label) so the existing STAR1-1.5B 4-behaviour metadata is never touched
(metadata.json skip-trap). Self-check asserts non-zero rows for both behaviours afterwards.
"""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

INERT = ("deduction", "initializing")
SHORT = "STAR1-1.5B-6label"

import src.annotation as _ann                    # noqa: E402
import src.activation_extraction as _ax          # noqa: E402

_ann.TARGET_BEHAVIOURS = tuple(INERT)
_ax.TARGET_BEHAVIOURS = tuple(INERT)             # rebind the from-import too

sys.argv = [
    "04_extract_activations.py",
    "--model", "star1-1.5b",
    "--tokenizer-alias", "1.5b",                 # C1 rule: byte-identical input_ids
    "--layers", "12", "16",
    "--short-name", SHORT,
]
runpy.run_path(str(Path(__file__).parent / "04_extract_activations.py"), run_name="__main__")

# POST-RUN NOTE (2026-08-08): with a registry --model, 04 derives the output dir from the
# registry short_name and IGNORES --short-name (that flag only applies with --model-path).
# The 2026-08-08 run therefore wrote to data/activations/STAR1-1.5B/ — which was EMPTY on the
# pod volume (July arms were verified-synced then cleaned), so nothing was clobbered; only this
# self-check's path was wrong, mislabelling a successful extraction as FAILED:s2.
out = Path(__file__).parent / "data/activations" / "STAR1-1.5B"
if not (out / "metadata.json").exists():
    out = Path(__file__).parent / "data/activations" / SHORT
meta = json.loads((out / "metadata.json").read_text())
n = meta.get("n_extracted", meta)
for b in INERT:
    rows = (n or {}).get(b, 0)
    assert rows and rows > 0, f"self-check FAILED: {b} has 0 extracted rows ({out})"
    print(f"self-check OK: {b} rows={rows}")
