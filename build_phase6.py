#!/usr/bin/env python3
"""Phase 6 — build steering vectors, each behaviour at its manifold-peak (PR-trough) layer.

Output dir is deliberately DISTINCT from 06_build_steering.py's
(results/steering_vectors/R1-1.5B = the canonical all-behaviours-at-layer-27
build): the two builders write identical filenames, so sharing a directory
meant whichever ran last silently clobbered the other and Phase 7 evaluated
whichever geometry happened to be on disk. Point 07 at this dir explicitly to
evaluate the per-behaviour-peak variant."""
import sys, json
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from src.steering import build_steering_vectors, save_steering_vectors
from src.annotation import TARGET_BEHAVIOURS

from src.config import PEAK_LAYERS as PEAK, provenance, require_file  # single source

ACT = Path("data/activations/R1-1.5B")
OUT = Path("results/steering_vectors/R1-1.5B-peak")
require_file(ACT, "run 04_extract_activations.py first to produce activations")

by_layer = defaultdict(list)
for b, L in PEAK.items():
    by_layer[L].append(b)

assembled = {}
for L in sorted(by_layer):
    res = build_steering_vectors(ACT, layer=L)        # all behaviours at L; off-set = others at L
    for b in by_layer[L]:
        if b in res:
            assembled[b] = res[b]

prov = provenance()
prov["builder"] = "build_phase6.py (per-behaviour peak layers)"
save_steering_vectors(assembled, OUT, provenance=prov)

print("\n" + "="*92)
print(f"{'behaviour':24s} {'layer':>5s} {'auto_k':>6s} {'n_on':>6s} {'n_off':>7s} "
      f"{'cos(single,manifold)':>20s} {'energy in manifold':>18s}")
print("-"*92)
rows = {}
for b in TARGET_BEHAVIOURS:
    d = assembled[b]
    r = d["single_direction"]
    rp = d["manifold_projected"]["auto"]
    cos = float(np.dot(r, rp))           # = ||proj(r)|| since r unit, rp normalized projection
    energy = cos**2                       # fraction of DOM-direction energy inside the manifold
    cos1 = float(np.dot(r, d["manifold_projected"][1]))
    cos10 = float(np.dot(r, d["manifold_projected"][10]))
    rows[b] = dict(layer=d["layer"], auto_k=d["auto_k"], n_on=d["n_on"], n_off=d["n_off"],
                   cos_auto=cos, energy=energy, cos_k1=cos1, cos_k10=cos10)
    print(f"{b:24s} {d['layer']:>5d} {d['auto_k']:>6d} {d['n_on']:>6d} {d['n_off']:>7d} "
          f"{cos:>20.3f} {energy*100:>16.1f}%")
print("="*92)
print("\ncos(single, manifold@k) convergence — how the projection sharpens as k grows:")
print(f"{'behaviour':24s} {'k=1':>7s} {'k=10':>7s} {'k=auto':>7s}")
for b in TARGET_BEHAVIOURS:
    print(f"{b:24s} {rows[b]['cos_k1']:>7.3f} {rows[b]['cos_k10']:>7.3f} {rows[b]['cos_auto']:>7.3f}")

json.dump(rows, open(OUT/"phase6_comparison.json","w"), indent=2)
print(f"\nSaved vectors + comparison -> {OUT}")
