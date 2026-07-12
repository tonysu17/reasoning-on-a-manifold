#!/usr/bin/env python3
"""pt06: per-seed direction analysis for the spillover recipe-specificity claim.

Three seeds per recipe (42 original, 43, 44 replicated on matched hardware):
within-recipe direction stability, cross-recipe separation, and per-seed alignment
to the full-SFT (STAR1) translation direction. Completes the seed-level hardening
the thesis declares pending. Directions = normalised mean row-paired displacement,
behaviours pooled, per layer (12, 16), as in pt04.

Run:  python3 pt06_perseed_directions.py
Out:  results/safety_posttrain/pt06_perseed.json
"""
import json
from itertools import combinations, product
from pathlib import Path

import numpy as np

BEH = ["backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"]
ACT = Path("data/activations")
SAFETY = {"42": "R1-1.5B-lora-safety1000", "43": "R1-1.5B-lora-safety1000-s43", "44": "R1-1.5B-lora-safety1000-s44"}
CONTROL = {"42": "R1-1.5B-lora-control1000", "43": "R1-1.5B-lora-control1000-s43", "44": "R1-1.5B-lora-control1000-s44"}
OUT = Path("results/safety_posttrain/pt06_perseed.json")


def direction(arm: str, layer: int) -> np.ndarray:
    D = np.concatenate([
        np.load(ACT / arm / f"{b}_layer{layer}.npy").astype(np.float64)
        - np.load(ACT / "R1-1.5B" / f"{b}_layer{layer}.npy").astype(np.float64)
        for b in BEH])
    return D.mean(axis=0)


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


report = {"seeds": list(SAFETY), "layers": {}}
for L in (12, 16):
    sd = {s: direction(a, L) for s, a in SAFETY.items()}
    cd = {s: direction(a, L) for s, a in CONTROL.items()}
    full = direction("STAR1-1.5B", L)

    within_s = {f"{a}-{b}": round(cos(sd[a], sd[b]), 4) for a, b in combinations(SAFETY, 2)}
    within_c = {f"{a}-{b}": round(cos(cd[a], cd[b]), 4) for a, b in combinations(CONTROL, 2)}
    cross = {f"s{a}-c{b}": round(cos(sd[a], cd[b]), 4) for a, b in product(SAFETY, CONTROL)}
    to_full_s = {s: round(cos(d, full), 4) for s, d in sd.items()}
    to_full_c = {s: round(cos(d, full), 4) for s, d in cd.items()}

    sep = min(to_full_s.values()) > max(to_full_c.values())
    report["layers"][str(L)] = {
        "within_safety": within_s, "within_control": within_c, "cross_recipe": cross,
        "cos_to_full_safety": to_full_s, "cos_to_full_control": to_full_c,
        "complete_separation_3v3": sep,
        "exact_perm_p_if_separated": round(1 / 20, 3) if sep else None,
        "gap_minS_maxC": round(min(to_full_s.values()) - max(to_full_c.values()), 4),
    }
    print(f"== L{L}")
    print(json.dumps(report["layers"][str(L)], indent=1))

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(report, open(OUT, "w"), indent=2)
print(f"written {OUT}")
