#!/usr/bin/env python3
"""Make auto-only copies of the L16/L27 vector dirs for the trimmed bake-off:
keep single_direction + manifold_kauto, drop manifold_k{1,3,5,10}. Edits the
metadata k_values so the (metadata-driven) loader only pulls the auto vector."""
import json
from pathlib import Path
import numpy as np
from src.steering import load_steering_vectors

for L in ["L16", "L27"]:
    d = Path(f"results/steering_vectors/R1-1.5B__{L}_auto")
    for k in ["1", "3", "5", "10"]:
        for f in d.glob(f"*_manifold_k{k}.npy"):
            f.unlink()
    m = json.load(open(d / "metadata.json"))
    for b in m:
        if not b.startswith("_") and isinstance(m[b], dict):
            m[b]["k_values"] = ["auto"]
    json.dump(m, open(d / "metadata.json", "w"), indent=2)
    v = load_steering_vectors(d)
    arms, bad = set(), 0
    for b, dd in v.items():
        if isinstance(dd, dict):
            arms |= set(dd.get("manifold_projected", {}).keys())
            if not np.isfinite(np.asarray(dd["single_direction"])).all():
                bad += 1
    print(f"{L}_auto: behaviours={len(v)} manifold_ks={sorted(map(str,arms))} "
          f"layer={v['backtracking']['layer']} nonfinite={bad}")
