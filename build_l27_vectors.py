#!/usr/bin/env python3
"""Build steering vectors at L27 (the E2 forward-causal layer + Huang's published
layer + sealed-α* layer) for the E8 steering grid, TRIMMED to k∈{3,5} — the E4
stability/energy sweet spot — to keep generation tractable (k=1 weak, k=10/auto
reach the unstable spectral tail). Same 50-task hold-out + OFF=other-3 apparatus
as the other builders. Writes results/steering_vectors/R1-1.5B__L27_k35."""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from src.steering import build_steering_vectors, save_steering_vectors
from src.annotation import TARGET_BEHAVIOURS
from src.config import provenance
from src.task_gen import load_tasks, stratified_eval_split

LAYER = 27
K_VALUES = [3, 5]
ACT = Path("data/activations/R1-1.5B")
OUT = Path("results/steering_vectors/R1-1.5B__L27_k35")

_test, _rule = stratified_eval_split(load_tasks(Path("data/tasks_final.json")), 50)
EXCLUDE = {t["id"] for t in _test}
print(f"Hold-out: excluding {len(EXCLUDE)} eval tasks ({_rule})")

res = build_steering_vectors(ACT, layer=LAYER, exclude_chain_ids=EXCLUDE,
                             annotated_path=Path("data/annotated_R1-1.5B.json"),
                             k_values=K_VALUES)
assembled = {b: res[b] for b in TARGET_BEHAVIOURS if b in res}

prov = provenance()
prov["builder"] = "build_l27_vectors.py (L27, k={3,5}, E8 generation build)"
prov["layers"] = {b: LAYER for b in TARGET_BEHAVIOURS}
prov["k_values"] = K_VALUES
prov["holdout"] = {"n_tasks": len(EXCLUDE), "rule": "src.task_gen.stratified_eval_split"}
save_steering_vectors(assembled, OUT, provenance=prov)

print("\n" + "=" * 80)
for b in TARGET_BEHAVIOURS:
    if b not in assembled:
        print(f"{b:24s} MISSING"); continue
    d = assembled[b]
    r = np.asarray(d["single_direction"])
    r3 = np.asarray(d["manifold_projected"][3])
    print(f"{b:24s} L{d['layer']:<3d} auto_k={d['auto_k']:<3d} n_on={d['n_on']:<6d} "
          f"cos(single,k3)={float(np.dot(r, r3)):.4f}")
print(f"\nSaved -> {OUT}")
