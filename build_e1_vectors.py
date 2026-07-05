#!/usr/bin/env python3
"""Rebuild steering vectors at the layers IDENTIFIED BY E1 (our faithful Venhoff
attribution patching on OUR data/annotator), rather than Venhoff's published
17/18/15/18. These are the *operational* vectors Phase-7 steers with if E1's
argmax is the chosen layer.

Mirror of build_venhoff_vectors.py in every respect (same builder, same 50-task
hold-out, same OFF=other-3 construction) EXCEPT the per-behaviour layer is read
from E1's results/venhoff_attribution/R1-1.5B/layer_effects.json -> argmax_layer.

Writes to results/steering_vectors/R1-1.5B__E1 so it does not clobber the Venhoff
anchor build (R1-1.5B__venhoff) or the canonical all-L27 build (R1-1.5B/). The
layer CHOICE is still a review gate (compare argmax vs published vs the L27
forward signal) — this script just makes the vectors ready the instant E1 lands.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from src.steering import build_steering_vectors, save_steering_vectors
from src.annotation import TARGET_BEHAVIOURS
from src.config import provenance
from src.task_gen import load_tasks, stratified_eval_split

PUBLISHED = {"backtracking": 17, "uncertainty-estimation": 18,
             "example-testing": 15, "adding-knowledge": 18}
E1_JSON = Path("results/venhoff_attribution/R1-1.5B/layer_effects.json")
ACT = Path("data/activations/R1-1.5B")
OUT = Path("results/steering_vectors/R1-1.5B__E1")

if not E1_JSON.exists():
    sys.exit(f"E1 results not found at {E1_JSON} — run 07e first.")

curves = json.loads(E1_JSON.read_text())
E1_LAYERS = {}
for b in TARGET_BEHAVIOURS:
    d = curves.get(b)
    if not isinstance(d, dict) or "argmax_layer" not in d:
        sys.exit(f"No argmax_layer for {b} in {E1_JSON}")
    E1_LAYERS[b] = int(d["argmax_layer"])

print("E1-identified layers (argmax) vs published:")
for b in TARGET_BEHAVIOURS:
    mark = "==" if E1_LAYERS[b] == PUBLISHED[b] else "!="
    print(f"  {b:24s} E1={E1_LAYERS[b]:>2d}  {mark}  published={PUBLISHED[b]:>2d}")

_test_tasks, _rule = stratified_eval_split(load_tasks(Path("data/tasks_final.json")), 50)
EXCLUDE = {t["id"] for t in _test_tasks}
print(f"Hold-out: excluding {len(EXCLUDE)} eval tasks ({_rule}) from vector construction")

by_layer = defaultdict(list)
for b, L in E1_LAYERS.items():
    by_layer[L].append(b)

assembled = {}
for L in sorted(by_layer):
    res = build_steering_vectors(ACT, layer=L, exclude_chain_ids=EXCLUDE,
                                 annotated_path=Path("data/annotated_R1-1.5B.json"))
    for b in by_layer[L]:
        if b in res:
            assembled[b] = res[b]

prov = provenance()
prov["builder"] = "build_e1_vectors.py (layers = E1 faithful-attribution argmax)"
prov["layers"] = E1_LAYERS
prov["published_layers"] = PUBLISHED
prov["holdout"] = {"n_tasks": len(EXCLUDE), "rule": "src.task_gen.stratified_eval_split"}
save_steering_vectors(assembled, OUT, provenance=prov)

print("\n" + "=" * 96)
print(f"{'behaviour':24s} {'layer':>5s} {'auto_k':>6s} {'n_on':>6s} {'n_off':>7s} "
      f"{'cos(single,k_auto)':>18s} {'cos(single,k1)':>14s} {'E_in_manifold':>13s}")
print("-" * 96)
for b in TARGET_BEHAVIOURS:
    d = assembled[b]
    r = np.asarray(d["single_direction"])
    r_auto = np.asarray(d["manifold_projected"]["auto"])
    r_k1 = np.asarray(d["manifold_projected"][1])
    cos_auto = float(np.dot(r, r_auto))
    cos_k1 = float(np.dot(r, r_k1))
    print(f"{b:24s} {d['layer']:>5d} {d['auto_k']:>6d} {d['n_on']:>6d} {d['n_off']:>7d} "
          f"{cos_auto:>18.4f} {cos_k1:>14.4f} {cos_auto**2:>13.4f}")
print(f"\nSaved -> {OUT}")
