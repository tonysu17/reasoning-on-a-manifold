#!/usr/bin/env python3
"""Path A — rebuild steering vectors at Venhoff's EXACT published per-behaviour
layers for DeepSeek-R1-Distill-Qwen-1.5B (arXiv:2506.18167, Table 2 /
utils.steering_config):

    backtracking            -> layer 17
    uncertainty-estimation  -> layer 18
    example-testing         -> layer 15
    adding-knowledge        -> layer 18

Writes to a DISTINCT dir (results/steering_vectors/R1-1.5B__venhoff) so it does
not clobber the canonical all-L27 build (R1-1.5B/) or the PR-trough -peak build
(R1-1.5B-peak/). Reuses src.steering.build_steering_vectors, so the construction
is OUR current apparatus (single = diff-of-means vs the other-3 behaviours,
unit-norm; manifold = top-k PCA of the ON activations, k in {1,3,5,10,auto}) at
Venhoff's layers. NOTE: a *pure* Venhoff replication would also use his vector
definition (behaviour-mean - OVERALL-mean) + additive steering; that is handled
alongside the attribution-patching reimplementation. Same 50-task hold-out as
06/07 via the shared split.
"""
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from src.steering import build_steering_vectors, save_steering_vectors
from src.annotation import TARGET_BEHAVIOURS
from src.config import provenance
from src.task_gen import load_tasks, stratified_eval_split

VENHOFF_LAYERS = {
    "backtracking": 17,
    "uncertainty-estimation": 18,
    "example-testing": 15,
    "adding-knowledge": 18,
}
ACT = Path("data/activations/R1-1.5B")
OUT = Path("results/steering_vectors/R1-1.5B__venhoff")

_test_tasks, _rule = stratified_eval_split(load_tasks(Path("data/tasks_final.json")), 50)
EXCLUDE = {t["id"] for t in _test_tasks}
print(f"Hold-out: excluding {len(EXCLUDE)} eval tasks ({_rule}) from vector construction")

by_layer = defaultdict(list)
for b, L in VENHOFF_LAYERS.items():
    by_layer[L].append(b)

assembled = {}
for L in sorted(by_layer):
    res = build_steering_vectors(ACT, layer=L, exclude_chain_ids=EXCLUDE,
                                 annotated_path=Path("data/annotated_R1-1.5B.json"))
    for b in by_layer[L]:
        if b in res:
            assembled[b] = res[b]

prov = provenance()
prov["builder"] = "build_venhoff_vectors.py (Venhoff Table-2 per-behaviour layers 17/18/15/18)"
prov["layers"] = VENHOFF_LAYERS
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
