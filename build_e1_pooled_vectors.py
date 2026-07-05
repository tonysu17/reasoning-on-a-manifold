#!/usr/bin/env python3
"""Build POOLED-PCA (Huang) manifold steering vectors at the E1-argmax
per-behaviour layers — the layers WE determined via faithful Venhoff attribution
(results/venhoff_attribution/R1-1.5B/layer_effects.json):

    backtracking 17,  uncertainty-estimation 15,  example-testing 15,  adding-knowledge 17

k={3,5} (the E4 stability/energy sweet spot). Pooled construction (E3/E4: better
than ON-only). Writes results/steering_vectors/R1-1.5B__E1_pooled for the E8 grid.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from src.huang_manifold import build_huang_manifold_vectors, save_steering_vectors
from src.annotation import TARGET_BEHAVIOURS
from src.config import provenance
from src.task_gen import load_tasks, stratified_eval_split

E1_JSON = Path("results/venhoff_attribution/R1-1.5B/layer_effects.json")
ACT = Path("data/activations/R1-1.5B")
OUT = Path("results/steering_vectors/R1-1.5B__E1_pooled")
K_VALUES = [3, 5]

curves = json.loads(E1_JSON.read_text())
layers = {}
for b in TARGET_BEHAVIOURS:
    d = curves.get(b)
    if not isinstance(d, dict) or "argmax_layer" not in d:
        sys.exit(f"no argmax_layer for {b} in {E1_JSON}")
    layers[b] = int(d["argmax_layer"])
print("E1-argmax layers:", layers)

_test, _rule = stratified_eval_split(load_tasks(Path("data/tasks_final.json")), 50)
EXCLUDE = {t["id"] for t in _test}
print(f"hold-out: excluding {len(EXCLUDE)} eval tasks ({_rule})")

# Pass ALL 6 labels so OFF = other-5 (incl. initializing + deduction) and the
# pooled PCA spans the full 6-label reasoning cloud. Then
#   single = unit(μ_b − μ_other-5) == the FAITHFUL Venhoff unit(μ_b − μ_overall_6)
# (they differ only by a positive scalar N_rest/N_total → identical after
# unit-normalisation). Only the 4 TARGET behaviours (those in `layers`) are built
# as steering arms; init/deduction enter only as OFF / pooled-PCA rows.
ALL_LABELS = ["initializing", "deduction", "adding-knowledge",
              "example-testing", "uncertainty-estimation", "backtracking"]
assembled = build_huang_manifold_vectors(
    ACT, layers=layers, behaviours=ALL_LABELS,
    k_values=K_VALUES, huang_k=5, exclude_chain_ids=EXCLUDE,
    annotated_path=Path("data/annotated_R1-1.5B.json"))

prov = provenance()
prov["builder"] = ("build_e1_pooled_vectors.py (pooled manifold @ E1-argmax layers, k={3,5}; "
                   "single = Venhoff 6-label overall: OFF=other-5 incl initializing+deduction)")
prov["layers"] = layers
prov["k_values"] = K_VALUES
prov["holdout"] = {"n_tasks": len(EXCLUDE), "rule": "src.task_gen.stratified_eval_split"}
save_steering_vectors(assembled, OUT, provenance=prov)

print("\nsaved ->", OUT)
for b in TARGET_BEHAVIOURS:
    if b not in assembled:
        print(f"  {b:24s} MISSING"); continue
    d = assembled[b]
    r = np.asarray(d["single_direction"]); r3 = np.asarray(d["manifold_projected"][3])
    print(f"  {b:24s} L{d['layer']:<3d} auto_k={d.get('auto_k')} n_on={d.get('n_on')} "
          f"cos(single,k3)={float(np.dot(r, r3)):.3f}")
