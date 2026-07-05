#!/usr/bin/env python3
"""Standalone side-by-side diagnostic: energy of the diff-of-means direction r in
the top-k POOLED-PCA subspace (Huang fix) vs the top-k ON-only PCA subspace (our
old construction), at the Venhoff per-behaviour layers. Computed directly from
activations because the build script's printed table is stuck in its stdout
buffer behind the slow robustness battery. cos(manifold_k, single) = sqrt(energy)
for an orthogonal projection of a unit r, so we report both.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from src.task_gen import load_tasks, stratified_eval_split
from src.row_provenance import chain_ids_for, require_aligned
from src.annotation import TARGET_BEHAVIOURS

ACT = Path("data/activations/R1-1.5B")
ANNOT = Path("data/annotated_R1-1.5B.json")
LAYERS = {"backtracking": 17, "uncertainty-estimation": 18,
          "example-testing": 15, "adding-knowledge": 18}

test, _ = stratified_eval_split(load_tasks(Path("data/tasks_final.json")), 50)
EXCL = {t["id"] for t in test}
cid = chain_ids_for(ACT, ANNOT, list(TARGET_BEHAVIOURS))


def load(b, L):
    X = np.load(ACT / f"{b}_layer{L}.npy").astype(np.float64)
    c = require_aligned(b, X.shape[0], cid.get(b), context="diag")
    return X[~np.isin(c, list(EXCL))]


def energy(comp, r):
    """||P_k r||^2 / ||r||^2 for orthonormal rows comp (k,d)."""
    co = comp @ r
    return float(co @ co) / float(r @ r)


print(f"{'behaviour':24}{'L':>3}  k |  cos_POOL  E_POOL  |  cos_only  E_only")
print("-" * 70)
for b, L in LAYERS.items():
    on = load(b, L)
    off = np.concatenate([load(o, L) for o in TARGET_BEHAVIOURS if o != b])
    r = on.mean(0) - off.mean(0)
    Pp = PCA(svd_solver="full").fit(np.vstack([on, off])).components_   # pooled
    Po = PCA(svd_solver="full").fit(on).components_                      # ON-only
    for k in (1, 3, 5, 10):
        Ep = energy(Pp[:k], r)
        Eo = energy(Po[:k], r)
        print(f"{b:24}{L:>3} {k:2d} |  {Ep**0.5:7.3f}  {Ep:6.3f}  |  {Eo**0.5:7.3f}  {Eo:6.3f}")
    print()
print("cos_POOL = cos(pooled-PCA manifold_k, single);  cos_only = old ON-only.")
