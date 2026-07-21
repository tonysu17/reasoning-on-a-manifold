#!/usr/bin/env python3
"""pt13: minimum-detectable-effect for the CONTRACTION instrument (ledger §F7).

The spillover "translation without contraction" (§B2, STAR1) and "RLVR leaves
geometry unchanged" (§B5) claims both rest on one instrument's near-null: the
participation-ratio effective dimension delta, dPR = d_eff(post) - d_eff(base)
(``src.safety_posttrain.spillover.d_eff``). Observed dPR ranges -0.045 to -0.682
across the eight behaviour-by-layer cells. Without an injection-recovery curve
those small negatives are ambiguous: "no contraction" vs "a contraction below the
instrument's sensitivity". This is the exact analogue of pt05's rotation MDE
(a 5 deg injected rotation reads +0.55-0.72, which licensed "bounded, not null").

Method: inject a KNOWN contraction into each base matrix and record the dPR the
instrument reads. The observed dPR is negative, so the calibrating operation is
tail-variance removal (scale singular values beyond rank k by c<1): the spectrum
concentrates, d_eff falls, dPR goes negative -- the same shape a post-training
"collapse toward a lower-dimensional manifold" would produce. Reported against an
interpretable axis: the fraction of total variance the injected contraction
removes. d_eff depends only on the singular values, so no matrix reconstruction
is needed and the curve is exact and deterministic.

Run:  python3 pt13_contraction_mde.py
Out:  results/safety_posttrain/pt13_contraction_mde.json
"""

import json
from pathlib import Path

import numpy as np

from src.safety_posttrain.spillover import d_eff

ACT = Path("data/activations")
BASE = "R1-1.5B"
BEHAVIOURS = ["backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"]
LAYERS = [12, 16]
K = 5                                   # the subspace rank the geometry battery uses
C_GRID = [1.0, 0.9, 0.75, 0.5, 0.25, 0.0]   # tail-scale factors (1.0 = identity sanity)
OUT = Path("results/safety_posttrain/pt13_contraction_mde.json")

# Observed dPR from spillover_star1_full.json (the null being calibrated).
OBSERVED = {
    "backtracking_L12": -0.273, "uncertainty-estimation_L12": -0.241,
    "example-testing_L12": -0.045, "adding-knowledge_L12": -0.164,
    "backtracking_L16": -0.223, "uncertainty-estimation_L16": -0.317,
    "example-testing_L16": -0.310, "adding-knowledge_L16": -0.682,
}


def d_eff_from_lambda(lam: np.ndarray) -> float:
    denom = float((lam ** 2).sum())
    return float((lam.sum() ** 2) / denom) if denom > 0 else 0.0


def contraction_curve(X: np.ndarray, k: int, c_grid) -> dict:
    """dPR and variance-removed for a tail-shrink of factor c (svd-only, exact)."""
    Xc = X.astype(np.float64) - X.astype(np.float64).mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam0 = s ** 2
    d0 = d_eff_from_lambda(lam0)
    tot0 = lam0.sum()
    out = {"d_eff_base": round(d0, 3), "d_eff_from_svd_check": round(d_eff(X), 3),
           "n": int(X.shape[0]), "grid": {}}
    for c in c_grid:
        lam = lam0.copy()
        lam[k:] *= c ** 2                      # scale tail singular values by c -> variance by c^2
        dpr = d_eff_from_lambda(lam) - d0
        var_removed = 1.0 - float(lam.sum() / tot0)
        out["grid"][str(c)] = {"dPR": round(dpr, 4),
                               "variance_removed_frac": round(var_removed, 4)}
    return out


def map_observed(curve: dict, observed_dpr: float) -> dict:
    """Interpolate: what variance-removal fraction produces the observed dPR."""
    cs = sorted(curve["grid"], key=float, reverse=True)     # c=1.0 -> 0.0
    dprs = [curve["grid"][c]["dPR"] for c in cs]            # 0 -> most negative
    vrem = [curve["grid"][c]["variance_removed_frac"] for c in cs]
    # dPR is monotone decreasing along the grid; interpolate on -dPR
    x = [-d for d in dprs]
    y = vrem
    target = -observed_dpr
    if target <= x[0]:
        return {"variance_removed_at_observed": 0.0, "note": "observed dPR within identity noise"}
    if target >= x[-1]:
        return {"variance_removed_at_observed": vrem[-1], "note": "observed exceeds full-collapse grid"}
    vr = float(np.interp(target, x, y))
    return {"variance_removed_at_observed": round(vr, 4)}


def main():
    report = {"instrument": "participation-ratio d_eff; tail-shrink injection beyond rank k",
              "k": K, "c_grid": C_GRID, "base": BASE, "cells": {}}
    for beh in BEHAVIOURS:
        for L in LAYERS:
            key = f"{beh}_L{L}"
            X = np.load(ACT / BASE / f"{beh}_layer{L}.npy")
            curve = contraction_curve(X, K, C_GRID)
            curve["observed_dPR"] = OBSERVED.get(key)
            if curve["observed_dPR"] is not None:
                curve.update(map_observed(curve, curve["observed_dPR"]))
            report["cells"][key] = curve
            g = curve["grid"]
            print(f"{key}: base d_eff {curve['d_eff_base']:.1f} | "
                  f"dPR@c=0.9 {g['0.9']['dPR']:+.3f} (var-rm {g['0.9']['variance_removed_frac']:.1%}) | "
                  f"dPR@c=0.5 {g['0.5']['dPR']:+.3f} ({g['0.5']['variance_removed_frac']:.1%}) | "
                  f"observed {curve['observed_dPR']:+.3f} -> var-rm "
                  f"{curve.get('variance_removed_at_observed', float('nan')):.1%}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
