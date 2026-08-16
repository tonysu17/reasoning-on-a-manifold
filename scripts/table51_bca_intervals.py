"""Paired BCa intervals for Table 5.1 (thesis Priority 4 / T3-13).

Endpoint: paired steered-minus-unsteered change in the annotated target-behaviour
fraction, per (behaviour, operator). Fraction and pairing follow
src/evaluation.py: behaviour_fraction = n_target / n_annotated_spans, and the
shared vanilla generation is the baseline for every behaviour on the same task.

Gate: reproduce the published point estimates in Table 5.1 before reporting
intervals. If the reproduction fails, nothing is written.
"""
import json, sys
from collections import defaultdict
import numpy as np
from scipy.stats import norm

ROOT = "/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold"
ANN = f"{ROOT}/results/eval/R1-1.5B__E1/annotated_steered.json"

TARGETS = ["backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"]
OPS = ["single_direction", "manifold_k3", "manifold_k5"]
SHARED, VANILLA = "shared", "vanilla"

# Published Table 5.1 values: (pp change, relative %, n) keyed (behaviour, op)
PUBLISHED = {
    ("backtracking", "single_direction"): (-5.52, -50.7, 49),
    ("backtracking", "manifold_k3"):      (-3.81, -35.0, 49),
    ("backtracking", "manifold_k5"):      (-5.74, -52.1, 48),
    ("uncertainty-estimation", "single_direction"): (-2.49, -11.6, 49),
    ("uncertainty-estimation", "manifold_k3"):      (+0.01, +0.04, 47),
    ("uncertainty-estimation", "manifold_k5"):      (+2.05, +9.6, 49),
    ("example-testing", "single_direction"): (-4.45, -47.3, 49),
    ("example-testing", "manifold_k3"):      (-1.01, -10.8, 49),
    ("example-testing", "manifold_k5"):      (-1.35, -14.3, 49),
    ("adding-knowledge", "single_direction"): (-0.10, -1.6, 48),
    ("adding-knowledge", "manifold_k3"):      (-0.89, -14.2, 49),
    ("adding-knowledge", "manifold_k5"):      (+1.27, +19.8, 48),
}


def behaviour_fraction(anns, target):
    if not anns:
        return None
    return sum(1 for a in anns if a["label"] == target) / len(anns)


def load():
    recs = json.load(open(ANN))
    idx = {}
    for r in recs:
        if not r.get("annotation_complete"):
            continue
        anns = r.get("annotations")
        if not anns:
            continue
        idx[(r["task_id"], r["behaviour"], r["method"])] = anns
    return idx


def paired(idx, beh, op):
    """Return (deltas, vanilla_fractions) over tasks with both arms annotated."""
    d, v = [], []
    for (task, b, m), anns in idx.items():
        if b != beh or m != op:
            continue
        van = idx.get((task, SHARED, VANILLA))
        if van is None:
            continue
        fa, fv = behaviour_fraction(anns, beh), behaviour_fraction(van, beh)
        if fa is None or fv is None:
            continue
        d.append(fa - fv); v.append(fv)
    return np.array(d), np.array(v)


def bca(x, B=10000, alpha=0.05, seed=42):
    """BCa interval for the mean of paired differences x."""
    rng = np.random.default_rng(seed)
    n = len(x)
    theta = x.mean()
    boot = np.array([rng.choice(x, n, replace=True).mean() for _ in range(B)])
    # bias correction
    prop = (boot < theta).mean()
    prop = min(max(prop, 1.0 / B), 1 - 1.0 / B)
    z0 = norm.ppf(prop)
    # acceleration via jackknife
    jack = np.array([np.delete(x, i).mean() for i in range(n)])
    jbar = jack.mean()
    num = ((jbar - jack) ** 3).sum()
    den = 6.0 * (((jbar - jack) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0
    zl, zu = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    def adj(z):
        return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo, hi = adj(zl), adj(zu)
    return float(np.quantile(boot, lo)), float(np.quantile(boot, hi)), float(theta)


def main():
    idx = load()
    print(f"annotated records indexed: {len(idx)}\n")
    rows, failures = [], []
    print(f"{'cell':46s} {'n':>3s} {'pp':>7s} {'pub':>7s} {'rel%':>7s} {'pub':>7s}  gate")
    for beh in TARGETS:
        for op in OPS:
            d, v = paired(idx, beh, op)
            if len(d) == 0:
                failures.append((beh, op, "no paired tasks")); continue
            pp = d.mean() * 100
            rel = (d.mean() / v.mean() * 100) if v.mean() > 0 else float("nan")
            pub_pp, pub_rel, pub_n = PUBLISHED[(beh, op)]
            ok_pp = abs(pp - pub_pp) <= 0.02
            ok_n = len(d) == pub_n
            gate = "OK" if (ok_pp and ok_n) else f"MISMATCH(pp={ok_pp},n={ok_n})"
            if not (ok_pp and ok_n):
                failures.append((beh, op, gate))
            print(f"{beh+'/'+op:46s} {len(d):3d} {pp:7.2f} {pub_pp:7.2f} "
                  f"{rel:7.1f} {pub_rel:7.1f}  {gate}")
            rows.append((beh, op, d, v, pp, rel, len(d)))
    if failures:
        print("\nGATE FAILED — point estimates not reproduced; no intervals written.")
        for f in failures:
            print("  ", f)
        sys.exit(1)
    print("\nGate passed: all 12 point estimates and paired counts reproduced.\n")
    out = {}
    print(f"{'cell':46s} {'pp':>7s}  {'BCa 95%':>20s}   excl 0")
    for beh, op, d, v, pp, rel, n in rows:
        lo, hi, th = bca(d)
        lo_pp, hi_pp = lo * 100, hi * 100
        excl = (lo_pp < 0 and hi_pp < 0) or (lo_pp > 0 and hi_pp > 0)
        out[f"{beh}|{op}"] = {"n": n, "pp": round(pp, 2), "rel_pct": round(rel, 1),
                              "bca_lo_pp": round(lo_pp, 2), "bca_hi_pp": round(hi_pp, 2),
                              "excludes_zero": bool(excl), "B": 10000, "seed": 42}
        print(f"{beh+'/'+op:46s} {pp:7.2f}  [{lo_pp:8.2f}, {hi_pp:8.2f}]   {excl}")
    dest = f"{ROOT}/results/eval/R1-1.5B__E1/table51_bca_intervals.json"
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nwritten {dest}")


if __name__ == "__main__":
    main()
