#!/usr/bin/env python3
"""P2/H1 — does a separable, low-dimensional DSR object exist in gpt-oss-20b,
and does it survive the capability control? Runs the sealed spec in
``results/safety/p2_h1/H1_PREREG.md`` (main repo). CPU-only, seed 0.

Usage: python3 p2_h1_analysis.py [--acts-root ...] [--out-root ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import provenance
from src.intrinsic_dim import levina_bickel_estimate, twoNN_estimate
from src.safety.capability import (
    DEFAULT_ALIGNMENT_MAX, DEFAULT_RETENTION_MIN, capability_direction, partial_out,
)
from src.safety.fingerprint import (
    bootstrap_separation_ci, layer_at_fraction, separation_heldout,
    separation_permutation_null,
)
from src.safety.refusal_direction import refusal_direction

SEED = 0
CITABLE = ["harm_recognition", "spec_citation", "decision"]
FRACTIONS = {"f25": 0.25, "f50": 0.50, "f75": 0.75}  # primary = f50


def load_class(root: Path, name: str, layers) -> tuple[dict, list]:
    rows = json.loads((root / f"{name}_rows.json").read_text())
    mats = {L: np.load(root / f"{name}_layer{L}.npy") for L in layers}
    return mats, rows


def dedupe_union(label_data: dict, layers) -> tuple[dict, list]:
    """Union of citable-label rows, deduped by (chain_id, text)."""
    seen, keep = set(), []          # keep: (label, row_idx)
    rows_out = []
    for lab in CITABLE:
        mats, rows = label_data[lab]
        for i, r in enumerate(rows):
            key = (r["chain_id"], r["text"])
            if key in seen:
                continue
            seen.add(key)
            keep.append((lab, i))
            rows_out.append({**r, "src_label": lab})
    union = {L: np.stack([label_data[lab][0][L][i] for lab, i in keep])
             for L in layers}
    return union, rows_out


def arm_map(chains_path: Path) -> dict:
    chains = json.loads(chains_path.read_text())
    return ({c["task_id"]: c["arm"] for c in chains},
            {c["task_id"]: c.get("difficulty") for c in chains})


def leg_b(X_d, g_d, X_g, g_g) -> dict:
    held = separation_heldout(X_d, X_g, g_d, g_g, n_folds=5, seed=SEED)
    perm = separation_permutation_null(X_d, X_g, n_perm=1000, seed=SEED)
    boot = bootstrap_separation_ci(X_d, X_g, n_boot=1000, seed=SEED)
    folds = held["cohens_d_folds"]
    pos = sum(1 for d in folds if d > 0)
    passed = (perm["p_value"] < 0.05 and held["n_folds_used"] >= 4
              and pos >= 4)
    return {"heldout": held, "permutation": perm, "bootstrap": boot,
            "folds_positive": pos, "PASS": bool(passed)}


def leg_a(X_d, X_g, rng) -> dict:
    n = X_d.shape[0]
    if n < 50:
        return {"verdict": "under-powered", "n": int(n)}
    two = twoNN_estimate(X_d)
    lb = levina_bickel_estimate(X_d)
    two_d = {"estimate": two.estimate, "ci": [two.ci_low, two.ci_high]}
    lb_d = {"estimate": lb.estimate, "ci": [lb.ci_low, lb.ci_high]}
    null = []
    for _ in range(100):
        idx = rng.choice(X_g.shape[0], size=n, replace=False)
        null.append(twoNN_estimate(X_g[idx], n_bootstrap=0).estimate)
    null = np.array(null)
    p5 = float(np.percentile(null, 5))
    # PCA-90 descriptive
    Xc = X_d - X_d.mean(0)
    sv = np.linalg.svd(Xc, compute_uv=False) ** 2
    pca90 = int(np.searchsorted(np.cumsum(sv) / sv.sum(), 0.90) + 1)
    return {"n": int(n), "twoNN": two_d, "levina_bickel": lb_d,
            "generic_null_twoNN": {"mean": float(null.mean()), "p5": p5,
                                   "p95": float(np.percentile(null, 95))},
            "pca90_dim": pca90,
            "low_dimensional": bool(two.estimate < p5)}


def leg_c(X_d, X_g, cap_hard, cap_easy) -> dict:
    cap_axis = capability_direction(cap_hard, cap_easy)
    safety_axis = refusal_direction(X_d, X_g)
    cos = float(abs(np.dot(safety_axis, cap_axis)))

    def insample_d(a, b):
        v = refusal_direction(a, b)
        pa, pb = a @ v, b @ v
        sp = np.sqrt((pa.var(ddof=1) + pb.var(ddof=1)) / 2)
        return float((pa.mean() - pb.mean()) / sp) if sp > 0 else 0.0

    d_raw = insample_d(X_d, X_g)
    d_ctl = insample_d(partial_out(X_d, cap_axis), partial_out(X_g, cap_axis))
    retention = float(d_ctl / d_raw) if d_raw != 0 else float("nan")
    return {"collinearity_abs_cos": cos, "d_raw": d_raw, "d_controlled": d_ctl,
            "retention": retention,
            "PASS": bool(retention >= DEFAULT_RETENTION_MIN
                         and cos <= DEFAULT_ALIGNMENT_MAX)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acts-root", default="../reasoning-on-manifold/results/safety/p2_activations")
    p.add_argument("--chains", default="../reasoning-on-manifold/data/chains_gpt-oss-20b_p0.json")
    p.add_argument("--out-root", default="../reasoning-on-manifold/results/safety/p2_h1")
    args = p.parse_args()

    root = Path(__file__).parent / args.acts_root
    out = Path(__file__).parent / args.out_root
    out.mkdir(parents=True, exist_ok=True)

    layer_ids = sorted({int(f.stem.rsplit("layer", 1)[1])
                        for f in root.glob("generic_layer*.npy")})
    layers = {k: layer_at_fraction(layer_ids, f) for k, f in FRACTIONS.items()}
    Ls = sorted(set(layers.values()))

    label_data = {lab: load_class(root, lab, Ls) for lab in CITABLE}
    gen_mats, gen_rows = load_class(root, "generic", Ls)
    dsr_mats, dsr_rows = dedupe_union(label_data, Ls)
    arms, difficulty = arm_map(Path(__file__).parent / args.chains)

    def arm_of(r):
        return arms.get(r["chain_id"], "?")

    rng = np.random.default_rng(SEED)
    results = {"layers": layers, "n_dsr_union": len(dsr_rows),
               "dsr_by_arm": {}, "generic_by_arm": {}}
    for r in dsr_rows:
        results["dsr_by_arm"][arm_of(r)] = results["dsr_by_arm"].get(arm_of(r), 0) + 1
    for r in gen_rows:
        results["generic_by_arm"][arm_of(r)] = results["generic_by_arm"].get(arm_of(r), 0) + 1

    # row selections (indices constant across layers)
    d_ben = [i for i, r in enumerate(dsr_rows) if arm_of(r) == "benign"]
    g_ben = [i for i, r in enumerate(gen_rows) if arm_of(r) == "benign"]
    cap_g = [i for i, r in enumerate(gen_rows) if arm_of(r) == "capability"]
    cap_hard_i = [i for i in cap_g if difficulty.get(gen_rows[i]["chain_id"]) == 5]
    cap_easy_i = [i for i in cap_g if difficulty.get(gen_rows[i]["chain_id"]) == 4]
    results["primary_n"] = {"dsr_benign": len(d_ben), "generic_benign": len(g_ben),
                            "cap_hard_rows": len(cap_hard_i),
                            "cap_easy_rows": len(cap_easy_i)}

    for tag, L in layers.items():
        Xd, Xg = dsr_mats[L], gen_mats[L]
        gd = [r["chain_id"] for r in dsr_rows]
        gg = [r["chain_id"] for r in gen_rows]
        block = {}
        # PRIMARY: benign-only
        block["primary_benign"] = leg_b(
            Xd[d_ben], [gd[i] for i in d_ben], Xg[g_ben], [gg[i] for i in g_ben])
        # secondary: all arms
        block["secondary_all"] = leg_b(Xd, gd, Xg, gg)
        if tag == "f50":
            block["leg_a_lowdim"] = leg_a(Xd[d_ben], Xg[g_ben], rng)
            block["leg_c_capability"] = leg_c(
                Xd[d_ben], Xg[g_ben], Xg[cap_hard_i], Xg[cap_easy_i])
            # descriptive per-label separations (benign-only, held-out)
            per = {}
            for lab in CITABLE:
                mats, rows = label_data[lab]
                sel = [i for i, r in enumerate(rows) if arms.get(r["chain_id"]) == "benign"]
                if len(sel) >= 5:
                    per[lab] = separation_heldout(
                        mats[L][sel], Xg[g_ben],
                        [rows[i]["chain_id"] for i in sel],
                        [gg[i] for i in g_ben], n_folds=5, seed=SEED)
            block["per_label_benign_heldout"] = per
        results[tag] = block

    # verdict (sealed rule): legs b AND c at primary layer, primary contrast
    b_pass = results["f50"]["primary_benign"]["PASS"]
    c_pass = results["f50"]["leg_c_capability"]["PASS"]
    a = results["f50"]["leg_a_lowdim"]
    results["VERDICT"] = {
        "H1": "SUPPORTED" if (b_pass and c_pass) else "NOT SUPPORTED",
        "leg_b_separation": bool(b_pass), "leg_c_capability": bool(c_pass),
        "leg_a_low_dimensional": a.get("low_dimensional", "under-powered"),
    }
    results["provenance"] = provenance(args=args)
    (out / "h1_results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results["VERDICT"], indent=1))
    print("primary layer:", layers["f50"], "| n:", results["primary_n"])
    print("wrote", out / "h1_results.json")


if __name__ == "__main__":
    main()
