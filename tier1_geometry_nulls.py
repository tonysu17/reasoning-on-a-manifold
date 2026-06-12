#!/usr/bin/env python3
"""
Tier-1 R1.1 - The keystone null fix (confound CF-3).

The chain-stratified permutation null in 05b is run ONLY on the top-k variance
ratio (`05b:191`, statistic_fn=top_k_variance_ratio). But the load-bearing
geometry numbers are the *intrinsic dimension* (the compression gap) and the
*curvature* ratio. Those two have never been tested against the chain confound.
This script re-runs the SAME null machinery (src/nulls.py) with:

  statistic_fn in { twoNN intrinsic dim, Levina-Bickel intrinsic dim,
                    local-vs-global curvature ratio }

For each behaviour and statistic we report the real value, the chain-stratified
permutation null (the primary, CF-2-aware test: shuffle labels WITHIN chains,
preserving per-chain composition), the cross-chain null, and a one-sided
p-value. tail='lower' because the claims are "dim is LOW" and "ratio is LOW
(curved)" — real should sit *below* the null for the claim to hold.

This is CPU-only, reads existing Phase-4 activations + Phase-3 labels, and writes
only under results/tier1_robustness/. No GPU, no API, nothing overwritten.

Estimators run with n_bootstrap=0 (no nested CI) for speed inside the null;
CIs are R1.2's job (chain-block bootstrap).

Usage:
  # quick timing + sanity smoke
  python tier1_geometry_nulls.py --n-resamples 5 --statistics twonn levina
  # full run (background it)
  python tier1_geometry_nulls.py --n-resamples 200 --statistics twonn levina
  # add the (slower) curvature null
  python tier1_geometry_nulls.py --n-resamples 200 --statistics curvature --curv-k 30
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("tier1_nulls")

from src.annotation import TARGET_BEHAVIOURS              # noqa: E402
from src.config import STEERING_LAYERS                    # noqa: E402
from nulls import (chain_stratified_permutation_null,     # noqa: E402
                   cross_chain_permutation_null)
from intrinsic_dim import twoNN_estimate, levina_bickel_estimate   # noqa: E402
from curvature import local_vs_global_dim_ratio           # noqa: E402


def _load_05b():
    spec = importlib.util.spec_from_file_location(
        "_p5b", ROOT / "05b_geometric_diagnostics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_statistic(name: str, max_points: int, curv_k: int, n_anchors: int, pca_topk: int):
    """Return (fn, pretty_name, tail). fn: (np.ndarray)->float, fast (no inner CI)."""
    cap_rng = np.random.default_rng(20260608)

    def cap(X):
        if max_points and X.shape[0] > max_points:
            idx = cap_rng.choice(X.shape[0], max_points, replace=False)
            return X[idx]
        return X

    if name == "twonn":
        def f(X):
            X = cap(X)
            if X.shape[0] < 10:
                return float("nan")
            return twoNN_estimate(X, n_bootstrap=0).estimate
        return f, "twoNN_intrinsic_dim", "lower"

    if name == "levina":
        def f(X):
            X = cap(X)
            if X.shape[0] < 32:
                return float("nan")
            return levina_bickel_estimate(X, n_bootstrap=0).estimate
        return f, "levina_bickel_intrinsic_dim", "lower"

    if name == "curvature":
        from sklearn.decomposition import PCA

        def f(X):
            X = cap(X)
            if X.shape[0] < curv_k + 2:
                return float("nan")
            nc = min(pca_topk, X.shape[0] - 1, X.shape[1])
            Xp = PCA(n_components=nc, svd_solver="full").fit_transform(X.astype(np.float64))
            return local_vs_global_dim_ratio(
                Xp, k=curv_k, n_anchors=n_anchors, n_bootstrap=0).mean
        return f, f"curvature_local_vs_global_k{curv_k}", "lower"

    raise ValueError(f"unknown statistic {name!r}")


def build_layer_matrix(p5b, act_dir: Path, annotated_p: Path, layer: int):
    """Cross-behaviour activation matrix + chain ids + labels at one layer
    (same construction as 05b.run_null_hierarchy_at_layer).

    Hard-fails on chain-id misalignment (the old behaviour-as-chain proxy made
    the within-chain permutation a no-op) and removes exact-duplicate rows
    before the null — duplicates concentrate within behaviour labels, so the
    real per-behaviour matrix is duplicate-rich while permuted resamples are
    duplicate-poor, biasing 'real < null' anti-conservatively (CF-13). The
    June-8 layer-27 results predate this fix and are superseded.
    """
    from src.row_provenance import dedup_rows, duplicate_fraction, require_aligned

    acts, present = {}, []
    for b in TARGET_BEHAVIOURS:
        f = act_dir / f"{b}_layer{layer}.npy"
        if f.exists():
            acts[b] = np.load(f)
            present.append(b)
        else:
            logger.warning(f"missing {f.name}; {b} skipped at layer {layer}")
    chain_ids = p5b.load_chain_id_map(annotated_p, present, act_dir=act_dir)
    parts_X, parts_c, parts_l = [], [], []
    for b in present:
        X = acts[b]
        c = require_aligned(b, X.shape[0], chain_ids.get(b), context="tier1 nulls")
        parts_X.append(X)
        parts_c.append(np.asarray(c))
        parts_l.append(np.array([b] * X.shape[0]))
    X_all = np.vstack(parts_X)
    c_all = np.concatenate(parts_c)
    l_all = np.concatenate(parts_l)
    dup = duplicate_fraction(X_all)
    if dup > 0:
        logger.info(f"  removing {dup:.1%} exact-duplicate rows before the null (CF-13)")
        X_all, c_all, l_all = dedup_rows(X_all, c_all, l_all)
    return (X_all, c_all, l_all, present)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-short", default="R1-1.5B")
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="Layers (default: model steering layer).")
    ap.add_argument("--n-resamples", type=int, default=200)
    ap.add_argument("--statistics", nargs="+", default=["twonn", "levina"],
                    choices=["twonn", "levina", "curvature"])
    ap.add_argument("--max-points", type=int, default=0,
                    help="Subsample cap per statistic eval (0 = full N, honest but slower).")
    ap.add_argument("--curv-k", type=int, default=30)
    ap.add_argument("--n-anchors", type=int, default=200)
    ap.add_argument("--pca-topk", type=int, default=50,
                    help="PCA projection before curvature (matches 05b).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    layers = args.layers if args.layers is not None else [STEERING_LAYERS.get(args.model_short, 27)]
    act_dir = ROOT / f"data/activations/{args.model_short}"
    annotated_p = ROOT / f"data/annotated_{args.model_short}.json"
    out_dir = ROOT / f"results/tier1_robustness/{args.model_short}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not act_dir.exists():
        logger.error(f"Activations not found at {act_dir}.")
        sys.exit(1)

    p5b = _load_05b()
    stats = [make_statistic(s, args.max_points, args.curv_k, args.n_anchors, args.pca_topk)
             for s in args.statistics]

    for layer in layers:
        logger.info(f"=== Layer {layer}: building cross-behaviour matrix ===")
        X_all, chain_all, label_all, present = build_layer_matrix(p5b, act_dir, annotated_p, layer)
        logger.info(f"  X_all={X_all.shape}  chains={np.unique(chain_all).size}  behaviours={present}")

        results: dict = {}
        for b in present:
            results[b] = {}
            for fn, pretty, tail in stats:
                logger.info(f"  [{b}] {pretty}: chain-strat + cross-chain null "
                            f"(n_resamples={args.n_resamples}, tail={tail}) ...")
                t = time.time()
                cs = chain_stratified_permutation_null(
                    X_all, chain_all, label_all, b,
                    statistic_fn=fn, statistic_name=pretty,
                    n_resamples=args.n_resamples, random_state=args.seed, tail=tail)
                cc = cross_chain_permutation_null(
                    X_all, label_all, b,
                    statistic_fn=fn, statistic_name=pretty,
                    n_resamples=args.n_resamples, random_state=args.seed, tail=tail)
                dt = time.time() - t
                results[b][pretty] = {
                    "chain_strat": asdict(cs),
                    "cross_chain": asdict(cc),
                    "tail": tail,
                    "runtime_s": dt,
                }
                logger.info(f"    real={cs.real_value:.3f}  chain-strat null "
                            f"mean={cs.null_mean:.3f} p={cs.p_value:.4f}  "
                            f"cross-chain p={cc.p_value:.4f}  ({dt:.1f}s)")

        payload = {
            "model_short": args.model_short,
            "layer": int(layer),
            "n_resamples": args.n_resamples,
            "max_points": args.max_points,
            "statistics": args.statistics,
            "confound": "CF-3 (null tested the wrong statistic) + CF-2 (chain) + CF-13 (exact-duplicate rows removed before estimation)",
            "note": "tail='lower': claim is dim/curvature-ratio LOW, so real must sit BELOW the null.",
            "results": results,
        }
        json_path = out_dir / f"geometry_nulls_layer{layer}.json"
        json_path.write_text(json.dumps(payload, indent=2))
        write_md(payload, out_dir / f"geometry_nulls_layer{layer}.md")
        logger.info(f"  wrote {json_path.name}")

    print("\nDONE. Results under", out_dir)


def write_md(payload: dict, path: Path) -> None:
    L = payload["layer"]
    res = payload["results"]
    lines = [
        f"# Tier-1 R1.1 - Geometry nulls ({payload['model_short']}, layer {L})",
        "",
        "CF-3 fix: the chain-stratified null applied to the *intrinsic-dim* and",
        "*curvature* numbers (not just the variance ratio). tail='lower' — the",
        "claim holds only if the real value sits **below** the null and p is small.",
        "A p-value near the null mean (real ~= null) means the statistic is a",
        "property of the chains, not the behaviour.",
        "",
        f"n_resamples={payload['n_resamples']}, max_points={payload['max_points']}",
        "",
        "| Behaviour | Statistic | real | chain-strat null mean | p (chain-strat) | p (cross-chain) |",
        "|-----------|-----------|-----:|----------------------:|----------------:|----------------:|",
    ]
    for b, perstat in res.items():
        for pretty, r in perstat.items():
            cs = r["chain_strat"]
            cc = r["cross_chain"]
            def f(x, p=3):
                return f"{x:.{p}f}" if isinstance(x, (int, float)) and np.isfinite(x) else "-"
            lines.append(f"| {b} | {pretty} | {f(cs['real_value'])} | {f(cs['null_mean'])} | "
                         f"{f(cs['p_value'],4)} | {f(cc['p_value'],4)} |")
    lines += [
        "",
        "**Interpretation key.** chain-strat is the primary test (controls chain",
        "identity + composition + sample size). If real < null with small p, the",
        "behaviour carries genuinely lower-dimensional / more-curved structure than",
        "a chain-matched random relabelling. If p is large, the headline number was",
        "the chain confound (CF-2) wearing a behaviour label.",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
