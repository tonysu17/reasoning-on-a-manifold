#!/usr/bin/env python3
"""
Tier-1 R1.0 - Effective-N quantification (confound CF-2, the keystone).

The headline geometry uses 5k-16k *sentences* per behaviour, but those sentences
come from only ~1000 reasoning chains. Sentences inside one chain are
autocorrelated, so the i.i.d. assumption behind every intrinsic-dim / curvature
estimator is violated and the number of *independent* units is far below the raw
N. This script computes the honest denominator:

  - n_sentences, n_chains, sentences-per-chain distribution per behaviour
  - intraclass correlation (ICC(1)) of the activation signal at the steering
    layer, via a one-way random-effects decomposition on principal-component
    scores (PC1 headline + mean over top-K PCs as a sensitivity)
  - design effect  Deff = 1 + (n0 - 1) * ICC   and   n_eff = N / Deff

This does NOT re-derive any geometry number; it contextualises them. It only
reads existing Phase-4 activations + Phase-3 labels and writes under
results/tier1_robustness/ (touches nothing else). CPU, seconds. No GPU/API.

Usage:
  python tier1_effective_n.py --model-short R1-1.5B
  python tier1_effective_n.py --model-short R1-1.5B --layer 27 --n-pcs 10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("tier1_effN")

from src.annotation import TARGET_BEHAVIOURS          # noqa: E402
from src.config import STEERING_LAYERS                # noqa: E402


def _load_05b():
    """Import 05b_geometric_diagnostics by path (filename starts with a digit)
    purely to reuse its canonical `load_chain_id_map` — keeps the chain-id
    reconstruction (the find_sentence_offset filter) in ONE place so it cannot
    drift from the geometry pipeline."""
    spec = importlib.util.spec_from_file_location(
        "_p5b", ROOT / "05b_geometric_diagnostics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def icc_oneway(y: np.ndarray, inv: np.ndarray, counts: np.ndarray):
    """One-way random-effects ICC(1) for a scalar feature `y` whose group index
    per row is `inv` (0..G-1) and whose group sizes are `counts`.

    Unbalanced-group adjusted via n0 = (N - sum(n_g^2)/N) / (G-1).
    Returns (icc, n0, G). ICC can be slightly negative (clamped to 0 for Deff).
    """
    G = counts.size
    N = y.size
    if G < 2 or N <= G:
        return float("nan"), float("nan"), int(G)
    grand = y.mean()
    sums = np.bincount(inv, weights=y, minlength=G)
    gmeans = sums / counts
    ssb = float((counts * (gmeans - grand) ** 2).sum())
    ssw = float(((y - gmeans[inv]) ** 2).sum())
    msb = ssb / (G - 1)
    msw = ssw / (N - G)
    n0 = (N - (counts ** 2).sum() / N) / (G - 1)
    denom = msb + (n0 - 1) * msw
    icc = (msb - msw) / denom if denom > 0 else float("nan")
    return float(icc), float(n0), int(G)


def analyse_behaviour(X: np.ndarray, cids, n_pcs: int) -> dict:
    """Effective-N stats for one behaviour at one layer."""
    N = int(X.shape[0])
    out: dict = {"n_sentences": N, "d": int(X.shape[1])}

    if cids is None:
        out["error"] = "chain_ids unavailable (annotation file missing / no labels)"
        return out
    if len(cids) != N:
        # Row alignment is required for ICC; chain COUNT is still reported but
        # flagged. A mismatch means the find_sentence_offset filter did not
        # reproduce Phase 4 exactly for this behaviour.
        out["row_alignment_mismatch"] = {"n_chain_ids": int(len(cids)), "n_rows": N}

    uniq, inv, counts = np.unique(np.asarray(cids), return_inverse=True, return_counts=True)
    sizes = counts.astype(float)
    out["n_chains"] = int(uniq.size)
    out["sentences_per_chain"] = {
        "mean": float(sizes.mean()),
        "median": float(np.median(sizes)),
        "min": int(sizes.min()),
        "max": int(sizes.max()),
        "p90": float(np.percentile(sizes, 90)),
    }
    # Hard floor on independent units: one sentence per chain.
    out["n_eff_floor_n_chains"] = int(uniq.size)

    if "row_alignment_mismatch" in out:
        out["icc_note"] = "ICC skipped: rows and chain_ids are not aligned."
        return out

    # PC scores at this layer (svd_solver='full' for reproducibility, matching
    # the project convention in src/pca.py and src/nulls.py).
    from sklearn.decomposition import PCA
    k = int(min(n_pcs, X.shape[0] - 1, X.shape[1]))
    pca = PCA(n_components=k, svd_solver="full")
    scores = pca.fit_transform(X.astype(np.float64))
    var_ratio = pca.explained_variance_ratio_[:k]

    icc_per_pc, deff_per_pc, neff_per_pc = [], [], []
    n0_ref = None
    for j in range(k):
        icc, n0, _ = icc_oneway(scores[:, j], inv, counts.astype(float))
        n0_ref = n0
        icc_clamped = max(0.0, icc) if np.isfinite(icc) else float("nan")
        deff = 1.0 + (n0 - 1.0) * icc_clamped if np.isfinite(icc_clamped) else float("nan")
        neff = N / deff if (np.isfinite(deff) and deff > 0) else float("nan")
        icc_per_pc.append(icc)
        deff_per_pc.append(deff)
        neff_per_pc.append(neff)

    icc_pc1 = icc_per_pc[0]
    # variance-weighted mean ICC over the top-k PCs (a single summary number)
    w = var_ratio / var_ratio.sum() if var_ratio.sum() > 0 else np.ones(k) / k
    icc_vw = float(np.nansum(np.array(icc_per_pc) * w))

    out["icc"] = {
        "n0_adjusted_group_size": n0_ref,
        "pc1": icc_pc1,
        "var_weighted_topk": icc_vw,
        "per_pc": [float(v) for v in icc_per_pc],
        "explained_variance_ratio_per_pc": [float(v) for v in var_ratio],
    }
    # Design effect + effective N reported under both summaries.
    def _deff_neff(icc):
        icc_c = max(0.0, icc) if np.isfinite(icc) else float("nan")
        deff = 1.0 + (n0_ref - 1.0) * icc_c if np.isfinite(icc_c) else float("nan")
        neff = N / deff if (np.isfinite(deff) and deff > 0) else float("nan")
        return deff, neff
    deff1, neff1 = _deff_neff(icc_pc1)
    deffw, neffw = _deff_neff(icc_vw)
    out["design_effect"] = {
        "pc1": {"deff": deff1, "n_eff": neff1},
        "var_weighted_topk": {"deff": deffw, "n_eff": neffw},
    }
    return out


def write_md(payload: dict, path: Path) -> None:
    L = payload["layer"]
    rows = payload["per_behaviour"]
    lines = [
        f"# Tier-1 R1.0 - Effective N ({payload['model_short']}, layer {L})",
        "",
        "Confound CF-2 (chain / effective-N). Sentences within a chain are not",
        "independent; the honest denominator for every geometry estimator is",
        "closer to the number of *chains* than the number of *sentences*. ICC is",
        "the intraclass correlation of the PC scores at this layer; the design",
        "effect Deff = 1 + (n0-1)*ICC inflates variance, so n_eff = N / Deff.",
        "",
        "| Behaviour | N sent | N chains | sent/chain (mean / med / max) | ICC PC1 | ICC vw | Deff(PC1) | n_eff(PC1) |",
        "|-----------|-------:|---------:|------------------------------|--------:|-------:|----------:|-----------:|",
    ]
    for b in TARGET_BEHAVIOURS:
        r = rows.get(b)
        if r is None:
            continue
        if "error" in r:
            lines.append(f"| {b} | {r.get('n_sentences','-')} | - | _{r['error']}_ |  |  |  |  |")
            continue
        spc = r["sentences_per_chain"]
        icc = r.get("icc", {})
        de = r.get("design_effect", {}).get("pc1", {})
        def f(x, p=2):
            return f"{x:.{p}f}" if isinstance(x, (int, float)) and np.isfinite(x) else "-"
        lines.append(
            f"| {b} | {r['n_sentences']} | {r['n_chains']} | "
            f"{spc['mean']:.1f} / {spc['median']:.0f} / {spc['max']} | "
            f"{f(icc.get('pc1', float('nan')),3)} | {f(icc.get('var_weighted_topk', float('nan')),3)} | "
            f"{f(de.get('deff', float('nan')),1)} | {f(de.get('n_eff', float('nan')),0)} |")
    lines += [
        "",
        "**Reading it:** n_eff(PC1) is the effective independent sample size implied",
        "by the dominant direction's within-chain correlation. If n_eff collapses",
        "toward N chains, the raw N (5k-16k) is illusory and every CI / null built",
        "on raw N is over-confident (this is exactly what R1.1/R1.2 then correct).",
        "n_chains is the hard floor (one sentence per chain).",
        "",
        f"_Generated by tier1_effective_n.py; n_pcs={payload['n_pcs']}, seed-free (svd_solver=full)._",
    ]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-short", default="R1-1.5B")
    ap.add_argument("--layer", type=int, default=None,
                    help="Layer for the ICC (default: model steering layer).")
    ap.add_argument("--n-pcs", type=int, default=10,
                    help="Number of PCs to compute ICC over. Default 10.")
    args = ap.parse_args()

    layer = args.layer if args.layer is not None else STEERING_LAYERS.get(args.model_short, 27)
    act_dir = ROOT / f"data/activations/{args.model_short}"
    annotated_p = ROOT / f"data/annotated_{args.model_short}.json"
    out_dir = ROOT / f"results/tier1_robustness/{args.model_short}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not act_dir.exists():
        logger.error(f"Activations not found at {act_dir}.")
        sys.exit(1)

    p5b = _load_05b()
    logger.info(f"Loading chain_ids (sidecar-first) for {annotated_p.name} ...")
    chain_ids = p5b.load_chain_id_map(annotated_p, list(TARGET_BEHAVIOURS),
                                      act_dir=act_dir)

    per_behaviour = {}
    for b in TARGET_BEHAVIOURS:
        fpath = act_dir / f"{b}_layer{layer}.npy"
        if not fpath.exists():
            logger.warning(f"Skipping {b}: missing {fpath.name}")
            continue
        X = np.load(fpath)
        cids = chain_ids.get(b)
        logger.info(f"[{b}] N={X.shape[0]} d={X.shape[1]} "
                    f"chain_ids={'None' if cids is None else len(cids)}")
        per_behaviour[b] = analyse_behaviour(X, cids, args.n_pcs)

    payload = {
        "model_short": args.model_short,
        "layer": int(layer),
        "n_pcs": int(args.n_pcs),
        "confound": "CF-2 (chain / effective N)",
        "per_behaviour": per_behaviour,
    }
    json_path = out_dir / "effective_n.json"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path = out_dir / "effective_n.md"
    write_md(payload, md_path)

    print("\n" + "=" * 64)
    print(f"Tier-1 R1.0 effective-N complete - layer {layer}")
    print("=" * 64)
    for b, r in per_behaviour.items():
        if "error" in r:
            print(f"  {b:24s} {r['error']}")
            continue
        de = r.get("design_effect", {}).get("pc1", {})
        icc = r.get("icc", {}).get("pc1", float("nan"))
        neff = de.get("n_eff", float("nan"))
        print(f"  {b:24s} N={r['n_sentences']:6d}  chains={r['n_chains']:5d}  "
              f"ICC(PC1)={icc:6.3f}  n_eff(PC1)={neff:8.1f}")
    print(f"\nJSON: {json_path}")
    print(f"MD:   {md_path}")


if __name__ == "__main__":
    main()
