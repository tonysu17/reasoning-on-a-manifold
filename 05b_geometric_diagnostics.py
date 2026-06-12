#!/usr/bin/env python3
"""
Phase 5b - Geometric diagnostics with null-hypothesis hierarchy.

Extends Phase 5 (05_pca_analysis.py) with the methodology pre-registered in
the revised companion document, Section 2.5:

  - Intrinsic dimension estimators (TwoNN, Levina-Bickel, correlation dim)
  - Curvature diagnostics (local-vs-global dim ratio, geodesic/Euclidean,
    tangent-space variation)
  - Null-hypothesis hierarchy (chain-stratified permutation, cross-chain
    permutation, MP isotropic diagnostic)

For each target behaviour at the focus layer, computes the full battery and
writes JSON results plus a summary table.

Inputs:
  data/activations/<model>/<behaviour>_layer<N>.npy  (from Phase 4)
  data/annotated_<model>.json                         (for chain_id-per-sentence)

Outputs:
  results/geometric/<model>/diagnostics_layer<N>.json
  results/geometric/<model>/summary_layer<N>.md

Runtime: ~5-15 minutes on CPU per layer (dominated by null Monte Carlo).

Usage:
  python 05b_geometric_diagnostics.py --layer 27
  python 05b_geometric_diagnostics.py --layer 27 --n-resamples 100 --k 10 30 100
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from intrinsic_dim import all_id_estimators
from curvature     import all_diagnostics, summary_table
from nulls         import full_null_hierarchy, top_k_variance_ratio, participation_ratio

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# Constants matching 05_pca_analysis.py for consistency

from src.annotation import TARGET_BEHAVIOURS
from src.config import STEERING_LAYERS  # single source: configs/config.yaml
                                        # (previously omitted QwenMath-1.5B here)


# Activation + chain-id loading

def load_activations(act_dir: Path, behaviour: str, layer: int) -> np.ndarray:
    """Load the per-behaviour activation matrix for one layer."""
    fname = act_dir / f"{behaviour}_layer{layer}.npy"
    if not fname.exists():
        raise FileNotFoundError(f"Missing activation file: {fname}")
    return np.load(fname)


# Sidecar-first row provenance + duplicate hygiene (single source of truth;
# replaces the replay loader and the silent proxy-chain fallback — CF-13).
from src.row_provenance import (
    chain_ids_for,
    dedup_rows,
    duplicate_fraction,
    require_aligned,
)


def load_chain_id_map(annotated_path: Path, target_behaviours,
                      act_dir: "Path | None" = None) -> dict:
    """Chain_id per activation row per behaviour.

    Prefers the extraction-time row_index.json sidecar in *act_dir* (exact by
    construction); falls back to an occurrence-aware replay of the annotation
    file for legacy extractions. Returns dict {behaviour: ndarray | None} —
    validate with src.row_provenance.require_aligned before use.
    """
    return chain_ids_for(act_dir if act_dir is not None else Path("."),
                         annotated_path, list(target_behaviours))


# Diagnostics per behaviour at one layer

def diagnose_behaviour(
    X:             np.ndarray,
    chain_ids:     "np.ndarray | None",
    behaviour:     str,
    n_resamples:   int,
    k_values:      tuple,
    intrinsic_dim_for_tangent: int,
    pca_topk_for_curvature:    int,
    variance_threshold:        float = 0.90,
) -> dict:
    """Run the full diagnostic battery on one behaviour's activation matrix."""
    out: dict = {"N_raw": int(X.shape[0]), "d": int(X.shape[1])}

    # Exact-duplicate hygiene BEFORE any kNN-based estimator: duplicate rows
    # put zero-distance neighbours into TwoNN/LB/geodesic graphs (CF-13).
    out["duplicate_fraction"] = duplicate_fraction(X)
    if out["duplicate_fraction"] > 0:
        (X,) = dedup_rows(X)
        logger.info(f"  [{behaviour}] removed {out['duplicate_fraction']:.1%} "
                    f"exact-duplicate rows → N={X.shape[0]}")
    out["N"] = int(X.shape[0])

    # ID estimators on deduplicated data
    logger.info(f"  [{behaviour}] intrinsic dimension estimators (N={X.shape[0]})...")
    t = time.time()
    id_results = all_id_estimators(X, n_bootstrap=100)
    out["intrinsic_dim"] = [asdict(r) for r in id_results]
    out["intrinsic_dim_runtime_s"] = time.time() - t

    # Project to top-K PCA subspace before curvature (per companion 2.5 caveat
    # to avoid ambient-dim noise dominating)
    logger.info(f"  [{behaviour}] projecting to top-{pca_topk_for_curvature} PCA for curvature...")
    from sklearn.decomposition import PCA
    n_components = min(pca_topk_for_curvature, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components)
    X_proj = pca.fit_transform(X)

    # Curvature
    logger.info(f"  [{behaviour}] curvature diagnostics (k sweep)...")
    t = time.time()
    curv = all_diagnostics(X_proj, k_values=k_values,
                            intrinsic_dim=intrinsic_dim_for_tangent,
                            variance_threshold=variance_threshold,
                            n_bootstrap=100)
    out["curvature"] = [asdict(r) for r in curv]
    out["curvature_runtime_s"] = time.time() - t

    # Null hierarchy on the top-10 variance ratio
    # (Note: full hierarchy needs the labels for ALL sentences, not just this behaviour,
    #  so it's run at the caller level, not here.)
    return out


def run_null_hierarchy_at_layer(
    activations_by_behaviour: dict,
    chain_ids_by_behaviour:   dict,
    n_resamples:              int,
) -> dict:
    """Build the cross-behaviour activation matrix and labels, then run the
    null hierarchy for each target behaviour.

    This requires the activation matrices to be concatenable across behaviours
    (same dimension). The label vector marks which sentences belong to which
    behaviour. Chain IDs are concatenated in matching order.
    """
    behaviours = sorted(activations_by_behaviour.keys())
    parts_X, parts_chain, parts_label = [], [], []
    for b in behaviours:
        X = activations_by_behaviour[b]
        # Hard error on missing/mismatched chain ids: the old proxy fallback
        # (one chain per behaviour) made the within-chain permutation a NO-OP,
        # silently returning p≈1.0.
        cids = require_aligned(b, X.shape[0], chain_ids_by_behaviour.get(b),
                               context="05b null hierarchy")
        parts_X.append(X)
        parts_chain.append(cids)
        parts_label.append(np.array([b] * X.shape[0]))

    X_all     = np.vstack(parts_X)
    chain_all = np.concatenate(parts_chain)
    label_all = np.concatenate(parts_label)

    dup = duplicate_fraction(X_all)
    if dup > 0:
        logger.info(f"  removing {dup:.1%} exact-duplicate rows before the "
                    f"null hierarchy (CF-13)")
        X_all, chain_all, label_all = dedup_rows(X_all, chain_all, label_all)

    results = {}
    for b in behaviours:
        logger.info(f"  null hierarchy for '{b}' (n_resamples={n_resamples})...")
        t = time.time()
        nh = full_null_hierarchy(
            X_all, chain_all, label_all, b,
            statistic_fn=top_k_variance_ratio,
            statistic_name="top10_var_ratio",
            n_resamples=n_resamples,
        )
        results[b] = {k: asdict(v) for k, v in nh.items()}
        results[b]["_runtime_s"] = time.time() - t
    return results


# Markdown summary

def write_summary(per_behaviour: dict, null_results: dict, args, out_path: Path) -> None:
    lines = [
        f"# Phase 5b diagnostics - {args.model_short} layer {args.layer}",
        "",
        f"k sweep: {args.k}",
        f"PCA topk for curvature: {args.pca_topk}",
        f"Null hierarchy resamples: {args.n_resamples}",
        "",
        "## Intrinsic dimension estimates",
        "",
        "| Behaviour | N | TwoNN | Levina-Bickel | Correlation dim |",
        "|-----------|---|-------|---------------|------------------|",
    ]
    for b in TARGET_BEHAVIOURS:
        if b not in per_behaviour:
            continue
        N = per_behaviour[b]["N"]
        ests = {r["estimator"]: r for r in per_behaviour[b]["intrinsic_dim"]}
        def fmt(name):
            if name not in ests:
                return "-"
            r = ests[name]
            return f"{r['estimate']:.1f} [{r['ci_low']:.1f}, {r['ci_high']:.1f}]"
        lines.append(f"| {b} | {N} | {fmt('twoNN')} | {fmt('levina_bickel')} | {fmt('correlation_dim')} |")

    lines += [
        "",
        "## Curvature diagnostics (post-PCA projection)",
        "",
        "Local-vs-global PCA dim ratio (close to 1 = flat, lower = curved):",
        "",
        "| Behaviour | k=10 | k=30 | k=100 |",
        "|-----------|------|------|--------|",
    ]
    def fmt_curv(beh, diag_name, k):
        if beh not in per_behaviour:
            return "-"
        for r in per_behaviour[beh]["curvature"]:
            if r["diagnostic"] == diag_name and r["k"] == k:
                return f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        return "-"
    for b in TARGET_BEHAVIOURS:
        lines.append(f"| {b} | {fmt_curv(b, 'local_vs_global_dim_ratio', 10)} | {fmt_curv(b, 'local_vs_global_dim_ratio', 30)} | {fmt_curv(b, 'local_vs_global_dim_ratio', 100)} |")

    lines += [
        "",
        "Geodesic / Euclidean ratio (close to 1 = flat, higher = curved):",
        "",
        "| Behaviour | k=10 | k=30 | k=100 |",
        "|-----------|------|------|--------|",
    ]
    for b in TARGET_BEHAVIOURS:
        lines.append(f"| {b} | {fmt_curv(b, 'geodesic_euclidean_ratio', 10)} | {fmt_curv(b, 'geodesic_euclidean_ratio', 30)} | {fmt_curv(b, 'geodesic_euclidean_ratio', 100)} |")

    lines += [
        "",
        "Tangent-space variation (degrees, close to 0 = flat, higher = curved):",
        "",
        "| Behaviour | k=10 | k=30 | k=100 |",
        "|-----------|------|------|--------|",
    ]
    for b in TARGET_BEHAVIOURS:
        lines.append(f"| {b} | {fmt_curv(b, 'tangent_space_variation_deg', 10)} | {fmt_curv(b, 'tangent_space_variation_deg', 30)} | {fmt_curv(b, 'tangent_space_variation_deg', 100)} |")

    if null_results:
        lines += [
            "",
            "## Null hypothesis hierarchy (top-10 variance ratio)",
            "",
            "Primary: chain-stratified permutation. Secondary: cross-chain permutation. Tertiary: MP isotropic (finite-sample inflation diagnostic only).",
            "",
            "| Behaviour | real | chain-strat null (mean, 95% CI), p | cross-chain null (mean, p) | MP null mean |",
            "|-----------|------|-------------------------------------|----------------------------|--------------|",
        ]
        for b in TARGET_BEHAVIOURS:
            if b not in null_results:
                continue
            cs = null_results[b]["chain_strat"]
            cc = null_results[b]["cross_chain"]
            mp = null_results[b]["mp"]
            lines.append(
                f"| {b} | {cs['real_value']:.4f} | "
                f"{cs['null_mean']:.4f} [{cs['null_p2_5']:.4f}, {cs['null_p97_5']:.4f}], p={cs['p_value']:.4f} | "
                f"{cc['null_mean']:.4f}, p={cc['p_value']:.4f} | "
                f"{mp['null_mean']:.4f} |"
            )

    out_path.write_text("\n".join(lines))


# Main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-short", default="R1-1.5B")
    parser.add_argument("--layer", type=int, default=None,
                        help="Single target layer (default: model's steering layer). "
                             "Use --layers for multi-layer runs.")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Multiple target layers — overrides --layer if set. "
                             "Used in the multi-criteria triangulation workflow.")
    parser.add_argument("--n-resamples", type=int, default=500,
                        help="Monte Carlo resamples per null. Default 500.")
    parser.add_argument("--k", nargs="+", type=int, default=[10, 30, 100],
                        help="k-NN neighborhood sizes for curvature sweep.")
    parser.add_argument("--pca-topk", type=int, default=50,
                        help="Project to top-k PCA before curvature diagnostics. Default 50.")
    parser.add_argument("--intrinsic-dim-for-tangent", type=int, default=5,
                        help="Intrinsic dim used by tangent-space-variation. Default 5.")
    parser.add_argument("--skip-nulls", action="store_true",
                        help="Skip the null-hypothesis hierarchy (faster).")
    args = parser.parse_args()

    # Determine which layers to process
    if args.layers is not None:
        layers_to_run = args.layers
    elif args.layer is not None:
        layers_to_run = [args.layer]
    else:
        layers_to_run = [STEERING_LAYERS.get(args.model_short, 27)]

    act_dir       = Path(f"data/activations/{args.model_short}")
    annotated_p   = Path(f"data/annotated_{args.model_short}.json")
    out_dir       = Path(f"results/geometric/{args.model_short}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not act_dir.exists():
        logger.error(f"Activations not found at {act_dir}. Run 04_extract_activations.py first.")
        sys.exit(1)

    logger.info(f"Running 5b across {len(layers_to_run)} layer(s): {layers_to_run}")
    for layer_idx, current_layer in enumerate(layers_to_run):
        logger.info(f"=== Layer {current_layer} ({layer_idx+1}/{len(layers_to_run)}) ===")
        args.layer = current_layer

        # Load per-behaviour activations and (optionally) chain IDs
        activations_by_beh = {}
        for b in TARGET_BEHAVIOURS:
            try:
                activations_by_beh[b] = load_activations(act_dir, b, args.layer)
                logger.info(f"Loaded {b}: shape={activations_by_beh[b].shape}")
            except FileNotFoundError as e:
                logger.warning(f"Skipping {b}: {e}")

        if not activations_by_beh:
            logger.error("No activation files found for any target behaviour.")
            sys.exit(1)

        chain_ids_by_beh = load_chain_id_map(annotated_p, list(activations_by_beh.keys()),
                                             act_dir=act_dir)

        # Per-behaviour diagnostics
        per_behaviour = {}
        for b, X in activations_by_beh.items():
            logger.info(f"=== {b} ===")
            per_behaviour[b] = diagnose_behaviour(
                X, chain_ids_by_beh.get(b), b,
                n_resamples=args.n_resamples,
                k_values=tuple(args.k),
                intrinsic_dim_for_tangent=args.intrinsic_dim_for_tangent,
                pca_topk_for_curvature=args.pca_topk,
            )

        # Null hierarchy across behaviours
        null_results = {}
        if not args.skip_nulls:
            logger.info("=== Null hypothesis hierarchy ===")
            null_results = run_null_hierarchy_at_layer(
                activations_by_beh, chain_ids_by_beh, args.n_resamples,
            )

        # Persist
        payload = {
            "model_short": args.model_short,
            "layer": args.layer,
            "args": vars(args),
            "per_behaviour": per_behaviour,
            "null_hierarchy": null_results,
        }
        json_path = out_dir / f"diagnostics_layer{args.layer}.json"

        def _make_serializable(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            if isinstance(obj, (set, frozenset, tuple)):
                return list(obj)
            raise TypeError(f"Not JSON serialisable: {type(obj).__name__}")

        json_path.write_text(json.dumps(payload, indent=2, default=_make_serializable))

        md_path = out_dir / f"summary_layer{args.layer}.md"
        write_summary(per_behaviour, null_results, args, md_path)

        print("\n" + "=" * 60)
        print(f"Phase 5b complete - layer {args.layer}")
        print("=" * 60)
        print(f"JSON: {json_path}")
        print(f"MD:   {md_path}")


if __name__ == "__main__":
    main()
