#!/usr/bin/env python3
"""
Phase 5 — PCA manifold analysis.

For each target behaviour at each layer, fits PCA to the activation matrix
and computes dimensionality metrics.  Produces:
  - results/pca/<model>/summary_layer<N>.json
  - results/pca/<model>/<behaviour>_components/eigenvalues/cumvar_layer<N>.npy
  - Console table: effective dimensionality across behaviours at the steering layer

Requirements:
  pip install scikit-learn numpy        (no GPU needed)
  Input: data/activations/<model>/      (from Phase 4)

Runtime: ~2–5 minutes on CPU
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from src.pca import (
    analyse_across_layers,
    analyse_at_layer,
    print_dimensionality_table,
    save_pca_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.annotation import TARGET_BEHAVIOURS

# Huang et al.'s recommended steering layers — single source: configs/config.yaml
from src.config import STEERING_LAYERS, provenance




# ── Per-layer null hierarchy (added) ──────────────────────────────────────────

# Single source of truth (was a verbatim copy; drift here misaligns rows).
from src.text_offsets import find_sentence_offset as _find_sentence_offset


def _load_chain_id_map(annotated_path, target_behaviours):
    """Return {behaviour: ndarray of chain_ids per row in extraction order}.

    The order MUST match how activation_extraction.py iterated through chains
    and annotations. To keep counts aligned with the activation matrices, we
    apply Phase 4's same find_sentence_offset filter — annotations whose text
    couldn't be located in the chain were skipped during extraction, so we skip
    them here too.
    """
    with open(annotated_path) as f:
        chains = json.load(f)
    per_beh = {b: [] for b in target_behaviours}
    n_filtered = {b: 0 for b in target_behaviours}
    for chain in chains:
        chain_text = chain.get("chain", "")
        cid = chain.get("chain_id") or chain["task_id"]
        for ann in chain.get("annotations", []):
            label = ann.get("label", "")
            if label not in per_beh:
                continue
            # Apply the same filter Phase 4 applied
            if _find_sentence_offset(chain_text, ann.get("text", "")) is None:
                n_filtered[label] += 1
                continue
            per_beh[label].append(cid)
    for b, n in n_filtered.items():
        if n > 0:
            logger.info(f"  chain_id loader filtered {n} unlocatable {b} annotations (Phase 4 skipped these too)")
    return {b: (np.array(v) if v else None) for b, v in per_beh.items()}


def _build_pooled(act_dir, layer, target_behaviours, chain_id_map):
    """Return X_pooled, chain_ids_pooled, labels_pooled at one layer."""
    X_parts, chain_parts, label_parts = [], [], []
    for b in target_behaviours:
        path = act_dir / f"{b}_layer{layer}.npy"
        if not path.exists():
            continue
        X_b = np.load(path)
        cids_b = chain_id_map.get(b)
        if cids_b is None or len(cids_b) != X_b.shape[0]:
            logger.warning(f"  chain_ids length mismatch for {b} layer {layer}: "
                           f"{None if cids_b is None else len(cids_b)} vs N={X_b.shape[0]}; using proxy")
            cids_b = np.array([f"{b}_proxy"] * X_b.shape[0])
        X_parts.append(X_b)
        chain_parts.append(cids_b)
        label_parts.append(np.array([b] * X_b.shape[0]))
    return np.vstack(X_parts), np.concatenate(chain_parts), np.concatenate(label_parts)


def compute_per_layer_nulls(act_dir, annotated_path, layers, target_behaviours,
                             n_resamples=100):
    """Run chain-stratified permutation null at every layer for every target
    behaviour. Returns a nested dict {behaviour: {layer: {real, null_mean, p_value}}}.
    """
    from src.nulls import chain_stratified_permutation_null, top_k_variance_ratio
    chain_id_map = _load_chain_id_map(annotated_path, target_behaviours)
    results = {b: {} for b in target_behaviours}
    for L in layers:
        try:
            X, chains, labels = _build_pooled(act_dir, L, target_behaviours, chain_id_map)
        except Exception as e:
            logger.warning(f"  layer {L}: build_pooled failed ({e}); skipping")
            continue
        for b in target_behaviours:
            if b not in labels:
                continue
            try:
                r = chain_stratified_permutation_null(
                    activations=X, chain_ids=chains, labels=labels,
                    target_label=b,
                    statistic_fn=top_k_variance_ratio,
                    statistic_name="top10_var_ratio",
                    n_resamples=n_resamples,
                )
                results[b][int(L)] = {
                    "real_value": r.real_value,
                    "null_mean":  r.null_mean,
                    "null_p2_5":  r.null_p2_5,
                    "null_p97_5": r.null_p97_5,
                    "p_value":    r.p_value,
                }
            except Exception as e:
                logger.warning(f"  null at {b}@L{L} failed: {e}")
        logger.info(f"  layer {L}: nulls computed for {len(results)} behaviours")
    return results

def main():
    parser = argparse.ArgumentParser(description="Phase 5: PCA manifold analysis")
    parser.add_argument("--model-short", default="R1-1.5B")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Layers to analyse (default: all available)")
    parser.add_argument("--focus-layer", type=int, default=None,
                        help="Primary layer to display in summary table "
                             "(default: Huang's recommended layer)")
    parser.add_argument("--with-nulls", action="store_true",
                        help="Compute chain-stratified permutation null p-value "
                             "at every layer per behaviour (B=100 by default).")
    parser.add_argument("--null-resamples", type=int, default=100,
                        help="Number of resamples for the per-layer null (default 100).")
    parser.add_argument("--annotated", type=Path, default=None,
                        help="Path to annotated chains JSON (needed for chain IDs).")
    args = parser.parse_args()

    act_dir = Path(f"data/activations/{args.model_short}")
    if not act_dir.exists():
        logger.error(f"Activations not found at {act_dir}. Run 04_extract_activations.py first.")
        sys.exit(1)

    save_dir = Path(f"results/pca/{args.model_short}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Discover available layers from filenames
    if args.layers is None:
        # Pick the first behaviour that has any extractions, so layer discovery
         # doesn't depend on backtracking specifically being non-empty.
        npy_files = []
        for beh in TARGET_BEHAVIOURS:
            npy_files = list(act_dir.glob(f"{beh}_layer*.npy"))
            if npy_files:
                break
        layers = sorted(int(p.stem.split("layer")[1]) for p in npy_files)
        if not layers:
            logger.error("No activation files found.")
            sys.exit(1)
        logger.info(f"Found layers: {layers}")
    else:
        layers = args.layers

    focus_layer = args.focus_layer or STEERING_LAYERS.get(args.model_short, layers[-1])

    logger.info(f"Running PCA across {len(layers)} layers × {len(TARGET_BEHAVIOURS)} behaviours …")
    all_results = analyse_across_layers(act_dir, TARGET_BEHAVIOURS, layers)

    # Save per-layer results
    for layer in layers:
        layer_res = {beh: all_results[beh][layer]
                     for beh in TARGET_BEHAVIOURS if layer in all_results[beh]}
        save_pca_results(layer_res, save_dir, layer=layer, provenance=provenance(args))

    # Optional: per-layer chain-stratified null
    if args.with_nulls:
        annotated_path = args.annotated or Path(f"data/annotated_{args.model_short}.json")
        if not annotated_path.exists():
            logger.warning(f"--with-nulls: annotated file {annotated_path} not found; skipping nulls")
        else:
            logger.info(f"Computing chain-stratified null at all layers (B={args.null_resamples})...")
            null_results = compute_per_layer_nulls(
                act_dir, annotated_path, layers, TARGET_BEHAVIOURS,
                n_resamples=args.null_resamples,
            )
            null_path = save_dir / "null_pvalues_per_layer.json"
            with open(null_path, "w") as f:
                json.dump(null_results, f, indent=2)
            logger.info(f"Per-layer null p-values saved → {null_path}")

    # Save layer-wise d_eff profiles (for plotting)
    profiles = {}
    for beh in TARGET_BEHAVIOURS:
        profiles[beh] = {
            "layers": [],
            "d_eff_70": [],
            "d_eff_90": [],
            "participation_ratio": [],
        }
        for layer in sorted(all_results[beh]):
            r = all_results[beh][layer]
            if "error" not in r:
                profiles[beh]["layers"].append(layer)
                profiles[beh]["d_eff_70"].append(r["d_eff_70"])
                profiles[beh]["d_eff_90"].append(r["d_eff_90"])
                profiles[beh]["participation_ratio"].append(r["participation_ratio"])

    with open(save_dir / "layer_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)

    # Print the key comparison table at the focus layer
    focus_res = {beh: all_results[beh].get(focus_layer, {"error": "missing"})
                 for beh in TARGET_BEHAVIOURS}
    print_dimensionality_table(focus_res, layer=focus_layer)

    print(f"\nResults saved → {save_dir}")
    print("Next step: run  06_build_steering.py")


if __name__ == "__main__":
    main()
