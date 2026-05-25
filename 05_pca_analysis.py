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

# Huang et al.'s recommended steering layers
STEERING_LAYERS = {"R1-1.5B": 27, "R1-7B": 27, "R1-8B": 31}


def main():
    parser = argparse.ArgumentParser(description="Phase 5: PCA manifold analysis")
    parser.add_argument("--model-short", default="R1-1.5B")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Layers to analyse (default: all available)")
    parser.add_argument("--focus-layer", type=int, default=None,
                        help="Primary layer to display in summary table "
                             "(default: Huang's recommended layer)")
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
        save_pca_results(layer_res, save_dir, layer=layer)

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
