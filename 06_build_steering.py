#!/usr/bin/env python3
"""
Phase 6 — Steering vector construction.

Builds two steering vectors per target behaviour at the specified layer:
  1. Single-direction (Venhoff-style): difference of means, unit-normalised
  2. Manifold-projected (our method):  same vector projected onto the top-k
     PCA subspace of the behaviour's activation manifold

Output: results/steering_vectors/<model>/

Requirements:
  pip install scikit-learn numpy        (no GPU needed)
  Input: data/activations/<model>/      (from Phase 4)

Runtime: <1 minute
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.steering import build_steering_vectors, save_steering_vectors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.config import STEERING_LAYERS, provenance  # single source: configs/config.yaml


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Build steering vectors")
    parser.add_argument("--model-short", default="R1-1.5B")
    parser.add_argument("--layer", type=int, default=None,
                        help="Transformer layer (default: Huang's recommended layer)")
    parser.add_argument("--k-values", nargs="+", default=["1", "3", "5", "10", "auto"],
                        help="k values for manifold projection (default: 1 3 5 10 auto)")
    args = parser.parse_args()

    act_dir = Path(f"data/activations/{args.model_short}")
    if not act_dir.exists():
        logger.error(f"Activations not found at {act_dir}. Run 04_extract_activations.py first.")
        sys.exit(1)

    layer = args.layer or STEERING_LAYERS.get(args.model_short, 27)
    k_values = [int(k) if k != "auto" else "auto" for k in args.k_values]

    logger.info(f"Building steering vectors at layer {layer} …")
    vectors = build_steering_vectors(
        activations_dir=act_dir,
        layer=layer,
        k_values=k_values,
    )

    # Canonical all-behaviours-at-one-layer build. The per-behaviour-peak
    # builder (build_phase6.py) writes to <model>-peak/ so the two can no
    # longer clobber each other's identically-named .npy files.
    save_dir = Path(f"results/steering_vectors/{args.model_short}")
    prov = provenance(args)
    prov["builder"] = f"06_build_steering.py (all behaviours at layer {layer})"
    save_steering_vectors(vectors, save_dir, provenance=prov)

    print(f"\n{'='*55}")
    print(f"Steering vectors built at layer {layer}")
    print(f"{'Behaviour':<30s} {'n_on':>6s} {'n_off':>6s} {'auto_k':>7s}")
    print("-" * 55)
    for beh, data in vectors.items():
        print(f"{beh:<30s} {data['n_on']:>6d} {data['n_off']:>6d} {data['auto_k']:>7d}")

    print(f"\nSaved → {save_dir}")
    print("Next step: run  07_evaluate_steering.py")


if __name__ == "__main__":
    main()
