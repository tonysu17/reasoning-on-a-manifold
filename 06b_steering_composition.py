#!/usr/bin/env python3
"""
Phase 6b - Steering vector composition test.

Companion document Section 2.6: "Composition experiments combine steering
vectors for distinct behaviours (e.g., backtracking + uncertainty-estimation)
and test whether observed effects sum linearly. Linear addition would
support the Venhoff single-direction picture; non-additive composition
would support the curved-manifold picture in which behaviours are tangent
vectors to a shared structure."

This script:
  1. Loads Phase 6 steering vectors for each pair of target behaviours.
  2. Computes three composed vectors per pair (a, b):
       v_sum    = v_a + v_b                           (linear)
       v_proj   = v_a + v_b projected onto top-50 PCA   (manifold-aware)
       v_tan    = v_a + v_b restricted to local tangent bundle (Phase 5b)
  3. Computes diagnostic quantities WITHOUT inference:
       cosine similarity between v_sum, v_proj, v_tan
       ||v_sum - v_proj|| / ||v_sum||  (relative magnitude of off-manifold component)
  4. Pre-registers what these diagnostics would predict under each picture:
       Flat picture: cos(v_sum, v_proj) ~= 1, off-manifold ratio ~= 0
       Curved picture: off-manifold ratio scales with the pair-specific
                       curvature magnitudes
  5. The actual behavioural test requires Phase 7 inference and is run by
     07_evaluate_steering.py with --composition mode (to be wired separately).

Outputs:
  results/composition/<model>/composition_layer<L>.json
  results/composition/<model>/summary_layer<L>.md

Usage:
  python 06b_steering_composition.py --layer 27
"""

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


from src.annotation import TARGET_BEHAVIOURS


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu == 0 or nv == 0:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def load_steering_vector(steer_dir: Path, behaviour: str, layer: int) -> np.ndarray:
    p = steer_dir / f"{behaviour}_layer{layer}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Steering vector missing: {p}")
    return np.load(p)


def load_activations(act_dir: Path, behaviour: str, layer: int) -> np.ndarray:
    p = act_dir / f"{behaviour}_layer{layer}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Activations missing: {p}")
    return np.load(p)


def project_to_pooled_subspace(v: np.ndarray, X: np.ndarray, top_k: int = 50) -> np.ndarray:
    """Project v onto the top-k PCA subspace of pooled activations X."""
    centered = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    d = min(top_k, vt.shape[0])
    B = vt[:d].T   # (D, d)
    coeffs = B.T @ v
    return B @ coeffs


def project_to_tangent_bundle(v: np.ndarray, X: np.ndarray, tangent_dim: int = 10) -> np.ndarray:
    """Project v onto the tangent-bundle (top-k PCA of X centered at its mean)."""
    return project_to_pooled_subspace(v, X, top_k=tangent_dim)


def compose_pair(
    v_a: np.ndarray, v_b: np.ndarray,
    pooled_act: np.ndarray,
    beh_a_act:  np.ndarray,
    beh_b_act:  np.ndarray,
    pca_topk:        int = 50,
    tangent_dim:     int = 10,
) -> dict:
    """Compute the three composed vectors and their pairwise diagnostics."""
    v_sum = v_a + v_b

    # Manifold-projection composition: project the sum onto the pooled top-50 subspace
    v_proj = project_to_pooled_subspace(v_sum, pooled_act, top_k=pca_topk)

    # Tangent-bundle composition: separately project each onto its OWN behaviour's
    # tangent bundle, then sum. This corresponds to "behaviours as tangent vectors
    # to a shared manifold" picture.
    v_a_tan = project_to_tangent_bundle(v_a, beh_a_act, tangent_dim=tangent_dim)
    v_b_tan = project_to_tangent_bundle(v_b, beh_b_act, tangent_dim=tangent_dim)
    v_tan = v_a_tan + v_b_tan

    off_manifold_ratio = float(np.linalg.norm(v_sum - v_proj) / max(1e-8, np.linalg.norm(v_sum)))
    off_tangent_ratio = float(np.linalg.norm(v_sum - v_tan) / max(1e-8, np.linalg.norm(v_sum)))

    return {
        "norm_v_a":          float(np.linalg.norm(v_a)),
        "norm_v_b":          float(np.linalg.norm(v_b)),
        "norm_v_sum":        float(np.linalg.norm(v_sum)),
        "norm_v_proj":       float(np.linalg.norm(v_proj)),
        "norm_v_tan":        float(np.linalg.norm(v_tan)),
        "cos_a_b":           cosine(v_a, v_b),
        "cos_sum_proj":      cosine(v_sum, v_proj),
        "cos_sum_tan":       cosine(v_sum, v_tan),
        "cos_proj_tan":      cosine(v_proj, v_tan),
        "off_manifold_ratio":  off_manifold_ratio,
        "off_tangent_ratio":   off_tangent_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-short", default="R1-1.5B")
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--behaviours", nargs="+", default=TARGET_BEHAVIOURS)
    parser.add_argument("--pca-topk", type=int, default=50)
    parser.add_argument("--tangent-dim", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(f"results/composition/{args.model_short}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    steer_dir = Path(f"results/steering_vectors/{args.model_short}")
    act_dir   = Path(f"data/activations/{args.model_short}")

    if not steer_dir.exists():
        logger.error(f"Steering vectors not found at {steer_dir}. Run Phase 6 first.")
        sys.exit(1)
    if not act_dir.exists():
        logger.error(f"Activations not found at {act_dir}. Run Phase 4 first.")
        sys.exit(1)

    # Load all needed vectors + activations
    vectors = {}
    activations = {}
    for b in args.behaviours:
        try:
            vectors[b]      = load_steering_vector(steer_dir, b, args.layer)
            activations[b]  = load_activations(act_dir, b, args.layer)
            logger.info(f"Loaded {b}: ||v||={np.linalg.norm(vectors[b]):.4f}, N_act={activations[b].shape[0]}")
        except FileNotFoundError as e:
            logger.warning(f"Skipping {b}: {e}")

    behaviours_have = sorted(set(vectors) & set(activations))
    if len(behaviours_have) < 2:
        logger.error("Need at least 2 behaviours with both steering vectors and activations")
        sys.exit(1)

    # Pooled activations across the available behaviours (for the v_proj subspace)
    pooled = np.vstack([activations[b] for b in behaviours_have])
    logger.info(f"Pooled activations: N={pooled.shape[0]}, d={pooled.shape[1]}")

    pair_results = {}
    for a, b in itertools.combinations(behaviours_have, 2):
        logger.info(f"=== pair: {a} + {b} ===")
        r = compose_pair(
            vectors[a], vectors[b],
            pooled,
            activations[a], activations[b],
            pca_topk=args.pca_topk,
            tangent_dim=args.tangent_dim,
        )
        pair_results[f"{a}+{b}"] = r
        logger.info(f"  cos(v_a, v_b)            = {r['cos_a_b']:.3f}")
        logger.info(f"  cos(v_sum, v_proj)       = {r['cos_sum_proj']:.3f}")
        logger.info(f"  off-manifold ratio       = {r['off_manifold_ratio']:.3f}")
        logger.info(f"  off-tangent ratio        = {r['off_tangent_ratio']:.3f}")

    payload = {
        "model_short": args.model_short, "layer": args.layer,
        "args": vars(args),
        "pairs": pair_results,
    }
    def _ser(o):
        if isinstance(o, Path): return str(o)
        raise TypeError
    (args.out_dir / f"composition_layer{args.layer}.json").write_text(json.dumps(payload, indent=2, default=_ser))

    # Markdown summary
    md = [
        f"# Steering composition - {args.model_short} layer {args.layer}",
        "",
        "Pre-registered predictions:",
        "- **Flat picture (Venhoff)**: cos(v_sum, v_proj) ~= 1, off-manifold ratio ~= 0.",
        "- **Curved-manifold picture**: cos(v_sum, v_proj) < 1, off-manifold ratio > 0,",
        "  with magnitude correlated with the pair-specific curvature scales kappa_a, kappa_b.",
        "",
        "## Diagnostics per pair",
        "",
        "| Pair | cos(v_a,v_b) | ||v_sum|| | cos(v_sum, v_proj) | off-manifold ratio | off-tangent ratio |",
        "|------|---------------|-----------|--------------------|---------------------|---------------------|",
    ]
    for pair, r in pair_results.items():
        md.append(
            f"| {pair} | {r['cos_a_b']:.3f} | {r['norm_v_sum']:.3f} | "
            f"{r['cos_sum_proj']:.3f} | {r['off_manifold_ratio']:.3f} | {r['off_tangent_ratio']:.3f} |"
        )
    md += [
        "",
        "## Interpretation",
        "",
        "Off-manifold ratio > 0.1 across multiple pairs is evidence against the flat-subspace picture.",
        "Cross-pair correlation between off-manifold ratio and per-behaviour curvature (from Phase 5b)",
        "tests whether the deviation magnitude scales with the geometry as predicted.",
    ]
    (args.out_dir / f"summary_layer{args.layer}.md").write_text("\n".join(md))

    print(f"\nResults: {args.out_dir}")


if __name__ == "__main__":
    main()
