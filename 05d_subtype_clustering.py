#!/usr/bin/env python3
"""
Phase 5d — Sub-type discovery via clustering.

For each target behaviour, cluster the activations at a specified layer
(typically the d_eff peak layer from Phase 5's layer sweep) to discover
semantically distinct sub-types of that behaviour. This produces the
sub-type steering vectors used downstream in Phase 6.

Methodology:
  1. Load behaviour's activation matrix at the focus layer.
  2. Project onto top-k PCA subspace (k = d_eff_70 from Phase 5).
  3. K-means in PC space, sweep k_clusters ∈ [2, 8].
  4. Select best cluster count by silhouette score.
  5. For each cluster:
      - cluster centroid in full activation space (1536-d)
      - cluster labels (per-instance assignment)
      - example sentences from annotations

Inputs:
  data/activations/{model_short}/{behaviour}_layer{N}.npy
  data/annotated_{model_short}.json
  results/pca/{model_short}/layer_profiles.json  (for d_eff peak layer)

Outputs:
  results/clustering/{model_short}/{behaviour}_layer{L}/
    centroids.npy              # (n_clusters, 1536)
    cluster_labels.npy         # (N_instances,)
    pc_coords.npy              # (N_instances, k_pca)  for visualisation
    silhouette_sweep.json      # silhouette score per k_clusters tried
    examples.json              # per-cluster example sentences
    summary.json               # n_clusters, n_per_cluster, etc.

Usage:
  python 05d_subtype_clustering.py --model-short R1-1.5B
  python 05d_subtype_clustering.py --model-short R1-1.5B --layer 18 --behaviours backtracking
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.annotation import TARGET_BEHAVIOURS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _annotation_index_to_text(annotated_path, target_behaviours):
    """Return {behaviour: list of (chain_id, sentence_text)} in extraction order."""
    with open(annotated_path) as f:
        chains = json.load(f)
    out = {b: [] for b in target_behaviours}
    for chain in chains:
        cid = chain.get("chain_id") or chain["task_id"]
        for ann in chain.get("annotations", []):
            lbl = ann.get("label", "")
            if lbl in out:
                out[lbl].append((cid, ann.get("text", "")))
    return out


def cluster_behaviour(
    X: np.ndarray,
    sentences: list,
    pca_topk: int,
    k_range: tuple = (2, 8),
    random_state: int = 42,
) -> dict:
    """K-means with silhouette selection.

    Args:
        X:          (N, hidden_dim) activation matrix
        sentences:  list of (chain_id, text) of length N
        pca_topk:   number of PCs to use as clustering space
        k_range:    (min_k, max_k) for sweep

    Returns dict with: best_k, silhouette_per_k, labels, centroids_full_space,
                       pc_coords, examples_per_cluster
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    N, d = X.shape
    pca_topk = min(pca_topk, N - 1, d, 100)
    if pca_topk < 2:
        return {"error": f"too few samples or dims for clustering (N={N}, d={d})"}

    # Project to top-k PCA
    pca = PCA(n_components=pca_topk, random_state=random_state)
    X_pc = pca.fit_transform(X)

    # Sweep k_clusters
    sweep = {}
    best_k, best_score, best_labels = None, -np.inf, None
    k_min, k_max = k_range
    k_max = min(k_max, N - 1)
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_pc)
        # silhouette only valid if more than one cluster has > 1 member
        unique, counts = np.unique(labels, return_counts=True)
        if (counts >= 2).sum() < 2:
            sweep[k] = float("nan")
            continue
        score = float(silhouette_score(X_pc, labels))
        sweep[k] = score
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels

    if best_labels is None:
        return {"error": "no valid clustering found"}

    # Compute full-space centroids (in original 1536-d, not PC space)
    centroids_full = np.array([X[best_labels == c].mean(axis=0)
                                for c in range(best_k)])

    # Pick example sentences per cluster (5 per cluster, randomly)
    rng = np.random.default_rng(random_state)
    examples = {}
    for c in range(best_k):
        members = np.where(best_labels == c)[0]
        n_pick = min(5, len(members))
        picked = rng.choice(members, size=n_pick, replace=False)
        examples[str(c)] = [
            {"chain_id": sentences[i][0], "text": sentences[i][1][:300]}
            for i in picked
        ]

    return {
        "best_k":            int(best_k),
        "best_silhouette":   float(best_score),
        "silhouette_per_k":  {str(k): float(v) for k, v in sweep.items()},
        "labels":            best_labels.astype(np.int32),
        "centroids_full":    centroids_full.astype(np.float32),
        "pc_coords":         X_pc[:, :3].astype(np.float32),   # 3D for scatter plots
        "examples":          examples,
        "n_per_cluster":     {str(c): int((best_labels == c).sum())
                              for c in range(best_k)},
    }


def _resolve_focus_layer(model_short, pca_dir, behaviour, fallback=27):
    """Find the manifold-peak layer for behaviour from Phase 5's layer_profiles.json.

    The manifold hypothesis predicts a *low*-dimensional curved manifold, so the
    layer where structure is strongest is the one with the LOWEST participation
    ratio (variance most concentrated). We therefore take argmin(participation_ratio).
    The earlier argmax(d_eff_70) rule returned layer 0 whenever d_eff saturated at
    the PCA component cap, which is why clustering ran at the wrong layer.
    """
    profile_path = pca_dir / "layer_profiles.json"
    if not profile_path.exists():
        logger.warning(f"  layer_profiles.json not found; using fallback layer {fallback}")
        return fallback
    profiles = json.load(open(profile_path))
    if behaviour not in profiles:
        return fallback
    bp = profiles[behaviour]
    pr = bp.get("participation_ratio")
    layers = bp.get("layers")
    if not pr or not layers:
        return fallback
    return int(layers[int(np.argmin(pr))])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-short", default="R1-1.5B")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to cluster at (default: d_eff peak per behaviour from Phase 5).")
    parser.add_argument("--behaviours", nargs="+", default=TARGET_BEHAVIOURS)
    parser.add_argument("--k-range", nargs=2, type=int, default=[2, 8],
                        help="Min/max number of clusters to sweep (default 2 8).")
    parser.add_argument("--pca-topk", type=int, default=None,
                        help="PCA components for clustering space (default: d_eff_70 per behaviour).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    act_dir   = Path(f"data/activations/{args.model_short}")
    annot_p   = Path(f"data/annotated_{args.model_short}.json")
    pca_dir   = Path(f"results/pca/{args.model_short}")
    out_root  = Path(f"results/clustering/{args.model_short}")
    out_root.mkdir(parents=True, exist_ok=True)

    if not act_dir.exists():
        logger.error(f"Activations not found at {act_dir}"); sys.exit(1)
    if not annot_p.exists():
        logger.error(f"Annotated file not found at {annot_p}"); sys.exit(1)

    sentence_index = _annotation_index_to_text(annot_p, args.behaviours)

    for beh in args.behaviours:
        # Layer choice
        L = args.layer if args.layer is not None else _resolve_focus_layer(args.model_short, pca_dir, beh)
        logger.info(f"=== {beh} at layer {L} ===")

        act_path = act_dir / f"{beh}_layer{L}.npy"
        if not act_path.exists():
            logger.warning(f"  missing {act_path}; skipping {beh}")
            continue

        X = np.load(act_path)
        sentences = sentence_index.get(beh, [])
        if len(sentences) != X.shape[0]:
            logger.warning(f"  sentence index length ({len(sentences)}) != X rows ({X.shape[0]}); proceeding without examples")
            sentences = [("", "")] * X.shape[0]

        # PCA topk — get from layer_profiles.json if available
        if args.pca_topk is not None:
            pca_topk = args.pca_topk
        else:
            try:
                profiles = json.load(open(pca_dir / "layer_profiles.json"))
                idx = profiles[beh]["layers"].index(L)
                pca_topk = profiles[beh]["d_eff_70"][idx]
            except (FileNotFoundError, KeyError, ValueError):
                pca_topk = 10

        logger.info(f"  N={X.shape[0]}, using top-{pca_topk} PCs as clustering space, k_range={args.k_range}")

        result = cluster_behaviour(
            X, sentences, pca_topk=pca_topk,
            k_range=tuple(args.k_range),
            random_state=args.seed,
        )

        if "error" in result:
            logger.warning(f"  clustering failed: {result['error']}")
            continue

        # Save per-behaviour outputs
        beh_dir = out_root / f"{beh}_layer{L}"
        beh_dir.mkdir(parents=True, exist_ok=True)
        np.save(beh_dir / "centroids.npy",       result["centroids_full"])
        np.save(beh_dir / "cluster_labels.npy",  result["labels"])
        np.save(beh_dir / "pc_coords.npy",       result["pc_coords"])
        with open(beh_dir / "silhouette_sweep.json", "w") as f:
            json.dump(result["silhouette_per_k"], f, indent=2)
        with open(beh_dir / "examples.json", "w") as f:
            json.dump(result["examples"], f, indent=2)
        with open(beh_dir / "summary.json", "w") as f:
            json.dump({
                "behaviour":       beh,
                "layer":           L,
                "n_instances":     int(X.shape[0]),
                "pca_topk":        int(pca_topk),
                "best_k":          result["best_k"],
                "best_silhouette": result["best_silhouette"],
                "n_per_cluster":   result["n_per_cluster"],
                "silhouette_per_k": result["silhouette_per_k"],
            }, f, indent=2)
        logger.info(f"  best_k={result['best_k']} (silhouette={result['best_silhouette']:.3f})")
        logger.info(f"  saved → {beh_dir}")

    print(f"\nResults: {out_root}")


if __name__ == "__main__":
    main()
