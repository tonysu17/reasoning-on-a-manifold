"""
Phase 5: PCA manifold analysis — the core empirical contribution.

For each behaviour at each layer, fits a PCA to the activation matrix and
computes dimensionality metrics that answer the central question:
  "Does this behaviour occupy a 1-D direction (Venhoff) or a multi-dimensional
   manifold (Huang)?"

Key metrics:
  d_eff(p)         — smallest k such that top-k PCs explain ≥ p of variance
  participation_ratio (PR) — (Σλ)² / Σ(λ²), a sample-size-robust estimate
                             of effective dimensionality

The comparison table (d_eff_70 and PR across all four behaviours) is the
primary result of Phase 3 / Section 4 of the paper.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


# ── Single-behaviour analysis ─────────────────────────────────────────────────

def analyse_behaviour(
    activation_matrix: np.ndarray,
    max_components: int = 50,
) -> dict:
    """
    Fit PCA to one behaviour's activation matrix and return all metrics.

    Args:
        activation_matrix: float32 array of shape (N_instances, hidden_dim)
        max_components:    maximum PCA components to fit

    Returns dict with:
        cumulative_variance, eigenvalues, explained_variance_ratio,
        components  (numpy arrays — serialise separately)
        mean        (numpy array)
        d_eff_50/70/80/90/95  (int)
        participation_ratio   (float)
        n_samples, hidden_dim
    """
    N, d = activation_matrix.shape
    k = min(N - 1, d, max_components)

    if k < 2:
        logger.warning(f"Too few samples or dims (N={N}, d={d}) for PCA")
        return {
            "n_samples": N, "hidden_dim": d, "error": "insufficient_data",
            "d_eff_50": 1, "d_eff_70": 1, "d_eff_80": 1,
            "d_eff_90": 1, "d_eff_95": 1, "participation_ratio": 1.0,
        }

    pca = PCA(n_components=k)
    pca.fit(activation_matrix)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    eigvals = pca.explained_variance_

    d_effs = {}
    for label, thresh in [("d_eff_50", 0.50), ("d_eff_70", 0.70), ("d_eff_80", 0.80),
                           ("d_eff_90", 0.90), ("d_eff_95", 0.95)]:
        idx = int(np.searchsorted(cumvar, thresh))
        d_effs[label] = min(idx + 1, k)

    pr = float((eigvals.sum() ** 2) / (np.sum(eigvals ** 2) + 1e-12))

    return {
        "cumulative_variance": cumvar,
        "eigenvalues": eigvals,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "components": pca.components_,
        "mean": pca.mean_,
        **d_effs,
        "participation_ratio": pr,
        "n_samples": N,
        "hidden_dim": d,
    }


# ── Multi-behaviour / multi-layer ─────────────────────────────────────────────

def analyse_at_layer(
    activations_dir: Path,
    behaviours: list[str],
    layer: int,
    max_components: int = 50,
) -> dict[str, dict]:
    """Run PCA on all behaviours at one layer."""
    activations_dir = Path(activations_dir)
    results = {}
    for beh in behaviours:
        path = activations_dir / f"{beh}_layer{layer}.npy"
        if not path.exists():
            logger.warning(f"Missing: {path}")
            continue
        mat = np.load(path)
        logger.info(f"  {beh}: {mat.shape[0]} instances × {mat.shape[1]} dims")
        results[beh] = analyse_behaviour(mat, max_components)
    return results


def analyse_across_layers(
    activations_dir: Path,
    behaviours: list[str],
    layers: list[int],
    max_components: int = 50,
) -> dict[str, dict[int, dict]]:
    """Run PCA for all behaviours across multiple layers."""
    results: dict[str, dict[int, dict]] = {b: {} for b in behaviours}
    for layer in layers:
        logger.info(f"Layer {layer}:")
        layer_res = analyse_at_layer(activations_dir, behaviours, layer, max_components)
        for beh, r in layer_res.items():
            results[beh][layer] = r
    return results


# ── Saving / loading ──────────────────────────────────────────────────────────

def save_pca_results(results: dict, save_dir: Path, layer: Optional[int] = None) -> None:
    """
    Save PCA results to *save_dir*.
    Large arrays (components, eigenvalues, cumvar) go to .npy files.
    Scalar metrics go to a summary JSON.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_layer{layer}" if layer is not None else ""
    summary = {}

    for beh, data in results.items():
        if "components" not in data:
            summary[beh] = data
            continue
        np.save(save_dir / f"{beh}_components{suffix}.npy", data["components"])
        np.save(save_dir / f"{beh}_eigenvalues{suffix}.npy", data["eigenvalues"])
        np.save(save_dir / f"{beh}_cumvar{suffix}.npy", data["cumulative_variance"])
        np.save(save_dir / f"{beh}_mean{suffix}.npy", data["mean"])
        summary[beh] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in data.items()
            if k not in ("components", "eigenvalues", "cumulative_variance",
                         "explained_variance_ratio", "mean")
        }

    with open(save_dir / f"summary{suffix}.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"PCA results saved → {save_dir}")


def load_pca_summary(save_dir: Path, layer: Optional[int] = None) -> dict:
    suffix = f"_layer{layer}" if layer is not None else ""
    with open(Path(save_dir) / f"summary{suffix}.json") as f:
        return json.load(f)


def load_pca_components(save_dir: Path, behaviour: str, layer: int) -> np.ndarray:
    return np.load(Path(save_dir) / f"{behaviour}_components_layer{layer}.npy")


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_dimensionality_table(results: dict, layer: Optional[int] = None) -> None:
    header = "Effective Dimensionality"
    if layer is not None:
        header += f" at Layer {layer}"
    print(f"\n{header}")
    print(f"{'Behaviour':<30s} {'N':>5s} {'d50':>5s} {'d70':>5s} {'d90':>5s} {'d95':>5s} {'PR':>6s}")
    print("─" * 60)
    for beh, data in results.items():
        if "error" in data:
            print(f"  {beh:<28s}  {'(insufficient data)'}")
            continue
        print(
            f"{beh:<30s} {data['n_samples']:>5d} "
            f"{data['d_eff_50']:>5d} {data['d_eff_70']:>5d} "
            f"{data['d_eff_90']:>5d} {data['d_eff_95']:>5d} "
            f"{data['participation_ratio']:>6.1f}"
        )
