"""
Curvature diagnostics for activation manifolds.

Three complementary diagnostics distinguish a flat linear subspace
from a curved manifold:

  1. Local-vs-global PCA dimension ratio
     dim_local / dim_global, where dim_local is fit in a k-NN ball.
     Flat manifold => 1.0; curved manifold => < 1.0.

  2. Geodesic-to-Euclidean distance ratio
     For pairs (i, j), compute graph-shortest-path / Euclidean. Mean over
     pairs measures global curvature. Flat manifold => 1.0; curved => > 1.0.

  3. Tangent-space variation
     Mean principal angle between local PCA bases at different points.
     Flat => 0 (constant tangent space); curved => > 0.

All three are sweepable over the k-NN neighborhood size k. Stable signals
across k are credible; k-dependent signals are flagged as artefactual.

Companion document Section 2.5 (Phase 5: geometric analysis).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# Result containers

@dataclass
class CurvatureResult:
    """Per-(diagnostic, k) result with bootstrap CI."""
    diagnostic: str
    k:          int
    mean:       float
    ci_low:     float
    ci_high:    float
    n_samples:  int
    raw:        np.ndarray = field(default_factory=lambda: np.array([]))


# Helpers

def _knn_graph(X: np.ndarray, k: int):
    """Symmetric k-NN graph as a sparse COO. Returns (rows, cols, dists, neighbors).

    neighbors[i] is an array of length k of the k nearest neighbor indices of i
    (excluding i itself). Computed via sklearn NearestNeighbors for memory/speed.
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(X)
    dists, idxs = nn.kneighbors(X)
    # Drop self-loop (column 0)
    dists = dists[:, 1:]
    idxs  = idxs[:, 1:]
    rows = np.repeat(np.arange(X.shape[0]), k)
    cols = idxs.flatten()
    return rows, cols, dists.flatten(), idxs


# 1. Local vs global PCA dimension ratio

def local_pca_dim(local_pts: np.ndarray, variance_threshold: float = 0.90) -> int:
    """Smallest k such that top-k components explain >= variance_threshold."""
    if local_pts.shape[0] < 2:
        return 0
    centered = local_pts - local_pts.mean(axis=0, keepdims=True)
    # SVD on centered points; sqv ~ eigenvalues of covariance up to scale
    _, sv, _ = np.linalg.svd(centered, full_matrices=False)
    eig = sv ** 2
    if eig.sum() == 0:
        return 0
    cum = np.cumsum(eig) / eig.sum()
    return int(np.searchsorted(cum, variance_threshold) + 1)


def local_vs_global_dim_ratio(
    X: np.ndarray,
    k: int = 30,
    variance_threshold: float = 0.90,
    n_anchors: int = 200,
    random_state: int = 42,
    n_bootstrap: int = 100,
) -> CurvatureResult:
    """Mean ratio dim_local(p) / dim_baseline, averaged over anchor points p.

    dim_local  = PCA dim of the k nearest neighbours of an anchor.
    dim_baseline = PCA dim of a RANDOM set of k points (matched sample size).

    Flat manifold:   ratio ~= 1 (a local patch and a random k-set span the same
                     effective dimension).
    Curved manifold: ratio < 1 (curvature flattens the local patch, so the
                     k neighbours span fewer effective dimensions than k points
                     drawn from across the whole cloud).

    CRITICAL calibration note: the baseline MUST be sample-size-matched to the
    local neighbourhood. The original implementation divided by the PCA dim of
    ALL N points, which conflated curvature with the trivial fact that a k-point
    neighbourhood can express at most k-1 dimensions while the full cloud can
    express many more — so a perfectly FLAT subspace of dimension > k scored
    << 1 (empirically 0.29-0.85 on flat Gaussian data). Matching the sample
    size removes that confound. See tests/test_curvature.py::test_flat_*.

    n_bootstrap resamples the set of anchor points to estimate CI.
    """
    rng = np.random.default_rng(random_state)
    N, _ = X.shape
    if N < k + 1:
        raise ValueError(f"Need at least k+1 points ({k+1}), got {N}.")

    _, _, _, neighbors = _knn_graph(X, k)
    n_anchors = min(n_anchors, N)
    anchor_idx = rng.choice(N, n_anchors, replace=False)

    # Matched-size baseline: PCA dim of random k-point subsets, averaged.
    n_baseline = min(n_anchors, N)
    baseline_dims = np.array([
        local_pca_dim(X[rng.choice(N, k, replace=False)], variance_threshold)
        for _ in range(n_baseline)
    ])
    baseline_dims = baseline_dims[baseline_dims > 0]
    if baseline_dims.size == 0:
        return CurvatureResult("local_vs_global_dim_ratio", k, float("nan"),
                                float("nan"), float("nan"), N)
    dim_baseline = float(baseline_dims.mean())

    local_dims = np.array([
        local_pca_dim(X[neighbors[i]], variance_threshold) for i in anchor_idx
    ])
    ratios = local_dims / dim_baseline

    # Bootstrap
    boots = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(ratios, ratios.size, replace=True)
        boots[b] = sample.mean()
    return CurvatureResult(
        diagnostic="local_vs_global_dim_ratio",
        k=k,
        mean=float(ratios.mean()),
        ci_low=float(np.percentile(boots, 2.5)),
        ci_high=float(np.percentile(boots, 97.5)),
        n_samples=N,
        raw=ratios,
    )


# 2. Geodesic / Euclidean distance ratio

def geodesic_euclidean_ratio(
    X: np.ndarray,
    k: int = 30,
    n_pairs: int = 500,
    random_state: int = 42,
    n_bootstrap: int = 100,
) -> CurvatureResult:
    """Mean ratio (graph shortest path) / (Euclidean) over n_pairs random pairs.

    Flat manifold: ratio ~= 1 (geodesics are straight lines).
    Curved manifold: ratio > 1 (geodesics bend along the manifold).
    """
    from scipy.sparse.csgraph import shortest_path
    from scipy.sparse import csr_matrix

    rng = np.random.default_rng(random_state)
    N = X.shape[0]
    if N < k + 1:
        raise ValueError(f"Need at least k+1 points ({k+1}), got {N}.")

    rows, cols, dists, _ = _knn_graph(X, k)
    W = csr_matrix((dists, (rows, cols)), shape=(N, N))
    # Symmetrise by taking the LARGER of the two directed edges. d(i,j)=d(j,i),
    # so for a one-directional kNN edge (reverse entry is an implicit 0),
    # max() keeps the true distance. (W + W.T)/2 instead HALVES such edges,
    # shortening graph geodesics below the straight-line distance and pushing
    # the flat-manifold ratio to ~0.73 when it must be >= 1. See tests/.
    W = W.maximum(W.T)

    n_pairs = min(n_pairs, N * (N - 1) // 2)
    pairs_i = rng.choice(N, n_pairs, replace=True)
    pairs_j = rng.choice(N, n_pairs, replace=True)
    valid = pairs_i != pairs_j
    pairs_i = pairs_i[valid]
    pairs_j = pairs_j[valid]

    unique_sources = np.unique(pairs_i)
    dist_matrix = shortest_path(W, method="D", directed=False, indices=unique_sources)
    source_to_row = {src: idx for idx, src in enumerate(unique_sources)}

    geodesic = dist_matrix[[source_to_row[s] for s in pairs_i], pairs_j]
    euclid   = np.linalg.norm(X[pairs_i] - X[pairs_j], axis=1)

    finite = np.isfinite(geodesic) & (euclid > 0)
    if not finite.any():
        return CurvatureResult("geodesic_euclidean_ratio", k, float("nan"),
                                float("nan"), float("nan"), N)
    ratios = geodesic[finite] / euclid[finite]

    boots = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(ratios, ratios.size, replace=True)
        boots[b] = sample.mean()
    return CurvatureResult(
        diagnostic="geodesic_euclidean_ratio",
        k=k,
        mean=float(ratios.mean()),
        ci_low=float(np.percentile(boots, 2.5)),
        ci_high=float(np.percentile(boots, 97.5)),
        n_samples=N,
        raw=ratios,
    )


# 3. Tangent-space variation

def tangent_space_variation(
    X: np.ndarray,
    k: int = 30,
    intrinsic_dim: int = 5,
    n_anchor_pairs: int = 300,
    random_state: int = 42,
    n_bootstrap: int = 100,
) -> CurvatureResult:
    """Mean principal angle (degrees) between local PCA bases at different
    anchor pairs.

    intrinsic_dim controls how many components define the "tangent space".
    Flat manifold: principal angles ~= 0.
    Curved manifold: principal angles > 0; grows with curvature.
    """
    rng = np.random.default_rng(random_state)
    N = X.shape[0]
    if N < k + 1:
        raise ValueError(f"Need at least k+1 points ({k+1}), got {N}.")

    _, _, _, neighbors = _knn_graph(X, k)

    # Compute local PCA bases at all anchors
    n_anchors = min(2 * int(np.sqrt(n_anchor_pairs)) + 10, N)
    anchor_idx = rng.choice(N, n_anchors, replace=False)
    bases = []
    for i in anchor_idx:
        local = X[neighbors[i]]
        centered = local - local.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        d = min(intrinsic_dim, vt.shape[0])
        bases.append(vt[:d].T)  # columns are basis vectors

    # Sample pairs
    pair_a = rng.choice(len(bases), n_anchor_pairs, replace=True)
    pair_b = rng.choice(len(bases), n_anchor_pairs, replace=True)
    valid = pair_a != pair_b
    pair_a = pair_a[valid]
    pair_b = pair_b[valid]

    angles_deg = np.empty(len(pair_a))
    for idx, (a, b) in enumerate(zip(pair_a, pair_b)):
        Ba, Bb = bases[a], bases[b]
        M = Ba.T @ Bb
        s = np.linalg.svd(M, compute_uv=False)
        s = np.clip(s, -1.0, 1.0)
        principal_angles = np.arccos(s)
        angles_deg[idx] = np.degrees(principal_angles.mean())

    boots = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(angles_deg, angles_deg.size, replace=True)
        boots[b] = sample.mean()
    return CurvatureResult(
        diagnostic="tangent_space_variation_deg",
        k=k,
        mean=float(angles_deg.mean()),
        ci_low=float(np.percentile(boots, 2.5)),
        ci_high=float(np.percentile(boots, 97.5)),
        n_samples=N,
        raw=angles_deg,
    )


# Sweep wrapper

def all_diagnostics(
    X: np.ndarray,
    k_values: Iterable[int] = (10, 30, 100),
    intrinsic_dim: int = 5,
    variance_threshold: float = 0.90,
    n_bootstrap: int = 100,
    random_state: int = 42,
) -> list:
    """Run all three diagnostics over a k-sweep. Returns a list of
    CurvatureResult objects."""
    out = []
    for k in k_values:
        if X.shape[0] <= k + 1:
            logger.warning(f"Skipping k={k}: N={X.shape[0]} too small.")
            continue
        try:
            out.append(local_vs_global_dim_ratio(
                X, k=k, variance_threshold=variance_threshold,
                n_bootstrap=n_bootstrap, random_state=random_state))
        except Exception as e:
            logger.error(f"local_vs_global_dim_ratio failed at k={k}: {e}")
        try:
            out.append(geodesic_euclidean_ratio(
                X, k=k, n_bootstrap=n_bootstrap, random_state=random_state))
        except Exception as e:
            logger.error(f"geodesic_euclidean_ratio failed at k={k}: {e}")
        try:
            out.append(tangent_space_variation(
                X, k=k, intrinsic_dim=intrinsic_dim,
                n_bootstrap=n_bootstrap, random_state=random_state))
        except Exception as e:
            logger.error(f"tangent_space_variation failed at k={k}: {e}")
    return out


# Convenience: compact summary

def summary_table(results: list) -> str:
    """Pretty-print a list of CurvatureResult objects."""
    lines = [f"{'diagnostic':35s} {'k':>4s} {'mean':>10s} {'95% CI':>22s} {'N':>6s}"]
    lines.append("-" * 80)
    for r in results:
        ci = f"[{r.ci_low:.4f}, {r.ci_high:.4f}]"
        lines.append(f"{r.diagnostic:35s} {r.k:>4d} {r.mean:>10.4f} {ci:>22s} {r.n_samples:>6d}")
    return "\n".join(lines)
