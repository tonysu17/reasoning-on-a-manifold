"""
src/cbs/geometry.py — Per-sentence geometric tests for CBS-tier and
cross-domain labels (M2).

Purpose
-------
At each saved layer, test whether CBS tier (and, separately, the binary
cross-domain flag) correlates with four geometric quantities of the
sentence-level activation vector:

  1. centroid distance,
  2. out-of-subspace residual against the union of behaviour subspaces,
  3. local intrinsic dimension via TwoNN over k-NN,
  4. principal angles between behaviour-subspace pairs.

Each test is run twice (once under each label) so both signals are
preserved (synthesis §M2.1).

Validation
----------
Shuffle test (|Cliff's delta| < 0.05); reversal test (JT sign flip);
category-stratified rerun; smoke unit tests in src/cbs/tests/test_geometry.py.

Milestone
---------
M2 (synthesis §M2).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ── Centroid distance ───────────────────────────────────────────────────────

def centroid_distance(X: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Per-row L2 distance ||x_i - mu||.

    Parameters
    ----------
    X        : (N, d)
    centroid : (d,)  — typically `X.mean(axis=0)` from a baseline cohort.

    Returns
    -------
    (N,) array of distances.
    """
    X = np.asarray(X)
    centroid = np.asarray(centroid).reshape(-1)
    if X.shape[1] != centroid.shape[0]:
        raise ValueError(f"shape mismatch: X.shape={X.shape}, "
                         f"centroid.shape={centroid.shape}")
    return np.linalg.norm(X - centroid[None, :], axis=1)


# ── Out-of-subspace residual ────────────────────────────────────────────────

def out_of_subspace_residual(
    X: np.ndarray,
    union_basis: np.ndarray,
) -> np.ndarray:
    """Per-row ||(I - V V^T) x||_2 / ||x||_2.

    `union_basis` is a (d, k) orthonormal basis (typically the output of
    `build_union_basis`). Ratio is clipped to [0, 1]; zero-norm rows return 0.
    """
    X = np.asarray(X)
    V = np.asarray(union_basis)
    if V.ndim != 2 or X.shape[1] != V.shape[0]:
        raise ValueError(f"shape mismatch: X.shape={X.shape}, V.shape={V.shape}")
    coef = X @ V                       # (N, k)
    proj = coef @ V.T                  # (N, d)
    residual = X - proj
    res_norm = np.linalg.norm(residual, axis=1)
    x_norm = np.linalg.norm(X, axis=1)
    safe = np.where(x_norm > 0, x_norm, 1.0)
    out = res_norm / safe
    out[x_norm == 0] = 0.0
    return np.clip(out, 0.0, 1.0)


# ── Principal angles ────────────────────────────────────────────────────────

def principal_angles(
    V_a: np.ndarray,
    V_b: np.ndarray,
    top_k: int = 10,
) -> np.ndarray:
    """Top-k principal angles (radians) between span(V_a) and span(V_b),
    sorted ascending. Standard SVD-based formulation: singular values of
    V_a^T V_b are cosines of the principal angles."""
    V_a = np.asarray(V_a)
    V_b = np.asarray(V_b)
    if V_a.ndim != 2 or V_b.ndim != 2 or V_a.shape[0] != V_b.shape[0]:
        raise ValueError(f"shape mismatch: V_a={V_a.shape}, V_b={V_b.shape}")
    s = np.linalg.svd(V_a.T @ V_b, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s)
    angles.sort()
    return angles[:top_k]


# ── Union-of-behaviour-subspaces basis ──────────────────────────────────────

def build_union_basis(
    per_behaviour_pcs: dict[str, np.ndarray],
    variance_threshold: float = 0.95,
    per_behaviour_weights: "dict[str, np.ndarray] | None" = None,
) -> np.ndarray:
    """Concatenate per-behaviour top PCs, orthonormalise via SVD, return basis
    covering `variance_threshold` of the joint variance (default 0.95).

    `per_behaviour_pcs` maps behaviour name -> (d, k_b) PC directions (columns).

    IMPORTANT (calibration): PC *directions* are unit-norm, so WITHOUT weighting
    the SVD singular values reflect only the geometric overlap of the directions,
    not how much activation variance each captures — `variance_threshold` is then
    a cut on direction-spectral-energy, NOT activation variance. Pass
    `per_behaviour_weights` (behaviour -> (k_b,) explained variance / eigenvalues)
    to scale each column by sqrt(eigenvalue) so the threshold means activation
    variance as advertised. See AUDIT.md §5 and tests/test_cbs_geometry_extra.py.
    """
    if not per_behaviour_pcs:
        raise ValueError("per_behaviour_pcs is empty")
    d = next(iter(per_behaviour_pcs.values())).shape[0]
    blocks = []
    for beh, V in per_behaviour_pcs.items():
        if V.ndim != 2 or V.shape[0] != d:
            raise ValueError(f"PC block shape mismatch: expected (d={d}, _), "
                             f"got {V.shape} for {beh!r}")
        if per_behaviour_weights is not None and beh in per_behaviour_weights:
            w = np.asarray(per_behaviour_weights[beh], dtype=float)
            if w.shape[0] != V.shape[1]:
                raise ValueError(f"weights for {beh!r} have {w.shape[0]} entries, "
                                 f"expected {V.shape[1]}")
            V = V * np.sqrt(np.maximum(w, 0.0))[None, :]
        blocks.append(V)
    stacked = np.concatenate(blocks, axis=1)
    # Use SVD to get orthonormal columns ordered by singular value (variance).
    U, S, _ = np.linalg.svd(stacked, full_matrices=False)
    if S.size == 0 or np.sum(S * S) == 0:
        return np.zeros((d, 0))
    cumvar = np.cumsum(S * S) / np.sum(S * S)
    n_keep = int(np.searchsorted(cumvar, variance_threshold) + 1)
    n_keep = max(1, min(n_keep, U.shape[1]))
    return U[:, :n_keep]


# ── Cliff's delta ───────────────────────────────────────────────────────────

def cliffs_delta(a: Sequence, b: Sequence) -> float:
    """Cliff's delta = P(A > B) - P(B > A). Range: [-1, 1].
    Returns 0.0 if either group is empty."""
    a = np.asarray(a)
    b = np.asarray(b)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0.0
    # Brute force is O(na*nb) which is fine at thesis scale (N <= a few hundred).
    A = a[:, None]
    B = b[None, :]
    gt = np.sum(A > B)
    lt = np.sum(A < B)
    return float(gt - lt) / (na * nb)


# ── Jonckheere-Terpstra trend test (1 -> 2 -> 3) ────────────────────────────

def jonckheere_terpstra(
    values: Sequence,
    tiers: Sequence,
) -> dict:
    """Jonckheere-Terpstra ordinal-trend test.

    Tests H0: no difference in distribution across tier groups, vs
    H1: there is an ordered trend (positive or negative).

    Returns
    -------
    {
      "statistic":       JT statistic,
      "expected":        E[JT] under H0,
      "variance":        Var[JT] under H0,
      "z":               (JT - E) / sqrt(Var),
      "p_value":         two-sided p,
      "trend_direction": -1, 0, or +1,
      "n_total":         total sample size,
      "n_per_tier":      {tier: count},
    }
    """
    values = np.asarray(values)
    tiers = np.asarray(tiers)
    if values.shape != tiers.shape:
        raise ValueError(f"shape mismatch: values={values.shape}, "
                         f"tiers={tiers.shape}")
    unique_tiers = np.unique(tiers)
    if unique_tiers.size < 2:
        return {
            "statistic": 0.0, "expected": 0.0, "variance": 0.0,
            "z": 0.0, "p_value": 1.0, "trend_direction": 0,
            "n_total": int(values.size),
            "n_per_tier": {int(t): int(np.sum(tiers == t)) for t in unique_tiers},
        }
    sorted_tiers = sorted(unique_tiers.tolist())
    groups = [values[tiers == t] for t in sorted_tiers]

    # JT statistic = sum_{i<j} U_{ij}, where U_{ij} = #(x in tier_i, y in tier_j with x < y) + 0.5 * ties.
    JT = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            x = groups[i][:, None]
            y = groups[j][None, :]
            JT += float(np.sum(x < y) + 0.5 * np.sum(x == y))

    n = int(values.size)
    ni = np.array([g.size for g in groups], dtype=float)
    sum_ni2 = float(np.sum(ni * ni))
    mean_jt = (n * n - sum_ni2) / 4.0
    # NOTE: this is the NO-TIES variance. It is exact for continuous inputs
    # (ties are measure-zero) and CONSERVATIVE under ties. The statistic itself
    # credits ties (+0.5). We warn rather than silently approximate if a
    # discretized input arrives with non-negligible ties (AUDIT.md §5).
    vs = np.sort(values)
    n_ties = int(np.sum(vs[1:] == vs[:-1])) if n > 1 else 0
    if n > 1 and n_ties / (n - 1) > 0.05:
        logger.warning("jonckheere_terpstra: %.0f%% of values are tied; the "
                       "no-ties variance is used (conservative) so the p-value "
                       "is approximate.", 100.0 * n_ties / (n - 1))
    var_jt = (n * n * (2 * n + 3) - float(np.sum(ni * ni * (2 * ni + 3)))) / 72.0

    if var_jt <= 0:
        z = 0.0
        p = 1.0
    else:
        z = (JT - mean_jt) / np.sqrt(var_jt)
        # Two-sided p via normal approximation.
        # Use scipy if available, otherwise erfc.
        try:
            from scipy.stats import norm
            p = float(2.0 * (1.0 - norm.cdf(abs(z))))
        except ImportError:
            from math import erfc, sqrt
            p = float(erfc(abs(z) / sqrt(2.0)))

    direction = 1 if JT > mean_jt else (-1 if JT < mean_jt else 0)
    return {
        "statistic": float(JT),
        "expected": float(mean_jt),
        "variance": float(var_jt),
        "z": float(z),
        "p_value": float(p),
        "trend_direction": int(direction),
        "n_total": n,
        "n_per_tier": {int(t): int(g.size) for t, g in zip(sorted_tiers, groups)},
    }


# ── Local intrinsic dimension via TwoNN over k-NN ───────────────────────────

def _twoNN_point_estimate(X: np.ndarray, fraction: float = 0.9) -> float:
    """Point estimate (no bootstrap) of TwoNN intrinsic dim on a k-NN cloud."""
    from sklearn.neighbors import NearestNeighbors
    n = X.shape[0]
    if n < 3:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    dists, _ = nn.kneighbors(X)
    r1, r2 = dists[:, 1], dists[:, 2]
    mask = r1 > 0
    mu = r2[mask] / r1[mask]
    mu = mu[mu > 1.0]
    if mu.size < 2:
        return float("nan")
    mu_sorted = np.sort(mu)
    cutoff = max(2, int(np.ceil(fraction * mu_sorted.size)))
    mu_fit = mu_sorted[:cutoff]
    F = np.arange(1, mu_fit.size + 1) / mu_fit.size
    x = np.log(mu_fit)
    y = -np.log(1.0 - F + 1e-12)
    denom = float(np.sum(x * x))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def local_intrinsic_dim(
    X: np.ndarray,
    k: int = 20,
    estimator: str = "mle",
) -> np.ndarray:
    """Per-row local intrinsic-dimension estimate over each point's k-NN cloud.

    estimator="mle" (default): Levina-Bickel maximum-likelihood estimator,
        m_k(i) = [ (1/(k-1)) * sum_{j=1}^{k-1} log(r_k(i) / r_j(i)) ]^{-1},
        where r_j(i) is the distance from point i to its j-th nearest
        neighbour. This is the standard, low-bias local ID estimator.

    estimator="twoNN": LEGACY per-row TwoNN fit on the k-NN cloud. Retained for
        backward compatibility ONLY — it is severely upward-biased on small
        clouds (empirically returns ~9-13 for a true dimension of 3). Do not
        use for new analysis. See tests/test_intrinsic_dim.py.

    Returns (N,) array of estimates; NaN entries indicate degenerate clouds
    (too few neighbours or zero distances). N < k + 1 produces all-NaN.
    """
    X = np.asarray(X)
    n = X.shape[0]
    if n < k + 1:
        return np.full(n, np.nan)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists, indices = nn.kneighbors(X)

    if estimator == "twoNN":
        out = np.empty(n)
        for i in range(n):
            out[i] = _twoNN_point_estimate(X[indices[i]])
        return out
    if estimator != "mle":
        raise ValueError(f"unsupported estimator: {estimator!r}")

    # Levina-Bickel MLE. dists[:, 0] is the self-distance (0); use cols 1..k.
    r = dists[:, 1:k + 1]                 # (n, k): r_1 <= ... <= r_k
    rk = r[:, -1:]                        # (n, 1): r_k
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratios = np.log(rk / r[:, :-1])           # (n, k-1): log(r_k / r_j)
        denom = log_ratios.sum(axis=1) / (k - 1)
        m = 1.0 / denom
    m = np.where(np.isfinite(m), m, np.nan)
    m[(r <= 0).any(axis=1)] = np.nan      # degenerate (coincident) neighbours
    return m


# ── Bootstrap CI ────────────────────────────────────────────────────────────

def bootstrap_ci(
    fn: Callable[..., float],
    *args: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
    paired: bool = True,
    **kwargs: Any,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a scalar statistic.

    Parameters
    ----------
    fn          : callable returning a scalar (or anything castable to float).
    *args       : numpy arrays passed positionally to `fn` after resampling.
    paired      : if True, all args resampled with the same index (assumed
                  same length). If False, each arg gets its own independent
                  resampling (use for two-group statistics with different
                  group sizes, e.g. Cliff's delta).
    n_bootstrap : number of bootstrap iterations.
    ci          : nominal coverage (e.g. 0.95).
    rng         : numpy Generator (`np.random.default_rng()`).

    Returns
    -------
    (lo, hi) percentile-bootstrap CI.
    """
    if rng is None:
        rng = np.random.default_rng()
    arr_args = [np.asarray(a) for a in args]
    if paired:
        if not arr_args:
            raise ValueError("bootstrap_ci(paired=True) needs at least one array arg")
        n = arr_args[0].shape[0]
        for a in arr_args[1:]:
            if a.shape[0] != n:
                raise ValueError(f"paired bootstrap needs same-length args; "
                                 f"got {a.shape[0]} vs {n}")
        stats = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            stats[b] = float(fn(*[a[idx] for a in arr_args], **kwargs))
    else:
        stats = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            resampled = []
            for a in arr_args:
                idx = rng.integers(0, a.shape[0], size=a.shape[0])
                resampled.append(a[idx])
            stats[b] = float(fn(*resampled, **kwargs))
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


# ── Holm correction ─────────────────────────────────────────────────────────

def holm_correction(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down correction. Returns adjusted p-values aligned with input."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p.copy()
    order = np.argsort(p)
    sorted_p = p[order]
    adj_sorted = np.empty(n)
    running = 0.0
    for i in range(n):
        running = max(running, sorted_p[i] * (n - i))
        adj_sorted[i] = min(running, 1.0)
    adj = np.empty(n)
    adj[order] = adj_sorted
    return adj


__all__ = [
    "centroid_distance",
    "out_of_subspace_residual",
    "principal_angles",
    "build_union_basis",
    "cliffs_delta",
    "jonckheere_terpstra",
    "local_intrinsic_dim",
    "bootstrap_ci",
    "holm_correction",
]
