"""
Intrinsic dimension estimators for per-behaviour activation manifolds.

Three estimators with bootstrap CIs:

  TwoNN (Facco et al. 2017)
    Uses the ratio of distances to 1st and 2nd nearest neighbours.
    Cumulative distribution of mu = r2/r1 follows Pareto with shape = id.
    Robust against curvature and non-uniform density at the local scale.

  Levina-Bickel MLE (Levina & Bickel 2004)
    Maximum-likelihood estimator using distances to k nearest neighbours.
    Reported across multiple k for stability.

  Correlation dimension (Grassberger-Procaccia)
    Slope of log(correlation sum) vs log(radius). Captures fractal-style
    dimension for non-integer cases.

Convergent estimates across the three support claims about effective
dimension; divergent estimates flag the methodological caveat that
intrinsic dimension is an estimand whose value depends on the estimator.

Companion document Section 2.5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IDResult:
    estimator:  str
    estimate:   float
    ci_low:     float
    ci_high:    float
    n_samples:  int
    extras:     dict


def _subsample_bootstrap(X, fit_fn, rng, n_bootstrap, frac=0.8, min_points=4):
    """CI by SUBSAMPLING points (m = frac*N, without replacement) and recomputing
    fit_fn end-to-end. Subsampling (not n-out-of-n resampling) avoids duplicate
    points whose zero-distance NN neighbours would corrupt these estimators;
    recomputing on points — rather than resampling derived mu / pairwise
    distances, which are dependent — gives an honest, not-too-narrow CI.
    See AUDIT.md §5 (#16). Returns the finite bootstrap draws."""
    N = X.shape[0]
    m = max(min_points, min(int(round(frac * N)), N - 1))
    out = []
    for _ in range(n_bootstrap):
        idx = rng.choice(N, m, replace=False)
        try:
            v = fit_fn(X[idx])
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            continue
        if np.isfinite(v):
            out.append(float(v))
    return np.asarray(out)


def _ci(boots):
    if boots.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# TwoNN

def twoNN_estimate(X: np.ndarray, fraction: float = 0.9,
                   random_state: int = 42, n_bootstrap: int = 200) -> IDResult:
    """TwoNN (Facco et al. 2017) intrinsic dimension estimator.

    Given mu_i = r2_i / r1_i (the ratio of distances to the 1st and 2nd
    nearest neighbours), the cumulative distribution F(mu) = 1 - mu^(-d) on
    the right side fits as a linear regression of -log(1 - F) vs log(mu).
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(random_state)

    def _fit(Y):
        """TwoNN point estimate on point cloud Y (raises ValueError if degenerate)."""
        nn = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(Y)
        dists, _ = nn.kneighbors(Y)
        r1, r2 = dists[:, 1], dists[:, 2]
        m = r1 > 0
        mu = r2[m] / r1[m]
        mu = mu[mu > 1.0]  # mu == 1 means coincident; drop
        if mu.size < 2:
            raise ValueError("insufficient non-degenerate mu")
        mu_sorted = np.sort(mu)
        n_mu = mu_sorted.size
        # Empirical CDF over the FULL sample as i/(n+1) (so F < 1 everywhere),
        # THEN keep the lower `fraction` (the linear regime). Computing
        # F = arange(1,cutoff+1)/cutoff on the *truncated* set instead forces
        # F=1.0 at the cutoff; -log(1-F) then explodes and that single
        # high-leverage point dominates the through-origin slope, inflating the
        # estimate ~35-50% (dim 5 -> ~6.7, dim 8 -> ~10.2). See tests/.
        F_full = np.arange(1, n_mu + 1) / (n_mu + 1)
        cutoff = max(2, int(np.ceil(fraction * n_mu)))
        x = np.log(mu_sorted[:cutoff])
        yv = -np.log(1 - F_full[:cutoff])
        denom = float((x * x).sum())
        if denom <= 0:
            raise ValueError("degenerate fit")
        return float((x * yv).sum() / denom)

    try:
        d_hat = _fit(X)
    except ValueError:
        return IDResult("twoNN", float("nan"), float("nan"), float("nan"), X.shape[0], {})

    # CI by point-subsample bootstrap (recompute the fit on resampled points),
    # not by resampling the per-point mu values (which fixes the kNN graph and
    # gives too-narrow CIs). See AUDIT.md §5 (#16).
    boots = _subsample_bootstrap(X, _fit, rng, n_bootstrap)
    ci_low, ci_high = _ci(boots)
    return IDResult(
        estimator="twoNN",
        estimate=d_hat,
        ci_low=ci_low,
        ci_high=ci_high,
        n_samples=int(X.shape[0]),
        extras={"fraction": fraction},
    )


# Levina-Bickel MLE

def levina_bickel_estimate(X: np.ndarray, k_values=(5, 10, 20, 30),
                            random_state: int = 42, n_bootstrap: int = 200) -> IDResult:
    """Levina-Bickel maximum-likelihood estimator over a range of k.

    For each point i and each k,
      m_k(i) = ( (1/(k-1)) sum_{j=1..k-1} log(r_k(i) / r_j(i)) )^{-1}
    Returns the median across i and the mean across k_values, plus a
    bootstrap CI on the per-point estimates.
    """
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(random_state)
    # Cap k_max at N-1 so small samples still run
    k_max = min(max(k_values) + 1, X.shape[0] - 1)
    if k_max < 3:
        return IDResult("levina_bickel", float("nan"), float("nan"), float("nan"), X.shape[0], {})
    # Restrict k_values to those that fit
    k_values = tuple(k for k in k_values if k <= k_max - 1)
    if not k_values:
        return IDResult("levina_bickel", float("nan"), float("nan"), float("nan"), X.shape[0], {})
    nn = NearestNeighbors(n_neighbors=k_max, algorithm="auto").fit(X)
    dists, _ = nn.kneighbors(X)
    # dists[:, 0] is the self-distance (zero); shift index
    dists = dists[:, 1:]  # now col j is the (j+1)th nearest neighbour distance

    estimates_per_k = []
    per_point_estimates = []
    for k in k_values:
        # Need columns 0..k-1 (which are r1..rk)
        if k > dists.shape[1]:
            continue
        sub = dists[:, :k]   # shape (N, k)
        rk = sub[:, -1:]     # shape (N, 1)
        mask = (sub > 0).all(axis=1)
        if not mask.any():
            continue
        log_ratios = np.log(rk[mask] / sub[mask, :-1])  # shape (N_valid, k-1)
        m_k = 1.0 / ((1.0 / (k - 1)) * log_ratios.sum(axis=1))
        finite = np.isfinite(m_k) & (m_k > 0)
        if finite.any():
            estimates_per_k.append(float(np.median(m_k[finite])))
            per_point_estimates.append(m_k[finite])

    if not estimates_per_k:
        return IDResult("levina_bickel", float("nan"), float("nan"), float("nan"), X.shape[0], {})

    point_pool = np.concatenate(per_point_estimates)
    if n_bootstrap > 0 and point_pool.size > 0:
        boots = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            sample = rng.choice(point_pool, point_pool.size, replace=True)
            boots[b] = np.median(sample)
    else:
        # n_bootstrap=0 (e.g. point estimate inside a permutation null) or an
        # empty pool: skip the CI rather than crash on np.percentile([]). Matches
        # twoNN_estimate, which routes through the empty-guarded _ci helper.
        boots = np.empty(0)
    ci_low, ci_high = _ci(boots)

    return IDResult(
        estimator="levina_bickel",
        estimate=float(np.mean(estimates_per_k)),
        ci_low=ci_low,
        ci_high=ci_high,
        n_samples=int(X.shape[0]),
        extras={"per_k_estimates": estimates_per_k, "k_values": list(k_values)},
    )


# Correlation dimension (Grassberger-Procaccia)

def correlation_dimension_estimate(X: np.ndarray,
                                     n_radii: int = 20,
                                     random_state: int = 42,
                                     n_bootstrap: int = 100,
                                     subsample: int = 2000) -> IDResult:
    """Grassberger-Procaccia correlation dimension.

    C(r) = (1/N^2) sum_{i != j} I(|x_i - x_j| < r)
    log C(r) ~= d * log(r) for r in the "scaling" range.

    We fit the slope in a robust middle-decade of r values.
    """
    from scipy.spatial.distance import pdist
    rng = np.random.default_rng(random_state)
    N = X.shape[0]
    if N > subsample:
        X = X[rng.choice(N, subsample, replace=False)]
        N = subsample

    def _fit(Y):
        """Correlation-dimension slope on point cloud Y (raises if degenerate)."""
        dd = pdist(Y)
        if dd.size == 0 or dd.max() == 0:
            raise ValueError("degenerate distances")
        rmin = float(np.percentile(dd, 2))
        rmax = float(np.percentile(dd, 98))
        if rmin <= 0 or rmax <= rmin:
            raise ValueError("degenerate radii")
        radii = np.logspace(np.log10(rmin), np.log10(rmax), n_radii)
        Cr = np.array([float((dd < r).sum()) / dd.size for r in radii])
        v = Cr > 0
        if v.sum() < 3:
            raise ValueError("too few valid radii")
        xx = np.log(radii[v]); yy = np.log(Cr[v])
        lo, hi = int(0.2 * xx.size), int(0.8 * xx.size)
        if hi - lo < 3:
            lo, hi = 0, xx.size
        return float(np.polyfit(xx[lo:hi], yy[lo:hi], 1)[0])

    try:
        slope = _fit(X)
    except ValueError:
        return IDResult("correlation_dim", float("nan"), float("nan"), float("nan"), N, {})

    # CI by point-subsample bootstrap (recompute the slope on resampled points),
    # NOT by resampling the pairwise distances — those are mutually dependent
    # (each point appears in N-1 pairs), so that bootstrap badly understates the
    # CI. See AUDIT.md §5 (#16).
    boots = _subsample_bootstrap(X, _fit, rng, n_bootstrap)
    ci_low, ci_high = _ci(boots)
    return IDResult(
        estimator="correlation_dim",
        estimate=float(slope),
        ci_low=ci_low,
        ci_high=ci_high,
        n_samples=int(N),
        extras={"n_radii": n_radii, "subsample": subsample},
    )


# Convenience: run all three

def all_id_estimators(X: np.ndarray,
                      random_state: int = 42,
                      n_bootstrap: int = 100) -> list:
    out = []
    try:
        out.append(twoNN_estimate(X, random_state=random_state, n_bootstrap=n_bootstrap))
    except Exception as e:
        logger.error(f"twoNN failed: {e}")
    try:
        out.append(levina_bickel_estimate(X, random_state=random_state, n_bootstrap=n_bootstrap))
    except Exception as e:
        logger.error(f"levina_bickel failed: {e}")
    try:
        out.append(correlation_dimension_estimate(X, random_state=random_state, n_bootstrap=n_bootstrap))
    except Exception as e:
        logger.error(f"correlation_dim failed: {e}")
    return out
