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
    nn = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(X)
    dists, _ = nn.kneighbors(X)
    r1, r2 = dists[:, 1], dists[:, 2]
    mask = r1 > 0
    mu = r2[mask] / r1[mask]
    mu = mu[mu > 1.0]  # mu == 1 means coincident; drop
    if mu.size == 0:
        return IDResult("twoNN", float("nan"), float("nan"), float("nan"), X.shape[0], {})

    mu_sorted = np.sort(mu)
    n_mu = mu_sorted.size
    # Empirical CDF over the FULL sample as i/(n+1) (so F < 1 everywhere), THEN
    # keep the lower `fraction` (the linear regime). The previous code computed
    # F = arange(1, cutoff+1)/cutoff on the *truncated* set, which forces F=1.0
    # at the cutoff; then -log(1 - F) -> huge and that single high-leverage
    # point dominates the through-origin least-squares slope, inflating the
    # estimate by ~35-50% (dim 5 -> ~6.7, dim 8 -> ~10.2). See tests/.
    F_full = np.arange(1, n_mu + 1) / (n_mu + 1)
    cutoff = max(2, int(np.ceil(fraction * n_mu)))
    mu_fit = mu_sorted[:cutoff]
    F = F_full[:cutoff]
    x = np.log(mu_fit)
    y = -np.log(1 - F)
    # Linear regression through origin
    d_hat = float((x * y).sum() / (x * x).sum())

    # Bootstrap CI on mu sample
    boots = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = np.sort(rng.choice(mu, mu.size, replace=True))
        Fb = np.arange(1, sample.size + 1) / (sample.size + 1)
        s = sample[:cutoff]
        Fbc = Fb[:cutoff]
        xb = np.log(s); yb = -np.log(1 - Fbc)
        boots[b] = (xb * yb).sum() / (xb * xb).sum()

    return IDResult(
        estimator="twoNN",
        estimate=d_hat,
        ci_low=float(np.percentile(boots, 2.5)),
        ci_high=float(np.percentile(boots, 97.5)),
        n_samples=int(X.shape[0]),
        extras={"fraction": fraction, "n_mu": int(mu.size)},
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
    boots = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(point_pool, point_pool.size, replace=True)
        boots[b] = np.median(sample)

    return IDResult(
        estimator="levina_bickel",
        estimate=float(np.mean(estimates_per_k)),
        ci_low=float(np.percentile(boots, 2.5)),
        ci_high=float(np.percentile(boots, 97.5)),
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
    rng = np.random.default_rng(random_state)
    N = X.shape[0]
    if N > subsample:
        idx = rng.choice(N, subsample, replace=False)
        X = X[idx]
        N = subsample

    # Pairwise distances (upper triangle)
    from scipy.spatial.distance import pdist
    d = pdist(X)
    if d.size == 0 or d.max() == 0:
        return IDResult("correlation_dim", float("nan"), float("nan"), float("nan"), N, {})

    # Use log-spaced radii between robust quantiles
    rmin = float(np.percentile(d, 2))
    rmax = float(np.percentile(d, 98))
    if rmin <= 0 or rmax <= rmin:
        return IDResult("correlation_dim", float("nan"), float("nan"), float("nan"), N, {})
    radii = np.logspace(np.log10(rmin), np.log10(rmax), n_radii)
    Cr = np.array([float((d < r).sum()) / d.size for r in radii])
    valid = Cr > 0
    if valid.sum() < 3:
        return IDResult("correlation_dim", float("nan"), float("nan"), float("nan"), N, {})
    x = np.log(radii[valid]); y = np.log(Cr[valid])
    # Fit in the middle 60% (drop the saturation tails)
    lo, hi = int(0.2 * x.size), int(0.8 * x.size)
    if hi - lo < 3:
        lo, hi = 0, x.size
    slope, intercept = np.polyfit(x[lo:hi], y[lo:hi], 1)

    # Bootstrap on the pairwise-distance set
    boots = np.empty(n_bootstrap)
    boots[:] = np.nan
    for b in range(n_bootstrap):
        s = rng.choice(d, d.size, replace=True)
        Cb = np.array([float((s < r).sum()) / s.size for r in radii])
        vb = Cb > 0
        if vb.sum() < 3:
            continue
        xb = np.log(radii[vb]); yb = np.log(Cb[vb])
        lo, hi = int(0.2 * xb.size), int(0.8 * xb.size)
        if hi - lo < 3:
            lo, hi = 0, xb.size
        boots[b] = np.polyfit(xb[lo:hi], yb[lo:hi], 1)[0]
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        ci_low, ci_high = float("nan"), float("nan")
    else:
        ci_low  = float(np.percentile(boots, 2.5))
        ci_high = float(np.percentile(boots, 97.5))

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
