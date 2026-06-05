"""Regression tests for the point-subsample bootstrap CIs (AUDIT.md §5 #16).

The old CIs resampled *derived* quantities (per-pair geodesic ratios, per-point
mu), which are dependent, producing absurdly tight bands (e.g. [0.575, 0.587]).
The fix recomputes the whole statistic on resampled POINTS (subsampling without
replacement, to avoid duplicate-point kNN degeneracy). These tests lock in that
the CIs are now non-degenerate, finite, bracket-ish the estimate, and
deterministic — without re-asserting the point estimates (covered elsewhere).
"""

import numpy as np
import pytest

from src.curvature import local_vs_global_dim_ratio, geodesic_euclidean_ratio
from src.intrinsic_dim import twoNN_estimate, correlation_dimension_estimate
from tests.synthetic import flat_subspace


def _nondegenerate(lo, hi):
    return np.isfinite(lo) and np.isfinite(hi) and hi > lo


def test_curvature_ci_is_nondegenerate_and_brackets():
    X = flat_subspace(140, 8, seed=1)
    r = local_vs_global_dim_ratio(X, k=12, n_bootstrap=25)
    assert _nondegenerate(r.ci_low, r.ci_high), (r.ci_low, r.ci_high)
    # estimate sits within (or essentially within) the band
    assert r.ci_low - 0.1 <= r.mean <= r.ci_high + 0.1


def test_geodesic_ci_is_nondegenerate():
    X = flat_subspace(140, 5, seed=2)
    r = geodesic_euclidean_ratio(X, k=12, n_pairs=200, n_bootstrap=20)
    assert _nondegenerate(r.ci_low, r.ci_high)


def test_twoNN_ci_is_nondegenerate_and_brackets():
    X = flat_subspace(400, 5, noise=0.0, seed=3)
    r = twoNN_estimate(X, n_bootstrap=25)
    assert _nondegenerate(r.ci_low, r.ci_high)
    assert r.ci_low - 1.0 <= r.estimate <= r.ci_high + 1.0


def test_correlation_dim_ci_is_nondegenerate():
    X = flat_subspace(400, 5, noise=0.0, seed=4)
    r = correlation_dimension_estimate(X, n_bootstrap=20)
    assert _nondegenerate(r.ci_low, r.ci_high)


@pytest.mark.parametrize("fn,kw", [
    (lambda X: local_vs_global_dim_ratio(X, k=12, n_bootstrap=20), {}),
    (lambda X: twoNN_estimate(X, n_bootstrap=20), {}),
])
def test_ci_is_deterministic(fn, kw):
    X = flat_subspace(150, 6, seed=5)
    a = fn(X)
    b = fn(X)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
