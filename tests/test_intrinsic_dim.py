"""Known-answer regression tests for src/intrinsic_dim.py and the local
intrinsic-dimension estimator in src/cbs/geometry.py.

The CBS `local_intrinsic_dim` was ~3-4x biased (returned 8-13 for true dim 3);
these tests pin the corrected Levina-Bickel MLE to the truth and keep the
legacy biased estimator documented as such.
"""

import numpy as np
import pytest

from src.intrinsic_dim import (
    twoNN_estimate,
    levina_bickel_estimate,
    correlation_dimension_estimate,
)
from src.cbs.geometry import local_intrinsic_dim
from tests.synthetic import flat_subspace


# ── global estimators recover planted dimension ──────────────────────────────

@pytest.mark.parametrize("true_dim", [3, 5, 8])
def test_twoNN_recovers_planted_dim(true_dim):
    X = flat_subspace(n=800, dim=true_dim, noise=0.0, seed=true_dim)
    est = twoNN_estimate(X, n_bootstrap=50).estimate
    # TwoNN is approximately unbiased on flat data given enough samples.
    assert abs(est - true_dim) <= max(1.0, 0.25 * true_dim), f"twoNN={est:.2f} vs {true_dim}"


@pytest.mark.parametrize("true_dim", [3, 5, 8])
def test_levina_bickel_recovers_planted_dim(true_dim):
    X = flat_subspace(n=800, dim=true_dim, noise=0.0, seed=10 + true_dim)
    est = levina_bickel_estimate(X, n_bootstrap=50).estimate
    assert abs(est - true_dim) <= max(1.5, 0.35 * true_dim), f"LB={est:.2f} vs {true_dim}"


def test_correlation_dimension_runs_and_is_positive():
    X = flat_subspace(500, 5, noise=0.0, seed=1)
    r = correlation_dimension_estimate(X, n_bootstrap=30)
    assert r.estimate > 0


def test_twoNN_small_sample_returns_finite_or_nan_not_crash():
    X = flat_subspace(8, 3, seed=0)
    r = twoNN_estimate(X, n_bootstrap=10)
    assert np.isnan(r.estimate) or r.estimate > 0


# ── corrected LOCAL intrinsic dim (the one that was broken) ──────────────────

@pytest.mark.parametrize("true_dim", [3, 5])
@pytest.mark.parametrize("k", [20, 40])
def test_local_mle_recovers_dim(true_dim, k):
    """The default (MLE) local estimator must recover the planted dimension to
    within a small tolerance — NOT the 3-4x overshoot of the legacy version."""
    X = flat_subspace(400, true_dim, noise=0.0, seed=true_dim + k)
    med = np.nanmedian(local_intrinsic_dim(X, k=k))
    assert abs(med - true_dim) <= 1.5, f"MLE local dim={med:.2f} vs {true_dim} (k={k})"


def test_legacy_twoNN_local_is_biased_high():
    """Document the legacy estimator's bias so nobody silently relies on it."""
    X = flat_subspace(400, 3, noise=0.0, seed=2)
    legacy = np.nanmedian(local_intrinsic_dim(X, k=20, estimator="twoNN"))
    mle = np.nanmedian(local_intrinsic_dim(X, k=20, estimator="mle"))
    assert legacy > mle + 2.0  # legacy overshoots; mle is near 3


def test_local_intrinsic_dim_small_N_all_nan():
    X = flat_subspace(10, 3, seed=0)
    out = local_intrinsic_dim(X, k=20)
    assert out.shape == (10,) and np.all(np.isnan(out))


def test_local_intrinsic_dim_rejects_unknown_estimator():
    X = flat_subspace(50, 3, seed=0)
    with pytest.raises(ValueError):
        local_intrinsic_dim(X, k=10, estimator="bogus")
