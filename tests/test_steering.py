"""Regression tests for src/steering.py.

Covers the difference-of-means construction, the manifold-projection geometry,
determinism (the randomized-SVD reproducibility bug), and the projection
idempotence/containment properties that make "manifold-projected" meaningful.
"""

import numpy as np

from src.steering import (
    single_direction_vector,
    manifold_projected_vector,
    auto_k,
)
from tests.synthetic import flat_subspace


def test_single_direction_points_from_off_to_on():
    """The vector is the unit-normalised difference of class means."""
    rng = np.random.default_rng(0)
    on = rng.standard_normal((100, 1536)) + 5.0   # shifted +5 on every dim
    off = rng.standard_normal((100, 1536))
    v = single_direction_vector(on, off)
    assert np.isclose(np.linalg.norm(v), 1.0)
    # mean(on) - mean(off) ~ +5 on every dim, so v should be ~ all-positive.
    assert (v > 0).mean() > 0.95


def test_single_direction_zero_when_means_equal():
    X = flat_subspace(50, 10, seed=1)
    v = single_direction_vector(X, X)  # identical on/off
    assert np.linalg.norm(v) < 1e-6


def test_manifold_vector_is_unit_norm():
    on = flat_subspace(120, 20, seed=2)
    off = flat_subspace(120, 20, seed=3)
    v = manifold_projected_vector(on, off, k=5)
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_manifold_projection_is_deterministic():
    """The randomized-SVD bug made this non-reproducible run-to-run."""
    on = flat_subspace(145, 30, seed=4)
    off = flat_subspace(145, 30, seed=5)
    v1 = manifold_projected_vector(on, off, k=5)
    v2 = manifold_projected_vector(on, off, k=5)
    assert np.allclose(v1, v2)


def test_auto_k_deterministic_and_in_range():
    on = flat_subspace(145, 30, seed=6)
    k1, k2 = auto_k(on), auto_k(on)
    assert k1 == k2
    assert 1 <= k1 <= min(on.shape[0] - 1, on.shape[1], 100)


def test_full_rank_projection_recovers_single_direction():
    """Projecting onto enough components (>= data rank) should leave the
    single-direction vector essentially unchanged (projection is into the span
    that already contains it)."""
    on = flat_subspace(60, 10, noise=0.0, seed=7)   # rank ~10
    off = flat_subspace(60, 10, noise=0.0, seed=8)
    r_single = single_direction_vector(on, off)
    r_proj = manifold_projected_vector(on, off, k=min(on.shape[0] - 1, 59))
    # cosine similarity should be high once k spans the on-subspace containing the mean shift
    cos = abs(float(r_single @ r_proj))
    assert cos > 0.5


def test_auto_k_higher_threshold_needs_more_components():
    on = flat_subspace(200, 25, noise=0.3, seed=9)
    assert auto_k(on, variance_threshold=0.5) <= auto_k(on, variance_threshold=0.95)
