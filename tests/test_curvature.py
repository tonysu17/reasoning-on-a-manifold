"""Known-answer regression tests for src/curvature.py.

The headline assertion is the one that the original code FAILED: a perfectly
flat linear subspace must score ~1.0 on the local-vs-global dimension ratio.
Before the calibration fix it scored 0.29-0.85 on flat data, making the
"evidence for curvature" (0.13-0.24 in PROGRESS.md) indistinguishable from the
flat null. These tests lock the calibration in place.
"""

import numpy as np
import pytest

from src.curvature import (
    local_vs_global_dim_ratio,
    geodesic_euclidean_ratio,
    tangent_space_variation,
    local_pca_dim,
)
from tests.synthetic import flat_subspace, sphere, helix


# ── local-vs-global dimension ratio: the calibration that was broken ─────────

@pytest.mark.parametrize("dim", [5, 10, 30])
@pytest.mark.parametrize("k", [10, 30])
def test_flat_subspace_scores_near_one(dim, k):
    """A flat linear subspace must give ratio ~1.0 regardless of dim or k."""
    X = flat_subspace(n=200, dim=dim, seed=dim + k)
    r = local_vs_global_dim_ratio(X, k=k, n_bootstrap=40)
    assert 0.9 <= r.mean <= 1.1, f"flat dim={dim} k={k} gave {r.mean:.3f}, expected ~1.0"


def test_curved_sphere_scores_below_one():
    """A curved 2-sphere must score clearly below 1 (local patch flatter than
    a global sample of the same size)."""
    X = sphere(n=300, seed=3)
    r = local_vs_global_dim_ratio(X, k=20, n_bootstrap=40)
    assert r.mean < 0.9, f"sphere gave {r.mean:.3f}, expected < 0.9"


def test_flat_strictly_higher_than_curved():
    """Ordering must hold: flat ratio > curved ratio (the diagnostic's whole point)."""
    flat = local_vs_global_dim_ratio(flat_subspace(300, 10, seed=7), k=20, n_bootstrap=40).mean
    curved = local_vs_global_dim_ratio(sphere(300, seed=7), k=20, n_bootstrap=40).mean
    assert flat > curved + 0.1


def test_ratio_is_deterministic():
    X = flat_subspace(150, 8, seed=1)
    a = local_vs_global_dim_ratio(X, k=15, n_bootstrap=30, random_state=42).mean
    b = local_vs_global_dim_ratio(X, k=15, n_bootstrap=30, random_state=42).mean
    assert a == b


def test_raises_when_too_few_points():
    X = flat_subspace(10, 3, seed=0)
    with pytest.raises(ValueError):
        local_vs_global_dim_ratio(X, k=30)


# ── geodesic / Euclidean ratio ───────────────────────────────────────────────

def test_geodesic_flat_near_one():
    """On a flat subspace, graph geodesics ~ Euclidean distances => ratio ~1."""
    X = flat_subspace(300, 5, seed=2)
    r = geodesic_euclidean_ratio(X, k=15, n_pairs=300, n_bootstrap=30)
    assert r.mean >= 0.99, f"flat geodesic ratio {r.mean:.3f} should be ~1 (>=1)"


def test_geodesic_curved_exceeds_flat():
    """A curved manifold bends geodesics => ratio strictly larger than flat."""
    flat = geodesic_euclidean_ratio(flat_subspace(300, 3, seed=5), k=12, n_bootstrap=30).mean
    curved = geodesic_euclidean_ratio(helix(300, seed=5), k=12, n_bootstrap=30).mean
    assert curved > flat


# ── tangent-space variation ──────────────────────────────────────────────────

def test_tangent_variation_flat_is_small():
    """A flat subspace has a constant tangent space => small principal angles."""
    X = flat_subspace(300, 6, seed=4)
    r = tangent_space_variation(X, k=30, intrinsic_dim=6, n_bootstrap=30)
    assert r.mean < 20.0, f"flat tangent variation {r.mean:.1f} deg should be small"


def test_tangent_variation_curved_larger_than_flat():
    flat = tangent_space_variation(flat_subspace(300, 3, seed=6), k=30, intrinsic_dim=3, n_bootstrap=30).mean
    curved = tangent_space_variation(sphere(300, seed=6), k=30, intrinsic_dim=2, n_bootstrap=30).mean
    assert curved > flat


# ── local_pca_dim primitive ──────────────────────────────────────────────────

def test_local_pca_dim_recovers_flat_dimension():
    """At 100% variance threshold, local_pca_dim of a dim-D subspace is D."""
    X = flat_subspace(200, 7, noise=0.0, seed=9)
    assert local_pca_dim(X, variance_threshold=0.999) == 7


def test_local_pca_dim_degenerate_returns_zero():
    assert local_pca_dim(np.zeros((5, 1536))) == 0
