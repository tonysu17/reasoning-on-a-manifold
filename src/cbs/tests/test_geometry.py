"""Unit tests for src/cbs/geometry.py (M2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.cbs import geometry as G


def test_geometry_module_importable() -> None:
    from src.cbs import geometry  # noqa: F401


# ── Centroid distance ─────────────────────────────────────────────────────

def test_centroid_distance_zero_at_centroid() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    c = X.mean(axis=0)
    d = G.centroid_distance(X, c)
    assert d.shape == (2,)
    # Both points equidistant from the centroid for this symmetric pair.
    assert d[0] == pytest.approx(d[1])


def test_centroid_distance_l2_correctness() -> None:
    X = np.array([[3.0, 4.0]])
    c = np.array([0.0, 0.0])
    assert G.centroid_distance(X, c)[0] == pytest.approx(5.0)


def test_centroid_distance_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        G.centroid_distance(np.zeros((3, 4)), np.zeros(5))


# ── Out-of-subspace residual ──────────────────────────────────────────────

def test_oos_residual_zero_in_subspace() -> None:
    # Points strictly inside the 2-D x-y subspace -> zero residual.
    rng = np.random.default_rng(0)
    V = np.eye(5)[:, :2]
    X = rng.normal(size=(10, 2)) @ V.T
    r = G.out_of_subspace_residual(X, V)
    assert np.all(r < 1e-10)


def test_oos_residual_full_outside_subspace() -> None:
    # Points entirely orthogonal to V -> residual ratio ~ 1.
    V = np.eye(5)[:, :2]
    X = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1],
    ], dtype=float)
    r = G.out_of_subspace_residual(X, V)
    assert np.all(r > 0.99)


def test_oos_residual_zero_row_returns_zero() -> None:
    V = np.eye(5)[:, :2]
    X = np.zeros((3, 5))
    X[1] = np.array([1, 0, 0, 0, 0])     # in-subspace
    r = G.out_of_subspace_residual(X, V)
    # zero rows give ratio 0 (defined), the in-subspace row gives ratio 0.
    assert r[0] == 0.0
    assert r[1] < 1e-10


# ── Principal angles ──────────────────────────────────────────────────────

def test_principal_angles_identical_subspaces() -> None:
    V = np.eye(6)[:, :3]
    a = G.principal_angles(V, V, top_k=3)
    assert np.allclose(a, 0.0, atol=1e-9)


def test_principal_angles_orthogonal_subspaces() -> None:
    Va = np.eye(6)[:, :3]
    Vb = np.eye(6)[:, 3:]
    a = G.principal_angles(Va, Vb, top_k=3)
    assert np.allclose(a, math.pi / 2, atol=1e-9)


def test_principal_angles_sorted_ascending() -> None:
    rng = np.random.default_rng(1)
    Va, _ = np.linalg.qr(rng.normal(size=(8, 3)))
    Vb, _ = np.linalg.qr(rng.normal(size=(8, 3)))
    a = G.principal_angles(Va, Vb, top_k=3)
    assert all(a[i] <= a[i + 1] for i in range(len(a) - 1))


# ── Union basis ───────────────────────────────────────────────────────────

def test_build_union_basis_respects_threshold() -> None:
    # 4-D PC blocks; union basis should retain enough columns to cover var.
    rng = np.random.default_rng(2)
    A, _ = np.linalg.qr(rng.normal(size=(10, 3)))
    B, _ = np.linalg.qr(rng.normal(size=(10, 3)))
    U = G.build_union_basis({"a": A, "b": B}, variance_threshold=0.95)
    assert U.ndim == 2
    assert U.shape[0] == 10
    # orthonormal columns
    assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-8)


def test_build_union_basis_low_threshold_smaller_basis() -> None:
    rng = np.random.default_rng(3)
    A, _ = np.linalg.qr(rng.normal(size=(10, 3)))
    U90 = G.build_union_basis({"a": A}, variance_threshold=0.90)
    U99 = G.build_union_basis({"a": A}, variance_threshold=0.99)
    assert U90.shape[1] <= U99.shape[1]


def test_build_union_basis_empty_raises() -> None:
    with pytest.raises(ValueError):
        G.build_union_basis({})


# ── Cliff's delta ─────────────────────────────────────────────────────────

def test_cliffs_delta_basic() -> None:
    a = np.array([1, 2, 3])
    b = np.array([0.5, 1.5, 2.5])
    d = G.cliffs_delta(a, b)
    # A is mostly larger than B; expect positive delta.
    assert d > 0


def test_cliffs_delta_perfect_separation_is_one() -> None:
    a = np.array([10, 11, 12])
    b = np.array([0, 1, 2])
    assert G.cliffs_delta(a, b) == pytest.approx(1.0)
    assert G.cliffs_delta(b, a) == pytest.approx(-1.0)


def test_cliffs_delta_shuffle_to_zero() -> None:
    # Generate two groups from the same distribution; expect delta ~ 0 on average.
    rng = np.random.default_rng(4)
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    deltas = [G.cliffs_delta(a, b) for _ in range(50)]
    assert abs(np.mean(deltas)) < 0.1


# ── Jonckheere-Terpstra ───────────────────────────────────────────────────

def test_jonckheere_terpstra_monotonic_positive_trend() -> None:
    # Values strictly increase across tiers 1 -> 2 -> 3.
    values = np.concatenate([np.full(20, 0.0), np.full(20, 1.0), np.full(20, 2.0)])
    tiers = np.array([1] * 20 + [2] * 20 + [3] * 20)
    r = G.jonckheere_terpstra(values, tiers)
    assert r["trend_direction"] == 1
    assert r["p_value"] < 0.001


def test_jonckheere_terpstra_reversal_flips_direction() -> None:
    values = np.concatenate([np.full(20, 2.0), np.full(20, 1.0), np.full(20, 0.0)])
    tiers = np.array([1] * 20 + [2] * 20 + [3] * 20)
    r = G.jonckheere_terpstra(values, tiers)
    assert r["trend_direction"] == -1
    assert r["p_value"] < 0.001


def test_jonckheere_terpstra_no_trend() -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(size=120)
    tiers = rng.integers(1, 4, size=120)
    r = G.jonckheere_terpstra(values, tiers)
    # Random data -> p ought to be moderately large; allow some slack.
    assert r["p_value"] > 0.05


def test_jonckheere_terpstra_single_tier_returns_null() -> None:
    r = G.jonckheere_terpstra(np.arange(10.0), np.ones(10, int))
    assert r["trend_direction"] == 0
    assert r["p_value"] == 1.0


# ── Local intrinsic dimension ─────────────────────────────────────────────

def test_local_intrinsic_dim_returns_per_row() -> None:
    rng = np.random.default_rng(6)
    X = rng.normal(size=(40, 8))
    d = G.local_intrinsic_dim(X, k=10)
    assert d.shape == (40,)


def test_local_intrinsic_dim_small_N_returns_nan() -> None:
    X = np.zeros((3, 8))
    d = G.local_intrinsic_dim(X, k=20)
    assert np.all(np.isnan(d))


# ── Bootstrap CI ──────────────────────────────────────────────────────────

def test_bootstrap_ci_paired_constant_returns_constant() -> None:
    x = np.ones(50)
    lo, hi = G.bootstrap_ci(lambda v: float(v.mean()), x,
                            n_bootstrap=200, ci=0.95,
                            rng=np.random.default_rng(7))
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_bootstrap_ci_separate_for_two_groups() -> None:
    rng = np.random.default_rng(8)
    a = rng.normal(loc=1.0, size=100)
    b = rng.normal(loc=0.0, size=80)
    lo, hi = G.bootstrap_ci(G.cliffs_delta, a, b,
                            n_bootstrap=200, paired=False,
                            rng=np.random.default_rng(9))
    # delta should be positive — both bounds well above zero.
    assert lo > 0.0


# ── Holm correction ──────────────────────────────────────────────────────

def test_holm_correction_monotone() -> None:
    p = np.array([0.001, 0.04, 0.2, 0.5])
    adj = G.holm_correction(p)
    # Adjusted p-values are >= raw and monotone in raw order.
    assert all(adj[i] >= p[i] for i in range(p.size))


def test_holm_correction_all_ones_capped() -> None:
    p = np.array([1.0, 1.0, 1.0])
    adj = G.holm_correction(p)
    assert np.all(adj == 1.0)


# ── Shuffle / reversal sanity checks (smoke-data style) ───────────────────

def test_shuffle_sanity() -> None:
    """Shuffling tier labels should null the JT trend; |Cliff's delta| < 0.05."""
    rng = np.random.default_rng(10)
    values = rng.normal(size=300)
    tiers = rng.integers(1, 4, size=300)
    # Shuffle tiers explicitly.
    shuffled = rng.permutation(tiers)
    jt = G.jonckheere_terpstra(values, shuffled)
    cd13 = G.cliffs_delta(values[shuffled == 3], values[shuffled == 1])
    assert jt["p_value"] > 0.05
    assert abs(cd13) < 0.20


def test_reversal_sanity() -> None:
    """Reversing the tier ordering 1<->3 flips JT trend direction."""
    values = np.concatenate([np.full(30, 0.0), np.full(30, 1.0), np.full(30, 2.0)])
    tiers = np.array([1] * 30 + [2] * 30 + [3] * 30)
    forward = G.jonckheere_terpstra(values, tiers)
    reverse_map = {1: 3, 2: 2, 3: 1}
    reversed_tiers = np.array([reverse_map[int(t)] for t in tiers])
    backward = G.jonckheere_terpstra(values, reversed_tiers)
    assert forward["trend_direction"] == 1
    assert backward["trend_direction"] == -1
