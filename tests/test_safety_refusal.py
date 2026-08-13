"""Known-answer tests for src/safety/refusal_direction.py — the S4 post-training
fingerprint engine. Synthetic activations with planted geometry (a known refusal
axis, controllable harmful/harmless separation) so a bug shows up as a numeric
disagreement rather than passing silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.safety.refusal_direction import (
    refusal_direction, project, directional_ablation,
    separation, recipe_fingerprint,
    recipe_direction_cosine, cross_recipe_cosines,
    category_refusal_subspace, recipe_principal_angles,
    linear_cka, effort_engagement,
)


def _unit(d: int, axis: int) -> np.ndarray:
    u = np.zeros(d)
    u[axis] = 1.0
    return u


def _clusters(n=300, d=8, sep=4.0, axis=0, seed=0, noise=1.0):
    """harmful centred at +sep·e_axis, harmless at the origin (noise σ=1)."""
    rng = np.random.default_rng(seed)
    u = _unit(d, axis)
    harmful = rng.standard_normal((n, d)) * noise + sep * u
    harmless = rng.standard_normal((n, d)) * noise
    return harmful, harmless


# ── Direction extraction ──────────────────────────────────────────────────────

def test_refusal_direction_recovers_planted_axis():
    harmful, harmless = _clusters(sep=5.0, axis=2, d=10, seed=1)
    r = refusal_direction(harmful, harmless)
    assert r.shape == (10,)
    assert np.isclose(np.linalg.norm(r), 1.0, atol=1e-6)
    # cosine with the true axis e_2 should be ~1
    assert abs(r @ _unit(10, 2)) > 0.97


def test_project_orders_harmful_above_harmless():
    harmful, harmless = _clusters(sep=4.0, seed=2)
    r = refusal_direction(harmful, harmless)
    assert project(harmful, r).mean() > project(harmless, r).mean()


# ── Separation fingerprint ────────────────────────────────────────────────────

def test_separation_scales_with_class_distance():
    big_h, big_l = _clusters(sep=4.0, seed=3)
    none_h, none_l = _clusters(sep=0.0, seed=3)
    big = separation(big_h, big_l)
    none = separation(none_h, none_l)
    assert big["cohens_d"] > 2.0
    assert big["auroc"] > 0.95
    # no real separation: weak fingerprint (self-fit inflation stays small)
    assert none["cohens_d"] < 0.5
    assert none["auroc"] < 0.65
    assert big["dom_norm"] > none["dom_norm"]


def test_directional_ablation_removes_separation():
    harmful, harmless = _clusters(sep=4.0, seed=4)
    r = refusal_direction(harmful, harmless)
    h_ab = directional_ablation(harmful, r)
    l_ab = directional_ablation(harmless, r)
    # after removing the refusal component, projection onto r is ~0 for both
    assert abs(project(h_ab, r).mean()) < 1e-6
    assert abs(project(l_ab, r).mean()) < 1e-6
    assert separation(h_ab, l_ab, direction=r)["cohens_d"] < 0.2


def test_recipe_fingerprint_orders_recipes():
    """H2 demonstration: deliberative-alignment (sharp) > RLHF (medium) >
    reasoning-distillation (~absent)."""
    per_recipe = {
        "deliberative": dict(zip(("harmful", "harmless"), _clusters(sep=5.0, seed=5))),
        "rlhf":         dict(zip(("harmful", "harmless"), _clusters(sep=2.0, seed=6))),
        "distill":      dict(zip(("harmful", "harmless"), _clusters(sep=0.0, seed=7))),
    }
    fp = recipe_fingerprint(per_recipe)
    assert fp["deliberative"]["cohens_d"] > fp["rlhf"]["cohens_d"] > fp["distill"]["cohens_d"]
    assert fp["deliberative"]["auroc"] > 0.95
    assert fp["distill"]["auroc"] < 0.65


# ── Cross-recipe: cosine + principal angles (same ambient dim) ────────────────

def test_recipe_direction_cosine_same_and_orthogonal():
    assert recipe_direction_cosine(_unit(8, 0), _unit(8, 0)) == pytest.approx(1.0)
    assert recipe_direction_cosine(_unit(8, 0), -_unit(8, 0)) == pytest.approx(1.0)  # |cos|
    assert recipe_direction_cosine(_unit(8, 0), _unit(8, 3)) == pytest.approx(0.0, abs=1e-9)


def test_cross_recipe_cosines_pairs():
    dirs = {"a": _unit(6, 0), "b": _unit(6, 0), "c": _unit(6, 1)}
    cos = cross_recipe_cosines(dirs)
    assert cos[("a", "b")] == pytest.approx(1.0)
    assert cos[("a", "c")] == pytest.approx(0.0, abs=1e-9)


def test_recipe_principal_angles_same_and_orthogonal():
    d = 6
    A = np.column_stack([_unit(d, 0), _unit(d, 1)])
    B_same = np.column_stack([_unit(d, 0), _unit(d, 1)])
    B_orth = np.column_stack([_unit(d, 2), _unit(d, 3)])
    same = recipe_principal_angles(A, B_same, top_k=2)
    orth = recipe_principal_angles(A, B_orth, top_k=2)
    assert np.max(same) < 1e-6
    assert np.min(orth) > np.pi / 2 - 1e-6


def test_category_refusal_subspace_is_orthonormal_and_spans_axes():
    d, n = 10, 200
    rng = np.random.default_rng(8)
    harmless = rng.standard_normal((n, d))
    harmful_by_cat = {
        f"cat{i}": rng.standard_normal((n, d)) + 4.0 * _unit(d, i)
        for i in range(3)
    }
    basis = category_refusal_subspace(harmful_by_cat, harmless)
    assert basis.shape == (d, 3)
    # orthonormal columns
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-6)
    # spans ~ span(e0, e1, e2): principal angles to that subspace ~ 0
    target = np.column_stack([_unit(d, 0), _unit(d, 1), _unit(d, 2)])
    angles = recipe_principal_angles(basis, target, top_k=3)
    assert np.max(angles) < 0.25  # radians (~14°), tolerant of cluster noise


# ── Cross-architecture: linear CKA (different ambient dim) ────────────────────

def test_linear_cka_high_for_linear_embedding_low_for_random():
    rng = np.random.default_rng(9)
    N, d1, d2 = 200, 8, 12
    X = rng.standard_normal((N, d1))
    # isometric embedding into a wider space (orthonormal rows) -> CKA ~ 1
    Q, _ = np.linalg.qr(rng.standard_normal((d2, d1)))   # (d2, d1) orthonormal cols
    A = Q.T                                              # (d1, d2) orthonormal rows
    Y = X @ A
    Z = rng.standard_normal((N, d2))                     # unrelated
    assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-9)
    assert linear_cka(X, Y) > 0.99
    assert linear_cka(X, Z) < 0.30


def test_linear_cka_requires_paired_rows():
    with pytest.raises(ValueError):
        linear_cka(np.zeros((5, 3)), np.zeros((6, 4)))


# ── gpt-oss reasoning-effort engagement ───────────────────────────────────────

def test_effort_engagement_increases_with_effort():
    d = 8
    r = _unit(d, 0)
    rng = np.random.default_rng(10)
    acts_by_effort = {
        "low":    rng.standard_normal((100, d)) + 0.5 * r,
        "medium": rng.standard_normal((100, d)) + 1.0 * r,
        "high":   rng.standard_normal((100, d)) + 2.0 * r,
    }
    eng = effort_engagement(acts_by_effort, r)
    assert eng["low"] < eng["medium"] < eng["high"]
