"""Tests for the new Phase-7 steering ARMS (synthetic vectors — no model).

Covers the adversarial-review fixes in src/steered_inference.py:
  * k-sweep arms at every built k (manifold_k1..manifold_auto), not just auto;
  * random-SUBSPACE-of-equal-k control (QR of a seeded Gaussian, R replicates);
  * energy-matched scale arithmetic;
  * orthogonal-complement (off-subspace) arm;
  * per-(behaviour,k) cos + retained-energy geometry bounds.

The model-dependent generation path is NOT exercised here; only the pure vector
construction + bookkeeping that _build_arms performs.
"""

import numpy as np
import pytest

from src.steered_inference import (
    _build_arms,
    manifold_method_label,
    random_subspace_projection,
    orthogonal_complement_vector,
    cosine,
    retained_energy,
    energy_matched_scale,
    random_direction_like,
)


def _fake_vecs(d=64, auto_k=12, seed=0):
    """A vecs dict shaped like src.steering.load_steering_vectors output."""
    rng = np.random.default_rng(seed)
    single = rng.standard_normal(d); single /= np.linalg.norm(single)
    man = {}
    for k in [1, 3, 5, 10, "auto"]:
        v = rng.standard_normal(d); v /= np.linalg.norm(v)
        man[k] = v
    return {"layer": 27, "single_direction": single,
            "manifold_projected": man, "auto_k": auto_k}


# ── k-sweep ───────────────────────────────────────────────────────────────────

def test_ksweep_builds_every_k_not_just_auto():
    arms, geom = _build_arms("backtracking", _fake_vecs(), 27)
    methods = {a["method"] for a in arms}
    for k in (1, 3, 5, 10):
        assert f"manifold_k{k}" in methods
    assert "manifold_auto" in methods
    # the OLD code only ran manifold_projected@auto — guard against regressing
    assert "manifold_projected" not in methods
    assert set(geom) == {1, 3, 5, 10, "auto"}


def test_manifold_label_helper():
    assert manifold_method_label(3) == "manifold_k3"
    assert manifold_method_label("auto") == "manifold_auto"


# ── geometry bounds (cos + retained energy) ───────────────────────────────────

def test_geometry_records_cos_and_retained_energy():
    arms, geom = _build_arms("backtracking", _fake_vecs(), 27)
    for k, g in geom.items():
        assert -1.0 <= g["cos_single_manifold"] <= 1.0
        assert 0.0 <= g["retained_energy"] <= 1.0 + 1e-9


def test_retained_energy_equals_abs_cos_for_unit_single():
    """When r_single is unit-norm, ‖P_k r‖/‖r‖ == |cos(r, p)| (p is unit)."""
    v = _fake_vecs()
    r = v["single_direction"]; p = v["manifold_projected"]["auto"]
    assert abs(retained_energy(r, p) - abs(cosine(r, p))) < 1e-9


def test_full_rank_subspace_recovers_high_retained_energy():
    """A manifold vector equal to the single direction retains all energy."""
    d = 32
    rng = np.random.default_rng(3)
    r = rng.standard_normal(d); r /= np.linalg.norm(r)
    assert abs(retained_energy(r, r) - 1.0) < 1e-9
    assert abs(cosine(r, r) - 1.0) < 1e-9


# ── random-subspace control ───────────────────────────────────────────────────

def test_random_subspace_is_unit_norm_and_seeded():
    rng = np.random.default_rng(5)
    r = rng.standard_normal(48); r /= np.linalg.norm(r)
    v1 = random_subspace_projection(r, 5, "seedkey|A")
    v2 = random_subspace_projection(r, 5, "seedkey|A")
    v3 = random_subspace_projection(r, 5, "seedkey|B")
    assert np.isclose(np.linalg.norm(v1), 1.0)
    assert np.allclose(v1, v2)               # same key -> identical
    assert not np.allclose(v1, v3)           # different key -> different subspace


def test_random_subspace_replicates_present_and_distinct():
    arms, _ = _build_arms("backtracking", _fake_vecs(), 27, n_random_subspaces=3)
    reps = [a for a in arms if a["method"] == "random_subspace_k5"]
    assert len(reps) == 3
    assert {a["subspace_replicate"] for a in reps} == {0, 1, 2}
    vs = [a["vector"] for a in reps]
    # replicates use different random subspaces -> not identical
    assert not np.allclose(vs[0], vs[1])
    assert not np.allclose(vs[1], vs[2])


def test_random_subspace_full_rank_recovers_direction():
    """Projecting onto a k=d random subspace == the (renormalised) original."""
    d = 16
    rng = np.random.default_rng(7)
    r = rng.standard_normal(d); r /= np.linalg.norm(r)
    v = random_subspace_projection(r, d, "k=d")
    assert abs(abs(float(v @ r)) - 1.0) < 1e-6      # spans whole space


def test_random_subspace_matches_manifold_renormalisation():
    """The random-subspace control must be renormalised IDENTICALLY to the
    manifold vector: both are a unit-norm projection of the single direction."""
    rng = np.random.default_rng(9)
    r = rng.standard_normal(40); r /= np.linalg.norm(r)
    v = random_subspace_projection(r, 4, "x")
    assert np.isclose(np.linalg.norm(v), 1.0)


# ── orthogonal-complement arm ─────────────────────────────────────────────────

def test_orthogonal_complement_is_orthogonal_to_subspace_vector():
    rng = np.random.default_rng(11)
    d = 50
    r = rng.standard_normal(d); r /= np.linalg.norm(r)
    p = rng.standard_normal(d); p /= np.linalg.norm(p)
    perp = orthogonal_complement_vector(r, p)
    assert np.isclose(np.linalg.norm(perp), 1.0)
    assert abs(float(perp @ p)) < 1e-6          # orthogonal to the in-subspace part


def test_orthogonal_complement_reconstructs_single():
    """r ≈ (r·p)p + ‖r_perp‖·perp  — the projection decomposition holds."""
    rng = np.random.default_rng(13)
    d = 24
    r = rng.standard_normal(d)
    p = rng.standard_normal(d); p /= np.linalg.norm(p)
    perp = orthogonal_complement_vector(r, p)
    along = float(r @ p) * p
    perp_norm = float(np.linalg.norm(r - along))
    recon = along + perp_norm * perp
    assert np.allclose(recon, r, atol=1e-6)


def test_orthogonal_complement_arm_present():
    arms, _ = _build_arms("backtracking", _fake_vecs(), 27)
    oc = [a for a in arms if a["method"] == "orthogonal_complement"]
    assert len(oc) == 1
    assert np.isclose(np.linalg.norm(oc[0]["vector"]), 1.0)
    assert "complement_of_k" in oc[0]


# ── energy-matched scale ──────────────────────────────────────────────────────

def test_energy_matched_scale_rescales_up():
    # behaviour delivers more projection energy than a random direction ->
    # the random arm must be scaled UP (>1) to match.
    assert energy_matched_scale(94.6, 4.9) == pytest.approx(94.6 / 4.9)
    assert energy_matched_scale(94.6, 4.9) > 1.0


def test_energy_matched_scale_degenerate_falls_back_to_one():
    assert energy_matched_scale(94.6, 0.0) == 1.0
    assert energy_matched_scale(94.6, None) == 1.0
    assert energy_matched_scale(None, 4.9) == 1.0


def test_energy_matched_arm_present_with_default_scale():
    arms, _ = _build_arms("backtracking", _fake_vecs(), 27)
    em = [a for a in arms if a["method"] == "energy_matched_random"]
    assert len(em) == 1
    # scale is a placeholder 1.0 until calibrated against the model in the runner
    assert em[0]["energy_scale"] == 1.0
    assert np.isclose(np.linalg.norm(em[0]["vector"]), 1.0)


# ── arm gating ────────────────────────────────────────────────────────────────

def test_control_arms_can_be_disabled():
    arms, _ = _build_arms("backtracking", _fake_vecs(), 27,
                          include_random_control=False,
                          include_random_subspace=False,
                          include_energy_matched=False,
                          include_orthogonal_complement=False)
    methods = {a["method"] for a in arms}
    # only single + the k-sweep manifold arms remain
    assert methods == {"single_direction", "manifold_k1", "manifold_k3",
                       "manifold_k5", "manifold_k10", "manifold_auto"}


def test_norm_matched_and_energy_matched_use_distinct_seeds():
    """The sanity floor and the energy floor must be DIFFERENT random vectors."""
    arms, _ = _build_arms("backtracking", _fake_vecs(), 27)
    rd = next(a["vector"] for a in arms if a["method"] == "random_direction")
    em = next(a["vector"] for a in arms if a["method"] == "energy_matched_random")
    assert not np.allclose(rd, em)
