"""Tests for src/safety/fingerprint.py — de-confounded scoring (F2) + length
controls (F6). Planted geometry so held-out d, the permutation null, and the
bootstrap CI behave as designed; deterministic helpers checked for stability.
"""

from __future__ import annotations

import numpy as np

from src.safety.fingerprint import (
    bootstrap_separation_ci, layer_at_fraction, length_normalised_engagement,
    separation_heldout, separation_permutation_null, subsample_to_min,
)


def _unit(d, axis):
    u = np.zeros(d)
    u[axis] = 1.0
    return u


def _clusters(n=100, d=8, sep=5.0, seed=0):
    rng = np.random.default_rng(seed)
    harmful = rng.standard_normal((n, d)) + sep * _unit(d, 0)
    harmless = rng.standard_normal((n, d))
    return harmful, harmless


# ── F2 held-out grouped scoring ───────────────────────────────────────────────

def test_heldout_d_large_when_separated():
    h, l = _clusters(sep=5.0, seed=1)
    gh = list(range(len(h)))
    gl = list(range(len(h), 2 * len(h)))
    res = separation_heldout(h, l, gh, gl, n_folds=5, seed=0)
    assert res["n_folds_used"] == 5
    assert res["cohens_d_mean"] > 1.0
    assert res["auroc_mean"] > 0.8


def test_heldout_d_near_zero_when_random():
    h, l = _clusters(sep=0.0, seed=2)
    gh = list(range(len(h)))
    gl = list(range(len(h), 2 * len(h)))
    res = separation_heldout(h, l, gh, gl, n_folds=5, seed=0)
    assert abs(res["cohens_d_mean"]) < 0.8  # no real signal survives the held-out split


# ── F2 permutation null ───────────────────────────────────────────────────────

def test_permutation_null_significant_when_separated():
    h, l = _clusters(sep=5.0, seed=3)
    res = separation_permutation_null(h, l, n_perm=200, seed=0)
    assert res["observed_d"] > res["null_p95"]
    assert res["p_value"] < 0.05


def test_permutation_null_nonsignificant_when_random():
    h, l = _clusters(sep=0.0, seed=4)
    res = separation_permutation_null(h, l, n_perm=200, seed=0)
    assert res["p_value"] > 0.05  # in-sample d is just diff-of-means bias


# ── F2 bootstrap CI + pre-registered layer ────────────────────────────────────

def test_bootstrap_ci_brackets_and_positive_when_separated():
    h, l = _clusters(sep=5.0, seed=5)
    res = bootstrap_separation_ci(h, l, n_boot=200, seed=0)
    assert res["ci_low"] < res["cohens_d_mean"] < res["ci_high"]
    assert res["ci_low"] > 0


def test_layer_at_fraction():
    assert layer_at_fraction(list(range(24)), 0.75) == 17
    assert layer_at_fraction(list(range(24)), 0.0) == 0
    assert layer_at_fraction(list(range(24)), 1.0) == 23


# ── F6 length controls ────────────────────────────────────────────────────────

def test_subsample_to_min_equalises_and_is_deterministic():
    rng = np.random.default_rng(0)
    by_key = {"low": rng.standard_normal((10, 4)),
              "med": rng.standard_normal((4, 4)),
              "high": rng.standard_normal((7, 4))}
    out = subsample_to_min(by_key, seed=0)
    assert all(v.shape[0] == 4 for v in out.values())
    out2 = subsample_to_min(by_key, seed=0)
    assert all(np.array_equal(out[k], out2[k]) for k in out)


def test_length_normalised_engagement_per_token():
    eng = length_normalised_engagement({"low": 10.0, "high": 30.0},
                                       {"low": 100, "high": 100})
    assert abs(eng["low"] - 0.1) < 1e-9
    assert abs(eng["high"] - 0.3) < 1e-9
    eng_b = length_normalised_engagement({"low": 10.0}, {"low": 100}, baseline=0.05)
    assert abs(eng_b["low"] - 0.05) < 1e-9
