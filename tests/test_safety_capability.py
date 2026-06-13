"""Tests for src/safety/capability.py — the H1 capability control (F3 / CF-11).

Planted geometry: when the safety axis is orthogonal to the capability axis the
control passes; when "safety" IS the capability axis it fails on both legs.
"""

from __future__ import annotations

import numpy as np

from src.safety.capability import (
    capability_control, capability_direction, difficulty_matched_indices,
)


def _unit(d, axis):
    u = np.zeros(d)
    u[axis] = 1.0
    return u


def _cluster(n=300, d=8, sep=5.0, axis=0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)) + sep * _unit(d, axis)


def test_capability_direction_recovers_axis():
    hard = _cluster(axis=1, seed=1)
    easy = _cluster(sep=0.0, seed=2)
    c = capability_direction(hard, easy)
    assert abs(c @ _unit(8, 1)) > 0.95


def test_difficulty_matching_nearest_each_used_once():
    pairs = difficulty_matched_indices([1.0, 2.0, 3.0], [2.1, 1.1, 3.05])
    # 1↔1.1 (j=1), 2↔2.1 (j=0), 3↔3.05 (j=2)
    assert dict(pairs) == {0: 1, 1: 0, 2: 2}


def test_difficulty_matching_tolerance_drops_far_pairs():
    pairs = difficulty_matched_indices([1.0, 10.0], [1.05, 1.1], tolerance=0.5)
    # only the close pair survives; 10.0 has no match within tolerance
    assert pairs == [(0, 0)] or pairs == [(0, 1)]


def test_control_passes_when_safety_orthogonal_to_capability():
    harmful = _cluster(axis=0, sep=5.0, seed=1)
    harmless = _cluster(axis=0, sep=0.0, seed=2)
    hard = _cluster(axis=1, sep=5.0, seed=3)     # capability on a DIFFERENT axis
    easy = _cluster(axis=1, sep=0.0, seed=4)
    res = capability_control(harmful, harmless, hard, easy)
    assert res["passed"] is True
    assert res["survives_partialling"] is True
    assert res["axis_distinct_from_capability"] is True
    assert res["retention"] > 0.5


def test_control_fails_when_safety_is_capability():
    # harmful == hard, harmless == easy, all on the same axis 0
    harmful = _cluster(axis=0, sep=5.0, seed=1)
    harmless = _cluster(axis=0, sep=0.0, seed=2)
    hard = _cluster(axis=0, sep=5.0, seed=5)
    easy = _cluster(axis=0, sep=0.0, seed=6)
    res = capability_control(harmful, harmless, hard, easy)
    assert res["passed"] is False
    # collinear axes and/or collapsed separation after partialling
    assert (res["axis_distinct_from_capability"] is False
            or res["survives_partialling"] is False)
