"""Tests for the genuine two-sample cross-model bootstrap (AUDIT.md §5, #15).

09_cbs_geometry now persists `effect_size_boots` (the bootstrap distribution of
each effect size). cross_model_compare uses both models' distributions for a
real two-sample bootstrap, falling back to the normal-from-CI approximation only
when the arrays are absent.
"""

import numpy as np

from src.cbs.comparison import cross_model_compare

_KEY = dict(layer=27, behaviour="adding-knowledge",
            statistic="centroid_distance", label_scheme="cbs_tier")


def _rec(eff, ci, boots=None):
    r = {**_KEY, "effect_size": eff, "effect_size_ci95": ci}
    if boots is not None:
        r["effect_size_boots"] = list(boots)
    return r


def _compared(out):
    return [r for r in out["records"] if "delta" in r][0]


def test_bootstrap_used_and_separated_gives_small_p():
    r1 = {"results": [_rec(0.8, [0.7, 0.9], np.linspace(0.7, 0.9, 200))]}
    base = {"results": [_rec(0.1, [0.0, 0.2], np.linspace(0.0, 0.2, 200))]}
    rec = _compared(cross_model_compare(r1, base, seed=0))
    assert rec["p_method"] == "two_sample_bootstrap"
    assert rec["p_value"] < 0.05


def test_bootstrap_overlap_gives_large_p():
    r1 = {"results": [_rec(0.5, [0.3, 0.7], np.linspace(0.3, 0.7, 200))]}
    base = {"results": [_rec(0.5, [0.3, 0.7], np.linspace(0.3, 0.7, 200))]}
    rec = _compared(cross_model_compare(r1, base, seed=0))
    assert rec["p_method"] == "two_sample_bootstrap"
    assert rec["p_value"] > 0.2


def test_falls_back_to_normal_approx_without_boots():
    r1 = {"results": [_rec(0.8, [0.7, 0.9])]}      # no effect_size_boots
    base = {"results": [_rec(0.1, [0.0, 0.2])]}
    rec = _compared(cross_model_compare(r1, base, seed=0))
    assert rec["p_method"] == "normal_approx_from_ci"


def test_bootstrap_is_deterministic():
    r1 = {"results": [_rec(0.6, [0.4, 0.8], np.linspace(0.4, 0.8, 200))]}
    base = {"results": [_rec(0.3, [0.1, 0.5], np.linspace(0.1, 0.5, 200))]}
    p1 = _compared(cross_model_compare(r1, base, seed=7))["p_value"]
    p2 = _compared(cross_model_compare(r1, base, seed=7))["p_value"]
    assert p1 == p2
