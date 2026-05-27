"""Unit tests for src/cbs/comparison.py (M6)."""

from __future__ import annotations

import numpy as np
import pytest

from src.cbs import comparison as C


def test_comparison_module_importable() -> None:
    from src.cbs import comparison  # noqa: F401


def _record(layer: int, behaviour: str, statistic: str, label_scheme: str,
            effect_size: float, ci_lo: float = -0.1, ci_hi: float = 0.1) -> dict:
    return {
        "layer": layer, "behaviour": behaviour, "statistic": statistic,
        "label_scheme": label_scheme, "labels_source": "real",
        "effect_size": effect_size, "effect_size_ci95": [ci_lo, ci_hi],
    }


def test_cross_model_compare_pairs_matching_records() -> None:
    r1 = {"results": [
        _record(17, "adding-knowledge", "centroid_distance", "cbs_tier", 0.3, 0.1, 0.5),
        _record(17, "adding-knowledge", "centroid_distance", "cross_domain", 0.4),
    ]}
    base = {"results": [
        _record(17, "adding-knowledge", "centroid_distance", "cbs_tier", 0.0, -0.1, 0.1),
        _record(17, "adding-knowledge", "centroid_distance", "cross_domain", 0.0),
    ]}
    out = C.cross_model_compare(r1, base)
    assert out["n_compared"] == 2
    assert out["n_missing"] == 0
    deltas = [r["delta"] for r in out["records"] if "delta" in r]
    assert all(d > 0 for d in deltas)


def test_cross_model_compare_flags_missing() -> None:
    r1 = {"results": [_record(17, "X", "Y", "cbs_tier", 0.1)]}
    base = {"results": []}
    out = C.cross_model_compare(r1, base)
    assert out["n_missing"] == 1
    assert out["n_compared"] == 0


def test_cross_model_compare_skips_controls() -> None:
    """shuffle_control / reversal_control records must be filtered out."""
    r1 = {"results": [
        {"layer": 17, "behaviour": "X", "statistic": "Y", "label_scheme": "z",
         "labels_source": "shuffle_control", "effect_size": 1.0,
         "effect_size_ci95": [0.5, 1.5]},
    ]}
    base = {"results": [
        _record(17, "X", "Y", "z", 0.1),
    ]}
    out = C.cross_model_compare(r1, base)
    assert out["n_compared"] == 0
    assert out["n_missing"] == 1


def test_trajectory_wasserstein_runs() -> None:
    rng = np.random.default_rng(0)
    r1_s = rng.normal(loc=0.0, scale=1.0, size=(20, 3))
    r1_f = rng.normal(loc=2.0, scale=1.0, size=(20, 3))
    base_s = rng.normal(loc=0.0, scale=1.0, size=(20, 3))
    base_f = rng.normal(loc=2.0, scale=1.0, size=(20, 3))
    out = C.trajectory_wasserstein(r1_s, r1_f, base_s, base_f)
    assert out["w2_r1_success_vs_r1_failure"] > 0
    assert out["backend"] in {"pot", "sort1d"}


def test_cross_model_classifier_well_separated() -> None:
    rng = np.random.default_rng(1)
    d = 8
    pos = rng.normal(loc=+1.0, scale=0.4, size=(100, d))
    neg = rng.normal(loc=-1.0, scale=0.4, size=(100, d))
    Xtrain = np.concatenate([pos, neg])
    ytrain = np.concatenate([np.ones(100), np.zeros(100)])
    pos2 = rng.normal(loc=+1.0, scale=0.4, size=(50, d))
    neg2 = rng.normal(loc=-1.0, scale=0.4, size=(50, d))
    Xtest = np.concatenate([pos2, neg2])
    ytest = np.concatenate([np.ones(50), np.zeros(50)])
    acc = C.cross_model_classifier(Xtrain, ytrain, Xtest, ytest)
    assert acc > 0.85


def test_cross_model_compare_schema() -> None:
    """Quick schema check: comparator output is JSON-serialisable."""
    import json
    r1 = {"results": [_record(17, "X", "Y", "cbs_tier", 0.1)]}
    base = {"results": [_record(17, "X", "Y", "cbs_tier", 0.0)]}
    out = C.cross_model_compare(r1, base)
    json.dumps(out)
