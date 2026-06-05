"""Regression tests for the Group-A CBS-internal fixes (AUDIT.md §5):
build_union_basis eigenvalue weighting, JT tie handling, schema coercion,
config-sourced truncation cap, and the paired matched-pair CI.
"""

import numpy as np
import pytest

from src.cbs.geometry import build_union_basis, jonckheere_terpstra
from src.cbs.schemas import CBSResult
from src.cbs.cohort import is_truncated, DEFAULT_MAX_NEW_TOKENS


# ── build_union_basis eigenvalue weighting ───────────────────────────────────

def _orthonormal_block(d, k, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    return Q[:, :k]


def test_union_basis_weighted_is_orthonormal():
    pcs = {"a": _orthonormal_block(50, 3, 1), "b": _orthonormal_block(50, 3, 2)}
    weights = {"a": np.array([100.0, 10.0, 0.1]), "b": np.array([100.0, 5.0, 0.2])}
    B = build_union_basis(pcs, variance_threshold=0.95, per_behaviour_weights=weights)
    # columns orthonormal
    assert np.allclose(B.T @ B, np.eye(B.shape[1]), atol=1e-8)


def test_union_basis_weighting_changes_kept_dimension():
    """With skewed eigenvalues, weighting concentrates energy -> fewer dims kept
    than the unweighted (direction-energy) cut."""
    pcs = {"a": _orthonormal_block(50, 3, 1), "b": _orthonormal_block(50, 3, 2)}
    weights = {"a": np.array([100.0, 0.5, 0.1]), "b": np.array([100.0, 0.5, 0.1])}
    unweighted = build_union_basis(pcs, 0.95).shape[1]
    weighted = build_union_basis(pcs, 0.95, per_behaviour_weights=weights).shape[1]
    assert weighted <= unweighted


def test_union_basis_weight_length_mismatch_raises():
    pcs = {"a": _orthonormal_block(50, 3, 1)}
    with pytest.raises(ValueError):
        build_union_basis(pcs, per_behaviour_weights={"a": np.array([1.0, 2.0])})


# ── JT tie handling ──────────────────────────────────────────────────────────

def test_jt_with_ties_runs_and_keeps_direction():
    # Increasing trend across tiers, with deliberate ties.
    values = [1, 1, 2, 2, 2, 3, 3, 4, 4, 5]
    tiers = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    out = jonckheere_terpstra(values, tiers)
    assert np.isfinite(out["z"])
    assert out["trend_direction"] == 1  # values rise with tier despite ties


def test_jt_continuous_no_warning_still_correct():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(0, 1, 30), rng.normal(1, 1, 30), rng.normal(2, 1, 30)])
    tiers = np.array([1] * 30 + [2] * 30 + [3] * 30)
    out = jonckheere_terpstra(values, tiers)
    assert out["trend_direction"] == 1 and out["p_value"] < 0.05


# ── schema coercion ──────────────────────────────────────────────────────────

def test_cbsresult_coerces_string_tier():
    r = CBSResult(tier="3", knowledge_domain="algebra", cross_domain=True,
                  rationale="x", confidence="high")
    assert r.tier == 3 and isinstance(r.tier, int)


def test_cbsresult_coerces_cross_domain_yes():
    r = CBSResult(tier=2, knowledge_domain="geometry", cross_domain="yes",
                  rationale="x", confidence="low")
    assert r.cross_domain is True


def test_cbsresult_still_rejects_bad_tier():
    with pytest.raises(ValueError):
        CBSResult(tier=7, knowledge_domain="algebra", cross_domain=False,
                  rationale="x", confidence="high")


# ── cohort cap sourced from config ───────────────────────────────────────────

def test_truncation_cap_matches_config():
    # config.yaml chains.max_new_tokens is 8192; loader must pick it up.
    from src.config import load_config
    assert DEFAULT_MAX_NEW_TOKENS == int(load_config()["chains"]["max_new_tokens"])


def test_is_truncated_uses_cap():
    capped = {"n_tokens": DEFAULT_MAX_NEW_TOKENS, "chain": "thinking..."}
    closed = {"n_tokens": DEFAULT_MAX_NEW_TOKENS, "chain": "done</think>"}
    assert is_truncated(capped) is True
    assert is_truncated(closed) is False


# ── 09_cbs_geometry real-label loader (replaces the synthetic-mislabel stub) ──

def test_load_real_labels_maps_tiers(tmp_path):
    """_load_real_labels must align real cbs_tier/cross_domain to the Phase-4
    row order (build_row_index), so geometry on real annotations is no longer
    silently synthetic-but-labelled-'real' (AUDIT.md §5)."""
    import json
    import importlib
    mod = importlib.import_module("09_cbs_geometry")

    chains = [
        {"task_id": "c0", "annotations": [
            {"label": "deduction"},  # i=0, not a target behaviour
            {"label": "adding-knowledge", "cbs_tier": 3, "cbs_cross_domain": True},   # i=1
            {"label": "adding-knowledge", "cbs_tier": 1, "cbs_cross_domain": False},  # i=2
        ]},
        {"task_id": "c1", "annotations": [
            {"label": "adding-knowledge", "cbs_tier": 2, "cbs_cross_domain": False},  # i=0
        ]},
    ]
    f = tmp_path / "cbs.json"
    f.write_text(json.dumps(chains))

    out = mod._load_real_labels(f, ["adding-knowledge"])
    tiers, cds = out["adding-knowledge"]
    # row order: (c0,1),(c0,2),(c1,0)
    assert list(tiers) == [3, 1, 2]
    assert list(cds) == [True, False, False]

