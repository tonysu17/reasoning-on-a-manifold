"""Offline tests for PT-B2 (no model loading, no network, no spend)."""
from __future__ import annotations

import numpy as np
import pytest

import ptb2_concentration as P


def test_checkpoint_table_is_base_plus_six_arms():
    assert len(P.CHECKPOINTS) == 7
    assert P.CHECKPOINTS["base"].endswith("R1-1.5B")
    for recipe in ("safety", "control"):
        for seed in P.SEEDS:
            d = P.CHECKPOINTS[f"{recipe}-s{seed}"]
            assert f"lora-{recipe}1000" in d
            assert (seed == "42") == (not d.endswith(("-s43", "-s44")))
    assert (P.ROOT / P.BASE_DIR / "row_index.json").exists()


def test_estimator_matches_reference_on_random_clouds():
    from src.nulls import top_k_variance_ratio
    rng = np.random.default_rng(0)
    for n, d in ((300, 64), (50, 200), (1000, 128)):
        X = rng.normal(size=(n, d)).astype(np.float32)
        X[:, :3] *= 20.0                      # plant real concentration
        ref = top_k_variance_ratio(X, k=P.TOP_K)
        fast = P.top_k_ratio_fast(X)
        assert abs(ref - fast) < P.VALIDATION_TOL, (n, d, ref, fast)


def test_estimator_edge_cases():
    assert np.isnan(P.top_k_ratio_fast(np.zeros((1, 5), dtype=np.float32)))
    # A cloud with zero variance has no defined ratio.
    assert np.isnan(P.top_k_ratio_fast(np.ones((10, 5), dtype=np.float32)))
    # k >= d must saturate at 1.0.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4)).astype(np.float32)
    assert P.top_k_ratio_fast(X, k=10) == pytest.approx(1.0, abs=1e-6)


def test_chain_bootstrap_resamples_whole_chains():
    ids = np.array(["c1"] * 3 + ["c2"] * 2 + ["c3"] * 4)
    rng = np.random.default_rng(0)
    for _ in range(25):
        idx = P.chain_bootstrap_indices(ids, rng)
        picked = ids[idx]
        # every present chain appears with its FULL row count (or a multiple)
        for c, size in (("c1", 3), ("c2", 2), ("c3", 4)):
            n = int((picked == c).sum())
            assert n % size == 0
        assert len(idx) >= 1


def test_seed_permutation_floor_is_quarter():
    r = P.seed_permutation_p([0.01, 0.02, 0.03])
    assert r["p_two_sided"] == pytest.approx(0.25)
    assert r["floor"] == pytest.approx(0.25)
    # a mixed-sign set cannot be more extreme than the all-positive case
    r2 = P.seed_permutation_p([0.01, -0.02, 0.03])
    assert r2["p_two_sided"] >= 0.25


def test_holm_is_monotone_and_bounded():
    adj = P.holm({"a": 0.001, "b": 0.02, "c": 0.30, "d": 0.9})
    assert adj["a"] <= adj["b"] <= adj["c"] <= adj["d"]
    assert all(0.0 <= v <= 1.0 for v in adj.values())
    assert adj["a"] == pytest.approx(0.004)     # 4 * 0.001
    # step-down monotonicity: a later value never drops below an earlier one
    assert adj["d"] == pytest.approx(0.9)


def test_bca_interval_brackets_the_estimate():
    rng = np.random.default_rng(3)
    boot = rng.normal(0.5, 0.1, size=2000)
    jack = rng.normal(0.5, 0.01, size=100)
    lo, hi = P.bca_interval(0.5, boot, jack)
    assert lo < 0.5 < hi
    assert hi - lo < 1.0


def test_bca_degenerate_falls_back_to_percentile():
    boot = np.full(500, 0.3)
    jack = np.full(50, 0.3)
    lo, hi = P.bca_interval(0.3, boot, jack)
    assert lo == pytest.approx(0.3) and hi == pytest.approx(0.3)


def test_prereg_pins_the_binding_rules():
    text = P.PREREG.read_text()
    for needle in ("fixed-top-ten variance concentration", "within-session",
                   "descriptive only", "1e-6", "0.25", "37,851"):
        assert needle in text, needle


def test_layers_do_not_overlap_rq1_depths():
    rq1_depths = {11, 14, 17, 20, 27}
    assert not (set(P.LAYERS) & rq1_depths), (
        "PT-B2 layers must not silently look comparable to RQ1 cells")
