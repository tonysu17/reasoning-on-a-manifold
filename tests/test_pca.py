"""Known-answer + determinism regression tests for src/pca.py.

Determinism is the key one: the default sklearn solver picks a randomized SVD
for these (N<<d) shapes, which is non-reproducible without pinning. The code now
uses svd_solver="full"; these tests fail if that regresses.
"""

import numpy as np

from src.pca import analyse_behaviour
from tests.synthetic import flat_subspace


def test_pca_is_deterministic():
    """Two fits on the same matrix must give identical components/d_eff."""
    X = flat_subspace(145, 30, seed=3)  # N<d shape that triggers randomized SVD by default
    a = analyse_behaviour(X)
    b = analyse_behaviour(X)
    assert np.allclose(a["components"], b["components"])
    assert a["d_eff_90"] == b["d_eff_90"]
    assert a["participation_ratio"] == b["participation_ratio"]


def test_d_eff_recovers_flat_dimension():
    """A clean dim-D subspace explains ~100% variance in exactly D components,
    so d_eff at every threshold <=95% should equal D."""
    X = flat_subspace(300, 6, noise=0.0, seed=4)
    res = analyse_behaviour(X)
    assert res["d_eff_90"] == 6
    assert res["d_eff_95"] == 6


def test_participation_ratio_isotropic_is_high():
    """Near-isotropic data has participation ratio ~ number of components."""
    X = flat_subspace(300, 20, noise=0.0, seed=5)
    res = analyse_behaviour(X)
    # PR for ~isotropic 20-d data should be close to 20 (allow finite-sample slack).
    assert 12 <= res["participation_ratio"] <= 20


def test_insufficient_data_path():
    """N=1 must hit the graceful 'insufficient_data' branch, not crash."""
    res = analyse_behaviour(np.zeros((1, 1536)))
    assert res.get("error") == "insufficient_data"
    assert res["participation_ratio"] == 1.0


def test_d_eff_monotone_in_threshold():
    X = flat_subspace(200, 15, noise=0.5, seed=6)
    res = analyse_behaviour(X)
    assert (res["d_eff_50"] <= res["d_eff_70"] <= res["d_eff_80"]
            <= res["d_eff_90"] <= res["d_eff_95"])
