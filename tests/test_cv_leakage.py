"""Regression test for the chain-leakage fix in src/cbs/matching.cv_probe
(and the M4/M5 probes that consume it).

Construction: each chain is a tight cluster around its own random centroid, and
labels are assigned PER CHAIN but INDEPENDENTLY of the centroid. There is no
real label signal — accuracy should be ~chance. An ungrouped CV nonetheless
scores high because sentences from a chain appear in both train and test, so the
probe memorises chain centroids (each of which carries one label). A chain-aware
split holds whole chains out and correctly collapses to chance. This is exactly
the leakage that was inflating the verification-gradient / fail-stop accuracy.
"""

import numpy as np

from src.cbs.matching import cv_probe, verification_gradient


def _chain_clustered_no_signal(n_chains=24, per_chain=15, d=64, seed=0):
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for c in range(n_chains):
        centroid = rng.standard_normal(d) * 5.0     # well-separated per-chain clusters
        label = c % 2                               # label is arbitrary w.r.t. centroid
        for _ in range(per_chain):
            X.append(centroid + rng.standard_normal(d) * 0.1)
            y.append(label)
            groups.append(f"chain{c}")
    return np.array(X), np.array(y), np.array(groups)


def test_ungrouped_cv_leaks_and_overstates_accuracy():
    X, y, _ = _chain_clustered_no_signal(seed=1)
    accs, _ = cv_probe(X, y, groups=None, cv_folds=5, seed=0)
    assert np.mean(accs) > 0.85, "expected ungrouped CV to leak (high accuracy)"


def test_grouped_cv_collapses_to_chance():
    X, y, groups = _chain_clustered_no_signal(seed=1)
    accs, _ = cv_probe(X, y, groups=groups, cv_folds=5, seed=0)
    # No real label signal => chain-aware CV must be near chance.
    assert np.mean(accs) < 0.70, f"grouped CV should be ~chance, got {np.mean(accs):.2f}"


def test_grouped_is_lower_than_ungrouped():
    X, y, groups = _chain_clustered_no_signal(seed=2)
    ungrouped = np.mean(cv_probe(X, y, None, cv_folds=5, seed=0)[0])
    grouped = np.mean(cv_probe(X, y, groups, cv_folds=5, seed=0)[0])
    assert grouped < ungrouped


def test_verification_gradient_accepts_groups_and_reports_flag():
    X, y, groups = _chain_clustered_no_signal(seed=3)
    correct = X[y == 1]
    incorrect = X[y == 0]
    cg = groups[y == 1]
    ig = groups[y == 0]
    out = verification_gradient(correct, incorrect,
                                correct_groups=cg, incorrect_groups=ig)
    assert out["grouped"] is True
    assert out["cv_accuracy_mean"] < 0.70   # leak-free => chance on no-signal data


def test_verification_gradient_without_groups_still_runs():
    """Backward-compat: omitting groups must not crash (just warns + may leak)."""
    X, y, _ = _chain_clustered_no_signal(seed=4)
    out = verification_gradient(X[y == 1], X[y == 0])
    assert "cv_accuracy_mean" in out and out["grouped"] is False
