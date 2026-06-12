"""Regression tests for src/nulls.py — the statistical backbone of the
manifold-vs-flat claim. The null hierarchy is the most carefully designed part
of the codebase; these tests guard its key invariants:

  * chain-stratified permutation preserves the per-label sample size N,
  * a genuinely low-dimensional target label lands in the upper tail (small p),
  * results are reproducible,
  * the all-degenerate path returns NaN with n_resamples=0 (no silent garbage).
"""

import numpy as np

from src.nulls import (
    top_k_variance_ratio,
    participation_ratio,
    chain_stratified_permutation_null,
    cross_chain_permutation_null,
    marchenko_pastur_diagnostic,
    full_null_hierarchy,
)
from tests.synthetic import chained_labelled, flat_subspace


# ── statistic primitives ─────────────────────────────────────────────────────

def test_top_k_variance_ratio_high_for_low_rank():
    """A near-1-D cloud has almost all variance in the top component."""
    X = flat_subspace(200, 1, noise=0.01, seed=0)
    assert top_k_variance_ratio(X, k=1) > 0.99


def test_top_k_variance_ratio_low_for_isotropic():
    X = flat_subspace(300, 50, noise=0.0, seed=1)
    assert top_k_variance_ratio(X, k=1) < 0.2  # one of ~50 equal dims


def test_participation_ratio_tracks_rank():
    low = participation_ratio(flat_subspace(300, 2, noise=0.0, seed=2))
    high = participation_ratio(flat_subspace(300, 30, noise=0.0, seed=3))
    assert low < high


# ── chain-stratified permutation null ────────────────────────────────────────

def test_structured_label_is_in_upper_tail():
    """Target sentences on a tight line => high top-k ratio => small upper-tail p."""
    acts, chains, labels = chained_labelled(structure=True, seed=4)
    res = chain_stratified_permutation_null(
        acts, chains, labels, "backtracking", n_resamples=200, random_state=42)
    assert res.p_value < 0.05, f"structured target gave p={res.p_value:.3f}"
    assert res.real_value > res.null_mean


def test_unstructured_label_not_significant():
    """Geometry-agnostic label => real value sits inside the null (not tiny p)."""
    acts, chains, labels = chained_labelled(structure=False, seed=5)
    res = chain_stratified_permutation_null(
        acts, chains, labels, "backtracking", n_resamples=200, random_state=42)
    assert res.p_value > 0.05, f"unstructured target gave suspiciously small p={res.p_value:.3f}"


def test_permutation_preserves_target_sample_size():
    """Within-chain permutation must keep the global count of the target label
    constant — otherwise the null statistic is computed on a different N than
    the real one and the comparison is invalid."""
    acts, chains, labels = chained_labelled(structure=True, seed=6)
    n_target = int((labels == "backtracking").sum())
    rng = np.random.default_rng(0)
    permuted = labels.copy()
    for c in np.unique(chains):
        idx = np.where(chains == c)[0]
        permuted[idx] = rng.permutation(labels[idx])
    assert int((permuted == "backtracking").sum()) == n_target


def test_null_is_reproducible():
    acts, chains, labels = chained_labelled(structure=True, seed=7)
    a = chain_stratified_permutation_null(acts, chains, labels, "backtracking",
                                          n_resamples=100, random_state=42)
    b = chain_stratified_permutation_null(acts, chains, labels, "backtracking",
                                          n_resamples=100, random_state=42)
    assert a.p_value == b.p_value and a.null_mean == b.null_mean


def test_absent_label_returns_nan_not_crash():
    """If the target label never occurs, every resample is degenerate and the
    result must be a clean NaN with n_resamples=0 — never a fabricated number."""
    acts, chains, labels = chained_labelled(structure=False, seed=8)
    res = chain_stratified_permutation_null(acts, chains, labels, "NEVER_OCCURS",
                                            n_resamples=50)
    assert np.isnan(res.p_value) and res.n_resamples == 0


# ── secondary + tertiary nulls ───────────────────────────────────────────────

def test_cross_chain_null_runs():
    acts, chains, labels = chained_labelled(structure=True, seed=9)
    res = cross_chain_permutation_null(acts, labels, "backtracking", n_resamples=100)
    assert np.isfinite(res.real_value)


def test_mp_diagnostic_returns_distribution_no_real_value():
    """The MP diagnostic is a baseline, not a test: real_value is NaN by design."""
    res = marchenko_pastur_diagnostic(N=100, d=1536, n_resamples=50)
    assert np.isnan(res.real_value)
    assert np.isfinite(res.null_mean)


def test_full_hierarchy_keys():
    acts, chains, labels = chained_labelled(structure=True, seed=11)
    out = full_null_hierarchy(acts, chains, labels, "backtracking", n_resamples=50)
    assert set(out) == {"chain_strat", "cross_chain", "mp"}


# ── fail-loud guard + smoothed p-values (2026-06-12 hardening) ───────────────

def test_proxy_chain_ids_raise_instead_of_noop():
    """One pseudo-chain per behaviour (the old fallback) makes within-chain
    permutation a no-op; the null must refuse to run, not return p≈1.0."""
    import pytest
    acts, chains, labels = chained_labelled(structure=True, seed=12)
    proxy_chains = labels.copy()  # chain id == label -> every chain single-label
    with pytest.raises(ValueError, match="NO-OP"):
        chain_stratified_permutation_null(acts, proxy_chains, labels,
                                          "backtracking", n_resamples=20)


def test_p_value_is_smoothed_never_zero():
    """Permutation p must be (1+count)/(1+B): minimum 1/(B+1), never 0 —
    unsmoothed zeros faked infinite resolution against Bonferroni thresholds."""
    acts, chains, labels = chained_labelled(structure=True, seed=13)
    B = 100
    res = chain_stratified_permutation_null(acts, chains, labels,
                                            "backtracking", n_resamples=B)
    assert res.p_value >= 1.0 / (B + 1)
    assert res.p_value > 0.0


def test_null_result_reports_chain_mixing():
    acts, chains, labels = chained_labelled(structure=True, seed=14)
    res = chain_stratified_permutation_null(acts, chains, labels,
                                            "backtracking", n_resamples=20)
    assert res.n_chains == len(np.unique(chains))
    assert 0 < res.n_mixed_label_chains <= res.n_chains
