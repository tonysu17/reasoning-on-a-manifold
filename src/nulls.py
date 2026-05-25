"""
Null-hypothesis machinery for per-behaviour structural claims.

Companion document Section 2.5 (revised) commits to a hierarchy:

  Primary:    chain-stratified permutation null
              (within-chain label shuffles; controls for chain identity,
               within-chain drift, ambient covariance, sample sizes)

  Secondary:  cross-chain permutation null
              (global label shuffles; complementary diagnostic for whether
               structure is driven by behaviour-level or chain-level effects)

  Tertiary:   Marchenko-Pastur diagnostic
              (isotropic Gaussian null; quantifies finite-sample inflation
               in top-k variance ratios at d, N, NOT a structural test)

Each function returns Monte Carlo null distributions for a statistic, plus
a one-sided p-value of the real data against the null.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


# Result container

@dataclass
class NullResult:
    """One null test's outcome."""
    null_name: str               # 'chain_strat_perm', 'cross_chain_perm', 'mp_isotropic'
    statistic_name: str          # what we computed (e.g. 'top10_var_ratio')
    real_value: float
    null_mean: float
    null_std: float
    null_p2_5: float             # 2.5th percentile of null
    null_p97_5: float            # 97.5th percentile of null
    p_value: float               # one-sided P(null >= real) for upper-tail tests
    n_resamples: int
    tail: str                    # 'upper' (real > null) or 'lower' (real < null)


# Statistic computers

def top_k_variance_ratio(activations: np.ndarray, k: int = 10) -> float:
    """Cumulative explained variance in the top k principal components."""
    if activations.shape[0] < 2:
        return float("nan")
    pca = PCA(n_components=min(k, activations.shape[0] - 1, activations.shape[1]))
    pca.fit(activations)
    return float(pca.explained_variance_ratio_.cumsum()[-1])


def participation_ratio(activations: np.ndarray) -> float:
    """PR = (sum(eig))^2 / sum(eig^2). Sample-size-robust effective dimension."""
    if activations.shape[0] < 2:
        return float("nan")
    pca = PCA(n_components=min(activations.shape[0] - 1, activations.shape[1]))
    pca.fit(activations)
    e = pca.explained_variance_
    if (e ** 2).sum() == 0:
        return float("nan")
    return float((e.sum() ** 2) / (e ** 2).sum())


# Primary null: chain-stratified permutation

def chain_stratified_permutation_null(
    activations:    np.ndarray,
    chain_ids:      np.ndarray,
    labels:         np.ndarray,
    target_label,
    statistic_fn:   Callable[[np.ndarray], float] = top_k_variance_ratio,
    statistic_name: str = "top10_var_ratio",
    n_resamples:    int = 500,
    random_state:   int = 42,
    tail:           str = "upper",
) -> NullResult:
    """For each resample, permute labels WITHIN each chain (preserving the
    label distribution per chain), then recompute the statistic on the
    sentences whose permuted label == target_label.

    Args:
        activations:   (N_sentences, hidden_dim)
        chain_ids:     (N_sentences,) which chain each sentence is from
        labels:        (N_sentences,) per-sentence behaviour label
        target_label:  the label whose conditional structure we are testing
        statistic_fn:  function taking an activation matrix and returning a scalar
        tail:          'upper' if larger statistic => more structure (default for variance ratio)
    """
    rng = np.random.default_rng(random_state)

    # Real statistic on the actually-target-labelled sentences
    real_mask = (labels == target_label)
    real_value = statistic_fn(activations[real_mask])

    # Build a per-chain index map for fast within-chain permutation
    unique_chains = np.unique(chain_ids)
    chain_to_idx = {c: np.where(chain_ids == c)[0] for c in unique_chains}

    null_stats = np.empty(n_resamples)
    null_stats[:] = np.nan
    for r in range(n_resamples):
        permuted_labels = labels.copy()
        for c, idxs in chain_to_idx.items():
            permuted_labels[idxs] = rng.permutation(labels[idxs])
        mask = (permuted_labels == target_label)
        try:
            null_stats[r] = statistic_fn(activations[mask])
        except (np.linalg.LinAlgError, ValueError):
            pass  # expected for degenerate resamples
        except Exception as e:
            if r == 0:  # log once per loop
                logger.warning(f"  unexpected error in resample: {type(e).__name__}: {e}")

    valid = np.isfinite(null_stats)
    if not valid.any():
        logger.warning(f"chain_strat_perm: all {n_resamples} resamples produced non-finite statistics; check for module-import errors or N-too-small")
        return NullResult(null_name="chain_strat_perm", statistic_name=statistic_name,
                          real_value=real_value, null_mean=float("nan"),
                          null_std=float("nan"), null_p2_5=float("nan"),
                          null_p97_5=float("nan"), p_value=float("nan"),
                          n_resamples=0, tail=tail)
    null_stats = null_stats[valid]
    if tail == "upper":
        p_val = float((null_stats >= real_value).sum() / null_stats.size)
    else:
        p_val = float((null_stats <= real_value).sum() / null_stats.size)
    return NullResult(
        null_name="chain_strat_perm",
        statistic_name=statistic_name,
        real_value=float(real_value),
        null_mean=float(null_stats.mean()),
        null_std=float(null_stats.std()),
        null_p2_5=float(np.percentile(null_stats, 2.5)),
        null_p97_5=float(np.percentile(null_stats, 97.5)),
        p_value=p_val,
        n_resamples=int(null_stats.size),
        tail=tail,
    )


# Secondary null: cross-chain (global) permutation

def cross_chain_permutation_null(
    activations:    np.ndarray,
    labels:         np.ndarray,
    target_label,
    statistic_fn:   Callable[[np.ndarray], float] = top_k_variance_ratio,
    statistic_name: str = "top10_var_ratio",
    n_resamples:    int = 500,
    random_state:   int = 42,
    tail:           str = "upper",
) -> NullResult:
    """Permute labels globally (ignoring chain_id), then recompute the
    statistic on sentences whose permuted label == target_label.

    Excess of real-data over this null *but not over chain_strat* indicates
    structure driven by cross-chain (category-level) effects rather than
    behaviour-level."""
    rng = np.random.default_rng(random_state)

    real_mask = (labels == target_label)
    real_value = statistic_fn(activations[real_mask])

    null_stats = np.empty(n_resamples)
    null_stats[:] = np.nan
    for r in range(n_resamples):
        permuted_labels = rng.permutation(labels)
        mask = (permuted_labels == target_label)
        try:
            null_stats[r] = statistic_fn(activations[mask])
        except (np.linalg.LinAlgError, ValueError):
            pass  # expected for degenerate resamples
        except Exception as e:
            if r == 0:  # log once per loop
                logger.warning(f"  unexpected error in resample: {type(e).__name__}: {e}")

    valid = np.isfinite(null_stats)
    if not valid.any():
        logger.warning(f"cross_chain_perm: all {n_resamples} resamples produced non-finite statistics")
        return NullResult(null_name="cross_chain_perm", statistic_name=statistic_name,
                          real_value=real_value, null_mean=float("nan"),
                          null_std=float("nan"), null_p2_5=float("nan"),
                          null_p97_5=float("nan"), p_value=float("nan"),
                          n_resamples=0, tail=tail)
    null_stats = null_stats[valid]
    if tail == "upper":
        p_val = float((null_stats >= real_value).sum() / null_stats.size)
    else:
        p_val = float((null_stats <= real_value).sum() / null_stats.size)
    return NullResult(
        null_name="cross_chain_perm",
        statistic_name=statistic_name,
        real_value=float(real_value),
        null_mean=float(null_stats.mean()),
        null_std=float(null_stats.std()),
        null_p2_5=float(np.percentile(null_stats, 2.5)),
        null_p97_5=float(np.percentile(null_stats, 97.5)),
        p_value=p_val,
        n_resamples=int(null_stats.size),
        tail=tail,
    )


# Tertiary null: Marchenko-Pastur isotropic Gaussian

def marchenko_pastur_diagnostic(
    N:              int,
    d:              int,
    statistic_fn:   Callable[[np.ndarray], float] = top_k_variance_ratio,
    statistic_name: str = "top10_var_ratio",
    sigma:          float = 1.0,
    n_resamples:    int = 500,
    random_state:   int = 42,
) -> NullResult:
    """Monte Carlo null: draw matched-(N, d) samples from N(0, sigma^2 I)
    and compute the statistic distribution.

    This is reported alongside the real statistic as a *finite-sample
    inflation diagnostic*. Excess over MP null does NOT imply per-behaviour
    structure (activation covariance is far from isotropic); it only
    quantifies the inflation in top-k variance ratios attributable to
    sampling alone.
    """
    rng = np.random.default_rng(random_state)
    null_stats = np.empty(n_resamples)
    null_stats[:] = np.nan
    for r in range(n_resamples):
        X = sigma * rng.standard_normal((N, d))
        try:
            null_stats[r] = statistic_fn(X)
        except (np.linalg.LinAlgError, ValueError):
            pass
        except Exception as e:
            if r == 0:
                logger.warning(f"  unexpected error in mp_isotropic resample: {type(e).__name__}: {e}")
    valid = np.isfinite(null_stats)
    if not valid.any():
        logger.warning(f"mp_isotropic: all {n_resamples} resamples produced non-finite statistics")
        return NullResult(null_name="mp_isotropic", statistic_name=statistic_name,
                          real_value=float("nan"), null_mean=float("nan"),
                          null_std=float("nan"), null_p2_5=float("nan"),
                          null_p97_5=float("nan"), p_value=float("nan"),
                          n_resamples=0, tail="upper")
    null_stats = null_stats[valid]
    return NullResult(
        null_name="mp_isotropic",
        statistic_name=statistic_name,
        real_value=float("nan"),  # No "real value" for this diagnostic; it's only a baseline
        null_mean=float(null_stats.mean()),
        null_std=float(null_stats.std()),
        null_p2_5=float(np.percentile(null_stats, 2.5)),
        null_p97_5=float(np.percentile(null_stats, 97.5)),
        p_value=float("nan"),
        n_resamples=int(null_stats.size),
        tail="upper",
    )


# Convenience: run the full hierarchy for one behaviour

def full_null_hierarchy(
    activations:    np.ndarray,
    chain_ids:      np.ndarray,
    labels:         np.ndarray,
    target_label,
    statistic_fn:   Callable[[np.ndarray], float] = top_k_variance_ratio,
    statistic_name: str = "top10_var_ratio",
    n_resamples:    int = 500,
    random_state:   int = 42,
    tail:           str = "upper",
) -> dict:
    """Run all three nulls for one (statistic, target_label) and return a
    dict with keys {'chain_strat', 'cross_chain', 'mp'}."""
    return {
        "chain_strat": chain_stratified_permutation_null(
            activations, chain_ids, labels, target_label,
            statistic_fn=statistic_fn, statistic_name=statistic_name,
            n_resamples=n_resamples, random_state=random_state, tail=tail,
        ),
        "cross_chain": cross_chain_permutation_null(
            activations, labels, target_label,
            statistic_fn=statistic_fn, statistic_name=statistic_name,
            n_resamples=n_resamples, random_state=random_state, tail=tail,
        ),
        "mp": marchenko_pastur_diagnostic(
            N=int((labels == target_label).sum()),
            d=activations.shape[1],
            statistic_fn=statistic_fn, statistic_name=statistic_name,
            n_resamples=n_resamples, random_state=random_state,
        ),
    }
