"""
src/predict/nulls_predict.py — Null hypotheses for the residual correctness signal.

Two nulls, both reusing the project's Phipson-Smyth smoothed permutation p-value
(src/nulls._smoothed_p) and the NullResult container, so results are directly
comparable to the rest of the pipeline:

  step_shuffle_null      — permute the ORDER of reasoning steps within each chain
                           and recompute the whole residual-AUC statistic. Tests
                           whether the signal needs the genuine temporal order or
                           survives on the static cloud of steps alone.

  label_permutation_null — permute the chain-level correctness labels (optionally
                           within difficulty/category strata) and recompute the
                           grouped AUC. Tests whether the residual features carry
                           correctness information beyond chance / strata
                           composition. NOTE: chain-level labels are one-per-chain,
                           so the *within-chain* label shuffle in src/nulls.py does
                           not apply here (it would be a no-op and trip its guard);
                           this is the correct across-chain analogue.

Milestone: Predictive Geometry — Rung 1.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from src.nulls import NullResult, _smoothed_p
from src.predict.evaluation import grouped_auc
from src.predict.trajectory_dataset import step_shuffle_within_chain

__all__ = ["step_shuffle_null", "label_permutation_null"]


def _summarise(null_stats: np.ndarray, real_value: float, tail: str,
               null_name: str, statistic_name: str) -> NullResult:
    valid = np.isfinite(null_stats)
    if not valid.any():
        return NullResult(
            null_name=null_name, statistic_name=statistic_name,
            real_value=float(real_value), null_mean=float("nan"),
            null_std=float("nan"), null_p2_5=float("nan"), null_p97_5=float("nan"),
            p_value=float("nan"), n_resamples=0, tail=tail,
        )
    ns = null_stats[valid]
    return NullResult(
        null_name=null_name, statistic_name=statistic_name,
        real_value=float(real_value),
        null_mean=float(ns.mean()), null_std=float(ns.std()),
        null_p2_5=float(np.percentile(ns, 2.5)),
        null_p97_5=float(np.percentile(ns, 97.5)),
        p_value=_smoothed_p(ns, real_value, tail),
        n_resamples=int(ns.size), tail=tail,
    )


def step_shuffle_null(
    datasets,
    statistic_fn: Callable[[list], float],
    *,
    n_resamples: int = 200,
    seed: int = 42,
    tail: str = "upper",
) -> NullResult:
    """Within-chain step-order permutation null.

    `statistic_fn(datasets) -> float` is the statistic to recompute on each
    shuffle (e.g. a partial of `residual_auc_statistic` binding the labels).
    `tail='upper'` because a real signal should give a HIGHER AUC than shuffled
    order.
    """
    rng = np.random.default_rng(seed)
    real = statistic_fn(list(datasets))
    null_stats = np.full(n_resamples, np.nan)
    for r in range(n_resamples):
        shuffled = step_shuffle_within_chain(datasets, rng)
        try:
            null_stats[r] = statistic_fn(shuffled)
        except Exception:  # noqa: BLE001 — degenerate resamples are expected
            pass
    return _summarise(null_stats, real, tail, "step_shuffle", "residual_auc")


def _permute_within_strata(
    labels: np.ndarray, strata: Optional[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """Permute labels globally, or independently within each stratum."""
    out = labels.copy()
    if strata is None:
        return rng.permutation(out)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        out[idx] = rng.permutation(out[idx])
    return out


def label_permutation_null(
    features,
    chain_ids,
    labels,
    *,
    strata=None,
    n_resamples: int = 200,
    seed: int = 42,
    n_splits: int = 5,
    eval_seed: int = 42,
    tail: str = "upper",
) -> NullResult:
    """Across-chain correctness-label permutation null on the grouped AUC.

    `features` are per-chain (row order == `chain_ids`); `labels` is the aligned
    {0,1,nan} array (see evaluation.align_labels). Rows with NaN labels are
    dropped first; `strata` (optional, same length) restricts the permutation to
    within-stratum shuffles so difficulty/category composition is held fixed.
    """
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    chain_ids = np.asarray(chain_ids, dtype=object)

    mask = ~np.isnan(labels)
    feats, labs, cids = features[mask], labels[mask], chain_ids[mask]
    strat = np.asarray(strata, dtype=object)[mask] if strata is not None else None

    def _auc(y: np.ndarray) -> float:
        return grouped_auc(feats, y, cids, seed=eval_seed, n_splits=n_splits)["auc_oof"]

    real = _auc(labs)
    rng = np.random.default_rng(seed)
    null_stats = np.full(n_resamples, np.nan)
    for r in range(n_resamples):
        try:
            null_stats[r] = _auc(_permute_within_strata(labs, strat, rng))
        except Exception:  # noqa: BLE001
            pass
    return _summarise(null_stats, real, tail, "label_perm", "auc_oof")
