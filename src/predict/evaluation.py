"""
src/predict/evaluation.py — Chain-grouped correctness evaluation + baselines.

Purpose
-------
The atom-1 test: does the predictor's RESIDUAL geometry predict chain-level
reasoning correctness, under chain-grouped CV, better than the baselines that
prior work already used? Baselines provided here:
  * raw_curvature_features  — Rung-0 / SSP-style trajectory smoothness on raw x
  * raw_step_features       — persistence residual ||x_{t+1} - x_t|| (no model)
  * length_features         — chain length + step gap + truncation (CF-8 control)
(A token-NLL baseline needs logits and is deferred to a GPU re-extraction.)

Every probe is chain-grouped: each chain is one row and its own group, so
GroupKFold simply holds out whole chains — there is no within-chain leakage to
inflate the AUC (the CF-2 discipline, same failure mode guarded in
tests/test_cv_leakage.py).

Milestone: Predictive Geometry — Rung 1.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.predict.predictor import RidgeConfig, chain_residual_features, oof_residuals
from src.predict.trajectory_dataset import make_supervised_pairs

__all__ = [
    "grouped_auc",
    "align_labels",
    "raw_curvature_features",
    "raw_step_features",
    "length_features",
    "residual_auc_statistic",
]

_STABILITY_TOL = 0.15  # per-fold AUC std above which we flag the probe unstable


def align_labels(chain_ids, label_map: dict | None) -> np.ndarray:
    """Map chain ids -> {0.0, 1.0, nan}, aligned to `chain_ids` row order.

    `label_map[cid]` may be a bool or a {'correct': ...} record. Missing or None
    labels become NaN so downstream code can mask them out.
    """
    label_map = label_map or {}
    out = np.full(len(chain_ids), np.nan, dtype=float)
    for i, cid in enumerate(chain_ids):
        v = label_map.get(cid)
        if isinstance(v, dict):
            v = v.get("correct")
        if v is not None:
            out[i] = 1.0 if bool(v) else 0.0
    return out


def grouped_auc(
    features,
    labels,
    groups,
    *,
    seed: int = 42,
    n_splits: int = 5,
    standardize: bool = True,
) -> dict:
    """Chain-grouped logistic-probe ROC-AUC for `features` predicting `labels`.

    Rows with NaN labels are dropped. Returns a dict with the out-of-fold AUC
    (`auc_oof`, computed on pooled held-out predictions), the mean/std of
    per-fold AUCs, a `stable` flag (fold std <= 0.15), and counts. Returns NaNs
    gracefully when there is too little signal (one class, < 2 groups, < 6 rows).
    """
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    groups = np.asarray(groups, dtype=object)

    out = {
        "auc_oof": float("nan"),
        "auc_fold_mean": float("nan"),
        "auc_fold_std": float("nan"),
        "stable": False,
        "n": 0,
        "n_pos": 0,
        "n_splits": 0,
    }

    mask = ~np.isnan(labels)
    X, y, g = features[mask], labels[mask].astype(int), groups[mask]
    out["n"] = int(X.shape[0])
    out["n_pos"] = int(y.sum())
    if X.shape[0] < 6 or np.unique(y).size < 2:
        return out
    uniq_groups = np.unique(g)
    k = min(n_splits, uniq_groups.size)
    if k < 2:
        return out

    oof = np.full(y.shape[0], np.nan)
    fold_aucs: list[float] = []
    gkf = GroupKFold(n_splits=k)
    for tr, te in gkf.split(X, y, groups=g):
        if np.unique(y[tr]).size < 2:
            continue
        if standardize:
            scaler = StandardScaler().fit(X[tr])
            Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        else:
            Xtr, Xte = X[tr], X[te]
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        oof[te] = p
        if np.unique(y[te]).size == 2:
            fold_aucs.append(float(roc_auc_score(y[te], p)))

    valid = ~np.isnan(oof)
    if valid.sum() > 0 and np.unique(y[valid]).size == 2:
        out["auc_oof"] = float(roc_auc_score(y[valid], oof[valid]))
    if fold_aucs:
        out["auc_fold_mean"] = float(np.mean(fold_aucs))
        out["auc_fold_std"] = float(np.std(fold_aucs))
        out["stable"] = bool(out["auc_fold_std"] <= _STABILITY_TOL)
    out["n_splits"] = k
    return out


# ── Baseline feature extractors (each returns (features, chain_ids, names)) ──

def raw_curvature_features(datasets):
    """Rung-0 / SSP-style smoothness on the RAW trajectory (no learned model).

    Uses the project's arc-length-reparameterised Frenet curvature
    (src/cbs/trajectory.curvature_sequence) so this baseline is exactly the
    predictor-free quantity prior work measured.
    """
    from src.cbs.schemas import ChainTrajectory
    from src.cbs.trajectory import curvature_sequence

    names = ["curv_mean", "curv_max", "arc_len", "arc_per_step"]
    chain_ids: list[object] = []
    rows: list[list[float]] = []
    for ds in datasets:
        traj = ChainTrajectory(chain_id=ds.chain_id, layer=0, X=ds.X)
        kappa = curvature_sequence(traj)
        finite = kappa[np.isfinite(kappa)]
        diffs = (
            np.linalg.norm(np.diff(ds.X, axis=0), axis=1)
            if ds.T > 1
            else np.zeros(0)
        )
        arc = float(diffs.sum())
        rows.append(
            [
                float(finite.mean()) if finite.size else 0.0,
                float(finite.max()) if finite.size else 0.0,
                arc,
                arc / max(1, ds.T - 1),
            ]
        )
        chain_ids.append(ds.chain_id)
    return np.asarray(rows, dtype=float).reshape(-1, len(names)), np.asarray(
        chain_ids, dtype=object
    ), names


def raw_step_features(datasets):
    """Persistence-baseline residual: ||x_{t+1} - x_t|| stats (no learned model).

    This is the residual a trivial predictor f(x_t)=x_t would leave; a learned
    predictor that cannot beat this contributes nothing.
    """
    names = ["step_mean", "step_std", "step_max"]
    chain_ids: list[object] = []
    rows: list[list[float]] = []
    for ds in datasets:
        if ds.T > 1:
            steps = np.linalg.norm(np.diff(ds.X, axis=0), axis=1)
            rows.append([float(steps.mean()), float(steps.std()), float(steps.max())])
        else:
            rows.append([0.0, 0.0, 0.0])
        chain_ids.append(ds.chain_id)
    return np.asarray(rows, dtype=float).reshape(-1, len(names)), np.asarray(
        chain_ids, dtype=object
    ), names


def length_features(datasets):
    """Length / gap / truncation control (CF-8): the contrast must beat these."""
    names = ["n_steps", "mean_gap", "truncated"]
    chain_ids: list[object] = []
    rows: list[list[float]] = []
    for ds in datasets:
        gaps = np.diff(ds.orig_indices) if ds.T > 1 else np.zeros(0)
        rows.append(
            [
                float(ds.T),
                float(gaps.mean()) if gaps.size else 0.0,
                float(bool(ds.truncated)),
            ]
        )
        chain_ids.append(ds.chain_id)
    return np.asarray(rows, dtype=float).reshape(-1, len(names)), np.asarray(
        chain_ids, dtype=object
    ), names


def residual_auc_statistic(
    datasets,
    label_map: dict,
    *,
    ridge_cfg: RidgeConfig | None = None,
    n_splits: int = 5,
    seed: int = 42,
    max_gap: int | None = None,
) -> float:
    """End-to-end residual-feature AUC as a single scalar.

    Builds pairs -> OOF residuals -> per-chain residual features -> chain-grouped
    AUC. Used as the statistic for the step-shuffle null. Returns NaN if the
    pipeline cannot be evaluated (too few chains/labels).
    """
    pairs = make_supervised_pairs(datasets, max_gap=max_gap)
    if pairs["X_hist"].shape[0] == 0:
        return float("nan")
    try:
        res = oof_residuals(pairs, ridge_cfg or RidgeConfig(n_splits=n_splits))
    except ValueError:
        return float("nan")
    feats, cids, _ = chain_residual_features(res)
    labels = align_labels(cids, label_map)
    return grouped_auc(feats, labels, cids, seed=seed, n_splits=n_splits)["auc_oof"]
