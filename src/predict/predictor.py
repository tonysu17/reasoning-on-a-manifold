"""
src/predict/predictor.py — Linear next-step predictor (Rung 1) + residual geometry.

Purpose
-------
Rung 1 of the ladder: a ridge next-step predictor x_t -> x_{t+1} with NO
anti-collapse term, so its residual geometry is uncontaminated by any
isotropy/rank-inflating regulariser. (Reading absolute intrinsic-dim or isotropy
off a JEPA latent whose loss inflates exactly those quantities is circular; the
residual of a plain linear map has no such penalty.) Out-of-fold predictions are
produced under chain-grouped CV so a chain never trains and tests together
(CF-2 / effective-N).

The residual r_t = x_{t+1} - f(x_t) is the object of study: per-chain residual
statistics (magnitude, growth, direction churn) are the candidate
correctness/branch-point signal.

Validation
----------
* AR(1) recovery: on x_{t+1} = A x_t + noise, the out-of-fold residual norm
  tracks the noise scale and is far below the raw step size (unit test).
* Out-of-fold integrity: predictions for a chain come only from folds that did
  not train on it (guaranteed by GroupKFold over chain_id).

Milestone
---------
Predictive Geometry — Rung 1.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

__all__ = [
    "RidgeConfig",
    "ResidualResult",
    "oof_residuals",
    "chain_residual_features",
    "RESIDUAL_FEATURE_NAMES",
]

RESIDUAL_FEATURE_NAMES: list[str] = [
    "resid_mean",      # mean residual norm over the chain's pairs
    "resid_std",       # spread of residual norm (bursty vs uniform error)
    "resid_max",       # largest single-step prediction error (sharpest branch)
    "resid_slope",     # trend of residual norm across the trajectory
    "resid_dir_churn", # mean (1 - cos) between consecutive residual vectors
]


@dataclass
class RidgeConfig:
    """Hyperparameters for the Rung-1 ridge next-step predictor."""

    alpha: float = 10.0      # ridge penalty (d=1536 is high-dim; regularise)
    standardize: bool = True # standardise inputs per training fold
    n_splits: int = 5        # chain-grouped CV folds
    seed: int = 42           # reserved for API symmetry (GroupKFold is deterministic)
    target: str = "delta"    # "delta": predict x_{t+1}-x_t (residual connection;
                             #   residual = the UNpredictable part of the step, and
                             #   the predictor can only improve on persistence).
                             # "absolute": predict x_{t+1} directly (ablation).


@dataclass
class ResidualResult:
    """Out-of-fold next-step residuals and their provenance."""

    residuals: np.ndarray  # (n, d) OOF residual vectors x_{t+1} - f(x_t)
    pred: np.ndarray       # (n, d) OOF predictions f(x_t)
    groups: np.ndarray     # (n,) chain_id per pair
    gaps: np.ndarray       # (n,) original-span gap per pair
    pos: np.ndarray        # (n,) position t within the retained trajectory


def oof_residuals(pairs: dict, config: RidgeConfig | None = None) -> ResidualResult:
    """Fit the ridge predictor under chain-grouped CV; return OOF residuals.

    `pairs` is the dict from `make_supervised_pairs`. For each GroupKFold fold a
    fresh scaler+Ridge is fit on the training pairs and used to predict the held
    -out pairs, so every residual is genuinely out-of-fold. Raises ValueError if
    fewer than two distinct chains are present (CV is undefined).
    """
    config = config or RidgeConfig()
    if config.target not in ("delta", "absolute"):
        raise ValueError(f"target must be 'delta' or 'absolute', got {config.target!r}")
    Xh = np.asarray(pairs["X_hist"], dtype=np.float64)
    Xn = np.asarray(pairs["X_next"], dtype=np.float64)
    groups = np.asarray(pairs["groups"], dtype=object)

    uniq = np.unique(groups)
    if uniq.size < 2:
        raise ValueError(
            f"oof_residuals needs >=2 distinct chains for grouped CV, got {uniq.size}"
        )
    n_splits = min(config.n_splits, uniq.size)

    # Fit on the displacement (delta) or the absolute next embedding.
    fit_target = (Xn - Xh) if config.target == "delta" else Xn
    pred = np.full_like(Xn, np.nan)  # predicted NEXT embedding either way
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(Xh, groups=groups):
        if config.standardize:
            scaler = StandardScaler().fit(Xh[tr])
            Xtr, Xte = scaler.transform(Xh[tr]), scaler.transform(Xh[te])
        else:
            Xtr, Xte = Xh[tr], Xh[te]
        model = Ridge(alpha=config.alpha)
        model.fit(Xtr, fit_target[tr])
        yhat = model.predict(Xte)
        pred[te] = (Xh[te] + yhat) if config.target == "delta" else yhat

    residuals = Xn - pred
    return ResidualResult(
        residuals=residuals,
        pred=pred,
        groups=groups,
        gaps=np.asarray(pairs["gaps"], dtype=int),
        pos=np.asarray(pairs["pos"], dtype=int),
    )


def _safe_slope(y: np.ndarray) -> float:
    """OLS slope of y against its index; 0 for fewer than two points."""
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=float)
    dx = x - x.mean()
    denom = float(np.sum(dx * dx))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(dx * (y - y.mean())) / denom)


def _dir_churn(residual_block: np.ndarray) -> float:
    """Mean (1 - cosine) between consecutive residual vectors; 0 if < 2 rows."""
    R = np.asarray(residual_block, dtype=float)
    if R.shape[0] < 2:
        return 0.0
    norms = np.linalg.norm(R, axis=1, keepdims=True)
    U = R / np.maximum(norms, 1e-12)
    cos = np.sum(U[1:] * U[:-1], axis=1)
    return float(np.mean(1.0 - cos))


def chain_residual_features(res: ResidualResult):
    """Aggregate out-of-fold residuals to per-chain features.

    Returns
    -------
    (features, chain_ids, feature_names)
      features    : (m, k) one row per chain, columns = RESIDUAL_FEATURE_NAMES
      chain_ids   : (m,) object array of chain ids (row order of `features`)
      feature_names : list[str]

    Rows whose OOF prediction is NaN (none, in normal use) are ignored. Per-chain
    pairs are ordered by `pos` before computing slope and direction churn so the
    temporal features are meaningful.
    """
    rnorm = np.linalg.norm(res.residuals, axis=1)

    idx_by_chain: dict[object, list[int]] = defaultdict(list)
    for i, c in enumerate(res.groups):
        if np.isfinite(rnorm[i]):
            idx_by_chain[c].append(i)

    chain_ids: list[object] = []
    rows: list[list[float]] = []
    for cid, idxs in idx_by_chain.items():
        idxs_arr = np.asarray(idxs, dtype=int)
        order = np.argsort(res.pos[idxs_arr], kind="stable")
        ordered = idxs_arr[order]
        rn = rnorm[ordered]
        rows.append(
            [
                float(rn.mean()),
                float(rn.std()),
                float(rn.max()),
                _safe_slope(rn),
                _dir_churn(res.residuals[ordered]),
            ]
        )
        chain_ids.append(cid)

    features = np.asarray(rows, dtype=float).reshape(-1, len(RESIDUAL_FEATURE_NAMES))
    return features, np.asarray(chain_ids, dtype=object), list(RESIDUAL_FEATURE_NAMES)
