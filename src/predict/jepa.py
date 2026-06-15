"""
src/predict/jepa.py — Small JEPA-style next-step predictor (Rung 2).

Purpose
-------
Rung 2 of the ladder: a compact *learned* next-step predictor f(x_t) -> x_{t+1}
(displacement by default, so the residual still measures the UNpredictable part
of the step and the model can only improve on persistence), trained with a
selectable anti-collapse term. It exists only to *beat* the Rung-1 linear ridge
(src/predict/predictor.py); if it cannot, the linear null stands.

The object of study is unchanged from Rung 1: the residual r_t = x_{t+1} - f(x_t)
under chain-grouped CV. `oof_residuals_jepa` is a deliberate drop-in for
`oof_residuals` — it returns the SAME `ResidualResult` container, so the whole
downstream stack (chain_residual_features, grouped_auc, the nulls) works on JEPA
residuals with no other change.

Anti-collapse, and why it lives in the LOSS, not the geometry read-out
---------------------------------------------------------------------
A learned predictor of a smooth target can cheat by collapsing its outputs onto a
low-rank / constant set ("everything maps to the mean step"), which would shrink
the residual artificially. JEPA-family methods add a term that keeps the
prediction distribution non-degenerate. Two are offered:

  'barlow'  Barlow-Twins redundancy reduction. Standardise the predicted batch
            and the target batch over the batch dimension, form the
            cross-correlation matrix C = (P_std^T T_std)/B, and push C -> I:
            diagonal -> 1 (invariance: each predicted coordinate tracks its
            target) and off-diagonal -> 0 (redundancy reduction: coordinates are
            decorrelated, so the prediction cannot collapse onto a few
            directions). Barlow-Twins, Zbontar et al. 2021 (arXiv:2103.03230),
            adapted here to a (prediction, target) pair rather than two augmented
            views.

  'sigreg'  A LeJEPA-style Sketched Isotropic Gaussian Regularisation
            (arXiv:2511.08544). Draw K = n_proj fixed random UNIT directions,
            project the standardised predictions onto each, and penalise every
            1-D marginal's departure from a standard Gaussian via a cheap
            moment-matching statistic (an Epps-Pulley / characteristic-function
            flavoured surrogate: match mean->0, variance->1, skew->0, and
            excess-kurtosis->0). A sum of many 1-D Gaussianity penalties on random
            projections is a sketch of "the joint is isotropic Gaussian"
            (Cramér-Wold); it both spreads variance across directions
            (anti-collapse) and discourages spiky/degenerate marginals. This is
            an APPROXIMATION of the full SIGReg energy-distance statistic, chosen
            to stay CPU-light and dependency-free.

Loss = MSE(pred, target) + lambda_ * anti_collapse.

CRITICAL (circularity): we never read absolute intrinsic-dim / isotropy off the
JEPA *latent* — that would be inflated by the very regulariser above. Geometry is
read only off the RESIDUAL (x_{t+1} - pred), and only relative/functional claims
(JEPA-vs-ridge, success-vs-failure) are made. See src/predict/__init__.py and
CONFOUNDS_AND_REMEDIATION.md.

Window vs single-step
---------------------
`JEPAPredictor` consumes a single state x_t. To extend to a causal window
x_{t-w+1..t} without touching the training loop, build the supervised pairs with a
history-stacked `X_hist` (concatenate / pool the last w steps into the feature
vector) and set `input_dim = w * d`; the MLP and the whole OOF path are unchanged
because they only ever see X_hist as an opaque (n, input_dim) matrix. The
predicted-NEXT reconstruction (add x_t back when target='delta') uses only the
LAST d columns of X_hist, which is exactly the current behaviour when w == 1.

Validation
----------
* AR(1) recovery: on x_{t+1} = A x_t + small noise the OOF JEPA residual norm is
  far below the persistence step norm (unit test test_ar1_recovery).
* Anti-collapse raises the predictions' effective covariance rank vs 'none' on
  collapse-prone data (unit test test_anticollapse_raises_variance).
* Chain-grouped OOF integrity: a chain never appears in train and test of the
  same fold (unit test test_chain_grouped_no_leakage); residuals are genuinely
  out-of-fold by construction (GroupKFold over pairs['groups']).

Milestone
---------
Predictive Geometry — Rung 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

# Reuse the Rung-1 container verbatim so JEPA residuals are a drop-in for the
# whole downstream stack (chain_residual_features / grouped_auc / nulls).
from src.predict.predictor import ResidualResult

__all__ = [
    "JEPAConfig",
    "JEPAPredictor",
    "barlow_twins_loss",
    "sigreg_loss",
    "oof_residuals_jepa",
]

_ANTI_COLLAPSE = ("none", "barlow", "sigreg")


@dataclass
class JEPAConfig:
    """Hyperparameters for the Rung-2 JEPA next-step predictor.

    Fields
    ------
    hidden_dim   : width of the single MLP hidden layer.
    anti_collapse: 'none' | 'barlow' | 'sigreg' (default 'barlow').
    lambda_      : weight on the anti-collapse term in the loss.
    epochs       : full-batch (or mini-batch) training epochs per CV fold.
    lr           : Adam learning rate.
    batch_size   : mini-batch size; if >= n_train the fold trains full-batch.
    n_splits     : chain-grouped CV folds.
    target       : 'delta' (predict x_{t+1}-x_t; residual = unpredictable part,
                   model can only improve on persistence) or 'absolute'
                   (predict x_{t+1} directly; ablation).
    seed         : global determinism seed (torch + numpy paths).
    n_proj       : number of random unit projections for the 'sigreg' sketch.
    """

    hidden_dim: int = 512
    anti_collapse: str = "barlow"
    lambda_: float = 1.0
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 256
    n_splits: int = 5
    target: str = "delta"
    seed: int = 42
    n_proj: int = 64


class JEPAPredictor(torch.nn.Module):
    """A small MLP next-step predictor: input_dim -> hidden_dim -> output_dim.

    One hidden layer with a GELU non-linearity (deliberately tiny: this is a
    CPU-bound Rung-2 probe, not a capacity contest). By default input_dim ==
    output_dim == d and the module predicts the displacement x_{t+1}-x_t; the
    caller adds x_t back to recover the next embedding.

    Causal window: pass input_dim = w * d and feed a history-stacked X_hist (the
    last w states concatenated). Nothing else changes — the module treats its
    input as an opaque feature vector. See the module docstring.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── anti-collapse terms ──────────────────────────────────────────────────────

def _standardize_cols(z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Standardise each column over the batch dimension (mean 0, std 1)."""
    mean = z.mean(dim=0, keepdim=True)
    std = z.std(dim=0, unbiased=False, keepdim=True)
    return (z - mean) / (std + eps)


def barlow_twins_loss(pred: torch.Tensor, target: torch.Tensor,
                      eps: float = 1e-5) -> torch.Tensor:
    """Barlow-Twins redundancy-reduction term on (prediction, target).

    Standardise both batches over the batch dim, form the cross-correlation
    C = (P_std^T @ T_std) / B, and drive C toward the identity: the diagonal to 1
    (invariance — each predicted coordinate co-varies with its target) and the
    off-diagonal to 0 (redundancy reduction — predicted coordinates are
    decorrelated, so outputs cannot collapse onto a low-rank set). Returns a
    scalar; smaller is better. Off-diagonal weight 1/D keeps the two parts on a
    comparable scale across widths.
    """
    B = pred.shape[0]
    if B < 2:
        return pred.new_zeros(())
    p = _standardize_cols(pred, eps)
    t = _standardize_cols(target, eps)
    c = (p.T @ t) / B                       # (D, D) cross-correlation
    D = c.shape[0]
    on_diag = ((torch.diagonal(c) - 1.0) ** 2).sum()
    off = c - torch.diag(torch.diagonal(c))
    off_diag = (off ** 2).sum() / max(1, D)
    return on_diag + off_diag


def _gaussianity_moment_penalty(u: torch.Tensor) -> torch.Tensor:
    """Per-projection departure-from-N(0,1) penalty via low-order moments.

    For each 1-D projection column u_k: match mean->0, variance->1, skewness->0,
    and excess kurtosis->0 (a standard Gaussian has skew 0, kurtosis 3). This is
    an Epps-Pulley / moment-matching surrogate for a full characteristic-function
    Gaussianity test — cheap and smooth. Returns a scalar (sum over projections).
    """
    mean = u.mean(dim=0)
    var = u.var(dim=0, unbiased=False)
    std = torch.sqrt(var + 1e-8)
    centred = u - mean
    m3 = (centred ** 3).mean(dim=0)
    m4 = (centred ** 4).mean(dim=0)
    skew = m3 / (std ** 3 + 1e-8)
    kurt_excess = m4 / (var ** 2 + 1e-8) - 3.0
    pen = (mean ** 2) + (var - 1.0) ** 2 + skew ** 2 + kurt_excess ** 2
    return pen.sum()


def sigreg_loss(pred: torch.Tensor, n_proj: int, generator: torch.Generator,
                eps: float = 1e-5) -> torch.Tensor:
    """LeJEPA-style sketched isotropic-Gaussian regulariser (approximate).

    Standardise the predicted batch, draw `n_proj` fixed random UNIT directions,
    project onto each, and penalise every 1-D marginal's deviation from a
    standard Gaussian (mean/variance/skew/kurtosis moment matching). By
    Cramér-Wold, isotropy of all 1-D projections characterises an isotropic joint
    Gaussian, so summing many such penalties both SPREADS variance across
    directions (anti-collapse) and discourages spiky marginals. Approximation of
    the exact SIGReg energy-distance statistic; chosen to stay CPU-light. Returns
    a scalar (averaged over projections); smaller is better.
    """
    B, D = pred.shape
    if B < 2:
        return pred.new_zeros(())
    p = _standardize_cols(pred, eps)
    dirs = torch.randn(D, n_proj, generator=generator, dtype=pred.dtype,
                       device=pred.device)
    dirs = dirs / (dirs.norm(dim=0, keepdim=True) + 1e-8)  # unit directions
    proj = p @ dirs                                        # (B, n_proj)
    return _gaussianity_moment_penalty(proj) / max(1, n_proj)


def _anti_collapse_term(name: str, pred: torch.Tensor, target: torch.Tensor,
                        n_proj: int, generator: torch.Generator) -> torch.Tensor:
    if name == "none":
        return pred.new_zeros(())
    if name == "barlow":
        return barlow_twins_loss(pred, target)
    if name == "sigreg":
        return sigreg_loss(pred, n_proj, generator)
    raise ValueError(f"anti_collapse must be one of {_ANTI_COLLAPSE}, got {name!r}")


# ── training (one fold) ──────────────────────────────────────────────────────

def _train_fold(
    X: np.ndarray,
    Y: np.ndarray,
    config: JEPAConfig,
    generator: torch.Generator,
) -> JEPAPredictor:
    """Train one JEPAPredictor on (standardised X -> Y). Deterministic, CPU."""
    device = torch.device("cpu")
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)
    Yt = torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float32)).to(device)
    n = Xt.shape[0]

    model = JEPAPredictor(Xt.shape[1], Yt.shape[1], hidden_dim=config.hidden_dim)
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    mse = torch.nn.MSELoss()
    bs = max(1, min(config.batch_size, n))

    for _ in range(config.epochs):
        # Deterministic per-epoch shuffle driven by the shared CPU generator.
        perm = torch.randperm(n, generator=generator, device=device)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            xb, yb = Xt[idx], Yt[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb)
            if config.anti_collapse != "none":
                loss = loss + config.lambda_ * _anti_collapse_term(
                    config.anti_collapse, pred, yb, config.n_proj, generator
                )
            loss.backward()
            opt.step()
    model.eval()
    return model


# ── out-of-fold residuals (drop-in for predictor.oof_residuals) ──────────────

def oof_residuals_jepa(pairs: dict, config: JEPAConfig | None = None) -> ResidualResult:
    """Fit the JEPA predictor under chain-grouped CV; return OOF residuals.

    Drop-in replacement for `src.predict.predictor.oof_residuals`: same `pairs`
    input (from make_supervised_pairs), same `ResidualResult` output, so
    chain_residual_features / grouped_auc / the nulls consume it unchanged.

    Per fold: standardise inputs (scaler FIT ON TRAIN only), train the JEPA for
    config.epochs on the displacement (target='delta') or absolute next embedding,
    predict held-out pairs, and reconstruct the predicted NEXT embedding (add x_t
    back when target='delta'). residual = X_next - pred.

    Determinism: torch.manual_seed(seed), a single explicit CPU torch.Generator
    threads the per-epoch shuffles and the sigreg sketch, and
    torch.use_deterministic_algorithms(True) is set for the duration (restored on
    exit). Raises ValueError if fewer than two distinct chains are present.
    """
    config = config or JEPAConfig()
    if config.target not in ("delta", "absolute"):
        raise ValueError(
            f"target must be 'delta' or 'absolute', got {config.target!r}"
        )
    if config.anti_collapse not in _ANTI_COLLAPSE:
        raise ValueError(
            f"anti_collapse must be one of {_ANTI_COLLAPSE}, got {config.anti_collapse!r}"
        )

    Xh = np.asarray(pairs["X_hist"], dtype=np.float64)
    Xn = np.asarray(pairs["X_next"], dtype=np.float64)
    groups = np.asarray(pairs["groups"], dtype=object)

    uniq = np.unique(groups)
    if uniq.size < 2:
        raise ValueError(
            f"oof_residuals_jepa needs >=2 distinct chains for grouped CV, got {uniq.size}"
        )
    n_splits = min(config.n_splits, uniq.size)

    # The model targets the displacement (delta) or the absolute next embedding;
    # `pred` always holds the reconstructed NEXT embedding (so residual = Xn-pred).
    fit_target = (Xn - Xh) if config.target == "delta" else Xn
    pred = np.full_like(Xn, np.nan)

    prev_det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        gkf = GroupKFold(n_splits=n_splits)
        for fold_i, (tr, te) in enumerate(gkf.split(Xh, groups=groups)):
            # Fresh, fold-dependent but fully reproducible RNG state.
            torch.manual_seed(config.seed + fold_i)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(config.seed + fold_i)

            # Standardise inputs on TRAIN statistics only (CF-2 hygiene).
            mu = Xh[tr].mean(axis=0, keepdims=True)
            sd = Xh[tr].std(axis=0, keepdims=True)
            sd = np.where(sd < 1e-8, 1.0, sd)
            Xtr = (Xh[tr] - mu) / sd
            Xte = (Xh[te] - mu) / sd

            model = _train_fold(Xtr, fit_target[tr], config, gen)
            with torch.no_grad():
                yhat = model(
                    torch.from_numpy(np.ascontiguousarray(Xte, dtype=np.float32))
                ).numpy().astype(np.float64)
            pred[te] = (Xh[te] + yhat) if config.target == "delta" else yhat
    finally:
        torch.use_deterministic_algorithms(prev_det)

    residuals = Xn - pred
    return ResidualResult(
        residuals=residuals,
        pred=pred,
        groups=groups,
        gaps=np.asarray(pairs["gaps"], dtype=int),
        pos=np.asarray(pairs["pos"], dtype=int),
    )
