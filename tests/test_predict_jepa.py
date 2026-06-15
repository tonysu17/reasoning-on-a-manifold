"""
tests/test_predict_jepa.py — Unit tests for src/predict/jepa.py (Rung 2).

Covers: AR(1) recovery (the learned JEPA next-step predictor beats persistence);
the anti-collapse terms ('barlow', 'sigreg') materially raise the effective
covariance rank of the predictions versus 'none' on collapse-prone data; and
chain-grouped OOF integrity (no chain id appears in both train and test of any
fold).

Deliberately small/fast (d<=16, <=20 epochs, <=40 chains) so it runs on CPU in
seconds. Mirrors tests/test_predict.py.

Runs under pytest, or standalone:
  PYTHONPATH=<repo> python3 tests/test_predict_jepa.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict.trajectory_dataset import (  # noqa: E402
    StepDataset,
    make_supervised_pairs,
)
from src.predict.predictor import ResidualResult  # noqa: E402
from src.predict.jepa import (  # noqa: E402
    JEPAConfig,
    JEPAPredictor,
    barlow_twins_loss,
    sigreg_loss,
    oof_residuals_jepa,
)


# ── synthetic data ──────────────────────────────────────────────────────────

def _ar1_datasets(K, T, d, noise, seed, A=None):
    """K chains following a SHARED linear map x_{t+1} = A x_t + noise*eps.

    A shared map across chains is exactly what a cross-chain CV predictor can
    learn (mirrors the AR(1) helper in tests/test_predict.py). Returns (datasets, A).
    """
    rng = np.random.default_rng(seed)
    if A is None:
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        A = 0.85 * Q
    datasets = []
    for k in range(K):
        X = np.zeros((T, d), dtype=np.float32)
        X[0] = rng.standard_normal(d)
        nz = noise(k) if callable(noise) else noise
        for t in range(1, T):
            X[t] = A @ X[t - 1] + nz * rng.standard_normal(d)
        datasets.append(
            StepDataset(
                chain_id=f"chain{k}",
                X=X,
                orig_indices=np.arange(T, dtype=int),
                behaviours=["deduction"] * T,
            )
        )
    return datasets, A


def _participation_ratio(Z: np.ndarray) -> float:
    """Effective covariance rank of rows of Z: (sum λ)^2 / sum(λ^2), λ>=0.

    A smooth, threshold-free proxy for "how many directions carry the variance".
    1.0 means a single direction (full collapse); D means isotropic. More robust
    for a unit test than a hard numerical-rank cutoff.
    """
    Z = np.asarray(Z, dtype=np.float64)
    Z = Z - Z.mean(axis=0, keepdims=True)
    cov = (Z.T @ Z) / max(1, Z.shape[0] - 1)
    w = np.linalg.eigvalsh(cov)
    w = np.clip(w, 0.0, None)
    s1 = float(w.sum())
    s2 = float((w * w).sum())
    if s2 <= 0.0:
        return 0.0
    return (s1 * s1) / s2


def _oof_predictions(pairs, res: ResidualResult) -> np.ndarray:
    """Recover the OOF *delta* predictions (pred_next - x_hist) for rank checks.

    Reading effective rank off the predicted DISPLACEMENT (the thing the model
    actually emits) is the honest target of the anti-collapse term — and it never
    touches the residual geometry used for science.
    """
    return res.pred - np.asarray(pairs["X_hist"], dtype=np.float64)


# ── tests ───────────────────────────────────────────────────────────────────

def test_ar1_recovery():
    """OOF JEPA residual norm << persistence step norm on a shared linear map."""
    cfg = JEPAConfig(hidden_dim=64, anti_collapse="barlow", lambda_=1.0,
                     epochs=20, lr=5e-3, batch_size=128, n_splits=4, seed=0)
    datasets, _ = _ar1_datasets(K=40, T=12, d=8, noise=0.05, seed=1)

    pairs = make_supervised_pairs(datasets)
    res = oof_residuals_jepa(pairs, cfg)

    resid_norm = float(np.linalg.norm(res.residuals, axis=1).mean())
    step_norm = float(
        np.linalg.norm(pairs["X_next"] - pairs["X_hist"], axis=1).mean()
    )
    # Learned predictor leaves far less than the persistence (x_{t+1}=x_t) residual.
    assert resid_norm < 0.6 * step_norm, (resid_norm, step_norm)


def test_ar1_recovery_determinism():
    """Same config -> bitwise-identical OOF residuals (CPU determinism contract)."""
    cfg = JEPAConfig(hidden_dim=32, anti_collapse="barlow", epochs=10, lr=5e-3,
                     batch_size=128, n_splits=3, seed=7)
    datasets, _ = _ar1_datasets(K=18, T=10, d=6, noise=0.05, seed=2)
    pairs = make_supervised_pairs(datasets)
    r1 = oof_residuals_jepa(pairs, cfg).residuals
    r2 = oof_residuals_jepa(pairs, cfg).residuals
    assert np.array_equal(r1, r2), float(np.abs(r1 - r2).max())


def _collapse_prone_pairs(K, T, d, seed):
    """Pairs whose MSE-optimal next-step map is near-low-rank (collapse-prone).

    The displacement target depends on x_t only through a SINGLE shared direction
    (rank-1 signal) plus small isotropic noise. An MSE-only learner is happy to
    emit predictions living on ~one direction (low effective rank); a working
    anti-collapse term must push the predicted displacements to spread across more
    directions. Built as chains so the OOF/grouped machinery applies unchanged.
    """
    rng = np.random.default_rng(seed)
    u_in = rng.standard_normal(d)
    u_in /= np.linalg.norm(u_in)
    u_out = rng.standard_normal(d)
    u_out /= np.linalg.norm(u_out)
    datasets = []
    for k in range(K):
        X = np.zeros((T, d), dtype=np.float32)
        X[0] = rng.standard_normal(d)
        for t in range(1, T):
            score = float(X[t - 1] @ u_in)
            delta = 0.6 * score * u_out + 0.02 * rng.standard_normal(d)
            X[t] = X[t - 1] + delta.astype(np.float32)
        datasets.append(
            StepDataset(chain_id=f"chain{k}", X=X,
                        orig_indices=np.arange(T, dtype=int),
                        behaviours=["deduction"] * T)
        )
    return make_supervised_pairs(datasets)


def test_anticollapse_raises_variance():
    """'barlow' and 'sigreg' raise predicted-displacement effective rank vs 'none'."""
    pairs = _collapse_prone_pairs(K=40, T=10, d=12, seed=3)

    def eff_rank(anti):
        cfg = JEPAConfig(hidden_dim=64, anti_collapse=anti, lambda_=5.0,
                         epochs=20, lr=5e-3, batch_size=128, n_splits=4,
                         seed=0, n_proj=32)
        res = oof_residuals_jepa(pairs, cfg)
        return _participation_ratio(_oof_predictions(pairs, res))

    pr_none = eff_rank("none")
    pr_barlow = eff_rank("barlow")
    pr_sigreg = eff_rank("sigreg")

    # Both anti-collapse variants spread the predictions across materially more
    # directions than plain MSE (clear inequality, not a hair's breadth).
    assert pr_barlow > 1.5 * pr_none, (pr_none, pr_barlow)
    assert pr_sigreg > 1.5 * pr_none, (pr_none, pr_sigreg)


def test_anticollapse_terms_are_well_behaved():
    """Direct sanity on the loss terms: collapsed pred -> larger penalty than spread.

    Guards the terms themselves (independent of the optimiser): a near-constant /
    rank-1 prediction batch should incur a strictly larger anti-collapse penalty
    than a spread, decorrelated one.
    """
    torch.manual_seed(0)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    B, D = 256, 8

    target = torch.randn(B, D, generator=gen)
    collapsed = torch.randn(B, 1, generator=gen).repeat(1, D)  # rank-1
    spread = torch.randn(B, D, generator=gen)                  # full-rank

    assert barlow_twins_loss(collapsed, target).item() > \
        barlow_twins_loss(spread, target).item()
    assert sigreg_loss(collapsed, 32, gen).item() > \
        sigreg_loss(spread, 32, gen).item()


def test_chain_grouped_no_leakage():
    """No chain id appears in both train and test indices of any GroupKFold fold.

    Reproduces the exact split oof_residuals_jepa uses (GroupKFold over
    pairs['groups']) and asserts disjoint chain sets per fold — the CF-2 guard.
    """
    from sklearn.model_selection import GroupKFold

    datasets, _ = _ar1_datasets(K=20, T=8, d=6, noise=0.05, seed=4)
    pairs = make_supervised_pairs(datasets)
    groups = np.asarray(pairs["groups"], dtype=object)

    n_splits = min(5, np.unique(groups).size)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(pairs["X_hist"], groups=groups):
        train_chains = set(groups[tr].tolist())
        test_chains = set(groups[te].tolist())
        assert train_chains.isdisjoint(test_chains), (
            train_chains & test_chains
        )
        # Sanity: every pair landed in exactly one side of the split.
        assert len(tr) + len(te) == groups.shape[0]


def test_oof_residual_result_shape_and_dropin():
    """oof_residuals_jepa returns a ResidualResult aligned to the input pairs.

    Confirms it is a structural drop-in for predictor.oof_residuals: same
    container, residuals == X_next - pred, and provenance arrays carried through.
    """
    datasets, _ = _ar1_datasets(K=16, T=8, d=6, noise=0.05, seed=5)
    pairs = make_supervised_pairs(datasets)
    cfg = JEPAConfig(hidden_dim=32, epochs=8, lr=5e-3, n_splits=4, seed=0)
    res = oof_residuals_jepa(pairs, cfg)

    assert isinstance(res, ResidualResult)
    n, d = pairs["X_hist"].shape
    assert res.residuals.shape == (n, d)
    assert res.pred.shape == (n, d)
    assert res.groups.shape[0] == n and res.gaps.shape[0] == n and res.pos.shape[0] == n
    # residual == X_next - pred, and no NaNs (every pair is covered by some fold).
    recon = np.asarray(pairs["X_next"], dtype=np.float64) - res.pred
    assert np.allclose(res.residuals, recon, atol=1e-9)
    assert np.isfinite(res.residuals).all()


def test_jepa_predictor_window_shape():
    """JEPAPredictor accepts input_dim != output_dim (causal-window extension)."""
    w, d = 3, 6
    model = JEPAPredictor(input_dim=w * d, output_dim=d, hidden_dim=16)
    x = torch.zeros(5, w * d)
    assert model(x).shape == (5, d)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
