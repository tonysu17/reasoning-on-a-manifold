"""
tests/test_predict.py — Unit tests for src/predict (Predictive Geometry, Rung 1).

Covers: supervised-pair construction + gap accounting; within-chain step shuffle;
AR(1) recovery (the learned predictor beats persistence and a shuffled control);
chain-grouped AUC honesty (no-signal -> ~0.5, planted -> high); and both nulls
(label permutation and step shuffle) collapsing a planted effect.

Runs under pytest, or standalone:  PYTHONPATH=<repo> python3 tests/test_predict.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict.trajectory_dataset import (  # noqa: E402
    StepDataset,
    make_supervised_pairs,
    step_shuffle_within_chain,
)
from src.predict.predictor import (  # noqa: E402
    RidgeConfig,
    chain_residual_features,
    oof_residuals,
)
from src.predict.evaluation import (  # noqa: E402
    align_labels,
    grouped_auc,
    residual_auc_statistic,
)
from src.predict.nulls_predict import (  # noqa: E402
    label_permutation_null,
    step_shuffle_null,
)
from src.predict.labels import parse_verdict  # noqa: E402


# ── synthetic data ──────────────────────────────────────────────────────────

def _ar1_datasets(K, T, d, noise, seed, A=None, correct_for=None):
    """K chains following a SHARED linear map x_{t+1} = A x_t + noise*eps.

    `correct_for(k) -> bool|None` optionally assigns a chain-level label.
    Returns (datasets, A).
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
                correct=None if correct_for is None else correct_for(k),
            )
        )
    return datasets, A


# ── tests ───────────────────────────────────────────────────────────────────

def test_make_supervised_pairs_shapes_and_gaps():
    ds = StepDataset(chain_id="c", X=np.zeros((4, 3), np.float32),
                     orig_indices=np.array([0, 1, 3, 7]))
    pairs = make_supervised_pairs([ds])
    assert pairs["X_hist"].shape == (3, 3)
    assert pairs["X_next"].shape == (3, 3)
    assert list(pairs["gaps"]) == [1, 2, 4]
    assert list(pairs["pos"]) == [0, 1, 2]
    # max_gap drops the wide pairs, keeping only the adjacent one.
    adj = make_supervised_pairs([ds], max_gap=1)
    assert adj["X_hist"].shape == (1, 3)
    assert list(adj["gaps"]) == [1]


def test_step_shuffle_preserves_membership():
    rng = np.random.default_rng(0)
    X = np.arange(15, dtype=np.float32).reshape(5, 3)
    ds = StepDataset(chain_id="c", X=X, orig_indices=np.arange(5),
                     behaviours=list("abcde"))
    out = step_shuffle_within_chain([ds], rng)[0]
    assert out.T == ds.T
    # same multiset of rows, order changed (with high probability), labels travel
    assert {tuple(r) for r in out.X} == {tuple(r) for r in ds.X}
    assert sorted(out.behaviours) == sorted(ds.behaviours)
    assert sorted(out.orig_indices.tolist()) == list(range(5))


def test_ar1_recovery_beats_persistence_and_shuffle():
    cfg = RidgeConfig(alpha=1.0, n_splits=4)
    datasets, _ = _ar1_datasets(K=40, T=12, d=8, noise=0.05, seed=1)

    pairs = make_supervised_pairs(datasets)
    res = oof_residuals(pairs, cfg)
    resid_norm = float(np.linalg.norm(res.residuals, axis=1).mean())
    step_norm = float(
        np.linalg.norm(pairs["X_next"] - pairs["X_hist"], axis=1).mean()
    )
    # learned predictor leaves far less than the persistence (x_{t+1}=x_t) residual
    assert resid_norm < 0.4 * step_norm, (resid_norm, step_norm)

    # shuffling step order destroys the learnable relationship -> larger residual
    rng = np.random.default_rng(7)
    sh_pairs = make_supervised_pairs(step_shuffle_within_chain(datasets, rng))
    sh_resid = float(np.linalg.norm(oof_residuals(sh_pairs, cfg).residuals, axis=1).mean())
    assert sh_resid > 1.8 * resid_norm, (sh_resid, resid_norm)


def test_grouped_auc_no_signal_is_chance():
    rng = np.random.default_rng(2)
    m = 80
    feats = rng.standard_normal((m, 4))
    labels = rng.integers(0, 2, size=m).astype(float)
    cids = np.array([f"c{i}" for i in range(m)], dtype=object)
    out = grouped_auc(feats, labels, cids, n_splits=5)
    assert 0.35 <= out["auc_oof"] <= 0.65, out  # honest: no leakage inflation


def test_grouped_auc_planted_signal_detected():
    rng = np.random.default_rng(3)
    m = 80
    labels = rng.integers(0, 2, size=m).astype(float)
    feats = np.column_stack([
        labels + 0.5 * rng.standard_normal(m),   # informative column
        rng.standard_normal(m),
    ])
    cids = np.array([f"c{i}" for i in range(m)], dtype=object)
    out = grouped_auc(feats, labels, cids, n_splits=5)
    assert out["auc_oof"] > 0.8, out


def test_label_permutation_null():
    rng = np.random.default_rng(4)
    m = 80
    labels = rng.integers(0, 2, size=m).astype(float)
    cids = np.array([f"c{i}" for i in range(m)], dtype=object)
    planted = np.column_stack([labels + 0.5 * rng.standard_normal(m),
                               rng.standard_normal(m)])
    null_planted = label_permutation_null(planted, cids, labels,
                                          n_resamples=200, seed=4)
    assert null_planted.real_value > 0.8
    assert null_planted.p_value < 0.05, null_planted

    noise_feats = rng.standard_normal((m, 2))
    null_none = label_permutation_null(noise_feats, cids, labels,
                                       n_resamples=200, seed=5)
    assert null_none.p_value > 0.05, null_none


def _scrambled_label_datasets(K, T, d, seed):
    """Correct chains follow the shared AR(1) map IN ORDER; incorrect chains have
    the SAME static cloud of points but scrambled order. The only thing that
    distinguishes the classes is temporal predictability, so a step-order shuffle
    must erase the separation (a static-cloud statistic would not be erased)."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    A = 0.85 * Q
    datasets = []
    for k in range(K):
        X = np.zeros((T, d), dtype=np.float32)
        X[0] = rng.standard_normal(d)
        for t in range(1, T):
            X[t] = A @ X[t - 1] + 0.05 * rng.standard_normal(d)
        correct = (k % 2 == 0)
        if not correct:
            X = X[rng.permutation(T)]  # same cloud, no temporal predictability
        datasets.append(StepDataset(chain_id=f"chain{k}", X=X,
                                    orig_indices=np.arange(T, dtype=int),
                                    correct=correct))
    return datasets


def test_step_shuffle_null_kills_order_dependent_signal():
    # Separation lives ONLY in temporal predictability -> shuffling must kill it.
    datasets = _scrambled_label_datasets(K=40, T=12, d=8, seed=6)
    label_map = {ds.chain_id: ds.correct for ds in datasets}
    cfg = RidgeConfig(alpha=1.0, n_splits=4)

    def stat(dsets):
        return residual_auc_statistic(dsets, label_map, ridge_cfg=cfg, n_splits=4)

    real = stat(datasets)
    assert real > 0.75, real
    nr = step_shuffle_null(datasets, stat, n_resamples=60, seed=6)
    assert nr.real_value > nr.null_mean, nr
    assert nr.p_value < 0.05, nr


def test_step_shuffle_null_ignores_static_magnitude_signal():
    # A residual-MAGNITUDE signal (low- vs high-noise chains) is a property of the
    # static point cloud, not of order; the step-shuffle null must NOT call it
    # significant. This documents precisely what the null does and does not test.
    def noise(k):
        return 0.03 if (k % 2 == 0) else 0.30
    datasets, _ = _ar1_datasets(K=40, T=12, d=8, noise=noise, seed=6,
                                correct_for=lambda k: (k % 2 == 0))
    label_map = {ds.chain_id: ds.correct for ds in datasets}
    cfg = RidgeConfig(alpha=1.0, n_splits=4)

    def stat(dsets):
        return residual_auc_statistic(dsets, label_map, ridge_cfg=cfg, n_splits=4)

    nr = step_shuffle_null(datasets, stat, n_resamples=60, seed=6)
    assert nr.p_value > 0.05, nr


def test_parse_verdict_robustness():
    assert parse_verdict('{"verdict":"correct","confidence":"high","rationale":"ok"}')["correct"] is True
    assert parse_verdict('```json\n{"verdict":"incorrect","confidence":"low","rationale":"x"}\n```')["correct"] is False
    assert parse_verdict('text {"verdict":"uncertain","confidence":"low","rationale":"cut"} end')["correct"] is None
    assert parse_verdict("This proof is incorrect.")["correct"] is False  # keyword fallback
    assert parse_verdict("")["correct"] is None                            # never a fake label
    out = parse_verdict('{"verdict":"correct","confidence":"medium","rationale":"valid"}')
    assert out["confidence"] == "medium" and out["rationale"] == "valid"


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
