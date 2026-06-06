"""Unit tests for the configurable span-pooling (_pool) in activation extraction.

Pooling is the methodological knob behind "mean vs last-token": last() reads the
context-complete final token; mean() smears the within-span trajectory. These
tests pin the semantics so a regression can't silently change what gets pooled.
"""

import numpy as np
import pytest
import torch

from src.activation_extraction import _pool, POOLING_MODES


def _slice():
    # 4 positions, hidden=3. Rows chosen so each mode gives a distinct answer.
    return torch.tensor([
        [0.0, 0.0, 0.0],   # first
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 9.0],   # last
    ])


def test_last_returns_final_execution_token():
    out = _pool(_slice(), "last").numpy()
    assert np.allclose(out, [0.0, 0.0, 9.0])


def test_first_returns_onset_token():
    out = _pool(_slice(), "first").numpy()
    assert np.allclose(out, [0.0, 0.0, 0.0])


def test_mean_averages_all_positions():
    out = _pool(_slice(), "mean").numpy()
    assert np.allclose(out, [0.25, 0.5, 2.25])


def test_max_is_elementwise():
    out = _pool(_slice(), "max").numpy()
    assert np.allclose(out, [1.0, 2.0, 9.0])


def test_mean_loses_order_information():
    """The headline caution: mean is permutation-invariant, last is not."""
    s = _slice()
    perm = s[torch.tensor([3, 2, 1, 0])]
    assert np.allclose(_pool(s, "mean").numpy(), _pool(perm, "mean").numpy())   # mean: same
    assert not np.allclose(_pool(s, "last").numpy(), _pool(perm, "last").numpy())  # last: differs


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        _pool(_slice(), "median")


def test_all_modes_listed_are_supported():
    for m in POOLING_MODES:
        assert _pool(_slice(), m).shape == (3,)


def test_pooling_sweep_report(tmp_path):
    """The auto-sweep report: cos(primary, mode) per behaviour + d_eff per mode.
    Built from synthetic activation dirs — no model needed."""
    from src.activation_extraction import _write_pooling_sweep_report
    rng = np.random.default_rng(0)
    behs = ["backtracking", "uncertainty-estimation"]
    L = 27
    (tmp_path / "pool_last").mkdir()
    for b in behs:
        base = rng.standard_normal((40, 64))
        np.save(tmp_path / f"{b}_layer{L}.npy", base)                                  # primary (mean)
        np.save(tmp_path / "pool_last" / f"{b}_layer{L}.npy",
                base + 0.01 * rng.standard_normal((40, 64)))                            # 'last' ≈ primary
    rep = _write_pooling_sweep_report(tmp_path, primary="mean",
                                      modes=["mean", "last"], layers=[L], behaviours=behs)
    assert (tmp_path / "pooling_sweep.json").exists()
    for b in behs:
        # mean vs itself is exactly 1; near-identical 'last' is high.
        assert rep["behaviours"][b]["cos_vs_primary"]["mean"] == pytest.approx(1.0, abs=1e-6)
        assert rep["behaviours"][b]["cos_vs_primary"]["last"] > 0.9
        assert rep["behaviours"][b]["d_eff_70"]["mean"] is not None
