"""Known-answer tests for the spillover gating nulls (src/safety_posttrain/nulls.py).

Synthetic ground truth:
- two samples from the SAME anisotropic Gaussian -> excess angle ~ 0, p large;
- post rotated so its top-k subspace genuinely moves -> excess > 0, p small;
- post = base + fixed translation -> paired coherence ~ 1, beats shuffled null;
- parity_check catches row-order and token_start divergence.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.safety_posttrain import nulls as N

D = 64
SEED = 7


def _aniso(n: int, rng: np.random.Generator, rot: np.ndarray | None = None) -> np.ndarray:
    """Anisotropic Gaussian with a well-defined top-5 subspace."""
    scales = np.concatenate([np.array([10, 8, 6, 5, 4], dtype=float),
                             0.5 * np.ones(D - 5)])
    X = rng.standard_normal((n, D)) * scales
    return X @ rot.T if rot is not None else X


def _plane_rotation(theta_deg: float) -> np.ndarray:
    """Rotate dims (0, 5): moves signal axis 0 toward a noise axis."""
    R = np.eye(D)
    t = np.radians(theta_deg)
    R[0, 0], R[0, 5] = np.cos(t), -np.sin(t)
    R[5, 0], R[5, 5] = np.sin(t), np.cos(t)
    return R


def test_null_case_no_excess():
    rng = np.random.default_rng(SEED)
    Xb, Xp = _aniso(400, rng), _aniso(400, rng)
    rep = N.gated_angle_report(Xb, Xp, k=5, m=200, n_perm=200, n_rep=5, seed=SEED)
    assert rep["p_within"] > 0.01
    assert abs(rep["excess_deg"]) < 5.0
    # split-half floors should be of the same order as the null
    assert rep["split_half_base_deg"] < rep["null_mean_deg"] + 10


def test_rotated_subspace_detected():
    rng = np.random.default_rng(SEED)
    Xb = _aniso(400, rng)
    Xp = _aniso(400, rng, rot=_plane_rotation(45.0))
    rep = N.gated_angle_report(Xb, Xp, k=5, m=200, n_perm=200, n_rep=5, seed=SEED)
    assert rep["p_within"] <= 0.05
    assert rep["excess_deg"] > 3.0


def test_paired_translation_coherent():
    rng = np.random.default_rng(SEED)
    Xb = _aniso(300, rng)
    v = np.zeros(D)
    v[3] = 25.0
    Xp = Xb + v + 0.1 * rng.standard_normal(Xb.shape)
    rep = N.paired_displacement(Xb, Xp, n_perm=200, seed=SEED)
    assert rep["coherence"] > 0.95
    assert rep["coherence_p_perm"] <= 0.05


def test_paired_noise_incoherent():
    rng = np.random.default_rng(SEED)
    Xb = _aniso(300, rng)
    Xp = Xb + rng.standard_normal(Xb.shape)      # pure isotropic per-row noise
    rep = N.paired_displacement(Xb, Xp, n_perm=200, seed=SEED)
    assert rep["coherence"] < 0.5
    assert rep["coherence_p_perm"] > 0.01


def _write_row_index(d, rows):
    d.mkdir(parents=True, exist_ok=True)
    (d / "row_index.json").write_text(json.dumps({"version": 1, "rows": rows}))


def test_parity_ok_and_token_divergence(tmp_path):
    rows = {"backtracking": [
        {"chain_id": "C1", "annotation_index": 0, "char_offset": 10, "token_start": 5},
        {"chain_id": "C2", "annotation_index": 3, "char_offset": 99, "token_start": 40},
    ]}
    a, b = tmp_path / "a", tmp_path / "b"
    _write_row_index(a, rows)
    _write_row_index(b, rows)
    assert N.parity_check(a, b, ["backtracking"])["ok"]

    shifted = {"backtracking": [dict(r, token_start=r["token_start"] + 1)
                                for r in rows["backtracking"]]}
    _write_row_index(b, shifted)
    rep = N.parity_check(a, b, ["backtracking"])
    assert not rep["ok"]
    assert "token_start" in rep["behaviours"]["backtracking"]["first_mismatch"]


def test_parity_row_mismatch(tmp_path):
    rows = {"backtracking": [
        {"chain_id": "C1", "annotation_index": 0, "char_offset": 10, "token_start": 5}]}
    other = {"backtracking": [
        {"chain_id": "C9", "annotation_index": 0, "char_offset": 10, "token_start": 5}]}
    a, b = tmp_path / "a", tmp_path / "b"
    _write_row_index(a, rows)
    _write_row_index(b, other)
    rep = N.parity_check(a, b, ["backtracking"])
    assert not rep["ok"]
    assert rep["behaviours"]["backtracking"]["row_identity_ok"] is False
