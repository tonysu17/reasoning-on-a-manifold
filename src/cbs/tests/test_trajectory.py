"""Unit tests for src/cbs/trajectory.py (M3)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.cbs import trajectory as T
from src.cbs.schemas import ChainTrajectory
from src.cbs.cohort import is_truncated


def test_trajectory_module_importable() -> None:
    from src.cbs import trajectory  # noqa: F401


# ── Cohort helper ────────────────────────────────────────────────────────

def test_is_truncated_basic() -> None:
    truncated = {"n_tokens": 8192, "chain": "thinking thinking thinking"}
    clean = {"n_tokens": 1024, "chain": "thinking</think>"}
    boundary_at_max_but_closed = {"n_tokens": 8192, "chain": "ok</think>"}
    short_unclosed = {"n_tokens": 512, "chain": "unfinished"}
    assert is_truncated(truncated)
    assert not is_truncated(clean)
    assert not is_truncated(boundary_at_max_but_closed)
    assert not is_truncated(short_unclosed)


# ── arc_length_sequence ──────────────────────────────────────────────────

def test_arc_length_zero_at_start() -> None:
    X = np.array([[0.0], [1.0], [3.0]])
    traj = ChainTrajectory(chain_id="t", layer=0, X=X)
    s = T.arc_length_sequence(traj)
    assert s.shape == (3,)
    assert s[0] == 0.0
    assert s[1] == pytest.approx(1.0)
    assert s[2] == pytest.approx(3.0)


def test_arc_length_empty_trajectory() -> None:
    traj = ChainTrajectory(chain_id="t", layer=0, X=np.zeros((0, 5)))
    s = T.arc_length_sequence(traj)
    assert s.shape == (0,)


def test_total_arc_length_straight_line_matches_euclidean() -> None:
    X = np.zeros((10, 3))
    X[:, 0] = np.arange(10)
    traj = ChainTrajectory(chain_id="line", layer=0, X=X)
    direct = float(np.linalg.norm(X[-1] - X[0]))
    total = T.total_arc_length(traj)
    assert total == pytest.approx(direct)


# ── curvature_sequence — straight line ───────────────────────────────────

def test_straight_line_zero_curvature() -> None:
    # Straight line in R^5 along x-axis.
    X = np.zeros((20, 5), dtype=np.float32)
    X[:, 0] = np.arange(20, dtype=np.float32)
    traj = ChainTrajectory(chain_id="line", layer=0, X=X)
    kappa = T.curvature_sequence(traj)
    assert np.isnan(kappa[0])
    assert np.isnan(kappa[-1])
    interior = kappa[1:-1]
    assert np.all(np.abs(interior) < 1e-6)


# ── curvature_sequence — synthetic helix ─────────────────────────────────

def test_synthetic_helix_constant_curvature() -> None:
    """Helix r(t)=(R cos t, R sin t, c t); analytical curvature kappa = R / (R^2 + c^2).

    The discrete arc-length-reparameterised formula in `curvature_sequence`
    should recover this within 5% for a moderately-sampled helix embedded
    in high-dim space.
    """
    R, c = 1.0, 0.3
    T_pts = 200
    t = np.linspace(0.0, 6.0 * np.pi, T_pts)
    helix = np.stack([R * np.cos(t), R * np.sin(t), c * t], axis=1)
    # Embed in R^1536 by zero-padding.
    d = 1536
    X = np.zeros((T_pts, d), dtype=np.float32)
    X[:, :3] = helix
    traj = ChainTrajectory(chain_id="helix", layer=0, X=X)
    kappa = T.curvature_sequence(traj)
    interior = kappa[1:-1]
    expected = R / (R * R + c * c)        # 0.9174
    mean_kappa = float(np.mean(interior))
    rel_err = abs(mean_kappa - expected) / expected
    assert rel_err < 0.05, (
        f"mean kappa = {mean_kappa:.4f}, expected {expected:.4f} "
        f"(rel err {rel_err:.3f})"
    )
    # Std across interior should also be small — constant-curvature property.
    assert float(np.std(interior)) < 0.05


# ── curvature_sequence — degenerate cases ────────────────────────────────

def test_degenerate_short_trajectory() -> None:
    # T = 2 is too short for interior curvature
    traj = ChainTrajectory(chain_id="t2", layer=0, X=np.zeros((2, 5)))
    kappa = T.curvature_sequence(traj)
    assert kappa.shape == (2,)
    assert np.all(np.isnan(kappa))

    # T = 1
    traj = ChainTrajectory(chain_id="t1", layer=0, X=np.zeros((1, 5)))
    kappa = T.curvature_sequence(traj)
    assert kappa.shape == (1,)
    assert np.all(np.isnan(kappa))


def test_duplicate_consecutive_points_yield_nan_curvature() -> None:
    X = np.zeros((5, 3))
    X[1] = [1, 0, 0]
    X[2] = [1, 0, 0]                     # duplicate
    X[3] = [1, 1, 0]
    X[4] = [2, 1, 0]
    traj = ChainTrajectory(chain_id="dup", layer=0, X=X)
    kappa = T.curvature_sequence(traj)
    assert np.isnan(kappa[2])            # zero step on either side -> NaN
    # Other interior points still get valid values.
    assert not np.isnan(kappa[3])


# ── subspace dynamics ───────────────────────────────────────────────────

def test_subspace_visit_argmax() -> None:
    # Two orthogonal subspaces in R^4. Point in span(a) -> visit "a".
    Va = np.eye(4)[:, :2]
    Vb = np.eye(4)[:, 2:]
    X = np.array([
        [1, 0, 0, 0],                    # in span(a)
        [0, 1, 0, 0],                    # in span(a)
        [0, 0, 1, 0],                    # in span(b)
        [0, 0, 0, 1],                    # in span(b)
    ], dtype=float)
    traj = ChainTrajectory(chain_id="t", layer=0, X=X)
    visits = T.subspace_visit_sequence(traj, {"a": Va, "b": Vb})
    assert visits == ["a", "a", "b", "b"]


def test_cross_subspace_returns_metrics() -> None:
    Va = np.eye(4)[:, :2]
    Vb = np.eye(4)[:, 2:]
    X = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
    ], dtype=float)
    traj = ChainTrajectory(chain_id="t", layer=0, X=X)
    out = T.cross_subspace_returns(traj, {"a": Va, "b": Vb})
    assert out["visit_sequence"] == ["a", "b", "a", "b", "b"]
    # 3 inter-state transitions (a->b, b->a, a->b)
    assert out["n_transitions"] == 3
    # Return rate over compressed [a, b, a, b]: 2 revisits out of 4 visits.
    assert out["return_rate"] == pytest.approx(2 / 4)


# ── trajectory_cone_angle ────────────────────────────────────────────────

def test_cone_angle_zero_for_aligned_points() -> None:
    X = np.array([
        [1, 0, 0],
        [2, 0, 0],
        [5, 0, 0],
    ], dtype=float)
    traj = ChainTrajectory(chain_id="t", layer=0, X=X)
    assert T.trajectory_cone_angle(traj) == pytest.approx(0.0, abs=1e-9)


def test_cone_angle_orthogonal_directions() -> None:
    # Two orthogonal directions -> max angle 90°.
    X = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
    ], dtype=float)
    traj = ChainTrajectory(chain_id="t", layer=0, X=X)
    angle = T.trajectory_cone_angle(traj)
    assert angle > 0.3 and angle < math.pi


# ── build_row_index ─────────────────────────────────────────────────────

def test_build_row_index_orders_by_chain_then_span() -> None:
    chains = [
        {"task_id": "T1", "annotations": [
            {"label": "deduction", "text": "skip"},
            {"label": "adding-knowledge", "text": "row 0"},
            {"label": "backtracking", "text": "row 0 bt"},
        ]},
        {"task_id": "T2", "annotations": [
            {"label": "adding-knowledge", "text": "row 1"},
            {"label": "backtracking", "text": "row 1 bt"},
        ]},
    ]
    idx = T.build_row_index(chains, target_behaviours=("adding-knowledge",
                                                         "backtracking"))
    assert idx["adding-knowledge"] == [("T1", 1), ("T2", 0)]
    assert idx["backtracking"] == [("T1", 2), ("T2", 1)]


# ── build_trajectory ────────────────────────────────────────────────────

def test_build_trajectory_uses_lookup() -> None:
    chain = {"task_id": "C1", "annotations": [
        {"label": "initializing", "text": "skip me"},
        {"label": "adding-knowledge", "text": "AK1"},
        {"label": "deduction", "text": "skip me too"},
        {"label": "backtracking", "text": "BT1"},
    ], "n_tokens": 100, "chain": "ok</think>"}
    activations = {
        "adding-knowledge": np.array([[1.0, 0.0, 0.0]]),
        "backtracking":     np.array([[0.0, 1.0, 0.0]]),
    }
    row_index = T.build_row_index([chain],
                                  target_behaviours=("adding-knowledge",
                                                       "backtracking"))
    traj = T.build_trajectory(chain, activations_dir=".", layer=0,
                              row_index=row_index, activations=activations,
                              target_behaviours=("adding-knowledge",
                                                  "backtracking"))
    assert traj.chain_id == "C1"
    assert traj.T == 2
    assert traj.behaviours == ["adding-knowledge", "backtracking"]
    assert traj.sentence_ids == ["C1:1", "C1:3"]
    # CBS fields default to 0 / None when absent on the span.
    assert traj.cbs_tiers == [0, 0]
    assert traj.cross_domain == [None, None]
    assert traj.truncated is False


def test_build_trajectory_truncated_flag_propagates() -> None:
    chain = {"task_id": "C2", "annotations": [
        {"label": "adding-knowledge", "text": "AK"},
    ], "n_tokens": 8192, "chain": "missing closing tag"}
    activations = {"adding-knowledge": np.array([[1.0, 0.0]])}
    row_index = T.build_row_index([chain],
                                  target_behaviours=("adding-knowledge",))
    traj = T.build_trajectory(chain, activations_dir=".", layer=0,
                              row_index=row_index, activations=activations,
                              target_behaviours=("adding-knowledge",))
    assert traj.truncated is True
