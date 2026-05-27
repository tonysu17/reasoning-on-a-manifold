"""Unit tests for src/cbs/ablation.py (M5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.cbs import ablation as A


def test_ablation_module_importable() -> None:
    from src.cbs import ablation  # noqa: F401


# ── build_v_cbs ──────────────────────────────────────────────────────────

def test_build_v_cbs_unit_norm() -> None:
    rng = np.random.default_rng(0)
    t3 = rng.normal(loc=+1.0, scale=0.5, size=(50, 32))
    t1 = rng.normal(loc=-1.0, scale=0.5, size=(50, 32))
    v = A.build_v_cbs(t3, t1)
    assert v.shape == (32,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_build_v_cbs_direction_matches_mean_difference() -> None:
    t3 = np.array([[2.0, 0.0]])
    t1 = np.array([[0.0, 0.0]])
    v = A.build_v_cbs(t3, t1)
    assert np.allclose(v, np.array([1.0, 0.0]))


def test_build_v_cbs_raises_when_means_coincide() -> None:
    X = np.zeros((10, 5))
    with pytest.raises(ValueError):
        A.build_v_cbs(X, X)


def test_build_v_cbs_raises_on_empty() -> None:
    with pytest.raises(ValueError):
        A.build_v_cbs(np.zeros((0, 5)), np.zeros((5, 5)))


# ── validate_v_cbs ───────────────────────────────────────────────────────

def test_validate_v_cbs_passes_well_separated_data() -> None:
    rng = np.random.default_rng(1)
    d = 32
    t3 = rng.normal(loc=+1.0, scale=0.5, size=(60, d))
    t1 = rng.normal(loc=-1.0, scale=0.5, size=(60, d))
    v = A.build_v_cbs(t3, t1)
    # Adding-knowledge centroid orthogonal to v_cbs.
    centroid = np.zeros(d)
    centroid[0] = 1.0      # arbitrary unit direction
    # The probe will get high accuracy because the data is well-separated;
    # cosine to "centroid" will be small if t3/t1 means lie along a
    # different axis. Construct t3, t1 along direction [0, 1, 0, ..., 0]
    # for clean orthogonality.
    t3 = np.zeros((60, d)); t3[:, 1] = rng.normal(loc=+2.0, size=60)
    t1 = np.zeros((60, d)); t1[:, 1] = rng.normal(loc=-2.0, size=60)
    # add small noise
    t3 += rng.normal(scale=0.1, size=t3.shape)
    t1 += rng.normal(scale=0.1, size=t1.shape)
    v = A.build_v_cbs(t3, t1)
    out = A.validate_v_cbs(v, centroid, t3, t1, cv_folds=5, seed=0)
    assert out["passes"], out
    assert out["cv_probe_accuracy_mean"] >= 0.7
    assert out["cv_probe_accuracy_std"] <= 0.15
    assert abs(out["cosine_sim_with_knowledge_centroid"]) < 0.5


def test_validate_v_cbs_failstop_cosine() -> None:
    """If v_cbs is parallel to the adding-knowledge centroid, |cos|=1 -> fail."""
    d = 16
    v = np.zeros(d); v[0] = 1.0
    centroid = v.copy()
    rng = np.random.default_rng(2)
    t3 = rng.normal(size=(50, d))
    t1 = rng.normal(size=(50, d))
    out = A.validate_v_cbs(v, centroid, t3, t1, cv_folds=5)
    assert out["passes"] is False
    # At least the cosine failure should appear.
    assert any("cos" in f for f in out["failures"])


def test_validate_v_cbs_failstop_low_probe_accuracy() -> None:
    """When tier-3 and tier-1 are indistinguishable, probe accuracy ~ 0.5
    -> fail."""
    d = 16
    rng = np.random.default_rng(3)
    # same distribution for both tiers
    t3 = rng.normal(size=(60, d))
    t1 = rng.normal(size=(60, d))
    # v_cbs is the small mean-diff direction (mostly noise)
    v = A.build_v_cbs(t3, t1)
    # centroid orthogonal-ish
    centroid = np.zeros(d); centroid[-1] = 1.0
    out = A.validate_v_cbs(v, centroid, t3, t1, cv_folds=5, seed=0)
    assert out["passes"] is False
    assert any("cv_probe_accuracy_mean" in f for f in out["failures"])


# ── CBSAblationModel ─────────────────────────────────────────────────────

def test_cbs_ablation_model_requires_unit_norm() -> None:
    """Constructor rejects non-unit vectors. Model/tokenizer are
    None placeholders — the constructor only checks the vector."""
    v_bad = np.array([2.0, 0.0, 0.0])     # norm 2.0
    with pytest.raises(ValueError):
        A.CBSAblationModel(model=_FakeModel(), tokenizer=None,
                           v_cbs=v_bad, layer=27)


class _FakeModel:
    """Minimal stand-in: SteeredModel.__init__ touches model.parameters() so
    we expose a no-op iterator producing tensors with a device + dtype."""

    def parameters(self):
        import torch
        t = torch.zeros(1)
        return iter([t])


def test_cbs_ablation_model_accepts_unit_vector() -> None:
    import torch
    v = np.zeros(8, dtype=np.float32); v[0] = 1.0
    m = A.CBSAblationModel(model=_FakeModel(), tokenizer=None,
                            v_cbs=v, layer=27, alpha=1.0)
    assert m.layer == 27
    assert m.alpha == 1.0
    assert m.mode == "subtract"


# ── construct_task_sets ──────────────────────────────────────────────────

def _toy_correct(task_id: str, tier3_count: int) -> dict:
    return {
        "task_id": task_id,
        "answer_correct": True,
        "annotations": [{"label": "adding-knowledge",
                          "text": f"sentence {i}",
                          "cbs_tier": 3 if i < tier3_count else 1}
                         for i in range(5)],
    }


def _toy_incorrect(task_id: str) -> dict:
    return {
        "task_id": task_id, "answer_correct": False,
        "annotations": [{"label": "adding-knowledge", "text": "x",
                          "cbs_tier": 3}],
    }


def test_construct_task_sets_size_floor() -> None:
    # Only 10 textbook + 10 bridge candidates; floor=50 should raise.
    chains = (
        [_toy_correct(f"A{i}", 0) for i in range(10)]
        + [_toy_correct(f"B{i}", 2) for i in range(10)]
        + [_toy_incorrect(f"C{i}") for i in range(20)]
    )
    with pytest.raises(RuntimeError) as exc:
        A.construct_task_sets(chains, target_per_set=100, floor=50)
    assert "Insufficient" in str(exc.value)


def test_construct_task_sets_returns_both_sets_when_enough() -> None:
    chains = (
        [_toy_correct(f"A{i}", 0) for i in range(60)]
        + [_toy_correct(f"B{i}", 1) for i in range(60)]
        + [_toy_incorrect(f"C{i}") for i in range(20)]
    )
    out = A.construct_task_sets(chains, target_per_set=50, floor=50)
    assert len(out["set_a_textbook"]) == 50
    assert len(out["set_b_bridge"]) == 50
    assert out["n_set_a"] == 60
    assert out["n_set_b"] == 60


# ── selectivity_ratio ────────────────────────────────────────────────────

def test_selectivity_ratio_basic() -> None:
    assert A.selectivity_ratio(0.4, 0.1) == pytest.approx(4.0)
    assert A.selectivity_ratio(0.0, 0.5) == pytest.approx(0.0)


def test_selectivity_ratio_zero_denominator_returns_nan() -> None:
    assert np.isnan(A.selectivity_ratio(0.4, 0.0))
