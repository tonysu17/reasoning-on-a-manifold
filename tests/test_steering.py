"""Regression tests for src/steering.py.

Covers the difference-of-means construction, the manifold-projection geometry,
determinism (the randomized-SVD reproducibility bug), and the projection
idempotence/containment properties that make "manifold-projected" meaningful.
"""

import json

import numpy as np
import pytest

from src.steering import (
    single_direction_vector,
    manifold_projected_vector,
    auto_k,
    build_steering_vectors,
    save_steering_vectors,
    load_steering_vectors,
)
from tests.synthetic import flat_subspace


def test_single_direction_points_from_off_to_on():
    """The vector is the unit-normalised difference of class means."""
    rng = np.random.default_rng(0)
    on = rng.standard_normal((100, 1536)) + 5.0   # shifted +5 on every dim
    off = rng.standard_normal((100, 1536))
    v = single_direction_vector(on, off)
    assert np.isclose(np.linalg.norm(v), 1.0)
    # mean(on) - mean(off) ~ +5 on every dim, so v should be ~ all-positive.
    assert (v > 0).mean() > 0.95


def test_single_direction_zero_when_means_equal():
    X = flat_subspace(50, 10, seed=1)
    v = single_direction_vector(X, X)  # identical on/off
    assert np.linalg.norm(v) < 1e-6


def test_manifold_vector_is_unit_norm():
    on = flat_subspace(120, 20, seed=2)
    off = flat_subspace(120, 20, seed=3)
    v = manifold_projected_vector(on, off, k=5)
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_manifold_projection_is_deterministic():
    """The randomized-SVD bug made this non-reproducible run-to-run."""
    on = flat_subspace(145, 30, seed=4)
    off = flat_subspace(145, 30, seed=5)
    v1 = manifold_projected_vector(on, off, k=5)
    v2 = manifold_projected_vector(on, off, k=5)
    assert np.allclose(v1, v2)


def test_auto_k_deterministic_and_in_range():
    on = flat_subspace(145, 30, seed=6)
    k1, k2 = auto_k(on), auto_k(on)
    assert k1 == k2
    assert 1 <= k1 <= min(on.shape[0] - 1, on.shape[1], 100)


def test_full_rank_projection_recovers_single_direction():
    """Projecting onto enough components (>= data rank) should leave the
    single-direction vector essentially unchanged (projection is into the span
    that already contains it)."""
    on = flat_subspace(60, 10, noise=0.0, seed=7)   # rank ~10
    off = flat_subspace(60, 10, noise=0.0, seed=8)
    r_single = single_direction_vector(on, off)
    r_proj = manifold_projected_vector(on, off, k=min(on.shape[0] - 1, 59))
    # cosine similarity should be high once k spans the on-subspace containing the mean shift
    cos = abs(float(r_single @ r_proj))
    assert cos > 0.5


def test_auto_k_higher_threshold_needs_more_components():
    on = flat_subspace(200, 25, noise=0.3, seed=9)
    assert auto_k(on, variance_threshold=0.5) <= auto_k(on, variance_threshold=0.95)


def test_save_provenance_written_and_load_skips_it(tmp_path):
    """Provenance is recorded in metadata.json but must not be loaded as a behaviour."""
    import json
    from src.steering import save_steering_vectors, load_steering_vectors
    vecs = {"backtracking": {"layer": 27, "single_direction": np.ones(4),
                             "manifold_projected": {1: np.ones(4)},
                             "n_on": 10, "n_off": 20, "auto_k": 1}}
    save_steering_vectors(vecs, tmp_path, provenance={"git_commit": "abc123", "seed": 42})
    meta = json.load(open(tmp_path / "metadata.json"))
    assert meta["_provenance"]["git_commit"] == "abc123"
    loaded = load_steering_vectors(tmp_path)
    assert set(loaded) == {"backtracking"}  # _provenance skipped, not a behaviour


# ── Phase-7 pre-flight: degenerate / silent-NaN guards ────────────────────────

def test_single_direction_empty_input_fails_loud():
    """Empty ON or OFF must raise, not silently emit a NaN 'unit' vector.

    mean(axis=0) of an empty matrix is all-NaN; `NaN < 1e-10` is False, so the
    old near-zero guard let a NaN vector through to steering.
    """
    off = flat_subspace(20, 5, seed=0)
    with pytest.raises(ValueError):
        single_direction_vector(np.empty((0, 1536)), off)
    with pytest.raises(ValueError):
        single_direction_vector(off, np.empty((0, off.shape[1])))


def test_single_direction_zero_returns_exact_zero_not_tiny_vector():
    """When means coincide the result is the exact zero vector (a no-op for
    Phase-7 steering), never a tiny non-unit vector masquerading as a direction."""
    X = flat_subspace(50, 10, seed=1)
    v = single_direction_vector(X, X)
    assert np.all(v == 0.0)
    assert np.isfinite(v).all()


def test_manifold_vector_never_nan():
    """Projected vector must be finite for ordinary inputs (no NaN leakage)."""
    on = flat_subspace(120, 20, seed=2)
    off = flat_subspace(120, 20, seed=3)
    for k in (1, 3, 5, 10):
        v = manifold_projected_vector(on, off, k=k)
        assert np.isfinite(v).all()
        assert np.isclose(np.linalg.norm(v), 1.0)


def test_manifold_k_is_clamped_above_available_components():
    """k far larger than min(N-1, dim) must clamp, not crash, and stay unit-norm."""
    on = flat_subspace(20, 5, seed=4)   # max usable comps = min(19, 1536) = 19
    off = flat_subspace(20, 5, seed=5)
    v = manifold_projected_vector(on, off, k=1000)
    assert v.shape == (1536,)
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_manifold_k1_is_unit_norm_and_finite():
    on = flat_subspace(120, 20, seed=6)
    off = flat_subspace(120, 20, seed=7)
    v = manifold_projected_vector(on, off, k=1)
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert np.isfinite(v).all()


def test_manifold_too_few_on_rows_falls_back_to_single():
    """ON with <2 rows cannot fit PCA; fall back to the single-direction vector."""
    on = flat_subspace(1, 5, seed=8)
    off = flat_subspace(20, 5, seed=9)
    v = manifold_projected_vector(on, off, k=3)
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert np.allclose(v, single_direction_vector(on, off))


def test_manifold_projection_equals_VVt_r():
    """The projection is the orthogonal projector onto the top-k PCA subspace:
    r_proj ∝ (Vᵀ V) r. Verify against an independent PCA fit (METHODOLOGY §3)."""
    from sklearn.decomposition import PCA
    on = flat_subspace(120, 8, seed=10)
    off = flat_subspace(120, 8, seed=11)
    r = single_direction_vector(on, off)
    k = 4
    pca = PCA(n_components=k, svd_solver="full").fit(on)
    V = pca.components_
    expected = (V.T @ V) @ r
    expected = expected / np.linalg.norm(expected)
    got = manifold_projected_vector(on, off, k=k)
    # both unit-norm; sign is fixed because we renormalise the same projected vector
    assert np.allclose(got, expected, atol=1e-6)
    # residual orthogonal to the subspace (defining property of a projection)
    resid = r - (V.T @ V) @ r
    assert np.abs(V @ resid).max() < 1e-6


def test_auto_k_unreachable_threshold_returns_cap():
    """A threshold that cumulative variance can never reach returns max_k, not 0
    or an out-of-range index."""
    on = flat_subspace(50, 10, seed=12)
    max_k = min(on.shape[0] - 1, on.shape[1], 100)
    assert auto_k(on, variance_threshold=1.5) == max_k


def test_auto_k_degenerate_threshold_is_at_least_one():
    on = flat_subspace(50, 10, seed=13)
    assert auto_k(on, variance_threshold=0.0) == 1
    assert auto_k(on, variance_threshold=-0.5) == 1
    assert auto_k(np.empty((0, 5))) == 1   # no rows


# ── build_steering_vectors: OFF concat, hold-out drop, fail-loud ──────────────

def _fake_activation_dir(tmp_path, exclude_chain="EXCL", n_excl_rows=10, seed=0):
    """Write 4 behaviour matrices + a row_index.json sidecar; first
    `n_excl_rows` rows of each carry chain id `exclude_chain`."""
    from src.annotation import TARGET_BEHAVIOURS
    rng = np.random.default_rng(seed)
    rows = {}
    for i, b in enumerate(TARGET_BEHAVIOURS):
        N = 30 + i * 5
        X = (rng.standard_normal((N, 1536)).astype(np.float32) + i)  # separable means
        np.save(tmp_path / f"{b}_layer27.npy", X)
        rows[b] = [{"chain_id": (exclude_chain if r < n_excl_rows else f"keep{i}_{r}")}
                   for r in range(N)]
    json.dump({"sentence_matching": "exact", "rows": rows},
              open(tmp_path / "row_index.json", "w"))
    return TARGET_BEHAVIOURS


def test_build_off_is_concat_of_other_behaviours(tmp_path):
    behs = _fake_activation_dir(tmp_path)
    res = build_steering_vectors(tmp_path, layer=27)
    # n_off for each behaviour = sum of the OTHER behaviours' row counts
    totals = {b: res[b]["n_on"] for b in behs}
    grand = sum(totals.values())
    for b in behs:
        assert res[b]["n_off"] == grand - totals[b]
        assert res[b]["single_direction"].shape == (1536,)
        assert set(res[b]["manifold_projected"]) == {1, 3, 5, 10, "auto"}


def test_build_holdout_actually_drops_rows(tmp_path):
    behs = _fake_activation_dir(tmp_path, n_excl_rows=10)
    full = build_steering_vectors(tmp_path, layer=27)
    held = build_steering_vectors(tmp_path, layer=27, exclude_chain_ids={"EXCL"})
    for b in behs:
        assert held[b]["n_excluded"] == 10
        assert held[b]["n_on"] == full[b]["n_on"] - 10


def test_build_holdout_mismatch_fails_loud(tmp_path):
    """A sidecar whose row count disagrees with the matrices must raise, never
    silently skip the hold-out (the keystone out-of-sample guarantee)."""
    behs = _fake_activation_dir(tmp_path)
    # corrupt the sidecar: 3 ids per behaviour vs 30+ rows in the matrices
    json.dump({"sentence_matching": "exact",
               "rows": {b: [{"chain_id": "x"}] * 3 for b in behs}},
              open(tmp_path / "row_index.json", "w"))
    with pytest.raises(RuntimeError):
        build_steering_vectors(tmp_path, layer=27, exclude_chain_ids={"EXCL"})


def test_build_missing_activation_file_is_skipped(tmp_path):
    behs = _fake_activation_dir(tmp_path)
    (tmp_path / f"{behs[0]}_layer27.npy").unlink()  # drop one behaviour's file
    res = build_steering_vectors(tmp_path, layer=27)
    assert behs[0] not in res                # missing file → skipped, no crash
    assert set(res) == set(behs[1:])


# ── save/load roundtrip incl. the k="auto" key as dict-key AND filename ───────

def test_save_load_roundtrip_with_auto_key(tmp_path):
    rng = np.random.default_rng(0)
    def unit(d=4):
        v = rng.standard_normal(d); return v / np.linalg.norm(v)
    vecs = {"backtracking": {
        "layer": 27,
        "single_direction": unit(),
        "manifold_projected": {1: unit(), 3: unit(), "auto": unit()},
        "n_on": 100, "n_off": 200, "auto_k": 7, "n_excluded": 5,
    }}
    save_steering_vectors(vecs, tmp_path)
    # the "auto" arm is written as ..._manifold_kauto.npy
    assert (tmp_path / "backtracking_manifold_kauto.npy").exists()
    assert (tmp_path / "backtracking_manifold_k1.npy").exists()
    loaded = load_steering_vectors(tmp_path)["backtracking"]
    assert loaded["auto_k"] == 7
    assert loaded["n_excluded"] == 5
    # int keys come back as ints, the "auto" key as the string "auto"
    keys = set(loaded["manifold_projected"])
    assert keys == {1, 3, "auto"}
    assert np.allclose(loaded["manifold_projected"]["auto"], vecs["backtracking"]["manifold_projected"]["auto"])
    assert np.allclose(loaded["single_direction"], vecs["backtracking"]["single_direction"])


def test_save_backs_up_existing_metadata(tmp_path):
    """A re-run must back up the prior metadata.json so a shorter run can't
    silently clobber a good artifact."""
    base = {"layer": 27, "single_direction": np.ones(4),
            "manifold_projected": {1: np.ones(4)},
            "n_on": 1, "n_off": 1, "auto_k": 1}
    save_steering_vectors({"backtracking": dict(base)}, tmp_path)
    save_steering_vectors({"uncertainty-estimation": dict(base)}, tmp_path)
    assert (tmp_path / "metadata.json.bak").exists()
    bak = json.load(open(tmp_path / "metadata.json.bak"))
    assert "backtracking" in bak  # the first write survives in the backup
