"""Tests for src/huang_manifold.py — the Huang-faithful POOLED-class PCA.

The keystone test (`test_between_class_axis_lands_in_top1_pooled_but_not_on_only`)
pins the actual fix: when the between-class mean-separation axis is a HIGH-variance
direction of the *pooled* cloud but a LOW-variance direction within ON alone, the
pooled construction recovers it in the top-1 component while the ON-only
construction does NOT. This is the whole point of pooling (Huang Eq. 5).

Other tests cover: pooled projector identity, projector idempotence/orthonormality,
hand-computable energy, unit-norm/finite/clamping/fallback guards, the robustness
primitives (principal angles on identical vs orthogonal subspaces, energy
monotonicity, Horn/eigengap sanity), build hold-out, and save/load roundtrip.

Stub-testable: pure numpy + sklearn, no GPU, no real activation files.
"""

import json

import numpy as np
import pytest
from sklearn.decomposition import PCA

from src.huang_manifold import (
    pooled_manifold_vector,
    pooled_auto_k,
    energy_in_subspace,
    build_huang_manifold_vectors,
    on_only_manifold_components,
    pca_basis,
    principal_angles,
    subspace_energy,
    split_half_stability,
    eigengap_report,
    horn_parallel_analysis,
    random_subspace_null,
    save_steering_vectors,
    load_steering_vectors,
)
from src.steering import single_direction_vector, manifold_projected_vector
from tests.synthetic import flat_subspace


# ── Helpers ────────────────────────────────────────────────────────────────--

def _two_clusters(sep_dir, sep, n=200, d=1536, within_scale=1.0,
                  within_dim=20, seed=0):
    """Two clusters separated by `sep` along unit `sep_dir`, with within-class
    Gaussian spread filling `within_dim` directions STRICTLY ORTHOGONAL to
    `sep_dir` (each within-direction unit-norm, scaled by `within_scale`).

    Building the within-class basis orthogonal to `sep_dir` makes the geometry
    exact: axis `sep_dir` carries ONLY the between-class separation (zero
    within-class variance), so the pooled top-1 PC is exactly `sep_dir` when the
    separation dominates, while the ON-only top PCs are exactly the within-class
    directions. `within_scale` controls per-direction within-class sd, so the
    separation-vs-within ratio is set by (sep, within_scale, within_dim).

    Returns (on, off) where mean(on)−mean(off) = sep · sep_dir exactly.
    """
    rng = np.random.default_rng(seed)
    sep_dir = sep_dir / np.linalg.norm(sep_dir)
    # Random basis, then project OUT sep_dir and orthonormalise → within-class
    # variance is strictly orthogonal to the separation axis.
    raw = rng.standard_normal((d, within_dim))
    raw = raw - np.outer(sep_dir, sep_dir @ raw)        # remove sep_dir component
    Q, _ = np.linalg.qr(raw)                            # (d, within_dim) orthonormal
    coords_on = within_scale * rng.standard_normal((n, within_dim))
    coords_off = within_scale * rng.standard_normal((n, within_dim))
    on = coords_on @ Q.T + 0.5 * sep * sep_dir
    off = coords_off @ Q.T - 0.5 * sep * sep_dir
    return on.astype(np.float32), off.astype(np.float32)


# ── KEYSTONE: the actual fix ──────────────────────────────────────────────────

def test_between_class_axis_lands_in_top1_pooled_but_not_on_only():
    """The between-class separation axis is a top-1 POOLED PC (so pooled
    cos(k1, single) is high) but NOT recovered by the ON-only fit.

    Construction: a LARGE class separation along a known axis, with MODERATE
    within-class spread filling other dims. Pooling adds a rank-1 term ∝ r·rᵀ
    along the separation axis → it dominates the pooled spectrum. ON-only
    centering removes the separation, leaving only within-ON variance.
    """
    d = 200
    sep_dir = np.zeros(d)
    sep_dir[0] = 1.0  # separation along axis 0
    # Big separation, small within-class spread in OTHER dims → axis 0 dominates
    # the pooled cloud but is absent from each class's own variance.
    on, off = _two_clusters(sep_dir, sep=40.0, n=300, d=d,
                            within_scale=1.0, within_dim=30, seed=1)

    r = single_direction_vector(on, off)
    pooled_k1 = pooled_manifold_vector(on, off, k=1)
    only_k1 = manifold_projected_vector(on, off, k=1)

    cos_pooled = abs(float(r @ pooled_k1))
    cos_only = abs(float(r @ only_k1))

    # Pooled top-1 essentially IS the separation axis → cos ≈ 1.
    assert cos_pooled > 0.9, f"pooled cos(k1,single)={cos_pooled}"
    # ON-only top-1 is a within-ON direction, not the separation axis → low cos.
    assert cos_only < 0.5, f"ON-only cos(k1,single)={cos_only}"
    # And the fix strictly improves the energy in the low-k subspace.
    assert cos_pooled > cos_only

    # The top-1 pooled component aligns with the known separation axis.
    U1 = pca_basis(np.concatenate([on, off]), 1)[0][:, 0]
    assert abs(float(U1 @ sep_dir)) > 0.9


def test_pooled_energy_higher_than_on_only_at_k1():
    """Energy of r in the top-1 subspace is higher under pooled than ON-only —
    the synthetic mirror of the real diagnostic's acceptance criterion."""
    d = 200
    sep_dir = np.zeros(d); sep_dir[0] = 1.0
    on, off = _two_clusters(sep_dir, sep=30.0, n=250, d=d,
                            within_scale=1.0, within_dim=25, seed=2)
    r = single_direction_vector(on, off)
    U_pooled = pca_basis(np.concatenate([on, off]), 1)[0]   # (d,1)
    U_only = on_only_manifold_components(on, 1).T            # (d,1)
    e_pooled = energy_in_subspace(r, U_pooled.T)
    e_only = energy_in_subspace(r, U_only.T)
    assert e_pooled > e_only
    assert e_pooled > 0.5


def test_pooled_differs_from_on_only_on_nontrivial_data():
    """Guard against accidentally reintroducing the ON-only fit: on data with
    different ON/OFF covariances the two constructions must differ."""
    rng = np.random.default_rng(3)
    d = 120
    on = (rng.standard_normal((150, 8)) @ rng.standard_normal((8, d))).astype(np.float32) + 3.0
    off = (rng.standard_normal((150, 8)) @ rng.standard_normal((8, d))).astype(np.float32)
    v_pooled = pooled_manifold_vector(on, off, k=2)
    v_only = manifold_projected_vector(on, off, k=2)
    assert not np.allclose(v_pooled, v_only)


# ── Pooled projector identity (mirrors test_manifold_projection_equals_VVt_r) ──

def test_pooled_projection_equals_VVt_r_on_pooled_fit():
    """pooled_manifold_vector == renorm((UᵀU) r) where U = top-k PCA of the
    POOLED set (this is the test that pins the fix — fit on vstack(on,off))."""
    on = flat_subspace(120, 8, seed=10).astype(np.float32) + 1.0
    off = flat_subspace(120, 8, seed=11).astype(np.float32)
    r = single_direction_vector(on, off)
    k = 4
    pca = PCA(n_components=k, svd_solver="full").fit(np.concatenate([on, off]))
    U = pca.components_
    expected = (U.T @ U) @ r
    expected = expected / np.linalg.norm(expected)
    got = pooled_manifold_vector(on, off, k=k)
    assert np.allclose(got, expected, atol=1e-6)
    # residual orthogonal to the subspace (defining property of a projection)
    resid = r - (U.T @ U) @ r
    assert np.abs(U @ resid).max() < 1e-6


def test_pooled_projector_idempotent_and_orthonormal():
    """U has orthonormal rows (UUᵀ = I_k) and P = UᵀU is an idempotent projector
    (P² = P, symmetric)."""
    on = flat_subspace(120, 10, seed=12).astype(np.float32) + 2.0
    off = flat_subspace(120, 10, seed=13).astype(np.float32)
    U = pca_basis(np.concatenate([on, off]), 5)[0]   # (d,5) orthonormal columns
    # columns orthonormal
    assert np.allclose(U.T @ U, np.eye(5), atol=1e-6)
    P = U @ U.T
    assert np.allclose(P @ P, P, atol=1e-6)          # idempotent
    assert np.allclose(P, P.T, atol=1e-6)            # symmetric


# ── Hand-computable energy ────────────────────────────────────────────────────

def test_energy_in_subspace_handcomputable():
    """For an axis-aligned orthonormal basis, energy = sum of squared coords on
    those axes / squared norm. r = [3,4,0] onto axis-0 only → 9/25 = 0.36."""
    r = np.array([3.0, 4.0, 0.0])
    U_axis0 = np.array([[1.0, 0.0, 0.0]])           # (1,3)
    assert np.isclose(energy_in_subspace(r, U_axis0), 9.0 / 25.0)
    U_axis01 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (2,3)
    assert np.isclose(energy_in_subspace(r, U_axis01), 1.0)  # fully captured
    U_axis2 = np.array([[0.0, 0.0, 1.0]])
    assert np.isclose(energy_in_subspace(r, U_axis2), 0.0)


def test_energy_full_rank_is_one():
    on = flat_subspace(60, 10, noise=0.0, seed=14).astype(np.float32) + 1.0
    off = flat_subspace(60, 10, noise=0.0, seed=15).astype(np.float32)
    r = single_direction_vector(on, off)
    pooled_n = on.shape[0] + off.shape[0]
    U = pca_basis(np.concatenate([on, off]), min(pooled_n - 1, 25))[0]
    assert energy_in_subspace(r, U.T) > 0.99


# ── Guards (ported from tests/test_steering.py) ───────────────────────────────

def test_pooled_vector_is_unit_norm_and_finite():
    on = flat_subspace(120, 20, seed=2).astype(np.float32) + 1.0
    off = flat_subspace(120, 20, seed=3).astype(np.float32)
    for k in (1, 3, 5, 10):
        v = pooled_manifold_vector(on, off, k=k)
        assert np.isfinite(v).all()
        assert np.isclose(np.linalg.norm(v), 1.0)


def test_pooled_deterministic():
    """full-SVD solver → reproducible run-to-run (the randomized-SVD bug)."""
    on = flat_subspace(145, 30, seed=4).astype(np.float32) + 1.0
    off = flat_subspace(145, 30, seed=5).astype(np.float32)
    v1 = pooled_manifold_vector(on, off, k=5)
    v2 = pooled_manifold_vector(on, off, k=5)
    assert np.allclose(v1, v2)


def test_pooled_k_is_clamped_above_available_components():
    on = flat_subspace(20, 5, seed=4).astype(np.float32)
    off = flat_subspace(20, 5, seed=5).astype(np.float32)
    v = pooled_manifold_vector(on, off, k=10_000)
    assert v.shape == (1536,)
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_pooled_too_few_rows_falls_back_to_single():
    """A pooled set with <2 rows cannot fit PCA → fall back to single direction."""
    on = flat_subspace(1, 5, seed=8).astype(np.float32)
    off = np.empty((0, 5), dtype=np.float32)
    # off empty → single_direction_vector raises (loud), as designed.
    with pytest.raises(ValueError):
        pooled_manifold_vector(on, off, k=3)
    # one row each → pooled_n=2, n_components clamps to 1, projection succeeds.
    off1 = flat_subspace(1, 5, seed=9).astype(np.float32)
    v = pooled_manifold_vector(on, off1, k=3)
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_pooled_empty_input_raises():
    off = flat_subspace(20, 5, seed=0).astype(np.float32)
    with pytest.raises(ValueError):
        pooled_manifold_vector(np.empty((0, 1536)), off, k=3)


def test_pooled_auto_k_in_range_and_monotone():
    on = flat_subspace(200, 25, noise=0.3, seed=9).astype(np.float32) + 1.0
    off = flat_subspace(200, 25, noise=0.3, seed=10).astype(np.float32)
    k_lo = pooled_auto_k(on, off, variance_threshold=0.5)
    k_hi = pooled_auto_k(on, off, variance_threshold=0.95)
    assert 1 <= k_lo <= k_hi
    assert k_hi <= min(on.shape[0] + off.shape[0] - 1, on.shape[1], 100)


# ── Robustness primitives ─────────────────────────────────────────────────────

def test_principal_angles_identical_subspace_is_zero():
    Q = pca_basis(flat_subspace(100, 10, seed=20).astype(np.float32), 5)[0]
    ang = principal_angles(Q, Q)
    assert ang.shape == (5,)
    assert np.all(ang < 1e-6)


def test_principal_angles_orthogonal_subspace_is_ninety_deg():
    """Two subspaces built from disjoint coordinate axes are orthogonal → all
    principal angles = 90°."""
    d = 50
    Q1 = np.zeros((d, 3)); Q1[0, 0] = Q1[1, 1] = Q1[2, 2] = 1.0
    Q2 = np.zeros((d, 3)); Q2[3, 0] = Q2[4, 1] = Q2[5, 2] = 1.0
    ang = np.degrees(principal_angles(Q1, Q2))
    assert np.allclose(ang, 90.0, atol=1e-6)


def test_principal_angles_ascending_and_bounded():
    Q1 = pca_basis(flat_subspace(120, 8, seed=21).astype(np.float32), 4)[0]
    Q2 = pca_basis(flat_subspace(120, 8, seed=22).astype(np.float32), 4)[0]
    ang = principal_angles(Q1, Q2)
    assert np.all(np.diff(ang) >= -1e-9)             # ascending
    assert np.all(ang >= 0) and np.all(ang <= np.pi / 2 + 1e-9)


def test_pca_basis_orthonormal_and_eigvals_descending():
    X = flat_subspace(150, 12, noise=0.2, seed=23).astype(np.float32)
    Q, ev = pca_basis(X, 6)
    assert np.allclose(Q.T @ Q, np.eye(6), atol=1e-6)
    assert np.all(np.diff(ev) <= 1e-9)               # descending


def test_subspace_energy_monotone_in_k_and_one_at_full_rank():
    X = flat_subspace(120, 10, noise=0.1, seed=24).astype(np.float32)
    R = flat_subspace(80, 10, noise=0.1, seed=25).astype(np.float32)
    # Energy of R in span(top-k of X) is non-decreasing in k (adding orthonormal
    # basis directions can only add captured variance) — holds for ANY R.
    energies = [subspace_energy(R, pca_basis(X, k)[0]) for k in range(1, 11)]
    assert all(b >= a - 1e-9 for a, b in zip(energies, energies[1:]))
    # X's OWN variance is fully captured by X's full-rank PCA basis (energy → 1).
    Q_full_X = pca_basis(X, min(X.shape[0] - 1, 30))[0]
    assert subspace_energy(X, Q_full_X) > 0.99


def test_random_subspace_null_beats_random_for_low_rank():
    """A genuine low-rank ON manifold beats random k-subspaces by a wide margin;
    random energy ≈ k/d."""
    rng = np.random.default_rng(26)
    d = 200
    on = (rng.standard_normal((400, 5)) @ rng.standard_normal((5, d))).astype(np.float32)
    off = (rng.standard_normal((400, 5)) @ rng.standard_normal((5, d))).astype(np.float32)
    res = random_subspace_null(on, off, k=10, n_rand=100)
    assert res["e_on"] > res["rand_on"]["p97_5"]
    assert res["fold_over_random"] > 5
    assert np.isclose(res["baseline_k_over_d"], 10 / d)
    # random mean is close to k/d
    assert abs(res["rand_on"]["mean"] - 10 / d) < 0.02


def test_random_subspace_null_subspace_X_controls_the_manifold():
    """``subspace_X`` selects which cloud defines the manifold Q — the battery
    fix that lets us test the POOLED subspace we steer with, not ON-only. ON and
    OFF live in disjoint 5-dim subspaces; building Q from ON concentrates ON
    energy, while building Q from OFF collapses ON energy and lifts OFF energy.
    Proves the knob genuinely swaps the subspace under test."""
    rng = np.random.default_rng(71)
    d = 200
    on = np.zeros((400, d), np.float32)
    off = np.zeros((400, d), np.float32)
    on[:, :5] = rng.standard_normal((400, 5))         # ON variance in dims 0–4
    off[:, 5:10] = rng.standard_normal((400, 5))      # OFF variance in dims 5–9
    default = random_subspace_null(on, off, k=5, n_rand=50)                 # Q from ON
    swapped = random_subspace_null(on, off, k=5, n_rand=50, subspace_X=off)  # Q from OFF
    assert default["e_on"] > 0.9 and default["e_off"] < 0.1
    assert swapped["e_on"] < 0.1 and swapped["e_off"] > 0.9


def test_eigengap_report_flat_subspace_has_gap_at_true_rank():
    """A noiseless rank-r flat subspace has a huge eigengap at k=r."""
    X = flat_subspace(300, 5, noise=0.01, seed=27).astype(np.float32)
    rep = eigengap_report(X, k=5, ks_scan=range(1, 12))
    # eigenvalue 5 >> eigenvalue 6 → large relative gap at k=5
    assert rep["rel_gap_k"] > 0.5
    assert rep["lambda_k"] > rep["lambda_k_plus_1"]


def test_split_half_stability_low_angle_for_real_structure():
    """Split-half of a stable low-rank subspace gives small principal angles and
    high overlap."""
    rng = np.random.default_rng(28)
    d = 120
    X = (rng.standard_normal((600, 6)) @ rng.standard_normal((6, d))
         + 0.05 * rng.standard_normal((600, d))).astype(np.float32)
    res = split_half_stability(X, k=6, n_boot=30)
    assert res["split"] == "row-level"
    assert res["mean_cos2"]["mean"] > 0.9
    assert res["max_angle_deg"]["p95"] < 25


def test_split_half_chain_aware_mode_selected_when_ids_given():
    rng = np.random.default_rng(29)
    d = 80
    X = (rng.standard_normal((200, 5)) @ rng.standard_normal((5, d))).astype(np.float32)
    cids = np.repeat(np.arange(40), 5)               # 40 chains, 5 rows each
    res = split_half_stability(X, k=5, chain_ids=cids, n_boot=20)
    assert res["split"] == "chain-aware"
    assert res["n_effective"] > 0


def test_horn_parallel_analysis_recovers_signal_components():
    """Horn keeps the leading run above the column-permuted null. A rank-3 signal
    embedded above weak isotropic noise → k_horn >= 3."""
    rng = np.random.default_rng(30)
    d = 60
    signal = rng.standard_normal((400, 3)) @ (5.0 * rng.standard_normal((3, d)))
    noise = rng.standard_normal((400, d))
    X = (signal + noise).astype(np.float32)
    res = horn_parallel_analysis(X, n_perm=20, k_huang=3, max_components=20)
    assert res["k_horn"] >= 3
    assert res["ratio_at_k_huang"] > 1.0


# ── build_huang_manifold_vectors: OFF concat, hold-out, per-layer dict ─────────

def _fake_activation_dir(tmp_path, layers, exclude_chain="EXCL",
                         n_excl_rows=10, seed=0):
    """Write 4 behaviour matrices (each at its own layer) + a row_index.json
    sidecar; first `n_excl_rows` rows of each carry chain id `exclude_chain`.
    Means are separable across behaviours so the diff-of-means is non-degenerate.
    """
    from src.annotation import TARGET_BEHAVIOURS
    rng = np.random.default_rng(seed)
    rows = {}
    for i, b in enumerate(TARGET_BEHAVIOURS):
        N = 40 + i * 5
        X = (rng.standard_normal((N, 1536)).astype(np.float32) + 3.0 * i)
        np.save(tmp_path / f"{b}_layer{layers[b]}.npy", X)
        rows[b] = [{"chain_id": (exclude_chain if r < n_excl_rows else f"keep{i}_{r}")}
                   for r in range(N)]
    json.dump({"sentence_matching": "exact", "rows": rows},
              open(tmp_path / "row_index.json", "w"))
    return list(TARGET_BEHAVIOURS)


def test_build_off_is_concat_of_other_behaviours_same_layer(tmp_path):
    """OFF for a behaviour = concat of the OTHER behaviours AT THE SAME LAYER.
    Use a single shared layer so all four matrices co-exist at one L."""
    from src.annotation import TARGET_BEHAVIOURS
    layers = {b: 27 for b in TARGET_BEHAVIOURS}
    behs = _fake_activation_dir(tmp_path, layers)
    res = build_huang_manifold_vectors(tmp_path, layers=layers)
    totals = {b: res[b]["n_on"] for b in behs}
    grand = sum(totals.values())
    for b in behs:
        assert res[b]["n_off"] == grand - totals[b]
        assert res[b]["single_direction"].shape == (1536,)
        assert set(res[b]["manifold_projected"]) == {1, 3, 5, 10, "auto"}
        assert res[b]["huang_k"] == 10
        assert "energy_in_manifold" in res[b]
        assert "cos_single_manifold" in res[b]


def test_build_per_behaviour_layers_grouping(tmp_path):
    """With DIFFERENT layers per behaviour, OFF is only the behaviours sharing
    that layer (Venhoff layers 17/18/15/18 → L18 has two behaviours)."""
    from src.annotation import TARGET_BEHAVIOURS
    layers = {"backtracking": 17, "uncertainty-estimation": 18,
              "example-testing": 15, "adding-knowledge": 18}
    _fake_activation_dir(tmp_path, layers)
    res = build_huang_manifold_vectors(tmp_path, layers=layers)
    # backtracking @17 is alone at its layer → no OFF behaviours → skipped
    assert "backtracking" not in res
    assert "example-testing" not in res        # alone @15
    # uncertainty(18) and adding-knowledge(18) share L18 → each is the other's OFF
    assert "uncertainty-estimation" in res
    assert "adding-knowledge" in res
    assert res["uncertainty-estimation"]["n_off"] == res["adding-knowledge"]["n_on"]


def test_build_holdout_actually_drops_rows(tmp_path):
    from src.annotation import TARGET_BEHAVIOURS
    layers = {b: 27 for b in TARGET_BEHAVIOURS}
    behs = _fake_activation_dir(tmp_path, layers, n_excl_rows=10)
    full = build_huang_manifold_vectors(tmp_path, layers=layers)
    held = build_huang_manifold_vectors(tmp_path, layers=layers,
                                        exclude_chain_ids={"EXCL"})
    for b in behs:
        assert held[b]["n_excluded"] == 10
        assert held[b]["n_on"] == full[b]["n_on"] - 10


def test_build_holdout_mismatch_fails_loud(tmp_path):
    """A sidecar whose row count disagrees with the matrices must raise."""
    from src.annotation import TARGET_BEHAVIOURS
    layers = {b: 27 for b in TARGET_BEHAVIOURS}
    behs = _fake_activation_dir(tmp_path, layers)
    json.dump({"sentence_matching": "exact",
               "rows": {b: [{"chain_id": "x"}] * 3 for b in behs}},
              open(tmp_path / "row_index.json", "w"))
    with pytest.raises(RuntimeError):
        build_huang_manifold_vectors(tmp_path, layers=layers,
                                     exclude_chain_ids={"EXCL"})


def test_build_missing_activation_file_skipped(tmp_path):
    from src.annotation import TARGET_BEHAVIOURS
    layers = {b: 27 for b in TARGET_BEHAVIOURS}
    behs = _fake_activation_dir(tmp_path, layers)
    (tmp_path / f"{behs[0]}_layer27.npy").unlink()
    res = build_huang_manifold_vectors(tmp_path, layers=layers)
    assert behs[0] not in res


# ── Save/load roundtrip (reuses src.steering I/O) ─────────────────────────────

def test_save_load_roundtrip_huang_artefacts(tmp_path):
    from src.annotation import TARGET_BEHAVIOURS
    layers = {b: 27 for b in TARGET_BEHAVIOURS}
    _fake_activation_dir(tmp_path, layers)
    res = build_huang_manifold_vectors(tmp_path, layers=layers)
    out = tmp_path / "out_huang"
    save_steering_vectors(res, out, provenance={"construction": "huang_pooled_pca"})
    assert (out / "backtracking_manifold_kauto.npy").exists()
    assert (out / "backtracking_manifold_k10.npy").exists()
    assert (out / "backtracking_single.npy").exists()
    loaded = load_steering_vectors(out)
    assert set(loaded) == set(TARGET_BEHAVIOURS)
    b0 = list(TARGET_BEHAVIOURS)[0]
    assert np.allclose(loaded[b0]["single_direction"], res[b0]["single_direction"])
    assert np.allclose(loaded[b0]["manifold_projected"][10],
                       res[b0]["manifold_projected"][10])
