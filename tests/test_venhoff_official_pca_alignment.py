import numpy as np
import pytest

import venhoff_official_pca_alignment as alignment_module
from venhoff_official_pca_alignment import (
    alignment,
    compare_directions,
    covariance_pca,
    cosine,
    load_selected_rows,
    unit,
)


def test_alignment_names_norm_and_squared_energy_separately():
    components = np.eye(4)[:2]
    direction = unit(np.array([1.0, 1.0, 1.0, 1.0]))
    result = alignment(direction, components, [1, 2])

    assert result["by_k"]["1"]["captured_squared_norm"] == pytest.approx(0.25)
    assert result["by_k"]["1"]["retained_norm"] == pytest.approx(0.5)
    assert result["by_k"]["2"]["captured_squared_norm"] == pytest.approx(0.5)
    assert result["by_k"]["2"]["retained_norm"] == pytest.approx(np.sqrt(0.5))


def test_projection_cosine_equals_retained_norm():
    components = np.eye(5)[[0, 2, 4]]
    direction = unit(np.array([2.0, 1.0, -3.0, 4.0, 5.0]))
    result = alignment(direction, components, [3])
    projected = unit(components.T @ (components @ direction))

    assert cosine(direction, projected) == pytest.approx(
        result["by_k"]["3"]["retained_norm"]
    )


def test_compare_directions_reports_signed_cosine_and_energy_delta():
    components = np.eye(3)[:1]
    current = unit(np.array([1.0, 1.0, 0.0]))
    official = unit(np.array([1.0, 0.0, 1.0]))
    result = compare_directions(current, official, components, [1])

    assert result["direction_cosine_current_vs_official"] == pytest.approx(0.5)
    assert result["official_minus_current"]["1"][
        "official_minus_current_captured_squared_norm"
    ] == pytest.approx(0.0)


def test_alignment_rejects_nonorthonormal_basis():
    with pytest.raises(ValueError, match="not sufficiently orthonormal"):
        alignment(np.ones(3), np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), [1])


def test_float64_covariance_pca_matches_centered_svd():
    rng = np.random.default_rng(17)
    # A large offset makes this sensitive to accidental float32 X.T @ X
    # accumulation followed by cancellation of n * mean * mean.T.
    matrix = (1_000.0 + rng.normal(size=(80, 6))).astype(np.float32)
    components, eigenvalues, mean, metadata = covariance_pca(
        matrix, max_components=5
    )

    centered = matrix.astype(np.float64) - matrix.astype(np.float64).mean(axis=0)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    expected_eigenvalues = np.square(singular_values) / (matrix.shape[0] - 1)

    assert np.allclose(mean, matrix.astype(np.float64).mean(axis=0))
    assert np.allclose(eigenvalues, expected_eigenvalues[:5], rtol=1e-10, atol=1e-10)
    # Component signs are arbitrary, so compare their projectors.
    assert np.allclose(
        components[:3].T @ components[:3],
        vt[:3].T @ vt[:3],
        rtol=1e-9,
        atol=1e-9,
    )
    assert metadata["numeric_method"].startswith("mean-centred float64")


def test_row_split_uses_exact_holdout_ids(tmp_path, monkeypatch):
    current = tmp_path / "current"
    archive = tmp_path / "archive"
    current.mkdir()
    archive.mkdir()
    np.save(
        current / "backtracking_layer17.npy",
        np.arange(12, dtype=np.float32).reshape(4, 3),
    )
    row_indices = {
        "current": {
            "rows": {
                "backtracking": [
                    {"chain_id": "train-a"},
                    {"chain_id": "heldout-a"},
                    {"chain_id": "train-b"},
                    {"chain_id": "heldout-b"},
                ]
            }
        },
        "archive": {"rows": {}},
    }
    monkeypatch.setattr(alignment_module, "CURRENT", current)
    monkeypatch.setattr(alignment_module, "ARCHIVE", archive)

    training, training_counts, _ = load_selected_rows(
        "backtracking",
        17,
        row_indices,
        {"heldout-a", "heldout-b"},
        "holdout_excluded_e1_training",
    )
    heldout, heldout_counts, _ = load_selected_rows(
        "backtracking",
        17,
        row_indices,
        {"heldout-a", "heldout-b"},
        "heldout_50_task",
    )

    assert training[:, 0].tolist() == [0.0, 6.0]
    assert heldout[:, 0].tolist() == [3.0, 9.0]
    assert training_counts == {
        "raw": 4,
        "heldout_rows": 2,
        "training_rows": 2,
        "selected_rows": 2,
    }
    assert heldout_counts == training_counts
