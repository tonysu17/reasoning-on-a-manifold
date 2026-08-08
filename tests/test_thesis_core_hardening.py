import hashlib
import importlib.util
import json
import unicodedata
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.intrinsic_dim import correlation_dimension_estimate


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "thesis_core_hardening.py"


def test_planned_harness_module_exists():
    assert SCRIPT.is_file()


def _load_harness():
    spec = importlib.util.spec_from_file_location("thesis_core_hardening", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_required_public_symbols_exist():
    harness = _load_harness()
    required = {
        "correlation_dimension_point",
        "canonical_occurrence_digest",
        "build_common_complete_case_grid",
        "run_within_chain_cdim_null",
        "run_chain_stability_band",
        "run_truncation_sensitivity",
        "run_pooling_window_family",
        "run_curvature_diagnostic",
        "run_resource_benchmark",
        "run_empirical_backtracking_resource_pilot",
        "load_empirical_backtracking_l27_inputs",
        "build_primary_family_document",
        "write_primary_family_result",
        "execute_registered_primary_from_disk",
        "load_registered_secondary_family_inputs",
        "build_secondary_family_document",
        "write_secondary_family_result",
        "execute_registered_secondary_from_disk",
        "load_registered_curvature_inputs",
        "build_h3_family_documents",
        "build_curvature_family_document",
        "write_scope_family_result",
        "execute_registered_h3_from_disk",
        "execute_registered_curvature_from_disk",
        "write_resource_only_benchmark",
        "write_resource_benchmark_markdown",
        "main",
    }
    assert required <= set(vars(harness))


def test_correlation_dimension_wrapper_is_exact_and_rejects_invalid_inputs():
    harness = _load_harness()
    rng = np.random.default_rng(910)
    x = rng.normal(size=(140, 4))
    x32 = x.astype(np.float32)
    expected = correlation_dimension_estimate(
        x32.astype(np.float64),
        n_radii=20,
        random_state=42,
        n_bootstrap=0,
        subsample=2000,
    ).estimate
    assert harness.correlation_dimension_point(x32) == expected
    assert np.isfinite(expected) and expected > 0

    for invalid in (
        np.ones((99, 3)),
        np.ones((100, 3)),
        np.r_[np.ones((99, 3)), [[np.nan, 0, 0]]],
    ):
        with pytest.raises(ValueError):
            harness.correlation_dimension_point(invalid)


def test_participation_ratio_matches_full_centered_svd_definition():
    harness = _load_harness()
    x = np.array([[1, 0], [2, 1], [4, 0], [8, 1]], dtype=np.float32)
    xc = x.astype(np.float64) - x.astype(np.float64).mean(axis=0)
    s = np.linalg.svd(xc, full_matrices=False, compute_uv=False)
    eigenvalues = s**2 / (len(x) - 1)
    expected = eigenvalues.sum() ** 2 / (eigenvalues**2).sum()
    assert harness._participation_ratio(x) == expected
    with pytest.raises(ValueError):
        harness._participation_ratio(np.ones((1, 2)))
    with pytest.raises(ValueError):
        harness._participation_ratio(np.ones((3, 2)))


def test_registered_wrappers_convert_only_numeric_domain_errors(monkeypatch):
    harness = _load_harness()

    def cdim_numeric_failure(*args, **kwargs):
        raise ValueError("numeric estimator domain")

    monkeypatch.setattr(
        "src.intrinsic_dim.correlation_dimension_estimate",
        cdim_numeric_failure,
    )
    with pytest.raises(harness.EstimatorInvalidError, match="numeric domain"):
        harness.correlation_dimension_point(np.ones((100, 2)))

    def svd_numeric_failure(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(np.linalg, "svd", svd_numeric_failure)
    with pytest.raises(harness.EstimatorInvalidError, match="numeric domain"):
        harness._participation_ratio(np.arange(8, dtype=float).reshape(4, 2))


def test_default_curvature_wrapper_converts_numeric_domain_errors(monkeypatch):
    harness = _load_harness()
    records = []
    rows = []
    for chain_index in range(15):
        for occurrence_index in range(2):
            records.append(_record("A", f"c{chain_index:02}", occurrence_index))
            rows.append([chain_index, occurrence_index, chain_index / 10])

    def curvature_numeric_failure(values, **settings):
        raise ValueError("curvature numeric domain")

    monkeypatch.setattr(
        "src.curvature.local_vs_global_dim_ratio",
        curvature_numeric_failure,
    )
    result = harness.run_curvature_diagnostic(
        np.asarray(rows),
        records,
        target_label="A",
        behaviour_code=1,
        cell_index=88,
        n_replicates=1,
        valid_threshold=1,
    )
    assert result["status"] == "UNRUN"
    assert result["draws"][0]["failure"] == "paired_estimator_invalid"


def test_canonical_digest_exact_serialization_and_nfc_normalization():
    harness = _load_harness()
    kwargs = dict(
        family_code=1,
        annotator_code=1,
        layer=27,
        behaviour_code=2,
        pooling_code=1,
        window_code=1,
        truncation_code=0,
        chain_id="cafe\u0301",
        annotation_index=3,
        char_offset=17,
        token_start=22,
        replicate_index=-1,
        scope_key="",
    )
    payload = [
        "thesis-core-hardening-v1",
        1,
        1,
        27,
        2,
        1,
        1,
        0,
        unicodedata.normalize("NFC", kwargs["chain_id"]),
        3,
        17,
        22,
        -1,
        "",
    ]
    expected = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    assert harness.canonical_occurrence_digest(**kwargs) == expected
    kwargs["chain_id"] = "café"
    assert harness.canonical_occurrence_digest(**kwargs) == expected
    kwargs["annotation_index"] = None
    with pytest.raises((TypeError, ValueError)):
        harness.canonical_occurrence_digest(**kwargs)


def test_seed_coordinates_and_lower_tail_and_holm_known_answers():
    harness = _load_harness()
    expected_perm = np.random.default_rng(
        np.random.SeedSequence([20260726, 1, 5, 71, 9])
    ).integers(0, 2**32, size=8)
    actual_perm = harness._permutation_rng(20260726, 5, 71, 9).integers(
        0, 2**32, size=8
    )
    np.testing.assert_array_equal(actual_perm, expected_perm)

    expected_stability = np.random.default_rng(
        np.random.SeedSequence([20260726, 2, 1, 0, 4])
    ).integers(0, 2**32, size=8)
    actual_stability = harness._chain_stability_rng(1, 0, 4).integers(
        0, 2**32, size=8
    )
    np.testing.assert_array_equal(actual_stability, expected_stability)

    assert harness._lower_tail_p(2.0, [1.0, 2.0, 3.0, 4.0]) == 3 / 5
    np.testing.assert_allclose(
        harness._holm_adjust([0.01, 0.04, 0.03, 0.002]),
        [0.03, 0.06, 0.06, 0.008],
    )


def _record(
    behaviour,
    chain_id,
    annotation_index,
    *,
    char_offset=None,
    token_start=None,
    record_id=None,
    n_positions=None,
):
    record = {
        "behaviour": behaviour,
        "chain_id": chain_id,
        "annotation_index": annotation_index,
        "char_offset": annotation_index * 10 if char_offset is None else char_offset,
        "token_start": annotation_index * 3 if token_start is None else token_start,
        "record_id": record_id or f"{chain_id}:{annotation_index}",
    }
    if n_positions is not None:
        record["n_positions"] = n_positions
    return record


def test_alignment_occurrence_and_vector_duplicate_audit():
    harness = _load_harness()
    repeated = [
        _record("A", "c1", 0, record_id="z"),
        _record("A", "c1", 0, record_id="a"),
        _record("A", "c1", 1),
    ]
    x = np.array([[9.0, 9.0], [1.0, 2.0], [3.0, 4.0]])
    audited = harness._validate_and_deduplicate(x, repeated)
    assert audited["records"][0]["record_id"] == "a"
    assert audited["audit"]["occurrence_duplicate_rows"] == 1

    vector_repeated = [
        _record("A", "c1", 2),
        _record("A", "c1", 1),
        _record("A", "c1", 3),
    ]
    x2 = np.array([[5.0, 6.0], [5.0, 6.0], [7.0, 8.0]])
    audited2 = harness._validate_and_deduplicate(x2, vector_repeated)
    assert [r["annotation_index"] for r in audited2["records"]] == [1, 3]
    assert audited2["audit"]["vector_duplicate_rows"] == 1

    signed_zero = harness._validate_and_deduplicate(
        np.array([[-0.0, 1.0], [0.0, 1.0]]),
        [_record("A", "c1", 0), _record("A", "c1", 1)],
    )
    assert signed_zero["audit"]["vector_duplicate_rows"] == 1


def test_duplicate_conflicts_and_proxy_or_misaligned_rows_hard_fail():
    harness = _load_harness()
    duplicate_vector = np.array([[1.0, 2.0], [1.0, 2.0]])
    with pytest.raises(harness.DuplicateAuditError, match="cross-chain") as cross_chain:
        harness._validate_and_deduplicate(
            duplicate_vector,
            [_record("A", "c1", 0), _record("A", "c2", 0)],
        )
    assert cross_chain.value.audit["cross_chain_groups"] == 1
    assert cross_chain.value.audit["cross_label_groups"] == 0
    assert cross_chain.value.audit["vector_duplicate_groups"] == 1
    assert cross_chain.value.audit["duplicate_fraction"] == 0.5

    with pytest.raises(harness.DuplicateAuditError, match="cross-label") as cross_label:
        harness._validate_and_deduplicate(
            duplicate_vector,
            [_record("A", "c1", 0), _record("B", "c1", 1)],
        )
    assert cross_label.value.audit["cross_chain_groups"] == 0
    assert cross_label.value.audit["cross_label_groups"] == 1
    with pytest.raises(ValueError, match="row alignment"):
        harness._validate_and_deduplicate(
            np.ones((2, 2)), [_record("A", "c1", 0)]
        )
    with pytest.raises(ValueError, match="proxy"):
        harness._validate_and_deduplicate(
            np.ones((1, 2)), [_record("A", "proxy-0", 0)]
        )


def test_amended_policy_drops_all_same_window_cross_label_members():
    harness = _load_harness()
    duplicate_vector = np.array(
        [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [3.0, 4.0]]
    )
    records = [
        _record(
            "A", "c1", 1, char_offset=10, token_start=5, n_positions=11
        ),
        _record(
            "B", "c1", 8, char_offset=10, token_start=5, n_positions=11
        ),
        _record(
            "B", "c1", 9, char_offset=10, token_start=5, n_positions=11
        ),
        _record(
            "A", "c1", 10, char_offset=20, token_start=8, n_positions=11
        ),
    ]
    with pytest.raises(harness.DuplicateAuditError, match="cross-label"):
        harness._validate_and_deduplicate(duplicate_vector, records)
    audited = harness._validate_and_deduplicate(
        duplicate_vector,
        records,
        cross_label_policy="drop_all_same_extraction_window",
    )
    assert audited["kept_indices"] == [3]
    assert audited["audit"]["deduplicated_rows"] == 1
    assert audited["audit"]["cross_label_groups"] == 1
    assert audited["audit"]["cross_label_same_window_groups_dropped"] == 1
    assert audited["audit"]["cross_label_same_window_rows_dropped"] == 3
    assert audited["audit"]["unresolved_cross_label_groups"] == 0
    assert audited["audit"]["cross_label_policy"] == (
        "drop_all_same_extraction_window"
    )


def test_amended_policy_still_rejects_distinct_window_or_cross_chain_conflicts():
    harness = _load_harness()
    duplicate_vector = np.array([[1.0, 2.0], [1.0, 2.0]])
    with pytest.raises(
        harness.DuplicateAuditError, match="cross-label"
    ) as distinct_window:
        harness._validate_and_deduplicate(
            duplicate_vector,
            [
                _record(
                    "A",
                    "c1",
                    1,
                    char_offset=10,
                    token_start=5,
                    n_positions=11,
                ),
                _record(
                    "B",
                    "c1",
                    2,
                    char_offset=20,
                    token_start=8,
                    n_positions=11,
                ),
            ],
            cross_label_policy="drop_all_same_extraction_window",
        )
    assert distinct_window.value.audit["unresolved_cross_label_groups"] == 1
    with pytest.raises(
        harness.DuplicateAuditError, match="cross-chain"
    ):
        harness._validate_and_deduplicate(
            duplicate_vector,
            [
                _record(
                    "A",
                    "c1",
                    1,
                    char_offset=10,
                    token_start=5,
                    n_positions=11,
                ),
                _record(
                    "B",
                    "c2",
                    2,
                    char_offset=10,
                    token_start=5,
                    n_positions=11,
                ),
            ],
            cross_label_policy="drop_all_same_extraction_window",
        )


def test_registered_null_reports_amended_duplicate_and_sample_provenance():
    harness = _load_harness()
    labels = np.asarray(["A", "B", "A", "B", "A", "B"])
    chains = np.asarray(["c1", "c1", "c1", "c1", "c2", "c2"])
    records = [
        _record("A", "c1", 0, char_offset=10, token_start=2, n_positions=11),
        _record("B", "c1", 1, char_offset=10, token_start=2, n_positions=11),
        _record("A", "c1", 2, char_offset=20, token_start=4, n_positions=11),
        _record("B", "c1", 3, char_offset=30, token_start=6, n_positions=11),
        _record("A", "c2", 0, char_offset=10, token_start=2, n_positions=11),
        _record("B", "c2", 1, char_offset=20, token_start=4, n_positions=11),
    ]
    rows = np.asarray(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ]
    )
    result = harness.run_within_chain_cdim_null(
        rows,
        labels,
        chains,
        records,
        target_label="A",
        family_code=1,
        cell_index=0,
        behaviour_code=1,
        B=1,
        attempt_cap=1,
        cross_label_policy="drop_all_same_extraction_window",
        statistic_fn=lambda values: float(len(values) + 1),
    )
    assert result["status"] == "VALID"
    assert result["duplicate_audit"][
        "cross_label_same_window_groups_dropped"
    ] == 1
    assert result["row_counts"] == {
        "raw": 6,
        "deduplicated": 4,
        "analyzed": 2,
        "sentence_weighted": 2,
    }
    assert result["chain_counts"] == {
        "unique": 2,
        "eligible": 2,
        "selected": 2,
        "mixed_label": 2,
    }
    assert len(result["occurrence_list_sha256"]) == 64
    assert result["chain_attrition_reasons"] == {}


def test_equal_chain_selection_gives_each_chain_one_row():
    harness = _load_harness()
    records = [
        *[_record("A", "long", i) for i in range(5)],
        _record("A", "short", 0),
    ]
    selected = harness._select_equal_chain_indices(
        records,
        np.array(["A"] * 6),
        target_label="A",
        family_code=1,
        annotator_code=1,
        layer=27,
        behaviour_code=1,
        pooling_code=1,
        window_code=1,
        truncation_code=0,
        replicate_index=-1,
        scope_key="",
    )
    assert len(selected) == 2
    assert {records[i]["chain_id"] for i in selected} == {"long", "short"}


def test_within_chain_null_preserves_counts_retains_identity_and_is_deterministic():
    harness = _load_harness()
    labels = np.array(["A", "B", "A", "B", "A", "B"])
    chain_ids = np.array(["c1", "c1", "c2", "c2", "c3", "c3"])
    records = [
        _record(label, chain, i)
        for i, (label, chain) in enumerate(zip(labels, chain_ids))
    ]
    x = np.arange(12, dtype=float).reshape(6, 2)

    def finite_statistic(values):
        return float(np.mean(values[:, 0]) + 1)

    kwargs = dict(
        activations=x,
        labels=labels,
        chain_ids=chain_ids,
        occurrences=records,
        target_label="A",
        family_code=1,
        cell_index=0,
        behaviour_code=1,
        B=12,
        attempt_cap=20,
        statistic_fn=finite_statistic,
    )
    first = harness.run_within_chain_cdim_null(**kwargs)
    second = harness.run_within_chain_cdim_null(**kwargs)
    assert first == second
    assert first["valid"] == 12
    assert any(attempt["identity"] for attempt in first["attempts"])
    assert first["identity_attempts"] == sum(
        attempt["identity"] for attempt in first["attempts"]
    )
    for attempt in first["attempts"]:
        assert set(attempt) == {
            "attempt_index",
            "assignment_sha256",
            "identity",
            "status",
            "reason",
        }
        assert len(attempt["assignment_sha256"]) == 64
        int(attempt["assignment_sha256"], 16)
        assert "assignment" not in attempt
    projected_2750_bytes = (
        len(json.dumps(first["attempts"])) / len(first["attempts"]) * 2750
    )
    assert projected_2750_bytes < 1_000_000


def test_within_chain_null_accounts_for_estimator_invalid_attempts():
    harness = _load_harness()
    labels = np.array(["A", "B"])
    chain_ids = np.array(["c1", "c1"])
    records = [_record("A", "c1", 0), _record("B", "c1", 1)]
    x = np.array([[1.0], [0.0]])

    def invalid_for_second_row(values):
        return 1.0 if float(values[0, 0]) == 1.0 else float("nan")

    result = harness.run_within_chain_cdim_null(
        x,
        labels,
        chain_ids,
        records,
        target_label="A",
        family_code=1,
        cell_index=0,
        behaviour_code=1,
        B=10,
        attempt_cap=20,
        statistic_fn=invalid_for_second_row,
    )
    assert result["valid"] == 10
    assert result["attempted"] > result["valid"]
    assert result["invalid"] == result["attempted"] - result["valid"]
    assert result["invalid_reasons"] == {"estimator_invalid": result["invalid"]}
    assert any(a["status"] == "INVALID" for a in result["attempts"])


def test_public_entry_points_enforce_exact_cell_and_code_registry():
    harness = _load_harness()
    x = np.array([[1.0], [2.0]])
    labels = np.array(["A", "B"])
    chains = np.array(["c1", "c1"])
    records = [_record("A", "c1", 0), _record("B", "c1", 1)]
    common = dict(
        activations=x,
        labels=labels,
        chain_ids=chains,
        occurrences=records,
        target_label="A",
        behaviour_code=1,
        B=1,
        attempt_cap=1,
        statistic_fn=lambda values: 1.0,
    )
    with pytest.raises(ValueError, match="registry"):
        harness.run_within_chain_cdim_null(
            **common, family_code=99, cell_index=999
        )
    with pytest.raises(ValueError, match="registry"):
        harness.run_within_chain_cdim_null(
            **{**common, "behaviour_code": 2}, family_code=1, cell_index=0
        )
    with pytest.raises(ValueError, match="registry"):
        harness.run_chain_stability_band(
            x,
            records,
            target_label="A",
            family_code=1,
            cell_index=0,
            behaviour_code=2,
            n_replicates=1,
            valid_threshold=1,
            cdim_fn=lambda values: 1.0,
            pr_fn=lambda values: 1.0,
        )
    with pytest.raises(ValueError, match="registry"):
        harness.run_curvature_diagnostic(
            np.ones((22, 2)),
            [_record("A", f"c{i // 2}", i % 2) for i in range(22)],
            target_label="A",
            behaviour_code=2,
            cell_index=88,
            n_replicates=1,
            valid_threshold=1,
            ratio_fn=lambda values, **kwargs: 1.0,
        )
    with pytest.raises(ValueError, match="registry"):
        harness.run_truncation_sensitivity(
            np.ones((1, 2)),
            [_record("A", "c1", 0)],
            {"c1": {"n_tokens": 1, "chain": "</think>", "category": "x"}},
            target_label="A",
            behaviour_code=5,
            n_replicates=1,
            valid_threshold=1,
        )
    with pytest.raises(ValueError, match="registry"):
        harness.build_common_complete_case_grid(
            _six_representations(
                [_record("A", "c1", 0)], [[1.0, 2.0]]
            ),
            behaviour_codes={"A": 99},
        )


def test_permutation_rejects_only_registered_estimator_invalid_condition():
    harness = _load_harness()
    x = np.array([[1.0], [0.0]])
    labels = np.array(["A", "B"])
    chains = np.array(["c1", "c1"])
    records = [_record("A", "c1", 0), _record("B", "c1", 1)]

    calls = 0

    def registered_invalid(values):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 1.0
        raise harness.EstimatorInvalidError("registered invalid region")

    rejected = harness.run_within_chain_cdim_null(
        x,
        labels,
        chains,
        records,
        target_label="A",
        family_code=1,
        cell_index=0,
        behaviour_code=1,
        B=1,
        attempt_cap=1,
        statistic_fn=registered_invalid,
    )
    assert rejected["status"] == "UNRUN"
    assert rejected["invalid_reasons"] == {"estimator_invalid": 1}

    calls = 0

    def programming_failure(values):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 1.0
        raise RuntimeError("unexpected programming failure")

    with pytest.raises(RuntimeError, match="unexpected programming failure"):
        harness.run_within_chain_cdim_null(
            x,
            labels,
            chains,
            records,
            target_label="A",
            family_code=1,
            cell_index=0,
            behaviour_code=1,
            B=1,
            attempt_cap=1,
            statistic_fn=programming_failure,
        )


def test_permutation_runner_rejects_truncation_purpose_before_estimator():
    harness = _load_harness()
    calls = 0

    def statistic(values):
        nonlocal calls
        calls += 1
        return 1.0

    with pytest.raises(ValueError, match="purpose"):
        harness.run_within_chain_cdim_null(
            np.asarray([[1.0], [2.0]]),
            ["A", "B"],
            ["c1", "c1"],
            [_record("A", "c1", 0), _record("B", "c1", 1)],
            target_label="A",
            family_code=4,
            cell_index=48,
            behaviour_code=1,
            pooling_code=0,
            window_code=0,
            truncation_code=1,
            B=0,
            attempt_cap=0,
            statistic_fn=statistic,
        )
    assert calls == 0


def test_tiny_synthetic_permutation_recovers_low_dimension_signal():
    harness = _load_harness()
    rng = np.random.default_rng(123)
    n_chains = 100
    t = np.linspace(-3, 3, n_chains)
    low = np.column_stack([t, t**2, np.sin(t), np.zeros_like(t), np.ones_like(t)])
    low += rng.normal(scale=0.002, size=low.shape)
    broad = rng.normal(size=(n_chains, 5))
    x = np.empty((n_chains * 2, 5))
    x[0::2], x[1::2] = low, broad
    labels = np.tile(["A", "B"], n_chains)
    chain_ids = np.repeat([f"c{i:03}" for i in range(n_chains)], 2)
    records = [
        _record(label, chain, i)
        for i, (label, chain) in enumerate(zip(labels, chain_ids))
    ]
    result = harness.run_within_chain_cdim_null(
        x,
        labels,
        chain_ids,
        records,
        target_label="A",
        family_code=1,
        cell_index=0,
        behaviour_code=1,
        B=19,
        attempt_cap=25,
    )
    assert result["status"] == "VALID"
    assert result["p_value"] <= 0.15


def test_chain_stability_uses_registered_chain_draws_and_linear_quantiles():
    harness = _load_harness()
    rng = np.random.default_rng(444)
    n_chains = 125
    records = []
    rows = []
    for chain_index in range(n_chains):
        for occurrence_index in range(2):
            records.append(
                _record(
                    "A",
                    f"c{chain_index:03}",
                    occurrence_index,
                    char_offset=10 * occurrence_index,
                    token_start=occurrence_index,
                )
            )
            rows.append(rng.normal(size=4) + chain_index / n_chains)
    result = harness.run_chain_stability_band(
        np.asarray(rows),
        records,
        target_label="A",
        family_code=1,
        cell_index=0,
        behaviour_code=1,
        n_replicates=4,
        valid_threshold=3,
    )
    assert result["label"] == "chain stability band (not a confidence interval)"
    assert result["selected_chain_count"] == 100
    sorted_chains = np.array([f"c{i:03}" for i in range(n_chains)])
    expected = sorted_chains[
        np.sort(harness._chain_stability_rng(1, 0, 0).choice(125, 100, replace=False))
    ].tolist()
    assert result["draws"][0]["selected_chain_ids"] == expected
    cdim_draws = [draw["cdim"] for draw in result["draws"]]
    expected_quantiles = np.quantile(
        cdim_draws, [0.025, 0.5, 0.975], method="linear"
    )
    np.testing.assert_allclose(
        [
            result["cdim_stability"]["q025"],
            result["cdim_stability"]["median"],
            result["cdim_stability"]["q975"],
        ],
        expected_quantiles,
    )


@pytest.mark.parametrize(
    ("n_tokens", "text", "expected"),
    [
        (8191, "unfinished", False),
        (8192, "finished </think>   ", False),
        (8192, "unfinished", True),
        (9000, "unfinished </thinking>", True),
    ],
)
def test_truncation_boundary_rule(n_tokens, text, expected):
    harness = _load_harness()
    assert harness._is_truncated(n_tokens=n_tokens, chain_text=text) is expected


@pytest.mark.parametrize(
    ("chain_leg", "truncation_leg", "expected"),
    [
        ("PASS", "PASS", "PASS"),
        ("PASS", "MIXED", "MIXED"),
        ("PASS", "FAIL", "FAIL"),
        ("PASS", "UNRUN", "UNRUN"),
        ("FAIL", "PASS", "FAIL"),
        ("FAIL", "MIXED", "FAIL"),
        ("FAIL", "FAIL", "FAIL"),
        ("FAIL", "UNRUN", "UNRUN"),
        ("UNRUN", "PASS", "UNRUN"),
        ("UNRUN", "MIXED", "UNRUN"),
        ("UNRUN", "FAIL", "UNRUN"),
        ("UNRUN", "UNRUN", "UNRUN"),
    ],
)
def test_h3_cross_product_is_exhaustive(chain_leg, truncation_leg, expected):
    harness = _load_harness()
    assert harness._h3_cross_product(chain_leg, truncation_leg) == expected


def test_h3_family_operator_and_category_imbalance_threshold():
    harness = _load_harness()
    assert harness._h3_family_status(["PASS"] * 4) == "PASS"
    assert harness._h3_family_status(["FAIL"] * 4) == "FAIL"
    assert harness._h3_family_status(["PASS", "PASS", "FAIL", "MIXED"]) == "MIXED"
    assert harness._h3_family_status(["PASS", "PASS", "PASS", "UNRUN"]) == "UNRUN"

    exactly_ten = harness._category_share_imbalance(
        {"math": 55, "code": 45}, {"math": 45, "code": 55}
    )
    over_ten = harness._category_share_imbalance(
        {"math": 56, "code": 44}, {"math": 44, "code": 56}
    )
    assert exactly_ten == pytest.approx(0.10)
    assert harness._matching_required(exactly_ten) is False
    assert harness._matching_required(over_ten) is True
    assert harness._category_share_imbalance(
        {"cafe\u0301": 1}, {"café": 1}
    ) == 0.0


def test_h3_chain_stability_leg_is_computed_from_band_not_caller_verdict():
    harness = _load_harness()
    passing = {
        "status": "VALID",
        "valid_threshold": 3,
        "full_sample_cdim": 2.0,
        "full_sample_pr": 4.0,
        "cdim_stability": {
            "n_valid": 3,
            "q025": 1.5,
            "median": 2.1,
            "q975": 2.5,
        },
        "pr_stability": {
            "n_valid": 3,
            "q025": 3.0,
            "median": 4.2,
            "q975": 5.0,
        },
    }
    assert harness._classify_chain_stability_leg(passing) == "PASS"
    failing = {
        **passing,
        "cdim_stability": {
            **passing["cdim_stability"],
            "median": 4.0,
        },
    }
    assert harness._classify_chain_stability_leg(failing) == "FAIL"
    unrun = {**passing, "status": "UNRUN"}
    assert harness._classify_chain_stability_leg(unrun) == "UNRUN"


def test_truncation_occurrence_map_is_neutral_before_strata():
    harness = _load_harness()
    records = [
        _record("A", "complete", 0),
        _record("A", "complete", 1),
        _record("A", "truncated", 0),
        _record("A", "truncated", 1),
    ]
    neutral = harness._truncation_neutral_map(
        records,
        target_label="A",
        behaviour_code=1,
        replicate_index=-1,
    )
    complete = {key: value for key, value in neutral.items() if key == "complete"}
    truncated = {key: value for key, value in neutral.items() if key == "truncated"}
    assert complete["complete"] == neutral["complete"]
    assert truncated["truncated"] == neutral["truncated"]
    assert len(neutral) == 2


def _six_representations(records, base_rows):
    return {
        (pooling, window): {
            "records": [dict(record) for record in records],
            "activations": np.asarray(base_rows, dtype=float) + offset,
        }
        for offset, (pooling, window) in enumerate(
            [
                ("mean", "unclipped"),
                ("mean", "clipped"),
                ("first", "unclipped"),
                ("first", "clipped"),
                ("last", "unclipped"),
                ("last", "clipped"),
            ]
        )
    }


def test_common_complete_case_grid_is_joint_and_representation_neutral():
    harness = _load_harness()
    records = [
        _record("A", "c1", 0),
        _record("A", "c1", 1),
        _record("A", "c2", 0),
    ]
    representations = _six_representations(
        records, [[1.0, 1.0], [2.0, 2.0], [3.0, 4.0]]
    )
    # In one representation, two same-chain/same-label vectors collide.  Its
    # digest loser must disappear from every representation.
    representations[("last", "clipped")]["activations"][1] = representations[
        ("last", "clipped")
    ]["activations"][0]
    grid = harness.build_common_complete_case_grid(
        representations, behaviour_codes={"A": 1}
    )
    behaviour = grid["behaviours"]["A"]
    assert len(behaviour["occurrence_keys"]) == 2
    assert len(
        {
            cell["occurrence_list_sha256"]
            for cell in behaviour["representations"].values()
        }
    ) == 1
    for cell in behaviour["representations"].values():
        assert cell["occurrence_keys"] == behaviour["occurrence_keys"]
        assert cell["activations"].shape == (2, 2)


def test_common_grid_requires_all_six_and_hard_fails_cross_chain_duplicate():
    harness = _load_harness()
    records = [_record("A", "c1", 0), _record("A", "c2", 0)]
    representations = _six_representations(records, [[1.0, 2.0], [3.0, 4.0]])
    missing = dict(representations)
    missing.pop(("last", "clipped"))
    with pytest.raises(ValueError, match="six"):
        harness.build_common_complete_case_grid(missing, behaviour_codes={"A": 1})

    representations[("first", "clipped")]["activations"][:] = [9.0, 9.0]
    with pytest.raises(harness.DuplicateAuditError, match="cross-chain") as conflict:
        harness.build_common_complete_case_grid(
            representations, behaviour_codes={"A": 1}
        )
    assert conflict.value.audit["cross_chain_groups"] >= 1
    assert conflict.value.audit["cross_label_groups"] == 0
    assert conflict.value.audit["vector_duplicate_groups"] >= 1


def test_pooling_window_family_runs_all_24_cells_or_unrun():
    harness = _load_harness()
    records = []
    rows = []
    labels = ["A", "B", "C", "D"]
    for chain_index in range(2):
        for label_index, label in enumerate(labels):
            records.append(_record(label, f"c{chain_index}", label_index))
            rows.append([chain_index * 10 + label_index, label_index + 0.25])
    grid = harness.build_common_complete_case_grid(
        _six_representations(records, rows),
        behaviour_codes={"A": 1, "B": 2, "C": 3, "D": 4},
    )

    def positive_statistic(values):
        return float(np.mean(np.abs(values)) + 1)

    result = harness.run_pooling_window_family(
        grid,
        B=2,
        attempt_cap=3,
        statistic_fn=positive_statistic,
    )
    assert result["status"] == "VALID"
    assert len(result["cells"]) == 24
    assert len(result["holm_p"]) == 24
    for label in labels:
        digests = {
            cell["occurrence_list_sha256"]
            for cell in result["cells"]
            if cell["label"] == label
        }
        assert len(digests) == 1
    for cell in result["cells"]:
        assert cell["sentence_weighted_label"] == (
            "sentence-weighted correlation dimension"
        )
        assert cell["sentence_weighted_cdim"] > 0
        assert isinstance(cell["chain_attrition_reasons"], dict)

    broken = dict(grid)
    broken["status"] = "UNRUN"
    assert harness.run_pooling_window_family(broken, B=1)["status"] == "UNRUN"


def test_h4_reports_per_chain_complete_case_attrition_reasons():
    harness = _load_harness()
    records = [
        _record("A", "c1", 0),
        _record("A", "c2", 0),
        _record("A", "c3", 0),
    ]
    representations = _six_representations(
        records, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    )
    clipped = representations[("last", "clipped")]
    clipped["records"] = clipped["records"][:2]
    clipped["activations"] = clipped["activations"][:2]
    grid = harness.build_common_complete_case_grid(
        representations, behaviour_codes={"A": 1}
    )
    reasons = grid["behaviours"]["A"]["chain_attrition_reasons"]
    assert reasons["c3"] == ["missing_complete_case_representation:last/clipped"]
    assert grid["duplicate_audit"]["duplicate_fraction"] == 0.0


def test_h4_repeated_occurrence_collapse_uses_one_aligned_record_across_grid():
    harness = _load_harness()
    representations = {}
    for rep_index, representation in enumerate(
        [
            ("mean", "unclipped"),
            ("mean", "clipped"),
            ("first", "unclipped"),
            ("first", "clipped"),
            ("last", "unclipped"),
            ("last", "clipped"),
        ]
    ):
        a = _record("A", "c1", 0, record_id="a")
        z = _record("A", "c1", 0, record_id="z")
        # Metadata is deliberately arranged so independent whole-record
        # lexical choices disagree across representations.
        a["note"] = "z" if rep_index % 2 else "a"
        z["note"] = "a" if rep_index % 2 else "z"
        records = [z, a] if rep_index % 2 else [a, z]
        value_by_id = {
            "a": [100 + rep_index, 1.0],
            "z": [900 + rep_index, 9.0],
        }
        representations[representation] = {
            "records": records,
            "activations": np.asarray(
                [value_by_id[record["record_id"]] for record in records]
            ),
        }
    grid = harness.build_common_complete_case_grid(
        representations, behaviour_codes={"A": 1}
    )
    for rep_index, name in enumerate(grid["representation_order"]):
        assert (
            grid["behaviours"]["A"]["representations"][name]["activations"][0, 0]
            == 100 + rep_index
        )


def _synthetic_family_cells(harness, family):
    labels_order = ["A", "B", "C", "D"]
    labels = np.tile(labels_order, 2)
    chains = np.repeat(["c0", "c1"], 4)
    records = [
        _record(label, chain, i)
        for i, (label, chain) in enumerate(zip(labels, chains))
    ]
    activations = np.arange(16, dtype=float).reshape(8, 2) + 1
    cells = {}
    for coordinates in harness._expected_family_cells(family):
        behaviour = coordinates["behaviour_code"]
        cells[coordinates["cell_index"]] = {
            "activations": activations + coordinates["cell_index"],
            "labels": labels,
            "chain_ids": chains,
            "occurrences": records,
            "target_label": labels_order[behaviour - 1],
            **coordinates,
        }
    return cells


@pytest.mark.parametrize(
    ("family", "expected_count"),
    [("primary", 4), ("five_depth", 20), ("three_annotator", 24)],
)
def test_registered_family_orchestration_is_all_cells_with_whole_family_holm(
    family, expected_count
):
    harness = _load_harness()
    cells = _synthetic_family_cells(harness, family)
    result = harness._run_registered_cdim_family(
        cells,
        family=family,
        B=1,
        attempt_cap=1,
        statistic_fn=lambda values: float(np.mean(np.abs(values)) + 1),
    )
    assert result["status"] == "VALID"
    assert len(result["cells"]) == expected_count
    assert len(result["holm_p"]) == expected_count
    assert len(result["diagnostic_holm_p"]) == (
        expected_count if family == "primary" else 0
    )

    missing = dict(cells)
    missing.pop(min(missing))
    unrun = harness._run_registered_cdim_family(
        missing,
        family=family,
        B=1,
        attempt_cap=1,
        statistic_fn=lambda values: 1.0,
    )
    assert unrun["status"] == "UNRUN"
    assert unrun["decision"] == "UNRUN"


@pytest.mark.parametrize(
    ("family", "expected_count", "expected_seeds"),
    [
        ("primary", 4, {20260726, 20260727}),
        ("five_depth", 20, {20260726}),
        ("three_annotator", 24, {20260726}),
    ],
)
def test_registered_family_seed_reruns_are_limited_to_primary(
    monkeypatch, family, expected_count, expected_seeds
):
    harness = _load_harness()
    calls = []

    def fake_registered_null(**kwargs):
        calls.append((kwargs["cell_index"], kwargs["base_seed"]))
        return {
            "status": "VALID",
            "observed": float(kwargs["behaviour_code"]),
            "p_value": 0.01,
            "null_values": [2.0],
            "attempted": 1,
            "valid": 1,
            "invalid": 0,
            "invalid_reasons": {},
        }

    monkeypatch.setattr(harness, "run_within_chain_cdim_null", fake_registered_null)
    result = harness._run_registered_cdim_family(
        _synthetic_family_cells(harness, family),
        family=family,
        B=1,
        attempt_cap=1,
    )
    assert result["status"] == "VALID"
    assert len(calls) == expected_count * (2 if family == "primary" else 1)
    assert {seed for _, seed in calls} == expected_seeds
    assert len({cell for cell, _ in calls}) == expected_count
    if family == "primary":
        assert result["hypotheses"]["H1"]["decision"] == "PASS"
        assert result["hypotheses"]["H2"]["decision"] == "PASS"


def test_registered_family_forwards_amended_cross_label_policy(monkeypatch):
    harness = _load_harness()
    policies = []

    def fake_registered_null(**kwargs):
        policies.append(kwargs["cross_label_policy"])
        return {
            "status": "VALID",
            "observed": float(kwargs["behaviour_code"]),
            "p_value": 0.01,
            "null_values": [2.0],
            "attempted": 1,
            "valid": 1,
            "invalid": 0,
            "invalid_reasons": {},
            "duplicate_audit": {
                "raw_rows": 8,
                "deduplicated_rows": 8,
                "duplicate_fraction": 0.0,
                "occurrence_duplicate_groups": 0,
                "occurrence_duplicate_rows": 0,
                "vector_duplicate_groups": 0,
                "vector_duplicate_rows": 0,
                "cross_chain_groups": 0,
                "cross_label_groups": 0,
            },
            "occurrence_list_sha256": "a" * 64,
            "row_counts": {
                "raw": 8,
                "deduplicated": 8,
                "analyzed": 2,
                "sentence_weighted": 2,
            },
            "chain_counts": {
                "unique": 2,
                "eligible": 2,
                "selected": 2,
                "mixed_label": 2,
            },
            "chain_attrition_reasons": {},
        }

    monkeypatch.setattr(
        harness, "run_within_chain_cdim_null", fake_registered_null
    )
    result = harness._run_registered_cdim_family(
        _synthetic_family_cells(harness, "primary"),
        family="primary",
        B=1,
        attempt_cap=1,
        cross_label_policy="drop_all_same_extraction_window",
    )
    assert result["status"] == "VALID"
    assert policies == ["drop_all_same_extraction_window"] * 8


def test_family_adjudication_uses_diagnostic_seed_and_frozen_behaviour_rules():
    harness = _load_harness()
    primary_cells = harness._expected_family_cells("primary")
    passing = harness._adjudicate_registered_family(
        primary_cells, [0.01] * 4, [0.01] * 4, family="primary"
    )
    assert passing["decision"] == "PASS"
    unstable = harness._adjudicate_registered_family(
        primary_cells, [0.01] * 4, [0.01, 0.01, 0.01, 0.2], family="primary"
    )
    assert unstable["decision"] == "MIXED"

    depth_cells = harness._expected_family_cells("five_depth")
    depth = harness._adjudicate_registered_family(
        depth_cells, [0.01] * 20, [0.01] * 20, family="five_depth"
    )
    assert set(depth["behaviour_decisions"].values()) == {"PASS"}

    annotator_cells = harness._expected_family_cells("three_annotator")
    annotator_p = [0.01] * 24
    annotator_p[4] = 0.2  # Sonnet L16, behaviour 1.
    annotator = harness._adjudicate_registered_family(
        annotator_cells,
        annotator_p,
        annotator_p,
        family="three_annotator",
    )
    assert annotator["behaviour_decisions"][1] == "MIXED"


def test_family_orchestration_propagates_programming_value_errors(monkeypatch):
    harness = _load_harness()

    def programming_failure(**kwargs):
        raise ValueError("programming contract failure")

    monkeypatch.setattr(harness, "run_within_chain_cdim_null", programming_failure)
    with pytest.raises(ValueError, match="programming contract failure"):
        harness._run_registered_cdim_family(
            _synthetic_family_cells(harness, "primary"),
            family="primary",
            B=1,
            attempt_cap=1,
        )


def test_h1_descriptive_adjudication_is_separate_and_uses_exact_boundaries():
    harness = _load_harness()
    assert harness._adjudicate_primary_h1([1.0, 2.0, 3.0, 10.0]) == {
        "decision": "PASS",
        "behaviour_decisions": {
            1: "PASS",
            2: "PASS",
            3: "PASS",
            4: "PASS",
        },
    }
    assert harness._adjudicate_primary_h1([10.0, 11.0, 12.0, 13.0])[
        "decision"
    ] == "MIXED"
    assert harness._adjudicate_primary_h1([11.0, 12.0, 13.0, 14.0])[
        "decision"
    ] == "FAIL"
    for invalid in ([0.0, 2.0, 3.0, 4.0], [None, 2.0, 3.0, 4.0]):
        adjudication = harness._adjudicate_primary_h1(invalid)
        assert adjudication["decision"] == "UNRUN"
        assert "UNRUN" in adjudication["behaviour_decisions"].values()


def test_incomplete_primary_family_returns_separate_unrun_h1_and_h2_details():
    harness = _load_harness()
    cells = _synthetic_family_cells(harness, "primary")
    cells.pop(0)
    result = harness._run_registered_cdim_family(
        cells, family="primary", B=1, attempt_cap=1
    )
    assert result["status"] == "UNRUN"
    assert result["hypotheses"] == {
        "H1": {
            "decision": "UNRUN",
            "behaviour_decisions": {1: "UNRUN", 2: "UNRUN", 3: "UNRUN", 4: "UNRUN"},
        },
        "H2": {
            "decision": "UNRUN",
            "behaviour_decisions": {1: "UNRUN", 2: "UNRUN", 3: "UNRUN", 4: "UNRUN"},
            "monte_carlo_unstable_cells": [],
        },
    }


def test_curvature_wiring_uses_frozen_settings_and_discards_invalid_pairs():
    harness = _load_harness()
    records = []
    rows = []
    for chain_index in range(15):
        for occurrence_index in range(2):
            records.append(_record("A", f"c{chain_index:02}", occurrence_index))
            rows.append([chain_index, occurrence_index, chain_index / 10])
    returned = iter([1.0, 1.2, np.nan, 1.2, 1.0, 1.3, 1.0, 1.4])
    calls = []

    def ratio_fn(
        values, *, k, variance_threshold, n_anchors, random_state, n_bootstrap
    ):
        kwargs = {
            "k": k,
            "variance_threshold": variance_threshold,
            "n_anchors": n_anchors,
            "random_state": random_state,
            "n_bootstrap": n_bootstrap,
        }
        calls.append((values.dtype, values.shape, kwargs))
        return next(returned)

    result = harness.run_curvature_diagnostic(
        np.asarray(rows),
        records,
        target_label="A",
        behaviour_code=1,
        cell_index=88,
        n_replicates=4,
        valid_threshold=3,
        ratio_fn=ratio_fn,
    )
    assert result["status"] == "VALID"
    assert result["n_valid_pairs"] == 3
    assert result["draws"][1]["status"] == "INVALID"
    assert result["bounded_negative"] is True
    assert (
        result["wording"]
        == "no beyond-chain curvature detected by this diagnostic at L16"
    )
    assert len(calls) == 8
    for dtype, shape, kwargs in calls:
        assert dtype == np.dtype(np.float32)
        assert shape[0] == 12
        assert kwargs == {
            "k": 10,
            "variance_threshold": 0.90,
            "n_anchors": 150,
            "random_state": 42,
            "n_bootstrap": 0,
        }
    parent = np.random.SeedSequence([20260726, 4, 6, 88, 0])
    expected_chain_rng = np.random.default_rng(parent.spawn(2)[0])
    expected_indices = np.sort(
        expected_chain_rng.choice(15, 12, replace=False)
    )
    expected_chains = [f"c{i:02}" for i in expected_indices]
    assert result["draws"][0]["selected_chain_ids"] == expected_chains


def test_stability_truncation_h4_and_curvature_propagate_programming_errors(
    monkeypatch,
):
    harness = _load_harness()
    records = []
    rows = []
    for chain_index in range(15):
        for occurrence_index in range(2):
            records.append(_record("A", f"c{chain_index:02}", occurrence_index))
            rows.append([chain_index, occurrence_index, chain_index / 10])
    stability_calls = 0

    def stability_cdim(values):
        nonlocal stability_calls
        stability_calls += 1
        if stability_calls == 1:
            return 1.0
        raise TypeError("stability programming failure")

    with pytest.raises(TypeError, match="stability programming failure"):
        harness.run_chain_stability_band(
            np.asarray(rows),
            records,
            target_label="A",
            family_code=1,
            cell_index=0,
            behaviour_code=1,
            n_replicates=1,
            valid_threshold=1,
            cdim_fn=stability_cdim,
            pr_fn=lambda values: 1.0,
        )

    metadata = {}
    truncation_records = []
    truncation_rows = []
    for index in range(20):
        chain_id = f"t{index:02}"
        metadata[chain_id] = {
            "n_tokens": 7000 if index < 10 else 8192,
            "chain": "done </think>" if index < 10 else "unfinished",
            "category": "x",
        }
        truncation_records.append(_record("A", chain_id, 0))
        truncation_rows.append([index, index + 1])
    truncation_calls = 0

    def truncation_cdim(values):
        nonlocal truncation_calls
        truncation_calls += 1
        if truncation_calls == 1:
            raise TypeError("truncation programming failure")
        return 1.0

    with pytest.raises(TypeError, match="truncation programming failure"):
        harness.run_truncation_sensitivity(
            np.asarray(truncation_rows),
            truncation_records,
            metadata,
            target_label="A",
            behaviour_code=1,
            n_replicates=1,
            valid_threshold=1,
            cdim_fn=truncation_cdim,
            pr_fn=lambda values: 1.0,
        )

    labels = ["A", "B", "C", "D"]
    h4_records = []
    h4_rows = []
    for chain_index in range(2):
        for label_index, label in enumerate(labels):
            h4_records.append(_record(label, f"h{chain_index}", label_index))
            h4_rows.append([chain_index * 10 + label_index, label_index + 0.25])
    grid = harness.build_common_complete_case_grid(
        _six_representations(h4_records, h4_rows),
        behaviour_codes={"A": 1, "B": 2, "C": 3, "D": 4},
    )

    def h4_failure(values):
        raise TypeError("H4 programming failure")

    with pytest.raises(TypeError, match="H4 programming failure"):
        harness.run_pooling_window_family(
            grid, B=1, attempt_cap=1, statistic_fn=h4_failure
        )

    def curvature_failure(
        values, *, k, variance_threshold, n_anchors, random_state, n_bootstrap
    ):
        raise TypeError("curvature programming failure")

    with pytest.raises(TypeError, match="curvature programming failure"):
        harness.run_curvature_diagnostic(
            np.asarray(rows),
            records,
            target_label="A",
            behaviour_code=1,
            cell_index=88,
            n_replicates=1,
            valid_threshold=1,
            ratio_fn=curvature_failure,
        )

def test_resource_benchmark_formulas_gate_and_exception_firewall(capsys):
    harness = _load_harness()
    statuses = iter(["VALID", "INVALID", "VALID", "VALID"])
    clock = iter([10.0, 15.0])
    result = harness.run_resource_benchmark(
        lambda attempt_index: next(statuses),
        b_target=4,
        synthetic=True,
        input_hashes={"synthetic": "a" * 64},
        workers=2,
        scratch_bytes=100,
        output_bytes_per_attempt=10,
        clock_fn=lambda: next(clock),
        peak_rss_fn=lambda: 1234,
    )
    assert result["attempted"] == 4
    assert result["valid"] == 3
    assert result["invalid"] == 1
    assert result["seconds_per_attempt"] == 1.25
    assert result["projected_primary_seconds"] == 1.25 * 1.25 * 2750 * 4
    assert result["projected_secondary_seconds"] == 1.25 * 1.25 * 2750 * 44
    assert result["projected_H4_seconds"] == 1.25 * 1.25 * 2750 * 24
    assert result["projected_scratch_bytes"] == 137500
    assert result["gate"] == "FAIL"
    assert result["gates"] == {
        "primary": "FAIL",
        "secondary": "FAIL",
        "H4_cpu": "FAIL",
        "H4": "UNRUN",
    }
    assert capsys.readouterr() == ("", "")
    harness._assert_resource_firewall(result)

    clock2 = iter([0.0, 0.1])

    def secret_failure(_attempt_index):
        raise RuntimeError("observed_cdim=3.14; raw_p=0.001")

    failed = harness.run_resource_benchmark(
        secret_failure,
        b_target=1,
        synthetic=True,
        clock_fn=lambda: next(clock2),
        peak_rss_fn=lambda: 0,
    )
    serialized = json.dumps(failed)
    assert "3.14" not in serialized
    assert "0.001" not in serialized
    assert failed["invalid_reasons"] == {"exception": 1}


def test_resource_benchmark_retains_registered_invalid_reason_without_secret():
    harness = _load_harness()
    clock = iter([0.0, 0.1])
    result = harness.run_resource_benchmark(
        lambda _attempt_index: ("INVALID", "estimator_invalid"),
        b_target=1,
        synthetic=True,
        clock_fn=lambda: next(clock),
        peak_rss_fn=lambda: 0,
    )
    assert result["valid"] == 0
    assert result["invalid_reasons"] == {"estimator_invalid": 1}
    harness._assert_resource_firewall(result)


def test_resource_writer_is_atomic_and_rejects_forbidden_names(tmp_path):
    harness = _load_harness()
    clock = iter([1.0, 1.01])
    result = harness.run_resource_benchmark(
        lambda _attempt_index: "VALID",
        b_target=1,
        synthetic=True,
        clock_fn=lambda: next(clock),
        peak_rss_fn=lambda: 0,
    )
    destination = tmp_path / "synthetic-resource.json"
    harness.write_resource_only_benchmark(destination, result)
    assert json.loads(destination.read_text()) == result
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))
    with pytest.raises(ValueError, match="firewall"):
        harness._assert_resource_firewall(
            {"safe": {"observed_statistic": 1.0}}
        )
    with pytest.raises(ValueError, match="firewall"):
        harness.write_resource_only_benchmark(
            destination, {**result, "nested": {"holm_p": 0.2}}
        )
    markdown = tmp_path / "BENCHMARK.md"
    harness.write_resource_benchmark_markdown(markdown, result)
    rendered = markdown.read_text()
    assert "Resource-only B=1 benchmark" in rendered
    for forbidden in ("observed", "cdim", "null draw", "p-value", "holm"):
        assert forbidden not in rendered.casefold()


def test_cli_separates_synthetic_validation_from_acknowledged_pilot(
    tmp_path, monkeypatch
):
    harness = _load_harness()
    destination = tmp_path / "synthetic-resource.json"
    assert (
        harness.main(
            ["--validate-synthetic", "--resource-output", str(destination)]
        )
        == 0
    )
    assert json.loads(destination.read_text())["status"] == "COMPLETE"
    with pytest.raises(SystemExit):
        harness.main(["--resource-pilot"])
    dispatched = []

    def fake_execute(destination):
        dispatched.append(Path(destination))

    monkeypatch.setattr(
        harness, "execute_empirical_resource_pilot_from_disk", fake_execute
    )
    assert (
        harness.main(
            ["--resource-pilot", "--acknowledge-empirical-pilot"]
        )
        == 0
    )
    assert dispatched == [
        harness._PLANNED_EMPIRICAL_RESOURCE_DESTINATION.resolve()
    ]


def test_empirical_resource_pilot_returns_only_firewalled_resource_fields(
    monkeypatch,
):
    harness = _load_harness()
    monkeypatch.setattr(
        harness, "_frozen_empirical_pilot_authorized", lambda: True
    )
    rng = np.random.default_rng(20260726)
    labels = []
    chains = []
    records = []
    rows = []
    behaviour_order = (
        "backtracking",
        "uncertainty-estimation",
        "example-testing",
        "adding-knowledge",
    )
    for chain_index in range(100):
        chain_id = f"MATH_{chain_index:03}"
        for annotation_index, label in enumerate(behaviour_order):
            labels.append(label)
            chains.append(chain_id)
            records.append(
                _record(label, chain_id, annotation_index)
            )
            rows.append(rng.normal(size=5))
    clock = iter([10.0, 12.5])
    result = harness.run_empirical_backtracking_resource_pilot(
        np.asarray(rows),
        labels,
        chains,
        records,
        input_hashes={"synthetic_fixture": "a" * 64},
        statistic_fn=lambda values: float(values.shape[0]),
        clock_fn=lambda: next(clock),
        peak_rss_fn=lambda: 1234,
    )
    assert result["mode"] == "empirical"
    assert result["b_target"] == 25
    assert result["attempted"] == result["valid"] == 25
    assert result["invalid"] == 0
    harness._assert_resource_firewall(result)
    serialized = json.dumps(result).casefold()
    for forbidden in ("observed", "cdim", "null_values", "p_value", "holm"):
        assert forbidden not in serialized


def test_empirical_callback_uses_frozen_amended_duplicate_policy(monkeypatch):
    harness = _load_harness()
    monkeypatch.setattr(
        harness,
        "_frozen_empirical_cross_label_policy",
        lambda: "drop_all_same_extraction_window",
    )
    original = harness._validate_and_deduplicate
    policies = []

    def recording_audit(values, records, **kwargs):
        policies.append(kwargs.get("cross_label_policy"))
        return original(values, records, **kwargs)

    monkeypatch.setattr(
        harness, "_validate_and_deduplicate", recording_audit
    )
    rng = np.random.default_rng(20260726)
    labels = []
    chains = []
    records = []
    rows = []
    for chain_index in range(100):
        chain_id = f"MATH_{chain_index:03}"
        for annotation_index, label in enumerate(
            (
                "backtracking",
                "uncertainty-estimation",
                "example-testing",
                "adding-knowledge",
            )
        ):
            labels.append(label)
            chains.append(chain_id)
            records.append(
                _record(
                    label,
                    chain_id,
                    annotation_index,
                    n_positions=11,
                )
            )
            rows.append(rng.normal(size=5))
    harness._empirical_backtracking_attempt_callback(
        np.asarray(rows),
        labels,
        chains,
        records,
        statistic_fn=lambda values: float(values.shape[0]),
    )
    assert policies == ["drop_all_same_extraction_window"]


def test_frozen_empirical_authorization_supports_one_unconsumed_retry_gate(
    tmp_path, monkeypatch
):
    harness = _load_harness()
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(
        harness, "_FROZEN_HARDENING_CONFIG", config_path
    )

    def write_execution(original, original_consumed, retry, retry_consumed):
        config_path.write_text(
            yaml.safe_dump(
                {
                    "execution": {
                        "empirical_pilot_authorized_in_this_batch": original,
                        "empirical_pilot_consumed_in_this_batch": (
                            original_consumed
                        ),
                        "empirical_pilot_retry_authorized_in_this_batch": retry,
                        "empirical_pilot_retry_consumed_in_this_batch": (
                            retry_consumed
                        ),
                    }
                }
            )
        )

    write_execution(False, True, True, False)
    assert harness._frozen_empirical_pilot_authorized() is True
    write_execution(False, True, False, True)
    assert harness._frozen_empirical_pilot_authorized() is False
    write_execution(True, False, True, False)
    with pytest.raises(ValueError, match="exactly one"):
        harness._frozen_empirical_pilot_authorized()


def test_empirical_loader_hash_gates_before_loading(tmp_path):
    harness = _load_harness()
    config = {
        "preregistration": {
            "path": "prereg.md",
            "sha256": "0" * 64,
        },
        "primary_input_hashes": {
            "data/activations/R1-1.5B/metadata.json": "1" * 64,
        },
        "representation": {
            "primary": {
                "pooling": "mean",
                "clip_window_to_sentence_end": False,
            }
        },
        "model": {"hidden_width": 1536},
        "labels": [
            "backtracking",
            "uncertainty-estimation",
            "example-testing",
            "adding-knowledge",
        ],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="hash mismatch"):
        harness.load_empirical_backtracking_l27_inputs(
            repository_root=tmp_path,
            config_path=config_path,
        )


def test_empirical_executor_records_duplicate_preflight_stop_without_attempts(
    tmp_path, monkeypatch
):
    harness = _load_harness()
    destination = tmp_path / "backtracking_L27_resource_only.json"
    monkeypatch.setattr(
        harness, "_PLANNED_EMPIRICAL_RESOURCE_DESTINATION", destination
    )
    monkeypatch.setattr(
        harness,
        "load_empirical_backtracking_l27_inputs",
        lambda: {
            "activations": np.ones((1, 1)),
            "labels": ["backtracking"],
            "chain_ids": ["MATH_000"],
            "occurrences": [_record("backtracking", "MATH_000", 0)],
            "input_hashes": {"fixture": "a" * 64},
        },
    )

    def duplicate_stop(*args, **kwargs):
        raise harness.DuplicateAuditError(
            "do not expose vector details",
            {"cross_label_groups": 1},
        )

    monkeypatch.setattr(
        harness,
        "run_empirical_backtracking_resource_pilot",
        duplicate_stop,
    )
    result = harness.execute_empirical_resource_pilot_from_disk(destination)
    assert result["status"] == "STOPPED"
    assert result["attempted"] == result["valid"] == result["invalid"] == 0
    assert result["invalid_reasons"] == {"duplicate_hard_failure": 1}
    assert result["gates"] == {
        "primary": "FAIL",
        "secondary": "UNRUN",
        "H4_cpu": "UNRUN",
        "H4": "UNRUN",
    }
    assert json.loads(destination.read_text()) == result
    assert destination.with_name("BENCHMARK.md").is_file()
    harness._assert_resource_firewall(result)


def test_resource_reports_independent_primary_secondary_and_h4_gates():
    harness = _load_harness()
    clock = iter([0.0, 2.0])
    result = harness.run_resource_benchmark(
        lambda _attempt_index: "VALID",
        b_target=1,
        synthetic=True,
        clock_fn=lambda: next(clock),
        peak_rss_fn=lambda: 0,
    )
    assert result["gates"] == {
        "primary": "PASS",
        "secondary": "FAIL",
        "H4_cpu": "PASS",
        "H4": "UNRUN",
    }
    assert result["gate"] == result["gates"]["primary"]


def test_h4_overall_gate_requires_explicit_validated_gpu_prerequisites():
    harness = _load_harness()
    clock = iter([0.0, 2.0])
    prerequisites = {
        "free_scratch_bytes": 100 * 1024**3,
        "six_representation_alignment_smoke_pass": True,
        "projected_extraction_seconds": 24 * 60 * 60,
        "vram_bytes": 24 * 1024**3,
    }
    result = harness.run_resource_benchmark(
        lambda _attempt_index: "VALID",
        b_target=1,
        synthetic=True,
        gpu_prerequisites=prerequisites,
        clock_fn=lambda: next(clock),
        peak_rss_fn=lambda: 0,
    )
    assert result["gpu_prerequisites"] == prerequisites
    assert result["gates"]["H4_cpu"] == "PASS"
    assert result["gates"]["H4"] == "PASS"

    clock2 = iter([0.0, 2.0])
    insufficient = harness.run_resource_benchmark(
        lambda _attempt_index: "VALID",
        b_target=1,
        synthetic=True,
        gpu_prerequisites={**prerequisites, "vram_bytes": 24 * 1024**3 - 1},
        clock_fn=lambda: next(clock2),
        peak_rss_fn=lambda: 0,
    )
    assert insufficient["gates"]["H4_cpu"] == "PASS"
    assert insufficient["gates"]["H4"] == "UNRUN"


def test_resource_firewall_rejects_statistical_values_but_allows_normal_facts():
    harness = _load_harness()
    harness._assert_resource_firewall(
        {
            "software": {"python": "3.13.12", "numpy": "2.4.1"},
            "process": {"pid": 10, "workers": 2},
            "host": {
                "hostname": "cpu-nullhost-01",
                "system": "Darwin",
                "machine": "arm64",
                "cpu_count": 12,
            },
        }
    )
    for hidden in (
        "correlation dimension was 3.1",
        "kept null draws internally",
        "p-value equals 0.04",
        "observed_cdim=3.14",
        "observed-cdim",
        "p_value",
        "raw.p",
        "holm/p",
    ):
        with pytest.raises(ValueError, match="firewall"):
            harness._assert_resource_firewall({"software": {"python": hidden}})
    for hidden_key in ("observed-cdim", "p-value", "p.value", "raw-p", "holm.p"):
        with pytest.raises(ValueError, match="firewall"):
            harness._assert_resource_firewall({hidden_key: 1})


def test_empirical_resource_benchmark_requires_ack_and_open_frozen_gate():
    harness = _load_harness()
    callback_calls = 0
    clock_calls = 0

    def callback(_attempt_index):
        nonlocal callback_calls
        callback_calls += 1
        return "VALID"

    def clock():
        nonlocal clock_calls
        clock_calls += 1
        return 0.0

    for acknowledgement in (None, False, "yes", 1):
        kwargs = {}
        if acknowledgement is not None:
            kwargs["empirical_authorization_acknowledged"] = acknowledgement
        with pytest.raises(ValueError, match="authorization"):
            harness.run_resource_benchmark(
                callback,
                b_target=25,
                clock_fn=clock,
                peak_rss_fn=lambda: 0,
                **kwargs,
            )
    with pytest.raises(ValueError, match="frozen config"):
        harness.run_resource_benchmark(
            callback,
            b_target=25,
            empirical_authorization_acknowledged=True,
            clock_fn=clock,
            peak_rss_fn=lambda: 0,
        )
    with pytest.raises(ValueError, match="synthetic mode"):
        harness.run_resource_benchmark(
            callback,
            b_target=25,
            synthetic="false",
            empirical_authorization_acknowledged=True,
            clock_fn=clock,
            peak_rss_fn=lambda: 0,
        )
    assert callback_calls == 0
    assert clock_calls == 0


def test_synthetic_resource_benchmark_never_requires_empirical_authorization():
    harness = _load_harness()
    clock = iter([0.0, 1.0])
    result = harness.run_resource_benchmark(
        lambda _attempt_index: "VALID",
        b_target=2,
        synthetic=True,
        empirical_authorization_acknowledged="not-applicable",
        clock_fn=lambda: next(clock),
        peak_rss_fn=lambda: 0,
    )
    assert result["b_target"] == 2
    assert result["mode"] == "synthetic"


def test_empirical_b50_is_blocked_by_frozen_authorization_before_callback():
    harness = _load_harness()
    callback_calls = 0

    def callback(_):
        nonlocal callback_calls
        callback_calls += 1
        return "VALID"

    for field, value in (
        ("amendment_allow_b50", True),
        ("b25_completed_successfully", True),
    ):
        kwargs = {
            "b_target": 50,
            "amendment_allow_b50": True,
            "b25_completed_successfully": True,
            "empirical_authorization_acknowledged": True,
            field: value,
            "clock_fn": lambda: pytest.fail("clock must not run"),
            "peak_rss_fn": lambda: 0,
        }
        with pytest.raises(ValueError, match="frozen config"):
            harness.run_resource_benchmark(callback, **kwargs)
    assert callback_calls == 0


def test_resource_writer_enforces_synthetic_marker_and_empirical_destination(
    tmp_path,
):
    harness = _load_harness()
    clock = iter([0.0, 0.1])
    synthetic = harness.run_resource_benchmark(
        lambda _: "VALID",
        b_target=1,
        synthetic=True,
        clock_fn=lambda: next(clock),
        peak_rss_fn=lambda: 0,
    )
    planned = (
        ROOT
        / "results"
        / "robustness"
        / "cdim_null_pilot"
        / "R1-1.5B"
        / "backtracking_L27_resource_only.json"
    )
    with pytest.raises(ValueError, match="synthetic"):
        harness.write_resource_only_benchmark(planned, synthetic)
    with pytest.raises(ValueError, match="marked synthetic"):
        harness.write_resource_only_benchmark(tmp_path / "resource.json", synthetic)

    empirical_clock = iter([0.0, 1.0])
    empirical = harness.run_resource_benchmark(
        lambda _: "VALID",
        b_target=25,
        synthetic=True,
        clock_fn=lambda: next(empirical_clock),
        peak_rss_fn=lambda: 0,
    )
    empirical = {**empirical, "mode": "empirical"}
    with pytest.raises(ValueError, match="planned empirical destination"):
        harness.write_resource_only_benchmark(
            tmp_path / "synthetic-but-empirical.json", empirical
        )


def test_json_schemas_parse_and_freeze_top_level_contracts():
    result_path = (
        ROOT / "schemas" / "thesis_core_hardening_result_cell_v1.schema.json"
    )
    resource_path = (
        ROOT
        / "schemas"
        / "thesis_core_hardening_resource_benchmark_v1.schema.json"
    )
    provenance_path = (
        ROOT / "schemas" / "thesis_core_hardening_provenance_v1.schema.json"
    )
    result_schema = json.loads(result_path.read_text())
    resource_schema = json.loads(resource_path.read_text())
    provenance_schema = json.loads(provenance_path.read_text())
    for schema in (result_schema, resource_schema, provenance_schema):
        assert schema["$schema"].endswith("2020-12/schema")
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False

    assert set(result_schema["required"]) == set(result_schema["properties"])
    assert result_schema["properties"]["status"]["enum"] == [
        "UNRUN",
        "VALID",
        "INVALID",
        "STOPPED",
    ]
    assert result_schema["properties"]["cell_index"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 91,
    }
    conditional_text = json.dumps(result_schema["allOf"])
    assert '"minItems": 2500' in conditional_text
    assert '"maxItems": 2500' in conditional_text
    assert '"minimum": 60' in conditional_text
    assert '"maximum": 63' in conditional_text

    assert set(resource_schema["required"]) == set(resource_schema["properties"])
    forbidden_pattern = resource_schema["propertyNames"]["not"]["pattern"]
    for forbidden in (
        "observed_statistic",
        "some_cdim_value",
        "null_draw",
        "x_p_value_y",
        "raw_p",
        "holm_p",
    ):
        import re

        assert re.search(forbidden_pattern, forbidden)

    assert set(provenance_schema["required"]) == set(
        provenance_schema["properties"]
    )
    assert provenance_schema["properties"]["code_commit"]["type"] == [
        "string",
        "null",
    ]


def _valid_unrun_result_cell():
    return {
        "schema_version": "thesis-core-hardening-result-cell-v1",
        "status": "UNRUN",
        "family": "truncation",
        "hypothesis": "H3",
        "cell_index": 48,
        "codes": {
            "family": 4,
            "purpose": 3,
            "annotator": 1,
            "behaviour": 1,
            "pooling": 0,
            "window": 0,
            "truncation": 1,
        },
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "annotator": "Sonnet",
        "label": "backtracking",
        "pooling": "mean",
        "window": "unclipped",
        "layer_zero_based": 27,
        "input_hashes": {"synthetic": "a" * 64},
        "estimator_settings": {
            "correlation_dimension": {
                "dtype": "float64",
                "n_radii": 20,
                "random_state": 42,
                "n_bootstrap": 0,
                "subsample": 2000,
                "distance": "euclidean",
            },
            "participation_ratio": {
                "dtype": "float64",
                "column_centered": True,
                "svd": "full",
                "component_cap": None,
            },
            "curvature": None,
        },
        "preprocessing": {
            "cdim_centered": False,
            "cdim_scaled": False,
            "cdim_normalized": False,
            "cdim_whitened": False,
            "duplicate_order": ["occurrence_key", "exact_vector"],
        },
        "duplicate_audit": {
            "raw_rows": 0,
            "deduplicated_rows": 0,
            "duplicate_fraction": 0.0,
            "occurrence_duplicate_groups": 0,
            "occurrence_duplicate_rows": 0,
            "vector_duplicate_groups": 0,
            "vector_duplicate_rows": 0,
            "cross_chain_groups": 0,
            "cross_label_groups": 0,
        },
        "provenance": {
            "preregistration_path": "results/prereg/example.md",
            "preregistration_sha256": "b" * 64,
            "code_commit": None,
            "dirty": True,
            "dirty_paths": ["synthetic"],
            "python_version": "3.13.12",
            "package_versions": {"numpy": "2.4.1"},
            "argv": ["--synthetic"],
            "utc_start": "2026-07-26T00:00:00Z",
            "utc_end": "2026-07-26T00:00:01Z",
            "host": {
                "hostname": "cpu-01",
                "system": "Darwin",
                "machine": "arm64",
                "cpu_count": 12,
            },
            "process": {"pid": 1, "workers": 1},
            "wall_seconds": 1.0,
            "peak_rss_bytes": 1024,
        },
        "occurrence_list_sha256": "c" * 64,
        "row_counts": {
            "raw": 0,
            "deduplicated": 0,
            "analyzed": 0,
            "sentence_weighted": 0,
        },
        "chain_counts": {
            "unique": 0,
            "eligible": 0,
            "selected": 0,
            "mixed_label": 0,
        },
        "resamples": {
            "requested": 0,
            "attempt_cap": 0,
            "attempted": 0,
            "valid": 0,
            "invalid": 0,
            "invalid_reasons": {},
            "seed_coordinates": {
                "base_seed": 20260726,
                "purpose_code": 3,
                "family_code": 4,
                "cell_index": 48,
            },
        },
        "observed_cdim": None,
        "observed_pr": None,
        "sentence_weighted_cdim": None,
        "sentence_weighted_label": None,
        "chain_attrition_reasons": {},
        "raw_p": None,
        "holm_p": None,
        "null_cdim": [],
        "stability_draws": [],
        "matched_results": None,
        "decision": "UNRUN",
        "amendment_ids": [],
    }


def _valid_result_cell(family):
    document = _valid_unrun_result_cell()
    document["status"] = "VALID"
    document["decision"] = "PASS"
    document["row_counts"] = {
        "raw": 100,
        "deduplicated": 100,
        "analyzed": 100,
        "sentence_weighted": 100 if family == "pooling_window" else 0,
    }
    document["chain_counts"] = {
        "unique": 100,
        "eligible": 100,
        "selected": 80,
        "mixed_label": 60,
    }
    coordinates = {
        "primary": {
            "hypothesis": "H2",
            "cell_index": 0,
            "codes": {
                "family": 1,
                "purpose": 1,
                "annotator": 1,
                "behaviour": 1,
                "pooling": 1,
                "window": 1,
                "truncation": 0,
            },
            "layer": 27,
        },
        "five_depth": {
            "hypothesis": "SECONDARY",
            "cell_index": 4,
            "codes": {
                "family": 2,
                "purpose": 1,
                "annotator": 1,
                "behaviour": 1,
                "pooling": 1,
                "window": 1,
                "truncation": 0,
            },
            "layer": 11,
        },
        "three_annotator": {
            "hypothesis": "SECONDARY",
            "cell_index": 24,
            "codes": {
                "family": 3,
                "purpose": 1,
                "annotator": 1,
                "behaviour": 1,
                "pooling": 1,
                "window": 1,
                "truncation": 0,
            },
            "layer": 12,
        },
        "truncation": {
            "hypothesis": "H3",
            "cell_index": 48,
            "codes": {
                "family": 4,
                "purpose": 3,
                "annotator": 1,
                "behaviour": 1,
                "pooling": 0,
                "window": 0,
                "truncation": 1,
            },
            "layer": 27,
        },
        "pooling_window": {
            "hypothesis": "H4",
            "cell_index": 64,
            "codes": {
                "family": 5,
                "purpose": 1,
                "annotator": 1,
                "behaviour": 1,
                "pooling": 1,
                "window": 1,
                "truncation": 0,
            },
            "layer": 27,
        },
        "curvature": {
            "hypothesis": "CURVATURE_DIAGNOSTIC",
            "cell_index": 88,
            "codes": {
                "family": 6,
                "purpose": 4,
                "annotator": 1,
                "behaviour": 1,
                "pooling": 0,
                "window": 0,
                "truncation": 0,
            },
            "layer": 16,
        },
    }[family]
    document.update(
        {
            "family": family,
            "hypothesis": coordinates["hypothesis"],
            "cell_index": coordinates["cell_index"],
            "codes": coordinates["codes"],
            "layer_zero_based": coordinates["layer"],
        }
    )
    purpose = coordinates["codes"]["purpose"]
    family_code = coordinates["codes"]["family"]
    if family in {"primary", "five_depth", "three_annotator", "pooling_window"}:
        document["observed_cdim"] = 3.0
        document["raw_p"] = 0.01
        document["holm_p"] = 0.04
        document["null_cdim"] = [4.0] * 2500
        document["resamples"] = {
            "requested": 2500,
            "attempt_cap": 2750,
            "attempted": 2500,
            "valid": 2500,
            "invalid": 0,
            "invalid_reasons": {},
            "seed_coordinates": {
                "base_seed": 20260726,
                "purpose_code": purpose,
                "family_code": family_code,
                "cell_index": coordinates["cell_index"],
            },
        }
    else:
        document["observed_cdim"] = 3.0 if family == "truncation" else None
        document["observed_pr"] = 2.0 if family == "truncation" else None
        document["stability_draws"] = [
            {
                "replicate": index,
                "status": "VALID",
                "cdim": 3.0,
                "pr": 2.0,
                "failure": None,
            }
            for index in range(500)
        ]
        document["resamples"] = {
            "requested": 500,
            "attempt_cap": 500,
            "attempted": 500,
            "valid": 500,
            "invalid": 0,
            "invalid_reasons": {},
            "seed_coordinates": {
                "base_seed": 20260726,
                "purpose_code": purpose,
                "family_code": family_code,
                "cell_index": coordinates["cell_index"],
            },
        }
    if family == "pooling_window":
        document["observed_pr"] = 2.0
        document["sentence_weighted_cdim"] = 3.2
        document[
            "sentence_weighted_label"
        ] = "sentence-weighted correlation dimension"
    return document


def _synthetic_primary_family_result():
    harness = _load_harness()
    cells = []
    for coordinates in harness._expected_family_cells("primary"):
        behaviour_code = coordinates["behaviour_code"]
        cells.append(
            {
                **coordinates,
                "status": "VALID",
                "observed_cdim": float(behaviour_code + 2),
                "raw_p": 0.01,
                "holm_p": 0.04,
                "null_cdim": [5.0] * 2500,
                "resamples": {
                    "attempted": 2500,
                    "valid": 2500,
                    "invalid": 0,
                    "invalid_reasons": {},
                },
                "diagnostic_raw_p": 0.01,
                "diagnostic_holm_p": 0.04,
                "diagnostic_null_cdim": [5.0] * 2500,
                "diagnostic_resamples": {
                    "attempted": 2500,
                    "valid": 2500,
                    "invalid": 0,
                    "invalid_reasons": {},
                },
                "duplicate_audit": {
                    "raw_rows": 400,
                    "deduplicated_rows": 396,
                    "duplicate_fraction": 0.01,
                    "occurrence_duplicate_groups": 0,
                    "occurrence_duplicate_rows": 0,
                    "vector_duplicate_groups": 4,
                    "vector_duplicate_rows": 4,
                    "cross_chain_groups": 0,
                    "cross_label_groups": 1,
                    "cross_label_policy": (
                        "drop_all_same_extraction_window"
                    ),
                    "cross_label_same_window_groups_dropped": 1,
                    "cross_label_same_window_rows_dropped": 2,
                    "unresolved_cross_label_groups": 0,
                },
                "occurrence_list_sha256": (
                    f"{behaviour_code:x}" * 64
                )[:64],
                "row_counts": {
                    "raw": 400,
                    "deduplicated": 396,
                    "analyzed": 100,
                    "sentence_weighted": 120,
                },
                "chain_counts": {
                    "unique": 100,
                    "eligible": 100,
                    "selected": 100,
                    "mixed_label": 90,
                },
                "chain_attrition_reasons": {},
            }
        )
    decisions = {code: "PASS" for code in range(1, 5)}
    return {
        "status": "VALID",
        "decision": "PASS",
        "cells": cells,
        "holm_p": [0.04] * 4,
        "diagnostic_holm_p": [0.04] * 4,
        "behaviour_decisions": decisions,
        "monte_carlo_unstable_cells": [],
        "hypotheses": {
            "H1": {
                "decision": "PASS",
                "behaviour_decisions": decisions,
            },
            "H2": {
                "decision": "PASS",
                "behaviour_decisions": decisions,
                "monte_carlo_unstable_cells": [],
            },
        },
    }


def _synthetic_secondary_family_result(family):
    harness = _load_harness()
    cells = []
    for coordinates in harness._expected_family_cells(family):
        cells.append(
            {
                **coordinates,
                "status": "VALID",
                "observed_cdim": 3.0,
                "raw_p": 0.001,
                "holm_p": 0.04,
                "null_cdim": [5.0] * 2500,
                "resamples": {
                    "attempted": 2500,
                    "valid": 2500,
                    "invalid": 0,
                    "invalid_reasons": {},
                },
                "duplicate_audit": {
                    "raw_rows": 400,
                    "deduplicated_rows": 396,
                    "duplicate_fraction": 0.01,
                    "occurrence_duplicate_groups": 0,
                    "occurrence_duplicate_rows": 0,
                    "vector_duplicate_groups": 4,
                    "vector_duplicate_rows": 4,
                    "cross_chain_groups": 0,
                    "cross_label_groups": 1,
                },
                "occurrence_list_sha256": (
                    f"{coordinates['cell_index']:x}" * 64
                )[:64],
                "row_counts": {
                    "raw": 400,
                    "deduplicated": 396,
                    "analyzed": 100,
                    "sentence_weighted": 120,
                },
                "chain_counts": {
                    "unique": 100,
                    "eligible": 100,
                    "selected": 100,
                    "mixed_label": 90,
                },
                "chain_attrition_reasons": {},
            }
        )
    decisions = {code: "PASS" for code in range(1, 5)}
    return {
        "status": "VALID",
        "decision": "PASS",
        "cells": cells,
        "holm_p": [0.04] * len(cells),
        "diagnostic_holm_p": [],
        "behaviour_decisions": decisions,
        "monte_carlo_unstable_cells": [],
    }


def test_primary_family_document_builds_twelve_schema_valid_cells(tmp_path):
    harness = _load_harness()
    schema_path = (
        ROOT
        / "schemas"
        / "thesis_core_hardening_result_cell_v1.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    provenance = _valid_unrun_result_cell()["provenance"]
    document = harness.build_primary_family_document(
        _synthetic_primary_family_result(),
        input_hashes={"synthetic": "a" * 64},
        preregistration_path="results/prereg/example.md",
        preregistration_sha256="b" * 64,
        result_schema_sha256=hashlib.sha256(
            schema_path.read_bytes()
        ).hexdigest(),
        provenance=provenance,
        amendment_ids=["full-run-test"],
    )
    assert document["schema_version"] == (
        "thesis-core-hardening-primary-family-v1"
    )
    assert len(document["h1_cells"]) == 4
    assert len(document["h2_registered_cells"]) == 4
    assert len(document["h2_diagnostic_cells"]) == 4
    for cell in (
        document["h1_cells"]
        + document["h2_registered_cells"]
        + document["h2_diagnostic_cells"]
    ):
        _assert_schema_valid(cell, schema)
        harness._validate_result_cell_invariants(cell)
    destination = tmp_path / "primary_L27_cdim_null.json"
    monkeypatch_path = harness._PLANNED_PRIMARY_DESTINATION
    harness._PLANNED_PRIMARY_DESTINATION = destination
    try:
        harness.write_primary_family_result(destination, document)
        assert json.loads(destination.read_text()) == document
        with pytest.raises(FileExistsError):
            harness.write_primary_family_result(destination, document)
    finally:
        harness._PLANNED_PRIMARY_DESTINATION = monkeypatch_path


def test_full_primary_authorization_and_executor_are_one_time_and_atomic(
    tmp_path, monkeypatch
):
    harness = _load_harness()
    config_path = tmp_path / "authorization.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "execution": {
                    "cpu_only": True,
                    "gpu_enabled": False,
                    "api_enabled": False,
                    "full_primary_authorized_in_this_batch": True,
                    "full_primary_consumed_in_this_batch": False,
                    "load_real_activation_matrices_in_this_batch": True,
                    "synthetic_validation_only": False,
                }
            }
        )
    )
    monkeypatch.setattr(
        harness, "_FROZEN_HARDENING_CONFIG", config_path
    )
    assert harness._frozen_full_primary_authorized() is True
    blocked = yaml.safe_load(config_path.read_text())
    blocked["execution"]["full_primary_consumed_in_this_batch"] = True
    config_path.write_text(yaml.safe_dump(blocked))
    with pytest.raises(ValueError, match="not authorized"):
        harness._frozen_full_primary_authorized()

    destination = tmp_path / "primary_L27_cdim_null.json"
    monkeypatch.setattr(
        harness, "_PLANNED_PRIMARY_DESTINATION", destination
    )
    monkeypatch.setattr(
        harness, "_frozen_full_primary_authorized", lambda: True
    )
    monkeypatch.setattr(
        harness,
        "_frozen_registered_cpu_cross_label_policy",
        lambda: "drop_all_same_extraction_window",
    )
    monkeypatch.setattr(
        harness,
        "load_empirical_backtracking_l27_inputs",
        lambda: {
            "activations": np.ones((4, 2)),
            "labels": list(harness._PRIMARY_LABELS),
            "chain_ids": ["c1"] * 4,
            "occurrences": [
                _record(
                    label,
                    "c1",
                    index,
                    n_positions=11,
                )
                for index, label in enumerate(harness._PRIMARY_LABELS)
            ],
            "input_hashes": {"synthetic": "a" * 64},
        },
    )

    def fake_family(*args, **kwargs):
        progress = kwargs["progress_callback"]
        result = _synthetic_primary_family_result()
        progress(result["cells"][:1])
        return result

    monkeypatch.setattr(
        harness, "_run_registered_cdim_family", fake_family
    )
    actual_config = (
        ROOT
        / "configs"
        / "analysis"
        / "thesis_core_hardening_2026-07-26.yaml"
    )
    monkeypatch.setattr(
        harness, "_FROZEN_HARDENING_CONFIG", actual_config
    )
    document = harness.execute_registered_primary_from_disk(
        destination, argv=["--run-primary", "--acknowledge-full-analysis"]
    )
    assert json.loads(destination.read_text()) == document
    assert not destination.with_name(
        f".{destination.name}.progress.json"
    ).exists()
    with pytest.raises(FileExistsError):
        harness.execute_registered_primary_from_disk(destination)


def test_secondary_family_documents_are_complete_schema_valid_and_atomic(
    tmp_path,
):
    harness = _load_harness()
    schema_path = (
        ROOT
        / "schemas"
        / "thesis_core_hardening_result_cell_v1.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    provenance = _valid_unrun_result_cell()["provenance"]
    for family, expected_count in (("five_depth", 20), ("three_annotator", 24)):
        document = harness.build_secondary_family_document(
            _synthetic_secondary_family_result(family),
            family=family,
            input_hashes={"synthetic": "a" * 64},
            preregistration_path="results/prereg/example.md",
            preregistration_sha256="b" * 64,
            result_schema_sha256=hashlib.sha256(
                schema_path.read_bytes()
            ).hexdigest(),
            provenance=provenance,
            amendment_ids=["secondary-run-test"],
        )
        assert document["schema_version"] == (
            "thesis-core-hardening-secondary-family-v1"
        )
        assert document["family"] == family
        assert len(document["cells"]) == expected_count
        for cell in document["cells"]:
            _assert_schema_valid(cell, schema)
            harness._validate_result_cell_invariants(cell)
        destination = tmp_path / f"{family}.json"
        original = harness._PLANNED_SECONDARY_DESTINATIONS[family]
        harness._PLANNED_SECONDARY_DESTINATIONS[family] = destination
        try:
            harness.write_secondary_family_result(
                destination, document, family=family
            )
            assert json.loads(destination.read_text()) == document
            with pytest.raises(FileExistsError):
                harness.write_secondary_family_result(
                    destination, document, family=family
                )
        finally:
            harness._PLANNED_SECONDARY_DESTINATIONS[family] = original


def test_secondary_authorization_is_family_specific_and_executor_is_atomic(
    tmp_path, monkeypatch
):
    harness = _load_harness()
    config_path = tmp_path / "authorization.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "execution": {
                    "cpu_only": True,
                    "gpu_enabled": False,
                    "api_enabled": False,
                    "cpu_secondary_authorized_in_this_batch": True,
                    "five_depth_consumed_in_this_batch": False,
                    "three_annotator_consumed_in_this_batch": False,
                    "load_real_activation_matrices_in_this_batch": True,
                    "synthetic_validation_only": False,
                }
            }
        )
    )
    monkeypatch.setattr(harness, "_FROZEN_HARDENING_CONFIG", config_path)
    assert harness._frozen_cpu_secondary_authorized("five_depth") is True
    blocked = yaml.safe_load(config_path.read_text())
    blocked["execution"]["five_depth_consumed_in_this_batch"] = True
    config_path.write_text(yaml.safe_dump(blocked))
    with pytest.raises(ValueError, match="not authorized"):
        harness._frozen_cpu_secondary_authorized("five_depth")

    family = "five_depth"
    destination = tmp_path / "five_depth.json"
    monkeypatch.setitem(
        harness._PLANNED_SECONDARY_DESTINATIONS, family, destination
    )
    monkeypatch.setattr(
        harness, "_frozen_cpu_secondary_authorized", lambda _family: True
    )
    monkeypatch.setattr(
        harness,
        "_frozen_registered_cpu_cross_label_policy",
        lambda: "drop_all_same_extraction_window",
    )
    synthetic_inputs = {
        coordinates["cell_index"]: {
            **coordinates,
            "activations": np.ones((4, 2)),
            "labels": list(harness._PRIMARY_LABELS),
            "chain_ids": ["c1"] * 4,
            "occurrences": [
                _record(label, "c1", index, n_positions=11)
                for index, label in enumerate(harness._PRIMARY_LABELS)
            ],
            "target_label": harness._PRIMARY_LABELS[
                coordinates["behaviour_code"] - 1
            ],
        }
        for coordinates in harness._expected_family_cells(family)
    }
    monkeypatch.setattr(
        harness,
        "load_registered_secondary_family_inputs",
        lambda _family: {
            "cells": synthetic_inputs,
            "input_hashes": {"synthetic": "a" * 64},
        },
    )

    def fake_family(*args, **kwargs):
        result = _synthetic_secondary_family_result(family)
        kwargs["progress_callback"](result["cells"][:1])
        return result

    monkeypatch.setattr(harness, "_run_registered_cdim_family", fake_family)
    actual_config = (
        ROOT
        / "configs"
        / "analysis"
        / "thesis_core_hardening_2026-07-26.yaml"
    )
    monkeypatch.setattr(harness, "_FROZEN_HARDENING_CONFIG", actual_config)
    document = harness.execute_registered_secondary_from_disk(
        family,
        destination,
        argv=["--run-secondary", family, "--acknowledge-full-analysis"],
    )
    assert json.loads(destination.read_text()) == document
    assert not destination.with_name(
        f".{destination.name}.progress.json"
    ).exists()
    with pytest.raises(FileExistsError):
        harness.execute_registered_secondary_from_disk(family, destination)


def _synthetic_scope_behaviour_result():
    valid_draws = [
        {
            "replicate": index,
            "status": "VALID",
            "cdim": 5.0,
            "pr": 4.0,
            "failure": None,
        }
        for index in range(500)
    ]
    audit = {
        "raw_rows": 500,
        "deduplicated_rows": 500,
        "duplicate_fraction": 0.0,
        "occurrence_duplicate_groups": 0,
        "occurrence_duplicate_rows": 0,
        "vector_duplicate_groups": 0,
        "vector_duplicate_rows": 0,
        "cross_chain_groups": 0,
        "cross_label_groups": 0,
    }
    cdim_summary = {
        "n_valid": 500,
        "q025": 4.8,
        "median": 5.0,
        "q975": 5.2,
    }
    pr_summary = {
        "n_valid": 500,
        "q025": 3.8,
        "median": 4.0,
        "q975": 4.2,
    }
    band = {
        "status": "VALID",
        "full_sample_cdim": 5.0,
        "full_sample_pr": 4.0,
        "valid_threshold": 475,
        "cdim_stability": cdim_summary,
        "pr_stability": pr_summary,
        "draws": valid_draws,
        "duplicate_audit": audit,
        "occurrence_list_sha256": "1" * 64,
        "row_counts": {
            "raw": 500,
            "deduplicated": 500,
            "analyzed": 200,
            "sentence_weighted": 400,
        },
        "chain_counts": {
            "unique": 500,
            "eligible": 200,
            "selected": 200,
            "mixed_label": 490,
        },
    }

    def stratum(digest):
        return {
            "status": "VALID",
            "chain_ids": [f"c{index:03}" for index in range(200)],
            "n_chains": 200,
            "n_categories": 2,
            "category_counts": {"x": 100, "y": 100},
            "occurrence_list_sha256": digest * 64,
            "cdim": 5.0,
            "pr": 4.0,
            "cdim_stability": cdim_summary,
            "pr_stability": pr_summary,
            "stability_draws": valid_draws,
        }

    def matched(digest):
        value = stratum(digest)
        return {
            key: value[key]
            for key in (
                "chain_ids",
                "n_chains",
                "n_categories",
                "category_counts",
                "occurrence_list_sha256",
                "cdim",
                "pr",
                "cdim_stability",
                "pr_stability",
                "status",
            )
        }

    return {
        "status": "VALID",
        "strata": {
            "combined": stratum("2"),
            "complete": stratum("3"),
            "truncated": stratum("4"),
        },
        "category_share_imbalance": 0.2,
        "matching_required": True,
        "matched_results": {
            "matched_complete": matched("5"),
            "matched_truncated": matched("6"),
        },
        "unadjusted_pass": True,
        "matched_pass": True,
        "chain_stability": band,
        "chain_stability_leg": "PASS",
        "truncation_leg": "PASS",
        "h3_behaviour_status": "PASS",
        "descriptives": {"complete": {}, "truncated": {}},
        "duplicate_audit": audit,
        "pool_counts": {
            "unique_chains": 500,
            "mixed_label_chains": 490,
            "target_rows_after_deduplication": 400,
        },
    }


def test_h3_scope_documents_are_complete_and_schema_valid():
    harness = _load_harness()
    schema_path = (
        ROOT
        / "schemas"
        / "thesis_core_hardening_result_cell_v1.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    provenance = _valid_unrun_result_cell()["provenance"]
    chain, truncation = harness.build_h3_family_documents(
        {code: _synthetic_scope_behaviour_result() for code in range(1, 5)},
        input_hashes={"synthetic": "a" * 64},
        preregistration_path="results/prereg/example.md",
        preregistration_sha256="b" * 64,
        result_schema_sha256=hashlib.sha256(
            schema_path.read_bytes()
        ).hexdigest(),
        provenance=provenance,
        amendment_ids=["scope-test"],
    )
    assert chain["decision"] == "PASS"
    assert truncation["decision"] == "PASS"
    assert len(chain["cells"]) == 4
    assert len(truncation["cells"]) == 16
    assert [cell["cell_index"] for cell in truncation["cells"]] == list(
        range(48, 64)
    )
    for cell in chain["cells"] + truncation["cells"]:
        _assert_schema_valid(cell, schema)
        harness._validate_result_cell_invariants(cell)
    for cell in truncation["cells"][-4:]:
        assert cell["observed_cdim"] is None
        assert cell["observed_pr"] is None
        assert set(cell["matched_results"]) == {
            "matched_complete",
            "matched_truncated",
        }


def test_curvature_scope_document_maps_paired_ratios_without_p_values():
    harness = _load_harness()
    schema_path = (
        ROOT
        / "schemas"
        / "thesis_core_hardening_result_cell_v1.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    provenance = _valid_unrun_result_cell()["provenance"]
    audit = _synthetic_scope_behaviour_result()["duplicate_audit"]
    result = {
        "status": "VALID",
        "n_valid_pairs": 500,
        "valid_threshold": 475,
        "bounded_negative": True,
        "wording": "no beyond-chain curvature detected by this diagnostic at L16",
        "D_chain": 0.05,
        "D_control": 0.08,
        "chain_stability": {
            "n_valid": 500,
            "q025": 0.95,
            "median": 1.0,
            "q975": 1.05,
        },
        "control_stability": {
            "n_valid": 500,
            "q025": 0.9,
            "median": 1.0,
            "q975": 1.1,
        },
        "draws": [
            {
                "replicate": index,
                "status": "VALID",
                "ratio_chain": 1.0,
                "ratio_control": 1.0,
                "failure": None,
            }
            for index in range(500)
        ],
        "duplicate_audit": audit,
        "occurrence_list_sha256": "7" * 64,
        "row_counts": {
            "raw": 500,
            "deduplicated": 500,
            "analyzed": 200,
            "sentence_weighted": 400,
        },
        "chain_counts": {
            "unique": 500,
            "eligible": 250,
            "selected": 200,
            "mixed_label": 490,
        },
    }
    document = harness.build_curvature_family_document(
        {code: result for code in range(1, 5)},
        input_hashes={"synthetic": "a" * 64},
        preregistration_path="results/prereg/example.md",
        preregistration_sha256="b" * 64,
        result_schema_sha256=hashlib.sha256(
            schema_path.read_bytes()
        ).hexdigest(),
        provenance=provenance,
        amendment_ids=["scope-test"],
    )
    assert document["decision"] == "PASS"
    assert document["draw_field_semantics"] == {
        "cdim": "ratio_chain",
        "pr": "ratio_control",
    }
    for cell in document["cells"]:
        _assert_schema_valid(cell, schema)
        harness._validate_result_cell_invariants(cell)
        assert cell["raw_p"] is None and cell["holm_p"] is None


def test_result_schema_nested_contracts_are_strict_and_nonempty():
    schema = json.loads(
        (
            ROOT
            / "schemas"
            / "thesis_core_hardening_result_cell_v1.schema.json"
        ).read_text()
    )
    for definition in (
        "estimator_settings",
        "preprocessing",
        "duplicate_audit",
        "result_provenance",
        "row_counts",
        "chain_counts",
        "resamples",
        "seed_coordinates",
        "software_facts",
        "process_facts",
        "host_facts",
    ):
        assert definition in schema["$defs"]
        assert schema["$defs"][definition]["additionalProperties"] is False
        assert schema["$defs"][definition]["required"]
    assert schema["$defs"]["hash_manifest"]["minProperties"] == 1
    assert "dirty_paths" in schema["$defs"]["result_provenance"]["required"]
    assert schema["$defs"]["codes"]["properties"]["family"]["maximum"] == 6
    assert schema["$defs"]["codes"]["properties"]["purpose"]["maximum"] == 4
    assert "sentence_weighted_cdim" in schema["properties"]
    assert "sentence_weighted_label" in schema["properties"]
    assert "chain_attrition_reasons" in schema["properties"]


class _SchemaValidationError(AssertionError):
    pass


def _schema_type_matches(instance, expected):
    if expected == "null":
        return instance is None
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "object":
        return isinstance(instance, dict)
    raise AssertionError(f"unsupported schema type {expected!r}")


def _resolve_local_ref(root_schema, reference):
    assert reference.startswith("#/")
    value = root_schema
    for component in reference[2:].split("/"):
        value = value[component.replace("~1", "/").replace("~0", "~")]
    return value


def _json_equal(left, right):
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_equal(left[key], right[key]) for key in left
        )
    return left == right


def _schema_errors(instance, schema, root_schema=None, path="$"):
    import re

    if root_schema is None:
        root_schema = schema
    errors = []
    if "$ref" in schema:
        target = _resolve_local_ref(root_schema, schema["$ref"])
        errors.extend(_schema_errors(instance, target, root_schema, path))
        schema = {key: value for key, value in schema.items() if key != "$ref"}
    if "type" in schema:
        allowed = schema["type"]
        allowed = allowed if isinstance(allowed, list) else [allowed]
        if not any(_schema_type_matches(instance, value) for value in allowed):
            return [f"{path}: expected type {allowed!r}"]
    if "const" in schema and not _json_equal(instance, schema["const"]):
        errors.append(f"{path}: const mismatch")
    if "enum" in schema and not any(
        _json_equal(instance, candidate) for candidate in schema["enum"]
    ):
        errors.append(f"{path}: enum mismatch")
    if "allOf" in schema:
        for subschema in schema["allOf"]:
            errors.extend(_schema_errors(instance, subschema, root_schema, path))
    if "anyOf" in schema and not any(
        not _schema_errors(instance, subschema, root_schema, path)
        for subschema in schema["anyOf"]
    ):
        errors.append(f"{path}: no anyOf branch matched")
    if "oneOf" in schema:
        matches = sum(
            not _schema_errors(instance, subschema, root_schema, path)
            for subschema in schema["oneOf"]
        )
        if matches != 1:
            errors.append(f"{path}: expected exactly one oneOf branch, got {matches}")
    if "not" in schema and not _schema_errors(
        instance, schema["not"], root_schema, path
    ):
        errors.append(f"{path}: forbidden not schema matched")
    if "if" in schema:
        branch = "then" if not _schema_errors(
            instance, schema["if"], root_schema, path
        ) else "else"
        if branch in schema:
            errors.extend(
                _schema_errors(instance, schema[branch], root_schema, path)
            )

    if isinstance(instance, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                errors.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        for key, value in instance.items():
            if "propertyNames" in schema:
                errors.extend(
                    _schema_errors(
                        key, schema["propertyNames"], root_schema, f"{path}.<key>"
                    )
                )
            if key in properties:
                errors.extend(
                    _schema_errors(
                        value, properties[key], root_schema, f"{path}.{key}"
                    )
                )
            else:
                additional = schema.get("additionalProperties", True)
                if additional is False:
                    errors.append(f"{path}: additional property {key!r}")
                elif isinstance(additional, dict):
                    errors.extend(
                        _schema_errors(
                            value, additional, root_schema, f"{path}.{key}"
                        )
                    )
        if len(instance) < schema.get("minProperties", 0):
            errors.append(f"{path}: too few properties")
        if "maxProperties" in schema and len(instance) > schema["maxProperties"]:
            errors.append(f"{path}: too many properties")

    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            errors.append(f"{path}: too few items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: too many items")
        if schema.get("uniqueItems"):
            serialized = [
                json.dumps(value, sort_keys=True, allow_nan=False) for value in instance
            ]
            if len(serialized) != len(set(serialized)):
                errors.append(f"{path}: duplicate items")
        prefix = schema.get("prefixItems", [])
        for index, subschema in enumerate(prefix[: len(instance)]):
            errors.extend(
                _schema_errors(
                    instance[index], subschema, root_schema, f"{path}[{index}]"
                )
            )
        items = schema.get("items")
        if items is False and len(instance) > len(prefix):
            errors.append(f"{path}: items beyond prefix are forbidden")
        elif isinstance(items, dict):
            start = len(prefix) if prefix else 0
            for index, value in enumerate(instance[start:], start=start):
                errors.extend(
                    _schema_errors(
                        value, items, root_schema, f"{path}[{index}]"
                    )
                )

    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            errors.append(f"{path}: string too short")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: string too long")
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errors.append(f"{path}: pattern mismatch")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: below minimum")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: above maximum")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: below exclusive minimum")
        if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: above exclusive maximum")
    return errors


def _assert_schema_valid(instance, schema):
    errors = _schema_errors(instance, schema)
    if errors:
        raise _SchemaValidationError("\n".join(errors))


def test_dependency_free_schema_validation_covers_all_three_contracts():
    result_schema = json.loads(
        (
            ROOT
            / "schemas"
            / "thesis_core_hardening_result_cell_v1.schema.json"
        ).read_text()
    )
    resource_schema = json.loads(
        (
            ROOT
            / "schemas"
            / "thesis_core_hardening_resource_benchmark_v1.schema.json"
        ).read_text()
    )
    provenance_schema = json.loads(
        (
            ROOT
            / "schemas"
            / "thesis_core_hardening_provenance_v1.schema.json"
        ).read_text()
    )
    _assert_schema_valid(_valid_unrun_result_cell(), result_schema)
    with pytest.raises(_SchemaValidationError):
        _assert_schema_valid({}, result_schema)
    invalid = _valid_unrun_result_cell()
    invalid["estimator_settings"] = {}
    with pytest.raises(_SchemaValidationError):
        _assert_schema_valid(invalid, result_schema)
    invalid_boolean = _valid_unrun_result_cell()
    invalid_boolean["preprocessing"]["cdim_centered"] = 0
    with pytest.raises(_SchemaValidationError):
        _assert_schema_valid(invalid_boolean, result_schema)
    invalid_truncation = _valid_unrun_result_cell()
    invalid_truncation["status"] = "VALID"
    invalid_truncation["observed_cdim"] = 1.0
    invalid_truncation["null_cdim"] = [1.0]
    with pytest.raises(_SchemaValidationError):
        _assert_schema_valid(invalid_truncation, result_schema)

    harness = _load_harness()
    clock = iter([0.0, 0.1])
    resource = harness.run_resource_benchmark(
        lambda _: "VALID",
        b_target=1,
        synthetic=True,
        clock_fn=lambda: next(clock),
        peak_rss_fn=lambda: 0,
    )
    _assert_schema_valid(resource, resource_schema)
    with pytest.raises(_SchemaValidationError):
        _assert_schema_valid({**resource, "software": {}}, resource_schema)
    with pytest.raises(_SchemaValidationError):
        _assert_schema_valid(
            {**resource, "gates": {**resource["gates"], "extra": "PASS"}},
            resource_schema,
        )

    provenance = {
        "schema_version": "thesis-core-hardening-provenance-v1",
        "preregistration_path": "results/prereg/example.md",
        "preregistration_sha256": "a" * 64,
        "input_manifest": {"synthetic": "b" * 64},
        "code_commit": None,
        "dirty": True,
        "dirty_paths": ["synthetic"],
        "python_version": "3.13.12",
        "package_versions": {"numpy": "2.4.1"},
        "argv": ["--synthetic"],
        "utc_start": "2026-07-26T00:00:00Z",
        "utc_end": "2026-07-26T00:00:01Z",
        "host": {
            "hostname": "cpu-01",
            "system": "Darwin",
            "machine": "arm64",
            "cpu_count": 12,
        },
        "process": {"pid": 1, "workers": 1},
        "amendment_ids": [],
    }
    _assert_schema_valid(provenance, provenance_schema)
    with pytest.raises(_SchemaValidationError):
        _assert_schema_valid({**provenance, "process": {}}, provenance_schema)
    with pytest.raises(_SchemaValidationError):
        _assert_schema_valid(
            {**provenance, "preregistration_sha256": "not-a-hash"},
            provenance_schema,
        )


@pytest.mark.parametrize(
    "family",
    [
        "primary",
        "five_depth",
        "three_annotator",
        "truncation",
        "pooling_window",
        "curvature",
    ],
)
def test_result_schema_validates_representative_valid_family_cells(family):
    harness = _load_harness()
    schema = json.loads(
        (
            ROOT
            / "schemas"
            / "thesis_core_hardening_result_cell_v1.schema.json"
        ).read_text()
    )
    document = _valid_result_cell(family)
    _assert_schema_valid(document, schema)
    harness._validate_result_cell_invariants(document)


def test_result_schema_rejects_registry_status_and_count_inconsistencies():
    harness = _load_harness()
    schema = json.loads(
        (
            ROOT
            / "schemas"
            / "thesis_core_hardening_result_cell_v1.schema.json"
        ).read_text()
    )
    wrong_registry = _valid_result_cell("primary")
    wrong_registry["cell_index"] = 4
    wrong_registry["resamples"]["seed_coordinates"]["cell_index"] = 4
    with pytest.raises(_SchemaValidationError):
        _assert_schema_valid(wrong_registry, schema)

    missing_observed = _valid_result_cell("five_depth")
    missing_observed["observed_cdim"] = None
    with pytest.raises(_SchemaValidationError):
        _assert_schema_valid(missing_observed, schema)

    negative_null = _valid_result_cell("primary")
    negative_null["null_cdim"][0] = -1.0
    with pytest.raises(_SchemaValidationError):
        _assert_schema_valid(negative_null, schema)

    invalid_valid_draw = _valid_result_cell("curvature")
    invalid_valid_draw["stability_draws"][0]["cdim"] = None
    with pytest.raises(_SchemaValidationError):
        _assert_schema_valid(invalid_valid_draw, schema)

    inconsistent_counts = _valid_result_cell("three_annotator")
    inconsistent_counts["resamples"]["attempted"] = 2501
    with pytest.raises(ValueError, match="resample counts"):
        harness._validate_result_cell_invariants(inconsistent_counts)

    inconsistent_draw_counts = _valid_result_cell("truncation")
    inconsistent_draw_counts["stability_draws"][0] = {
        "replicate": 0,
        "status": "INVALID",
        "cdim": None,
        "pr": None,
        "failure": "estimator_invalid",
    }
    with pytest.raises(ValueError, match="draw counts"):
        harness._validate_result_cell_invariants(inconsistent_draw_counts)

    inferential_unrun = _valid_unrun_result_cell()
    inferential_unrun.update(
        {
            "family": "primary",
            "hypothesis": "H2",
            "cell_index": 0,
            "codes": {
                "family": 1,
                "purpose": 1,
                "annotator": 1,
                "behaviour": 1,
                "pooling": 1,
                "window": 1,
                "truncation": 0,
            },
            "layer_zero_based": 27,
        }
    )
    inferential_unrun["resamples"]["seed_coordinates"].update(
        {"purpose_code": 1, "family_code": 1, "cell_index": 0}
    )
    _assert_schema_valid(inferential_unrun, schema)
    harness._validate_result_cell_invariants(inferential_unrun)

    invalid = dict(inferential_unrun)
    invalid["status"] = "INVALID"
    _assert_schema_valid(invalid, schema)
    harness._validate_result_cell_invariants(invalid)

def test_frozen_yaml_config_has_exact_registry_and_consumed_pilot_gate():
    config_path = (
        ROOT / "configs" / "analysis" / "thesis_core_hardening_2026-07-26.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    assert config["status"] == "UNRUN"
    assert config["model"]["id"] == (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )
    assert config["labels"] == [
        "backtracking",
        "uncertainty-estimation",
        "example-testing",
        "adding-knowledge",
    ]
    assert config["layers"]["primary_zero_based"] == 27
    assert config["layers"]["curvature_zero_based"] == 16
    assert config["resampling"] == {
        "permutation_B": 2500,
        "permutation_attempt_cap": 2750,
        "resource_pilot_B": 25,
        "chain_stability_replicates": 500,
        "chain_stability_valid_threshold": 475,
    }
    assert config["seeds"]["registered"] == 20260726
    assert config["seeds"]["diagnostic"] == 20260727
    assert config["cell_index_registry"]["maximum"] == 91
    assert config["execution"]["gpu_enabled"] is False
    assert config["execution"]["api_enabled"] is False
    assert config["execution"]["empirical_pilot_authorized_in_this_batch"] is False
    assert config["execution"]["empirical_pilot_consumed_in_this_batch"] is True
    assert config["execution"]["empirical_pilot_retry_authorized_in_this_batch"] is False
    assert config["execution"]["empirical_pilot_retry_consumed_in_this_batch"] is True
    assert config["execution"]["empirical_B50_authorized_in_this_batch"] is False
    assert config["execution"]["full_primary_authorized_in_this_batch"] is False
    assert config["execution"]["full_primary_consumed_in_this_batch"] is True
    assert config["execution"]["cpu_secondary_authorized_in_this_batch"] is False
    assert config["execution"]["five_depth_consumed_in_this_batch"] is True
    assert config["execution"]["three_annotator_consumed_in_this_batch"] is True
    assert (
        config["execution"][
            "cpu_scope_diagnostics_authorized_in_this_batch"
        ]
        is False
    )
    assert config["execution"]["h3_consumed_in_this_batch"] is True
    assert config["execution"]["curvature_consumed_in_this_batch"] is True
    assert config["execution"]["load_real_activation_matrices_in_this_batch"] is False
    assert config["execution"]["synthetic_validation_only"] is True
    assert config["thresholds"]["resource"]["reported_gates"] == [
        "primary",
        "secondary",
        "H4_cpu",
        "H4",
    ]
    assert config["execution"]["resource_modes"]["empirical"][
        "B50_requires_successful_B25"
    ] is True
    assert config["duplicate_handling"] == {
        "cross_chain_policy": "hard_fail",
        "empirical_resource_pilot_cross_label_policy": (
            "drop_all_same_extraction_window"
        ),
        "registered_cpu_family_cross_label_policy": (
            "drop_all_same_extraction_window"
        ),
        "scope": "amended_registered_cpu_families",
    }
    assert config["execution"]["h4_gpu_prerequisites"] == {
        "free_scratch_bytes_min": 100 * 1024**3,
        "six_representation_alignment_smoke_pass_required": True,
        "projected_extraction_seconds_max": 24 * 60 * 60,
        "vram_bytes_min": 24 * 1024**3,
    }


def test_truncation_sensitivity_runs_exhaustive_matched_operator_on_synthetic_data():
    harness = _load_harness()
    rng = np.random.default_rng(222)
    records = []
    metadata = {}
    rows = []
    for index in range(500):
        chain_id = f"c{index:03}"
        complete = index < 250
        within_group = index if complete else index - 250
        if complete:
            category = "x" if within_group < 175 else "y"
            n_tokens = 7000
            text = "finished </think>"
        else:
            category = "x" if within_group < 75 else "y"
            n_tokens = 8192
            text = "unfinished"
        metadata[chain_id] = {
            "n_tokens": n_tokens,
            "chain": text,
            "category": category,
        }
        records.append(_record("A", chain_id, 0, token_start=index % 100))
        rows.append(rng.normal(size=4))
    result = harness.run_truncation_sensitivity(
        np.asarray(rows),
        records,
        metadata,
        target_label="A",
        behaviour_code=1,
        n_replicates=3,
        valid_threshold=2,
    )
    assert result["status"] == "VALID"
    assert result["matching_required"] is True
    assert result["category_share_imbalance"] == pytest.approx(0.4)
    assert set(result["strata"]) == {"combined", "complete", "truncated"}
    assert result["strata"]["combined"]["n_chains"] == 500
    assert result["matched_results"]["matched_complete"]["n_chains"] == 150
    assert result["matched_results"]["matched_truncated"]["n_chains"] == 150
    assert result["truncation_leg"] in {"PASS", "MIXED", "FAIL"}
    assert result["h3_behaviour_status"] in {"PASS", "MIXED", "FAIL"}
    assert result["chain_stability_leg"] in {"PASS", "FAIL"}
