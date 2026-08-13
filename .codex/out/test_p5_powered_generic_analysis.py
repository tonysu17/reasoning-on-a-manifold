from __future__ import annotations

import copy

import pytest

import p5_powered_generic_analysis as analysis


H = "a" * 64


def record(task, category, role, values):
    return {
        "task_id": task,
        "category": category,
        "checkpoint_role": role,
        "endpoints": dict(zip(analysis.ENDPOINTS, values)),
        "generation_row_sha256": H,
        "prefix_row_sha256": H,
        "annotation_row_sha256": H,
    }


def fixture_records():
    rows = []
    for index in range(10):
        category = f"category-{index % 2}"
        task = f"task-{index:02d}"
        control = [0.10 + index / 1000, 0.20, 0.30, 0.40]
        safety = [value + 0.05 for value in control]
        rows.append(record(task, category, analysis.CONTROL_ROLE, control))
        rows.append(record(task, category, analysis.SAFETY_ROLE, safety))
    return rows


def test_primary_analysis_is_task_paired_and_direction_is_safety_minus_control():
    result = analysis.analyze_primary(
        fixture_records(), bootstrap_draws=200, sign_flip_draws=500, seed=7
    )
    assert result["independent_unit"] == "task/prompt"
    assert result["multiple_testing"] == "Holm across four primary endpoints"
    for endpoint in analysis.ENDPOINTS:
        cell = result["endpoints"][endpoint]
        assert cell["n_planned_tasks_present"] == 10
        assert cell["n_complete_pairs"] == 10
        assert cell["paired_mean_difference"] == pytest.approx(0.05)
        assert cell["control_task_mean"] + 0.05 == pytest.approx(cell["safety_task_mean"])
        assert 0 <= cell["raw_sign_flip_p"] <= cell["holm_adjusted_p"] <= 1


def test_endpoint_specific_missingness_does_not_become_zero():
    rows = fixture_records()
    rows[1]["endpoints"]["backtracking"] = None
    result = analysis.analyze_primary(rows, bootstrap_draws=50, sign_flip_draws=100, seed=3)
    backtracking = result["endpoints"]["backtracking"]
    uncertainty = result["endpoints"]["uncertainty-estimation"]
    assert backtracking["n_complete_pairs"] == 9
    assert backtracking["missingness"] == {"missing_safety_endpoint": 1}
    assert uncertainty["n_complete_pairs"] == 10


def test_invalid_endpoint_role_hash_and_conflicting_duplicates_fail_closed():
    base = fixture_records()[0]
    bad_endpoint = copy.deepcopy(base)
    bad_endpoint["endpoints"].pop("backtracking")
    with pytest.raises(analysis.AnalysisError):
        analysis.validate_and_index([bad_endpoint])
    bad_role = copy.deepcopy(base)
    bad_role["checkpoint_role"] = "phase2_public_star1_shared"
    with pytest.raises(analysis.AnalysisError):
        analysis.validate_and_index([bad_role])
    bad_hash = copy.deepcopy(base)
    bad_hash["prefix_row_sha256"] = "short"
    with pytest.raises(analysis.AnalysisError):
        analysis.validate_and_index([bad_hash])
    conflict = copy.deepcopy(base)
    conflict["endpoints"]["backtracking"] = 0.9
    with pytest.raises(analysis.AnalysisError):
        analysis.validate_and_index([base, conflict])


def test_category_mismatch_and_nonfraction_values_fail_closed():
    rows = fixture_records()[:2]
    rows[1]["category"] = "different"
    with pytest.raises(analysis.AnalysisError):
        analysis.validate_and_index(rows)
    rows = fixture_records()[:2]
    rows[1]["endpoints"]["backtracking"] = 1.1
    with pytest.raises(analysis.AnalysisError):
        analysis.validate_and_index(rows)


def test_holm_requires_exact_primary_family():
    with pytest.raises(analysis.AnalysisError):
        analysis.holm_adjust({"backtracking": 0.1})
    adjusted = analysis.holm_adjust(dict(zip(analysis.ENDPOINTS, [0.01, 0.02, 0.03, 0.20])))
    assert adjusted["backtracking"] == pytest.approx(0.04)
    assert adjusted["uncertainty-estimation"] == pytest.approx(0.06)
    assert adjusted["example-testing"] == pytest.approx(0.06)
    assert adjusted["adding-knowledge"] == pytest.approx(0.20)
