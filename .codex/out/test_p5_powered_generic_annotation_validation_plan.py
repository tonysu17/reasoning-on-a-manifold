from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PLAN = HERE / "P5_POWERED_GENERIC_ANNOTATION_VALIDATION_PLAN_2026-08-09.json"


def load():
    return json.loads(PLAN.read_text())


def test_plan_is_non_executable_and_matches_approved_design():
    plan = load()
    assert plan["status"] == "prospective_non_executable_phase2_and_spend_gated"
    assert plan["execution_authorized"] is False
    assert plan["api_model_pod_human_calls_authorized"] == 0
    assert plan["approved_design_source"]["file_sha256"] == (
        "e36d44c20c6bde9dc197d0d2bccb6479c7134b654d8c29175f418a26375e2b6f"
    )


def test_population_and_validation_arithmetic():
    plan = load()
    population = plan["population"]
    validation = plan["validation_stage"]
    assert population["tasks"] == 100
    assert population["planned_rows"] == 500
    assert population["shared_phase2_rows"] + population["p5_owned_rows"] == 500
    assert validation["tasks"] == 20
    assert validation["tasks_per_category"] * population["categories"] == 20
    assert validation["tasks"] * validation["roles_per_task"] == validation["planned_rows"] == 100
    assert validation["planned_rows"] + validation["held_out_rows"] == 500


def test_gate_preserves_missingness_and_role_coverage():
    gate = load()["operational_gate"]
    assert gate["atomic_terminal_persistence"] == 1.0
    assert gate["minimum_global_valid_row_rate"] == 0.98
    assert gate["minimum_valid_rows"] == 98
    assert gate["minimum_per_role_valid_rate"] == 0.95
    assert gate["minimum_valid_rows_per_role"] == 19
    assert gate["missing_is_never_zero"] is True
    assert gate["same_sonnet_repeat_is_interannotator_evidence"] is False


def test_phase2_annotation_reuse_is_exact_and_raw_generation_is_immutable():
    reuse = load()["phase2_annotation_reuse"]
    assert reuse["annotation_reuse_default"] == "not_assumed"
    assert reuse["never_regenerate_shared_generation"] is True
    requirements = " ".join(reuse["eligible_only_if_all"])
    assert "complete P5 prefix" in requirements
    assert "source-unit" in requirements
    assert "causal-family outcome" in requirements


def test_human_audit_is_not_silently_authorized():
    audit = load()["human_audit"]
    assert audit["status"] == "prospective_not_authorized"
    assert audit["recommended_rows"] == 20
    assert "builder-annotator-scored" in audit["boundary"]
