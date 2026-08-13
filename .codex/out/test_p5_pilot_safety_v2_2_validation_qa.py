from __future__ import annotations

import json
from pathlib import Path

import pytest

import p5_preflight as p5
import p5_pilot_safety_v2_2_validation_qa as qa


HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE / "p5_runs" / "p5-pilot-20260808"
MANIFEST = HERE / "P5_PILOT_SAFETY_V2_2_PROPOSED_VALIDATION_RESCORING_2026-08-09.json"
REPORT = HERE / "P5_PILOT_SAFETY_V2_2_VALIDATION_INTEGRITY_AND_BLINDED_SIZING_2026-08-09.json"


@pytest.fixture(scope="module")
def report():
    return qa.build_report(RUN_ROOT, MANIFEST)


def test_exact_authorized_manifest_binding(report):
    sources = report["sources"]
    assert sources["authorized_manifest_internal_sha256"] == (
        qa.EXPECTED_MANIFEST_INTERNAL_SHA256
    )
    assert sources["authorized_manifest_file_sha256"] == (
        qa.EXPECTED_MANIFEST_FILE_SHA256
    )
    assert p5.sha256_file(MANIFEST) == qa.EXPECTED_MANIFEST_FILE_SHA256


def test_integrity_reconstructs_every_validation_chunk_and_assignment(report):
    integrity = report["integrity"]
    assert integrity["status"] == "pass"
    assert integrity["errors"] == []
    assert integrity["observed_validation_assignments"] == 48
    assert integrity["terminal_validation_chunks"] == 104
    assert integrity["unique_terminal_chunk_keys"] == 104
    assert integrity["generation_lineage_errors"] == 0
    assert integrity["chunk_plan_errors"] == 0
    assert integrity["deterministic_evidence_errors"] == 0
    assert integrity["aggregation_recomputation_errors"] == 0


def test_frozen_gate_fails_only_network_success(report):
    gate = report["validation_gate"]
    assert gate["status"] == "fail"
    assert gate["atomic_terminal_chunk_persistence_rate"] == 1.0
    assert gate["successful_response_chunks"] == 90
    assert gate["network_success_rate"] == pytest.approx(90 / 104)
    assert gate["complete_parser_and_evidence_chunks"] == 90
    assert gate["complete_parser_and_evidence_rate_among_successful_responses"] == 1.0
    assert gate["errors"] == ["network success rate below gate"]


def test_missingness_and_proxy_errors_are_explicit(report):
    scoring = report["scoring"]
    assert scoring["terminal_chunks"] == 104
    assert scoring["network_success_chunks"] == 90
    assert scoring["network_error_chunks"] == 14
    assert scoring["network_error_classes"] == {"ProxyProtocolError": 14}
    assert scoring["parser_or_evidence_failures_among_successful_responses"] == 0
    assert scoring["aggregation_status"] == {
        "resolved_complete": 41,
        "resolved_monotone_compliance_partial": 3,
        "unresolved_partial_chunks": 4,
    }
    assert scoring["missing_chunks_remain_unresolved"] is True


def test_exact_reported_accounting_and_hard_guards(report):
    budget = report["requests_and_budget"]
    assert budget["attempts_reserved"] == 104
    assert budget["attempts_completed"] == 104
    assert budget["retries"] == 0
    assert budget["reported_successful_response_cost_usd"] == pytest.approx(0.488511)
    assert budget["responses_without_cost_accounting"] == 14
    assert budget["total_billed_cost_known"] is False
    assert budget["conservative_authorized_cost_commitment_usd"] == pytest.approx(2.6)
    assert budget["maximum_reported_call_cost_usd"] == pytest.approx(0.011778)
    assert budget["minimum_reported_remaining_quota_usd"] == pytest.approx(
        131.87720394999997
    )
    assert budget["all_hard_guards_respected"] is True


def test_held_out_and_repeats_were_not_executed(report):
    integrity = report["integrity"]
    disposition = report["disposition"]
    assert integrity["held_out_assignment_rows"] == 0
    assert integrity["held_out_chunk_rows"] == 0
    assert integrity["held_out_journal_events"] == 0
    assert integrity["same_sonnet_repeat_rows"] == 0
    assert disposition["held_out_rescoring_executed"] is False
    assert disposition["current_manifest_may_resume_held_out"] is False


def test_blinded_sizing_uses_only_missingness(report):
    sizing = report["blinded_sizing_readiness"]
    assert sizing["arm_labelled_effect_differences_computed"] is False
    assert sizing["arm_labelled_effect_differences_reported"] is False
    assert sizing["endpoint_values_used_for_sizing"] is False
    assert sizing["powered_size_freeze"] == "not_permitted"
    harmful = sizing["endpoints"]["harmful_refusal"]
    benign = sizing["endpoints"]["benign_compliance"]
    assert (harmful["complete_pooled_prompt_pairs"], harmful["possible_pooled_prompt_pairs"]) == (
        8,
        12,
    )
    assert (benign["complete_pooled_prompt_pairs"], benign["possible_pooled_prompt_pairs"]) == (
        9,
        12,
    )
    assert harmful["powered_sample_size"] is None
    assert benign["powered_sample_size"] is None


def test_persisted_machine_report_matches_final_disposition():
    document = json.loads(REPORT.read_text())
    assert document["integrity"]["status"] == "pass"
    assert document["validation_gate"]["status"] == "fail"
    assert document["disposition"]["held_out_rescoring_executed"] is False


def test_frozen_v21_artifacts_remain_unchanged():
    assert p5.sha256_file(HERE / "p5_pilot_scorer_v2.py") == (
        "fc4f541b6d24e9611808f004cf719fb9a4e563805f133528d03a50a47ba09b65"
    )
    assert p5.sha256_file(RUN_ROOT / "safety_scores_v2.jsonl") == (
        "b2dfc032d1f7acd6e2fa301bc7f538cffaf6404d100cec13ce7a440f320075f3"
    )
    assert p5.sha256_file(RUN_ROOT / "scoring_call_journal_v2.jsonl") == (
        "8b0fbe01cbd176e727e6f5e3a23bd5e7f0cb5b1b3c5a90cae5fc081042bcad96"
    )
