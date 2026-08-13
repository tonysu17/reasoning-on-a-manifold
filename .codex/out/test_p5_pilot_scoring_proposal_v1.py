from __future__ import annotations

from pathlib import Path

import pytest

import p5_preflight as p5
import p5_pilot_scorer_v2 as scorer
import p5_pilot_scoring_proposal_v1 as proposal


HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE / "p5_runs" / "p5-pilot-20260808"
FINAL_PLAN = HERE / "P5_PILOT_SCORER_V2_1_FINAL_PRIMARY_ONLY_PLAN_2026-08-09.json"
CALIBRATION_RESULTS = RUN_ROOT / "cost_calibration_results_v1.jsonl"
CALIBRATION_JOURNAL = RUN_ROOT / "cost_calibration_call_journal_v1.jsonl"
PROPOSED = HERE / "P5_PILOT_SCORER_V2_1_PROPOSED_EXECUTABLE_PRIMARY_2026-08-09.json"


def test_final_plan_binding_is_exact():
    plan = scorer.load_v2_manifest(FINAL_PLAN)
    assert plan["manifest_sha256"] == proposal.EXPECTED_PLAN_INTERNAL_SHA256
    assert p5.sha256_file(FINAL_PLAN) == proposal.EXPECTED_PLAN_FILE_SHA256
    assert plan["dry_run_accounting"]["full_176_primary_only_initial_requests"] == 490


def test_calibration_binding_and_observed_summary_are_exact():
    summary = proposal.calibration_summary(CALIBRATION_RESULTS, CALIBRATION_JOURNAL)
    assert summary["results_sha256"] == proposal.EXPECTED_CALIBRATION_RESULTS_SHA256
    assert summary["journal_sha256"] == proposal.EXPECTED_CALIBRATION_JOURNAL_SHA256
    assert summary["n_calls"] == 2
    assert summary["n_retries"] == 0
    assert summary["total_observed_cost_usd"] == pytest.approx(0.01692)
    assert summary["maximum_observed_cost_usd"] == pytest.approx(0.01095)
    assert summary["minimum_observed_remaining_quota_usd"] == pytest.approx(
        135.65018394999998
    )


def test_proposed_manifest_has_only_primary_scope_and_exact_limits():
    document = scorer.load_v2_manifest(PROPOSED)
    guard = document["authorization_guard"]
    scope = document["execution_scope"]
    assert document["proposal_status"] == "proposed_awaiting_owner_exact_hash_approval"
    assert len(document["logical_assignments"]) == 176
    assert len(document["initial_network_request_plan"]) == 490
    assert all(row["pass_id"] == "primary" for row in document["logical_assignments"])
    assert all(row["pass_id"] == "primary" for row in document["initial_network_request_plan"])
    assert scope["same_sonnet_repeats"] == "absent/dropped"
    assert scope["optional_repeat_keys_authorized"] == 0
    assert guard["approved_request_ceiling"] == 600
    assert guard["approved_spend_ceiling_usd"] == 15.0
    assert guard["approved_max_cost_per_request_usd"] == 0.025
    assert guard["quota_stop_floor_usd"] == 5.0
    assert guard["global_retry_attempt_capacity"] == 110
    assert guard["per_chunk_retry_ceiling"] == 3
    assert 490 <= guard["approved_request_ceiling"]
    assert guard["approved_request_ceiling"] * guard[
        "approved_max_cost_per_request_usd"
    ] <= guard["approved_spend_ceiling_usd"]


def test_old_dry_manifest_refuses_execution_even_with_its_exact_hash():
    dry = scorer.load_v2_manifest(FINAL_PLAN)
    with pytest.raises(scorer.ScorerV2Error, match="dry-run/non-executable"):
        scorer.require_execution_authorisation(dry, dry["manifest_sha256"], authorised=True)


def test_proposed_manifest_requires_exact_hash_and_authorised_flag():
    document = scorer.load_v2_manifest(PROPOSED)
    with pytest.raises(scorer.ScorerV2Error, match="exact v2 manifest hash"):
        scorer.require_execution_authorisation(
            document, document["manifest_sha256"], authorised=False
        )
    with pytest.raises(scorer.ScorerV2Error, match="exact v2 manifest hash"):
        scorer.require_execution_authorisation(document, "0" * 64, authorised=True)
    scorer.require_execution_authorisation(
        document, document["manifest_sha256"], authorised=True
    )


def test_proposal_binds_scorer_test_generation_and_calibration_hashes():
    document = scorer.load_v2_manifest(PROPOSED)
    sources = document["sources"]
    assert sources["scorer_sha256"] == p5.sha256_file(Path(scorer.__file__).resolve())
    assert sources["test_sha256"] == p5.sha256_file(HERE / "test_p5_pilot_scorer_v2.py")
    assert sources["generation_snapshot"]["n_rows"] == 176
    assert sources["final_primary_plan"]["internal_sha256"] == (
        proposal.EXPECTED_PLAN_INTERNAL_SHA256
    )
    assert sources["final_primary_plan"]["file_sha256"] == proposal.EXPECTED_PLAN_FILE_SHA256
    assert sources["calibration"]["results_sha256"] == (
        proposal.EXPECTED_CALIBRATION_RESULTS_SHA256
    )
    assert sources["calibration"]["journal_sha256"] == (
        proposal.EXPECTED_CALIBRATION_JOURNAL_SHA256
    )
