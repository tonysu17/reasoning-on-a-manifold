from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import requests

import p5_preflight as p5
import p5_pilot_safety_v2_2_one_call_diagnostic as diagnostic


HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE / "p5_runs" / "p5-pilot-20260808"
PLAN = HERE / "P5_PILOT_SAFETY_V2_2_ONE_CALL_DIAGNOSTIC_PLAN_2026-08-09.json"
PROPOSED = HERE / "P5_PILOT_SAFETY_V2_2_PROPOSED_ONE_CALL_DIAGNOSTIC_2026-08-09.json"


class FakeResponse:
    def __init__(self, payload, *, status=200, headers=None):
        self.content = json.dumps(payload, separators=(",", ":")).encode()
        self.status_code = status
        self.headers = headers or {"Content-Type": "application/json; charset=utf-8"}


def target():
    binding, _ = diagnostic.target_binding(RUN_ROOT)
    return binding


def test_target_is_exact_frozen_repeated_failure_and_hashes_reverify():
    sources = diagnostic.verify_original_artifacts(RUN_ROOT)
    binding, prompt = diagnostic.target_binding(RUN_ROOT)
    assert len(sources) == 4
    assert binding["logical_assignment_id"] == "v2.2:base_r1:p5sp_weapons_h01"
    assert binding["chunk_index"] == 0
    assert binding["original_failed_call_id"] == diagnostic.TARGET_ORIGINAL_CALL_ID
    assert binding["generation_record_sha256"] == diagnostic.EXPECTED_GENERATION_RECORD_SHA256
    assert binding["chunk_plan_sha256"] == diagnostic.EXPECTED_CHUNK_PLAN_SHA256
    assert binding["diagnostic_prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert "prompt" not in binding


def test_plan_and_fixed_proposal_are_one_call_no_retry_and_excluded():
    plan = diagnostic.build_plan(RUN_ROOT)
    assert plan["status"] == "dry_run_non_executable"
    assert plan["proxy_or_model_calls_made"] == 0
    contract = plan["request_contract"]
    assert contract["model"] == diagnostic.MODEL
    assert contract["max_output_tokens"] == 400
    assert contract["timeout_seconds"] == 25
    assert contract["maximum_total_attempts"] == 1
    assert contract["automatic_retries"] == 0
    assert contract["maximum_cost_per_request_usd"] == 0.025
    assert contract["total_spend_ceiling_usd"] == 0.025
    assert contract["quota_stop_floor_usd"] == 5.0
    exclusions = plan["output_exclusions"]
    assert exclusions["annotations"] is True
    assert exclusions["validation_gate"] is True
    assert exclusions["recovery_outputs"] is True
    assert exclusions["v2_2_artifacts_mutated"] is False
    assert exclusions["v2_2_1_recovery_artifacts_mutated"] is False

    fixed_plan = diagnostic.load_manifest(PLAN)
    fixed_proposed = diagnostic.load_manifest(PROPOSED)
    assert fixed_plan["status"] == "dry_run_non_executable"
    assert fixed_proposed["proposal_status"] == "proposed_awaiting_owner_exact_hash_approval"
    guard = fixed_proposed["authorization_guard"]
    assert guard["approved_request_ceiling"] == 1
    assert guard["approved_retry_ceiling"] == 0
    assert guard["ceiling_product_usd"] <= guard["approved_spend_ceiling_usd"]
    assert guard["owner_approval_received"] is False
    assert guard["annotation_or_recovery_execution_authorized"] is False


def test_authorization_needs_flag_and_exact_internal_and_file_hashes():
    plan = diagnostic.load_manifest(PLAN)
    proposed = diagnostic.load_manifest(PROPOSED)
    proposed_file_hash = p5.sha256_file(PROPOSED)
    with pytest.raises(diagnostic.DiagnosticError, match="dry-run/non-executable"):
        diagnostic.require_authorisation(plan, PLAN, plan["manifest_sha256"], p5.sha256_file(PLAN), True)
    with pytest.raises(diagnostic.DiagnosticError, match="--authorised"):
        diagnostic.require_authorisation(proposed, PROPOSED, proposed["manifest_sha256"], proposed_file_hash, False)
    with pytest.raises(diagnostic.DiagnosticError, match="internal hash"):
        diagnostic.require_authorisation(proposed, PROPOSED, "0" * 64, proposed_file_hash, True)
    with pytest.raises(diagnostic.DiagnosticError, match="file hash"):
        diagnostic.require_authorisation(proposed, PROPOSED, proposed["manifest_sha256"], "0" * 64, True)
    diagnostic.require_authorisation(proposed, PROPOSED, proposed["manifest_sha256"], proposed_file_hash, True)


def test_empty_with_filter_marker_is_sanitized_before_extraction():
    secret_request_id = "provider-secret-request-id"
    response = FakeResponse(
        {
            "content": [],
            "stop_reason": "content_filter",
            "usage": {"input_tokens": 123, "output_tokens": 0, "cost": 0.01},
            "metadata": {"remaining_quota": {"remaining_budget": 100}},
        },
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "X-Request-Id": secret_request_id,
            "Authorization": "response-header-secret",
        },
    )
    telemetry, payload = diagnostic.response_shape_telemetry(response)
    assert payload is not None
    assert telemetry["http_status_class"] == "2xx"
    assert telemetry["response_content_type"] == "application/json"
    assert telemetry["content_container_type"] == "array"
    assert telemetry["content_container_count"] == 0
    assert telemetry["signal_field_presence"]["stop_field_present"] is True
    assert telemetry["signal_field_presence"]["filter_marker_present"] is True
    assert telemetry["usage_cost_usd"] == 0.01
    assert telemetry["remaining_quota_usd"] == 100
    assert telemetry["provider_request_id_sha256"] == hashlib.sha256(secret_request_id.encode()).hexdigest()
    serialized = json.dumps(telemetry)
    assert secret_request_id not in serialized
    assert "response-header-secret" not in serialized
    assert "content_filter" not in serialized
    assert diagnostic.classify_shape(telemetry, "") == "empty_scorer_text_with_filter_marker"


def test_empty_without_marker_and_nonempty_decision_paths_do_not_persist_text():
    empty = FakeResponse({"content": [], "usage": {"cost": 0.001}, "metadata": {"remaining_quota": {"remaining_budget": 99}}})
    empty_telemetry, empty_payload = diagnostic.response_shape_telemetry(empty)
    assert diagnostic.classify_shape(empty_telemetry, "") == "empty_scorer_text_without_filter_marker"
    assert empty_payload == {"content": [], "usage": {"cost": 0.001}, "metadata": {"remaining_quota": {"remaining_budget": 99}}}

    secret_text = "NEVER PERSIST THIS SCORER TEXT"
    nonempty = FakeResponse({"content": [{"type": "text", "text": secret_text}], "usage": {"cost": 0.002}, "metadata": {"remaining_quota": {"remaining_budget": 98}}})
    telemetry, payload = diagnostic.response_shape_telemetry(nonempty)
    assert diagnostic.classify_shape(telemetry, secret_text) == "nonempty_scorer_text"
    assert telemetry["text_block_count"] == 1
    assert telemetry["text_block_byte_lengths"] == [len(secret_text.encode())]
    assert telemetry["text_block_whitespace_only"] == [False]
    assert secret_text not in json.dumps(telemetry)
    assert payload is not None


def test_unsafe_response_key_and_block_type_are_hashed_not_persisted():
    unsafe_key = "unsafe key containing response text"
    unsafe_type = "unsafe type containing response text"
    response = FakeResponse({"content": [{"type": unsafe_type, "text": "x"}], unsafe_key: True})
    telemetry, _ = diagnostic.response_shape_telemetry(response)
    rendered = json.dumps(telemetry)
    assert unsafe_key not in rendered
    assert unsafe_type not in rendered
    assert "unsafe_name_sha256:" in rendered


def test_perform_call_makes_exactly_one_attempt_with_no_retry_and_no_secret_persistence():
    calls = []
    events = []
    secret_url = "https://proxy-secret.invalid/invoke"
    secret_key = "api-secret"
    secret_prompt = "prompt-secret"
    response = FakeResponse({"content": [], "usage": {"cost": 0.003}, "metadata": {"remaining_quota": {"remaining_budget": 97}}})

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return response

    binding = target()
    result = diagnostic.perform_one_call(
        url=secret_url,
        key=secret_key,
        prompt=secret_prompt,
        target=binding,
        emit_event=events.append,
        post=post,
    )
    assert len(calls) == 1
    assert calls[0][1]["timeout"] == 25
    assert calls[0][1]["json"]["max_tokens"] == 400
    assert len(events) == 2
    assert result["network_attempts"] == 1
    assert result["automatic_retries"] == 0
    assert result["outcome"] == "empty_scorer_text_without_filter_marker"
    persisted = json.dumps(events)
    assert secret_url not in persisted
    assert secret_key not in persisted
    assert secret_prompt not in persisted
    assert result["output_excluded_from_annotations"] is True
    assert result["output_excluded_from_validation_gate"] is True
    assert result["output_excluded_from_recovery"] is True


def test_timeout_is_terminal_after_one_attempt():
    calls = []
    events = []

    def post(*args, **kwargs):
        calls.append(1)
        raise requests.Timeout("must not be persisted")

    result = diagnostic.perform_one_call(
        url="https://example.invalid",
        key="secret",
        prompt="prompt",
        target=target(),
        emit_event=events.append,
        post=post,
    )
    assert len(calls) == 1
    assert len(events) == 2
    assert result["outcome"] == "inconclusive_transport_timeout"
    assert result["automatic_retries"] == 0
    assert "must not be persisted" not in json.dumps(result)


def test_plan_and_proposal_builders_never_touch_network(monkeypatch):
    monkeypatch.setattr(diagnostic.requests, "post", lambda *a, **k: pytest.fail("network called"))
    plan = diagnostic.build_plan(RUN_ROOT)
    proposal = diagnostic.build_proposal(PLAN)
    assert plan["proxy_or_model_calls_made"] == 0
    assert proposal["proxy_or_model_calls_made"] == 0
