from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import requests

import p5_preflight as p5
import p5_pilot_safety_v2_2_long_endpoint_comparison as longdiag


HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE / "p5_runs" / "p5-pilot-20260808"
PLAN = HERE / "P5_PILOT_SAFETY_V2_2_LONG_ENDPOINT_COMPARISON_PLAN_2026-08-09.json"
AUTHORIZED = HERE / "P5_PILOT_SAFETY_V2_2_AUTHORIZED_LONG_ENDPOINT_COMPARISON_2026-08-09.json"


class FakeResponse:
    status_code = 200
    headers = {"Content-Type": "application/json", "X-Request-Id": "must-be-hashed"}
    content = json.dumps(
        {
            "content": [],
            "usage": {"cost": 0.006},
            "metadata": {"remaining_quota": {"remaining_budget": 200}},
        },
        separators=(",", ":"),
    ).encode()


def test_approved_endpoint_is_exactly_hash_bound_and_not_returned():
    endpoint = longdiag.long_endpoint_from_approved_handoff()
    assert hashlib.sha256(endpoint.encode()).hexdigest() == longdiag.EXPECTED_LONG_ENDPOINT_SHA256
    plan = longdiag.build_plan(RUN_ROOT)
    rendered = json.dumps(plan)
    assert endpoint not in rendered
    assert plan["sources"]["long_endpoint_handoff"]["endpoint_url_persisted"] is False


def test_request_and_target_are_identical_to_completed_standard_diagnostic():
    target, prompt, standard = longdiag.standard_bindings(RUN_ROOT)
    plan = longdiag.build_plan(RUN_ROOT)
    assert plan["target"] == target == standard["source_target"]
    assert plan["request_identity"]["serialized_request_sha256"] == p5.sha256_json(longdiag.standard._request_body(prompt))
    assert plan["request_identity"]["request_body_identical_to_standard"] is True
    assert plan["request_identity"]["diagnostic_prompt_sha256"] == target["diagnostic_prompt_sha256"]
    assert plan["request_identity"]["model"] == longdiag.MODEL
    assert plan["request_identity"]["temperature"] == 0.0
    assert plan["request_identity"]["max_output_tokens"] == 400


def test_transport_changes_only_endpoint_and_remains_one_attempt():
    plan = longdiag.build_plan(RUN_ROOT)
    transport = plan["transport_contract"]
    assert transport["endpoint_class"] == "official_long_120_second"
    assert transport["only_endpoint_base_url_changes"] is True
    assert transport["client_timeout_seconds"] == 25
    assert transport["client_timeout_unchanged_from_standard"] is True
    assert transport["client_timeout_seconds"] <= 120
    assert transport["maximum_total_attempts"] == 1
    assert transport["automatic_retries"] == 0
    guards = plan["financial_guards"]
    assert guards["ceiling_product_usd"] <= guards["total_spend_ceiling_usd"]


def test_authorized_manifest_records_exact_direct_chat_scope():
    manifest = longdiag.load_manifest(AUTHORIZED)
    assert manifest["status"] == "authorized_for_execution"
    owner = manifest["owner_authorization"]
    assert owner["received"] is True
    assert owner["exact_quote"] == longdiag.OWNER_AUTHORIZATION_QUOTE
    assert owner["authorizes_one_long_endpoint_call_without_hash_repetition"] is True
    assert owner["authorized_attempts"] == 1
    assert owner["authorized_retries"] == 0
    assert owner["diagnostic_only"] is True
    assert manifest["authorization_guard"]["recovery_heldout_or_other_calls_authorized"] is False


def test_authorization_integrity_requires_flag_and_both_frozen_hashes():
    manifest = longdiag.load_manifest(AUTHORIZED)
    file_sha = p5.sha256_file(AUTHORIZED)
    with pytest.raises(longdiag.LongEndpointDiagnosticError, match="--authorised"):
        longdiag.require_authorisation(manifest, AUTHORIZED, manifest["manifest_sha256"], file_sha, False)
    with pytest.raises(longdiag.LongEndpointDiagnosticError, match="internal hash"):
        longdiag.require_authorisation(manifest, AUTHORIZED, "0" * 64, file_sha, True)
    with pytest.raises(longdiag.LongEndpointDiagnosticError, match="file hash"):
        longdiag.require_authorisation(manifest, AUTHORIZED, manifest["manifest_sha256"], "0" * 64, True)
    longdiag.require_authorisation(manifest, AUTHORIZED, manifest["manifest_sha256"], file_sha, True)


def test_long_call_is_one_attempt_sanitized_and_uses_identical_body():
    endpoint = longdiag.long_endpoint_from_approved_handoff()
    target, prompt, _ = longdiag.standard_bindings(RUN_ROOT)
    calls = []
    events = []

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeResponse()

    result = longdiag.perform_long_call(
        endpoint=endpoint,
        key="secret-key",
        prompt=prompt,
        target=target,
        emit_event=events.append,
        post=post,
    )
    assert len(calls) == 1
    assert calls[0][0][0] == endpoint
    assert calls[0][1]["json"] == longdiag.standard._request_body(prompt)
    assert calls[0][1]["timeout"] == 25
    assert len(events) == 2
    assert result["network_attempts"] == 1
    assert result["automatic_retries"] == 0
    assert result["outcome"] == "empty_scorer_text_without_filter_marker"
    persisted = json.dumps(events) + json.dumps(result)
    assert endpoint not in persisted
    assert "secret-key" not in persisted
    assert prompt not in persisted
    assert "must-be-hashed" not in persisted
    assert result["endpoint_url_persisted"] is False
    assert result["output_excluded_from_annotations"] is True
    assert result["output_excluded_from_validation_gate"] is True
    assert result["output_excluded_from_recovery"] is True


def test_timeout_stops_without_retry():
    endpoint = longdiag.long_endpoint_from_approved_handoff()
    target, prompt, _ = longdiag.standard_bindings(RUN_ROOT)
    calls = []
    events = []

    def post(*args, **kwargs):
        calls.append(1)
        raise requests.Timeout("not persisted")

    result = longdiag.perform_long_call(
        endpoint=endpoint,
        key="secret",
        prompt=prompt,
        target=target,
        emit_event=events.append,
        post=post,
    )
    assert len(calls) == 1
    assert len(events) == 2
    assert result["outcome"] == "inconclusive_transport_timeout"
    assert result["automatic_retries"] == 0
    assert "not persisted" not in json.dumps(result)


def test_planning_and_authorization_builders_make_no_call(monkeypatch):
    monkeypatch.setattr(longdiag.requests, "post", lambda *a, **k: pytest.fail("network called"))
    plan = longdiag.build_plan(RUN_ROOT)
    authorized = longdiag.build_authorized_manifest(PLAN)
    assert plan["proxy_or_model_calls_made_by_planning"] == 0
    assert authorized["proxy_or_model_calls_made_by_planning"] == 0

