from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

import p5_pilot_cost_calibration_v1 as calibration
import p5_pilot_scorer_v2 as scorer


HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE / "p5_runs" / "p5-pilot-20260808"
FINAL_PLAN = HERE / "P5_PILOT_SCORER_V2_1_FINAL_PRIMARY_ONLY_PLAN_2026-08-09.json"
CALIBRATION_MANIFEST = HERE / "P5_PILOT_COST_CALIBRATION_MANIFEST_2026-08-09.json"


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def proxy_payload(text: str = "calibration response") -> dict:
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"cost": 0.031},
        "metadata": {"remaining_quota": {"remaining_budget": 12.5}},
    }


def test_final_primary_plan_is_exactly_176_rows_and_490_requests():
    plan = scorer.load_v2_manifest(FINAL_PLAN)
    accounting = plan["dry_run_accounting"]
    assert accounting["generation_rows_snapshot"] == 176
    assert accounting["missing_generation_keys_vs_176"] == 0
    assert accounting["full_176_primary_only_initial_requests"] == 490
    assert accounting["initial_network_requests_current"] == 490
    assert plan["reliability_status"]["repeat_policy"] == "drop"


def test_selection_is_exactly_largest_generic_and_safety_request():
    plan = scorer.load_v2_manifest(FINAL_PLAN)
    descriptors, _ = calibration._request_descriptors(RUN_ROOT, plan)
    selected = calibration.deterministic_selections(descriptors)
    assert [row["request_class"] for row in selected] == ["generic", "safety"]
    for row in selected:
        class_sizes = [
            candidate["canonical_serialized_request_bytes"]
            for candidate in descriptors
            if candidate["request_class"] == row["request_class"]
        ]
        assert row["canonical_serialized_request_bytes"] == max(class_sizes)


def test_calibration_manifest_has_unset_ceilings_and_valid_internal_hash():
    document = calibration.load_manifest(CALIBRATION_MANIFEST)
    assert document["manifest_sha256"] == calibration.manifest_hash(document)
    assert document["status"] == "awaiting_hash_bound_authorization"
    assert document["transport"]["request_ceiling"] == 2
    assert document["transport"]["retry_ceiling"] == 0
    assert len(document["selections"]) == 2
    proposed = document["proposed_authorization"]
    assert proposed["execution_authorized"] is False
    assert proposed["approved_request_ceiling"] is None
    assert proposed["approved_spend_ceiling_usd"] is None
    assert proposed["approved_max_cost_per_call_usd"] is None
    assert proposed["quota_stop_floor_usd"] is None


def test_calibration_execution_requires_both_exact_hashes_and_limits():
    document = calibration.load_manifest(CALIBRATION_MANIFEST)
    file_hash = calibration.p5.sha256_file(CALIBRATION_MANIFEST)
    with pytest.raises(calibration.CalibrationAuthorizationError, match="--authorised"):
        calibration.require_authorization(
            CALIBRATION_MANIFEST,
            document,
            supplied_internal_hash=document["manifest_sha256"],
            supplied_file_hash=file_hash,
            authorised=False,
            approved_request_ceiling=2,
            approved_spend_ceiling_usd=1.0,
            approved_max_cost_per_call_usd=0.5,
            quota_stop_floor_usd=0.0,
        )
    with pytest.raises(calibration.CalibrationAuthorizationError, match="internal hash"):
        calibration.require_authorization(
            CALIBRATION_MANIFEST,
            document,
            supplied_internal_hash="0" * 64,
            supplied_file_hash=file_hash,
            authorised=True,
            approved_request_ceiling=2,
            approved_spend_ceiling_usd=1.0,
            approved_max_cost_per_call_usd=0.5,
            quota_stop_floor_usd=0.0,
        )
    limits = calibration.require_authorization(
        CALIBRATION_MANIFEST,
        document,
        supplied_internal_hash=document["manifest_sha256"],
        supplied_file_hash=file_hash,
        authorised=True,
        approved_request_ceiling=2,
        approved_spend_ceiling_usd=1.0,
        approved_max_cost_per_call_usd=0.5,
        quota_stop_floor_usd=0.0,
    )
    assert limits["approved_request_ceiling"] == 2


def test_calibration_proxy_call_exact_transport_and_cost_only_output(monkeypatch):
    document = calibration.load_manifest(CALIBRATION_MANIFEST)
    selection = document["selections"][0]
    plan = scorer.load_v2_manifest(FINAL_PLAN)
    _, prompt_index = calibration._request_descriptors(RUN_ROOT, plan)
    prompt = prompt_index[selection["selection_id"]]["prompt"]
    calls = []
    events = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(proxy_payload())

    monkeypatch.setattr(calibration.requests, "post", fake_post)
    result, payload = calibration.calibration_proxy_call(
        url="https://proxy.invalid",
        key="FAKE_CALIBRATION_KEY",
        prompt=prompt,
        selection=selection,
        emit_event=events.append,
    )
    assert len(calls) == 1
    request = calls[0][1]
    assert request["json"]["model"] == calibration.MODEL
    assert request["json"]["max_tokens"] == 800
    assert request["timeout"] == 25
    assert result["usage_cost_usd"] == 0.031
    assert result["remaining_quota_usd"] == 12.5
    assert result["raw_model_text_persisted"] is False
    assert result["calibration_only_excluded_from_annotations"] is True
    assert "calibration response" not in json.dumps(result)
    assert "FAKE_CALIBRATION_KEY" not in json.dumps(events)
    assert payload == proxy_payload()


def test_calibration_timeout_has_no_retry(monkeypatch):
    document = calibration.load_manifest(CALIBRATION_MANIFEST)
    selection = document["selections"][0]
    plan = scorer.load_v2_manifest(FINAL_PLAN)
    _, prompt_index = calibration._request_descriptors(RUN_ROOT, plan)
    prompt = prompt_index[selection["selection_id"]]["prompt"]
    calls = []
    events = []

    def timeout(url, **kwargs):
        calls.append(kwargs)
        raise requests.Timeout("timeout")

    monkeypatch.setattr(calibration.requests, "post", timeout)
    with pytest.raises(requests.Timeout):
        calibration.calibration_proxy_call(
            url="https://proxy.invalid",
            key="key",
            prompt=prompt,
            selection=selection,
            emit_event=events.append,
        )
    assert len(calls) == 1
    assert [row["event"] for row in events] == ["reserved", "completed"]
    assert events[-1]["status"] == "error_no_retry"


def test_calibration_output_paths_cannot_collide_with_annotation_outputs():
    assert calibration.RESULTS_NAME not in {scorer.BEHAVIOUR_OUTPUT, scorer.SAFETY_OUTPUT}
    assert calibration.JOURNAL_NAME != scorer.JOURNAL_OUTPUT
