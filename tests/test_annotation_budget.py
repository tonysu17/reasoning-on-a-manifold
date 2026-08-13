"""Fail-closed tests for durable Phase-2 annotation API-attempt budgets."""
from __future__ import annotations

import hashlib
import json

import pytest

from src import annotation
from src.annotation_budget import (
    AnnotationAttemptGuard,
    AnnotationAttemptLimitError,
    AnnotationCostLimitError,
    AnnotationCostTelemetryError,
    SCHEMA_VERSION,
)
from src.annotation_coverage import COVERAGE_RULE_VERSION

#: The executing operational parameters, so the guard fixtures bind to what the
#: pipeline actually sends rather than to invented numbers.
MAX_OUTPUT_TOKENS = 2800
MAX_PROMPT_CHARS = annotation.max_prompt_chars()


SCOPE_A = hashlib.sha256(b"scope-a").hexdigest()
SCOPE_B = hashlib.sha256(b"scope-b").hexdigest()
SCOPE_C = hashlib.sha256(b"scope-c").hexdigest()


def _write_manifest(
    tmp_path,
    *,
    planned=2,
    retries=1,
    total=3,
    per_scope=2,
    spend=10.0,
    max_output_tokens=MAX_OUTPUT_TOKENS,
    max_prompt_chars=MAX_PROMPT_CHARS,
    max_cost_per_attempt=0.05,
    prior_spend=0.0,
):
    source = tmp_path / "source.json"
    source.write_text('[{"chain":"bounded test"}]')
    code = tmp_path / "code.py"
    code.write_text("# hash-bound executing code\n")
    doc = {
        "schema": SCHEMA_VERSION,
        "status": "authorised",
        "model": annotation.ANNOTATION_MODEL,
        "annotation_window_tokens": 3000,
        "guards": {
            "planned_initial_requests": planned,
            "global_retry_capacity": retries,
            "max_total_attempts": total,
            "max_attempts_per_scope": per_scope,
            "approved_spend_ceiling_usd": spend,
            "max_cost_per_attempt_usd": max_cost_per_attempt,
            "remaining_quota_floor_usd": 5.0,
            "max_output_tokens": max_output_tokens,
            "max_prompt_chars": max_prompt_chars,
            "prior_committed_spend_usd": prior_spend,
        },
        "rate_card": {
            "input_usd_per_mtok": 3.0,
            "output_usd_per_mtok": 15.0,
            "tokens_per_char_upper": 0.4,
        },
        "coverage_rule_version": COVERAGE_RULE_VERSION,
        "source_artifacts": [{
            "path": "source.json",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }],
        "bound_artifacts": [{
            "path": "code.py",
            "sha256": hashlib.sha256(code.read_bytes()).hexdigest(),
        }],
    }
    manifest = tmp_path / "guard.json"
    manifest.write_text(json.dumps(doc, sort_keys=True))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return manifest, digest, source


def _load_guard(tmp_path, manifest, digest, *, journal="attempts.jsonl",
                max_output_tokens=MAX_OUTPUT_TOKENS,
                max_prompt_chars=MAX_PROMPT_CHARS):
    return AnnotationAttemptGuard.from_manifest(
        manifest,
        digest,
        tmp_path / journal,
        root=tmp_path,
        expected_model=annotation.ANNOTATION_MODEL,
        expected_annotation_window_tokens=3000,
        expected_bound_paths=("code.py",),
        expected_max_output_tokens=max_output_tokens,
        expected_max_prompt_chars=max_prompt_chars,
        expected_coverage_rule_version=COVERAGE_RULE_VERSION,
    )


def test_global_and_per_scope_limits_persist_across_restarts(tmp_path):
    manifest, digest, _ = _write_manifest(tmp_path)
    guard = _load_guard(tmp_path, manifest, digest)
    guard.reserve(SCOPE_A)
    guard.reserve(SCOPE_A)

    restarted = _load_guard(tmp_path, manifest, digest)
    with pytest.raises(AnnotationAttemptLimitError, match="per-chunk"):
        restarted.reserve(SCOPE_A)
    restarted.reserve(SCOPE_B)
    with pytest.raises(AnnotationAttemptLimitError, match="initial annotation"):
        restarted.reserve(SCOPE_C)

    summary = restarted.summary()
    assert summary["attempts_reserved"] == 3
    assert summary["max_scope_attempts_observed"] == 2


def test_retry_pool_cannot_consume_later_initial_attempts(tmp_path):
    manifest, digest, _ = _write_manifest(
        tmp_path, planned=3, retries=1, total=4, per_scope=3
    )
    guard = _load_guard(tmp_path, manifest, digest)
    guard.reserve(SCOPE_A)  # initial
    guard.reserve(SCOPE_A)  # sole authorized retry
    with pytest.raises(AnnotationAttemptLimitError, match="retry ceiling"):
        guard.reserve(SCOPE_A)
    # The two remaining initial scopes are still authorized.
    guard.reserve(SCOPE_B)
    guard.reserve(SCOPE_C)
    summary = guard.summary()
    assert summary["initial_attempts_reserved"] == 3
    assert summary["retry_attempts_reserved"] == 1


def test_failed_calls_cannot_gain_fresh_retries_after_restart(tmp_path, monkeypatch):
    manifest, digest, _ = _write_manifest(
        tmp_path, planned=1, retries=2, total=3, per_scope=3
    )
    calls = {"n": 0}

    def fail_proxy(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("synthetic transport failure")

    monkeypatch.setattr(annotation, "_proxy_call", fail_proxy)
    monkeypatch.setattr(annotation.time, "sleep", lambda *_: None)

    first = _load_guard(tmp_path, manifest, digest)
    assert annotation._annotate_single(
        "some chain", max_retries=3, max_tokens=MAX_OUTPUT_TOKENS,
        attempt_guard=first, attempt_scope=SCOPE_A,
    ) == []
    assert calls["n"] == 3

    restarted = _load_guard(tmp_path, manifest, digest)
    assert annotation._annotate_single(
        "some chain", max_retries=3, max_tokens=MAX_OUTPUT_TOKENS,
        attempt_guard=restarted, attempt_scope=SCOPE_A,
    ) == []
    assert calls["n"] == 3


def test_corrupt_or_foreign_journal_fails_closed(tmp_path):
    manifest, digest, _ = _write_manifest(tmp_path)
    guard = _load_guard(tmp_path, manifest, digest)
    guard.journal_path.write_text("not-json\n")
    with pytest.raises(AnnotationAttemptLimitError, match="corrupt"):
        guard.reserve(SCOPE_A)


def test_manifest_and_source_hashes_are_mandatory(tmp_path):
    manifest, digest, source = _write_manifest(tmp_path)
    with pytest.raises(AnnotationAttemptLimitError, match="manifest SHA-256 mismatch"):
        _load_guard(tmp_path, manifest, "0" * 64)

    source.write_text("changed after authorisation")
    with pytest.raises(AnnotationAttemptLimitError, match="hash-mismatched"):
        _load_guard(tmp_path, manifest, digest)


def test_manifest_cannot_authorise_more_than_three_attempts_per_scope(tmp_path):
    manifest, digest, _ = _write_manifest(
        tmp_path, planned=1, retries=3, total=4, per_scope=4
    )
    with pytest.raises(AnnotationAttemptLimitError, match="hard project limit of 3"):
        _load_guard(tmp_path, manifest, digest)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("approved_spend_ceiling_usd", "approved_spend"),
        ("max_cost_per_attempt_usd", "max_cost"),
        ("remaining_quota_floor_usd", "quota"),
    ],
)
def test_manifest_rejects_nonfinite_money_limits(tmp_path, field, message):
    manifest, _, _ = _write_manifest(tmp_path)
    doc = json.loads(manifest.read_text())
    doc["guards"][field] = float("nan")
    manifest.write_text(json.dumps(doc, sort_keys=True))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(AnnotationAttemptLimitError, match=message):
        _load_guard(tmp_path, manifest, digest)


def test_exact_initial_request_plan_is_required(tmp_path):
    manifest, digest, _ = _write_manifest(tmp_path)
    guard = _load_guard(tmp_path, manifest, digest)
    guard.assert_planned_initial_requests(2)
    with pytest.raises(AnnotationAttemptLimitError, match="request-plan mismatch"):
        guard.assert_planned_initial_requests(3)


def test_journal_contains_no_prompt_or_response_text(tmp_path):
    manifest, digest, _ = _write_manifest(tmp_path)
    guard = _load_guard(tmp_path, manifest, digest)
    guard.reserve(SCOPE_A)
    event = json.loads(guard.journal_path.read_text())
    assert set(event) == {
        "schema", "event", "reservation_id", "reserved_at_utc", "pid",
        "manifest_file_sha256", "scope_sha256", "ordinal", "scope_ordinal",
        "attempt_kind",
    }
    assert "bounded test" not in guard.journal_path.read_text()


def test_unaccounted_attempts_are_conservatively_cost_committed(tmp_path):
    manifest, digest, _ = _write_manifest(
        tmp_path, planned=3, retries=0, total=3, per_scope=1, spend=0.10
    )
    guard = _load_guard(tmp_path, manifest, digest)
    guard.reserve(SCOPE_A)
    guard.reserve(SCOPE_B)
    with pytest.raises(AnnotationCostLimitError, match="spend ceiling"):
        guard.reserve(SCOPE_C)
    assert guard.summary()["committed_cost_usd"] == pytest.approx(0.10)


def test_reported_cost_replaces_conservative_commitment(tmp_path):
    manifest, digest, _ = _write_manifest(
        tmp_path, planned=3, retries=0, total=3, per_scope=1, spend=0.10
    )
    guard = _load_guard(tmp_path, manifest, digest)
    event = guard.reserve(SCOPE_A)
    guard.record_response(event["reservation_id"], 0.01, 100.0)
    guard.reserve(SCOPE_B)
    summary = guard.summary()
    assert summary["reported_cost_usd"] == pytest.approx(0.01)
    assert summary["committed_cost_usd"] == pytest.approx(0.06)


def test_missing_success_telemetry_fails_closed(tmp_path):
    manifest, digest, _ = _write_manifest(tmp_path)
    guard = _load_guard(tmp_path, manifest, digest)
    event = guard.reserve(SCOPE_A)
    with pytest.raises(AnnotationCostTelemetryError):
        guard.record_response(event["reservation_id"], None, None)
    with pytest.raises(AnnotationCostTelemetryError, match="omitted"):
        guard.reserve(SCOPE_B)


def test_nonfinite_success_telemetry_fails_closed(tmp_path):
    manifest, digest, _ = _write_manifest(tmp_path)
    guard = _load_guard(tmp_path, manifest, digest)
    event = guard.reserve(SCOPE_A)
    with pytest.raises(AnnotationCostTelemetryError):
        guard.record_response(event["reservation_id"], float("nan"), 100.0)
    with pytest.raises(AnnotationCostTelemetryError, match="omitted"):
        guard.reserve(SCOPE_B)


def test_remaining_quota_floor_fails_closed(tmp_path):
    manifest, digest, _ = _write_manifest(tmp_path)
    guard = _load_guard(tmp_path, manifest, digest)
    event = guard.reserve(SCOPE_A)
    guard.record_response(event["reservation_id"], 0.01, 4.99)
    with pytest.raises(AnnotationCostLimitError, match="quota"):
        guard.reserve(SCOPE_B)


def test_annotation_call_records_cost_before_parsing(tmp_path, monkeypatch):
    manifest, digest, _ = _write_manifest(
        tmp_path, planned=1, retries=0, total=1, per_scope=1
    )
    guard = _load_guard(tmp_path, manifest, digest)
    good = '["deduction"]Therefore.["end-section"]'
    monkeypatch.setattr(
        annotation, "_proxy_call",
        lambda *args, **kwargs: annotation.ProxyText(
            good, usage_cost=0.012, remaining_quota=80.0
        ),
    )
    spans = annotation._annotate_single(
        "chain", max_retries=1, max_tokens=MAX_OUTPUT_TOKENS,
        attempt_guard=guard, attempt_scope=SCOPE_A
    )
    assert spans[0]["label"] == "deduction"
    summary = guard.summary()
    assert summary["reported_cost_usd"] == pytest.approx(0.012)
    assert summary["committed_cost_usd"] == pytest.approx(0.012)
