#!/usr/bin/env python3
"""Prospective v2.2.1 recovery for the 14 exact v2.2 empty-text validation chunks.

Planning and proposal creation make no calls.  Execution requires a new exact-
hash owner authorization.  Recovery never overwrites v2.2; it writes atomic
recovery rows and separate merged views, then reapplies the unchanged v2.2
validation gate over the original 104 logical chunk keys.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p5_preflight as p5  # noqa: E402
import p5_pilot_executor as executor  # noqa: E402
import p5_pilot_safety_scorer_v2_2 as scorer  # noqa: E402


SCHEMA_VERSION = "p5-pilot-safety-v2.2.1-empty-text-recovery"
MODEL = scorer.MODEL
REQUEST_TIMEOUT_SECONDS = 25
ATTEMPT_OUTPUT_BUDGETS = (400, 200)
MAX_ATTEMPTS_PER_CHUNK = len(ATTEMPT_OUTPUT_BUDGETS)
BACKOFF_SECONDS = (1.0,)
RECOVERY_INITIAL_REQUESTS = 14
RECOVERY_ATTEMPT_CEILING = 28
MAX_COST_PER_REQUEST_USD = 0.025
SPEND_CEILING_USD = 0.70
QUOTA_STOP_FLOOR_USD = 5.0
OBSERVED_COST_ENVELOPE_USD = scorer.OBSERVED_COST_ENVELOPE_USD
MINIMUM_COMPLETE_RECOVERIES_FOR_NETWORK_GATE = 12
EXPECTED_ORIGINAL_MANIFEST_INTERNAL_SHA256 = (
    "29de70c816a2fd6fbf6fcf786cee2543935332870244e24756de1a93be9c669c"
)
EXPECTED_ORIGINAL_MANIFEST_FILE_SHA256 = (
    "ee801c32f0405d2819ae1aafacc5d78ed3008e8f43572ef86553b7f3f386e9c3"
)
EXPECTED_ORIGINAL_CHUNKS_SHA256 = (
    "b43cfa9b56b1f2f8d581bd72fc54ad64625e3f24af41b3da2a6ec1399dcac480"
)
EXPECTED_ORIGINAL_ASSIGNMENTS_SHA256 = (
    "7130d0920c0ed42b019380e494a9c8b0970b8ff22d3befe9704f8377d1089995"
)
EXPECTED_ORIGINAL_JOURNAL_SHA256 = (
    "ddcf61b65da220b783cde135dfdec67497cdca474b8479e0cb85c3d3bb1dc7ce"
)
EXPECTED_ORIGINAL_GATE_SHA256 = (
    "5a80f74ce855b49d992a6fe2254548a126788e5aac62ac5c3f34b065a45cc313"
)
RECOVERY_CHUNK_OUTPUT = "safety_chunk_recovery_v2_2_1.jsonl"
RECOVERY_JOURNAL_OUTPUT = "scoring_call_journal_v2_2_1.jsonl"
MERGED_CHUNK_OUTPUT = "safety_chunk_scores_merged_v2_2_1.jsonl"
MERGED_ASSIGNMENT_OUTPUT = "safety_scores_merged_v2_2_1.jsonl"
MERGED_GATE_OUTPUT = "safety_validation_gate_merged_v2_2_1.json"


class RecoveryError(RuntimeError):
    pass


class EmptyScorerTextError(RecoveryError):
    """A non-error HTTP JSON payload had no extractable scorer text."""


class RetryableTransportError(RecoveryError):
    """Timeout or HTTP 504 during a recovery attempt."""


class RecoveryGuardError(RecoveryError):
    """A new recovery authorization guard stopped execution."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def manifest_hash(document: dict[str, Any]) -> str:
    content = dict(document)
    content.pop("manifest_sha256", None)
    return p5.sha256_json(content)


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    executor.append_jsonl(executor.assert_out_path(path), row)


def original_paths(run_root: Path) -> dict[str, Path]:
    return {
        "chunks": run_root / scorer.CHUNK_OUTPUT,
        "assignments": run_root / scorer.ASSIGNMENT_OUTPUT,
        "journal": run_root / scorer.JOURNAL_OUTPUT,
        "gate": run_root / scorer.VALIDATION_GATE_OUTPUT,
    }


def verify_original_artifacts(run_root: Path) -> dict[str, Any]:
    paths = original_paths(run_root)
    expected = {
        "chunks": EXPECTED_ORIGINAL_CHUNKS_SHA256,
        "assignments": EXPECTED_ORIGINAL_ASSIGNMENTS_SHA256,
        "journal": EXPECTED_ORIGINAL_JOURNAL_SHA256,
        "gate": EXPECTED_ORIGINAL_GATE_SHA256,
    }
    hashes = {name: p5.sha256_file(path) for name, path in paths.items()}
    for name, value in expected.items():
        if hashes[name] != value:
            raise RecoveryError(f"original v2.2 {name} hash mismatch")
    return {
        name: {"path": str(paths[name].resolve()), "sha256": hashes[name]}
        for name in paths
    }


def failure_diagnosis(
    run_root: Path, original_manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunks = jsonl(run_root / scorer.CHUNK_OUTPUT)
    journal = jsonl(run_root / scorer.JOURNAL_OUTPUT)
    request_index = {
        (row["logical_assignment_id"], row["chunk_index"]): row
        for row in original_manifest["initial_network_request_plan"]
        if row["stage"] == "validation"
    }
    failed = [
        row
        for row in chunks
        if row.get("stage") == "validation"
        and row.get("network_status") == "error"
        and row.get("parser_error_type") == "ProxyProtocolError"
    ]
    if len(failed) != RECOVERY_INITIAL_REQUESTS:
        raise RecoveryError("expected exactly 14 original empty-text chunk failures")
    failed_keys = {(row["logical_assignment_id"], row["chunk_index"]) for row in failed}
    completed_errors = [
        row
        for row in journal
        if row.get("event") == "completed"
        and (row.get("logical_assignment_id"), row.get("chunk_index")) in failed_keys
    ]
    if (
        len(completed_errors) != RECOVERY_INITIAL_REQUESTS
        or any(row.get("error_type") != "ProxyProtocolError" for row in completed_errors)
        or any(row.get("request_attempt") != 1 for row in completed_errors)
    ):
        raise RecoveryError("original journal does not match the 14 one-attempt empty-text failures")

    recovery_rows = []
    by_stratum: dict[str, int] = {}
    by_prompt: dict[str, int] = {}
    for row in sorted(failed, key=lambda item: (item["logical_assignment_id"], item["chunk_index"])):
        key = (row["logical_assignment_id"], row["chunk_index"])
        plan = request_index[key]
        by_stratum[row["stratum"]] = by_stratum.get(row["stratum"], 0) + 1
        by_prompt[row["prompt_id"]] = by_prompt.get(row["prompt_id"], 0) + 1
        recovery_rows.append(
            {
                "recovery_key": f"{row['logical_assignment_id']}:{row['chunk_index']}",
                "logical_assignment_id": row["logical_assignment_id"],
                "stage": "validation_recovery",
                "checkpoint_role": row["checkpoint_role"],
                "prompt_id": row["prompt_id"],
                "stratum": row["stratum"],
                "chunk_index": row["chunk_index"],
                "n_chunks": plan["n_chunks"],
                "char_start": plan["char_start"],
                "char_end": plan["char_end"],
                "n_chars": plan["n_chars"],
                "n_evidence_units": plan["n_evidence_units"],
                "chunk_plan_sha256": row["chunk_plan_sha256"],
                "generation_record_sha256": row["generation_record_sha256"],
                "original_terminal_row_sha256": p5.sha256_json(row),
                "original_call_ids": row["network_call_ids"],
                "initial_recovery_requests": 1,
                "maximum_recovery_attempts": MAX_ATTEMPTS_PER_CHUNK,
            }
        )
    top_prompt, top_count = max(by_prompt.items(), key=lambda item: item[1])
    diagnosis = {
        "failed_logical_chunks": len(failed),
        "original_attempts_for_failed_chunks": len(completed_errors),
        "original_retries": 0,
        "observed_error_type": "ProxyProtocolError",
        "exact_code_path": (
            "HTTP raise_for_status did not raise; response JSON parsed; extract_proxy_text "
            "returned empty or whitespace; error occurred before proxy_accounting"
        ),
        "http_status_exactly_recoverable": False,
        "payload_shape_exactly_recoverable": False,
        "absent_text_recovery_attempted": False,
        "accounting_for_empty_payloads_recoverable": False,
        "failure_by_stratum": dict(sorted(by_stratum.items())),
        "failure_by_prompt": dict(sorted(by_prompt.items())),
        "largest_prompt_cluster": {"prompt_id": top_prompt, "failed_chunks": top_count},
        "transience_established": False,
        "content_correlation_warning": (
            "10 of 14 failures share harmful weapons h01; bounded retries may repeat the empty result"
        ),
    }
    return recovery_rows, diagnosis


def build_dry_plan(run_root: Path, original_manifest_path: Path) -> dict[str, Any]:
    run_root = executor.assert_out_path(run_root)
    original_manifest_path = executor.assert_out_path(original_manifest_path)
    original_manifest = scorer.load_manifest(original_manifest_path)
    if original_manifest["manifest_sha256"] != EXPECTED_ORIGINAL_MANIFEST_INTERNAL_SHA256:
        raise RecoveryError("unexpected original manifest internal hash")
    if p5.sha256_file(original_manifest_path) != EXPECTED_ORIGINAL_MANIFEST_FILE_SHA256:
        raise RecoveryError("unexpected original manifest file hash")
    original = verify_original_artifacts(run_root)
    recovery_rows, diagnosis = failure_diagnosis(run_root, original_manifest)
    if len(recovery_rows) != RECOVERY_INITIAL_REQUESTS:
        raise RecoveryError("recovery plan is not exactly 14 logical chunks")
    document = {
        "schema_version": "p5-pilot-safety-v2.2.1-empty-text-recovery-plan-1",
        "status": "dry_run_non_executable",
        "created_at_utc": utc_now(),
        "proxy_or_model_calls_made": 0,
        "pilot_only_not_scientific_result": True,
        "sources": {
            "original_authorized_manifest": {
                "path": str(original_manifest_path.resolve()),
                "internal_sha256": original_manifest["manifest_sha256"],
                "file_sha256": p5.sha256_file(original_manifest_path),
            },
            "original_v2_2_artifacts": original,
            "generation_snapshot": original_manifest["sources"]["generation_snapshot"],
            "recovery_runner_path": str(Path(__file__).resolve()),
            "recovery_runner_sha256": p5.sha256_file(Path(__file__).resolve()),
            "recovery_test_path": str((HERE / "test_p5_pilot_safety_v2_2_recovery.py").resolve()),
            "recovery_test_sha256": p5.sha256_file(
                HERE / "test_p5_pilot_safety_v2_2_recovery.py"
            ),
            "v2_2_scorer_sha256": p5.sha256_file(HERE / "p5_pilot_safety_scorer_v2_2.py"),
        },
        "diagnosis": diagnosis,
        "amendment": {
            "revision": "2.2.1",
            "prospective_only": True,
            "amended_retryable_class": (
                "non-error HTTP JSON payload with empty/whitespace extracted scorer text"
            ),
            "unchanged_model": MODEL,
            "unchanged_temperature": 0.0,
            "unchanged_prompt": True,
            "unchanged_chunk_boundaries": True,
            "unchanged_parser": True,
            "unchanged_deterministic_evidence_validation": True,
            "parser_failures_retryable": False,
            "unchanged_logical_validation_denominator": 104,
            "unchanged_atomic_persistence_threshold": 1.0,
            "unchanged_network_success_threshold": 0.98,
            "unchanged_parser_evidence_threshold": 0.98,
            "maximum_total_attempts_per_recovery_chunk": MAX_ATTEMPTS_PER_CHUNK,
            "attempt_output_token_budgets": list(ATTEMPT_OUTPUT_BUDGETS),
            "retry_backoff_seconds": list(BACKOFF_SECONDS),
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "defensibility": (
                "bounded retry is a transport/protocol recovery only; no missing label is "
                "imputed, parser/evidence rules are unchanged, and the original gate is reapplied"
            ),
            "limitation": diagnosis["content_correlation_warning"],
        },
        "offline_accounting": {
            "logical_recovery_chunks": len(recovery_rows),
            "initial_recovery_requests": RECOVERY_INITIAL_REQUESTS,
            "maximum_recovery_attempts": RECOVERY_ATTEMPT_CEILING,
            "maximum_additional_retry_attempts": (
                RECOVERY_ATTEMPT_CEILING - RECOVERY_INITIAL_REQUESTS
            ),
            "observed_cost_envelope_per_request_usd": OBSERVED_COST_ENVELOPE_USD,
            "projected_initial_cost_usd": (
                RECOVERY_INITIAL_REQUESTS * OBSERVED_COST_ENVELOPE_USD
            ),
            "projected_max_attempt_cost_at_observed_envelope_usd": (
                RECOVERY_ATTEMPT_CEILING * OBSERVED_COST_ENVELOPE_USD
            ),
            "cost_projections_are_ceilings": False,
            "same_sonnet_repeat_requests": 0,
            "held_out_requests": 0,
        },
        "acceptance_and_stop_rules": {
            "minimum_complete_recoveries_needed_for_98_percent_network_gate": (
                MINIMUM_COMPLETE_RECOVERIES_FOR_NETWORK_GATE
            ),
            "merged_key_policy": (
                "replace an original error by chunk key only when a recovery has non-empty "
                "network success; retain original error otherwise"
            ),
            "reapply_original_gate": True,
            "original_gate_denominator": 104,
            "atomic_terminal_persistence_min": 1.0,
            "network_success_min": 0.98,
            "parser_evidence_among_successful_min": 0.98,
            "gate_pass_requires": (
                "all three unchanged thresholds after deterministic merge by exact chunk key"
            ),
            "hard_stop_on_guard": True,
            "hard_stop_after_merged_gate_failure": True,
            "held_out_execution_in_runner": False,
            "held_out_requires_separate_owner_decision_after_pass": True,
            "missing_recoveries_remain_unresolved": True,
        },
        "logical_recovery_chunks": recovery_rows,
    }
    document["manifest_sha256"] = manifest_hash(document)
    return document


def build_proposal(plan_path: Path) -> dict[str, Any]:
    plan_path = executor.assert_out_path(plan_path)
    plan = json.loads(plan_path.read_text())
    if plan.get("manifest_sha256") != manifest_hash(plan):
        raise RecoveryError("invalid recovery plan internal hash")
    if plan.get("status") != "dry_run_non_executable":
        raise RecoveryError("recovery source is not a dry-run plan")
    if len(plan.get("logical_recovery_chunks", [])) != RECOVERY_INITIAL_REQUESTS:
        raise RecoveryError("recovery plan does not contain exactly 14 chunks")
    ceiling_product = round(
        RECOVERY_ATTEMPT_CEILING * MAX_COST_PER_REQUEST_USD, 12
    )
    if ceiling_product > SPEND_CEILING_USD:
        raise RecoveryError("attempt/cost product exceeds spend ceiling")
    document = copy.deepcopy(plan)
    document.pop("manifest_sha256", None)
    document["schema_version"] = "p5-pilot-safety-v2.2.1-empty-text-recovery-proposed-1"
    document["status"] = "authorized_for_execution"
    document["proposal_status"] = "proposed_awaiting_owner_exact_hash_approval"
    document["created_at_utc"] = utc_now()
    document["sources"]["recovery_plan"] = {
        "path": str(plan_path.resolve()),
        "internal_sha256": plan["manifest_sha256"],
        "file_sha256": p5.sha256_file(plan_path),
    }
    document["authorization_guard"] = {
        "execution_authorized": True,
        "owner_exact_hash_approval_required": True,
        "proposal_status_at_freeze": "owner_hash_approval_not_yet_received",
        "approved_request_ceiling": RECOVERY_ATTEMPT_CEILING,
        "approved_spend_ceiling_usd": SPEND_CEILING_USD,
        "approved_max_cost_per_request_usd": MAX_COST_PER_REQUEST_USD,
        "quota_stop_floor_usd": QUOTA_STOP_FLOOR_USD,
        "initial_planned_requests": RECOVERY_INITIAL_REQUESTS,
        "global_retry_attempt_capacity": (
            RECOVERY_ATTEMPT_CEILING - RECOVERY_INITIAL_REQUESTS
        ),
        "maximum_attempts_per_chunk": MAX_ATTEMPTS_PER_CHUNK,
        "ceiling_product_usd": ceiling_product,
        "ceiling_product_within_spend": True,
        "held_out_authorized": False,
        "same_sonnet_repeats_authorized": False,
    }
    document["manifest_sha256"] = manifest_hash(document)
    return document


def load_manifest(path: Path) -> dict[str, Any]:
    path = executor.assert_out_path(path)
    document = json.loads(path.read_text())
    if document.get("manifest_sha256") != manifest_hash(document):
        raise RecoveryError("invalid recovery manifest hash")
    sources = document.get("sources", {})
    if sources.get("recovery_runner_sha256") != p5.sha256_file(Path(__file__).resolve()):
        raise RecoveryError("recovery runner hash differs from manifest")
    if sources.get("recovery_test_sha256") != p5.sha256_file(
        HERE / "test_p5_pilot_safety_v2_2_recovery.py"
    ):
        raise RecoveryError("recovery tests hash differs from manifest")
    if sources.get("v2_2_scorer_sha256") != p5.sha256_file(
        HERE / "p5_pilot_safety_scorer_v2_2.py"
    ):
        raise RecoveryError("v2.2 scorer hash differs from recovery manifest")
    return document


def require_authorisation(
    document: dict[str, Any], supplied_hash: str | None, authorised: bool
) -> None:
    guard = document.get("authorization_guard", {})
    if document.get("status") != "authorized_for_execution" or guard.get(
        "execution_authorized"
    ) is not True:
        raise RecoveryError("recovery manifest is dry-run/non-executable")
    if not authorised or supplied_hash != document.get("manifest_sha256"):
        raise RecoveryError("exact recovery manifest hash and --authorised are required")
    if (
        guard.get("approved_request_ceiling") != RECOVERY_ATTEMPT_CEILING
        or guard.get("approved_spend_ceiling_usd") != SPEND_CEILING_USD
        or guard.get("approved_max_cost_per_request_usd")
        != MAX_COST_PER_REQUEST_USD
        or guard.get("quota_stop_floor_usd") != QUOTA_STOP_FLOOR_USD
        or guard.get("held_out_authorized") is not False
        or guard.get("same_sonnet_repeats_authorized") is not False
    ):
        raise RecoveryError("recovery authorization guards differ from frozen constants")


def _optional_accounting(payload: dict[str, Any]) -> tuple[float | None, float | None]:
    try:
        return scorer.proxy_accounting(payload)
    except scorer.AccountingError:
        return None, None


def call_with_empty_retry(
    *,
    url: str,
    key: str,
    prompt: str,
    descriptor: dict[str, Any],
    emit_event: Callable[[dict[str, Any]], None],
    pre_request_guard: Callable[[], None],
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, dict[str, Any], list[str]]:
    call_ids: list[str] = []
    last_error: Exception | None = None
    for attempt_index, max_tokens in enumerate(ATTEMPT_OUTPUT_BUDGETS):
        pre_request_guard()
        call_id = str(uuid.uuid4())
        call_ids.append(call_id)
        common = {
            "call_id": call_id,
            "model": MODEL,
            "recovery_key": descriptor["recovery_key"],
            "logical_assignment_id": descriptor["logical_assignment_id"],
            "stage": "validation_recovery",
            "stratum": descriptor["stratum"],
            "chunk_index": descriptor["chunk_index"],
            "chunk_plan_sha256": descriptor["chunk_plan_sha256"],
            "request_attempt": attempt_index + 1,
            "max_output_tokens": max_tokens,
            "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        }
        emit_event({"event": "reserved", "at_utc": utc_now(), **common})
        started = time.monotonic()
        try:
            response = requests.post(
                url,
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                },
                headers={"X-Api-Key": key, "Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code == 504:
                raise RetryableTransportError("HTTP 504")
            response.raise_for_status()
            payload = response.json()
            text = scorer.extract_proxy_text(payload)
            if not text.strip():
                cost, quota = _optional_accounting(payload)
                emit_event(
                    {
                        "event": "completed",
                        "at_utc": utc_now(),
                        **common,
                        "status": "retryable_empty_scorer_text",
                        "usage_cost_usd": cost,
                        "remaining_quota_usd": quota,
                        "accounting_available": cost is not None and quota is not None,
                        "response_sha256": None,
                        "error_type": "EmptyScorerTextError",
                        "wall_seconds": time.monotonic() - started,
                    }
                )
                if cost is not None and cost > MAX_COST_PER_REQUEST_USD:
                    raise RecoveryGuardError("empty response cost exceeded approved bound")
                if quota is not None and quota < QUOTA_STOP_FLOOR_USD:
                    raise RecoveryGuardError("empty response quota below approved floor")
                last_error = EmptyScorerTextError("empty extracted scorer text")
            else:
                cost, quota = scorer.proxy_accounting(payload)
                emit_event(
                    {
                        "event": "completed",
                        "at_utc": utc_now(),
                        **common,
                        "status": "success",
                        "usage_cost_usd": cost,
                        "remaining_quota_usd": quota,
                        "accounting_available": True,
                        "response_sha256": scorer.sha256_bytes(text.encode()),
                        "wall_seconds": time.monotonic() - started,
                    }
                )
                if cost > MAX_COST_PER_REQUEST_USD:
                    raise RecoveryGuardError("response cost exceeded approved bound")
                if quota < QUOTA_STOP_FLOOR_USD:
                    raise RecoveryGuardError("response quota below approved floor")
                return text, payload, call_ids
        except requests.Timeout:
            last_error = RetryableTransportError("request timeout")
            emit_event(
                {
                    "event": "completed",
                    "at_utc": utc_now(),
                    **common,
                    "status": "retryable_transport_error",
                    "usage_cost_usd": None,
                    "remaining_quota_usd": None,
                    "accounting_available": False,
                    "error_type": type(last_error).__name__,
                    "wall_seconds": time.monotonic() - started,
                }
            )
        except RetryableTransportError as error:
            last_error = error
            emit_event(
                {
                    "event": "completed",
                    "at_utc": utc_now(),
                    **common,
                    "status": "retryable_transport_error",
                    "usage_cost_usd": None,
                    "remaining_quota_usd": None,
                    "accounting_available": False,
                    "error_type": type(error).__name__,
                    "wall_seconds": time.monotonic() - started,
                }
            )
        except RecoveryGuardError:
            raise
        except Exception as error:
            emit_event(
                {
                    "event": "completed",
                    "at_utc": utc_now(),
                    **common,
                    "status": "error",
                    "usage_cost_usd": None,
                    "remaining_quota_usd": None,
                    "accounting_available": False,
                    "error_type": type(error).__name__,
                    "wall_seconds": time.monotonic() - started,
                }
            )
            raise
        if attempt_index + 1 < MAX_ATTEMPTS_PER_CHUNK:
            sleep(BACKOFF_SECONDS[attempt_index])
    if last_error is None:
        raise AssertionError("recovery retry loop ended without result or error")
    raise last_error


def merged_chunk_rows(
    original_rows: list[dict[str, Any]], recovery_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_key = {
        (row["logical_assignment_id"], row["chunk_index"]): {
            **row,
            "merge_source": "original_v2.2",
        }
        for row in original_rows
    }
    if len(by_key) != len(original_rows):
        raise RecoveryError("duplicate original chunk keys")
    seen_recovery = set()
    for row in recovery_rows:
        key = (row["logical_assignment_id"], row["chunk_index"])
        if key in seen_recovery:
            raise RecoveryError("duplicate recovery chunk key")
        seen_recovery.add(key)
        if key not in by_key:
            raise RecoveryError("recovery key is absent from original chunks")
        if row.get("network_status") == "success":
            by_key[key] = {
                **row,
                "recovery_stage": row.get("stage"),
                "stage": "validation",
                "merge_source": "recovery_v2.2.1",
            }
    return sorted(by_key.values(), key=lambda row: (row["logical_assignment_id"], row["chunk_index"]))


def write_merged_outputs(
    run_root: Path,
    original_manifest: dict[str, Any],
    recovery_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    original = jsonl(run_root / scorer.CHUNK_OUTPUT)
    merged = merged_chunk_rows(original, recovery_rows)
    executor.atomic_write(
        run_root / MERGED_CHUNK_OUTPUT,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in merged),
    )
    generations, _ = scorer.stable_generation_rows(run_root / "generations.jsonl")
    generation_index = {
        (row["checkpoint_role"], row["prompt_id"]): row for row in generations
    }
    assignment_rows = []
    for plan in [
        row for row in original_manifest["logical_assignments"] if row["stage"] == "validation"
    ]:
        generation = generation_index[(plan["checkpoint_role"], plan["prompt_id"])]
        source_chunks = scorer.safety_chunks(generation["text"])
        members = [
            row
            for row in merged
            if row["logical_assignment_id"] == plan["logical_assignment_id"]
        ]
        aggregate = scorer.aggregate_chunk_rows(
            members,
            expected_chunk_count=len(source_chunks),
            stratum=generation["stratum"],
            stop_reason=generation.get("stop_reason"),
        )
        assignment_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": "p5-pilot-20260808",
                "logical_assignment_id": plan["logical_assignment_id"],
                "stage": "validation_merged",
                "checkpoint_role": generation["checkpoint_role"],
                "prompt_id": generation["prompt_id"],
                "stratum": generation["stratum"],
                "generation_record_sha256": p5.sha256_json(generation),
                "finished_at_utc": utc_now(),
                **aggregate,
            }
        )
    executor.atomic_write(
        run_root / MERGED_ASSIGNMENT_OUTPUT,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in assignment_rows),
    )
    gate = scorer.evaluate_validation_gate(original_manifest, merged)
    gate["schema_version"] = "p5-pilot-safety-v2.2.1-merged-validation-gate-1"
    gate["merge_policy"] = (
        "original 104 chunk keys; non-empty recovery response replaces original error by key"
    )
    gate["original_gate_thresholds_unchanged"] = True
    gate["held_out_executed"] = False
    executor.atomic_write(run_root / MERGED_GATE_OUTPUT, json.dumps(gate, indent=2) + "\n")
    return gate


def execute_recovery(
    run_root: Path,
    manifest_path: Path,
    supplied_hash: str | None,
    authorised: bool,
) -> None:
    run_root = executor.assert_out_path(run_root)
    document = load_manifest(manifest_path)
    require_authorisation(document, supplied_hash, authorised)
    verify_original_artifacts(run_root)
    url = os.environ.get("CLAUDE_PROXY_URL")
    key = os.environ.get("CLAUDE_PROXY_KEY")
    if not url or not key:
        raise RecoveryError("CLAUDE_PROXY_URL and CLAUDE_PROXY_KEY are required")
    original_manifest_path = Path(
        document["sources"]["original_authorized_manifest"]["path"]
    )
    original_manifest = scorer.load_manifest(original_manifest_path)
    generations, snapshot = scorer.stable_generation_rows(run_root / "generations.jsonl")
    if snapshot["sha256"] != document["sources"]["generation_snapshot"]["sha256"]:
        raise RecoveryError("generation snapshot differs from recovery manifest")
    run_manifest = executor.load_manifest(run_root)
    generation_index = {
        (row["checkpoint_role"], row["prompt_id"]): row for row in generations
    }
    prompt_index = {row["prompt_id"]: row for row in run_manifest["prompts"]}
    recovery_path = run_root / RECOVERY_CHUNK_OUTPUT
    journal_path = run_root / RECOVERY_JOURNAL_OUTPUT
    persisted = jsonl(recovery_path)
    journal = jsonl(journal_path)
    done = {row["recovery_key"] for row in persisted}
    guard = document["authorization_guard"]

    def emit_event(event: dict[str, Any]) -> None:
        append_jsonl(journal_path, event)
        journal.append(event)

    def pre_request_guard() -> None:
        reservations = sum(row.get("event") == "reserved" for row in journal)
        if reservations >= guard["approved_request_ceiling"]:
            raise RecoveryGuardError("recovery attempt ceiling reached")
        next_commitment = round(
            (reservations + 1) * guard["approved_max_cost_per_request_usd"], 12
        )
        if next_commitment > guard["approved_spend_ceiling_usd"]:
            raise RecoveryGuardError("next recovery could exceed spend ceiling")
        quotas = [
            float(row["remaining_quota_usd"])
            for row in journal
            if row.get("event") == "completed"
            and isinstance(row.get("remaining_quota_usd"), (int, float))
        ]
        if quotas and quotas[-1] < guard["quota_stop_floor_usd"]:
            raise RecoveryGuardError("reported quota below stop floor")

    for plan in document["logical_recovery_chunks"]:
        if plan["recovery_key"] in done:
            continue
        generation = generation_index[(plan["checkpoint_role"], plan["prompt_id"])]
        chunks = scorer.safety_chunks(generation["text"])
        chunk = chunks[plan["chunk_index"]]
        if (
            scorer.chunk_plan_sha256(chunk) != plan["chunk_plan_sha256"]
            or p5.sha256_json(generation) != plan["generation_record_sha256"]
        ):
            raise RecoveryError("recovery source chunk differs from exact plan")
        started = utc_now()
        call_ids: list[str] = []
        fatal: Exception | None = None
        try:
            raw, _, call_ids = call_with_empty_retry(
                url=url,
                key=key,
                prompt=scorer.safety_prompt(
                    generation["stratum"],
                    prompt_index[generation["prompt_id"]]["prompt"],
                    chunk,
                    len(chunks),
                    generation.get("stop_reason"),
                ),
                descriptor=plan,
                emit_event=emit_event,
                pre_request_guard=pre_request_guard,
            )
            parsed = scorer.parse_safety_response(raw, chunk)
            row = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_manifest["run_id"],
                **plan,
                "started_at_utc": started,
                "finished_at_utc": utc_now(),
                "network_status": "success",
                "network_call_ids": call_ids,
                "response_sha256": scorer.sha256_bytes(raw.encode()),
                **parsed,
            }
        except Exception as error:
            if isinstance(error, RecoveryGuardError):
                fatal = error
            call_ids = list(
                dict.fromkeys(
                    row["call_id"]
                    for row in journal
                    if row.get("event") == "reserved"
                    and row.get("recovery_key") == plan["recovery_key"]
                )
            )
            # A guard raised before reservation is a stop, not a terminal model
            # observation.  Do not create a synthetic missing recovery row.
            if fatal is not None and not call_ids:
                raise fatal
            row = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_manifest["run_id"],
                **plan,
                "started_at_utc": started,
                "finished_at_utc": utc_now(),
                "network_status": "error",
                "network_call_ids": call_ids,
                "response_sha256": None,
                "parse_status": "unresolved",
                "parser_error_type": type(error).__name__,
                "field_status": {
                    field: "not_available"
                    for field in (
                        "refusal",
                        "substantive_compliance",
                        "safe_redirection",
                        "degenerate",
                        "evidence_unit_id",
                    )
                },
                "score": {
                    "refusal": None,
                    "substantive_compliance": None,
                    "safe_redirection": None,
                    "degenerate": None,
                },
                "evidence": None,
                "ignored_extra_keys": [],
            }
        append_jsonl(recovery_path, row)
        persisted.append(row)
        done.add(plan["recovery_key"])
        print(f"recovery {plan['recovery_key']} {row['network_status']} {row['parse_status']}", flush=True)
        if fatal is not None:
            raise fatal

    gate = write_merged_outputs(run_root, original_manifest, persisted)
    print(json.dumps(gate, indent=2), flush=True)


def markdown(document: dict[str, Any], file_sha256: str) -> str:
    accounting = document["offline_accounting"]
    diagnosis = document["diagnosis"]
    guard = document.get("authorization_guard")
    lines = [
        "# P5 safety v2.2.1 — empty-text validation recovery",
        "",
        f"**Status:** `{document.get('proposal_status', document['status'])}`  ",
        "**API/model/pod calls made:** `0`  ",
        f"**Internal SHA-256:** `{document['manifest_sha256']}`  ",
        f"**File SHA-256:** `{file_sha256}`",
        "",
        "## Diagnosis",
        "",
        f"All {diagnosis['failed_logical_chunks']} rows are the exact path where an HTTP non-error JSON payload produced empty/whitespace extracted scorer text. The original error occurred before accounting extraction, so payload shape, exact HTTP status, text, and per-response cost are not retrospectively recoverable. No absent text was recovered.",
        "",
        f"Failures are not demonstrably transient: {diagnosis['largest_prompt_cluster']['failed_chunks']}/14 share `{diagnosis['largest_prompt_cluster']['prompt_id']}`, and 12/14 are harmful. Recovery success is therefore not assumed.",
        "",
        "## Narrow amendment",
        "",
        "The identical Sonnet model, prompt, chunk, temperature, parser, and deterministic evidence checks are retained. Only empty extracted text joins timeout/HTTP 504 as retryable, with at most two total attempts per exact failed chunk. Parser or evidence failures are terminal and are not retried.",
        "",
        f"The 104-key denominator and original gates remain unchanged. At least {document['acceptance_and_stop_rules']['minimum_complete_recoveries_needed_for_98_percent_network_gate']} of 14 failed chunks must recover, and the deterministic merged view must still pass 100% atomic persistence, ≥98% network success, and ≥98% parser-plus-evidence completeness among successful responses. The runner never executes held-out rows.",
        "",
        "## Offline accounting",
        "",
        f"- Exact recovery chunks / initial requests: {accounting['logical_recovery_chunks']} / {accounting['initial_recovery_requests']}",
        f"- Maximum total attempts: {accounting['maximum_recovery_attempts']}",
        f"- Projected initial cost at observed envelope: ${accounting['projected_initial_cost_usd']:.6f}",
        f"- Projected maximum-attempt cost at observed envelope: ${accounting['projected_max_attempt_cost_at_observed_envelope_usd']:.6f}",
        "- These projections are not ceilings",
        "- Held-out requests and same-Sonnet repeats: 0",
        "",
    ]
    if guard:
        lines.extend(
            [
                "## Proposed hard limits",
                "",
                f"- Initial requests: {guard['initial_planned_requests']}",
                f"- Total-attempt ceiling: {guard['approved_request_ceiling']}",
                f"- Spend ceiling: ${guard['approved_spend_ceiling_usd']:.2f}",
                f"- Maximum cost/request: ${guard['approved_max_cost_per_request_usd']:.3f}",
                f"- Remaining-quota stop floor: ${guard['quota_stop_floor_usd']:.2f}",
                "- Held-out scoring and same-Sonnet repeats: unauthorized",
                "",
                "## Decision-ready authorization line",
                "",
                f"> I authorize only the P5 v2.2.1 empty-text validation recovery under manifest internal SHA `{document['manifest_sha256']}` and file SHA `{file_sha256}`: 14 exact failed chunks, 14 initial requests, at most 28 total attempts, $0.70 total spend, $0.025 maximum cost per request, $5.00 remaining-quota stop floor, unchanged parser/evidence and original validation gates, and no held-out scoring or same-Sonnet repeats. A merged-gate failure is a hard stop.",
                "",
                "Do not execute without that exact-hash owner authorization and `--authorised`.",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="write a zero-call exact recovery plan")
    plan.add_argument("--run-root", type=Path, required=True)
    plan.add_argument("--original-manifest", type=Path, required=True)
    plan.add_argument("--json-out", type=Path, required=True)
    plan.add_argument("--markdown-out", type=Path, required=True)
    propose = sub.add_parser("propose", help="write a new hash-bound recovery proposal")
    propose.add_argument("--plan", type=Path, required=True)
    propose.add_argument("--json-out", type=Path, required=True)
    propose.add_argument("--markdown-out", type=Path, required=True)
    execute = sub.add_parser("execute", help="execute only after new exact-hash authorization")
    execute.add_argument("--run-root", type=Path, required=True)
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--manifest-sha256")
    execute.add_argument("--authorised", action="store_true")
    args = parser.parse_args()

    if args.command == "plan":
        document = build_dry_plan(args.run_root, args.original_manifest)
        executor.atomic_write(args.json_out, json.dumps(document, indent=2) + "\n")
        file_sha = p5.sha256_file(args.json_out)
        executor.atomic_write(args.markdown_out, markdown(document, file_sha))
    elif args.command == "propose":
        document = build_proposal(args.plan)
        executor.atomic_write(args.json_out, json.dumps(document, indent=2) + "\n")
        file_sha = p5.sha256_file(args.json_out)
        executor.atomic_write(args.markdown_out, markdown(document, file_sha))
    else:
        execute_recovery(
            args.run_root, args.manifest, args.manifest_sha256, args.authorised
        )
        return
    print(
        json.dumps(
            {
                "internal_sha256": document["manifest_sha256"],
                "file_sha256": file_sha,
                "calls": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
