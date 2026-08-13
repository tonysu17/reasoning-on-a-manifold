#!/usr/bin/env python3
"""Hash-bound, one-call response-shape diagnostic for one frozen v2.2 failure.

Plan/proposal creation is offline.  Execution requires the exact internal and
file hashes plus ``--authorised``.  The sole prospective response is inspected
for shape before scorer-text extraction; raw prompt, response, URL, keys, and
header values are never persisted.  This artifact never writes annotations,
validation gates, or recovery outputs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import requests


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p5_preflight as p5  # noqa: E402
import p5_pilot_executor as executor  # noqa: E402
import p5_pilot_safety_scorer_v2_2 as scorer  # noqa: E402


SCHEMA_VERSION = "p5-pilot-safety-v2.2-one-call-shape-diagnostic-1"
MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"
REQUEST_TIMEOUT_SECONDS = 25
MAX_OUTPUT_TOKENS = 400
MAX_ATTEMPTS = 1
MAX_RETRIES = 0
MAX_COST_PER_REQUEST_USD = 0.025
SPEND_CEILING_USD = 0.025
QUOTA_STOP_FLOOR_USD = 5.0
TARGET_LOGICAL_ASSIGNMENT_ID = "v2.2:base_r1:p5sp_weapons_h01"
TARGET_CHECKPOINT_ROLE = "base_r1"
TARGET_PROMPT_ID = "p5sp_weapons_h01"
TARGET_CHUNK_INDEX = 0
TARGET_ORIGINAL_CALL_ID = "12c12c00-5440-405a-8d90-87fd91bfd848"
EXPECTED_GENERATION_SNAPSHOT_SHA256 = (
    "2f2e9bc4d71c0e8f7ef71fe067d0e3f12b6fb2af31e83ee692884d70f6bbaf01"
)
EXPECTED_GENERATION_RECORD_SHA256 = (
    "8aa4e3a162e3ca5ded5f621a831460bd7f90410044c14b8c3b250b470a04ef30"
)
EXPECTED_CHUNK_PLAN_SHA256 = (
    "61f12de70109d291df79d844584cfb8d431ec04c278eb6aa5d09b11cdf900d77"
)
EXPECTED_ORIGINAL_MANIFEST_INTERNAL_SHA256 = (
    "29de70c816a2fd6fbf6fcf786cee2543935332870244e24756de1a93be9c669c"
)
EXPECTED_ORIGINAL_MANIFEST_FILE_SHA256 = (
    "ee801c32f0405d2819ae1aafacc5d78ed3008e8f43572ef86553b7f3f386e9c3"
)
EXPECTED_RUN_MANIFEST_SHA256 = (
    "ec8fbead64dfc0362d9c3646bdcaa9593e8beadb5d8ce38033c6f0d3e983c567"
)
EXPECTED_SCORER_SHA256 = (
    "e5fd7a894442d73a75c90ba45c6b0d4ed79bb4dd1f84993157c50087770a8914"
)
EXPECTED_ORIGINAL_ARTIFACT_HASHES = {
    scorer.CHUNK_OUTPUT: "b43cfa9b56b1f2f8d581bd72fc54ad64625e3f24af41b3da2a6ec1399dcac480",
    scorer.ASSIGNMENT_OUTPUT: "7130d0920c0ed42b019380e494a9c8b0970b8ff22d3befe9704f8377d1089995",
    scorer.JOURNAL_OUTPUT: "ddcf61b65da220b783cde135dfdec67497cdca474b8479e0cb85c3d3bb1dc7ce",
    scorer.VALIDATION_GATE_OUTPUT: "5a80f74ce855b49d992a6fe2254548a126788e5aac62ac5c3f34b065a45cc313",
}
ORIGINAL_MANIFEST_NAME = "P5_PILOT_SAFETY_V2_2_PROPOSED_VALIDATION_RESCORING_2026-08-09.json"
TEST_NAME = "test_p5_pilot_safety_v2_2_one_call_diagnostic.py"
DIAGNOSTIC_OUTPUT = "safety_v2_2_one_call_shape_diagnostic.json"
DIAGNOSTIC_JOURNAL = "safety_v2_2_one_call_shape_diagnostic_journal.jsonl"

SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
SAFE_MEDIA_TYPE = re.compile(r"^[a-z0-9.+-]+/[a-z0-9.+-]+$")
PROVIDER_REQUEST_ID_HEADERS = (
    "anthropic-request-id",
    "request-id",
    "x-request-id",
    "x-amzn-requestid",
)
FILTER_VALUE_MARKERS = (
    "block",
    "content_filter",
    "filter",
    "guardrail",
    "moderation",
    "safety",
)
USAGE_NUMERIC_ALLOWLIST = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "total_tokens",
    "cost",
)


class DiagnosticError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def manifest_hash(document: dict[str, Any]) -> str:
    content = dict(document)
    content.pop("manifest_sha256", None)
    return p5.sha256_json(content)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _safe_name(value: Any) -> str:
    if isinstance(value, str) and SAFE_NAME.fullmatch(value):
        return value
    encoded = str(value).encode("utf-8", errors="replace")
    return f"unsafe_name_sha256:{sha256_bytes(encoded)}"


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, (int, float)):
        return "number"
    return "other"


def _normalized_content_type(headers: Mapping[str, Any]) -> str | None:
    raw = headers.get("Content-Type") or headers.get("content-type")
    if not isinstance(raw, str):
        return None
    media_type = raw.split(";", 1)[0].strip().lower()
    return media_type if SAFE_MEDIA_TYPE.fullmatch(media_type) else "unrecognized"


def _provider_request_id_hash(headers: Mapping[str, Any]) -> tuple[str | None, str | None]:
    lowered = {str(key).lower(): value for key, value in headers.items()}
    for name in PROVIDER_REQUEST_ID_HEADERS:
        value = lowered.get(name)
        if isinstance(value, str) and value:
            return name, sha256_bytes(value.encode("utf-8"))
    return None, None


def _walk_signal_presence(value: Any) -> dict[str, bool]:
    presence = {
        "stop_field_present": False,
        "filter_field_present": False,
        "guardrail_field_present": False,
        "safety_field_present": False,
        "filter_marker_present": False,
    }

    def visit(item: Any, contextual_key: str | None = None) -> None:
        if isinstance(item, dict):
            for raw_key, child in item.items():
                key = str(raw_key).lower()
                if "stop" in key or "finish_reason" in key:
                    presence["stop_field_present"] = True
                if "filter" in key or "moderation" in key:
                    presence["filter_field_present"] = True
                    presence["filter_marker_present"] = True
                if "guardrail" in key:
                    presence["guardrail_field_present"] = True
                    presence["filter_marker_present"] = True
                if "safety" in key:
                    presence["safety_field_present"] = True
                    presence["filter_marker_present"] = True
                visit(child, key)
        elif isinstance(item, list):
            for child in item:
                visit(child, contextual_key)
        elif isinstance(item, str) and contextual_key is not None:
            if any(token in contextual_key for token in ("stop", "finish", "reason", "filter", "guardrail", "safety", "moderation")):
                normalized = item.strip().lower().replace("-", "_").replace(" ", "_")
                if any(marker in normalized for marker in FILTER_VALUE_MARKERS):
                    presence["filter_marker_present"] = True

    visit(value)
    return presence


def _text_blocks(content: Any) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    block_types: list[str] = []
    if isinstance(content, str):
        texts.append(content)
        block_types.append("string")
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                block_types.append("string")
                texts.append(block)
            elif isinstance(block, dict):
                block_types.append(_safe_name(block.get("type", "object_without_type")))
                if isinstance(block.get("text"), str):
                    texts.append(block["text"])
            else:
                block_types.append(_json_type(block))
    elif isinstance(content, dict):
        block_types.append(_safe_name(content.get("type", "object_without_type")))
        if isinstance(content.get("text"), str):
            texts.append(content["text"])
    elif content is not None:
        block_types.append(_json_type(content))
    return texts, block_types


def _numeric_usage(payload: Any) -> dict[str, float | int]:
    if not isinstance(payload, dict) or not isinstance(payload.get("usage"), dict):
        return {}
    result: dict[str, float | int] = {}
    for key in USAGE_NUMERIC_ALLOWLIST:
        value = payload["usage"].get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)) and float(value) >= 0:
            result[key] = value
    return result


def response_shape_telemetry(response: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Inspect a response without returning response text in persisted telemetry."""
    raw = bytes(response.content)
    headers = response.headers if isinstance(response.headers, Mapping) else {}
    request_header, request_id_hash = _provider_request_id_hash(headers)
    status_code = int(response.status_code)
    telemetry: dict[str, Any] = {
        "http_status_class": f"{status_code // 100}xx" if 100 <= status_code <= 599 else "invalid",
        "response_content_type": _normalized_content_type(headers),
        "raw_payload_sha256": sha256_bytes(raw),
        "raw_payload_bytes": len(raw),
        "provider_request_id_header_name": request_header,
        "provider_request_id_sha256": request_id_hash,
        "json_parse_status": "invalid_json",
        "top_level_json_type": None,
        "top_level_json_keys": [],
        "content_container_type": "absent",
        "content_container_count": 0,
        "block_type_names": [],
        "text_block_count": 0,
        "text_block_byte_lengths": [],
        "text_block_whitespace_only": [],
        "signal_field_presence": {
            "stop_field_present": False,
            "filter_field_present": False,
            "guardrail_field_present": False,
            "safety_field_present": False,
            "filter_marker_present": False,
        },
        "usage_numeric": {},
        "usage_cost_usd": None,
        "remaining_quota_usd": None,
        "accounting_parseable": False,
    }
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        return telemetry, None

    telemetry["json_parse_status"] = "parsed"
    telemetry["top_level_json_type"] = _json_type(parsed)
    if not isinstance(parsed, dict):
        return telemetry, None
    telemetry["top_level_json_keys"] = sorted(_safe_name(key) for key in parsed)
    content = parsed.get("content")
    telemetry["content_container_type"] = _json_type(content) if "content" in parsed else "absent"
    if isinstance(content, list):
        telemetry["content_container_count"] = len(content)
    elif content is not None:
        telemetry["content_container_count"] = 1
    texts, block_types = _text_blocks(content)
    telemetry["block_type_names"] = block_types
    telemetry["text_block_count"] = len(texts)
    telemetry["text_block_byte_lengths"] = [len(text.encode("utf-8")) for text in texts]
    telemetry["text_block_whitespace_only"] = [not text.strip() for text in texts]
    telemetry["signal_field_presence"] = _walk_signal_presence(parsed)
    telemetry["usage_numeric"] = _numeric_usage(parsed)
    try:
        cost, quota = scorer.proxy_accounting(parsed)
    except scorer.AccountingError:
        pass
    else:
        telemetry["usage_cost_usd"] = cost
        telemetry["remaining_quota_usd"] = quota
        telemetry["accounting_parseable"] = True
    return telemetry, parsed


def classify_shape(telemetry: dict[str, Any], extracted_text: str | None) -> str:
    if telemetry["http_status_class"] != "2xx":
        return "inconclusive_http_error"
    if telemetry["json_parse_status"] != "parsed" or extracted_text is None:
        return "inconclusive_invalid_or_nonobject_json"
    if extracted_text.strip():
        return "nonempty_scorer_text"
    if telemetry["signal_field_presence"]["filter_marker_present"]:
        return "empty_scorer_text_with_filter_marker"
    return "empty_scorer_text_without_filter_marker"


def _request_body(prompt: str) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
    }


def target_binding(run_root: Path) -> tuple[dict[str, Any], str]:
    run_root = executor.assert_out_path(run_root)
    original_manifest_path = HERE / ORIGINAL_MANIFEST_NAME
    if p5.sha256_file(original_manifest_path) != EXPECTED_ORIGINAL_MANIFEST_FILE_SHA256:
        raise DiagnosticError("original v2.2 manifest file hash mismatch")
    original_manifest = scorer.load_manifest(original_manifest_path)
    if original_manifest["manifest_sha256"] != EXPECTED_ORIGINAL_MANIFEST_INTERNAL_SHA256:
        raise DiagnosticError("original v2.2 manifest internal hash mismatch")
    if MODEL != scorer.MODEL or p5.sha256_file(HERE / "p5_pilot_safety_scorer_v2_2.py") != EXPECTED_SCORER_SHA256:
        raise DiagnosticError("frozen scorer/model binding mismatch")

    generation_path = run_root / "generations.jsonl"
    rows, snapshot = scorer.stable_generation_rows(generation_path)
    if snapshot["sha256"] != EXPECTED_GENERATION_SNAPSHOT_SHA256:
        raise DiagnosticError("generation snapshot hash mismatch")
    generation = next(
        row for row in rows
        if row["checkpoint_role"] == TARGET_CHECKPOINT_ROLE and row["prompt_id"] == TARGET_PROMPT_ID
    )
    generation_hash = scorer.sha256_json(generation)
    if generation_hash != EXPECTED_GENERATION_RECORD_SHA256:
        raise DiagnosticError("target generation-record hash mismatch")
    chunks = scorer.safety_chunks(generation["text"])
    chunk = chunks[TARGET_CHUNK_INDEX]
    chunk_hash = scorer.chunk_plan_sha256(chunk)
    if chunk_hash != EXPECTED_CHUNK_PLAN_SHA256:
        raise DiagnosticError("target chunk-plan hash mismatch")

    run_manifest_path = run_root / "run_manifest.json"
    if p5.sha256_file(run_manifest_path) != EXPECTED_RUN_MANIFEST_SHA256:
        raise DiagnosticError("run manifest hash mismatch")
    run_manifest = executor.load_manifest(run_root)
    prompt_row = next(row for row in run_manifest["prompts"] if row["prompt_id"] == TARGET_PROMPT_ID)
    user_prompt_hash = sha256_bytes(prompt_row["prompt"].encode("utf-8"))
    if user_prompt_hash != generation["prompt_text_sha256"]:
        raise DiagnosticError("source user-prompt hash mismatch")
    prompt = scorer.safety_prompt(
        generation["stratum"],
        prompt_row["prompt"],
        chunk,
        len(chunks),
        generation.get("stop_reason"),
    )

    planned = next(
        row for row in original_manifest["initial_network_request_plan"]
        if row["logical_assignment_id"] == TARGET_LOGICAL_ASSIGNMENT_ID
        and row["chunk_index"] == TARGET_CHUNK_INDEX
    )
    if planned["chunk_plan_sha256"] != chunk_hash or planned["stage"] != "validation":
        raise DiagnosticError("original request-plan binding mismatch")
    original_rows = _jsonl(run_root / scorer.CHUNK_OUTPUT)
    original = next(
        row for row in original_rows
        if row["logical_assignment_id"] == TARGET_LOGICAL_ASSIGNMENT_ID
        and row["chunk_index"] == TARGET_CHUNK_INDEX
    )
    if (
        original.get("network_status") != "error"
        or original.get("parser_error_type") != "ProxyProtocolError"
        or original.get("network_call_ids") != [TARGET_ORIGINAL_CALL_ID]
        or original.get("generation_record_sha256") != generation_hash
    ):
        raise DiagnosticError("target is not the exact retained v2.2 empty-text failure")

    binding = {
        "logical_assignment_id": TARGET_LOGICAL_ASSIGNMENT_ID,
        "stage": "prospective_shape_diagnostic_only",
        "checkpoint_role": TARGET_CHECKPOINT_ROLE,
        "prompt_id": TARGET_PROMPT_ID,
        "stratum": generation["stratum"],
        "chunk_index": TARGET_CHUNK_INDEX,
        "n_chunks": len(chunks),
        "char_start": chunk["char_start"],
        "char_end": chunk["char_end"],
        "n_chars": chunk["n_chars"],
        "n_evidence_units": chunk["n_units"],
        "generation_record_sha256": generation_hash,
        "source_user_prompt_sha256": user_prompt_hash,
        "chunk_plan_sha256": chunk_hash,
        "diagnostic_prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "serialized_request_sha256": p5.sha256_json(_request_body(prompt)),
        "original_terminal_row_sha256": p5.sha256_json(original),
        "original_failed_call_id": TARGET_ORIGINAL_CALL_ID,
        "original_failure_type": "ProxyProtocolError",
    }
    return binding, prompt


def verify_original_artifacts(run_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, expected in EXPECTED_ORIGINAL_ARTIFACT_HASHES.items():
        path = executor.assert_out_path(run_root) / name
        actual = p5.sha256_file(path)
        if actual != expected:
            raise DiagnosticError(f"original v2.2 artifact hash mismatch: {name}")
        result[name] = {"path": str(path.resolve()), "sha256": actual}
    return result


def build_plan(run_root: Path) -> dict[str, Any]:
    original = verify_original_artifacts(run_root)
    target, _ = target_binding(run_root)
    document: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}-plan",
        "status": "dry_run_non_executable",
        "created_at_utc": utc_now(),
        "proxy_or_model_calls_made": 0,
        "purpose": "prospective response-shape diagnosis only",
        "scientific_status": "diagnostic output excluded from all annotations and gates",
        "sources": {
            "generation_snapshot": {
                "path": str((run_root / "generations.jsonl").resolve()),
                "sha256": EXPECTED_GENERATION_SNAPSHOT_SHA256,
            },
            "run_manifest": {
                "path": str((run_root / "run_manifest.json").resolve()),
                "sha256": EXPECTED_RUN_MANIFEST_SHA256,
            },
            "original_v2_2_manifest": {
                "path": str((HERE / ORIGINAL_MANIFEST_NAME).resolve()),
                "internal_sha256": EXPECTED_ORIGINAL_MANIFEST_INTERNAL_SHA256,
                "file_sha256": EXPECTED_ORIGINAL_MANIFEST_FILE_SHA256,
            },
            "original_v2_2_artifacts": original,
            "frozen_v2_2_scorer": {
                "path": str((HERE / "p5_pilot_safety_scorer_v2_2.py").resolve()),
                "sha256": EXPECTED_SCORER_SHA256,
            },
            "diagnostic_runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": p5.sha256_file(Path(__file__).resolve()),
            },
            "diagnostic_tests": {
                "path": str((HERE / TEST_NAME).resolve()),
                "sha256": p5.sha256_file(HERE / TEST_NAME),
            },
        },
        "target": target,
        "request_contract": {
            "model": MODEL,
            "temperature": 0.0,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "initial_attempts": MAX_ATTEMPTS,
            "maximum_total_attempts": MAX_ATTEMPTS,
            "automatic_retries": MAX_RETRIES,
            "sequential": True,
            "maximum_cost_per_request_usd": MAX_COST_PER_REQUEST_USD,
            "total_spend_ceiling_usd": SPEND_CEILING_USD,
            "quota_stop_floor_usd": QUOTA_STOP_FLOOR_USD,
        },
        "telemetry_contract": {
            "captured_before_text_extraction": True,
            "captured": [
                "HTTP status class",
                "normalized response content type",
                "top-level JSON key names only",
                "content container type/count",
                "block type names",
                "text-block count/byte lengths/whitespace flags",
                "stop/filter/guardrail/safety field presence only",
                "parseable numeric usage/cost/quota",
                "in-memory raw payload SHA-256 and byte length",
                "hashed provider request ID when available",
            ],
            "never_persisted": [
                "raw response payload",
                "source or scorer prompt text",
                "extracted scorer text",
                "proxy URL",
                "API key",
                "raw header values or raw provider request ID",
            ],
        },
        "output_exclusions": {
            "annotations": True,
            "validation_gate": True,
            "recovery_outputs": True,
            "v2_2_artifacts_mutated": False,
            "v2_2_1_recovery_artifacts_mutated": False,
        },
        "decision_tree": {
            "empty_scorer_text_with_filter_marker": (
                "Do not retry or score; seek operator confirmation of the filter pathway. "
                "Original missingness and 98% validation gate remain unchanged."
            ),
            "empty_scorer_text_without_filter_marker": (
                "Do not retry automatically; treat as response-shape/proxy anomaly and let the "
                "owner decide whether a separately hash-authorized bounded recovery is warranted."
            ),
            "nonempty_scorer_text": (
                "Do not parse or annotate; intermittency becomes plausible but the original failure "
                "is not retrospectively reclassified. A recovery plan still needs separate approval."
            ),
            "inconclusive_http_transport_or_json_error": (
                "Stop after the one attempt; no retry. Request operator metadata before any recovery."
            ),
        },
        "prospective_output_paths": {
            "diagnostic": str((run_root / DIAGNOSTIC_OUTPUT).resolve()),
            "journal": str((run_root / DIAGNOSTIC_JOURNAL).resolve()),
        },
    }
    document["manifest_sha256"] = manifest_hash(document)
    return document


def build_proposal(plan_path: Path) -> dict[str, Any]:
    plan_path = executor.assert_out_path(plan_path)
    plan = json.loads(plan_path.read_text())
    if plan.get("manifest_sha256") != manifest_hash(plan):
        raise DiagnosticError("invalid diagnostic plan hash")
    if plan.get("status") != "dry_run_non_executable":
        raise DiagnosticError("diagnostic source plan must be non-executable")
    document = copy.deepcopy(plan)
    document.pop("manifest_sha256", None)
    document["schema_version"] = f"{SCHEMA_VERSION}-proposed-executable"
    document["status"] = "authorized_for_execution"
    document["proposal_status"] = "proposed_awaiting_owner_exact_hash_approval"
    document["created_at_utc"] = utc_now()
    document["sources"]["diagnostic_plan"] = {
        "path": str(plan_path.resolve()),
        "internal_sha256": plan["manifest_sha256"],
        "file_sha256": p5.sha256_file(plan_path),
    }
    document["authorization_guard"] = {
        "execution_capability_only": True,
        "owner_exact_internal_and_file_hash_approval_required": True,
        "owner_approval_received": False,
        "authorised_flag_required": True,
        "approved_request_ceiling": 1,
        "approved_retry_ceiling": 0,
        "approved_spend_ceiling_usd": SPEND_CEILING_USD,
        "approved_max_cost_per_request_usd": MAX_COST_PER_REQUEST_USD,
        "quota_stop_floor_usd": QUOTA_STOP_FLOOR_USD,
        "ceiling_product_usd": MAX_ATTEMPTS * MAX_COST_PER_REQUEST_USD,
        "ceiling_product_within_spend": (
            MAX_ATTEMPTS * MAX_COST_PER_REQUEST_USD <= SPEND_CEILING_USD
        ),
        "same_sonnet_repeats_authorized": False,
        "annotation_or_recovery_execution_authorized": False,
    }
    document["authorization_text_template"] = (
        "I authorize exactly one P5 v2.2 response-shape diagnostic call under the manifest "
        "with internal SHA <INTERNAL_SHA256> and file SHA <FILE_SHA256>: Sonnet model as bound, "
        "max 400 output tokens, 25-second timeout, no retry, $0.025 total/per-call ceiling, $5 "
        "quota floor, and output excluded from annotations, gates, and recovery."
    )
    document["manifest_sha256"] = manifest_hash(document)
    return document


def load_manifest(path: Path) -> dict[str, Any]:
    path = executor.assert_out_path(path)
    document = json.loads(path.read_text())
    if document.get("manifest_sha256") != manifest_hash(document):
        raise DiagnosticError("invalid diagnostic manifest internal hash")
    sources = document.get("sources", {})
    if sources.get("diagnostic_runner", {}).get("sha256") != p5.sha256_file(Path(__file__).resolve()):
        raise DiagnosticError("diagnostic runner hash mismatch")
    if sources.get("diagnostic_tests", {}).get("sha256") != p5.sha256_file(HERE / TEST_NAME):
        raise DiagnosticError("diagnostic test hash mismatch")
    if sources.get("frozen_v2_2_scorer", {}).get("sha256") != EXPECTED_SCORER_SHA256:
        raise DiagnosticError("frozen scorer hash binding mismatch")
    return document


def require_authorisation(
    document: dict[str, Any],
    manifest_path: Path,
    supplied_internal_sha256: str | None,
    supplied_file_sha256: str | None,
    authorised: bool,
) -> None:
    guard = document.get("authorization_guard", {})
    if document.get("status") != "authorized_for_execution" or document.get("proposal_status") != "proposed_awaiting_owner_exact_hash_approval":
        raise DiagnosticError("diagnostic manifest is dry-run/non-executable")
    if not authorised:
        raise DiagnosticError("--authorised is required")
    if supplied_internal_sha256 != document.get("manifest_sha256"):
        raise DiagnosticError("exact diagnostic internal hash is required")
    if supplied_file_sha256 != p5.sha256_file(executor.assert_out_path(manifest_path)):
        raise DiagnosticError("exact diagnostic file hash is required")
    if (
        guard.get("approved_request_ceiling") != 1
        or guard.get("approved_retry_ceiling") != 0
        or guard.get("approved_spend_ceiling_usd") != SPEND_CEILING_USD
        or guard.get("approved_max_cost_per_request_usd") != MAX_COST_PER_REQUEST_USD
        or guard.get("quota_stop_floor_usd") != QUOTA_STOP_FLOOR_USD
        or guard.get("same_sonnet_repeats_authorized") is not False
        or guard.get("annotation_or_recovery_execution_authorized") is not False
        or MAX_ATTEMPTS * MAX_COST_PER_REQUEST_USD > SPEND_CEILING_USD
    ):
        raise DiagnosticError("diagnostic authorization guards differ from frozen constants")


def perform_one_call(
    *,
    url: str,
    key: str,
    prompt: str,
    target: dict[str, Any],
    emit_event: Callable[[dict[str, Any]], None],
    post: Callable[..., Any] = requests.post,
) -> dict[str, Any]:
    """Make at most one call. Tests inject ``post``; production has no retry path."""
    call_id = str(uuid.uuid4())
    common = {
        "diagnostic_call_id": call_id,
        "model": MODEL,
        "logical_assignment_id": target["logical_assignment_id"],
        "chunk_index": target["chunk_index"],
        "generation_record_sha256": target["generation_record_sha256"],
        "chunk_plan_sha256": target["chunk_plan_sha256"],
        "diagnostic_prompt_sha256": target["diagnostic_prompt_sha256"],
        "request_attempt": 1,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
    }
    emit_event({"event": "reserved", "at_utc": utc_now(), **common})
    started = time.monotonic()
    try:
        response = post(
            url,
            json=_request_body(prompt),
            headers={"X-Api-Key": key, "Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        telemetry, payload = response_shape_telemetry(response)
        extracted_text = scorer.extract_proxy_text(payload) if isinstance(payload, dict) else None
        decision = classify_shape(telemetry, extracted_text)
        guard_trips: list[str] = []
        cost = telemetry["usage_cost_usd"]
        quota = telemetry["remaining_quota_usd"]
        if isinstance(cost, (int, float)) and cost > MAX_COST_PER_REQUEST_USD:
            guard_trips.append("per_call_cost_ceiling_exceeded")
        if isinstance(quota, (int, float)) and quota < QUOTA_STOP_FLOOR_USD:
            guard_trips.append("quota_floor_crossed")
        result = {
            "schema_version": SCHEMA_VERSION,
            "event": "completed",
            "at_utc": utc_now(),
            **common,
            "network_attempts": 1,
            "automatic_retries": 0,
            "outcome": decision,
            "hard_guard_trips": guard_trips,
            "telemetry": telemetry,
            "wall_seconds": time.monotonic() - started,
            "output_excluded_from_annotations": True,
            "output_excluded_from_validation_gate": True,
            "output_excluded_from_recovery": True,
            "scorer_text_extracted_in_memory_only": extracted_text is not None,
            "scorer_text_persisted": False,
        }
    except requests.Timeout:
        result = {
            "schema_version": SCHEMA_VERSION,
            "event": "completed",
            "at_utc": utc_now(),
            **common,
            "network_attempts": 1,
            "automatic_retries": 0,
            "outcome": "inconclusive_transport_timeout",
            "error_type": "Timeout",
            "telemetry": None,
            "hard_guard_trips": [],
            "wall_seconds": time.monotonic() - started,
            "output_excluded_from_annotations": True,
            "output_excluded_from_validation_gate": True,
            "output_excluded_from_recovery": True,
            "scorer_text_persisted": False,
        }
    except requests.RequestException:
        result = {
            "schema_version": SCHEMA_VERSION,
            "event": "completed",
            "at_utc": utc_now(),
            **common,
            "network_attempts": 1,
            "automatic_retries": 0,
            "outcome": "inconclusive_transport_error",
            "error_type": "RequestException",
            "telemetry": None,
            "hard_guard_trips": [],
            "wall_seconds": time.monotonic() - started,
            "output_excluded_from_annotations": True,
            "output_excluded_from_validation_gate": True,
            "output_excluded_from_recovery": True,
            "scorer_text_persisted": False,
        }
    emit_event(result)
    return result


def execute(
    run_root: Path,
    manifest_path: Path,
    supplied_internal_sha256: str | None,
    supplied_file_sha256: str | None,
    authorised: bool,
) -> dict[str, Any]:
    run_root = executor.assert_out_path(run_root)
    manifest_path = executor.assert_out_path(manifest_path)
    document = load_manifest(manifest_path)
    require_authorisation(
        document,
        manifest_path,
        supplied_internal_sha256,
        supplied_file_sha256,
        authorised,
    )
    verify_original_artifacts(run_root)
    target, prompt = target_binding(run_root)
    if target != document.get("target"):
        raise DiagnosticError("live target/prompt binding differs from authorized manifest")
    if p5.sha256_json(_request_body(prompt)) != target["serialized_request_sha256"]:
        raise DiagnosticError("serialized diagnostic request differs from authorized binding")

    output_path = run_root / DIAGNOSTIC_OUTPUT
    journal_path = run_root / DIAGNOSTIC_JOURNAL
    if output_path.exists() or journal_path.exists():
        raise DiagnosticError("diagnostic output or reservation already exists; refusing any rerun")
    url = os.environ.get("CLAUDE_PROXY_URL")
    key = os.environ.get("CLAUDE_PROXY_KEY")
    if not url or not key:
        raise DiagnosticError("configured proxy environment is required")

    events: list[dict[str, Any]] = []

    def emit_event(event: dict[str, Any]) -> None:
        safe_event = {
            **event,
            "manifest_internal_sha256": document["manifest_sha256"],
            "manifest_file_sha256": p5.sha256_file(manifest_path),
        }
        executor.append_jsonl(journal_path, safe_event)
        events.append(safe_event)

    result = perform_one_call(
        url=url,
        key=key,
        prompt=prompt,
        target=target,
        emit_event=emit_event,
    )
    final = {
        **result,
        "manifest_internal_sha256": document["manifest_sha256"],
        "manifest_file_sha256": p5.sha256_file(manifest_path),
        "journal_events": len(events),
        "source_target": target,
    }
    executor.atomic_write(output_path, json.dumps(final, indent=2, sort_keys=True) + "\n")
    return final


def write_json(path: Path, document: dict[str, Any]) -> str:
    path = executor.assert_out_path(path)
    executor.atomic_write(path, json.dumps(document, indent=2, sort_keys=True) + "\n")
    return p5.sha256_file(path)


def proposal_markdown(document: dict[str, Any], file_sha256: str) -> str:
    internal = document["manifest_sha256"]
    approval = (
        "I authorize exactly one P5 v2.2 response-shape diagnostic call under manifest internal "
        f"SHA `{internal}` and file SHA `{file_sha256}`: Sonnet model as bound, max 400 output "
        "tokens, 25-second timeout, no retry, $0.025 total/per-call ceiling, $5 quota floor, and "
        "output excluded from annotations, gates, and recovery."
    )
    return f"""# Proposed P5 v2.2 one-call response-shape diagnostic

Status: **awaiting owner approval; no call made**.

- Internal manifest SHA-256: `{internal}`
- File SHA-256: `{file_sha256}`
- Target: `{TARGET_CHECKPOINT_ROLE} / {TARGET_PROMPT_ID} / chunk {TARGET_CHUNK_INDEX}`
- Original failed call: `{TARGET_ORIGINAL_CALL_ID}`
- Requests: exactly 1; retries: 0
- Model: `{MODEL}`
- Output limit / timeout: {MAX_OUTPUT_TOKENS} tokens / {REQUEST_TIMEOUT_SECONDS}s
- Total and per-call ceiling: ${SPEND_CEILING_USD:.3f}
- Quota stop floor: ${QUOTA_STOP_FLOOR_USD:.2f}
- Diagnostic output is excluded from annotations, validation gates, and recovery.

## Decision tree

- Empty with a filter marker: do not retry or score; request operator confirmation.
- Empty without a filter marker: do not retry automatically; owner decides whether to authorize a separate bounded recovery.
- Nonempty: do not parse or annotate; this suggests possible intermittency but does not reclassify the original failure.
- HTTP, transport, or JSON error: stop after the single attempt and seek operator metadata.

## Exact authorization line

{approval}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--run-root", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    proposal_parser = sub.add_parser("propose")
    proposal_parser.add_argument("--plan", type=Path, required=True)
    proposal_parser.add_argument("--output", type=Path, required=True)
    proposal_parser.add_argument("--report", type=Path, required=True)
    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument("--run-root", type=Path, required=True)
    execute_parser.add_argument("--manifest", type=Path, required=True)
    execute_parser.add_argument("--manifest-sha256")
    execute_parser.add_argument("--manifest-file-sha256")
    execute_parser.add_argument("--authorised", action="store_true")
    args = parser.parse_args()

    if args.command == "plan":
        document = build_plan(args.run_root)
        file_sha = write_json(args.output, document)
        print(json.dumps({"status": document["status"], "internal_sha256": document["manifest_sha256"], "file_sha256": file_sha}))
    elif args.command == "propose":
        document = build_proposal(args.plan)
        file_sha = write_json(args.output, document)
        executor.atomic_write(
            executor.assert_out_path(args.report),
            proposal_markdown(document, file_sha),
        )
        print(json.dumps({"status": document["proposal_status"], "internal_sha256": document["manifest_sha256"], "file_sha256": file_sha}))
    elif args.command == "execute":
        result = execute(
            args.run_root,
            args.manifest,
            args.manifest_sha256,
            args.manifest_file_sha256,
            args.authorised,
        )
        print(json.dumps({"outcome": result["outcome"], "network_attempts": result["network_attempts"]}))


if __name__ == "__main__":
    main()
