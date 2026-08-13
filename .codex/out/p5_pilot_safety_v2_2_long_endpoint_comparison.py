#!/usr/bin/env python3
"""Directly authorized one-call long-endpoint comparison for P5 v2.2.

The request body is byte/content-bound to the completed standard-endpoint
diagnostic.  The only request-path change is the approved long base URL; the
25-second client timeout remains unchanged and is below the endpoint's 120-
second service limit.  No response, prompt, URL, key, or raw header value is
persisted.  Outputs are diagnostic-only and cannot enter annotations, gates,
or recovery.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p5_preflight as p5  # noqa: E402
import p5_pilot_executor as executor  # noqa: E402
import p5_pilot_safety_v2_2_one_call_diagnostic as standard  # noqa: E402


SCHEMA_VERSION = "p5-pilot-safety-v2.2-long-endpoint-comparison-1"
MODEL = standard.MODEL
MAX_OUTPUT_TOKENS = standard.MAX_OUTPUT_TOKENS
CLIENT_TIMEOUT_SECONDS = standard.REQUEST_TIMEOUT_SECONDS
LONG_ENDPOINT_SERVICE_LIMIT_SECONDS = 120
MAX_ATTEMPTS = 1
MAX_RETRIES = 0
MAX_COST_PER_REQUEST_USD = 0.025
SPEND_CEILING_USD = 0.025
QUOTA_STOP_FLOOR_USD = 5.0
OWNER_AUTHORIZATION_QUOTE = (
    "One official long-endpoint comparison. Send the identical diagnostic once through "
    "the 120-second endpoint supplied. Expected cost about $0.006; hard ceiling $0.025; "
    "diagnostic-only."
)
LONG_ENDPOINT_HANDOFF_NAME = "CLAUDE_LONG_REQUEST_ENDPOINT_HANDOFF_2026-08-09.md"
EXPECTED_LONG_ENDPOINT_HANDOFF_SHA256 = (
    "abd617419d4fc0b481576c8366ba9a98429c64577d39d3dd6f56b7805090dfdd"
)
EXPECTED_LONG_ENDPOINT_SHA256 = (
    "2e9bb6979f6ac76b368a67332ede97eab840112146ec0904b1c74e34c847b7a8"
)
STANDARD_MANIFEST_NAME = "P5_PILOT_SAFETY_V2_2_PROPOSED_ONE_CALL_DIAGNOSTIC_2026-08-09.json"
EXPECTED_STANDARD_MANIFEST_INTERNAL_SHA256 = (
    "23a494840511a5f7a18965489fbd17aa5c08ae11711137d8616341db1698d5e2"
)
EXPECTED_STANDARD_MANIFEST_FILE_SHA256 = (
    "ce6da07d9f80bc3ec07b254551c26e192c95d8a798c9ae754fa9bd1ed907db78"
)
EXPECTED_STANDARD_RUNNER_SHA256 = (
    "476b8db87e5ba7d864eb0164ecc83a97f3fe9c077d30455988309b02443b0fbb"
)
EXPECTED_STANDARD_OUTPUT_SHA256 = (
    "57b1ec877a71d3f8ae23ff97ae23763dec7d4def6f06f2f27ec4b3dd68eed759"
)
EXPECTED_STANDARD_JOURNAL_SHA256 = (
    "0def32e7143db9be09781493aaf41a3f88dc52c52622143c646c43a2dd07ca77"
)
EXPECTED_STANDARD_OUTCOME = "empty_scorer_text_without_filter_marker"
EXPECTED_STANDARD_RAW_PAYLOAD_SHA256 = (
    "04357631ff49178aa3434f82d97c1d818c56882db0f97dfcd9ce458c8b895700"
)
TEST_NAME = "test_p5_pilot_safety_v2_2_long_endpoint_comparison.py"
OUTPUT_NAME = "safety_v2_2_long_endpoint_shape_diagnostic.json"
JOURNAL_NAME = "safety_v2_2_long_endpoint_shape_diagnostic_journal.jsonl"


class LongEndpointDiagnosticError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def manifest_hash(document: dict[str, Any]) -> str:
    content = dict(document)
    content.pop("manifest_sha256", None)
    return p5.sha256_json(content)


def long_endpoint_from_approved_handoff() -> str:
    path = HERE / LONG_ENDPOINT_HANDOFF_NAME
    if p5.sha256_file(path) != EXPECTED_LONG_ENDPOINT_HANDOFF_SHA256:
        raise LongEndpointDiagnosticError("approved long-endpoint handoff hash mismatch")
    matches = re.findall(r"https://[^`\s]+", path.read_text())
    if len(matches) != 1:
        raise LongEndpointDiagnosticError("approved handoff must contain exactly one HTTPS endpoint")
    endpoint = matches[0]
    if hashlib.sha256(endpoint.encode("utf-8")).hexdigest() != EXPECTED_LONG_ENDPOINT_SHA256:
        raise LongEndpointDiagnosticError("approved long endpoint SHA-256 mismatch")
    return endpoint


def standard_bindings(run_root: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    run_root = executor.assert_out_path(run_root)
    if p5.sha256_file(HERE / "p5_pilot_safety_v2_2_one_call_diagnostic.py") != EXPECTED_STANDARD_RUNNER_SHA256:
        raise LongEndpointDiagnosticError("standard diagnostic runner hash mismatch")
    standard_manifest_path = HERE / STANDARD_MANIFEST_NAME
    if p5.sha256_file(standard_manifest_path) != EXPECTED_STANDARD_MANIFEST_FILE_SHA256:
        raise LongEndpointDiagnosticError("standard diagnostic manifest file hash mismatch")
    standard_manifest = standard.load_manifest(standard_manifest_path)
    if standard_manifest["manifest_sha256"] != EXPECTED_STANDARD_MANIFEST_INTERNAL_SHA256:
        raise LongEndpointDiagnosticError("standard diagnostic manifest internal hash mismatch")
    standard.verify_original_artifacts(run_root)
    target, prompt = standard.target_binding(run_root)
    if p5.sha256_json(standard._request_body(prompt)) != target["serialized_request_sha256"]:
        raise LongEndpointDiagnosticError("standard serialized request binding mismatch")

    output_path = run_root / standard.DIAGNOSTIC_OUTPUT
    journal_path = run_root / standard.DIAGNOSTIC_JOURNAL
    if p5.sha256_file(output_path) != EXPECTED_STANDARD_OUTPUT_SHA256:
        raise LongEndpointDiagnosticError("standard diagnostic output hash mismatch")
    if p5.sha256_file(journal_path) != EXPECTED_STANDARD_JOURNAL_SHA256:
        raise LongEndpointDiagnosticError("standard diagnostic journal hash mismatch")
    output = json.loads(output_path.read_text())
    if (
        output.get("outcome") != EXPECTED_STANDARD_OUTCOME
        or output.get("network_attempts") != 1
        or output.get("automatic_retries") != 0
        or output.get("source_target") != target
        or output.get("telemetry", {}).get("raw_payload_sha256")
        != EXPECTED_STANDARD_RAW_PAYLOAD_SHA256
    ):
        raise LongEndpointDiagnosticError("standard diagnostic result differs from frozen comparison source")
    return target, prompt, output


def build_plan(run_root: Path) -> dict[str, Any]:
    target, prompt, standard_output = standard_bindings(run_root)
    long_endpoint_from_approved_handoff()
    request_sha = p5.sha256_json(standard._request_body(prompt))
    if request_sha != target["serialized_request_sha256"]:
        raise LongEndpointDiagnosticError("long comparison request differs from standard request")
    document: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}-plan",
        "status": "dry_run_non_executable",
        "created_at_utc": utc_now(),
        "proxy_or_model_calls_made_by_planning": 0,
        "purpose": "official diagnostic-only standard-versus-long endpoint comparison",
        "sources": {
            "long_endpoint_handoff": {
                "path": str((HERE / LONG_ENDPOINT_HANDOFF_NAME).resolve()),
                "file_sha256": EXPECTED_LONG_ENDPOINT_HANDOFF_SHA256,
                "endpoint_url_sha256": EXPECTED_LONG_ENDPOINT_SHA256,
                "endpoint_url_persisted": False,
            },
            "standard_diagnostic_manifest": {
                "path": str((HERE / STANDARD_MANIFEST_NAME).resolve()),
                "internal_sha256": EXPECTED_STANDARD_MANIFEST_INTERNAL_SHA256,
                "file_sha256": EXPECTED_STANDARD_MANIFEST_FILE_SHA256,
            },
            "standard_diagnostic_output": {
                "path": str((run_root / standard.DIAGNOSTIC_OUTPUT).resolve()),
                "file_sha256": EXPECTED_STANDARD_OUTPUT_SHA256,
                "outcome": standard_output["outcome"],
                "raw_payload_sha256": EXPECTED_STANDARD_RAW_PAYLOAD_SHA256,
            },
            "standard_diagnostic_journal": {
                "path": str((run_root / standard.DIAGNOSTIC_JOURNAL).resolve()),
                "file_sha256": EXPECTED_STANDARD_JOURNAL_SHA256,
            },
            "standard_diagnostic_runner": {
                "path": str((HERE / "p5_pilot_safety_v2_2_one_call_diagnostic.py").resolve()),
                "file_sha256": EXPECTED_STANDARD_RUNNER_SHA256,
            },
            "long_comparison_runner": {
                "path": str(Path(__file__).resolve()),
                "file_sha256": p5.sha256_file(Path(__file__).resolve()),
            },
            "long_comparison_tests": {
                "path": str((HERE / TEST_NAME).resolve()),
                "file_sha256": p5.sha256_file(HERE / TEST_NAME),
            },
        },
        "target": target,
        "request_identity": {
            "serialized_request_sha256": request_sha,
            "diagnostic_prompt_sha256": target["diagnostic_prompt_sha256"],
            "generation_record_sha256": target["generation_record_sha256"],
            "chunk_plan_sha256": target["chunk_plan_sha256"],
            "model": MODEL,
            "temperature": 0.0,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "request_body_identical_to_standard": True,
        },
        "transport_contract": {
            "endpoint_class": "official_long_120_second",
            "endpoint_url_sha256": EXPECTED_LONG_ENDPOINT_SHA256,
            "endpoint_url_persisted": False,
            "service_limit_seconds": LONG_ENDPOINT_SERVICE_LIMIT_SECONDS,
            "client_timeout_seconds": CLIENT_TIMEOUT_SECONDS,
            "client_timeout_unchanged_from_standard": True,
            "only_endpoint_base_url_changes": True,
            "same_api_key_and_schema": True,
            "initial_attempts": 1,
            "maximum_total_attempts": 1,
            "automatic_retries": 0,
        },
        "financial_guards": {
            "maximum_cost_per_request_usd": MAX_COST_PER_REQUEST_USD,
            "total_spend_ceiling_usd": SPEND_CEILING_USD,
            "quota_stop_floor_usd": QUOTA_STOP_FLOOR_USD,
            "ceiling_product_usd": MAX_ATTEMPTS * MAX_COST_PER_REQUEST_USD,
            "ceiling_product_within_spend": MAX_ATTEMPTS * MAX_COST_PER_REQUEST_USD <= SPEND_CEILING_USD,
        },
        "telemetry_contract": copy.deepcopy(standard_manifest_telemetry()),
        "output_exclusions": {
            "annotations": True,
            "validation_gate": True,
            "recovery": True,
            "held_out": True,
            "standard_diagnostic_mutated": False,
            "frozen_v2_2_mutated": False,
        },
        "prospective_outputs": {
            "diagnostic": str((run_root / OUTPUT_NAME).resolve()),
            "journal": str((run_root / JOURNAL_NAME).resolve()),
        },
    }
    document["manifest_sha256"] = manifest_hash(document)
    return document


def standard_manifest_telemetry() -> dict[str, Any]:
    manifest = standard.load_manifest(HERE / STANDARD_MANIFEST_NAME)
    contract = copy.deepcopy(manifest["telemetry_contract"])
    contract["identical_to_standard_diagnostic"] = True
    return contract


def build_authorized_manifest(plan_path: Path) -> dict[str, Any]:
    plan_path = executor.assert_out_path(plan_path)
    plan = json.loads(plan_path.read_text())
    if plan.get("manifest_sha256") != manifest_hash(plan) or plan.get("status") != "dry_run_non_executable":
        raise LongEndpointDiagnosticError("invalid long comparison plan")
    document = copy.deepcopy(plan)
    document.pop("manifest_sha256", None)
    document["schema_version"] = f"{SCHEMA_VERSION}-authorized-executable"
    document["status"] = "authorized_for_execution"
    document["created_at_utc"] = utc_now()
    document["sources"]["long_comparison_plan"] = {
        "path": str(plan_path.resolve()),
        "internal_sha256": plan["manifest_sha256"],
        "file_sha256": p5.sha256_file(plan_path),
    }
    document["owner_authorization"] = {
        "received": True,
        "source": "Tony direct chat authorization before deterministic manifest construction",
        "exact_quote": OWNER_AUTHORIZATION_QUOTE,
        "authorizes_one_long_endpoint_call_without_hash_repetition": True,
        "authorized_attempts": 1,
        "authorized_retries": 0,
        "authorized_total_and_per_call_ceiling_usd": 0.025,
        "diagnostic_only": True,
    }
    document["authorization_guard"] = {
        "execution_authorized": True,
        "exact_internal_and_file_hash_cli_binding_required_for_integrity": True,
        "additional_owner_hash_repetition_required": False,
        "approved_request_ceiling": 1,
        "approved_retry_ceiling": 0,
        "approved_spend_ceiling_usd": SPEND_CEILING_USD,
        "approved_max_cost_per_request_usd": MAX_COST_PER_REQUEST_USD,
        "quota_stop_floor_usd": QUOTA_STOP_FLOOR_USD,
        "recovery_heldout_or_other_calls_authorized": False,
    }
    document["manifest_sha256"] = manifest_hash(document)
    return document


def load_manifest(path: Path) -> dict[str, Any]:
    path = executor.assert_out_path(path)
    document = json.loads(path.read_text())
    if document.get("manifest_sha256") != manifest_hash(document):
        raise LongEndpointDiagnosticError("invalid long comparison manifest internal hash")
    sources = document.get("sources", {})
    if sources.get("long_comparison_runner", {}).get("file_sha256") != p5.sha256_file(Path(__file__).resolve()):
        raise LongEndpointDiagnosticError("long comparison runner hash mismatch")
    if sources.get("long_comparison_tests", {}).get("file_sha256") != p5.sha256_file(HERE / TEST_NAME):
        raise LongEndpointDiagnosticError("long comparison tests hash mismatch")
    return document


def require_authorisation(
    document: dict[str, Any],
    manifest_path: Path,
    supplied_internal_sha256: str | None,
    supplied_file_sha256: str | None,
    authorised: bool,
) -> None:
    guard = document.get("authorization_guard", {})
    owner = document.get("owner_authorization", {})
    if document.get("status") != "authorized_for_execution" or guard.get("execution_authorized") is not True:
        raise LongEndpointDiagnosticError("long comparison manifest is non-executable")
    if owner.get("received") is not True or owner.get("exact_quote") != OWNER_AUTHORIZATION_QUOTE:
        raise LongEndpointDiagnosticError("direct owner authorization is absent or altered")
    if not authorised:
        raise LongEndpointDiagnosticError("--authorised is required")
    if supplied_internal_sha256 != document.get("manifest_sha256"):
        raise LongEndpointDiagnosticError("exact long comparison internal hash is required")
    if supplied_file_sha256 != p5.sha256_file(executor.assert_out_path(manifest_path)):
        raise LongEndpointDiagnosticError("exact long comparison file hash is required")
    if (
        guard.get("approved_request_ceiling") != 1
        or guard.get("approved_retry_ceiling") != 0
        or guard.get("approved_spend_ceiling_usd") != SPEND_CEILING_USD
        or guard.get("approved_max_cost_per_request_usd") != MAX_COST_PER_REQUEST_USD
        or guard.get("quota_stop_floor_usd") != QUOTA_STOP_FLOOR_USD
        or guard.get("recovery_heldout_or_other_calls_authorized") is not False
        or MAX_ATTEMPTS * MAX_COST_PER_REQUEST_USD > SPEND_CEILING_USD
    ):
        raise LongEndpointDiagnosticError("long comparison authorization guards differ from frozen scope")


def perform_long_call(
    *,
    endpoint: str,
    key: str,
    prompt: str,
    target: dict[str, Any],
    emit_event: Callable[[dict[str, Any]], None],
    post: Callable[..., Any] = requests.post,
) -> dict[str, Any]:
    if hashlib.sha256(endpoint.encode("utf-8")).hexdigest() != EXPECTED_LONG_ENDPOINT_SHA256:
        raise LongEndpointDiagnosticError("runtime endpoint differs from authorized long endpoint")

    def emit_wrapped(event: dict[str, Any]) -> None:
        emit_event(
            {
                **event,
                "endpoint_class": "official_long_120_second",
                "endpoint_url_sha256": EXPECTED_LONG_ENDPOINT_SHA256,
                "endpoint_url_persisted": False,
                "request_body_identical_to_standard": True,
            }
        )

    result = standard.perform_one_call(
        url=endpoint,
        key=key,
        prompt=prompt,
        target=target,
        emit_event=emit_wrapped,
        post=post,
    )
    return {
        **result,
        "schema_version": SCHEMA_VERSION,
        "endpoint_class": "official_long_120_second",
        "endpoint_url_sha256": EXPECTED_LONG_ENDPOINT_SHA256,
        "endpoint_url_persisted": False,
        "request_body_identical_to_standard": True,
    }


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
    require_authorisation(document, manifest_path, supplied_internal_sha256, supplied_file_sha256, authorised)
    target, prompt, standard_output = standard_bindings(run_root)
    endpoint = long_endpoint_from_approved_handoff()
    if target != document.get("target"):
        raise LongEndpointDiagnosticError("live target differs from authorized long comparison")
    request_sha = p5.sha256_json(standard._request_body(prompt))
    if request_sha != document["request_identity"]["serialized_request_sha256"]:
        raise LongEndpointDiagnosticError("live request differs from authorized long comparison")
    if document["transport_contract"]["client_timeout_seconds"] != standard.REQUEST_TIMEOUT_SECONDS:
        raise LongEndpointDiagnosticError("client timeout differs from standard comparison")

    output_path = run_root / OUTPUT_NAME
    journal_path = run_root / JOURNAL_NAME
    if output_path.exists() or journal_path.exists():
        raise LongEndpointDiagnosticError("long comparison output/reservation exists; refusing rerun")
    key = os.environ.get("CLAUDE_PROXY_KEY")
    if not key:
        raise LongEndpointDiagnosticError("configured proxy key is required")

    events: list[dict[str, Any]] = []

    def emit_event(event: dict[str, Any]) -> None:
        safe_event = {
            **event,
            "manifest_internal_sha256": document["manifest_sha256"],
            "manifest_file_sha256": p5.sha256_file(manifest_path),
        }
        executor.append_jsonl(journal_path, safe_event)
        events.append(safe_event)

    result = perform_long_call(
        endpoint=endpoint,
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
        "standard_comparison": {
            "outcome": standard_output["outcome"],
            "raw_payload_sha256": standard_output["telemetry"]["raw_payload_sha256"],
            "usage_cost_usd": standard_output["telemetry"]["usage_cost_usd"],
            "remaining_quota_usd": standard_output["telemetry"]["remaining_quota_usd"],
            "output_file_sha256": EXPECTED_STANDARD_OUTPUT_SHA256,
        },
    }
    executor.atomic_write(output_path, json.dumps(final, indent=2, sort_keys=True) + "\n")
    return final


def write_json(path: Path, document: dict[str, Any]) -> str:
    path = executor.assert_out_path(path)
    executor.atomic_write(path, json.dumps(document, indent=2, sort_keys=True) + "\n")
    return p5.sha256_file(path)


def authorization_markdown(document: dict[str, Any], file_sha256: str) -> str:
    return f"""# Authorized P5 v2.2 official long-endpoint comparison

Tony directly authorized this deterministic one-call construction and execution
before its hashes were frozen. No additional hash repetition is required.

- Exact authorization: “{OWNER_AUTHORIZATION_QUOTE}”
- Internal manifest SHA-256: `{document['manifest_sha256']}`
- File SHA-256: `{file_sha256}`
- Request-body SHA-256: `{document['request_identity']['serialized_request_sha256']}`
- Long endpoint URL SHA-256: `{EXPECTED_LONG_ENDPOINT_SHA256}`
- Model: `{MODEL}`
- Attempts / retries: 1 / 0
- Output tokens / client timeout: {MAX_OUTPUT_TOKENS} / {CLIENT_TIMEOUT_SECONDS}s
- Long endpoint service limit: {LONG_ENDPOINT_SERVICE_LIMIT_SECONDS}s
- Total and per-call ceiling: ${SPEND_CEILING_USD:.3f}
- Quota floor: ${QUOTA_STOP_FLOOR_USD:.2f}
- Use: diagnostic only; excluded from annotation, gates, held-out, and recovery.

The endpoint URL itself is not persisted in this manifest/report; it is loaded
from the exact approved handoff and verified by SHA-256 before execution.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--run-root", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    auth_parser = sub.add_parser("authorize")
    auth_parser.add_argument("--plan", type=Path, required=True)
    auth_parser.add_argument("--output", type=Path, required=True)
    auth_parser.add_argument("--report", type=Path, required=True)
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
    elif args.command == "authorize":
        document = build_authorized_manifest(args.plan)
        file_sha = write_json(args.output, document)
        executor.atomic_write(executor.assert_out_path(args.report), authorization_markdown(document, file_sha))
        print(json.dumps({"status": document["status"], "internal_sha256": document["manifest_sha256"], "file_sha256": file_sha}))
    elif args.command == "execute":
        result = execute(args.run_root, args.manifest, args.manifest_sha256, args.manifest_file_sha256, args.authorised)
        print(json.dumps({"outcome": result["outcome"], "network_attempts": result["network_attempts"], "endpoint_class": result["endpoint_class"]}))


if __name__ == "__main__":
    main()
