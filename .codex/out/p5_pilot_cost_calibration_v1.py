#!/usr/bin/env python3
"""Hash-gated two-call cost calibration for the final P5 v2.1 plan.

Planning is zero cost.  Execution requires the exact internal manifest hash,
the exact manifest file hash, an explicit authorization flag, and externally
approved cost/quota limits.  Calibration output is cost metadata only and is
never written to behavioural or safety annotation files.
"""

from __future__ import annotations

import argparse
import hashlib
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
import p5_pilot_scorer_v2 as scorer  # noqa: E402


SCHEMA_VERSION = "p5-pilot-cost-calibration-v1"
MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"
MAX_OUTPUT_TOKENS = 800
TIMEOUT_SECONDS = 25
REQUEST_CEILING = 2
RETRY_CEILING = 0
JOURNAL_NAME = "cost_calibration_call_journal_v1.jsonl"
RESULTS_NAME = "cost_calibration_results_v1.jsonl"


class CalibrationError(RuntimeError):
    pass


class CalibrationAuthorizationError(CalibrationError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def manifest_hash(document: dict[str, Any]) -> str:
    content = dict(document)
    content.pop("manifest_sha256", None)
    return p5.sha256_json(content)


def canonical_payload(prompt: str) -> tuple[dict[str, Any], bytes]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return payload, encoded


def _request_descriptors(
    run_root: Path, scoring_plan: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows, snapshot = scorer.stable_generation_snapshot(run_root / "generations.jsonl")
    expected = scoring_plan["sources"]["generation_snapshot"]
    if snapshot["sha256"] != expected["sha256"]:
        raise CalibrationError("generation snapshot differs from final scoring plan")
    if len(rows) != 176 or any(row["status"] != "success" for row in rows):
        raise CalibrationError("cost calibration requires 176 successful generation rows")
    run_manifest = executor.load_manifest(run_root)
    prompts = {row["prompt_id"]: row for row in run_manifest["prompts"]}
    row_index = {(row["checkpoint_role"], row["prompt_id"]): row for row in rows}
    descriptors: list[dict[str, Any]] = []
    prompts_by_selection: dict[str, dict[str, Any]] = {}
    for item in scoring_plan["initial_network_request_plan"]:
        if item["pass_id"] != "primary":
            raise CalibrationError("final calibration input must be a primary-only plan")
        generation = row_index[(item["checkpoint_role"], item["prompt_id"])]
        chunks = scorer.chunks_for_text(generation["text"], stratum=generation["stratum"])
        chunk = chunks[item["chunk_index"]]
        if generation["stratum"] == "generic":
            prompt = scorer.behaviour_prompt(chunk, len(chunks))
            request_class = "generic"
        else:
            prompt = scorer.safety_prompt(
                generation["stratum"],
                prompts[generation["prompt_id"]]["prompt"],
                chunk,
                len(chunks),
                generation.get("stop_reason"),
            )
            request_class = "safety"
        _, encoded = canonical_payload(prompt)
        selection_id = f"{request_class}:{item['logical_assignment_id']}:{item['chunk_index']}"
        descriptor = {
            "selection_id": selection_id,
            "request_class": request_class,
            "logical_assignment_id": item["logical_assignment_id"],
            "checkpoint_role": item["checkpoint_role"],
            "prompt_id": item["prompt_id"],
            "stratum": item["stratum"],
            "chunk_index": item["chunk_index"],
            "n_chunks": item["n_chunks"],
            "canonical_serialized_request_bytes": len(encoded),
            "canonical_request_sha256": hashlib.sha256(encoded).hexdigest(),
            "source_char_start": item["char_start"],
            "source_char_end": item["char_end"],
            "source_chars": item["n_chars"],
            "source_units": item["n_units"],
        }
        descriptors.append(descriptor)
        prompts_by_selection[selection_id] = {"prompt": prompt, "descriptor": descriptor}
    return descriptors, prompts_by_selection


def deterministic_selections(descriptors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for request_class in ("generic", "safety"):
        candidates = [row for row in descriptors if row["request_class"] == request_class]
        if not candidates:
            raise CalibrationError(f"no {request_class} requests in final plan")
        ranked = sorted(
            candidates,
            key=lambda row: (
                -row["canonical_serialized_request_bytes"],
                row["logical_assignment_id"],
                row["chunk_index"],
            ),
        )
        selected.append(ranked[0])
    if len({row["selection_id"] for row in selected}) != REQUEST_CEILING:
        raise CalibrationError("calibration selections are not two unique requests")
    return selected


def build_manifest(run_root: Path, scoring_plan_path: Path) -> dict[str, Any]:
    run_root = executor.assert_out_path(run_root)
    scoring_plan_path = executor.assert_out_path(scoring_plan_path)
    scoring_plan = scorer.load_v2_manifest(scoring_plan_path)
    accounting = scoring_plan["dry_run_accounting"]
    if (
        scoring_plan["status"] != "dry_run_not_authorized"
        or scoring_plan["reliability_status"]["repeat_policy"] != "drop"
        or accounting["generation_rows_snapshot"] != 176
        or accounting["missing_generation_keys_vs_176"] != 0
        or accounting["full_176_primary_only_initial_requests"] != 490
    ):
        raise CalibrationError("scoring plan is not the final 176-row/490-request primary plan")
    descriptors, _ = _request_descriptors(run_root, scoring_plan)
    selections = deterministic_selections(descriptors)
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "awaiting_hash_bound_authorization",
        "created_at_utc": utc_now(),
        "run_id": scoring_plan["run_id"],
        "purpose": "two-call cost calibration only",
        "proxy_calls_made_during_planning": 0,
        "sources": {
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": p5.sha256_file(Path(__file__).resolve()),
            "test_path": str((HERE / "test_p5_pilot_cost_calibration_v1.py").resolve()),
            "test_sha256": p5.sha256_file(HERE / "test_p5_pilot_cost_calibration_v1.py"),
            "scorer_path": str(Path(scorer.__file__).resolve()),
            "scorer_sha256": p5.sha256_file(Path(scorer.__file__).resolve()),
            "scoring_plan_path": str(scoring_plan_path.resolve()),
            "scoring_plan_internal_sha256": scoring_plan["manifest_sha256"],
            "scoring_plan_file_sha256": p5.sha256_file(scoring_plan_path),
            "generation_snapshot": scoring_plan["sources"]["generation_snapshot"],
        },
        "transport": {
            "model": MODEL,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "timeout_seconds": TIMEOUT_SECONDS,
            "temperature": 0.0,
            "sequential": True,
            "max_concurrency": 1,
            "request_ceiling": REQUEST_CEILING,
            "retry_ceiling": RETRY_CEILING,
            "retry_policy": "no retry; any failed or ambiguous request stops calibration",
        },
        "selection_rule": "largest canonical serialized generic request and largest canonical serialized safety request; ties by logical_assignment_id then chunk_index",
        "selections": selections,
        "output_policy": {
            "calibration_only": True,
            "excluded_from_behaviour_annotations": True,
            "excluded_from_safety_scores": True,
            "raw_model_text_persisted": False,
            "persisted_fields": "request/response hashes, usage.cost, remaining quota, timing, status",
            "journal_name": JOURNAL_NAME,
            "results_name": RESULTS_NAME,
        },
        "proposed_authorization": {
            "execution_authorized": False,
            "approved_request_ceiling": None,
            "approved_spend_ceiling_usd": None,
            "approved_max_cost_per_call_usd": None,
            "quota_stop_floor_usd": None,
            "note": "all ceilings remain unset until Tony gives exact hash-bound authorization",
        },
    }
    document["manifest_sha256"] = manifest_hash(document)
    return document


def load_manifest(path: Path) -> dict[str, Any]:
    path = executor.assert_out_path(path)
    document = json.loads(path.read_text())
    if document.get("manifest_sha256") != manifest_hash(document):
        raise CalibrationError("invalid calibration manifest internal hash")
    sources = document.get("sources", {})
    if sources.get("runner_sha256") != p5.sha256_file(Path(__file__).resolve()):
        raise CalibrationError("calibration runner differs from manifest")
    if sources.get("test_sha256") != p5.sha256_file(HERE / "test_p5_pilot_cost_calibration_v1.py"):
        raise CalibrationError("calibration tests differ from manifest")
    if sources.get("scorer_sha256") != p5.sha256_file(Path(scorer.__file__).resolve()):
        raise CalibrationError("v2.1 scorer differs from calibration manifest")
    return document


def require_authorization(
    manifest_path: Path,
    document: dict[str, Any],
    *,
    supplied_internal_hash: str | None,
    supplied_file_hash: str | None,
    authorised: bool,
    approved_request_ceiling: int | None,
    approved_spend_ceiling_usd: float | None,
    approved_max_cost_per_call_usd: float | None,
    quota_stop_floor_usd: float | None,
) -> dict[str, float | int]:
    actual_file_hash = p5.sha256_file(manifest_path)
    if not authorised:
        raise CalibrationAuthorizationError("--authorised is required")
    if supplied_internal_hash != document["manifest_sha256"]:
        raise CalibrationAuthorizationError("exact calibration internal hash is required")
    if supplied_file_hash != actual_file_hash:
        raise CalibrationAuthorizationError("exact calibration file hash is required")
    if approved_request_ceiling != REQUEST_CEILING:
        raise CalibrationAuthorizationError("calibration approval must authorize exactly two requests")
    numeric = {
        "approved_spend_ceiling_usd": approved_spend_ceiling_usd,
        "approved_max_cost_per_call_usd": approved_max_cost_per_call_usd,
        "quota_stop_floor_usd": quota_stop_floor_usd,
    }
    for field, value in numeric.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise CalibrationAuthorizationError(f"{field} must be an approved non-negative number")
    if approved_spend_ceiling_usd <= 0 or approved_max_cost_per_call_usd <= 0:
        raise CalibrationAuthorizationError("approved spend and per-call ceilings must be positive")
    if REQUEST_CEILING * approved_max_cost_per_call_usd > approved_spend_ceiling_usd:
        raise CalibrationAuthorizationError("two-call maximum exceeds approved spend ceiling")
    return {"approved_request_ceiling": approved_request_ceiling, **numeric}


def calibration_proxy_call(
    *,
    url: str,
    key: str,
    prompt: str,
    selection: dict[str, Any],
    emit_event: Callable[[dict[str, Any]], None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Perform exactly one calibration request.  There is deliberately no retry loop."""
    payload, encoded = canonical_payload(prompt)
    if hashlib.sha256(encoded).hexdigest() != selection["canonical_request_sha256"]:
        raise CalibrationError("reconstructed calibration request hash mismatch")
    call_id = str(uuid.uuid4())
    common = {
        "call_id": call_id,
        "selection_id": selection["selection_id"],
        "request_class": selection["request_class"],
        "model": MODEL,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "retry_number": 0,
    }
    emit_event({"event": "reserved", "at_utc": utc_now(), **common})
    started = time.monotonic()
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"X-Api-Key": key, "Content-Type": "application/json"},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        proxy_payload = response.json()
        text = scorer.extract_proxy_text(proxy_payload)
        if not text.strip():
            raise CalibrationError("proxy returned empty calibration response")
        cost, quota = scorer.proxy_accounting(proxy_payload)
        completed = {
            "event": "completed",
            "at_utc": utc_now(),
            **common,
            "status": "success",
            "usage_cost_usd": cost,
            "remaining_quota_usd": quota,
            "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "wall_seconds": time.monotonic() - started,
            "calibration_only_excluded_from_annotations": True,
        }
        emit_event(completed)
        result = {
            "schema_version": SCHEMA_VERSION,
            "selection_id": selection["selection_id"],
            "call_id": call_id,
            "status": "success",
            "usage_cost_usd": cost,
            "remaining_quota_usd": quota,
            "response_sha256": completed["response_sha256"],
            "raw_model_text_persisted": False,
            "calibration_only_excluded_from_annotations": True,
        }
        return result, proxy_payload
    except Exception as error:
        emit_event(
            {
                "event": "completed",
                "at_utc": utc_now(),
                **common,
                "status": "error_no_retry",
                "usage_cost_usd": None,
                "remaining_quota_usd": None,
                "error_type": type(error).__name__,
                "wall_seconds": time.monotonic() - started,
                "calibration_only_excluded_from_annotations": True,
            }
        )
        raise


def execute(
    run_root: Path,
    manifest_path: Path,
    *,
    supplied_internal_hash: str | None,
    supplied_file_hash: str | None,
    authorised: bool,
    approved_request_ceiling: int | None,
    approved_spend_ceiling_usd: float | None,
    approved_max_cost_per_call_usd: float | None,
    quota_stop_floor_usd: float | None,
) -> None:
    run_root = executor.assert_out_path(run_root)
    manifest_path = executor.assert_out_path(manifest_path)
    document = load_manifest(manifest_path)
    limits = require_authorization(
        manifest_path,
        document,
        supplied_internal_hash=supplied_internal_hash,
        supplied_file_hash=supplied_file_hash,
        authorised=authorised,
        approved_request_ceiling=approved_request_ceiling,
        approved_spend_ceiling_usd=approved_spend_ceiling_usd,
        approved_max_cost_per_call_usd=approved_max_cost_per_call_usd,
        quota_stop_floor_usd=quota_stop_floor_usd,
    )
    url = os.environ.get("CLAUDE_PROXY_URL")
    key = os.environ.get("CLAUDE_PROXY_KEY")
    if not url or not key:
        raise CalibrationAuthorizationError("CLAUDE_PROXY_URL and CLAUDE_PROXY_KEY are required")

    scoring_plan_path = Path(document["sources"]["scoring_plan_path"])
    scoring_plan = scorer.load_v2_manifest(scoring_plan_path)
    descriptors, prompt_index = _request_descriptors(run_root, scoring_plan)
    if deterministic_selections(descriptors) != document["selections"]:
        raise CalibrationError("deterministic selections differ from calibration manifest")

    journal_path = run_root / JOURNAL_NAME
    results_path = run_root / RESULTS_NAME
    journal = scorer.jsonl_rows(journal_path)
    results = scorer.jsonl_rows(results_path)
    reserved_ids = [row["call_id"] for row in journal if row.get("event") == "reserved"]
    completed_ids = {row["call_id"] for row in journal if row.get("event") == "completed"}
    dangling = [call_id for call_id in reserved_ids if call_id not in completed_ids]
    if dangling:
        raise CalibrationError("dangling calibration reservation; no implicit retry permitted")
    if len(reserved_ids) > REQUEST_CEILING:
        raise CalibrationError("calibration request ceiling already exceeded")
    done = {row["selection_id"] for row in results if row.get("status") == "success"}

    def emit_event(event: dict[str, Any]) -> None:
        scorer.append_jsonl(journal_path, event)
        journal.append(event)

    emit_event(
        {
            "event": "authorization_bound",
            "at_utc": utc_now(),
            "manifest_internal_sha256": document["manifest_sha256"],
            "manifest_file_sha256": p5.sha256_file(manifest_path),
            **limits,
            "retry_ceiling": RETRY_CEILING,
        }
    )
    for selection in document["selections"]:
        if selection["selection_id"] in done:
            continue
        reservations = sum(row.get("event") == "reserved" for row in journal)
        if reservations >= limits["approved_request_ceiling"]:
            raise CalibrationAuthorizationError("approved two-call ceiling reached")
        committed = reservations * limits["approved_max_cost_per_call_usd"]
        if committed + limits["approved_max_cost_per_call_usd"] > limits["approved_spend_ceiling_usd"]:
            raise CalibrationAuthorizationError("next calibration call could exceed approved spend")
        quotas = [
            row["remaining_quota_usd"]
            for row in journal
            if row.get("event") == "completed"
            and isinstance(row.get("remaining_quota_usd"), (int, float))
        ]
        if quotas and quotas[-1] < limits["quota_stop_floor_usd"]:
            raise CalibrationAuthorizationError("remaining quota is below approved floor")
        prompt = prompt_index[selection["selection_id"]]["prompt"]
        result, payload = calibration_proxy_call(
            url=url, key=key, prompt=prompt, selection=selection, emit_event=emit_event
        )
        scorer.append_jsonl(results_path, result)
        done.add(selection["selection_id"])
        cost, quota = scorer.proxy_accounting(payload)
        if cost > limits["approved_max_cost_per_call_usd"]:
            raise CalibrationAuthorizationError("observed calibration cost exceeded approved bound")
        if quota < limits["quota_stop_floor_usd"]:
            raise CalibrationAuthorizationError("remaining quota fell below approved floor")


def markdown_report(document: dict[str, Any], file_sha256: str) -> str:
    generic, safety = document["selections"]
    internal = document["manifest_sha256"]
    return "\n".join(
        [
            "# P5 v2.1 two-call cost calibration — authorization report",
            "",
            f"**Status:** `{document['status']}`  ",
            "**Proxy calls made during planning:** `0`  ",
            f"**Internal manifest SHA-256:** `{internal}`  ",
            f"**Manifest file SHA-256:** `{file_sha256}`",
            "",
            "## Deterministic selections",
            "",
            f"- Generic: `{generic['logical_assignment_id']}` chunk {generic['chunk_index']}; {generic['canonical_serialized_request_bytes']} canonical bytes; request SHA `{generic['canonical_request_sha256']}`.",
            f"- Safety: `{safety['logical_assignment_id']}` chunk {safety['chunk_index']}; {safety['canonical_serialized_request_bytes']} canonical bytes; request SHA `{safety['canonical_request_sha256']}`.",
            "",
            "Both calls use exact Sonnet, `max_tokens=800`, a 25-second timeout, temperature zero, sequential execution, and zero retries. Returned text is hashed but not persisted and is excluded from every annotation/scoring output.",
            "",
            "## Unset authorization fields",
            "",
            "- Approved request ceiling: `unset` (must equal 2)",
            "- Approved spend ceiling: `unset`",
            "- Approved maximum cost per call: `unset`",
            "- Remaining-quota stop floor: `unset`",
            "",
            "## Decision-ready authorization line",
            "",
            f"> I authorize the two calibration-only requests selected by P5 cost-calibration manifest internal SHA `{internal}` and file SHA `{file_sha256}`, with request ceiling `2`, total spend ceiling `$____`, maximum cost per call `$____`, and remaining-quota stop floor `$____`; exact Sonnet, 800 output tokens, 25-second timeout, sequential execution, zero retries, and exclusion of returned text from all pilot annotations/results are mandatory.",
            "",
            "Execution remains disabled until the blank ceilings are supplied together with both exact hashes and the explicit authorization flag.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--run-root", type=Path, required=True)
    plan.add_argument("--scoring-plan", type=Path, required=True)
    plan.add_argument("--json-out", type=Path, required=True)
    plan.add_argument("--markdown-out", type=Path, required=True)
    run = sub.add_parser("execute")
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--manifest-sha256")
    run.add_argument("--manifest-file-sha256")
    run.add_argument("--authorised", action="store_true")
    run.add_argument("--approved-request-ceiling", type=int)
    run.add_argument("--approved-spend-ceiling-usd", type=float)
    run.add_argument("--approved-max-cost-per-call-usd", type=float)
    run.add_argument("--quota-stop-floor-usd", type=float)
    args = parser.parse_args()
    if args.command == "plan":
        document = build_manifest(args.run_root, args.scoring_plan)
        executor.atomic_write(args.json_out, json.dumps(document, indent=2) + "\n")
        file_sha = p5.sha256_file(args.json_out)
        executor.atomic_write(args.markdown_out, markdown_report(document, file_sha))
        print(
            json.dumps(
                {
                    "manifest_sha256": document["manifest_sha256"],
                    "manifest_file_sha256": file_sha,
                    "selections": document["selections"],
                    "proxy_calls_made": 0,
                },
                indent=2,
            )
        )
    else:
        execute(
            args.run_root,
            args.manifest,
            supplied_internal_hash=args.manifest_sha256,
            supplied_file_hash=args.manifest_file_sha256,
            authorised=args.authorised,
            approved_request_ceiling=args.approved_request_ceiling,
            approved_spend_ceiling_usd=args.approved_spend_ceiling_usd,
            approved_max_cost_per_call_usd=args.approved_max_cost_per_call_usd,
            quota_stop_floor_usd=args.quota_stop_floor_usd,
        )


if __name__ == "__main__":
    main()
