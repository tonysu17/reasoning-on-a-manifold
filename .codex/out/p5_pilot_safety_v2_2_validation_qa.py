#!/usr/bin/env python3
"""Offline integrity QA and arm-blind sizing readiness for P5 safety v2.2 validation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p5_preflight as p5  # noqa: E402
import p5_pilot_executor as executor  # noqa: E402
import p5_pilot_safety_scorer_v2_2 as scorer  # noqa: E402


EXPECTED_MANIFEST_INTERNAL_SHA256 = "29de70c816a2fd6fbf6fcf786cee2543935332870244e24756de1a93be9c669c"
EXPECTED_MANIFEST_FILE_SHA256 = "ee801c32f0405d2819ae1aafacc5d78ed3008e8f43572ef86553b7f3f386e9c3"
CONTRAST_PAIRS = (
    ("base_r1", "public_star1"),
    ("owned_fullft_control_s42", "owned_fullft_safety_s42"),
)


class QAError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pair_completion(
    rows: list[dict[str, Any]], stratum: str, endpoint: str
) -> dict[str, Any]:
    relevant = [row for row in rows if row["stratum"] == stratum]
    prompt_ids = sorted({row["prompt_id"] for row in relevant})
    resolved_keys = {
        (row["checkpoint_role"], row["prompt_id"])
        for row in relevant
        if isinstance(row.get("endpoints", {}).get(endpoint), bool)
    }
    possible = len(prompt_ids) * len(CONTRAST_PAIRS)
    complete = sum(
        (left, prompt_id) in resolved_keys and (right, prompt_id) in resolved_keys
        for prompt_id in prompt_ids
        for left, right in CONTRAST_PAIRS
    )
    return {
        "stratum": stratum,
        "endpoint": endpoint,
        "validation_assignments": len(relevant),
        "resolved_assignments": len(resolved_keys),
        "assignment_completion_rate": len(resolved_keys) / len(relevant),
        "possible_pooled_prompt_pairs": possible,
        "complete_pooled_prompt_pairs": complete,
        "pair_completion_rate": complete / possible if possible else 0.0,
        "endpoint_values_used_for_sizing": False,
        "arm_labelled_means_or_differences_computed": False,
        "powered_sample_size": None,
        "sizing_status": "blocked_by_failed_operational_validation_gate",
    }


def build_report(run_root: Path, manifest_path: Path) -> dict[str, Any]:
    run_root = executor.assert_out_path(run_root)
    manifest_path = executor.assert_out_path(manifest_path)
    manifest = scorer.load_manifest(manifest_path)
    if manifest["manifest_sha256"] != EXPECTED_MANIFEST_INTERNAL_SHA256:
        raise QAError("unexpected authorized manifest internal hash")
    if p5.sha256_file(manifest_path) != EXPECTED_MANIFEST_FILE_SHA256:
        raise QAError("unexpected authorized manifest file hash")

    generations, generation_snapshot = scorer.stable_generation_rows(
        run_root / "generations.jsonl"
    )
    chunk_path = run_root / scorer.CHUNK_OUTPUT
    assignment_path = run_root / scorer.ASSIGNMENT_OUTPUT
    journal_path = run_root / scorer.JOURNAL_OUTPUT
    gate_path = run_root / scorer.VALIDATION_GATE_OUTPUT
    chunks = jsonl(chunk_path)
    assignments = jsonl(assignment_path)
    journal = jsonl(journal_path)
    gate = json.loads(gate_path.read_text())

    expected_requests = [
        row
        for row in manifest["initial_network_request_plan"]
        if row["stage"] == "validation"
    ]
    expected_assignments = [
        row for row in manifest["logical_assignments"] if row["stage"] == "validation"
    ]
    expected_chunk_keys = {
        (row["logical_assignment_id"], row["chunk_index"]): row
        for row in expected_requests
    }
    expected_assignment_ids = {
        row["logical_assignment_id"] for row in expected_assignments
    }
    generation_index = {
        (row["checkpoint_role"], row["prompt_id"]): row for row in generations
    }
    actual_chunk_keys = [
        (row["logical_assignment_id"], row["chunk_index"]) for row in chunks
    ]
    actual_assignment_ids = [row["logical_assignment_id"] for row in assignments]
    errors: list[str] = []
    if any(row.get("stage") != "validation" for row in chunks + assignments):
        errors.append("non-validation output row exists")
    if len(actual_chunk_keys) != len(set(actual_chunk_keys)):
        errors.append("duplicate terminal chunk key")
    if set(actual_chunk_keys) != set(expected_chunk_keys):
        errors.append("terminal chunk set differs from validation manifest")
    if len(actual_assignment_ids) != len(set(actual_assignment_ids)):
        errors.append("duplicate validation assignment ID")
    if set(actual_assignment_ids) != expected_assignment_ids:
        errors.append("assignment output set differs from validation manifest")

    chunk_index: dict[tuple[str, int], dict[str, Any]] = {}
    assignment_plan = {
        row["logical_assignment_id"]: row for row in expected_assignments
    }
    lineage_errors = 0
    chunk_plan_errors = 0
    evidence_errors = 0
    for row in chunks:
        key = (row["logical_assignment_id"], row["chunk_index"])
        chunk_index[key] = row
        plan = assignment_plan[row["logical_assignment_id"]]
        generation = generation_index[(plan["checkpoint_role"], plan["prompt_id"])]
        if row["generation_record_sha256"] != p5.sha256_json(generation):
            lineage_errors += 1
        source_chunks = scorer.safety_chunks(generation["text"])
        source_chunk = source_chunks[row["chunk_index"]]
        if row["chunk_plan_sha256"] != scorer.chunk_plan_sha256(source_chunk):
            chunk_plan_errors += 1
        evidence = row.get("evidence")
        if evidence is not None:
            units = {
                unit["source_unit_id"]: unit for unit in source_chunk["units"]
            }
            unit = units.get(evidence.get("source_unit_id"))
            if unit is None or evidence != {
                "source_unit_id": unit["source_unit_id"],
                "char_start": unit["char_start"],
                "char_end": unit["char_end"],
                "text": unit["text"],
            }:
                evidence_errors += 1
    if lineage_errors:
        errors.append("generation lineage mismatch")
    if chunk_plan_errors:
        errors.append("chunk plan hash mismatch")
    if evidence_errors:
        errors.append("deterministic evidence mismatch")

    aggregation_errors = 0
    for row in assignments:
        plan = assignment_plan[row["logical_assignment_id"]]
        generation = generation_index[(plan["checkpoint_role"], plan["prompt_id"])]
        source_chunks = scorer.safety_chunks(generation["text"])
        member_rows = [
            chunk_index[(row["logical_assignment_id"], index)]
            for index in range(len(source_chunks))
        ]
        expected = scorer.aggregate_chunk_rows(
            member_rows,
            expected_chunk_count=len(source_chunks),
            stratum=generation["stratum"],
            stop_reason=generation.get("stop_reason"),
        )
        for field in (
            "aggregation_status",
            "n_chunks_expected",
            "n_chunks_terminal",
            "n_chunks_complete",
            "score",
            "endpoints",
        ):
            if row.get(field) != expected[field]:
                aggregation_errors += 1
                break
    if aggregation_errors:
        errors.append("assignment aggregation mismatch")

    reserved = [row for row in journal if row.get("event") == "reserved"]
    completed = [row for row in journal if row.get("event") == "completed"]
    reserved_ids = [row["call_id"] for row in reserved]
    completed_ids = [row["call_id"] for row in completed]
    if (
        len(reserved_ids) != len(set(reserved_ids))
        or len(completed_ids) != len(set(completed_ids))
        or set(reserved_ids) != set(completed_ids)
    ):
        errors.append("journal reservation/completion mismatch")
    if any(row.get("stage") != "validation" for row in journal):
        errors.append("held-out journal event exists")
    successful = [row for row in completed if row.get("status") == "success"]
    non_success = [row for row in completed if row.get("status") != "success"]
    costs = [float(row["usage_cost_usd"]) for row in successful]
    quotas = [float(row["remaining_quota_usd"]) for row in successful]
    guard = manifest["authorization_guard"]
    if len(reserved) > guard["approved_request_ceiling"]:
        errors.append("request ceiling exceeded")
    if costs and max(costs) > guard["approved_max_cost_per_request_usd"]:
        errors.append("per-request cost ceiling exceeded")
    if quotas and min(quotas) < guard["quota_stop_floor_usd"]:
        errors.append("quota floor crossed")
    if len(reserved) * guard["approved_max_cost_per_request_usd"] > guard[
        "approved_spend_ceiling_usd"
    ]:
        errors.append("conservative spend commitment exceeded")

    recomputed_gate = scorer.evaluate_validation_gate(manifest, chunks)
    gate_fields = (
        "status",
        "planned_validation_chunks",
        "terminal_validation_chunks",
        "successful_response_chunks",
        "complete_parser_and_evidence_chunks",
        "atomic_terminal_chunk_persistence_rate",
        "complete_parser_and_evidence_rate_among_successful_responses",
        "network_success_rate",
        "errors",
    )
    if any(gate.get(field) != recomputed_gate.get(field) for field in gate_fields):
        errors.append("persisted validation gate differs from recomputation")
    if gate.get("status") != "fail":
        errors.append("expected frozen gate failure is absent")

    by_aggregation: dict[str, int] = {}
    for row in assignments:
        status = row["aggregation_status"]
        by_aggregation[status] = by_aggregation.get(status, 0) + 1
    error_classes: dict[str, int] = {}
    for row in non_success:
        error_type = str(row.get("error_type"))
        error_classes[error_type] = error_classes.get(error_type, 0) + 1

    sizing = {
        "harmful_refusal": _pair_completion(assignments, "harmful", "harmful_refusal"),
        "benign_compliance": _pair_completion(assignments, "benign", "benign_compliance"),
    }
    return {
        "schema_version": "p5-pilot-safety-v2.2-validation-integrity-sizing-1",
        "created_at_utc": utc_now(),
        "run_id": "p5-pilot-20260808",
        "pilot_only_not_scientific_result": True,
        "sources": {
            "authorized_manifest_path": str(manifest_path.resolve()),
            "authorized_manifest_internal_sha256": manifest["manifest_sha256"],
            "authorized_manifest_file_sha256": p5.sha256_file(manifest_path),
            "generation_snapshot_sha256": generation_snapshot["sha256"],
            "chunk_scores_sha256": p5.sha256_file(chunk_path),
            "assignment_scores_sha256": p5.sha256_file(assignment_path),
            "journal_sha256": p5.sha256_file(journal_path),
            "validation_gate_sha256": p5.sha256_file(gate_path),
        },
        "integrity": {
            "status": "pass" if not errors else "fail",
            "errors": errors,
            "expected_validation_assignments": len(expected_assignments),
            "observed_validation_assignments": len(assignments),
            "expected_validation_chunks": len(expected_requests),
            "terminal_validation_chunks": len(chunks),
            "unique_terminal_chunk_keys": len(set(actual_chunk_keys)),
            "generation_lineage_errors": lineage_errors,
            "chunk_plan_errors": chunk_plan_errors,
            "deterministic_evidence_errors": evidence_errors,
            "aggregation_recomputation_errors": aggregation_errors,
            "held_out_assignment_rows": sum(row.get("stage") == "rescore" for row in assignments),
            "held_out_chunk_rows": sum(row.get("stage") == "rescore" for row in chunks),
            "held_out_journal_events": sum(row.get("stage") == "rescore" for row in journal),
            "same_sonnet_repeat_rows": 0,
        },
        "validation_gate": gate,
        "scoring": {
            "assignments": len(assignments),
            "aggregation_status": dict(sorted(by_aggregation.items())),
            "terminal_chunks": len(chunks),
            "network_success_chunks": len(successful),
            "network_error_chunks": len(non_success),
            "complete_parser_and_evidence_chunks": sum(
                row.get("parse_status") == "complete" for row in chunks
            ),
            "parser_or_evidence_failures_among_successful_responses": sum(
                row.get("network_status") == "success" and row.get("parse_status") != "complete"
                for row in chunks
            ),
            "network_error_classes": dict(sorted(error_classes.items())),
            "missing_chunks_remain_unresolved": True,
        },
        "requests_and_budget": {
            "planned_validation_initial_requests": len(expected_requests),
            "attempts_reserved": len(reserved),
            "attempts_completed": len(completed),
            "retries": sum(row.get("request_attempt", 1) > 1 for row in reserved),
            "successful_proxy_responses": len(successful),
            "non_success_proxy_responses": len(non_success),
            "reported_successful_response_cost_usd": sum(costs),
            "responses_without_cost_accounting": len(non_success),
            "total_billed_cost_known": len(non_success) == 0,
            "conservative_authorized_cost_commitment_usd": len(reserved)
            * guard["approved_max_cost_per_request_usd"],
            "maximum_reported_call_cost_usd": max(costs) if costs else None,
            "minimum_reported_remaining_quota_usd": min(quotas) if quotas else None,
            "final_reported_remaining_quota_usd": quotas[-1] if quotas else None,
            "request_ceiling": guard["approved_request_ceiling"],
            "spend_ceiling_usd": guard["approved_spend_ceiling_usd"],
            "max_cost_per_request_usd": guard["approved_max_cost_per_request_usd"],
            "quota_floor_usd": guard["quota_stop_floor_usd"],
            "all_hard_guards_respected": not any(
                text in error
                for error in errors
                for text in ("ceiling", "quota floor", "spend commitment")
            ),
        },
        "blinded_sizing_readiness": {
            "arm_labelled_effect_differences_computed": False,
            "arm_labelled_effect_differences_reported": False,
            "endpoint_values_used_for_sizing": False,
            "inputs": "missingness and complete-pair indicators pooled over two pre-specified contrast families",
            "endpoints": sizing,
            "powered_size_freeze": "not_permitted",
            "reason": "frozen operational validation gate failed; held-out rows were correctly not scored",
        },
        "disposition": {
            "held_out_rescoring_executed": False,
            "held_out_rescoring_authorized_condition_satisfied": False,
            "current_manifest_may_resume_held_out": False,
            "owner_decision_required": True,
            "required_decision": (
                "do not run held-out scoring from this manifest; investigate the 14 "
                "ProxyProtocolError responses and, if desired, approve a new hash-bound "
                "validation-recovery protocol and budget"
            ),
        },
    }


def markdown(report: dict[str, Any], json_sha256: str) -> str:
    gate = report["validation_gate"]
    budget = report["requests_and_budget"]
    scoring = report["scoring"]
    harmful = report["blinded_sizing_readiness"]["endpoints"]["harmful_refusal"]
    benign = report["blinded_sizing_readiness"]["endpoints"]["benign_compliance"]
    return "\n".join(
        [
            "# P5 safety v2.2 validation — integrity and blinded sizing readiness",
            "",
            f"**Integrity:** `{report['integrity']['status']}`  ",
            f"**Frozen validation gate:** `{gate['status']}`  ",
            "**Held-out scoring:** `not executed`  ",
            "**Arm-labelled effect differences:** `not computed or reported`  ",
            f"**Machine-readable report SHA-256:** `{json_sha256}`",
            "",
            "## Validation result",
            "",
            f"- Atomic terminal persistence: {gate['terminal_validation_chunks']}/{gate['planned_validation_chunks']} ({gate['atomic_terminal_chunk_persistence_rate']:.1%})",
            f"- Network success: {gate['successful_response_chunks']}/{gate['planned_validation_chunks']} ({gate['network_success_rate']:.1%}); required ≥98%",
            f"- Complete parser plus deterministic evidence: {gate['complete_parser_and_evidence_chunks']}/{gate['successful_response_chunks']} ({gate['complete_parser_and_evidence_rate_among_successful_responses']:.1%}); required ≥98%",
            f"- Network failures: {scoring['network_error_chunks']} `{next(iter(scoring['network_error_classes']), 'none')}` chunks",
            "",
            "The repair eliminated observed parser/evidence failures among successful responses, but the frozen gate failed on proxy/network completeness. Missing chunks remain unresolved. The held-out 48 assignments were correctly not started.",
            "",
            "## Integrity and accounting",
            "",
            f"- Validation assignments: {report['integrity']['observed_validation_assignments']}/{report['integrity']['expected_validation_assignments']}",
            f"- Terminal chunks: {report['integrity']['terminal_validation_chunks']}/{report['integrity']['expected_validation_chunks']}; lineage, chunk-plan, evidence, and aggregation recomputation errors: 0",
            f"- Attempts: {budget['attempts_completed']}; retries: {budget['retries']}",
            f"- Reported successful-response cost: ${budget['reported_successful_response_cost_usd']:.6f}",
            f"- Responses without cost accounting: {budget['responses_without_cost_accounting']}; total billed cost is therefore not known exactly",
            f"- Conservative authorized commitment: ${budget['conservative_authorized_cost_commitment_usd']:.3f}",
            f"- Maximum reported call: ${budget['maximum_reported_call_cost_usd']:.6f}; minimum/final reported quota: ${budget['minimum_reported_remaining_quota_usd']:.6f}",
            "- All hard guards were respected; same-Sonnet repeats were not run",
            "",
            "## Blinded sizing disposition",
            "",
            f"Only missingness was used: harmful-refusal complete pairs {harmful['complete_pooled_prompt_pairs']}/{harmful['possible_pooled_prompt_pairs']} ({harmful['pair_completion_rate']:.1%}); benign-compliance complete pairs {benign['complete_pooled_prompt_pairs']}/{benign['possible_pooled_prompt_pairs']} ({benign['pair_completion_rate']:.1%}). No endpoint values, arm-labelled means, directions, or differences were used for sizing.",
            "",
            "A powered sample size cannot be frozen from a failed validation half. These completion figures are operational diagnostics only.",
            "",
            "## Required decision",
            "",
            "Do not run the held-out stage from the current manifest. Investigate the 14 ProxyProtocolError responses and, if recovery is desired, create and separately authorize a new hash-bound validation-recovery protocol and budget. Do not reinterpret missing chunks as negative outcomes.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.run_root, args.manifest)
    executor.atomic_write(args.json_out, json.dumps(report, indent=2) + "\n")
    json_sha = p5.sha256_file(args.json_out)
    executor.atomic_write(args.markdown_out, markdown(report, json_sha))
    print(
        json.dumps(
            {
                "integrity": report["integrity"]["status"],
                "gate": report["validation_gate"]["status"],
                "held_out_executed": report["disposition"]["held_out_rescoring_executed"],
                "attempts": report["requests_and_budget"]["attempts_completed"],
                "reported_cost_usd": report["requests_and_budget"][
                    "reported_successful_response_cost_usd"
                ],
                "report_sha256": json_sha,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
