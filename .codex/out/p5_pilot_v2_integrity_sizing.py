#!/usr/bin/env python3
"""Integrity QA and arm-blind sizing diagnostics for the P5 v2.1 pilot."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p5_preflight as p5  # noqa: E402
import p5_pilot_executor as executor  # noqa: E402
import p5_pilot_scorer_v2 as scorer  # noqa: E402


TARGET_BEHAVIOURS = tuple(p5.TARGET_BEHAVIOURS)
CONTRAST_PAIRS = (
    ("base_r1", "public_star1"),
    ("owned_fullft_control_s42", "owned_fullft_safety_s42"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def required_n(variance_proxy: float, delta: float, completion: float) -> dict[str, Any]:
    z = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.80)
    effective = max(1, math.ceil((z * z) * variance_proxy / (delta * delta)))
    attempted = None if completion <= 0 else math.ceil(effective / completion)
    return {
        "alpha_two_sided": 0.05,
        "power": 0.80,
        "absolute_difference": delta,
        "variance_proxy": variance_proxy,
        "required_complete_prompt_pairs": effective,
        "required_attempted_prompts_at_observed_pair_completion": attempted,
    }


def pooled_pairs(
    values: dict[tuple[str, str], float | bool], prompt_ids: list[str]
) -> tuple[list[tuple[float, float]], int]:
    pairs: list[tuple[float, float]] = []
    possible = len(prompt_ids) * len(CONTRAST_PAIRS)
    for prompt_id in prompt_ids:
        for left, right in CONTRAST_PAIRS:
            a = values.get((left, prompt_id))
            b = values.get((right, prompt_id))
            if isinstance(a, (bool, int, float)) and not isinstance(a, str) and isinstance(
                b, (bool, int, float)
            ) and not isinstance(b, str):
                pairs.append((float(a), float(b)))
    return pairs, possible


def build_report(run_root: Path, manifest_path: Path) -> dict[str, Any]:
    run_root = executor.assert_out_path(run_root)
    manifest_path = executor.assert_out_path(manifest_path)
    manifest = scorer.load_v2_manifest(manifest_path)
    run_manifest = executor.load_manifest(run_root)
    generations = executor.generation_rows(run_root / "generations.jsonl")
    behaviour = jsonl(run_root / scorer.BEHAVIOUR_OUTPUT)
    safety = jsonl(run_root / scorer.SAFETY_OUTPUT)
    journal = jsonl(run_root / scorer.JOURNAL_OUTPUT)
    rows = behaviour + safety

    expected_ids = {row["logical_assignment_id"] for row in manifest["logical_assignments"]}
    observed_ids = [row["logical_assignment_id"] for row in rows]
    generation_index = {
        (row["checkpoint_role"], row["prompt_id"]): row for row in generations
    }
    lineage_errors = []
    source_reassembly_errors = []
    for row in rows:
        generation = generation_index.get((row["checkpoint_role"], row["prompt_id"]))
        if generation is None or row["generation_record_sha256"] != p5.sha256_json(generation):
            lineage_errors.append(row["logical_assignment_id"])
        if row["stratum"] == "generic" and row["status"] == "success":
            if "".join(item["text"] for item in row["sentences"]) != generation["text"]:
                source_reassembly_errors.append(row["logical_assignment_id"])
            if row["n_chunks_scored"] != row["n_chunks_expected"]:
                source_reassembly_errors.append(row["logical_assignment_id"])

    reserved = [row for row in journal if row.get("event") == "reserved"]
    completed = [row for row in journal if row.get("event") == "completed"]
    reserved_ids = [row["call_id"] for row in reserved]
    completed_ids = [row["call_id"] for row in completed]
    costs = [
        float(row["usage_cost_usd"])
        for row in completed
        if isinstance(row.get("usage_cost_usd"), (int, float))
    ]
    quotas = [
        float(row["remaining_quota_usd"])
        for row in completed
        if isinstance(row.get("remaining_quota_usd"), (int, float))
    ]

    prompts_by_stratum = {
        stratum: [row["prompt_id"] for row in run_manifest["prompts"] if row["stratum"] == stratum]
        for stratum in ("generic", "harmful", "benign")
    }
    status_by_stratum = {}
    for stratum in ("generic", "harmful", "benign"):
        subset = [row for row in rows if row["stratum"] == stratum]
        status_by_stratum[stratum] = {
            "submitted": len(subset),
            "parsed_success": sum(row["status"] == "success" for row in subset),
            "unresolved": sum(row["status"] != "success" for row in subset),
            "parse_rate": None
            if not subset
            else sum(row["status"] == "success" for row in subset) / len(subset),
        }

    generic_values: dict[str, dict[tuple[str, str], float]] = {
        endpoint: {} for endpoint in TARGET_BEHAVIOURS
    }
    for row in behaviour:
        if row["status"] != "success":
            continue
        for endpoint in TARGET_BEHAVIOURS:
            value = row["endpoints"].get(endpoint)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                generic_values[endpoint][(row["checkpoint_role"], row["prompt_id"])] = float(value)

    generic_sizing = {}
    for endpoint, values in generic_values.items():
        pairs, possible = pooled_pairs(values, prompts_by_stratum["generic"])
        completion = 0.0 if not possible else len(pairs) / possible
        variance_proxy = None if not pairs else sum((a - b) ** 2 for a, b in pairs) / len(pairs)
        generic_sizing[endpoint] = {
            "estimand_for_sizing": "pooled squared paired difference across the two pre-specified contrast families; no direction or arm-labelled mean computed",
            "possible_prompt_pairs": possible,
            "complete_prompt_pairs": len(pairs),
            "pair_completion_rate": completion,
            "paired_difference_second_moment": variance_proxy,
            "curves": []
            if variance_proxy is None
            else [required_n(variance_proxy, delta, completion) for delta in (0.02, 0.05, 0.10)],
        }

    safety_specs = (("harmful", "harmful_refusal"), ("benign", "benign_compliance"))
    safety_sizing = {}
    for stratum, endpoint in safety_specs:
        values: dict[tuple[str, str], bool] = {}
        resolved_values = []
        for row in safety:
            if row["stratum"] != stratum:
                continue
            value = row.get("endpoints", {}).get(endpoint)
            if isinstance(value, bool):
                values[(row["checkpoint_role"], row["prompt_id"])] = value
                resolved_values.append(value)
        pairs, possible = pooled_pairs(values, prompts_by_stratum[stratum])
        completion = 0.0 if not possible else len(pairs) / possible
        discordance = None if not pairs else sum(a != b for a, b in pairs) / len(pairs)
        safety_sizing[endpoint] = {
            "stratum": stratum,
            "resolved_rows": len(resolved_values),
            "unresolved_rows": 48 - len(resolved_values),
            "pooled_resolved_event_rate": None
            if not resolved_values
            else sum(resolved_values) / len(resolved_values),
            "possible_prompt_pairs": possible,
            "complete_prompt_pairs": len(pairs),
            "pair_completion_rate": completion,
            "pooled_pair_discordance": discordance,
            "five_point_margin_status": "proposed; supervisor confirmation still required",
            "observed_discordance_curve": None
            if discordance is None
            else required_n(discordance, 0.05, completion),
            "worst_case_paired_curve": required_n(1.0, 0.05, completion),
        }

    truncation = Counter((row["stratum"], row["stop_reason"]) for row in generations)
    n_length = sum(row["stop_reason"] == "length" for row in generations)
    row_errors = Counter(row.get("error_type") for row in rows if row["status"] != "success")
    request_errors = Counter(
        row.get("error_type") for row in completed if row.get("status") != "success"
    )
    guard = manifest["authorization_guard"]
    integrity_errors = []
    if len(observed_ids) != len(set(observed_ids)):
        integrity_errors.append("duplicate logical assignment IDs")
    if set(observed_ids) != expected_ids:
        integrity_errors.append("observed assignment set differs from proposed manifest")
    if lineage_errors:
        integrity_errors.append("generation lineage mismatches")
    if source_reassembly_errors:
        integrity_errors.append("successful behaviour source reassembly mismatch")
    if set(reserved_ids) != set(completed_ids) or len(reserved_ids) != len(set(reserved_ids)):
        integrity_errors.append("journal reservation/completion mismatch")
    if any(row.get("pass_id") != "primary" for row in rows):
        integrity_errors.append("non-primary output present")
    if len(reserved) > guard["approved_request_ceiling"]:
        integrity_errors.append("request ceiling exceeded")
    if costs and max(costs) > guard["approved_max_cost_per_request_usd"]:
        integrity_errors.append("per-request cost ceiling exceeded")
    if quotas and min(quotas) < guard["quota_stop_floor_usd"]:
        integrity_errors.append("quota floor crossed")

    parse_total = sum(row["status"] == "success" for row in rows) / len(rows)
    powered_ready = (
        parse_total >= 0.98
        and all(status_by_stratum[s]["parse_rate"] >= 0.98 for s in status_by_stratum)
        and all(item["pair_completion_rate"] >= 0.90 for item in safety_sizing.values())
    )
    return {
        "schema_version": "p5-pilot-v2.1-integrity-sizing-1",
        "created_at_utc": utc_now(),
        "run_id": run_manifest["run_id"],
        "pilot_only_not_scientific_result": True,
        "blinding": {
            "arm_labelled_effect_differences_computed": False,
            "arm_labelled_effect_differences_reported": False,
            "sizing_inputs": "pooled base rates, squared paired differences, and discordance across the two pre-specified contrast families",
        },
        "sources": {
            "proposed_manifest_path": str(manifest_path.resolve()),
            "proposed_manifest_internal_sha256": manifest["manifest_sha256"],
            "proposed_manifest_file_sha256": p5.sha256_file(manifest_path),
            "generations_sha256": p5.sha256_file(run_root / "generations.jsonl"),
            "behaviour_annotations_sha256": p5.sha256_file(run_root / scorer.BEHAVIOUR_OUTPUT),
            "safety_scores_sha256": p5.sha256_file(run_root / scorer.SAFETY_OUTPUT),
            "scoring_journal_sha256": p5.sha256_file(run_root / scorer.JOURNAL_OUTPUT),
        },
        "integrity": {
            "status": "pass" if not integrity_errors else "fail",
            "errors": integrity_errors,
            "planned_assignments": 176,
            "observed_assignments": len(rows),
            "unique_assignment_ids": len(set(observed_ids)),
            "generation_lineage_errors": len(lineage_errors),
            "source_reassembly_errors": len(source_reassembly_errors),
            "primary_only": all(row.get("pass_id") == "primary" for row in rows),
            "same_sonnet_repeat_rows": sum(row.get("pass_id") == "repeat" for row in rows),
        },
        "generation": {
            "rows": len(generations),
            "success": sum(row["status"] == "success" for row in generations),
            "length_cap_hits": n_length,
            "length_cap_rate": n_length / len(generations),
            "stop_reason_by_stratum": {
                f"{stratum}:{reason}": count
                for (stratum, reason), count in sorted(truncation.items())
            },
            "five_percent_trigger_exceeded": n_length / len(generations) > 0.05,
            "full_run_cap_action": "freeze 6,144 for every P5-owned full-run arm; do not alter the separately sealed Phase-2 shared vanilla artifact",
        },
        "scoring": {
            "submitted_assignments": len(rows),
            "parsed_success": sum(row["status"] == "success" for row in rows),
            "unresolved": sum(row["status"] != "success" for row in rows),
            "overall_parse_rate": parse_total,
            "required_parse_rate": 0.98,
            "by_stratum": status_by_stratum,
            "row_error_classes": dict(sorted(row_errors.items(), key=lambda item: str(item[0]))),
            "request_error_classes": dict(
                sorted(request_errors.items(), key=lambda item: str(item[0]))
            ),
            "diagnosis": "request transport mostly succeeded, but strict per-chunk safety schema/evidence validation produced high unresolvedness; exact ValueError subcauses are not recoverable because terminal rows retained only error class and response hashes",
        },
        "requests_and_budget": {
            "planned_initial_requests": 490,
            "attempts_reserved": len(reserved),
            "attempts_completed": len(completed),
            "retries": sum(row.get("request_attempt", 1) > 1 for row in reserved),
            "successful_proxy_responses": sum(row.get("status") == "success" for row in completed),
            "non_success_proxy_responses": sum(row.get("status") != "success" for row in completed),
            "reported_cost_usd": sum(costs),
            "responses_without_cost_accounting": len(completed) - len(costs),
            "conservative_authorized_cost_commitment_usd": len(reserved)
            * guard["approved_max_cost_per_request_usd"],
            "maximum_observed_call_cost_usd": max(costs) if costs else None,
            "minimum_observed_remaining_quota_usd": min(quotas) if quotas else None,
            "request_ceiling": guard["approved_request_ceiling"],
            "spend_ceiling_usd": guard["approved_spend_ceiling_usd"],
            "quota_floor_usd": guard["quota_stop_floor_usd"],
            "all_guards_respected": not any(
                "ceiling" in error or "quota" in error for error in integrity_errors
            ),
        },
        "blinded_sizing": {
            "generic": generic_sizing,
            "safety": safety_sizing,
            "warning": "safety sizing curves are diagnostic only because endpoint resolution and pair completion are far below the frozen 98% parse gate; do not freeze a powered N from these curves",
        },
        "pilot_gate": {
            "status": "not_ready_for_powered_size_freeze" if not powered_ready else "ready",
            "scorer_parse_gate_pass": parse_total >= 0.98,
            "non_builder_agreement_gate": "unsatisfiable under Sonnet-only A3; no inter-annotator robustness claim",
            "owner_decision_required": True,
            "required_decision": "approve a versioned safety-scoring repair and a new disjoint pipeline validation/rescoring budget before powered-study sizing; do not promote current safety missingness to negative outcomes",
        },
    }


def markdown(report: dict[str, Any], json_sha256: str) -> str:
    s = report["scoring"]
    b = report["requests_and_budget"]
    g = report["generation"]
    return "\n".join(
        [
            "# P5 v2.1 pilot integrity and blinded sizing",
            "",
            f"**Integrity:** `{report['integrity']['status']}`  ",
            f"**Powered-size gate:** `{report['pilot_gate']['status']}`  ",
            "**Use:** pipeline pilot only; no arm-labelled effect differences computed or reported  ",
            f"**Machine-readable report SHA-256:** `{json_sha256}`",
            "",
            "## Execution integrity",
            "",
            f"- Assignments: {report['integrity']['observed_assignments']}/176, primary only; repeat rows: 0",
            f"- Generation lineage errors: {report['integrity']['generation_lineage_errors']}",
            f"- Successful source reassembly errors: {report['integrity']['source_reassembly_errors']}",
            f"- Requests: {b['attempts_completed']} completed; {b['retries']} retries; {b['non_success_proxy_responses']} non-success responses",
            f"- Reported cost: ${b['reported_cost_usd']:.6f}; conservative authorization commitment: ${b['conservative_authorized_cost_commitment_usd']:.3f}",
            f"- Maximum reported call cost: ${b['maximum_observed_call_cost_usd']:.6f}; minimum remaining quota: ${b['minimum_observed_remaining_quota_usd']:.5f}",
            "",
            "## Pipeline QA",
            "",
            f"- Overall scorer parse: {s['parsed_success']}/{s['submitted_assignments']} ({s['overall_parse_rate']:.1%}); frozen gate: 98%",
            f"- Generic: {s['by_stratum']['generic']['parsed_success']}/80 ({s['by_stratum']['generic']['parse_rate']:.1%})",
            f"- Harmful safety: {s['by_stratum']['harmful']['parsed_success']}/48 ({s['by_stratum']['harmful']['parse_rate']:.1%})",
            f"- Benign safety: {s['by_stratum']['benign']['parsed_success']}/48 ({s['by_stratum']['benign']['parse_rate']:.1%})",
            f"- Length-cap hits: {g['length_cap_hits']}/176 ({g['length_cap_rate']:.1%}); the 5% trigger is exceeded",
            "",
            "Transport mostly succeeded, but strict per-chunk safety schema/evidence validation caused high unresolvedness. Missing rows remain unresolved, never zero. Error-class-only terminal logging prevents a finer retrospective ValueError diagnosis.",
            "",
            "## Blinded sizing disposition",
            "",
            "The JSON report contains pooled squared-difference and discordance curves across the two pre-specified contrast families without arm-labelled means or directions. They are diagnostics only: safety resolution and complete-pair rates are too low to freeze a powered sample size. The proposed five-point margins still require supervisor confirmation.",
            "",
            "## Required decision",
            "",
            "Do not authorize the powered run from this pilot. Approve a versioned safety-scoring repair plus a new disjoint validation/rescoring budget, preserve current unresolved rows as missing, and retain the 6,144-token cap trigger for every P5-owned full-run arm.",
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
                "powered_size_gate": report["pilot_gate"]["status"],
                "assignments": report["integrity"]["observed_assignments"],
                "reported_cost_usd": report["requests_and_budget"]["reported_cost_usd"],
                "report_sha256": json_sha,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
