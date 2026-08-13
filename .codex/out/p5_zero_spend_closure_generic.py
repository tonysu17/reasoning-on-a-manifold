#!/usr/bin/env python3
"""Zero-spend P5 safety disposition, generic QA/sizing, and powered skeleton.

This module is offline-only: it imports no HTTP client and contains no model,
proxy, pod, or generation execution path.  It preserves arm blinding for pilot
distributions/sizing by using pooled endpoint distributions and unsigned
within-task squared contrast distances.  The independent unit is always the
task/prompt; checkpoint rows are nested repeated observations.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p5_preflight as p5  # noqa: E402
import p5_pilot_executor as executor  # noqa: E402


SCHEMA_VERSION = "p5-zero-spend-closure-generic-1"
TARGET_ENDPOINTS = (
    "backtracking",
    "uncertainty-estimation",
    "example-testing",
    "adding-knowledge",
)
CONTRASTS = {
    "contrast_family_A": ("base_r1", "public_star1"),
    "contrast_family_B": ("owned_fullft_control_s42", "owned_fullft_safety_s42"),
}
EFFECT_GRID = (0.02, 0.03, 0.05, 0.075, 0.10)
POWER_GRID = (0.80, 0.90)
ALPHA_SPECS = {
    "nominal_two_sided": 0.05,
    "eight_test_bonferroni_bound": 0.05 / 8,
}
N_CATEGORIES = 10
FINAL_TASK_CAP = 100

EXPECTED_HASHES = {
    "generations.jsonl": "2f2e9bc4d71c0e8f7ef71fe067d0e3f12b6fb2af31e83ee692884d70f6bbaf01",
    "behaviour_annotations_v2.jsonl": "f171cdb86306cae6bbcd9208a1ce60c237676e9eed10f1575695083ae23f8113",
    "safety_scores_v2.jsonl": "b2dfc032d1f7acd6e2fa301bc7f538cffaf6404d100cec13ce7a440f320075f3",
    "scoring_call_journal_v2.jsonl": "8b0fbe01cbd176e727e6f5e3a23bd5e7f0cb5b1b3c5a90cae5fc081042bcad96",
    "safety_chunk_scores_v2_2.jsonl": "b43cfa9b56b1f2f8d581bd72fc54ad64625e3f24af41b3da2a6ec1399dcac480",
    "safety_scores_v2_2.jsonl": "7130d0920c0ed42b019380e494a9c8b0970b8ff22d3befe9704f8377d1089995",
    "scoring_call_journal_v2_2.jsonl": "ddcf61b65da220b783cde135dfdec67497cdca474b8479e0cb85c3d3bb1dc7ce",
    "safety_validation_gate_v2_2.json": "5a80f74ce855b49d992a6fe2254548a126788e5aac62ac5c3f34b065a45cc313",
    "safety_v2_2_one_call_shape_diagnostic.json": "57b1ec877a71d3f8ae23ff97ae23763dec7d4def6f06f2f27ec4b3dd68eed759",
    "safety_v2_2_one_call_shape_diagnostic_journal.jsonl": "0def32e7143db9be09781493aaf41a3f88dc52c52622143c646c43a2dd07ca77",
    "safety_v2_2_long_endpoint_shape_diagnostic.json": "87e6e8eb95d51553193d624f928bb29cd56a0106a64efa7ef45a02f76f5dbbc0",
    "safety_v2_2_long_endpoint_shape_diagnostic_journal.jsonl": "d837534f3ee0201054ce032841c9cc6de6ec4851160cd3b979077fba2726921f",
}
V21_MANIFEST_NAME = "P5_PILOT_SCORER_V2_1_PROPOSED_EXECUTABLE_PRIMARY_2026-08-09.json"
V21_MANIFEST_INTERNAL = "af8754ae13658c5b82ea7ec1360e25f6fd39d674549f5512c80c338d0e6be803"
V21_MANIFEST_FILE = "988348a549480cbc63aa1437e493c7ed048ed428028246e9517a16a208e45592"
V22_MANIFEST_NAME = "P5_PILOT_SAFETY_V2_2_PROPOSED_VALIDATION_RESCORING_2026-08-09.json"
V22_MANIFEST_INTERNAL = "29de70c816a2fd6fbf6fcf786cee2543935332870244e24756de1a93be9c669c"
V22_MANIFEST_FILE = "ee801c32f0405d2819ae1aafacc5d78ed3008e8f43572ef86553b7f3f386e9c3"
RECOVERY_PROPOSAL_NAME = "P5_PILOT_SAFETY_V2_2_1_PROPOSED_EMPTY_TEXT_RECOVERY_2026-08-09.json"
RECOVERY_PROPOSAL_FILE = "c4263d449a0f0687bc56729a4158ff3d3c620b0834810e4261b81d298fbe84e9"
SHARED_VANILLA_CONTRACT_NAME = "P5_PHASE2_SHARED_VANILLA_CONTRACT_2026-08-08.md"
SHARED_VANILLA_CONTRACT_FILE = "391461ca66ef60015d4790a4226ffdb839edd9fdc2ba14fa8be2947acf1184a2"
TEST_NAME = "test_p5_zero_spend_closure_generic.py"
CURRENT_CORRECTED_LONG_HANDOFF_SHA256 = (
    "30e7319c13aec105d2cd5d06955130c96a4b137371a3cc41b5ee2fbe1fd7d012"
)
EXECUTION_BOUND_LONG_HANDOFF_SHA256 = (
    "abd617419d4fc0b481576c8366ba9a98429c64577d39d3dd6f56b7805090dfdd"
)
EXPECTED_LONG_HANDOFF_FAIL_CLOSED_TESTS = [
    "test_approved_endpoint_is_exactly_hash_bound_and_not_returned",
    "test_request_and_target_are_identical_to_completed_standard_diagnostic",
    "test_transport_changes_only_endpoint_and_remains_one_attempt",
    "test_long_call_is_one_attempt_sanitized_and_uses_identical_body",
    "test_timeout_stops_without_retry",
    "test_planning_and_authorization_builders_make_no_call",
]


class ClosureError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def internal_hash(document: dict[str, Any]) -> str:
    content = dict(document)
    content.pop("document_sha256", None)
    return p5.sha256_json(content)


def seal(document: dict[str, Any]) -> dict[str, Any]:
    document["document_sha256"] = internal_hash(document)
    return document


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    if not data:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "sample_variance": None,
            "sample_sd": None,
            "minimum": None,
            "q25": None,
            "q75": None,
            "maximum": None,
        }
    variance = statistics.variance(data) if len(data) > 1 else None
    return {
        "n": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "sample_variance": variance,
        "sample_sd": None if variance is None else math.sqrt(variance),
        "minimum": min(data),
        "q25": quantile(data, 0.25),
        "q75": quantile(data, 0.75),
        "maximum": max(data),
    }


def ceil_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def conservative_required_tasks(
    second_moment: float,
    delta: float,
    power: float,
    alpha: float,
    pair_completion_rate: float,
) -> dict[str, Any]:
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    complete = max(2, math.ceil((z * z) * second_moment / (delta * delta)))
    attempted = math.ceil(complete / pair_completion_rate) if pair_completion_rate > 0 else None
    balanced = None if attempted is None else ceil_multiple(attempted, N_CATEGORIES)
    return {
        "absolute_fraction_effect": delta,
        "power": power,
        "two_sided_alpha": alpha,
        "unsigned_paired_difference_second_moment": second_moment,
        "pair_completion_rate": pair_completion_rate,
        "required_complete_tasks": complete,
        "required_attempted_tasks": attempted,
        "required_category_balanced_tasks": balanced,
        "fits_current_100_task_manifest": balanced is not None and balanced <= FINAL_TASK_CAP,
    }


def verify_sources(run_root: Path) -> dict[str, Any]:
    run_root = executor.assert_out_path(run_root)
    sources: dict[str, Any] = {}
    for name, expected in EXPECTED_HASHES.items():
        path = run_root / name
        actual = p5.sha256_file(path)
        if actual != expected:
            raise ClosureError(f"source hash mismatch: {name}")
        sources[name] = {"path": str(path.resolve()), "sha256": actual}
    fixed = {
        V21_MANIFEST_NAME: V21_MANIFEST_FILE,
        V22_MANIFEST_NAME: V22_MANIFEST_FILE,
        RECOVERY_PROPOSAL_NAME: RECOVERY_PROPOSAL_FILE,
        SHARED_VANILLA_CONTRACT_NAME: SHARED_VANILLA_CONTRACT_FILE,
    }
    for name, expected in fixed.items():
        path = HERE / name
        actual = p5.sha256_file(path)
        if actual != expected:
            raise ClosureError(f"fixed document hash mismatch: {name}")
        sources[name] = {"path": str(path.resolve()), "sha256": actual}
    corrected_handoff = HERE / "CLAUDE_LONG_REQUEST_ENDPOINT_HANDOFF_2026-08-09.md"
    if p5.sha256_file(corrected_handoff) != CURRENT_CORRECTED_LONG_HANDOFF_SHA256:
        raise ClosureError("corrected long-endpoint handoff hash mismatch")
    sources["corrected_long_endpoint_handoff"] = {
        "path": str(corrected_handoff.resolve()),
        "sha256": CURRENT_CORRECTED_LONG_HANDOFF_SHA256,
        "execution_bound_precomparison_sha256": EXECUTION_BOUND_LONG_HANDOFF_SHA256,
    }
    sources["closure_builder"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": p5.sha256_file(Path(__file__).resolve()),
    }
    sources["closure_tests"] = {
        "path": str((HERE / TEST_NAME).resolve()),
        "sha256": p5.sha256_file(HERE / TEST_NAME),
    }
    return sources


def build_safety_disposition(run_root: Path) -> dict[str, Any]:
    sources = verify_sources(run_root)
    v21_safety = jsonl(run_root / "safety_scores_v2.jsonl")
    v22_chunks = jsonl(run_root / "safety_chunk_scores_v2_2.jsonl")
    v22_assignments = jsonl(run_root / "safety_scores_v2_2.jsonl")
    gate = json.loads((run_root / "safety_validation_gate_v2_2.json").read_text())
    standard_diag = json.loads((run_root / "safety_v2_2_one_call_shape_diagnostic.json").read_text())
    long_diag = json.loads((run_root / "safety_v2_2_long_endpoint_shape_diagnostic.json").read_text())
    held_out_rows = [row for row in v22_assignments if row.get("stage") == "rescore"]
    recovery_outputs = [
        run_root / "safety_chunk_recovery_v2_2_1.jsonl",
        run_root / "scoring_call_journal_v2_2_1.jsonl",
        run_root / "safety_chunk_scores_merged_v2_2_1.jsonl",
        run_root / "safety_scores_merged_v2_2_1.jsonl",
        run_root / "safety_validation_gate_merged_v2_2_1.json",
    ]
    if held_out_rows or any(path.exists() for path in recovery_outputs):
        raise ClosureError("held-out or recovery output exists; cannot assert closed/unexecuted")
    v21_errors = Counter(row.get("error_type") for row in v21_safety if row.get("status") != "success")
    v22_errors = Counter(
        row.get("parser_error_type") for row in v22_chunks if row.get("network_status") != "success"
    )
    assignment_status = Counter(row["aggregation_status"] for row in v22_assignments)
    document: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}-safety-disposition",
        "status": "immutable_closed_no_further_sonnet_safety_scoring",
        "created_at_utc": utc_now(),
        "proxy_model_or_pod_calls_made_by_closure": 0,
        "scope": "P5 Sonnet safety scoring path only; generic pilot closure is separate",
        "sources": sources,
        "v2_1": {
            "evidence_status": "current resource record of pipeline failure; not a scientific endpoint result",
            "authorized_manifest_internal_sha256": V21_MANIFEST_INTERNAL,
            "authorized_manifest_file_sha256": V21_MANIFEST_FILE,
            "planned_safety_assignments": 96,
            "terminal_safety_assignments": len(v21_safety),
            "parsed_success": sum(row.get("status") == "success" for row in v21_safety),
            "unresolved": sum(row.get("status") != "success" for row in v21_safety),
            "error_classes": dict(sorted(v21_errors.items(), key=lambda item: str(item[0]))),
            "disposition": "pipeline reliability gate failed; unresolved safety rows are not negative labels",
        },
        "v2_2_validation": {
            "evidence_status": "current resource record of failed disjoint pipeline validation",
            "authorized_manifest_internal_sha256": V22_MANIFEST_INTERNAL,
            "authorized_manifest_file_sha256": V22_MANIFEST_FILE,
            "validation_assignments": len(v22_assignments),
            "validation_chunks": len(v22_chunks),
            "terminal_chunk_persistence": gate["atomic_terminal_chunk_persistence_rate"],
            "network_success_chunks": gate["successful_response_chunks"],
            "network_error_chunks": len(v22_chunks) - gate["successful_response_chunks"],
            "network_success_rate": gate["network_success_rate"],
            "required_network_success_rate": 0.98,
            "parser_evidence_complete_chunks": gate["complete_parser_and_evidence_chunks"],
            "parser_evidence_rate_among_network_success": gate[
                "complete_parser_and_evidence_rate_among_successful_responses"
            ],
            "network_error_classes": dict(sorted(v22_errors.items(), key=lambda item: str(item[0]))),
            "assignment_aggregation_status": dict(sorted(assignment_status.items())),
            "gate_status": gate["status"],
            "interpretation": (
                "deterministic parser/evidence validation succeeded for every usable network response, "
                "but the predeclared proxy/network gate failed"
            ),
        },
        "diagnostics": {
            "standard_endpoint": {
                "attempts": standard_diag["network_attempts"],
                "outcome": standard_diag["outcome"],
                "http_status_class": standard_diag["telemetry"]["http_status_class"],
                "content_container_type": standard_diag["telemetry"]["content_container_type"],
                "content_container_count": standard_diag["telemetry"]["content_container_count"],
                "filter_marker_present": standard_diag["telemetry"]["signal_field_presence"]["filter_marker_present"],
                "observed_cost_usd": standard_diag["telemetry"]["usage_cost_usd"],
                "conclusion": "successful billed proxy wrapper prospectively reproduced an empty content array without an explicit filter marker",
            },
            "long_endpoint": {
                "attempts": long_diag["network_attempts"],
                "outcome": long_diag["outcome"],
                "http_status_class": long_diag["telemetry"]["http_status_class"],
                "top_level_keys": long_diag["telemetry"]["top_level_json_keys"],
                "accounting_parseable": long_diag["telemetry"]["accounting_parseable"],
                "observed_cost_usd": long_diag["telemetry"]["usage_cost_usd"],
                "conclusion": "returned an incompatible choices-style envelope; underlying text and exact cost were not retained or recovered",
            },
            "diagnostic_outputs_enter_annotations_or_gates": False,
        },
        "postexecution_handoff_integrity": {
            "execution_bound_precomparison_handoff_sha256": EXECUTION_BOUND_LONG_HANDOFF_SHA256,
            "current_corrected_handoff_sha256": CURRENT_CORRECTED_LONG_HANDOFF_SHA256,
            "reason_for_change": (
                "post-execution correction records that the long endpoint returned an incompatible "
                "choices-style envelope and must not be treated as a drop-in proxy replacement"
            ),
            "executed_runner_manifest_output_and_journal_mutated": False,
            "historical_tests_fail_closed_as_designed": True,
            "affected_historical_tests": EXPECTED_LONG_HANDOFF_FAIL_CLOSED_TESTS,
            "disposition": (
                "preserve both the corrected handoff and every execution-bound artifact; do not "
                "restore the stale handoff or edit/recreate the frozen runner, manifest, or tests"
            ),
        },
        "closure": {
            "v2_2_1_recovery_status": "superseded_unexecuted_closed",
            "held_out_status": "not_executed_closed",
            "same_sonnet_repeats_status": "not_authorized_not_executed",
            "long_endpoint_recovery_status": "prohibited_by_schema_and_accounting_incompatibility",
            "missingness_rule": "all failed/missing chunks and endpoints remain missing or unresolved; no zero, negative, or refusal imputation",
            "scientific_use": "none; safety pipeline outputs do not support arm comparisons",
            "reopening_rule": (
                "requires a new versioned protocol, documented response envelope and accounting, "
                "disjoint validation, hard spend guards, and new owner authorization"
            ),
            "owner_decision_required": False,
        },
    }
    return seal(document)


def build_generic_report(run_root: Path) -> dict[str, Any]:
    sources = verify_sources(run_root)
    annotations = jsonl(run_root / "behaviour_annotations_v2.jsonl")
    generations = [row for row in jsonl(run_root / "generations.jsonl") if row["stratum"] == "generic"]
    run_manifest = executor.load_manifest(run_root)
    prompt_index = {row["prompt_id"]: row for row in run_manifest["prompts"] if row["stratum"] == "generic"}
    generation_index = {(row["checkpoint_role"], row["prompt_id"]): row for row in generations}
    if len(annotations) != 80 or len(generations) != 80 or len(prompt_index) != 20:
        raise ClosureError("generic pilot cardinality differs from 20 tasks x 4 checkpoints")

    lineage_errors: list[str] = []
    reassembly_errors: list[str] = []
    values: dict[str, dict[tuple[str, str], float]] = {endpoint: {} for endpoint in TARGET_ENDPOINTS}
    for row in annotations:
        generation = generation_index[(row["checkpoint_role"], row["prompt_id"])]
        if row["generation_record_sha256"] != p5.sha256_json(generation):
            lineage_errors.append(row["logical_assignment_id"])
        if row["status"] == "success":
            if "".join(sentence["text"] for sentence in row["sentences"]) != generation["text"]:
                reassembly_errors.append(row["logical_assignment_id"])
            if row["n_chunks_scored"] != row["n_chunks_expected"]:
                reassembly_errors.append(row["logical_assignment_id"])
            for endpoint in TARGET_ENDPOINTS:
                endpoint_value = row["endpoints"].get(endpoint)
                if not isinstance(endpoint_value, (int, float)) or isinstance(endpoint_value, bool):
                    raise ClosureError("successful generic endpoint is non-numeric")
                values[endpoint][(row["checkpoint_role"], row["prompt_id"])] = float(endpoint_value)
    if lineage_errors or reassembly_errors:
        raise ClosureError("generic lineage/source integrity failure")

    success = [row for row in annotations if row["status"] == "success"]
    unresolved = [row for row in annotations if row["status"] != "success"]
    task_ids = sorted(prompt_index)
    task_complete_counts = Counter(
        sum((role, prompt_id) in values[TARGET_ENDPOINTS[0]] for role in set(row["checkpoint_role"] for row in annotations))
        for prompt_id in task_ids
    )
    role_completion = sorted(
        Counter(row["checkpoint_role"] for row in success).values()
    )
    unresolved_by_category = Counter(prompt_index[row["prompt_id"]]["category"] for row in unresolved)
    unresolved_errors = Counter(row.get("error_type") for row in unresolved)

    endpoint_distributions: dict[str, Any] = {}
    unsigned_pairing: dict[str, Any] = {}
    curve_cells: list[dict[str, Any]] = []
    for endpoint in TARGET_ENDPOINTS:
        endpoint_values = values[endpoint]
        pooled = list(endpoint_values.values())
        task_means: list[float] = []
        within_task_variances: list[float] = []
        for prompt_id in task_ids:
            task_values = [
                value for (role, current_prompt), value in endpoint_values.items()
                if current_prompt == prompt_id
            ]
            task_means.append(statistics.fmean(task_values))
            within_task_variances.append(statistics.variance(task_values) if len(task_values) > 1 else 0.0)
        endpoint_distributions[endpoint] = {
            "pooled_checkpoint_rows_descriptive_only_not_independent": {
                **describe(pooled),
                "mass_at_zero": sum(value == 0 for value in pooled) / len(pooled),
                "mass_at_one": sum(value == 1 for value in pooled) / len(pooled),
            },
            "task_cluster_means": describe(task_means),
            "within_task_checkpoint_variance": describe(within_task_variances),
        }
        unsigned_pairing[endpoint] = {}
        for contrast_id, (left, right) in CONTRASTS.items():
            squared = []
            for prompt_id in task_ids:
                a = endpoint_values.get((left, prompt_id))
                b = endpoint_values.get((right, prompt_id))
                if a is not None and b is not None:
                    squared.append((a - b) ** 2)
            completion = len(squared) / len(task_ids)
            second_moment = statistics.fmean(squared)
            unsigned_pairing[endpoint][contrast_id] = {
                "possible_task_pairs": len(task_ids),
                "complete_task_pairs": len(squared),
                "pair_completion_rate": completion,
                "unsigned_squared_difference_distribution": describe(squared),
                "variance_proxy_definition": (
                    "mean squared within-task paired distance; conservative for signed paired-difference variance; "
                    "no signed or arm-labelled mean was computed or retained"
                ),
            }
            for alpha_id, alpha in ALPHA_SPECS.items():
                for power in POWER_GRID:
                    for delta in EFFECT_GRID:
                        cell = conservative_required_tasks(second_moment, delta, power, alpha, completion)
                        curve_cells.append(
                            {
                                "endpoint": endpoint,
                                "contrast_family": contrast_id,
                                "alpha_specification": alpha_id,
                                **cell,
                            }
                        )

    sizing_ranges: dict[str, Any] = {}
    for alpha_id in ALPHA_SPECS:
        sizing_ranges[alpha_id] = {}
        for power in POWER_GRID:
            sizing_ranges[alpha_id][str(power)] = {}
            for delta in EFFECT_GRID:
                selected = [
                    row["required_category_balanced_tasks"]
                    for row in curve_cells
                    if row["alpha_specification"] == alpha_id
                    and row["power"] == power
                    and row["absolute_fraction_effect"] == delta
                ]
                sizing_ranges[alpha_id][str(power)][str(delta)] = {
                    "minimum_across_four_endpoints_and_two_contrast_families": min(selected),
                    "maximum_across_four_endpoints_and_two_contrast_families": max(selected),
                    "current_final_manifest_task_cap": FINAL_TASK_CAP,
                    "all_cells_fit_current_manifest": max(selected) <= FINAL_TASK_CAP,
                }

    cap_hits = [row for row in generations if row["stop_reason"] == "length"]
    cap_by_task = Counter(row["prompt_id"] for row in cap_hits)
    cap_counts_all_tasks = [cap_by_task.get(prompt_id, 0) for prompt_id in task_ids]
    cap_role_counts = sorted(Counter(row["checkpoint_role"] for row in cap_hits).values())
    token_stats = describe([float(row["n_tokens"]) for row in generations])
    category_counts = Counter(row["category"] for row in prompt_index.values())
    document: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}-generic-pilot-qa-sizing",
        "status": "arm_blinded_pilot_resource_record_not_scientific_result",
        "created_at_utc": utc_now(),
        "proxy_model_or_pod_calls_made": 0,
        "sources": sources,
        "blinding": {
            "arm_labelled_means_computed_or_reported": False,
            "signed_arm_contrasts_computed_or_reported": False,
            "arm_labelled_effect_directions_inspected": False,
            "allowed_inputs": "pooled distributions, task-cluster summaries, unsigned squared paired distances, completion, and truncation",
        },
        "design_grain": {
            "independent_unit": "task/prompt",
            "independent_pilot_tasks": len(task_ids),
            "checkpoint_rows_are": "four repeated observations nested within each task; never independent N",
            "planned_checkpoint_rows": 80,
            "categories": len(category_counts),
            "tasks_per_category": dict(sorted(category_counts.items())),
            "predeclared_contrast_families": 2,
            "contrast_identity_not_used_to_select_effect_size_or N": True,
        },
        "annotation_qa": {
            "planned_rows": 80,
            "parsed_rows": len(success),
            "unresolved_rows": len(unresolved),
            "row_parse_rate": len(success) / 80,
            "frozen_pipeline_gate": 0.98,
            "gate_pass": len(success) / 80 >= 0.98,
            "task_cluster_complete_checkpoint_count_distribution": {
                str(key): value for key, value in sorted(task_complete_counts.items())
            },
            "anonymized_checkpoint_completion_counts_sorted": role_completion,
            "unresolved_by_task": sorted(row["prompt_id"] for row in unresolved),
            "unresolved_by_category": dict(sorted(unresolved_by_category.items())),
            "unresolved_error_classes": dict(sorted(unresolved_errors.items())),
            "generation_lineage_errors": len(lineage_errors),
            "successful_source_reassembly_errors": len(reassembly_errors),
            "missingness_rule": "no imputation; endpoints missing for unresolved rows; each contrast uses complete task pairs and reports its denominator",
            "reliability_status": (
                "primary-only Sonnet annotation; deterministic lineage/reassembly/endpoint validation passed, "
                "but no independent annotator and therefore no inter-annotator robustness claim"
            ),
        },
        "cap_and_truncation": {
            "generation_rows": len(generations),
            "length_cap_tokens": 4096,
            "length_cap_hits": len(cap_hits),
            "length_cap_rate": len(cap_hits) / len(generations),
            "tasks_with_any_cap_hit": len(cap_by_task),
            "tasks_with_all_four_cap_hits": sum(count == 4 for count in cap_by_task.values()),
            "cap_hits_per_task_distribution": {
                str(count): sum(value == count for value in cap_counts_all_tasks)
                for count in sorted(set(cap_counts_all_tasks))
            },
            "anonymized_checkpoint_cap_hit_counts_sorted": cap_role_counts,
            "generated_token_distribution_pooled_descriptive_only": token_stats,
            "five_percent_trigger_exceeded": len(cap_hits) / len(generations) > 0.05,
            "transfer_limitation": (
                "pilot endpoint variance is specific to the full 4,096-token-capped pilot outputs; "
                "it transfers directly only to an analytically identical 4,096-token prefix estimand"
            ),
        },
        "endpoint_distributions": endpoint_distributions,
        "unsigned_paired_design_inputs": unsigned_pairing,
        "predeclared_sizing": {
            "estimand": "mean task-level paired endpoint difference within each predeclared contrast family",
            "independent_N": "number of tasks with complete pairs, not checkpoint rows",
            "effect_grid_absolute_fraction_points": list(EFFECT_GRID),
            "power_grid": list(POWER_GRID),
            "alpha_specifications": ALPHA_SPECS,
            "primary_hypothesis_count_for_bound": 8,
            "method": (
                "normal-approximation paired design using unsigned E[d^2] as a conservative variance proxy, "
                "inflated by observed contrast-specific pair completion and rounded up to 10-category balance"
            ),
            "curve_cells": curve_cells,
            "category_balanced_task_ranges": sizing_ranges,
            "bounded_worst_case_note": (
                "all endpoints lie in [0,1], so E[d^2] <= 1; if pilot transfer is rejected, "
                "use the E[d^2]=1 worst-case formula or run a new blinded pipeline pilot"
            ),
            "not_a_power_freeze": True,
        },
        "limitations": [
            "Only 20 independent tasks, two per category; variance proxies are imprecise.",
            "The 97.5% row parse rate misses the frozen 98% gate by one row.",
            "One contrast family has 18/20 complete task pairs; the other has 20/20.",
            "42.5% of generic checkpoint rows hit the 4,096-token cap, so full-completion estimands are not supported by this pilot.",
            "Endpoint rows share tasks and cannot be treated as 78 independent observations.",
            "Same-Sonnet repeats would not establish inter-annotator reliability and were not used.",
            "No arm-labelled pilot effects or directions were inspected; this report contains no scientific finding.",
        ],
        "qa_assessment": {
            "level": "share_with_noted_caveats_for_design_only",
            "generic_powered_execution_ready": False,
            "blocking_items": [
                "choose and hash the primary analytic prefix",
                "bind the Phase-2 shared-vanilla manifest and row hashes",
                "freeze the smallest effect and power from the predeclared grid",
                "resolve whether 100 tasks suffice or expand the final task manifest",
                "prospectively validate generic annotation at >=98% parse with exact response/accounting contract",
            ],
        },
    }
    return seal(document)


def build_powered_skeleton(generic_report: dict[str, Any]) -> dict[str, Any]:
    if generic_report.get("document_sha256") != internal_hash(generic_report):
        raise ClosureError("generic QA report is not internally sealed")
    document: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}-powered-generic-protocol-skeleton",
        "status": "draft_non_executable_owner_decisions_required",
        "created_at_utc": utc_now(),
        "execution_authorized": False,
        "proxy_model_or_pod_calls_authorized": 0,
        "source_generic_qa": {
            "internal_sha256": generic_report["document_sha256"],
            "generations_sha256": generic_report["sources"]["generations.jsonl"]["sha256"],
            "annotations_sha256": generic_report["sources"]["behaviour_annotations_v2.jsonl"]["sha256"],
        },
        "phase2_shared_vanilla_condition": {
            "contract_path": str((HERE / SHARED_VANILLA_CONTRACT_NAME).resolve()),
            "contract_sha256": SHARED_VANILLA_CONTRACT_FILE,
            "required_phase2_manifest_path": None,
            "required_phase2_manifest_file_sha256": None,
            "required_phase2_internal_ids_sha256": None,
            "required_canonical_artifact_sha256": None,
            "expected_tasks": 100,
            "expected_shared_roles": ["base_r1", "public_star1", "deepscaler"],
            "expected_terminal_rows": 300,
            "read_only_no_regeneration": True,
            "pre_generation_hash_gate_required": True,
        },
        "checkpoint_structure": {
            "independent_unit": "task/prompt",
            "repeated_observations_nested_within_task": True,
            "candidate_roles": [
                "Phase-2 base R1 shared vanilla",
                "Phase-2 public STAR1 shared vanilla",
                "Phase-2 DeepScaleR shared vanilla",
                "P5-owned matched full-FT control seed 42",
                "P5-owned full-FT safety seed 42",
            ],
            "primary_contrast_families": [
                "Phase-2 base R1 versus public STAR1",
                "P5-owned matched control versus safety full-FT",
            ],
            "deepscaler_status": "secondary observational unless explicitly added to primary multiplicity before freeze",
            "owned_role_generation_authorized": False,
            "owned_roles_must_match_phase2_task_bytes_tokenizer_template_and_decoding": True,
        },
        "analytic_prefix_decision": {
            "status": "owner_decision_required_before_any_scoring",
            "primary_prefix_generated_tokens": None,
            "recommended_choice": 4096,
            "recommendation_reason": (
                "matches the pilot estimand and its blinded variance inputs; all roles may preserve longer raw generations, "
                "but primary behavioural annotation would stop at the first 4,096 generated token IDs or earlier EOS"
            ),
            "required_definition": (
                "prefix over canonical generated token IDs before EOS under the common base-tokenizer alias; "
                "decode the prefix once, retain byte/token hashes, and form deterministic source units only within that prefix"
            ),
            "if_6144_or_8192_chosen": (
                "current pilot variance does not directly transfer; run a new arm-blinded pipeline/variance pilot or use bounded worst-case sizing"
            ),
            "full_completion_analysis": "secondary sensitivity only and separately labelled for cap/EOS support",
            "no_parser_side_clipping_before_raw_artifact_hash": True,
        },
        "primary_endpoints": {
            "names": list(TARGET_ENDPOINTS),
            "estimand": "fraction of deterministic source sentence/clause units within the frozen analytic prefix assigned each behaviour label",
            "denominator": "source units in the frozen prefix for that task-checkpoint row",
            "non_poolability": "v2.1 deterministic source-unit denominator is not pooled with superseded v1 model-generated segmentation",
        },
        "analysis": {
            "primary_estimator": "mean task-level paired difference separately within each predeclared contrast family",
            "standard_error": "task-clustered paired bootstrap, category-stratified; checkpoint rows never resampled independently",
            "bootstrap_draws": 10000,
            "bootstrap_seed": 20260808,
            "categories": 10,
            "multiple_testing": "Holm familywise correction across 4 endpoints x 2 primary contrasts (8 tests)",
            "missingness": (
                "complete pairs per endpoint/contrast; no imputation; report planned tasks, complete pairs, "
                "task/checkpoint missingness, and reasons"
            ),
            "truncation": "report EOS/cap/prefix support by task and checkpoint; cap-hit is observed truncation, not failure",
            "arm_labelled_pilot_effects_used_for_design": False,
        },
        "sample_size_freeze": {
            "selected_absolute_effect": None,
            "selected_power": None,
            "selected_alpha_strategy": None,
            "selected_tasks": None,
            "maximum_current_phase2_tasks": FINAL_TASK_CAP,
            "must_be_multiple_of_categories": N_CATEGORIES,
            "source_grid": generic_report["predeclared_sizing"]["category_balanced_task_ranges"],
            "freeze_rule": (
                "select delta/power/alpha without arm-labelled pilot results; choose the maximum required N across "
                "all four endpoints and both primary contrasts; if >100, expand prospectively or declare smaller effects underpowered"
            ),
        },
        "generic_annotation_validation": {
            "scorer_version_sha256": None,
            "prompt_template_sha256": None,
            "response_envelope_schema_sha256": None,
            "disjoint_validation_manifest_sha256": None,
            "minimum_row_parse_rate": 0.98,
            "minimum_atomic_terminal_persistence": 1.0,
            "independent_annotator_agreement_claim": False,
            "same_sonnet_repeats": "not a reliability measure and absent unless separately authorized for stability only",
            "missing_stays_missing": True,
        },
        "required_hashes_before_execution": [
            "Phase-2 canonical shared-vanilla manifest/file/internal IDs and every admitted row/shard",
            "final ordered 100-task manifest and prompt-text/input-ID hashes",
            "analytic-prefix definition and per-row prefix hashes",
            "owned checkpoint weights/configs and exact matched generation configuration",
            "generic scorer code/tests/prompt/envelope/parser hashes",
            "analysis code, endpoint family, contrast family, bootstrap seed, and multiplicity plan",
            "generation/scoring request counts, calibrated spend ceilings, retry limits, and quota floor",
        ],
        "hard_stops": [
            "Phase-2 shared artefact absent, corrupt, or provenance-incomplete",
            "analytic prefix not selected and hashed",
            "task bytes/input IDs differ across roles",
            "generic disjoint validation below 98% parse or 100% terminal persistence",
            "sample-size choice absent or exceeds available tasks without a prospective manifest expansion",
            "any attempt to treat checkpoint rows as independent tasks or missing endpoints as zero",
        ],
        "owner_decisions": [
            "Choose the primary analytic prefix: recommended 4,096 generated tokens; alternatives require new variance evidence.",
            "Choose the smallest effect from the frozen 0.02/0.03/0.05/0.075/0.10 grid and 80% or 90% power.",
            "Choose nominal versus conservative eight-test alpha for planning; Holm remains the analysis correction.",
            "Confirm whether the 100-task Phase-2 manifest is a hard maximum or may be prospectively expanded.",
            "Confirm whether DeepScaleR is secondary observational or part of the primary contrast family.",
        ],
    }
    return seal(document)


def write_json(path: Path, document: dict[str, Any]) -> str:
    path = executor.assert_out_path(path)
    executor.atomic_write(path, json.dumps(document, indent=2, sort_keys=True) + "\n")
    return p5.sha256_file(path)


def safety_markdown(document: dict[str, Any], file_sha: str) -> str:
    v21 = document["v2_1"]
    v22 = document["v2_2_validation"]
    return f"""# Immutable P5 Sonnet safety-scoring disposition

Status: **closed**. This is a pipeline/resource disposition, not a scientific safety result.

- Internal SHA-256: `{document['document_sha256']}`
- File SHA-256: `{file_sha}`
- Closure made no proxy, model, or pod calls.

## Evidence chain

- v2.1: {v21['parsed_success']}/96 safety assignments parsed; {v21['unresolved']} remain unresolved. The pipeline gate failed.
- v2.2 disjoint validation: {v22['parser_evidence_complete_chunks']}/{v22['network_success_chunks']} usable responses passed deterministic schema/evidence validation, but network success was {v22['network_success_rate']:.1%} versus the frozen 98% gate.
- Standard diagnostic: a billed HTTP-success proxy wrapper prospectively reproduced an empty `content` array without an explicit filter marker.
- Long-endpoint comparison: returned an incompatible `choices` envelope without parseable cost/quota accounting; underlying text was neither retained nor recovered.
- The post-execution handoff was intentionally corrected. Six historical tests now fail closed on the execution-bound old handoff hash; the corrected handoff and all executed artifacts are preserved unchanged.

## Final disposition

- v2.2.1 recovery: superseded, unexecuted, and closed.
- Held-out safety scoring: not executed and closed.
- Same-Sonnet repeats: unauthorized and unexecuted.
- Every failed/missing chunk and endpoint remains missing or unresolved. Nothing is imputed as zero, negative, compliant, or refusing.
- No safety arm comparison or powered safety claim is licensed.

Reopening requires a new versioned protocol, documented envelope/accounting, disjoint validation, hard guards, and new owner authorization.
"""


def generic_markdown(document: dict[str, Any], file_sha: str) -> str:
    qa = document["annotation_qa"]
    cap = document["cap_and_truncation"]
    ranges = document["predeclared_sizing"]["category_balanced_task_ranges"][
        "eight_test_bonferroni_bound"
    ]["0.8"]
    lines = [
        "# Arm-blinded P5 generic pilot QA and clustered sizing",
        "",
        "Status: **design resource only; no arm-labelled means, differences, directions, or scientific findings**.",
        "",
        f"- Internal SHA-256: `{document['document_sha256']}`",
        f"- File SHA-256: `{file_sha}`",
        "- Independent unit: task/prompt (20); checkpoint rows are nested repeated observations.",
        f"- Parsed annotations: {qa['parsed_rows']}/80 ({qa['row_parse_rate']:.1%}); frozen gate: 98% — **not passed**.",
        f"- Length-cap hits: {cap['length_cap_hits']}/80 ({cap['length_cap_rate']:.1%}); the 5% trigger is exceeded.",
        "- Complete task pairs: 18/20 in one anonymous contrast family and 20/20 in the other.",
        "",
        "## Descriptive and sizing interpretation",
        "",
        "Endpoint distributions are reported pooled across checkpoints and as task-cluster means/within-task variance. Pooled row counts are descriptive only; N is never 78. Paired sizing uses unsigned squared within-task distances as a conservative variance proxy, so no signed or arm-labelled pilot effect was computed.",
        "",
        "Conservative eight-test, 80%-power category-balanced task ranges:",
        "",
    ]
    for delta in EFFECT_GRID:
        cell = ranges[str(delta)]
        lines.append(
            f"- absolute effect {delta:.3f}: {cell['minimum_across_four_endpoints_and_two_contrast_families']}–{cell['maximum_across_four_endpoints_and_two_contrast_families']} tasks"
        )
    lines.extend(
        [
            "",
            "These are planning ranges, not a frozen N. The 20-task pilot is small, one contrast loses two complete pairs, and 42.5% of rows hit the 4,096-token cap. The current variance evidence transfers directly only to a 4,096-generated-token analytic-prefix estimand.",
            "",
            "## QA disposition",
            "",
            "Share with noted caveats for design only. Before powered P5: bind the Phase-2 shared vanilla, choose/hash the analytic prefix, choose effect/power/alpha, determine whether 100 tasks suffice, and prospectively validate generic annotation at at least 98% parse with an exact response/accounting contract.",
        ]
    )
    return "\n".join(lines) + "\n"


def skeleton_markdown(document: dict[str, Any], file_sha: str) -> str:
    decisions = "\n".join(f"- {item}" for item in document["owner_decisions"])
    return f"""# Draft powered P5 generic protocol/manifest skeleton

Status: **non-executable**. No generation, scoring, model, proxy, or pod call is authorized.

- Internal SHA-256: `{document['document_sha256']}`
- File SHA-256: `{file_sha}`
- Phase-2 canonical manifest SHA: unset/blocking
- Primary analytic prefix: unset/blocking (recommended: first 4,096 generated token IDs or earlier EOS)
- Selected effect, power, alpha strategy, and task N: unset/blocking

The independent unit is the task. Checkpoint outputs are repeated observations nested within task. Primary inference uses task-paired contrasts, 10-category-stratified bootstrap resampling, 10,000 draws, seed 20260808, complete-pair missingness, and Holm correction across four endpoints by two primary contrasts.

The shared Phase-2 vanilla artefact is read-only and must be bound by manifest, internal-ID, row, and shard hashes. P5 may not regenerate missing shared rows. P5-owned roles require a separate generation manifest and authorization using identical task bytes, tokenizer/template, and decoding.

## Owner decisions

{decisions}

Execution remains blocked until every required hash and validation field in the JSON skeleton is populated and a new exact spend/scope authorization is issued.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--safety-json", type=Path, required=True)
    parser.add_argument("--safety-md", type=Path, required=True)
    parser.add_argument("--generic-json", type=Path, required=True)
    parser.add_argument("--generic-md", type=Path, required=True)
    parser.add_argument("--skeleton-json", type=Path, required=True)
    parser.add_argument("--skeleton-md", type=Path, required=True)
    args = parser.parse_args()

    safety = build_safety_disposition(args.run_root)
    generic = build_generic_report(args.run_root)
    skeleton = build_powered_skeleton(generic)
    safety_file = write_json(args.safety_json, safety)
    generic_file = write_json(args.generic_json, generic)
    skeleton_file = write_json(args.skeleton_json, skeleton)
    executor.atomic_write(executor.assert_out_path(args.safety_md), safety_markdown(safety, safety_file))
    executor.atomic_write(executor.assert_out_path(args.generic_md), generic_markdown(generic, generic_file))
    executor.atomic_write(executor.assert_out_path(args.skeleton_md), skeleton_markdown(skeleton, skeleton_file))
    print(
        json.dumps(
            {
                "status": "offline_complete",
                "proxy_model_or_pod_calls": 0,
                "safety_internal_sha256": safety["document_sha256"],
                "generic_internal_sha256": generic["document_sha256"],
                "skeleton_internal_sha256": skeleton["document_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
