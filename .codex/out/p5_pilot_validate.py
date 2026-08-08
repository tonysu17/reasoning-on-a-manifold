#!/usr/bin/env python3
"""Blinded reliability and integrity validation for the P5 pipeline pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p5_preflight as p5  # noqa: E402
import p5_pilot_executor as executor  # noqa: E402
import p5_pilot_scorer as scorer  # noqa: E402


def masked_roles(manifest: dict[str, Any]) -> dict[str, str]:
    roles = [row["role"] for row in manifest["checkpoint_roles"]]
    ranked = sorted(
        roles,
        key=lambda role: hashlib.sha256(
            f"p5-pilot-blind|{manifest['authorisation_sha256']}|{role}".encode()
        ).hexdigest(),
    )
    return {role: f"arm_{chr(65 + index)}" for index, role in enumerate(ranked)}


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(left == right for left, right in pairs) / n
    left_true = sum(left for left, _ in pairs) / n
    right_true = sum(right for _, right in pairs) / n
    expected = left_true * right_true + (1 - left_true) * (1 - right_true)
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def agreement_record(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    table = Counter(pairs)
    n = len(pairs)
    return {
        "n_complete_pairs": n,
        "agreement": None if not n else sum(a == b for a, b in pairs) / n,
        "cohen_kappa": cohen_kappa(pairs),
        "contingency": {
            "primary_false_double_false": table[(False, False)],
            "primary_false_double_true": table[(False, True)],
            "primary_true_double_false": table[(True, False)],
            "primary_true_double_true": table[(True, True)],
        },
    }


def validate(run_root: Path) -> dict[str, Any]:
    run_root = executor.assert_out_path(run_root)
    manifest = executor.load_manifest(run_root)
    scoring_manifest = scorer.load_scoring_manifest(run_root)
    role_mask = masked_roles(manifest)
    generations = executor.generation_rows(run_root / "generations.jsonl")
    generation_errors = []
    keys = [p5.generation_key(row) for row in generations]
    if len(keys) != len(set(keys)):
        generation_errors.append("duplicate generation keys")
    expected_keys = {
        (manifest["run_id"], role, prompt["prompt_id"])
        for role in role_mask
        for prompt in manifest["prompts"]
    }
    unexpected = sorted(set(keys) - expected_keys)
    if unexpected:
        generation_errors.append(f"unexpected generation keys: {unexpected[:5]}")

    generation_by_arm: dict[str, Any] = {}
    for role, mask in role_mask.items():
        rows = [row for row in generations if row["checkpoint_role"] == role]
        successes = [row for row in rows if row["status"] == "success"]
        metrics = [
            p5.annotation_free_metrics(
                row["text"], n_tokens=row["n_tokens"], stop_reason=row["stop_reason"]
            )
            for row in successes
        ]
        generation_by_arm[mask] = {
            "n_rows": len(rows),
            "n_success": len(successes),
            "n_terminal_error": len(rows) - len(successes),
            "n_empty": sum(metric["empty"] for metric in metrics),
            "n_truncated": sum(metric["truncated"] is True for metric in metrics),
            "n_rep4_collapse": sum(metric["collapse_rep4_gt_0_8"] is True for metric in metrics),
            "truncation_rate_success": None
            if not metrics
            else sum(metric["truncated"] is True for metric in metrics) / len(metrics),
        }

    behaviour = scorer.jsonl_rows(run_root / "behaviour_annotations.jsonl")
    safety = scorer.jsonl_rows(run_root / "safety_scores.jsonl")
    journal = scorer.jsonl_rows(run_root / "scoring_call_journal.jsonl")
    reserved = [row for row in journal if row.get("event") == "reserved"]
    completed_calls = [row for row in journal if row.get("event") == "completed"]
    scoring_errors = []
    if len(reserved) > scoring_manifest["proxy_request_limit"]:
        scoring_errors.append("proxy request ceiling exceeded")

    generation_index = {
        (row["checkpoint_role"], row["prompt_id"]): row for row in generations
    }
    lineage_mismatches = []
    for row in behaviour + safety:
        generation = generation_index.get((row["checkpoint_role"], row["prompt_id"]))
        if generation is None or row["generation_record_sha256"] != p5.sha256_json(generation):
            lineage_mismatches.append((row["checkpoint_role"], row["prompt_id"], row.get("annotator_id") or row.get("scorer_id")))
    if lineage_mismatches:
        scoring_errors.append(f"generation lineage mismatches: {lineage_mismatches[:5]}")

    primary_behaviour = [row for row in behaviour if str(row.get("annotator_id", "")).startswith("primary:")]
    double_behaviour = [row for row in behaviour if str(row.get("annotator_id", "")).startswith("double:")]
    primary_safety = [row for row in safety if str(row.get("scorer_id", "")).startswith("primary:")]
    double_safety = [row for row in safety if str(row.get("scorer_id", "")).startswith("double:")]

    primary_safety_index = {
        (row["checkpoint_role"], row["prompt_id"]): row
        for row in primary_safety
        if row["status"] == "success"
    }
    double_safety_index = {
        (row["checkpoint_role"], row["prompt_id"]): row
        for row in double_safety
        if row["status"] == "success"
    }
    agreements = {}
    for endpoint in ("harmful_refusal", "benign_compliance"):
        pairs = []
        for key in sorted(set(primary_safety_index) & set(double_safety_index)):
            left = primary_safety_index[key]["endpoints"].get(endpoint)
            right = double_safety_index[key]["endpoints"].get(endpoint)
            if isinstance(left, bool) and isinstance(right, bool):
                pairs.append((left, right))
        agreements[endpoint] = agreement_record(pairs)

    generation_complete = len(generations) == 176
    scoring_complete = len(reserved) == 212 and len(completed_calls) == 212
    primary_rows = len(primary_behaviour) + len(primary_safety)
    double_rows = len(double_behaviour) + len(double_safety)
    primary_success = sum(row["status"] == "success" for row in primary_behaviour + primary_safety)
    double_success = sum(row["status"] == "success" for row in double_behaviour + double_safety)
    report = {
        "schema_version": "p5-pilot-validation-1",
        "run_id": manifest["run_id"],
        "pilot_only_not_scientific_result": True,
        "role_labels_blinded": True,
        "generation": {
            "complete": generation_complete,
            "n_rows": len(generations),
            "n_expected": 176,
            "n_missing_keys": len(expected_keys - set(keys)),
            "errors": generation_errors,
            "by_masked_arm": generation_by_arm,
        },
        "scoring": {
            "complete": scoring_complete,
            "proxy_requests_reserved": len(reserved),
            "proxy_requests_completed": len(completed_calls),
            "request_limit": scoring_manifest["proxy_request_limit"],
            "primary_rows": primary_rows,
            "primary_parse_rate": None if not primary_rows else primary_success / primary_rows,
            "double_rows": double_rows,
            "double_parse_rate": None if not double_rows else double_success / double_rows,
            "errors": scoring_errors,
            "safety_agreement": agreements,
        },
        "status": "complete" if generation_complete and scoring_complete and not generation_errors and not scoring_errors else "in_progress",
    }
    return report


def markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# P5 pilot validation",
            "",
            f"**Status:** `{report['status']}`  ",
            "**Use:** pipeline pilot only; not a scientific result  ",
            "**Checkpoint labels:** blinded in this report",
            "",
            f"- Generations: {report['generation']['n_rows']}/{report['generation']['n_expected']}",
            f"- Proxy calls: {report['scoring']['proxy_requests_completed']}/{report['scoring']['request_limit']}",
            f"- Primary parsed rows: {report['scoring']['primary_rows']}",
            f"- Double-scored rows: {report['scoring']['double_rows']}",
            "",
            "The report intentionally omits arm-labelled behavioural differences and cannot be cited as a thesis result.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.run_root)
    executor.atomic_write(args.json_out, json.dumps(report, indent=2) + "\n")
    executor.atomic_write(args.markdown_out, markdown(report))
    print(json.dumps({"status": report["status"], "generations": report["generation"]["n_rows"], "proxy_calls": report["scoring"]["proxy_requests_completed"]}, indent=2))


if __name__ == "__main__":
    main()
