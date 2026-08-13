#!/usr/bin/env python3
"""Build a hash-bound proposed executable manifest for P5 v2.1 primary scoring.

This builder is zero-call only.  The resulting manifest is executable-capable
under the scorer's existing exact-hash plus ``--authorised`` guard, but the
proposal records that owner hash approval was not present when it was frozen.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p5_preflight as p5  # noqa: E402
import p5_pilot_executor as executor  # noqa: E402
import p5_pilot_scorer_v2 as scorer  # noqa: E402


EXPECTED_PLAN_INTERNAL_SHA256 = "67f47ecd9f0b4da0299d4d0caff74ec6827cb90c8833f5f4430b7d78db44f2bd"
EXPECTED_PLAN_FILE_SHA256 = "97e6ee52d455f5e53b6750348cd768f2527a0b9a1e321ee8e7c42a1aa994d11b"
EXPECTED_CALIBRATION_RESULTS_SHA256 = "1d79ececf4b2548b1bfc2e0d4b4c6a4d1044a220d8168415b872b04ffada3988"
EXPECTED_CALIBRATION_JOURNAL_SHA256 = "23efb6c5cea32ebd021d14038a8510837bad621063d5da5a03afdce3edf91d09"
REQUEST_CEILING = 600
SPEND_CEILING_USD = 15.0
MAX_COST_PER_REQUEST_USD = 0.025
QUOTA_STOP_FLOOR_USD = 5.0
INITIAL_REQUESTS = 490
GLOBAL_RETRY_CAPACITY = REQUEST_CEILING - INITIAL_REQUESTS


class ProposalError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def calibration_summary(results_path: Path, journal_path: Path) -> dict[str, Any]:
    results_path = executor.assert_out_path(results_path)
    journal_path = executor.assert_out_path(journal_path)
    result_sha = p5.sha256_file(results_path)
    journal_sha = p5.sha256_file(journal_path)
    if result_sha != EXPECTED_CALIBRATION_RESULTS_SHA256:
        raise ProposalError("calibration result hash mismatch")
    if journal_sha != EXPECTED_CALIBRATION_JOURNAL_SHA256:
        raise ProposalError("calibration journal hash mismatch")
    results = _jsonl(results_path)
    journal = _jsonl(journal_path)
    if (
        len(results) != 2
        or any(row.get("status") != "success" for row in results)
        or any(row.get("calibration_only_excluded_from_annotations") is not True for row in results)
    ):
        raise ProposalError("calibration results are not two successful excluded calls")
    reserved = [row for row in journal if row.get("event") == "reserved"]
    completed = [row for row in journal if row.get("event") == "completed"]
    if len(reserved) != 2 or len(completed) != 2:
        raise ProposalError("calibration journal does not contain exactly two calls")
    if any(row.get("retry_number") != 0 for row in reserved + completed):
        raise ProposalError("calibration journal contains a retry")
    costs = [float(row["usage_cost_usd"]) for row in results]
    quotas = [float(row["remaining_quota_usd"]) for row in results]
    return {
        "results_path": str(results_path.resolve()),
        "results_sha256": result_sha,
        "journal_path": str(journal_path.resolve()),
        "journal_sha256": journal_sha,
        "n_calls": 2,
        "n_retries": 0,
        "total_observed_cost_usd": sum(costs),
        "maximum_observed_cost_usd": max(costs),
        "minimum_observed_remaining_quota_usd": min(quotas),
        "calibration_only_excluded_from_annotations": True,
    }


def build_proposal(
    plan_path: Path, results_path: Path, journal_path: Path
) -> dict[str, Any]:
    plan_path = executor.assert_out_path(plan_path)
    plan = scorer.load_v2_manifest(plan_path)
    if plan["manifest_sha256"] != EXPECTED_PLAN_INTERNAL_SHA256:
        raise ProposalError("final plan internal hash mismatch")
    if p5.sha256_file(plan_path) != EXPECTED_PLAN_FILE_SHA256:
        raise ProposalError("final plan file hash mismatch")
    accounting = plan["dry_run_accounting"]
    if (
        accounting["generation_rows_snapshot"] != 176
        or accounting["missing_generation_keys_vs_176"] != 0
        or accounting["full_176_primary_only_initial_requests"] != INITIAL_REQUESTS
        or len(plan["logical_assignments"]) != 176
        or len(plan["initial_network_request_plan"]) != INITIAL_REQUESTS
        or any(row["pass_id"] != "primary" for row in plan["logical_assignments"])
        or any(row["pass_id"] != "primary" for row in plan["initial_network_request_plan"])
        or plan["reliability_status"]["repeat_policy"] != "drop"
    ):
        raise ProposalError("final plan is not the exact 176-row/490-request primary-only plan")
    if REQUEST_CEILING * MAX_COST_PER_REQUEST_USD > SPEND_CEILING_USD:
        raise ProposalError("request ceiling times maximum request cost exceeds spend ceiling")
    if INITIAL_REQUESTS > REQUEST_CEILING:
        raise ProposalError("initial request plan exceeds proposed request ceiling")

    calibration = calibration_summary(results_path, journal_path)
    document = copy.deepcopy(plan)
    document.pop("manifest_sha256", None)
    document["schema_version"] = "p5-pilot-scoring-manifest-v2.1-proposed-executable-1"
    # The scorer recognises this capability status, but exact hash plus
    # --authorised remains mandatory.  proposal_status carries the owner state.
    document["status"] = "authorized_for_execution"
    document["proposal_status"] = "proposed_awaiting_owner_exact_hash_approval"
    document["created_at_utc"] = utc_now()
    document["proxy_calls_made_while_building_proposal"] = 0
    document["sources"]["final_primary_plan"] = {
        "path": str(plan_path.resolve()),
        "internal_sha256": EXPECTED_PLAN_INTERNAL_SHA256,
        "file_sha256": EXPECTED_PLAN_FILE_SHA256,
    }
    document["sources"]["calibration"] = calibration
    document["authorization_guard"] = {
        "execution_authorized": True,
        "authorization_semantics": "executable capability only; owner approval is supplied externally by exact manifest hash plus --authorised",
        "proposal_status_at_freeze": "owner_hash_approval_not_yet_received",
        "owner_exact_hash_approval_required": True,
        "approved_request_ceiling": REQUEST_CEILING,
        "approved_spend_ceiling_usd": SPEND_CEILING_USD,
        "approved_max_cost_per_request_usd": MAX_COST_PER_REQUEST_USD,
        "quota_stop_floor_usd": QUOTA_STOP_FLOOR_USD,
        "initial_planned_requests": INITIAL_REQUESTS,
        "global_retry_attempt_capacity": GLOBAL_RETRY_CAPACITY,
        "per_chunk_retry_ceiling": 3,
        "global_request_ceiling_overrides_per_chunk_retry_availability": True,
        "ceiling_product_usd": REQUEST_CEILING * MAX_COST_PER_REQUEST_USD,
        "ceiling_product_within_spend": (
            REQUEST_CEILING * MAX_COST_PER_REQUEST_USD <= SPEND_CEILING_USD
        ),
        "initial_plan_within_request_ceiling": INITIAL_REQUESTS <= REQUEST_CEILING,
    }
    document["execution_scope"] = {
        "pass": "primary",
        "logical_assignments": 176,
        "initial_network_requests": INITIAL_REQUESTS,
        "same_sonnet_repeats": "absent/dropped",
        "optional_repeat_keys_authorized": 0,
        "maximum_total_network_attempts": REQUEST_CEILING,
        "maximum_additional_retry_attempts": GLOBAL_RETRY_CAPACITY,
        "per_chunk_retry_policy": "up to 3 retries on timeout/HTTP 504 with halved output budget",
        "global_retry_policy": "all retry reservations count against the 600-attempt ceiling; at most 110 retry attempts exist globally",
    }
    document["manifest_sha256"] = scorer.manifest_hash(document)
    return document


def markdown_report(document: dict[str, Any], file_sha256: str) -> str:
    internal = document["manifest_sha256"]
    calibration = document["sources"]["calibration"]
    return "\n".join(
        [
            "# P5 v2.1 primary scoring — proposed hash-bound authorization",
            "",
            f"**Proposal status:** `{document['proposal_status']}`  ",
            "**Proxy/scoring calls made while building:** `0`  ",
            f"**Proposed manifest internal SHA-256:** `{internal}`  ",
            f"**Proposed manifest file SHA-256:** `{file_sha256}`",
            "",
            "## Bound execution limits",
            "",
            "- Primary assignments only: 176",
            "- Initial planned requests: 490",
            "- Total network-attempt ceiling: 600",
            "- Global retry-attempt capacity: 110",
            "- Per-chunk retry ceiling: 3, still subject to the 600-attempt global ceiling",
            "- Spend ceiling: $15.00",
            "- Approved maximum cost per request: $0.025",
            "- Remaining-quota stop floor: $5.00",
            "- Same-Sonnet repeats: absent and unauthorized",
            "",
            f"The ceiling product is `600 × $0.025 = $15.00`; `490 ≤ 600`. Calibration is bound by result SHA `{calibration['results_sha256']}` and journal SHA `{calibration['journal_sha256']}`. It observed two successful zero-retry calls costing `${calibration['total_observed_cost_usd']:.5f}` total, `${calibration['maximum_observed_cost_usd']:.5f}` maximum, with minimum remaining quota `${calibration['minimum_observed_remaining_quota_usd']:.5f}`.",
            "",
            "## Decision-ready owner authorization line",
            "",
            f"> I authorize P5 v2.1 primary scoring under proposed manifest internal SHA `{internal}` and file SHA `{file_sha256}`, with exactly the bound limits recorded there: 176 primary assignments, 490 initial requests, at most 600 total attempts (therefore at most 110 retries globally and no more than 3 per chunk), $15.00 total spend, $0.025 maximum cost per request, and $5.00 remaining-quota stop floor; same-Sonnet repeats remain unauthorized.",
            "",
            "Until the owner supplies this exact-hash approval, do not pass `--authorised` and do not execute scoring.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--calibration-results", type=Path, required=True)
    parser.add_argument("--calibration-journal", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    document = build_proposal(args.plan, args.calibration_results, args.calibration_journal)
    executor.atomic_write(args.json_out, json.dumps(document, indent=2) + "\n")
    file_sha = p5.sha256_file(args.json_out)
    executor.atomic_write(args.markdown_out, markdown_report(document, file_sha))
    print(
        json.dumps(
            {
                "manifest_sha256": document["manifest_sha256"],
                "manifest_file_sha256": file_sha,
                "proposal_status": document["proposal_status"],
                "proxy_calls_made": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
