#!/usr/bin/env python3
"""Freeze Tony's approved P5 powered-generic design without authorizing execution.

This is an offline document builder. It reads only frozen P5 design artefacts,
uses arm-blinded variance inputs, and makes no network/model/API/pod call.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any


HERE = Path(__file__).resolve().parent
SKELETON = HERE / "P5_POWERED_GENERIC_PROTOCOL_MANIFEST_SKELETON_2026-08-09.json"
GENERIC_QA = HERE / "P5_GENERIC_PILOT_ARM_BLINDED_QA_SIZING_2026-08-09.json"
RECOMMENDATION = HERE / "P5_GENERIC_OWNER_DECISION_RECOMMENDATION_2026-08-09.md"
OUTPUT_JSON = HERE / "P5_POWERED_GENERIC_PROTOCOL_APPROVED_DESIGN_2026-08-09.json"
OUTPUT_MD = HERE / "P5_POWERED_GENERIC_PROTOCOL_APPROVED_DESIGN_2026-08-09.md"

EXPECTED_FILES = {
    SKELETON: "abffad6d6452b3bc135e625aab2a6ea07415f5b76d62986ed9ba386514e0c78b",
    GENERIC_QA: "1368c2a7db408862e6a268eec794605ba05b12247841734baacebb1dfb8b336f",
    RECOMMENDATION: "8a6ab535ce3c0f3dc5d0ee755f8e1ecbe16b968eb36370511aa4bccb5952255e",
}
ENDPOINTS = (
    "backtracking",
    "uncertainty-estimation",
    "example-testing",
    "adding-knowledge",
)
DELTA = 0.05
POWER = 0.80
PLANNING_ALPHA = 0.05 / 4
TASKS = 100
CATEGORIES = 10


class FreezeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def internal_hash(document: dict[str, Any]) -> str:
    content = dict(document)
    content.pop("document_sha256", None)
    return sha256_json(content)


def verify_sources() -> None:
    for path, expected in EXPECTED_FILES.items():
        if sha256_file(path) != expected:
            raise FreezeError(f"source hash mismatch: {path.name}")


def required_tasks(second_moment: float, completion_rate: float) -> int:
    z = NormalDist().inv_cdf(1 - PLANNING_ALPHA / 2) + NormalDist().inv_cdf(POWER)
    complete = max(2, math.ceil((z * z) * second_moment / (DELTA * DELTA)))
    attempted = math.ceil(complete / completion_rate)
    return int(math.ceil(attempted / CATEGORIES) * CATEGORIES)


def build_document() -> dict[str, Any]:
    verify_sources()
    document = copy.deepcopy(json.loads(SKELETON.read_text()))
    generic = json.loads(GENERIC_QA.read_text())
    document.pop("document_sha256", None)
    document["schema_version"] = "p5-powered-generic-protocol-approved-design-1"
    document["status"] = "approved_design_non_executable_phase2_and_spend_gated"
    document["created_at_utc"] = "2026-08-09T00:00:00Z"
    document["execution_authorized"] = False
    document["proxy_model_or_pod_calls_authorized"] = 0
    document["owner_approval"] = {
        "approved_in_chat_date": "2026-08-09",
        "scope": "design choices only; no generation, annotation, model inference, API, pod, or human-annotation spend",
        "recommendation_path": str(RECOMMENDATION.resolve()),
        "recommendation_sha256": EXPECTED_FILES[RECOMMENDATION],
    }

    prefix = document["analytic_prefix_decision"]
    prefix["primary_prefix_generated_tokens"] = 4096
    prefix["status"] = "owner_approved_design_choice"
    prefix["eos_rule"] = "earlier EOS ends the prefix"
    prefix["raw_phase2_generation_contract"] = "unchanged sealed E8/Option-A raw generation; P5 prefix is analytic only"

    checkpoint = document["checkpoint_structure"]
    checkpoint["primary_contrast_families"] = [
        "P5-owned matched full-FT control seed 42 versus full-FT safety seed 42"
    ]
    checkpoint["secondary_contrasts"] = [
        "Phase-2 base R1 versus public STAR1",
        "every comparison involving DeepScaleR",
    ]
    checkpoint["deepscaler_status"] = "secondary observational"

    requirements: dict[str, int] = {}
    for endpoint in ENDPOINTS:
        cell = generic["unsigned_paired_design_inputs"][endpoint]["contrast_family_B"]
        requirements[endpoint] = required_tasks(
            cell["unsigned_squared_difference_distribution"]["mean"],
            cell["pair_completion_rate"],
        )
    if max(requirements.values()) > TASKS:
        raise FreezeError("approved 100-task design does not cover primary planning cells")

    size = document["sample_size_freeze"]
    size["freeze_rule"] = (
        "maximum conservative required N across the four endpoints in the single primary owned "
        "full-FT safety-versus-control contrast; arm-blinded unsigned paired variance only"
    )
    size["selected_absolute_effect"] = DELTA
    size["selected_alpha_strategy"] = {
        "planning": "Bonferroni bound across four primary endpoints",
        "planning_two_sided_alpha_per_endpoint": PLANNING_ALPHA,
        "analysis": "Holm familywise correction across four primary endpoints",
    }
    size["selected_power"] = POWER
    size["selected_tasks"] = TASKS
    size["primary_required_category_balanced_tasks"] = requirements
    size["maximum_primary_required_tasks"] = max(requirements.values())
    size["all_primary_cells_fit_selected_tasks"] = True
    size["secondary_power_boundary"] = (
        "100 tasks is not claimed to power every five-point public-STAR1/base or DeepScaleR cell"
    )

    analysis = document["analysis"]
    analysis["multiple_testing"] = "Holm familywise correction across four endpoints in one primary contrast"
    analysis["primary_contrast"] = (
        "owned matched full-FT control seed 42 versus owned full-FT safety seed 42"
    )
    analysis["secondary_contrasts"] = (
        "public STAR1 versus base R1 and every DeepScaleR contrast; estimates and intervals without equal power claim"
    )
    analysis["arm_labelled_pilot_effects_used_for_design"] = False

    document["owner_decisions"] = {
        "status": "resolved_for_design",
        "analytic_prefix": 4096,
        "smallest_absolute_effect": DELTA,
        "power": POWER,
        "planning_alpha_per_endpoint": PLANNING_ALPHA,
        "task_maximum": TASKS,
        "deepscaler": "secondary observational",
    }
    document["remaining_blockers"] = [
        "Phase-2 canonical shared-vanilla manifest, internal-ID, artefact/shard, and row hashes",
        "P5-owned checkpoint-row generation manifest, exact cost, and owner spend authorization",
        "generic annotation/scorer validation at the frozen operational gate",
        "analysis and output manifests bound to final shared-row and analytic-prefix hashes",
    ]
    document["document_sha256"] = internal_hash(document)
    return document


def markdown(document: dict[str, Any], json_file_sha256: str) -> str:
    req = document["sample_size_freeze"]["primary_required_category_balanced_tasks"]
    return f"""# Approved P5 powered-generic design freeze

**Status:** design approved; non-executable and Phase-2/spend-gated  
**Internal SHA-256:** `{document['document_sha256']}`  
**JSON file SHA-256:** `{json_file_sha256}`  
**Calls/spend authorized by this freeze:** none

## Frozen primary design

- Contrast: owned matched full-FT control seed 42 versus owned full-FT safety seed 42.
- Endpoints: backtracking, uncertainty estimation, example testing, and adding knowledge.
- Smallest absolute effect: 0.05 sentence fraction.
- Power: 80%.
- Planning: two-sided Bonferroni alpha 0.0125 per endpoint.
- Analysis: Holm across the four primary endpoints.
- Tasks: 100, balanced across ten categories.
- Analytic prefix: first 4,096 generated token IDs or earlier EOS.

Arm-blinded conservative task requirements are backtracking {req['backtracking']}, uncertainty estimation {req['uncertainty-estimation']}, example testing {req['example-testing']}, and adding knowledge {req['adding-knowledge']}. All fit the 100-task design. These are planning quantities, not outcome findings.

Public STAR1 versus base R1 and every DeepScaleR comparison are secondary. The design does not claim equal five-point power for every secondary cell. The owned comparison has one training seed and remains checkpoint-bounded rather than recipe-level replication.

## Remaining blockers

The Phase-2 shared-vanilla hashes, P5-owned generation manifest and spend authorization, prospective generic-annotation validation, and final prefix/analysis hash bindings are still absent. This document cannot launch generation, annotation, inference, API calls, pod work, or human annotation.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    document = build_document()
    if not args.write:
        print(json.dumps(document, sort_keys=True, indent=2))
        return
    OUTPUT_JSON.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    file_hash = sha256_file(OUTPUT_JSON)
    OUTPUT_MD.write_text(markdown(document, file_hash))
    print(json.dumps({
        "json": str(OUTPUT_JSON),
        "json_internal_sha256": document["document_sha256"],
        "json_file_sha256": file_hash,
        "md": str(OUTPUT_MD),
        "md_file_sha256": sha256_file(OUTPUT_MD),
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
