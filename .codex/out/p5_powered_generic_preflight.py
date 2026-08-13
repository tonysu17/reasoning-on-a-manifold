#!/usr/bin/env python3
"""Offline, fail-closed readiness audit for the approved powered P5 study.

This module binds inputs that already exist, records absent inputs as blockers,
and never generates text, loads a model, calls an API, launches a pod, annotates,
or authorises execution. It writes only under ``.codex/out`` when ``--write`` is
passed. Phase-2 and checkpoint paths are read-only.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DESIGN = HERE / "P5_POWERED_GENERIC_PROTOCOL_APPROVED_DESIGN_2026-08-09.json"
PREFIX_CONTRACT = HERE / "P5_4096_ANALYTIC_PREFIX_CONTRACT_2026-08-09.md"
PREFIX_CODE = HERE / "p5_analytic_prefix.py"
SHARED_CONTRACT = HERE / "P5_PHASE2_SHARED_VANILLA_CONTRACT_2026-08-08.md"
PH2_MANIFEST = ROOT / "results/prereg/phase2_task_manifest.json"
PH2_STATUS = ROOT / "results/ph2/PH2_STATUS"
PH2_BATTERY = ROOT / "results/ph2/battery"
CONTROL = ROOT / "checkpoints/pod_fullft/fullft_control_s42"
SAFETY = ROOT / "checkpoints/pod_fullft/fullft_safety_s42"
OUTPUT = HERE / "P5_POWERED_GENERIC_PREFLIGHT_READINESS_2026-08-09.json"

EXPECTED = {
    DESIGN: "e36d44c20c6bde9dc197d0d2bccb6479c7134b654d8c29175f418a26375e2b6f",
    PREFIX_CONTRACT: "3403bca6b48a57639ea98edfb3a16671973f67a589821673abb636774097a06a",
    PREFIX_CODE: "ac21e8885f10a77956b285f8a65d0240bfc9e54ff40836442cfe48213c9b9731",
    SHARED_CONTRACT: "391461ca66ef60015d4790a4226ffdb839edd9fdc2ba14fa8be2947acf1184a2",
}
EXPECTED_DESIGN_INTERNAL = "0bf279cbac7a7fae51c89642af9b1dd068274a2acb517622b30efa6221664e2a"
EXPECTED_CATEGORIES = 10
EXPECTED_TASKS_PER_CATEGORY = 10
EXPECTED_TASKS = 100
EXPECTED_ROLES = ("base", "star1", "deepscaler")
REQUIRED_GENERATION_FIELDS = {
    "generated_token_ids",
    "stop_reason",
    "raw_max_new_tokens",
    "generation_config_sha256",
}
SHARED_CHECKPOINT_FILES = (
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
REQUIRED_CHECKPOINT_FILES = set(SHARED_CHECKPOINT_FILES) | {
    "model.safetensors",
    "training_summary.json",
}


class PreflightError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def internal_hash(document: dict[str, Any]) -> str:
    body = dict(document)
    body.pop("internal_sha256", None)
    return sha256_json(body)


def display_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def verify_fixed_sources() -> dict[str, str]:
    found: dict[str, str] = {}
    for path, expected in EXPECTED.items():
        if not path.is_file():
            raise PreflightError(f"required P5 source missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise PreflightError(f"required P5 source hash drift: {path.name}")
        found[str(path.relative_to(ROOT))] = actual
    design = json.loads(DESIGN.read_text())
    if design.get("document_sha256") != EXPECTED_DESIGN_INTERNAL:
        raise PreflightError("approved-design internal hash drift")
    if design.get("execution_authorized") is not False:
        raise PreflightError("approved design must remain non-executable")
    return found


def ph2_ids_sha256(task_ids: list[str]) -> str:
    """Match ``ph2_manifest.manifest_hash`` exactly."""
    payload = json.dumps(sorted(task_ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def _task_suffix(task_id: str) -> int:
    match = re.search(r"_(\d+)$", task_id)
    if not match:
        raise PreflightError(f"task id lacks numeric suffix: {task_id}")
    return int(match.group(1))


def validate_phase2_manifest(path: Path = PH2_MANIFEST) -> dict[str, Any]:
    if not path.is_file():
        raise PreflightError("Phase-2 task manifest missing")
    document = json.loads(path.read_text())
    tasks = document.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != EXPECTED_TASKS:
        raise PreflightError("Phase-2 task manifest must contain exactly 100 tasks")
    rows: list[dict[str, Any]] = []
    ids: list[str] = []
    categories: collections.Counter[str] = collections.Counter()
    for task in tasks:
        if set(("id", "prompt", "category")) - set(task):
            raise PreflightError("Phase-2 task lacks id/prompt/category")
        task_id, prompt, category = task["id"], task["prompt"], task["category"]
        if not all(isinstance(value, str) and value for value in (task_id, prompt, category)):
            raise PreflightError("Phase-2 task identity values must be non-empty strings")
        ids.append(task_id)
        categories[category] += 1
        rows.append({
            "task_id": task_id,
            "category": category,
            "prompt_utf8_bytes": len(prompt.encode("utf-8")),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        })
    if len(set(ids)) != EXPECTED_TASKS:
        raise PreflightError("Phase-2 task IDs are not unique")
    if len(categories) != EXPECTED_CATEGORIES or set(categories.values()) != {
        EXPECTED_TASKS_PER_CATEGORY
    }:
        raise PreflightError(f"Phase-2 category balance invalid: {dict(categories)}")
    computed_ids = ph2_ids_sha256(ids)
    if document.get("ids_sha256") != computed_ids:
        raise PreflightError("Phase-2 ids_sha256 mismatch")
    if document.get("n") != EXPECTED_TASKS:
        raise PreflightError("Phase-2 manifest n is not 100")
    if document.get("per_category") != dict(categories):
        raise PreflightError("Phase-2 per_category metadata disagrees with task rows")
    actual_start = min(_task_suffix(task_id) for task_id in ids)
    generator_start = document.get("generator", {}).get("id_start")
    static_floor = document.get("id_start")
    if generator_start != actual_start:
        raise PreflightError("Phase-2 generator id_start disagrees with task IDs")
    if not isinstance(static_floor, int) or static_floor > actual_start:
        raise PreflightError("Phase-2 static id_start floor is invalid")
    return {
        "path": display_path(path),
        "file_sha256": sha256_file(path),
        "ids_sha256": computed_ids,
        "ordered_task_content_sha256": sha256_json(rows),
        "n_tasks": len(rows),
        "per_category": dict(sorted(categories.items())),
        "static_id_start_floor": static_floor,
        "actual_dynamic_id_start": actual_start,
        "task_rows": rows,
    }


def checkpoint_manifest(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise PreflightError(f"checkpoint directory missing: {path}")
    files = sorted(item for item in path.iterdir() if item.is_file())
    names = {item.name for item in files}
    missing = REQUIRED_CHECKPOINT_FILES - names
    if missing:
        raise PreflightError(f"checkpoint missing required files: {sorted(missing)}")
    rows = [
        {"name": item.name, "size": item.stat().st_size, "sha256": sha256_file(item)}
        for item in files
    ]
    return {
        "path": display_path(path),
        "files": rows,
        "directory_manifest_sha256": sha256_json(rows),
    }


def validate_checkpoint_pair(
    control_path: Path = CONTROL, safety_path: Path = SAFETY
) -> dict[str, Any]:
    control = checkpoint_manifest(control_path)
    safety = checkpoint_manifest(safety_path)
    c_files = {item["name"]: item["sha256"] for item in control["files"]}
    s_files = {item["name"]: item["sha256"] for item in safety["files"]}
    drift = [name for name in SHARED_CHECKPOINT_FILES if c_files[name] != s_files[name]]
    if drift:
        raise PreflightError(f"owned checkpoint tokenizer/config drift: {drift}")
    if c_files["model.safetensors"] == s_files["model.safetensors"]:
        raise PreflightError("owned safety/control model weights are byte-identical")
    return {
        "control": control,
        "safety": safety,
        "shared_tokenizer_template_config_sha256": sha256_json(
            {name: c_files[name] for name in SHARED_CHECKPOINT_FILES}
        ),
        "model_weights_distinct": True,
    }


def inspect_phase2_shared(task_ids: set[str]) -> dict[str, Any]:
    role_reports: dict[str, Any] = {}
    all_ready = True
    for role in EXPECTED_ROLES:
        path = PH2_BATTERY / f"{role}.json"
        if not path.is_file():
            role_reports[role] = {"path": str(path.relative_to(ROOT)), "status": "absent"}
            all_ready = False
            continue
        rows = json.loads(path.read_text())
        vanilla = [row for row in rows if row.get("method") == "vanilla"]
        ids = [row.get("task_id") for row in vanilla]
        missing_fields = sorted({
            field for row in vanilla for field in REQUIRED_GENERATION_FIELDS if field not in row
        })
        valid = (
            len(vanilla) == EXPECTED_TASKS
            and len(set(ids)) == EXPECTED_TASKS
            and set(ids) == task_ids
            and not missing_fields
        )
        role_reports[role] = {
            "path": str(path.relative_to(ROOT)),
            "file_sha256": sha256_file(path),
            "n_vanilla": len(vanilla),
            "task_set_matches": set(ids) == task_ids,
            "missing_lossless_generation_fields": missing_fields,
            "status": "ready" if valid else "ineligible",
        }
        all_ready = all_ready and valid
    canonical_candidates = [
        PH2_BATTERY / "shared_vanilla_300.json",
        PH2_BATTERY / "shared_vanilla_manifest.json",
    ]
    present = [path for path in canonical_candidates if path.is_file()]
    return {
        "status": "ready" if all_ready and present else "blocked",
        "role_shards": role_reports,
        "canonical_artifact_candidates_present": [
            {"path": str(path.relative_to(ROOT)), "file_sha256": sha256_file(path)}
            for path in present
        ],
        "requires_original_generated_token_ids": True,
        "decoded_text_retokenization_permitted": False,
    }


def build_report() -> dict[str, Any]:
    sources = verify_fixed_sources()
    phase2_manifest = validate_phase2_manifest()
    checkpoints = validate_checkpoint_pair()
    phase2_status = PH2_STATUS.read_text().strip() if PH2_STATUS.is_file() else "ABSENT"
    shared = inspect_phase2_shared({row["task_id"] for row in phase2_manifest["task_rows"]})
    blockers: list[str] = []
    if phase2_status not in {"COMPLETE", "DONE", "SUCCEEDED"}:
        blockers.append(f"Phase-2 status is {phase2_status!r}, not a completed state")
    if shared["status"] != "ready":
        blockers.append("canonical eligible 300-row Phase-2 shared-vanilla input absent")
    blockers.extend([
        "P5-owned 200-row generation artefact and immutable run manifest absent",
        "generic annotation validation has not passed its frozen gate",
        "powered annotation/generation spend has not been hash-bound or authorised",
    ])
    report: dict[str, Any] = {
        "schema_version": "p5-powered-generic-preflight-readiness-1",
        "snapshot_date": "2026-08-09",
        "execution_authorized": False,
        "network_model_api_pod_calls_made": 0,
        "status": "ready_for_authorisation" if not blockers else "blocked_non_executable",
        "fixed_p5_sources": sources,
        "phase2_status": phase2_status,
        "phase2_task_manifest": phase2_manifest,
        "owned_checkpoint_pair": checkpoints,
        "phase2_shared_vanilla": shared,
        "blockers": blockers,
        "boundary": (
            "This snapshot binds existing inputs only. It cannot launch generation, "
            "annotation, inference, API calls, pod work, or human annotation."
        ),
    }
    report["internal_sha256"] = internal_hash(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = build_report()
    if args.write:
        OUTPUT.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        print(json.dumps({
            "path": str(OUTPUT),
            "file_sha256": sha256_file(OUTPUT),
            "internal_sha256": report["internal_sha256"],
            "status": report["status"],
            "blockers": report["blockers"],
        }, sort_keys=True, indent=2))
    else:
        print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
