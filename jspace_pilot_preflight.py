#!/usr/bin/env python3
"""Offline preflight for the sealed R1 J-lens pilot.

This command performs no model load, network access, fitting, generation, or
result write. It validates the fixed local inputs and, optionally, a local
checkout of Anthropic's reference implementation against the sealed benchmark
manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = (
    ROOT / "results/prereg/JSPACE_R1_BENCHMARK_MANIFEST_2026-08-10.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_head(checkout: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def validate(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    root: Path = ROOT,
    jlens_checkout: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "errors": [f"manifest unreadable: {exc}"], "warnings": []}

    if manifest.get("execution_authorised") is not False:
        errors.append("manifest must keep execution_authorised=false")
    if manifest.get("status") != "sealed_specification_unrun":
        errors.append("manifest status is not sealed_specification_unrun")

    for relative, expected in manifest.get("fixed_inputs", {}).items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing fixed input: {relative}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            errors.append(f"hash mismatch: {relative}: {actual} != {expected}")

    tasks_path = root / "data/tasks_final.json"
    tasks: dict[str, dict[str, Any]] = {}
    if tasks_path.is_file():
        try:
            tasks = {row["id"]: row for row in json.loads(tasks_path.read_text())}
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"canonical task source malformed: {exc}")

    seen_ids: set[str] = set()
    for item in manifest.get("prompts", []):
        task_id = item.get("task_id")
        if not isinstance(task_id, str) or task_id in seen_ids:
            errors.append(f"missing or duplicate benchmark task_id: {task_id!r}")
            continue
        seen_ids.add(task_id)
        source = tasks.get(task_id)
        if source is None:
            errors.append(f"benchmark task absent from canonical source: {task_id}")
            continue
        prompt = item.get("prompt")
        if prompt != source.get("prompt"):
            errors.append(f"prompt text differs from canonical source: {task_id}")
            continue
        if item.get("category") != source.get("category"):
            errors.append(f"category differs from canonical source: {task_id}")
        actual_bytes = len(prompt.encode("utf-8"))
        if item.get("utf8_bytes") != actual_bytes:
            errors.append(f"UTF-8 byte count mismatch: {task_id}")
        actual_hash = sha256_text(prompt)
        if item.get("prompt_sha256") != actual_hash:
            errors.append(f"prompt hash mismatch: {task_id}")

    if len(seen_ids) != 5:
        errors.append(f"benchmark must contain exactly five unique prompts, found {len(seen_ids)}")

    metadata_path = root / "results/steering_vectors/R1-1.5B__E1_pooled/metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("backtracking", {}).get("layer") != 17:
                errors.append("primary backtracking vector is not registered at L17")
            holdout = metadata.get("_provenance", {}).get("holdout", {})
            if holdout.get("n_tasks") != 50:
                errors.append("primary vector metadata does not record a 50-task holdout")
        except json.JSONDecodeError as exc:
            errors.append(f"steering metadata malformed: {exc}")

    eval_ids_path = root / "results/eval/R1-1.5B__E1/eval_task_ids.json"
    if eval_ids_path.is_file():
        try:
            eval_ids = json.loads(eval_ids_path.read_text()).get("task_ids", [])
            if len(eval_ids) != 50 or len(set(eval_ids)) != 50:
                errors.append("evaluation split is not 50 unique task IDs")
        except json.JSONDecodeError as exc:
            errors.append(f"evaluation task manifest malformed: {exc}")

    output_root = root / manifest.get("output_root", "")
    if output_root.exists():
        warnings.append(f"prospective output root already exists: {output_root}")

    if jlens_checkout is not None:
        checkout = jlens_checkout.resolve()
        expected_commit = manifest.get("jlens", {}).get("commit")
        actual_commit = _git_head(checkout)
        if actual_commit != expected_commit:
            errors.append(
                f"J-lens checkout commit mismatch: {actual_commit} != {expected_commit}"
            )
        for relative, expected in manifest.get("jlens", {}).get("files_sha256", {}).items():
            path = checkout / relative
            if not path.is_file():
                errors.append(f"missing J-lens file: {relative}")
                continue
            actual = sha256_file(path)
            if actual != expected:
                errors.append(f"J-lens file hash mismatch: {relative}: {actual} != {expected}")
    else:
        warnings.append("J-lens checkout not supplied; repository commit/file checks skipped")

    return {
        "ok": not errors,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "execution_authorised": manifest.get("execution_authorised"),
        "n_prompts": len(seen_ids),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--jlens-checkout",
        type=Path,
        default=None,
        help="Optional local checkout to verify against the pinned commit and file hashes.",
    )
    args = parser.parse_args()
    report = validate(args.manifest, jlens_checkout=args.jlens_checkout)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

