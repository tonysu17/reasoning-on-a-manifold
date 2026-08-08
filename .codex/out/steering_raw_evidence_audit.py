#!/usr/bin/env python3
"""Recompute the thesis raw steering table into a non-authoritative audit file.

Inputs are read-only analysis-repository artefacts.  Output is restricted to
``.codex/out`` and is explicitly a candidate for later evidence-chain repair,
not an artefact of record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


BEHAVIOURS = (
    "backtracking",
    "uncertainty-estimation",
    "example-testing",
    "adding-knowledge",
)
METHODS = ("single_direction", "manifold_k3", "manifold_k5")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fraction(annotations: list[dict], target: str) -> float:
    if not annotations:
        raise ValueError("empty annotations are missing data, not zero behaviour")
    return sum(item.get("label") == target for item in annotations) / len(annotations)


def key(row: dict) -> tuple:
    return row["task_id"], row["behaviour"], row["method"], float(row["alpha"])


def source_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    resolved_out = args.out.resolve()
    if ".codex/out" not in resolved_out.as_posix():
        raise SystemExit("refusing to write outside .codex/out")

    generation_path = args.eval_dir / "steering_results.json"
    annotation_path = args.eval_dir / "annotated_steered.json"
    provenance_path = args.eval_dir / "provenance.json"
    generated = json.loads(generation_path.read_text())
    annotated = json.loads(annotation_path.read_text())
    provenance = json.loads(provenance_path.read_text())

    generated_index = {key(row): row for row in generated}
    if len(generated_index) != len(generated):
        raise SystemExit("generation keys are not unique")
    annotation_index = {key(row): row for row in annotated}
    if len(annotation_index) != len(annotated):
        raise SystemExit("annotation keys are not unique")
    orphan_annotations = sorted(set(annotation_index) - set(generated_index))
    if orphan_annotations:
        raise SystemExit(f"{len(orphan_annotations)} annotation rows lack generation rows")

    vanilla = {
        row["task_id"]: row
        for row in annotated
        if row["behaviour"] == "shared" and row["method"] == "vanilla" and row.get("annotations")
    }
    cells = []
    for behaviour in BEHAVIOURS:
        for method in METHODS:
            generated_cell = [
                row
                for row in generated
                if row["behaviour"] == behaviour
                and row["method"] == method
                and float(row["alpha"]) == 1.0
            ]
            annotated_cell = [
                row
                for row in annotated
                if row["behaviour"] == behaviour
                and row["method"] == method
                and float(row["alpha"]) == 1.0
            ]
            paired = [
                row
                for row in annotated_cell
                if row.get("annotations") and row["task_id"] in vanilla
            ]
            arm_values = [fraction(row["annotations"], behaviour) for row in paired]
            vanilla_values = [fraction(vanilla[row["task_id"]]["annotations"], behaviour) for row in paired]
            arm_mean = sum(arm_values) / len(arm_values)
            vanilla_mean = sum(vanilla_values) / len(vanilla_values)
            delta = arm_mean - vanilla_mean
            cells.append(
                {
                    "behaviour": behaviour,
                    "method": method,
                    "alpha": 1.0,
                    "n_generated": len(generated_cell),
                    "n_annotated_records": len(annotated_cell),
                    "n_paired_nonempty": len(paired),
                    "n_unpaired_or_empty": len(generated_cell) - len(paired),
                    "estimand": "mean over paired tasks of target-labelled sentence fraction in arm minus paired vanilla",
                    "paired_vanilla_mean": vanilla_mean,
                    "arm_mean": arm_mean,
                    "delta_fraction": delta,
                    "delta_percentage_points": 100.0 * delta,
                    "relative_change_percent": 100.0 * delta / vanilla_mean,
                }
            )

    repo = args.eval_dir.resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", str(repo), "diff", "--quiet"]) != 0
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None

    document = {
        "status": "recomputation candidate only; not an artefact of record and not thesis evidence",
        "purpose": "demonstrate the compact summary needed to repair snapshot reproducibility",
        "source_repository_state_at_audit": {"commit": commit, "tracked_worktree_dirty": dirty},
        "source_artifacts": {
            "generations": source_record(generation_path),
            "annotations": source_record(annotation_path),
            "run_provenance": source_record(provenance_path),
        },
        "record_counts": {
            "generated": len(generated),
            "annotated": len(annotated),
            "shared_vanilla_nonempty": len(vanilla),
        },
        "source_run_provenance_as_recorded": provenance,
        "unresolved_provenance": (
            "The source run record has git_commit null and no input hashes. This audit recovers arithmetic from current files but cannot retroactively establish execution lineage."
        ),
        "cells": cells,
    }
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    resolved_out.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
