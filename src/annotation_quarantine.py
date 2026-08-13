"""Versioned quarantine/migration for coverage-incomplete annotation rows.

The 2026-08-11 Phase-2 QA left a checkpoint whose rows are all marked
``annotation_complete: true`` while several do not exhaust the annotation
region.  Those rows must become eligible for reannotation without destroying
the evidence of what the annotator actually returned.

This module never edits a checkpoint in place and never deletes a row.  It
writes:

* ``<quarantine_dir>/<name>.v<N>.json`` — the ORIGINAL rows that failed
  coverage, each with the verdict explaining why it is eligible again;
* ``<quarantine_dir>/<name>.v<N>.manifest.json`` — SHA-256 of the source
  checkpoint before and after, the rule version, and the per-row reasons;
* the migrated checkpoint, containing only the rows that pass coverage.

``plan_quarantine`` is pure and side-effect free, so a dry run can be reviewed
and approved before anything is written.  ``apply_quarantine`` refuses unless
``approved=True`` and the checkpoint still hashes to what the plan inspected.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from src.annotation_coverage import (COVERAGE_RULE_VERSION, row_is_coverage_complete,
                                     validate_coverage)

#: Bump when the migration's own semantics change (not when the rule changes).
QUARANTINE_SCHEMA_VERSION = "ph2-annotation-quarantine-1"


class QuarantineError(RuntimeError):
    """The migration cannot proceed safely."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class QuarantinePlan:
    """What a migration would do, computed without touching anything."""

    checkpoint_path: Path
    checkpoint_sha256: str
    rule_version: str
    retained: tuple[dict, ...]
    quarantined: tuple[dict, ...]

    @property
    def n_total(self) -> int:
        return len(self.retained) + len(self.quarantined)

    def summary(self) -> dict:
        return {
            "schema": QUARANTINE_SCHEMA_VERSION,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "coverage_rule_version": self.rule_version,
            "n_rows": self.n_total,
            "n_retained": len(self.retained),
            "n_quarantined": len(self.quarantined),
            "quarantined_rows": [
                {
                    "task_id": row.get("task_id"),
                    "reasons": list(
                        (row.get("annotation_coverage") or {}).get("reasons", [])
                    ),
                    "coverage_fraction": (
                        row.get("annotation_coverage") or {}
                    ).get("coverage_fraction"),
                    "n_spans": len(row.get("annotations") or []),
                }
                for row in self.quarantined
            ],
        }


def plan_quarantine(
    checkpoint_path: Path,
    *,
    include_post_think: bool = False,
    revalidate: bool = True,
) -> QuarantinePlan:
    """Classify every checkpoint row as retained or eligible for reannotation.

    ``revalidate=True`` recomputes the coverage verdict from the stored spans
    and chain, which is what migrates rows written before the validator
    existed: they carry no verdict, so they cannot be judged any other way.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise QuarantineError(f"checkpoint not found: {checkpoint_path}")
    raw = checkpoint_path.read_bytes()
    rows = json.loads(raw)
    if not isinstance(rows, list):
        raise QuarantineError("checkpoint must be a list of annotated records")

    retained: list[dict] = []
    quarantined: list[dict] = []
    for row in rows:
        annotated = dict(row)
        if revalidate:
            report = validate_coverage(
                row.get("chain", ""), row.get("annotations") or [],
                include_post_think=include_post_think,
            )
            annotated["annotation_coverage"] = report.to_dict()
            annotated["annotation_coverage_complete"] = bool(
                row.get("annotation_complete", False) and report.complete
            )
            annotated["annotation_coverage_rule_version"] = COVERAGE_RULE_VERSION
        if row_is_coverage_complete(annotated):
            retained.append(annotated)
        else:
            quarantined.append(annotated)

    return QuarantinePlan(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=_sha256_bytes(raw),
        rule_version=COVERAGE_RULE_VERSION,
        retained=tuple(retained),
        quarantined=tuple(quarantined),
    )


def _next_version(quarantine_dir: Path, stem: str) -> int:
    existing = sorted(quarantine_dir.glob(f"{stem}.v*.json"))
    versions = []
    for path in existing:
        part = path.name[len(stem) + 2:].split(".", 1)[0]
        if part.isdigit():
            versions.append(int(part))
    return max(versions, default=0) + 1


def apply_quarantine(
    plan: QuarantinePlan,
    quarantine_dir: Path,
    *,
    approved: bool,
) -> dict:
    """Write the quarantine files and the migrated checkpoint.

    Refuses unless ``approved`` is True and the checkpoint is byte-identical to
    the one the plan was computed from, so an unreviewed or concurrently
    modified checkpoint can never be rewritten.
    """
    if not approved:
        raise QuarantineError(
            "quarantine migration requires explicit owner approval; nothing written"
        )
    current = _sha256_bytes(plan.checkpoint_path.read_bytes())
    if current != plan.checkpoint_sha256:
        raise QuarantineError(
            "checkpoint changed since the plan was computed; refusing to migrate"
        )
    if not plan.quarantined:
        raise QuarantineError("no coverage-incomplete rows to quarantine")

    quarantine_dir = Path(quarantine_dir)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    stem = plan.checkpoint_path.stem
    version = _next_version(quarantine_dir, stem)

    rows_path = quarantine_dir / f"{stem}.v{version}.json"
    manifest_path = quarantine_dir / f"{stem}.v{version}.manifest.json"
    rows_path.write_text(
        json.dumps(list(plan.quarantined), indent=2, ensure_ascii=False)
    )

    migrated = json.dumps(list(plan.retained), indent=2, ensure_ascii=False)
    manifest = {
        **plan.summary(),
        "version": version,
        "quarantined_rows_path": str(rows_path),
        "checkpoint_sha256_before": plan.checkpoint_sha256,
        "checkpoint_sha256_after": _sha256_bytes(migrated.encode()),
        "quarantined_rows_sha256": _sha256_bytes(rows_path.read_bytes()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    tmp = plan.checkpoint_path.with_suffix(".migrating")
    tmp.write_text(migrated)
    tmp.replace(plan.checkpoint_path)
    return manifest
