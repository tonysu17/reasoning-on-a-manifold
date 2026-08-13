#!/usr/bin/env python3
"""Offline preflight for the P5 replacement safety-evaluator validation set.

This module performs no network, model, API, pod, or human-annotation action.
It reproduces the frozen 24-response selection, verifies source and row hashes,
and emits private/public metadata in memory.  Checkpoint identity is excluded
from the public rows by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE / "p5_runs" / "p5-pilot-20260808"
GENERATIONS = RUN_ROOT / "generations.jsonl"
RUN_MANIFEST = RUN_ROOT / "run_manifest.json"
SPEC = HERE / "P5_SAFETY_HUMAN_VALIDATION_SELECTION_SPEC_2026-08-09.json"

EXPECTED_FILE_HASHES = {
    GENERATIONS: "2f2e9bc4d71c0e8f7ef71fe067d0e3f12b6fb2af31e83ee692884d70f6bbaf01",
    RUN_MANIFEST: "ec8fbead64dfc0362d9c3646bdcaa9593e8beadb5d8ce38033c6f0d3e983c567",
    SPEC: "15e5e777d55ab8fede7d17ca59dda21f90bcf5e42bee761a4ec183c73f5c3bf3",
}

PRIVATE_FIELDS = (
    "blind_id",
    "prompt_id",
    "pair_id",
    "stratum",
    "category",
    "checkpoint_role",
    "generation_record_sha256",
    "prompt_text_sha256",
    "n_tokens",
    "length_cap_hit",
)
HIDDEN_FROM_HUMANS = {
    "checkpoint_role",
    "checkpoint_id",
    "checkpoint_revision",
    "checkpoint_weight_sha256",
    "training_recipe",
    "automated_labels",
    "expected_effect_direction",
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
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_sources() -> dict[str, str]:
    verified: dict[str, str] = {}
    for path, expected in EXPECTED_FILE_HASHES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise PreflightError(f"source hash mismatch: {path.name}")
        verified[str(path.resolve())] = actual
    return verified


def reproduce_selection() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    verify_sources()
    spec = json.loads(SPEC.read_text())
    role_pairs = spec["selection_rule"]["role_pairs_by_pair_id"]
    source_rows = load_jsonl(GENERATIONS)
    private: list[dict[str, Any]] = []

    for generation in source_rows:
        pair_id = generation.get("pair_id")
        role = generation.get("checkpoint_role")
        if pair_id not in role_pairs or role not in role_pairs[pair_id]:
            continue
        generation_hash = sha256_json(generation)
        blind_hash = hashlib.sha256(
            f"p5-human-validation-2026-08-09|{generation_hash}".encode("utf-8")
        ).hexdigest()
        row = {
            "blind_id": f"p5hv-{blind_hash[:16]}",
            "prompt_id": generation["prompt_id"],
            "pair_id": pair_id,
            "stratum": generation["stratum"],
            "category": generation["category"],
            "checkpoint_role": role,
            "generation_record_sha256": generation_hash,
            "prompt_text_sha256": generation["prompt_text_sha256"],
            "n_tokens": generation["n_tokens"],
            "length_cap_hit": generation["n_tokens"] >= 4096,
        }
        if tuple(row) != PRIVATE_FIELDS:
            raise PreflightError("private row field order drift")
        private.append(row)

    private.sort(key=lambda row: row["blind_id"])
    public = [
        {key: value for key, value in row.items() if key != "checkpoint_role"}
        for row in private
    ]
    expected = spec["expected"]
    if len(private) != expected["n_rows"]:
        raise PreflightError("selection row count mismatch")
    if sha256_json(private) != expected["private_rows_sha256"]:
        raise PreflightError("private selection hash mismatch")
    if sha256_json(public) != expected["public_rows_sha256"]:
        raise PreflightError("public selection hash mismatch")
    if any(HIDDEN_FROM_HUMANS.intersection(row) for row in public):
        raise PreflightError("public selection leaks a hidden field")

    counts = {
        "strata": Counter(row["stratum"] for row in private),
        "categories": Counter(row["category"] for row in private),
        "checkpoint_roles": Counter(row["checkpoint_role"] for row in private),
    }
    if sorted(counts["strata"].values()) != [12, 12]:
        raise PreflightError("stratum balance mismatch")
    if set(counts["categories"].values()) != {4}:
        raise PreflightError("category balance mismatch")
    if set(counts["checkpoint_roles"].values()) != {6}:
        raise PreflightError("checkpoint-role balance mismatch")
    if sum(row["n_tokens"] for row in private) != expected["total_generated_tokens"]:
        raise PreflightError("token total mismatch")
    if sum(row["length_cap_hit"] for row in private) != expected["length_cap_hits"]:
        raise PreflightError("length-cap total mismatch")
    return private, public


def build_dry_report() -> dict[str, Any]:
    private, public = reproduce_selection()
    return {
        "schema_version": "p5-safety-human-validation-preflight-1",
        "status": "pass_zero_spend_non_executable",
        "network_model_api_pod_calls": 0,
        "human_annotations": 0,
        "selection": {
            "n_rows": len(private),
            "private_rows_sha256": sha256_json(private),
            "public_rows_sha256": sha256_json(public),
            "total_generated_tokens": sum(row["n_tokens"] for row in private),
            "length_cap_hits": sum(row["length_cap_hit"] for row in private),
            "strata": dict(sorted(Counter(row["stratum"] for row in private).items())),
            "categories": dict(sorted(Counter(row["category"] for row in private).items())),
            "checkpoint_role_counts_private_only": sorted(
                Counter(row["checkpoint_role"] for row in private).values()
            ),
        },
        "blinding": {
            "public_rows_contain_checkpoint_identity": False,
            "arm_labelled_effect_estimation_authorized": False,
            "automated_labels_visible_to_humans": False,
        },
        "execution_blockers": [
            "exact WildGuard model revision and local file hashes unset",
            "local inference environment and deterministic replay unset",
            "smoke-test pod ceiling not authorized",
            "human annotation scope and compensation not authorized",
            "Prometheus excluded unless a separate reference-answer protocol is frozen",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", required=True)
    args = parser.parse_args()
    if not args.dry_run:
        raise PreflightError("only --dry-run is implemented")
    print(json.dumps(build_dry_report(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
