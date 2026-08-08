"""Read-only provenance audit for exact activation-vector duplicates.

This diagnostic does not call an estimator, generate a permutation, modify
inputs, or authorize a pilot retry. It records occurrence provenance and
cryptographic vector identities without serializing vector values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs" / "analysis" / "thesis_core_hardening_2026-07-26.yaml"
)
OUTPUT_ROOT = (
    ROOT
    / "results"
    / "robustness"
    / "cdim_null_pilot"
    / "R1-1.5B"
)
DEFAULT_JSON = OUTPUT_ROOT / "DUPLICATE_PROVENANCE_AUDIT.json"
DEFAULT_MARKDOWN = OUTPUT_ROOT / "DUPLICATE_PROVENANCE_AUDIT.md"

_OCCURRENCE_FIELDS = (
    "behaviour",
    "chain_id",
    "annotation_index",
    "char_offset",
    "token_start",
)


def _nfc(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("occurrence string fields must be strings")
    return unicodedata.normalize("NFC", value)


def _integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("occurrence integer fields must be integers")
    return int(value)


def _normalized_record(record: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in _OCCURRENCE_FIELDS if field not in record]
    if missing:
        raise ValueError(f"occurrence record is missing fields: {missing}")
    chain_id = _nfc(record["chain_id"])
    if not chain_id:
        raise ValueError("occurrence record has an empty chain id")
    normalized = {
        "behaviour": _nfc(record["behaviour"]),
        "chain_id": chain_id,
        "annotation_index": _integer(record["annotation_index"]),
        "char_offset": _integer(record["char_offset"]),
        "token_start": _integer(record["token_start"]),
        "n_positions": _integer(record.get("n_positions", 0)),
        "source_index": _integer(record.get("source_index", 0)),
    }
    if normalized["n_positions"] < 0 or normalized["source_index"] < 0:
        raise ValueError("occurrence record contains a negative count/index")
    return normalized


def _occurrence_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(record[field] for field in _OCCURRENCE_FIELDS)


def _record_lex_key(record: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector_sha256(row: np.ndarray) -> str:
    values = np.asarray(row, dtype="<f8")
    if not np.isfinite(values).all():
        raise ValueError("activation vector contains a non-finite value")
    if np.any(values == 0.0):
        values = values.copy()
        values[values == 0.0] = 0.0
    digest = hashlib.sha256()
    digest.update(b"thesis-duplicate-provenance-vector-v1\0")
    digest.update(values.shape[0].to_bytes(8, "big", signed=False))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _occurrence_sha256(record: Mapping[str, Any]) -> str:
    payload = [
        "thesis-duplicate-provenance-occurrence-v1",
        *[record[field] for field in _OCCURRENCE_FIELDS],
        record["n_positions"],
        record["source_index"],
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _classify_group(
    *,
    cross_label: bool,
    cross_chain: bool,
    zero_vector: bool,
    constant_vector: bool,
    same_annotation_location: bool,
    same_extraction_window: bool,
) -> str:
    if cross_label and same_annotation_location:
        return "same_location_multilabel"
    if cross_label and same_extraction_window:
        return "same_extraction_window_multilabel_annotation_alias"
    if zero_vector:
        return "shared_zero_vector"
    if constant_vector:
        return "shared_constant_vector"
    if cross_chain:
        return "cross_chain_exact_match"
    if cross_label:
        return "same_chain_distinct_location_cross_label_exact_match"
    return "within_label_exact_match"


def _root_cause_classification(groups: Sequence[Mapping[str, Any]]) -> str:
    cross_label = [group for group in groups if group["cross_label"]]
    if not cross_label:
        return "no_cross_label_exact_vector_conflict"
    classes = {group["classification"] for group in cross_label}
    if classes == {"same_location_multilabel"}:
        return "same_occurrence_reused_across_labels"
    if classes == {
        "same_extraction_window_multilabel_annotation_alias"
    }:
        return "same_extraction_window_reused_across_annotation_aliases"
    if classes <= {"shared_zero_vector", "shared_constant_vector"}:
        return "sentinel_or_constant_activation_reuse"
    if any(group["cross_chain"] for group in cross_label):
        return "cross_chain_exact_activation_reuse"
    if classes == {
        "same_chain_distinct_location_cross_label_exact_match"
    }:
        return "same_chain_distinct_location_exact_activation_reuse"
    return "mixed_duplicate_sources"


def _pool_summary(
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> dict[str, Any]:
    labels = sorted({record["behaviour"] for record in records})
    chain_labels: dict[str, set[str]] = {}
    per_label: dict[str, dict[str, int]] = {}
    for label in labels:
        label_indices = [
            index
            for index in indices
            if records[index]["behaviour"] == label
        ]
        per_label[label] = {
            "rows": len(label_indices),
            "chains": len(
                {records[index]["chain_id"] for index in label_indices}
            ),
        }
    for index in indices:
        record = records[index]
        chain_labels.setdefault(record["chain_id"], set()).add(
            record["behaviour"]
        )
    chain_count = len(chain_labels)
    mixed_count = sum(len(value) > 1 for value in chain_labels.values())
    return {
        "rows": len(indices),
        "chains": chain_count,
        "mixed_label_chains": mixed_count,
        "mixed_label_chain_fraction": (
            float(mixed_count / chain_count) if chain_count else 0.0
        ),
        "per_label": per_label,
    }


def _remedy_feasibility(
    records: Sequence[Mapping[str, Any]],
    retained: Sequence[int],
    duplicate_partitions: Sequence[tuple[str, list[int]]],
    *,
    target_label: str,
    minimum_target_chains: int,
) -> dict[str, Any]:
    """Assess a symmetric removal rule using provenance counts only."""
    kept = set(retained)
    applicable_groups = 0
    unsupported_cross_label_groups = 0
    cross_chain_groups = 0
    proposed_rows: set[int] = set()
    within_label_losers: set[int] = set()
    for _, indices in duplicate_partitions:
        group_records = [records[index] for index in indices]
        labels = {record["behaviour"] for record in group_records}
        chains = {record["chain_id"] for record in group_records}
        extraction_windows = {
            (
                record["chain_id"],
                record["char_offset"],
                record["token_start"],
                record["n_positions"],
            )
            for record in group_records
        }
        if len(chains) > 1:
            cross_chain_groups += 1
            continue
        if len(labels) > 1:
            if len(extraction_windows) == 1:
                applicable_groups += 1
                proposed_rows.update(indices)
            else:
                unsupported_cross_label_groups += 1
            continue
        winner = min(
            indices, key=lambda index: _record_lex_key(records[index])
        )
        within_label_losers.update(
            index for index in indices if index != winner
        )

    before_indices = sorted(retained)
    after_indices = sorted(
        kept.difference(proposed_rows).difference(within_label_losers)
    )
    before_summary = _pool_summary(records, before_indices)
    after_summary = _pool_summary(records, after_indices)
    before_target = {
        records[index]["chain_id"]
        for index in before_indices
        if records[index]["behaviour"] == target_label
    }
    after_target = {
        records[index]["chain_id"]
        for index in after_indices
        if records[index]["behaviour"] == target_label
    }
    before_chains = {records[index]["chain_id"] for index in before_indices}
    after_chains = {records[index]["chain_id"] for index in after_indices}
    sample_minimum_met = len(after_target) >= minimum_target_chains
    mixed_gate_met = (
        after_summary["chains"] > 0
        and after_summary["mixed_label_chains"]
        >= 0.5 * after_summary["chains"]
    )
    positive = bool(
        applicable_groups > 0
        and unsupported_cross_label_groups == 0
        and cross_chain_groups == 0
        and not (before_target - after_target)
        and not (before_chains - after_chains)
        and sample_minimum_met
        and mixed_gate_met
    )
    return {
        "proposed_rule": (
            "drop_all_members_of_same_extraction_window_cross_label_groups"
        ),
        "target_label": target_label,
        "minimum_target_chains": minimum_target_chains,
        "applicable_groups": applicable_groups,
        "unsupported_cross_label_groups": unsupported_cross_label_groups,
        "cross_chain_groups": cross_chain_groups,
        "rows_removed_by_proposed_rule": len(proposed_rows),
        "within_label_duplicate_losers": len(within_label_losers),
        "pool_before": before_summary,
        "pool_after": after_summary,
        "target_chains_before": len(before_target),
        "target_chains_after": len(after_target),
        "target_chains_lost": len(before_target - after_target),
        "all_chains_lost": len(before_chains - after_chains),
        "mixed_chain_fraction_after": after_summary[
            "mixed_label_chain_fraction"
        ],
        "sample_minimum_met": sample_minimum_met,
        "mixed_chain_fraction_gate_met": mixed_gate_met,
        "positive": positive,
    }


def audit_exact_vector_provenance(
    activations: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    *,
    input_hashes: Mapping[str, str],
    target_label: str = "backtracking",
    minimum_target_chains: int = 100,
) -> dict[str, Any]:
    """Apply the frozen two-stage duplicate audit and retain provenance only."""
    values = np.asarray(activations, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(records):
        raise ValueError("activation rows and occurrence records are misaligned")
    if not np.isfinite(values).all():
        raise ValueError("activation matrix contains non-finite values")
    normalized = [_normalized_record(record) for record in records]

    occurrence_groups: dict[tuple[Any, ...], list[int]] = {}
    for index, record in enumerate(normalized):
        occurrence_groups.setdefault(_occurrence_key(record), []).append(index)
    occurrence_duplicate_groups = sum(
        len(indices) > 1 for indices in occurrence_groups.values()
    )
    occurrence_duplicate_rows = sum(
        max(0, len(indices) - 1) for indices in occurrence_groups.values()
    )
    retained = sorted(
        min(indices, key=lambda index: _record_lex_key(normalized[index]))
        for indices in occurrence_groups.values()
    )

    digest_partitions: dict[str, list[list[int]]] = {}
    for index in retained:
        vector_digest = _vector_sha256(values[index])
        partitions = digest_partitions.setdefault(vector_digest, [])
        for partition in partitions:
            if np.array_equal(values[index], values[partition[0]]):
                partition.append(index)
                break
        else:
            partitions.append([index])

    duplicate_partitions: list[tuple[str, list[int]]] = []
    for vector_digest, partitions in digest_partitions.items():
        duplicate_partitions.extend(
            (vector_digest, partition)
            for partition in partitions
            if len(partition) > 1
        )
    duplicate_partitions.sort(
        key=lambda item: (
            item[0],
            min(_record_lex_key(normalized[index]) for index in item[1]),
        )
    )

    groups: list[dict[str, Any]] = []
    classifications: dict[str, int] = {}
    for group_index, (vector_digest, indices) in enumerate(duplicate_partitions):
        group_records = [normalized[index] for index in indices]
        labels = sorted({record["behaviour"] for record in group_records})
        chain_ids = sorted({record["chain_id"] for record in group_records})
        annotation_locations = {
            (
                record["chain_id"],
                record["annotation_index"],
                record["char_offset"],
                record["token_start"],
            )
            for record in group_records
        }
        extraction_windows = {
            (
                record["chain_id"],
                record["char_offset"],
                record["token_start"],
                record["n_positions"],
            )
            for record in group_records
        }
        representative = values[indices[0]]
        zero_vector = bool(np.count_nonzero(representative) == 0)
        constant_vector = bool(
            representative.size > 0
            and np.all(representative == representative[0])
        )
        cross_label = len(labels) > 1
        cross_chain = len(chain_ids) > 1
        same_annotation_location = len(annotation_locations) == 1
        same_extraction_window = len(extraction_windows) == 1
        classification = _classify_group(
            cross_label=cross_label,
            cross_chain=cross_chain,
            zero_vector=zero_vector,
            constant_vector=constant_vector,
            same_annotation_location=same_annotation_location,
            same_extraction_window=same_extraction_window,
        )
        classifications[classification] = (
            classifications.get(classification, 0) + 1
        )
        occurrences = []
        for record in sorted(group_records, key=_record_lex_key):
            occurrences.append(
                {
                    "occurrence_sha256": _occurrence_sha256(record),
                    **record,
                }
            )
        groups.append(
            {
                "group_index": group_index,
                "vector_sha256": vector_digest,
                "size": len(indices),
                "labels": labels,
                "chain_ids": chain_ids,
                "cross_label": cross_label,
                "cross_chain": cross_chain,
                "zero_vector": zero_vector,
                "constant_vector": constant_vector,
                "same_location_ignoring_label": same_annotation_location,
                "same_annotation_location_ignoring_label": (
                    same_annotation_location
                ),
                "same_extraction_window": same_extraction_window,
                "classification": classification,
                "occurrences": occurrences,
            }
        )

    summary = {
        "raw_rows": int(values.shape[0]),
        "rows_after_occurrence_collapse": len(retained),
        "occurrence_duplicate_groups": int(occurrence_duplicate_groups),
        "occurrence_duplicate_rows": int(occurrence_duplicate_rows),
        "exact_vector_duplicate_groups": len(groups),
        "exact_vector_duplicate_rows": sum(group["size"] - 1 for group in groups),
        "cross_label_groups": sum(group["cross_label"] for group in groups),
        "cross_chain_groups": sum(group["cross_chain"] for group in groups),
        "zero_vector_groups": sum(group["zero_vector"] for group in groups),
        "constant_vector_groups": sum(
            group["constant_vector"] for group in groups
        ),
        "same_location_multilabel_groups": sum(
            group["classification"] == "same_location_multilabel"
            for group in groups
        ),
        "same_extraction_window_multilabel_groups": sum(
            group["classification"]
            == "same_extraction_window_multilabel_annotation_alias"
            for group in groups
        ),
        "classifications": dict(sorted(classifications.items())),
    }
    remedy_feasibility = _remedy_feasibility(
        normalized,
        retained,
        duplicate_partitions,
        target_label=target_label,
        minimum_target_chains=minimum_target_chains,
    )
    return {
        "schema_version": "thesis-duplicate-provenance-audit-v2",
        "status": "COMPLETE",
        "scope": {
            "model": "R1-1.5B",
            "annotator": "Sonnet",
            "layer_zero_based": 27,
            "pooling": "mean",
            "window": "unclipped",
            "labels": [
                "backtracking",
                "uncertainty-estimation",
                "example-testing",
                "adding-knowledge",
            ],
        },
        "input_hashes": dict(input_hashes),
        "summary": summary,
        "root_cause_classification": _root_cause_classification(groups),
        "remedy_feasibility": remedy_feasibility,
        "groups": groups,
        "execution_boundaries": {
            "estimator_run": False,
            "permutation_run": False,
            "pilot_rerun": False,
            "data_modified": False,
            "raw_vector_values_serialized": False,
        },
    }


def load_registered_l27_pool(
    *,
    repository_root: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Hash-gate and load the registered pool for provenance inspection."""
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else ROOT
    )
    frozen_config = (
        Path(config_path).resolve()
        if config_path is not None
        else CONFIG_PATH
    )
    config = yaml.safe_load(frozen_config.read_text(encoding="utf-8"))
    execution = config["execution"]
    if (
        execution["empirical_pilot_authorized_in_this_batch"] is not False
        or execution["empirical_B50_authorized_in_this_batch"] is not False
        or execution["gpu_enabled"] is not False
        or execution["api_enabled"] is not False
    ):
        raise ValueError("empirical execution gates must remain relocked")
    registered = {
        config["preregistration"]["path"]: config["preregistration"]["sha256"],
        **dict(config["primary_input_hashes"]),
    }
    actual_hashes: dict[str, str] = {}
    for relative_path, expected_digest in registered.items():
        path = root / relative_path
        if not path.is_file():
            raise ValueError("registered audit input is absent")
        actual_digest = _sha256_file(path)
        if actual_digest != expected_digest:
            raise ValueError("registered audit input hash mismatch")
        if relative_path in config["primary_input_hashes"]:
            actual_hashes[relative_path] = actual_digest

    labels = tuple(config["labels"])
    width = _integer(config["model"]["hidden_width"])
    activation_root = root / "data" / "activations" / "R1-1.5B"
    metadata = json.loads(
        (activation_root / "metadata.json").read_text(encoding="utf-8")
    )
    row_index = json.loads(
        (activation_root / "row_index.json").read_text(encoding="utf-8")
    )
    if (
        metadata.get("pooling") != "mean"
        or metadata.get("clip_window_to_sentence_end") is not False
        or metadata.get("sentence_matching") != "occurrence_aware_v1"
        or row_index.get("pooling") != "mean"
        or row_index.get("clip_window_to_sentence_end") is not False
        or row_index.get("sentence_matching") != "occurrence_aware_v1"
    ):
        raise ValueError("registered audit representation mismatch")

    matrix_parts: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    rows_by_label = row_index["rows"]
    for label in labels:
        registered_rows = rows_by_label[label]
        matrix = np.load(
            activation_root / f"{label}_layer27.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        if (
            matrix.ndim != 2
            or matrix.shape[0] != len(registered_rows)
            or matrix.shape[1] != width
        ):
            raise ValueError("registered audit row alignment failure")
        matrix_parts.append(np.asarray(matrix))
        for source_index, raw_record in enumerate(registered_rows):
            records.append(
                _normalized_record(
                    {
                        "behaviour": label,
                        "chain_id": raw_record["chain_id"],
                        "annotation_index": raw_record["annotation_index"],
                        "char_offset": raw_record["char_offset"],
                        "token_start": raw_record["token_start"],
                        "n_positions": raw_record.get("n_positions", 0),
                        "source_index": source_index,
                    }
                )
            )
    return {
        "activations": np.concatenate(matrix_parts, axis=0),
        "records": records,
        "input_hashes": actual_hashes,
    }


def _atomic_write(path: Path, text: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_audit_artifacts(
    json_path: str | os.PathLike[str],
    markdown_path: str | os.PathLike[str],
    report: Mapping[str, Any],
) -> None:
    """Write the complete provenance JSON and a compact human-readable summary."""
    json_text = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    summary = report["summary"]
    feasibility = report["remedy_feasibility"]
    markdown = "\n".join(
        [
            "# Exact-vector duplicate provenance audit",
            "",
            f"- Status: `{report['status']}`",
            f"- Root-cause classification: `{report['root_cause_classification']}`",
            f"- Raw rows checked: `{summary['raw_rows']}`",
            f"- Rows after occurrence collapse: `{summary['rows_after_occurrence_collapse']}`",
            f"- Repeated-occurrence groups: `{summary['occurrence_duplicate_groups']}`",
            f"- Exact-vector duplicate groups: `{summary['exact_vector_duplicate_groups']}`",
            f"- Cross-label groups: `{summary['cross_label_groups']}`",
            f"- Cross-chain groups: `{summary['cross_chain_groups']}`",
            f"- Zero-vector groups: `{summary['zero_vector_groups']}`",
            f"- Constant-vector groups: `{summary['constant_vector_groups']}`",
            f"- Same-location multilabel groups: `{summary['same_location_multilabel_groups']}`",
            f"- Same-window multilabel groups: `{summary['same_extraction_window_multilabel_groups']}`",
            "",
            "## Classifications",
            "",
            *(
                [
                    f"- `{name}`: `{count}`"
                    for name, count in summary["classifications"].items()
                ]
                or ["- None"]
            ),
            "",
            "## Remedy feasibility",
            "",
            f"- Proposed rule: `{feasibility['proposed_rule']}`",
            f"- Applicable groups: `{feasibility['applicable_groups']}`",
            f"- Unsupported cross-label groups: `{feasibility['unsupported_cross_label_groups']}`",
            f"- Rows removed by proposed rule: `{feasibility['rows_removed_by_proposed_rule']}`",
            f"- Target chains before/after: `{feasibility['target_chains_before']}` / `{feasibility['target_chains_after']}`",
            f"- Target chains lost: `{feasibility['target_chains_lost']}`",
            f"- All chains lost: `{feasibility['all_chains_lost']}`",
            f"- Mixed-label chain fraction after: `{feasibility['mixed_chain_fraction_after']:.6f}`",
            f"- Sample minimum met: `{str(feasibility['sample_minimum_met']).upper()}`",
            f"- Mixed-chain gate met: `{str(feasibility['mixed_chain_fraction_gate_met']).upper()}`",
            f"- Positive: `{str(feasibility['positive']).upper()}`",
            "",
            "## Boundaries",
            "",
            "- Vector values are excluded; only cryptographic identities and occurrence provenance are recorded.",
            "- No estimator, permutation, pilot retry, data modification, GPU, or API operation was performed.",
            "- Group-level occurrence details are in the companion JSON.",
            "",
        ]
    )
    _atomic_write(Path(json_path), json_text)
    _atomic_write(Path(markdown_path), markdown)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument(
        "--markdown-output", type=Path, default=DEFAULT_MARKDOWN
    )
    args = parser.parse_args(argv)
    loaded = load_registered_l27_pool()
    report = audit_exact_vector_provenance(
        loaded["activations"],
        loaded["records"],
        input_hashes=loaded["input_hashes"],
    )
    write_audit_artifacts(args.json_output, args.markdown_output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
