"""Preregistered thesis core-hardening analysis harness.

The module is intentionally data-source agnostic: execution functions accept
already aligned in-memory rows.  Loading and hash-gating the empirical inputs
is a separate, explicit CLI action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import socket
import sys
import tempfile
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


class EstimatorInvalidError(ValueError):
    """The registered estimator is undefined for an otherwise valid assignment."""


class DuplicateAuditError(ValueError):
    """Hard duplicate conflict carrying the complete structured audit."""

    def __init__(self, message: str, audit: Mapping[str, Any]):
        super().__init__(message)
        self.audit = dict(audit)


def _validate_behaviour_code(behaviour_code: int) -> int:
    code = _json_integer(behaviour_code)
    if code not in {1, 2, 3, 4}:
        raise ValueError("cell/code registry violation: behaviour code must be 1..4")
    return code


def _validate_cell_registry(
    *,
    family_code: int,
    cell_index: int,
    behaviour_code: int,
    annotator_code: int,
    layer: int,
    pooling_code: int,
    window_code: int,
    truncation_code: int,
    scope_key: str = "",
    purpose_code: int | None = None,
) -> None:
    """Enforce the exact Section 9 family/cell/code registry."""
    family = _json_integer(family_code)
    cell = _json_integer(cell_index)
    behaviour = _validate_behaviour_code(behaviour_code)
    annotator = _json_integer(annotator_code)
    frozen_layer = _json_integer(layer)
    pooling = _json_integer(pooling_code)
    window = _json_integer(window_code)
    truncation = _json_integer(truncation_code)
    if purpose_code is not None:
        purpose = _json_integer(purpose_code)
        permitted_families = {
            1: {1, 2, 3, 5},
            2: {1},
            3: {4},
            4: {6},
        }
        if purpose not in permitted_families or family not in permitted_families[purpose]:
            raise ValueError("cell/code registry violation: estimator purpose")
    valid = False
    if family == 1 and 0 <= cell <= 3:
        valid = (
            behaviour == cell + 1
            and annotator == 1
            and frozen_layer == 27
            and pooling == 1
            and window == 1
            and truncation == 0
        )
    elif family == 2 and 4 <= cell <= 23:
        offset = cell - 4
        expected_behaviour = offset // 5 + 1
        expected_layer = (11, 14, 17, 20, 27)[offset % 5]
        valid = (
            behaviour == expected_behaviour
            and annotator == 1
            and frozen_layer == expected_layer
            and pooling == 1
            and window == 1
            and truncation == 0
        )
    elif family == 3 and 24 <= cell <= 47:
        offset = cell - 24
        expected_behaviour = offset % 4 + 1
        arm = offset // 4
        expected_annotator = arm // 2 + 1
        expected_layer = (12, 16)[arm % 2]
        valid = (
            behaviour == expected_behaviour
            and annotator == expected_annotator
            and frozen_layer == expected_layer
            and pooling == 1
            and window == 1
            and truncation == 0
        )
    elif family == 4 and 48 <= cell <= 63:
        offset = cell - 48
        expected_behaviour = offset % 4 + 1
        expected_truncation = offset // 4 + 1
        valid = (
            behaviour == expected_behaviour
            and annotator == 1
            and frozen_layer == 27
            and pooling == 0
            and window == 0
            and truncation == expected_truncation
        )
    elif family == 5 and 64 <= cell <= 87:
        offset = cell - 64
        expected_behaviour = offset % 4 + 1
        representation = offset // 4
        expected_pooling = representation // 2 + 1
        expected_window = representation % 2 + 1
        neutral_h4 = (
            pooling == 0 and window == 0 and _nfc(scope_key) == "H4-common"
        )
        valid = (
            behaviour == expected_behaviour
            and annotator == 1
            and frozen_layer == 27
            and truncation == 0
            and (
                neutral_h4
                or (pooling == expected_pooling and window == expected_window)
            )
        )
    elif family == 6 and 88 <= cell <= 91:
        valid = (
            behaviour == cell - 88 + 1
            and annotator == 1
            and frozen_layer == 16
            and pooling == 0
            and window == 0
            and truncation == 0
        )
    if not valid:
        raise ValueError("cell/code registry violation")


def correlation_dimension_point(X: np.ndarray) -> float:
    """Return the frozen correlation-dimension point estimate."""
    from src.intrinsic_dim import correlation_dimension_estimate

    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 100:
        raise EstimatorInvalidError(
            "correlation dimension requires a 2-D matrix with N >= 100"
        )
    if not np.isfinite(values).all():
        raise EstimatorInvalidError(
            "correlation dimension input contains non-finite values"
        )
    try:
        estimate = float(
            correlation_dimension_estimate(
                values,
                n_radii=20,
                random_state=42,
                n_bootstrap=0,
                subsample=2000,
            ).estimate
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise EstimatorInvalidError(
            "correlation dimension numeric domain is invalid"
        ) from exc
    if not np.isfinite(estimate) or estimate <= 0:
        raise EstimatorInvalidError("correlation dimension estimate is invalid")
    return estimate


def _participation_ratio(X: np.ndarray) -> float:
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise EstimatorInvalidError(
            "participation ratio requires a 2-D matrix with N >= 2"
        )
    if not np.isfinite(values).all():
        raise EstimatorInvalidError(
            "participation ratio input contains non-finite values"
        )
    centered = values - np.mean(values, axis=0, dtype=np.float64)
    try:
        singular_values = np.linalg.svd(
            centered, full_matrices=False, compute_uv=False
        )
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise EstimatorInvalidError(
            "participation ratio numeric domain is invalid"
        ) from exc
    eigenvalues = singular_values**2 / (values.shape[0] - 1)
    denominator = float(np.sum(eigenvalues**2, dtype=np.float64))
    numerator = float(np.sum(eigenvalues, dtype=np.float64)) ** 2
    if (
        not np.isfinite(eigenvalues).all()
        or not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or denominator <= 0
    ):
        raise EstimatorInvalidError("participation ratio is degenerate")
    result = numerator / denominator
    if not np.isfinite(result) or result <= 0:
        raise EstimatorInvalidError("participation ratio is invalid")
    return float(result)


def _nfc(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("digest string fields must be strings")
    return unicodedata.normalize("NFC", value)


def _json_integer(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("digest integer fields must be integers")
    return int(value)


def canonical_occurrence_digest(
    *,
    family_code: int,
    annotator_code: int,
    layer: int,
    behaviour_code: int,
    pooling_code: int,
    window_code: int,
    truncation_code: int,
    chain_id: str,
    annotation_index: int,
    char_offset: int,
    token_start: int,
    replicate_index: int,
    scope_key: str,
) -> str:
    """Hash the exact Section 7 canonical occurrence payload."""
    payload = [
        "thesis-core-hardening-v1",
        *[
            _json_integer(value)
            for value in (
                family_code,
                annotator_code,
                layer,
                behaviour_code,
                pooling_code,
                window_code,
                truncation_code,
            )
        ],
        _nfc(chain_id),
        *[
            _json_integer(value)
            for value in (
                annotation_index,
                char_offset,
                token_start,
                replicate_index,
            )
        ],
        _nfc(scope_key),
    ]
    serialized = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _permutation_rng(
    base_seed: int, family_code: int, cell_index: int, attempt_index: int
) -> np.random.Generator:
    seed = np.random.SeedSequence(
        [
            _json_integer(base_seed),
            1,
            _json_integer(family_code),
            _json_integer(cell_index),
            _json_integer(attempt_index),
        ]
    )
    return np.random.default_rng(seed)


def _chain_stability_rng(
    family_code: int, cell_index: int, replicate_index: int
) -> np.random.Generator:
    seed = np.random.SeedSequence(
        [
            20260726,
            2,
            _json_integer(family_code),
            _json_integer(cell_index),
            _json_integer(replicate_index),
        ]
    )
    return np.random.default_rng(seed)


def _lower_tail_p(observed: float, null_values: Sequence[float]) -> float:
    obs = float(observed)
    null = np.asarray(null_values, dtype=np.float64)
    if not np.isfinite(obs) or null.ndim != 1 or not np.isfinite(null).all():
        raise ValueError("lower-tail p-value requires finite statistics")
    return float((1 + np.count_nonzero(null <= obs)) / (null.size + 1))


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Holm correction requires finite p-values")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty(values.size, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (values.size - rank) * values[index]
        running = max(running, float(candidate))
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty(values.size, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


_OCCURRENCE_FIELDS = (
    "behaviour",
    "chain_id",
    "annotation_index",
    "char_offset",
    "token_start",
)


def _occurrence_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    missing = [field for field in _OCCURRENCE_FIELDS if field not in record]
    if missing:
        raise ValueError(f"row alignment missing occurrence fields: {missing}")
    behaviour = _nfc(record["behaviour"])
    chain_id = _nfc(record["chain_id"])
    if not chain_id:
        raise ValueError("row alignment contains an empty chain id")
    if chain_id.casefold().startswith(("proxy", "row-", "index-")):
        raise ValueError("proxy chain ids are prohibited")
    return (
        behaviour,
        chain_id,
        _json_integer(record["annotation_index"]),
        _json_integer(record["char_offset"]),
        _json_integer(record["token_start"]),
    )


def _record_lex_key(record: Mapping[str, Any]) -> bytes:
    normalized = {
        str(key): (
            _nfc(value)
            if isinstance(value, str)
            else int(value)
            if isinstance(value, np.integer)
            else value
        )
        for key, value in record.items()
    }
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validate_and_deduplicate(
    activations: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    *,
    cross_label_policy: str = "hard_fail",
) -> dict[str, Any]:
    """Validate exact row alignment and apply the frozen two-stage audit."""
    if cross_label_policy not in {
        "hard_fail",
        "drop_all_same_extraction_window",
    }:
        raise ValueError("unknown cross-label duplicate policy")
    values = np.asarray(activations, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(records):
        raise ValueError(
            "row alignment failure: activation rows and occurrence records differ"
        )
    if not np.isfinite(values).all():
        raise ValueError("row alignment failure: activation values are non-finite")

    normalized_records = [dict(record) for record in records]
    occurrence_keys = [_occurrence_key(record) for record in normalized_records]
    occurrence_groups: dict[tuple[Any, ...], list[int]] = {}
    for index, key in enumerate(occurrence_keys):
        occurrence_groups.setdefault(key, []).append(index)

    keep_after_occurrence: list[int] = []
    occurrence_duplicate_groups = 0
    occurrence_duplicate_rows = 0
    for indices in occurrence_groups.values():
        if len(indices) > 1:
            occurrence_duplicate_groups += 1
            occurrence_duplicate_rows += len(indices) - 1
        winner = min(indices, key=lambda index: _record_lex_key(normalized_records[index]))
        keep_after_occurrence.append(winner)
    keep_after_occurrence.sort()

    vector_groups: dict[tuple[float, ...], list[int]] = {}
    for index in keep_after_occurrence:
        vector_groups.setdefault(tuple(values[index].tolist()), []).append(index)

    vector_duplicate_groups = 0
    vector_duplicate_rows = 0
    cross_chain_groups = 0
    cross_label_groups = 0
    cross_label_same_window_groups_dropped = 0
    cross_label_same_window_rows_dropped = 0
    unresolved_cross_label_groups = 0
    vector_winners: set[int] = set()
    for indices in vector_groups.values():
        if len(indices) == 1:
            vector_winners.add(indices[0])
            continue
        vector_duplicate_groups += 1
        vector_duplicate_rows += len(indices) - 1
        keys = [occurrence_keys[index] for index in indices]
        chains = {key[1] for key in keys}
        labels = {key[0] for key in keys}
        if len(chains) > 1:
            cross_chain_groups += 1
        if len(labels) > 1:
            cross_label_groups += 1
        if len(chains) > 1:
            winner = min(indices, key=lambda index: occurrence_keys[index])
            vector_winners.add(winner)
            continue
        if len(labels) > 1:
            if cross_label_policy == "drop_all_same_extraction_window":
                extraction_windows = {
                    (
                        key[1],
                        key[3],
                        key[4],
                        _json_integer(
                            normalized_records[index]["n_positions"]
                        ),
                    )
                    for index, key in zip(indices, keys)
                }
                if len(extraction_windows) == 1:
                    cross_label_same_window_groups_dropped += 1
                    cross_label_same_window_rows_dropped += len(indices)
                    continue
            unresolved_cross_label_groups += 1
        winner = min(indices, key=lambda index: occurrence_keys[index])
        vector_winners.add(winner)

    kept_indices = [
        index for index in keep_after_occurrence if index in vector_winners
    ]
    n_raw = int(values.shape[0])
    n_kept = len(kept_indices)
    audit = {
        "raw_rows": n_raw,
        "deduplicated_rows": n_kept,
        "duplicate_fraction": float((n_raw - n_kept) / n_raw) if n_raw else 0.0,
        "occurrence_duplicate_groups": occurrence_duplicate_groups,
        "occurrence_duplicate_rows": occurrence_duplicate_rows,
        "vector_duplicate_groups": vector_duplicate_groups,
        "vector_duplicate_rows": vector_duplicate_rows,
        "cross_chain_groups": cross_chain_groups,
        "cross_label_groups": cross_label_groups,
        "cross_label_policy": cross_label_policy,
        "cross_label_same_window_groups_dropped": (
            cross_label_same_window_groups_dropped
        ),
        "cross_label_same_window_rows_dropped": (
            cross_label_same_window_rows_dropped
        ),
        "unresolved_cross_label_groups": unresolved_cross_label_groups,
    }
    if cross_chain_groups or unresolved_cross_label_groups:
        conflict = "cross-chain" if cross_chain_groups else "cross-label"
        raise DuplicateAuditError(
            f"{conflict} exact-vector duplicate hard failure", audit
        )
    return {
        "activations": values[kept_indices],
        "records": [normalized_records[index] for index in kept_indices],
        "kept_indices": kept_indices,
        "audit": audit,
    }


def _select_equal_chain_indices(
    records: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    *,
    target_label: str,
    family_code: int,
    annotator_code: int = 1,
    layer: int = 27,
    behaviour_code: int,
    pooling_code: int = 1,
    window_code: int = 1,
    truncation_code: int = 0,
    replicate_index: int = -1,
    scope_key: str = "",
) -> list[int]:
    """Select at most one target-labelled occurrence per chain by digest."""
    if len(records) != len(labels):
        raise ValueError("row alignment failure in equal-chain selection")
    target = _nfc(target_label)
    candidates: dict[str, list[tuple[str, int]]] = {}
    for index, (record, assigned_label) in enumerate(zip(records, labels)):
        if _nfc(str(assigned_label)) != target:
            continue
        key = _occurrence_key(record)
        digest = canonical_occurrence_digest(
            family_code=family_code,
            annotator_code=annotator_code,
            layer=layer,
            behaviour_code=behaviour_code,
            pooling_code=pooling_code,
            window_code=window_code,
            truncation_code=truncation_code,
            chain_id=key[1],
            annotation_index=key[2],
            char_offset=key[3],
            token_start=key[4],
            replicate_index=replicate_index,
            scope_key=scope_key,
        )
        candidates.setdefault(key[1], []).append((digest, index))
    selected = [min(items)[1] for items in candidates.values()]
    return sorted(
        selected,
        key=lambda index: _nfc(str(records[index]["chain_id"])).encode("utf-8"),
    )


_REPRESENTATION_ORDER = (
    ("mean", "unclipped"),
    ("mean", "clipped"),
    ("first", "unclipped"),
    ("first", "clipped"),
    ("last", "unclipped"),
    ("last", "clipped"),
)


def _representation_key(value: Any) -> tuple[str, str]:
    if isinstance(value, str) and "/" in value:
        pooling, window = value.split("/", 1)
        return _nfc(pooling), _nfc(window)
    if isinstance(value, tuple) and len(value) == 2:
        return _nfc(value[0]), _nfc(value[1])
    raise ValueError("representation keys must be (pooling, window) pairs")


def _occurrence_list_digest(keys: Sequence[Sequence[Any]]) -> str:
    payload = [
        [
            _nfc(str(key[0])),
            _nfc(str(key[1])),
            _json_integer(key[2]),
            _json_integer(key[3]),
            _json_integer(key[4]),
        ]
        for key in keys
    ]
    serialized = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _assignment_digest(labels: Sequence[str]) -> str:
    """Hash an assignment without materializing or retaining a label list."""
    digest = hashlib.sha256()
    digest.update(b"thesis-core-hardening-assignment-v1\0")
    for raw_label in labels:
        encoded = _nfc(str(raw_label)).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _neutral_h4_digest(
    occurrence_key: Sequence[Any], behaviour_code: int
) -> str:
    return canonical_occurrence_digest(
        family_code=5,
        annotator_code=1,
        layer=27,
        behaviour_code=behaviour_code,
        pooling_code=0,
        window_code=0,
        truncation_code=0,
        chain_id=occurrence_key[1],
        annotation_index=occurrence_key[2],
        char_offset=occurrence_key[3],
        token_start=occurrence_key[4],
        replicate_index=-1,
        scope_key="H4-common",
    )


def _aligned_record_identity(record: Mapping[str, Any]) -> bytes:
    """Representation-neutral row identity for one-time occurrence collapse."""
    for field in ("record_id", "row_index", "source_index"):
        if field in record:
            value = record[field]
            if isinstance(value, str):
                value = _nfc(value)
            elif isinstance(value, np.integer):
                value = int(value)
            return json.dumps(
                [field, value],
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
    return _record_lex_key(record)


def build_common_complete_case_grid(
    representations: Mapping[Any, Mapping[str, Any]],
    *,
    behaviour_codes: Mapping[str, int],
) -> dict[str, Any]:
    """Build the joint six-representation H4 complete-case grid."""
    canonical_representations = {
        _representation_key(key): value for key, value in representations.items()
    }
    if set(canonical_representations) != set(_REPRESENTATION_ORDER):
        raise ValueError("all six mean/first/last x clipped/unclipped representations are required")
    normalized_codes = {
        _nfc(str(label)): _json_integer(code)
        for label, code in behaviour_codes.items()
    }
    if not normalized_codes:
        raise ValueError("behaviour code registry is empty")
    if (
        any(code not in {1, 2, 3, 4} for code in normalized_codes.values())
        or len(set(normalized_codes.values())) != len(normalized_codes)
    ):
        raise ValueError("cell/code registry violation in H4 behaviour codes")

    raw_by_representation: dict[tuple[str, str], dict[str, Any]] = {}
    for representation in _REPRESENTATION_ORDER:
        cell = canonical_representations[representation]
        values = np.asarray(cell.get("activations"), dtype=np.float64)
        records = list(cell.get("records", []))
        if values.ndim != 2 or values.shape[0] != len(records):
            raise ValueError(f"row alignment failure in {representation}")
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite activation in {representation}")
        groups: dict[tuple[Any, ...], list[int]] = {}
        for index, record in enumerate(records):
            key = _occurrence_key(record)
            if key[0] not in normalized_codes:
                raise ValueError(f"unregistered behaviour {key[0]!r}")
            groups.setdefault(key, []).append(index)
        raw_by_representation[representation] = {
            "values": values,
            "records": records,
            "groups": groups,
        }

    intersection_keys = set.intersection(
        *(
            set(raw_by_representation[representation]["groups"])
            for representation in _REPRESENTATION_ORDER
        )
    )
    per_representation: dict[
        tuple[str, str], dict[tuple[Any, ...], tuple[dict[str, Any], np.ndarray]]
    ] = {representation: {} for representation in _REPRESENTATION_ORDER}
    repeated_occurrence_losers = {
        representation: 0 for representation in _REPRESENTATION_ORDER
    }
    unaligned_occurrence_keys: set[tuple[Any, ...]] = set()
    for key in intersection_keys:
        identity_maps: dict[tuple[str, str], dict[bytes, list[int]]] = {}
        for representation in _REPRESENTATION_ORDER:
            raw = raw_by_representation[representation]
            identity_map: dict[bytes, list[int]] = {}
            indices = raw["groups"][key]
            repeated_occurrence_losers[representation] += len(indices) - 1
            for index in indices:
                identity_map.setdefault(
                    _aligned_record_identity(raw["records"][index]), []
                ).append(index)
            identity_maps[representation] = identity_map
        common_identities = set.intersection(
            *(set(identity_maps[representation]) for representation in _REPRESENTATION_ORDER)
        )
        if not common_identities:
            unaligned_occurrence_keys.add(key)
            continue
        retained_identity = min(common_identities)
        for representation in _REPRESENTATION_ORDER:
            raw = raw_by_representation[representation]
            winner = min(
                identity_maps[representation][retained_identity],
                key=lambda index: _record_lex_key(raw["records"][index]),
            )
            per_representation[representation][key] = (
                dict(raw["records"][winner]),
                raw["values"][winner],
            )

    common_keys = intersection_keys - unaligned_occurrence_keys
    joint_vector_losers: set[tuple[Any, ...]] = set()
    vector_duplicate_groups: dict[str, int] = {}
    cross_chain_groups = 0
    cross_label_groups = 0
    for representation in _REPRESENTATION_ORDER:
        vector_groups: dict[tuple[float, ...], list[tuple[Any, ...]]] = {}
        for key in common_keys:
            vector = per_representation[representation][key][1]
            vector_groups.setdefault(tuple(vector.tolist()), []).append(key)
        n_groups = 0
        for keys in vector_groups.values():
            if len(keys) == 1:
                continue
            n_groups += 1
            cross_chain = len({key[1] for key in keys}) > 1
            cross_label = len({key[0] for key in keys}) > 1
            cross_chain_groups += int(cross_chain)
            cross_label_groups += int(cross_label)
            if cross_chain or cross_label:
                winner = min(keys)
            else:
                behaviour_code = normalized_codes[keys[0][0]]
                winner = min(
                    keys,
                    key=lambda key: _neutral_h4_digest(key, behaviour_code),
                )
            joint_vector_losers.update(key for key in keys if key != winner)
        vector_duplicate_groups[f"{representation[0]}/{representation[1]}"] = n_groups

    surviving_keys = common_keys - joint_vector_losers
    raw_rows = sum(
        len(raw_by_representation[representation]["records"])
        for representation in _REPRESENTATION_ORDER
    )
    duplicate_rows_removed = sum(
        repeated_occurrence_losers.values()
    ) + len(joint_vector_losers) * len(_REPRESENTATION_ORDER)
    deduplicated_rows = max(0, raw_rows - duplicate_rows_removed)
    duplicate_audit = {
        "raw_rows": raw_rows,
        "deduplicated_rows": deduplicated_rows,
        "duplicate_fraction": (
            float((raw_rows - deduplicated_rows) / raw_rows)
            if raw_rows
            else 0.0
        ),
        "occurrence_duplicate_groups": sum(
            1
            for representation in _REPRESENTATION_ORDER
            for indices in raw_by_representation[representation]["groups"].values()
            if len(indices) > 1
        ),
        "occurrence_duplicate_rows": sum(
            repeated_occurrence_losers.values()
        ),
        "vector_duplicate_groups": sum(vector_duplicate_groups.values()),
        "vector_duplicate_rows": len(joint_vector_losers),
        "cross_chain_groups": cross_chain_groups,
        "cross_label_groups": cross_label_groups,
    }
    if cross_chain_groups or cross_label_groups:
        conflict = "cross-chain" if cross_chain_groups else "cross-label"
        raise DuplicateAuditError(
            f"{conflict} exact-vector duplicate hard failure", duplicate_audit
        )
    common_order = sorted(
        surviving_keys,
        key=lambda key: (
            key[0].encode("utf-8"),
            key[1].encode("utf-8"),
            key[2:],
        ),
    )
    common_representations: dict[str, dict[str, Any]] = {}
    for representation in _REPRESENTATION_ORDER:
        name = f"{representation[0]}/{representation[1]}"
        common_representations[name] = {
            "records": [
                per_representation[representation][key][0] for key in common_order
            ],
            "activations": np.asarray(
                [per_representation[representation][key][1] for key in common_order],
                dtype=np.float64,
            ),
        }

    behaviour_results: dict[str, Any] = {}
    for label, behaviour_code in sorted(
        normalized_codes.items(), key=lambda item: item[1]
    ):
        all_chains = {
            key[1]
            for representation in _REPRESENTATION_ORDER
            for key in raw_by_representation[representation]["groups"]
            if key[0] == label
        }
        selected_keys: list[tuple[Any, ...]] = []
        chain_groups: dict[str, list[tuple[Any, ...]]] = {}
        for key in common_order:
            if key[0] == label:
                chain_groups.setdefault(key[1], []).append(key)
        for chain_id in sorted(chain_groups, key=lambda value: value.encode("utf-8")):
            selected_keys.append(
                min(
                    chain_groups[chain_id],
                    key=lambda key: _neutral_h4_digest(key, behaviour_code),
                )
            )
        digest = _occurrence_list_digest(selected_keys)
        selected_chains = {key[1] for key in selected_keys}
        attrition_reasons: dict[str, list[str]] = {}
        for chain_id in sorted(
            all_chains - selected_chains, key=lambda value: value.encode("utf-8")
        ):
            missing_representations = []
            for representation in _REPRESENTATION_ORDER:
                has_chain = any(
                    key[0] == label and key[1] == chain_id
                    for key in raw_by_representation[representation]["groups"]
                )
                if not has_chain:
                    missing_representations.append(
                        f"{representation[0]}/{representation[1]}"
                    )
            if missing_representations:
                attrition_reasons[chain_id] = [
                    f"missing_complete_case_representation:{name}"
                    for name in missing_representations
                ]
            elif any(
                key[0] == label
                and key[1] == chain_id
                and key in unaligned_occurrence_keys
                for key in intersection_keys
            ):
                attrition_reasons[chain_id] = [
                    "no_fully_aligned_repeated_occurrence"
                ]
            elif any(
                key[0] == label
                and key[1] == chain_id
                and key in joint_vector_losers
                for key in common_keys
            ):
                attrition_reasons[chain_id] = ["joint_duplicate_loser"]
            else:
                attrition_reasons[chain_id] = [
                    "no_exact_occurrence_intersection"
                ]
        representation_results: dict[str, Any] = {}
        for representation in _REPRESENTATION_ORDER:
            name = f"{representation[0]}/{representation[1]}"
            representation_results[name] = {
                "activations": np.asarray(
                    [
                        per_representation[representation][key][1]
                        for key in selected_keys
                    ],
                    dtype=np.float64,
                ),
                "occurrence_keys": [list(key) for key in selected_keys],
                "occurrence_list_sha256": digest,
            }
        behaviour_results[label] = {
            "behaviour_code": behaviour_code,
            "chain_ids": [key[1] for key in selected_keys],
            "occurrence_keys": [list(key) for key in selected_keys],
            "occurrence_list_sha256": digest,
            "chain_attrition_reasons": attrition_reasons,
            "representations": representation_results,
        }

    return {
        "status": "VALID",
        "representation_order": [
            f"{pooling}/{window}" for pooling, window in _REPRESENTATION_ORDER
        ],
        "behaviour_codes": normalized_codes,
        "behaviours": behaviour_results,
        "common_representations": common_representations,
        "duplicate_audit": duplicate_audit,
        "audit": {
            "intersection_rows": len(common_keys),
            "unaligned_occurrence_rows": len(unaligned_occurrence_keys),
            "joint_vector_loser_rows": len(joint_vector_losers),
            "surviving_rows": len(surviving_keys),
            "occurrence_duplicate_losers": {
                f"{key[0]}/{key[1]}": value
                for key, value in repeated_occurrence_losers.items()
            },
            "vector_duplicate_groups": vector_duplicate_groups,
        },
    }


def run_within_chain_cdim_null(
    activations: np.ndarray,
    labels: Sequence[str],
    chain_ids: Sequence[str],
    occurrences: Sequence[Mapping[str, Any]],
    *,
    target_label: str,
    family_code: int,
    cell_index: int,
    behaviour_code: int,
    B: int = 2500,
    attempt_cap: int = 2750,
    base_seed: int = 20260726,
    annotator_code: int = 1,
    layer: int = 27,
    pooling_code: int = 1,
    window_code: int = 1,
    truncation_code: int = 0,
    scope_key: str = "",
    cross_label_policy: str = "hard_fail",
    statistic_fn: Callable[[np.ndarray], float] = correlation_dimension_point,
) -> dict[str, Any]:
    """Run the registered conditional-valid within-chain permutation null."""
    _validate_cell_registry(
        family_code=family_code,
        cell_index=cell_index,
        behaviour_code=behaviour_code,
        annotator_code=annotator_code,
        layer=layer,
        pooling_code=pooling_code,
        window_code=window_code,
        truncation_code=truncation_code,
        scope_key=scope_key,
        purpose_code=1,
    )
    if base_seed not in {20260726, 20260727}:
        raise ValueError("cell/code registry violation: unregistered base seed")
    values = np.asarray(activations, dtype=np.float64)
    assigned = np.asarray(labels, dtype=str)
    chains = np.asarray(chain_ids, dtype=str)
    if not (
        values.ndim == 2
        and len(assigned) == len(chains) == len(occurrences) == values.shape[0]
    ):
        raise ValueError("row alignment failure before permutation")
    for index, record in enumerate(occurrences):
        key = _occurrence_key(record)
        if key[0] != _nfc(str(assigned[index])) or key[1] != _nfc(str(chains[index])):
            raise ValueError("row alignment failure between arrays and occurrence keys")

    target = _nfc(target_label)
    raw_target_chains = {
        _nfc(str(chains[index]))
        for index in range(len(chains))
        if _nfc(str(assigned[index])) == target
    }
    audited = _validate_and_deduplicate(
        values,
        occurrences,
        cross_label_policy=cross_label_policy,
    )
    kept = np.asarray(audited["kept_indices"], dtype=int)
    values = audited["activations"]
    records = audited["records"]
    assigned = assigned[kept]
    chains = chains[kept]

    B = _json_integer(B)
    attempt_cap = _json_integer(attempt_cap)
    if B < 0 or attempt_cap < B:
        raise ValueError("attempt cap must be at least B and both must be non-negative")
    unique_chains = sorted(
        {_nfc(str(chain)) for chain in chains}, key=lambda value: value.encode("utf-8")
    )
    chain_to_indices = {
        chain: np.flatnonzero(chains == chain) for chain in unique_chains
    }
    mixed = sum(
        1 for indices in chain_to_indices.values() if np.unique(assigned[indices]).size > 1
    )
    if not unique_chains or mixed < 0.5 * len(unique_chains):
        raise ValueError("within-chain permutation informativeness gate failed")

    selector_kwargs = {
        "target_label": target_label,
        "family_code": family_code,
        "annotator_code": annotator_code,
        "layer": layer,
        "behaviour_code": behaviour_code,
        "pooling_code": pooling_code,
        "window_code": window_code,
        "truncation_code": truncation_code,
        "replicate_index": -1,
        "scope_key": scope_key,
    }
    observed_indices = _select_equal_chain_indices(
        records, assigned, **selector_kwargs
    )
    if not observed_indices:
        raise ValueError("observed assignment has no eligible target chains")
    try:
        observed = float(statistic_fn(values[observed_indices]))
    except EstimatorInvalidError as exc:
        raise EstimatorInvalidError(
            "observed assignment is estimator-invalid"
        ) from exc
    if not np.isfinite(observed) or observed <= 0:
        raise EstimatorInvalidError("observed assignment is estimator-invalid")
    observed_chains = {chains[index] for index in observed_indices}
    target_rows_after_deduplication = int(np.count_nonzero(assigned == target))
    chain_attrition_reasons = {
        chain_id: ["all_target_occurrences_removed_by_duplicate_remedy"]
        for chain_id in sorted(
            raw_target_chains.difference(
                {_nfc(str(chain)) for chain in observed_chains}
            ),
            key=lambda value: value.encode("utf-8"),
        )
    }
    occurrence_list_sha256 = _occurrence_list_digest(
        [_occurrence_key(records[index]) for index in observed_indices]
    )

    attempts: list[dict[str, Any]] = []
    null_values: list[float] = []
    invalid_reasons: dict[str, int] = {}
    for attempt_index in range(attempt_cap):
        if len(null_values) >= B:
            break
        rng = _permutation_rng(
            base_seed, family_code, cell_index, attempt_index
        )
        proposed = assigned.copy()
        for indices in chain_to_indices.values():
            proposed[indices] = rng.permutation(assigned[indices])

        for indices in chain_to_indices.values():
            if sorted(proposed[indices].tolist()) != sorted(assigned[indices].tolist()):
                raise RuntimeError("within-chain label counts changed")
        selected = _select_equal_chain_indices(
            records, proposed, **selector_kwargs
        )
        if {chains[index] for index in selected} != observed_chains:
            raise RuntimeError("target-chain eligibility changed under permutation")
        identity = bool(np.array_equal(proposed, assigned))
        attempt_record: dict[str, Any] = {
            "attempt_index": attempt_index,
            "assignment_sha256": _assignment_digest(proposed),
            "identity": identity,
        }
        try:
            statistic = float(statistic_fn(values[selected]))
        except EstimatorInvalidError:
            statistic = float("nan")
        if not np.isfinite(statistic) or statistic <= 0:
            attempt_record.update(
                {"status": "INVALID", "reason": "estimator_invalid"}
            )
            invalid_reasons["estimator_invalid"] = (
                invalid_reasons.get("estimator_invalid", 0) + 1
            )
        else:
            attempt_record.update({"status": "VALID", "reason": None})
            null_values.append(statistic)
        attempts.append(attempt_record)

    valid = len(null_values)
    attempted = len(attempts)
    complete = valid == B
    return {
        "status": "VALID" if complete else "UNRUN",
        "observed": observed,
        "null_values": null_values,
        "p_value": _lower_tail_p(observed, null_values) if complete else None,
        "requested": B,
        "attempted": attempted,
        "valid": valid,
        "invalid": attempted - valid,
        "invalid_reasons": invalid_reasons,
        "attempts": attempts,
        "identity_attempts": sum(attempt["identity"] for attempt in attempts),
        "selected_indices": observed_indices,
        "n_chains": len(unique_chains),
        "n_mixed_label_chains": mixed,
        "duplicate_audit": audited["audit"],
        "occurrence_list_sha256": occurrence_list_sha256,
        "row_counts": {
            "raw": audited["audit"]["raw_rows"],
            "deduplicated": audited["audit"]["deduplicated_rows"],
            "analyzed": len(observed_indices),
            "sentence_weighted": target_rows_after_deduplication,
        },
        "chain_counts": {
            "unique": len(unique_chains),
            "eligible": len(observed_chains),
            "selected": len(observed_indices),
            "mixed_label": mixed,
        },
        "chain_attrition_reasons": chain_attrition_reasons,
        "seed_coordinates": {
            "base_seed": int(base_seed),
            "purpose_code": 1,
            "family_code": int(family_code),
            "cell_index": int(cell_index),
        },
    }


def _quantile_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n_valid": 0, "q025": None, "median": None, "q975": None}
    q025, median, q975 = np.quantile(
        finite, [0.025, 0.5, 0.975], method="linear"
    )
    return {
        "n_valid": int(finite.size),
        "q025": float(q025),
        "median": float(median),
        "q975": float(q975),
    }


def run_chain_stability_band(
    activations: np.ndarray,
    occurrences: Sequence[Mapping[str, Any]],
    *,
    target_label: str,
    family_code: int,
    cell_index: int,
    behaviour_code: int,
    n_replicates: int = 500,
    valid_threshold: int = 475,
    fraction: float = 0.80,
    annotator_code: int = 1,
    layer: int = 27,
    pooling_code: int = 1,
    window_code: int = 1,
    truncation_code: int = 0,
    scope_key: str = "",
    cross_label_policy: str = "hard_fail",
    cdim_fn: Callable[[np.ndarray], float] = correlation_dimension_point,
    pr_fn: Callable[[np.ndarray], float] = _participation_ratio,
) -> dict[str, Any]:
    """Compute the registered 80%-chain deletion stability band."""
    _validate_cell_registry(
        family_code=family_code,
        cell_index=cell_index,
        behaviour_code=behaviour_code,
        annotator_code=annotator_code,
        layer=layer,
        pooling_code=pooling_code,
        window_code=window_code,
        truncation_code=truncation_code,
        scope_key=scope_key,
        purpose_code=2,
    )
    audited = _validate_and_deduplicate(
        activations,
        occurrences,
        cross_label_policy=cross_label_policy,
    )
    values = audited["activations"]
    records = audited["records"]
    labels = np.asarray([_occurrence_key(record)[0] for record in records])
    target = _nfc(target_label)
    eligible_chains = sorted(
        {
            _occurrence_key(record)[1]
            for record in records
            if _occurrence_key(record)[0] == target
        },
        key=lambda value: value.encode("utf-8"),
    )
    if not eligible_chains:
        raise ValueError("no eligible target-label chains")
    selected_chain_count = math.floor(float(fraction) * len(eligible_chains))
    if selected_chain_count < 1:
        raise ValueError("chain stability draw would select no chains")
    n_replicates = _json_integer(n_replicates)
    valid_threshold = _json_integer(valid_threshold)
    if not 0 <= valid_threshold <= n_replicates:
        raise ValueError("invalid stability threshold")

    selector = {
        "target_label": target,
        "family_code": family_code,
        "annotator_code": annotator_code,
        "layer": layer,
        "behaviour_code": behaviour_code,
        "pooling_code": pooling_code,
        "window_code": window_code,
        "truncation_code": truncation_code,
        "scope_key": scope_key,
    }
    full_indices = _select_equal_chain_indices(
        records, labels, replicate_index=-1, **selector
    )
    occurrence_list_sha256 = _occurrence_list_digest(
        [list(_occurrence_key(records[index])) for index in full_indices]
    )
    chain_to_labels: dict[str, set[str]] = {}
    for record in records:
        key = _occurrence_key(record)
        chain_to_labels.setdefault(key[1], set()).add(key[0])
    mixed_label_chains = sum(
        len(chain_labels) > 1 for chain_labels in chain_to_labels.values()
    )
    try:
        full_cdim = float(cdim_fn(values[full_indices]))
        full_pr = float(pr_fn(values[full_indices]))
    except EstimatorInvalidError as exc:
        raise EstimatorInvalidError(
            "full-sample stability statistic is invalid"
        ) from exc
    if (
        not np.isfinite(full_cdim)
        or full_cdim <= 0
        or not np.isfinite(full_pr)
        or full_pr <= 0
    ):
        raise EstimatorInvalidError("full-sample stability statistic is invalid")

    chain_array = np.asarray(eligible_chains)
    draws: list[dict[str, Any]] = []
    for replicate_index in range(n_replicates):
        rng = _chain_stability_rng(family_code, cell_index, replicate_index)
        chosen_positions = np.sort(
            rng.choice(
                len(eligible_chains), selected_chain_count, replace=False
            )
        )
        chosen_chains = chain_array[chosen_positions].tolist()
        chosen_set = set(chosen_chains)
        all_selected = _select_equal_chain_indices(
            records, labels, replicate_index=replicate_index, **selector
        )
        selected = [
            index
            for index in all_selected
            if _occurrence_key(records[index])[1] in chosen_set
        ]
        draw = {
            "replicate": replicate_index,
            "selected_chain_ids": chosen_chains,
            "cdim": None,
            "pr": None,
            "failure": None,
        }
        try:
            cdim = float(cdim_fn(values[selected]))
            pr = float(pr_fn(values[selected]))
            if (
                not np.isfinite(cdim)
                or cdim <= 0
                or not np.isfinite(pr)
                or pr <= 0
            ):
                raise EstimatorInvalidError(
                    "non-finite or non-positive stability statistic"
                )
        except EstimatorInvalidError:
            draw.update({"status": "INVALID", "failure": "estimator_invalid"})
        else:
            draw.update({"status": "VALID", "cdim": cdim, "pr": pr})
        draws.append(draw)

    cdim_summary = _quantile_summary(
        [draw["cdim"] for draw in draws if draw["cdim"] is not None]
    )
    pr_summary = _quantile_summary(
        [draw["pr"] for draw in draws if draw["pr"] is not None]
    )
    n_valid = min(cdim_summary["n_valid"], pr_summary["n_valid"])
    return {
        "status": "VALID" if n_valid >= valid_threshold else "UNRUN",
        "label": "chain stability band (not a confidence interval)",
        "full_sample_cdim": full_cdim,
        "full_sample_pr": full_pr,
        "n_eligible_chains": len(eligible_chains),
        "selected_chain_count": selected_chain_count,
        "n_replicates": n_replicates,
        "valid_threshold": valid_threshold,
        "n_valid": n_valid,
        "cdim_stability": cdim_summary,
        "pr_stability": pr_summary,
        "draws": draws,
        "duplicate_audit": audited["audit"],
        "occurrence_list_sha256": occurrence_list_sha256,
        "row_counts": {
            "raw": audited["audit"]["raw_rows"],
            "deduplicated": audited["audit"]["deduplicated_rows"],
            "analyzed": len(full_indices),
            "sentence_weighted": int(np.sum(labels == target)),
        },
        "chain_counts": {
            "unique": len(chain_to_labels),
            "eligible": len(eligible_chains),
            "selected": len(full_indices),
            "mixed_label": mixed_label_chains,
        },
    }


def _is_truncated(*, n_tokens: int, chain_text: str) -> bool:
    return _json_integer(n_tokens) >= 8192 and not _nfc(chain_text).strip().endswith(
        "</think>"
    )


def _h3_cross_product(chain_stability_leg: str, truncation_leg: str) -> str:
    if chain_stability_leg not in {"PASS", "FAIL", "UNRUN"}:
        raise ValueError("invalid chain-stability leg")
    if truncation_leg not in {"PASS", "MIXED", "FAIL", "UNRUN"}:
        raise ValueError("invalid truncation/category leg")
    if "UNRUN" in {chain_stability_leg, truncation_leg}:
        return "UNRUN"
    if chain_stability_leg == "FAIL":
        return "FAIL"
    return truncation_leg


def _h3_family_status(behaviour_statuses: Sequence[str]) -> str:
    statuses = list(behaviour_statuses)
    if len(statuses) != 4 or any(
        value not in {"PASS", "MIXED", "FAIL", "UNRUN"} for value in statuses
    ):
        raise ValueError("H3 requires exactly four exhaustive behaviour statuses")
    if "UNRUN" in statuses:
        return "UNRUN"
    if all(value == "PASS" for value in statuses):
        return "PASS"
    if all(value == "FAIL" for value in statuses):
        return "FAIL"
    return "MIXED"


def _classify_chain_stability_leg(band: Mapping[str, Any]) -> str:
    """Apply the exact Section 11 chain-stability leg operator."""
    if band.get("status") != "VALID":
        return "UNRUN"
    threshold = _json_integer(band["valid_threshold"])
    full_cdim = float(band["full_sample_cdim"])
    full_pr = float(band["full_sample_pr"])
    cdim_band = band["cdim_stability"]
    pr_band = band["pr_stability"]
    required_values = (
        full_cdim,
        full_pr,
        float(cdim_band["q025"]),
        float(cdim_band["median"]),
        float(cdim_band["q975"]),
        float(pr_band["q025"]),
        float(pr_band["median"]),
        float(pr_band["q975"]),
    )
    if (
        _json_integer(cdim_band["n_valid"]) < threshold
        or _json_integer(pr_band["n_valid"]) < threshold
        or not np.isfinite(required_values).all()
    ):
        return "UNRUN"
    passes = (
        cdim_band["q025"] <= full_cdim <= cdim_band["q975"]
        and pr_band["q025"] <= full_pr <= pr_band["q975"]
        and _relative_difference(full_cdim, cdim_band["median"]) <= 0.25
        and _relative_difference(full_pr, pr_band["median"]) <= 0.25
    )
    return "PASS" if passes else "FAIL"


def _category_share_imbalance(
    complete_counts: Mapping[str, int], truncated_counts: Mapping[str, int]
) -> float:
    normalized_complete: dict[str, int] = {}
    normalized_truncated: dict[str, int] = {}
    for key, value in complete_counts.items():
        category = _nfc(str(key))
        normalized_complete[category] = (
            normalized_complete.get(category, 0) + _json_integer(value)
        )
    for key, value in truncated_counts.items():
        category = _nfc(str(key))
        normalized_truncated[category] = (
            normalized_truncated.get(category, 0) + _json_integer(value)
        )
    complete_total = sum(normalized_complete.values())
    truncated_total = sum(normalized_truncated.values())
    if complete_total <= 0 or truncated_total <= 0:
        raise ValueError("category shares require non-empty complete and truncated groups")
    categories = set(normalized_complete) | set(normalized_truncated)
    return float(
        max(
            abs(
                normalized_complete.get(category, 0) / complete_total
                - normalized_truncated.get(category, 0) / truncated_total
            )
            for category in categories
        )
    )


def _matching_required(imbalance: float, threshold: float = 0.10) -> bool:
    value = float(imbalance)
    if not np.isfinite(value) or value < 0:
        raise ValueError("category imbalance must be finite and non-negative")
    return bool(value > float(threshold) + 1e-12)


def _truncation_neutral_map(
    occurrences: Sequence[Mapping[str, Any]],
    *,
    target_label: str,
    behaviour_code: int,
    replicate_index: int,
) -> dict[str, int]:
    labels = [_occurrence_key(record)[0] for record in occurrences]
    selected = _select_equal_chain_indices(
        occurrences,
        labels,
        target_label=target_label,
        family_code=4,
        annotator_code=1,
        layer=27,
        behaviour_code=behaviour_code,
        pooling_code=0,
        window_code=0,
        truncation_code=0,
        replicate_index=replicate_index,
        scope_key="",
    )
    return {
        _occurrence_key(occurrences[index])[1]: index for index in selected
    }


def _bands_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    required = ("q025", "q975")
    if any(left.get(key) is None or right.get(key) is None for key in required):
        return False
    return bool(
        max(float(left["q025"]), float(right["q025"]))
        <= min(float(left["q975"]), float(right["q975"]))
    )


def _category_match_rank(
    chain_id: str, category: str, behaviour_code: int
) -> str:
    return canonical_occurrence_digest(
        family_code=4,
        annotator_code=1,
        layer=27,
        behaviour_code=behaviour_code,
        pooling_code=0,
        window_code=0,
        truncation_code=4,
        chain_id=chain_id,
        annotation_index=-1,
        char_offset=-1,
        token_start=-1,
        replicate_index=-1,
        scope_key=category,
    )


def run_truncation_sensitivity(
    activations: np.ndarray,
    occurrences: Sequence[Mapping[str, Any]],
    chain_metadata: Mapping[str, Mapping[str, Any]]
    | Sequence[Mapping[str, Any]],
    *,
    target_label: str,
    behaviour_code: int,
    n_replicates: int = 500,
    valid_threshold: int = 475,
    cross_label_policy: str = "hard_fail",
    cdim_fn: Callable[[np.ndarray], float] = correlation_dimension_point,
    pr_fn: Callable[[np.ndarray], float] = _participation_ratio,
) -> dict[str, Any]:
    """Run the neutral-map truncation and category-matched H3 sensitivity."""
    behaviour_code = _validate_behaviour_code(behaviour_code)
    if isinstance(chain_metadata, Mapping):
        metadata_by_chain = {
            _nfc(str(key)): dict(value) for key, value in chain_metadata.items()
        }
    else:
        metadata_by_chain = {}
        for metadata in chain_metadata:
            chain_id = metadata.get("chain_id") or metadata.get("task_id")
            if chain_id is None:
                raise ValueError("chain metadata is missing chain_id/task_id")
            metadata_by_chain[_nfc(str(chain_id))] = dict(metadata)

    audited = _validate_and_deduplicate(
        activations,
        occurrences,
        cross_label_policy=cross_label_policy,
    )
    values = audited["activations"]
    records = audited["records"]
    normalized_target = _nfc(target_label)
    chain_to_labels: dict[str, set[str]] = {}
    target_rows_after_deduplication = 0
    for record in records:
        key = _occurrence_key(record)
        chain_to_labels.setdefault(key[1], set()).add(key[0])
        target_rows_after_deduplication += int(key[0] == normalized_target)
    mixed_label_chains = sum(
        len(chain_labels) > 1 for chain_labels in chain_to_labels.values()
    )
    n_replicates = _json_integer(n_replicates)
    valid_threshold = _json_integer(valid_threshold)
    if not 0 <= valid_threshold <= n_replicates:
        raise ValueError("invalid truncation stability threshold")
    full_map = _truncation_neutral_map(
        records,
        target_label=target_label,
        behaviour_code=behaviour_code,
        replicate_index=-1,
    )
    if not full_map:
        raise ValueError("no truncation-neutral target occurrences")
    for chain_id in full_map:
        if chain_id not in metadata_by_chain:
            raise ValueError(f"missing chain metadata for {chain_id}")
        metadata = metadata_by_chain[chain_id]
        if "n_tokens" not in metadata:
            raise ValueError(f"missing n_tokens for {chain_id}")
        if "chain" not in metadata and "text" not in metadata:
            raise ValueError(f"missing chain text for {chain_id}")
        if "category" not in metadata:
            raise ValueError(f"missing category for {chain_id}")

    neutral_maps = [
        _truncation_neutral_map(
            records,
            target_label=target_label,
            behaviour_code=behaviour_code,
            replicate_index=replicate_index,
        )
        for replicate_index in range(n_replicates)
    ]
    complete_chains = sorted(
        [
            chain_id
            for chain_id in full_map
            if not _is_truncated(
                n_tokens=metadata_by_chain[chain_id]["n_tokens"],
                chain_text=metadata_by_chain[chain_id].get(
                    "chain", metadata_by_chain[chain_id].get("text", "")
                ),
            )
        ],
        key=lambda value: value.encode("utf-8"),
    )
    truncated_chains = sorted(
        [chain_id for chain_id in full_map if chain_id not in set(complete_chains)],
        key=lambda value: value.encode("utf-8"),
    )
    combined_chains = sorted(full_map, key=lambda value: value.encode("utf-8"))

    def group_result(
        fixed_chains: Sequence[str], *, cell_index: int
    ) -> dict[str, Any]:
        ordered_chains = sorted(
            {_nfc(str(chain)) for chain in fixed_chains},
            key=lambda value: value.encode("utf-8"),
        )
        full_indices = [full_map[chain_id] for chain_id in ordered_chains]
        occurrence_keys = [
            list(_occurrence_key(records[index])) for index in full_indices
        ]
        category_counts: dict[str, int] = {}
        for chain_id in ordered_chains:
            category = _nfc(str(metadata_by_chain[chain_id]["category"]))
            category_counts[category] = category_counts.get(category, 0) + 1
        try:
            cdim = float(cdim_fn(values[full_indices]))
            pr = float(pr_fn(values[full_indices]))
            if (
                not np.isfinite(cdim)
                or cdim <= 0
                or not np.isfinite(pr)
                or pr <= 0
            ):
                raise EstimatorInvalidError("invalid full truncation statistic")
        except EstimatorInvalidError:
            cdim = None
            pr = None

        draws: list[dict[str, Any]] = []
        selected_count = math.floor(0.80 * len(ordered_chains))
        for replicate_index, neutral_map in enumerate(neutral_maps):
            draw = {
                "replicate": replicate_index,
                "status": "INVALID",
                "cdim": None,
                "pr": None,
                "failure": "estimator_invalid",
            }
            if selected_count > 0:
                rng = _chain_stability_rng(4, cell_index, replicate_index)
                positions = np.sort(
                    rng.choice(
                        len(ordered_chains), selected_count, replace=False
                    )
                )
                selected_chains = [
                    ordered_chains[position] for position in positions
                ]
                if all(chain_id in neutral_map for chain_id in selected_chains):
                    indices = [
                        neutral_map[chain_id] for chain_id in selected_chains
                    ]
                    try:
                        draw_cdim = float(cdim_fn(values[indices]))
                        draw_pr = float(pr_fn(values[indices]))
                        if (
                            not np.isfinite(draw_cdim)
                            or draw_cdim <= 0
                            or not np.isfinite(draw_pr)
                            or draw_pr <= 0
                        ):
                            raise EstimatorInvalidError(
                                "invalid truncation stability statistic"
                            )
                    except EstimatorInvalidError:
                        pass
                    else:
                        draw.update(
                            {
                                "status": "VALID",
                                "cdim": draw_cdim,
                                "pr": draw_pr,
                                "failure": None,
                            }
                        )
            draws.append(draw)
        cdim_stability = _quantile_summary(
            [draw["cdim"] for draw in draws if draw["cdim"] is not None]
        )
        pr_stability = _quantile_summary(
            [draw["pr"] for draw in draws if draw["pr"] is not None]
        )
        n_valid = min(
            cdim_stability["n_valid"], pr_stability["n_valid"]
        )
        if cdim is None or pr is None:
            status = "INVALID"
        elif n_valid < valid_threshold:
            status = "UNRUN"
        else:
            status = "VALID"
        return {
            "status": status,
            "chain_ids": ordered_chains,
            "n_chains": len(ordered_chains),
            "n_categories": len(category_counts),
            "category_counts": category_counts,
            "occurrence_list_sha256": _occurrence_list_digest(occurrence_keys),
            "cdim": cdim,
            "pr": pr,
            "cdim_stability": cdim_stability,
            "pr_stability": pr_stability,
            "stability_draws": draws,
        }

    behaviour_rank = _json_integer(behaviour_code) - 1
    strata = {
        "combined": group_result(
            combined_chains, cell_index=48 + behaviour_rank
        ),
        "complete": group_result(
            complete_chains, cell_index=52 + behaviour_rank
        ),
        "truncated": group_result(
            truncated_chains, cell_index=56 + behaviour_rank
        ),
    }
    complete_counts = strata["complete"]["category_counts"]
    truncated_counts = strata["truncated"]["category_counts"]
    try:
        imbalance = _category_share_imbalance(
            complete_counts, truncated_counts
        )
    except ValueError:
        imbalance = float("inf")
    matching_required = (
        _matching_required(imbalance) if np.isfinite(imbalance) else True
    )

    def valid_unadjusted() -> bool | None:
        if any(result["status"] != "VALID" for result in strata.values()):
            return None
        combined = strata["combined"]
        complete = strata["complete"]
        truncated = strata["truncated"]
        return bool(
            _relative_difference(complete["cdim"], combined["cdim"]) <= 0.25
            and _relative_difference(truncated["cdim"], combined["cdim"])
            <= 0.25
            and _relative_difference(complete["pr"], combined["pr"]) <= 0.25
            and _relative_difference(truncated["pr"], combined["pr"]) <= 0.25
            and _bands_overlap(
                complete["cdim_stability"], truncated["cdim_stability"]
            )
            and _bands_overlap(
                complete["pr_stability"], truncated["pr_stability"]
            )
        )

    unadjusted_pass = valid_unadjusted()
    matched_results: dict[str, Any] | None = None
    matched_pass: bool | None = None
    if matching_required and np.isfinite(imbalance):
        complete_by_category: dict[str, list[str]] = {}
        truncated_by_category: dict[str, list[str]] = {}
        for chain_id in complete_chains:
            category = _nfc(str(metadata_by_chain[chain_id]["category"]))
            complete_by_category.setdefault(category, []).append(chain_id)
        for chain_id in truncated_chains:
            category = _nfc(str(metadata_by_chain[chain_id]["category"]))
            truncated_by_category.setdefault(category, []).append(chain_id)
        matched_complete: list[str] = []
        matched_truncated: list[str] = []
        for category in sorted(
            set(complete_by_category) & set(truncated_by_category),
            key=lambda value: value.encode("utf-8"),
        ):
            n_match = min(
                len(complete_by_category[category]),
                len(truncated_by_category[category]),
            )
            ranked_complete = sorted(
                complete_by_category[category],
                key=lambda chain_id: _category_match_rank(
                    chain_id, category, behaviour_code
                ),
            )
            ranked_truncated = sorted(
                truncated_by_category[category],
                key=lambda chain_id: _category_match_rank(
                    chain_id, category, behaviour_code
                ),
            )
            matched_complete.extend(ranked_complete[:n_match])
            matched_truncated.extend(ranked_truncated[:n_match])
        matched_complete_result = group_result(
            matched_complete, cell_index=60 + behaviour_rank
        )
        matched_truncated_result = group_result(
            matched_truncated, cell_index=60 + behaviour_rank
        )

        def exact_matched_result(result: Mapping[str, Any]) -> dict[str, Any]:
            return {
                key: result[key]
                for key in (
                    "chain_ids",
                    "n_chains",
                    "n_categories",
                    "category_counts",
                    "occurrence_list_sha256",
                    "cdim",
                    "pr",
                    "cdim_stability",
                    "pr_stability",
                    "status",
                )
            }

        matched_results = {
            "matched_complete": exact_matched_result(matched_complete_result),
            "matched_truncated": exact_matched_result(
                matched_truncated_result
            ),
        }
        if (
            matched_complete_result["status"] == "VALID"
            and matched_truncated_result["status"] == "VALID"
        ):
            matched_pass = bool(
                _relative_difference(
                    matched_complete_result["cdim"],
                    matched_truncated_result["cdim"],
                )
                <= 0.25
                and _relative_difference(
                    matched_complete_result["pr"],
                    matched_truncated_result["pr"],
                )
                <= 0.25
                and _bands_overlap(
                    matched_complete_result["cdim_stability"],
                    matched_truncated_result["cdim_stability"],
                )
                and _bands_overlap(
                    matched_complete_result["pr_stability"],
                    matched_truncated_result["pr_stability"],
                )
            )

    if unadjusted_pass is None:
        truncation_leg = "UNRUN"
    elif not matching_required:
        truncation_leg = "PASS" if unadjusted_pass else "FAIL"
    elif matched_pass is None:
        truncation_leg = "UNRUN"
    elif unadjusted_pass and matched_pass:
        truncation_leg = "PASS"
    elif not unadjusted_pass and not matched_pass:
        truncation_leg = "FAIL"
    else:
        truncation_leg = "MIXED"

    descriptive: dict[str, Any] = {}
    record_counts: dict[str, int] = {}
    for record in records:
        key = _occurrence_key(record)
        if key[0] == _nfc(target_label):
            record_counts[key[1]] = record_counts.get(key[1], 0) + 1
    for name, chain_list in (
        ("complete", complete_chains),
        ("truncated", truncated_chains),
    ):
        positions = [
            _occurrence_key(records[full_map[chain_id]])[4]
            / max(_json_integer(metadata_by_chain[chain_id]["n_tokens"]), 1)
            for chain_id in chain_list
        ]
        descriptive[name] = {
            "behaviour_rows_per_chain_mean": (
                float(np.mean([record_counts[chain] for chain in chain_list]))
                if chain_list
                else None
            ),
            "normalised_within_chain_positions": positions,
        }
    try:
        chain_stability = run_chain_stability_band(
            activations,
            occurrences,
            target_label=target_label,
            family_code=1,
            cell_index=behaviour_code - 1,
            behaviour_code=behaviour_code,
            n_replicates=n_replicates,
            valid_threshold=valid_threshold,
            cross_label_policy=cross_label_policy,
            cdim_fn=cdim_fn,
            pr_fn=pr_fn,
        )
    except EstimatorInvalidError:
        chain_stability = {
            "status": "UNRUN",
            "valid_threshold": valid_threshold,
        }
    chain_stability_leg = _classify_chain_stability_leg(chain_stability)
    h3_status = _h3_cross_product(chain_stability_leg, truncation_leg)
    overall_valid = h3_status != "UNRUN"
    return {
        "status": "VALID" if overall_valid else "UNRUN",
        "strata": strata,
        "category_share_imbalance": imbalance,
        "matching_required": matching_required,
        "matched_results": matched_results,
        "unadjusted_pass": unadjusted_pass,
        "matched_pass": matched_pass,
        "chain_stability": chain_stability,
        "chain_stability_leg": chain_stability_leg,
        "truncation_leg": truncation_leg,
        "h3_behaviour_status": h3_status,
        "descriptives": descriptive,
        "duplicate_audit": audited["audit"],
        "pool_counts": {
            "unique_chains": len(chain_to_labels),
            "mixed_label_chains": mixed_label_chains,
            "target_rows_after_deduplication": (
                target_rows_after_deduplication
            ),
        },
    }


def _relative_difference(a: float, b: float) -> float:
    return float(abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), 1e-12))


def _expected_family_cells(family: str) -> list[dict[str, int]]:
    cells: list[dict[str, int]] = []
    if family == "primary":
        for behaviour_code in range(1, 5):
            cells.append(
                {
                    "family_code": 1,
                    "cell_index": behaviour_code - 1,
                    "behaviour_code": behaviour_code,
                    "annotator_code": 1,
                    "layer": 27,
                    "pooling_code": 1,
                    "window_code": 1,
                    "truncation_code": 0,
                }
            )
    elif family == "five_depth":
        for behaviour_code in range(1, 5):
            for depth_rank, layer in enumerate((11, 14, 17, 20, 27)):
                cells.append(
                    {
                        "family_code": 2,
                        "cell_index": 4 + 5 * (behaviour_code - 1) + depth_rank,
                        "behaviour_code": behaviour_code,
                        "annotator_code": 1,
                        "layer": layer,
                        "pooling_code": 1,
                        "window_code": 1,
                        "truncation_code": 0,
                    }
                )
    elif family == "three_annotator":
        for annotator_code in range(1, 4):
            for layer_rank, layer in enumerate((12, 16)):
                for behaviour_code in range(1, 5):
                    cells.append(
                        {
                            "family_code": 3,
                            "cell_index": (
                                24
                                + (
                                    (
                                        (annotator_code - 1) * 2
                                        + layer_rank
                                    )
                                    * 4
                                )
                                + behaviour_code
                                - 1
                            ),
                            "behaviour_code": behaviour_code,
                            "annotator_code": annotator_code,
                            "layer": layer,
                            "pooling_code": 1,
                            "window_code": 1,
                            "truncation_code": 0,
                        }
                    )
    else:
        raise ValueError("unknown registered cdim family")
    return cells


def _adjudicate_registered_family(
    cells: Sequence[Mapping[str, Any]],
    holm_p: Sequence[float],
    diagnostic_holm_p: Sequence[float],
    *,
    family: str,
) -> dict[str, Any]:
    expected_count = len(_expected_family_cells(family))
    if len(cells) != len(holm_p) or len(cells) != expected_count:
        raise ValueError("registered family adjudication requires every cell")
    if family == "primary" and len(diagnostic_holm_p) != expected_count:
        raise ValueError("primary adjudication requires every diagnostic-seed cell")
    if family != "primary" and len(diagnostic_holm_p) not in {0, expected_count}:
        raise ValueError("secondary adjudication received a partial diagnostic family")
    main_pass = [float(value) <= 0.05 for value in holm_p]
    diagnostic_pass = (
        [float(value) <= 0.05 for value in diagnostic_holm_p]
        if family == "primary"
        else []
    )
    unstable = (
        [
            int(cells[index]["cell_index"])
            for index in range(len(cells))
            if main_pass[index] != diagnostic_pass[index]
        ]
        if family == "primary"
        else []
    )
    behaviour_decisions: dict[int, str] = {}
    if family == "primary":
        for index, cell in enumerate(cells):
            behaviour_decisions[int(cell["behaviour_code"])] = (
                "PASS"
                if main_pass[index] and diagnostic_pass[index]
                else "MIXED"
                if main_pass[index] != diagnostic_pass[index]
                else "FAIL"
            )
        n_pass = sum(value == "PASS" for value in behaviour_decisions.values())
        if n_pass == 4 and not unstable:
            decision = "PASS"
        elif n_pass == 0 and not unstable:
            decision = "FAIL"
        else:
            decision = "MIXED"
    elif family == "five_depth":
        for behaviour_code in range(1, 5):
            indices = [
                index
                for index, cell in enumerate(cells)
                if int(cell["behaviour_code"]) == behaviour_code
            ]
            n_stable_pass = sum(main_pass[index] for index in indices)
            behaviour_decisions[behaviour_code] = (
                "PASS" if n_stable_pass >= 4 else "MIXED"
            )
        n_pass = sum(value == "PASS" for value in behaviour_decisions.values())
        decision = "PASS" if n_pass == 4 else "FAIL" if n_pass == 0 else "MIXED"
    elif family == "three_annotator":
        for behaviour_code in range(1, 5):
            indices = [
                index
                for index, cell in enumerate(cells)
                if int(cell["behaviour_code"]) == behaviour_code
                and int(cell["layer"]) == 16
            ]
            robust = len(indices) == 3 and all(main_pass[index] for index in indices)
            behaviour_decisions[behaviour_code] = (
                "PASS" if robust else "MIXED"
            )
        n_pass = sum(value == "PASS" for value in behaviour_decisions.values())
        decision = "PASS" if n_pass == 4 else "FAIL" if n_pass == 0 else "MIXED"
    else:
        raise ValueError("unknown registered cdim family")
    return {
        "decision": decision,
        "behaviour_decisions": behaviour_decisions,
        "monte_carlo_unstable_cells": unstable,
    }


def _adjudicate_primary_h1(
    observed_values: Sequence[float | None],
) -> dict[str, Any]:
    """Classify the four-cell descriptive low-cdim H1 contract."""
    if len(observed_values) != 4:
        raise ValueError("H1 adjudication requires exactly four primary cells")
    behaviour_decisions: dict[int, str] = {}
    invalid = False
    for behaviour_code, raw_value in enumerate(observed_values, start=1):
        if (
            raw_value is None
            or isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float, np.number))
            or not np.isfinite(float(raw_value))
            or float(raw_value) <= 0
        ):
            behaviour_decisions[behaviour_code] = "UNRUN"
            invalid = True
        else:
            behaviour_decisions[behaviour_code] = (
                "PASS" if float(raw_value) <= 10.0 else "FAIL"
            )
    if invalid:
        decision = "UNRUN"
    else:
        passing = sum(value == "PASS" for value in behaviour_decisions.values())
        decision = "PASS" if passing == 4 else "FAIL" if passing == 0 else "MIXED"
    return {
        "decision": decision,
        "behaviour_decisions": behaviour_decisions,
    }


def _unrun_registered_family(
    *, family: str, reason: str, cells: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "UNRUN",
        "decision": "UNRUN",
        "reason": reason,
        "cells": list(cells),
        "holm_p": [],
        "diagnostic_holm_p": [],
        "behaviour_decisions": {},
    }
    if family == "primary":
        unrun_behaviours = {code: "UNRUN" for code in range(1, 5)}
        result["behaviour_decisions"] = dict(unrun_behaviours)
        result["hypotheses"] = {
            "H1": _adjudicate_primary_h1([None, None, None, None]),
            "H2": {
                "decision": "UNRUN",
                "behaviour_decisions": dict(unrun_behaviours),
                "monte_carlo_unstable_cells": [],
            },
        }
    return result


def _run_registered_cdim_family(
    cell_inputs: Mapping[int, Mapping[str, Any]],
    *,
    family: str,
    B: int = 2500,
    attempt_cap: int = 2750,
    cross_label_policy: str = "hard_fail",
    statistic_fn: Callable[[np.ndarray], float] = correlation_dimension_point,
    progress_callback: Callable[[Sequence[Mapping[str, Any]]], None]
    | None = None,
) -> dict[str, Any]:
    expected = _expected_family_cells(family)
    expected_indices = {cell["cell_index"] for cell in expected}
    if set(cell_inputs) != expected_indices:
        return _unrun_registered_family(
            family=family,
            reason="registered family is incomplete",
            cells=[],
        )
    outputs: list[dict[str, Any]] = []
    for coordinates in expected:
        supplied = cell_inputs[coordinates["cell_index"]]
        if any(
            int(supplied.get(key, -1)) != value
            for key, value in coordinates.items()
        ):
            return _unrun_registered_family(
                family=family,
                reason="cell/code registry mismatch",
                cells=outputs,
            )
        common_arguments = {
            "activations": supplied["activations"],
            "labels": supplied["labels"],
            "chain_ids": supplied["chain_ids"],
            "occurrences": supplied["occurrences"],
            "target_label": supplied["target_label"],
            **coordinates,
            "B": B,
            "attempt_cap": attempt_cap,
            "cross_label_policy": cross_label_policy,
            "statistic_fn": statistic_fn,
        }
        try:
            registered = run_within_chain_cdim_null(
                **common_arguments, base_seed=20260726
            )
            diagnostic = (
                run_within_chain_cdim_null(
                    **common_arguments, base_seed=20260727
                )
                if family == "primary"
                else None
            )
        except EstimatorInvalidError:
            return _unrun_registered_family(
                family=family,
                reason=f"invalid registered cell {coordinates['cell_index']}",
                cells=outputs,
            )
        if registered["status"] != "VALID" or (
            diagnostic is not None and diagnostic["status"] != "VALID"
        ):
            return _unrun_registered_family(
                family=family,
                reason=f"under-resampled registered cell {coordinates['cell_index']}",
                cells=outputs,
            )
        output = {
            **coordinates,
            "status": "VALID",
            "observed_cdim": registered["observed"],
            "raw_p": registered["p_value"],
            "null_cdim": registered["null_values"],
            "resamples": {
                "attempted": registered["attempted"],
                "valid": registered["valid"],
                "invalid": registered["invalid"],
                "invalid_reasons": registered["invalid_reasons"],
            },
        }
        for field in (
            "duplicate_audit",
            "occurrence_list_sha256",
            "row_counts",
            "chain_counts",
            "chain_attrition_reasons",
        ):
            if field in registered:
                output[field] = registered[field]
        if diagnostic is not None:
            output["diagnostic_raw_p"] = diagnostic["p_value"]
            output["diagnostic_null_cdim"] = diagnostic["null_values"]
            output["diagnostic_resamples"] = {
                "attempted": diagnostic["attempted"],
                "valid": diagnostic["valid"],
                "invalid": diagnostic["invalid"],
                "invalid_reasons": diagnostic["invalid_reasons"],
            }
        outputs.append(output)
        if progress_callback is not None:
            progress_callback(outputs)
    holm = _holm_adjust([cell["raw_p"] for cell in outputs])
    diagnostic_holm = (
        _holm_adjust([cell["diagnostic_raw_p"] for cell in outputs])
        if family == "primary"
        else []
    )
    for index, (output, adjusted) in enumerate(zip(outputs, holm)):
        output["holm_p"] = adjusted
        if family == "primary":
            output["diagnostic_holm_p"] = diagnostic_holm[index]
    adjudication = _adjudicate_registered_family(
        expected, holm, diagnostic_holm, family=family
    )
    result = {
        "status": "VALID",
        "cells": outputs,
        "holm_p": holm,
        "diagnostic_holm_p": diagnostic_holm,
        **adjudication,
    }
    if family == "primary":
        h1 = _adjudicate_primary_h1(
            [cell["observed_cdim"] for cell in outputs]
        )
        result["hypotheses"] = {
            "H1": h1,
            "H2": {
                "decision": adjudication["decision"],
                "behaviour_decisions": adjudication["behaviour_decisions"],
                "monte_carlo_unstable_cells": adjudication[
                    "monte_carlo_unstable_cells"
                ],
            },
        }
    return result


def run_pooling_window_family(
    grid: Mapping[str, Any],
    *,
    B: int = 2500,
    attempt_cap: int = 2750,
    base_seed: int = 20260726,
    diagnostic_base_seed: int = 20260727,
    statistic_fn: Callable[[np.ndarray], float] = correlation_dimension_point,
) -> dict[str, Any]:
    """Run the all-or-UNRUN 24-cell H4 family on a common grid."""
    if grid.get("status") != "VALID":
        return {
            "status": "UNRUN",
            "decision": "UNRUN",
            "reason": "common complete-case grid is not valid",
            "cells": [],
            "holm_p": [],
            "diagnostic_holm_p": [],
        }
    representation_order = list(grid.get("representation_order", []))
    expected_order = [
        f"{pooling}/{window}" for pooling, window in _REPRESENTATION_ORDER
    ]
    behaviour_codes = dict(grid.get("behaviour_codes", {}))
    if representation_order != expected_order or sorted(behaviour_codes.values()) != [
        1,
        2,
        3,
        4,
    ]:
        return {
            "status": "UNRUN",
            "decision": "UNRUN",
            "reason": "H4 requires the exact six-by-four registry",
            "cells": [],
            "holm_p": [],
            "diagnostic_holm_p": [],
        }

    cells: list[dict[str, Any]] = []
    for representation_rank, representation_name in enumerate(representation_order):
        pooling_rank = representation_rank // 2
        window_rank = representation_rank % 2
        common = grid["common_representations"].get(representation_name)
        if not common:
            return {
                "status": "UNRUN",
                "decision": "UNRUN",
                "reason": f"missing representation {representation_name}",
                "cells": cells,
                "holm_p": [],
                "diagnostic_holm_p": [],
            }
        records = common["records"]
        values = np.asarray(common["activations"], dtype=np.float64)
        labels = np.asarray([_occurrence_key(record)[0] for record in records])
        chain_ids = np.asarray([_occurrence_key(record)[1] for record in records])
        for label, behaviour_code in sorted(
            behaviour_codes.items(), key=lambda item: item[1]
        ):
            behaviour_rank = behaviour_code - 1
            cell_index = (
                64 + ((pooling_rank * 2 + window_rank) * 4) + behaviour_rank
            )
            try:
                registered = run_within_chain_cdim_null(
                    values,
                    labels,
                    chain_ids,
                    records,
                    target_label=label,
                    family_code=5,
                    cell_index=cell_index,
                    behaviour_code=behaviour_code,
                    B=B,
                    attempt_cap=attempt_cap,
                    base_seed=base_seed,
                    pooling_code=0,
                    window_code=0,
                    truncation_code=0,
                    scope_key="H4-common",
                    statistic_fn=statistic_fn,
                )
                diagnostic = run_within_chain_cdim_null(
                    values,
                    labels,
                    chain_ids,
                    records,
                    target_label=label,
                    family_code=5,
                    cell_index=cell_index,
                    behaviour_code=behaviour_code,
                    B=B,
                    attempt_cap=attempt_cap,
                    base_seed=diagnostic_base_seed,
                    pooling_code=0,
                    window_code=0,
                    truncation_code=0,
                    scope_key="H4-common",
                    statistic_fn=statistic_fn,
                )
                selected_matrix = grid["behaviours"][label]["representations"][
                    representation_name
                ]["activations"]
                pr = _participation_ratio(selected_matrix)
                sentence_weighted_cdim = float(
                    statistic_fn(values[labels == label])
                )
                if (
                    not np.isfinite(sentence_weighted_cdim)
                    or sentence_weighted_cdim <= 0
                ):
                    raise EstimatorInvalidError(
                        "sentence-weighted cdim is invalid"
                    )
            except EstimatorInvalidError:
                return {
                    "status": "UNRUN",
                    "decision": "UNRUN",
                    "reason": f"invalid H4 cell {cell_index}",
                    "cells": cells,
                    "holm_p": [],
                    "diagnostic_holm_p": [],
                }
            if registered["status"] != "VALID" or diagnostic["status"] != "VALID":
                return {
                    "status": "UNRUN",
                    "decision": "UNRUN",
                    "reason": f"under-resampled H4 cell {cell_index}",
                    "cells": cells,
                    "holm_p": [],
                    "diagnostic_holm_p": [],
                }
            expected_digest = grid["behaviours"][label]["occurrence_list_sha256"]
            cells.append(
                {
                    "status": "VALID",
                    "cell_index": cell_index,
                    "label": label,
                    "pooling": representation_name.split("/")[0],
                    "window": representation_name.split("/")[1],
                    "occurrence_list_sha256": expected_digest,
                    "observed_cdim": registered["observed"],
                    "observed_pr": pr,
                    "sentence_weighted_label": (
                        "sentence-weighted correlation dimension"
                    ),
                    "sentence_weighted_cdim": sentence_weighted_cdim,
                    "chain_attrition_reasons": grid["behaviours"][label][
                        "chain_attrition_reasons"
                    ],
                    "null_cdim": registered["null_values"],
                    "raw_p": registered["p_value"],
                    "diagnostic_raw_p": diagnostic["p_value"],
                    "resamples": {
                        "attempted": registered["attempted"],
                        "valid": registered["valid"],
                        "invalid": registered["invalid"],
                        "invalid_reasons": registered["invalid_reasons"],
                    },
                }
            )

    if len(cells) != 24:
        return {
            "status": "UNRUN",
            "decision": "UNRUN",
            "reason": "H4 did not produce all 24 cells",
            "cells": cells,
            "holm_p": [],
            "diagnostic_holm_p": [],
        }
    holm = _holm_adjust([cell["raw_p"] for cell in cells])
    diagnostic_holm = _holm_adjust(
        [cell["diagnostic_raw_p"] for cell in cells]
    )
    for cell, adjusted, diagnostic_adjusted in zip(
        cells, holm, diagnostic_holm
    ):
        cell["holm_p"] = adjusted
        cell["diagnostic_holm_p"] = diagnostic_adjusted

    baseline = {
        cell["label"]: cell
        for cell in cells
        if cell["pooling"] == "mean" and cell["window"] == "unclipped"
    }
    failed_behaviours: set[str] = set()
    for cell in cells:
        reference = baseline[cell["label"]]
        passes = (
            _relative_difference(
                cell["observed_cdim"], reference["observed_cdim"]
            )
            <= 0.25
            and _relative_difference(cell["observed_pr"], reference["observed_pr"])
            <= 0.25
            and cell["holm_p"] <= 0.05
            and ((cell["holm_p"] <= 0.05) == (cell["diagnostic_holm_p"] <= 0.05))
        )
        cell["passes"] = passes
        if not passes:
            failed_behaviours.add(cell["label"])
    if not failed_behaviours:
        decision = "PASS"
    elif len(failed_behaviours) == 4:
        decision = "FAIL"
    else:
        decision = "MIXED"
    return {
        "status": "VALID",
        "decision": decision,
        "cells": cells,
        "holm_p": holm,
        "diagnostic_holm_p": diagnostic_holm,
        "failed_behaviours": sorted(failed_behaviours),
    }


def run_curvature_diagnostic(
    activations: np.ndarray,
    occurrences: Sequence[Mapping[str, Any]],
    *,
    target_label: str,
    behaviour_code: int,
    cell_index: int,
    n_replicates: int = 500,
    valid_threshold: int = 475,
    fraction: float = 0.80,
    cross_label_policy: str = "hard_fail",
    ratio_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run the paired L16 local/global curvature diagnostic."""
    _validate_cell_registry(
        family_code=6,
        cell_index=cell_index,
        behaviour_code=behaviour_code,
        annotator_code=1,
        layer=16,
        pooling_code=0,
        window_code=0,
        truncation_code=0,
        purpose_code=4,
    )
    if ratio_fn is None:
        from src.curvature import local_vs_global_dim_ratio

        def registered_ratio_fn(values: np.ndarray, **settings: Any) -> Any:
            try:
                return local_vs_global_dim_ratio(values, **settings)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                raise EstimatorInvalidError(
                    "curvature ratio numeric domain is invalid"
                ) from exc

        ratio_fn = registered_ratio_fn
    audited = _validate_and_deduplicate(
        activations,
        occurrences,
        cross_label_policy=cross_label_policy,
    )
    values = np.asarray(audited["activations"], dtype=np.float32)
    records = audited["records"]
    labels = np.asarray([_occurrence_key(record)[0] for record in records])
    target = _nfc(target_label)
    full_indices = _select_equal_chain_indices(
        records,
        labels,
        target_label=target,
        family_code=6,
        annotator_code=1,
        layer=16,
        behaviour_code=behaviour_code,
        pooling_code=0,
        window_code=0,
        truncation_code=0,
        replicate_index=-1,
        scope_key="",
    )
    occurrence_list_sha256 = _occurrence_list_digest(
        [list(_occurrence_key(records[index])) for index in full_indices]
    )
    chain_to_labels: dict[str, set[str]] = {}
    for record in records:
        key = _occurrence_key(record)
        chain_to_labels.setdefault(key[1], set()).add(key[0])
    mixed_label_chains = sum(
        len(chain_labels) > 1 for chain_labels in chain_to_labels.values()
    )
    eligible_chains = sorted(
        {
            _occurrence_key(record)[1]
            for record in records
            if _occurrence_key(record)[0] == target
        },
        key=lambda value: value.encode("utf-8"),
    )
    selected_count = math.floor(float(fraction) * len(eligible_chains))
    if selected_count < 11 or values.shape[0] < selected_count:
        raise ValueError("curvature diagnostic requires at least 11 selected points")
    n_replicates = _json_integer(n_replicates)
    valid_threshold = _json_integer(valid_threshold)
    if not 0 <= valid_threshold <= n_replicates:
        raise ValueError("invalid curvature valid-pair threshold")

    settings = {
        "k": 10,
        "variance_threshold": 0.90,
        "n_anchors": 150,
        "random_state": 42,
        "n_bootstrap": 0,
    }
    chain_array = np.asarray(eligible_chains)
    draws: list[dict[str, Any]] = []
    for replicate_index in range(n_replicates):
        parent = np.random.SeedSequence(
            [20260726, 4, 6, _json_integer(cell_index), replicate_index]
        )
        chain_seed, control_seed = parent.spawn(2)
        chain_rng = np.random.default_rng(chain_seed)
        control_rng = np.random.default_rng(control_seed)
        positions = np.sort(
            chain_rng.choice(len(eligible_chains), selected_count, replace=False)
        )
        chosen_chains = chain_array[positions].tolist()
        chosen_set = set(chosen_chains)
        selected_all = _select_equal_chain_indices(
            records,
            labels,
            target_label=target,
            family_code=6,
            annotator_code=1,
            layer=16,
            behaviour_code=behaviour_code,
            pooling_code=0,
            window_code=0,
            truncation_code=0,
            replicate_index=replicate_index,
            scope_key="",
        )
        selected = [
            index
            for index in selected_all
            if _occurrence_key(records[index])[1] in chosen_set
        ]
        control = np.sort(
            control_rng.choice(values.shape[0], selected_count, replace=False)
        )
        draw: dict[str, Any] = {
            "replicate": replicate_index,
            "selected_chain_ids": chosen_chains,
            "ratio_chain": None,
            "ratio_control": None,
            "failure": None,
        }
        try:
            chain_result = ratio_fn(values[selected].copy(), **settings)
            control_result = ratio_fn(values[control].copy(), **settings)
            chain_ratio = float(
                chain_result.mean if hasattr(chain_result, "mean") else chain_result
            )
            control_ratio = float(
                control_result.mean
                if hasattr(control_result, "mean")
                else control_result
            )
            if (
                not np.isfinite(chain_ratio)
                or chain_ratio <= 0
                or not np.isfinite(control_ratio)
                or control_ratio <= 0
            ):
                raise EstimatorInvalidError("invalid paired curvature ratio")
        except EstimatorInvalidError:
            draw.update({"status": "INVALID", "failure": "paired_estimator_invalid"})
        else:
            draw.update(
                {
                    "status": "VALID",
                    "ratio_chain": chain_ratio,
                    "ratio_control": control_ratio,
                }
            )
        draws.append(draw)

    valid_draws = [draw for draw in draws if draw["status"] == "VALID"]
    n_valid = len(valid_draws)
    chain_summary = _quantile_summary(
        [draw["ratio_chain"] for draw in valid_draws]
    )
    control_summary = _quantile_summary(
        [draw["ratio_control"] for draw in valid_draws]
    )
    if n_valid < valid_threshold:
        return {
            "status": "UNRUN",
            "n_valid_pairs": n_valid,
            "valid_threshold": valid_threshold,
            "bounded_negative": None,
            "wording": None,
            "chain_stability": chain_summary,
            "control_stability": control_summary,
            "draws": draws,
            "settings": settings,
        }
    chain_values = np.asarray(
        [draw["ratio_chain"] for draw in valid_draws], dtype=np.float64
    )
    control_values = np.asarray(
        [draw["ratio_control"] for draw in valid_draws], dtype=np.float64
    )
    d_chain = float(np.median(np.abs(chain_values - 1.0)))
    d_control = float(np.median(np.abs(control_values - 1.0)))
    bounded_negative = bool(
        chain_summary["q025"] <= 1.0 <= chain_summary["q975"]
        and d_chain <= d_control
    )
    wording = (
        "no beyond-chain curvature detected by this diagnostic at L16"
        if bounded_negative
        else "mixed/positive controlled-layer curvature diagnostic at L16"
    )
    return {
        "status": "VALID",
        "n_valid_pairs": n_valid,
        "valid_threshold": valid_threshold,
        "bounded_negative": bounded_negative,
        "wording": wording,
        "D_chain": d_chain,
        "D_control": d_control,
        "chain_stability": chain_summary,
        "control_stability": control_summary,
        "draws": draws,
        "settings": settings,
        "duplicate_audit": audited["audit"],
        "occurrence_list_sha256": occurrence_list_sha256,
        "row_counts": {
            "raw": audited["audit"]["raw_rows"],
            "deduplicated": audited["audit"]["deduplicated_rows"],
            "analyzed": selected_count,
            "sentence_weighted": int(np.sum(labels == target)),
        },
        "chain_counts": {
            "unique": len(chain_to_labels),
            "eligible": len(eligible_chains),
            "selected": selected_count,
            "mixed_label": mixed_label_chains,
        },
    }


def _validate_result_cell_invariants(document: Mapping[str, Any]) -> None:
    """Enforce result-cell arithmetic and registry relations JSON Schema cannot."""
    family_codes = {
        "primary": 1,
        "five_depth": 2,
        "three_annotator": 3,
        "truncation": 4,
        "pooling_window": 5,
        "curvature": 6,
    }
    family = document["family"]
    if family not in family_codes:
        raise ValueError("result cell has an unknown family")
    codes = document["codes"]
    if _json_integer(codes["family"]) != family_codes[family]:
        raise ValueError("result cell family/code registry mismatch")
    purpose = _json_integer(codes["purpose"])
    permitted_hypotheses = {
        "primary": {"H1", "H2", "H3"},
        "five_depth": {"SECONDARY"},
        "three_annotator": {"SECONDARY"},
        "truncation": {"H3"},
        "pooling_window": {"H4"},
        "curvature": {"CURVATURE_DIAGNOSTIC"},
    }
    if document["hypothesis"] not in permitted_hypotheses[family]:
        raise ValueError("result cell family/hypothesis registry mismatch")
    expected_purpose = (
        2
        if family == "primary" and document["hypothesis"] == "H3"
        else {
            "primary": 1,
            "five_depth": 1,
            "three_annotator": 1,
            "truncation": 3,
            "pooling_window": 1,
            "curvature": 4,
        }[family]
    )
    if purpose != expected_purpose:
        raise ValueError("result cell estimator-purpose registry mismatch")
    _validate_cell_registry(
        family_code=codes["family"],
        cell_index=document["cell_index"],
        behaviour_code=codes["behaviour"],
        annotator_code=codes["annotator"],
        layer=document["layer_zero_based"],
        pooling_code=codes["pooling"],
        window_code=codes["window"],
        truncation_code=codes["truncation"],
        purpose_code=purpose,
    )

    resamples = document["resamples"]
    attempted = _json_integer(resamples["attempted"])
    valid = _json_integer(resamples["valid"])
    invalid = _json_integer(resamples["invalid"])
    if attempted != valid + invalid:
        raise ValueError("result resample counts are inconsistent")
    invalid_reason_total = sum(
        _json_integer(count) for count in resamples["invalid_reasons"].values()
    )
    if invalid_reason_total != invalid:
        raise ValueError("result resample counts do not match invalid reasons")
    if attempted > _json_integer(resamples["attempt_cap"]):
        raise ValueError("result resample counts exceed the attempt cap")
    seed = resamples["seed_coordinates"]
    if (
        _json_integer(seed["purpose_code"]) != purpose
        or _json_integer(seed["family_code"]) != family_codes[family]
        or _json_integer(seed["cell_index"]) != _json_integer(document["cell_index"])
    ):
        raise ValueError("result resample seed coordinates are inconsistent")

    status = document["status"]
    inferential = (
        family in {"five_depth", "three_annotator", "pooling_window"}
        or (family == "primary" and document["hypothesis"] == "H2")
    )
    if status != "VALID":
        if (
            document["observed_cdim"] is not None
            or document["observed_pr"] is not None
            or document["raw_p"] is not None
            or document["holm_p"] is not None
            or document["null_cdim"]
        ):
            raise ValueError("non-valid result cell exposes result statistics")
        return

    def positive_number(value: Any) -> bool:
        return bool(
            not isinstance(value, bool)
            and isinstance(value, (int, float, np.number))
            and np.isfinite(float(value))
            and float(value) > 0
        )

    if inferential:
        if not positive_number(document["observed_cdim"]):
            raise ValueError("valid inferential result lacks observed cdim")
        if not all(
            not isinstance(value, bool)
            and isinstance(value, (int, float, np.number))
            and np.isfinite(float(value))
            and 0 <= float(value) <= 1
            for value in (document["raw_p"], document["holm_p"])
        ):
            raise ValueError("valid inferential result lacks p-values")
        if (
            _json_integer(resamples["requested"]) != 2500
            or _json_integer(resamples["attempt_cap"]) != 2750
            or valid != 2500
            or len(document["null_cdim"]) != 2500
        ):
            raise ValueError("valid inferential resample counts are inconsistent")
        if family == "pooling_window":
            if not positive_number(document["observed_pr"]) or not positive_number(
                document["sentence_weighted_cdim"]
            ):
                raise ValueError("valid H4 result lacks relevant observed statistics")
        elif document["observed_pr"] is not None:
            raise ValueError("cdim-only inferential result has an unexpected PR")
    else:
        if document["raw_p"] is not None or document["holm_p"] is not None:
            raise ValueError("noninferential result contains p-values")
        if document["null_cdim"]:
            raise ValueError("noninferential result contains null cdim draws")
        matched_truncation = (
            family == "truncation"
            and 60 <= _json_integer(document["cell_index"]) <= 63
        )
        if family == "primary" and document["hypothesis"] == "H1":
            if not positive_number(document["observed_cdim"]):
                raise ValueError("valid H1 result lacks observed cdim")
        elif family in {"primary", "truncation"} and not matched_truncation:
            if not positive_number(document["observed_cdim"]) or not positive_number(
                document["observed_pr"]
            ):
                raise ValueError("valid H3 result lacks observed statistics")
        if (
            family in {"truncation", "curvature"}
            and not matched_truncation
        ) or (
            family == "primary" and document["hypothesis"] == "H3"
        ):
            valid_draws = sum(
                draw["status"] == "VALID" for draw in document["stability_draws"]
            )
            if (
                _json_integer(resamples["requested"]) != 500
                or _json_integer(resamples["attempt_cap"]) != 500
                or attempted != 500
                or valid < 475
                or len(document["stability_draws"]) != 500
            ):
                raise ValueError("valid diagnostic resample counts are inconsistent")
            if valid_draws != valid:
                raise ValueError(
                    "valid diagnostic draw counts are inconsistent"
                )
        if matched_truncation:
            if (
                document["observed_cdim"] is not None
                or document["observed_pr"] is not None
                or document["stability_draws"]
                or document["matched_results"] is None
                or any(
                    result["status"] != "VALID"
                    for result in document["matched_results"].values()
                )
            ):
                raise ValueError(
                    "valid category-matched cell must expose only two valid results"
                )


_RESULT_CELL_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "thesis_core_hardening_result_cell_v1.schema.json"
)
_PLANNED_PRIMARY_DESTINATION = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "robustness"
    / "core_hardening"
    / "R1-1.5B"
    / "primary_L27_cdim_null.json"
)
_PLANNED_SECONDARY_DESTINATIONS = {
    "five_depth": (
        Path(__file__).resolve().parents[1]
        / "results"
        / "robustness"
        / "core_hardening"
        / "R1-1.5B"
        / "five_depth_cdim_null.json"
    ),
    "three_annotator": (
        Path(__file__).resolve().parents[1]
        / "results"
        / "robustness"
        / "core_hardening"
        / "R1-1.5B"
        / "three_annotator_cdim_null.json"
    ),
}
_PLANNED_SCOPE_DESTINATIONS = {
    "chain_stability": (
        Path(__file__).resolve().parents[1]
        / "results"
        / "robustness"
        / "core_hardening"
        / "R1-1.5B"
        / "chain_stability.json"
    ),
    "truncation": (
        Path(__file__).resolve().parents[1]
        / "results"
        / "robustness"
        / "core_hardening"
        / "R1-1.5B"
        / "truncation_sensitivity.json"
    ),
    "curvature": (
        Path(__file__).resolve().parents[1]
        / "results"
        / "robustness"
        / "core_hardening"
        / "R1-1.5B"
        / "curvature_L16.json"
    ),
}
_PRIMARY_AMENDMENT_IDS = (
    "2026-07-26-2345-same-window-duplicate-remedy",
    "2026-07-26-2358-full-run-authorization",
)
_SECONDARY_AMENDMENT_IDS = (
    "2026-07-26-2345-same-window-duplicate-remedy",
    "2026-07-26-2358-full-run-authorization",
    "2026-07-27-secondary-implementation-lock",
)
_SCOPE_AMENDMENT_IDS = (
    "2026-07-26-2345-same-window-duplicate-remedy",
    "2026-07-26-2358-full-run-authorization",
    "2026-07-27-scope-diagnostic-implementation-lock",
)
_PRIMARY_LABELS = (
    "backtracking",
    "uncertainty-estimation",
    "example-testing",
    "adding-knowledge",
)
_SECONDARY_ANNOTATORS = {
    1: "Sonnet",
    2: "Qwen3-235B",
    3: "Nova-Pro",
}


def _public_duplicate_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "raw_rows",
        "deduplicated_rows",
        "duplicate_fraction",
        "occurrence_duplicate_groups",
        "occurrence_duplicate_rows",
        "vector_duplicate_groups",
        "vector_duplicate_rows",
        "cross_chain_groups",
        "cross_label_groups",
    )
    if any(field not in audit for field in fields):
        raise ValueError("duplicate audit is incomplete")
    return {field: audit[field] for field in fields}


def _result_cell_settings() -> dict[str, Any]:
    return {
        "correlation_dimension": {
            "dtype": "float64",
            "n_radii": 20,
            "random_state": 42,
            "n_bootstrap": 0,
            "subsample": 2000,
            "distance": "euclidean",
        },
        "participation_ratio": {
            "dtype": "float64",
            "column_centered": True,
            "svd": "full",
            "component_cap": None,
        },
        "curvature": None,
    }


def _result_cell_preprocessing() -> dict[str, Any]:
    return {
        "cdim_centered": False,
        "cdim_scaled": False,
        "cdim_normalized": False,
        "cdim_whitened": False,
        "duplicate_order": ["occurrence_key", "exact_vector"],
    }


def _validate_result_cell_contract(document: Mapping[str, Any]) -> None:
    schema = json.loads(_RESULT_CELL_SCHEMA_PATH.read_text(encoding="utf-8"))
    if set(document) != set(schema["required"]):
        raise ValueError("result cell fields do not match the frozen schema")
    _validate_result_cell_invariants(document)
    json.dumps(document, allow_nan=False)


def _build_primary_result_cell(
    cell: Mapping[str, Any],
    *,
    hypothesis: str,
    role: str,
    decision: str,
    input_hashes: Mapping[str, str],
    provenance: Mapping[str, Any],
    amendment_ids: Sequence[str],
) -> dict[str, Any]:
    if role not in {"descriptive", "registered", "diagnostic"}:
        raise ValueError("unknown primary result-cell role")
    behaviour_code = _json_integer(cell["behaviour_code"])
    label = _PRIMARY_LABELS[behaviour_code - 1]
    if hypothesis == "H1":
        requested = attempt_cap = attempted = valid = invalid = 0
        invalid_reasons: dict[str, int] = {}
        base_seed = 20260726
        raw_p = holm_p = None
        null_cdim: list[float] = []
    elif hypothesis == "H2" and role in {"registered", "diagnostic"}:
        prefix = "" if role == "registered" else "diagnostic_"
        resamples = cell[f"{prefix}resamples"]
        requested = 2500
        attempt_cap = 2750
        attempted = _json_integer(resamples["attempted"])
        valid = _json_integer(resamples["valid"])
        invalid = _json_integer(resamples["invalid"])
        invalid_reasons = dict(resamples["invalid_reasons"])
        base_seed = 20260726 if role == "registered" else 20260727
        raw_p = float(cell[f"{prefix}raw_p"])
        holm_p = float(cell[f"{prefix}holm_p"])
        null_cdim = [
            float(value) for value in cell[f"{prefix}null_cdim"]
        ]
    else:
        raise ValueError("invalid primary hypothesis/role combination")
    document = {
        "schema_version": "thesis-core-hardening-result-cell-v1",
        "status": "VALID",
        "family": "primary",
        "hypothesis": hypothesis,
        "cell_index": _json_integer(cell["cell_index"]),
        "codes": {
            "family": 1,
            "purpose": 1,
            "annotator": 1,
            "behaviour": behaviour_code,
            "pooling": 1,
            "window": 1,
            "truncation": 0,
        },
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "annotator": "Sonnet",
        "label": label,
        "pooling": "mean",
        "window": "unclipped",
        "layer_zero_based": 27,
        "input_hashes": dict(input_hashes),
        "estimator_settings": _result_cell_settings(),
        "preprocessing": _result_cell_preprocessing(),
        "duplicate_audit": _public_duplicate_audit(
            cell["duplicate_audit"]
        ),
        "provenance": dict(provenance),
        "occurrence_list_sha256": cell["occurrence_list_sha256"],
        "row_counts": dict(cell["row_counts"]),
        "chain_counts": dict(cell["chain_counts"]),
        "resamples": {
            "requested": requested,
            "attempt_cap": attempt_cap,
            "attempted": attempted,
            "valid": valid,
            "invalid": invalid,
            "invalid_reasons": invalid_reasons,
            "seed_coordinates": {
                "base_seed": base_seed,
                "purpose_code": 1,
                "family_code": 1,
                "cell_index": _json_integer(cell["cell_index"]),
            },
        },
        "observed_cdim": float(cell["observed_cdim"]),
        "observed_pr": None,
        "sentence_weighted_cdim": None,
        "sentence_weighted_label": None,
        "chain_attrition_reasons": dict(
            cell["chain_attrition_reasons"]
        ),
        "raw_p": raw_p,
        "holm_p": holm_p,
        "null_cdim": null_cdim,
        "stability_draws": [],
        "matched_results": None,
        "decision": decision,
        "amendment_ids": list(amendment_ids),
    }
    _validate_result_cell_contract(document)
    return document


def _validate_primary_family_document(document: Mapping[str, Any]) -> None:
    expected_fields = {
        "schema_version",
        "status",
        "family",
        "model_id",
        "preregistration",
        "cell_schema",
        "input_hashes",
        "amendment_ids",
        "registered_seed",
        "diagnostic_seed",
        "hypotheses",
        "h1_cells",
        "h2_registered_cells",
        "h2_diagnostic_cells",
        "provenance",
    }
    if set(document) != expected_fields:
        raise ValueError("primary family fields do not match the frozen envelope")
    if (
        document["schema_version"]
        != "thesis-core-hardening-primary-family-v1"
        or document["status"] != "VALID"
        or document["family"] != "primary"
        or document["registered_seed"] != 20260726
        or document["diagnostic_seed"] != 20260727
    ):
        raise ValueError("primary family envelope registry mismatch")
    for field in (
        "h1_cells",
        "h2_registered_cells",
        "h2_diagnostic_cells",
    ):
        cells = document[field]
        if len(cells) != 4:
            raise ValueError("primary family envelope is incomplete")
        for cell in cells:
            _validate_result_cell_contract(cell)
    json.dumps(document, allow_nan=False)


def build_primary_family_document(
    family_result: Mapping[str, Any],
    *,
    input_hashes: Mapping[str, str],
    preregistration_path: str,
    preregistration_sha256: str,
    result_schema_sha256: str,
    provenance: Mapping[str, Any],
    amendment_ids: Sequence[str],
) -> dict[str, Any]:
    """Build the sealed primary envelope from one complete registered family."""
    if family_result.get("status") != "VALID":
        raise ValueError("cannot seal an incomplete primary family")
    cells = list(family_result["cells"])
    if len(cells) != 4:
        raise ValueError("cannot seal a partial primary family")
    h1_decisions = family_result["hypotheses"]["H1"][
        "behaviour_decisions"
    ]
    h2_decisions = family_result["hypotheses"]["H2"][
        "behaviour_decisions"
    ]
    h1_cells = [
        _build_primary_result_cell(
            cell,
            hypothesis="H1",
            role="descriptive",
            decision=h1_decisions[_json_integer(cell["behaviour_code"])],
            input_hashes=input_hashes,
            provenance=provenance,
            amendment_ids=amendment_ids,
        )
        for cell in cells
    ]
    registered_cells = [
        _build_primary_result_cell(
            cell,
            hypothesis="H2",
            role="registered",
            decision=h2_decisions[_json_integer(cell["behaviour_code"])],
            input_hashes=input_hashes,
            provenance=provenance,
            amendment_ids=amendment_ids,
        )
        for cell in cells
    ]
    diagnostic_cells = [
        _build_primary_result_cell(
            cell,
            hypothesis="H2",
            role="diagnostic",
            decision=h2_decisions[_json_integer(cell["behaviour_code"])],
            input_hashes=input_hashes,
            provenance=provenance,
            amendment_ids=amendment_ids,
        )
        for cell in cells
    ]
    document = {
        "schema_version": "thesis-core-hardening-primary-family-v1",
        "status": "VALID",
        "family": "primary",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "preregistration": {
            "path": preregistration_path,
            "sha256": preregistration_sha256,
        },
        "cell_schema": {
            "path": (
                "schemas/"
                "thesis_core_hardening_result_cell_v1.schema.json"
            ),
            "sha256": result_schema_sha256,
        },
        "input_hashes": dict(input_hashes),
        "amendment_ids": list(amendment_ids),
        "registered_seed": 20260726,
        "diagnostic_seed": 20260727,
        "hypotheses": json.loads(
            json.dumps(
                family_result["hypotheses"],
                sort_keys=True,
                allow_nan=False,
            )
        ),
        "h1_cells": h1_cells,
        "h2_registered_cells": registered_cells,
        "h2_diagnostic_cells": diagnostic_cells,
        "provenance": dict(provenance),
    }
    _validate_primary_family_document(document)
    return document


def _build_secondary_result_cell(
    cell: Mapping[str, Any],
    *,
    family: str,
    input_hashes: Mapping[str, str],
    provenance: Mapping[str, Any],
    amendment_ids: Sequence[str],
) -> dict[str, Any]:
    if family not in {"five_depth", "three_annotator"}:
        raise ValueError("unknown registered CPU secondary family")
    behaviour_code = _json_integer(cell["behaviour_code"])
    annotator_code = _json_integer(cell["annotator_code"])
    family_code = 2 if family == "five_depth" else 3
    resamples = cell["resamples"]
    holm_p = float(cell["holm_p"])
    document = {
        "schema_version": "thesis-core-hardening-result-cell-v1",
        "status": "VALID",
        "family": family,
        "hypothesis": "SECONDARY",
        "cell_index": _json_integer(cell["cell_index"]),
        "codes": {
            "family": family_code,
            "purpose": 1,
            "annotator": annotator_code,
            "behaviour": behaviour_code,
            "pooling": 1,
            "window": 1,
            "truncation": 0,
        },
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "annotator": _SECONDARY_ANNOTATORS[annotator_code],
        "label": _PRIMARY_LABELS[behaviour_code - 1],
        "pooling": "mean",
        "window": "unclipped",
        "layer_zero_based": _json_integer(cell["layer"]),
        "input_hashes": dict(input_hashes),
        "estimator_settings": _result_cell_settings(),
        "preprocessing": _result_cell_preprocessing(),
        "duplicate_audit": _public_duplicate_audit(
            cell["duplicate_audit"]
        ),
        "provenance": dict(provenance),
        "occurrence_list_sha256": cell["occurrence_list_sha256"],
        "row_counts": dict(cell["row_counts"]),
        "chain_counts": dict(cell["chain_counts"]),
        "resamples": {
            "requested": 2500,
            "attempt_cap": 2750,
            "attempted": _json_integer(resamples["attempted"]),
            "valid": _json_integer(resamples["valid"]),
            "invalid": _json_integer(resamples["invalid"]),
            "invalid_reasons": dict(resamples["invalid_reasons"]),
            "seed_coordinates": {
                "base_seed": 20260726,
                "purpose_code": 1,
                "family_code": family_code,
                "cell_index": _json_integer(cell["cell_index"]),
            },
        },
        "observed_cdim": float(cell["observed_cdim"]),
        "observed_pr": None,
        "sentence_weighted_cdim": None,
        "sentence_weighted_label": None,
        "chain_attrition_reasons": dict(
            cell["chain_attrition_reasons"]
        ),
        "raw_p": float(cell["raw_p"]),
        "holm_p": holm_p,
        "null_cdim": [float(value) for value in cell["null_cdim"]],
        "stability_draws": [],
        "matched_results": None,
        "decision": "PASS" if holm_p <= 0.05 else "FAIL",
        "amendment_ids": list(amendment_ids),
    }
    _validate_result_cell_contract(document)
    return document


def _validate_secondary_family_document(
    document: Mapping[str, Any],
) -> None:
    expected_fields = {
        "schema_version",
        "status",
        "family",
        "model_id",
        "preregistration",
        "cell_schema",
        "input_hashes",
        "amendment_ids",
        "registered_seed",
        "decision",
        "behaviour_decisions",
        "cells",
        "provenance",
    }
    if set(document) != expected_fields:
        raise ValueError(
            "secondary family fields do not match the frozen envelope"
        )
    family = document["family"]
    if (
        document["schema_version"]
        != "thesis-core-hardening-secondary-family-v1"
        or document["status"] != "VALID"
        or family not in {"five_depth", "three_annotator"}
        or document["registered_seed"] != 20260726
        or document["decision"] not in {"PASS", "MIXED", "FAIL"}
    ):
        raise ValueError("secondary family envelope registry mismatch")
    expected = _expected_family_cells(family)
    cells = document["cells"]
    if len(cells) != len(expected):
        raise ValueError("secondary family envelope is incomplete")
    if [cell["cell_index"] for cell in cells] != [
        coordinates["cell_index"] for coordinates in expected
    ]:
        raise ValueError("secondary family envelope order is not registered")
    if set(document["behaviour_decisions"]) != {"1", "2", "3", "4"}:
        raise ValueError("secondary behaviour decisions are incomplete")
    for cell in cells:
        _validate_result_cell_contract(cell)
        if cell["family"] != family:
            raise ValueError("secondary cell family does not match envelope")
    json.dumps(document, allow_nan=False)


def build_secondary_family_document(
    family_result: Mapping[str, Any],
    *,
    family: str,
    input_hashes: Mapping[str, str],
    preregistration_path: str,
    preregistration_sha256: str,
    result_schema_sha256: str,
    provenance: Mapping[str, Any],
    amendment_ids: Sequence[str],
) -> dict[str, Any]:
    """Build a sealed complete registered CPU-secondary family envelope."""
    if family_result.get("status") != "VALID":
        raise ValueError("cannot seal an incomplete secondary family")
    expected = _expected_family_cells(family)
    cells = list(family_result["cells"])
    if len(cells) != len(expected):
        raise ValueError("cannot seal a partial secondary family")
    for cell, coordinates in zip(cells, expected):
        if any(
            _json_integer(cell[key]) != value
            for key, value in coordinates.items()
        ):
            raise ValueError("secondary cell/code registry mismatch")
    document = {
        "schema_version": "thesis-core-hardening-secondary-family-v1",
        "status": "VALID",
        "family": family,
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "preregistration": {
            "path": preregistration_path,
            "sha256": preregistration_sha256,
        },
        "cell_schema": {
            "path": (
                "schemas/"
                "thesis_core_hardening_result_cell_v1.schema.json"
            ),
            "sha256": result_schema_sha256,
        },
        "input_hashes": dict(input_hashes),
        "amendment_ids": list(amendment_ids),
        "registered_seed": 20260726,
        "decision": family_result["decision"],
        "behaviour_decisions": json.loads(
            json.dumps(
                family_result["behaviour_decisions"],
                sort_keys=True,
                allow_nan=False,
            )
        ),
        "cells": [
            _build_secondary_result_cell(
                cell,
                family=family,
                input_hashes=input_hashes,
                provenance=provenance,
                amendment_ids=amendment_ids,
            )
            for cell in cells
        ],
        "provenance": dict(provenance),
    }
    _validate_secondary_family_document(document)
    return document


def _diagnostic_stability_draws(
    draws: Sequence[Mapping[str, Any]],
    *,
    left_field: str,
    right_field: str,
) -> list[dict[str, Any]]:
    """Project internal diagnostic draws into the frozen cell schema."""
    projected: list[dict[str, Any]] = []
    for draw in draws:
        status = draw["status"]
        if status not in {"VALID", "INVALID", "UNRUN"}:
            raise ValueError("diagnostic draw has an invalid status")
        projected.append(
            {
                "replicate": _json_integer(draw["replicate"]),
                "status": status,
                "cdim": (
                    float(draw[left_field]) if status == "VALID" else None
                ),
                "pr": (
                    float(draw[right_field]) if status == "VALID" else None
                ),
                "failure": None if status == "VALID" else str(draw["failure"]),
            }
        )
    return projected


def _diagnostic_resamples(
    draws: Sequence[Mapping[str, Any]],
    *,
    family_code: int,
    purpose_code: int,
    cell_index: int,
) -> dict[str, Any]:
    valid = sum(draw["status"] == "VALID" for draw in draws)
    reasons: dict[str, int] = {}
    for draw in draws:
        if draw["status"] != "VALID":
            reason = str(draw.get("failure") or "diagnostic_invalid")
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "requested": 500,
        "attempt_cap": 500,
        "attempted": len(draws),
        "valid": valid,
        "invalid": len(draws) - valid,
        "invalid_reasons": reasons,
        "seed_coordinates": {
            "base_seed": 20260726,
            "purpose_code": purpose_code,
            "family_code": family_code,
            "cell_index": cell_index,
        },
    }


def _build_chain_stability_result_cell(
    result: Mapping[str, Any],
    *,
    behaviour_code: int,
    decision: str,
    input_hashes: Mapping[str, str],
    provenance: Mapping[str, Any],
    amendment_ids: Sequence[str],
) -> dict[str, Any]:
    if result.get("status") != "VALID":
        raise ValueError("cannot seal an incomplete chain-stability cell")
    cell_index = _json_integer(behaviour_code) - 1
    draws = _diagnostic_stability_draws(
        result["draws"], left_field="cdim", right_field="pr"
    )
    document = {
        "schema_version": "thesis-core-hardening-result-cell-v1",
        "status": "VALID",
        "family": "primary",
        "hypothesis": "H3",
        "cell_index": cell_index,
        "codes": {
            "family": 1,
            "purpose": 2,
            "annotator": 1,
            "behaviour": behaviour_code,
            "pooling": 1,
            "window": 1,
            "truncation": 0,
        },
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "annotator": "Sonnet",
        "label": _PRIMARY_LABELS[behaviour_code - 1],
        "pooling": "mean",
        "window": "unclipped",
        "layer_zero_based": 27,
        "input_hashes": dict(input_hashes),
        "estimator_settings": _result_cell_settings(),
        "preprocessing": _result_cell_preprocessing(),
        "duplicate_audit": _public_duplicate_audit(
            result["duplicate_audit"]
        ),
        "provenance": dict(provenance),
        "occurrence_list_sha256": result["occurrence_list_sha256"],
        "row_counts": dict(result["row_counts"]),
        "chain_counts": dict(result["chain_counts"]),
        "resamples": _diagnostic_resamples(
            draws,
            family_code=1,
            purpose_code=2,
            cell_index=cell_index,
        ),
        "observed_cdim": float(result["full_sample_cdim"]),
        "observed_pr": float(result["full_sample_pr"]),
        "sentence_weighted_cdim": None,
        "sentence_weighted_label": None,
        "chain_attrition_reasons": {},
        "raw_p": None,
        "holm_p": None,
        "null_cdim": [],
        "stability_draws": draws,
        "matched_results": None,
        "decision": decision,
        "amendment_ids": list(amendment_ids),
    }
    _validate_result_cell_contract(document)
    return document


def _matched_occurrence_digest(
    matched_results: Mapping[str, Mapping[str, Any]],
) -> str:
    payload = [
        matched_results["matched_complete"]["occurrence_list_sha256"],
        matched_results["matched_truncated"]["occurrence_list_sha256"],
    ]
    serialized = json.dumps(
        payload, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _build_truncation_result_cell(
    behaviour_result: Mapping[str, Any],
    *,
    behaviour_code: int,
    truncation_code: int,
    input_hashes: Mapping[str, str],
    provenance: Mapping[str, Any],
    amendment_ids: Sequence[str],
) -> dict[str, Any]:
    behaviour_rank = _json_integer(behaviour_code) - 1
    if truncation_code not in {1, 2, 3, 4}:
        raise ValueError("invalid truncation stratum code")
    cell_index = 48 + 4 * (truncation_code - 1) + behaviour_rank
    label = _PRIMARY_LABELS[behaviour_rank]
    decision = behaviour_result["h3_behaviour_status"]
    pool_counts = behaviour_result["pool_counts"]
    if truncation_code == 4:
        matched_results = behaviour_result["matched_results"]
        if (
            not behaviour_result["matching_required"]
            or matched_results is None
            or any(
                result["status"] != "VALID"
                for result in matched_results.values()
            )
        ):
            raise ValueError("required category-matched H3 result is incomplete")
        n_analyzed = sum(
            _json_integer(result["n_chains"])
            for result in matched_results.values()
        )
        occurrence_list_sha256 = _matched_occurrence_digest(matched_results)
        status = "VALID"
        observed_cdim = observed_pr = None
        draws: list[dict[str, Any]] = []
        resamples = {
            "requested": 0,
            "attempt_cap": 0,
            "attempted": 0,
            "valid": 0,
            "invalid": 0,
            "invalid_reasons": {},
            "seed_coordinates": {
                "base_seed": 20260726,
                "purpose_code": 3,
                "family_code": 4,
                "cell_index": cell_index,
            },
        }
        top_level_matched = json.loads(
            json.dumps(matched_results, sort_keys=True, allow_nan=False)
        )
    else:
        stratum_name = {1: "combined", 2: "complete", 3: "truncated"}[
            truncation_code
        ]
        stratum = behaviour_result["strata"][stratum_name]
        if stratum["status"] != "VALID":
            raise ValueError("cannot seal an incomplete truncation stratum")
        draws = _diagnostic_stability_draws(
            stratum["stability_draws"],
            left_field="cdim",
            right_field="pr",
        )
        resamples = _diagnostic_resamples(
            draws,
            family_code=4,
            purpose_code=3,
            cell_index=cell_index,
        )
        status = "VALID"
        observed_cdim = float(stratum["cdim"])
        observed_pr = float(stratum["pr"])
        n_analyzed = _json_integer(stratum["n_chains"])
        occurrence_list_sha256 = stratum["occurrence_list_sha256"]
        top_level_matched = None
    document = {
        "schema_version": "thesis-core-hardening-result-cell-v1",
        "status": status,
        "family": "truncation",
        "hypothesis": "H3",
        "cell_index": cell_index,
        "codes": {
            "family": 4,
            "purpose": 3,
            "annotator": 1,
            "behaviour": behaviour_code,
            "pooling": 0,
            "window": 0,
            "truncation": truncation_code,
        },
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "annotator": "Sonnet",
        "label": label,
        "pooling": "not_applicable",
        "window": "not_applicable",
        "layer_zero_based": 27,
        "input_hashes": dict(input_hashes),
        "estimator_settings": _result_cell_settings(),
        "preprocessing": _result_cell_preprocessing(),
        "duplicate_audit": _public_duplicate_audit(
            behaviour_result["duplicate_audit"]
        ),
        "provenance": dict(provenance),
        "occurrence_list_sha256": occurrence_list_sha256,
        "row_counts": {
            "raw": behaviour_result["duplicate_audit"]["raw_rows"],
            "deduplicated": behaviour_result["duplicate_audit"][
                "deduplicated_rows"
            ],
            "analyzed": n_analyzed,
            "sentence_weighted": pool_counts[
                "target_rows_after_deduplication"
            ],
        },
        "chain_counts": {
            "unique": pool_counts["unique_chains"],
            "eligible": n_analyzed,
            "selected": n_analyzed,
            "mixed_label": pool_counts["mixed_label_chains"],
        },
        "resamples": resamples,
        "observed_cdim": observed_cdim,
        "observed_pr": observed_pr,
        "sentence_weighted_cdim": None,
        "sentence_weighted_label": None,
        "chain_attrition_reasons": {},
        "raw_p": None,
        "holm_p": None,
        "null_cdim": [],
        "stability_draws": draws,
        "matched_results": top_level_matched,
        "decision": decision,
        "amendment_ids": list(amendment_ids),
    }
    _validate_result_cell_contract(document)
    return document


def _validate_scope_family_document(
    document: Mapping[str, Any], *, kind: str
) -> None:
    expected_fields = {
        "schema_version",
        "status",
        "kind",
        "model_id",
        "preregistration",
        "cell_schema",
        "input_hashes",
        "amendment_ids",
        "decision",
        "behaviour_decisions",
        "cells",
        "summaries",
        "provenance",
    }
    if set(document) != expected_fields:
        raise ValueError("scope-family fields do not match the frozen envelope")
    if (
        document["schema_version"]
        != "thesis-core-hardening-scope-family-v1"
        or document["status"] != "VALID"
        or document["kind"] != kind
        or document["decision"] not in {"PASS", "MIXED", "FAIL"}
        or set(document["behaviour_decisions"]) != {"1", "2", "3", "4"}
    ):
        raise ValueError("scope-family envelope registry mismatch")
    expected_count = 4 if kind == "chain_stability" else 16
    if len(document["cells"]) != expected_count:
        raise ValueError("scope-family envelope is incomplete")
    for cell in document["cells"]:
        _validate_result_cell_contract(cell)
    json.dumps(document, allow_nan=False)


def build_h3_family_documents(
    behaviour_results: Mapping[int, Mapping[str, Any]],
    *,
    input_hashes: Mapping[str, str],
    preregistration_path: str,
    preregistration_sha256: str,
    result_schema_sha256: str,
    provenance: Mapping[str, Any],
    amendment_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the two sealed H3 envelopes from one exhaustive computation."""
    if set(behaviour_results) != {1, 2, 3, 4}:
        raise ValueError("H3 requires all four behaviour results")
    if any(
        result.get("status") != "VALID"
        for result in behaviour_results.values()
    ):
        raise ValueError("cannot seal an incomplete H3 family")
    h3_decisions = {
        str(code): result["h3_behaviour_status"]
        for code, result in sorted(behaviour_results.items())
    }
    h3_decision = _h3_family_status(list(h3_decisions.values()))
    if h3_decision == "UNRUN":
        raise ValueError("cannot seal an UNRUN H3 family")
    chain_decisions = {
        str(code): result["chain_stability_leg"]
        for code, result in sorted(behaviour_results.items())
    }
    chain_values = list(chain_decisions.values())
    chain_decision = (
        "PASS"
        if all(value == "PASS" for value in chain_values)
        else "FAIL"
        if all(value == "FAIL" for value in chain_values)
        else "MIXED"
    )
    common = {
        "schema_version": "thesis-core-hardening-scope-family-v1",
        "status": "VALID",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "preregistration": {
            "path": preregistration_path,
            "sha256": preregistration_sha256,
        },
        "cell_schema": {
            "path": (
                "schemas/"
                "thesis_core_hardening_result_cell_v1.schema.json"
            ),
            "sha256": result_schema_sha256,
        },
        "input_hashes": dict(input_hashes),
        "amendment_ids": list(amendment_ids),
        "provenance": dict(provenance),
    }
    chain_document = {
        **common,
        "kind": "chain_stability",
        "decision": chain_decision,
        "behaviour_decisions": chain_decisions,
        "cells": [
            _build_chain_stability_result_cell(
                behaviour_results[code]["chain_stability"],
                behaviour_code=code,
                decision=behaviour_results[code]["chain_stability_leg"],
                input_hashes=input_hashes,
                provenance=provenance,
                amendment_ids=amendment_ids,
            )
            for code in range(1, 5)
        ],
        "summaries": {
            str(code): {
                "cdim_stability": behaviour_results[code][
                    "chain_stability"
                ]["cdim_stability"],
                "pr_stability": behaviour_results[code][
                    "chain_stability"
                ]["pr_stability"],
            }
            for code in range(1, 5)
        },
    }
    truncation_document = {
        **common,
        "kind": "truncation",
        "decision": h3_decision,
        "behaviour_decisions": h3_decisions,
        "cells": [
            _build_truncation_result_cell(
                behaviour_results[code],
                behaviour_code=code,
                truncation_code=truncation_code,
                input_hashes=input_hashes,
                provenance=provenance,
                amendment_ids=amendment_ids,
            )
            for truncation_code in range(1, 5)
            for code in range(1, 5)
        ],
        "summaries": {
            str(code): {
                "category_share_imbalance": behaviour_results[code][
                    "category_share_imbalance"
                ],
                "matching_required": behaviour_results[code][
                    "matching_required"
                ],
                "unadjusted_pass": behaviour_results[code][
                    "unadjusted_pass"
                ],
                "matched_pass": behaviour_results[code]["matched_pass"],
                "chain_stability_leg": behaviour_results[code][
                    "chain_stability_leg"
                ],
                "truncation_leg": behaviour_results[code]["truncation_leg"],
                "h3_behaviour_status": behaviour_results[code][
                    "h3_behaviour_status"
                ],
                "descriptives": behaviour_results[code]["descriptives"],
            }
            for code in range(1, 5)
        },
    }
    _validate_scope_family_document(
        chain_document, kind="chain_stability"
    )
    _validate_scope_family_document(
        truncation_document, kind="truncation"
    )
    return chain_document, truncation_document


def _curvature_result_cell_settings() -> dict[str, Any]:
    settings = _result_cell_settings()
    settings["curvature"] = {
        "dtype": "float32",
        "k": 10,
        "variance_threshold": 0.90,
        "n_anchors": 150,
        "random_state": 42,
        "n_bootstrap": 0,
    }
    return settings


def _build_curvature_result_cell(
    result: Mapping[str, Any],
    *,
    behaviour_code: int,
    input_hashes: Mapping[str, str],
    provenance: Mapping[str, Any],
    amendment_ids: Sequence[str],
) -> dict[str, Any]:
    if result.get("status") != "VALID":
        raise ValueError("cannot seal an incomplete curvature cell")
    cell_index = 88 + _json_integer(behaviour_code) - 1
    draws = _diagnostic_stability_draws(
        result["draws"],
        left_field="ratio_chain",
        right_field="ratio_control",
    )
    document = {
        "schema_version": "thesis-core-hardening-result-cell-v1",
        "status": "VALID",
        "family": "curvature",
        "hypothesis": "CURVATURE_DIAGNOSTIC",
        "cell_index": cell_index,
        "codes": {
            "family": 6,
            "purpose": 4,
            "annotator": 1,
            "behaviour": behaviour_code,
            "pooling": 0,
            "window": 0,
            "truncation": 0,
        },
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "annotator": "Sonnet",
        "label": _PRIMARY_LABELS[behaviour_code - 1],
        "pooling": "not_applicable",
        "window": "not_applicable",
        "layer_zero_based": 16,
        "input_hashes": dict(input_hashes),
        "estimator_settings": _curvature_result_cell_settings(),
        "preprocessing": _result_cell_preprocessing(),
        "duplicate_audit": _public_duplicate_audit(
            result["duplicate_audit"]
        ),
        "provenance": dict(provenance),
        "occurrence_list_sha256": result["occurrence_list_sha256"],
        "row_counts": dict(result["row_counts"]),
        "chain_counts": dict(result["chain_counts"]),
        "resamples": _diagnostic_resamples(
            draws,
            family_code=6,
            purpose_code=4,
            cell_index=cell_index,
        ),
        "observed_cdim": None,
        "observed_pr": None,
        "sentence_weighted_cdim": None,
        "sentence_weighted_label": None,
        "chain_attrition_reasons": {},
        "raw_p": None,
        "holm_p": None,
        "null_cdim": [],
        "stability_draws": draws,
        "matched_results": None,
        "decision": "PASS" if result["bounded_negative"] else "MIXED",
        "amendment_ids": list(amendment_ids),
    }
    _validate_result_cell_contract(document)
    return document


def _validate_curvature_family_document(
    document: Mapping[str, Any],
) -> None:
    expected_fields = {
        "schema_version",
        "status",
        "family",
        "model_id",
        "preregistration",
        "cell_schema",
        "input_hashes",
        "amendment_ids",
        "decision",
        "behaviour_decisions",
        "draw_field_semantics",
        "cells",
        "summaries",
        "provenance",
    }
    if set(document) != expected_fields:
        raise ValueError("curvature-family fields do not match the frozen envelope")
    if (
        document["schema_version"]
        != "thesis-core-hardening-curvature-family-v1"
        or document["status"] != "VALID"
        or document["family"] != "curvature"
        or document["decision"] not in {"PASS", "MIXED"}
        or document["draw_field_semantics"]
        != {"cdim": "ratio_chain", "pr": "ratio_control"}
        or set(document["behaviour_decisions"]) != {"1", "2", "3", "4"}
        or len(document["cells"]) != 4
    ):
        raise ValueError("curvature-family envelope registry mismatch")
    for expected_index, cell in enumerate(document["cells"], start=88):
        _validate_result_cell_contract(cell)
        if cell["cell_index"] != expected_index:
            raise ValueError("curvature cells are not in registered order")
    json.dumps(document, allow_nan=False)


def build_curvature_family_document(
    behaviour_results: Mapping[int, Mapping[str, Any]],
    *,
    input_hashes: Mapping[str, str],
    preregistration_path: str,
    preregistration_sha256: str,
    result_schema_sha256: str,
    provenance: Mapping[str, Any],
    amendment_ids: Sequence[str],
) -> dict[str, Any]:
    if set(behaviour_results) != {1, 2, 3, 4} or any(
        result.get("status") != "VALID"
        for result in behaviour_results.values()
    ):
        raise ValueError("cannot seal an incomplete curvature family")
    behaviour_decisions = {
        str(code): (
            "PASS" if behaviour_results[code]["bounded_negative"] else "MIXED"
        )
        for code in range(1, 5)
    }
    document = {
        "schema_version": "thesis-core-hardening-curvature-family-v1",
        "status": "VALID",
        "family": "curvature",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "preregistration": {
            "path": preregistration_path,
            "sha256": preregistration_sha256,
        },
        "cell_schema": {
            "path": (
                "schemas/"
                "thesis_core_hardening_result_cell_v1.schema.json"
            ),
            "sha256": result_schema_sha256,
        },
        "input_hashes": dict(input_hashes),
        "amendment_ids": list(amendment_ids),
        "decision": (
            "PASS"
            if all(value == "PASS" for value in behaviour_decisions.values())
            else "MIXED"
        ),
        "behaviour_decisions": behaviour_decisions,
        "draw_field_semantics": {
            "cdim": "ratio_chain",
            "pr": "ratio_control",
        },
        "cells": [
            _build_curvature_result_cell(
                behaviour_results[code],
                behaviour_code=code,
                input_hashes=input_hashes,
                provenance=provenance,
                amendment_ids=amendment_ids,
            )
            for code in range(1, 5)
        ],
        "summaries": {
            str(code): {
                "n_valid_pairs": behaviour_results[code]["n_valid_pairs"],
                "bounded_negative": behaviour_results[code][
                    "bounded_negative"
                ],
                "wording": behaviour_results[code]["wording"],
                "D_chain": behaviour_results[code]["D_chain"],
                "D_control": behaviour_results[code]["D_control"],
                "chain_stability": behaviour_results[code][
                    "chain_stability"
                ],
                "control_stability": behaviour_results[code][
                    "control_stability"
                ],
            }
            for code in range(1, 5)
        },
        "provenance": dict(provenance),
    }
    _validate_curvature_family_document(document)
    return document


def _atomic_write_json_file(
    destination: Path,
    document: Mapping[str, Any],
    *,
    replace: bool,
) -> None:
    destination = destination.resolve()
    if not replace and destination.exists():
        raise FileExistsError("refusing to overwrite a sealed result")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                document,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if not replace and destination.exists():
            raise FileExistsError("refusing to overwrite a sealed result")
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_primary_family_result(
    destination: str | os.PathLike[str],
    document: Mapping[str, Any],
) -> None:
    """Validate and atomically seal the registered primary family."""
    destination_path = Path(destination).resolve()
    if destination_path != _PLANNED_PRIMARY_DESTINATION.resolve():
        raise ValueError("primary family requires the planned destination")
    _validate_primary_family_document(document)
    _atomic_write_json_file(destination_path, document, replace=False)


def write_secondary_family_result(
    destination: str | os.PathLike[str],
    document: Mapping[str, Any],
    *,
    family: str,
) -> None:
    """Validate and atomically seal one registered CPU-secondary family."""
    if family not in _PLANNED_SECONDARY_DESTINATIONS:
        raise ValueError("unknown registered CPU secondary family")
    destination_path = Path(destination).resolve()
    if (
        destination_path
        != _PLANNED_SECONDARY_DESTINATIONS[family].resolve()
    ):
        raise ValueError("secondary family requires its planned destination")
    if document.get("family") != family:
        raise ValueError("secondary destination/family mismatch")
    _validate_secondary_family_document(document)
    _atomic_write_json_file(destination_path, document, replace=False)


def write_scope_family_result(
    destination: str | os.PathLike[str],
    document: Mapping[str, Any],
    *,
    kind: str,
) -> None:
    """Validate and atomically seal one registered CPU scope diagnostic."""
    if kind not in _PLANNED_SCOPE_DESTINATIONS:
        raise ValueError("unknown scope diagnostic")
    destination_path = Path(destination).resolve()
    if destination_path != _PLANNED_SCOPE_DESTINATIONS[kind].resolve():
        raise ValueError("scope diagnostic requires its planned destination")
    if kind == "curvature":
        _validate_curvature_family_document(document)
    else:
        _validate_scope_family_document(document, kind=kind)
    _atomic_write_json_file(destination_path, document, replace=False)


_RESOURCE_FIELDS = {
    "schema_version",
    "status",
    "mode",
    "input_hashes",
    "b_target",
    "attempted",
    "valid",
    "invalid",
    "workers",
    "invalid_reasons",
    "wall_seconds",
    "seconds_per_attempt",
    "projected_primary_seconds",
    "projected_secondary_seconds",
    "projected_H4_seconds",
    "peak_rss_bytes",
    "scratch_bytes",
    "output_bytes_per_attempt",
    "projected_scratch_bytes",
    "software",
    "process",
    "host",
    "gpu_prerequisites",
    "gate",
    "gates",
}


def _resource_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.findall(r"[a-z0-9]+", normalized)


def _contains_forbidden_resource_terms(value: str, *, key: bool) -> bool:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    tokens = _resource_tokens(normalized)
    pairs = set(zip(tokens, tokens[1:]))
    if (
        "cdim" in tokens
        or "correlationdimension" in normalized
        or ("correlation", "dimension") in pairs
        or ("p", "value") in pairs
        or ("raw", "p") in pairs
        or ("holm", "p") in pairs
    ):
        return True
    if "observed" in tokens or "null" in tokens:
        return True
    if key and (
        normalized.startswith("observed")
        or normalized.startswith("null")
        or "cdim" in normalized
    ):
        return True
    return False


def _forbidden_resource_key(key: str) -> bool:
    return _contains_forbidden_resource_terms(key, key=True)


def _assert_resource_firewall(value: Any) -> None:
    """Reject forbidden statistical field names without echoing their content."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _forbidden_resource_key(str(key)):
                raise ValueError("resource-output firewall violation")
            _assert_resource_firewall(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_resource_firewall(nested)
    elif isinstance(value, str):
        if _contains_forbidden_resource_terms(value, key=False):
            raise ValueError("resource-output firewall violation")


_GPU_PREREQUISITE_FIELDS = {
    "free_scratch_bytes",
    "six_representation_alignment_smoke_pass",
    "projected_extraction_seconds",
    "vram_bytes",
}


def _validate_gpu_prerequisites(
    prerequisites: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if prerequisites is None:
        return None
    if not isinstance(prerequisites, Mapping) or set(prerequisites) != (
        _GPU_PREREQUISITE_FIELDS
    ):
        raise ValueError("GPU prerequisites do not match the exact fact registry")
    normalized = dict(prerequisites)
    for field in ("free_scratch_bytes", "vram_bytes"):
        if (
            isinstance(normalized[field], bool)
            or not isinstance(normalized[field], int)
            or normalized[field] < 0
        ):
            raise ValueError("GPU prerequisite byte count is invalid")
    seconds = normalized["projected_extraction_seconds"]
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not np.isfinite(seconds)
        or seconds < 0
    ):
        raise ValueError("GPU prerequisite time is invalid")
    if not isinstance(
        normalized["six_representation_alignment_smoke_pass"], bool
    ):
        raise ValueError("GPU prerequisite alignment fact is invalid")
    return normalized


def _gpu_prerequisites_pass(prerequisites: Mapping[str, Any] | None) -> bool:
    return bool(
        prerequisites is not None
        and prerequisites["free_scratch_bytes"] >= 100 * 1024**3
        and prerequisites["six_representation_alignment_smoke_pass"] is True
        and prerequisites["projected_extraction_seconds"] <= 24 * 60 * 60
        and prerequisites["vram_bytes"] >= 24 * 1024**3
    )


def _default_peak_rss_bytes() -> int:
    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and the BSDs conventionally report KiB.
    return peak if sys.platform == "darwin" else peak * 1024


def _validate_resource_document(document: Mapping[str, Any]) -> None:
    _assert_resource_firewall(document)
    if set(document) != _RESOURCE_FIELDS:
        raise ValueError("resource document does not match its exact field registry")
    if document["schema_version"] != "thesis-core-hardening-resource-benchmark-v1":
        raise ValueError("invalid resource schema version")
    if document["status"] not in {"UNRUN", "COMPLETE", "INVALID", "STOPPED"}:
        raise ValueError("invalid resource status")
    if document["mode"] not in {"synthetic", "empirical"}:
        raise ValueError("invalid resource mode")
    if document["gate"] not in {"PASS", "FAIL", "UNRUN"}:
        raise ValueError("invalid resource gate")
    if (
        set(document["gates"]) != {"primary", "secondary", "H4_cpu", "H4"}
        or any(
            gate not in {"PASS", "FAIL", "UNRUN"}
            for gate in document["gates"].values()
        )
        or document["gate"] != document["gates"]["primary"]
    ):
        raise ValueError("invalid resource gates")
    _validate_gpu_prerequisites(document["gpu_prerequisites"])
    for path, digest in document["input_hashes"].items():
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError("input hashes must map strings to strings")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("input hash is not lowercase SHA-256")
    for field in (
        "b_target",
        "attempted",
        "valid",
        "invalid",
        "workers",
        "peak_rss_bytes",
        "scratch_bytes",
        "output_bytes_per_attempt",
        "projected_scratch_bytes",
    ):
        if (
            isinstance(document[field], bool)
            or not isinstance(document[field], int)
            or document[field] < 0
        ):
            raise ValueError("resource integer field is invalid")
    for field in (
        "wall_seconds",
        "seconds_per_attempt",
        "projected_primary_seconds",
        "projected_secondary_seconds",
        "projected_H4_seconds",
    ):
        value = document[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("resource numeric field is invalid")
        if not np.isfinite(value) or value < 0:
            raise ValueError("resource numeric field is invalid")
    if document["valid"] + document["invalid"] != document["attempted"]:
        raise ValueError("resource attempt counts are inconsistent")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in document["invalid_reasons"].values()
    ):
        raise ValueError("resource invalid reasons are malformed")
    software = document["software"]
    process = document["process"]
    host = document["host"]
    if (
        not isinstance(software, Mapping)
        or set(software) != {"python", "numpy"}
        or not all(isinstance(value, str) for value in software.values())
    ):
        raise ValueError("resource software facts are malformed")
    if (
        not isinstance(process, Mapping)
        or set(process) != {"pid", "workers"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in process.values()
        )
    ):
        raise ValueError("resource process facts are malformed")
    if (
        not isinstance(host, Mapping)
        or set(host) != {"hostname", "system", "machine", "cpu_count"}
        or not all(
            isinstance(host[field], str)
            for field in ("hostname", "system", "machine")
        )
        or (
            host["cpu_count"] is not None
            and (
                isinstance(host["cpu_count"], bool)
                or not isinstance(host["cpu_count"], int)
                or host["cpu_count"] < 0
            )
        )
    ):
        raise ValueError("resource host facts are malformed")


_FROZEN_HARDENING_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "analysis"
    / "thesis_core_hardening_2026-07-26.yaml"
)


def _frozen_empirical_pilot_authorized() -> bool:
    """Read the separately recorded spend authorization from the frozen config."""
    import yaml

    config = yaml.safe_load(_FROZEN_HARDENING_CONFIG.read_text(encoding="utf-8"))
    execution = config["execution"]
    original = execution["empirical_pilot_authorized_in_this_batch"]
    original_consumed = execution["empirical_pilot_consumed_in_this_batch"]
    retry = execution["empirical_pilot_retry_authorized_in_this_batch"]
    retry_consumed = execution["empirical_pilot_retry_consumed_in_this_batch"]
    if not all(
        isinstance(value, bool)
        for value in (original, original_consumed, retry, retry_consumed)
    ):
        raise ValueError("frozen config empirical gates must be boolean")
    if original and retry:
        raise ValueError("exactly one empirical pilot gate may be authorized")
    if (original and original_consumed) or (retry and retry_consumed):
        raise ValueError("an authorized empirical pilot gate is already consumed")
    return bool(original or retry)


def _frozen_empirical_b50_authorized() -> bool:
    """Read the separate B=50 authorization, which remains false this batch."""
    import yaml

    config = yaml.safe_load(_FROZEN_HARDENING_CONFIG.read_text(encoding="utf-8"))
    authorization = config["execution"]["empirical_B50_authorized_in_this_batch"]
    if not isinstance(authorization, bool):
        raise ValueError("frozen config B=50 authorization must be boolean")
    return authorization


def _frozen_full_primary_authorized() -> bool:
    """Require the one-time frozen gate for the full CPU primary family."""
    import yaml

    config = yaml.safe_load(_FROZEN_HARDENING_CONFIG.read_text(encoding="utf-8"))
    execution = config["execution"]
    fields = {
        name: execution[name]
        for name in (
            "cpu_only",
            "gpu_enabled",
            "api_enabled",
            "full_primary_authorized_in_this_batch",
            "full_primary_consumed_in_this_batch",
            "load_real_activation_matrices_in_this_batch",
            "synthetic_validation_only",
        )
    }
    if not all(isinstance(value, bool) for value in fields.values()):
        raise ValueError("frozen full-primary gates must be boolean")
    if (
        fields["cpu_only"] is not True
        or fields["gpu_enabled"] is not False
        or fields["api_enabled"] is not False
        or fields["full_primary_authorized_in_this_batch"] is not True
        or fields["full_primary_consumed_in_this_batch"] is not False
        or fields["load_real_activation_matrices_in_this_batch"] is not True
        or fields["synthetic_validation_only"] is not False
    ):
        raise ValueError("full primary family is not authorized")
    return True


def _frozen_cpu_secondary_authorized(family: str) -> bool:
    """Require the one-time frozen gate for one registered CPU secondary."""
    import yaml

    if family not in {"five_depth", "three_annotator"}:
        raise ValueError("unknown registered CPU secondary family")
    config = yaml.safe_load(_FROZEN_HARDENING_CONFIG.read_text(encoding="utf-8"))
    execution = config["execution"]
    consumed_field = f"{family}_consumed_in_this_batch"
    fields = {
        name: execution[name]
        for name in (
            "cpu_only",
            "gpu_enabled",
            "api_enabled",
            "cpu_secondary_authorized_in_this_batch",
            consumed_field,
            "load_real_activation_matrices_in_this_batch",
            "synthetic_validation_only",
        )
    }
    if not all(isinstance(value, bool) for value in fields.values()):
        raise ValueError("frozen CPU-secondary gates must be boolean")
    if (
        fields["cpu_only"] is not True
        or fields["gpu_enabled"] is not False
        or fields["api_enabled"] is not False
        or fields["cpu_secondary_authorized_in_this_batch"] is not True
        or fields[consumed_field] is not False
        or fields["load_real_activation_matrices_in_this_batch"] is not True
        or fields["synthetic_validation_only"] is not False
    ):
        raise ValueError(f"{family} CPU secondary is not authorized")
    return True


def _frozen_cpu_scope_authorized(kind: str) -> bool:
    """Require the one-time frozen gate for an available CPU scope diagnostic."""
    import yaml

    if kind not in {"h3", "curvature"}:
        raise ValueError("unknown registered CPU scope diagnostic")
    config = yaml.safe_load(_FROZEN_HARDENING_CONFIG.read_text(encoding="utf-8"))
    execution = config["execution"]
    consumed_field = f"{kind}_consumed_in_this_batch"
    fields = {
        name: execution[name]
        for name in (
            "cpu_only",
            "gpu_enabled",
            "api_enabled",
            "cpu_scope_diagnostics_authorized_in_this_batch",
            consumed_field,
            "load_real_activation_matrices_in_this_batch",
            "synthetic_validation_only",
        )
    }
    if not all(isinstance(value, bool) for value in fields.values()):
        raise ValueError("frozen CPU-scope gates must be boolean")
    if (
        fields["cpu_only"] is not True
        or fields["gpu_enabled"] is not False
        or fields["api_enabled"] is not False
        or fields[
            "cpu_scope_diagnostics_authorized_in_this_batch"
        ] is not True
        or fields[consumed_field] is not False
        or fields["load_real_activation_matrices_in_this_batch"] is not True
        or fields["synthetic_validation_only"] is not False
    ):
        raise ValueError(f"{kind} CPU scope diagnostic is not authorized")
    return True


def _frozen_empirical_cross_label_policy() -> str:
    """Read the amended resource-pilot-only duplicate policy."""
    import yaml

    config = yaml.safe_load(_FROZEN_HARDENING_CONFIG.read_text(encoding="utf-8"))
    duplicate_handling = config["duplicate_handling"]
    if (
        duplicate_handling["cross_chain_policy"] != "hard_fail"
        or duplicate_handling["scope"]
        != "amended_registered_cpu_families"
    ):
        raise ValueError("frozen empirical duplicate policy scope is invalid")
    policy = duplicate_handling[
        "empirical_resource_pilot_cross_label_policy"
    ]
    if policy != "drop_all_same_extraction_window":
        raise ValueError("frozen empirical cross-label policy is invalid")
    return policy


def _frozen_registered_cpu_cross_label_policy() -> str:
    """Read the amended duplicate policy for registered CPU families."""
    import yaml

    config = yaml.safe_load(_FROZEN_HARDENING_CONFIG.read_text(encoding="utf-8"))
    duplicate_handling = config["duplicate_handling"]
    if (
        duplicate_handling["cross_chain_policy"] != "hard_fail"
        or duplicate_handling["scope"]
        != "amended_registered_cpu_families"
    ):
        raise ValueError("frozen registered CPU duplicate policy is invalid")
    policy = duplicate_handling[
        "registered_cpu_family_cross_label_policy"
    ]
    if policy != "drop_all_same_extraction_window":
        raise ValueError("frozen registered CPU cross-label policy is invalid")
    return policy


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _secondary_arm_specs(family: str) -> tuple[dict[str, Any], ...]:
    if family == "five_depth":
        return (
            {
                "annotator_code": 1,
                "activation_root": "data/activations/R1-1.5B",
                "annotation": "data/annotated_R1-1.5B.json",
                "layers": (11, 14, 17, 20, 27),
            },
        )
    if family == "three_annotator":
        return (
            {
                "annotator_code": 1,
                "activation_root": "data/activations/R1-1.5B",
                "annotation": "data/annotated_R1-1.5B.json",
                "layers": (12, 16),
            },
            {
                "annotator_code": 2,
                "activation_root": "data/activations/R1-1.5B-qwenspans",
                "annotation": "data/annotated_R1-1.5B__qwen3-235b.json",
                "layers": (12, 16),
            },
            {
                "annotator_code": 3,
                "activation_root": "data/activations/R1-1.5B-novaspans",
                "annotation": "data/annotated_R1-1.5B__nova-pro.json",
                "layers": (12, 16),
            },
        )
    raise ValueError("unknown registered CPU secondary family")


def _secondary_expected_paths(family: str) -> set[str]:
    paths: set[str] = set()
    for arm in _secondary_arm_specs(family):
        activation_root = arm["activation_root"]
        paths.update(
            {
                f"{activation_root}/metadata.json",
                f"{activation_root}/row_index.json",
                arm["annotation"],
            }
        )
        paths.update(
            f"{activation_root}/{label}_layer{layer}.npy"
            for layer in arm["layers"]
            for label in _PRIMARY_LABELS
        )
    return paths


def _registered_secondary_manifest(
    preregistration_text: str, *, family: str
) -> dict[str, str]:
    """Resolve the exact secondary path/hash registry without guessing."""
    observed: dict[str, set[str]] = {}
    pattern = re.compile(
        r"`((?:data|src)/[^`]+)`\s*\|\s*`([0-9a-f]{64})`"
    )
    for relative_path, digest in pattern.findall(preregistration_text):
        observed.setdefault(relative_path, set()).add(digest)
    manifest: dict[str, str] = {}
    for relative_path in sorted(_secondary_expected_paths(family)):
        digests = observed.get(relative_path, set())
        if len(digests) != 1:
            raise ValueError(
                "secondary input path/hash pairing is absent or ambiguous"
            )
        manifest[relative_path] = next(iter(digests))
    return manifest


def load_registered_secondary_family_inputs(
    family: str,
    *,
    repository_root: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Hash-gate and load one complete registered CPU secondary family."""
    import yaml

    if family not in {"five_depth", "three_annotator"}:
        raise ValueError("unknown registered CPU secondary family")
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    frozen_config = (
        Path(config_path).resolve()
        if config_path is not None
        else _FROZEN_HARDENING_CONFIG.resolve()
    )
    config = yaml.safe_load(frozen_config.read_text(encoding="utf-8"))
    preregistration = config["preregistration"]
    preregistration_path = root / preregistration["path"]
    if (
        not preregistration_path.is_file()
        or _sha256_file(preregistration_path) != preregistration["sha256"]
    ):
        raise ValueError("preregistration hash mismatch before secondary load")
    manifest = _registered_secondary_manifest(
        preregistration_path.read_text(encoding="utf-8"), family=family
    )
    actual_hashes: dict[str, str] = {}
    for relative_path, expected_digest in manifest.items():
        path = root / relative_path
        if not path.is_file():
            raise ValueError("registered secondary input is absent")
        actual_digest = _sha256_file(path)
        if actual_digest != expected_digest:
            raise ValueError("secondary input hash mismatch against registry")
        actual_hashes[relative_path] = actual_digest

    width = _json_integer(config["model"]["hidden_width"])
    pooled_inputs: dict[tuple[int, int], dict[str, Any]] = {}
    for arm in _secondary_arm_specs(family):
        annotator_code = _json_integer(arm["annotator_code"])
        activation_root_relative = arm["activation_root"]
        activation_root = root / activation_root_relative
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
            raise ValueError(
                "secondary row provenance is not the frozen representation"
            )
        rows_by_label = row_index.get("rows")
        if not isinstance(rows_by_label, Mapping):
            raise ValueError("secondary row index is malformed")
        assigned_labels: list[str] = []
        chain_ids: list[str] = []
        occurrences: list[dict[str, Any]] = []
        row_counts: dict[str, int] = {}
        for label in _PRIMARY_LABELS:
            registered_rows = rows_by_label.get(label)
            if not isinstance(registered_rows, list):
                raise ValueError(
                    "secondary row index is missing a registered label"
                )
            row_counts[label] = len(registered_rows)
            for source_index, raw_record in enumerate(registered_rows):
                record = {
                    "behaviour": label,
                    "chain_id": raw_record["chain_id"],
                    "annotation_index": raw_record["annotation_index"],
                    "char_offset": raw_record["char_offset"],
                    "token_start": raw_record["token_start"],
                    "n_positions": raw_record["n_positions"],
                    "source_index": source_index,
                }
                _occurrence_key(record)
                occurrences.append(record)
                assigned_labels.append(label)
                chain_ids.append(record["chain_id"])
        for layer in arm["layers"]:
            matrix_parts: list[np.ndarray] = []
            for label in _PRIMARY_LABELS:
                matrix = np.load(
                    activation_root / f"{label}_layer{layer}.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                )
                if (
                    matrix.ndim != 2
                    or matrix.shape[0] != row_counts[label]
                    or matrix.shape[1] != width
                ):
                    raise ValueError(
                        "row alignment failure in secondary activation matrix"
                    )
                matrix_parts.append(np.asarray(matrix))
            activations = np.concatenate(matrix_parts, axis=0)
            if not (
                activations.shape[0]
                == len(assigned_labels)
                == len(chain_ids)
                == len(occurrences)
            ):
                raise ValueError(
                    "row alignment failure in secondary activation pool"
                )
            pooled_inputs[(annotator_code, int(layer))] = {
                "activations": activations,
                "labels": assigned_labels,
                "chain_ids": chain_ids,
                "occurrences": occurrences,
            }

    cells: dict[int, dict[str, Any]] = {}
    for coordinates in _expected_family_cells(family):
        pool = pooled_inputs[
            (coordinates["annotator_code"], coordinates["layer"])
        ]
        cells[coordinates["cell_index"]] = {
            **coordinates,
            **pool,
            "target_label": _PRIMARY_LABELS[
                coordinates["behaviour_code"] - 1
            ],
        }
    return {"cells": cells, "input_hashes": actual_hashes}


def load_registered_curvature_inputs(
    *,
    repository_root: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Hash-gate and load the canonical Sonnet L16 curvature row pool."""
    import yaml

    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    frozen_config = (
        Path(config_path).resolve()
        if config_path is not None
        else _FROZEN_HARDENING_CONFIG.resolve()
    )
    config = yaml.safe_load(frozen_config.read_text(encoding="utf-8"))
    preregistration = config["preregistration"]
    preregistration_path = root / preregistration["path"]
    if (
        not preregistration_path.is_file()
        or _sha256_file(preregistration_path) != preregistration["sha256"]
    ):
        raise ValueError("preregistration hash mismatch before curvature load")
    preregistration_text = preregistration_path.read_text(encoding="utf-8")
    expected_paths = {
        "src/curvature.py",
        "data/activations/R1-1.5B/metadata.json",
        "data/activations/R1-1.5B/row_index.json",
        "data/annotated_R1-1.5B.json",
        *{
            f"data/activations/R1-1.5B/{label}_layer16.npy"
            for label in _PRIMARY_LABELS
        },
    }
    observed: dict[str, set[str]] = {}
    pattern = re.compile(
        r"`((?:data|src)/[^`]+)`\s*\|\s*`([0-9a-f]{64})`"
    )
    for relative_path, digest in pattern.findall(preregistration_text):
        observed.setdefault(relative_path, set()).add(digest)
    manifest: dict[str, str] = {}
    for relative_path in sorted(expected_paths):
        digests = observed.get(relative_path, set())
        if len(digests) != 1:
            raise ValueError(
                "curvature input path/hash pairing is absent or ambiguous"
            )
        manifest[relative_path] = next(iter(digests))
    actual_hashes: dict[str, str] = {}
    for relative_path, expected_digest in manifest.items():
        path = root / relative_path
        if not path.is_file() or _sha256_file(path) != expected_digest:
            raise ValueError("curvature input hash mismatch against registry")
        actual_hashes[relative_path] = expected_digest

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
        raise ValueError("curvature row provenance is not frozen representation")
    width = _json_integer(config["model"]["hidden_width"])
    rows_by_label = row_index.get("rows")
    if not isinstance(rows_by_label, Mapping):
        raise ValueError("curvature row index is malformed")
    matrix_parts: list[np.ndarray] = []
    occurrences: list[dict[str, Any]] = []
    for label in _PRIMARY_LABELS:
        registered_rows = rows_by_label.get(label)
        if not isinstance(registered_rows, list):
            raise ValueError("curvature row index is missing a registered label")
        matrix = np.load(
            activation_root / f"{label}_layer16.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        if (
            matrix.ndim != 2
            or matrix.shape[0] != len(registered_rows)
            or matrix.shape[1] != width
        ):
            raise ValueError("row alignment failure in curvature activation matrix")
        matrix_parts.append(np.asarray(matrix))
        for source_index, raw_record in enumerate(registered_rows):
            record = {
                "behaviour": label,
                "chain_id": raw_record["chain_id"],
                "annotation_index": raw_record["annotation_index"],
                "char_offset": raw_record["char_offset"],
                "token_start": raw_record["token_start"],
                "n_positions": raw_record["n_positions"],
                "source_index": source_index,
            }
            _occurrence_key(record)
            occurrences.append(record)
    activations = np.concatenate(matrix_parts, axis=0)
    if activations.shape[0] != len(occurrences):
        raise ValueError("row alignment failure in curvature row pool")
    return {
        "activations": activations,
        "occurrences": occurrences,
        "input_hashes": actual_hashes,
    }


def load_empirical_backtracking_l27_inputs(
    *,
    repository_root: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Hash-gate and load the canonical four-label Sonnet/L27 row pool."""
    import yaml

    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    frozen_config = (
        Path(config_path).resolve()
        if config_path is not None
        else _FROZEN_HARDENING_CONFIG.resolve()
    )
    config = yaml.safe_load(frozen_config.read_text(encoding="utf-8"))
    preregistration = config["preregistration"]
    expected_hashes = dict(config["primary_input_hashes"])
    paths_to_check = {
        preregistration["path"]: preregistration["sha256"],
        **expected_hashes,
    }
    actual_hashes: dict[str, str] = {}
    for relative_path, expected_digest in paths_to_check.items():
        path = root / relative_path
        if not path.is_file():
            raise ValueError("input hash mismatch: registered file is absent")
        actual_digest = _sha256_file(path)
        if actual_digest != expected_digest:
            raise ValueError("input hash mismatch against frozen registry")
        if relative_path in expected_hashes:
            actual_hashes[relative_path] = actual_digest

    primary = config["representation"]["primary"]
    if (
        primary["pooling"] != "mean"
        or primary["clip_window_to_sentence_end"] is not False
    ):
        raise ValueError("registered primary representation is not canonical")
    labels = tuple(config["labels"])
    if labels != (
        "backtracking",
        "uncertainty-estimation",
        "example-testing",
        "adding-knowledge",
    ):
        raise ValueError("registered primary label order is not canonical")

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
        raise ValueError("empirical row provenance is not the frozen representation")

    width = _json_integer(config["model"]["hidden_width"])
    matrix_parts: list[np.ndarray] = []
    assigned_labels: list[str] = []
    chain_ids: list[str] = []
    occurrences: list[dict[str, Any]] = []
    rows_by_label = row_index.get("rows")
    if not isinstance(rows_by_label, Mapping):
        raise ValueError("empirical row index is malformed")
    for label in labels:
        registered_rows = rows_by_label.get(label)
        if not isinstance(registered_rows, list):
            raise ValueError("empirical row index is missing a registered label")
        matrix_path = activation_root / f"{label}_layer27.npy"
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        if (
            matrix.ndim != 2
            or matrix.shape[0] != len(registered_rows)
            or matrix.shape[1] != width
        ):
            raise ValueError("row alignment failure in empirical L27 matrix")
        matrix_parts.append(np.asarray(matrix))
        for source_index, raw_record in enumerate(registered_rows):
            record = {
                "behaviour": label,
                "chain_id": raw_record["chain_id"],
                "annotation_index": raw_record["annotation_index"],
                "char_offset": raw_record["char_offset"],
                "token_start": raw_record["token_start"],
                "n_positions": raw_record["n_positions"],
                "source_index": source_index,
            }
            _occurrence_key(record)
            occurrences.append(record)
            assigned_labels.append(label)
            chain_ids.append(record["chain_id"])
    activations = np.concatenate(matrix_parts, axis=0)
    if not (
        activations.shape[0]
        == len(assigned_labels)
        == len(chain_ids)
        == len(occurrences)
    ):
        raise ValueError("row alignment failure in empirical row pool")
    return {
        "activations": activations,
        "labels": assigned_labels,
        "chain_ids": chain_ids,
        "occurrences": occurrences,
        "input_hashes": actual_hashes,
    }


def _empirical_backtracking_attempt_callback(
    activations: np.ndarray,
    labels: Sequence[str],
    chain_ids: Sequence[str],
    occurrences: Sequence[Mapping[str, Any]],
    *,
    statistic_fn: Callable[[np.ndarray], float],
) -> Callable[[int], str | tuple[str, str]]:
    """Prepare the registered cell and return a scalar-discarding attempt callback."""
    values = np.asarray(activations, dtype=np.float64)
    assigned = np.asarray(labels, dtype=str)
    chains = np.asarray(chain_ids, dtype=str)
    if not (
        values.ndim == 2
        and len(assigned) == len(chains) == len(occurrences) == values.shape[0]
    ):
        raise ValueError("row alignment failure before empirical pilot")
    for index, record in enumerate(occurrences):
        key = _occurrence_key(record)
        if (
            key[0] != _nfc(str(assigned[index]))
            or key[1] != _nfc(str(chains[index]))
        ):
            raise ValueError("row alignment failure in empirical pilot records")

    audited = _validate_and_deduplicate(
        values,
        occurrences,
        cross_label_policy=_frozen_empirical_cross_label_policy(),
    )
    kept = np.asarray(audited["kept_indices"], dtype=int)
    values = audited["activations"]
    records = audited["records"]
    assigned = assigned[kept]
    chains = chains[kept]
    unique_chains = sorted(
        {_nfc(str(chain)) for chain in chains},
        key=lambda value: value.encode("utf-8"),
    )
    chain_to_indices = {
        chain: np.flatnonzero(chains == chain) for chain in unique_chains
    }
    mixed = sum(
        1
        for indices in chain_to_indices.values()
        if np.unique(assigned[indices]).size > 1
    )
    if not unique_chains or mixed < 0.5 * len(unique_chains):
        raise ValueError("within-chain permutation informativeness gate failed")

    selector_kwargs = {
        "target_label": "backtracking",
        "family_code": 1,
        "annotator_code": 1,
        "layer": 27,
        "behaviour_code": 1,
        "pooling_code": 1,
        "window_code": 1,
        "truncation_code": 0,
        "replicate_index": -1,
        "scope_key": "",
    }
    observed_indices = _select_equal_chain_indices(
        records, assigned, **selector_kwargs
    )
    if len(observed_indices) < 100:
        raise ValueError("empirical pilot has fewer than 100 equal-chain points")
    observed_chains = {chains[index] for index in observed_indices}
    try:
        observed_check = float(statistic_fn(values[observed_indices]))
    except EstimatorInvalidError as exc:
        raise EstimatorInvalidError(
            "empirical pilot observed assignment is estimator-invalid"
        ) from exc
    if not np.isfinite(observed_check) or observed_check <= 0:
        raise EstimatorInvalidError(
            "empirical pilot observed assignment is estimator-invalid"
        )
    del observed_check

    def attempt(attempt_index: int) -> str | tuple[str, str]:
        rng = _permutation_rng(20260726, 1, 0, attempt_index)
        proposed = assigned.copy()
        for indices in chain_to_indices.values():
            proposed[indices] = rng.permutation(assigned[indices])
        for indices in chain_to_indices.values():
            if sorted(proposed[indices].tolist()) != sorted(
                assigned[indices].tolist()
            ):
                raise RuntimeError("within-chain label counts changed")
        selected = _select_equal_chain_indices(
            records, proposed, **selector_kwargs
        )
        if {chains[index] for index in selected} != observed_chains:
            raise RuntimeError(
                "target-chain eligibility changed under permutation"
            )
        try:
            attempt_check = float(statistic_fn(values[selected]))
        except EstimatorInvalidError:
            return ("INVALID", "estimator_invalid")
        if not np.isfinite(attempt_check) or attempt_check <= 0:
            return ("INVALID", "estimator_invalid")
        del attempt_check
        return "VALID"

    return attempt


def run_empirical_backtracking_resource_pilot(
    activations: np.ndarray,
    labels: Sequence[str],
    chain_ids: Sequence[str],
    occurrences: Sequence[Mapping[str, Any]],
    *,
    input_hashes: Mapping[str, str],
    statistic_fn: Callable[[np.ndarray], float] = correlation_dimension_point,
    clock_fn: Callable[[], float] = time.perf_counter,
    peak_rss_fn: Callable[[], int] = _default_peak_rss_bytes,
) -> dict[str, Any]:
    """Run only the authorized B=25 backtracking/L27 resource benchmark."""
    attempt = _empirical_backtracking_attempt_callback(
        activations,
        labels,
        chain_ids,
        occurrences,
        statistic_fn=statistic_fn,
    )
    return run_resource_benchmark(
        attempt,
        b_target=25,
        synthetic=False,
        empirical_authorization_acknowledged=True,
        input_hashes=input_hashes,
        workers=1,
        scratch_bytes=0,
        output_bytes_per_attempt=0,
        cell_count=4,
        clock_fn=clock_fn,
        peak_rss_fn=peak_rss_fn,
    )


def run_resource_benchmark(
    attempt_callback: Callable[[int], str | bool],
    *,
    b_target: int = 25,
    synthetic: bool = False,
    empirical_authorization_acknowledged: bool = False,
    amendment_allow_b50: bool = False,
    b25_completed_successfully: bool = False,
    gpu_prerequisites: Mapping[str, Any] | None = None,
    input_hashes: Mapping[str, str] | None = None,
    workers: int = 1,
    scratch_bytes: int = 0,
    output_bytes_per_attempt: int = 0,
    cell_count: int = 4,
    alignment_failure: bool = False,
    duplicate_hard_failure: bool = False,
    clock_fn: Callable[[], float] = time.perf_counter,
    peak_rss_fn: Callable[[], int] = _default_peak_rss_bytes,
) -> dict[str, Any]:
    """Measure resource-only attempt validity without retaining statistics."""
    if not isinstance(synthetic, bool):
        raise ValueError("synthetic mode flag must be boolean")
    if not synthetic:
        if not isinstance(empirical_authorization_acknowledged, bool):
            raise ValueError(
                "empirical authorization acknowledgement must be boolean"
            )
        if empirical_authorization_acknowledged is not True:
            raise ValueError(
                "explicit empirical authorization acknowledgement is required"
            )
        if not _frozen_empirical_pilot_authorized():
            raise ValueError(
                "empirical pilot is not authorized by the frozen config"
            )
    if not isinstance(amendment_allow_b50, bool) or not isinstance(
        b25_completed_successfully, bool
    ):
        raise ValueError("B=50 sequencing flags must be boolean")
    b_target = _json_integer(b_target)
    workers = _json_integer(workers)
    scratch_bytes = _json_integer(scratch_bytes)
    output_bytes_per_attempt = _json_integer(output_bytes_per_attempt)
    cell_count = _json_integer(cell_count)
    if min(b_target, workers, scratch_bytes, output_bytes_per_attempt, cell_count) < 0:
        raise ValueError("resource counts must be non-negative")
    if workers == 0 or cell_count == 0:
        raise ValueError("workers and cell_count must be positive")
    if not synthetic and not (
        b_target == 25 or (amendment_allow_b50 and b_target == 50)
    ):
        raise ValueError("empirical resource pilot requires frozen B=25")
    if (
        not synthetic
        and b_target == 50
        and (
            not bool(b25_completed_successfully)
            or not _frozen_empirical_b50_authorized()
        )
    ):
        raise ValueError(
            "empirical B=50 is not authorized by the frozen config"
        )
    normalized_gpu_prerequisites = _validate_gpu_prerequisites(gpu_prerequisites)

    started = float(clock_fn())
    valid = 0
    invalid_reasons: dict[str, int] = {}
    for attempt_index in range(b_target):
        try:
            outcome = attempt_callback(attempt_index)
        except Exception:
            outcome = "INVALID"
            reason = "exception"
        else:
            reason = "attempt_invalid"
            if (
                isinstance(outcome, tuple)
                and len(outcome) == 2
                and outcome[0] == "INVALID"
                and isinstance(outcome[1], str)
                and re.fullmatch(r"[a-z][a-z0-9_]*", outcome[1])
                and not _contains_forbidden_resource_terms(
                    outcome[1], key=False
                )
            ):
                outcome, reason = outcome
        if outcome is True or outcome == "VALID":
            valid += 1
        elif outcome is False or outcome == "INVALID":
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
        else:
            invalid_reasons["invalid_status"] = (
                invalid_reasons.get("invalid_status", 0) + 1
            )
    ended = float(clock_fn())
    wall_seconds = max(0.0, ended - started)
    attempted = b_target
    invalid = attempted - valid
    seconds_per_attempt = wall_seconds / attempted if attempted else 0.0
    projected_primary = 1.25 * seconds_per_attempt * 2750 * 4
    projected_secondary = 1.25 * seconds_per_attempt * 2750 * 44
    projected_h4 = 1.25 * seconds_per_attempt * 2750 * 24
    projected_scratch = math.ceil(
        1.25
        * max(
            scratch_bytes,
            output_bytes_per_attempt * 2750 * cell_count,
        )
    )
    peak_rss = max(0, int(peak_rss_fn()))
    valid_rate = valid / attempted if attempted else 0.0
    common_quality_passes = (
        attempted > 0
        and peak_rss <= 48 * 1024**3
        and valid_rate >= 0.99
        and not bool(alignment_failure)
        and not bool(duplicate_hard_failure)
        and projected_scratch <= 10 * 1024**3
    )
    h4_cpu = (
        "PASS"
        if common_quality_passes and projected_h4 <= 48 * 60 * 60
        else ("FAIL" if attempted else "UNRUN")
    )
    gates = {
        "primary": (
            "PASS"
            if common_quality_passes and projected_primary <= 24 * 60 * 60
            else ("FAIL" if attempted else "UNRUN")
        ),
        "secondary": (
            "PASS"
            if common_quality_passes and projected_secondary <= 72 * 60 * 60
            else ("FAIL" if attempted else "UNRUN")
        ),
        "H4_cpu": h4_cpu,
        "H4": (
            "PASS"
            if h4_cpu == "PASS"
            and _gpu_prerequisites_pass(normalized_gpu_prerequisites)
            else "UNRUN"
        ),
    }
    document = {
        "schema_version": "thesis-core-hardening-resource-benchmark-v1",
        "status": "COMPLETE" if attempted else "UNRUN",
        "mode": "synthetic" if synthetic else "empirical",
        "input_hashes": dict(input_hashes or {}),
        "b_target": b_target,
        "attempted": attempted,
        "valid": valid,
        "invalid": invalid,
        "workers": workers,
        "invalid_reasons": invalid_reasons,
        "wall_seconds": float(wall_seconds),
        "seconds_per_attempt": float(seconds_per_attempt),
        "projected_primary_seconds": float(projected_primary),
        "projected_secondary_seconds": float(projected_secondary),
        "projected_H4_seconds": float(projected_h4),
        "peak_rss_bytes": peak_rss,
        "scratch_bytes": scratch_bytes,
        "output_bytes_per_attempt": output_bytes_per_attempt,
        "projected_scratch_bytes": projected_scratch,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "process": {"pid": os.getpid(), "workers": workers},
        "host": {
            "hostname": socket.gethostname(),
            "system": platform.system(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "gpu_prerequisites": normalized_gpu_prerequisites,
        "gate": gates["primary"],
        "gates": gates,
    }
    _validate_resource_document(document)
    return document


_PLANNED_EMPIRICAL_RESOURCE_DESTINATION = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "robustness"
    / "cdim_null_pilot"
    / "R1-1.5B"
    / "backtracking_L27_resource_only.json"
)


def write_resource_only_benchmark(
    destination: str | os.PathLike[str], document: Mapping[str, Any]
) -> None:
    """Validate and atomically write the public resource-only document."""
    _validate_resource_document(document)
    destination_path = Path(destination).resolve()
    planned_destination = _PLANNED_EMPIRICAL_RESOURCE_DESTINATION.resolve()
    if document["mode"] == "synthetic":
        if destination_path == planned_destination:
            raise ValueError(
                "synthetic benchmark cannot use the planned empirical destination"
            )
        if "synthetic" not in destination_path.name.casefold():
            raise ValueError(
                "synthetic benchmark destination must be marked synthetic"
            )
    elif destination_path != planned_destination:
        raise ValueError(
            "empirical benchmark requires the planned empirical destination"
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=destination_path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                document,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_resource_benchmark_markdown(
    destination: str | os.PathLike[str], document: Mapping[str, Any]
) -> None:
    """Atomically write a human-readable view of only firewall-safe fields."""
    _validate_resource_document(document)
    lines = [
        f"# Resource-only B={document['b_target']} benchmark",
        "",
        f"- Status: `{document['status']}`",
        f"- Mode: `{document['mode']}`",
        f"- Attempts: `{document['attempted']}`",
        f"- Valid: `{document['valid']}`",
        f"- Invalid: `{document['invalid']}`",
        f"- Wall seconds: `{document['wall_seconds']:.6f}`",
        f"- Seconds per attempt: `{document['seconds_per_attempt']:.6f}`",
        f"- Peak RSS bytes: `{document['peak_rss_bytes']}`",
        f"- Projected primary seconds: `{document['projected_primary_seconds']:.6f}`",
        f"- Projected secondary seconds: `{document['projected_secondary_seconds']:.6f}`",
        f"- Projected H4 seconds: `{document['projected_H4_seconds']:.6f}`",
        f"- Projected scratch bytes: `{document['projected_scratch_bytes']}`",
        "",
        "## Gate decisions",
        "",
        *[
            f"- {name}: `{status}`"
            for name, status in document["gates"].items()
        ],
        "",
        "## Invalid reasons",
        "",
        *(
            [
                f"- {reason}: `{count}`"
                for reason, count in sorted(
                    document["invalid_reasons"].items()
                )
            ]
            or ["- None"]
        ),
        "",
        "## Input hashes",
        "",
        *[
            f"- `{path}`: `{digest}`"
            for path, digest in sorted(document["input_hashes"].items())
        ],
        "",
    ]
    rendered = "\n".join(lines)
    destination_path = Path(destination).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=destination_path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _stopped_empirical_resource_document(
    input_hashes: Mapping[str, str],
    *,
    reason: str,
) -> dict[str, Any]:
    """Build a firewall-safe zero-attempt document for a preflight hard stop."""
    if (
        not isinstance(reason, str)
        or re.fullmatch(r"[a-z][a-z0-9_]*", reason) is None
        or _contains_forbidden_resource_terms(reason, key=False)
    ):
        raise ValueError("invalid resource stop reason")
    document = {
        "schema_version": "thesis-core-hardening-resource-benchmark-v1",
        "status": "STOPPED",
        "mode": "empirical",
        "input_hashes": dict(input_hashes),
        "b_target": 25,
        "attempted": 0,
        "valid": 0,
        "invalid": 0,
        "workers": 1,
        "invalid_reasons": {reason: 1},
        "wall_seconds": 0.0,
        "seconds_per_attempt": 0.0,
        "projected_primary_seconds": 0.0,
        "projected_secondary_seconds": 0.0,
        "projected_H4_seconds": 0.0,
        "peak_rss_bytes": _default_peak_rss_bytes(),
        "scratch_bytes": 0,
        "output_bytes_per_attempt": 0,
        "projected_scratch_bytes": 0,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "process": {"pid": os.getpid(), "workers": 1},
        "host": {
            "hostname": socket.gethostname(),
            "system": platform.system(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "gpu_prerequisites": None,
        "gate": "FAIL",
        "gates": {
            "primary": "FAIL",
            "secondary": "UNRUN",
            "H4_cpu": "UNRUN",
            "H4": "UNRUN",
        },
    }
    _validate_resource_document(document)
    return document


def execute_empirical_resource_pilot_from_disk(
    destination: str | os.PathLike[str],
) -> dict[str, Any]:
    """Load, run, and write the one authorized empirical resource pilot."""
    destination_path = Path(destination).resolve()
    if destination_path != _PLANNED_EMPIRICAL_RESOURCE_DESTINATION.resolve():
        raise ValueError(
            "empirical benchmark requires the planned empirical destination"
        )
    inputs = load_empirical_backtracking_l27_inputs()
    try:
        document = run_empirical_backtracking_resource_pilot(
            inputs["activations"],
            inputs["labels"],
            inputs["chain_ids"],
            inputs["occurrences"],
            input_hashes=inputs["input_hashes"],
        )
    except DuplicateAuditError:
        document = _stopped_empirical_resource_document(
            inputs["input_hashes"], reason="duplicate_hard_failure"
        )
    except EstimatorInvalidError:
        document = _stopped_empirical_resource_document(
            inputs["input_hashes"], reason="estimator_invalid"
        )
    except ValueError:
        document = _stopped_empirical_resource_document(
            inputs["input_hashes"], reason="preflight_failure"
        )
    write_resource_only_benchmark(destination_path, document)
    write_resource_benchmark_markdown(
        destination_path.with_name("BENCHMARK.md"), document
    )
    return document


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def execute_registered_primary_from_disk(
    destination: str | os.PathLike[str],
    *,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Execute and atomically seal the complete registered CPU primary family."""
    _frozen_full_primary_authorized()
    destination_path = Path(destination).resolve()
    if destination_path != _PLANNED_PRIMARY_DESTINATION.resolve():
        raise ValueError("primary family requires the planned destination")
    if destination_path.exists():
        raise FileExistsError("refusing to rerun a sealed primary family")
    progress_path = destination_path.with_name(
        f".{destination_path.name}.progress.json"
    )
    if progress_path.exists():
        raise FileExistsError(
            "an unresolved primary-family progress artifact already exists"
        )

    started_utc = _utc_now()
    started = time.perf_counter()
    inputs = load_empirical_backtracking_l27_inputs()
    cells: dict[int, dict[str, Any]] = {}
    for coordinates in _expected_family_cells("primary"):
        behaviour_code = coordinates["behaviour_code"]
        cells[coordinates["cell_index"]] = {
            **coordinates,
            "activations": inputs["activations"],
            "labels": inputs["labels"],
            "chain_ids": inputs["chain_ids"],
            "occurrences": inputs["occurrences"],
            "target_label": _PRIMARY_LABELS[behaviour_code - 1],
        }

    def record_progress(completed: Sequence[Mapping[str, Any]]) -> None:
        _atomic_write_json_file(
            progress_path,
            {
                "schema_version": (
                    "thesis-core-hardening-primary-progress-v1"
                ),
                "status": "IN_PROGRESS",
                "completed_cells": len(completed),
                "registered_seed": 20260726,
                "diagnostic_seed": 20260727,
                "cells": list(completed),
            },
            replace=True,
        )

    family_result = _run_registered_cdim_family(
        cells,
        family="primary",
        B=2500,
        attempt_cap=2750,
        cross_label_policy=_frozen_registered_cpu_cross_label_policy(),
        progress_callback=record_progress,
    )
    if family_result.get("status") != "VALID":
        raise RuntimeError("registered primary family did not complete")

    root = Path(__file__).resolve().parents[1]
    config = __import__("yaml").safe_load(
        _FROZEN_HARDENING_CONFIG.read_text(encoding="utf-8")
    )
    preregistration = config["preregistration"]
    result_schema_sha256 = _sha256_file(_RESULT_CELL_SCHEMA_PATH)
    input_hashes = {
        **inputs["input_hashes"],
        "scripts/thesis_core_hardening.py": _sha256_file(
            Path(__file__).resolve()
        ),
        str(_FROZEN_HARDENING_CONFIG.resolve().relative_to(root)): (
            _sha256_file(_FROZEN_HARDENING_CONFIG)
        ),
        str(_RESULT_CELL_SCHEMA_PATH.resolve().relative_to(root)): (
            result_schema_sha256
        ),
    }
    ended = time.perf_counter()
    ended_utc = _utc_now()
    provenance = {
        "preregistration_path": preregistration["path"],
        "preregistration_sha256": preregistration["sha256"],
        "code_commit": None,
        "dirty": True,
        "dirty_paths": [
            "configs/analysis/thesis_core_hardening_2026-07-26.yaml",
            "results/prereg/THESIS_CORE_HARDENING_PREREG_2026-07-26.md",
            "scripts/thesis_core_hardening.py",
            "tests/test_thesis_core_hardening.py",
        ],
        "python_version": platform.python_version(),
        "package_versions": {"numpy": np.__version__},
        "argv": list(argv or []),
        "utc_start": started_utc,
        "utc_end": ended_utc,
        "host": {
            "hostname": socket.gethostname(),
            "system": platform.system(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "process": {"pid": os.getpid(), "workers": 1},
        "wall_seconds": max(0.0, ended - started),
        "peak_rss_bytes": _default_peak_rss_bytes(),
    }
    document = build_primary_family_document(
        family_result,
        input_hashes=input_hashes,
        preregistration_path=preregistration["path"],
        preregistration_sha256=preregistration["sha256"],
        result_schema_sha256=result_schema_sha256,
        provenance=provenance,
        amendment_ids=_PRIMARY_AMENDMENT_IDS,
    )
    write_primary_family_result(destination_path, document)
    try:
        os.unlink(progress_path)
    except FileNotFoundError:
        pass
    return document


def execute_registered_secondary_from_disk(
    family: str,
    destination: str | os.PathLike[str],
    *,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Execute and atomically seal one complete registered CPU secondary."""
    _frozen_cpu_secondary_authorized(family)
    if family not in _PLANNED_SECONDARY_DESTINATIONS:
        raise ValueError("unknown registered CPU secondary family")
    destination_path = Path(destination).resolve()
    if (
        destination_path
        != _PLANNED_SECONDARY_DESTINATIONS[family].resolve()
    ):
        raise ValueError("secondary family requires its planned destination")
    if destination_path.exists():
        raise FileExistsError("refusing to rerun a sealed secondary family")
    progress_path = destination_path.with_name(
        f".{destination_path.name}.progress.json"
    )
    if progress_path.exists():
        raise FileExistsError(
            "an unresolved secondary-family progress artifact already exists"
        )

    started_utc = _utc_now()
    started = time.perf_counter()
    inputs = load_registered_secondary_family_inputs(family)

    def record_progress(completed: Sequence[Mapping[str, Any]]) -> None:
        _atomic_write_json_file(
            progress_path,
            {
                "schema_version": (
                    "thesis-core-hardening-secondary-progress-v1"
                ),
                "status": "IN_PROGRESS",
                "family": family,
                "completed_cells": len(completed),
                "registered_seed": 20260726,
                "cells": list(completed),
            },
            replace=True,
        )

    family_result = _run_registered_cdim_family(
        inputs["cells"],
        family=family,
        B=2500,
        attempt_cap=2750,
        cross_label_policy=_frozen_registered_cpu_cross_label_policy(),
        progress_callback=record_progress,
    )
    if family_result.get("status") != "VALID":
        raise RuntimeError(f"registered {family} family did not complete")

    root = Path(__file__).resolve().parents[1]
    config = __import__("yaml").safe_load(
        _FROZEN_HARDENING_CONFIG.read_text(encoding="utf-8")
    )
    preregistration = config["preregistration"]
    result_schema_sha256 = _sha256_file(_RESULT_CELL_SCHEMA_PATH)
    input_hashes = {
        **inputs["input_hashes"],
        "scripts/thesis_core_hardening.py": _sha256_file(
            Path(__file__).resolve()
        ),
        str(_FROZEN_HARDENING_CONFIG.resolve().relative_to(root)): (
            _sha256_file(_FROZEN_HARDENING_CONFIG)
        ),
        str(_RESULT_CELL_SCHEMA_PATH.resolve().relative_to(root)): (
            result_schema_sha256
        ),
    }
    ended = time.perf_counter()
    ended_utc = _utc_now()
    provenance = {
        "preregistration_path": preregistration["path"],
        "preregistration_sha256": preregistration["sha256"],
        "code_commit": None,
        "dirty": True,
        "dirty_paths": [
            "configs/analysis/thesis_core_hardening_2026-07-26.yaml",
            "results/prereg/THESIS_CORE_HARDENING_PREREG_2026-07-26.md",
            "scripts/thesis_core_hardening.py",
            "tests/test_thesis_core_hardening.py",
        ],
        "python_version": platform.python_version(),
        "package_versions": {"numpy": np.__version__},
        "argv": list(argv or []),
        "utc_start": started_utc,
        "utc_end": ended_utc,
        "host": {
            "hostname": socket.gethostname(),
            "system": platform.system(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "process": {"pid": os.getpid(), "workers": 1},
        "wall_seconds": max(0.0, ended - started),
        "peak_rss_bytes": _default_peak_rss_bytes(),
    }
    document = build_secondary_family_document(
        family_result,
        family=family,
        input_hashes=input_hashes,
        preregistration_path=preregistration["path"],
        preregistration_sha256=preregistration["sha256"],
        result_schema_sha256=result_schema_sha256,
        provenance=provenance,
        amendment_ids=_SECONDARY_AMENDMENT_IDS,
    )
    write_secondary_family_result(
        destination_path, document, family=family
    )
    try:
        os.unlink(progress_path)
    except FileNotFoundError:
        pass
    return document


def _scope_result_provenance(
    *,
    preregistration: Mapping[str, Any],
    started_utc: str,
    started: float,
    elapsed_before_start: float,
    argv: Sequence[str] | None,
) -> dict[str, Any]:
    return {
        "preregistration_path": preregistration["path"],
        "preregistration_sha256": preregistration["sha256"],
        "code_commit": None,
        "dirty": True,
        "dirty_paths": [
            "configs/analysis/thesis_core_hardening_2026-07-26.yaml",
            "results/prereg/THESIS_CORE_HARDENING_PREREG_2026-07-26.md",
            "scripts/thesis_core_hardening.py",
            "tests/test_thesis_core_hardening.py",
        ],
        "python_version": platform.python_version(),
        "package_versions": {"numpy": np.__version__},
        "argv": list(argv or []),
        "utc_start": started_utc,
        "utc_end": _utc_now(),
        "host": {
            "hostname": socket.gethostname(),
            "system": platform.system(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "process": {"pid": os.getpid(), "workers": 1},
        "wall_seconds": max(
            0.0, elapsed_before_start + time.perf_counter() - started
        ),
        "peak_rss_bytes": _default_peak_rss_bytes(),
    }


def _scope_output_input_hashes(
    empirical_hashes: Mapping[str, str],
) -> tuple[dict[str, str], str, Mapping[str, Any]]:
    import yaml

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        _FROZEN_HARDENING_CONFIG.read_text(encoding="utf-8")
    )
    result_schema_sha256 = _sha256_file(_RESULT_CELL_SCHEMA_PATH)
    hashes = {
        **empirical_hashes,
        "scripts/thesis_core_hardening.py": _sha256_file(
            Path(__file__).resolve()
        ),
        str(_FROZEN_HARDENING_CONFIG.resolve().relative_to(root)): (
            _sha256_file(_FROZEN_HARDENING_CONFIG)
        ),
        str(_RESULT_CELL_SCHEMA_PATH.resolve().relative_to(root)): (
            result_schema_sha256
        ),
    }
    return hashes, result_schema_sha256, config["preregistration"]


def execute_registered_h3_from_disk(
    *,
    argv: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute and seal the exhaustive L27 chain/truncation H3 diagnostic."""
    _frozen_cpu_scope_authorized("h3")
    chain_destination = _PLANNED_SCOPE_DESTINATIONS["chain_stability"].resolve()
    truncation_destination = _PLANNED_SCOPE_DESTINATIONS["truncation"].resolve()
    if chain_destination.exists() or truncation_destination.exists():
        raise FileExistsError("refusing to rerun a sealed H3 diagnostic")
    progress_path = truncation_destination.with_name(
        ".h3_scope.progress.json"
    )
    started = time.perf_counter()
    started_utc = _utc_now()
    inputs = load_empirical_backtracking_l27_inputs()
    input_hashes, result_schema_sha256, preregistration = (
        _scope_output_input_hashes(inputs["input_hashes"])
    )
    runner_sha256 = input_hashes["scripts/thesis_core_hardening.py"]
    elapsed_before_start = 0.0
    behaviour_results: dict[int, Mapping[str, Any]] = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("schema_version")
            != "thesis-core-hardening-h3-progress-v1"
            or progress.get("status") != "IN_PROGRESS"
            or progress.get("runner_sha256") != runner_sha256
            or progress.get("preregistration_sha256")
            != preregistration["sha256"]
            or progress.get("empirical_input_hashes")
            != inputs["input_hashes"]
        ):
            raise ValueError("H3 progress artifact does not match frozen inputs")
        behaviour_results = {
            _json_integer(int(code)): result
            for code, result in progress["behaviour_results"].items()
        }
        if not set(behaviour_results).issubset({1, 2, 3, 4}):
            raise ValueError("H3 progress artifact has an invalid behaviour set")
        elapsed_before_start = float(progress["elapsed_wall_seconds"])
        started_utc = str(progress["utc_start"])
    chain_records = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "data"
            / "chains_R1-1.5B.json"
        ).read_text(encoding="utf-8")
    )
    chain_metadata = {
        _nfc(str(record["task_id"])): record for record in chain_records
    }
    for behaviour_code in range(1, 5):
        if behaviour_code in behaviour_results:
            continue
        result = run_truncation_sensitivity(
            inputs["activations"],
            inputs["occurrences"],
            chain_metadata,
            target_label=_PRIMARY_LABELS[behaviour_code - 1],
            behaviour_code=behaviour_code,
            n_replicates=500,
            valid_threshold=475,
            cross_label_policy=_frozen_registered_cpu_cross_label_policy(),
        )
        if result.get("status") != "VALID":
            raise RuntimeError(
                f"H3 behaviour {behaviour_code} did not satisfy validity gates"
            )
        behaviour_results[behaviour_code] = result
        elapsed = (
            elapsed_before_start + time.perf_counter() - started
        )
        _atomic_write_json_file(
            progress_path,
            {
                "schema_version": "thesis-core-hardening-h3-progress-v1",
                "status": "IN_PROGRESS",
                "runner_sha256": runner_sha256,
                "preregistration_sha256": preregistration["sha256"],
                "empirical_input_hashes": inputs["input_hashes"],
                "utc_start": started_utc,
                "elapsed_wall_seconds": elapsed,
                "completed_behaviours": len(behaviour_results),
                "behaviour_results": behaviour_results,
            },
            replace=True,
        )
    provenance = _scope_result_provenance(
        preregistration=preregistration,
        started_utc=started_utc,
        started=started,
        elapsed_before_start=elapsed_before_start,
        argv=argv,
    )
    chain_document, truncation_document = build_h3_family_documents(
        behaviour_results,
        input_hashes=input_hashes,
        preregistration_path=preregistration["path"],
        preregistration_sha256=preregistration["sha256"],
        result_schema_sha256=result_schema_sha256,
        provenance=provenance,
        amendment_ids=_SCOPE_AMENDMENT_IDS,
    )
    write_scope_family_result(
        chain_destination, chain_document, kind="chain_stability"
    )
    write_scope_family_result(
        truncation_destination, truncation_document, kind="truncation"
    )
    try:
        os.unlink(progress_path)
    except FileNotFoundError:
        pass
    return chain_document, truncation_document


def execute_registered_curvature_from_disk(
    *,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Execute and seal the complete four-behaviour L16 curvature diagnostic."""
    _frozen_cpu_scope_authorized("curvature")
    destination = _PLANNED_SCOPE_DESTINATIONS["curvature"].resolve()
    if destination.exists():
        raise FileExistsError("refusing to rerun a sealed curvature diagnostic")
    progress_path = destination.with_name(".curvature_scope.progress.json")
    started = time.perf_counter()
    started_utc = _utc_now()
    inputs = load_registered_curvature_inputs()
    input_hashes, result_schema_sha256, preregistration = (
        _scope_output_input_hashes(inputs["input_hashes"])
    )
    runner_sha256 = input_hashes["scripts/thesis_core_hardening.py"]
    elapsed_before_start = 0.0
    behaviour_results: dict[int, Mapping[str, Any]] = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("schema_version")
            != "thesis-core-hardening-curvature-progress-v1"
            or progress.get("status") != "IN_PROGRESS"
            or progress.get("runner_sha256") != runner_sha256
            or progress.get("preregistration_sha256")
            != preregistration["sha256"]
            or progress.get("empirical_input_hashes")
            != inputs["input_hashes"]
        ):
            raise ValueError(
                "curvature progress artifact does not match frozen inputs"
            )
        behaviour_results = {
            _json_integer(int(code)): result
            for code, result in progress["behaviour_results"].items()
        }
        if not set(behaviour_results).issubset({1, 2, 3, 4}):
            raise ValueError(
                "curvature progress artifact has an invalid behaviour set"
            )
        elapsed_before_start = float(progress["elapsed_wall_seconds"])
        started_utc = str(progress["utc_start"])
    for behaviour_code in range(1, 5):
        if behaviour_code in behaviour_results:
            continue
        result = run_curvature_diagnostic(
            inputs["activations"],
            inputs["occurrences"],
            target_label=_PRIMARY_LABELS[behaviour_code - 1],
            behaviour_code=behaviour_code,
            cell_index=88 + behaviour_code - 1,
            n_replicates=500,
            valid_threshold=475,
            cross_label_policy=_frozen_registered_cpu_cross_label_policy(),
        )
        if result.get("status") != "VALID":
            raise RuntimeError(
                f"curvature behaviour {behaviour_code} did not satisfy validity gates"
            )
        behaviour_results[behaviour_code] = result
        elapsed = elapsed_before_start + time.perf_counter() - started
        _atomic_write_json_file(
            progress_path,
            {
                "schema_version": (
                    "thesis-core-hardening-curvature-progress-v1"
                ),
                "status": "IN_PROGRESS",
                "runner_sha256": runner_sha256,
                "preregistration_sha256": preregistration["sha256"],
                "empirical_input_hashes": inputs["input_hashes"],
                "utc_start": started_utc,
                "elapsed_wall_seconds": elapsed,
                "completed_behaviours": len(behaviour_results),
                "behaviour_results": behaviour_results,
            },
            replace=True,
        )
    provenance = _scope_result_provenance(
        preregistration=preregistration,
        started_utc=started_utc,
        started=started,
        elapsed_before_start=elapsed_before_start,
        argv=argv,
    )
    document = build_curvature_family_document(
        behaviour_results,
        input_hashes=input_hashes,
        preregistration_path=preregistration["path"],
        preregistration_sha256=preregistration["sha256"],
        result_schema_sha256=result_schema_sha256,
        provenance=provenance,
        amendment_ids=_SCOPE_AMENDMENT_IDS,
    )
    write_scope_family_result(destination, document, kind="curvature")
    try:
        os.unlink(progress_path)
    except FileNotFoundError:
        pass
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-synthetic",
        action="store_true",
        help="run the non-empirical resource/firewall validation path",
    )
    mode.add_argument(
        "--resource-pilot",
        action="store_true",
        help="select the empirical B=25 pilot path",
    )
    mode.add_argument(
        "--run-primary",
        action="store_true",
        help="execute the complete registered CPU primary family",
    )
    mode.add_argument(
        "--run-secondary",
        choices=("five_depth", "three_annotator"),
        help="execute one complete registered CPU secondary family",
    )
    mode.add_argument(
        "--run-scope",
        choices=("h3", "curvature"),
        help="execute one complete registered CPU scope diagnostic",
    )
    parser.add_argument("--resource-output", type=Path)
    parser.add_argument(
        "--acknowledge-empirical-pilot",
        action="store_true",
        help="explicitly acknowledge that the empirical pilot may run estimators",
    )
    parser.add_argument(
        "--acknowledge-full-analysis",
        action="store_true",
        help="explicitly acknowledge the full statistical primary run",
    )
    args = parser.parse_args(argv)
    if args.resource_pilot and not args.acknowledge_empirical_pilot:
        parser.error(
            "--resource-pilot requires --acknowledge-empirical-pilot"
        )
    if args.run_primary and not args.acknowledge_full_analysis:
        parser.error(
            "--run-primary requires --acknowledge-full-analysis"
        )
    if args.run_secondary and not args.acknowledge_full_analysis:
        parser.error(
            "--run-secondary requires --acknowledge-full-analysis"
        )
    if args.run_scope and not args.acknowledge_full_analysis:
        parser.error(
            "--run-scope requires --acknowledge-full-analysis"
        )
    if args.run_primary:
        execute_registered_primary_from_disk(
            _PLANNED_PRIMARY_DESTINATION.resolve(),
            argv=list(argv or sys.argv[1:]),
        )
        return 0
    if args.run_secondary:
        execute_registered_secondary_from_disk(
            args.run_secondary,
            _PLANNED_SECONDARY_DESTINATIONS[
                args.run_secondary
            ].resolve(),
            argv=list(argv or sys.argv[1:]),
        )
        return 0
    if args.run_scope == "h3":
        execute_registered_h3_from_disk(
            argv=list(argv or sys.argv[1:]),
        )
        return 0
    if args.run_scope == "curvature":
        execute_registered_curvature_from_disk(
            argv=list(argv or sys.argv[1:]),
        )
        return 0
    if args.resource_pilot:
        destination = (
            args.resource_output.resolve()
            if args.resource_output is not None
            else _PLANNED_EMPIRICAL_RESOURCE_DESTINATION.resolve()
        )
        execute_empirical_resource_pilot_from_disk(destination)
        return 0
    document = run_resource_benchmark(
        lambda _attempt_index: "VALID",
        b_target=1,
        synthetic=True,
        input_hashes={},
        workers=1,
        scratch_bytes=0,
        output_bytes_per_attempt=0,
    )
    if args.resource_output is not None:
        write_resource_only_benchmark(args.resource_output, document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
