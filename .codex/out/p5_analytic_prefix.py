#!/usr/bin/env python3
"""Fail-closed construction of the approved P5 4,096-token prefix.

The module is pure and offline.  It never loads a model or tokenizer itself;
the caller supplies the already hash-bound base-tokenizer decoder.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable


PREFIX_TOKENS = 4096
EOS_TOKEN_ID = 151643
SCHEMA_VERSION = "p5-analytic-prefix-4096-1"


class PrefixError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_generated_ids(row: dict[str, Any]) -> list[int]:
    if "generated_token_ids" not in row:
        raise PrefixError("lossless generated_token_ids are required")
    ids = row["generated_token_ids"]
    if not isinstance(ids, list) or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in ids
    ):
        raise PrefixError("generated_token_ids must be a list of non-negative integers")
    if row.get("n_tokens") != len(ids):
        raise PrefixError("n_tokens does not equal generated_token_ids length")
    if EOS_TOKEN_ID in ids:
        raise PrefixError("stored generated IDs must exclude terminating EOS")
    if row.get("stop_reason") not in {"eos", "length", "other"}:
        raise PrefixError("invalid stop_reason")
    raw_cap = row.get("raw_max_new_tokens")
    if not isinstance(raw_cap, int) or raw_cap <= 0:
        raise PrefixError("raw_max_new_tokens must be a positive integer")
    if row["stop_reason"] == "length" and len(ids) != raw_cap:
        raise PrefixError("length stop does not equal the recorded raw generation cap")
    if len(ids) > raw_cap:
        raise PrefixError("generated token count exceeds the recorded raw cap")
    return ids


def make_prefix_record(
    row: dict[str, Any],
    decode: Callable[[list[int]], str],
    *,
    tokenizer_sha256: str,
    tokenizer_config_sha256: str,
    source_row_sha256: str | None = None,
) -> dict[str, Any]:
    ids = validate_generated_ids(row)
    computed_source_hash = sha256_json(row)
    if source_row_sha256 is not None and source_row_sha256 != computed_source_hash:
        raise PrefixError("source row hash mismatch")
    prefix_ids = ids[:PREFIX_TOKENS]
    prefix_text = decode(prefix_ids)
    if not isinstance(prefix_text, str):
        raise PrefixError("decoder must return text")
    if len(ids) <= PREFIX_TOKENS and row.get("text") != prefix_text:
        raise PrefixError("complete source text does not match exact generated-ID decode")

    if len(ids) > PREFIX_TOKENS:
        prefix_stop = "analytic_prefix_cap"
    elif row["stop_reason"] == "eos":
        prefix_stop = "eos"
    else:
        prefix_stop = row["stop_reason"]

    required_identity = ("checkpoint_role", "task_id", "generation_config_sha256")
    missing = [field for field in required_identity if not row.get(field)]
    if missing:
        raise PrefixError(f"missing source identity fields: {','.join(missing)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_role": row["checkpoint_role"],
        "task_id": row["task_id"],
        "source_row_sha256": computed_source_hash,
        "source_generation_config_sha256": row["generation_config_sha256"],
        "tokenizer_sha256": tokenizer_sha256,
        "tokenizer_config_sha256": tokenizer_config_sha256,
        "prefix_max_tokens": PREFIX_TOKENS,
        "prefix_n_tokens": len(prefix_ids),
        "prefix_stop_reason": prefix_stop,
        "prefix_token_ids": prefix_ids,
        "prefix_token_ids_sha256": sha256_json(prefix_ids),
        "prefix_text": prefix_text,
        "prefix_text_sha256": sha256_text(prefix_text),
        "source_n_tokens": len(ids),
        "source_stop_reason": row["stop_reason"],
        "source_raw_max_new_tokens": row["raw_max_new_tokens"],
    }


def validate_unique_prefix_rows(rows: list[dict[str, Any]]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row["checkpoint_role"], row["task_id"])
        digest = sha256_json(row)
        if key in seen and seen[key] != digest:
            raise PrefixError("conflicting duplicate prefix row")
        seen[key] = digest
