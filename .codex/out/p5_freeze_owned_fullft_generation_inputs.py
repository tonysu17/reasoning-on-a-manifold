#!/usr/bin/env python3
"""Freeze zero-spend inputs for the owned full-FT P5 generation pair.

The output is deliberately non-executable. It binds task bytes, tokenizer input
IDs, checkpoint files, and the prospective decoding contract, but contains no
pod runner, spend ceiling, or execution authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import p5_powered_generic_preflight as preflight


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "P5_OWNED_FULLFT_GENERATION_INPUT_MANIFEST_2026-08-09.json"
PREFLIGHT_CODE_SHA256 = "2d3477ba73945867956250362b83aaee468ca62b07006bdf7d53397d00a22dc4"
PREFLIGHT_SNAPSHOT = HERE / "P5_POWERED_GENERIC_PREFLIGHT_READINESS_2026-08-09.json"
PREFLIGHT_SNAPSHOT_SHA256 = "99ba17b90e8c0ee3ce9b181133c40095eae7533072bb6bc82821fbd3a24059b3"
RUN_SEED = 20260808
MAX_NEW_TOKENS = 4096
EOS_TOKEN_ID = 151643
PAD_TOKEN_ID = 151643
ROLES = (
    "p5_owned_fullft_control_s42",
    "p5_owned_fullft_safety_s42",
)


class GenerationFreezeError(RuntimeError):
    pass


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def internal_hash(document: dict[str, Any]) -> str:
    body = dict(document)
    body.pop("internal_sha256", None)
    return sha256_json(body)


def verify_sources() -> None:
    if preflight.sha256_file(Path(preflight.__file__)) != PREFLIGHT_CODE_SHA256:
        raise GenerationFreezeError("P5 preflight code hash drift")
    if preflight.sha256_file(PREFLIGHT_SNAPSHOT) != PREFLIGHT_SNAPSHOT_SHA256:
        raise GenerationFreezeError("P5 preflight snapshot hash drift")


def prompt_input_records(tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise GenerationFreezeError("transformers is required for offline tokenization") from error

    tokenizer = AutoTokenizer.from_pretrained(
        preflight.CONTROL,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.eos_token_id != EOS_TOKEN_ID or tokenizer.pad_token_id != PAD_TOKEN_ID:
        raise GenerationFreezeError("owned tokenizer EOS/pad IDs drift from frozen contract")
    records: list[dict[str, Any]] = []
    for task in tasks:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": task["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
        )
        input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        if not isinstance(input_ids, list) or not input_ids or any(
            isinstance(token, bool) or not isinstance(token, int) for token in input_ids
        ):
            raise GenerationFreezeError("chat-template tokenization returned invalid IDs")
        records.append({
            "task_id": task["id"],
            "category": task["category"],
            "prompt_sha256": hashlib.sha256(task["prompt"].encode()).hexdigest(),
            "n_input_tokens": len(input_ids),
            "input_ids_sha256": sha256_json(input_ids),
        })
    runtime = {
        "loader_contract": "AutoTokenizer local_files_only=true trust_remote_code=false use_fast=true",
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "model_max_length": tokenizer.model_max_length,
        "chat_template_sha256": preflight.sha256_file(
            preflight.CONTROL / "chat_template.jinja"
        ),
        "tokenizer_json_sha256": preflight.sha256_file(
            preflight.CONTROL / "tokenizer.json"
        ),
        "tokenizer_config_sha256": preflight.sha256_file(
            preflight.CONTROL / "tokenizer_config.json"
        ),
        "phase2_base_alias_equivalence": (
            "pending comparison with the final Phase-2 tokenizer/template hashes; mismatch blocks generation"
        ),
    }
    return records, runtime


def build_document() -> dict[str, Any]:
    verify_sources()
    phase2 = preflight.validate_phase2_manifest()
    checkpoints = preflight.validate_checkpoint_pair()
    phase2_document = json.loads(preflight.PH2_MANIFEST.read_text())
    input_rows, tokenizer_runtime = prompt_input_records(phase2_document["tasks"])
    document: dict[str, Any] = {
        "schema_version": "p5-owned-fullft-generation-input-manifest-1",
        "status": "frozen_inputs_non_executable_runner_cost_and_authorisation_absent",
        "created_date": "2026-08-09",
        "execution_authorized": False,
        "network_model_api_pod_calls_authorized": 0,
        "source_preflight": {
            "code_path": str(Path(preflight.__file__).relative_to(ROOT)),
            "code_sha256": PREFLIGHT_CODE_SHA256,
            "snapshot_path": str(PREFLIGHT_SNAPSHOT.relative_to(ROOT)),
            "snapshot_file_sha256": PREFLIGHT_SNAPSHOT_SHA256,
        },
        "phase2_task_source": {
            key: phase2[key]
            for key in (
                "path",
                "file_sha256",
                "ids_sha256",
                "ordered_task_content_sha256",
                "n_tasks",
                "per_category",
            )
        },
        "roles": list(ROLES),
        "expected_terminal_rows": 200,
        "independent_unit": "task/prompt",
        "checkpoint_rows": "paired repeated observations nested within task",
        "checkpoint_pair": checkpoints,
        "tokenizer_preflight": tokenizer_runtime,
        "ordered_prompt_inputs": input_rows,
        "ordered_prompt_inputs_sha256": sha256_json(input_rows),
        "generation_contract": {
            "samples_per_task_role": 1,
            "max_new_tokens": MAX_NEW_TOKENS,
            "primary_analytic_prefix_tokens": 4096,
            "do_sample": False,
            "temperature_semantics": 0.0,
            "temperature_argument": "omit when do_sample=false",
            "seed": RUN_SEED,
            "seed_semantics": "recorded but inert under greedy decoding",
            "eos_token_id": EOS_TOKEN_ID,
            "pad_token_id": PAD_TOKEN_ID,
            "custom_stop_strings": [],
            "stop_at_think_close": False,
            "chat_template": "local checkpoint template, one user message, add_generation_prompt=true",
            "stored_generated_token_ids": "required; exclude a terminal EOS token",
            "stored_stop_reason": "required exact enum: eos or length; other requires prospective schema amendment",
            "stored_text": "decode the stored generated IDs once with skip_special_tokens=true",
            "default_generation_config_override": (
                "checkpoint generation_config defaults to sampling; the runner must explicitly "
                "override do_sample=false and must not inherit temperature/top_p"
            ),
            "cap_boundary": (
                "P5-owned rows stop at the approved 4096-token analytic boundary. "
                "Phase-2 shared rows retain sealed 8192-token raw generation and are reduced "
                "to the same 4096-token P5 estimand analytically."
            ),
        },
        "required_terminal_row_fields": [
            "schema_version",
            "task_id",
            "category",
            "checkpoint_role",
            "source_task_manifest_file_sha256",
            "prompt_sha256",
            "input_ids_sha256",
            "generation_config_sha256",
            "generated_token_ids",
            "n_tokens",
            "stop_reason",
            "raw_max_new_tokens",
            "text",
            "text_sha256",
            "checkpoint_directory_manifest_sha256",
            "tokenizer_config_sha256",
            "started_utc",
            "finished_utc",
            "row_sha256",
        ],
        "hard_stops": [
            "Phase-2 task-manifest file or ordered task-content hash drift",
            "control/safety task bytes or input IDs differ",
            "checkpoint, tokenizer, template, or generation-config hash drift",
            "runner inherits checkpoint sampling defaults",
            "generated token IDs or exact stop reason absent",
            "row conflict, overwrite, or missing terminal row",
            "runner/environment/cost manifest absent",
            "owner has not authorized the exact runner/manifest/budget hashes",
        ],
        "remaining_before_execution": [
            "build and test a hash-bound pod runner in Claude's pod lane",
            "pin environment/container and cumulative duration/cost guards",
            "dry-plan exact request/row count and expected pod cost",
            "obtain Tony's explicit authorization for the returned hashes and ceiling",
        ],
        "boundary": (
            "This manifest freezes inputs only. It performs no generation and grants no "
            "permission for model loading, network access, API calls, or pod spend."
        ),
    }
    document["internal_sha256"] = internal_hash(document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    document = build_document()
    if args.write:
        OUTPUT.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
        print(json.dumps({
            "path": str(OUTPUT),
            "file_sha256": preflight.sha256_file(OUTPUT),
            "internal_sha256": document["internal_sha256"],
            "expected_terminal_rows": document["expected_terminal_rows"],
            "execution_authorized": document["execution_authorized"],
        }, sort_keys=True, indent=2))
    else:
        print(json.dumps(document, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
