#!/usr/bin/env python3
"""Resumable P5 pilot executor confined to .codex/out.

The executor freezes one run manifest before output, generates the 44 x 4
pilot grid with a common base-tokenizer alias, and refuses model-touching work
without Tony's authorisation plus the frozen content hash.  It never writes to
results/, Phase-2 files, or the thesis evidence snapshot.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p5_preflight as p5  # noqa: E402


RUN_ID = "p5-pilot-20260808"
RUN_ROOT_NAME = "p5_runs"
GENERATION_LIMIT = 176
SCORING_LIMIT = 212
SEED = 20260808


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def assert_out_path(path: Path) -> Path:
    resolved = path.resolve()
    if ".codex/out" not in resolved.as_posix():
        raise SystemExit(f"refusing path outside .codex/out: {resolved}")
    return resolved


def atomic_write(path: Path, text: str) -> None:
    path = assert_out_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def file_sha(path: Path) -> str:
    return p5.sha256_file(path)


def prompt_bundle(repo_root: Path) -> list[dict[str, Any]]:
    generic_path = repo_root / "data" / "tasks_pilot.json"
    safety_path = HERE / "P5_SAFETY_PILOT_MANIFEST_2026-08-08.json"
    generic = p5.load_prompt_manifest_rows(generic_path)
    safety = p5.load_prompt_manifest_rows(safety_path)
    generic_validation = p5.validate_prompt_rows(generic, kind="generic")
    safety_validation = p5.validate_prompt_rows(safety, kind="safety", require_pair_id=True)
    if generic_validation["status"] != "pass" or safety_validation["status"] != "pass":
        raise SystemExit("pilot prompt validation failed")
    rows = []
    for row in generic:
        rows.append(
            {
                "prompt_id": row["id"],
                "prompt": row["prompt"],
                "category": row["category"],
                "stratum": "generic",
                "pair_id": None,
            }
        )
    for row in safety:
        rows.append(
            {
                "prompt_id": row["id"],
                "prompt": row["prompt"],
                "category": row["category"],
                "stratum": "harmful" if row["harmful"] else "benign",
                "pair_id": row["pair_id"],
            }
        )
    if len(rows) != 44 or len({row["prompt_id"] for row in rows}) != 44:
        raise SystemExit("pilot prompt bundle must contain 44 unique prompts")
    return rows


def git_state(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"], text=True
        ).strip()
    )
    return {"commit": commit, "dirty": dirty}


def frozen_double_score_keys(prompts: list[dict[str, Any]], roles: list[str]) -> list[dict[str, str]]:
    """Select 36 output keys before generation, balanced by role and stratum."""
    by_stratum: dict[str, list[str]] = {}
    for prompt in prompts:
        by_stratum.setdefault(prompt["stratum"], []).append(prompt["prompt_id"])
    selected = []
    # Aggregate allocation: generic=16, harmful=10, benign=10; every role=9.
    allocations = {
        roles[0]: {"generic": 4, "harmful": 3, "benign": 2},
        roles[1]: {"generic": 4, "harmful": 3, "benign": 2},
        roles[2]: {"generic": 4, "harmful": 2, "benign": 3},
        roles[3]: {"generic": 4, "harmful": 2, "benign": 3},
    }
    for role in roles:
        for stratum in ("generic", "harmful", "benign"):
            ranked = sorted(
                by_stratum[stratum],
                key=lambda prompt_id: hashlib.sha256(
                    f"p5-double|{SEED}|{role}|{prompt_id}".encode()
                ).hexdigest(),
            )
            for prompt_id in ranked[: allocations[role][stratum]]:
                selected.append(
                    {"checkpoint_role": role, "prompt_id": prompt_id, "stratum": stratum}
                )
    if len(selected) != 36 or len({(row["checkpoint_role"], row["prompt_id"]) for row in selected}) != 36:
        raise SystemExit("double-score subset must contain 36 unique output keys")
    return selected


def freeze_manifest(repo_root: Path, run_root: Path) -> dict[str, Any]:
    run_root = assert_out_path(run_root)
    manifest_path = run_root / "run_manifest.json"
    readiness = p5.build_readiness(repo_root, deep_hash=True)
    if readiness["status"] != "ready_for_spend_gated_pilot":
        raise SystemExit(f"preflight is not ready: {readiness['blocking_gate_ids']}")
    inventory_rows = readiness["checkpoint_inventory"]["checkpoints"]
    roles = [row["role"] for row in inventory_rows]
    if (
        len(roles) != 4
        or any(row["missing_files"] for row in inventory_rows)
        or any(row["weight_hash_matches"] is not True for row in inventory_rows)
    ):
        raise SystemExit("four fully verified checkpoint roles are required")
    input_gate_path = HERE / "P5_INPUT_ID_GATE_2026-08-08.json"
    input_gate = json.loads(input_gate_path.read_text())
    if input_gate.get("status") != "pass" or input_gate.get("validation_errors") != []:
        raise SystemExit("input-ID gate is not a complete pass")
    prompts = prompt_bundle(repo_root)
    expected_ids = {row["prompt_id"] for row in input_gate["records"]}
    if {row["prompt_id"] for row in prompts} != expected_ids:
        raise SystemExit("prompt bundle does not match the frozen input-ID gate")

    generation_config = {
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": 4096,
        "num_return_sequences": 1,
        "pad_token_policy": "base tokenizer eos token",
        "eos_policy": "base tokenizer/model EOS; common across arms",
        "seed": SEED,
        "tokenizer_alias": "base_r1 for every checkpoint role",
        "dtype": "bfloat16 on Apple MPS; bfloat16 on CUDA",
    }
    payload = {
        "schema_version": "p5-pilot-run-manifest-1",
        "run_id": RUN_ID,
        "frozen_before_output": True,
        "created_at_utc": utc_now(),
        "repository": {"path": str(repo_root), **git_state(repo_root)},
        "software_sources": {
            "executor_path": str(Path(__file__).resolve()),
            "executor_sha256": file_sha(Path(__file__).resolve()),
            "preflight_path": str(HERE / "p5_preflight.py"),
            "preflight_sha256": file_sha(HERE / "p5_preflight.py"),
            "test_path": str(HERE / "test_p5_preflight.py"),
            "test_sha256": file_sha(HERE / "test_p5_preflight.py"),
        },
        "prompt_sources": {
            "generic": {
                "path": str(repo_root / "data" / "tasks_pilot.json"),
                "sha256": file_sha(repo_root / "data" / "tasks_pilot.json"),
                "n": 20,
            },
            "safety": {
                "path": str(HERE / "P5_SAFETY_PILOT_MANIFEST_2026-08-08.json"),
                "sha256": file_sha(HERE / "P5_SAFETY_PILOT_MANIFEST_2026-08-08.json"),
                "n": 24,
            },
            "bundle_sha256": p5.sha256_json(prompts),
        },
        "input_id_gate": {
            "path": str(input_gate_path),
            "file_sha256": file_sha(input_gate_path),
            "record_manifest_sha256": input_gate["record_manifest_sha256"],
            "tokenizer_sha256": input_gate["tokenizer_sha256"],
            "tokenizer_config_sha256": input_gate["tokenizer_config_sha256"],
            "chat_template_sha256": input_gate["chat_template_sha256"],
        },
        "checkpoint_roles": inventory_rows,
        "generation_config": generation_config,
        "generation_config_sha256": p5.sha256_json(generation_config),
        "scoring": {
            "primary_calls": 176,
            "double_score_calls": 36,
            "call_limit": SCORING_LIMIT,
            "primary_model": "amazon.nova-pro-v1:0",
            "double_score_model": "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "safety_rubric_version": "p5-safety-rubric-1",
            "safety_rubric_sha256": file_sha(HERE / "P5_SAFETY_SCORER_RUBRIC_2026-08-08.md"),
            "double_score_keys": frozen_double_score_keys(prompts, roles),
        },
        "budget": {
            "generation_limit": GENERATION_LIMIT,
            "scoring_limit": SCORING_LIMIT,
            "generation_ceiling_usd": 3,
            "scoring_ceiling_usd": 15,
            "total_ceiling_usd": 18,
        },
        "prompts": prompts,
    }
    payload["authorisation_sha256"] = p5.sha256_json(payload)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        # Creation time may differ only if someone tries to recreate from scratch;
        # once frozen, never overwrite it.
        if existing != payload:
            raise SystemExit(f"refusing to overwrite frozen manifest: {manifest_path}")
        return existing
    atomic_write(manifest_path, rendered)
    return payload


def load_manifest(run_root: Path) -> dict[str, Any]:
    path = assert_out_path(run_root) / "run_manifest.json"
    if not path.exists():
        raise SystemExit("run manifest is absent; run freeze first")
    manifest = json.loads(path.read_text())
    content = dict(manifest)
    recorded = content.pop("authorisation_sha256", None)
    if recorded != p5.sha256_json(content):
        raise SystemExit("run manifest authorisation hash is invalid")
    return manifest


def require_authorisation(manifest: dict[str, Any], supplied_hash: str | None, authorised: bool) -> None:
    p5.require_spend_authorisation(
        "pilot-generate",
        authorised=authorised,
        manifest_sha256=supplied_hash,
        expected_manifest_sha256=manifest["authorisation_sha256"],
    )


def environment_record() -> dict[str, Any]:
    import torch
    import transformers

    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        ),
    }
    record["sha256"] = p5.sha256_json(record)
    return record


def generation_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        errors = p5.validate_generation_row(row)
        if errors:
            raise SystemExit(f"invalid existing generation row {line_number}: {'; '.join(errors)}")
        rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path = assert_out_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def generate(repo_root: Path, run_root: Path, manifest_hash: str | None, authorised: bool, roles: list[str]) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_root = assert_out_path(run_root)
    manifest = load_manifest(run_root)
    require_authorisation(manifest, manifest_hash, authorised)
    if file_sha(Path(__file__).resolve()) != manifest["software_sources"]["executor_sha256"]:
        raise SystemExit("executor source changed after the run manifest was frozen")
    if file_sha(HERE / "p5_preflight.py") != manifest["software_sources"]["preflight_sha256"]:
        raise SystemExit("preflight source changed after the run manifest was frozen")
    records_path = run_root / "generations.jsonl"
    existing = generation_rows(records_path)
    if len(existing) > GENERATION_LIMIT:
        raise SystemExit("generation call limit already exceeded")
    done = {p5.generation_key(row) for row in existing}
    role_inventory = {row["role"]: row for row in manifest["checkpoint_roles"]}
    unknown = sorted(set(roles) - set(role_inventory))
    if unknown:
        raise SystemExit(f"unknown roles: {unknown}")
    input_gate = json.loads((HERE / "P5_INPUT_ID_GATE_2026-08-08.json").read_text())
    expected_inputs = {row["prompt_id"]: row for row in input_gate["records"]}
    specs = {row["role"]: row for row in p5.checkpoint_specs(repo_root)}
    base_root = Path(specs["base_r1"]["root"])
    tokenizer = AutoTokenizer.from_pretrained(base_root, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    env = environment_record()
    git = git_state(repo_root)
    hardware = f"{platform.machine()} {env['device']}"
    prompts = manifest["prompts"]

    for role in roles:
        spec = specs[role]
        checkpoint = role_inventory[role]
        root = Path(spec["root"])
        if not root.exists():
            raise SystemExit(f"checkpoint unavailable locally for {role}: {root}")
        dtype = torch.bfloat16 if env["device"] in {"mps", "cuda"} else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        device = torch.device(env["device"])
        model.to(device).eval()
        for prompt in prompts:
            key = (RUN_ID, role, prompt["prompt_id"])
            if key in done:
                continue
            if len(existing) >= GENERATION_LIMIT:
                raise SystemExit("refusing to exceed 176 generation records")
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_hash = p5.input_ids_sha256(input_ids[0].tolist())
            expected = expected_inputs[prompt["prompt_id"]]
            if input_hash != expected["input_ids_sha256"]:
                raise SystemExit(f"input-ID mismatch for {role}/{prompt['prompt_id']}")
            input_ids = input_ids.to(device)
            attention_mask = torch.ones_like(input_ids)
            torch.manual_seed(SEED)
            attempt_id = str(uuid.uuid4())
            started = utc_now()
            started_clock = time.monotonic()
            row: dict[str, Any]
            try:
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=False,
                        max_new_tokens=manifest["generation_config"]["max_new_tokens"],
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                new_ids = output[0, input_ids.shape[1] :]
                token_ids = new_ids.detach().cpu().tolist()
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
                eos_ids = tokenizer.eos_token_id
                stopped_eos = bool(token_ids and eos_ids is not None and token_ids[-1] == eos_ids)
                stop_reason = (
                    "eos"
                    if stopped_eos
                    else "length"
                    if len(token_ids) >= manifest["generation_config"]["max_new_tokens"]
                    else "other"
                )
                row = {
                    "schema_version": p5.SCHEMA_VERSION,
                    "run_id": RUN_ID,
                    "code_commit": git["commit"],
                    "git_dirty": git["dirty"],
                    "environment_sha256": env["sha256"],
                    "hardware": hardware,
                    "checkpoint_role": role,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "checkpoint_revision": checkpoint.get("revision"),
                    "checkpoint_weight_sha256": checkpoint["observed_weight_sha256"],
                    "checkpoint_config_sha256": checkpoint["config_sha256"],
                    "tokenizer_sha256": manifest["input_id_gate"]["tokenizer_sha256"],
                    "tokenizer_config_sha256": manifest["input_id_gate"]["tokenizer_config_sha256"],
                    "chat_template_sha256": manifest["input_id_gate"]["chat_template_sha256"],
                    "prompt_id": prompt["prompt_id"],
                    "prompt_text_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
                    "prompt_manifest_sha256": manifest["prompt_sources"]["bundle_sha256"],
                    "input_ids_sha256": input_hash,
                    "generation_config_sha256": manifest["generation_config_sha256"],
                    "generation_seed": SEED,
                    "attempt_id": attempt_id,
                    "started_at_utc": started,
                    "finished_at_utc": utc_now(),
                    "attempt": 1,
                    "status": "success",
                    "text": text,
                    "n_tokens": len(token_ids),
                    "stop_reason": stop_reason,
                    "wall_seconds": time.monotonic() - started_clock,
                    "stratum": prompt["stratum"],
                    "category": prompt["category"],
                    "pair_id": prompt["pair_id"],
                }
            except Exception as error:
                row = {
                    "schema_version": p5.SCHEMA_VERSION,
                    "run_id": RUN_ID,
                    "code_commit": git["commit"],
                    "git_dirty": git["dirty"],
                    "environment_sha256": env["sha256"],
                    "hardware": hardware,
                    "checkpoint_role": role,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "checkpoint_revision": checkpoint.get("revision"),
                    "checkpoint_weight_sha256": checkpoint["observed_weight_sha256"],
                    "checkpoint_config_sha256": checkpoint["config_sha256"],
                    "tokenizer_sha256": manifest["input_id_gate"]["tokenizer_sha256"],
                    "tokenizer_config_sha256": manifest["input_id_gate"]["tokenizer_config_sha256"],
                    "chat_template_sha256": manifest["input_id_gate"]["chat_template_sha256"],
                    "prompt_id": prompt["prompt_id"],
                    "prompt_text_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
                    "prompt_manifest_sha256": manifest["prompt_sources"]["bundle_sha256"],
                    "input_ids_sha256": input_hash,
                    "generation_config_sha256": manifest["generation_config_sha256"],
                    "generation_seed": SEED,
                    "attempt_id": attempt_id,
                    "started_at_utc": started,
                    "finished_at_utc": utc_now(),
                    "attempt": 1,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "wall_seconds": time.monotonic() - started_clock,
                    "stratum": prompt["stratum"],
                    "category": prompt["category"],
                    "pair_id": prompt["pair_id"],
                }
            errors = p5.validate_generation_row(row)
            if errors:
                raise SystemExit("refusing invalid generation row: " + "; ".join(errors))
            append_jsonl(records_path, row)
            existing.append(row)
            done.add(key)
            print(
                f"{len(existing):03d}/{GENERATION_LIMIT} {role} {prompt['prompt_id']} "
                f"{row['status']} tokens={row.get('n_tokens')} seconds={row['wall_seconds']:.1f}",
                flush=True,
            )
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def status(run_root: Path) -> dict[str, Any]:
    run_root = assert_out_path(run_root)
    manifest = load_manifest(run_root)
    rows = generation_rows(run_root / "generations.jsonl")
    counts: dict[str, int] = {}
    errors = 0
    for row in rows:
        counts[row["checkpoint_role"]] = counts.get(row["checkpoint_role"], 0) + 1
        errors += row["status"] == "error"
    return {
        "run_id": manifest["run_id"],
        "authorisation_sha256": manifest["authorisation_sha256"],
        "n_generation_rows": len(rows),
        "n_generation_errors": errors,
        "generation_by_role": counts,
        "generation_complete": len(rows) == GENERATION_LIMIT,
        "generation_limit": GENERATION_LIMIT,
        "scoring_limit": SCORING_LIMIT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=HERE / RUN_ROOT_NAME / RUN_ID,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("status")
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--authorised", action="store_true")
    generate_parser.add_argument("--manifest-sha256")
    generate_parser.add_argument("--role", action="append")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    run_root = args.run_root.resolve()
    if args.command == "freeze":
        manifest = freeze_manifest(repo_root, run_root)
        print(json.dumps({"run_root": str(run_root), "authorisation_sha256": manifest["authorisation_sha256"]}, indent=2))
    elif args.command == "status":
        print(json.dumps(status(run_root), indent=2))
    elif args.command == "generate":
        manifest = load_manifest(run_root)
        roles = args.role or [row["role"] for row in manifest["checkpoint_roles"]]
        generate(repo_root, run_root, args.manifest_sha256, args.authorised, roles)


if __name__ == "__main__":
    main()
