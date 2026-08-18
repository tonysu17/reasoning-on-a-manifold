#!/usr/bin/env python3
"""Result-blind runner for the sealed 1.5B atomic-hop/order-operations study.

This module is intentionally split into five stages:

``fetch``
    Fetch and hash-check the two upstream task files.  No model is imported.
``preflight``
    Check the sealed checkout, pinned local lenses, task schemas and input
    population.  No model is imported or loaded.
``campaign``
    Freeze one sealed, immutable pre-forward authority shared by A0 and O1.
``atomic``
    Run the two-arm behavioural atomic-hop competence audit.
``orderops``
    Extract exact J-lens and logit-lens ranks for both checkpoints.  This
    stage writes rank arrays but does not calculate between-checkpoint tests.
``analyse``
    Calculate the sealed item-paired summaries from an ``orderops`` bundle.

The model-bearing stages refuse to start unless HEAD is the supplied full
seal commit, the preregistration is tracked and unchanged at that commit, and
there are no working-tree changes outside this experiment's isolated output
root.  Importing this file never imports torch, transformers or jlens.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback
import unicodedata
import urllib.request
import uuid
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "rom-jspace-1p5b-atomic-orderops-v1"
PROTOCOL_PATH = Path("results/prereg/JSPACE_1P5B_ATOMIC_ORDEROPS_PREREG_2026-08-18.md")
OUTPUT_ROOT = Path("results/jspace_r1_pilot/followups/atomic_orderops")

JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
JLENS_RAW = (
    "https://raw.githubusercontent.com/anthropics/jacobian-lens/"
    f"{JLENS_COMMIT}/data/evaluations"
)
UPSTREAM_FILES = {
    "multihop": {
        "filename": "lens-eval-multihop.json",
        "sha256": "50b7e4c9255291c0ca2a8e94615be9f44531fa57bb1a844e4f9616056d987416",
        "n_items": 93,
    },
    "orderops": {
        "filename": "lens-eval-order-ops.json",
        "sha256": "b203206d16ff628152cc86f3838604e06cb54776f3e14fa1c34f150db8bc7560",
        "n_items": 55,
    },
    "typo": {
        "filename": "lens-eval-typo.json",
        "sha256": "9d05e16b7234a57d0773d120a4e1c4e94fd3bc2235a8125d4200a70e60ab17aa",
        "n_items": 96,
    },
}
AUX_UPSTREAM_FILES = {
    "evaluation_readme": {
        "filename": "README.md",
        "sha256": "e061d9cce02a1cc651d58a81927833b760d3cef65bf4995126ecbe372a0ebe07",
    }
}

ELIGIBILITY_MANIFEST = Path("results/prereg/jspace_r1_eval_eligibility_manifest.json")
ELIGIBILITY_SHA256 = "a193ce15ff18d1852a870703dba104763a01b06740c99ec8fd955e3a15927f7c"
INPUT_LOCK = Path("results/prereg/JSPACE_1P5B_ATOMIC_ORDEROPS_INPUTS_2026-08-18.json")
ATOMIC_TASK_FILE = Path("results/prereg/JSPACE_ATOMIC_MULTIHOP_81_TASKS_2026-08-18.json")
ATOMIC_TASK_SHA256 = "44951afb2e454a4d3e3ce114c9a40c9cb0cec1ab990b89e82e7ecaba79e81497"
REQUIREMENTS_LOCK = Path("results/prereg/jspace_1p5b_atomic_orderops_requirements_lock.txt")
BUNDLE_MANIFEST = Path("results/prereg/JSPACE_1P5B_ATOMIC_ORDEROPS_BUNDLE_MANIFEST_2026-08-18.json")
RUNNER_PATH = Path("jspace_1p5b_atomic_orderops.py")
TEST_PATH = Path("tests/test_jspace_1p5b_atomic_orderops.py")
SHARED_SCORER_PATH = Path("jspace_phase1_scoring.py")

MODEL_CELLS: dict[str, dict[str, Any]] = {
    "base": {
        "input_lock_key": "math_base",
        "model": "Qwen/Qwen2.5-Math-1.5B",
        "revision": "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2",
        "lens_path": "data/jlens_local/qwen2.5-math-1.5b_wikitext100.pt",
        "lens_sha256": "ce4f034dd4eabc8d814913b4ed476aa1f9ea3f44e27ed92a408200aa7a1417c8",
        "lens_size_bytes": 127_411_048,
        "expected_lens_n_prompts": 100,
        "expected_head_vocab_size": 151936,
        "expected_tokenizer_len": 151665,
        "expected_special_ids": tuple(range(151643, 151657)),
        "effective_force_bos": False,
        "expected_bos_token_id": None,
    },
    "distill": {
        "input_lock_key": "r1_distill",
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "revision": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
        "lens_path": "data/jlens_local/r1-distill-qwen-1.5b_wikitext100_merged.fp32.pt",
        "lens_sha256": "6b5f1043b3c3fa3fcd8d4d69919772a2d763c145f996477bf715fe4b9b89f672",
        "lens_size_bytes": 254_812_153,
        "expected_lens_n_prompts": 100,
        "expected_head_vocab_size": 151936,
        "expected_tokenizer_len": 151665,
        "expected_special_ids": (151646, 151643),
        "effective_force_bos": True,
        "expected_bos_token_id": 151646,
    },
}

HF_SNAPSHOT_FILES = {
    "config.json": "config_sha256",
    "tokenizer.json": "tokenizer_json_sha256",
    "tokenizer_config.json": "tokenizer_config_sha256",
    "generation_config.json": "generation_config_sha256",
    "model.safetensors": "model_safetensors_sha256",
}

EXPECTED_N_LAYERS = 28
EXPECTED_D_MODEL = 1536
SOURCE_LAYERS = tuple(range(27))
SOURCE_BANDS = {
    "early": tuple(range(0, 9)),
    "mid": tuple(range(9, 18)),
    "late": tuple(range(18, 27)),
}
MAX_SEQ_LEN = 128
K = 25

ATOMIC_GENERATION_SEED = 20260818
ATOMIC_SWAP_SEED = 20260818
ATOMIC_BOOTSTRAP_SEED = 20260819
SWAP_SEED = 20260820
BOOTSTRAP_SEED = 20260821
LABEL_PERMUTATION_SEED = 20260822
TYPO_LABEL_PERMUTATION_SEED = 20260823
N_BOOTSTRAP = 20_000
N_CHECKPOINT_SWAPS = 99_999
N_LABEL_PERMUTATIONS = 99_999
EXPECTED_PURE_TEST_COUNT = 57
ORDEROPS_READOUT_ARMS = ("fp32_j", "fp32_logit", "bf16_j", "bf16_logit")
COMPLETION_STAGE_FILES = {
    "atomic": ("manifest.json", "derivation_manifest.json"),
    "orderops-rank-extraction": ("manifest.json", "rank_derivation_manifest.json"),
    "orderops-analysis": ("report.json", "derivation_manifest.json"),
}

ORDEROPS_LEXICAL_EXCLUSIONS = frozenset(
    {"nested-sub-add-mult", "add-add-add", "square-mult"}
)
ORDEROPS_ANNOTATION_EXCLUSIONS = frozenset({"mult-div-mult"})
ORDEROPS_PRIMARY_EXCLUSIONS = (
    ORDEROPS_LEXICAL_EXCLUSIONS | ORDEROPS_ANNOTATION_EXCLUSIONS
)
ORDEROPS_EXPECTED_KEYS = frozenset(
    {
        "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13",
        "15", "16", "20", "24", "addition", "subtraction", "multiplication",
        "division", "mod", "squared",
    }
)
OPERATION_KEYS = frozenset(
    {"addition", "subtraction", "multiplication", "division", "mod", "squared"}
)
OPERATION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "addition": ("+", "plus", "add", "addition"),
    "subtraction": ("-", "minus", "subtract", "subtraction"),
    "multiplication": ("*", "×", "times", "multiply", "multiplication"),
    "division": ("/", "÷", "divide", "division"),
    "mod": ("%", "mod", "modulo", "remainder"),
    "squared": ("^2", "**2", "square", "squared"),
}

ATOMIC_MAX_NEW_TOKENS = 16
ATOMIC_WRAPPER = (
    "Question: {question}\n"
    "Give only the shortest correct answer; do not explain.\n"
    "Answer:"
)
ATOMIC_QUESTION_ARMS = {
    "hop1": ("hop1_question", "bridge_aliases"),
    "hop2": ("hop2_question", "target_aliases"),
    "composite": ("composite_question", "target_aliases"),
}
ATOMIC_CONTINUITY_ARM = "source_prompt_continuity"
ORDEROPS_COT_WRAPPER = (
    'Solve this arithmetic problem step by step and end with the final answer: "{prompt}"'
)
ORDEROPS_DIRECT_MAX_NEW_TOKENS = 16
ORDEROPS_COT_MAX_NEW_TOKENS = 128


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n").encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_validate_input_lock(repo: Path) -> dict[str, Any]:
    path = repo / INPUT_LOCK
    try:
        lock = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError("input lock is missing or malformed") from exc
    if lock.get("schema_version") != "rom-jspace-1p5b-atomic-orderops-input-lock-v1":
        raise RuntimeError("unexpected input-lock schema")
    for cell, spec in MODEL_CELLS.items():
        entry = lock["models"][spec["input_lock_key"]]
        if entry["id"] != spec["model"] or entry["revision"] != spec["revision"]:
            raise RuntimeError(f"{cell}: input-lock model identity drift")
        if int(entry["expected_n_layers"]) != EXPECTED_N_LAYERS:
            raise RuntimeError(f"{cell}: input-lock layer-count drift")
        if int(entry["expected_d_model"]) != EXPECTED_D_MODEL:
            raise RuntimeError(f"{cell}: input-lock d_model drift")
        if int(entry["expected_head_vocab_size"]) != spec["expected_head_vocab_size"]:
            raise RuntimeError(f"{cell}: input-lock head-vocabulary drift")
        if int(entry["expected_tokenizer_len"]) != spec["expected_tokenizer_len"]:
            raise RuntimeError(f"{cell}: input-lock tokenizer-length drift")
    return lock


def verify_runtime_environment(repo: Path) -> dict[str, Any]:
    """Enforce the sealed interpreter/package versions without loading a model."""
    lock = load_and_validate_input_lock(repo)
    expected = lock["provenance"]["runtime_versions"]
    actual = {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "numpy": importlib.metadata.version("numpy"),
        "tokenizers": importlib.metadata.version("tokenizers"),
    }
    if actual != expected:
        raise RuntimeError(f"sealed runtime-version mismatch: {actual} != {expected}")
    requirements = lock["provenance"]["requirements_lock"]
    if sha256_file(repo / requirements["path"]) != requirements["sha256"]:
        raise RuntimeError("requirements-lock hash mismatch")
    expected_freeze = sorted(
        line.strip() for line in (repo / requirements["path"]).read_text().splitlines()
        if line.strip()
    )
    frozen = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        capture_output=True, text=True, check=True,
    )
    actual_freeze = sorted(line.strip() for line in frozen.stdout.splitlines() if line.strip())
    if actual_freeze != expected_freeze:
        raise RuntimeError("installed environment differs from the complete requirements lock")
    try:
        direct = json.loads(importlib.metadata.distribution("jlens").read_text("direct_url.json"))
        commit = direct["vcs_info"]["commit_id"]
    except Exception as exc:
        raise RuntimeError("cannot establish installed J-Lens commit") from exc
    if commit != JLENS_COMMIT:
        raise RuntimeError(f"installed J-Lens commit drift: {commit}")
    return {
        **actual,
        "requirements_lock_exact": True,
        "requirements_distribution_count": len(expected_freeze),
        "requirements_freeze_sha256": sha256_bytes(
            ("\n".join(expected_freeze) + "\n").encode("utf-8")
        ),
    }


def run_sealed_pure_tests(repo: Path) -> dict[str, Any]:
    """Run the frozen model-free tests in the exact execution interpreter."""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        TEST_PATH.as_posix(),
    ]
    completed = subprocess.run(
        command, cwd=repo, env=environment, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "sealed pure tests failed:\n" + completed.stdout + "\n" + completed.stderr
        )
    summary_line = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    match = re.fullmatch(
        rf"{EXPECTED_PURE_TEST_COUNT} passed in [0-9]+(?:\.[0-9]+)?s",
        summary_line,
    )
    if match is None or completed.stderr.strip():
        raise RuntimeError(
            "sealed pure-test receipt is unexpected:\n"
            + completed.stdout + "\n" + completed.stderr
        )
    return {
        "test_file_sha256": sha256_file(repo / TEST_PATH),
        "fixed_selection": TEST_PATH.as_posix(),
        "collected_count": EXPECTED_PURE_TEST_COUNT,
        "passed_count": EXPECTED_PURE_TEST_COUNT,
        "status": "pass",
    }


def verified_hf_snapshot(
    repo: Path,
    cell_name: str,
    *,
    include_model_weights: bool,
    cache_dir: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve the pinned HF snapshot and verify every registered file byte."""
    from huggingface_hub import snapshot_download

    lock = load_and_validate_input_lock(repo)
    spec = MODEL_CELLS[cell_name]
    entry = lock["models"][spec["input_lock_key"]]
    filenames = [
        name for name in HF_SNAPSHOT_FILES
        if include_model_weights or name != "model.safetensors"
    ]
    transport_cache = cache_dir or (Path(tempfile.gettempdir()) / "rom-jspace-hf-transport")
    transport_snapshot = Path(snapshot_download(
        repo_id=spec["model"], revision=spec["revision"],
        allow_patterns=filenames,
        cache_dir=str(transport_cache),
    ))
    materialized = (
        transport_cache / "materialized" / cell_name / str(spec["revision"])
    )
    materialized.mkdir(parents=True, exist_ok=True)
    if materialized.is_symlink():
        raise RuntimeError(f"{cell_name}: materialized snapshot root is a symlink")
    verified: dict[str, Any] = {}
    for filename in filenames:
        source = transport_snapshot / filename
        if not source.is_file():
            raise RuntimeError(f"{cell_name}: pinned snapshot lacks {filename}")
        actual = sha256_file(source)
        expected = entry[HF_SNAPSHOT_FILES[filename]]
        if actual != expected:
            raise RuntimeError(
                f"{cell_name}: {filename} hash mismatch {actual} != {expected}"
            )
        path = materialized / filename
        if path.exists():
            if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"{cell_name}: existing materialized {filename} is unsafe")
        else:
            atomic_copy_regular_file(source, path)
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"{cell_name}: materialized {filename} failed verification")
        verified[filename] = {
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }
    return materialized, {
        "model": spec["model"], "revision": spec["revision"],
        "snapshot_path": str(materialized), "files": verified,
        "model_weights_verified": bool(include_model_weights),
    }


def verify_existing_hf_snapshot(
    repo: Path,
    cell_name: str,
    snapshot: Path,
    *,
    include_model_weights: bool = True,
) -> dict[str, Any]:
    """Re-hash a preflight-resolved local snapshot without network access."""
    lock = load_and_validate_input_lock(repo)
    spec = MODEL_CELLS[cell_name]
    entry = lock["models"][spec["input_lock_key"]]
    filenames = [
        name for name in HF_SNAPSHOT_FILES
        if include_model_weights or name != "model.safetensors"
    ]
    verified: dict[str, Any] = {}
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise RuntimeError(f"{cell_name}: verified snapshot root is unsafe")
    for filename in filenames:
        path = snapshot / filename
        if path.is_symlink():
            raise RuntimeError(f"{cell_name}: required snapshot file is a symlink: {filename}")
        if not path.is_file():
            raise RuntimeError(f"{cell_name}: verified snapshot lacks {filename}")
        actual = sha256_file(path)
        expected = entry[HF_SNAPSHOT_FILES[filename]]
        if actual != expected:
            raise RuntimeError(f"{cell_name}: local {filename} hash drift")
        verified[filename] = {"sha256": actual, "size_bytes": path.stat().st_size}
    return {
        "model": spec["model"],
        "revision": spec["revision"],
        "snapshot_path": str(snapshot),
        "files": verified,
        "model_weights_verified": bool(include_model_weights),
    }


def verify_snapshot_architecture(
    snapshot: Path, cell_name: str, *, include_model_weights: bool
) -> dict[str, Any]:
    """Assert both checkpoint architectures from config and weight headers."""
    spec = MODEL_CELLS[cell_name]
    config_path = snapshot / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise RuntimeError(f"{cell_name}: unsafe architecture config")
    config = json.loads(config_path.read_text())
    observed = {
        "num_hidden_layers": int(config.get("num_hidden_layers", -1)),
        "hidden_size": int(config.get("hidden_size", -1)),
        "vocab_size": int(config.get("vocab_size", -1)),
    }
    expected = {
        "num_hidden_layers": EXPECTED_N_LAYERS,
        "hidden_size": EXPECTED_D_MODEL,
        "vocab_size": int(spec["expected_head_vocab_size"]),
    }
    if observed != expected:
        raise RuntimeError(f"{cell_name}: config architecture drift {observed} != {expected}")
    receipt: dict[str, Any] = {
        "config": observed,
        "weight_header_verified": False,
        "model_weight_dtypes": [],
        "output_domain_source": None,
        "output_domain_shape": None,
    }
    if include_model_weights:
        from safetensors import safe_open

        weight_path = snapshot / "model.safetensors"
        if weight_path.is_symlink() or not weight_path.is_file():
            raise RuntimeError(f"{cell_name}: unsafe architecture weight file")
        with safe_open(weight_path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            dtypes = sorted({str(handle.get_slice(key).get_dtype()) for key in keys})
            candidates = [
                key for key in ("lm_head.weight", "model.embed_tokens.weight")
                if key in keys
            ]
            if not candidates:
                raise RuntimeError(f"{cell_name}: cannot establish output-domain weight shape")
            source = candidates[0]
            shape = [int(x) for x in handle.get_slice(source).get_shape()]
        if shape != [spec["expected_head_vocab_size"], EXPECTED_D_MODEL]:
            raise RuntimeError(f"{cell_name}: output-domain weight-header drift {shape}")
        if dtypes != ["BF16"]:
            raise RuntimeError(f"{cell_name}: model weight-dtype drift {dtypes}")
        receipt.update({
            "weight_header_verified": True,
            "model_weight_dtypes": dtypes,
            "output_domain_source": source,
            "output_domain_shape": shape,
        })
    return receipt


def fsync_parent_directory(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_mkdir_chain(path: Path) -> None:
    """Create every absent directory one level at a time and fsync its parent."""
    missing: list[Path] = []
    probe = path
    while not probe.exists():
        missing.append(probe)
        if probe == probe.parent:
            raise RuntimeError(f"cannot establish directory ancestor for {path}")
        probe = probe.parent
    if probe.is_symlink() or not probe.is_dir():
        raise RuntimeError(f"unsafe directory ancestor for {path}: {probe}")
    for directory in reversed(missing):
        directory.mkdir(exist_ok=False)
        fsync_parent_directory(directory)


def atomic_write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        fsync_parent_directory(path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_copy_regular_file(source: Path, destination: Path) -> None:
    """Materialize a verified cache blob as an ordinary, non-symlink file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise RuntimeError(f"refusing to replace symlinked required file: {destination}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with source.open("rb") as reader, os.fdopen(fd, "wb") as writer:
            for chunk in iter(lambda: reader.read(8 << 20), b""):
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(tmp_name, destination)
        fsync_parent_directory(destination)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value))


def atomic_savez(path: Path, **arrays: Any) -> None:
    """Write a compressed NumPy archive through an adjacent atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **arrays)
        with open(tmp_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        fsync_parent_directory(path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def preserve_unbound_files(
    run_dir: Path, names: Sequence[str], *, label: str
) -> Path | None:
    """Move final-looking but unbound bytes aside before deterministic recovery."""
    candidates = [run_dir / name for name in names if (run_dir / name).exists()]
    if not candidates:
        return None
    for path in candidates:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"unsafe unbound {label} payload: {path}")
    recovery = run_dir / "finalization_recovery" / (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-" + uuid.uuid4().hex[:8] + "-" + label
    )
    durable_mkdir_chain(recovery)
    for path in candidates:
        destination = recovery / path.name
        os.replace(path, destination)
        fsync_parent_directory(path)
        fsync_parent_directory(destination)
    return recovery


def append_retry_event(path: Path, event: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise RuntimeError(f"refusing to append through symlink: {path}")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def completed_prefix_receipt(run_dir: Path) -> dict[str, Any]:
    """Hash every retained run byte except retry-control and incomplete markers."""
    excluded = {
        "INCOMPLETE", "ANALYSIS_INCOMPLETE", "retry_log.jsonl",
        "stage_exception.json", "stage_exception_after_resume.json",
        "stage_exception_consumed.json",
    }
    files: dict[str, Any] = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"completed prefix contains a symlink: {path}")
        if path.is_file() and path.name not in excluded:
            relative = path.relative_to(run_dir).as_posix()
            files[relative] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    return {
        "files": files,
        "digest": sha256_bytes(canonical_json_bytes(files)),
    }


def write_stage_exception_record(
    run_dir: Path,
    *,
    stage: str,
    exc: BaseException,
    traceback_text: str,
) -> tuple[Path, dict[str, Any]]:
    """Atomically retain the sole authoritative technical-failure receipt."""
    consumed = (run_dir / "stage_exception_consumed.json").exists()
    destination = run_dir / (
        "stage_exception_after_resume.json" if consumed else "stage_exception.json"
    )
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"refusing to overwrite technical-failure receipt: {destination}")
    preflight = json.loads((run_dir / "preflight.json").read_text())
    fingerprint = preflight["device_fingerprint"]
    prefix = completed_prefix_receipt(run_dir)
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "stage_exception",
        "stage": stage,
        "seal_commit": json.loads((run_dir / "seal.json").read_text())["git_commit"],
        "configuration_digest": sha256_file(run_dir / "preflight.json"),
        "completed_prefix_digest": prefix["digest"],
        "completed_prefix_files": prefix["files"],
        "execution_fingerprint": fingerprint,
        "execution_fingerprint_digest": sha256_bytes(canonical_json_bytes(fingerprint)),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback_sha256": sha256_bytes(traceback_text.encode("utf-8")),
        "timestamp_utc": utc_now(),
        "after_resume": consumed,
    }
    atomic_write_json(destination, record)
    if json.loads(destination.read_text()) != record:
        raise RuntimeError("atomically written technical-failure receipt did not revalidate")
    return destination, record


def validate_resumable_stage_exception(
    run_dir: Path,
    *,
    stage: str,
    seal: Mapping[str, Any],
    sealed_preflight: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    path = run_dir / "stage_exception.json"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("resume lacks a safe atomic stage_exception record")
    if (run_dir / "stage_exception_consumed.json").exists():
        raise RuntimeError("the stage_exception record has already been consumed")
    record = json.loads(path.read_text())
    required = {
        "schema_version", "record_type", "stage", "seal_commit",
        "configuration_digest", "completed_prefix_digest", "completed_prefix_files",
        "execution_fingerprint", "execution_fingerprint_digest", "exception_type",
        "exception_message", "traceback_sha256", "timestamp_utc", "after_resume",
    }
    if set(record) != required:
        raise RuntimeError("stage_exception record key-set drift")
    if (record["schema_version"] != SCHEMA_VERSION
            or record["record_type"] != "stage_exception"
            or record["stage"] != stage or record["after_resume"] is not False):
        raise RuntimeError("stage_exception record identity drift")
    if record["seal_commit"] != seal["git_commit"]:
        raise RuntimeError("stage_exception seal drift")
    if record["configuration_digest"] != sha256_file(run_dir / "preflight.json"):
        raise RuntimeError("stage_exception configuration digest drift")
    if record["execution_fingerprint"] != sealed_preflight["device_fingerprint"]:
        raise RuntimeError("stage_exception execution fingerprint drift")
    if record["execution_fingerprint_digest"] != sha256_bytes(
        canonical_json_bytes(record["execution_fingerprint"])
    ):
        raise RuntimeError("stage_exception fingerprint digest drift")
    prefix = completed_prefix_receipt(run_dir)
    if (record["completed_prefix_digest"] != prefix["digest"]
            or record["completed_prefix_files"] != prefix["files"]):
        raise RuntimeError("stage_exception completed-prefix digest drift")
    if re.fullmatch(r"[0-9a-f]{64}", str(record["traceback_sha256"])) is None:
        raise RuntimeError("stage_exception traceback hash is malformed")
    if not isinstance(record["timestamp_utc"], str) or not isinstance(record["exception_type"], str):
        raise RuntimeError("stage_exception timestamp/type is malformed")
    return path, record


def bundle_manifest_input_paths(repo: Path) -> list[Path]:
    """Expand every explicitly sealed bundle entry into a direct input."""
    manifest = json.loads((repo / BUNDLE_MANIFEST).read_text())
    paths: list[Path] = []
    for relative in sorted(manifest.get("files", {})):
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"unsafe sealed-bundle input path: {relative}")
        paths.append(repo / rel)
    return paths


def stage_derivation_extra_inputs(run_dir: Path, stage: str) -> list[Path]:
    preflight_record = json.loads((run_dir / "preflight.json").read_text())
    snapshots = preflight_record["tokenizer_only_orderops"]["verified_hf_snapshots"]
    paths = [
        *(Path(record["path"]) for record in preflight_record["lenses"].values()),
        *snapshot_input_paths(snapshots),
        *fetched_input_paths(run_dir),
        *campaign_input_paths(run_dir),
        stage_selection_path(run_dir),
    ]
    if stage == "orderops-analysis":
        paths.append(run_dir / "RANKS_COMPLETE")
    elif stage not in {"atomic", "orderops-rank-extraction"}:
        raise RuntimeError(f"unregistered derivation stage: {stage}")
    return paths


def derivation_input_table(
    repo: Path, extra_inputs: Sequence[Path]
) -> dict[str, Any]:
    fixed_repo_inputs = {
        PROTOCOL_PATH, INPUT_LOCK, ATOMIC_TASK_FILE, REQUIREMENTS_LOCK,
        BUNDLE_MANIFEST, RUNNER_PATH, TEST_PATH, SHARED_SCORER_PATH,
        ELIGIBILITY_MANIFEST,
    }
    repo_paths = {repo / relative for relative in fixed_repo_inputs}
    repo_paths.update(bundle_manifest_input_paths(repo))
    inputs: dict[str, Any] = {}
    for path in sorted(repo_paths):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"derivation input is missing or unsafe: {path}")
        relative = path.relative_to(repo).as_posix()
        inputs[relative] = {
            "sha256": sha256_file(path), "size_bytes": path.stat().st_size
        }
    for supplied in extra_inputs:
        path = Path(os.path.abspath(supplied))
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"derivation input is missing or unsafe: {path}")
        key = str(path)
        record = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        if key in inputs and inputs[key] != record:
            raise RuntimeError(f"conflicting derivation input record: {path}")
        inputs[key] = record
    return inputs


def derivation_output_table(
    run_dir: Path, filename: str, *, stage: str
) -> dict[str, Any]:
    """Hash the final payloads owned by one stage.

    Rank extraction and analysis intentionally share a run directory.  The
    rank authority must remain valid after analysis adds its own files, so
    each stage has an explicit output namespace instead of rescanning all
    later-stage bytes as if they belonged to the earlier derivation.
    """
    if stage not in COMPLETION_STAGE_FILES:
        raise RuntimeError(f"unregistered derivation-output stage: {stage}")
    outputs: dict[str, Any] = {}
    for path in sorted(run_dir.rglob("*")):
        relative = path.relative_to(run_dir)
        if ("finalization_recovery" in relative.parts
                or "analysis_recovery" in relative.parts):
            continue
        if stage == "orderops-analysis":
            if relative.as_posix() not in {"report.json", "REPORT.md"}:
                continue
        elif stage == "orderops-rank-extraction" and relative.as_posix() in {
            "report.json", "REPORT.md", "derivation_manifest.json",
        }:
            continue
        if path.is_file() and path.name not in {
            filename, "INCOMPLETE", "ANALYSIS_INCOMPLETE", "COMPLETE",
            "RANKS_COMPLETE", "ANALYSIS_COMPLETE",
        }:
            if path.is_symlink():
                raise RuntimeError(f"unsafe derivation output: {path}")
            outputs[relative.as_posix()] = {
                "sha256": sha256_file(path), "size_bytes": path.stat().st_size
            }
    if not outputs:
        raise RuntimeError("derivation has no final evidence outputs")
    return outputs


def validate_derivation_manifest(
    repo: Path,
    run_dir: Path,
    path: Path,
    *,
    stage: str,
) -> dict[str, Any]:
    if path.parent != run_dir or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{stage}: derivation manifest path is unsafe")
    payload = json.loads(path.read_text())
    if set(payload) != {"schema_version", "stage", "created_utc", "inputs", "outputs"}:
        raise RuntimeError(f"{stage}: derivation manifest key-set drift")
    if (payload["schema_version"] != SCHEMA_VERSION or payload["stage"] != stage
            or not isinstance(payload["created_utc"], str)):
        raise RuntimeError(f"{stage}: derivation manifest identity drift")
    expected_inputs = derivation_input_table(
        repo, stage_derivation_extra_inputs(run_dir, stage)
    )
    expected_outputs = derivation_output_table(run_dir, path.name, stage=stage)
    if payload["inputs"] != expected_inputs:
        raise RuntimeError(f"{stage}: derivation input table drift")
    if payload["outputs"] != expected_outputs:
        raise RuntimeError(f"{stage}: derivation output table drift")
    return payload


def write_derivation_manifest(
    repo: Path,
    run_dir: Path,
    *,
    stage: str,
    extra_inputs: Sequence[Path] = (),
    filename: str = "derivation_manifest.json",
) -> Path:
    """Hash and then independently validate every direct input and final payload."""
    registered_extra = stage_derivation_extra_inputs(run_dir, stage)
    if {str(Path(os.path.abspath(path))) for path in extra_inputs} != {
        str(Path(os.path.abspath(path))) for path in registered_extra
    }:
        raise RuntimeError(f"{stage}: caller derivation-input set differs from registration")
    destination = run_dir / filename
    atomic_write_json(destination, {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "created_utc": utc_now(),
        "inputs": derivation_input_table(repo, registered_extra),
        "outputs": derivation_output_table(run_dir, filename, stage=stage),
    })
    validate_derivation_manifest(repo, run_dir, destination, stage=stage)
    return destination


def snapshot_input_paths(records: Mapping[str, Mapping[str, Any]]) -> list[Path]:
    """Return every verified HF file as a direct derivation input."""
    paths: list[Path] = []
    for record in records.values():
        snapshot = Path(str(record["snapshot_path"]))
        for filename in sorted(record["files"]):
            path = snapshot / filename
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"unsafe verified HF derivation input: {path}")
            paths.append(path)
    return paths


def fetched_input_paths(run_dir: Path) -> list[Path]:
    paths = [
        run_dir / "input_files" / str(spec["filename"])
        for spec in (*UPSTREAM_FILES.values(), *AUX_UPSTREAM_FILES.values())
    ]
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"fetched derivation input is missing or unsafe: {path}")
    return paths


def completion_marker_payload(
    repo: Path,
    run_dir: Path,
    *,
    stage: str,
    manifest_name: str,
    derivation_name: str,
) -> dict[str, Any]:
    expected_names = COMPLETION_STAGE_FILES.get(stage)
    if expected_names != (manifest_name, derivation_name):
        raise RuntimeError(f"{stage}: unregistered completion authority filenames")
    manifest_path = run_dir / manifest_name
    derivation_path = run_dir / derivation_name
    if (manifest_path.parent != run_dir or derivation_path.parent != run_dir
            or manifest_path.is_symlink() or derivation_path.is_symlink()
            or not manifest_path.is_file() or not derivation_path.is_file()):
        raise RuntimeError(f"{stage}: cannot complete without manifest and derivation")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("stage") != stage):
        raise RuntimeError(f"{stage}: stage manifest identity drift")
    derivation = validate_derivation_manifest(
        repo, run_dir, derivation_path, stage=stage
    )
    final_evidence = derivation.get("outputs")
    if not isinstance(final_evidence, dict) or not final_evidence:
        raise RuntimeError(f"{stage}: derivation has no final evidence table")
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "manifest": {
            "path": manifest_name,
            "sha256": sha256_file(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        },
        "derivation": {
            "path": derivation_name,
            "sha256": sha256_file(derivation_path),
            "size_bytes": derivation_path.stat().st_size,
        },
        "final_evidence": final_evidence,
    }


def validate_completion_marker(
    repo: Path, run_dir: Path, marker_name: str, stage: str
) -> dict[str, Any]:
    path = run_dir / marker_name
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{stage}: completion marker is missing or unsafe")
    marker = json.loads(path.read_text())
    if set(marker) != {
        "schema_version", "stage", "manifest", "derivation", "final_evidence"
    }:
        raise RuntimeError(f"{stage}: completion marker key-set drift")
    if marker.get("schema_version") != SCHEMA_VERSION or marker.get("stage") != stage:
        raise RuntimeError(f"{stage}: completion marker schema drift")
    expected_names = COMPLETION_STAGE_FILES.get(stage)
    if expected_names is None:
        raise RuntimeError(f"{stage}: unregistered completion stage")
    for key in ("manifest", "derivation"):
        record = marker.get(key, {})
        if set(record) != {"path", "sha256", "size_bytes"}:
            raise RuntimeError(f"{stage}: marker-bound {key} record drift")
        expected_name = expected_names[0 if key == "manifest" else 1]
        if record.get("path") != expected_name:
            raise RuntimeError(f"{stage}: marker-bound {key} path drift")
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise RuntimeError(f"{stage}: unsafe marker-bound {key} path")
        target = run_dir / relative
        if target.parent != run_dir or target.is_symlink() or not target.is_file():
            raise RuntimeError(f"{stage}: marker-bound {key} is missing or unsafe")
        if sha256_file(target) != record.get("sha256"):
            raise RuntimeError(f"{stage}: marker-bound {key} hash mismatch")
        if target.stat().st_size != int(record.get("size_bytes", -1)):
            raise RuntimeError(f"{stage}: marker-bound {key} size mismatch")
    manifest_payload = json.loads((run_dir / expected_names[0]).read_text())
    if (manifest_payload.get("schema_version") != SCHEMA_VERSION
            or manifest_payload.get("stage") != stage):
        raise RuntimeError(f"{stage}: marker-bound stage manifest identity drift")
    derivation = validate_derivation_manifest(
        repo, run_dir, run_dir / expected_names[1], stage=stage
    )
    if marker.get("final_evidence") != derivation.get("outputs"):
        raise RuntimeError(f"{stage}: completion marker evidence table drift")
    for relative, record in marker["final_evidence"].items():
        rel = Path(relative)
        if (rel.is_absolute() or ".." in rel.parts
                or not isinstance(record, dict)
                or set(record) != {"sha256", "size_bytes"}):
            raise RuntimeError(f"{stage}: unsafe marker-bound evidence path")
        target = run_dir / rel
        if target.is_symlink() or not target.is_file():
            raise RuntimeError(f"{stage}: marker-bound evidence is missing or unsafe: {relative}")
        if sha256_file(target) != record.get("sha256"):
            raise RuntimeError(f"{stage}: marker-bound evidence hash mismatch: {relative}")
        if target.stat().st_size != int(record.get("size_bytes", -1)):
            raise RuntimeError(f"{stage}: marker-bound evidence size mismatch: {relative}")
    return marker


def finalize_stage(
    repo: Path,
    run_dir: Path,
    *,
    stage: str,
    marker_name: str,
    manifest_name: str,
    derivation_name: str,
    incomplete_name: str = "INCOMPLETE",
) -> None:
    marker = completion_marker_payload(
        repo, run_dir, stage=stage, manifest_name=manifest_name,
        derivation_name=derivation_name,
    )
    atomic_write_json(run_dir / marker_name, marker)
    validate_completion_marker(repo, run_dir, marker_name, stage)
    incomplete = run_dir / incomplete_name
    if incomplete.exists():
        if incomplete.is_symlink():
            raise RuntimeError(f"{stage}: unsafe incomplete marker")
        incomplete.unlink()
        fsync_parent_directory(incomplete)


def atomic_report_markdown(report: Mapping[str, Any]) -> str:
    k = report["co_primary"]["delta_k"]
    x = report["co_primary"]["delta_x"]
    lines = [
        "# A0 — 1.5B atomic-hop behavioural audit", "",
        "Exploratory follow-up under the sealed protocol. Independent unit: source item.", "",
        "| Estimand | Estimate | 95% paired-bootstrap interval | Holm p |",
        "|---|---:|---:|---:|",
    ]
    for label, result in (("Delta K", k), ("Delta X", x)):
        boot = result["bootstrap"]
        swap = result["checkpoint_vector_swap"]
        lines.append(
            f"| {label} | {boot['estimate']:.4f} | "
            f"[{boot['percentile_95_ci'][0]:.4f}, {boot['percentile_95_ci'][1]:.4f}] | "
            f"{swap['holm_adjusted_two_sided_p']:.6f} |"
        )
    lines += ["", "## Behavioural scoring sensitivities", "",
              "| Cell | Arm | Exact | Whole-label | Substring |",
              "|---|---|---:|---:|---:|"]
    for cell, arms in report["behavioral_scoring_sensitivities"].items():
        for arm, values in arms.items():
            lines.append(
                f"| {cell} | {arm} | {values['exact_rate']:.4f} | "
                f"{values['whole_term_rate']:.4f} | {values['substring_rate']:.4f} |"
            )
    lines += ["", f"Registered pattern: **{report['pattern']}**.", "",
              "This operational composition contrast is not causal mediation.", ""]
    return "\n".join(lines)


def orderops_report_markdown(report: Mapping[str, Any]) -> str:
    if report.get("analysis_stopped_before_checkpoint_inference"):
        return "\n".join([
            "# O1 — 1.5B order-of-operations J-Lens readout", "",
            "The sealed same-runtime typo repeatability control failed.", "",
            "No O1 checkpoint contrast, bootstrap, permutation test, or reasoning interpretation was calculated.",
            "", "Status: **instrument-invalid in this runtime**.", "",
        ])
    delta = report["co_primary"]["delta_j"]
    did = report["co_primary"]["j_minus_logit_difference_in_differences"]
    lines = [
        "# O1 — 1.5B order-of-operations J-Lens readout", "",
        "Exploratory, checkpoint-local, and lens-fit-bounded follow-up.", "",
        "| Estimand | Estimate | 95% paired-bootstrap interval | Holm p |",
        "|---|---:|---:|---:|",
    ]
    for label, result in (("Delta J (distill - base)", delta), ("DiD", did)):
        boot = result["bootstrap"]
        swap = result["checkpoint_vector_swap"]
        lines.append(
            f"| {label} | {boot['estimate']:.4f} | "
            f"[{boot['percentile_95_ci'][0]:.4f}, {boot['percentile_95_ci'][1]:.4f}] | "
            f"{swap['holm_adjusted_two_sided_p']:.6f} |"
        )
    lines += [
        "", f"Assay-presence gate: **{'pass' if report['assay_presence_gate']['pass'] else 'fail'}**.",
        f"Same-runtime typo control: **{'pass' if report['same_runtime_typo_control']['integrity_pass'] else 'fail'}**.",
        f"Registered pair pattern: **{report['pair_pattern']}**.", "",
        "Ranks do not establish a causal effect of distillation or a global workspace.", "",
    ]
    return "\n".join(lines)


def run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=check
    )


def _status_path(line: str) -> str:
    """Extract a path from a porcelain-v1 status line (renames use destination)."""
    body = line[3:]
    if " -> " in body:
        body = body.split(" -> ", 1)[1]
    return body.strip().strip('"')


def validate_sealed_checkout(
    repo: Path,
    seal_commit: str,
    *,
    allowed_untracked_root: Path | None = None,
) -> dict[str, str]:
    """Require an exact sealed checkout before any model-dependent action.

    Existing untracked experiment outputs may be allowed so that atomic and
    order-ops stages can coexist.  Tracked modifications are never allowed,
    nor are untracked files elsewhere in the repository.
    """
    if re.fullmatch(r"[0-9a-f]{40}", seal_commit) is None:
        raise RuntimeError("--seal-commit must be a full lowercase 40-hex commit")
    head = run_git(repo, "rev-parse", "HEAD").stdout.strip()
    if head != seal_commit:
        raise RuntimeError(f"HEAD {head!r} does not equal seal commit {seal_commit!r}")

    required_paths = (
        PROTOCOL_PATH, INPUT_LOCK, ATOMIC_TASK_FILE, REQUIREMENTS_LOCK,
        BUNDLE_MANIFEST, RUNNER_PATH, TEST_PATH, SHARED_SCORER_PATH,
    )
    current_hashes: dict[str, str] = {}
    for relative in required_paths:
        path = repo / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"sealed bundle file is missing: {relative}")
        tracked = run_git(repo, "ls-files", "--error-unmatch", str(relative), check=False)
        if tracked.returncode != 0:
            raise RuntimeError(f"sealed bundle file is not tracked: {relative}")
        committed = run_git(repo, "show", f"{seal_commit}:{relative.as_posix()}").stdout.encode()
        current = path.read_bytes()
        if committed != current:
            raise RuntimeError(f"working bytes differ from seal commit: {relative}")
        current_hashes[relative.as_posix()] = sha256_bytes(current)

    allowed_rel: str | None = None
    if allowed_untracked_root is not None:
        try:
            allowed_rel = allowed_untracked_root.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError:
            # A durable output directory outside the clean execution checkout
            # cannot dirty this checkout and therefore needs no status waiver.
            allowed_rel = None
    dirty = run_git(repo, "status", "--porcelain=v1", "--untracked-files=all").stdout.splitlines()
    rejected: list[str] = []
    for line in dirty:
        path = _status_path(line)
        allowed = (
            line.startswith("?? ")
            and allowed_rel is not None
            and (path == allowed_rel or path.startswith(allowed_rel + "/"))
        )
        if not allowed:
            rejected.append(line)
    if rejected:
        raise RuntimeError("checkout is not seal-clean:\n" + "\n".join(rejected))
    bundle = validate_bundle_manifest(repo)
    return {
        "git_commit": head,
        "git_dirty": False,
        "protocol_path": PROTOCOL_PATH.as_posix(),
        "protocol_sha256": current_hashes[PROTOCOL_PATH.as_posix()],
        "sealed_bundle_sha256": current_hashes,
        "bundle_manifest_sha256": sha256_file(repo / BUNDLE_MANIFEST),
        "bundle_manifest_file_count": len(bundle["files"]),
    }


def validate_bundle_manifest(repo: Path) -> dict[str, Any]:
    """Verify every non-self entry in the sealed bundle manifest."""
    path = repo / BUNDLE_MANIFEST
    try:
        manifest = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError("bundle manifest is missing or malformed") from exc
    if manifest.get("schema_version") != "rom-jspace-1p5b-seal-bundle-v1":
        raise RuntimeError("unexpected bundle-manifest schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("bundle manifest has no file table")
    mandatory = {
        PROTOCOL_PATH.as_posix(), INPUT_LOCK.as_posix(), ATOMIC_TASK_FILE.as_posix(),
        REQUIREMENTS_LOCK.as_posix(), RUNNER_PATH.as_posix(), TEST_PATH.as_posix(),
        SHARED_SCORER_PATH.as_posix(), ELIGIBILITY_MANIFEST.as_posix(),
    }
    if not mandatory <= set(files):
        raise RuntimeError(
            f"bundle manifest is missing mandatory files: {sorted(mandatory - set(files))}"
        )
    for relative, record in files.items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"unsafe path in bundle manifest: {relative}")
        candidate = repo / rel
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError(f"bundle-manifest file is absent: {relative}")
        actual_size = candidate.stat().st_size
        actual_sha = sha256_file(candidate)
        if int(record.get("size_bytes", -1)) != actual_size:
            raise RuntimeError(f"bundle-manifest size mismatch: {relative}")
        if record.get("sha256") != actual_sha:
            raise RuntimeError(f"bundle-manifest hash mismatch: {relative}")
    return manifest


def validate_output_root(repo: Path, outroot: Path) -> Path:
    """Resolve an isolated output root and reject broad or ambiguous paths.

    Absolute roots are supported because clean execution checkouts commonly
    write into the main repository's durable results tree.  The final three
    path components are fixed to prevent accidental broad writes.
    """
    candidate = outroot if outroot.is_absolute() else repo / outroot
    resolved = candidate.resolve(strict=False)
    if tuple(resolved.parts[-4:]) != (
        "results", "jspace_r1_pilot", "followups", "atomic_orderops"
    ):
        raise RuntimeError(
            "output root must end in results/jspace_r1_pilot/followups/atomic_orderops"
        )
    if len(resolved.parts) < 5 or resolved == Path(resolved.anchor):
        raise RuntimeError("output root is too broad")
    probe = candidate
    while probe != probe.parent:
        if probe.is_symlink():
            raise RuntimeError(f"output path contains a symlink: {probe}")
        if probe == repo:
            break
        probe = probe.parent
    return resolved


def reject_symlinks_beneath(root: Path) -> None:
    """Reject every symlink in an existing run tree before any resume/write."""
    if root.is_symlink():
        raise RuntimeError(f"run directory is a symlink: {root}")
    if not root.is_dir():
        raise RuntimeError(f"run directory is missing: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"run tree contains a symlink: {path}")


def validate_mutable_cache_dir(repo: Path, cache_dir: Path, outroot: Path) -> Path:
    """Keep all mutable download/cache state outside checkout and evidence output."""
    candidate = cache_dir.resolve(strict=False)
    repo_resolved = repo.resolve()
    outroot_resolved = outroot.resolve(strict=False)
    if (candidate == repo_resolved or repo_resolved in candidate.parents
            or candidate in repo_resolved.parents):
        raise RuntimeError("mutable cache directory must be outside the sealed checkout")
    if (candidate == outroot_resolved or outroot_resolved in candidate.parents
            or candidate in outroot_resolved.parents):
        raise RuntimeError("mutable cache directory must be outside the evidence output root")
    probe = candidate
    while probe != probe.parent:
        if probe.is_symlink():
            raise RuntimeError(f"mutable cache path contains a symlink: {probe}")
        probe = probe.parent
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def execution_device_fingerprint(torch: Any) -> dict[str, Any]:
    """Freeze the backend properties that can change numerical execution."""
    device, dtype = _device_and_dtype(torch)
    probe_input = {
        "schema": "rom-bf16-probe-v1",
        "a": [1, 2, 3, 4, 5, 6, 7, 8],
        "b": [8, 7, 6, 5, 4, 3, 2, 1],
        "expression": "a*b+a-b",
    }
    try:
        a = torch.tensor(probe_input["a"], device=device, dtype=torch.bfloat16)
        b = torch.tensor(probe_input["b"], device=device, dtype=torch.bfloat16)
        probe_values = [float(x) for x in (a * b + a - b).float().cpu().tolist()]
        probe_payload = {**probe_input, "result_float32": probe_values}
        probe_digest = sha256_bytes(canonical_json_bytes(probe_payload))
        probe_supported = True
    except Exception as exc:
        raise RuntimeError(f"{device}: deterministic BF16 arithmetic probe failed") from exc
    record: dict[str, Any] = {
        "backend": device,
        "device": device,
        "device_index": 0,
        "requested_dtype": str(dtype),
        "forward_dtype": str(dtype),
        "observed_model_dtype": None,
        "torch_version": str(torch.__version__),
        "platform": platform.platform(),
        "platform_machine": platform.machine(),
        "accelerator_runtime_version": None,
        "accelerator_compute_capability": None,
        "bf16_execution_supported": False,
        "bf16_support_basis": "unsupported backend",
        "bf16_probe_digest": probe_digest,
        "bf16_probe_result_float32": probe_values,
        "mps_available": bool(
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ),
    }
    if device == "cuda":
        index = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(index)
        record.update({
            "device_index": index,
            "physical_device_name": str(props.name),
            "accelerator_runtime_version": str(torch.version.cuda),
            "accelerator_compute_capability": [
                int(x) for x in torch.cuda.get_device_capability(index)
            ],
            "bf16_execution_supported": bool(
                probe_supported and torch.cuda.is_bf16_supported()
            ),
            "bf16_support_basis": (
                "successful deterministic BF16 probe and "
                "torch.cuda.is_bf16_supported()"
            ),
        })
        record["cuda"] = {
            "index": index,
            "name": str(props.name),
            "capability": [int(x) for x in torch.cuda.get_device_capability(index)],
            "total_memory_bytes": int(props.total_memory),
        }
    elif device == "mps":
        profiler = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, check=True,
        )
        displays = json.loads(profiler.stdout).get("SPDisplaysDataType", [])
        gpu_rows = [row for row in displays if row.get("sppci_device_type") == "spdisplays_gpu"]
        if not gpu_rows:
            raise RuntimeError("MPS fingerprint cannot identify the physical GPU")
        gpu = sorted(gpu_rows, key=lambda row: str(row.get("_name", "")))[0]
        metal = str(gpu.get("spdisplays_mtlgpufamilysupport", "unknown"))
        record.update({
            "physical_device_name": str(gpu.get("sppci_model") or gpu.get("_name")),
            "accelerator_runtime_version": f"macOS-{platform.mac_ver()[0]}",
            "accelerator_compute_capability": metal,
            "bf16_execution_supported": bool(
                probe_supported
                and torch.backends.mps.is_built()
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_macos_or_newer(14, 0)
            ),
            "bf16_support_basis": (
                "successful deterministic BF16 probe plus built/available MPS "
                "backend on macOS 14+"
            ),
        })
        record["mps"] = {
            "built": bool(torch.backends.mps.is_built()),
            "gpu_cores": str(gpu.get("sppci_cores", "unknown")),
            "metal_family": metal,
        }
    else:
        processor = platform.processor() or platform.machine()
        record.update({
            "physical_device_name": processor,
            "accelerator_runtime_version": "cpu",
            "accelerator_compute_capability": str(
                getattr(torch.backends.cpu, "get_cpu_capability", lambda: "unknown")()
            ),
        })
        record["cpu"] = {"processor": processor}
    return record


def bind_observed_model_dtype(
    fingerprint: Mapping[str, Any],
    architecture_receipts: Mapping[str, Mapping[str, Any]],
    *,
    require_weight_headers: bool,
) -> dict[str, Any]:
    """Bind the execution fingerprint to dtypes observed in both weight headers."""
    bound = dict(fingerprint)
    dtype_sets = {
        cell: list(architecture_receipts[cell].get("model_weight_dtypes", []))
        for cell in MODEL_CELLS
    }
    if require_weight_headers:
        if dtype_sets != {cell: ["BF16"] for cell in MODEL_CELLS}:
            raise RuntimeError(f"globally observed model-weight dtype drift: {dtype_sets}")
        bound["observed_model_dtype"] = "torch.bfloat16"
    else:
        bound["observed_model_dtype"] = None
    return bound


def execution_binding(run_dir: Path) -> dict[str, Any]:
    """Bind every resumable payload to the seal, preflight and device."""
    seal_path = run_dir / "seal.json"
    preflight_path = run_dir / "preflight.json"
    return {
        "schema_version": SCHEMA_VERSION,
        "seal_sha256": sha256_file(seal_path),
        "preflight_sha256": sha256_file(preflight_path),
        "campaign_authority_sha256": sha256_file(
            run_dir / "campaign_authority.json"
        ),
        "stage_selection_sha256": sha256_file(stage_selection_path(run_dir)),
        "run_identity_sha256": sha256_file(run_dir / "run_identity.json"),
        "device_fingerprint": json.loads(preflight_path.read_text())["device_fingerprint"],
    }


def assert_binding(actual: Mapping[str, Any], expected: Mapping[str, Any], context: str) -> None:
    if dict(actual) != dict(expected):
        raise RuntimeError(f"{context}: execution binding drift")


def run_identity_payload(run_dir: Path, stage: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "seal_sha256": sha256_file(run_dir / "seal.json"),
        "preflight_sha256": sha256_file(run_dir / "preflight.json"),
        "campaign_authority_sha256": sha256_file(
            run_dir / "campaign_authority.json"
        ),
        "stage_selection_sha256": sha256_file(stage_selection_path(run_dir)),
    }


def validate_run_identity(run_dir: Path, stage: str) -> dict[str, Any]:
    path = run_dir / "run_identity.json"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("run identity is missing or unsafe")
    payload = json.loads(path.read_text())
    expected = run_identity_payload(run_dir, stage)
    if set(payload) != set(expected) or payload != expected:
        raise RuntimeError("run identity drift")
    return payload


def partial_completed_prefix_digest(
    *,
    stage: str,
    binding: Mapping[str, Any],
    ordered_identifiers: Sequence[Any],
    payload_sha256: str,
) -> str:
    """Digest the exact semantic prefix without recursively hashing its envelope."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "execution_binding": dict(binding),
        "ordered_identifiers": list(ordered_identifiers),
        "payload_sha256": payload_sha256,
    }
    return sha256_bytes(canonical_json_bytes(payload))


def new_run_dir(outroot: Path, stage: str) -> Path:
    run_id = f"{stage}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    durable_mkdir_chain(outroot)
    parent = outroot / stage
    if parent.exists():
        raise RuntimeError(f"the sealed campaign already selected a {stage} run")
    durable_mkdir_chain(parent)
    path = parent / run_id
    path.mkdir(exist_ok=False)
    fsync_parent_directory(path)
    atomic_write_bytes(path / "INCOMPLETE", b"created before model-dependent execution\n")
    return path


def stage_selection_path(run_dir: Path) -> Path:
    return run_dir.parent / "STAGE_SELECTED.json"


def expected_stage_selection_payload(run_dir: Path, stage: str) -> dict[str, Any]:
    campaign_reference = json.loads(
        (run_dir / "campaign_authority.json").read_text()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "stage-selection",
        "stage": stage,
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "seal_sha256": sha256_file(run_dir / "seal.json"),
        "preflight_sha256": sha256_file(run_dir / "preflight.json"),
        "campaign_authority_sha256": sha256_file(
            run_dir / "campaign_authority.json"
        ),
        "campaign_selection_sha256": campaign_reference[
            "campaign_selection_sha256"
        ],
    }


def write_stage_selection(run_dir: Path, stage: str) -> dict[str, Any]:
    """Select the sole fresh run permitted for one campaign stage."""
    if {path.name for path in run_dir.parent.iterdir()} != {run_dir.name}:
        raise RuntimeError(f"{stage}: stage parent is not an unclaimed sole-run directory")
    payload = expected_stage_selection_payload(run_dir, stage)
    atomic_write_json(stage_selection_path(run_dir), payload)
    return validate_stage_selection(run_dir, stage)


def validate_stage_selection(run_dir: Path, stage: str) -> dict[str, Any]:
    parent = run_dir.parent
    path = stage_selection_path(run_dir)
    if (parent.name != stage or parent.is_symlink() or path.is_symlink()
            or not path.is_file()):
        raise RuntimeError(f"{stage}: stage selection is missing or unsafe")
    if {candidate.name for candidate in parent.iterdir()} != {
        run_dir.name, "STAGE_SELECTED.json"
    }:
        raise RuntimeError(f"{stage}: stage parent does not contain one selected run")
    payload = json.loads(path.read_text())
    expected = expected_stage_selection_payload(run_dir, stage)
    if set(payload) != set(expected) or payload != expected:
        raise RuntimeError(f"{stage}: stage selection authority drift")
    return payload


def new_campaign_dir(outroot: Path) -> Path:
    campaign_id = (
        "campaign-"
        + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-" + uuid.uuid4().hex[:12]
    )
    durable_mkdir_chain(outroot)
    parent = outroot / "campaign"
    if not parent.exists():
        durable_mkdir_chain(parent)
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError("campaign authority parent is unsafe")
    if any(parent.iterdir()):
        raise RuntimeError("the sealed output root already has a selected campaign")
    path = parent / campaign_id
    path.mkdir(parents=False, exist_ok=False)
    fsync_parent_directory(path)
    return path


def write_campaign_authority(
    campaign_dir: Path,
    *,
    seal: Mapping[str, Any],
    sealed_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the one immutable pre-forward authority shared by A0 and O1."""
    campaign_dir = campaign_dir.resolve()
    campaign_id = campaign_dir.name
    if re.fullmatch(r"campaign-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}", campaign_id) is None:
        raise RuntimeError("campaign authority directory has an invalid UUID name")
    if campaign_dir.parent.is_symlink() or not campaign_dir.parent.is_dir():
        raise RuntimeError("campaign authority parent is unsafe")
    if {path.name for path in campaign_dir.parent.iterdir()} != {campaign_id}:
        raise RuntimeError("campaign authority is not the sole preselection candidate")
    if any(campaign_dir.iterdir()):
        raise RuntimeError("campaign authority directory is not empty")
    atomic_write_json(campaign_dir / "seal.json", dict(seal))
    atomic_write_json(campaign_dir / "preflight.json", dict(sealed_preflight))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "campaign-preflight",
        "created_utc": utc_now(),
        "campaign_id": campaign_id,
        "campaign_dir": str(campaign_dir),
        "seal_sha256": sha256_file(campaign_dir / "seal.json"),
        "preflight_sha256": sha256_file(campaign_dir / "preflight.json"),
        "execution_fingerprint_digest": sha256_bytes(canonical_json_bytes(
            sealed_preflight["device_fingerprint"]
        )),
    }
    atomic_write_json(campaign_dir / "campaign_manifest.json", manifest)
    ready = {
        "schema_version": SCHEMA_VERSION,
        "stage": "campaign-preflight",
        "campaign_id": campaign_id,
        "campaign_dir": str(campaign_dir),
        "manifest_path": "campaign_manifest.json",
        "manifest_sha256": sha256_file(campaign_dir / "campaign_manifest.json"),
        "seal_sha256": manifest["seal_sha256"],
        "preflight_sha256": manifest["preflight_sha256"],
    }
    atomic_write_json(campaign_dir / "CAMPAIGN_READY", ready)
    selection = {
        "schema_version": SCHEMA_VERSION,
        "stage": "campaign-selection",
        "campaign_id": campaign_id,
        "campaign_dir": str(campaign_dir),
        "ready_path": f"{campaign_id}/CAMPAIGN_READY",
        "ready_sha256": sha256_file(campaign_dir / "CAMPAIGN_READY"),
        "manifest_sha256": sha256_file(campaign_dir / "campaign_manifest.json"),
        "seal_sha256": manifest["seal_sha256"],
        "preflight_sha256": manifest["preflight_sha256"],
    }
    atomic_write_json(campaign_dir.parent / "CAMPAIGN_SELECTED.json", selection)
    return validate_campaign_authority(
        campaign_dir, seal=seal, sealed_preflight=sealed_preflight
    )


def validate_campaign_authority(
    campaign_dir: Path,
    *,
    seal: Mapping[str, Any],
    sealed_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    if campaign_dir.is_symlink():
        raise RuntimeError("campaign authority directory is a symlink")
    campaign_dir = campaign_dir.resolve()
    campaign_id = campaign_dir.name
    if re.fullmatch(r"campaign-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}", campaign_id) is None:
        raise RuntimeError("campaign authority directory has an invalid UUID name")
    selection_path = campaign_dir.parent / "CAMPAIGN_SELECTED.json"
    if selection_path.is_symlink() or not selection_path.is_file():
        raise RuntimeError("campaign selection authority is missing or unsafe")
    if {path.name for path in campaign_dir.parent.iterdir()} != {
        campaign_id, "CAMPAIGN_SELECTED.json"
    }:
        raise RuntimeError("campaign parent does not contain one unique selection")
    reject_symlinks_beneath(campaign_dir)
    expected_names = {
        "seal.json", "preflight.json", "campaign_manifest.json", "CAMPAIGN_READY"
    }
    actual_names = {path.name for path in campaign_dir.iterdir()}
    if actual_names != expected_names:
        raise RuntimeError("campaign authority file-set drift")
    stored_seal = json.loads((campaign_dir / "seal.json").read_text())
    stored_preflight = json.loads((campaign_dir / "preflight.json").read_text())
    if stored_seal != dict(seal) or stored_preflight != dict(sealed_preflight):
        raise RuntimeError("campaign authority differs from current sealed preflight")
    manifest = json.loads((campaign_dir / "campaign_manifest.json").read_text())
    if set(manifest) != {
        "schema_version", "stage", "created_utc", "campaign_id", "campaign_dir",
        "seal_sha256",
        "preflight_sha256", "execution_fingerprint_digest",
    }:
        raise RuntimeError("campaign manifest key-set drift")
    if (manifest["schema_version"] != SCHEMA_VERSION
            or manifest["stage"] != "campaign-preflight"
            or manifest["campaign_id"] != campaign_id
            or manifest["campaign_dir"] != str(campaign_dir)
            or not isinstance(manifest["created_utc"], str)):
        raise RuntimeError("campaign manifest identity drift")
    expected_manifest = {
        "seal_sha256": sha256_file(campaign_dir / "seal.json"),
        "preflight_sha256": sha256_file(campaign_dir / "preflight.json"),
        "execution_fingerprint_digest": sha256_bytes(canonical_json_bytes(
            sealed_preflight["device_fingerprint"]
        )),
    }
    if any(manifest[key] != value for key, value in expected_manifest.items()):
        raise RuntimeError("campaign manifest digest drift")
    ready = json.loads((campaign_dir / "CAMPAIGN_READY").read_text())
    if set(ready) != {
        "schema_version", "stage", "campaign_id", "campaign_dir",
        "manifest_path", "manifest_sha256",
        "seal_sha256", "preflight_sha256",
    }:
        raise RuntimeError("campaign ready-marker key-set drift")
    if (ready["schema_version"] != SCHEMA_VERSION
            or ready["stage"] != "campaign-preflight"
            or ready["campaign_id"] != campaign_id
            or ready["campaign_dir"] != str(campaign_dir)
            or ready["manifest_path"] != "campaign_manifest.json"
            or ready["manifest_sha256"]
            != sha256_file(campaign_dir / "campaign_manifest.json")
            or ready["seal_sha256"] != expected_manifest["seal_sha256"]
            or ready["preflight_sha256"] != expected_manifest["preflight_sha256"]):
        raise RuntimeError("campaign ready-marker drift")
    selection = json.loads(selection_path.read_text())
    expected_selection = {
        "schema_version": SCHEMA_VERSION,
        "stage": "campaign-selection",
        "campaign_id": campaign_id,
        "campaign_dir": str(campaign_dir),
        "ready_path": f"{campaign_id}/CAMPAIGN_READY",
        "ready_sha256": sha256_file(campaign_dir / "CAMPAIGN_READY"),
        "manifest_sha256": sha256_file(campaign_dir / "campaign_manifest.json"),
        "seal_sha256": ready["seal_sha256"],
        "preflight_sha256": ready["preflight_sha256"],
    }
    if selection != expected_selection:
        raise RuntimeError("campaign selection authority drift")
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "campaign-preflight-reference",
        "campaign_id": campaign_id,
        "campaign_dir": str(campaign_dir),
        "campaign_manifest_sha256": ready["manifest_sha256"],
        "campaign_ready_sha256": sha256_file(campaign_dir / "CAMPAIGN_READY"),
        "campaign_selection_sha256": sha256_file(selection_path),
        "seal_sha256": ready["seal_sha256"],
        "preflight_sha256": ready["preflight_sha256"],
    }


def campaign_input_paths(run_dir: Path) -> list[Path]:
    reference_path = run_dir / "campaign_authority.json"
    if reference_path.is_symlink() or not reference_path.is_file():
        raise RuntimeError("run lacks a safe campaign authority reference")
    reference = json.loads(reference_path.read_text())
    campaign_dir = Path(str(reference.get("campaign_dir", "")))
    paths = [
        reference_path,
        campaign_dir.parent / "CAMPAIGN_SELECTED.json",
        campaign_dir / "seal.json",
        campaign_dir / "preflight.json",
        campaign_dir / "campaign_manifest.json",
        campaign_dir / "CAMPAIGN_READY",
    ]
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"campaign derivation input is missing or unsafe: {path}")
    return paths


def resolve_lens_paths(
    repo: Path, base_lens: Path | None, distill_lens: Path | None
) -> dict[str, Path]:
    supplied = {"base": base_lens, "distill": distill_lens}
    resolved: dict[str, Path] = {}
    for cell, override in supplied.items():
        configured = Path(MODEL_CELLS[cell]["lens_path"]) if override is None else override
        path = configured if configured.is_absolute() else repo / configured
        resolved[cell] = Path(os.path.abspath(path))
    return resolved


def verify_lens_files(lens_paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    for cell, spec in MODEL_CELLS.items():
        path = lens_paths[cell]
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"missing pinned {cell} lens: {path}")
        size = path.stat().st_size
        if size != spec["lens_size_bytes"]:
            raise RuntimeError(
                f"{cell} lens size mismatch: {size} != {spec['lens_size_bytes']}"
            )
        got = sha256_file(path)
        if got != spec["lens_sha256"]:
            raise RuntimeError(f"{cell} lens hash mismatch: {got}")
        verified[cell] = {"path": str(path), "size_bytes": size, "sha256": got}
    return verified


def paired_behavior_token_ids(
    tokenizers: Mapping[str, Any],
    atomic_tasks: Sequence[Mapping[str, Any]],
    orderops_items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Freeze all ordinary no-special-token behavioural input IDs."""
    specifications: list[tuple[str, str, str, str]] = []
    for task in atomic_tasks:
        specifications.append((
            "atomic", str(task["name"]), str(task["arm"]), str(task["input_text"])
        ))
    for item in orderops_items:
        name = str(item["name"])
        prompt = str(item["prompt"])
        specifications.extend((
            ("orderops", name, "direct", prompt),
            ("orderops", name, "cot", ORDEROPS_COT_WRAPPER.format(prompt=prompt)),
            ("orderops", name, "teacher_forced_joint", prompt + str(item["target"])),
        ))
    records: list[dict[str, Any]] = []
    for population, name, arm, text in specifications:
        paired = {
            cell: [
                int(x) for x in tokenizer(text, add_special_tokens=False)["input_ids"]
            ]
            for cell, tokenizer in tokenizers.items()
        }
        if paired["base"] != paired["distill"]:
            raise RuntimeError(f"{population}/{name}/{arm}: ordinary prefix IDs differ")
        if not paired["base"] or len(paired["base"]) > MAX_SEQ_LEN:
            raise RuntimeError(f"{population}/{name}/{arm}: invalid frozen token length")
        records.append({
            "population": population,
            "name": name,
            "arm": arm,
            "token_ids": paired["base"],
        })
    if len(records) != 489:
        raise RuntimeError(f"paired behavioural tokenization count drift: {len(records)}")
    return records


def behavior_token_id_lookup(preflight_record: Mapping[str, Any]) -> dict[tuple[str, str, str], list[int]]:
    rows = preflight_record["tokenizer_only_orderops"]["ordinary_behavior_prefix_ids"]
    lookup: dict[tuple[str, str, str], list[int]] = {}
    for row in rows:
        key = (str(row["population"]), str(row["name"]), str(row["arm"]))
        if key in lookup:
            raise RuntimeError(f"duplicate frozen behavioural tokenization: {key}")
        lookup[key] = [int(x) for x in row["token_ids"]]
    if len(lookup) != 489:
        raise RuntimeError("sealed behavioural tokenization table must have 489 rows")
    return lookup


def tokenizer_only_orderops_preflight(
    repo: Path,
    items: Sequence[Mapping[str, Any]],
    typo_items: Sequence[Mapping[str, Any]] | None = None,
    atomic_tasks: Sequence[Mapping[str, Any]] = (),
    *,
    include_model_weights: bool = False,
    hf_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Run eligibility and joint-boundary checks without importing a model."""
    from transformers import AutoTokenizer

    snapshots = {
        cell: verified_hf_snapshot(
            repo, cell, include_model_weights=include_model_weights,
            cache_dir=hf_cache_dir,
        )
        for cell in MODEL_CELLS
    }
    architecture_receipts = {
        cell: verify_snapshot_architecture(
            path, cell, include_model_weights=include_model_weights
        )
        for cell, (path, _) in snapshots.items()
    }
    tokenizers = {
        cell: AutoTokenizer.from_pretrained(path, local_files_only=True)
        for cell, (path, _) in snapshots.items()
    }
    for cell, tokenizer in tokenizers.items():
        spec = MODEL_CELLS[cell]
        if len(tokenizer) != spec["expected_tokenizer_len"]:
            raise RuntimeError(f"{cell}: tokenizer length drift")
        if tuple(int(x) for x in tokenizer.all_special_ids) != spec["expected_special_ids"]:
            raise RuntimeError(f"{cell}: tokenizer special-ID drift")
        if tokenizer.bos_token_id != spec["expected_bos_token_id"]:
            raise RuntimeError(f"{cell}: tokenizer BOS drift")
        if tokenizer.eos_token_id != 151643 or tokenizer.pad_token_id != 151643:
            raise RuntimeError(f"{cell}: tokenizer EOS/PAD drift")
    domains = {cell: tokenizer_domain(tokenizer, 151936)
               for cell, tokenizer in tokenizers.items()}
    eligibility = derive_common_orderops_eligibility(items, tokenizers, domains)
    typo_eligibility = (
        derive_common_typo_eligibility(typo_items, tokenizers, domains)
        if typo_items is not None else None
    )
    locators: dict[str, dict[str, dict[str, Any]]] = {cell: {} for cell in MODEL_CELLS}
    shortcut_differences: list[str] = []
    max_prefix = 0
    max_joint = 0
    for item in items:
        for cell, tokenizer in tokenizers.items():
            locator = locate_joint_readout(
                tokenizer, item["prompt"], item["target"],
                force_bos=bool(MODEL_CELLS[cell]["effective_force_bos"]),
            )
            locators[cell][str(item["name"])] = locator
            max_prefix = max(max_prefix, len(locator["causal_prefix_token_ids"]))
            max_joint = max(
                max_joint,
                len(locator["joint_token_ids_no_special"]) + int(locator["effective_force_bos"]),
            )
        base = locators["base"][str(item["name"])]
        distill = locators["distill"][str(item["name"])]
        if base["joint_token_ids_no_special"] != distill["joint_token_ids_no_special"]:
            raise RuntimeError(f"{item['name']}: paired joint token IDs differ")
        if base["joint_offsets"] != distill["joint_offsets"]:
            raise RuntimeError(f"{item['name']}: paired joint offsets differ")
        if base["causal_prefix_token_ids"] != distill["causal_prefix_token_ids"][1:]:
            raise RuntimeError(f"{item['name']}: causal prefixes differ beyond registered BOS")
        prompt_ids = tokenizers["base"](item["prompt"], add_special_tokens=False)["input_ids"]
        if int(prompt_ids[-1]) != int(base["predecessor_token_id"]):
            shortcut_differences.append(str(item["name"]))
    expected_shortcut = [
        "word-add-mult", "word-mult-sub", "word-parens", "word-sub-mult",
        "word-add-add", "word-div-sub",
    ]
    if shortcut_differences != expected_shortcut:
        raise RuntimeError(f"prompt-alone shortcut audit drift: {shortcut_differences}")
    if max_joint != 27:
        raise RuntimeError(f"maximum joint-token count drift: {max_joint}")
    typo_locators: dict[str, dict[str, dict[str, Any]]] = {cell: {} for cell in MODEL_CELLS}
    if typo_items is not None:
        for item in typo_items:
            name = str(item["name"])
            for cell, tokenizer in tokenizers.items():
                typo_locators[cell][name] = locate_final_prompt_readout(
                    tokenizer, str(item["prompt"]),
                    force_bos=bool(MODEL_CELLS[cell]["effective_force_bos"]),
                )
            if (typo_locators["base"][name]["causal_prefix_token_ids"]
                    != typo_locators["distill"][name]["causal_prefix_token_ids"][1:]):
                raise RuntimeError(f"{name}: paired typo prefixes differ beyond registered BOS")
    behavior_ids = paired_behavior_token_ids(tokenizers, atomic_tasks, items)
    return {
        "model_loaded": False,
        "model_forward_performed": False,
        "verified_hf_snapshots": {cell: record for cell, (_, record) in snapshots.items()},
        "architecture_receipts": architecture_receipts,
        "domains": domains,
        "eligibility": eligibility,
        "typo_eligibility": typo_eligibility,
        "locators": locators,
        "max_joint_tokens_including_distill_bos": max_joint,
        "max_causal_prefix_tokens_including_distill_bos": max_prefix,
        "paired_joint_ids_identical": True,
        "paired_offsets_identical": True,
        "causal_prefix_ids_identical_except_registered_bos": True,
        "ordinary_behavior_prefix_ids": behavior_ids,
        "ordinary_behavior_prefix_ids_sha256": sha256_bytes(
            canonical_json_bytes(behavior_ids)
        ),
        "typo_locators": typo_locators,
        "prompt_alone_shortcut_differs_n": len(shortcut_differences),
        "prompt_alone_shortcut_differs_items": shortcut_differences,
    }


def fetch_pinned_file(cache_dir: Path, spec: Mapping[str, Any]) -> tuple[Path, bytes]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / str(spec["filename"])
    if path.is_symlink():
        raise RuntimeError(f"pinned-input cache path is a symlink: {path}")
    raw: bytes
    if path.is_file():
        raw = path.read_bytes()
    else:
        with urllib.request.urlopen(f"{JLENS_RAW}/{spec['filename']}", timeout=60) as response:
            raw = response.read()
        if sha256_bytes(raw) != spec["sha256"]:
            raise RuntimeError(f"download hash mismatch for {spec['filename']}")
        atomic_write_bytes(path, raw)
    got = sha256_bytes(raw)
    if got != spec["sha256"]:
        raise RuntimeError(
            f"pinned input hash mismatch for {spec['filename']}: {got} != {spec['sha256']}"
        )
    return path, raw


def load_upstream_inputs(cache_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    out: dict[str, list[dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    for key, spec in UPSTREAM_FILES.items():
        _, raw = fetch_pinned_file(cache_dir, spec)
        obj = json.loads(raw)
        items = obj.get("items")
        if not isinstance(items, list) or len(items) != spec["n_items"]:
            raise RuntimeError(f"{key} population mismatch: expected {spec['n_items']}")
        names = [item.get("name") for item in items]
        if len(names) != len(set(names)) or any(not isinstance(x, str) for x in names):
            raise RuntimeError(f"{key} item names are missing or duplicated")
        for item in items:
            if not isinstance(item.get("prompt"), str):
                raise RuntimeError(f"{key}/{item.get('name')}: malformed prompt")
            if key != "typo" and not isinstance(item.get("target"), str):
                raise RuntimeError(f"{key}/{item.get('name')}: malformed target")
            labels = item.get("intermediates")
            if not isinstance(labels, list) or not labels or not all(isinstance(x, str) for x in labels):
                raise RuntimeError(f"{key}/{item.get('name')}: malformed intermediates")
        out[key] = items
        hashes[str(spec["filename"])] = sha256_bytes(raw)
    for spec in AUX_UPSTREAM_FILES.values():
        _, raw = fetch_pinned_file(cache_dir, spec)
        hashes[str(spec["filename"])] = sha256_bytes(raw)
    validate_orderops_items(out["orderops"])
    return out, hashes


def install_fetched_inputs(run_dir: Path, cache_dir: Path) -> dict[str, str]:
    """Retain exact fetched bytes in the isolated run bundle without overwrite."""
    destination = run_dir / "input_files"
    destination.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for spec in (*UPSTREAM_FILES.values(), *AUX_UPSTREAM_FILES.values()):
        _, raw = fetch_pinned_file(cache_dir, spec)
        path = destination / str(spec["filename"])
        if path.exists():
            if path.read_bytes() != raw:
                raise RuntimeError(f"resume input bytes differ: {path.name}")
        else:
            atomic_write_bytes(path, raw)
        hashes[path.name] = sha256_file(path)
    return hashes


_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def number_to_cardinal(value: int) -> str:
    """Conventional lowercase English cardinal for 0--999, without ``and``."""
    if not 0 <= value <= 999:
        raise ValueError("numeric synonym domain is 0..999")
    if value < 20:
        return _ONES[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_ONES[ones]}"
    hundreds, rem = divmod(value, 100)
    prefix = f"{_ONES[hundreds]} hundred"
    return prefix if rem == 0 else f"{prefix} {number_to_cardinal(rem)}"


def semantic_synonyms(key: str) -> tuple[str, ...]:
    if re.fullmatch(r"(?:0|[1-9][0-9]{0,2})", key):
        return (key, number_to_cardinal(int(key)))
    try:
        return OPERATION_SYNONYMS[key]
    except KeyError as exc:
        raise ValueError(f"unregistered order-ops semantic key: {key!r}") from exc


def validate_orderops_items(items: Sequence[Mapping[str, Any]]) -> None:
    if len(items) != 55:
        raise RuntimeError(f"order-ops must contain 55 items, got {len(items)}")
    keys: set[str] = set()
    for item in items:
        labels = item["intermediates"]
        if len(labels) != 2:
            raise RuntimeError(f"{item['name']}: expected numeric+operation pair")
        numeric = [x for x in labels if x not in OPERATION_KEYS]
        operation = [x for x in labels if x in OPERATION_KEYS]
        if len(numeric) != 1 or len(operation) != 1 or not numeric[0].isdigit():
            raise RuntimeError(f"{item['name']}: invalid numeric+operation pair {labels!r}")
        keys.update(labels)
    if keys != ORDEROPS_EXPECTED_KEYS:
        raise RuntimeError(f"order-ops semantic-key set drift: {sorted(keys)}")
    names = {str(x["name"]) for x in items}
    if not ORDEROPS_LEXICAL_EXCLUSIONS <= names:
        raise RuntimeError("registered prompt-leakage exclusions are absent")
    if not ORDEROPS_ANNOTATION_EXCLUSIONS <= names:
        raise RuntimeError("registered annotation exclusion is absent")
    if len(names - ORDEROPS_LEXICAL_EXCLUSIONS) != 52:
        raise RuntimeError("non-lexically-present task population must contain 52 items")
    if len(names - ORDEROPS_PRIMARY_EXCLUSIONS) != 51:
        raise RuntimeError("annotation-valid pre-token primary must contain 51 items")
    mismatch = next(item for item in items if item["name"] == "mult-div-mult")
    if (mismatch["prompt"] != "2 * 6 / 4 * 3 = "
            or mismatch["target"] != "9"
            or mismatch["intermediates"] != ["12", "multiplication"]):
        raise RuntimeError("registered mult-div-mult annotation exception drift")


def tokenizer_domain(tokenizer: Any, head_vocab_size: int) -> dict[str, int]:
    tok_len = int(len(tokenizer))
    scored = min(tok_len, int(head_vocab_size))
    if scored <= K:
        raise RuntimeError("ranked tokenizer vocabulary is unexpectedly small")
    return {
        "tokenizer_len": tok_len,
        "head_vocab_size": int(head_vocab_size),
        "ranked_vocab_size": scored,
    }


def eligible_synonym_ids(tokenizer: Any, key: str, common_vocab_size: int) -> dict[str, int]:
    """Return one-token, non-special variants; each is prefixed by one ASCII space."""
    special_ids = {int(x) for x in (getattr(tokenizer, "all_special_ids", None) or [])}
    eligible: dict[str, int] = {}
    seen_token_ids: set[int] = set()
    for variant in semantic_synonyms(key):
        scored = " " + variant
        encoded = tokenizer(scored, add_special_tokens=False)["input_ids"]
        ids = encoded[0] if encoded and isinstance(encoded[0], list) else encoded
        if len(ids) == 1:
            token_id = int(ids[0])
            if (0 <= token_id < common_vocab_size and token_id not in special_ids
                    and token_id not in seen_token_ids):
                eligible[variant] = token_id
                seen_token_ids.add(token_id)
    return eligible


def derive_common_orderops_eligibility(
    items: Sequence[Mapping[str, Any]],
    tokenizers: Mapping[str, Any],
    domains: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Freeze model-specific synonym IDs under one common paired rank domain."""
    if set(tokenizers) != set(MODEL_CELLS):
        raise RuntimeError("eligibility requires exactly the base and distill tokenizers")
    common_vocab = min(int(domains[c]["ranked_vocab_size"]) for c in MODEL_CELLS)
    keys = sorted({key for item in items for key in item["intermediates"]})
    by_cell: dict[str, dict[str, dict[str, int]]] = {}
    for cell, tokenizer in tokenizers.items():
        by_cell[cell] = {key: eligible_synonym_ids(tokenizer, key, common_vocab) for key in keys}
    # Freeze the same variant-string set in both cells, while retaining each
    # tokenizer's own token ID for those strings.
    common_variants: dict[str, list[str]] = {}
    for key in keys:
        variants = set.intersection(*(set(by_cell[cell][key]) for cell in MODEL_CELLS))
        common_variants[key] = [x for x in semantic_synonyms(key) if x in variants]
        for cell in MODEL_CELLS:
            by_cell[cell][key] = {
                variant: by_cell[cell][key][variant] for variant in common_variants[key]
            }
    common_keys = [key for key in keys if common_variants[key]]
    if any(key not in common_keys for key in OPERATION_KEYS):
        raise RuntimeError("an operation control lacks a paired one-token synonym")
    retained = [
        str(item["name"])
        for item in items
        if all(key in common_keys for key in item["intermediates"])
    ]
    excluded = [str(item["name"]) for item in items if str(item["name"]) not in retained]
    if excluded != ["mult-div-left"] or len(retained) != 54:
        raise RuntimeError(f"unexpected paired eligibility exclusions: {excluded}")
    analysis_names = [x for x in retained if x not in ORDEROPS_ANNOTATION_EXCLUSIONS]
    if len(analysis_names) != 53:
        raise RuntimeError(
            f"annotation-valid paired sensitivity must contain 53 items, got {len(analysis_names)}"
        )
    primary_names = [x for x in retained if x not in ORDEROPS_PRIMARY_EXCLUSIONS]
    if len(primary_names) != 50:
        raise RuntimeError(
            f"annotation-valid non-lexically-present primary must contain 50 items, got {len(primary_names)}"
        )
    return {
        "common_ranked_vocab_size": common_vocab,
        "semantic_keys": common_keys,
        "common_variant_strings_by_key": {key: common_variants[key] for key in common_keys},
        "token_ids_by_cell": {
            cell: {key: by_cell[cell][key] for key in common_keys} for cell in MODEL_CELLS
        },
        "retained_item_names": retained,
        "n_retained": len(retained),
        "analysis_item_names": analysis_names,
        "n_analysis": len(analysis_names),
        "primary_item_names": primary_names,
        "n_primary": len(primary_names),
        "common_token_exclusions": excluded,
        "prompt_copy_exclusions": sorted(ORDEROPS_LEXICAL_EXCLUSIONS),
        "annotation_exclusions": sorted(ORDEROPS_ANNOTATION_EXCLUSIONS),
        "primary_exclusions": sorted(ORDEROPS_PRIMARY_EXCLUSIONS),
    }


def derive_common_typo_eligibility(
    items: Sequence[Mapping[str, Any]],
    tokenizers: Mapping[str, Any],
    domains: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Freeze the 96 canonical typo labels under the paired vocabulary."""
    if len(items) != 96:
        raise RuntimeError("same-runtime typo control requires 96 items")
    common_vocab = min(int(domains[cell]["ranked_vocab_size"]) for cell in MODEL_CELLS)
    keys = sorted({str(item["intermediates"][0]) for item in items})
    if any(len(item["intermediates"]) != 1 for item in items):
        raise RuntimeError("typo control expects exactly one label per item")
    by_cell: dict[str, dict[str, dict[str, int]]] = {cell: {} for cell in MODEL_CELLS}
    for cell, tokenizer in tokenizers.items():
        specials = {int(x) for x in tokenizer.all_special_ids}
        for key in keys:
            ids = tokenizer(" " + key, add_special_tokens=False)["input_ids"]
            token_ids = ids[0] if ids and isinstance(ids[0], list) else ids
            eligible = (
                len(token_ids) == 1 and 0 <= int(token_ids[0]) < common_vocab
                and int(token_ids[0]) not in specials
            )
            by_cell[cell][key] = {key: int(token_ids[0])} if eligible else {}
    missing = {cell: [key for key in keys if not by_cell[cell][key]] for cell in MODEL_CELLS}
    if any(missing.values()):
        raise RuntimeError(f"typo common eligibility drift: {missing}")
    return {
        "common_ranked_vocab_size": common_vocab,
        "semantic_keys": keys,
        "token_ids_by_cell": by_cell,
        "retained_item_names": [str(item["name"]) for item in items],
        "n_retained": len(items),
    }


def locate_final_prompt_readout(tokenizer: Any, prompt: str, *, force_bos: bool) -> dict[str, Any]:
    encoded = tokenizer(prompt, add_special_tokens=False)
    ids_raw = encoded["input_ids"]
    ids = ids_raw[0] if ids_raw and isinstance(ids_raw[0], list) else ids_raw
    ids = [int(x) for x in ids]
    if not ids:
        raise RuntimeError("empty prompt tokenization")
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if force_bos:
        if bos_id is None:
            raise RuntimeError("forced-BOS prompt has no BOS token")
        ids = [int(bos_id), *ids]
    if len(ids) > MAX_SEQ_LEN:
        raise RuntimeError("prompt exceeds fixed max sequence length")
    return {
        "prompt_utf8_sha256": sha256_bytes(prompt.encode("utf-8")),
        "causal_prefix_token_ids": ids,
        "readout_index": len(ids) - 1,
        "readout_token_id": ids[-1],
        "effective_force_bos": bool(force_bos),
        "effective_bos_token_id": int(bos_id) if force_bos else None,
    }


def locate_joint_readout(tokenizer: Any, prompt: str, target: str, *, force_bos: bool) -> dict[str, Any]:
    """Locate the official predecessor using joint prompt+target tokenization.

    The first token whose offset overlaps the target character interval is the
    first target token.  The causal prefix ends immediately before it.  A BOS
    is prepended only when the fitted lens's effective encoding did so.
    """
    joint = prompt + target
    encoded = tokenizer(
        joint,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids_raw = encoded["input_ids"]
    offsets_raw = encoded["offset_mapping"]
    ids = ids_raw[0] if ids_raw and isinstance(ids_raw[0], list) else ids_raw
    offsets = offsets_raw[0] if offsets_raw and isinstance(offsets_raw[0], list) and offsets_raw[0] and isinstance(offsets_raw[0][0], (list, tuple)) else offsets_raw
    ids = [int(x) for x in ids]
    offsets = [(int(a), int(b)) for a, b in offsets]
    if len(ids) != len(offsets):
        raise RuntimeError("joint token IDs and offsets have different lengths")
    target_start, target_end = len(prompt), len(joint)
    first_target = next(
        (idx for idx, (start, end) in enumerate(offsets)
         if end > target_start and start < target_end),
        None,
    )
    if first_target is None or first_target == 0:
        raise RuntimeError("could not locate a predecessor token before the target")
    prefix = ids[:first_target]
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if force_bos:
        if bos_id is None:
            raise RuntimeError("effective_force_bos is true but tokenizer has no BOS ID")
        if not prefix or prefix[0] != int(bos_id):
            prefix = [int(bos_id), *prefix]
    elif bos_id is not None and prefix and prefix[0] == int(bos_id):
        raise RuntimeError("unexpected BOS in add_special_tokens=False joint tokenization")
    if len(prefix) > MAX_SEQ_LEN:
        raise RuntimeError(f"joint causal prefix exceeds max length {MAX_SEQ_LEN}")
    predecessor = first_target - 1
    return {
        "joint_utf8_sha256": sha256_bytes(joint.encode("utf-8")),
        "target_character_interval": [target_start, target_end],
        "joint_token_ids_no_special": ids,
        "joint_offsets": [[a, b] for a, b in offsets],
        "first_target_token_index_no_special": first_target,
        "predecessor_token_index_no_special": predecessor,
        "predecessor_token_id": ids[predecessor],
        "predecessor_offset": list(offsets[predecessor]),
        "causal_prefix_token_ids": prefix,
        "effective_force_bos": bool(force_bos),
        "effective_bos_token_id": int(bos_id) if force_bos else None,
    }


def deterministic_exact_ranks(logits: np.ndarray, token_ids: Sequence[int]) -> np.ndarray:
    """Exact 1-based ranks: descending logit, then ascending token ID on ties."""
    values = np.asarray(logits)
    if values.ndim < 1:
        raise ValueError("logits must have a vocabulary axis")
    vocab = values.shape[-1]
    ids = np.asarray(token_ids, dtype=np.int64)
    if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= vocab):
        raise ValueError("token IDs fall outside the logit vocabulary")
    flat = values.reshape(-1, vocab)
    ranks = np.empty((flat.shape[0], len(ids)), dtype=np.int64)
    vocabulary_ids = np.arange(vocab)
    for col, token_id in enumerate(ids.tolist()):
        target = flat[:, token_id][:, None]
        before = (flat > target) | ((flat == target) & (vocabulary_ids[None, :] < token_id))
        ranks[:, col] = 1 + before.sum(axis=1, dtype=np.int64)
    return ranks.reshape(*values.shape[:-1], len(ids))


def semantic_rank_bank_numpy(
    logits: np.ndarray,
    keys: Sequence[str],
    token_ids_by_key: Mapping[str, Mapping[str, int]],
) -> np.ndarray:
    rows = []
    for key in keys:
        token_ids = list(token_ids_by_key[key].values())
        synonym_ranks = deterministic_exact_ranks(logits, token_ids)
        rows.append(synonym_ranks.min(axis=-1))
    return np.stack(rows, axis=-1).astype(np.int32)


def per_item_layer_persistence_at_25(ranks: np.ndarray, layers: Sequence[int]) -> np.ndarray:
    """Fraction of selected source layers at which the item has rank <=25."""
    array = np.asarray(ranks)
    if array.ndim != 2:
        raise ValueError("ranks must be [items, source_layers]")
    layer_ids = np.asarray(layers, dtype=np.int64)
    if np.any(layer_ids < 0) or np.any(layer_ids >= array.shape[1]):
        raise ValueError("layer index outside rank array")
    return (array[:, layer_ids] <= K).mean(axis=1)


def atomic_supportive_bootstrap(
    hb1: np.ndarray,
    hb2: np.ndarray,
    hd1: np.ndarray,
    hd2: np.ndarray,
    mb: np.ndarray,
    md: np.ndarray,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = ATOMIC_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap state proportions and the derived common-known subset."""
    vectors = [np.asarray(x, dtype=np.int8) for x in (hb1, hb2, hd1, hd2, mb, md)]
    if any(x.shape != vectors[0].shape or x.ndim != 1 for x in vectors):
        raise ValueError("atomic supportive vectors must be equal-length 1D arrays")
    if any(not np.isin(x, [0, 1]).all() for x in vectors):
        raise ValueError("atomic supportive vectors must be binary")
    hb1, hb2, hd1, hd2, mb, md = vectors
    kb, kd = hb1 * hb2, hd1 * hd2
    states = {
        "base": {
            "both": (hb1 == 1) & (hb2 == 1),
            "hop1_only": (hb1 == 1) & (hb2 == 0),
            "hop2_only": (hb1 == 0) & (hb2 == 1),
            "neither": (hb1 == 0) & (hb2 == 0),
        },
        "distill": {
            "both": (hd1 == 1) & (hd2 == 1),
            "hop1_only": (hd1 == 1) & (hd2 == 0),
            "hop2_only": (hd1 == 0) & (hd2 == 1),
            "neither": (hd1 == 0) & (hd2 == 0),
        },
    }
    rng = np.random.Generator(np.random.PCG64(seed))
    state_draws = {
        cell: {name: [] for name in table} for cell, table in states.items()
    }
    common_n: list[np.ndarray] = []
    common_base: list[np.ndarray] = []
    common_distill: list[np.ndarray] = []
    common_rd: list[np.ndarray] = []
    remaining = n_bootstrap
    while remaining:
        size = min(2_000, remaining)
        indices = rng.integers(0, len(kb), size=(size, len(kb)))
        for cell, table in states.items():
            for name, mask in table.items():
                state_draws[cell][name].append(mask[indices].mean(axis=1))
        common = (kb[indices] == 1) & (kd[indices] == 1)
        n_common = common.sum(axis=1)
        common_n.append(n_common.astype(float))
        denominator = np.where(n_common > 0, n_common, np.nan)
        base_rate = (mb[indices] * common).sum(axis=1) / denominator
        distill_rate = (md[indices] * common).sum(axis=1) / denominator
        common_base.append(base_rate)
        common_distill.append(distill_rate)
        common_rd.append(distill_rate - base_rate)
        remaining -= size

    def interval(draws: Sequence[np.ndarray]) -> tuple[list[float] | None, int]:
        values = np.concatenate(draws)
        finite = values[np.isfinite(values)]
        invalid = int(len(values) - len(finite))
        if invalid:
            return None, invalid
        low, high = np.quantile(finite, [0.025, 0.975], method="linear")
        return [float(low), float(high)], 0

    state_output: dict[str, Any] = {}
    for cell, table in states.items():
        state_output[cell] = {}
        for name, mask in table.items():
            ci, invalid = interval(state_draws[cell][name])
            if invalid:
                raise RuntimeError("atomic state bootstrap unexpectedly produced non-finite draws")
            state_output[cell][name] = {
                "count": int(mask.sum()),
                "proportion": float(mask.mean()),
                "percentile_95_ci": ci,
            }
    state_output["paired_distill_minus_base"] = {}
    for name in states["base"]:
        base_draws = np.concatenate(state_draws["base"][name])
        distill_draws = np.concatenate(state_draws["distill"][name])
        difference_draws = [distill_draws - base_draws]
        ci, invalid = interval(difference_draws)
        if invalid:
            raise RuntimeError("atomic state-difference bootstrap produced non-finite draws")
        state_output["paired_distill_minus_base"][name] = {
            "estimate": float(
                states["distill"][name].mean() - states["base"][name].mean()
            ),
            "percentile_95_ci": ci,
        }
    observed_common = (kb == 1) & (kd == 1)
    common_output: dict[str, Any] = {
        "n_items": int(observed_common.sum()),
        "shown": bool(observed_common.sum() >= 20),
        "bootstrap_n": int(n_bootstrap),
        "bootstrap_seed": int(seed),
        "derived_subset_recomputed_within_each_draw": True,
        "n_items_percentile_95_ci": interval(common_n)[0],
    }
    if common_output["shown"]:
        base_ci, empty_base = interval(common_base)
        distill_ci, empty_distill = interval(common_distill)
        difference_ci, empty_difference = interval(common_rd)
        empty_draws = max(empty_base, empty_distill, empty_difference)
        common_output.update({
            "base_composite_rate": float(mb[observed_common].mean()),
            "base_composite_rate_percentile_95_ci": base_ci,
            "distill_composite_rate": float(md[observed_common].mean()),
            "distill_composite_rate_percentile_95_ci": distill_ci,
            "distill_minus_base_risk_difference": float(
                (md[observed_common] - mb[observed_common]).mean()
            ),
            "distill_minus_base_risk_difference_percentile_95_ci": difference_ci,
            "conditional_intervals_available": empty_draws == 0,
            "empty_recomputed_common_known_draws": empty_draws,
            "empty_draw_rule": "withhold all conditional intervals; no redraw, deletion, or imputation",
        })
    return {"atomic_state": state_output, "common_known_composite": common_output}


def paired_bootstrap_mean_difference(
    first: Sequence[float],
    second: Sequence[float],
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 2_000,
) -> dict[str, Any]:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError("paired vectors must be equal-length one-dimensional arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired vectors contain non-finite values")
    rng = np.random.Generator(np.random.PCG64(seed))
    delta = a - b
    draws: list[np.ndarray] = []
    remaining = n_bootstrap
    while remaining:
        size = min(chunk_size, remaining)
        indices = rng.integers(0, len(delta), size=(size, len(delta)))
        draws.append(delta[indices].mean(axis=1))
        remaining -= size
    distribution = np.concatenate(draws)
    low90, low95, high95, high90 = np.quantile(
        distribution, [0.05, 0.025, 0.975, 0.95], method="linear"
    )
    return {
        "n_items": len(delta),
        "estimand": "paired mean(first - second)",
        "estimate": float(delta.mean()),
        "bootstrap_n": int(n_bootstrap),
        "bootstrap_seed": int(seed),
        "percentile_95_ci": [float(low95), float(high95)],
        "percentile_90_ci": [float(low90), float(high90)],
    }


def checkpoint_vector_swap_test(
    differences: Sequence[float],
    *,
    n_swaps: int = N_CHECKPOINT_SWAPS,
    seed: int = SWAP_SEED,
    chunk_size: int = 4_000,
) -> dict[str, Any]:
    """Paired randomization test swapping the whole checkpoint vector by item."""
    delta = np.asarray(differences, dtype=np.float64)
    if delta.ndim != 1 or len(delta) < 2 or not np.isfinite(delta).all():
        raise ValueError("differences must be a finite one-dimensional item vector")
    observed = float(delta.mean())
    rng = np.random.Generator(np.random.PCG64(seed))
    extreme = 0
    remaining = n_swaps
    tolerance = 1e-15
    while remaining:
        size = min(chunk_size, remaining)
        signs = rng.integers(0, 2, size=(size, len(delta)), dtype=np.int8) * 2 - 1
        null = (signs * delta).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(null) + tolerance >= abs(observed)))
        remaining -= size
    return {
        "n_items": len(delta),
        "estimand": "paired mean checkpoint-vector difference",
        "estimate": observed,
        "n_random_swaps": int(n_swaps),
        "seed": int(seed),
        "two_sided_p": float((extreme + 1) / (n_swaps + 1)),
        "plus_one_correction": True,
    }


def _gather_assigned_key_ranks(rank_bank: np.ndarray, assigned_keys: np.ndarray) -> np.ndarray:
    """Gather [batch,item,layer,component] ranks for assigned whole label pairs."""
    bank = np.asarray(rank_bank)
    assigned = np.asarray(assigned_keys, dtype=np.int64)
    if bank.ndim != 3 or assigned.ndim != 3 or assigned.shape[1] != bank.shape[0]:
        raise ValueError("rank bank or assigned label-pair shape mismatch")
    # Advanced indexing returns [batch,item,component,layer]; transpose once.
    gathered = bank[np.arange(bank.shape[0])[None, :, None], :, assigned]
    return np.transpose(gathered, (0, 1, 3, 2))


def synchronized_whole_pair_permutation_test(
    rank_banks: Mapping[str, np.ndarray],
    label_pairs: np.ndarray,
    *,
    prompt_mask: np.ndarray | None = None,
    layers: Sequence[int] = SOURCE_BANDS["mid"],
    n_permutations: int = N_LABEL_PERMUTATIONS,
    seed: int = LABEL_PERMUTATION_SEED,
    chunk_size: int = 512,
) -> dict[str, Any]:
    """Permute intact numeric+operation pairs synchronously across all banks.

    ``label_pairs`` has one row per prompt and keeps duplicate rows.  Each
    random permutation shuffles row indices rather than unique values, so the
    duplicate-pair multiplicities are preserved exactly.  Component 0 is the
    non-lexically-present numeric endpoint; component 1 is the prompt-present operation
    positive control.
    """
    if not rank_banks:
        raise ValueError("at least one rank bank is required")
    pairs = np.asarray(label_pairs, dtype=np.int64)
    n_items = pairs.shape[0]
    if pairs.shape != (n_items, 2):
        raise ValueError("label_pairs must be [items,2]")
    for name, bank in rank_banks.items():
        if np.asarray(bank).ndim != 3 or np.asarray(bank).shape[0] != n_items:
            raise ValueError(f"rank bank {name!r} has incompatible shape")
    mask = np.ones(n_items, dtype=bool) if prompt_mask is None else np.asarray(prompt_mask, dtype=bool)
    if mask.shape != (n_items,) or mask.sum() < 2:
        raise ValueError("prompt mask must retain at least two items")
    layer_ids = np.asarray(layers, dtype=np.int64)
    actual_assigned = pairs[None, :, :]
    observed: dict[str, list[float]] = {}
    for name, bank in rank_banks.items():
        hits = _gather_assigned_key_ranks(bank, actual_assigned)[0, mask][:, layer_ids] <= K
        observed[name] = [float(hits[..., c].mean()) for c in range(2)]

    rng = np.random.Generator(np.random.PCG64(seed))
    extreme = {name: np.zeros(2, dtype=np.int64) for name in rank_banks}
    remaining = n_permutations
    while remaining:
        size = min(chunk_size, remaining)
        # Sorting independent random keys yields synchronized row permutations.
        permutations = np.argsort(rng.random((size, n_items)), axis=1, kind="stable")
        assigned = pairs[permutations]
        for name, bank in rank_banks.items():
            hits = _gather_assigned_key_ranks(bank, assigned)[:, mask][:, :, layer_ids] <= K
            null = hits.mean(axis=(1, 2))  # [batch, numeric/operation]
            for component in range(2):
                extreme[name][component] += int(
                    np.count_nonzero(null[:, component] + 1e-15 >= observed[name][component])
                )
        remaining -= size
    return {
        "n_prompts": int(mask.sum()),
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "whole_label_pairs_preserved": True,
        "duplicates_preserved": True,
        "synchronized_across_banks": True,
        "components": [
            "numeric_non_lexically_present_endpoint",
            "operation_context_positive_control",
        ],
        "banks": {
            name: {
                "observed": observed[name],
                "empirical_upper_p": [
                    float((int(extreme[name][c]) + 1) / (n_permutations + 1))
                    for c in range(2)
                ],
            }
            for name in rank_banks
        },
    }


def synchronized_numeric_assay_gate(
    base_j_rank_bank: np.ndarray,
    distill_j_rank_bank: np.ndarray,
    label_pairs: np.ndarray,
    *,
    layers: Sequence[int] = SOURCE_BANDS["mid"],
    n_permutations: int = N_LABEL_PERMUTATIONS,
    seed: int = LABEL_PERMUTATION_SEED,
    chunk_size: int = 512,
) -> dict[str, Any]:
    """Registered O1 gate on the exact paired complete-case primary subset."""
    base = np.asarray(base_j_rank_bank)
    distill = np.asarray(distill_j_rank_bank)
    pairs = np.asarray(label_pairs, dtype=np.int64)
    if base.shape != distill.shape or base.ndim != 3:
        raise ValueError("paired J rank banks must share [item,layer,key] shape")
    if pairs.shape != (base.shape[0], 2) or base.shape[0] < 2:
        raise ValueError("label pairs do not match rank banks")
    layer_ids = np.asarray(layers, dtype=np.int64)

    secondary_bands = {**SOURCE_BANDS, "all": SOURCE_LAYERS}

    def statistics_for_assignment(assigned: np.ndarray) -> np.ndarray:
        b_all = _gather_assigned_key_ranks(base, assigned)[..., 0] <= K
        d_all = _gather_assigned_key_ranks(distill, assigned)[..., 0] <= K
        persistence = (
            b_all[:, :, layer_ids].mean(axis=(1, 2))
            + d_all[:, :, layer_ids].mean(axis=(1, 2))
        ) / 2.0
        columns = [persistence]
        for band_layers in secondary_bands.values():
            indices = np.asarray(band_layers, dtype=np.int64)
            union = (
                b_all[:, :, indices].any(axis=2).mean(axis=1)
                + d_all[:, :, indices].any(axis=2).mean(axis=1)
            ) / 2.0
            columns.append(union)
        return np.stack(columns, axis=1)

    observed_vector = statistics_for_assignment(pairs[None, :, :])[0]
    observed = float(observed_vector[0])
    rng = np.random.Generator(np.random.PCG64(seed))
    null_parts: list[np.ndarray] = []
    remaining = n_permutations
    while remaining:
        size = min(chunk_size, remaining)
        permutations = np.argsort(
            rng.random((size, len(pairs))), axis=1, kind="stable"
        )
        null_parts.append(statistics_for_assignment(pairs[permutations]))
        remaining -= size
    null = np.concatenate(null_parts)
    primary_null = null[:, 0]
    q95 = float(np.quantile(primary_null, 0.95, method="higher"))
    p_upper = float(
        (1 + np.count_nonzero(primary_null >= observed)) / (len(primary_null) + 1)
    )
    secondary: dict[str, Any] = {}
    for index, band in enumerate(secondary_bands, start=1):
        values = null[:, index]
        obs = float(observed_vector[index])
        secondary[band] = {
            "observed_mean_checkpoint_averaged_band_union": obs,
            "null_p95_higher": float(np.quantile(values, 0.95, method="higher")),
            "empirical_upper_p_plus_one": float(
                (1 + np.count_nonzero(values >= obs)) / (len(values) + 1)
            ),
        }
    return {
        "n_items": int(len(pairs)),
        "population": "paired technically complete non-lexically-present numeric primary",
        "statistic": "mean_i[(A_J_base + A_J_distill)/2]",
        "observed": observed,
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "null_p95_higher": q95,
        "empirical_upper_p_plus_one": p_upper,
        "exceeds_null_p95": bool(observed > q95),
        "p_at_most_0_05": bool(p_upper <= 0.05),
        "pass": bool(observed > q95 and p_upper <= 0.05),
        "whole_numeric_families_permuted": True,
        "duplicate_families_preserved": True,
        "assignment_synchronized_across_checkpoints_layers": True,
        "null_mean": float(primary_null.mean()),
        "null_sd": float(primary_null.std(ddof=1)),
        "secondary_band_union_nulls": secondary,
    }


def synchronized_typo_union_null(
    rank_banks: Mapping[str, np.ndarray],
    label_indices: Sequence[int],
    *,
    n_permutations: int = N_LABEL_PERMUTATIONS,
    seed: int = TYPO_LABEL_PERMUTATION_SEED,
    chunk_size: int = 512,
) -> dict[str, Any]:
    """Same item-label assignments for the paired typo runtime controls."""
    labels = np.asarray(label_indices, dtype=np.int64)
    if set(rank_banks) != set(MODEL_CELLS):
        raise ValueError("typo null requires base and distill rank banks")
    n_items = len(labels)
    for cell, bank in rank_banks.items():
        if np.asarray(bank).ndim != 3 or np.asarray(bank).shape[0] != n_items:
            raise ValueError(f"{cell}: typo rank-bank shape mismatch")

    def assigned_union(bank: np.ndarray, assigned: np.ndarray) -> np.ndarray:
        # assigned [batch,item]; gathered [batch,item,layer]
        gathered = bank[np.arange(n_items)[None, :], :, assigned]
        return (gathered <= K).any(axis=2).mean(axis=1)

    observed = {
        cell: float(assigned_union(np.asarray(bank), labels[None, :])[0])
        for cell, bank in rank_banks.items()
    }
    rng = np.random.Generator(np.random.PCG64(seed))
    null_parts = {cell: [] for cell in MODEL_CELLS}
    remaining = n_permutations
    while remaining:
        size = min(chunk_size, remaining)
        permutations = np.argsort(rng.random((size, n_items)), axis=1, kind="stable")
        assigned = labels[permutations]
        for cell, bank in rank_banks.items():
            null_parts[cell].append(assigned_union(np.asarray(bank), assigned))
        remaining -= size
    cells: dict[str, Any] = {}
    for cell in MODEL_CELLS:
        null = np.concatenate(null_parts[cell])
        q95 = float(np.quantile(null, 0.95, method="higher"))
        p = float((1 + np.count_nonzero(null >= observed[cell])) / (len(null) + 1))
        passed = observed[cell] >= 0.75 and observed[cell] > q95 and p <= 0.05
        cells[cell] = {
            "n_items": n_items,
            "any_layer_union_pass_at_25": observed[cell],
            "null_p95_higher": q95,
            "empirical_upper_p_plus_one": p,
            "at_least_0_75": bool(observed[cell] >= 0.75),
            "above_null_p95": bool(observed[cell] > q95),
            "p_at_most_0_05": bool(p <= 0.05),
            "pass": bool(passed),
        }
    return {
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "assignments_synchronized_across_checkpoints": True,
        "duplicate_labels_preserved": True,
        "cells": cells,
        "integrity_pass": bool(all(x["pass"] for x in cells.values())),
    }


def normalize_atomic_answer(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.splitlines()[0] if normalized.splitlines() else ""
    normalized = re.sub(r"\s+", " ", normalized).strip()
    while True:
        prior = normalized
        normalized = normalized.rstrip(" .,:;!?").strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] \
                and normalized[0] in ('"', "'", "`"):
            normalized = normalized[1:-1].strip()
        if normalized == prior:
            break
    return normalized.casefold()


_CARDINAL_TO_DIGIT = {number_to_cardinal(i): str(i) for i in range(1000)}


def atomic_answer_aliases(labels: Iterable[str]) -> tuple[str, ...]:
    """Normalize exactly the presealed aliases; never expand them after the seal."""
    return tuple(sorted({normalize_atomic_answer(str(label)) for label in labels}))


def score_atomic_answer(text: str, labels: Iterable[str]) -> dict[str, Any]:
    answer = normalize_atomic_answer(text)
    aliases = atomic_answer_aliases(labels)
    exact = answer in aliases
    boundary = False
    substring = False
    for alias in aliases:
        if not alias:
            continue
        pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
        if re.search(pattern, answer):
            boundary = True
        if alias in answer:
            substring = True
    return {
        "normalized_answer": answer,
        "normalized_aliases": list(aliases),
        "exact_primary": bool(exact),
        "whole_term_sensitivity": bool(boundary),
        "substring_sensitivity": bool(substring),
    }


def orderops_target_aliases(target: str) -> tuple[str, ...]:
    normalized = normalize_atomic_answer(target)
    if normalized.isdigit() and 0 <= int(normalized) <= 999:
        return tuple(sorted({normalized, number_to_cardinal(int(normalized))}))
    if normalized in _CARDINAL_TO_DIGIT:
        return tuple(sorted({normalized, _CARDINAL_TO_DIGIT[normalized]}))
    return (normalized,)


def _normalize_cot_segment(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text.replace("\r\n", "\n"))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    while True:
        prior = normalized
        normalized = normalized.rstrip(" .,:;!?").strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] \
                and normalized[0] in ('"', "'", "`"):
            normalized = normalized[1:-1].strip()
        if normalized == prior:
            break
    return normalized.casefold()


def score_orderops_cot(text: str, aliases: Iterable[str]) -> dict[str, Any]:
    canonical_aliases = tuple(sorted({_normalize_cot_segment(x) for x in aliases}))
    normalized_crlf = text.replace("\r\n", "\n")
    nonempty = [line for line in normalized_crlf.split("\n") if line.strip()]
    final_line = _normalize_cot_segment(nonempty[-1] if nonempty else "")
    entire = _normalize_cot_segment(normalized_crlf)

    def bounded(haystack: str) -> bool:
        return any(
            bool(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", haystack))
            for alias in canonical_aliases if alias
        )

    return {
        "normalized_final_nonempty_line": final_line,
        "normalized_entire_continuation": entire,
        "normalized_aliases": list(canonical_aliases),
        "bounded_final_line_primary": bounded(final_line),
        "bounded_anywhere_sensitivity": bounded(entire),
    }


def force_incorrect_on_token_cap(
    score: Mapping[str, Any], *, reached_token_cap: bool
) -> dict[str, Any]:
    """Apply the sealed rule that a capped behavioural continuation is incorrect."""
    result = dict(score)
    if reached_token_cap:
        for key in (
            "exact_primary", "whole_term_sensitivity", "substring_sensitivity",
            "bounded_final_line_primary", "bounded_anywhere_sensitivity",
        ):
            if key in result:
                result[key] = False
        result["forced_incorrect_reason"] = "generation_token_cap"
    return result


def validate_generation_stop_state(
    generated_ids: Sequence[int],
    *,
    stopped_on_eos: bool,
    stopped_on_lf: bool,
    reached_token_cap: bool,
    max_new_tokens: int,
    context: str,
) -> None:
    """Validate mutually exclusive logical stop reasons for stored continuations."""
    length = len(generated_ids)
    if stopped_on_eos:
        if stopped_on_lf or reached_token_cap or length >= max_new_tokens:
            raise RuntimeError(f"{context}: impossible EOS stop state")
    elif stopped_on_lf:
        if reached_token_cap or length > max_new_tokens:
            raise RuntimeError(f"{context}: impossible line-feed stop state")
    elif not reached_token_cap or length != max_new_tokens:
        raise RuntimeError(f"{context}: uncensored continuation lacks token-cap state")


def joint_target_tokenization(tokenizer: Any, prompt: str, target: str) -> dict[str, Any]:
    joint = prompt + target
    encoded = tokenizer(
        joint, add_special_tokens=False, return_offsets_mapping=True
    )
    ids_raw, offsets_raw = encoded["input_ids"], encoded["offset_mapping"]
    ids = ids_raw[0] if ids_raw and isinstance(ids_raw[0], list) else ids_raw
    offsets = offsets_raw[0] if offsets_raw and isinstance(offsets_raw[0], list) and offsets_raw[0] and isinstance(offsets_raw[0][0], (list, tuple)) else offsets_raw
    ids = [int(x) for x in ids]
    offsets = [(int(a), int(b)) for a, b in offsets]
    start, end = len(prompt), len(joint)
    target_positions = [
        index for index, (left, right) in enumerate(offsets)
        if right > start and left < end
    ]
    if not target_positions or target_positions[0] == 0:
        raise RuntimeError("teacher-forced target lacks a causal predecessor")
    if len(ids) > MAX_SEQ_LEN:
        raise RuntimeError("teacher-forced joint sequence exceeds max length")
    return {
        "joint_token_ids": ids,
        "joint_offsets": [[a, b] for a, b in offsets],
        "target_token_positions": target_positions,
        "target_token_ids": [ids[index] for index in target_positions],
        "target_character_interval": [start, end],
        "manual_bos": False,
    }


def load_frozen_atomic_tasks(
    repo: Path,
    multihop_items: Sequence[Mapping[str, Any]],
    eligibility_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate and expand the sealed, manually authored 81-item task file."""
    task_path = repo / ATOMIC_TASK_FILE
    got = sha256_file(task_path)
    if got != ATOMIC_TASK_SHA256:
        raise RuntimeError(f"frozen atomic task hash mismatch: {got}")
    frozen = json.loads(task_path.read_text())
    if frozen.get("wrapper") != ATOMIC_WRAPPER:
        raise RuntimeError("parsed frozen atomic wrapper differs from the protocol string")
    rows = frozen.get("items")
    if not isinstance(rows, list) or len(rows) != 81:
        raise RuntimeError("frozen atomic task file must contain 81 items")

    eligible_rows = [
        x for x in eligibility_manifest["evaluations"]["lens-eval-multihop"]["items"]
        if x.get("item_eligible") and x.get("n_eligible_labels", 0) > 0
    ]
    if len(eligible_rows) != 81:
        raise RuntimeError("source eligibility no longer contains 81 atomic items")
    upstream = {str(x["name"]): x for x in multihop_items}
    if [x["name"] for x in rows] != [x["name"] for x in eligible_rows]:
        raise RuntimeError("frozen atomic item order differs from the eligibility manifest")
    if len({x["name"] for x in rows}) != 81:
        raise RuntimeError("frozen atomic task names are not unique")

    expanded: list[dict[str, Any]] = []
    for item_index, row in enumerate(rows):
        source = upstream.get(str(row["name"]))
        if source is None or source["prompt"] != row["source_prompt"]:
            raise RuntimeError(f"frozen atomic source mismatch: {row['name']}")
        for arm, (question_field, alias_field) in ATOMIC_QUESTION_ARMS.items():
            expanded.append({
                "item_index": item_index,
                "name": row["name"],
                "arm": arm,
                "input_text": ATOMIC_WRAPPER.format(question=row[question_field]),
                "accepted_labels": [str(x) for x in row[alias_field]],
                "fact_status": row["fact_status"],
            })
        expanded.append({
            "item_index": item_index,
            "name": row["name"],
            "arm": ATOMIC_CONTINUITY_ARM,
            "input_text": row["source_prompt"],
            "accepted_labels": [str(x) for x in row["target_aliases"]],
            "fact_status": row["fact_status"],
        })
    return expanded, frozen


def _device_and_dtype(torch: Any) -> tuple[str, Any]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.bfloat16


def _load_hf_cell(
    repo: Path,
    cell_name: str,
    *,
    expected_snapshot: Mapping[str, Any],
    expected_device: Mapping[str, Any],
) -> tuple[Any, Any, Any, str, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = MODEL_CELLS[cell_name]
    device, dtype = _device_and_dtype(torch)
    snapshot = Path(str(expected_snapshot["snapshot_path"]))
    snapshot_manifest = verify_existing_hf_snapshot(
        repo, cell_name, snapshot, include_model_weights=True
    )
    if snapshot_manifest != dict(expected_snapshot):
        raise RuntimeError(f"{cell_name}: verified HF snapshot differs from sealed preflight")
    architecture = verify_snapshot_architecture(
        snapshot, cell_name, include_model_weights=True
    )
    current_device = execution_device_fingerprint(torch)
    if architecture.get("model_weight_dtypes") == ["BF16"]:
        current_device["observed_model_dtype"] = "torch.bfloat16"
    if current_device != dict(expected_device):
        raise RuntimeError(f"{cell_name}: execution device differs from sealed preflight")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, dtype=dtype, attn_implementation="eager"
    ).to(device)
    model.eval()
    observed_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    expected_observed = str(expected_device["observed_model_dtype"])
    if observed_dtypes != [expected_observed]:
        raise RuntimeError(
            f"{cell_name}: observed model parameter dtypes {observed_dtypes} "
            f"!= {[expected_observed]}"
        )
    expected_bos = spec["expected_bos_token_id"]
    actual_bos = getattr(tokenizer, "bos_token_id", None)
    if actual_bos != expected_bos:
        raise RuntimeError(f"{cell_name}: BOS drift {actual_bos!r} != {expected_bos!r}")
    if len(tokenizer) != spec["expected_tokenizer_len"]:
        raise RuntimeError(f"{cell_name}: tokenizer length drift")
    if tuple(int(x) for x in tokenizer.all_special_ids) != spec["expected_special_ids"]:
        raise RuntimeError(f"{cell_name}: tokenizer special-ID drift")
    if tokenizer.eos_token_id != 151643 or tokenizer.pad_token_id != 151643:
        raise RuntimeError(f"{cell_name}: tokenizer EOS/PAD drift")
    config = model.config
    if int(config.num_hidden_layers) != EXPECTED_N_LAYERS:
        raise RuntimeError(f"{cell_name}: model layer-count drift")
    if int(config.hidden_size) != EXPECTED_D_MODEL:
        raise RuntimeError(f"{cell_name}: model d_model drift")
    if int(model.get_output_embeddings().weight.shape[0]) != spec["expected_head_vocab_size"]:
        raise RuntimeError(f"{cell_name}: model output-head vocabulary drift")
    return tokenizer, model, dtype, device, snapshot_manifest


def _explicit_raw_ids(tokenizer: Any, text: str, force_bos: bool) -> list[int]:
    ids = [int(x) for x in tokenizer(text, add_special_tokens=False)["input_ids"]]
    if force_bos:
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            raise RuntimeError("forced-BOS cell has no BOS ID")
        ids = [int(bos_id), *ids]
    if len(ids) > MAX_SEQ_LEN:
        raise RuntimeError("atomic prompt exceeds fixed max sequence length")
    return ids


def holm_adjust_two(p_first: float, p_second: float) -> list[float]:
    values = np.asarray([p_first, p_second], dtype=np.float64)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(2, dtype=np.float64)
    adjusted[order[0]] = min(1.0, 2.0 * values[order[0]])
    adjusted[order[1]] = max(adjusted[order[0]], values[order[1]])
    return adjusted.tolist()


def _atomic_outcomes_from_records(
    records: Sequence[Mapping[str, Any]], frozen: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    n_items = len(frozen["items"])
    output: dict[str, np.ndarray] = {}
    by_key = {(str(x["cell"]), int(x["item_index"]), str(x["arm"])): x for x in records}
    expected = len(MODEL_CELLS) * n_items * 4
    if len(by_key) != expected or len(records) != expected:
        raise RuntimeError(f"atomic records must contain {expected} unique cell/item/arm rows")
    for cell in MODEL_CELLS:
        for arm in (*ATOMIC_QUESTION_ARMS, ATOMIC_CONTINUITY_ARM):
            output[f"{cell}_{arm}"] = np.asarray(
                [bool(by_key[(cell, index, arm)]["score"]["exact_primary"])
                 for index in range(n_items)],
                dtype=np.int8,
            )
    return output


def analyse_atomic_records(
    records: Sequence[Mapping[str, Any]], frozen: Mapping[str, Any]
) -> dict[str, Any]:
    outcomes = _atomic_outcomes_from_records(records, frozen)
    hb1, hb2, mb = outcomes["base_hop1"], outcomes["base_hop2"], outcomes["base_composite"]
    hd1, hd2, md = outcomes["distill_hop1"], outcomes["distill_hop2"], outcomes["distill_composite"]
    kb, kd = hb1 * hb2, hd1 * hd2
    delta_k_items = kb.astype(float) - kd.astype(float)
    delta_x_items = (kd - md).astype(float) - (kb - mb).astype(float)

    k_boot = paired_bootstrap_mean_difference(
        kb, kd, seed=ATOMIC_BOOTSTRAP_SEED
    )
    # The generic paired bootstrap reports mean(first-second).  Encode ΔX as
    # first=KD-MD and second=KB-MB exactly as preregistered.
    x_boot = paired_bootstrap_mean_difference(
        kd - md, kb - mb, seed=ATOMIC_BOOTSTRAP_SEED
    )
    k_swap = checkpoint_vector_swap_test(
        delta_k_items, seed=ATOMIC_SWAP_SEED
    )
    x_swap = checkpoint_vector_swap_test(
        delta_x_items, seed=ATOMIC_SWAP_SEED
    )
    adjusted = holm_adjust_two(k_swap["two_sided_p"], x_swap["two_sided_p"])
    k_swap["holm_adjusted_two_sided_p"] = adjusted[0]
    x_swap["holm_adjusted_two_sided_p"] = adjusted[1]

    stable_mask = np.asarray(
        [row["fact_status"] == "stable" for row in frozen["items"]], dtype=bool
    )
    if int(stable_mask.sum()) != 60:
        raise RuntimeError("frozen stable-only atomic sensitivity must contain 60 items")
    sensitivity = {
        "n_items": int(stable_mask.sum()),
        "delta_k": paired_bootstrap_mean_difference(
            kb[stable_mask], kd[stable_mask], seed=ATOMIC_BOOTSTRAP_SEED
        ),
        "delta_x": paired_bootstrap_mean_difference(
            (kd - md)[stable_mask], (kb - mb)[stable_mask], seed=ATOMIC_BOOTSTRAP_SEED
        ),
        "p_values": "not calculated by protocol",
    }

    supportive_bootstrap = atomic_supportive_bootstrap(
        hb1, hb2, hd1, hd2, mb, md,
        n_bootstrap=N_BOOTSTRAP, seed=ATOMIC_BOOTSTRAP_SEED,
    )

    margin = 0.10
    k_material = k_boot["percentile_95_ci"][0] > margin
    x_material = x_boot["percentile_95_ci"][0] > margin
    k_equiv = (k_boot["percentile_90_ci"][0] > -margin
               and k_boot["percentile_90_ci"][1] < margin)
    x_equiv = (x_boot["percentile_90_ci"][0] > -margin
               and x_boot["percentile_90_ci"][1] < margin)
    if k_material and x_equiv:
        pattern = "factual-dominant"
    elif x_material and k_equiv:
        pattern = "composition-dominant"
    elif k_material and x_material:
        pattern = "mixed"
    else:
        pattern = "indeterminate"

    return {
        "independent_unit": "source multihop item",
        "n_items": len(kb),
        "co_primary": {
            "delta_k": {"definition": "mean(K_base-K_distill)", "bootstrap": k_boot,
                        "checkpoint_vector_swap": k_swap},
            "delta_x": {"definition": "mean[(K_distill-M_distill)-(K_base-M_base)]",
                        "bootstrap": x_boot, "checkpoint_vector_swap": x_swap},
        },
        "pattern": pattern,
        "practical_equivalence_margin": [-margin, margin],
        "supportive": {
            "atomic_state_bootstrap": supportive_bootstrap["atomic_state"],
            "hop1_risk_difference_base_minus_distill": paired_bootstrap_mean_difference(
                hb1, hd1, seed=ATOMIC_BOOTSTRAP_SEED
            ),
            "hop2_risk_difference_base_minus_distill": paired_bootstrap_mean_difference(
                hb2, hd2, seed=ATOMIC_BOOTSTRAP_SEED
            ),
            "composite_risk_difference_base_minus_distill": paired_bootstrap_mean_difference(
                mb, md, seed=ATOMIC_BOOTSTRAP_SEED
            ),
            "common_known_composite": supportive_bootstrap["common_known_composite"],
            "source_prompt_continuity_rates": {
                cell: float(outcomes[f"{cell}_{ATOMIC_CONTINUITY_ARM}"].mean())
                for cell in MODEL_CELLS
            },
        },
        "stable_only_sensitivity": sensitivity,
        "interpretation_boundary": "operational composition contrast; not causal mediation",
    }


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def validate_atomic_partial(
    path: Path,
    *,
    tasks: Sequence[Mapping[str, Any]],
    binding: Mapping[str, Any],
    sealed_preflight: Mapping[str, Any],
    tokenizers: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("atomic partial checkpoint is missing or unsafe")
    payload = json.loads(path.read_text())
    if set(payload) != {
        "schema_version", "stage", "execution_binding", "records_sha256",
        "completed_prefix_digest", "records",
    }:
        raise RuntimeError("atomic partial checkpoint key-set drift")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("stage") != "atomic":
        raise RuntimeError("atomic partial checkpoint schema drift")
    assert_binding(payload.get("execution_binding", {}), binding, "atomic partial")
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise RuntimeError("atomic partial records are malformed")
    if payload.get("records_sha256") != sha256_bytes(canonical_json_bytes(rows)):
        raise RuntimeError("atomic partial records digest mismatch")
    identifiers = [
        [str(row.get("cell")), int(row.get("task_index", -1))] for row in rows
    ]
    expected_prefix = partial_completed_prefix_digest(
        stage="atomic", binding=binding, ordered_identifiers=identifiers,
        payload_sha256=str(payload["records_sha256"]),
    )
    if payload.get("completed_prefix_digest") != expected_prefix:
        raise RuntimeError("atomic partial completed-prefix digest mismatch")
    expected_order = [(cell, index) for cell in MODEL_CELLS for index in range(len(tasks))]
    if len(rows) > len(expected_order):
        raise RuntimeError("atomic partial has too many records")
    token_ids = behavior_token_id_lookup(sealed_preflight)
    expected_device = sealed_preflight["device_fingerprint"]["device"]
    expected_dtype = sealed_preflight["device_fingerprint"]["forward_dtype"]
    validated: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError("atomic partial row is not an object")
        if set(row) != {
            "cell", "model", "model_revision", "device", "dtype", "task_index",
            "item_index", "task_name", "arm", "input_token_ids",
            "generated_token_ids", "generated_text", "n_new_tokens_stored",
            "stopped_on_eos", "stopped_on_lf", "reached_token_cap", "score",
        }:
            raise RuntimeError(f"atomic partial row key-set drift at row {position}")
        cell, task_index = expected_order[position]
        task = tasks[task_index]
        exact_identity = {
            "cell": cell,
            "task_index": task_index,
            "item_index": int(task["item_index"]),
            "task_name": str(task["name"]),
            "arm": str(task["arm"]),
        }
        if any(row.get(key) != value for key, value in exact_identity.items()):
            raise RuntimeError(f"atomic partial identity drift at row {position}")
        spec = MODEL_CELLS[cell]
        if row.get("model") != spec["model"] or row.get("model_revision") != spec["revision"]:
            raise RuntimeError(f"atomic partial model drift at row {position}")
        if row.get("device") != expected_device or row.get("dtype") != expected_dtype:
            raise RuntimeError(f"atomic partial device/dtype drift at row {position}")
        expected_ids = token_ids[("atomic", str(task["name"]), str(task["arm"]))]
        if row.get("input_token_ids") != expected_ids:
            raise RuntimeError(f"atomic partial input-token drift at row {position}")
        generated = row.get("generated_token_ids")
        if (not isinstance(generated, list) or len(generated) > ATOMIC_MAX_NEW_TOKENS
                or any(not isinstance(x, int) or x < 0 or x >= 151936 for x in generated)):
            raise RuntimeError(f"atomic partial generated-token payload malformed at row {position}")
        if 151643 in generated:
            raise RuntimeError(f"atomic partial retained an EOS token at row {position}")
        if row.get("n_new_tokens_stored") != len(generated):
            raise RuntimeError(f"atomic partial token count drift at row {position}")
        if not isinstance(row.get("generated_text"), str):
            raise RuntimeError(f"atomic partial generated text malformed at row {position}")
        decoded = tokenizers[cell].decode(generated, skip_special_tokens=False)
        if row["generated_text"] != decoded:
            raise RuntimeError(f"atomic partial token-to-text decode drift at row {position}")
        for field in ("stopped_on_eos", "stopped_on_lf", "reached_token_cap"):
            if not isinstance(row.get(field), bool):
                raise RuntimeError(f"atomic partial stop flag malformed at row {position}")
        eos = bool(row["stopped_on_eos"])
        lf = bool(row["stopped_on_lf"])
        cap = bool(row["reached_token_cap"])
        if lf != ("\n" in decoded):
            raise RuntimeError(f"atomic partial logical line-stop drift at row {position}")
        validate_generation_stop_state(
            generated, stopped_on_eos=eos, stopped_on_lf=lf,
            reached_token_cap=cap, max_new_tokens=ATOMIC_MAX_NEW_TOKENS,
            context=f"atomic partial row {position}",
        )
        expected_score = force_incorrect_on_token_cap(
            score_atomic_answer(row["generated_text"], task["accepted_labels"]),
            reached_token_cap=bool(row["reached_token_cap"]),
        )
        if row.get("score") != expected_score:
            raise RuntimeError(f"atomic partial score drift at row {position}")
        validated.append(dict(row))
    return validated


def write_atomic_partial(
    path: Path, records: Sequence[Mapping[str, Any]], binding: Mapping[str, Any]
) -> None:
    rows = [dict(row) for row in records]
    records_sha = sha256_bytes(canonical_json_bytes(rows))
    identifiers = [[str(row["cell"]), int(row["task_index"])] for row in rows]
    atomic_write_json(path, {
        "schema_version": SCHEMA_VERSION,
        "stage": "atomic",
        "execution_binding": dict(binding),
        "records_sha256": records_sha,
        "completed_prefix_digest": partial_completed_prefix_digest(
            stage="atomic", binding=binding, ordered_identifiers=identifiers,
            payload_sha256=records_sha,
        ),
        "records": rows,
    })


def validate_atomic_final_bundle(
    run_dir: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Re-read every promoted A0 payload before granting completion authority."""
    paths = {
        "generations": run_dir / "generations.jsonl",
        "report": run_dir / "report.json",
        "markdown": run_dir / "REPORT.md",
        "manifest": run_dir / "manifest.json",
    }
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"atomic final {name} payload is missing or unsafe")
    expected_generations = jsonl_bytes(records)
    if paths["generations"].read_bytes() != expected_generations:
        raise RuntimeError("atomic final generations differ from validated partial records")
    parsed_rows = [json.loads(line) for line in paths["generations"].read_text().splitlines()]
    if parsed_rows != [dict(row) for row in records]:
        raise RuntimeError("atomic final generation rows failed round-trip validation")
    if json.loads(paths["report"].read_text()) != dict(report):
        raise RuntimeError("atomic final report differs from recomputed report")
    expected_markdown = atomic_report_markdown(report).encode("utf-8")
    if paths["markdown"].read_bytes() != expected_markdown:
        raise RuntimeError("atomic final markdown differs from recomputed rendering")
    canonical_manifest = json.loads(canonical_json_bytes(dict(manifest)))
    if json.loads(paths["manifest"].read_text()) != canonical_manifest:
        raise RuntimeError("atomic final manifest differs from recomputed manifest")
    if manifest.get("generation_jsonl_sha256") != sha256_file(paths["generations"]):
        raise RuntimeError("atomic manifest generation hash drift")
    if manifest.get("report_sha256") != sha256_file(paths["report"]):
        raise RuntimeError("atomic manifest report hash drift")


def run_atomic_stage(
    repo: Path,
    run_dir: Path,
    cache_dir: Path,
    input_hashes: Mapping[str, str],
    sealed_preflight: Mapping[str, Any],
    lens_paths: Mapping[str, Path],
) -> None:
    import torch
    from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

    started_wall = time.time()
    started_utc = utc_now()

    inputs, _ = load_upstream_inputs(cache_dir)
    got = sha256_file(repo / ELIGIBILITY_MANIFEST)
    if got != ELIGIBILITY_SHA256:
        raise RuntimeError(f"eligibility manifest hash drift: {got}")
    eligibility = json.loads((repo / ELIGIBILITY_MANIFEST).read_text())
    tasks, frozen = load_frozen_atomic_tasks(repo, inputs["multihop"], eligibility)
    task_hash = sha256_file(repo / ATOMIC_TASK_FILE)
    records_path = run_dir / "generations.jsonl"
    partial_path = run_dir / "partial_atomic.json"
    binding = execution_binding(run_dir)
    if (run_dir / "COMPLETE").exists():
        validate_completion_marker(repo, run_dir, "COMPLETE", "atomic")
        if (run_dir / "INCOMPLETE").exists():
            (run_dir / "INCOMPLETE").unlink()
            fsync_parent_directory(run_dir / "INCOMPLETE")
        return
    plan_path = run_dir / "atomic_plan.json"
    plan = {
        "task_list_sha256": task_hash,
        "task_population": 81,
        "n_tasks_per_cell": len(tasks),
        "models": {cell: {"model": spec["model"], "revision": spec["revision"]}
                   for cell, spec in MODEL_CELLS.items()},
        "execution_binding": binding,
    }
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise RuntimeError("atomic resume plan differs from the frozen task plan")
    else:
        atomic_write_json(plan_path, plan)

    model_snapshots = sealed_preflight["tokenizer_only_orderops"]["verified_hf_snapshots"]
    if not all(record.get("model_weights_verified") for record in model_snapshots.values()):
        raise RuntimeError("atomic stage lacks globally verified model weights")
    if not all(
        record.get("weight_header_verified")
        for record in sealed_preflight["tokenizer_only_orderops"]["architecture_receipts"].values()
    ):
        raise RuntimeError("atomic stage lacks global architecture-header verification")
    for cell_name, record in model_snapshots.items():
        current = verify_existing_hf_snapshot(
            repo, cell_name, Path(record["snapshot_path"]), include_model_weights=True
        )
        if current != record:
            raise RuntimeError(f"{cell_name}: global HF snapshot drift before atomic forward")
    validation_tokenizers = {
        cell: AutoTokenizer.from_pretrained(
            Path(model_snapshots[cell]["snapshot_path"]), local_files_only=True
        )
        for cell in MODEL_CELLS
    }
    records: list[dict[str, Any]] = []
    if partial_path.exists():
        records = validate_atomic_partial(
            partial_path, tasks=tasks, binding=binding,
            sealed_preflight=sealed_preflight, tokenizers=validation_tokenizers,
        )
    elif records_path.exists():
        raise RuntimeError("atomic final generations exist without a bound partial checkpoint")
    existing = {(str(row["cell"]), int(row["task_index"])): row for row in records}
    summaries: dict[str, Any] = {}
    runtime_cells: dict[str, Any] = {}
    expected_device = sealed_preflight["device_fingerprint"]
    torch.manual_seed(ATOMIC_GENERATION_SEED)
    for cell_name, spec in MODEL_CELLS.items():
        if all((cell_name, index) in existing for index in range(len(tasks))):
            continue
        tokenizer, model, dtype, device, snapshot_manifest = _load_hf_cell(
            repo, cell_name, expected_snapshot=model_snapshots[cell_name],
            expected_device=expected_device,
        )
        if snapshot_manifest != model_snapshots[cell_name]:
            raise RuntimeError(f"{cell_name}: atomic snapshot changed during load")
        runtime_cells[cell_name] = {"device": device, "dtype": str(dtype)}
        if tokenizer.eos_token_id != 151643:
            raise RuntimeError(f"{cell_name}: behavioural EOS drift")
        pad_id = 151643
        for index, task in enumerate(tasks):
            if (cell_name, index) in existing:
                continue
            ids = _explicit_raw_ids(tokenizer, task["input_text"], False)
            frozen_ids = behavior_token_id_lookup(sealed_preflight)[
                ("atomic", str(task["name"]), str(task["arm"]))
            ]
            if ids != frozen_ids:
                raise RuntimeError(f"{cell_name}/{task['name']}/{task['arm']}: token drift")
            tensor = torch.tensor([ids], dtype=torch.long, device=device)

            class StopAfterGeneratedNewline(StoppingCriteria):
                def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                    continuation = tokenizer.decode(
                        input_ids[0, len(ids):], skip_special_tokens=False
                    )
                    return "\n" in continuation

            with torch.inference_mode():
                generated = model.generate(
                    input_ids=tensor,
                    max_new_tokens=ATOMIC_MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=pad_id,
                    eos_token_id=151643,
                    stopping_criteria=StoppingCriteriaList([StopAfterGeneratedNewline()]),
                )
            raw_continuation_ids = [int(x) for x in generated[0, len(ids):].tolist()]
            stopped_on_eos = bool(
                raw_continuation_ids and raw_continuation_ids[-1] == 151643
            )
            continuation_ids = list(raw_continuation_ids)
            if stopped_on_eos:
                continuation_ids = continuation_ids[:-1]
            text = tokenizer.decode(continuation_ids, skip_special_tokens=False)
            stopped_on_lf = "\n" in text
            reached_token_cap = bool(
                not stopped_on_eos and not stopped_on_lf
                and len(raw_continuation_ids) >= ATOMIC_MAX_NEW_TOKENS
            )
            score = force_incorrect_on_token_cap(
                score_atomic_answer(text, task["accepted_labels"]),
                reached_token_cap=reached_token_cap,
            )
            record = {
                "cell": cell_name,
                "model": spec["model"],
                "model_revision": spec["revision"],
                "device": device,
                "dtype": str(dtype),
                "task_index": index,
                "item_index": task["item_index"],
                "task_name": task["name"],
                "arm": task["arm"],
                "input_token_ids": ids,
                "generated_token_ids": continuation_ids,
                "generated_text": text,
                "n_new_tokens_stored": len(continuation_ids),
                "stopped_on_eos": stopped_on_eos,
                "stopped_on_lf": stopped_on_lf,
                "reached_token_cap": reached_token_cap,
                "score": score,
            }
            records.append(record)
            write_atomic_partial(partial_path, records, binding)
            records = validate_atomic_partial(
                partial_path, tasks=tasks, binding=binding,
                sealed_preflight=sealed_preflight, tokenizers=validation_tokenizers,
            )
            existing[(cell_name, index)] = record
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    records = validate_atomic_partial(
        partial_path, tasks=tasks, binding=binding, sealed_preflight=sealed_preflight,
        tokenizers=validation_tokenizers,
    )
    preserve_unbound_files(
        run_dir,
        ("generations.jsonl", "report.json", "REPORT.md", "manifest.json",
         "derivation_manifest.json"),
        label="atomic",
    )
    atomic_write_bytes(records_path, jsonl_bytes(records))
    completed = {(row["cell"], int(row["task_index"])): row for row in records}
    expected_keys = {(cell, index) for cell in MODEL_CELLS for index in range(len(tasks))}
    if set(completed) != expected_keys:
        raise RuntimeError("atomic checkpoint is incomplete after generation loop")
    for cell_name in MODEL_CELLS:
        summaries[cell_name] = {}
        for arm in (*ATOMIC_QUESTION_ARMS, ATOMIC_CONTINUITY_ARM):
            selected = [record for (cell, _), record in completed.items()
                        if cell == cell_name and record["arm"] == arm]
            exact = sum(bool(x["score"]["exact_primary"]) for x in selected)
            whole = sum(bool(x["score"]["whole_term_sensitivity"]) for x in selected)
            substring = sum(bool(x["score"]["substring_sensitivity"]) for x in selected)
            summaries[cell_name][arm] = {
                "n": len(selected),
                "exact_hits": exact,
                "exact_rate": exact / len(selected),
                "whole_term_hits": whole,
                "whole_term_rate": whole / len(selected),
                "substring_hits": substring,
                "substring_rate": substring / len(selected),
            }
    runtime_cells = {
        cell: {
            "device": next(row["device"] for row in records if row["cell"] == cell),
            "dtype": next(row["dtype"] for row in records if row["cell"] == cell),
        }
        for cell in MODEL_CELLS
    }
    report = analyse_atomic_records(records, frozen)
    report["behavioral_scoring_sensitivities"] = summaries
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "atomic",
        "seal": json.loads((run_dir / "seal.json").read_text()),
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "wall_seconds": round(time.time() - started_wall, 3),
        "models": MODEL_CELLS,
        "verified_hf_snapshots": model_snapshots,
        "device_fingerprint": sealed_preflight["device_fingerprint"],
        "model_dtype": "bfloat16",
        "decoding": "greedy",
        "max_new_tokens": ATOMIC_MAX_NEW_TOKENS,
        "seed": ATOMIC_GENERATION_SEED,
        "behaviour_manual_bos": False,
        "eos_token_id": 151643,
        "pad_token_id": 151643,
        "task_population": 81,
        "arms": [*ATOMIC_QUESTION_ARMS, ATOMIC_CONTINUITY_ARM],
        "task_list_sha256": task_hash,
        "frozen_atomic_task_path": ATOMIC_TASK_FILE.as_posix(),
        "eligibility_manifest_sha256": got,
        "upstream_file_sha256": dict(input_hashes),
        "scoring": "NFKC + casefold + whitespace collapse + outer quote/terminal punctuation strip; exact alias primary",
        "summaries": summaries,
        "generation_jsonl_sha256": sha256_file(records_path),
        "report_sha256": sha256_bytes(canonical_json_bytes(report)),
        "runtime_cells": runtime_cells,
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "tokenizers": __import__("tokenizers").__version__,
            "platform": platform.platform(),
        },
    }
    atomic_write_json(run_dir / "report.json", report)
    atomic_write_bytes(run_dir / "REPORT.md", atomic_report_markdown(report).encode("utf-8"))
    atomic_write_json(run_dir / "manifest.json", manifest)
    validate_atomic_final_bundle(
        run_dir, records=records, report=report, manifest=manifest,
    )
    derivation = write_derivation_manifest(
        repo, run_dir, stage="atomic",
        extra_inputs=[
            *lens_paths.values(), *snapshot_input_paths(model_snapshots),
            *fetched_input_paths(run_dir), *campaign_input_paths(run_dir),
            stage_selection_path(run_dir),
        ],
    )
    finalize_stage(
        repo, run_dir, stage="atomic", marker_name="COMPLETE",
        manifest_name="manifest.json", derivation_name=derivation.name,
    )


def _semantic_rank_bank_torch(logits: Any, keys: Sequence[str], ids_by_key: Mapping[str, Mapping[str, int]]) -> np.ndarray:
    """GPU-capable exact ranks without a vocabulary argsort."""
    import torch

    if logits.ndim != 2:
        raise RuntimeError("expected logits [source_layers,vocabulary]")
    vocab_size = logits.shape[1]
    vocab_ids = torch.arange(vocab_size, device=logits.device)
    columns = []
    for key in keys:
        ranks_for_synonyms = []
        for token_id in ids_by_key[key].values():
            token_id = int(token_id)
            target = logits[:, token_id:token_id + 1]
            before = (logits > target) | ((logits == target) & (vocab_ids[None, :] < token_id))
            rank = 1 + before.sum(dim=1, dtype=torch.int64)
            ranks_for_synonyms.append(rank)
        columns.append(torch.stack(ranks_for_synonyms, dim=1).amin(dim=1))
    return torch.stack(columns, dim=1).to(torch.int32).cpu().numpy()


def _deterministic_top25_torch(logits: Any) -> tuple[np.ndarray, int]:
    # Reuse the already validated deterministic primitive.  The logit tensor
    # has already been restricted to the registered tokenizer-ID prefix.
    from jspace_phase1_scoring import deterministic_topk_ids

    ids, ties = deterministic_topk_ids(logits, k=K)
    return ids.to(dtype=__import__("torch").int32).cpu().numpy(), int(ties)


def _capture_prefix_residuals(lens_model: Any, prefix_ids: Sequence[int], source_layers: Sequence[int], device: str) -> Any:
    import torch
    from jlens.hooks import ActivationRecorder

    input_ids = torch.tensor([list(prefix_ids)], dtype=torch.long, device=device)
    with torch.inference_mode(), ActivationRecorder(lens_model.layers, at=list(source_layers)) as recorder:
        lens_model.forward(input_ids)
        acts = torch.stack(
            [recorder.activations[layer][0, -1, :].detach() for layer in source_layers], dim=0
        )
    return acts[:, None, :]


def _readout_logits(
    lens_model: Any,
    residual: Any,
    jacobian_stack: Any | None,
    *,
    ranked_vocab_size: int,
    unembed: Any,
    readout_dtype: str,
) -> Any:
    import torch

    if jacobian_stack is None:
        transported = residual
    else:
        transported = torch.bmm(residual.float(), jacobian_stack.transpose(1, 2))
    flat = transported.reshape(-1, transported.shape[-1])
    if readout_dtype == "float32":
        logits = unembed(flat.float())
    elif readout_dtype == "bfloat16":
        logits = unembed(flat.to(torch.bfloat16))
    else:
        raise ValueError(readout_dtype)
    return logits[:, :ranked_vocab_size]


def _generate_raw_behavior(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    device: str,
    max_new_tokens: int,
    stop_on_lf: bool,
) -> dict[str, Any]:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    ids = [int(x) for x in tokenizer(text, add_special_tokens=False)["input_ids"]]
    if not ids:
        raise RuntimeError("empty behavioural prompt")
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    class StopAfterLF(StoppingCriteria):
        def __call__(self, generated: Any, scores: Any, **kwargs: Any) -> bool:
            continuation = tokenizer.decode(
                generated[0, len(ids):], skip_special_tokens=False
            )
            return stop_on_lf and "\n" in continuation

    criteria = StoppingCriteriaList([StopAfterLF()]) if stop_on_lf else None
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=151643,
            pad_token_id=151643,
            stopping_criteria=criteria,
        )
    raw_ids = [int(x) for x in output[0, len(ids):].tolist()]
    stopped_on_eos = bool(raw_ids and raw_ids[-1] == 151643)
    stored_ids = raw_ids[:-1] if stopped_on_eos else raw_ids
    decoded = tokenizer.decode(stored_ids, skip_special_tokens=False)
    if stop_on_lf and "\n" in decoded:
        decoded_for_score = decoded.split("\n", 1)[0]
    else:
        decoded_for_score = decoded
    stopped_on_lf = bool(stop_on_lf and "\n" in decoded)
    return {
        "input_token_ids": ids,
        "generated_token_ids": stored_ids,
        "generated_text": decoded,
        "text_before_lf": decoded_for_score,
        "stopped_on_eos": stopped_on_eos,
        "stopped_on_lf": stopped_on_lf,
        "reached_token_cap": bool(
            not stopped_on_eos and not stopped_on_lf and len(raw_ids) >= max_new_tokens
        ),
        "manual_bos": False,
    }


def _teacher_forced_target_logprob(
    model: Any, tokenizer: Any, prompt: str, target: str, *, device: str
) -> dict[str, Any]:
    import torch

    tokenization = joint_target_tokenization(tokenizer, prompt, target)
    ids = tokenization["joint_token_ids"]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0].float()
        log_probs = torch.log_softmax(logits, dim=-1)
    values = []
    for position in tokenization["target_token_positions"]:
        token_id = ids[position]
        values.append(float(log_probs[position - 1, token_id].item()))
    if not values or not np.isfinite(values).all():
        raise RuntimeError("non-finite teacher-forced target log-probability")
    return {
        **tokenization,
        "token_log_probabilities": values,
        "sum_log_probability": float(sum(values)),
        "mean_log_probability": float(np.mean(values)),
    }


def _run_orderops_behavior_cell(
    run_dir: Path,
    cell_name: str,
    model: Any,
    tokenizer: Any,
    items: Sequence[Mapping[str, Any]],
    *,
    device: str,
    binding: Mapping[str, Any],
    sealed_preflight: Mapping[str, Any],
) -> tuple[Path, list[dict[str, Any]]]:
    """Resume-safe Direct/CoT/teacher-forced behaviour for all 55 items."""
    if tokenizer.eos_token_id != 151643:
        raise RuntimeError(f"{cell_name}: behavioural EOS drift")
    path = run_dir / f"behavior__{cell_name}.jsonl"
    partial_path = run_dir / f"partial_behavior__{cell_name}.json"
    records: list[dict[str, Any]] = []
    token_ids = behavior_token_id_lookup(sealed_preflight)

    def validate_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if len(rows) > len(items):
            raise RuntimeError(f"{cell_name}: too many behavioural partial rows")
        validated: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            item = items[index]
            name = str(item["name"])
            aliases = orderops_target_aliases(str(item["target"]))
            if set(row) != {
                "cell", "name", "target", "target_aliases", "direct", "cot",
                "teacher_forced",
            }:
                raise RuntimeError(f"{cell_name}/{name}: behavioural row key-set drift")
            if (row.get("cell") != cell_name or row.get("name") != name
                    or row.get("target") != item["target"]
                    or row.get("target_aliases") != list(aliases)):
                raise RuntimeError(f"{cell_name}/{name}: behavioural identity drift")
            for arm, max_tokens, stop_on_lf in (
                ("direct", ORDEROPS_DIRECT_MAX_NEW_TOKENS, True),
                ("cot", ORDEROPS_COT_MAX_NEW_TOKENS, False),
            ):
                payload = row.get(arm)
                if not isinstance(payload, dict) or set(payload) != {
                    "input_token_ids", "generated_token_ids", "generated_text",
                    "text_before_lf", "stopped_on_eos", "stopped_on_lf",
                    "reached_token_cap", "manual_bos", "score",
                }:
                    raise RuntimeError(f"{cell_name}/{name}/{arm}: malformed payload")
                expected_ids = token_ids[("orderops", name, arm)]
                if payload.get("input_token_ids") != expected_ids or payload.get("manual_bos") is not False:
                    raise RuntimeError(f"{cell_name}/{name}/{arm}: input-token drift")
                generated = payload.get("generated_token_ids")
                if (not isinstance(generated, list)
                        or len(generated) > max_tokens
                        or any(not isinstance(x, int) or x < 0 or x >= 151936 for x in generated)):
                    raise RuntimeError(f"{cell_name}/{name}/{arm}: generated-token drift")
                if 151643 in generated:
                    raise RuntimeError(f"{cell_name}/{name}/{arm}: retained EOS token drift")
                decoded = tokenizer.decode(generated, skip_special_tokens=False)
                if payload.get("generated_text") != decoded:
                    raise RuntimeError(f"{cell_name}/{name}/{arm}: token-to-text decode drift")
                for field in ("generated_text", "text_before_lf"):
                    if not isinstance(payload.get(field), str):
                        raise RuntimeError(f"{cell_name}/{name}/{arm}: malformed text")
                for field in ("stopped_on_eos", "stopped_on_lf", "reached_token_cap"):
                    if not isinstance(payload.get(field), bool):
                        raise RuntimeError(f"{cell_name}/{name}/{arm}: malformed stop flag")
                expected_text_before = (
                    payload["generated_text"].split("\n", 1)[0]
                    if stop_on_lf and "\n" in payload["generated_text"]
                    else payload["generated_text"]
                )
                if payload["text_before_lf"] != expected_text_before:
                    raise RuntimeError(f"{cell_name}/{name}/{arm}: line-stop text drift")
                eos = bool(payload["stopped_on_eos"])
                lf = bool(payload["stopped_on_lf"])
                cap = bool(payload["reached_token_cap"])
                if lf != (stop_on_lf and "\n" in decoded):
                    raise RuntimeError(f"{cell_name}/{name}/{arm}: logical line-stop drift")
                validate_generation_stop_state(
                    generated, stopped_on_eos=eos, stopped_on_lf=lf,
                    reached_token_cap=cap, max_new_tokens=max_tokens,
                    context=f"{cell_name}/{name}/{arm}",
                )
                score = (
                    score_atomic_answer(payload["text_before_lf"], aliases)
                    if arm == "direct"
                    else score_orderops_cot(payload["generated_text"], aliases)
                )
                expected_score = force_incorrect_on_token_cap(
                    score, reached_token_cap=bool(payload["reached_token_cap"])
                )
                if payload.get("score") != expected_score:
                    raise RuntimeError(f"{cell_name}/{name}/{arm}: score drift")
            teacher = row.get("teacher_forced")
            if not isinstance(teacher, dict) or set(teacher) != {
                "joint_token_ids", "joint_offsets", "target_token_positions",
                "target_token_ids", "target_character_interval", "manual_bos",
                "token_log_probabilities", "sum_log_probability",
                "mean_log_probability",
            }:
                raise RuntimeError(f"{cell_name}/{name}: teacher-forced key-set drift")
            expected_teacher = joint_target_tokenization(
                tokenizer, str(item["prompt"]), str(item["target"])
            )
            if expected_teacher["joint_token_ids"] != token_ids[
                ("orderops", name, "teacher_forced_joint")
            ]:
                raise RuntimeError(f"{cell_name}/{name}: frozen teacher token IDs drift")
            if any(
                teacher.get(key) != value for key, value in expected_teacher.items()
            ):
                raise RuntimeError(f"{cell_name}/{name}: teacher-forced tokenization drift")
            logs = teacher.get("token_log_probabilities")
            if (not isinstance(logs, list) or len(logs) != len(expected_teacher["target_token_ids"])
                    or not np.isfinite(np.asarray(logs, dtype=float)).all()):
                raise RuntimeError(f"{cell_name}/{name}: teacher-forced score payload drift")
            if not math.isclose(float(teacher.get("sum_log_probability")), float(sum(logs)), abs_tol=1e-10):
                raise RuntimeError(f"{cell_name}/{name}: teacher-forced sum drift")
            if not math.isclose(float(teacher.get("mean_log_probability")), float(np.mean(logs)), abs_tol=1e-10):
                raise RuntimeError(f"{cell_name}/{name}: teacher-forced mean drift")
            validated.append(dict(row))
        return validated

    if partial_path.exists():
        if partial_path.is_symlink():
            raise RuntimeError(f"{cell_name}: unsafe behavioural partial")
        partial = json.loads(partial_path.read_text())
        if set(partial) != {
            "schema_version", "stage", "cell", "execution_binding",
            "records_sha256", "completed_prefix_digest", "records",
        }:
            raise RuntimeError(f"{cell_name}: behavioural partial key-set drift")
        if (partial.get("schema_version") != SCHEMA_VERSION
                or partial.get("stage") != "orderops-behavior"
                or partial.get("cell") != cell_name):
            raise RuntimeError(f"{cell_name}: behavioural partial schema drift")
        assert_binding(partial.get("execution_binding", {}), binding, f"{cell_name} behavior")
        rows = partial.get("records")
        if not isinstance(rows, list) or partial.get("records_sha256") != sha256_bytes(canonical_json_bytes(rows)):
            raise RuntimeError(f"{cell_name}: behavioural partial digest mismatch")
        behavior_identifiers = [str(row.get("name")) for row in rows]
        expected_prefix = partial_completed_prefix_digest(
            stage=f"orderops-behavior:{cell_name}", binding=binding,
            ordered_identifiers=behavior_identifiers,
            payload_sha256=str(partial["records_sha256"]),
        )
        if partial.get("completed_prefix_digest") != expected_prefix:
            raise RuntimeError(f"{cell_name}: behavioural completed-prefix digest mismatch")
        records = validate_records(rows)
    elif path.exists():
        raise RuntimeError(f"{cell_name}: final behavior exists without a bound partial")
    for item in items[len(records):]:
        aliases = orderops_target_aliases(str(item["target"]))
        direct = _generate_raw_behavior(
            model, tokenizer, str(item["prompt"]), device=device,
            max_new_tokens=ORDEROPS_DIRECT_MAX_NEW_TOKENS, stop_on_lf=True,
        )
        direct["score"] = force_incorrect_on_token_cap(
            score_atomic_answer(direct["text_before_lf"], aliases),
            reached_token_cap=bool(direct["reached_token_cap"]),
        )
        cot = _generate_raw_behavior(
            model, tokenizer,
            ORDEROPS_COT_WRAPPER.format(prompt=item["prompt"]),
            device=device, max_new_tokens=ORDEROPS_COT_MAX_NEW_TOKENS,
            stop_on_lf=False,
        )
        cot["score"] = force_incorrect_on_token_cap(
            score_orderops_cot(cot["generated_text"], aliases),
            reached_token_cap=bool(cot["reached_token_cap"]),
        )
        teacher = _teacher_forced_target_logprob(
            model, tokenizer, str(item["prompt"]), str(item["target"]), device=device
        )
        record = {
            "cell": cell_name,
            "name": item["name"],
            "target": item["target"],
            "target_aliases": list(aliases),
            "direct": direct,
            "cot": cot,
            "teacher_forced": teacher,
        }
        records.append(record)
        records_sha = sha256_bytes(canonical_json_bytes(records))
        atomic_write_json(partial_path, {
            "schema_version": SCHEMA_VERSION,
            "stage": "orderops-behavior",
            "cell": cell_name,
            "execution_binding": dict(binding),
            "records_sha256": records_sha,
            "completed_prefix_digest": partial_completed_prefix_digest(
                stage=f"orderops-behavior:{cell_name}", binding=binding,
                ordered_identifiers=[str(row["name"]) for row in records],
                payload_sha256=records_sha,
            ),
            "records": records,
        })
        records = validate_records(records)
    if len(records) != 55:
        raise RuntimeError(f"{cell_name}: behaviour must complete all 55 items")
    atomic_write_bytes(path, jsonl_bytes(records))
    return path, records


def array_bundle_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        if name in {"payload_sha256", "completed_prefix_digest"}:
            continue
        array = np.asarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(canonical_json_bytes(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_npz_no_pickle(path: Path) -> dict[str, np.ndarray]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"unsafe or missing NumPy payload: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _validate_rank_and_top25_arrays(
    ranks: np.ndarray,
    top25: np.ndarray,
    *,
    expected_shape: tuple[int, int, int],
    ranked_vocab_size: int,
    context: str,
) -> None:
    if ranks.dtype != np.int32 or ranks.shape != expected_shape:
        raise RuntimeError(f"{context}: rank dtype/shape drift {ranks.dtype}/{ranks.shape}")
    if ranks.size and (int(ranks.min()) < 1 or int(ranks.max()) > ranked_vocab_size):
        raise RuntimeError(f"{context}: rank outside registered vocabulary")
    expected_top = (expected_shape[0], expected_shape[1], K)
    if top25.dtype != np.int32 or top25.shape != expected_top:
        raise RuntimeError(f"{context}: top-25 dtype/shape drift")
    if top25.size:
        if int(top25.min()) < 0 or int(top25.max()) >= ranked_vocab_size:
            raise RuntimeError(f"{context}: top-25 ID outside registered vocabulary")
        if np.any(np.diff(np.sort(top25, axis=-1), axis=-1) == 0):
            raise RuntimeError(f"{context}: duplicate ID within a top-25 row")


def validate_rank_partial(
    path: Path,
    *,
    cell: str,
    expected_names: Sequence[str],
    semantic_keys: Sequence[str],
    expected_label_pairs: np.ndarray,
    ranked_vocab_size: int,
    binding: Mapping[str, Any],
    expected_locators: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]], dict[str, int], list[dict[str, Any]]]:
    arrays = load_npz_no_pickle(path)
    expected_keys = {
        "schema_version", "stage", "cell", "completed_item_names",
        "semantic_keys", "source_layers", "label_pairs",
        "execution_binding_json", "locators_json", "payload_sha256",
        "completed_prefix_digest",
    }
    for arm in ORDEROPS_READOUT_ARMS:
        expected_keys.update({f"ranks__{arm}", f"top25__{arm}", f"boundary_ties__{arm}"})
    if set(arrays) != expected_keys:
        raise RuntimeError(f"{cell}: rank partial key-set drift")
    if (str(arrays["schema_version"].item()) != SCHEMA_VERSION
            or str(arrays["stage"].item()) != "orderops-rank-partial"
            or str(arrays["cell"].item()) != cell):
        raise RuntimeError(f"{cell}: rank partial envelope identity drift")
    if arrays["semantic_keys"].astype(str).tolist() != list(semantic_keys):
        raise RuntimeError(f"{cell}: rank partial semantic-key drift")
    if (arrays["source_layers"].dtype != np.int32
            or arrays["source_layers"].tolist() != list(SOURCE_LAYERS)):
        raise RuntimeError(f"{cell}: rank partial source-layer drift")
    if (arrays["label_pairs"].dtype != np.int32
            or not np.array_equal(arrays["label_pairs"], expected_label_pairs)):
        raise RuntimeError(f"{cell}: rank partial label-pair drift")
    stored_digest = str(arrays["payload_sha256"].item())
    if stored_digest != array_bundle_digest(arrays):
        raise RuntimeError(f"{cell}: rank partial payload digest mismatch")
    stored_binding = json.loads(str(arrays["execution_binding_json"].item()))
    assert_binding(stored_binding, binding, f"{cell} rank partial")
    names = arrays["completed_item_names"].astype(str).tolist()
    if names != list(expected_names[:len(names)]) or not 0 < len(names) <= len(expected_names):
        raise RuntimeError(f"{cell}: rank partial item-prefix drift")
    expected_prefix = partial_completed_prefix_digest(
        stage=f"orderops-rank:{cell}", binding=binding,
        ordered_identifiers=names, payload_sha256=stored_digest,
    )
    if str(arrays["completed_prefix_digest"].item()) != expected_prefix:
        raise RuntimeError(f"{cell}: rank partial completed-prefix digest mismatch")
    locators = json.loads(str(arrays["locators_json"].item()))
    expected_locator_rows = [
        {"name": name, **dict(expected_locators[name])} for name in names
    ]
    if locators != expected_locator_rows:
        raise RuntimeError(f"{cell}: rank partial locator drift")
    banks: dict[str, list[np.ndarray]] = {}
    top25: dict[str, list[np.ndarray]] = {}
    ties: dict[str, int] = {}
    for arm in ORDEROPS_READOUT_ARMS:
        ranks = arrays[f"ranks__{arm}"]
        top = arrays[f"top25__{arm}"]
        _validate_rank_and_top25_arrays(
            ranks, top,
            expected_shape=(len(names), len(SOURCE_LAYERS), len(semantic_keys)),
            ranked_vocab_size=ranked_vocab_size,
            context=f"{cell}/{arm} partial",
        )
        tie_array = arrays[f"boundary_ties__{arm}"]
        if tie_array.dtype != np.int64 or tie_array.shape != () or int(tie_array) < 0:
            raise RuntimeError(f"{cell}/{arm}: partial boundary-tie drift")
        banks[arm] = [row for row in ranks]
        top25[arm] = [row for row in top]
        ties[arm] = int(tie_array)
    return banks, top25, ties, locators


def validate_typo_partial(
    path: Path,
    *,
    cell: str,
    expected_names: Sequence[str],
    semantic_keys: Sequence[str],
    expected_label_indices: np.ndarray,
    ranked_vocab_size: int,
    binding: Mapping[str, Any],
    expected_locators: Mapping[str, Mapping[str, Any]],
) -> tuple[list[np.ndarray], list[np.ndarray], int, list[dict[str, Any]]]:
    arrays = load_npz_no_pickle(path)
    expected_keys = {
        "schema_version", "stage", "cell", "completed_item_names",
        "semantic_keys", "source_layers", "label_indices",
        "ranks", "top25", "boundary_ties",
        "execution_binding_json", "locators_json", "payload_sha256",
        "completed_prefix_digest",
    }
    if set(arrays) != expected_keys:
        raise RuntimeError(f"{cell}: typo partial key-set drift")
    if (str(arrays["schema_version"].item()) != SCHEMA_VERSION
            or str(arrays["stage"].item()) != "typo-rank-partial"
            or str(arrays["cell"].item()) != cell):
        raise RuntimeError(f"{cell}: typo partial envelope identity drift")
    if arrays["semantic_keys"].astype(str).tolist() != list(semantic_keys):
        raise RuntimeError(f"{cell}: typo partial semantic-key drift")
    if (arrays["source_layers"].dtype != np.int32
            or arrays["source_layers"].tolist() != list(SOURCE_LAYERS)):
        raise RuntimeError(f"{cell}: typo partial source-layer drift")
    if (arrays["label_indices"].dtype != np.int32
            or not np.array_equal(arrays["label_indices"], expected_label_indices)):
        raise RuntimeError(f"{cell}: typo partial label-index drift")
    if str(arrays["payload_sha256"].item()) != array_bundle_digest(arrays):
        raise RuntimeError(f"{cell}: typo partial payload digest mismatch")
    assert_binding(
        json.loads(str(arrays["execution_binding_json"].item())),
        binding, f"{cell} typo partial",
    )
    names = arrays["completed_item_names"].astype(str).tolist()
    if names != list(expected_names[:len(names)]) or not 0 < len(names) <= len(expected_names):
        raise RuntimeError(f"{cell}: typo partial item-prefix drift")
    stored_digest = str(arrays["payload_sha256"].item())
    expected_prefix = partial_completed_prefix_digest(
        stage=f"typo-rank:{cell}", binding=binding,
        ordered_identifiers=names, payload_sha256=stored_digest,
    )
    if str(arrays["completed_prefix_digest"].item()) != expected_prefix:
        raise RuntimeError(f"{cell}: typo partial completed-prefix digest mismatch")
    locators = json.loads(str(arrays["locators_json"].item()))
    expected_rows = [{"name": name, **dict(expected_locators[name])} for name in names]
    if locators != expected_rows:
        raise RuntimeError(f"{cell}: typo partial locator drift")
    _validate_rank_and_top25_arrays(
        arrays["ranks"], arrays["top25"],
        expected_shape=(len(names), len(SOURCE_LAYERS), len(semantic_keys)),
        ranked_vocab_size=ranked_vocab_size,
        context=f"{cell}/typo partial",
    )
    ties = arrays["boundary_ties"]
    if ties.dtype != np.int64 or ties.shape != () or int(ties) < 0:
        raise RuntimeError(f"{cell}: typo partial boundary-tie drift")
    return ([row for row in arrays["ranks"]], [row for row in arrays["top25"]],
            int(ties), locators)


def validate_orderops_rank_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    cell: str,
    expected_names: Sequence[str],
    semantic_keys: Sequence[str],
    expected_label_pairs: np.ndarray,
    typo_names: Sequence[str],
    typo_keys: Sequence[str],
    expected_typo_labels: np.ndarray,
    ranked_vocab_size: int,
) -> None:
    expected_keys = {
        "label_pairs", "item_names", "semantic_keys", "typo_fp32_j",
        "typo_top25__fp32_j", "typo_boundary_ties__fp32_j",
        "typo_label_indices", "typo_item_names", "typo_semantic_keys",
    }
    for arm in ORDEROPS_READOUT_ARMS:
        expected_keys.update({arm, f"top25__{arm}", f"boundary_ties__{arm}"})
    if set(arrays) != expected_keys:
        raise RuntimeError(f"{cell}: final rank payload key-set drift")
    if arrays["item_names"].astype(str).tolist() != list(expected_names):
        raise RuntimeError(f"{cell}: final orderops item order drift")
    if arrays["semantic_keys"].astype(str).tolist() != list(semantic_keys):
        raise RuntimeError(f"{cell}: final semantic-key order drift")
    if arrays["label_pairs"].dtype != np.int32 or not np.array_equal(
        arrays["label_pairs"], expected_label_pairs
    ):
        raise RuntimeError(f"{cell}: final label-pair drift")
    for arm in ORDEROPS_READOUT_ARMS:
        _validate_rank_and_top25_arrays(
            arrays[arm], arrays[f"top25__{arm}"],
            expected_shape=(len(expected_names), len(SOURCE_LAYERS), len(semantic_keys)),
            ranked_vocab_size=ranked_vocab_size,
            context=f"{cell}/{arm} final",
        )
        ties = arrays[f"boundary_ties__{arm}"]
        if ties.dtype != np.int64 or ties.shape != () or int(ties) < 0:
            raise RuntimeError(f"{cell}/{arm}: final boundary-tie drift")
    if arrays["typo_item_names"].astype(str).tolist() != list(typo_names):
        raise RuntimeError(f"{cell}: final typo item order drift")
    if arrays["typo_semantic_keys"].astype(str).tolist() != list(typo_keys):
        raise RuntimeError(f"{cell}: final typo semantic-key drift")
    if arrays["typo_label_indices"].dtype != np.int32 or not np.array_equal(
        arrays["typo_label_indices"], expected_typo_labels
    ):
        raise RuntimeError(f"{cell}: final typo-label drift")
    _validate_rank_and_top25_arrays(
        arrays["typo_fp32_j"], arrays["typo_top25__fp32_j"],
        expected_shape=(len(typo_names), len(SOURCE_LAYERS), len(typo_keys)),
        ranked_vocab_size=ranked_vocab_size,
        context=f"{cell}/typo final",
    )
    typo_ties = arrays["typo_boundary_ties__fp32_j"]
    if typo_ties.dtype != np.int64 or typo_ties.shape != () or int(typo_ties) < 0:
        raise RuntimeError(f"{cell}: final typo boundary-tie drift")


def validate_locator_bundle(
    bundle: Mapping[str, Any],
    *,
    cell: str,
    order_names: Sequence[str],
    typo_names: Sequence[str],
    tokenizer_preflight: Mapping[str, Any],
) -> None:
    expected = {
        "orderops": [
            {"name": name, **dict(tokenizer_preflight["locators"][cell][name])}
            for name in order_names
        ],
        "typo": [
            {"name": name, **dict(tokenizer_preflight["typo_locators"][cell][name])}
            for name in typo_names
        ],
    }
    if dict(bundle) != expected:
        raise RuntimeError(f"{cell}: final locator bundle drift")


def validate_behavior_payload(
    path: Path,
    *,
    cell: str,
    items: Sequence[Mapping[str, Any]],
    sealed_preflight: Mapping[str, Any],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{cell}: unsafe or missing behavior payload")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != len(items):
        raise RuntimeError(f"{cell}: behavior payload row-count drift")
    frozen_ids = behavior_token_id_lookup(sealed_preflight)
    for index, (row, item) in enumerate(zip(rows, items, strict=True)):
        name = str(item["name"])
        aliases = orderops_target_aliases(str(item["target"]))
        if set(row) != {
            "cell", "name", "target", "target_aliases", "direct", "cot",
            "teacher_forced",
        }:
            raise RuntimeError(f"{cell}/{name}: behavior row key-set drift")
        if (row.get("cell") != cell or row.get("name") != name
                or row.get("target") != item["target"]
                or row.get("target_aliases") != list(aliases)):
            raise RuntimeError(f"{cell}/{name}: behavior identity drift")
        for arm in ("direct", "cot"):
            payload = row.get(arm, {})
            if set(payload) != {
                "input_token_ids", "generated_token_ids", "generated_text",
                "text_before_lf", "stopped_on_eos", "stopped_on_lf",
                "reached_token_cap", "manual_bos", "score",
            }:
                raise RuntimeError(f"{cell}/{name}/{arm}: behavior key-set drift")
            if payload.get("input_token_ids") != frozen_ids[("orderops", name, arm)]:
                raise RuntimeError(f"{cell}/{name}/{arm}: behavior token drift")
            if payload.get("manual_bos") is not False or not isinstance(payload.get("generated_text"), str):
                raise RuntimeError(f"{cell}/{name}/{arm}: behavior schema drift")
            generated = payload.get("generated_token_ids")
            maximum = (
                ORDEROPS_DIRECT_MAX_NEW_TOKENS if arm == "direct"
                else ORDEROPS_COT_MAX_NEW_TOKENS
            )
            if (not isinstance(generated, list) or len(generated) > maximum
                    or any(not isinstance(x, int) or x < 0 or x >= 151936 for x in generated)):
                raise RuntimeError(f"{cell}/{name}/{arm}: generated-token drift")
            if 151643 in generated:
                raise RuntimeError(f"{cell}/{name}/{arm}: retained EOS token drift")
            decoded = tokenizer.decode(generated, skip_special_tokens=False)
            if payload.get("generated_text") != decoded:
                raise RuntimeError(f"{cell}/{name}/{arm}: token-to-text decode drift")
            for field in ("stopped_on_eos", "stopped_on_lf", "reached_token_cap"):
                if not isinstance(payload.get(field), bool):
                    raise RuntimeError(f"{cell}/{name}/{arm}: stop-flag drift")
            expected_before_lf = (
                payload["generated_text"].split("\n", 1)[0]
                if arm == "direct" and "\n" in payload["generated_text"]
                else payload["generated_text"]
            )
            if payload.get("text_before_lf") != expected_before_lf:
                raise RuntimeError(f"{cell}/{name}/{arm}: line-stop text drift")
            cap = bool(payload.get("reached_token_cap"))
            eos = bool(payload.get("stopped_on_eos"))
            lf = bool(payload.get("stopped_on_lf"))
            if lf != (arm == "direct" and "\n" in decoded):
                raise RuntimeError(f"{cell}/{name}/{arm}: logical line-stop drift")
            validate_generation_stop_state(
                generated, stopped_on_eos=eos, stopped_on_lf=lf,
                reached_token_cap=cap, max_new_tokens=maximum,
                context=f"{cell}/{name}/{arm}",
            )
            score = (
                score_atomic_answer(str(payload.get("text_before_lf", "")), aliases)
                if arm == "direct" else score_orderops_cot(payload["generated_text"], aliases)
            )
            if payload.get("score") != force_incorrect_on_token_cap(score, reached_token_cap=cap):
                raise RuntimeError(f"{cell}/{name}/{arm}: behavior score drift")
        teacher = row.get("teacher_forced", {})
        if set(teacher) != {
            "joint_token_ids", "joint_offsets", "target_token_positions",
            "target_token_ids", "target_character_interval", "manual_bos",
            "token_log_probabilities", "sum_log_probability", "mean_log_probability",
        }:
            raise RuntimeError(f"{cell}/{name}: teacher-forced key-set drift")
        expected_joint = frozen_ids[("orderops", name, "teacher_forced_joint")]
        if teacher.get("joint_token_ids") != expected_joint or teacher.get("manual_bos") is not False:
            raise RuntimeError(f"{cell}/{name}: teacher-forced token drift")
        expected_teacher = joint_target_tokenization(
            tokenizer, str(item["prompt"]), str(item["target"])
        )
        if any(teacher.get(key) != value for key, value in expected_teacher.items()):
            raise RuntimeError(f"{cell}/{name}: teacher-forced locator drift")
        positions = teacher.get("target_token_positions", [])
        target_ids = teacher.get("target_token_ids", [])
        if (not isinstance(positions, list) or not positions
                or any(not isinstance(position, int) or position <= 0
                       or position >= len(expected_joint) for position in positions)
                or target_ids != [expected_joint[position] for position in positions]
                or len(teacher.get("joint_offsets", [])) != len(expected_joint)):
            raise RuntimeError(f"{cell}/{name}: teacher-forced locator drift")
        logs = np.asarray(teacher.get("token_log_probabilities", []), dtype=float)
        if len(logs) != len(teacher.get("target_token_ids", [])) or not np.isfinite(logs).all():
            raise RuntimeError(f"{cell}/{name}: teacher-forced score drift")
        if not math.isclose(float(teacher.get("sum_log_probability")), float(logs.sum()), abs_tol=1e-10):
            raise RuntimeError(f"{cell}/{name}: teacher-forced sum drift")
        if not math.isclose(float(teacher.get("mean_log_probability")), float(logs.mean()), abs_tol=1e-10):
            raise RuntimeError(f"{cell}/{name}: teacher-forced mean drift")
    return rows


def validate_completed_cell_checkpoint(
    path: Path, *, cell: str, binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the exact completed-cell envelope before trusting its payloads."""
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{cell}: completed-cell checkpoint is missing or unsafe")
    checkpoint = json.loads(path.read_text())
    expected_keys = {
        "schema_version", "stage", "cell", "execution_binding",
        "array_sha256", "locator_sha256", "behavior_sha256", "cell_manifest",
        "completed_utc",
    }
    if set(checkpoint) != expected_keys:
        raise RuntimeError(f"{cell}: completed-cell checkpoint key-set drift")
    if (checkpoint.get("schema_version") != SCHEMA_VERSION
            or checkpoint.get("stage") != "orderops-cell-complete"
            or checkpoint.get("cell") != cell):
        raise RuntimeError(f"{cell}: completed-cell checkpoint identity drift")
    assert_binding(
        checkpoint.get("execution_binding", {}), binding,
        f"{cell} completed cell",
    )
    for key in ("array_sha256", "locator_sha256", "behavior_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(checkpoint.get(key, ""))) is None:
            raise RuntimeError(f"{cell}: completed-cell {key} is malformed")
    if not isinstance(checkpoint.get("cell_manifest"), dict):
        raise RuntimeError(f"{cell}: completed-cell manifest is malformed")
    if not isinstance(checkpoint.get("completed_utc"), str):
        raise RuntimeError(f"{cell}: completed-cell timestamp is malformed")
    return checkpoint


def expected_orderops_cell_manifest(
    *,
    cell: str,
    verified_hf_snapshot: Mapping[str, Any],
    lens_record: Mapping[str, Any],
    vocabulary_domain: Mapping[str, Any],
    device_fingerprint: Mapping[str, Any],
    boundary_ties: Mapping[str, int],
    typo_boundary_ties: int,
) -> dict[str, Any]:
    spec = MODEL_CELLS[cell]
    expected_tie_keys = set(ORDEROPS_READOUT_ARMS)
    if set(boundary_ties) != expected_tie_keys or any(
        not isinstance(value, int) or value < 0 for value in boundary_ties.values()
    ):
        raise RuntimeError(f"{cell}: boundary-tie summary is malformed")
    return {
        "model": spec["model"],
        "model_revision": spec["revision"],
        "verified_hf_snapshot": dict(verified_hf_snapshot),
        "forward_dtype": "bfloat16",
        "primary_readout_dtype": "float32",
        "precision_sensitivity_readout_dtype": "bfloat16",
        "lens_path": str(lens_record["path"]),
        "lens_size_bytes": int(lens_record["size_bytes"]),
        "lens_sha256": str(lens_record["sha256"]),
        "lens_n_prompts": int(spec["expected_lens_n_prompts"]),
        "source_layers": list(SOURCE_LAYERS),
        "source_bands": {key: list(value) for key, value in SOURCE_BANDS.items()},
        "vocabulary_domain": dict(vocabulary_domain),
        "ranked_common_vocabulary_includes_specials": True,
        "special_tokens_excluded_as_labels_only": True,
        "effective_force_bos": bool(spec["effective_force_bos"]),
        "effective_bos_token_id": spec["expected_bos_token_id"],
        "device": str(device_fingerprint["device"]),
        "device_fingerprint": dict(device_fingerprint),
        "boundary_ties": {key: int(boundary_ties[key]) for key in ORDEROPS_READOUT_ARMS},
        "typo_boundary_ties_fp32_j": int(typo_boundary_ties),
        "typo_control_n": 96,
        "behavior_n": 55,
        "behavior_manual_bos": False,
        "behavior_eos_and_pad_token_id": 151643,
        "behavior_generation_seed": ATOMIC_GENERATION_SEED,
    }


def run_orderops_stage(
    repo: Path,
    run_dir: Path,
    cache_dir: Path,
    input_hashes: Mapping[str, str],
    lens_paths: Mapping[str, Path],
    sealed_preflight: Mapping[str, Any],
) -> None:
    import torch
    import jlens
    from jlens import JacobianLens

    started_wall = time.time()
    started_utc = utc_now()
    binding = execution_binding(run_dir)
    if (run_dir / "RANKS_COMPLETE").exists():
        validate_completion_marker(
            repo, run_dir, "RANKS_COMPLETE", "orderops-rank-extraction"
        )
        if (run_dir / "INCOMPLETE").exists():
            (run_dir / "INCOMPLETE").unlink()
            fsync_parent_directory(run_dir / "INCOMPLETE")
        return

    inputs, _ = load_upstream_inputs(cache_dir)
    items = inputs["orderops"]
    official_order_items = list(items)
    typo_items = inputs["typo"]

    tokenizer_preflight = sealed_preflight["tokenizer_only_orderops"]
    model_snapshots = tokenizer_preflight["verified_hf_snapshots"]
    if not all(record.get("model_weights_verified") for record in model_snapshots.values()):
        raise RuntimeError("orderops stage lacks globally verified model weights")
    if not all(
        record.get("weight_header_verified")
        for record in tokenizer_preflight["architecture_receipts"].values()
    ):
        raise RuntimeError("orderops stage lacks global architecture-header verification")
    for cell_name, record in model_snapshots.items():
        current = verify_existing_hf_snapshot(
            repo, cell_name, Path(record["snapshot_path"]), include_model_weights=True
        )
        if current != record:
            raise RuntimeError(f"{cell_name}: global HF snapshot drift before any forward")
    verified_lenses = verify_lens_files(lens_paths)

    # Recompute from the two verified local tokenizers and exact-match the
    # globally frozen eligibility before either checkpoint is scored.
    from transformers import AutoTokenizer
    tokenizers = {
        cell: AutoTokenizer.from_pretrained(
            Path(model_snapshots[cell]["snapshot_path"]), local_files_only=True
        )
        for cell in MODEL_CELLS
    }
    # Both pinned heads are 151936 rows; this is rechecked after each model load.
    provisional_domains = {cell: tokenizer_domain(tok, 151936) for cell, tok in tokenizers.items()}
    eligibility = derive_common_orderops_eligibility(items, tokenizers, provisional_domains)
    typo_eligibility = derive_common_typo_eligibility(
        typo_items, tokenizers, provisional_domains
    )
    if eligibility != tokenizer_preflight["eligibility"]:
        raise RuntimeError("orderops eligibility differs from sealed verified-local preflight")
    if typo_eligibility != tokenizer_preflight["typo_eligibility"]:
        raise RuntimeError("typo eligibility differs from sealed verified-local preflight")
    eligibility_payload = {
        "orderops": eligibility, "typo": typo_eligibility,
    }
    eligibility_path = run_dir / "eligibility.json"
    if eligibility_path.exists():
        if eligibility_path.is_symlink() or json.loads(eligibility_path.read_text()) != eligibility_payload:
            raise RuntimeError("resumed eligibility payload drift")
    else:
        atomic_write_json(eligibility_path, eligibility_payload)
    retained_names = set(eligibility["retained_item_names"])
    items = [item for item in items if str(item["name"]) in retained_names]
    if [item["name"] for item in items] != eligibility["retained_item_names"]:
        raise RuntimeError("paired eligible item order drift")
    keys = eligibility["semantic_keys"]
    key_index = {key: idx for idx, key in enumerate(keys)}
    label_pairs = np.asarray(
        [[key_index[next(x for x in item["intermediates"] if x not in OPERATION_KEYS)],
          key_index[next(x for x in item["intermediates"] if x in OPERATION_KEYS)]]
         for item in items],
        dtype=np.int32,
    )
    item_names = [str(x["name"]) for x in items]
    typo_names = [str(x["name"]) for x in typo_items]
    typo_keys = typo_eligibility["semantic_keys"]
    typo_key_index = {key: index for index, key in enumerate(typo_keys)}
    typo_label_indices = np.asarray(
        [typo_key_index[str(item["intermediates"][0])] for item in typo_items],
        dtype=np.int32,
    )
    torch.manual_seed(ATOMIC_GENERATION_SEED)

    cell_manifests: dict[str, Any] = {}
    locators_by_cell: dict[str, list[dict[str, Any]]] = {}
    array_hashes: dict[str, str] = {}
    for cell_name, spec in MODEL_CELLS.items():
        array_path = run_dir / f"ranks__{cell_name}.npz"
        locator_cell_path = run_dir / f"locators__{cell_name}.json"
        checkpoint_path = run_dir / f"cell__{cell_name}.json"
        if checkpoint_path.exists():
            checkpoint = validate_completed_cell_checkpoint(
                checkpoint_path, cell=cell_name, binding=binding,
            )
            if checkpoint["cell_manifest"].get("verified_hf_snapshot") != model_snapshots[cell_name]:
                raise RuntimeError(f"{cell_name}: resumed HF snapshot provenance drift")
            if sha256_file(array_path) != checkpoint["array_sha256"]:
                raise RuntimeError(f"{cell_name}: completed rank checkpoint hash mismatch")
            if sha256_file(locator_cell_path) != checkpoint["locator_sha256"]:
                raise RuntimeError(f"{cell_name}: completed locator checkpoint hash mismatch")
            behavior_path = run_dir / f"behavior__{cell_name}.jsonl"
            if sha256_file(behavior_path) != checkpoint["behavior_sha256"]:
                raise RuntimeError(f"{cell_name}: completed behavior checkpoint hash mismatch")
            completed_arrays = load_npz_no_pickle(array_path)
            validate_orderops_rank_arrays(
                completed_arrays, cell=cell_name, expected_names=item_names,
                semantic_keys=keys, expected_label_pairs=label_pairs,
                typo_names=typo_names, typo_keys=typo_keys,
                expected_typo_labels=typo_label_indices,
                ranked_vocab_size=int(eligibility["common_ranked_vocab_size"]),
            )
            completed_boundary_ties = {
                arm: int(completed_arrays[f"boundary_ties__{arm}"])
                for arm in ORDEROPS_READOUT_ARMS
            }
            expected_cell_manifest = expected_orderops_cell_manifest(
                cell=cell_name,
                verified_hf_snapshot=model_snapshots[cell_name],
                lens_record=verified_lenses[cell_name],
                vocabulary_domain=provisional_domains[cell_name],
                device_fingerprint=sealed_preflight["device_fingerprint"],
                boundary_ties=completed_boundary_ties,
                typo_boundary_ties=int(
                    completed_arrays["typo_boundary_ties__fp32_j"]
                ),
            )
            if checkpoint["cell_manifest"] != expected_cell_manifest:
                raise RuntimeError(f"{cell_name}: completed-cell provenance drift")
            locator_bundle = json.loads(locator_cell_path.read_text())
            validate_locator_bundle(
                locator_bundle, cell=cell_name, order_names=item_names,
                typo_names=typo_names, tokenizer_preflight=tokenizer_preflight,
            )
            validate_behavior_payload(
                behavior_path, cell=cell_name, items=official_order_items,
                sealed_preflight=sealed_preflight,
                tokenizer=tokenizers[cell_name],
            )
            cell_manifests[cell_name] = checkpoint["cell_manifest"]
            locators_by_cell[cell_name] = locator_bundle
            array_hashes[array_path.name] = checkpoint["array_sha256"]
            array_hashes[locator_cell_path.name] = checkpoint["locator_sha256"]
            array_hashes[behavior_path.name] = checkpoint["behavior_sha256"]
            array_hashes[checkpoint_path.name] = sha256_file(checkpoint_path)
            continue

        preserve_unbound_files(
            run_dir,
            (array_path.name, locator_cell_path.name, f"behavior__{cell_name}.jsonl"),
            label=f"orderops-{cell_name}",
        )

        tokenizer, hf_model, dtype, device, snapshot_manifest = _load_hf_cell(
            repo, cell_name, expected_snapshot=model_snapshots[cell_name],
            expected_device=sealed_preflight["device_fingerprint"],
        )
        lens_path = lens_paths[cell_name]
        size = lens_path.stat().st_size
        if size != spec["lens_size_bytes"]:
            raise RuntimeError(f"{cell_name}: lens size mismatch {size}")
        got_lens = sha256_file(lens_path)
        if got_lens != spec["lens_sha256"]:
            raise RuntimeError(f"{cell_name}: lens hash mismatch {got_lens}")
        lens = JacobianLens.load(str(lens_path))
        if int(lens.n_prompts) != int(spec["expected_lens_n_prompts"]):
            raise RuntimeError(f"{cell_name}: lens prompt-count drift")
        lens_model = jlens.from_hf(hf_model, tokenizer)
        source_layers = list(lens.source_layers)
        if tuple(source_layers) != SOURCE_LAYERS:
            raise RuntimeError(f"{cell_name}: source-layer drift {source_layers}")
        if int(lens.d_model) != EXPECTED_D_MODEL or int(lens_model.d_model) != EXPECTED_D_MODEL:
            raise RuntimeError(f"{cell_name}: d_model drift")
        if int(lens_model.n_layers) != EXPECTED_N_LAYERS:
            raise RuntimeError(f"{cell_name}: layer-count drift")
        head_vocab = int(lens_model._lm_head.weight.shape[0])
        domain = tokenizer_domain(tokenizer, head_vocab)
        ranked_vocab = eligibility["common_ranked_vocab_size"]
        if domain != provisional_domains[cell_name] or ranked_vocab > domain["ranked_vocab_size"]:
            raise RuntimeError(f"{cell_name}: vocabulary-domain drift")
        jacobians = torch.stack(
            [lens.jacobians[layer].to(device=device, dtype=torch.float32) for layer in source_layers],
            dim=0,
        )

        # Primary readout is explicitly FP32 (D5 style); BF16 is sensitivity only.
        final_norm_fp32 = copy.deepcopy(lens_model._final_norm).float().to(device)
        lm_head_fp32 = copy.deepcopy(lens_model._lm_head).float().to(device)

        def unembed_fp32(x: Any) -> Any:
            return lm_head_fp32(final_norm_fp32(x.float()))

        ids_by_key = eligibility["token_ids_by_cell"][cell_name]
        banks: dict[str, list[np.ndarray]] = {name: [] for name in ORDEROPS_READOUT_ARMS}
        top25: dict[str, list[np.ndarray]] = {name: [] for name in banks}
        boundary_ties = {name: 0 for name in banks}
        locators: list[dict[str, Any]] = []
        partial_path = run_dir / f"partial__{cell_name}.npz"
        start_index = 0
        if partial_path.exists():
            banks, top25, boundary_ties, locators = validate_rank_partial(
                partial_path, cell=cell_name, expected_names=item_names,
                semantic_keys=keys, expected_label_pairs=label_pairs,
                ranked_vocab_size=int(ranked_vocab),
                binding=binding,
                expected_locators=tokenizer_preflight["locators"][cell_name],
            )
            start_index = len(locators)

        for item_index, item in enumerate(items[start_index:], start=start_index):
            locator = locate_joint_readout(
                tokenizer, item["prompt"], item["target"],
                force_bos=bool(spec["effective_force_bos"]),
            )
            expected_locator = tokenizer_preflight["locators"][cell_name][str(item["name"])]
            if locator != expected_locator:
                raise RuntimeError(f"{cell_name}/{item['name']}: sealed locator drift")
            residual = _capture_prefix_residuals(
                lens_model, locator["causal_prefix_token_ids"], source_layers, device
            )
            logits_by_arm = {
                "fp32_j": _readout_logits(
                    lens_model, residual, jacobians, ranked_vocab_size=ranked_vocab,
                    unembed=unembed_fp32, readout_dtype="float32",
                ),
                "fp32_logit": _readout_logits(
                    lens_model, residual, None, ranked_vocab_size=ranked_vocab,
                    unembed=unembed_fp32, readout_dtype="float32",
                ),
                "bf16_j": _readout_logits(
                    lens_model, residual, jacobians, ranked_vocab_size=ranked_vocab,
                    unembed=lens_model.unembed, readout_dtype="bfloat16",
                ),
                "bf16_logit": _readout_logits(
                    lens_model, residual, None, ranked_vocab_size=ranked_vocab,
                    unembed=lens_model.unembed, readout_dtype="bfloat16",
                ),
            }
            for arm, logits in logits_by_arm.items():
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f"{cell_name}/{item['name']}/{arm}: non-finite logits")
                banks[arm].append(_semantic_rank_bank_torch(logits, keys, ids_by_key))
                top, ties = _deterministic_top25_torch(logits)
                top25[arm].append(top)
                boundary_ties[arm] += ties
            locators.append({"name": item["name"], **locator})
            partial_arrays: dict[str, Any] = {
                "schema_version": np.asarray(SCHEMA_VERSION),
                "stage": np.asarray("orderops-rank-partial"),
                "cell": np.asarray(cell_name),
                "completed_item_names": np.asarray(item_names[:item_index + 1], dtype="U64"),
                "semantic_keys": np.asarray(keys, dtype="U32"),
                "source_layers": np.asarray(SOURCE_LAYERS, dtype=np.int32),
                "label_pairs": label_pairs,
                "execution_binding_json": np.asarray(
                    json.dumps(binding, sort_keys=True, separators=(",", ":"))
                ),
                "locators_json": np.asarray(
                    json.dumps(locators, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                ),
            }
            for arm in banks:
                partial_arrays[f"ranks__{arm}"] = np.stack(banks[arm]).astype(np.int32)
                partial_arrays[f"top25__{arm}"] = np.stack(top25[arm]).astype(np.int32)
                partial_arrays[f"boundary_ties__{arm}"] = np.asarray(
                    boundary_ties[arm], dtype=np.int64
                )
            partial_arrays["payload_sha256"] = np.asarray(array_bundle_digest(partial_arrays))
            partial_arrays["completed_prefix_digest"] = np.asarray(
                partial_completed_prefix_digest(
                    stage=f"orderops-rank:{cell_name}", binding=binding,
                    ordered_identifiers=item_names[:item_index + 1],
                    payload_sha256=str(partial_arrays["payload_sha256"].item()),
                )
            )
            atomic_savez(partial_path, **partial_arrays)
            banks, top25, boundary_ties, locators = validate_rank_partial(
                partial_path, cell=cell_name, expected_names=item_names,
                semantic_keys=keys, expected_label_pairs=label_pairs,
                ranked_vocab_size=int(ranked_vocab),
                binding=binding,
                expected_locators=tokenizer_preflight["locators"][cell_name],
            )

        # Same-runtime 96-item typo repeatability control.  It uses the same
        # BF16 forward, FP32 J transport, FP32 final norm/head and vocabulary.
        typo_ids_by_key = typo_eligibility["token_ids_by_cell"][cell_name]
        typo_ranks: list[np.ndarray] = []
        typo_top25: list[np.ndarray] = []
        typo_ties = 0
        typo_locators: list[dict[str, Any]] = []
        typo_partial_path = run_dir / f"partial_typo__{cell_name}.npz"
        typo_start = 0
        if typo_partial_path.exists():
            typo_ranks, typo_top25, typo_ties, typo_locators = validate_typo_partial(
                typo_partial_path, cell=cell_name, expected_names=typo_names,
                semantic_keys=typo_keys, expected_label_indices=typo_label_indices,
                ranked_vocab_size=int(ranked_vocab),
                binding=binding,
                expected_locators=tokenizer_preflight["typo_locators"][cell_name],
            )
            typo_start = len(typo_locators)
        for typo_index, item in enumerate(typo_items[typo_start:], start=typo_start):
            locator = locate_final_prompt_readout(
                tokenizer, str(item["prompt"]),
                force_bos=bool(spec["effective_force_bos"]),
            )
            expected_locator = tokenizer_preflight["typo_locators"][cell_name][str(item["name"])]
            if locator != expected_locator:
                raise RuntimeError(f"{cell_name}/{item['name']}: sealed typo locator drift")
            residual = _capture_prefix_residuals(
                lens_model, locator["causal_prefix_token_ids"], source_layers, device
            )
            logits = _readout_logits(
                lens_model, residual, jacobians, ranked_vocab_size=ranked_vocab,
                unembed=unembed_fp32, readout_dtype="float32",
            )
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"{cell_name}/{item['name']}/typo: non-finite logits")
            typo_ranks.append(_semantic_rank_bank_torch(logits, typo_keys, typo_ids_by_key))
            top, ties = _deterministic_top25_torch(logits)
            typo_top25.append(top)
            typo_ties += ties
            typo_locators.append({"name": item["name"], **locator})
            typo_partial_arrays = {
                "schema_version": np.asarray(SCHEMA_VERSION),
                "stage": np.asarray("typo-rank-partial"),
                "cell": np.asarray(cell_name),
                "completed_item_names": np.asarray(
                    [str(x["name"]) for x in typo_items[:typo_index + 1]], dtype="U64"
                ),
                "semantic_keys": np.asarray(typo_keys, dtype="U64"),
                "source_layers": np.asarray(SOURCE_LAYERS, dtype=np.int32),
                "label_indices": typo_label_indices,
                "ranks": np.stack(typo_ranks).astype(np.int32),
                "top25": np.stack(typo_top25).astype(np.int32),
                "boundary_ties": np.asarray(typo_ties, dtype=np.int64),
                "execution_binding_json": np.asarray(
                    json.dumps(binding, sort_keys=True, separators=(",", ":"))
                ),
                "locators_json": np.asarray(
                    json.dumps(typo_locators, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                ),
            }
            typo_partial_arrays["payload_sha256"] = np.asarray(
                array_bundle_digest(typo_partial_arrays)
            )
            typo_partial_arrays["completed_prefix_digest"] = np.asarray(
                partial_completed_prefix_digest(
                    stage=f"typo-rank:{cell_name}", binding=binding,
                    ordered_identifiers=[
                        str(x["name"]) for x in typo_items[:typo_index + 1]
                    ],
                    payload_sha256=str(
                        typo_partial_arrays["payload_sha256"].item()
                    ),
                )
            )
            atomic_savez(typo_partial_path, **typo_partial_arrays)
            typo_ranks, typo_top25, typo_ties, typo_locators = validate_typo_partial(
                typo_partial_path, cell=cell_name, expected_names=typo_names,
                semantic_keys=typo_keys, expected_label_indices=typo_label_indices,
                ranked_vocab_size=int(ranked_vocab),
                binding=binding,
                expected_locators=tokenizer_preflight["typo_locators"][cell_name],
            )

        arrays = {name: np.stack(rows).astype(np.int32) for name, rows in banks.items()}
        arrays.update({f"top25__{name}": np.stack(rows).astype(np.int32)
                       for name, rows in top25.items()})
        arrays.update({f"boundary_ties__{name}": np.asarray(value, dtype=np.int64)
                       for name, value in boundary_ties.items()})
        arrays["label_pairs"] = label_pairs
        arrays["item_names"] = np.asarray(item_names, dtype="U64")
        arrays["semantic_keys"] = np.asarray(keys, dtype="U32")
        arrays["typo_fp32_j"] = np.stack(typo_ranks).astype(np.int32)
        arrays["typo_top25__fp32_j"] = np.stack(typo_top25).astype(np.int32)
        arrays["typo_boundary_ties__fp32_j"] = np.asarray(typo_ties, dtype=np.int64)
        arrays["typo_label_indices"] = typo_label_indices
        arrays["typo_item_names"] = np.asarray(
            [str(item["name"]) for item in typo_items], dtype="U64"
        )
        arrays["typo_semantic_keys"] = np.asarray(typo_keys, dtype="U64")
        validate_orderops_rank_arrays(
            arrays, cell=cell_name, expected_names=item_names,
            semantic_keys=keys, expected_label_pairs=label_pairs,
            typo_names=typo_names, typo_keys=typo_keys,
            expected_typo_labels=typo_label_indices,
            ranked_vocab_size=int(ranked_vocab),
        )
        behavior_path, behavior_records = _run_orderops_behavior_cell(
            run_dir, cell_name, hf_model, tokenizer, official_order_items,
            device=device, binding=binding, sealed_preflight=sealed_preflight,
        )
        validate_behavior_payload(
            behavior_path, cell=cell_name, items=official_order_items,
            sealed_preflight=sealed_preflight, tokenizer=tokenizer,
        )
        atomic_savez(array_path, **arrays)
        validate_orderops_rank_arrays(
            load_npz_no_pickle(array_path), cell=cell_name, expected_names=item_names,
            semantic_keys=keys, expected_label_pairs=label_pairs,
            typo_names=typo_names, typo_keys=typo_keys,
            expected_typo_labels=typo_label_indices,
            ranked_vocab_size=int(ranked_vocab),
        )
        array_hashes[array_path.name] = sha256_file(array_path)
        locator_bundle = {"orderops": locators, "typo": typo_locators}
        validate_locator_bundle(
            locator_bundle, cell=cell_name, order_names=item_names,
            typo_names=typo_names, tokenizer_preflight=tokenizer_preflight,
        )
        atomic_write_json(locator_cell_path, locator_bundle)
        array_hashes[locator_cell_path.name] = sha256_file(locator_cell_path)
        array_hashes[behavior_path.name] = sha256_file(behavior_path)
        locators_by_cell[cell_name] = locator_bundle
        cell_manifest = expected_orderops_cell_manifest(
            cell=cell_name, verified_hf_snapshot=snapshot_manifest,
            lens_record=verified_lenses[cell_name], vocabulary_domain=domain,
            device_fingerprint=sealed_preflight["device_fingerprint"],
            boundary_ties=boundary_ties, typo_boundary_ties=typo_ties,
        )
        cell_manifests[cell_name] = cell_manifest
        atomic_write_json(checkpoint_path, {
            "schema_version": SCHEMA_VERSION,
            "stage": "orderops-cell-complete",
            "cell": cell_name,
            "execution_binding": binding,
            "array_sha256": array_hashes[array_path.name],
            "locator_sha256": array_hashes[locator_cell_path.name],
            "behavior_sha256": array_hashes[behavior_path.name],
            "cell_manifest": cell_manifest,
            "completed_utc": utc_now(),
        })
        array_hashes[checkpoint_path.name] = sha256_file(checkpoint_path)
        del lens, lens_model, hf_model, final_norm_fp32, lm_head_fp32, jacobians
        if device == "cuda":
            torch.cuda.empty_cache()

    preserve_unbound_files(
        run_dir,
        ("joint_readout_locators.json", "manifest.json",
         "rank_derivation_manifest.json"),
        label="orderops-joint",
    )
    locator_path = run_dir / "joint_readout_locators.json"
    atomic_write_json(locator_path, locators_by_cell)
    array_hashes[locator_path.name] = sha256_file(locator_path)
    array_hashes[eligibility_path.name] = sha256_file(eligibility_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "orderops-rank-extraction",
        "seal": json.loads((run_dir / "seal.json").read_text()),
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "wall_seconds": round(time.time() - started_wall, 3),
        "jlens_commit": JLENS_COMMIT,
        "upstream_file_sha256": dict(input_hashes),
        "n_items_official_source": 55,
        "n_items_token_eligible_extraction": 54,
        "n_items_annotation_valid_sensitivity": 53,
        "n_items_non_lexically_present_primary": 50,
        "prompt_copy_exclusions": sorted(ORDEROPS_LEXICAL_EXCLUSIONS),
        "annotation_exclusions": sorted(ORDEROPS_ANNOTATION_EXCLUSIONS),
        "primary_exclusions": sorted(ORDEROPS_PRIMARY_EXCLUSIONS),
        "operation_endpoint_status": "prompt-present context positive control; never reasoning endpoint",
        "readout_locator": "joint prompt+target offsets; predecessor of first target-overlap token",
        "effective_bos_is_asymmetric": True,
        "behavior_generation_seed": ATOMIC_GENERATION_SEED,
        "device_fingerprint": sealed_preflight["device_fingerprint"],
        "eligibility_sha256": sha256_file(run_dir / "eligibility.json"),
        "cells": cell_manifests,
        "payload_sha256": array_hashes,
        "analysis_status": "not calculated by this stage",
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "jlens": getattr(jlens, "__version__", "unknown"),
            "transformers": __import__("transformers").__version__,
            "tokenizers": __import__("tokenizers").__version__,
            "platform": platform.platform(),
        },
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    # Re-load every final payload and compare all registered structure before
    # allowing a completion marker.
    _load_and_verify_orderops_bundle(
        run_dir, sealed_preflight=sealed_preflight,
        orderops_items=official_order_items, typo_items=typo_items,
    )
    derivation = write_derivation_manifest(
        repo, run_dir, stage="orderops-rank-extraction",
        extra_inputs=[
            *lens_paths.values(), *snapshot_input_paths(model_snapshots),
            *fetched_input_paths(run_dir), *campaign_input_paths(run_dir),
            stage_selection_path(run_dir),
        ],
        filename="rank_derivation_manifest.json",
    )
    finalize_stage(
        repo, run_dir, stage="orderops-rank-extraction", marker_name="RANKS_COMPLETE",
        manifest_name="manifest.json", derivation_name=derivation.name,
    )


def _load_and_verify_orderops_bundle(
    run_dir: Path,
    *,
    sealed_preflight: Mapping[str, Any] | None = None,
    orderops_items: Sequence[Mapping[str, Any]] | None = None,
    typo_items: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    reject_symlinks_beneath(run_dir)
    if sealed_preflight is None:
        sealed_preflight = json.loads((run_dir / "preflight.json").read_text())
    if orderops_items is None:
        payload = json.loads((run_dir / "input_files" / "lens-eval-order-ops.json").read_text())
        orderops_items = payload["items"]
    if typo_items is None:
        payload = json.loads((run_dir / "input_files" / "lens-eval-typo.json").read_text())
        typo_items = payload["items"]
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("orderops manifest is missing or unsafe")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("stage") != "orderops-rank-extraction"):
        raise RuntimeError("not an order-ops rank-extraction bundle")
    if manifest.get("device_fingerprint") != sealed_preflight["device_fingerprint"]:
        raise RuntimeError("rank manifest device fingerprint drift")
    tokenizer_preflight = sealed_preflight["tokenizer_only_orderops"]
    snapshot_records = tokenizer_preflight["verified_hf_snapshots"]
    from transformers import AutoTokenizer
    validation_tokenizers: dict[str, Any] = {}
    for cell in MODEL_CELLS:
        snapshot_path = Path(str(snapshot_records[cell]["snapshot_path"]))
        current_snapshot = verify_existing_hf_snapshot(
            Path(__file__).resolve().parent, cell, snapshot_path,
            include_model_weights=True,
        )
        if current_snapshot != snapshot_records[cell]:
            raise RuntimeError(f"{cell}: verified HF snapshot drift in bundle validation")
        architecture = verify_snapshot_architecture(
            snapshot_path, cell, include_model_weights=True
        )
        if not architecture.get("weight_header_verified"):
            raise RuntimeError(f"{cell}: architecture header is not verified")
        validation_tokenizers[cell] = AutoTokenizer.from_pretrained(
            snapshot_path, local_files_only=True
        )
    eligibility = tokenizer_preflight["eligibility"]
    typo_eligibility = tokenizer_preflight["typo_eligibility"]
    expected_eligibility = {"orderops": eligibility, "typo": typo_eligibility}
    eligibility_path = run_dir / "eligibility.json"
    if json.loads(eligibility_path.read_text()) != expected_eligibility:
        raise RuntimeError("final eligibility payload drift")
    if sha256_file(eligibility_path) != manifest.get("eligibility_sha256"):
        raise RuntimeError("final eligibility digest drift")
    expected_payload_names = {
        "eligibility.json", "joint_readout_locators.json",
        *{f"ranks__{cell}.npz" for cell in MODEL_CELLS},
        *{f"locators__{cell}.json" for cell in MODEL_CELLS},
        *{f"behavior__{cell}.jsonl" for cell in MODEL_CELLS},
        *{f"cell__{cell}.json" for cell in MODEL_CELLS},
    }
    payload_hashes = manifest.get("payload_sha256")
    if not isinstance(payload_hashes, dict) or set(payload_hashes) != expected_payload_names:
        raise RuntimeError("rank manifest payload table is incomplete or has extras")
    for name, expected_hash in payload_hashes.items():
        path = run_dir / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"rank payload hash mismatch: {name}")

    retained = set(eligibility["retained_item_names"])
    scored_items = [item for item in orderops_items if str(item["name"]) in retained]
    item_names = [str(item["name"]) for item in scored_items]
    typo_names = [str(item["name"]) for item in typo_items]
    keys = eligibility["semantic_keys"]
    key_index = {key: index for index, key in enumerate(keys)}
    label_pairs = np.asarray([
        [key_index[next(x for x in item["intermediates"] if x not in OPERATION_KEYS)],
         key_index[next(x for x in item["intermediates"] if x in OPERATION_KEYS)]]
        for item in scored_items
    ], dtype=np.int32)
    typo_keys = typo_eligibility["semantic_keys"]
    typo_key_index = {key: index for index, key in enumerate(typo_keys)}
    typo_labels = np.asarray([
        typo_key_index[str(item["intermediates"][0])] for item in typo_items
    ], dtype=np.int32)
    binding = execution_binding(run_dir)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    locators: dict[str, Any] = {}
    for cell in MODEL_CELLS:
        array_path = run_dir / f"ranks__{cell}.npz"
        arrays[cell] = load_npz_no_pickle(array_path)
        validate_orderops_rank_arrays(
            arrays[cell], cell=cell, expected_names=item_names,
            semantic_keys=keys, expected_label_pairs=label_pairs,
            typo_names=typo_names, typo_keys=typo_keys,
            expected_typo_labels=typo_labels,
            ranked_vocab_size=int(eligibility["common_ranked_vocab_size"]),
        )
        locator_path = run_dir / f"locators__{cell}.json"
        locators[cell] = json.loads(locator_path.read_text())
        validate_locator_bundle(
            locators[cell], cell=cell, order_names=item_names,
            typo_names=typo_names, tokenizer_preflight=tokenizer_preflight,
        )
        behavior_path = run_dir / f"behavior__{cell}.jsonl"
        validate_behavior_payload(
            behavior_path, cell=cell, items=orderops_items,
            sealed_preflight=sealed_preflight,
            tokenizer=validation_tokenizers[cell],
        )
        checkpoint = validate_completed_cell_checkpoint(
            run_dir / f"cell__{cell}.json", cell=cell, binding=binding,
        )
        if checkpoint.get("array_sha256") != sha256_file(array_path):
            raise RuntimeError(f"{cell}: cell checkpoint rank hash drift")
        if checkpoint.get("locator_sha256") != sha256_file(locator_path):
            raise RuntimeError(f"{cell}: cell checkpoint locator hash drift")
        if checkpoint.get("behavior_sha256") != sha256_file(behavior_path):
            raise RuntimeError(f"{cell}: cell checkpoint behavior hash drift")
        cell_manifest = checkpoint.get("cell_manifest", {})
        expected_cell_manifest = expected_orderops_cell_manifest(
            cell=cell,
            verified_hf_snapshot=tokenizer_preflight["verified_hf_snapshots"][cell],
            lens_record=sealed_preflight["lenses"][cell],
            vocabulary_domain=tokenizer_preflight["domains"][cell],
            device_fingerprint=sealed_preflight["device_fingerprint"],
            boundary_ties={
                arm: int(arrays[cell][f"boundary_ties__{arm}"])
                for arm in ORDEROPS_READOUT_ARMS
            },
            typo_boundary_ties=int(
                arrays[cell]["typo_boundary_ties__fp32_j"]
            ),
        )
        if cell_manifest != expected_cell_manifest:
            raise RuntimeError(f"{cell}: cell manifest does not match frozen provenance")
        if cell_manifest != manifest.get("cells", {}).get(cell):
            raise RuntimeError(f"{cell}: cell manifest drift")
        if cell_manifest.get("device_fingerprint") != sealed_preflight["device_fingerprint"]:
            raise RuntimeError(f"{cell}: cell device fingerprint drift")
        if (cell_manifest.get("verified_hf_snapshot")
                != tokenizer_preflight["verified_hf_snapshots"][cell]):
            raise RuntimeError(f"{cell}: cell HF snapshot drift")
    combined = json.loads((run_dir / "joint_readout_locators.json").read_text())
    if combined != locators:
        raise RuntimeError("combined locator payload drift")
    for key in (
        "item_names", "semantic_keys", "label_pairs", "typo_item_names",
        "typo_semantic_keys", "typo_label_indices",
    ):
        if not np.array_equal(arrays["base"][key], arrays["distill"][key]):
            raise RuntimeError(f"paired payload drift in {key}")
    return manifest, arrays


def _endpoint_ranks(bank: np.ndarray, label_pairs: np.ndarray, component: int) -> np.ndarray:
    item_indices = np.arange(bank.shape[0])[:, None]
    layer_indices = np.arange(bank.shape[1])[None, :]
    keys = label_pairs[:, component][:, None]
    return bank[item_indices, layer_indices, keys]


def summarize_orderops_behavior(run_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    records: dict[str, list[dict[str, Any]]] = {}
    for cell in MODEL_CELLS:
        path = run_dir / f"behavior__{cell}.jsonl"
        expected_hash = manifest["payload_sha256"][path.name]
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"behaviour payload hash mismatch: {cell}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 55 or len({row["name"] for row in rows}) != 55:
            raise RuntimeError(f"{cell}: behaviour payload must have 55 unique items")
        records[cell] = rows
    if [x["name"] for x in records["base"]] != [x["name"] for x in records["distill"]]:
        raise RuntimeError("paired behaviour item order drift")

    summaries: dict[str, Any] = {}
    cot_vectors: dict[str, np.ndarray] = {}
    direct_vectors: dict[str, np.ndarray] = {}
    for cell, rows in records.items():
        direct = np.asarray([row["direct"]["score"]["exact_primary"] for row in rows], dtype=bool)
        cot = np.asarray(
            [row["cot"]["score"]["bounded_final_line_primary"] for row in rows], dtype=bool
        )
        cot_anywhere = np.asarray(
            [row["cot"]["score"]["bounded_anywhere_sensitivity"] for row in rows], dtype=bool
        )
        teacher_sum = np.asarray(
            [row["teacher_forced"]["sum_log_probability"] for row in rows], dtype=float
        )
        teacher_mean = np.asarray(
            [row["teacher_forced"]["mean_log_probability"] for row in rows], dtype=float
        )
        if not np.isfinite(teacher_sum).all() or not np.isfinite(teacher_mean).all():
            raise RuntimeError(f"{cell}: non-finite teacher-forced behavior summary")
        direct_vectors[cell], cot_vectors[cell] = direct, cot
        summaries[cell] = {
            "n_paired_complete": len(rows),
            "direct_first_line_exact_rate": float(direct.mean()),
            "cot_final_line_bounded_rate": float(cot.mean()),
            "cot_anywhere_bounded_sensitivity_rate": float(cot_anywhere.mean()),
            "teacher_forced_sum_logprob_mean_over_items": float(teacher_sum.mean()),
            "teacher_forced_mean_token_logprob_mean_over_items": float(teacher_mean.mean()),
            "reasoning_competence_at_least_0_80": bool(cot.mean() >= 0.80),
        }
    common = cot_vectors["base"] & cot_vectors["distill"]
    return {
        "behavior_never_filters_readout": True,
        "manual_bos": False,
        "eos_and_pad_token_id": 151643,
        "cells": summaries,
        "both_checkpoints_cot_at_least_0_80": bool(
            all(summaries[cell]["reasoning_competence_at_least_0_80"] for cell in MODEL_CELLS)
        ),
        "common_cot_correct_subset": {
            "n_items": int(common.sum()),
            "base_direct_rate": float(direct_vectors["base"][common].mean()) if common.any() else None,
            "distill_direct_rate": float(direct_vectors["distill"][common].mean()) if common.any() else None,
        },
    }


def analyse_orderops_bundle(run_dir: Path) -> dict[str, Any]:
    manifest, arrays = _load_and_verify_orderops_bundle(run_dir)
    item_names = arrays["base"]["item_names"].astype(str)
    label_pairs = arrays["base"]["label_pairs"].astype(np.int64)
    analysis_mask = ~np.isin(item_names, list(ORDEROPS_ANNOTATION_EXCLUSIONS))
    if int(analysis_mask.sum()) != 53:
        raise RuntimeError("annotation-valid analysis mask drift")
    primary_mask = ~np.isin(item_names, list(ORDEROPS_PRIMARY_EXCLUSIONS))
    if int(primary_mask.sum()) != 50:
        raise RuntimeError("annotation-valid non-lexically-present primary mask drift")
    if not np.array_equal(
        arrays["base"]["typo_label_indices"], arrays["distill"]["typo_label_indices"]
    ):
        raise RuntimeError("paired typo label-index drift")
    typo_control = synchronized_typo_union_null(
        {
            "base": arrays["base"]["typo_fp32_j"],
            "distill": arrays["distill"]["typo_fp32_j"],
        },
        arrays["base"]["typo_label_indices"],
    )
    if not typo_control["integrity_pass"]:
        return {
            "schema_version": SCHEMA_VERSION,
            "stage": "orderops-analysis",
            "source_manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "same_runtime_typo_control": typo_control,
            "analysis_stopped_before_checkpoint_inference": True,
            "co_primary": None,
            "pair_pattern": None,
            "evidence_status": "instrument-invalid-in-this-runtime",
            "interpretation": "not a negative order-operations result",
            "finished_utc": utc_now(),
        }

    item_metrics: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    summaries: dict[str, Any] = {}
    reporting_bands = {**SOURCE_BANDS, "all": SOURCE_LAYERS}
    for precision in ("fp32", "bf16"):
        item_metrics[precision] = {}
        summaries[precision] = {}
        for cell in MODEL_CELLS:
            item_metrics[precision][cell] = {}
            summaries[precision][cell] = {}
            for method in ("j", "logit"):
                bank = arrays[cell][f"{precision}_{method}"]
                numeric = _endpoint_ranks(bank, label_pairs, 0)
                operation = _endpoint_ranks(bank, label_pairs, 1)
                numeric_mid = per_item_layer_persistence_at_25(numeric, SOURCE_BANDS["mid"])
                operation_mid = per_item_layer_persistence_at_25(operation, SOURCE_BANDS["mid"])
                item_metrics[precision][cell][method] = {
                    "numeric_mid": numeric_mid,
                    "operation_mid": operation_mid,
                    "numeric_ranks": numeric,
                    "operation_ranks": operation,
                }
                band_outputs: dict[str, Any] = {}
                for band, layers in reporting_bands.items():
                    layer_ids = np.asarray(layers, dtype=np.int64)
                    numeric_hits = numeric[:, layer_ids] <= K
                    operation_hits = operation[:, layer_ids] <= K
                    numeric_persistence = numeric_hits.mean(axis=1)
                    operation_persistence = operation_hits.mean(axis=1)
                    numeric_union = numeric_hits.any(axis=1).astype(float)
                    operation_union = operation_hits.any(axis=1).astype(float)
                    best_numeric_rank = numeric[:, layer_ids].min(axis=1)
                    q25, median, q75 = np.quantile(
                        best_numeric_rank[analysis_mask],
                        [0.25, 0.5, 0.75], method="linear"
                    )
                    band_outputs[band] = {
                        "numeric_layer_persistence_primary50": float(
                            numeric_persistence[primary_mask].mean()
                        ),
                        "numeric_layer_persistence_annotation_valid53": float(
                            numeric_persistence[analysis_mask].mean()
                        ),
                        "numeric_band_union_primary50": float(
                            numeric_union[primary_mask].mean()
                        ),
                        "numeric_band_union_annotation_valid53": float(
                            numeric_union[analysis_mask].mean()
                        ),
                        "operation_layer_persistence_context_control_annotation_valid53": float(
                            operation_persistence[analysis_mask].mean()
                        ),
                        "operation_band_union_context_control_annotation_valid53": float(
                            operation_union[analysis_mask].mean()
                        ),
                        "official_two_intermediate_band_union_annotation_valid53": float(
                            ((numeric_union[analysis_mask]
                              + operation_union[analysis_mask]) / 2.0).mean()
                        ),
                        "numeric_best_rank_annotation_valid53": {
                            "median": float(median), "q25": float(q25), "q75": float(q75)
                        },
                        "numeric_band_mean_reciprocal_rank_annotation_valid53": float(
                            np.mean(1.0 / numeric[analysis_mask][:, layer_ids])
                        ),
                    }
                summaries[precision][cell][method] = {
                    "numeric_non_lexically_present_mid_layer_persistence_at_25": float(
                        numeric_mid[primary_mask].mean()
                    ),
                    "numeric_annotation_valid53_mid_layer_persistence_at_25_sensitivity": float(
                        numeric_mid[analysis_mask].mean()
                    ),
                    "official55_accounting": {
                        "scored_annotation_valid": 53,
                        "not_available_common_token": ["mult-div-left"],
                        "excluded_annotation_mismatch": ["mult-div-mult"],
                        "token_eligible_extraction_retained": 54,
                        "imputed": False,
                    },
                    "operation_context_control_mid_layer_persistence_at_25": float(
                        operation_mid[analysis_mask].mean()
                    ),
                    "numeric_non_lexically_present_band_layer_persistence_at_25": {
                        band: float(per_item_layer_persistence_at_25(numeric, layers)[primary_mask].mean())
                        for band, layers in SOURCE_BANDS.items()
                    },
                    "bands": band_outputs,
                }
            summaries[precision][cell]["j_minus_logit_numeric_persistence_primary50"] = {
                band: float(
                    per_item_layer_persistence_at_25(
                        item_metrics[precision][cell]["j"]["numeric_ranks"], layers
                    )[primary_mask].mean()
                    - per_item_layer_persistence_at_25(
                        item_metrics[precision][cell]["logit"]["numeric_ranks"], layers
                    )[primary_mask].mean()
                )
                for band, layers in reporting_bands.items()
            }

    # Primary checkpoint contrast: FP32 J-lens numeric mid-band layer persistence@25.
    base_primary = item_metrics["fp32"]["base"]["j"]["numeric_mid"][primary_mask]
    distill_primary = item_metrics["fp32"]["distill"]["j"]["numeric_mid"][primary_mask]
    bootstrap = paired_bootstrap_mean_difference(distill_primary, base_primary)
    swap = checkpoint_vector_swap_test(distill_primary - base_primary)

    # Secondary transport-specific difference-in-differences, same whole-item swap.
    base_did = (
        item_metrics["fp32"]["base"]["j"]["numeric_mid"]
        - item_metrics["fp32"]["base"]["logit"]["numeric_mid"]
    )[primary_mask]
    distill_did = (
        item_metrics["fp32"]["distill"]["j"]["numeric_mid"]
        - item_metrics["fp32"]["distill"]["logit"]["numeric_mid"]
    )[primary_mask]
    did_bootstrap = paired_bootstrap_mean_difference(distill_did, base_did)
    did_swap = checkpoint_vector_swap_test(distill_did - base_did)
    adjusted = holm_adjust_two(swap["two_sided_p"], did_swap["two_sided_p"])
    swap["holm_adjusted_two_sided_p"] = adjusted[0]
    did_swap["holm_adjusted_two_sided_p"] = adjusted[1]

    rank_banks = {
        f"{cell}_{method}": arrays[cell][f"fp32_{method}"][primary_mask]
        for cell in MODEL_CELLS for method in ("j", "logit")
    }
    label_null_primary = synchronized_whole_pair_permutation_test(
        rank_banks, label_pairs[primary_mask]
    )
    assay_gate = synchronized_numeric_assay_gate(
        arrays["base"]["fp32_j"][primary_mask],
        arrays["distill"]["fp32_j"][primary_mask],
        label_pairs[primary_mask],
    )
    behavior = summarize_orderops_behavior(run_dir, manifest)
    margin = 0.10
    recurrence = (assay_gate["pass"] and bootstrap["percentile_95_ci"][1] < -margin)
    transport_specific = recurrence and did_bootstrap["percentile_95_ci"][1] < -margin
    equivalent = (
        assay_gate["pass"]
        and bootstrap["percentile_90_ci"][0] > -margin
        and bootstrap["percentile_90_ci"][1] < margin
        and did_bootstrap["percentile_90_ci"][0] > -margin
        and did_bootstrap["percentile_90_ci"][1] < margin
    )
    if not assay_gate["pass"]:
        pair_pattern = "assay-inconclusive"
    elif transport_specific:
        pair_pattern = "material-recurrence-transport-specific"
    elif recurrence:
        pair_pattern = "material-recurrence-transport-specificity-not-established"
    elif equivalent:
        pair_pattern = "practical-equivalence"
    else:
        pair_pattern = "indeterminate"
    behavioral_claims_suspended = not behavior["both_checkpoints_cot_at_least_0_80"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": "orderops-analysis",
        "source_manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "independent_unit": "official order-operations item",
        "bands": {key: list(value) for key, value in SOURCE_BANDS.items()},
        "co_primary": {
            "delta_j": {
            "population": "50 annotation-valid, non-lexically-present, paired-token-eligible numeric items",
            "excluded_prompt_leakage_items": sorted(ORDEROPS_LEXICAL_EXCLUSIONS),
            "excluded_annotation_mismatch_items": sorted(
                ORDEROPS_ANNOTATION_EXCLUSIONS
            ),
            "common_token_exclusion": ["mult-div-left"],
            "estimand": "paired mean(distill - base) FP32 J-lens mid-band layer persistence@25",
            "bootstrap": bootstrap,
            "checkpoint_vector_swap": swap,
            },
            "j_minus_logit_difference_in_differences": {
            "estimand": "paired mean[(J-logit)_distill - (J-logit)_base] mid-band layer persistence@25",
            "bootstrap": did_bootstrap,
            "checkpoint_vector_swap": did_swap,
            },
        },
        "assay_presence_gate": assay_gate,
        "same_runtime_typo_control": typo_control,
        "behavioral_competence": behavior,
        "pair_pattern": pair_pattern,
        "reasoning_competence_claims_suspended": behavioral_claims_suspended,
        "practical_equivalence_margin": [-margin, margin],
        "whole_label_pair_permutation": label_null_primary,
        "summaries": summaries,
        "precision_status": {
            "fp32": "primary D5-style final norm and LM head",
            "bf16": "ordinary unembed sensitivity only",
        },
        "operation_status": "prompt-present context positive control only",
        "finished_utc": utc_now(),
    }
    return report


def preflight(
    repo: Path,
    cache_dir: Path,
    seal_commit: str | None,
    outroot: Path,
    lens_paths: Mapping[str, Path],
) -> dict[str, Any]:
    cache_dir = validate_mutable_cache_dir(repo, cache_dir, outroot)
    seal = None
    pure_tests = None
    if seal_commit:
        seal = validate_sealed_checkout(repo, seal_commit, allowed_untracked_root=outroot)
        pure_tests = run_sealed_pure_tests(repo)
    inputs, hashes = load_upstream_inputs(cache_dir)
    if sha256_file(repo / ELIGIBILITY_MANIFEST) != ELIGIBILITY_SHA256:
        raise RuntimeError("pinned R1 eligibility manifest hash mismatch")
    lenses = verify_lens_files(lens_paths)
    eligibility = json.loads((repo / ELIGIBILITY_MANIFEST).read_text())
    atomic_tasks, frozen_atomic = load_frozen_atomic_tasks(
        repo, inputs["multihop"], eligibility
    )
    tokenizer_preflight = tokenizer_only_orderops_preflight(
        repo, inputs["orderops"], inputs["typo"], atomic_tasks,
        include_model_weights=bool(seal_commit),
        hf_cache_dir=cache_dir / "huggingface",
    )
    import torch
    device_fingerprint = bind_observed_model_dtype(
        execution_device_fingerprint(torch),
        tokenizer_preflight["architecture_receipts"],
        require_weight_headers=bool(seal_commit),
    )
    if (device_fingerprint["backend"] not in {"cuda", "mps"}
            or not device_fingerprint["bf16_execution_supported"]):
        raise RuntimeError("selected accelerator lacks the registered BF16 execution support")
    if seal_commit:
        # Downloads and every other mutable cache are required to live outside
        # the checkout.  Recheck immediately after all global pre-forward gates.
        second_seal = validate_sealed_checkout(
            repo, seal_commit, allowed_untracked_root=outroot
        )
        if second_seal != seal:
            raise RuntimeError("sealed checkout changed during preflight")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "preflight-pass",
        "model_loaded": False,
        "runtime_versions": verify_runtime_environment(repo),
        "seal": seal,
        "pure_tests": pure_tests,
        "device_fingerprint": device_fingerprint,
        "mutable_cache_dir": str(cache_dir),
        "upstream_file_sha256": hashes,
        "populations": {key: len(value) for key, value in inputs.items()},
        "orderops_non_lexically_present_task_n_before_annotation_audit": 52,
        "orderops_annotation_valid_primary_n_before_tokenization": 51,
        "orderops_anticipated_common_eligible_primary_n": 50,
        "orderops_anticipated_annotation_valid_sensitivity_n": 53,
        "frozen_atomic_tasks": {
            "path": ATOMIC_TASK_FILE.as_posix(),
            "sha256": sha256_file(repo / ATOMIC_TASK_FILE),
            "n_items": len(frozen_atomic["items"]),
            "wrapper_matches_protocol": frozen_atomic["wrapper"] == ATOMIC_WRAPPER,
        },
        "tokenizer_only_orderops": tokenizer_preflight,
        "lenses": lenses,
    }


def default_cache_dir() -> Path:
    return Path(tempfile.gettempdir()) / "rom-jspace-1p5b-pinned-inputs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--outroot", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--base-lens", type=Path, default=None,
        help="absolute or repo-relative path to the pinned base lens",
    )
    parser.add_argument(
        "--distill-lens", type=Path, default=None,
        help="absolute or repo-relative path to the pinned distilled lens",
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("fetch", help="fetch and hash-check pinned task files; no models")
    pre = sub.add_parser("preflight", help="validate sealed inputs; no models")
    pre.add_argument("--seal-commit", default=None)
    campaign = sub.add_parser(
        "campaign", help="freeze the shared sealed preflight before any model forward"
    )
    campaign.add_argument("--seal-commit", required=True)
    for stage in ("atomic", "orderops"):
        command = sub.add_parser(stage)
        command.add_argument("--seal-commit", required=True)
        command.add_argument("--campaign-dir", type=Path, required=True)
        command.add_argument(
            "--resume-run-dir", type=Path, default=None,
            help="resume an incomplete run directory created by the same stage",
        )
    analyse = sub.add_parser("analyse")
    analyse.add_argument("--run-dir", type=Path, required=True)
    analyse.add_argument("--seal-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = args.repo_root.resolve()
    outroot = validate_output_root(repo, args.outroot)
    cache_dir = validate_mutable_cache_dir(repo, args.cache_dir, outroot)
    lens_paths = resolve_lens_paths(repo, args.base_lens, args.distill_lens)
    if args.stage == "fetch":
        inputs, hashes = load_upstream_inputs(cache_dir)
        print(json.dumps({"model_loaded": False, "hashes": hashes,
                          "populations": {k: len(v) for k, v in inputs.items()}}, indent=2))
        return 0
    if args.stage == "preflight":
        print(json.dumps(
            preflight(repo, cache_dir, args.seal_commit, outroot, lens_paths), indent=2
        ))
        return 0
    if args.stage == "campaign":
        sealed_preflight = preflight(
            repo, cache_dir, args.seal_commit, outroot, lens_paths
        )
        seal = sealed_preflight["seal"]
        if seal is None:  # pragma: no cover - --seal-commit is required
            raise RuntimeError("campaign preflight lacks a seal")
        campaign_dir = new_campaign_dir(outroot)
        reference = write_campaign_authority(
            campaign_dir, seal=seal, sealed_preflight=sealed_preflight
        )
        print(json.dumps({"campaign_dir": str(campaign_dir), **reference}, indent=2))
        return 0
    if args.stage == "analyse":
        verify_runtime_environment(repo)
        seal = validate_sealed_checkout(
            repo, args.seal_commit, allowed_untracked_root=outroot
        )
        run_dir = args.run_dir.resolve()
        if run_dir.parent != (outroot / "orderops").resolve(strict=False):
            raise RuntimeError("analysis run directory must be inside the isolated output root")
        reject_symlinks_beneath(run_dir)
        validate_stage_selection(run_dir, "orderops")
        if not (run_dir / "RANKS_COMPLETE").is_file() or (run_dir / "INCOMPLETE").exists():
            raise RuntimeError("analysis requires a completed rank-extraction bundle")
        validate_completion_marker(
            repo, run_dir, "RANKS_COMPLETE", "orderops-rank-extraction"
        )
        prior_seal = json.loads((run_dir / "seal.json").read_text())
        if prior_seal != seal:
            raise RuntimeError("rank bundle was created under a different seal")
        validate_run_identity(run_dir, "orderops")
        stored_preflight = json.loads((run_dir / "preflight.json").read_text())
        stored_campaign_reference = json.loads(
            (run_dir / "campaign_authority.json").read_text()
        )
        current_campaign_reference = validate_campaign_authority(
            Path(str(stored_campaign_reference.get("campaign_dir", ""))),
            seal=prior_seal, sealed_preflight=stored_preflight,
        )
        if current_campaign_reference != stored_campaign_reference:
            raise RuntimeError("rank bundle campaign authority drift")
        rank_manifest = json.loads((run_dir / "manifest.json").read_text())
        verified_lenses = verify_lens_files(lens_paths)
        verified_snapshots: dict[str, Any] = {}
        for cell in MODEL_CELLS:
            cell_manifest = rank_manifest.get("cells", {}).get(cell, {})
            lens_record = verified_lenses[cell]
            if (cell_manifest.get("lens_sha256") != lens_record["sha256"]
                    or int(cell_manifest.get("lens_size_bytes", -1))
                    != lens_record["size_bytes"]):
                raise RuntimeError(f"{cell}: analysis lens provenance differs from rank bundle")
            snapshot_record = cell_manifest.get("verified_hf_snapshot", {})
            snapshot_path = Path(str(snapshot_record.get("snapshot_path", "")))
            current_snapshot = verify_existing_hf_snapshot(
                repo, cell, snapshot_path, include_model_weights=True
            )
            if current_snapshot != snapshot_record:
                raise RuntimeError(f"{cell}: analysis HF snapshot differs from rank bundle")
            architecture = verify_snapshot_architecture(
                snapshot_path, cell, include_model_weights=True
            )
            if not architecture.get("weight_header_verified"):
                raise RuntimeError(f"{cell}: analysis architecture receipt is incomplete")
            verified_snapshots[cell] = current_snapshot
        report_path = run_dir / "report.json"
        analysis_marker_valid = False
        if (run_dir / "ANALYSIS_COMPLETE").exists():
            try:
                validate_completion_marker(
                    repo, run_dir, "ANALYSIS_COMPLETE", "orderops-analysis"
                )
                analysis_marker_valid = True
            except Exception:
                if not (run_dir / "ANALYSIS_INCOMPLETE").is_file():
                    raise
                preserve_unbound_files(
                    run_dir,
                    ("ANALYSIS_COMPLETE", "report.json", "REPORT.md",
                     "derivation_manifest.json"),
                    label="invalid-analysis-completion",
                )
        if analysis_marker_valid:
            analysis_incomplete = run_dir / "ANALYSIS_INCOMPLETE"
            if analysis_incomplete.exists():
                if analysis_incomplete.is_symlink():
                    raise RuntimeError("unsafe analysis incomplete marker")
                analysis_incomplete.unlink()
                fsync_parent_directory(analysis_incomplete)
            report = json.loads(report_path.read_text())
            print(json.dumps(
                report["co_primary"] if report["co_primary"] is not None
                else {
                    "evidence_status": report.get("evidence_status"),
                    "pair_pattern": report.get("pair_pattern"),
                }, indent=2,
            ))
            return 0
        preserve_unbound_files(
            run_dir, ("report.json", "REPORT.md", "derivation_manifest.json"),
            label="analysis",
        )
        atomic_write_json(run_dir / "ANALYSIS_INCOMPLETE", {
            "schema_version": SCHEMA_VERSION,
            "stage": "orderops-analysis",
            "seal_sha256": sha256_file(run_dir / "seal.json"),
        })
        report = analyse_orderops_bundle(run_dir)
        report["seal"] = seal
        atomic_write_json(report_path, report)
        atomic_write_bytes(
            run_dir / "REPORT.md", orderops_report_markdown(report).encode("utf-8")
        )
        derivation = write_derivation_manifest(
            repo, run_dir, stage="orderops-analysis",
            extra_inputs=[
                *lens_paths.values(),
                *snapshot_input_paths(verified_snapshots),
                *fetched_input_paths(run_dir),
                *campaign_input_paths(run_dir),
                stage_selection_path(run_dir),
                run_dir / "RANKS_COMPLETE",
            ],
        )
        finalize_stage(
            repo, run_dir, stage="orderops-analysis", marker_name="ANALYSIS_COMPLETE",
            manifest_name="report.json", derivation_name=derivation.name,
            incomplete_name="ANALYSIS_INCOMPLETE",
        )
        print(json.dumps(
            report["co_primary"] if report["co_primary"] is not None
            else {
                "evidence_status": report.get("evidence_status"),
                "pair_pattern": report.get("pair_pattern"),
            },
            indent=2,
        ))
        return 0

    sealed_preflight = preflight(
        repo, cache_dir, args.seal_commit, outroot, lens_paths
    )
    seal = sealed_preflight["seal"]
    if seal is None:  # pragma: no cover - model stages require --seal-commit
        raise RuntimeError("model-bearing stage lacks a sealed preflight")
    campaign_dir = args.campaign_dir.resolve()
    campaign_parent = (outroot / "campaign").resolve(strict=False)
    if campaign_dir.parent != campaign_parent:
        raise RuntimeError("campaign authority is outside the isolated campaign root")
    campaign_reference = validate_campaign_authority(
        campaign_dir, seal=seal, sealed_preflight=sealed_preflight
    )
    inputs, hashes = load_upstream_inputs(cache_dir)
    del inputs
    if args.resume_run_dir is None:
        run_dir = new_run_dir(outroot, args.stage)
        atomic_write_json(run_dir / "seal.json", seal)
        atomic_write_json(run_dir / "preflight.json", sealed_preflight)
        atomic_write_json(run_dir / "campaign_authority.json", campaign_reference)
        write_stage_selection(run_dir, args.stage)
        atomic_write_json(
            run_dir / "run_identity.json", run_identity_payload(run_dir, args.stage)
        )
        atomic_write_bytes(run_dir / "retry_log.jsonl", b"")
    else:
        run_dir = args.resume_run_dir.resolve()
        if run_dir.parent != (outroot / args.stage).resolve(strict=False):
            raise RuntimeError("resume directory is outside the isolated stage output root")
        reject_symlinks_beneath(run_dir)
        validate_stage_selection(run_dir, args.stage)
        marker_name = "COMPLETE" if args.stage == "atomic" else "RANKS_COMPLETE"
        marker_stage = "atomic" if args.stage == "atomic" else "orderops-rank-extraction"
        marker_exists = (run_dir / marker_name).is_file()
        invalid_incomplete_marker: str | None = None
        if marker_exists:
            try:
                validate_completion_marker(repo, run_dir, marker_name, marker_stage)
            except Exception:
                if not (run_dir / "INCOMPLETE").is_file():
                    raise
                marker_exists = False
                invalid_incomplete_marker = marker_name
        if not marker_exists and not (run_dir / "INCOMPLETE").is_file():
            raise RuntimeError("incomplete resume directory lacks INCOMPLETE marker")
        prior_seal = json.loads((run_dir / "seal.json").read_text())
        if prior_seal != seal:
            raise RuntimeError("resume directory was created under a different seal")
        prior_preflight = json.loads((run_dir / "preflight.json").read_text())
        if prior_preflight != sealed_preflight:
            raise RuntimeError("resume directory preflight differs from the current sealed preflight")
        stored_campaign = json.loads((run_dir / "campaign_authority.json").read_text())
        if stored_campaign != campaign_reference:
            raise RuntimeError("resume directory campaign authority differs")
        validate_run_identity(run_dir, args.stage)
        retry_path = run_dir / "retry_log.jsonl"
        prior_events = [json.loads(line) for line in retry_path.read_text().splitlines()]
        if not marker_exists:
            if sum(event.get("event") == "resume_requested" for event in prior_events) >= 1:
                raise RuntimeError("the single permitted technical resume has already been used")
            exception_path, exception_record = validate_resumable_stage_exception(
                run_dir, stage=args.stage, seal=seal,
                sealed_preflight=sealed_preflight,
            )
            exception_sha = sha256_file(exception_path)
            last_event = prior_events[-1] if prior_events else {}
            event_traceback = str(last_event.get("traceback", ""))
            if (last_event.get("event") != "stage_exception"
                    or last_event.get("stage") != args.stage
                    or last_event.get("stage_exception_sha256") != exception_sha
                    or last_event.get("traceback_sha256")
                    != exception_record["traceback_sha256"]
                    or sha256_bytes(event_traceback.encode("utf-8"))
                    != exception_record["traceback_sha256"]):
                raise RuntimeError("retry log does not bind the validated stage_exception")
            resume_event = {
                "utc": utc_now(), "stage": args.stage,
                "event": "resume_requested",
                "seal_commit": seal["git_commit"],
                "configuration_changed": False,
                "prior_python_exception_recorded": True,
                "stage_exception_sha256": exception_sha,
                "configuration_digest": exception_record["configuration_digest"],
                "completed_prefix_digest": exception_record["completed_prefix_digest"],
                "execution_fingerprint_digest": exception_record[
                    "execution_fingerprint_digest"
                ],
            }
            append_retry_event(retry_path, resume_event)
            atomic_write_json(run_dir / "stage_exception_consumed.json", {
                "schema_version": SCHEMA_VERSION,
                "record_type": "stage_exception_consumed",
                "stage_exception_sha256": exception_sha,
                "resume_event_sha256": sha256_bytes(canonical_json_bytes(resume_event)),
                "consumed_utc": resume_event["utc"],
            })
            if invalid_incomplete_marker is not None:
                preserve_unbound_files(
                    run_dir, (invalid_incomplete_marker,),
                    label=f"invalid-{args.stage}-completion-marker",
                )
    started = time.time()
    try:
        reject_symlinks_beneath(run_dir)
        installed_hashes = install_fetched_inputs(run_dir, cache_dir)
        if installed_hashes != hashes:
            raise RuntimeError("installed upstream input hashes differ from preflight hashes")
        current_device = bind_observed_model_dtype(
            execution_device_fingerprint(__import__("torch")),
            sealed_preflight["tokenizer_only_orderops"]["architecture_receipts"],
            require_weight_headers=True,
        )
        if current_device != sealed_preflight["device_fingerprint"]:
            raise RuntimeError("execution device differs from sealed preflight")
        current_seal = validate_sealed_checkout(
            repo, args.seal_commit, allowed_untracked_root=outroot
        )
        if current_seal != seal:
            raise RuntimeError("sealed checkout changed immediately before model load")
        if args.stage == "atomic":
            run_atomic_stage(
                repo, run_dir, cache_dir, hashes, sealed_preflight, lens_paths
            )
        elif args.stage == "orderops":
            run_orderops_stage(
                repo, run_dir, cache_dir, hashes, lens_paths, sealed_preflight
            )
        else:  # pragma: no cover - argparse makes this unreachable
            raise AssertionError(args.stage)
    except Exception as exc:
        traceback_text = traceback.format_exc()
        completed_marker = "COMPLETE" if args.stage == "atomic" else "RANKS_COMPLETE"
        completed_stage = (
            "atomic" if args.stage == "atomic" else "orderops-rank-extraction"
        )
        if (run_dir / completed_marker).is_file():
            try:
                validate_completion_marker(
                    repo, run_dir, completed_marker, completed_stage
                )
            except Exception:
                pass
            else:
                # Completion is already authoritative.  In particular, never
                # mutate marker-bound retry bytes because cleanup/fsync failed.
                raise
        exception_path, exception_record = write_stage_exception_record(
            run_dir, stage=args.stage, exc=exc, traceback_text=traceback_text
        )
        append_retry_event(run_dir / "retry_log.jsonl", {
            "utc": exception_record["timestamp_utc"], "stage": args.stage,
            "event": "stage_exception",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback_text,
            "traceback_sha256": exception_record["traceback_sha256"],
            "stage_exception_path": exception_path.name,
            "stage_exception_sha256": sha256_file(exception_path),
        })
        raise
    print(json.dumps({"stage": args.stage, "run_dir": str(run_dir),
                      "wall_seconds": round(time.time() - started, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
