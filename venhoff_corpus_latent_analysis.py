#!/usr/bin/env python3
"""Reconstruct Venhoff's Qwen-1.5B activation cloud and compare PCA spaces.

This is a zero-API, post-hoc sensitivity analysis over the 500 already
annotated responses released with Venhoff et al.'s code.  It has three stages:

``audit``
    Re-run the released annotation parser and first-occurrence token locator
    without loading a model.  The accepted-row counts must exactly match the
    released mean-vector tensor.

``extract``
    Re-run the released responses through a pinned local
    DeepSeek-R1-Distill-Qwen-1.5B snapshot and save row-level residual-stream
    activations at L15/L17/L18.  Two span-location arms share each forward pass:

    * ``released`` reproduces Venhoff's ``response_text.find(text)`` rule;
    * ``occurrence_aware`` searches sequentially inside the released thinking
      process, preventing a repeated quotation from binding to the prompt or
      to an earlier occurrence.

``analyse``
    Validate reconstructed released means/counts against the official ``.pt``;
    fit exact row-pooled six-label PCA clouds; report 70%-variance dimension,
    participation ratio, and top-ten variance concentration; and compute the
    2x2 vector/subspace comparison

        direction {thesis/hybrid, official Venhoff}
        x PCA cloud {thesis, reconstructed Venhoff}

    at k={3,5,10}.  The occurrence-aware Venhoff cloud is retained as a
    sensitivity rather than silently replacing the released convention.

The full run is deliberately guarded by ``--confirm-full``.  Pilot output is
written to a separate path and cannot be analysed as a full result unless
``--allow-pilot-analysis`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
OFFICIAL_REPO = Path("/Users/tonysu/steering-thinking-llms")
OFFICIAL_VARS = OFFICIAL_REPO / "train-steering-vectors/results/vars"
RESPONSES = OFFICIAL_VARS / "responses_deepseek-r1-distill-qwen-1.5b.json"
OFFICIAL_MEANS = OFFICIAL_VARS / "mean_vectors_deepseek-r1-distill-qwen-1.5b.pt"
MODEL_CACHE_ROOT = (
    Path("/Users/tonysu/.cache/huggingface/hub")
    / "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"
)
MODEL_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
MODEL_SNAPSHOT = MODEL_CACHE_ROOT / "snapshots" / MODEL_REVISION

DEFAULT_DATA_OUT = (
    ROOT / "data/activations/Venhoff-R1-1.5B__released500_l15-l17-l18"
)
DEFAULT_RESULT_OUT = (
    ROOT / "results/pca/R1-1.5B__venhoff_released_corpus_latent_v3"
)
THESIS_BASIS_ROOT = (
    ROOT / "results/pca/R1-1.5B__venhoff_official_direction_alignment"
)
HYBRID_ROOT = ROOT / "results/steering_vectors/R1-1.5B__venhoff_bridge_hybrid"
THESIS_CURRENT = ROOT / "data/activations/R1-1.5B"
THESIS_ARCHIVE = ROOT / "data/activations/_volume_R1-1.5B_6label_archive"
THESIS_EVAL_IDS = ROOT / "results/eval/R1-1.5B__E1/eval_task_ids.json"
THESIS_ARCHIVED_LABELS = {"initializing", "deduction"}

EXPECTED_RESPONSE_SHA256 = (
    "5f6f9ed1fd2f60608cd7c08b1253a43ebbc8ad1b268a46ba1f1d7d48bea8b4af"
)
EXPECTED_MEANS_SHA256 = (
    "bbf7bcac3758df3236d485c41dcf3033a53fd6333a4abd555290e8ef71419412"
)
EXPECTED_OFFICIAL_REPO_COMMIT = "93259bc3410c99293351df41141cd16b4110422a"
EXPECTED_EXTRACTION_BUILDER_SHA256 = (
    "edd13037943d9339d15bc14b3f5f67bf4e96e483159337a990263fc1a07a9a46"
)
OFFICIAL_RUNTIME_PINS = {
    "torch": "2.5.1",
    "transformers": "4.47.1",
    "nnsight": "0.4.5",
}

LAYERS = (15, 17, 18)
ARMS = ("released", "occurrence_aware")
PCA_LABELS = (
    "initializing",
    "deduction",
    "adding-knowledge",
    "example-testing",
    "uncertainty-estimation",
    "backtracking",
)
TARGETS = (
    "backtracking",
    "uncertainty-estimation",
    "example-testing",
    "adding-knowledge",
)
VENHOFF_LAYERS = {
    "backtracking": 17,
    "uncertainty-estimation": 18,
    "example-testing": 15,
    "adding-knowledge": 18,
}
K_VALUES = (3, 5, 10)
ANNOTATION_RE = re.compile(r'\["(\S+?)"\](.*?)\["end-section"\]', re.DOTALL)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    tmp.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, **arrays)
    tmp.replace(path)


def git_output(cwd: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def verify_official_runtime_pins(environment_path: Path) -> dict[str, Any]:
    """Verify the author repository's explicit package/runtime pins."""
    text = environment_path.read_text()
    missing = [
        f"{package}=={version}"
        for package, version in OFFICIAL_RUNTIME_PINS.items()
        if f"- {package}=={version}" not in text
    ]
    cuda_lines = [
        line.strip()[2:]
        for line in text.splitlines()
        if line.strip().startswith("- nvidia-") and "-cu12==" in line
    ]
    if missing or not cuda_lines:
        raise ValueError(
            "Official environment pin audit failed: "
            f"missing={missing}, cuda_dependency_count={len(cuda_lines)}"
        )
    return {
        "source": str(environment_path),
        "source_sha256": sha256(environment_path),
        "pinned_packages": dict(OFFICIAL_RUNTIME_PINS),
        "cuda_stack_pinned": True,
        "cuda_dependencies": cuda_lines,
        "hf_model_revision_pinned": False,
        "interpretation": (
            "Venhoff's released environment pins the software/CUDA runtime; "
            "the model identifier is pinned only by name, not by Hugging Face revision."
        ),
    }


def analysis_disposition(
    reconstruction_summary: dict[str, Any],
) -> dict[str, Any]:
    """Make the reconstruction gate a load-bearing top-level claim boundary."""
    gate = bool(reconstruction_summary.get("high_fidelity_gate"))
    return {
        "empirical_evidence_status": "exploratory",
        "confirmatory": False,
        "protocol_marker": "amended local-runtime replay",
        "strict_high_fidelity_gate": gate,
        "venhoff_cloud_pca_claim_licensed": gate,
        "actual_disposition": (
            "conditional local-runtime sensitivity"
            if gate
            else "unlicensed conditional local-runtime sensitivity"
        ),
        "claim_boundary": (
            "The strict reconstruction gate did not pass. Therefore the locally "
            "replayed activation-cloud PCA and every alignment cell using a local "
            "Venhoff cloud are not a reproduced Venhoff PCA result and are unlicensed "
            "for a claim about Venhoff's original activation cloud. They may be used "
            "only as explicitly conditional local-runtime sensitivities."
            if not gate
            else
            "The strict reconstruction gate passed; local cloud results remain "
            "post-hoc exploratory and bounded by corpus/protocol differences."
        ),
    }


def parse_segments(annotation: str | None) -> list[dict[str, Any]]:
    """Match the released regex exactly and preserve annotation order."""
    segments: list[dict[str, Any]] = []
    for annotation_index, match in enumerate(
        ANNOTATION_RE.finditer(annotation or "")
    ):
        text = match.group(2).strip()
        if not text:
            continue
        segments.append(
            {
                "annotation_index": annotation_index,
                "label": match.group(1).strip(),
                "text": text,
            }
        )
    return segments


def char_to_token_map(offsets: Iterable[Iterable[int]]) -> dict[int, int]:
    """Reproduce Venhoff's character-to-token dictionary construction."""
    mapping: dict[int, int] = {}
    for token_index, pair in enumerate(offsets):
        start, end = int(pair[0]), int(pair[1])
        for char_position in range(start, end):
            mapping[char_position] = token_index
    return mapping


def token_bounds(
    start_char: int | None,
    text: str,
    mapping: dict[int, int],
) -> tuple[int, int] | None:
    if start_char is None or start_char < 0 or not text:
        return None
    token_start = mapping.get(start_char)
    token_end = mapping.get(start_char + len(text) - 1)
    if token_end is not None:
        token_end += 1
    if token_start is None or token_end is None or token_start >= token_end:
        return None
    return int(token_start), int(token_end)


def released_char_locations(
    response_text: str,
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Venhoff's exact first-occurrence rule: ``response_text.find(text)``."""
    located = []
    for segment in segments:
        text = segment["text"]
        start = response_text.find(text)
        located.append(
            {
                **segment,
                "start_char": start if start >= 0 else None,
                "end_char": start + len(text) if start >= 0 else None,
                "resolution": "global_first_occurrence" if start >= 0 else "missing",
                "n_exact_occurrences": response_text.count(text),
            }
        )
    return located


def _all_occurrences(text: str, needle: str, start: int = 0) -> list[int]:
    positions: list[int] = []
    cursor = max(0, start)
    while True:
        position = text.find(needle, cursor)
        if position < 0:
            return positions
        positions.append(position)
        cursor = position + max(1, len(needle))


def occurrence_aware_char_locations(
    response_text: str,
    thinking_process: str,
    segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the thesis's canonical occurrence-aware rule inside thinking text.

    This mirrors ``src.text_offsets.locate_annotation_offsets``: exact search
    from a forward cursor, exact from-start fallback for out-of-order
    annotations, then a 40-character-prefix sensitivity for truncated labels.
    The cursor advances by one character, which binds ordinary verbatim repeats
    successively while preserving nested/overlapping annotated spans.
    """
    marker = response_text.find("<think>")
    search_floor = marker + len("<think>") if marker >= 0 else 0
    thinking_start = response_text.find(thinking_process, search_floor)
    boundary_status = "exact_thinking_process"
    if thinking_start < 0:
        thinking_start = search_floor
        boundary_status = "fallback_after_think_marker"
    thinking_stop = (
        thinking_start + len(thinking_process)
        if thinking_process and boundary_status == "exact_thinking_process"
        else len(response_text)
    )

    thinking_region = response_text[thinking_start:thinking_stop]
    cursor = 0
    located: list[dict[str, Any]] = []
    for segment in segments:
        text = segment["text"]
        all_positions = _all_occurrences(thinking_region, text, 0)
        local_position = thinking_region.find(text, cursor)
        resolution = "sequential_inside_thinking"
        matched_text_length = len(text)
        if local_position < 0:
            local_position = thinking_region.find(text)
            resolution = "from_start_fallback_inside_thinking"
        if local_position < 0:
            prefix = text[:40].strip()
            if len(prefix) >= 10:
                local_position = thinking_region.find(prefix, cursor)
                resolution = "sequential_prefix40_inside_thinking"
                if local_position < 0:
                    local_position = thinking_region.find(prefix)
                    resolution = "from_start_prefix40_inside_thinking"
                matched_text_length = len(prefix)
        if local_position >= 0:
            position = thinking_start + local_position
            end = position + matched_text_length
            cursor = max(cursor, local_position + 1)
            start_value: int | None = position
            end_value: int | None = end
        else:
            start_value = None
            end_value = None
            resolution = "missing"
        located.append(
            {
                **segment,
                "start_char": start_value,
                "end_char": end_value,
                "resolution": resolution,
                "n_exact_occurrences_in_thinking": len(all_positions),
                "matched_text_length": matched_text_length if local_position >= 0 else None,
            }
        )
    return located, {
        "thinking_start_char": thinking_start,
        "thinking_stop_char": thinking_stop,
        "thinking_boundary_status": boundary_status,
    }


def attach_token_bounds(
    located: list[dict[str, Any]],
    mapping: dict[int, int],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for location in located:
        matched_length = int(location.get("matched_text_length") or len(location["text"]))
        bounds = token_bounds(
            location["start_char"], location["text"][:matched_length], mapping
        )
        if bounds is None:
            output.append({**location, "accepted": False, "skip_reason": "token_mapping"})
            continue
        token_start, token_end = bounds
        pool_start = token_start - 1
        pool_stop = min(token_end - 1, token_start + 10)
        if pool_stop <= pool_start:
            output.append({**location, "accepted": False, "skip_reason": "empty_pool"})
            continue
        output.append(
            {
                **location,
                "accepted": True,
                "token_start": token_start,
                "token_end": token_end,
                "pool_start": pool_start,
                "pool_stop": pool_stop,
            }
        )
    return output


def locate_record(
    record: dict[str, Any], tokenizer: Any
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    response = record["full_response"]
    segments = parse_segments(record.get("annotated_thinking"))
    offsets = tokenizer(
        response, return_offsets_mapping=True, add_special_tokens=True
    )["offset_mapping"]
    mapping = char_to_token_map(offsets)
    released = attach_token_bounds(
        released_char_locations(response, segments), mapping
    )
    corrected_chars, corrected_boundary = occurrence_aware_char_locations(
        response, record.get("thinking_process", ""), segments
    )
    corrected = attach_token_bounds(corrected_chars, mapping)
    return {"released": released, "occurrence_aware": corrected}, {
        "n_tokens": len(offsets),
        **corrected_boundary,
    }


def expected_counts_from_means(means: dict[str, Any]) -> dict[str, int]:
    return {label: int(cell["count"]) for label, cell in means.items()}


def static_audit(
    records: list[dict[str, Any]], tokenizer: Any, official_means: dict[str, Any]
) -> dict[str, Any]:
    counts = {arm: Counter() for arm in ARMS}
    overall = Counter()
    missing = {arm: Counter() for arm in ARMS}
    changed = Counter()
    boundary = Counter()
    duplicate_released = Counter()
    token_lengths: list[int] = []

    for index, record in enumerate(records):
        arms, metadata = locate_record(record, tokenizer)
        token_lengths.append(metadata["n_tokens"])
        boundary[metadata["thinking_boundary_status"]] += 1
        by_ann = {
            arm: {x["annotation_index"]: x for x in locations}
            for arm, locations in arms.items()
        }
        for arm, locations in arms.items():
            accepted_any = False
            for item in locations:
                if item.get("accepted"):
                    counts[arm][item["label"]] += 1
                    accepted_any = True
                else:
                    missing[arm][item["label"]] += 1
            if accepted_any:
                overall[arm] += 1
        for annotation_index, released in by_ann["released"].items():
            corrected = by_ann["occurrence_aware"].get(annotation_index)
            if released.get("n_exact_occurrences", 0) > 1:
                duplicate_released[released["label"]] += 1
            if corrected is None:
                continue
            key = f"{released.get('accepted', False)}->{corrected.get('accepted', False)}"
            changed[key] += 1
            if (
                released.get("accepted")
                and corrected.get("accepted")
                and released.get("start_char") != corrected.get("start_char")
            ):
                changed["accepted_position_changed"] += 1

    expected = expected_counts_from_means(official_means)
    released_observed = dict(counts["released"])
    released_observed["overall"] = int(overall["released"])
    count_deltas = {
        label: released_observed.get(label, 0) - count
        for label, count in expected.items()
    }
    exact = all(delta == 0 for delta in count_deltas.values())
    return {
        "n_records": len(records),
        "token_lengths": {
            "min": int(min(token_lengths)),
            "median": float(np.median(token_lengths)),
            "p90": float(np.percentile(token_lengths, 90)),
            "p99": float(np.percentile(token_lengths, 99)),
            "max": int(max(token_lengths)),
            "sum": int(sum(token_lengths)),
        },
        "released_counts": released_observed,
        "occurrence_aware_counts": {
            **dict(counts["occurrence_aware"]),
            "overall": int(overall["occurrence_aware"]),
        },
        "official_counts": expected,
        "released_minus_official_counts": count_deltas,
        "released_counts_exactly_match_official": exact,
        "missing_by_arm_and_label": {
            arm: dict(values) for arm, values in missing.items()
        },
        "location_comparison": dict(changed),
        "released_multi_occurrence_segments_by_label": dict(duplicate_released),
        "thinking_boundary_status": dict(boundary),
    }


def validate_inputs() -> dict[str, Any]:
    required = [
        RESPONSES,
        OFFICIAL_MEANS,
        MODEL_SNAPSHOT / "config.json",
        MODEL_SNAPSHOT / "tokenizer.json",
        MODEL_SNAPSHOT / "tokenizer_config.json",
        MODEL_SNAPSHOT / "model.safetensors",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required local inputs: " + ", ".join(missing))
    response_sha = sha256(RESPONSES)
    means_sha = sha256(OFFICIAL_MEANS)
    if response_sha != EXPECTED_RESPONSE_SHA256:
        raise ValueError(
            f"Released response SHA changed: {response_sha} != {EXPECTED_RESPONSE_SHA256}"
        )
    if means_sha != EXPECTED_MEANS_SHA256:
        raise ValueError(
            f"Released mean SHA changed: {means_sha} != {EXPECTED_MEANS_SHA256}"
        )
    official_commit = git_output(OFFICIAL_REPO, "rev-parse", "HEAD")
    if official_commit != EXPECTED_OFFICIAL_REPO_COMMIT:
        raise ValueError(
            f"Official repository commit changed: {official_commit} != "
            f"{EXPECTED_OFFICIAL_REPO_COMMIT}"
        )
    ref = (MODEL_CACHE_ROOT / "refs/main").read_text().strip()
    if ref != MODEL_REVISION:
        raise ValueError(f"Local model ref changed: {ref} != {MODEL_REVISION}")
    return {
        "responses_sha256": response_sha,
        "official_means_sha256": means_sha,
        "official_repository_commit": official_commit,
        "model_revision": ref,
        "model_config_sha256": sha256(MODEL_SNAPSHOT / "config.json"),
        "tokenizer_json_sha256": sha256(MODEL_SNAPSHOT / "tokenizer.json"),
        "tokenizer_config_sha256": sha256(MODEL_SNAPSHOT / "tokenizer_config.json"),
        "model_weight_file_size_bytes": (MODEL_SNAPSHOT / "model.safetensors").stat().st_size,
        "model_weight_blob": os.path.realpath(MODEL_SNAPSHOT / "model.safetensors"),
    }


def load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SNAPSHOT, local_files_only=True, use_fast=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_official_means() -> dict[str, Any]:
    import torch

    return torch.load(OFFICIAL_MEANS, map_location="cpu", weights_only=True)


def select_device(requested: str) -> str:
    import torch

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_dtype(device: str, requested: str) -> Any:
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if requested != "auto":
        return mapping[requested]
    if device in {"cuda", "mps"}:
        return torch.bfloat16
    return torch.float32


def load_model(device: str, dtype: Any) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    # The CausalLM wrapper is loaded to match Venhoff's released code, but the
    # extraction calls only its decoder backbone, avoiding a needless full-vocab
    # logits tensor for 16 x ~1,000-token batches.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_SNAPSHOT,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.set_grad_enabled(False)
    return model


@contextmanager
def capture_layers(model: Any, layers: Iterable[int]):
    cache: dict[int, Any] = {}
    handles = []

    def hook_for(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            import torch

            hidden = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden):
                raise TypeError(f"Unexpected layer-{layer} output: {type(hidden)}")
            cache[layer] = hidden.detach().to(device="cpu", dtype=torch.float32)

        return hook

    try:
        for layer in layers:
            handles.append(model.model.layers[layer].register_forward_hook(hook_for(layer)))
        yield cache
    finally:
        for handle in handles:
            handle.remove()


def batch_key(arm: str, label: str, layer: int) -> str:
    return f"{arm}__{label}__layer{layer}"


def source_record_digest(records: list[dict[str, Any]], start: int, stop: int) -> str:
    selected = []
    for index in range(start, stop):
        record = records[index]
        selected.append(
            {
                "source_json_index": index,
                "full_response": record["full_response"],
                "thinking_process": record.get("thinking_process", ""),
                "annotated_thinking": record.get("annotated_thinking", ""),
            }
        )
    return canonical_sha256(selected)


def shard_complete(
    shard_npz: Path,
    shard_rows: Path,
    shard_meta: Path,
    expected_digest: str,
) -> bool:
    if not (shard_npz.is_file() and shard_rows.is_file() and shard_meta.is_file()):
        return False
    try:
        meta = load_json(shard_meta)
        return (
            meta.get("complete") is True
            and meta.get("source_record_digest") == expected_digest
            and meta.get("npz_sha256") == sha256(shard_npz)
            and meta.get("rows_sha256") == sha256(shard_rows)
        )
    except Exception:
        return False


def _row_record(
    *,
    arm: str,
    source_index: int,
    location: dict[str, Any],
    response_sha: str,
) -> dict[str, Any]:
    return {
        "chain_id": f"venhoff-qwen1p5b-{source_index:03d}",
        "source_json_index": source_index,
        "arm": arm,
        "annotation_index": location["annotation_index"],
        "label": location["label"],
        "text_sha256": hashlib.sha256(location["text"].encode("utf-8")).hexdigest(),
        "response_sha256": response_sha,
        "start_char": location["start_char"],
        "end_char": location["end_char"],
        "token_start": location["token_start"],
        "token_end": location["token_end"],
        "pool_start": location["pool_start"],
        "pool_stop": location["pool_stop"],
        "resolution": location["resolution"],
    }


def extract_batch(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    source_indices: list[int],
    device: str,
    labels: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    import torch

    texts = [records[index]["full_response"] for index in source_indices]
    encoded = tokenizer(
        texts,
        padding="longest",
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids_cpu = encoded["input_ids"]
    # Match the released implementation, which derives the mask from the pad id
    # rather than using the tokenizer-returned mask.
    attention_cpu = (input_ids_cpu != tokenizer.pad_token_id).long()
    inputs = {
        "input_ids": input_ids_cpu.to(device),
        "attention_mask": attention_cpu.to(device),
        "use_cache": False,
        "return_dict": False,
    }
    started = time.perf_counter()
    with torch.inference_mode(), capture_layers(model, LAYERS) as cache:
        output = model.model(**inputs)
    elapsed = time.perf_counter() - started
    del output, inputs

    accumulated: dict[str, list[np.ndarray]] = defaultdict(list)
    rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        arm: {label: [] for label in labels} for arm in ARMS
    }
    skipped = {arm: Counter() for arm in ARMS}
    changed_positions = 0

    for batch_index, source_index in enumerate(source_indices):
        record = records[source_index]
        locations, locate_meta = locate_record(record, tokenizer)
        non_padding = int(attention_cpu[batch_index].sum().item())
        pad_length = int((attention_cpu[batch_index] == 0).sum().item())
        if non_padding != locate_meta["n_tokens"]:
            raise ValueError(
                f"Tokenization mismatch at source index {source_index}: "
                f"batch={non_padding}, offsets={locate_meta['n_tokens']}"
            )
        per_layer = {
            layer: cache[layer][batch_index, pad_length : pad_length + non_padding]
            for layer in LAYERS
        }
        response_sha = hashlib.sha256(record["full_response"].encode("utf-8")).hexdigest()

        release_by_index = {
            x["annotation_index"]: x for x in locations["released"]
        }
        corrected_by_index = {
            x["annotation_index"]: x for x in locations["occurrence_aware"]
        }
        for ann_index, released in release_by_index.items():
            corrected = corrected_by_index.get(ann_index)
            if (
                corrected
                and released.get("accepted")
                and corrected.get("accepted")
                and released["start_char"] != corrected["start_char"]
            ):
                changed_positions += 1

        for arm in ARMS:
            accepted_locations = [
                location
                for location in locations[arm]
                if location.get("accepted") and location["label"] in labels
            ]
            for location in locations[arm]:
                if not location.get("accepted"):
                    skipped[arm][location["label"]] += 1
            for location in accepted_locations:
                label = location["label"]
                row = _row_record(
                    arm=arm,
                    source_index=source_index,
                    location=location,
                    response_sha=response_sha,
                )
                rows[arm][label].append(row)
                for layer in LAYERS:
                    pooled = per_layer[layer][
                        location["pool_start"] : location["pool_stop"]
                    ].mean(dim=0)
                    if not torch.isfinite(pooled).all():
                        raise ValueError(
                            f"Non-finite pooled row at record={source_index}, "
                            f"arm={arm}, label={label}, layer={layer}"
                        )
                    accumulated[batch_key(arm, label, layer)].append(
                        pooled.numpy().astype(np.float32, copy=False)
                    )

            if accepted_locations:
                overall_start = min(x["token_start"] for x in accepted_locations)
                overall_stop = max(x["token_end"] for x in accepted_locations)
                overall_location = {
                    "annotation_index": None,
                    "label": "overall",
                    "text": "",
                    "start_char": min(x["start_char"] for x in accepted_locations),
                    "end_char": max(x["end_char"] for x in accepted_locations),
                    "token_start": overall_start,
                    "token_end": overall_stop,
                    "pool_start": overall_start,
                    "pool_stop": overall_stop,
                    "resolution": f"{arm}_min_to_max_valid_label_region",
                }
                overall_row = _row_record(
                    arm=arm,
                    source_index=source_index,
                    location=overall_location,
                    response_sha=response_sha,
                )
                rows[arm]["overall"].append(overall_row)
                for layer in LAYERS:
                    pooled = per_layer[layer][overall_start:overall_stop].mean(dim=0)
                    if not torch.isfinite(pooled).all():
                        raise ValueError(
                            f"Non-finite overall row at record={source_index}, "
                            f"arm={arm}, layer={layer}"
                        )
                    accumulated[batch_key(arm, "overall", layer)].append(
                        pooled.numpy().astype(np.float32, copy=False)
                    )

    arrays: dict[str, np.ndarray] = {}
    hidden_size = int(model.config.hidden_size)
    for arm in ARMS:
        for label in labels:
            n_rows = len(rows[arm][label])
            for layer in LAYERS:
                key = batch_key(arm, label, layer)
                values = accumulated.get(key, [])
                if len(values) != n_rows:
                    raise AssertionError(f"Row/array mismatch for {key}")
                arrays[key] = (
                    np.stack(values).astype(np.float32, copy=False)
                    if values
                    else np.empty((0, hidden_size), dtype=np.float32)
                )
    return arrays, rows, {
        "elapsed_seconds": elapsed,
        "batch_size": len(source_indices),
        "max_padded_tokens": int(input_ids_cpu.shape[1]),
        "total_non_padding_tokens": int(attention_cpu.sum().item()),
        "changed_accepted_positions": changed_positions,
        "skipped": {arm: dict(values) for arm, values in skipped.items()},
    }


def write_shard(
    shard_root: Path,
    batch_number: int,
    start: int,
    stop: int,
    source_digest: str,
    arrays: dict[str, np.ndarray],
    rows: dict[str, dict[str, list[dict[str, Any]]]],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    prefix = f"batch_{batch_number:04d}"
    npz_path = shard_root / f"{prefix}.npz"
    rows_path = shard_root / f"{prefix}.rows.json"
    meta_path = shard_root / f"{prefix}.meta.json"
    atomic_npz(npz_path, arrays)
    atomic_json(rows_path, rows)
    meta = {
        "complete": True,
        "batch_number": batch_number,
        "source_start_inclusive": start,
        "source_stop_exclusive": stop,
        "source_record_digest": source_digest,
        "runtime": runtime,
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "npz_sha256": sha256(npz_path),
        "rows_sha256": sha256(rows_path),
    }
    atomic_json(meta_path, meta)
    return meta


def shard_paths(shard_root: Path, batch_number: int) -> tuple[Path, Path, Path]:
    prefix = f"batch_{batch_number:04d}"
    return (
        shard_root / f"{prefix}.npz",
        shard_root / f"{prefix}.rows.json",
        shard_root / f"{prefix}.meta.json",
    )


def finalise_extraction(
    out: Path,
    n_records: int,
    batch_size: int,
    labels: tuple[str, ...],
    base_metadata: dict[str, Any],
) -> dict[str, Any]:
    shard_root = out / "_shards"
    n_batches = math.ceil(n_records / batch_size)
    shards = [shard_paths(shard_root, index) for index in range(n_batches)]
    if any(not all(path.is_file() for path in paths) for paths in shards):
        raise FileNotFoundError("Cannot finalise: one or more batch shards are missing")

    output_hashes: dict[str, str] = {}
    row_index: dict[str, dict[str, list[dict[str, Any]]]] = {
        arm: {label: [] for label in labels} for arm in ARMS
    }
    for _npz, rows_path, _meta in shards:
        rows = load_json(rows_path)
        for arm in ARMS:
            for label in labels:
                row_index[arm][label].extend(rows[arm][label])

    for arm in ARMS:
        arm_dir = out / arm
        for label in labels:
            expected_rows = len(row_index[arm][label])
            for layer in LAYERS:
                key = batch_key(arm, label, layer)
                pieces = []
                for npz_path, _rows, _meta in shards:
                    with np.load(npz_path, allow_pickle=False) as archive:
                        pieces.append(np.asarray(archive[key], dtype=np.float32))
                matrix = (
                    np.concatenate(pieces, axis=0)
                    if pieces
                    else np.empty((0, 1536), dtype=np.float32)
                )
                if matrix.shape != (expected_rows, 1536):
                    raise ValueError(
                        f"Final shape mismatch for {key}: {matrix.shape} vs "
                        f"{(expected_rows, 1536)}"
                    )
                path = arm_dir / f"{label}_layer{layer}.npy"
                atomic_npy(path, matrix)
                output_hashes[path.relative_to(out).as_posix()] = sha256(path)
                del matrix, pieces

    row_path = out / "row_index.json"
    atomic_json(row_path, row_index)
    output_hashes[row_path.relative_to(out).as_posix()] = sha256(row_path)
    counts = {
        arm: {label: len(rows) for label, rows in labels_rows.items()}
        for arm, labels_rows in row_index.items()
    }
    shard_meta = [load_json(paths[2]) for paths in shards]
    metadata = {
        **base_metadata,
        "complete": True,
        "n_records_processed": n_records,
        "batch_size": batch_size,
        "layers": list(LAYERS),
        "arms": list(ARMS),
        "labels": list(labels),
        "counts": counts,
        "n_batches": n_batches,
        "runtime": {
            "forward_seconds_sum": float(
                sum(cell["runtime"]["elapsed_seconds"] for cell in shard_meta)
            ),
            "total_non_padding_tokens": int(
                sum(cell["runtime"]["total_non_padding_tokens"] for cell in shard_meta)
            ),
            "per_batch": [cell["runtime"] for cell in shard_meta],
        },
        "output_sha256": output_hashes,
        "row_index_file": "row_index.json",
        "shards_retained_for_resume_and_audit": True,
    }
    atomic_json(out / "metadata.json", metadata)
    return metadata


def run_extract(args: argparse.Namespace, audit: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    import torch
    import transformers

    records = load_json(RESPONSES)
    official_means = load_official_means()
    labels = tuple(official_means.keys())
    if "overall" not in labels:
        raise ValueError("Official tensor has no overall mean")
    if not audit["released_counts_exactly_match_official"]:
        raise ValueError("Static released-count audit failed; refusing model extraction")

    full = args.limit is None
    if full and not args.confirm_full:
        raise ValueError("Full 500-response extraction requires --confirm-full")
    n_records = len(records) if full else min(args.limit, len(records))
    if n_records <= 0:
        raise ValueError("--limit must be positive")
    if full and args.batch_size != 16 and not args.allow_batch_sensitivity:
        raise ValueError(
            "The released run used batch_size=16; pass --batch-size 16 or "
            "explicitly mark a batch sensitivity with --allow-batch-sensitivity"
        )

    out = args.data_out.resolve()
    metadata_path = out / "metadata.json"
    if metadata_path.exists() and load_json(metadata_path).get("complete"):
        raise FileExistsError(f"Refusing to overwrite complete extraction: {out}")
    out.mkdir(parents=True, exist_ok=True)
    shard_root = out / "_shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    dtype = select_dtype(device, args.dtype)
    execution_contract_sha256 = canonical_sha256(
        {
            "builder_sha256": sha256(Path(__file__)),
            "model_revision": MODEL_REVISION,
            "device": device,
            "dtype": str(dtype),
            "batch_size": args.batch_size,
            "layers": LAYERS,
            "arms": ARMS,
        }
    )
    base_metadata = {
        "schema_version": "venhoff-corpus-activations-v1",
        "status": {
            "empirical": "post-hoc exploratory reconstruction",
            "confirmatory": False,
            "claim_boundary": (
                "Replays released responses/annotations through a pinned local model "
                "snapshot. Venhoff pinned the software/CUDA runtime in environment.yaml "
                "but did not pin an HF model revision; this local MPS replay uses a "
                "different software/backend runtime, so numerical fidelity is established "
                "only by comparison with released means."
            ),
        },
        "api_cost_usd": 0.0,
        "source_protocol": {
            "released_arm": (
                "first exact occurrence in full decoded response; token slice "
                "[start-1:min(end-1,start+10)]; overall min(start):max(end)"
            ),
            "occurrence_aware_arm": (
                "sequential exact occurrences restricted to released thinking_process; "
                "same token pooling and overall rules"
            ),
            "released_original_batch_size": 16,
            "executed_batch_size": args.batch_size,
            "batch_sensitivity": args.batch_size != 16,
            "model_call": "decoder backbone only; no lm_head logits; inference mode",
        },
        "provenance": {
            **provenance,
            "builder": Path(__file__).name,
            "builder_sha256": sha256(Path(__file__)),
            "analysis_repo_git_commit": git_output(ROOT, "rev-parse", "HEAD"),
            "analysis_repo_git_dirty": bool(git_output(ROOT, "status", "--porcelain")),
            "python": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": device,
            "dtype": str(dtype),
            "execution_contract_sha256": execution_contract_sha256,
        },
        "static_audit": audit,
    }
    atomic_json(out / "run_contract.json", {
        **base_metadata,
        "complete": False,
        "n_records_planned": n_records,
        "limit": args.limit,
    })

    tokenizer = load_tokenizer()
    print(f"Loading pinned model on {device} ({dtype})...", flush=True)
    model = load_model(device, dtype)
    n_batches = math.ceil(n_records / args.batch_size)
    wall_started = time.perf_counter()
    processed_this_invocation = 0
    for batch_number in range(n_batches):
        start = batch_number * args.batch_size
        stop = min(start + args.batch_size, n_records)
        digest = canonical_sha256(
            {
                "source_records": source_record_digest(records, start, stop),
                "execution_contract_sha256": execution_contract_sha256,
            }
        )
        npz_path, rows_path, meta_path = shard_paths(shard_root, batch_number)
        if shard_complete(npz_path, rows_path, meta_path, digest):
            print(f"batch {batch_number + 1}/{n_batches}: verified existing shard", flush=True)
            continue
        source_indices = list(range(start, stop))
        arrays, rows, runtime = extract_batch(
            model, tokenizer, records, source_indices, device, labels
        )
        write_shard(
            shard_root,
            batch_number,
            start,
            stop,
            digest,
            arrays,
            rows,
            runtime,
        )
        processed_this_invocation += stop - start
        rate = runtime["total_non_padding_tokens"] / runtime["elapsed_seconds"]
        print(
            f"batch {batch_number + 1}/{n_batches}: {stop-start} records, "
            f"{runtime['total_non_padding_tokens']} tokens, "
            f"{runtime['elapsed_seconds']:.2f}s ({rate:.1f} tok/s)",
            flush=True,
        )
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    metadata = finalise_extraction(
        out, n_records, args.batch_size, labels, base_metadata
    )
    metadata["runtime"]["this_invocation_wall_seconds"] = time.perf_counter() - wall_started
    metadata["runtime"]["records_processed_this_invocation"] = processed_this_invocation
    atomic_json(metadata_path, metadata)
    print(f"Extraction complete -> {metadata_path}", flush=True)
    return metadata


def unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"Cannot normalise vector with norm={norm}")
    return value / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0))


def alignment(direction: np.ndarray, components: np.ndarray) -> dict[str, Any]:
    u = unit(direction)
    U = np.asarray(components, dtype=np.float64)
    if U.ndim != 2 or U.shape[1] != u.shape[0]:
        raise ValueError(f"Basis/vector mismatch: {U.shape} vs {u.shape}")
    gram = U[: max(K_VALUES)] @ U[: max(K_VALUES)].T
    if not np.allclose(gram, np.eye(max(K_VALUES)), atol=3e-4, rtol=3e-4):
        raise ValueError("PCA component rows are not orthonormal")
    squared = np.square(U @ u)
    cumulative = np.cumsum(squared)
    return {
        "by_k": {
            str(k): {
                "captured_squared_norm": float(np.clip(cumulative[k - 1], 0, 1)),
                "retained_norm": float(math.sqrt(np.clip(cumulative[k - 1], 0, 1))),
                "subspace_angle_degrees": float(
                    math.degrees(math.acos(math.sqrt(np.clip(cumulative[k - 1], 0, 1))))
                ),
            }
            for k in K_VALUES
        },
        "random_isotropic_expected_captured_squared_norm": {
            str(k): k / u.size for k in K_VALUES
        },
    }


def pca_cloud(matrices: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not matrices:
        raise ValueError("No matrices supplied")
    X = np.concatenate([np.asarray(x, dtype=np.float32) for x in matrices], axis=0)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError(f"Invalid PCA matrix shape {X.shape}")
    n_rows, hidden_dim = X.shape
    mean = X.mean(axis=0, dtype=np.float64)
    second_moment = np.asarray(X.T @ X, dtype=np.float64)
    covariance = (second_moment - n_rows * np.outer(mean, mean)) / (n_rows - 1)
    covariance = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = eigenvectors[:, order].T
    total = float(eigenvalues.sum())
    if total <= 0:
        raise ValueError("Non-positive total PCA variance")
    ratios = eigenvalues / total
    cumulative = np.cumsum(ratios)
    hits = np.flatnonzero(cumulative >= 0.70)
    d_eff70 = int(hits[0] + 1) if hits.size else hidden_dim
    participation_ratio = float(total**2 / np.square(eigenvalues).sum())
    metrics = {
        "n_rows": int(n_rows),
        "hidden_dim": int(hidden_dim),
        "total_variance": total,
        "d_eff70_variance_threshold_dimension": d_eff70,
        "participation_ratio": participation_ratio,
        "fixed_top10_variance_concentration": float(
            cumulative[min(10, hidden_dim) - 1]
        ),
        "cumulative_variance": {
            str(k): float(cumulative[min(k, hidden_dim) - 1]) for k in K_VALUES
        },
        "n_positive_eigenvalues": int((eigenvalues > 0).sum()),
    }
    return components, eigenvalues, mean, metrics


def compare_mean(reconstructed: np.ndarray, official: np.ndarray) -> dict[str, float]:
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    official = np.asarray(official, dtype=np.float64)
    difference = reconstructed - official
    official_norm = float(np.linalg.norm(official))
    reconstructed_norm = float(np.linalg.norm(reconstructed))
    return {
        "cosine": cosine(reconstructed, official),
        "relative_l2_error": float(np.linalg.norm(difference) / official_norm),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "reconstructed_norm": reconstructed_norm,
        "official_norm": official_norm,
        "norm_ratio": reconstructed_norm / official_norm,
    }


def official_numpy(means: dict[str, Any], label: str, layer: int) -> np.ndarray:
    return means[label]["mean"][layer].detach().cpu().numpy().astype(np.float64)


def load_thesis_cloud_matrices(
    layer: int,
    current_rows: dict[str, Any],
    archive_rows: dict[str, Any],
    excluded_ids: set[str],
) -> tuple[list[np.ndarray], dict[str, int], list[Path]]:
    """Load the exact mixed-vintage thesis comparison cloud at one layer."""
    matrices: list[np.ndarray] = []
    counts: dict[str, int] = {}
    paths: list[Path] = []
    for label in PCA_LABELS:
        archived = label in THESIS_ARCHIVED_LABELS
        root = THESIS_ARCHIVE if archived else THESIS_CURRENT
        rows = archive_rows if archived else current_rows
        path = root / f"{label}_layer{layer}.npy"
        matrix = np.load(path, allow_pickle=False)
        provenance_rows = rows.get("rows", {}).get(label)
        if provenance_rows is None or len(provenance_rows) != matrix.shape[0]:
            raise ValueError(
                f"Thesis row provenance mismatch for {label}@L{layer}: "
                f"matrix={matrix.shape[0]}, rows="
                f"{None if provenance_rows is None else len(provenance_rows)}"
            )
        chain_ids = np.asarray(
            [row["chain_id"] for row in provenance_rows], dtype=object
        )
        keep = ~np.isin(chain_ids, list(excluded_ids))
        retained = np.asarray(matrix[keep], dtype=np.float32)
        matrices.append(retained)
        counts[label] = int(retained.shape[0])
        paths.append(path)
    return matrices, counts, paths


def basis_validation(
    reconstructed: np.ndarray, saved: np.ndarray
) -> dict[str, Any]:
    """Sign/rotation-invariant top-k subspace agreement."""
    cells: dict[str, Any] = {}
    for k in K_VALUES:
        cross = reconstructed[:k] @ saved[:k].T
        singular = np.linalg.svd(cross, compute_uv=False)
        cells[str(k)] = {
            "minimum_principal_cosine": float(singular.min()),
            "mean_principal_cosine": float(singular.mean()),
            "maximum_principal_angle_degrees": float(
                np.degrees(np.arccos(np.clip(singular.min(), -1.0, 1.0)))
            ),
        }
    return cells


def run_analysis(args: argparse.Namespace, provenance: dict[str, Any]) -> dict[str, Any]:
    data_root = args.data_out.resolve()
    extraction_meta_path = data_root / "metadata.json"
    if not extraction_meta_path.is_file():
        raise FileNotFoundError(f"Missing extraction metadata: {extraction_meta_path}")
    extraction = load_json(extraction_meta_path)
    if extraction.get("complete") is not True:
        raise ValueError("Extraction is incomplete")
    extraction_builder_sha = extraction.get("provenance", {}).get("builder_sha256")
    if extraction_builder_sha != EXPECTED_EXTRACTION_BUILDER_SHA256:
        raise ValueError(
            "Unexpected extraction builder SHA: "
            f"{extraction_builder_sha} != {EXPECTED_EXTRACTION_BUILDER_SHA256}"
        )
    if extraction.get("n_records_processed") != 500 and not args.allow_pilot_analysis:
        raise ValueError(
            "PCA result requires all 500 records; use --allow-pilot-analysis only "
            "for mechanical testing"
        )
    official_runtime = verify_official_runtime_pins(OFFICIAL_REPO / "environment.yaml")
    required = [
        OFFICIAL_MEANS,
        OFFICIAL_REPO / "environment.yaml",
        OFFICIAL_REPO / "train-steering-vectors/train_vectors.py",
        OFFICIAL_REPO / "utils/utils.py",
        HYBRID_ROOT / "metadata.json",
        THESIS_BASIS_ROOT / "venhoff_official_pca_alignment.json",
        THESIS_CURRENT / "metadata.json",
        THESIS_CURRENT / "row_index.json",
        THESIS_ARCHIVE / "metadata.json",
        THESIS_ARCHIVE / "row_index.json",
        THESIS_EVAL_IDS,
    ]
    for behaviour, layer in VENHOFF_LAYERS.items():
        required.extend(
            [
                HYBRID_ROOT / f"{behaviour}_single.npy",
                THESIS_BASIS_ROOT / f"pooled_components_layer{layer}.npy",
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing analysis inputs: " + ", ".join(missing))

    current_metadata = load_json(THESIS_CURRENT / "metadata.json")
    archive_metadata = load_json(THESIS_ARCHIVE / "metadata.json")
    if current_metadata.get("clip_window_to_sentence_end") is not False:
        raise ValueError("Unexpected current thesis target-label extraction contract")
    if archive_metadata.get("clip_window_to_sentence_end") is not True:
        raise ValueError("Unexpected archived thesis inert-label extraction contract")
    current_rows = load_json(THESIS_CURRENT / "row_index.json")
    archive_rows = load_json(THESIS_ARCHIVE / "row_index.json")
    excluded_ids = set(load_json(THESIS_EVAL_IDS)["task_ids"])
    if len(excluded_ids) != 50:
        raise ValueError(f"Expected 50 thesis holdout IDs, found {len(excluded_ids)}")

    out = args.result_out.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite result directory: {out}")
    out.mkdir(parents=True)
    means = load_official_means()
    official_labels = tuple(means.keys())
    verified_input_hashes: dict[str, str] = {}
    hash_cache: dict[Path, str] = {}

    def record_hash(path: Path) -> str:
        path = path.resolve()
        if path not in hash_cache:
            hash_cache[path] = sha256(path)
        key = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        verified_input_hashes[key] = hash_cache[path]
        return hash_cache[path]

    def load_extracted(arm: str, label: str, layer: int) -> np.ndarray:
        path = data_root / arm / f"{label}_layer{layer}.npy"
        relative = path.relative_to(data_root).as_posix()
        expected = extraction.get("output_sha256", {}).get(relative)
        observed = record_hash(path)
        if expected is None or observed != expected:
            raise ValueError(
                f"Extraction hash mismatch for {relative}: {observed} != {expected}"
            )
        return np.load(path, allow_pickle=False)

    for path in required:
        record_hash(path)

    reconstruction: dict[str, Any] = {
        "released_mean_cells": {},
        "released_direction_cells": {},
        "count_check": {},
    }
    for label in official_labels:
        observed_count = extraction["counts"]["released"].get(label, 0)
        official_count = int(means[label]["count"])
        reconstruction["count_check"][label] = {
            "reconstructed": observed_count,
            "official": official_count,
            "exact": observed_count == official_count,
        }
        reconstruction["released_mean_cells"][label] = {}
        for layer in LAYERS:
            matrix = load_extracted("released", label, layer)
            reconstructed_mean = matrix.mean(axis=0, dtype=np.float64)
            reconstruction["released_mean_cells"][label][str(layer)] = compare_mean(
                reconstructed_mean, official_numpy(means, label, layer)
            )

    for behaviour in TARGETS:
        layer = VENHOFF_LAYERS[behaviour]
        target = load_extracted("released", behaviour, layer).mean(
            axis=0, dtype=np.float64
        )
        overall = load_extracted("released", "overall", layer).mean(
            axis=0, dtype=np.float64
        )
        reconstructed_direction = target - overall
        official_direction = (
            official_numpy(means, behaviour, layer)
            - official_numpy(means, "overall", layer)
        )
        reconstruction["released_direction_cells"][behaviour] = {
            "layer": layer,
            **compare_mean(reconstructed_direction, official_direction),
        }
    all_counts_exact = all(
        cell["exact"] for cell in reconstruction["count_check"].values()
    )
    min_mean_cosine = min(
        cell["cosine"]
        for label in reconstruction["released_mean_cells"].values()
        for cell in label.values()
    )
    min_direction_cosine = min(
        cell["cosine"]
        for cell in reconstruction["released_direction_cells"].values()
    )
    reconstruction["summary"] = {
        "all_counts_exact": all_counts_exact,
        "minimum_released_mean_cosine": min_mean_cosine,
        "minimum_target_minus_overall_direction_cosine": min_direction_cosine,
        "high_fidelity_gate": bool(
            all_counts_exact and min_mean_cosine >= 0.999 and min_direction_cosine >= 0.98
        ),
        "gate_definition": (
            "all released counts exact; every selected-layer mean cosine >=0.999; "
            "every protocol-layer target-minus-overall direction cosine >=0.98"
        ),
    }

    cloud_results: dict[str, dict[str, Any]] = {
        "thesis_mixed_vintage": {},
        **{arm: {} for arm in ARMS},
    }
    cloud_bases: dict[tuple[str, int], np.ndarray] = {}
    output_hashes: dict[str, str] = {}
    for arm in ARMS:
        for layer in LAYERS:
            matrices = [
                load_extracted(arm, label, layer)
                for label in PCA_LABELS
            ]
            components, eigenvalues, mean, metrics = pca_cloud(matrices)
            cloud_bases[(arm, layer)] = components
            component_path = out / f"{arm}_components_layer{layer}.npy"
            eigenvalue_path = out / f"{arm}_eigenvalues_layer{layer}.npy"
            mean_path = out / f"{arm}_mean_layer{layer}.npy"
            atomic_npy(component_path, components.astype(np.float32))
            atomic_npy(eigenvalue_path, eigenvalues.astype(np.float64))
            atomic_npy(mean_path, mean.astype(np.float64))
            for path in (component_path, eigenvalue_path, mean_path):
                output_hashes[path.name] = sha256(path)
            metrics.update(
                {
                    "labels": list(PCA_LABELS),
                    "counts_by_label": {
                        label: int(matrix.shape[0])
                        for label, matrix in zip(PCA_LABELS, matrices)
                    },
                    "unit": "span row nested within released response/chain",
                    "pca_method": "mean-centred row-pooled exact covariance eigendecomposition",
                    "component_file": component_path.name,
                    "eigenvalue_file": eigenvalue_path.name,
                    "mean_file": mean_path.name,
                }
            )
            cloud_results[arm][str(layer)] = metrics
            del matrices

    thesis_basis_validation: dict[str, Any] = {}
    for layer in LAYERS:
        matrices, counts, source_paths = load_thesis_cloud_matrices(
            layer, current_rows, archive_rows, excluded_ids
        )
        for path in source_paths:
            record_hash(path)
        components, eigenvalues, mean, metrics = pca_cloud(matrices)
        cloud_bases[("thesis_mixed_vintage", layer)] = components
        component_path = out / f"thesis_mixed_vintage_components_layer{layer}.npy"
        eigenvalue_path = out / f"thesis_mixed_vintage_eigenvalues_layer{layer}.npy"
        mean_path = out / f"thesis_mixed_vintage_mean_layer{layer}.npy"
        atomic_npy(component_path, components.astype(np.float32))
        atomic_npy(eigenvalue_path, eigenvalues.astype(np.float64))
        atomic_npy(mean_path, mean.astype(np.float64))
        for path in (component_path, eigenvalue_path, mean_path):
            output_hashes[path.name] = sha256(path)
        metrics.update(
            {
                "labels": list(PCA_LABELS),
                "counts_by_label": counts,
                "unit": "span row nested within thesis response/chain",
                "pca_method": "mean-centred row-pooled exact covariance eigendecomposition",
                "source_contract": (
                    "held-out 50 task IDs; current unclipped target-label rows plus "
                    "archived sentence-clipped initializing/deduction rows"
                ),
                "component_file": component_path.name,
                "eigenvalue_file": eigenvalue_path.name,
                "mean_file": mean_path.name,
            }
        )
        cloud_results["thesis_mixed_vintage"][str(layer)] = metrics
        saved_path = THESIS_BASIS_ROOT / f"pooled_components_layer{layer}.npy"
        saved = np.load(saved_path, allow_pickle=False)
        record_hash(saved_path)
        thesis_basis_validation[str(layer)] = basis_validation(components, saved)
        del matrices

    metric_deltas: dict[str, Any] = {}
    for arm in ARMS:
        metric_deltas[arm] = {}
        for layer in LAYERS:
            thesis = cloud_results["thesis_mixed_vintage"][str(layer)]
            venhoff = cloud_results[arm][str(layer)]
            metric_deltas[arm][str(layer)] = {
                "venhoff_minus_thesis_d_eff70": (
                    venhoff["d_eff70_variance_threshold_dimension"]
                    - thesis["d_eff70_variance_threshold_dimension"]
                ),
                "venhoff_minus_thesis_participation_ratio": (
                    venhoff["participation_ratio"] - thesis["participation_ratio"]
                ),
                "venhoff_minus_thesis_fixed_top10_variance_concentration": (
                    venhoff["fixed_top10_variance_concentration"]
                    - thesis["fixed_top10_variance_concentration"]
                ),
            }

    comparisons: dict[str, Any] = {}
    occurrence_sensitivity: dict[str, Any] = {}
    for behaviour in TARGETS:
        layer = VENHOFF_LAYERS[behaviour]
        hybrid_path = HYBRID_ROOT / f"{behaviour}_single.npy"
        record_hash(hybrid_path)
        directions = {
            "thesis_hybrid": np.load(hybrid_path, allow_pickle=False),
            "official_venhoff": (
                official_numpy(means, behaviour, layer)
                - official_numpy(means, "overall", layer)
            ),
        }
        clouds = {
            "thesis_mixed_vintage": cloud_bases[("thesis_mixed_vintage", layer)],
            "venhoff_released": cloud_bases[("released", layer)],
        }
        cell: dict[str, Any] = {
            "layer": layer,
            "direction_cosine_thesis_hybrid_vs_official": cosine(
                directions["thesis_hybrid"], directions["official_venhoff"]
            ),
            "cells": {},
        }
        for direction_name, direction in directions.items():
            cell["cells"][direction_name] = {
                cloud_name: alignment(direction, components)
                for cloud_name, components in clouds.items()
            }
        comparisons[behaviour] = cell
        occurrence_sensitivity[behaviour] = {
            direction_name: alignment(
                direction, cloud_bases[("occurrence_aware", layer)]
            )
            for direction_name, direction in directions.items()
        }

    disposition = analysis_disposition(reconstruction["summary"])
    current_builder_sha = sha256(Path(__file__))
    result_licensing = {
        "exact_official_vector_to_thesis_basis": {
            "licensed": True,
            "empirical_evidence_status": "exploratory",
            "scope": (
                "Only two_by_two_alignment.*.cells.official_venhoff."
                "thesis_mixed_vintage: exact released official direction from the "
                "hashed .pt measured against the independently reconstructed thesis "
                "basis. This result does not depend on the local model replay."
            ),
            "limitations": (
                "Vector-to-subspace alignment only; not a Venhoff-cloud PCA result, "
                "not a steering-effect estimate, and still inherits the mixed-vintage "
                "thesis-basis provenance boundary."
            ),
        },
        "local_replay_cloud": {
            "licensed": bool(disposition["venhoff_cloud_pca_claim_licensed"]),
            "actual_status": disposition["actual_disposition"],
            "scope": (
                "pca_cloud_metrics.released, pca_cloud_metrics.occurrence_aware, "
                "all two_by_two_alignment cells using venhoff_released, and all "
                "occurrence_aware_alignment_sensitivity cells"
            ),
            "reason": disposition["claim_boundary"],
        },
    }

    result = {
        "schema_version": "venhoff-corpus-latent-analysis-v3",
        "status": disposition,
        "result_licensing": result_licensing,
        "api_cost_usd": 0.0,
        "definitions": {
            "d_eff70_variance_threshold_dimension": (
                "smallest PCA component count reaching 70% cumulative variance"
            ),
            "participation_ratio": "(sum eigenvalues)^2 / sum(eigenvalues^2)",
            "fixed_top10_variance_concentration": (
                "fraction of total variance captured by the first ten PCs"
            ),
            "captured_squared_norm": "E_k = ||U_k u||^2 for unit direction u",
            "retained_norm": "R_k = ||U_k u|| = sqrt(E_k)",
        },
        "released_reconstruction_validation": reconstruction,
        "pca_cloud_metrics": cloud_results,
        "pca_metric_deltas_venhoff_minus_thesis": metric_deltas,
        "thesis_basis_validation_against_prior_reconstruction": thesis_basis_validation,
        "two_by_two_alignment": comparisons,
        "occurrence_aware_alignment_sensitivity": occurrence_sensitivity,
        "provenance": {
            **provenance,
            "builder": Path(__file__).name,
            "builder_sha256": current_builder_sha,
            "analysis_repo_git_commit": git_output(ROOT, "rev-parse", "HEAD"),
            "analysis_repo_git_dirty": bool(git_output(ROOT, "status", "--porcelain")),
            "extraction_root": str(data_root),
            "extraction_metadata_sha256": sha256(extraction_meta_path),
            "thesis_basis_root": str(THESIS_BASIS_ROOT),
            "hybrid_direction_root": str(HYBRID_ROOT),
            "input_sha256": verified_input_hashes,
            "provenance_status": (
                "unresolved provenance — the extraction records a builder SHA but the "
                "exact executed source file was not preserved; the local runtime also "
                "differs from Venhoff's pinned software/CUDA environment"
            ),
            "official_runtime": official_runtime,
            "local_replay_runtime": {
                "torch": extraction.get("provenance", {}).get("torch_version"),
                "transformers": extraction.get("provenance", {}).get(
                    "transformers_version"
                ),
                "device": extraction.get("provenance", {}).get("device"),
                "dtype": extraction.get("provenance", {}).get("dtype"),
                "model_revision": extraction.get("provenance", {}).get(
                    "model_revision"
                ),
            },
            "runtime_difference": (
                "Official: CUDA, torch 2.5.1, transformers 4.47.1, NNSight 0.4.5. "
                "Local replay: MPS, torch 2.11.0, transformers 5.4.0; the extraction "
                "metadata records the exact local dtype. Only the official HF model "
                "revision was unpinned."
            ),
            "extraction_builder_lineage": {
                "executed_builder_sha256": extraction_builder_sha,
                "current_source_sha256": current_builder_sha,
                "sha_matches_current_source": extraction_builder_sha
                == current_builder_sha,
                "exact_executed_source_preserved": False,
                "assessment": (
                    "Subsequent visible edits appear confined to analysis/reporting, "
                    "but the repository cannot establish byte identity or reconstruct "
                    "the exact executed extraction source from the hash alone."
                ),
            },
            "extraction_metadata_correction": (
                "The immutable extraction metadata says Venhoff did not pin a runtime. "
                "That statement is incorrect: environment.yaml pins the software/CUDA "
                "runtime. V3 supersedes that statement without modifying the extraction "
                "artefact; only the HF model revision was unpinned."
            ),
            "output_sha256": output_hashes,
        },
    }
    report_path = out / "venhoff_corpus_latent_analysis.json"
    atomic_json(report_path, result)
    print(f"Analysis complete -> {report_path}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("audit", "extract", "analyse", "all"), default="audit"
    )
    parser.add_argument("--data-out", type=Path, default=DEFAULT_DATA_OUT)
    parser.add_argument("--result-out", type=Path, default=DEFAULT_RESULT_OUT)
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=None,
        help="Optional JSON path for the no-model preflight audit",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Pilot record count. Omit only with --confirm-full.",
    )
    parser.add_argument("--confirm-full", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--allow-batch-sensitivity", action="store_true")
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--allow-pilot-analysis", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    provenance = validate_inputs()
    tokenizer = load_tokenizer()
    means = load_official_means()
    records = load_json(RESPONSES)
    audit = static_audit(records, tokenizer, means)
    if args.audit_output:
        atomic_json(args.audit_output.resolve(), {
            "schema_version": "venhoff-corpus-static-audit-v1",
            "api_cost_usd": 0.0,
            "audit": audit,
            "provenance": provenance,
        })
    print(
        json.dumps(
            {
                "released_counts_exactly_match_official": audit[
                    "released_counts_exactly_match_official"
                ],
                "released_counts": audit["released_counts"],
                "occurrence_aware_counts": audit["occurrence_aware_counts"],
                "location_comparison": audit["location_comparison"],
                "token_lengths": audit["token_lengths"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if not audit["released_counts_exactly_match_official"]:
        raise ValueError("Released static counts do not match official means")
    if args.stage == "audit":
        return 0
    if args.stage in {"extract", "all"}:
        run_extract(args, audit, provenance)
    if args.stage in {"analyse", "all"}:
        run_analysis(args, provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
