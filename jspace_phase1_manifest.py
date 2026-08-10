#!/usr/bin/env python3
"""Create the immutable Phase-1 corpus and external-eval eligibility manifests.

This is a pre-execution command.  It loads no model, computes no Jacobian or
readout, and refuses to overwrite either output.  The effective tokenizer
configuration mirrors ``jlens.from_hf(..., force_bos=True)``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import datasets
import numpy as np
import transformers
from transformers import AutoTokenizer


DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_SPLIT = "train"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
SEED = 20260810
MAX_SEQ_LEN = 128
N_SELECTED = 120
NEAR_OVERLAP_THRESHOLD = 0.80
SHINGLE_N = 5
EVAL_ORDER = (
    "lens-eval-association",
    "lens-eval-typo",
    "lens-eval-multihop",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def shingles(ids: list[int], n: int = SHINGLE_N) -> set[tuple[int, ...]]:
    if len(ids) < n:
        return {tuple(ids)} if ids else set()
    return {tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)}


def jaccard(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def tokenizer_file_hashes(snapshot: Path) -> dict[str, str]:
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "config.json",
        "generation_config.json",
    )
    return {name: sha256_file(snapshot / name) for name in names if (snapshot / name).is_file()}


def effective_tokenizer(snapshot: Path) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if (
        getattr(tokenizer, "bos_token_id", None) is not None
        and hasattr(tokenizer, "add_bos_token")
    ):
        tokenizer.add_bos_token = True
    return tokenizer


def encode(tokenizer: Any, text: str) -> list[int]:
    # These defaults deliberately match HFLensModel.encode except for tensor
    # materialisation and truncation; callers truncate explicitly when needed.
    return list(tokenizer(text, truncation=False)["input_ids"])


def source_hashes(root: Path, eval_root: Path) -> dict[str, str]:
    paths = [
        root / "data/tasks_final.json",
        root / "results/eval/R1-1.5B__E1/eval_task_ids.json",
        root / "results/prereg/JSPACE_R1_STEERING_PILOT_PREREG_2026-08-10.md",
        *(eval_root / f"{slug}.json" for slug in EVAL_ORDER),
        eval_root / "README.md",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing fixed inputs: {missing}")
    return {str(path): sha256_file(path) for path in paths}


def load_external_rows(root: Path, eval_root: Path, tokenizer: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slug in EVAL_ORDER:
        payload = json.loads((eval_root / f"{slug}.json").read_text())
        for item in payload["items"]:
            prompt = item["prompt"]
            rows.append(
                {
                    "source": slug,
                    "name": item["name"],
                    "text": prompt,
                    "token_ids_128": encode(tokenizer, prompt)[:MAX_SEQ_LEN],
                }
            )

    task_rows = json.loads((root / "data/tasks_final.json").read_text())
    by_id = {row["id"]: row for row in task_rows}
    eval_ids = json.loads(
        (root / "results/eval/R1-1.5B__E1/eval_task_ids.json").read_text()
    )["task_ids"]
    if len(eval_ids) != 50 or len(set(eval_ids)) != 50:
        raise ValueError("causal evaluation task manifest is not 50 unique task IDs")
    for task_id in eval_ids:
        prompt = by_id[task_id]["prompt"]
        rows.append(
            {
                "source": "causal_tasks_50",
                "name": task_id,
                "text": prompt,
                "token_ids_128": encode(tokenizer, prompt)[:MAX_SEQ_LEN],
            }
        )
    return rows


def leakage_audit(selected: list[dict[str, Any]], external: list[dict[str, Any]]) -> dict[str, Any]:
    external_prepared = []
    for row in external:
        external_prepared.append(
            {
                **row,
                "normalised": normalise_text(row["text"]),
                "shingles": shingles(row["token_ids_128"]),
            }
        )

    flagged: list[dict[str, Any]] = []
    maxima: list[dict[str, Any]] = []
    for row in selected:
        ids = row["fit_token_ids_128"]
        row_shingles = shingles(ids)
        norm = normalise_text(row["prompt"])
        best: dict[str, Any] | None = None
        for candidate in external_prepared:
            score = jaccard(row_shingles, candidate["shingles"])
            comparison = {
                "selected_order": row["selected_order"],
                "split": row["split"],
                "source_index": row["source_index"],
                "external_source": candidate["source"],
                "external_name": candidate["name"],
                "token_5gram_jaccard": score,
                "normalised_text_equal": norm == candidate["normalised"],
                "token_prefix_equal": ids == candidate["token_ids_128"],
            }
            if best is None or score > best["token_5gram_jaccard"]:
                best = comparison
            if (
                comparison["normalised_text_equal"]
                or comparison["token_prefix_equal"]
                or score >= NEAR_OVERLAP_THRESHOLD
            ):
                flagged.append(comparison)
        assert best is not None
        maxima.append(best)
    return {
        "method": {
            "normalisation": "Unicode_NFKC_casefold_whitespace_collapse",
            "token_sequence": "effective_tokenizer_first_128_ids",
            "near_metric": "set_Jaccard_of_5_token_shingles",
            "near_threshold_inclusive": NEAR_OVERLAP_THRESHOLD,
        },
        "n_external_prompts": len(external),
        "flagged": flagged,
        "max_near_match_by_selected_row": maxima,
        "pass": not flagged,
    }


def selected_corpus(dataset: Any, tokenizer: Any, batch_size: int) -> tuple[list[dict[str, Any]], int]:
    eligible: list[int] = []
    for start in range(0, len(dataset), batch_size):
        texts = dataset[start : min(start + batch_size, len(dataset))]["text"]
        nonblank_positions = [i for i, text in enumerate(texts) if text.strip()]
        if not nonblank_positions:
            continue
        nonblank = [texts[i] for i in nonblank_positions]
        encoded = tokenizer(nonblank, truncation=False, add_special_tokens=True)["input_ids"]
        eligible.extend(
            start + nonblank_positions[j]
            for j, ids in enumerate(encoded)
            if len(ids) >= MAX_SEQ_LEN
        )

    if len(eligible) < N_SELECTED:
        raise ValueError(f"only {len(eligible)} eligible rows; need {N_SELECTED}")
    rng = np.random.Generator(np.random.PCG64(SEED))
    selected_indices = rng.permutation(np.asarray(eligible, dtype=np.int64))[:N_SELECTED]

    rows: list[dict[str, Any]] = []
    for order, source_index_value in enumerate(selected_indices.tolist()):
        source_index = int(source_index_value)
        prompt = dataset[source_index]["text"]
        ids = encode(tokenizer, prompt)
        if len(ids) < MAX_SEQ_LEN:
            raise AssertionError("selected row no longer meets eligibility rule")
        split = "fit_a" if order < 50 else "fit_b" if order < 100 else "heldout"
        rows.append(
            {
                "selected_order": order,
                "split": split,
                "source_index": source_index,
                "utf8_sha256": sha256_bytes(prompt.encode("utf-8")),
                "utf8_bytes": len(prompt.encode("utf-8")),
                "token_count": len(ids),
                "token_ids": ids,
                "fit_token_ids_128": ids[:MAX_SEQ_LEN],
                "fit_prefix_sha256": sha256_bytes(
                    np.asarray(ids[:MAX_SEQ_LEN], dtype="<i8").tobytes()
                ),
                "prompt": prompt,
            }
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["fit_prefix_sha256"]].append(row)
    duplicates = [
        [
            {
                "selected_order": row["selected_order"],
                "split": row["split"],
                "source_index": row["source_index"],
            }
            for row in group
        ]
        for group in groups.values()
        if len(group) > 1
    ]
    if duplicates:
        raise ValueError(
            "the first 120 eligible-row permutation entries contain duplicate "
            f"128-token prefixes; amendment/resampling required: {duplicates}"
        )
    return rows, len(eligible)


def evaluation_eligibility(eval_root: Path, tokenizer: Any) -> dict[str, Any]:
    evaluations: dict[str, Any] = {}
    for slug in EVAL_ORDER:
        payload = json.loads((eval_root / f"{slug}.json").read_text())
        items: list[dict[str, Any]] = []
        for item_index, item in enumerate(payload["items"]):
            prompt_ids = encode(tokenizer, item["prompt"])
            if not prompt_ids:
                raise ValueError(f"empty prompt tokenization: {slug}:{item['name']}")
            if len(prompt_ids) > MAX_SEQ_LEN:
                raise ValueError(
                    f"external prompt exceeds max_seq_len={MAX_SEQ_LEN}: "
                    f"{slug}:{item['name']}:{len(prompt_ids)}"
                )
            labels = []
            for label_index, label in enumerate(item["intermediates"]):
                leading_ids = list(
                    tokenizer(" " + label, add_special_tokens=False, truncation=False)[
                        "input_ids"
                    ]
                )
                bare_ids = list(
                    tokenizer(label, add_special_tokens=False, truncation=False)[
                        "input_ids"
                    ]
                )
                labels.append(
                    {
                        "label_index": label_index,
                        "label": label,
                        "scored_form": " " + label,
                        "scored_token_ids": leading_ids,
                        "eligible": len(leading_ids) == 1,
                        "bare_token_ids_diagnostic": bare_ids,
                    }
                )
            eligible_labels = [row for row in labels if row["eligible"]]
            items.append(
                {
                    "item_index": item_index,
                    "name": item["name"],
                    "prompt": item["prompt"],
                    "prompt_utf8_sha256": sha256_bytes(item["prompt"].encode("utf-8")),
                    "prompt_token_ids": prompt_ids,
                    "readout_position": len(prompt_ids) - 1,
                    "readout_rule": "final_prompt_token",
                    "target_locator_only": item.get("target"),
                    "labels": labels,
                    "n_eligible_labels": len(eligible_labels),
                    "item_eligible": bool(eligible_labels),
                }
            )
        evaluations[slug] = {
            "n_items_total": len(items),
            "n_items_eligible": sum(row["item_eligible"] for row in items),
            "n_labels_total": sum(len(row["labels"]) for row in items),
            "n_labels_eligible": sum(row["n_eligible_labels"] for row in items),
            "items": items,
        }
    return evaluations


def publish_many(payloads: dict[Path, bytes]) -> None:
    for output in payloads:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    parent = next(iter(payloads)).parent
    if any(path.parent != parent for path in payloads):
        raise ValueError("atomic manifest publication requires one output directory")
    lock = parent / ".jspace_phase1_manifest.lock"
    lock.mkdir()
    staged: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for output, content in payloads.items():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{output.name}.", suffix=".tmp", dir=parent
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            staged[output] = temporary
        for output, temporary in staged.items():
            os.link(temporary, output)
            published.append(output)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        for output in published:
            output.unlink(missing_ok=True)
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        shutil.rmtree(lock, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--jlens-checkout", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--corpus-output", type=Path, required=True)
    parser.add_argument("--eligibility-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    root = args.root.resolve()
    model_snapshot = args.model_snapshot.resolve()
    eval_root = args.jlens_checkout.resolve() / "data/evaluations"
    if not model_snapshot.is_dir() or not eval_root.is_dir():
        raise FileNotFoundError("model snapshot or pinned evaluation directory missing")
    jlens_commit = os.popen(
        f"git -C {str(args.jlens_checkout.resolve())!r} rev-parse HEAD"
    ).read().strip()
    if jlens_commit != "581d398613e5602a5af361e1c34d3a92ea82ba8e":
        raise ValueError(f"J-lens checkout commit mismatch: {jlens_commit}")

    tokenizer = effective_tokenizer(model_snapshot)
    dataset = datasets.load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
        cache_dir=str(args.dataset_cache.resolve()),
    )
    rows, n_eligible = selected_corpus(dataset, tokenizer, args.batch_size)
    external = load_external_rows(root, eval_root, tokenizer)
    overlap = leakage_audit(rows, external)
    if not overlap["pass"]:
        raise ValueError(f"leakage audit failed: {overlap['flagged']}")
    evaluations = evaluation_eligibility(eval_root, tokenizer)
    for slug, report in evaluations.items():
        if report["n_items_eligible"] < 50:
            raise ValueError(f"{slug} has fewer than 50 eligible items")

    environment = {
        "python": platform.python_version(),
        "datasets": datasets.__version__,
        "numpy": np.__version__,
        "transformers": transformers.__version__,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "tokenizer_add_bos_token": getattr(tokenizer, "add_bos_token", None),
        "tokenizer_bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "tokenizer_eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "tokenizer_vocab_size": len(tokenizer),
    }
    fixed_hashes = source_hashes(root, eval_root)
    corpus = {
        "schema_version": "rom-jspace-r1-fit-manifest-v1",
        "status": "sealed_pre_execution_input",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "tokenizer_snapshot": str(model_snapshot),
        "tokenizer_files_sha256": tokenizer_file_hashes(model_snapshot),
        "effective_tokenizer_environment": environment,
        "dataset": {
            "id": DATASET_ID,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "revision": DATASET_REVISION,
            "fingerprint": dataset._fingerprint,
            "n_source_rows": len(dataset),
            "n_eligible_nonblank_rows_at_least_128_tokens": n_eligible,
            "cache_files": dataset.cache_files,
        },
        "selection": {
            "bit_generator": "numpy.random.PCG64",
            "numpy_version": np.__version__,
            "seed": SEED,
            "rule": "permutation_of_eligible_source_row_indices_first_120",
            "split_rule": "first_50_fit_a_next_50_fit_b_last_20_heldout",
            "max_seq_len": MAX_SEQ_LEN,
            "prompt_format": "raw_text_no_chat_template",
            "force_bos": True,
        },
        "fixed_input_sha256": fixed_hashes,
        "deduplication": {
            "key": "sha256_little_endian_int64_first_128_effective_token_ids",
            "duplicate_groups": [],
            "pass": True,
        },
        "leakage_audit": overlap,
        "rows": rows,
    }
    eligibility = {
        "schema_version": "rom-jspace-r1-eval-eligibility-v1",
        "status": "sealed_pre_readout_input",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "jlens_commit": jlens_commit,
        "tokenizer_files_sha256": tokenizer_file_hashes(model_snapshot),
        "effective_tokenizer_environment": environment,
        "fixed_input_sha256": fixed_hashes,
        "scoring_contract": {
            "prompt_rendering": "raw_prompt_string_no_chat_template",
            "max_seq_len": MAX_SEQ_LEN,
            "readout_position": "final_prompt_token_for_all_three_evaluations",
            "multihop_target_field": "locator_only_not_scored",
            "scored_fields": "intermediates",
            "target_encoding": "one_leading_ASCII_space_plus_label_add_special_tokens_false",
            "single_token_eligibility": "exactly_one_effective_R1_token",
            "partial_item_rule": "drop_ineligible_labels_keep_item_if_at_least_one_eligible_label",
            "zero_eligible_item_rule": "exclude_item_before_any_ranks",
            "minimum_eligible_items_per_evaluation": 50,
        },
        "evaluation_order": list(EVAL_ORDER),
        "evaluations": evaluations,
    }
    publish_many(
        {
            args.corpus_output.resolve(): canonical_json_bytes(corpus),
            args.eligibility_output.resolve(): canonical_json_bytes(eligibility),
        }
    )
    print(
        json.dumps(
            {
                "corpus_output": str(args.corpus_output.resolve()),
                "corpus_sha256": sha256_file(args.corpus_output.resolve()),
                "eligibility_output": str(args.eligibility_output.resolve()),
                "eligibility_sha256": sha256_file(args.eligibility_output.resolve()),
                "n_source_rows": len(dataset),
                "n_eligible": n_eligible,
                "selected_split_counts": {split: sum(row["split"] == split for row in rows) for split in ("fit_a", "fit_b", "heldout")},
                "external_eligible_items": {slug: report["n_items_eligible"] for slug, report in evaluations.items()},
                "leakage_pass": overlap["pass"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
