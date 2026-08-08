#!/usr/bin/env python3
"""Read-only record-level train/evaluation overlap audit for the P5 protocol.

The script reads existing JSON inputs and writes only to an explicitly supplied
path under ``.codex/out``.  It never edits training data, evaluation data,
results, preregistrations, or ledgers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def shingles(text: str, width: int = 5) -> frozenset[tuple[str, ...]]:
    words = WORD_RE.findall(normalise(text))
    if len(words) < width:
        return frozenset({tuple(words)}) if words else frozenset()
    return frozenset(tuple(words[i : i + width]) for i in range(len(words) - width + 1))


def load_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    if isinstance(rows, dict) and isinstance(rows.get("tasks"), list):
        rows = rows["tasks"]
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON array or an object with a tasks array")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str):
            raise ValueError(f"{path}: row {index} lacks a string prompt")
    return rows


def audit_one(eval_rows: list[dict], train_rows: list[dict], threshold: float) -> dict:
    train_norm: dict[str, list[int]] = defaultdict(list)
    train_shingles: list[frozenset[tuple[str, ...]]] = []
    inverted: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for index, row in enumerate(train_rows):
        train_norm[normalise(row["prompt"])].append(index)
        grams = shingles(row["prompt"])
        train_shingles.append(grams)
        for gram in grams:
            inverted[gram].add(index)

    exact, near = [], []
    best_scores = []
    for eval_index, row in enumerate(eval_rows):
        norm = normalise(row["prompt"])
        exact_indices = train_norm.get(norm, [])
        if exact_indices:
            for train_index in exact_indices:
                exact.append(
                    {
                        "eval_index": eval_index,
                        "eval_id": row.get("id"),
                        "eval_harmful": row.get("harmful"),
                        "eval_source": row.get("source"),
                        "train_index": train_index,
                        "train_id": train_rows[train_index].get("id"),
                        "train_source": train_rows[train_index].get("source"),
                        "normalised_exact": True,
                    }
                )
            best_scores.append(1.0)
            continue

        grams = shingles(row["prompt"])
        candidate_indices: set[int] = set()
        for gram in grams:
            candidate_indices.update(inverted.get(gram, ()))
        best_score, best_index = 0.0, None
        for train_index in candidate_indices:
            other = train_shingles[train_index]
            union = grams | other
            score = len(grams & other) / len(union) if union else 0.0
            if score > best_score:
                best_score, best_index = score, train_index
        best_scores.append(best_score)
        if best_index is not None and best_score >= threshold:
            near.append(
                {
                    "eval_index": eval_index,
                    "eval_id": row.get("id"),
                    "eval_harmful": row.get("harmful"),
                    "eval_source": row.get("source"),
                    "train_index": best_index,
                    "train_id": train_rows[best_index].get("id"),
                    "train_source": train_rows[best_index].get("source"),
                    "word_5gram_jaccard": round(best_score, 6),
                }
            )

    exact_eval_indices = {item["eval_index"] for item in exact}
    near_eval_indices = {item["eval_index"] for item in near}
    by_stratum = {}
    for harmful in (True, False):
        indices = {i for i, row in enumerate(eval_rows) if row.get("harmful") is harmful}
        by_stratum["harmful" if harmful else "benign"] = {
            "n_eval": len(indices),
            "n_normalised_exact": len(indices & exact_eval_indices),
            "n_near_only": len(indices & near_eval_indices),
        }

    return {
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "n_normalised_exact": len(exact_eval_indices),
        "n_near_only": len(near_eval_indices),
        "near_rule": f"word 5-gram Jaccard >= {threshold:.3f} after NFKC/case/whitespace normalisation; exact matches excluded",
        "by_eval_stratum": by_stratum,
        "normalised_exact_matches": exact,
        "near_only_matches": near,
        "maximum_nonexact_similarity": round(max((s for s in best_scores if s < 1.0), default=0.0), 6),
    }


def parse_training(value: str) -> tuple[str, Path]:
    role, separator, raw_path = value.partition("=")
    if not separator or not role or not raw_path:
        raise argparse.ArgumentTypeError("training inputs must be ROLE=PATH")
    return role, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--training", action="append", type=parse_training, required=True)
    parser.add_argument("--near-threshold", type=float, default=0.80)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    resolved_out = args.out.resolve()
    if ".codex/out" not in resolved_out.as_posix():
        raise SystemExit("refusing to write outside .codex/out")
    if not 0.0 < args.near_threshold <= 1.0:
        raise SystemExit("--near-threshold must be in (0, 1]")

    eval_path = args.eval.resolve()
    eval_rows = load_rows(eval_path)
    if len({normalise(row["prompt"]) for row in eval_rows}) != len(eval_rows):
        raise SystemExit("evaluation file contains duplicate normalised prompts")

    document = {
        "status": "planning audit; not a scientific result or thesis evidence artefact",
        "evaluation": {
            "path": str(eval_path),
            "bytes": eval_path.stat().st_size,
            "sha256": sha256(eval_path),
            "n": len(eval_rows),
            "n_harmful": sum(row.get("harmful") is True for row in eval_rows),
            "n_benign": sum(row.get("harmful") is False for row in eval_rows),
        },
        "normalisation": "Unicode NFKC, casefold, collapse whitespace",
        "training_roles": {},
        "admission_consequence": (
            "Any exact or near-overlapping evaluation prompt is excluded from the final held-out safety benchmark for that checkpoint role."
        ),
    }
    for role, path in args.training:
        resolved = path.resolve()
        rows = load_rows(resolved)
        result = audit_one(eval_rows, rows, args.near_threshold)
        result["input"] = {
            "path": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": sha256(resolved),
        }
        document["training_roles"][role] = result

    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    resolved_out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
