#!/usr/bin/env python3
"""Build leakage-safe, balance-audited KTO labels for the pt18 planning arm.

The problem split is created before any backtracking threshold is computed. The strict
training file uses only chains with an existing correctness verdict and exact-matches the
high/low classes within (correctness, length-band) strata. A larger candidate file retains
the quartile arms with explicit missing-correctness status; it is not training-authorised.
No training, generation, annotation, or result-directory write occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
ANNOTATED = ROOT / "data/annotated_R1-1.5B.json"
TASKS = ROOT / "data/tasks_final.json"
CORRECTNESS = ROOT / "data/correctness_R1-1.5B_pilot.json"
OUT = ROOT / ".codex/out"
DEFAULT_SEED = 20260802
SPLIT_FRACTIONS = {"train": 0.60, "discovery": 0.20, "eval": 0.20}
LENGTH_BANDS = (
    ("lt_2048", 0, 2048),
    ("2048_4095", 2048, 4096),
    ("4096_8191", 4096, 8192),
    ("cap_8192_plus", 8192, None),
)
TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True)
class SplitAudit:
    counts: dict[str, int]
    intersections: dict[str, int]
    union_count: int
    zero_overlap: bool


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def problem_split(problem_ids_by_category: dict[str, list[str]], seed: int) -> dict[str, list[str]]:
    """Category-stratified 60/20/20 split; thresholds are deliberately absent here."""
    rng = np.random.default_rng(seed)
    splits = {name: [] for name in SPLIT_FRACTIONS}
    for category in sorted(problem_ids_by_category):
        ids = np.asarray(sorted(problem_ids_by_category[category]), dtype=object)
        ids = ids[rng.permutation(len(ids))]
        n = len(ids)
        n_train = int(round(SPLIT_FRACTIONS["train"] * n))
        n_discovery = int(round(SPLIT_FRACTIONS["discovery"] * n))
        cuts = {
            "train": ids[:n_train],
            "discovery": ids[n_train:n_train + n_discovery],
            "eval": ids[n_train + n_discovery:],
        }
        for split, values in cuts.items():
            splits[split].extend(map(str, values.tolist()))
    return {name: sorted(values) for name, values in splits.items()}


def leakage_check(splits: dict[str, Iterable[str]]) -> SplitAudit:
    """Prove that no problem id appears in more than one split; raise on any leak."""
    sets = {name: set(values) for name, values in splits.items()}
    expected = set(SPLIT_FRACTIONS)
    if set(sets) != expected:
        raise ValueError(f"expected split keys {sorted(expected)}, got {sorted(sets)}")
    intersections = {}
    names = sorted(sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            intersections[f"{left}&{right}"] = len(sets[left] & sets[right])
    audit = SplitAudit(
        counts={name: len(values) for name, values in sets.items()},
        intersections=intersections,
        union_count=len(set().union(*sets.values())),
        zero_overlap=all(value == 0 for value in intersections.values()),
    )
    if not audit.zero_overlap:
        raise ValueError(f"problem leakage detected: {audit.intersections}")
    return audit


def length_band(n_tokens: int) -> str:
    for name, low, high in LENGTH_BANDS:
        if n_tokens >= low and (high is None or n_tokens < high):
            return name
    raise AssertionError(f"unreachable length band for {n_tokens}")


def lexical_token_entropy_bits(text: str) -> float:
    """Shannon entropy of lexical token types; a proxy, not model predictive entropy."""
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0.0
    counts = np.asarray(list(Counter(tokens).values()), dtype=float)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def chain_metrics(row: dict, task: dict, correctness: dict | None) -> dict:
    annotations = row.get("annotations") or []
    n_backtracking = sum(annotation.get("label") == "backtracking"
                         for annotation in annotations)
    n_tokens = int(row.get("n_tokens") or 0)
    if n_tokens <= 0:
        raise ValueError(f"{row.get('task_id')} has invalid n_tokens={n_tokens}")
    correctness_value = None if correctness is None else correctness.get("correct")
    correctness_observed = isinstance(correctness_value, bool)
    return {
        "problem_id": str(row["task_id"]),
        "task_id": str(row["task_id"]),
        "category": str(task.get("category", row.get("category", "unknown"))),
        "difficulty_proxy": str(task.get("difficulty", "unknown")),
        "n_tokens": n_tokens,
        "length_band": length_band(n_tokens),
        "n_annotated_sentences": len(annotations),
        "n_backtracking_sentences": int(n_backtracking),
        "backtracking_per_1k_tokens": 1000.0 * n_backtracking / n_tokens,
        "lexical_token_entropy_bits": lexical_token_entropy_bits(row.get("chain", "")),
        "correctness": bool(correctness_value) if correctness_observed else None,
        "correctness_status": "observed" if correctness_observed else "unobserved",
        "annotation_complete": bool(row.get("annotation_complete")),
        "prompt": row.get("prompt", ""),
        "completion": row.get("chain", ""),
    }


def select_rank_quartiles(train_rows: list[dict], seed: int) -> tuple[list[dict], list[dict], dict]:
    """Exact rank quartiles; seeded tie breaks prevent task-id order from selecting zero ties."""
    if len(train_rows) < 4:
        raise ValueError("need at least four complete train rows for quartiles")
    rng = np.random.default_rng(seed)
    tie_break = {row["problem_id"]: float(rng.random()) for row in train_rows}
    ordered = sorted(train_rows, key=lambda row: (
        row["backtracking_per_1k_tokens"], tie_break[row["problem_id"]]))
    n_each = len(ordered) // 4
    low = [dict(row, source_quartile="bottom") for row in ordered[:n_each]]
    high = [dict(row, source_quartile="top") for row in ordered[-n_each:]]
    if set(row["problem_id"] for row in low) & set(row["problem_id"] for row in high):
        raise AssertionError("rank quartiles overlap")
    thresholds = {
        "n_complete_train": len(train_rows),
        "n_each": n_each,
        "bottom_selected_max": max(row["backtracking_per_1k_tokens"] for row in low),
        "top_selected_min": min(row["backtracking_per_1k_tokens"] for row in high),
        "numpy_q25_linear": float(np.quantile(
            [row["backtracking_per_1k_tokens"] for row in train_rows], 0.25)),
        "numpy_q75_linear": float(np.quantile(
            [row["backtracking_per_1k_tokens"] for row in train_rows], 0.75)),
        "tie_rule": "exact rank quartiles; seeded random order within equal rates",
    }
    return low, high, thresholds


def exact_match_correctness_length(low: list[dict], high: list[dict],
                                   seed: int) -> tuple[list[dict], dict]:
    """Keep only observed correctness and exact-match both classes within sealed strata."""
    rng = np.random.default_rng(seed)
    by_class: dict[str, dict[tuple[bool, str], list[dict]]] = {
        "bottom": defaultdict(list), "top": defaultdict(list)}
    for label, rows in (("bottom", low), ("top", high)):
        for row in rows:
            if row["correctness_status"] != "observed":
                continue
            by_class[label][(bool(row["correctness"]), row["length_band"])].append(row)

    matched = []
    audit = {"strata": {}, "dropped_missing_correctness": {
        "bottom": sum(row["correctness_status"] != "observed" for row in low),
        "top": sum(row["correctness_status"] != "observed" for row in high),
    }}
    strata = sorted(set(by_class["bottom"]) | set(by_class["top"]), key=str)
    for stratum in strata:
        bottom = list(by_class["bottom"].get(stratum, []))
        top = list(by_class["top"].get(stratum, []))
        rng.shuffle(bottom)
        rng.shuffle(top)
        keep = min(len(bottom), len(top))
        selected_bottom, selected_top = bottom[:keep], top[:keep]
        matched.extend(dict(row, desirable=False, preference_label="undesirable")
                       for row in selected_bottom)
        matched.extend(dict(row, desirable=True, preference_label="desirable")
                       for row in selected_top)
        key = f"correct={stratum[0]}|length={stratum[1]}"
        audit["strata"][key] = {
            "available_bottom": len(bottom), "available_top": len(top),
            "kept_per_class": keep,
        }
    matched.sort(key=lambda row: row["problem_id"])
    counts = Counter(row["desirable"] for row in matched)
    if counts[True] != counts[False]:
        raise AssertionError(f"matched label imbalance: {counts}")
    return matched, audit


def shuffled_control(rows: list[dict], seed: int) -> list[dict]:
    """Shuffle labels within correctness/length strata, preserving exact covariate counts."""
    rng = np.random.default_rng(seed)
    grouped: dict[tuple[bool, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(bool(row["correctness"]), row["length_band"])].append(row)
    output = []
    for stratum in sorted(grouped, key=str):
        group = sorted(grouped[stratum], key=lambda row: row["problem_id"])
        labels = np.asarray([row["desirable"] for row in group], dtype=bool)
        shuffled = labels[rng.permutation(len(labels))]
        for row, label in zip(group, shuffled):
            copied = dict(row)
            copied["desirable"] = bool(label)
            copied["preference_label"] = "desirable" if label else "undesirable"
            copied["control"] = "label-shuffled within correctness x length stratum"
            output.append(copied)
    return sorted(output, key=lambda row: row["problem_id"])


def _numeric_summary(rows: list[dict], field: str) -> dict:
    values = np.asarray([row[field] for row in rows], dtype=float)
    return {
        "mean": float(values.mean()) if values.size else None,
        "median": float(np.median(values)) if values.size else None,
        "sd": float(values.std(ddof=1)) if values.size > 1 else None,
        "min": float(values.min()) if values.size else None,
        "max": float(values.max()) if values.size else None,
    }


def balance_summary(rows: list[dict]) -> dict:
    result = {}
    for desirable, label in ((True, "desirable"), (False, "undesirable")):
        group = [row for row in rows if row["desirable"] is desirable]
        result[label] = {
            "n": len(group),
            "correctness": dict(sorted(Counter(str(row["correctness"]) for row in group).items())),
            "length_band": dict(sorted(Counter(row["length_band"] for row in group).items())),
            "difficulty_proxy": dict(sorted(Counter(row["difficulty_proxy"] for row in group).items())),
            "category": dict(sorted(Counter(row["category"] for row in group).items())),
            "n_tokens": _numeric_summary(group, "n_tokens"),
            "backtracking_per_1k_tokens": _numeric_summary(group, "backtracking_per_1k_tokens"),
            "lexical_token_entropy_bits": _numeric_summary(group, "lexical_token_entropy_bits"),
        }
    return result


def _compact_training_row(row: dict) -> dict:
    return {key: row[key] for key in (
        "problem_id", "task_id", "prompt", "completion", "desirable", "preference_label",
        "source_quartile", "backtracking_per_1k_tokens", "n_backtracking_sentences",
        "n_annotated_sentences", "n_tokens", "length_band", "correctness",
        "correctness_status", "difficulty_proxy", "category", "lexical_token_entropy_bits",
    ) if key in row}


def render_report(meta: dict, balance: dict, shuffled_balance: dict) -> str:
    leak = meta["leakage_check"]
    thresholds = meta["quartiles"]
    lines = [
        "# KTO label balance — pt18 construction only",
        "",
        "**Status:** labels built; no training run. **The strict file is an audit subset, not "
        "training-ready**, because the available correctness pilot leaves too few matched "
        "high-backtracking chains.",
        "",
        "## Leakage and split order",
        "",
        f"Problem-level split was created first with seed {meta['seed']}, stratified by the ten "
        "task categories: train/discovery/eval = "
        f"{leak['counts']['train']}/{leak['counts']['discovery']}/{leak['counts']['eval']}.",
        f"Pairwise problem-id intersections: `{leak['intersections']}`. "
        f"**Zero overlap: {leak['zero_overlap']}**.",
        "Backtracking quartiles and all matching decisions were computed only after this check, "
        "and only inside the train split.",
        "",
        "## Label construction",
        "",
        f"Among {thresholds['n_complete_train']} annotation-complete train chains, exact rank "
        f"quartiles selected {thresholds['n_each']} bottom and {thresholds['n_each']} top "
        "candidates. Bottom selected max = "
        f"{thresholds['bottom_selected_max']:.6f} and top selected min = "
        f"{thresholds['top_selected_min']:.6f} backtracking-labelled sentences per 1,000 "
        "generated tokens.",
        "Ties were ordered by a seeded random key; this matters because the linear 25th "
        f"percentile is {thresholds['numpy_q25_linear']:.6f} and many chains have zero "
        "backtracking labels.",
        f"Strict exact matching retained {meta['strict_n_per_class']} chains per class across "
        "(observed correctness x fixed length-band) strata.",
        "",
        "Correctness coverage is limited to the existing 200-chain pilot, whose provenance says "
        "`allow_truncated=false`. Missingness is therefore not plausibly random with respect to "
        "length: unknown correctness was not treated as a value and was excluded from the strict "
        "file. The larger candidate file is audit-only until correctness is completed under a "
        "sealed rule that covers cap-hit chains.",
        "",
        "## Strict matched balance",
        "",
        "| class | n | correctness counts | length-band counts | difficulty proxy | "
        "mean tokens | mean lexical token H (bits) | mean bt/1k |",
        "|---|---:|---|---|---|---:|---:|---:|",
    ]
    for label in ("desirable", "undesirable"):
        row = balance[label]
        lines.append(
            f"| {label} | {row['n']} | `{row['correctness']}` | `{row['length_band']}` | "
            f"`{row['difficulty_proxy']}` | {row['n_tokens']['mean']:.1f} | "
            f"{row['lexical_token_entropy_bits']['mean']:.3f} | "
            f"{row['backtracking_per_1k_tokens']['mean']:.3f} |")
    lines += [
        "",
        "Correctness and length-band counts are equal by construction. Difficulty and lexical "
        "token entropy are reported diagnostics, not additional matching variables.",
        "",
        "`lexical_token_entropy_bits` is Shannon entropy over lower-cased lexical token types in "
        "the generated chain. It is a deterministic text-diversity proxy, **not** next-token "
        "predictive entropy from the model; predictive-entropy coverage is not available for the "
        "full label set.",
        "",
        "## Shuffled control",
        "",
        f"The control contains {sum(row['n'] for row in shuffled_balance.values())} chains. "
        "Labels were permuted within the same correctness x length strata, so class totals and "
        "those two matching margins are preserved while label assignment is randomized subject "
        "to the fixed seed. With only four strict rows this control is an implementation audit, "
        "not an inferential comparator; a permutation may retain some original labels.",
        "",
        "## Files",
        "",
        "- `kto_labels_splits.json`: frozen problem IDs and leakage audit.",
        "- `kto_labels_candidates.json`: all top/bottom train-quartile candidates; includes "
        "missing correctness and is not training-authorised.",
        "- `kto_labels_train.json`: strict correctness-observed, exactly matched **audit "
        "subset**; not training-ready at the current coverage.",
        "- `kto_labels_shuffled_control.json`: within-stratum label-shuffled control.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--annotated", type=Path, default=ANNOTATED)
    parser.add_argument("--tasks", type=Path, default=TASKS)
    parser.add_argument("--correctness", type=Path, default=CORRECTNESS)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()

    annotated = json.loads(args.annotated.read_text())
    tasks = json.loads(args.tasks.read_text())
    correctness = json.loads(args.correctness.read_text())
    correctness_provenance_path = args.correctness.with_name(
        args.correctness.stem + ".provenance.json")
    correctness_provenance = (json.loads(correctness_provenance_path.read_text())
                              if correctness_provenance_path.exists() else None)
    tasks_by_id = {str(task["id"]): task for task in tasks}
    annotated_by_id = {str(row["task_id"]): row for row in annotated}
    if set(tasks_by_id) != set(annotated_by_id):
        raise ValueError("tasks_final and annotated problem-id sets differ")

    by_category: dict[str, list[str]] = defaultdict(list)
    for task_id, task in tasks_by_id.items():
        by_category[str(task["category"])].append(task_id)
    splits = problem_split(dict(by_category), args.seed)
    leak = leakage_check(splits)

    train_rows = []
    for problem_id in splits["train"]:
        row = annotated_by_id[problem_id]
        if not row.get("annotation_complete") or not row.get("annotations"):
            continue
        train_rows.append(chain_metrics(
            row, tasks_by_id[problem_id], correctness.get(problem_id)))

    low, high, thresholds = select_rank_quartiles(train_rows, args.seed + 1)
    candidates = sorted(
        [dict(row, desirable=False, preference_label="undesirable") for row in low]
        + [dict(row, desirable=True, preference_label="desirable") for row in high],
        key=lambda row: row["problem_id"])
    matched, matching_audit = exact_match_correctness_length(low, high, args.seed + 2)
    control = shuffled_control(matched, args.seed + 3)
    balance = balance_summary(matched)
    control_balance = balance_summary(control)

    meta = {
        "script": "pt18_build_kto_labels.py",
        "status": "label construction only; no training",
        "seed": args.seed,
        "split_rule": SPLIT_FRACTIONS,
        "leakage_check": {
            "counts": leak.counts, "intersections": leak.intersections,
            "union_count": leak.union_count, "zero_overlap": leak.zero_overlap,
        },
        "quartiles": thresholds,
        "matching": matching_audit,
        "strict_n_per_class": balance["desirable"]["n"],
        "strict_training_ready": False,
        "strict_training_blocker": (
            "pilot correctness coverage is asymmetric across quartiles and exact "
            "correctness x length matching leaves too few chains for the intended pt18 run"),
        "correctness_provenance": correctness_provenance,
        "sources": {
            str(args.annotated): sha256(args.annotated),
            str(args.tasks): sha256(args.tasks),
            str(args.correctness): sha256(args.correctness),
            **({str(correctness_provenance_path): sha256(correctness_provenance_path)}
               if correctness_provenance_path.exists() else {}),
        },
        "length_bands": [
            {"name": name, "low_inclusive": low, "high_exclusive": high}
            for name, low, high in LENGTH_BANDS],
        "entropy_metric": "lexical token-type Shannon entropy in bits; not predictive entropy",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "kto_labels_splits.json").write_text(json.dumps({
        "metadata": meta, "splits": splits}, indent=2) + "\n")
    (args.out_dir / "kto_labels_candidates.json").write_text(json.dumps({
        "metadata": meta,
        "warning": "audit candidates; missing correctness rows are not training-authorised",
        "rows": [_compact_training_row(row) for row in candidates],
    }, indent=2) + "\n")
    (args.out_dir / "kto_labels_train.json").write_text(json.dumps({
        "metadata": meta, "balance": balance,
        "rows": [_compact_training_row(row) for row in matched],
    }, indent=2) + "\n")
    (args.out_dir / "kto_labels_shuffled_control.json").write_text(json.dumps({
        "metadata": meta, "balance": control_balance,
        "rows": [_compact_training_row(row) | {"control": row["control"]} for row in control],
    }, indent=2) + "\n")
    (args.out_dir / "KTO_BALANCE.md").write_text(
        render_report(meta, balance, control_balance))
    print(json.dumps({
        "split_counts": leak.counts,
        "zero_problem_overlap": leak.zero_overlap,
        "quartile_candidates_per_class": thresholds["n_each"],
        "strict_matched_per_class": balance["desirable"]["n"],
        "output_dir": str(args.out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
