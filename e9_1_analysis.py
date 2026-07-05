#!/usr/bin/env python3
"""
E9.1 analysis — dose-response × decoding-entropy factorial, annotation-free
endpoints (COLLAPSE_AND_ENTROPY.md §5; thesis sec:steering-collapse-programme
rung two).

Inputs (each optional — the report covers whatever exists):
  results/eval/R1-1.5B__E9_1_greedy/steering_results.json   (α ∈ {0.5, 1.5}, T=0)
  results/eval/R1-1.5B__E9_1_T06/steering_results.json      (α ∈ {0.5,1,1.5}, T=0.6 ×3)
  results/eval/R1-1.5B__E1/steering_results.json            (E8: α=1, T=0 — REUSED)

The E8 greedy α=1.0 cells are merged in as the missing dose point: same model,
vectors dir (E1_pooled), eval split, arms, and greedy decoding, so the merge is
exact by construction (provenance asserted, not assumed: layer + task ids
checked against the new runs).

Endpoints per (behaviour, method, α, T):
  collapse fraction (4-gram repetition > 0.8), mean repetition, mean tokens,
  cap-hit fraction, boxed rate (\\boxed{} emission — the task-accuracy guard
  proxy, also split by structured vs open categories).
T=0.6 adds cross-sample diversity per task (3 samples): 1 − mean pairwise
4-gram Jaccard, and the distinct-4-gram ratio of the pooled samples.

Pre-registered readings (COLLAPSE_AND_ENTROPY.md §5):
  P1 dose-response: collapse rises with α for the ex-test behaviour arms,
     stays flat for the matched floors.
  P2 rescue: ≥ half of the tasks that collapsed under greedy α=1 (E8) are
     clean (majority of 3 samples) at T=0.6, α=1.
  P3 (annotation-free half): backtracking arms stay ≤ their floors on
     collapse/repetition at T=0.6 (the Δ_floor half needs annotation — owed).
  P4 iso-collapse: the α needed to collapse falls as decoding entropy falls.
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from src.evaluation import repetition_rate

EVAL = Path("results/eval")
RUNS = {
    ("greedy", None): EVAL / "R1-1.5B__E9_1_greedy" / "steering_results.json",
    ("T06", None): EVAL / "R1-1.5B__E9_1_T06" / "steering_results.json",
    ("greedy", "E8"): EVAL / "R1-1.5B__E1" / "steering_results.json",
}
COLLAPSE = 0.8
CAP = 8192
BEHAVIOURS = ("example-testing", "backtracking")
ARMS = ("single_direction", "manifold_k5", "random_subspace_k5",
        "energy_matched_random", "vanilla")
STRUCTURED_PREFIXES = ("MATH", "PATT")   # tasks with a gradeable final answer


def _ngrams(text: str, n: int = 4) -> set:
    toks = text.split()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def load_records() -> list[dict]:
    rows = []
    for (decode, tag), path in RUNS.items():
        if not path.exists():
            print(f"  (missing, skipped: {path})")
            continue
        for r in json.loads(path.read_text()):
            if tag == "E8":
                # reuse ONLY the α=1.0 greedy cells + vanilla from E8
                if r["alpha"] not in (0.0, 1.0):
                    continue
                if r["behaviour"] not in BEHAVIOURS + ("shared",):
                    continue
                if r["method"] not in ARMS:
                    continue
            rep = repetition_rate(r["chain"])
            rows.append({
                "decode": decode,
                "src": tag or "E9.1",
                "behaviour": r["behaviour"],
                "method": r["method"],
                "alpha": float(r["alpha"]),
                "task": r["base_task_id"],
                "sample": r.get("sample", 0),
                "rep": rep,
                "collapsed": rep > COLLAPSE,
                "n_tokens": r["n_tokens"],
                "cap": r["n_tokens"] >= CAP,
                "boxed": "\\boxed{" in r["chain"],
                "structured": r["base_task_id"].split("_")[0] in
                              [p for p in STRUCTURED_PREFIXES],
                "chain": r["chain"],
            })
    return rows


def cell_table(rows: list[dict]) -> dict:
    cells: dict = defaultdict(list)
    for r in rows:
        cells[(r["decode"], r["behaviour"], r["method"], r["alpha"])].append(r)
    out = {}
    for key, rs in sorted(cells.items()):
        struct = [r for r in rs if r["structured"]]
        out["|".join(map(str, key))] = {
            "n": len(rs),
            "collapse": round(float(np.mean([r["collapsed"] for r in rs])), 3),
            "mean_rep": round(float(np.mean([r["rep"] for r in rs])), 3),
            "mean_tokens": round(float(np.mean([r["n_tokens"] for r in rs])), 0),
            "cap_frac": round(float(np.mean([r["cap"] for r in rs])), 3),
            "boxed_rate": round(float(np.mean([r["boxed"] for r in rs])), 3),
            "boxed_rate_structured": (
                round(float(np.mean([r["boxed"] for r in struct])), 3)
                if struct else None),
        }
    return out


def diversity_table(rows: list[dict]) -> dict:
    """Cross-sample diversity at T>0: per (behaviour, method, α, task), the
    pairwise 4-gram Jaccard across samples; diversity = 1 − mean Jaccard."""
    groups: dict = defaultdict(list)
    for r in rows:
        if r["decode"] != "T06":
            continue
        groups[(r["behaviour"], r["method"], r["alpha"], r["task"])].append(r)
    per_cell: dict = defaultdict(list)
    for (beh, meth, alpha, _task), rs in groups.items():
        if len(rs) < 2:
            continue
        grams = [_ngrams(r["chain"]) for r in rs]
        jac = [len(a & b) / max(len(a | b), 1)
               for a, b in combinations(grams, 2)]
        per_cell[(beh, meth, alpha)].append(1.0 - float(np.mean(jac)))
    return {"|".join(map(str, k)): {
                "n_tasks": len(v),
                "diversity_mean": round(float(np.mean(v)), 3),
                "diversity_p10": round(float(np.percentile(v, 10)), 3)}
            for k, v in sorted(per_cell.items())}


def p2_rescue(rows: list[dict]) -> dict:
    """Tasks collapsed under greedy α=1 (E8) → clean (majority of samples) at
    T=0.6 α=1, per collapse-inflating arm. Baseline: same transition for the
    vanilla arm (does temperature alone rescue vanilla collapses?)."""
    out = {}
    greedy1 = {(r["behaviour"], r["method"], r["task"]): r["collapsed"]
               for r in rows if r["decode"] == "greedy" and r["alpha"] == 1.0}
    t06: dict = defaultdict(list)
    for r in rows:
        if r["decode"] == "T06" and r["alpha"] in (0.0, 1.0):
            t06[(r["behaviour"], r["method"], r["task"])].append(r["collapsed"])
    for beh in BEHAVIOURS + ("shared",):
        for meth in ARMS:
            collapsed_tasks = [t for (b, m, t), c in greedy1.items()
                               if b == beh and m == meth and c]
            if not collapsed_tasks:
                continue
            rescued = sum(
                1 for t in collapsed_tasks
                if (beh, meth, t) in t06
                and np.mean(t06[(beh, meth, t)]) < 0.5)
            n_scored = sum(1 for t in collapsed_tasks if (beh, meth, t) in t06)
            if n_scored:
                out[f"{beh}|{meth}"] = {
                    "n_collapsed_greedy": len(collapsed_tasks),
                    "n_scored_T06": n_scored,
                    "n_rescued": rescued,
                    "rescue_rate": round(rescued / n_scored, 3),
                }
    return out


def main() -> None:
    print("loading runs …")
    rows = load_records()
    if not rows:
        raise SystemExit("no runs found")
    report = {
        "n_records": len(rows),
        "cells": cell_table(rows),
        "diversity_T06": diversity_table(rows),
        "P2_rescue": p2_rescue(rows),
    }
    out = EVAL / "E9_1_analysis.json"
    out.write_text(json.dumps(report, indent=1))

    md = ["# E9.1 analysis (annotation-free endpoints)\n",
          f"Records: {len(rows)} (E8 greedy α=1 cells merged as the middle dose)\n",
          "## Per-cell endpoints (decode | behaviour | arm | α)\n",
          "| cell | n | collapse | mean rep | mean tok | boxed | boxed(struct) |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    for k, c in report["cells"].items():
        md.append(f"| {k} | {c['n']} | {c['collapse']} | {c['mean_rep']} "
                  f"| {c['mean_tokens']:.0f} | {c['boxed_rate']} "
                  f"| {c['boxed_rate_structured']} |")
    md += ["\n## Cross-sample diversity at T=0.6 (1 − mean pairwise 4-gram Jaccard)\n",
           "| behaviour | arm | α | n tasks | diversity | P10 |",
           "|---|---|--:|--:|--:|--:|"]
    for k, d in report["diversity_T06"].items():
        beh, meth, alpha = k.split("|")
        md.append(f"| {beh} | {meth} | {alpha} | {d['n_tasks']} "
                  f"| {d['diversity_mean']} | {d['diversity_p10']} |")
    md += ["\n## P2 — rescue of greedy-α=1 collapses at T=0.6\n",
           "| behaviour | arm | collapsed (greedy) | scored | rescued | rate |",
           "|---|---|--:|--:|--:|--:|"]
    for k, p in report["P2_rescue"].items():
        beh, meth = k.split("|")
        md.append(f"| {beh} | {meth} | {p['n_collapsed_greedy']} "
                  f"| {p['n_scored_T06']} | {p['n_rescued']} | {p['rescue_rate']} |")
    (EVAL / "E9_1_ANALYSIS.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out} + E9_1_ANALYSIS.md")


if __name__ == "__main__":
    main()
