#!/usr/bin/env python3
"""
E9 secondary analysis: per-cell collapse table from the executed E8 run
(COLLAPSE_AND_ENTROPY.md §2; thesis tab:steer-collapse).

Reads results/eval/R1-1.5B__E1/steering_results.json and reports, per
(behaviour, arm): mean per-chain 4-gram repetition, collapsed fraction
(repetition > 0.8 — the loop-to-cap mode; the distribution is bimodal so the
threshold is uncritical), cap-hit fraction, paired Δrepetition vs the shared
vanilla baseline, and the collapse transitions vs vanilla (arm-only /
vanilla-only). Writes collapse_table.json next to the input.

Post-hoc / not pre-registered: this is the reproducible source for the numbers
quoted in the thesis steering chapter's collapse section.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.evaluation import repetition_rate

EVAL_DIR = Path("results/eval/R1-1.5B__E1")
COLLAPSE_THRESHOLD = 0.8
CAP_TOKENS = 8192


def main() -> None:
    records = json.loads((EVAL_DIR / "steering_results.json").read_text())

    per_cell: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        rep = repetition_rate(r["chain"])
        key = (r["behaviour"], r["method"])
        cell = per_cell[key]
        cell["rep"].append(rep)
        cell["cap"].append(r["n_tokens"] >= CAP_TOKENS)
        cell["task"].append(r["base_task_id"])

    # vanilla is shared across behaviours (behaviour == "shared")
    van_key = next(k for k in per_cell if k[1] == "vanilla")
    van = per_cell.pop(van_key)
    van_rep = {}
    for t, rep in zip(van["task"], van["rep"]):
        van_rep.setdefault(t, []).append(rep)
    van_rep = {t: float(np.mean(v)) for t, v in van_rep.items()}
    van_collapsed = {t: v > COLLAPSE_THRESHOLD for t, v in van_rep.items()}

    table = {"vanilla": {
        "n": len(van["rep"]),
        "mean_rep": round(float(np.mean(van["rep"])), 3),
        "frac_collapsed": round(float(np.mean([r > COLLAPSE_THRESHOLD for r in van["rep"]])), 3),
        "frac_cap": round(float(np.mean(van["cap"])), 3),
    }}

    for (beh, arm), cell in sorted(per_cell.items()):
        # replicate arms (random subspaces) pool: average per task first
        task_rep: dict[str, list] = defaultdict(list)
        for t, rep in zip(cell["task"], cell["rep"]):
            task_rep[t].append(rep)
        t_rep = {t: float(np.mean(v)) for t, v in task_rep.items()}

        paired = [(t_rep[t], van_rep[t]) for t in t_rep if t in van_rep]
        d_rep = [a - v for a, v in paired]
        arm_only = sum(1 for a, v in paired
                       if a > COLLAPSE_THRESHOLD and v <= COLLAPSE_THRESHOLD)
        van_only = sum(1 for a, v in paired
                       if a <= COLLAPSE_THRESHOLD and v > COLLAPSE_THRESHOLD)

        table[f"{beh}|{arm}"] = {
            "n_chains": len(cell["rep"]),
            "n_tasks": len(t_rep),
            "mean_rep": round(float(np.mean(cell["rep"])), 3),
            "frac_collapsed": round(float(np.mean(
                [r > COLLAPSE_THRESHOLD for r in cell["rep"]])), 3),
            "frac_cap": round(float(np.mean(cell["cap"])), 3),
            "delta_rep_vs_vanilla": round(float(np.mean(d_rep)), 3),
            "collapse_arm_only": arm_only,
            "collapse_vanilla_only": van_only,
        }

    out = EVAL_DIR / "collapse_table.json"
    out.write_text(json.dumps(table, indent=1))

    hdr = f"{'cell':44s} {'meanrep':>8s} {'coll>0.8':>9s} {'cap':>6s} {'Δrep':>7s} {'A-only':>7s} {'V-only':>7s}"
    print(hdr)
    print("-" * len(hdr))
    v = table["vanilla"]
    print(f"{'vanilla (shared)':44s} {v['mean_rep']:8.3f} {v['frac_collapsed']:9.3f} "
          f"{v['frac_cap']:6.3f} {'—':>7s} {'—':>7s} {'—':>7s}")
    for k, c in table.items():
        if k == "vanilla":
            continue
        print(f"{k:44s} {c['mean_rep']:8.3f} {c['frac_collapsed']:9.3f} "
              f"{c['frac_cap']:6.3f} {c['delta_rep_vs_vanilla']:+7.3f} "
              f"{c['collapse_arm_only']:7d} {c['collapse_vanilla_only']:7d}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
