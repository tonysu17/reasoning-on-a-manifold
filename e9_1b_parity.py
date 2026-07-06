#!/usr/bin/env python3
"""
E9.1b parity analysis — the amplify/sign test (COLLAPSE_AND_ENTROPY.md §5
E9.1b; thesis rung two-b). Pre-registered P5-P8 BEFORE any amplify cell
existed (2026-07-05).

Reads the amplify run (add-mode, greedy) plus the matched SUBTRACT cells
(E9.1 greedy α∈{0.5}, E8 α=1.0) and reports, per (behaviour, arm, α):
collapse under subtract / vanilla / amplify — the parity read. H-D
(deliberation-cycle) predicts an ODD response per behaviour; generic-damage
accounts predict EVEN. Paired evidence: per-task collapse transitions vs the
SAME run's vanilla (McNemar counts) + two-sided sign test.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from src.evaluation import repetition_rate

EVAL = Path("results/eval")
RUNS = {
    "amplify": EVAL / "R1-1.5B__E9_1b_amp" / "steering_results.json",
    "subtract_new": EVAL / "R1-1.5B__E9_1_greedy" / "steering_results.json",
    "subtract_e8": EVAL / "R1-1.5B__E1" / "steering_results.json",
}
COLLAPSE = 0.8
ARMS = ("single_direction", "manifold_k5", "random_subspace_k5",
        "energy_matched_random")


def load() -> dict:
    """rows[(mode, behaviour, method, alpha)][task] = mean collapse (0/1)."""
    rows: dict = defaultdict(lambda: defaultdict(list))
    van: dict = defaultdict(lambda: defaultdict(list))   # per mode-source
    for tag, path in RUNS.items():
        if not path.exists():
            continue
        mode = "amplify" if tag == "amplify" else "subtract"
        for r in json.loads(path.read_text()):
            if tag == "subtract_e8" and r["alpha"] not in (0.0, 1.0):
                continue
            if r["method"] not in ARMS + ("vanilla",):
                continue
            c = repetition_rate(r["chain"]) > COLLAPSE
            if r["method"] == "vanilla":
                van[tag][r["base_task_id"]].append(c)
            else:
                rows[(mode, r["behaviour"], r["method"], float(r["alpha"]))][
                    r["base_task_id"]].append(c)
    return rows, van


def main() -> None:
    rows, van = load()
    # amplify run has its own vanilla; subtract cells pair with their run's
    van_amp = {t: float(np.mean(v)) for t, v in van["amplify"].items()}
    van_sub = {t: float(np.mean(v)) for t, v in
               {**van.get("subtract_e8", {}), **van.get("subtract_new", {})}.items()}

    lines = ["# E9.1b parity report (amplify vs subtract, greedy)\n",
             "Vanilla collapse: amplify-run "
             f"{np.mean(list(van_amp.values())):.2f} / subtract-runs "
             f"{np.mean(list(van_sub.values())):.2f}\n",
             "| behaviour | arm | α | subtract | amplify | parity | amp arm-only/van-only (p) |",
             "|---|---|--:|--:|--:|---|---|"]
    verdicts = {}
    for (mode, beh, meth, alpha) in sorted(rows):
        if mode != "amplify":
            continue
        amp_cell = rows[("amplify", beh, meth, alpha)]
        amp = float(np.mean([np.mean(v) for v in amp_cell.values()]))
        sub_cell = rows.get(("subtract", beh, meth, alpha), {})
        sub = (float(np.mean([np.mean(v) for v in sub_cell.values()]))
               if sub_cell else float("nan"))
        vbar = np.mean(list(van_amp.values()))
        # paired transitions vs the amplify run's own vanilla
        arm_only = sum(1 for t, v in amp_cell.items()
                       if np.mean(v) > 0.5 and van_amp.get(t, 1) <= 0.5)
        van_only = sum(1 for t, v in amp_cell.items()
                       if np.mean(v) <= 0.5 and van_amp.get(t, 0) > 0.5)
        p = binomtest(arm_only, arm_only + van_only).pvalue \
            if (arm_only + van_only) else 1.0
        parity = ("ODD" if (not np.isnan(sub)) and
                  (amp - vbar) * (sub - vbar) < 0 else
                  "even/flat" if not np.isnan(sub) else "—")
        lines.append(f"| {beh} | {meth} | {alpha} | "
                     f"{'' if np.isnan(sub) else f'{sub:.2f}'} | {amp:.2f} "
                     f"| {parity} | {arm_only}/{van_only} (p={p:.3f}) |")
        verdicts[f"{beh}|{meth}|{alpha}"] = {
            "subtract": None if np.isnan(sub) else round(sub, 3),
            "amplify": round(amp, 3), "parity": parity,
            "arm_only": arm_only, "van_only": van_only, "p_sign": round(p, 4)}

    out_md = EVAL / "E9_1B_PARITY.md"
    out_md.write_text("\n".join(lines) + "\n")
    (EVAL / "E9_1b_parity.json").write_text(json.dumps(verdicts, indent=1))
    print("\n".join(lines))
    print(f"\nwrote {out_md}")


if __name__ == "__main__":
    main()
