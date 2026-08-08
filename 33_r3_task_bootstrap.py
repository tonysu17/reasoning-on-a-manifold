#!/usr/bin/env python3
"""33 — R3 P-R2.1 task-level bootstrap re-analysis (RESULTS_LEDGER §F item 6).

Why this exists
---------------
`32_r3_strategy.py::_matched_entropy_gap` bootstraps the 25 *interpolated*
grid points, which `np.interp` derives deterministically from 6 cell means:
that CI measures the interpolation curve's shape, not sampling variability,
and `n_grid` narrows it for free. The 2026-07-19 ledger downgrade bars citing
it as significance. This script replaces the resampling unit with the one that
is actually exchangeable — the 64 tasks — and propagates the resampling
through the entire statistic: per-cell (strategy-H, accuracy) recomputation →
frontier sort → overlap window → grid interpolation → mean gap.

Design points
-------------
* Input is `results/r3_strategy/full_report.json` `per_task` only (no
  regeneration, no re-annotation; the frozen lexical classifier's labels are
  untouched — the CF-T judged-label caveat stands regardless of this result).
* Tasks are resampled ONCE per replicate and applied to every cell (paired
  cluster bootstrap): cells share the same 64 tasks, and the pump/thermostat
  frontiers share the vanilla_T0.6 anchor, so unpaired resampling would
  destroy the very matching the statistic depends on.
* The overlap window is recomputed inside every replicate. Replicates whose
  frontiers stop overlapping contribute no gap and are counted separately —
  silently dropping them would bias toward replicates that resemble the
  observed configuration.
* Self-gate: the observed statistic recomputed here must match the frozen
  pipeline's `P_R2_1_reframed` mean gap to 1e-9 before any bootstrap runs.
  If the reimplementation cannot reproduce the point estimate, its CI is
  meaningless and the script aborts.

Verdict rule (sealed before running, same bar shape as the prereg's):
  CI95 excludes 0 from below  → gap survives task-level resampling
  CI95 straddles 0            → directional only (thesis wording stays)
  CI95 excludes 0 from above  → thermostat wins; escalate, do not paper over
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "results" / "r3_strategy" / "full_report.json"
OUT_JSON = ROOT / "results" / "r3_strategy" / "task_bootstrap.json"
OUT_MD = ROOT / "results" / "r3_strategy" / "TASK_BOOTSTRAP.md"

PUMP_CELLS = {  # knob value → cell name; α=0 IS the shared anchor
    0.0: "vanilla_T0.6",
    0.5: "pump_subtract_a0.5",
    1.0: "pump_subtract_a1",
    1.5: "pump_subtract_a1.5",
}
THERMO_CELLS = {
    0.3: "vanilla_T0.3",
    0.6: "vanilla_T0.6",
    0.9: "vanilla_T0.9",
    1.2: "vanilla_T1.2",
}
N_GRID = 25          # integration rule only; CI no longer depends on it
B = 2000
SEED = 20260719


def _cell_point(per_task_cell: dict, tids: list[str]) -> tuple[float, float] | None:
    """(strategy-H, accuracy) for one cell under a task multiset, mirroring
    `_full_cell_metrics`: cell H = mean of per-task H over tasks where it is
    defined; cell accuracy = mean over rows (tasks have equal n=8, so the
    task-mean equals the row-mean)."""
    hs = [per_task_cell[t]["strategy_entropy"] for t in tids
          if per_task_cell[t]["strategy_entropy"] is not None]
    accs = [per_task_cell[t]["accuracy"] for t in tids]
    if not hs:
        return None
    return float(np.mean(hs)), float(np.mean(accs))


def _gap(per_task: dict, tids: list[str]) -> float | None:
    """Mean pump−thermo accuracy gap on the shared-entropy grid, or None if
    the frontiers do not overlap under this task multiset."""
    def frontier(cells: dict) -> list[tuple[float, float]] | None:
        pts = []
        for cell in cells.values():
            p = _cell_point(per_task[cell], tids)
            if p is None:
                return None
            pts.append(p)
        return sorted(pts)

    pump, thermo = frontier(PUMP_CELLS), frontier(THERMO_CELLS)
    if pump is None or thermo is None:
        return None
    lo = max(pump[0][0], thermo[0][0])
    hi = min(pump[-1][0], thermo[-1][0])
    if not hi > lo:
        return None
    grid = np.linspace(lo, hi, N_GRID)

    def interp(pts):
        return np.interp(grid, [p[0] for p in pts], [p[1] for p in pts])

    return float(np.mean(interp(pump) - interp(thermo)))


def main() -> None:
    report = json.loads(REPORT.read_text())
    per_task = report["per_task"]
    tids = sorted(per_task["vanilla_T0.6"].keys())
    assert len(tids) == 64, f"expected 64 tasks, got {len(tids)}"
    for cell in set(PUMP_CELLS.values()) | set(THERMO_CELLS.values()):
        assert sorted(per_task[cell].keys()) == tids, f"task mismatch in {cell}"

    # Self-gate: reproduce the frozen pipeline's point estimate exactly.
    observed = _gap(per_task, tids)
    frozen = report["P_R2_1_reframed"]["mean_gap_pump_minus_thermo"]
    if observed is None or abs(observed - frozen) > 1e-9:
        raise SystemExit(
            f"SELF-GATE FAILED: recomputed gap {observed} != frozen {frozen}; "
            "reimplementation does not reproduce the pipeline — CI would be "
            "meaningless. Aborting before any bootstrap."
        )

    rng = np.random.default_rng(SEED)
    gaps, no_overlap = [], 0
    for _ in range(B):
        sample = [tids[i] for i in rng.integers(0, len(tids), len(tids))]
        g = _gap(per_task, sample)
        if g is None:
            no_overlap += 1
        else:
            gaps.append(g)
    gaps = np.array(gaps)
    lo_ci, hi_ci = np.percentile(gaps, [2.5, 97.5])
    p_le_0 = float(np.mean(gaps <= 0))

    verdict = (
        "gap SURVIVES task-level resampling (CI95 excludes 0 from below)"
        if lo_ci > 0 else
        "thermostat wins under task-level resampling (CI95 excludes 0 from above)"
        if hi_ci < 0 else
        "DIRECTIONAL ONLY — gap does not survive task-level resampling "
        "(CI95 straddles 0)"
    )

    result = {
        "prereg_note": (
            "Re-analysis of P-R2.1 (reframed) with the resampling unit moved "
            "from interpolated grid points to tasks (paired cluster bootstrap "
            "over the 64 shared tasks). Classifier labels untouched (frozen "
            "lexical range-finder; CF-T judged-label caveat unaffected). "
            "Supersedes the grid-point CI of full_report.json for citation."
        ),
        "observed_gap": observed,
        "frozen_pipeline_gap": frozen,
        "B": B,
        "seed": SEED,
        "n_grid": N_GRID,
        "n_no_overlap_replicates": no_overlap,
        "ci95": [float(lo_ci), float(hi_ci)],
        "p_gap_le_0": p_le_0,
        "bootstrap_mean": float(np.mean(gaps)),
        "bootstrap_sd": float(np.std(gaps)),
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(result, indent=1))

    OUT_MD.write_text(
        "# R3 P-R2.1 — task-level bootstrap re-analysis\n\n"
        f"Supersedes the grid-point CI in `FULL_REPORT.md` (defect: resampled "
        f"deterministic interpolants; see RESULTS_LEDGER 2026-07-19 downgrade).\n\n"
        f"- observed gap (pump−thermo, matched strategy-H): "
        f"**{observed:+.4f}** (reproduces frozen pipeline exactly)\n"
        f"- task-level bootstrap (B={B}, paired over 64 tasks, seed {SEED}): "
        f"CI95 **[{lo_ci:+.4f}, {hi_ci:+.4f}]**, P(gap≤0) = {p_le_0:.4f}\n"
        f"- replicates without frontier overlap: {no_overlap}/{B}\n\n"
        f"**{verdict}**\n\n"
        "Caveats unchanged by this re-analysis: frozen lexical classifier "
        "(CF-T, judged-label replication owed); no arm raises strategy-H "
        "above the shared anchor, so the comparison is a descent contrast "
        "against modest temperature (T0.9/T1.2 fall below the overlap "
        "window); pump α=0 ≡ vanilla_T0.6 pins both frontiers at one end.\n"
    )
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
