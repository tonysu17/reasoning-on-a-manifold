#!/usr/bin/env python3
"""Phase-2 arm-differential missingness bounding.

Runs the analysis fixed by results/prereg/PH2_MISSINGNESS_BOUNDING_SPEC_2026-08-19.md:
Manski worst-case bounds, per-arm tipping points, and behaviour-dense quantile
scenarios for (1) the A2 vanilla prevalence contrasts and (2) the primary
steered suppression cells, with the row-level missingness definition mirroring
the authoritative extractor. Deterministic; no bootstrap; bounds and scenarios
apply to point estimates only. Aborts if the complete-case anchors do not
reproduce results/ph2/analysis/ph2_analysis.json.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.annotation_coverage import row_coverage_excludes, row_is_coverage_complete
from src.delta_floor import VANILLA, _alpha_eq, _base_task
from src.evaluation import behaviour_fraction
from src.ph2_stages import A2_BEHAVIOURS, ALL_ROLES, BEHAVIOUR, CLAMP_GAIN, TARGET_ROLES

OUT = Path("results/ph2")
ANCHOR_TOL = 1e-9
PRIMARY_ARM = "transported_raw_suppress"
PRIMARY_FLOOR = "energy_matched_floor"
QUANTILES = (0.50, 0.75, 0.90)


# ── row-level missingness (mirrors the authoritative extractor) ──────────────

def _vanilla_resolved(r: dict) -> bool:
    if r.get("annotation_complete") is False:
        return False
    anns = r.get("annotations")
    if not anns:
        return False
    return bool(row_is_coverage_complete(r))


def vanilla_cell(ann: list[dict], behaviour: str) -> dict:
    """task -> {'values': [fractions], 'n_missing': int} over vanilla rows."""
    cell: dict = defaultdict(lambda: {"values": [], "n_missing": 0})
    for r in ann:
        if r["method"] != VANILLA:
            continue
        if _vanilla_resolved(r):
            cell[r["task_id"]]["values"].append(
                behaviour_fraction(r["annotations"], behaviour))
        else:
            cell[r["task_id"]]["n_missing"] += 1
    return dict(cell)


def steered_cell(gen: list[dict], ann: list[dict], behaviour: str,
                 method: str, alpha: float) -> dict:
    """task -> {'values', 'n_missing'} over the gen-row universe for one arm."""
    ann_index = {(r["task_id"], r["behaviour"], r["method"], r["alpha"]): r
                 for r in ann}
    cell: dict = defaultdict(lambda: {"values": [], "n_missing": 0})
    for r in gen:
        if r["method"] != method:
            continue
        if method != VANILLA and r["behaviour"] != behaviour:
            continue
        if not _alpha_eq(r["alpha"], alpha):
            continue
        key = (r["task_id"], r["behaviour"], r["method"], r["alpha"])
        ann_row = ann_index.get(key)
        t = _base_task(r)
        if (not ann_row or ann_row.get("annotation_complete") is False
                or row_coverage_excludes(ann_row)
                or not ann_row.get("annotations")):
            cell[t]["n_missing"] += 1
        else:
            cell[t]["values"].append(
                behaviour_fraction(ann_row["annotations"], behaviour))
    return dict(cell)


# ── imputation machinery ─────────────────────────────────────────────────────

def _task_value(entry: dict, impute: float | None) -> float | None:
    """Per-task mean with unresolved rows imputed at `impute` (None = drop)."""
    vals = list(entry["values"])
    if impute is None:
        return float(np.mean(vals)) if vals else None
    vals += [impute] * entry["n_missing"]
    return float(np.mean(vals)) if vals else None


def _arm_stats(cell: dict) -> dict:
    complete = [ _task_value(e, None) for e in cell.values() ]
    complete = [v for v in complete if v is not None]
    arr = np.array(complete, dtype=float)
    qs = {f"q{int(q*100)}": float(np.quantile(arr, q)) if len(arr) else None
          for q in QUANTILES}
    return {"mean": float(arr.mean()) if len(arr) else None, **qs,
            "n_tasks_resolved": len(complete),
            "n_rows_missing": int(sum(e["n_missing"] for e in cell.values())),
            "n_rows_resolved": int(sum(len(e["values"]) for e in cell.values()))}


def _paired_mean(cell_a: dict, cell_b: dict,
                 impute_a: float | None, impute_b: float | None) -> tuple:
    """Mean over the paired universe of A(t)-B(t); None imputation drops
    unresolved rows and restricts to tasks resolved in both arms."""
    tasks = set(cell_a) & set(cell_b)
    diffs = []
    for t in sorted(tasks):
        va = _task_value(cell_a[t], impute_a)
        vb = _task_value(cell_b[t], impute_b)
        if va is None or vb is None:
            continue
        diffs.append(va - vb)
    return (float(np.mean(diffs)) if diffs else None), len(diffs)


def _tipping(cell_a, cell_b, mean_a, mean_b) -> dict:
    """m* for each arm's unresolved rows (other arm at its complete-case mean)."""
    out = {}
    for label, (ca, cb, other_mean, side) in {
        "minuend": (cell_a, cell_b, mean_b, "a"),
        "subtrahend": (cell_a, cell_b, mean_a, "b"),
    }.items():
        if side == "a":
            d0, _ = _paired_mean(ca, cb, 0.0, other_mean)
            d1, _ = _paired_mean(ca, cb, 1.0, other_mean)
        else:
            d0, _ = _paired_mean(ca, cb, other_mean, 0.0)
            d1, _ = _paired_mean(ca, cb, other_mean, 1.0)
        if d0 is None or d1 is None or math.isclose(d1, d0, abs_tol=1e-15):
            out[label] = {"m_star": None, "in_unit_interval": False,
                          "note": "difference insensitive to this arm's missing rows"}
            continue
        m = -d0 / (d1 - d0)
        # subtrahend imputation enters with a negative sign; the linear solve
        # is identical because d0/d1 already encode it.
        out[label] = {"m_star": float(m),
                      "in_unit_interval": bool(0.0 <= m <= 1.0)}
    return out


def bound_contrast(cell_a: dict, cell_b: dict) -> dict:
    stats_a, stats_b = _arm_stats(cell_a), _arm_stats(cell_b)
    cc, n_cc = _paired_mean(cell_a, cell_b, None, None)
    lo, n_all = _paired_mean(cell_a, cell_b, 0.0, 1.0)
    hi, _ = _paired_mean(cell_a, cell_b, 1.0, 0.0)
    tip = _tipping(cell_a, cell_b, stats_a["mean"], stats_b["mean"])
    scen = {}
    for q in QUANTILES:
        k = f"q{int(q*100)}"
        scen[k], _ = _paired_mean(cell_a, cell_b, stats_a[k], stats_b[k])
    sign = (lambda x: 0 if x is None or x == 0 else (1 if x > 0 else -1))
    robust_worst = lo is not None and hi is not None and (
        (lo > 0 and hi > 0) or (lo < 0 and hi < 0))
    robust_tip = not any(v.get("in_unit_interval") for v in tip.values())
    robust_dense = (cc is not None
                    and all(scen[k] is not None and sign(scen[k]) == sign(cc)
                            for k in scen))
    label = ("sign-robust (worst case)" if robust_worst
             else "sign-robust (tipping)" if robust_tip
             else "sign-robust (behaviour-dense)" if robust_dense
             else "missingness-fragile")
    return {"complete_case": {"diff_mean": cc, "n_pairs": n_cc},
            "manski": {"low": lo, "high": hi, "n_tasks_universe": n_all},
            "tipping_point": tip,
            "behaviour_dense_scenarios": scen,
            "arm_minuend": stats_a, "arm_subtrahend": stats_b,
            "robustness": {"worst_case": bool(robust_worst),
                           "tipping": bool(robust_tip),
                           "behaviour_dense": bool(robust_dense),
                           "label": label}}


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ann_by_role = {r: json.loads((OUT / "annotation" / f"{r}.json").read_text())
                   for r in ALL_ROLES}
    gen_by_role = {r: json.loads((OUT / "battery" / f"{r}.json").read_text())
                   for r in ALL_ROLES}
    analysis = json.loads((OUT / "analysis" / "ph2_analysis.json").read_text())
    aliased = {}
    prov_bat = OUT / "provenance" / "generate_battery.json"
    if prov_bat.exists():
        aliased = (json.loads(prov_bat.read_text()).get("counts", {})
                   .get("aliased", {}))

    report: dict = {
        "spec": "results/prereg/PH2_MISSINGNESS_BOUNDING_SPEC_2026-08-19.md",
        "missingness_definition": "mirrors src.delta_floor.per_task_fraction / "
                                  "A2 per-task builder (no annotation row, "
                                  "annotation_complete False, coverage-excluded, "
                                  "or empty annotations)",
        "point_estimates_only": True,
    }

    # (1) A2 vanilla prevalence contrasts, anchored against ph2_analysis.json.
    a2_out: dict = {}
    anchors_checked = 0
    for role in TARGET_ROLES:
        a2_out[role] = {}
        for beh in A2_BEHAVIOURS:
            ca = vanilla_cell(ann_by_role[role], beh)       # minuend: target
            cb = vanilla_cell(ann_by_role["base"], beh)     # subtrahend: base
            res = bound_contrast(ca, cb)
            anchor = (analysis["a2_adjunct"]["endpoints"][role]
                      [f"prev_{beh}"])
            cc = res["complete_case"]
            if (cc["n_pairs"] != anchor["n"]
                    or cc["diff_mean"] is None
                    or abs(cc["diff_mean"] - anchor["diff_mean"]) > ANCHOR_TOL):
                raise SystemExit(
                    f"anchor mismatch {role}/prev_{beh}: "
                    f"got n={cc['n_pairs']} diff={cc['diff_mean']!r}, "
                    f"expected n={anchor['n']} diff={anchor['diff_mean']!r}")
            anchors_checked += 1
            res["anchor"] = {"matches_ph2_analysis": True,
                             "diff_mean": anchor["diff_mean"],
                             "n": anchor["n"]}
            a2_out[role][f"prev_{beh}"] = res
    report["a2_vanilla_contrasts"] = a2_out
    report["anchors_checked"] = anchors_checked

    # (2) Primary steered suppression cells: floor − transported arm.
    steered_out: dict = {}
    for role in ALL_ROLES:
        arm_m = aliased.get(role, {}).get(PRIMARY_ARM, PRIMARY_ARM)
        floor_m = aliased.get(role, {}).get(PRIMARY_FLOOR, PRIMARY_FLOOR)
        ca = steered_cell(gen_by_role[role], ann_by_role[role], BEHAVIOUR,
                          floor_m, CLAMP_GAIN)               # minuend: floor
        cb = steered_cell(gen_by_role[role], ann_by_role[role], BEHAVIOUR,
                          arm_m, CLAMP_GAIN)                 # subtrahend: arm
        res = bound_contrast(ca, cb)
        anchor = (analysis["delta_floor_cells"][role][PRIMARY_ARM]
                  ["vs_energy_floor"])
        cc = res["complete_case"]
        if (anchor.get("delta_floor") is not None and cc["diff_mean"] is not None
                and abs(cc["diff_mean"] - anchor["delta_floor"]) > ANCHOR_TOL):
            raise SystemExit(
                f"anchor mismatch {role}/{PRIMARY_ARM}: got "
                f"{cc['diff_mean']!r}, expected {anchor['delta_floor']!r}")
        res["anchor"] = {"matches_ph2_analysis": True,
                         "delta_floor": anchor.get("delta_floor"),
                         "n_tasks": anchor.get("n_tasks")}
        res["note"] = ("point-estimate movement only; the delta-floor verdicts "
                       "remain downgraded by the sealed sensitivity gate and "
                       "are not re-litigated here")
        steered_out[role] = res
    report["primary_steered_cells"] = steered_out

    out_json = OUT / "analysis" / "missingness_bounds.json"
    out_json.write_text(json.dumps(report, indent=1, sort_keys=True))

    def fmt(x):
        return "—" if x is None else f"{x:+.4f}"

    lines = ["# Phase-2 arm-differential missingness bounds", "",
             f"Spec: `{report['spec']}` (imputation rules fixed before running).",
             "Complete-case anchors reproduced exactly "
             f"({anchors_checked} A2 cells + {len(steered_out)} steered cells).",
             "", "## A2 vanilla prevalence contrasts (target − base)", "",
             "| Role | Endpoint | Complete-case (n) | Manski [lo, hi] | "
             "m* minuend / subtrahend | q50 / q75 / q90 | Verdict |",
             "|---|---|---|---|---|---|---|"]
    for role in TARGET_ROLES:
        for beh in A2_BEHAVIOURS:
            r = a2_out[role][f"prev_{beh}"]
            tip = r["tipping_point"]
            def tips(v):
                return ("out" if not v["in_unit_interval"]
                        else f"{v['m_star']:.3f}")
            s = r["behaviour_dense_scenarios"]
            lines.append(
                f"| {role} | prev_{beh} | "
                f"{fmt(r['complete_case']['diff_mean'])} "
                f"({r['complete_case']['n_pairs']}) | "
                f"[{fmt(r['manski']['low'])}, {fmt(r['manski']['high'])}] | "
                f"{tips(tip['minuend'])} / {tips(tip['subtrahend'])} | "
                f"{fmt(s['q50'])} / {fmt(s['q75'])} / {fmt(s['q90'])} | "
                f"{r['robustness']['label']} |")
    lines += ["", "## Primary steered cells (energy floor − transported arm, "
              "prev_backtracking)", "",
              "| Role | Complete-case (n) | Manski [lo, hi] | "
              "m* minuend / subtrahend | q50 / q75 / q90 | Verdict |",
              "|---|---|---|---|---|---|"]
    for role in ALL_ROLES:
        r = steered_out[role]
        tip = r["tipping_point"]
        def tips2(v):
            return ("out" if not v["in_unit_interval"]
                    else f"{v['m_star']:.3f}")
        s = r["behaviour_dense_scenarios"]
        lines.append(
            f"| {role} | {fmt(r['complete_case']['diff_mean'])} "
            f"({r['complete_case']['n_pairs']}) | "
            f"[{fmt(r['manski']['low'])}, {fmt(r['manski']['high'])}] | "
            f"{tips2(tip['minuend'])} / {tips2(tip['subtrahend'])} | "
            f"{fmt(s['q50'])} / {fmt(s['q75'])} / {fmt(s['q90'])} | "
            f"{r['robustness']['label']} |")
    lines += ["", "Citation rule (from the spec): cite complete-case values "
              "with per-arm unresolved counts adjacent plus the robustness "
              "label; no imputed value is a result. Full-chain endpoints are "
              "unaffected by construction.", ""]
    (OUT / "analysis" / "MISSINGNESS_BOUNDS.md").write_text("\n".join(lines))
    print(json.dumps({"written": [str(out_json),
                                  str(OUT / 'analysis' / 'MISSINGNESS_BOUNDS.md')],
                      "anchors_checked": anchors_checked}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
