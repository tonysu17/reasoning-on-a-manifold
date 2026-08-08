#!/usr/bin/env python3
"""pt13b — re-derive the contraction injection-recovery curve on the R1-compression
instrument (windowed PR @ L17, E9.0 grid W=128/S=64), fixing the pt13 instrument mismatch.

Pre-registered in results/prereg/PHASE0_TRANSPORT_FREEZE_2026-08-02.md §1.1 (written first).
Substrate: the locally stored 192-token state excerpts (full sequences are not on local disk;
pod stage s3 validates the excerpt-derived curve on full sequences).

Gates (must all pass before any mapping is reported):
  A  — aggregation identity: reproduce report.json paired pr medians (deepscaler, star1) to 1e-9
  B1 — grid identity: len(pr_17) == floor((T-128)/64)+1 on >=95% of the 200-task set
  B2 — excerpt sanity: median excerpt PR within +/-20% of median stored full-sequence PR
  C  — null identity: c=1.0 SVD round-trip |delta| < 1e-6

Mapping set (pre-committed): deepscaler and star1 only (qwenmath grid differs — excluded).
All mapped numbers are calibrated bounds under the isotropic tail-shrink family;
family membership is unverified until pod s3. CPU-only, deterministic, ~1-2 min.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from src.loop_geometry import windowed_state_metrics  # noqa: E402

ROOT = Path(__file__).parent
R0_OUT = ROOT / "results/r0_entropy_ladder/R1-1.5B"
STATE_SHARDS = ROOT / "results/loop_geometry/R1-1.5B/shards"
RC_OUT = ROOT / "results/r1_compression"
OUT_JSON = ROOT / "results/safety_posttrain/pt13b_l17_curve.json"
OUT_MD = ROOT / "results/safety_posttrain/PT13B_L17_CURVE.md"

WINDOW, STRIDE = 128, 64                      # E9.0 grid — asserted, not assumed
C_GRID = [1.0, 0.9, 0.75, 0.5, 0.25, 0.0]
K_GRID = [5, 2, 10]                           # k=5 primary (mirrors pt13), {2,10} sensitivity
LAYERS = [17, 16]                             # 17 primary, 16 sensitivity
MAPPED_ARMS = ("deepscaler", "star1")         # qwenmath excluded (grid differs)


def _r1_pr(tid: str, layer: int) -> float | None:
    """Mirror 30_r1_compression._rep_summaries for the r1 arm."""
    ep = R0_OUT / "ent_shards" / f"{tid}.npz"
    sp = STATE_SHARDS / f"{tid}.npz"
    if not ep.exists() or not sp.exists():
        return None
    with np.load(sp) as z:
        return float(np.nanmean(z[f"pr_{layer}"]))


def gate_a(report: dict) -> dict:
    out = {}
    for arm in MAPPED_ARMS:
        ref, s = {}, {}
        sample = json.loads((R0_OUT / "sample.json").read_text())
        for tid in sample["loop"] + sample["clean"]:
            v = _r1_pr(tid, 17)
            if v is not None:
                ref[tid] = v
        for p in sorted((RC_OUT / arm / "ent_shards").glob("*.npz")):
            with np.load(p) as z:
                s[p.stem] = float(np.nanmean(z["pr"]))
        common = sorted(set(s) & set(ref))
        d = [s[t] - ref[t] for t in common
             if np.isfinite(s[t]) and np.isfinite(ref[t])]
        med = float(np.median(d))
        want = report["rep_paired_vs_r1"][arm]["pr"]["median_delta"]
        out[arm] = {"recomputed_median": med, "report_median": want,
                    "n": len(d), "pass": bool(abs(med - want) < 1e-9)}
    return out


def main() -> None:
    report = json.loads((RC_OUT / "report.json").read_text())
    assert report["layer"] == 17

    gates: dict = {"A": gate_a(report)}
    if not all(v["pass"] for v in gates["A"].values()):
        _finish(gates, None, None, report, failed="Gate A")
        return

    sample = json.loads((R0_OUT / "sample.json").read_text())
    tids = [t for t in sample["loop"] + sample["clean"]
            if (STATE_SHARDS / f"{t}.npz").exists()
            and (R0_OUT / "ent_shards" / f"{t}.npz").exists()]

    # Gate B1: grid identity on the analysis set
    grid_ok = 0
    for tid in tids:
        with np.load(STATE_SHARDS / f"{tid}.npz") as z:
            T, npr = int(z["n_gen_tokens"]), len(z["pr_17"])
        if npr == (T - WINDOW) // STRIDE + 1:
            grid_ok += 1
    gates["B1"] = {"n": len(tids), "grid_ok": grid_ok,
                   "pass": bool(grid_ok >= 0.95 * len(tids))}

    # Injection on excerpts
    curves: dict = {}
    excerpt_med, stored_med = [], []
    gate_c_worst = 0.0
    for layer in LAYERS:
        acc: dict = {k: {c: {"dpr": [], "var_removed": []} for c in C_GRID} for k in K_GRID}
        for tid in tids:
            with np.load(STATE_SHARDS / f"{tid}.npz") as z:
                X = z[f"Xout_{layer}"].astype(np.float64)
                stored = float(np.nanmean(z[f"pr_{layer}"]))
            if X.shape[0] < WINDOW:
                continue
            mu = X.mean(axis=0, keepdims=True)
            U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
            base = float(np.nanmean(
                windowed_state_metrics(X, window=WINDOW, stride=STRIDE)["pr"]))
            if layer == 17:
                excerpt_med.append(base)
                stored_med.append(stored)
            e_tot = float((s ** 2).sum())
            for k in K_GRID:
                for c in C_GRID:
                    s2 = s.copy()
                    s2[k:] *= c
                    Xr = (U * s2) @ Vt + mu
                    pr = float(np.nanmean(
                        windowed_state_metrics(Xr, window=WINDOW, stride=STRIDE)["pr"]))
                    acc[k][c]["dpr"].append(pr - base)
                    acc[k][c]["var_removed"].append(1.0 - float((s2 ** 2).sum()) / e_tot)
                    if c == 1.0:
                        gate_c_worst = max(gate_c_worst, abs(pr - base))
        curves[layer] = {
            k: {"c": C_GRID,
                "median_dpr": [float(np.median(acc[k][c]["dpr"])) for c in C_GRID],
                "mean_var_removed": [float(np.mean(acc[k][c]["var_removed"])) for c in C_GRID],
                "n_tasks": len(acc[k][C_GRID[0]]["dpr"])}
            for k in K_GRID}

    b2_ratio = float(np.median(excerpt_med) / np.median(stored_med))
    gates["B2"] = {"median_excerpt_pr": float(np.median(excerpt_med)),
                   "median_stored_pr": float(np.median(stored_med)),
                   "ratio": b2_ratio,
                   "pass": bool(0.8 <= b2_ratio <= 1.2),
                   "root_cause": ("Xout_* is a CLASS-SELECTED NON-CONTIGUOUS token subsample "
                                  "(18_loop_geometry.py select_class_token_indices, "
                                  "tokens-per-class=192, out-of-loop tokens only), not a "
                                  "sequence excerpt; the windowed instrument is undefined on "
                                  "it. No local full-sequence source exists (main + "
                                  "_pod_archive checked). Mapping deferred to pod s3.")}
    gates["C"] = {"worst_null_delta": gate_c_worst, "pass": bool(gate_c_worst < 1e-6)}

    # Mapping (only if all gates pass)
    mapping = None
    failing = [k for k, g in gates.items()
               if not (g["pass"] if "pass" in g else all(v["pass"] for v in g.values()))]
    if failing:
        _finish(gates, curves, None, report,
                failed=f"Gate(s) {'+'.join(failing)} — substrate inadequate locally; "
                       "mapping deferred to pod s3 (freeze §1.1 labelling rule)")
        return
    if True:
        mapping = {}
        cur = curves[17][5]
        # monotone: dpr 0 -> negative as c drops; interpolate observed on (dpr -> var_removed)
        xs = np.array(cur["median_dpr"])[::-1]          # ascendingly negative -> 0
        ys = np.array(cur["mean_var_removed"])[::-1]
        for arm in MAPPED_ARMS:
            obs = report["rep_paired_vs_r1"][arm]["pr"]["median_delta"]
            in_range = bool(xs.min() <= obs <= xs.max())
            mapping[arm] = {
                "observed_median_dpr": obs,
                "var_removed_interp": float(np.interp(obs, xs, ys)) if in_range else None,
                "in_curve_range": in_range,
                "label": ("calibrated bound under isotropic tail-shrink family; "
                          "family membership unverified (pod s3)"),
            }
    _finish(gates, curves, mapping, report)


def _finish(gates, curves, mapping, report, failed: str | None = None) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True, cwd=ROOT).stdout.strip())
    out = {"script": "pt13b_l17_curve.py", "date": "2026-08-02",
           "prereg": "results/prereg/PHASE0_TRANSPORT_FREEZE_2026-08-02.md §1.1",
           "instrument": f"windowed PR @L17, W={WINDOW}/S={STRIDE}, excerpt substrate",
           "git_commit": commit, "git_dirty": dirty,
           "gates": gates, "curves": curves, "mapping": mapping,
           "failed": failed}
    OUT_JSON.write_text(json.dumps(out, indent=1))

    lines = ["# PT13b — contraction curve on the R1-compression instrument (L17 windowed PR)",
             "", f"Date 2026-08-02 · prereg §1.1 of PHASE0_TRANSPORT_FREEZE · commit {commit[:8]}"
             f"{' (dirty)' if dirty else ''}", ""]
    if failed:
        lines += [f"**FAILED: {failed} — no mapping reported.**", ""]
        if gates.get("B2"):
            lines += [f"- B2 ratio {gates['B2'].get('ratio', float('nan')):.3f}; "
                      f"root cause: {gates['B2'].get('root_cause', 'n/a')}", ""]
        if curves:
            lines += ["Substrate-invalid curve retained below for the audit record only — "
                      "NOT valid for mapping.", ""]
            cur = curves[17][5]
            lines += ["| c | median dPR (invalid substrate) | mean var removed |", "|---|---|---|"]
            for c, d, v in zip(cur["c"], cur["median_dpr"], cur["mean_var_removed"]):
                lines += [f"| {c} | {d:+.4f} | {v:.4f} |"]
    else:
        ga = gates["A"]
        lines += ["## Gates",
                  f"- A aggregation identity: deepscaler {'PASS' if ga['deepscaler']['pass'] else 'FAIL'},"
                  f" star1 {'PASS' if ga['star1']['pass'] else 'FAIL'} (to 1e-9)",
                  f"- B1 grid identity: {gates['B1']['grid_ok']}/{gates['B1']['n']}"
                  f" ({'PASS' if gates['B1']['pass'] else 'FAIL'})",
                  f"- B2 excerpt sanity: ratio {gates['B2']['ratio']:.3f}"
                  f" ({'PASS' if gates['B2']['pass'] else 'FAIL'})",
                  f"- C null identity: worst |delta| {gates['C']['worst_null_delta']:.2e}"
                  f" ({'PASS' if gates['C']['pass'] else 'FAIL'})", ""]
        if curves:
            cur = curves[17][5]
            lines += ["## Curve (L17, k=5 primary)", "",
                      "| c | median dPR | mean var removed |", "|---|---|---|"]
            for c, d, v in zip(cur["c"], cur["median_dpr"], cur["mean_var_removed"]):
                lines += [f"| {c} | {d:+.4f} | {v:.4f} |"]
            lines += [""]
        if mapping:
            lines += ["## Mapping (pre-committed arms; excerpt-substrate caveat applies)", ""]
            for arm, m in mapping.items():
                vr = m["var_removed_interp"]
                lines += [f"- **{arm}**: observed median dPR {m['observed_median_dpr']:+.4f} → "
                          + (f"≈ **{100*vr:.2f}% variance removed**" if vr is not None
                             else "OUT OF CURVE RANGE (no interpolation)")
                          + f" — {m['label']}"]
            lines += ["", "Excerpt substrate: curve derived on stored 192-token onset-locked "
                      "excerpts (2 windows/task); full-sequence validation = pod batch s3."]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"gates": {k: (v if "pass" not in v else v["pass"])
                                if not isinstance(v, dict) or "pass" not in v
                                else v["pass"] for k, v in gates.items()},
                      "mapping": mapping, "failed": failed}, indent=1, default=str))


if __name__ == "__main__":
    main()
