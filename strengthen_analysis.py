#!/usr/bin/env python3
"""Free, no-budget strengthening of the E8 result (steps 1 + 2), all from the
existing 1629 annotated chains.

STEP 1 — degeneration-immune endpoints. Δ_floor (= floor − arm, +ve ⇒ arm
suppresses) recomputed per cell on:
  * count  : raw #behaviour-sentences  (immune to denominator/dilution inflation)
  * per1k  : #behaviour-sentences per 1000 tokens (length-controlled rate)
  * frac@nodeg : the ORIGINAL fraction metric, but restricted to task-pairs where
                 NEITHER chain is degenerate (line-dup ratio ≤ 0.5)
Paired BCa bootstrap + Holm across the 12-cell family, per endpoint.

STEP 2 — cross-behaviour specificity (an 'active' control, free). Every chain is
labelled for all 4 target behaviours, so for each steering vector we measure how
much it suppresses ITS OWN behaviour vs the OTHER three (per-1k; the own-minus-other
contrast cancels within-chain dilution since looping hits all behaviours equally).
Reports the 4×4 steer×measure suppression matrix + an own-vs-other paired test
(absolute per-1k and base-rate-controlled relative %).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np  # noqa: E402
from src.delta_floor import (paired_bootstrap_mean, empirical_two_sided_p,  # noqa: E402
                             _base_task, floor_for_arm, _alpha_eq, VANILLA)

EVAL = Path("results/eval/R1-1.5B__E1")
TARGETS = ["backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"]
ARMS = ["single_direction", "manifold_k3", "manifold_k5"]
DUP_MAX = 0.5  # a chain with >50% duplicate lines is treated as degenerate

steered = json.loads((EVAL / "steering_results.json").read_text())
annotated = json.loads((EVAL / "annotated_steered.json").read_text())
ANN = {(r["task_id"], r["behaviour"], r["method"], r["alpha"]): r.get("annotations", [])
       for r in annotated if r.get("annotation_complete")}


def cnt(anns, b):
    return sum(1 for a in anns if a.get("label") == b)


def dup_ratio(txt):
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    return 1 - len(set(lines)) / len(lines) if len(lines) >= 6 else 0.0


def metric(anns, ntok, b, kind):
    c = cnt(anns, b)
    if kind == "count":
        return float(c)
    if kind == "per1k":
        return c / ntok * 1000 if ntok else 0.0
    return c / len(anns) if anns else 0.0   # frac


def holm(pmap):
    items = sorted(pmap.items(), key=lambda kv: kv[1])
    m, run, adj = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
    return adj


def per_task(beh, meth, kind, max_dup=None):
    acc = defaultdict(list)
    for r in steered:
        if r["method"] != meth:
            continue
        if meth != VANILLA and r["behaviour"] != beh:
            continue
        if not _alpha_eq(r["alpha"], 1.0):
            continue
        key = (r["task_id"], r["behaviour"], r["method"], r["alpha"])
        anns = ANN.get(key)
        if not anns:
            continue
        if max_dup is not None and dup_ratio(r.get("chain", "")) > max_dup:
            continue
        acc[_base_task(r)].append(metric(anns, r.get("n_tokens", 0), beh, kind))
    return {t: float(np.mean(v)) for t, v in acc.items() if v}


def cell_delta(beh, arm, kind, max_dup=None):
    fl = floor_for_arm(arm)
    a, f = per_task(beh, arm, kind, max_dup), per_task(beh, fl, kind, max_dup)
    sh = sorted(set(a) & set(f))
    if len(sh) < 2:
        return None
    diffs = [f[t] - a[t] for t in sh]
    res, boot = paired_bootstrap_mean(diffs, n_resamples=10000, return_distribution=True)
    return {"n": len(sh), "delta": res.estimate, "lo": res.ci_low, "hi": res.ci_high,
            "p": empirical_two_sided_p(boot, res.estimate)}


# ── STEP 1 ──────────────────────────────────────────────────────────────────
print("=" * 100)
print("STEP 1 — degeneration-immune Δ_floor  (+ve ⇒ arm suppresses behaviour beyond its matched floor)")
print("=" * 100)
report = {"step1": {}, "step2": {}}
for kind in ["count", "per1k", "frac_nodeg"]:
    md = DUP_MAX if kind == "frac_nodeg" else None
    realkind = "frac" if kind == "frac_nodeg" else kind
    cells = {(b, a): cell_delta(b, a, realkind, md) for b in TARGETS for a in ARMS}
    hp = holm({k: c["p"] for k, c in cells.items() if c})
    print(f"\n--- {kind} ---")
    print(f"  {'behaviour':24s} {'arm':16s} {'N':>3s} {'Δ':>8s} {'95% CI':>18s} {'raw p':>7s} {'Holm':>6s}")
    for b in TARGETS:
        for a in ARMS:
            c = cells[(b, a)]
            if not c:
                continue
            sig = "  <== sig" if hp[(b, a)] < 0.05 else ""
            ci = f"[{c['lo']:+.3f},{c['hi']:+.3f}]" if realkind != "count" else f"[{c['lo']:+.2f},{c['hi']:+.2f}]"
            dv = f"{c['delta']:+.3f}" if realkind != "count" else f"{c['delta']:+.2f}"
            print(f"  {b:24s} {a:16s} {c['n']:>3d} {dv:>8s} {ci:>18s} {c['p']:>7.4f} {hp[(b,a)]:>6.3f}{sig}")
            report["step1"].setdefault(kind, {})[f"{b}|{a}"] = {**c, "holm": hp[(b, a)]}

# ── STEP 2 ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("STEP 2 — cross-behaviour specificity  (does steering for B suppress B more than the other 3 targets?)")
print("=" * 100)
van = {}
for r in steered:
    if r["method"] != VANILLA:
        continue
    anns = ANN.get((r["task_id"], r["behaviour"], r["method"], r["alpha"]))
    if not anns:
        continue
    van[_base_task(r)] = {b: metric(anns, r.get("n_tokens", 0), b, "per1k") for b in TARGETS}

matrix = {}
for B in TARGETS:
    arm_rates = {}
    for r in steered:
        if r["method"] != "single_direction" or r["behaviour"] != B or not _alpha_eq(r["alpha"], 1.0):
            continue
        anns = ANN.get((r["task_id"], r["behaviour"], r["method"], r["alpha"]))
        if not anns:
            continue
        arm_rates[_base_task(r)] = {b: metric(anns, r.get("n_tokens", 0), b, "per1k") for b in TARGETS}
    rows = {b: [] for b in TARGETS}
    spec_abs, spec_rel = [], []
    for t in sorted(set(arm_rates) & set(van)):
        supp = {b: van[t][b] - arm_rates[t][b] for b in TARGETS}
        for b in TARGETS:
            rows[b].append(supp[b])
        others = [supp[b] for b in TARGETS if b != B]
        spec_abs.append(supp[B] - float(np.mean(others)))
        rel = {b: (van[t][b] - arm_rates[t][b]) / van[t][b] if van[t][b] > 0 else 0.0 for b in TARGETS}
        spec_rel.append(rel[B] - float(np.mean([rel[b] for b in TARGETS if b != B])))
    matrix[B] = {b: float(np.mean(rows[b])) for b in TARGETS}
    ra, ba = paired_bootstrap_mean(spec_abs, n_resamples=10000, return_distribution=True)
    rr, br = paired_bootstrap_mean(spec_rel, n_resamples=10000, return_distribution=True)
    report["step2"][B] = {
        "matrix_per1k": matrix[B], "n": len(spec_abs),
        "own_minus_other_abs": {"delta": ra.estimate, "ci": [ra.ci_low, ra.ci_high], "p": empirical_two_sided_p(ba, ra.estimate)},
        "own_minus_other_rel": {"delta": rr.estimate, "ci": [rr.ci_low, rr.ci_high], "p": empirical_two_sided_p(br, rr.estimate)},
    }

print("\n  4×4 suppression matrix (per-1k tokens; rows = steering vector, cols = measured behaviour; DIAGONAL = own)")
abbr = {"backtracking": "backtr", "uncertainty-estimation": "uncert", "example-testing": "ex-test", "adding-knowledge": "add-kn"}
print(f"  {'steer \\\\ measure':18s}" + "".join(f"{abbr[b]:>9s}" for b in TARGETS))
for B in TARGETS:
    print(f"  {abbr[B]:18s}" + "".join(f"{matrix[B][b]:>8.3f}{'*' if b==B else ' '}" for b in TARGETS))
print("\n  own-vs-other specificity test (single_direction arm):")
print(f"  {'behaviour':24s} {'N':>3s} {'own−other (abs/1k)':>20s} {'p':>7s} {'own−other (rel %)':>20s} {'p':>7s}")
for B in TARGETS:
    s = report["step2"][B]
    a, r = s["own_minus_other_abs"], s["own_minus_other_rel"]
    print(f"  {B:24s} {s['n']:>3d} {a['delta']:>+18.3f}   {a['p']:>5.4f} {r['delta']*100:>+18.1f}   {r['p']:>5.4f}")

(EVAL / "strengthen_report.json").write_text(json.dumps(report, indent=2))
print(f"\nsaved -> {EVAL/'strengthen_report.json'}")
