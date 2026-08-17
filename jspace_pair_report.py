#!/usr/bin/env python3
"""Registered evaluator for the 7B matched pair (sealed sheet E1-E5).

Run locally after pulling the pod bundles. Reads the newest d3 bundles for
labels {math-7b-base, r1-distill-7b}, the 1.5B pair (Phase-1 distill arrays +
base-control bundle), and the hosted 7B-Instruct cell; evaluates E1-E5
verbatim; writes results/jspace_r1_pilot/diagnostics/pair7b/{report.json,REPORT.md}.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

DIAG = "results/jspace_r1_pilot/diagnostics"
MID = range(10, 19)   # sealed band, zero-based, 28-layer models
LATE = range(19, 27)
ELIG = "results/prereg/jspace_r1_eval_eligibility_manifest.json"
PHASE1_NPZ = ("/Users/tonysu/.codex/jspace_phase1_watch_root/results/jspace_r1_pilot/phase1/"
              "jspace-p1-20260810T224338Z-996077031021-validated-a5/external_readout_arrays.npz")


def newest(pattern):
    hits = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not hits:
        raise SystemExit(f"missing bundle: {pattern}")
    return hits[-1]


def load_cell(label):
    d = json.load(open(os.path.join(newest(f"{DIAG}/d3/*{label}*"), "report.json")))
    return d["results"]


def mh(cell):
    r = cell["lens-eval-multihop"]
    prof = r["layer_profile_pass_at_25"]
    return {
        "union": r["any_layer_union_pass_at_25"],
        "logit_union": r["logit_lens_comparator"]["any_layer_union_pass_at_25"],
        "delta_J_minus_logit": r["any_layer_union_pass_at_25"]
        - r["logit_lens_comparator"]["any_layer_union_pass_at_25"],
        "mid_max": max(prof[i] for i in MID if i < len(prof)),
        "late_max": max(prof[i] for i in LATE if i < len(prof)),
        "peak_layer": r["peak_layer"],
        "p": r["permutation"]["empirical_upper_p"],
    }


def hit_items(top, labels):
    out = set()
    for i in range(top.shape[0]):
        seen = set(int(v) for v in top[i].reshape(-1))
        if any(l in seen for l in labels[i]):
            out.add(i)
    return out


def main() -> int:
    base = load_cell("math-7b-base")
    dist = load_cell("r1-distill-7b")
    cells = {"math-7b-base": mh(base), "r1-distill-7b": mh(dist)}
    anchor = mh(load_cell("qwen2.5-7b-it")) if glob.glob(f"{DIAG}/d2/*qwen2.5-7b-it*") else None
    if anchor is None:
        d2 = json.load(open(os.path.join(newest(f"{DIAG}/d2/*qwen2.5-7b-it*"), "report.json")))

    e = {}
    e["E1_reduction_replicates"] = {
        "distill_union": cells["r1-distill-7b"]["union"],
        "base_union": cells["math-7b-base"]["union"],
        "supported": cells["r1-distill-7b"]["union"] < cells["math-7b-base"]["union"],
        "ratio": (cells["r1-distill-7b"]["union"] / cells["math-7b-base"]["union"]
                  if cells["math-7b-base"]["union"] else None),
        "comparison_1p5b": {"base": 0.4074, "distill": 0.1852, "ratio": 0.1852 / 0.4074},
    }
    e["E2_location_late_band"] = {
        k: {"mid_max": v["mid_max"], "late_max": v["late_max"],
            "confined_to_late": v["mid_max"] <= 0.05 and v["late_max"] > v["mid_max"]}
        for k, v in cells.items()}
    e["E3_format_delta"] = {
        "math_7b_delta": cells["math-7b-base"]["delta_J_minus_logit"],
        "distill_7b_delta": cells["r1-distill-7b"]["delta_J_minus_logit"],
        "reference": {"math_1p5b": -0.056, "qwen3_1p7b": +0.037, "qwen2p5_7b_it": +0.093},
        "reading": ("scale_account" if cells["math-7b-base"]["delta_J_minus_logit"] > 0.03
                    else "math_narrowing_account"
                    if cells["math-7b-base"]["delta_J_minus_logit"] < 0.0
                    else "intermediate"),
    }
    assoc_b = base["lens-eval-association"]
    assoc_d = dist["lens-eval-association"]
    e["E4_association"] = {
        "math_7b_union": assoc_b["any_layer_union_pass_at_25"],
        "math_7b_p": assoc_b["permutation"]["empirical_upper_p"],
        "math_7b_qualifies": assoc_b["qualifies_instrument_level"],
        "distill_7b_union": assoc_d["any_layer_union_pass_at_25"],
        "general_7b_it_reference": 0.0612,
    }
    # E5: item overlap using stored npz + shared eligible-item order
    elig = json.load(open(ELIG))
    items = [i for i in elig["evaluations"]["lens-eval-multihop"]["items"] if i["item_eligible"]]
    names = [i["name"] for i in items]
    labels = [[int(l["scored_token_ids"][0]) for l in it["labels"] if l["eligible"]] for it in items]
    tb = np.load(os.path.join(newest(f"{DIAG}/d3/*math-7b-base*"), "top25__lens_eval_multihop.npz"))["lens"]
    td = np.load(os.path.join(newest(f"{DIAG}/d3/*r1-distill-7b*"), "top25__lens_eval_multihop.npz"))["lens"]
    hb, hd = hit_items(tb, labels), hit_items(td, labels)
    e["E5_selectivity"] = {
        "base_hit_items": len(hb), "distill_hit_items": len(hd),
        "overlap": len(hb & hd),
        "lost_in_distill": sorted(names[i] for i in hb - hd),
        "gained_in_distill": sorted(names[i] for i in hd - hb),
        "note": "qualitative post-hoc reading of item names; no inferential weight",
    }
    e["typo_internal_control"] = {
        k: {"union": c["lens-eval-typo"]["any_layer_union_pass_at_25"],
            "qualifies": c["lens-eval-typo"]["qualifies_instrument_level"]}
        for k, c in (("math-7b-base", base), ("r1-distill-7b", dist))}

    outdir = os.path.join(DIAG, "pair7b")
    os.makedirs(outdir, exist_ok=True)
    json.dump(e, open(os.path.join(outdir, "report.json"), "w"), indent=1, sort_keys=True)
    md = ["# 7B matched-pair registered evaluation (sealed sheet E1–E5)", ""]
    for k, v in e.items():
        md.append(f"## {k}\n```json\n{json.dumps(v, indent=1)}\n```")
    open(os.path.join(outdir, "REPORT.md"), "w").write("\n".join(md) + "\n")
    print(json.dumps(e, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
