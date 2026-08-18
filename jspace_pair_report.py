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


def cell_dir(label, stage="d3", execution_run_id=None):
    hits = sorted(glob.glob(f"{DIAG}/{stage}/*{label}*"))
    if execution_run_id is not None:
        selected = []
        for path in hits:
            manifest_path = os.path.join(path, "manifest.json")
            if os.path.exists(manifest_path):
                manifest = json.load(open(manifest_path))
                if manifest.get("execution_run_id") == execution_run_id:
                    selected.append(path)
        hits = selected
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one bundle for {stage}/{label}"
                         f" run_id={execution_run_id!r}, got {hits}")
    return hits[0]


def load_cell(label, stage="d3"):
    d = json.load(open(os.path.join(cell_dir(label, stage), "report.json")))
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


def hit_items(top, items):
    if top.shape[0] != len(items):
        raise SystemExit(f"top25 rows {top.shape[0]} != eligibility rows {len(items)}")
    out = set()
    for i in range(top.shape[0]):
        seen = set(int(v) for v in top[i].reshape(-1))
        if any(int(l) in seen for l in items[i]["scored_token_ids"]):
            out.add(items[i]["name"])
    return out


def main() -> int:
    run_id_path = "data/jlens_local/RUN_ID.txt"
    if not os.path.exists(run_id_path):
        raise SystemExit(f"missing synchronized run id: {run_id_path}")
    execution_run_id = open(run_id_path).read().strip()
    base_dir = cell_dir("math-7b-base", execution_run_id=execution_run_id)
    dist_dir = cell_dir("r1-distill-7b", execution_run_id=execution_run_id)
    base = json.load(open(os.path.join(base_dir, "report.json")))["results"]
    dist = json.load(open(os.path.join(dist_dir, "report.json")))["results"]
    cells = {"math-7b-base": mh(base), "r1-distill-7b": mh(dist)}
    anchor_cell = (load_cell("qwen2.5-7b-it", stage="d2")
                   if glob.glob(f"{DIAG}/d2/*qwen2.5-7b-it*") else None)
    anchor = mh(anchor_cell) if anchor_cell is not None else None

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
        k: {"mid_max": v["mid_max"], "late_max": v["late_max"]}
        for k, v in cells.items()}
    e["E2_location_late_band"]["protocol_note"] = (
        "The sealed sheet registered a directional expectation, not a numerical "
        "threshold. Report both continuous maxima; no Boolean operationalises "
        "'confined to late' or 'approximately zero'.")
    e["E3_format_delta"] = {
        "math_7b_delta": cells["math-7b-base"]["delta_J_minus_logit"],
        "distill_7b_delta": cells["r1-distill-7b"]["delta_J_minus_logit"],
        "reference": {"math_1p5b": -0.056, "qwen3_1p7b": +0.037,
                      "qwen2p5_7b_it": (anchor["delta_J_minus_logit"]
                                        if anchor is not None else +0.093)},
        "protocol_note": ("The sealed sheet registered competing directions, not "
                          "categorical thresholds; report the continuous deltas."),
    }
    assoc_b = base["lens-eval-association"]
    assoc_d = dist["lens-eval-association"]
    e["E4_association"] = {
        "math_7b_union": assoc_b["any_layer_union_pass_at_25"],
        "math_7b_p": assoc_b["permutation"]["empirical_upper_p"],
        "math_7b_qualifies": assoc_b["qualifies_instrument_level"],
        "distill_7b_union": assoc_d["any_layer_union_pass_at_25"],
        "general_7b_it_reference": (anchor_cell["lens-eval-association"]
                                    ["any_layer_union_pass_at_25"]
                                    if anchor_cell is not None else 0.0612),
    }
    # E5: item overlap under each model's own saved eligibility and token IDs.
    eb = json.load(open(os.path.join(base_dir, "eligibility.json")))
    ed = json.load(open(os.path.join(dist_dir, "eligibility.json")))
    ib = eb["lens-eval-multihop"]["items"]
    id_ = ed["lens-eval-multihop"]["items"]
    tb = np.load(os.path.join(base_dir, "top25__lens_eval_multihop.npz"))["lens"]
    td = np.load(os.path.join(dist_dir, "top25__lens_eval_multihop.npz"))["lens"]
    hb, hd = hit_items(tb, ib), hit_items(td, id_)
    eligible_b = {i["name"] for i in ib}
    eligible_d = {i["name"] for i in id_}
    common = eligible_b & eligible_d
    hb_common, hd_common = hb & common, hd & common
    e["E5_selectivity"] = {
        "common_eligible_items": len(common),
        "base_only_eligible_items": sorted(eligible_b - eligible_d),
        "distill_only_eligible_items": sorted(eligible_d - eligible_b),
        "base_hit_items_common_eligible": len(hb_common),
        "distill_hit_items_common_eligible": len(hd_common),
        "overlap": len(hb_common & hd_common),
        "lost_in_distill": sorted(hb_common - hd_common),
        "gained_in_distill": sorted(hd_common - hb_common),
        "note": ("common-eligible item names only; qualitative post-hoc reading; "
                 "no inferential weight"),
    }
    e["typo_internal_control"] = {
        k: {"union": c["lens-eval-typo"]["any_layer_union_pass_at_25"],
            "qualifies": c["lens-eval-typo"]["qualifies_instrument_level"]}
        for k, c in (("math-7b-base", base), ("r1-distill-7b", dist))}
    e["protocol_amendment"] = (
        "results/prereg/JSPACE_7B_PAIR_AMENDMENT_1_2026-08-18.md")
    e["execution_run_id"] = execution_run_id

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
