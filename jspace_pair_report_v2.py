#!/usr/bin/env python3
"""Versioned v2 reporting evaluator for the 7B matched pair (sealed sheet E1-E5).

Post-execution reporting/provenance correction over jspace_pair_report.py (v1).
It changes reporting only, never the scientific estimands: all primary 7B values
must equal the preserved v1 report. v1->v2 changes:

  1. exact 1.5B and hosted reference values are loaded from committed artefacts
     instead of rounded literals;
  2. complete E2 per-layer profiles are emitted alongside the fixed-band maxima;
  3. E5 emits the retained item names as well as lost and gained sets;
  4. the sealed sheet and both pre-execution amendments are recorded;
  5. a provenance block binds execution source commit, evaluator identity,
     run UUIDs, input paths+hashes, anchor identity, generation UTC, schema,
     and output version;
  6. the run ID and output destination are explicit arguments;
  7. the meaning of E1 `supported: true` is stated in the report; and
  8. no post-result numerical threshold is applied to E2, E3, or E5.

Deterministic: identical invocation over identical inputs produces
byte-identical outputs (generation UTC is an explicit argument, documented as
the only operator-supplied field).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import numpy as np

DIAG = "results/jspace_r1_pilot/diagnostics"
MID = range(10, 19)   # sealed band, zero-based, 28-layer models
LATE = range(19, 27)
SCHEMA_VERSION = "rom-jspace-pair7b-report-v2"
OUTPUT_VERSION = "v2"

SEALED_DOCS = [
    "results/prereg/JSPACE_7B_PAIR_SHEET_2026-08-17.md",
    "results/prereg/JSPACE_7B_PAIR_AMENDMENT_1_2026-08-18.md",
    "results/prereg/JSPACE_7B_PAIR_AMENDMENT_2_2026-08-18.md",
]

# Committed reference artefacts (exact values are loaded from these, never typed).
REF_MATH15B = f"{DIAG}/d3/d3-qwen2.5-math-1.5b-base-2008de267117"
REF_R15B_D5 = f"{DIAG}/d5/d5-c480b291f5aa"
REF_QWEN3 = f"{DIAG}/d3/d3-qwen3-1.7b-3bd7b4c16291"
REF_ANCHOR = f"{DIAG}/d2/d2-qwen2.5-7b-it-f1b6cdfdff03"

V1_OUTPUTS = {
    f"{DIAG}/pair7b/report.json":
        "6bc2714a85b8a03a560c43b16c5ffeaaf5edc8fad8324ffe39878cb1273f3813",
    f"{DIAG}/pair7b/REPORT.md":
        "864f957ba75cd6854b1e20d3a0b7c86901e2052ddf2ab20c8ad6e05de593e42e",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
        "layer_profile_pass_at_25": prof,
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


def load_report(d):
    return json.load(open(os.path.join(d, "report.json")))["results"]


def manifest_of(d):
    return json.load(open(os.path.join(d, "manifest.json")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True,
                    help="execution run id the 7B bundles must carry")
    ap.add_argument("--generated-utc", required=True,
                    help="ISO-8601 UTC stamp recorded as the generation time; "
                         "an explicit argument so reruns are byte-identical")
    ap.add_argument("--evaluator-commit", required=True,
                    help="analysis commit at which this evaluator is tracked")
    ap.add_argument("--outdir", default=f"{DIAG}/pair7b",
                    help="versioned output destination")
    args = ap.parse_args()

    base_dir = cell_dir("math-7b-base", execution_run_id=args.run_id)
    dist_dir = cell_dir("r1-distill-7b", execution_run_id=args.run_id)
    base = load_report(base_dir)
    dist = load_report(dist_dir)
    cells = {"math-7b-base": mh(base), "r1-distill-7b": mh(dist)}

    # Exact reference values from committed artefacts.
    math15b = mh(load_report(REF_MATH15B))
    qwen3 = mh(load_report(REF_QWEN3))
    anchor = mh(load_report(REF_ANCHOR))
    anchor_assoc = load_report(REF_ANCHOR)["lens-eval-association"]
    d5 = json.load(open(os.path.join(REF_R15B_D5, "report.json")))["results"]
    # Validated Phase-1 R1-1.5B BF16 union, recomputed from the stored Phase-1
    # arrays by the sealed D5 precision diagnostic.
    r15b_union = d5["lens-eval-multihop"]["union_pass_at_25_bf16"]

    e = {}
    e["E1_reduction_replicates"] = {
        "distill_union": cells["r1-distill-7b"]["union"],
        "base_union": cells["math-7b-base"]["union"],
        "supported": cells["r1-distill-7b"]["union"] < cells["math-7b-base"]["union"],
        "supported_meaning": (
            "supported=true records only that the registered E1 directional "
            "inequality (distill any-layer union < base any-layer union) held. "
            "It is not a paired base-vs-distill model-difference test, not an "
            "effect-size threshold, and not a causal claim about distillation. "
            "Each cell's permutation p tests token labels within that cell "
            "only."),
        "ratio": (cells["r1-distill-7b"]["union"] / cells["math-7b-base"]["union"]
                  if cells["math-7b-base"]["union"] else None),
        "comparison_1p5b": {
            "base": math15b["union"],
            "distill": r15b_union,
            "ratio": r15b_union / math15b["union"],
            "sources": {
                "base": f"{REF_MATH15B}/report.json",
                "distill": (f"{REF_R15B_D5}/report.json "
                            "(union_pass_at_25_bf16; validated Phase-1 arrays)"),
            },
        },
    }
    e["E2_location_late_band"] = {
        k: {"mid_max": v["mid_max"], "late_max": v["late_max"],
            "layer_profile_pass_at_25": v["layer_profile_pass_at_25"]}
        for k, v in cells.items()}
    e["E2_location_late_band"]["protocol_note"] = (
        "The sealed sheet registered a directional expectation, not a numerical "
        "threshold. Report both continuous maxima and the full per-layer "
        "profiles; no Boolean operationalises 'confined to late' or "
        "'approximately zero'.")
    e["E3_format_delta"] = {
        "math_7b_delta": cells["math-7b-base"]["delta_J_minus_logit"],
        "distill_7b_delta": cells["r1-distill-7b"]["delta_J_minus_logit"],
        "reference": {
            "math_1p5b": math15b["delta_J_minus_logit"],
            "qwen3_1p7b": qwen3["delta_J_minus_logit"],
            "qwen2p5_7b_it": anchor["delta_J_minus_logit"],
            "sources": {
                "math_1p5b": f"{REF_MATH15B}/report.json",
                "qwen3_1p7b": f"{REF_QWEN3}/report.json",
                "qwen2p5_7b_it": f"{REF_ANCHOR}/report.json",
            },
        },
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
        "general_7b_it_reference": anchor_assoc["any_layer_union_pass_at_25"],
    }
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
        "retained_in_distill": sorted(hb_common & hd_common),
        "lost_in_distill": sorted(hb_common - hd_common),
        "gained_in_distill": sorted(hd_common - hb_common),
        "note": ("common-eligible item names only; qualitative post-hoc reading; "
                 "no inferential weight"),
    }
    e["typo_internal_control"] = {
        k: {"union": c["lens-eval-typo"]["any_layer_union_pass_at_25"],
            "qualifies": c["lens-eval-typo"]["qualifies_instrument_level"]}
        for k, c in (("math-7b-base", base), ("r1-distill-7b", dist))}

    inputs = {}
    for d in (base_dir, dist_dir):
        for name in ("report.json", "manifest.json", "eligibility.json",
                     "top25__lens_eval_multihop.npz"):
            p = os.path.join(d, name)
            inputs[p] = sha256(p)
    for d in (REF_MATH15B, REF_QWEN3, REF_ANCHOR, REF_R15B_D5):
        for name in ("report.json", "manifest.json"):
            p = os.path.join(d, name)
            inputs[p] = sha256(p)
    for p in SEALED_DOCS:
        inputs[p] = sha256(p)

    mb, md_ = manifest_of(base_dir), manifest_of(dist_dir)
    e["provenance"] = {
        "schema_version": SCHEMA_VERSION,
        "output_version": OUTPUT_VERSION,
        "execution_run_id": args.run_id,
        "execution_source_commit": mb["git_commit"],
        "run_uuids": {"math-7b-base": mb["run_uuid"],
                      "r1-distill-7b": md_["run_uuid"]},
        "model_revisions": {"math-7b-base": mb["model_revision"],
                            "r1-distill-7b": md_["model_revision"]},
        "anchor_identity": {
            "model": manifest_of(REF_ANCHOR)["model"],
            "model_revision": manifest_of(REF_ANCHOR)["model_revision"],
            "run_uuid": manifest_of(REF_ANCHOR)["run_uuid"]},
        "sealed_documents": {p: inputs[p] for p in SEALED_DOCS},
        "evaluator": {
            "v1": {"path": "jspace_pair_report.py",
                   "sha256": sha256("jspace_pair_report.py")},
            "v2": {"path": "jspace_pair_report_v2.py",
                   "sha256": sha256("jspace_pair_report_v2.py"),
                   "git_commit": args.evaluator_commit}},
        "v1_outputs_preserved": V1_OUTPUTS,
        "generated_utc": args.generated_utc,
        "correction_type": (
            "post-execution reporting/provenance correction; estimands "
            "unchanged; see REPORTING_ADDENDUM_2026-08-18.md"),
    }

    if mb["git_commit"] != md_["git_commit"]:
        raise SystemExit("7B bundles disagree on execution source commit")
    for p, want in V1_OUTPUTS.items():
        got = sha256(p)
        if got != want:
            raise SystemExit(f"v1 output hash changed: {p} {got}")

    os.makedirs(args.outdir, exist_ok=True)
    report_path = os.path.join(args.outdir, "report_v2.json")
    md_path = os.path.join(args.outdir, "REPORT_v2.md")
    json.dump(e, open(report_path, "w"), indent=1, sort_keys=True)
    md = ["# 7B matched-pair registered evaluation — v2 reporting correction",
          "",
          "Reporting/provenance correction over the preserved v1 report; all "
          "primary 7B values are unchanged. See REPORTING_ADDENDUM_2026-08-18.md.",
          ""]
    for k in sorted(e):
        md.append(f"## {k}\n```json\n{json.dumps(e[k], indent=1, sort_keys=True)}\n```")
    open(md_path, "w").write("\n".join(md) + "\n")

    derivation = {
        "schema_version": "rom-jspace-pair7b-derivation-v2",
        "generated_utc": args.generated_utc,
        "invocation": ("python3 jspace_pair_report_v2.py "
                       f"--run-id {args.run_id} "
                       f"--generated-utc {args.generated_utc} "
                       f"--evaluator-commit {args.evaluator_commit} "
                       f"--outdir {args.outdir}"),
        "inputs_sha256": dict(sorted(inputs.items())),
        "evaluator_sha256": {
            "jspace_pair_report.py": sha256("jspace_pair_report.py"),
            "jspace_pair_report_v2.py": sha256("jspace_pair_report_v2.py")},
        "v1_outputs_preserved": V1_OUTPUTS,
        "outputs_sha256": {report_path: sha256(report_path),
                           md_path: sha256(md_path)},
        "determinism_note": (
            "identical invocation over identical inputs reproduces all outputs "
            "byte-identically; --generated-utc is the only operator-supplied "
            "field"),
    }
    json.dump(derivation,
              open(os.path.join(args.outdir, "DERIVATION_MANIFEST_v2.json"), "w"),
              indent=1, sort_keys=True)
    print(json.dumps({"written": [report_path, md_path,
                                  os.path.join(args.outdir, "DERIVATION_MANIFEST_v2.json")]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
