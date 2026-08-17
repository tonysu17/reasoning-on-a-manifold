#!/usr/bin/env python3
"""D4 — prompt-specific vs corpus-averaged Jacobian (R1-1.5B), two arms.

Sealed protocol §6 (seal 752dee7) + AMENDMENT 2 (length-eligible skip-16
selection) + AMENDMENT 3 (position-local arm).

Arm "skip16"  (A2): the 8 length-eligible multihop items with lowest
    SHA-256(ASCII name); released estimator, skip_first=16.
Arm "local"   (A3): ALL 81 eligible multihop items; released estimator with
    skip_first = seq_len - 2 -> the exact position-local Jacobian at the last
    maskable position (one token before the Phase-1 readout position; the
    released mask can never include the final position — declared in A3).

Per item and arm: bridge-label censored rank (>25 -> 26) at source layers
17 and 25 under (i) the per-prompt lens and (ii) the Phase-1 merged lens on
the same activations. Descriptive only; no threshold; non-gating.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid

import numpy as np
import torch

import jspace_diag_common as C

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_REV = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
ELIG_MANIFEST = "results/prereg/jspace_r1_eval_eligibility_manifest.json"
ELIG_SHA256 = "a193ce15ff18d1852a870703dba104763a01b06740c99ec8fd955e3a15927f7c"
MERGED_SHA256 = "6b5f1043b3c3fa3fcd8d4d69919772a2d763c145f996477bf715fe4b9b89f672"
AMEND2 = "results/prereg/JSPACE_DIAG_AMENDMENT_2_2026-08-17.md"
AMEND3 = "results/prereg/JSPACE_DIAG_AMENDMENT_3_2026-08-17.md"
LAYERS = [17, 25]
SKIP_FIRST = 16
DIM_BATCH = 8
N_ITEMS_SKIP16 = 8
CENSOR = 26


def rank_of(top25_row: np.ndarray, token_id: int) -> int:
    hit = np.nonzero(top25_row == token_id)[0]
    return int(hit[0]) + 1 if hit.size else CENSOR


def best_rank(top25: np.ndarray, labels: list[int], layer_ix: int) -> int:
    return min(rank_of(top25[layer_ix], lab) for lab in labels)


def layer_summary(rows: list[dict], per_key: str) -> dict:
    out = {}
    for i, l in enumerate(LAYERS):
        k = str(l)
        usable = [r for r in rows if r.get(per_key)]
        if not usable:
            out[k] = None
            continue
        out[k] = {
            "n": len(usable),
            f"{per_key}_median": float(np.median([r[per_key][k] for r in usable])),
            "merged_median": float(np.median([r["merged_rank"][k] for r in usable])),
            "n_prompt_specific_better": int(sum(r[per_key][k] < r["merged_rank"][k] for r in usable)),
            "n_merged_better": int(sum(r["merged_rank"][k] < r[per_key][k] for r in usable)),
            "n_tied": int(sum(r[per_key][k] == r["merged_rank"][k] for r in usable)),
            "n_prompt_specific_in_top25": int(sum(r[per_key][k] <= 25 for r in usable)),
            "n_merged_in_top25": int(sum(r["merged_rank"][k] <= 25 for r in usable)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-lens", default="/workspace/artifacts/merged.fp32.pt")
    ap.add_argument("--selection", default="amended", choices=["sealed", "amended"],
                    help="skip16-arm item rule: 'amended' requires sealed AMENDMENT 2")
    ap.add_argument("--arms", default="skip16,local",
                    help="comma list; 'local' requires sealed AMENDMENT 3")
    ap.add_argument("--outroot", default="results/jspace_r1_pilot/diagnostics")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    import jlens
    from jlens import JacobianLens
    from jlens.fitting import jacobian_for_prompt
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.selection == "amended" and not os.path.exists(AMEND2):
        raise SystemExit("--selection amended requires sealed AMENDMENT 2; not found")
    if "local" in arms and not os.path.exists(AMEND3):
        raise SystemExit("arm 'local' requires sealed AMENDMENT 3; not found")

    t0 = time.time()
    run_uuid = "d4-" + uuid.uuid4().hex[:12]
    outdir = os.path.join(args.outroot, "d4", run_uuid)
    os.makedirs(outdir, exist_ok=True)

    got = C.sha256_file(ELIG_MANIFEST)
    assert got == ELIG_SHA256, f"eligibility manifest hash mismatch: {got}"
    merged_sha = C.sha256_file(args.merged_lens)
    assert merged_sha == MERGED_SHA256, f"merged lens hash mismatch: {merged_sha}"
    elig = json.load(open(ELIG_MANIFEST))

    items = [i for i in elig["evaluations"]["lens-eval-multihop"]["items"]
             if i.get("item_eligible") and i.get("n_eligible_labels", 0) > 0]
    ranked = sorted(items, key=lambda it: hashlib.sha256(it["name"].encode("ascii")).hexdigest())
    if args.selection == "sealed":
        skip16_items = ranked[:N_ITEMS_SKIP16]
    else:
        skip16_items = [it for it in ranked
                        if len(it["prompt_token_ids"]) > SKIP_FIRST + 1][:N_ITEMS_SKIP16]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REV,
                                              dtype=torch.bfloat16,
                                              attn_implementation="eager").to(device)
    lens_model = jlens.from_hf(hf, tok)
    domain = C.vocabulary_domain(lens_model)
    merged = JacobianLens.load(args.merged_lens)
    merged_stack = C.jacobian_stack(merged, LAYERS, device)

    def item_labels(item):
        return [int(l["scored_token_ids"][0]) for l in item["labels"] if l.get("eligible")]

    def merged_ranks(item):
        top_m, _, ids = C.readout_item(lens_model, item["prompt"], merged_stack, LAYERS, domain)
        assert ids == item["prompt_token_ids"], f"prompt token IDs differ for {item['name']}"
        labs = item_labels(item)
        return {str(l): best_rank(top_m, labs, i) for i, l in enumerate(LAYERS)}

    def prompt_specific_ranks(item, skip_first):
        labs = item_labels(item)
        jac, seq_len, n_valid = jacobian_for_prompt(
            lens_model, item["prompt"], source_layers=LAYERS, dim_batch=DIM_BATCH,
            max_seq_len=C.MAX_SEQ_LEN, skip_first=skip_first)
        single = JacobianLens(jacobians=jac, n_prompts=1, d_model=lens_model.d_model)
        stack = C.jacobian_stack(single, LAYERS, device)
        top_s, _, _ = C.readout_item(lens_model, item["prompt"], stack, LAYERS, domain)
        ranks = {str(l): best_rank(top_s, labs, i) for i, l in enumerate(LAYERS)}
        del jac, single, stack
        return ranks, int(seq_len), int(n_valid)

    report: dict = {"arms": {}}

    # ---------------- arm "skip16" (A2) ----------------
    if "skip16" in arms:
        rows, n_fitted = [], 0
        for item in skip16_items:
            row = {"name": item["name"], "n_prompt_tokens": len(item["prompt_token_ids"]),
                   "bridge_labels": item_labels(item), "merged_rank": merged_ranks(item)}
            try:
                row["skip16_rank"], row["seq_len"], row["n_valid_positions"] = \
                    prompt_specific_ranks(item, SKIP_FIRST)
                row["status"] = "fitted"
                n_fitted += 1
            except Exception as exc:
                row["status"] = "insufficient_valid_positions"
                row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                row["skip16_rank"] = None
            rows.append(row)
            print(f"[d4:skip16] {row['name']}: {row['status']} merged={row['merged_rank']} "
                  f"single={row.get('skip16_rank')} nvp={row.get('n_valid_positions')}", flush=True)
        report["arms"]["skip16"] = {
            "amendment": "AMENDMENT 2", "skip_first": SKIP_FIRST,
            "selection_mode": args.selection, "n_selected": len(skip16_items),
            "n_fitted": n_fitted, "per_item": rows,
            "layer_summary": layer_summary(rows, "skip16_rank"),
            "caveat": ("four items rest on 1-3 valid positions (A2); n_valid_positions is shown "
                       "beside every rank and conclusions lean on the 19-22-position items"),
        }

    # ---------------- arm "local" (A3) ----------------
    if "local" in arms:
        rows = []
        for n_done, item in enumerate(ranked):
            seq_len = len(item["prompt_token_ids"])
            row = {"name": item["name"], "n_prompt_tokens": seq_len,
                   "bridge_labels": item_labels(item), "merged_rank": merged_ranks(item)}
            row["local_rank"], row["seq_len"], row["n_valid_positions"] = \
                prompt_specific_ranks(item, seq_len - 2)
            assert row["n_valid_positions"] == 1, f"{item['name']}: expected 1 valid position"
            rows.append(row)
            if (n_done + 1) % 10 == 0 or n_done + 1 == len(ranked):
                print(f"[d4:local] {n_done+1}/{len(ranked)}", flush=True)
        report["arms"]["local"] = {
            "amendment": "AMENDMENT 3",
            "skip_first_rule": "seq_len - 2 (single position; one token before the Phase-1 readout "
                               "position — the released mask cannot include the final position)",
            "n_items": len(rows), "per_item": rows,
            "layer_summary": layer_summary(rows, "local_rank"),
        }

    report["interpretation"] = (
        "descriptive only; no threshold. Registered directions: averaging-destroys (account d) => "
        "prompt-specific ranks materially better than merged (esp. L25); never-existed => both poor. "
        "A3 asymmetry: the local map is exact, so poor local ranks are strong evidence AGAINST (d).")

    manifest = {
        "schema_version": "rom-jspace-d4-prompt-specific-v2", "run_uuid": run_uuid,
        "protocol": "results/prereg/JSPACE_R1_DIAGNOSTIC_PROTOCOL_2026-08-16.md",
        "seal_commit": "752dee7",
        "amendments": ["JSPACE_DIAG_AMENDMENT_1_2026-08-17.md",
                       "JSPACE_DIAG_AMENDMENT_2_2026-08-17.md",
                       "JSPACE_DIAG_AMENDMENT_3_2026-08-17.md"],
        "arms": arms, "model": MODEL_ID, "model_revision": MODEL_REV,
        "model_dtype": "bfloat16", "attn_implementation": "eager",
        "layers": LAYERS, "dim_batch": DIM_BATCH, "merged_lens_sha256": merged_sha,
        "eligibility_manifest_sha256": ELIG_SHA256, "vocabulary_domain": domain,
        "skip16_selected_item_names": [it["name"] for it in skip16_items],
        "env": C.environment(), "git_commit": C.git_head(),
        "finished_utc": C.utc_now(), "wall_seconds": round(time.time() - t0, 1),
    }

    md = ["# D4 — prompt-specific vs corpus-averaged Jacobian (two arms)", "",
          f"Run `{run_uuid}`; sealed §6 + A2 + A3; arms: {', '.join(arms)}. "
          "Descriptive, non-gating.", ""]
    if "skip16" in arms:
        a = report["arms"]["skip16"]
        md += [f"## Arm skip16 (A2) — {a['n_fitted']}/{a['n_selected']} fitted", "",
               "| Item | tokens | valid pos | merged L17 | skip16 L17 | merged L25 | skip16 L25 |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        for r in a["per_item"]:
            s = r.get("skip16_rank") or {}
            md.append(f"| {r['name']} | {r['n_prompt_tokens']} | {r.get('n_valid_positions','—')} | "
                      f"{r['merged_rank']['17']} | {s.get('17','—')} | "
                      f"{r['merged_rank']['25']} | {s.get('25','—')} |")
        md += ["", f"Summary: `{json.dumps(a['layer_summary'])}`", ""]
    if "local" in arms:
        a = report["arms"]["local"]
        md += [f"## Arm local (A3) — all {a['n_items']} items, exact single-position Jacobian", "",
               f"Summary: `{json.dumps(a['layer_summary'])}`", "",
               "Per-item table in report.json. Rank 26 = outside top-25 (censored).", ""]
    md += [f"_{report['interpretation']}_", ""]
    C.write_bundle(outdir, manifest, report, "\n".join(md) + "\n")
    print(f"[d4] DONE {run_uuid} arms={arms} -> {outdir}", flush=True)
    for arm in arms:
        print(f"[d4] {arm} summary: {json.dumps(report['arms'][arm]['layer_summary'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
