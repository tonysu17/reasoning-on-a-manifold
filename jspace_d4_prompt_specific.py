#!/usr/bin/env python3
"""D4 — prompt-specific vs corpus-averaged Jacobian (R1-1.5B).

Sealed protocol §6 (seal 752dee7) + AMENDMENT 1 §3.

Items: the 8 eligible multihop items with lowest SHA-256(ASCII item name),
ascending hex -- fixed before any computation. Per item, fit a single-prompt
Jacobian through the pinned released path and compare the bridge-label rank at
the final prompt position against the Phase-1 merged lens, at source layers 17
and 25. Ranks > 25 are censored to 26 (Phase-1 rank domain).

KNOWN CONSTRAINT (surfaced 2026-08-17, before execution): the released
skip-first-16 rule needs seq_len > 17, and 5 of the 8 sealed items are shorter.
Those items CANNOT yield a single-prompt Jacobian; they are recorded as
``insufficient_valid_positions`` and NOT substituted (substitution would be
selection after inspection). --selection amended enables the pre-declared
length-eligible variant ONLY if AMENDMENT 2 is sealed and committed.
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
LAYERS = [17, 25]
SKIP_FIRST = 16
DIM_BATCH = 8
N_ITEMS = 8
CENSOR = 26


def rank_of(top25_row: np.ndarray, token_id: int) -> int:
    hit = np.nonzero(top25_row == token_id)[0]
    return int(hit[0]) + 1 if hit.size else CENSOR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-lens", default="/workspace/artifacts/merged.fp32.pt")
    ap.add_argument("--selection", default="sealed", choices=["sealed", "amended"])
    ap.add_argument("--outroot", default="results/jspace_r1_pilot/diagnostics")
    args = ap.parse_args()

    import jlens
    from jlens import JacobianLens
    from jlens.fitting import jacobian_for_prompt
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
        selected = ranked[:N_ITEMS]
    else:
        amend = "results/prereg/JSPACE_DIAG_AMENDMENT_2_2026-08-17.md"
        if not os.path.exists(amend):
            raise SystemExit("--selection amended requires a sealed AMENDMENT 2; not found")
        selected = [it for it in ranked if len(it["prompt_token_ids"]) > SKIP_FIRST + 1][:N_ITEMS]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REV,
                                              dtype=torch.bfloat16,
                                              attn_implementation="eager").to(device)
    lens_model = jlens.from_hf(hf, tok)
    domain = C.vocabulary_domain(lens_model)
    merged = JacobianLens.load(args.merged_lens)
    merged_stack = C.jacobian_stack(merged, LAYERS, device)

    rows, n_fitted = [], 0
    for item in selected:
        name, prompt = item["name"], item["prompt"]
        labels = [int(l["scored_token_ids"][0]) for l in item["labels"] if l.get("eligible")]
        n_tokens = len(item["prompt_token_ids"])
        row = {"name": name, "n_prompt_tokens": n_tokens,
               "sha256_prefix": hashlib.sha256(name.encode("ascii")).hexdigest()[:8],
               "bridge_labels": labels}
        # merged-lens comparator (always computable)
        top_m, _, ids = C.readout_item(lens_model, prompt, merged_stack, LAYERS, domain)
        assert ids == item["prompt_token_ids"], f"prompt token IDs differ for {name}"
        row["merged_rank"] = {str(l): min(rank_of(top_m[i], lab) for lab in labels)
                              for i, l in enumerate(LAYERS)}
        # single-prompt Jacobian
        try:
            jac, seq_len, n_valid = jacobian_for_prompt(
                lens_model, prompt, source_layers=LAYERS, dim_batch=DIM_BATCH,
                max_seq_len=C.MAX_SEQ_LEN, skip_first=SKIP_FIRST)
            single = JacobianLens(jacobians=jac, n_prompts=1, d_model=lens_model.d_model)
            stack = C.jacobian_stack(single, LAYERS, device)
            top_s, _, _ = C.readout_item(lens_model, prompt, stack, LAYERS, domain)
            row["single_rank"] = {str(l): min(rank_of(top_s[i], lab) for lab in labels)
                                  for i, l in enumerate(LAYERS)}
            row["status"] = "fitted"
            row["seq_len"] = int(seq_len)
            row["n_valid_positions"] = int(n_valid)
            n_fitted += 1
            del jac, single, stack
        except Exception as exc:
            row["status"] = "insufficient_valid_positions"
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            row["single_rank"] = None
        rows.append(row)
        print(f"[d4] {name}: tokens={n_tokens} status={row['status']} "
              f"merged={row['merged_rank']} single={row.get('single_rank')}", flush=True)

    fitted = [r for r in rows if r["status"] == "fitted"]
    medians = {}
    for l in LAYERS:
        k = str(l)
        if fitted:
            medians[k] = {
                "merged_median_rank": float(np.median([r["merged_rank"][k] for r in fitted])),
                "single_median_rank": float(np.median([r["single_rank"][k] for r in fitted])),
                "n_single_better": int(sum(r["single_rank"][k] < r["merged_rank"][k] for r in fitted)),
                "n_merged_better": int(sum(r["merged_rank"][k] < r["single_rank"][k] for r in fitted)),
                "n_tied": int(sum(r["single_rank"][k] == r["merged_rank"][k] for r in fitted)),
            }
        else:
            medians[k] = None

    report = {
        "selection_mode": args.selection,
        "n_selected": len(selected), "n_fitted": n_fitted,
        "n_insufficient_valid_positions": len(selected) - n_fitted,
        "per_item": rows, "layer_summary": medians,
        "interpretation": ("descriptive only; no threshold. Averaging-destroys predicts "
                           "single-prompt ranks materially better (esp. L25); never-existed "
                           "predicts both poor everywhere. Censored rank 26 = outside top-25."),
        "coverage_caveat": (f"{len(selected) - n_fitted} of {len(selected)} sealed items could not "
                            "yield a single-prompt Jacobian under the released skip-16 rule "
                            "(prompt shorter than 18 tokens). Not substituted; see protocol §9.4."),
    }
    manifest = {
        "schema_version": "rom-jspace-d4-prompt-specific-v1", "run_uuid": run_uuid,
        "protocol": "results/prereg/JSPACE_R1_DIAGNOSTIC_PROTOCOL_2026-08-16.md",
        "seal_commit": "752dee7", "amendment": "JSPACE_DIAG_AMENDMENT_1_2026-08-17.md",
        "model": MODEL_ID, "model_revision": MODEL_REV, "model_dtype": "bfloat16",
        "attn_implementation": "eager", "layers": LAYERS, "skip_first": SKIP_FIRST,
        "dim_batch": DIM_BATCH, "merged_lens_sha256": merged_sha,
        "eligibility_manifest_sha256": ELIG_SHA256, "vocabulary_domain": domain,
        "selected_item_names": [it["name"] for it in selected],
        "env": C.environment(), "git_commit": C.git_head(),
        "finished_utc": C.utc_now(), "wall_seconds": round(time.time() - t0, 1),
    }
    md = ["# D4 — prompt-specific vs corpus-averaged Jacobian", "",
          f"Run `{run_uuid}`; sealed §6; selection `{args.selection}`; "
          f"{n_fitted}/{len(selected)} items yielded a single-prompt Jacobian.", "",
          "| Item | tokens | status | merged L17 | single L17 | merged L25 | single L25 |",
          "|---|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        s = r.get("single_rank") or {}
        md.append(f"| {r['name']} | {r['n_prompt_tokens']} | {r['status']} | "
                  f"{r['merged_rank']['17']} | {s.get('17','—')} | "
                  f"{r['merged_rank']['25']} | {s.get('25','—')} |")
    md += ["", f"**Layer summary:** {json.dumps(medians)}", "",
           f"_{report['coverage_caveat']}_", ""]
    C.write_bundle(outdir, manifest, report, "\n".join(md) + "\n")
    print(f"[d4] DONE {run_uuid} fitted={n_fitted}/{len(selected)} -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
