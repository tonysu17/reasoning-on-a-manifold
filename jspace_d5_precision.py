#!/usr/bin/env python3
"""D5 — FP32 vs BF16 unembedding margins (R1-1.5B).

Sealed protocol §7 (seal 752dee7) as clarified by AMENDMENT 1 §2: the stored
npz holds top-25 ID sets, not logits, so the FP32 arm re-extracts residuals
with the pinned model (BF16 forward, exactly as Phase 1), then applies the
FP32 merged lens and an FP32 copy of the final norm + unembedding.

Endpoint (unchanged): per suite, the count of (label, layer) hits crossing the
25/26 boundary in either direction versus the stored BF16 top-25 sets, and any
change to the any-layer union counts. Registered expectation: association
union changes by at most +/-1 item.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
import uuid

import numpy as np
import torch

import jspace_diag_common as C
import jspace_phase1_scoring as scoring

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_REV = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
ELIG_MANIFEST = "results/prereg/jspace_r1_eval_eligibility_manifest.json"
ELIG_SHA256 = "a193ce15ff18d1852a870703dba104763a01b06740c99ec8fd955e3a15927f7c"
MERGED_SHA256 = "6b5f1043b3c3fa3fcd8d4d69919772a2d763c145f996477bf715fe4b9b89f672"
NPZ_SHA256 = "02ee47f173b034db744758ac915460d1bd7ed5c347bd4022ca97dd95760c6289"
HEAD_VOCAB = 151936      # Phase-1 fixed domain
TOK_VOCAB = 151665


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-lens", default="/workspace/artifacts/merged.fp32.pt")
    ap.add_argument("--stored-npz", default="/workspace/artifacts/external_readout_arrays.npz")
    ap.add_argument("--outroot", default="results/jspace_r1_pilot/diagnostics")
    args = ap.parse_args()

    import jlens
    from jlens import JacobianLens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    run_uuid = "d5-" + uuid.uuid4().hex[:12]
    outdir = os.path.join(args.outroot, "d5", run_uuid)
    os.makedirs(outdir, exist_ok=True)

    for path, expect, label in ((ELIG_MANIFEST, ELIG_SHA256, "eligibility"),
                                (args.merged_lens, MERGED_SHA256, "merged lens"),
                                (args.stored_npz, NPZ_SHA256, "stored npz")):
        got = C.sha256_file(path)
        assert got == expect, f"{label} hash mismatch: {got}"
    elig = json.load(open(ELIG_MANIFEST))
    stored = np.load(args.stored_npz)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REV,
                                              dtype=torch.bfloat16).to(device)
    lens_model = jlens.from_hf(hf, tok)          # BF16 forward, as Phase 1
    merged = JacobianLens.load(args.merged_lens)
    source_layers = list(merged.source_layers)
    stack = C.jacobian_stack(merged, source_layers, device)
    domain = {"head_vocab_size": HEAD_VOCAB, "scored_vocab_size": TOK_VOCAB,
              "tokenizer_len": len(tok), "head_rows_excluded": HEAD_VOCAB - TOK_VOCAB,
              "tokenizer_ids_beyond_head": 0}

    # FP32 copies of the read-out path (norm + unembedding); forward stays BF16
    norm32 = copy.deepcopy(lens_model._final_norm).float().to(device)
    head32 = copy.deepcopy(lens_model._lm_head).float().to(device)

    def unembed_fp32(residual: torch.Tensor) -> torch.Tensor:
        return head32(norm32(residual.float()))

    results = {}
    for slug in C.SUITES:
        key = slug.replace("-", "_")
        bf16_top = stored[f"{key}__top25__merged"]          # [items, 27, 25]
        items = [i for i in elig["evaluations"][slug]["items"] if i["item_eligible"]]
        assert len(items) == bf16_top.shape[0], f"{slug}: item count mismatch"
        fp32_rows, labels = [], []
        for i, item in enumerate(items):
            top, _, ids = C.readout_item(lens_model, item["prompt"], stack,
                                         source_layers, domain, unembed=unembed_fp32)
            assert ids == item["prompt_token_ids"], f"{slug}:{item['name']} token IDs differ"
            fp32_rows.append(top)
            labels.append([int(l["scored_token_ids"][0]) for l in item["labels"] if l["eligible"]])
            if (i + 1) % 25 == 0:
                print(f"[d5] {slug} {i+1}/{len(items)}", flush=True)
        fp32_top = np.stack(fp32_rows)
        assert fp32_top.shape == bf16_top.shape, f"{slug}: shape {fp32_top.shape} vs {bf16_top.shape}"

        gained = lost = 0
        for it in range(fp32_top.shape[0]):
            for lay in range(fp32_top.shape[1]):
                s_b = set(int(v) for v in bf16_top[it, lay])
                s_f = set(int(v) for v in fp32_top[it, lay])
                for lab in labels[it]:
                    in_b, in_f = lab in s_b, lab in s_f
                    gained += int(in_f and not in_b)
                    lost += int(in_b and not in_f)
        n_layers = fp32_top.shape[1]
        union_b = scoring.pass_at_25(bf16_top, labels, layers=range(n_layers))
        union_f = scoring.pass_at_25(fp32_top, labels, layers=range(n_layers))
        items_b = sum(any(lab in set(int(v) for v in bf16_top[i].reshape(-1)) for lab in labels[i])
                      for i in range(len(labels)))
        items_f = sum(any(lab in set(int(v) for v in fp32_top[i].reshape(-1)) for lab in labels[i])
                      for i in range(len(labels)))
        results[slug] = {
            "n_items": int(fp32_top.shape[0]),
            "label_layer_hits_gained_in_fp32": gained,
            "label_layer_hits_lost_in_fp32": lost,
            "union_pass_at_25_bf16": union_b, "union_pass_at_25_fp32": union_f,
            "union_delta": union_f - union_b,
            "items_with_any_hit_bf16": int(items_b), "items_with_any_hit_fp32": int(items_f),
            "items_with_any_hit_delta": int(items_f - items_b),
            "top25_sets_identical": bool(np.array_equal(bf16_top, fp32_top)),
        }
        np.savez_compressed(os.path.join(outdir, f"fp32_top25__{key}.npz"), fp32=fp32_top)
        print(f"[d5] {slug}: gained={gained} lost={lost} "
              f"items {items_b}->{items_f} union {union_b:.4f}->{union_f:.4f}", flush=True)

    assoc_delta = abs(results["lens-eval-association"]["items_with_any_hit_delta"])
    verdict = {
        "association_item_delta": results["lens-eval-association"]["items_with_any_hit_delta"],
        "registered_expectation_met": bool(assoc_delta <= 1),
        "registered_expectation": "association union changes by at most +/-1 item",
        "consequence": ("account (e) precision artefact REJECTED as an explanation of the "
                        "association failure" if assoc_delta <= 1 else
                        "association result is precision-sensitive; revisit before any "
                        "attribution to model capability"),
    }
    manifest = {
        "schema_version": "rom-jspace-d5-precision-v1", "run_uuid": run_uuid,
        "protocol": "results/prereg/JSPACE_R1_DIAGNOSTIC_PROTOCOL_2026-08-16.md",
        "seal_commit": "752dee7", "amendment": "JSPACE_DIAG_AMENDMENT_1_2026-08-17.md",
        "model": MODEL_ID, "model_revision": MODEL_REV,
        "forward_dtype": "bfloat16", "readout_dtype": "float32",
        "merged_lens_sha256": MERGED_SHA256, "stored_npz_sha256": NPZ_SHA256,
        "eligibility_manifest_sha256": ELIG_SHA256,
        "source_layers": source_layers, "vocabulary_domain": domain,
        "env": C.environment(), "git_commit": C.git_head(),
        "finished_utc": C.utc_now(), "wall_seconds": round(time.time() - t0, 1),
    }
    md = ["# D5 — FP32 vs BF16 readout margins", "",
          f"Run `{run_uuid}`; sealed §7 + amendment 1 §2. BF16 forward, FP32 lens/norm/unembed.", "",
          "| Suite | items | hits gained | hits lost | items w/ hit BF16→FP32 | union BF16→FP32 |",
          "|---|---:|---:|---:|---:|---:|"]
    for s, r in results.items():
        md.append(f"| {s} | {r['n_items']} | {r['label_layer_hits_gained_in_fp32']} | "
                  f"{r['label_layer_hits_lost_in_fp32']} | "
                  f"{r['items_with_any_hit_bf16']}→{r['items_with_any_hit_fp32']} | "
                  f"{r['union_pass_at_25_bf16']:.4f}→{r['union_pass_at_25_fp32']:.4f} |")
    md += ["", f"**Verdict:** {json.dumps(verdict)}", ""]
    C.write_bundle(outdir, manifest, {"results": results, "verdict": verdict}, "\n".join(md) + "\n")
    print(f"[d5] DONE {run_uuid} -> {outdir}", flush=True)
    print(json.dumps(verdict, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
