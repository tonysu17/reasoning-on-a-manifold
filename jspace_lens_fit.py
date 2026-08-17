#!/usr/bin/env python3
"""Generic 100-prompt WikiText J-lens fit for any pinned model.

Sealed sheets: JSPACE_BASE_CONTROL_SHEET_2026-08-17.md (1.5B base cell) and
JSPACE_7B_PAIR_SHEET_2026-08-17.md (7B pair). Fit prompts, recipe, and the
registered OOM fallback ladder (dim_batch 8 -> 4 -> 2; identical estimator,
attempted value recorded) are fixed by the sheets. Fit only; scoring chains
separately through jspace_d2d3_readout.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import torch

import jspace_diag_common as C

FIT_MANIFEST = "results/prereg/jspace_r1_fit_manifest.json"
FIT_MANIFEST_SHA = "59c8695f4d8bc37e8a6cfb592bba09fc72afaec7f5eca6b415f443cb90defa6e"
DIM_BATCH_LADDER = [8, 4, 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out-dir", default="data/jlens_local")
    ap.add_argument("--out-name", default=None, help="basename; default derived from model id")
    ap.add_argument("--save-dtype", default="fp16", choices=["fp16", "fp32"])
    args = ap.parse_args()

    import jlens

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.time()
    name = args.out_name or args.model.split("/")[-1].lower().replace(".", "p")
    os.makedirs(args.out_dir, exist_ok=True)
    out_lens = os.path.join(args.out_dir, f"{name}_wikitext100.pt")
    ckpt = os.path.join(args.out_dir, f"{name}_wikitext100.ckpt.pt")

    got = C.sha256_file(FIT_MANIFEST)
    assert got == FIT_MANIFEST_SHA, f"fit manifest hash mismatch: {got}"
    rows = json.load(open(FIT_MANIFEST))["rows"]
    prompts = [r["prompt"] for r in rows if r.get("split") in ("fit_a", "fit_b")]
    assert len(prompts) == 100, f"expected 100 fit prompts, got {len(prompts)}"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        attn_implementation="eager").to(device)
    lens_model = jlens.from_hf(hf, tok)
    print(f"[fit:{name}] {args.model}@{args.revision[:8]} on {device}; "
          f"n_layers={lens_model.n_layers} d_model={lens_model.d_model}", flush=True)

    lens = None
    used_dim_batch = None
    for db in DIM_BATCH_LADDER:
        try:
            lens = jlens.fit(lens_model, prompts, dim_batch=db,
                             max_seq_len=C.MAX_SEQ_LEN, skip_first=16,
                             checkpoint_path=ckpt, checkpoint_every=1, resume=True)
            used_dim_batch = db
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[fit:{name}] OOM at dim_batch={db}; registered fallback to next rung",
                  flush=True)
    if lens is None:
        raise SystemExit(f"[fit:{name}] OOM at every registered dim_batch rung")

    dtype = torch.float16 if args.save_dtype == "fp16" else torch.float32
    lens.save(out_lens, dtype=dtype)
    sha = C.sha256_file(out_lens)
    meta = {
        "schema_version": "rom-jspace-generic-fit-v1",
        "sheets": ["results/prereg/JSPACE_BASE_CONTROL_SHEET_2026-08-17.md",
                   "results/prereg/JSPACE_7B_PAIR_SHEET_2026-08-17.md"],
        "model": args.model, "model_revision": args.revision, "device": device,
        "n_prompts": lens.n_prompts, "source_layers": lens.source_layers,
        "d_model": lens.d_model, "dim_batch_used": used_dim_batch,
        "dim_batch_ladder": DIM_BATCH_LADDER, "max_seq_len": C.MAX_SEQ_LEN,
        "skip_first": 16, "save_dtype": args.save_dtype,
        "fit_manifest_sha256": FIT_MANIFEST_SHA,
        "lens_path": out_lens, "lens_sha256": sha,
        "env": C.environment(), "git_commit": C.git_head(),
        "finished_utc": C.utc_now(), "wall_seconds": round(time.time() - t0, 1),
    }
    json.dump(meta, open(os.path.join(args.out_dir, f"{name}_fit_meta.json"), "w"),
              indent=1, sort_keys=True)
    print(f"[fit:{name}] DONE lens={out_lens} sha256={sha} dim_batch={used_dim_batch} "
          f"wall={meta['wall_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
