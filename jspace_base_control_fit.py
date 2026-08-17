#!/usr/bin/env python3
"""Base-control cell: fit a 100-prompt WikiText J-lens on Qwen2.5-Math-1.5B.

Sealed sheet: results/prereg/JSPACE_BASE_CONTROL_SHEET_2026-08-17.md.
Fit only — scoring is chained separately through jspace_d2d3_readout.py.
Resume-safe via the released checkpointing (checkpoint_every=1, resume=True).
"""
from __future__ import annotations

import json
import logging
import os
import time

import torch

import jspace_diag_common as C

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
MODEL_REV = "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
FIT_MANIFEST = "results/prereg/jspace_r1_fit_manifest.json"
FIT_MANIFEST_SHA = "59c8695f4d8bc37e8a6cfb592bba09fc72afaec7f5eca6b415f443cb90defa6e"
OUT_DIR = "data/jlens_local"
OUT_LENS = os.path.join(OUT_DIR, "qwen2.5-math-1.5b_wikitext100.pt")
CKPT = os.path.join(OUT_DIR, "qwen2.5-math-1.5b_wikitext100.ckpt.pt")


def main() -> int:
    import jlens

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    got = C.sha256_file(FIT_MANIFEST)
    assert got == FIT_MANIFEST_SHA, f"fit manifest hash mismatch: {got}"
    rows = json.load(open(FIT_MANIFEST))["rows"]
    prompts = [r["prompt"] for r in rows if r.get("split") in ("fit_a", "fit_b")]
    assert len(prompts) == 100, f"expected 100 fit prompts, got {len(prompts)}"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REV, dtype=torch.bfloat16,
        attn_implementation="eager").to(device)
    lens_model = jlens.from_hf(hf, tok)
    print(f"[base-fit] {MODEL_ID}@{MODEL_REV[:8]} on {device}; "
          f"n_layers={lens_model.n_layers} d_model={lens_model.d_model}", flush=True)

    lens = jlens.fit(lens_model, prompts, dim_batch=8, max_seq_len=C.MAX_SEQ_LEN,
                     skip_first=16, checkpoint_path=CKPT, checkpoint_every=1,
                     resume=True)
    lens.save(OUT_LENS)
    sha = C.sha256_file(OUT_LENS)
    meta = {
        "schema_version": "rom-jspace-base-control-fit-v1",
        "sheet": "results/prereg/JSPACE_BASE_CONTROL_SHEET_2026-08-17.md",
        "model": MODEL_ID, "model_revision": MODEL_REV, "device": device,
        "n_prompts": lens.n_prompts, "source_layers": lens.source_layers,
        "d_model": lens.d_model, "dim_batch": 8, "max_seq_len": C.MAX_SEQ_LEN,
        "skip_first": 16, "fit_manifest_sha256": FIT_MANIFEST_SHA,
        "lens_path": OUT_LENS, "lens_sha256": sha,
        "env": C.environment(), "git_commit": C.git_head(),
        "finished_utc": C.utc_now(), "wall_seconds": round(time.time() - t0, 1),
    }
    json.dump(meta, open(os.path.join(OUT_DIR, "qwen2.5-math-1.5b_fit_meta.json"), "w"),
              indent=1, sort_keys=True)
    print(f"[base-fit] DONE lens={OUT_LENS} sha256={sha} "
          f"wall={meta['wall_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
