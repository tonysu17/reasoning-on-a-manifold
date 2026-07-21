#!/usr/bin/env python3
"""P2 — gpt-oss-20b DSR span-activation extraction (the F1-corrected H1 input).

For every v2-annotated P0 chain: one batch-1 teacher-forced forward pass over
``prompt + chain``, residual-stream hidden states cached per layer (CPU side),
then per-sentence mean-pooling into per-DSR-label matrices via
``src.safety.cot_extraction`` — including the ``__generic__`` complement class
(consensus-unlabelled sentences) that H1's separation leg reads against.

Memory profile (the OOM audit this script exists to honour):
  * weights: MXFP4 dequantized to bf16 ≈ 42 GB -> REQUIRES a ~48 GB card
    (A40/A6000). ``pod_p2_extract.sh`` refuses to start on less.
  * forward: batch 1, no_grad, no KV growth (single pass), hidden states are
    copied to CPU by the hook cache -> GPU overhead ≈ activations of one
    sequence (~hundreds of MB at 4k tokens).
  * peak VRAM is logged per chain to the provenance file; the run aborts with
    a clear message if any forward OOMs rather than silently truncating.

Sharded + resumable: each chain writes ``{out}/shards/{task_id}.npz`` (pooled
vectors only — tiny); a rerun skips existing shards; ``--assemble`` merges
shards into ``{label}_layer{L}.npy`` + ``{label}_rows.json`` + provenance.

Usage (pod):
    python3 p2_extract_dsr.py --chains data/dsr_annotated_v2.json \
        --out-root results/safety/p2_activations [--layers 8 12 16 20] [--smoke 3]
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import MODELS_BY_CLI, provenance
from src.safety.cot_extraction import pool_dsr_spans
from src.safety.deliberation import DSR_LABELS
from src.text_offsets import find_sentence_offset

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

GENERIC_KEY = "__generic__"

# Same sentence discipline as the labelling apps (human_labelling/build_tasks.py).
_BOUNDARY = re.compile(r"(?:(?<=[.!?])\s+)|(?:\n{2,})")


def _split_sentences(text: str, min_chars: int = 30) -> list:
    spans, start = [], 0
    for m in _BOUNDARY.finditer(text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    merged = []
    for s, e in spans:
        if merged and (e - s) < min_chars:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def augment_with_generic(chain: dict) -> dict:
    """Add empty-label pseudo-spans for sentences no consensus span covers.

    ``aggregate_dsr`` only emits LABELLED runs, so the consensus contains no
    record of generic sentences at all (caught by the local smoke run:
    __generic__ pooled 0 rows). H1's separation leg needs that complement
    class, so we sentence-split the chain and add ``dsr_labels: []`` spans for
    every sentence that does not overlap a labelled region; with
    ``unlabelled_key`` set these pool under ``__generic__``. Returns a copy —
    the annotated file on disk is never mutated.
    """
    text = chain.get("chain", "")
    spans = chain.get("dsr_consensus", {}).get("spans", [])
    labelled = []
    for sp in spans:
        off = find_sentence_offset(text, sp.get("text", ""))
        if off is not None:
            labelled.append((off, off + len(sp.get("text", ""))))
    generic = []
    for s, e in _split_sentences(text):
        if not any(s < le and ls < e for ls, le in labelled):
            generic.append({"text": text[s:e], "dsr_labels": [],
                            "decision_type": None})
    out = dict(chain)
    out["dsr_consensus"] = {**chain.get("dsr_consensus", {}),
                            "spans": list(spans) + generic}
    return out


def extract_shards(args, chains, out: Path) -> None:
    import torch

    from src.chain_gen import load_model
    from src.hooks import ActivationCache
    from src.model_adapters import locate_decoder_layers

    spec = MODELS_BY_CLI[args.model]
    logger.info(f"loading {spec['id']} (bf16, MXFP4 dequantized — needs ~42 GB VRAM)")
    model, tokenizer = load_model(
        spec["id"], dtype="bfloat16", cache_dir=args.cache_dir,
        attn_implementation=args.attn_impl)
    model.eval()

    n_layers = len(locate_decoder_layers(model))
    layers = args.layers or list(range(n_layers))
    logger.info(f"{n_layers} decoder layers; extracting {len(layers)}: {layers}")

    shards = out / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    vram_log = []
    for i, chain in enumerate(chains):
        tid = str(chain["task_id"]).replace("/", "_").replace(":", "_")
        shard = shards / f"{tid}.npz"
        if shard.exists():
            continue
        full_text = chain.get("prompt", "") + chain.get("chain", "")
        enc = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(model.device) for k, v in enc.items()}
        seq_len = inputs["input_ids"].shape[1]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with ActivationCache(model, layers=layers) as cache:
            with torch.no_grad():
                model(**inputs)

            def _pool(L, positions, _cache=cache):
                return (_cache.mean_at_positions(L, positions, batch_idx=0)
                        .float().numpy().astype(np.float32))

            acc, rows = pool_dsr_spans(
                augment_with_generic(chain), offsets, _pool, layers,
                seq_len=seq_len, unlabelled_key=GENERIC_KEY)
        peak = (torch.cuda.max_memory_allocated() / 2**20
                if torch.cuda.is_available() else 0)
        vram_log.append({"task_id": chain["task_id"], "seq_len": int(seq_len),
                         "peak_vram_mib": round(peak), "sec": round(time.time() - t0, 1)})
        arrays, meta = {}, {}
        for lab in list(DSR_LABELS) + [GENERIC_KEY]:
            for L in layers:
                if acc[lab][L]:
                    arrays[f"{lab}__{L}"] = np.stack(acc[lab][L])
            meta[lab] = rows[lab]
        np.savez_compressed(shard, **arrays)
        shard.with_suffix(".rows.json").write_text(json.dumps(
            {"meta": meta, "arm": chain.get("arm"), "label": chain.get("label"),
             "difficulty": chain.get("difficulty"),
             "dsr_complete": chain.get("dsr_complete")}))
        logger.info(f"[{i+1}/{len(chains)}] {tid}: seq {seq_len} tok, "
                    f"peak {peak:.0f} MiB, {time.time()-t0:.1f}s")
    (out / "vram_log.json").write_text(json.dumps(vram_log, indent=1))


def assemble(args, chains, out: Path) -> None:
    shards = out / "shards"
    labels = list(DSR_LABELS) + [GENERIC_KEY]
    acc = {lab: {} for lab in labels}
    rows = {lab: [] for lab in labels}
    n_shards = 0
    for chain in chains:
        tid = str(chain["task_id"]).replace("/", "_").replace(":", "_")
        shard = shards / f"{tid}.npz"
        if not shard.exists():
            logger.warning(f"missing shard {tid} — assembly incomplete")
            continue
        n_shards += 1
        z = np.load(shard)
        meta = json.loads(shard.with_suffix(".rows.json").read_text())["meta"]
        for key in z.files:
            lab, L = key.rsplit("__", 1)
            acc[lab].setdefault(int(L), []).append(z[key])
        for lab in labels:
            rows[lab].extend(meta.get(lab, []))
    for lab in labels:
        safe = lab.strip("_")
        for L, mats in sorted(acc[lab].items()):
            m = np.concatenate(mats)
            np.save(out / f"{safe}_layer{L}.npy", m)
        (out / f"{safe}_rows.json").write_text(json.dumps(rows[lab]))
        logger.info(f"{lab}: {len(rows[lab])} rows")
    report = {
        "n_chains": len(chains), "n_shards": n_shards,
        "rows_per_label": {lab: len(rows[lab]) for lab in labels},
        "schema_version": "v2",
        "provenance": provenance(args=args),
    }
    (out / "P2_EXTRACTION_REPORT.json").write_text(json.dumps(report, indent=1))
    logger.info(json.dumps(report["rows_per_label"], indent=1))
    if n_shards < len(chains):
        logger.error(f"only {n_shards}/{len(chains)} shards — NOT complete")
        sys.exit(2)


def main():
    p = argparse.ArgumentParser(description="P2: gpt-oss DSR span extraction")
    p.add_argument("--chains", required=True, help="dsr_annotated_v2.json")
    p.add_argument("--model", default="gpt-oss-20b")
    p.add_argument("--layers", nargs="+", type=int, default=None)
    p.add_argument("--out-root", default="results/safety/p2_activations")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--attn-impl", default="eager",
                   help="eager = attention sinks off-Hopper (P0 precedent)")
    p.add_argument("--smoke", type=int, default=None, help="only N chains")
    p.add_argument("--assemble", action="store_true",
                   help="merge shards into label/layer matrices (no GPU needed)")
    args = p.parse_args()

    chains = json.loads(Path(args.chains).read_text())
    complete = [c for c in chains if c.get("dsr_complete")]
    if len(complete) < len(chains):
        logger.warning(f"{len(chains) - len(complete)} incomplete chains EXCLUDED")
    if args.smoke:
        complete = complete[:args.smoke]
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)

    if args.assemble:
        assemble(args, complete, out)
    else:
        extract_shards(args, complete, out)


if __name__ == "__main__":
    main()
