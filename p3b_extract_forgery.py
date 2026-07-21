"""P3b phase C — extract residual activations at the injection span for every
forgery variant (genuine / forged-attacker / forged-paraphrase).

One teacher-forced forward per record's ``text`` (host chain with the span
inserted); mean-pool the residual stream over the tokens covering the inserted
span [injection_char, injection_char+injection_len). Same loader/memory profile
as P2 (bf16 MXFP4 dequantize, batch 1, hook-to-CPU) — needs ~48 GB VRAM.

Deduped forwards: the genuine records are identical across the two manifests, so
each distinct (text) is run once and reused. Sharded + resumable.

Usage (pod): python3 p3b_extract_forgery.py --manifests manifest_attacker.jsonl \
             manifest_paraphrase.jsonl --layers 6 11 12 18 [--smoke N]
Assemble:    python3 p3b_extract_forgery.py --assemble
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import MODELS_BY_CLI, provenance


def _span_token_positions(offsets, char_lo, char_hi):
    """Token indices whose char span overlaps [char_lo, char_hi)."""
    return [i for i, (a, b) in enumerate(offsets)
            if b > char_lo and a < char_hi and not (a == 0 and b == 0)]


def extract(args, records, out: Path):
    import torch

    from src.chain_gen import load_model
    from src.hooks import ActivationCache
    from src.model_adapters import locate_decoder_layers

    spec = MODELS_BY_CLI[args.model]
    model, tokenizer = load_model(spec["id"], dtype="bfloat16",
                                  cache_dir=args.cache_dir,
                                  attn_implementation=args.attn_impl)
    model.eval()
    n_layers = len(locate_decoder_layers(model))
    layers = args.layers or list(range(n_layers))

    shards = out / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    # dedupe identical variant texts across manifests
    seen_text: dict = {}
    vram = []
    # Fail-fast on memory: run the LONGEST text first, so an OOM surfaces in the
    # first forward rather than hours in. (Peak VRAM tracks sequence length;
    # P2's peak was 48.3 GB on its longest chain, ~2x headroom here.)
    records = sorted(records, key=lambda r: -len(r["text"]))
    for i, r in enumerate(records):
        key = r["text"]
        rid = f"{r['pair_id']}__{r['variant']}__{r['style_source']}".replace("/", "_").replace(":", "_")
        shard = shards / f"{rid}.npz"
        if shard.exists():
            continue
        if key in seen_text:                       # reuse identical forward
            src = seen_text[key]
            (shards / f"{rid}.link.json").write_text(json.dumps({"same_as": src}))
            continue
        enc = tokenizer(r["text"], return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(model.device) for k, v in enc.items()}
        positions = _span_token_positions(offsets, r["injection_char"],
                                           r["injection_char"] + r["injection_len"])
        if not positions:
            (shards / f"{rid}.empty.json").write_text(json.dumps({"reason": "no span tokens"}))
            continue
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with ActivationCache(model, layers=layers) as cache:
            with torch.no_grad():
                model(**inputs)
            vecs = {str(L): cache.mean_at_positions(L, positions, batch_idx=0)
                    .float().numpy().astype(np.float32) for L in layers}
        peak = (torch.cuda.max_memory_allocated() / 2**20
                if torch.cuda.is_available() else 0)
        np.savez_compressed(shard, **vecs)
        (shards / f"{rid}.meta.json").write_text(json.dumps(
            {"pair_id": r["pair_id"], "chain_id": r["chain_id"],
             "variant": r["variant"], "style_source": r["style_source"],
             "cv_fold": r["cv_fold"], "n_span_tokens": len(positions)}))
        seen_text[key] = rid
        vram.append({"rid": rid, "peak_vram_mib": round(peak), "sec": round(time.time()-t0, 1)})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()          # prevent fragmentation creep across forwards
        if i == 0 or (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(records)}] {rid}: {len(positions)} span tok, peak {peak:.0f} MiB "
                  f"(longest-first; headroom check on forward 1)")
    (out / "vram_log.json").write_text(json.dumps(vram, indent=1))


def assemble(args, records, out: Path):
    shards = out / "shards"
    rows, mats = [], {}
    for r in records:
        rid = f"{r['pair_id']}__{r['variant']}__{r['style_source']}".replace("/", "_").replace(":", "_")
        shard = shards / f"{rid}.npz"
        link = shards / f"{rid}.link.json"
        if link.exists():
            shard = shards / f"{json.loads(link.read_text())['same_as']}.npz"
        if not shard.exists():
            continue
        z = np.load(shard)
        for L in z.files:
            mats.setdefault(L, []).append(z[L])
        rows.append({"pair_id": r["pair_id"], "chain_id": r["chain_id"],
                     "variant": r["variant"], "style_source": r["style_source"],
                     "cv_fold": r["cv_fold"]})
    for L, v in mats.items():
        np.save(out / f"acts_layer{L}.npy", np.stack(v))
    (out / "rows.json").write_text(json.dumps(rows))
    rep = {"n_rows": len(rows), "layers": sorted(int(l) for l in mats),
           "by_variant_style": {}, "provenance": provenance(args=args)}
    from collections import Counter
    rep["by_variant_style"] = dict(Counter(f"{r['variant']}/{r['style_source']}" for r in rows))
    (out / "P3B_EXTRACTION_REPORT.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["by_variant_style"], indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="../reasoning-on-manifold/results/safety/p3_forgery")
    p.add_argument("--manifests", nargs="+",
                   default=["manifest_attacker.jsonl", "manifest_paraphrase.jsonl"])
    p.add_argument("--model", default="gpt-oss-20b")
    p.add_argument("--layers", nargs="+", type=int, default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--attn-impl", default="eager")
    p.add_argument("--smoke", type=int, default=None)
    p.add_argument("--assemble", action="store_true")
    args = p.parse_args()

    root = Path(__file__).parent / args.root
    records = []
    for m in args.manifests:
        for line in (root / m).read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    if args.smoke:
        records = records[:args.smoke]
    out = root / "activations"
    out.mkdir(parents=True, exist_ok=True)
    if args.assemble:
        assemble(args, records, out)
    else:
        extract(args, records, out)


if __name__ == "__main__":
    main()
