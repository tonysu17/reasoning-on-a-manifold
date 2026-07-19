#!/usr/bin/env python3
"""Post-training spillover — Step 8: surprisal control (C4) + entropy battery.

One teacher-forced corpus pass with BOTH the base and the post-trained model in
memory, on BYTE-IDENTICAL token sequences (tokenised once, exactly like
04_extract_activations.py: full_text = prompt + chain, occurrence-aware span
offsets, optional --tokenizer-alias so e.g. STAR1's extra BOS never diverges the
input_ids). Per target token t it computes, in fp32:

    nll_base[t]  = -log p_base(x_t | x_<t)      (a) base surprisal
    nll_post[t]  = -log p_post(x_t | x_<t)      (b) post surprisal
    H_base[t], H_post[t]                        (c) predictive entropy of each
                                                    model's next-token distribution
    kl[t] = KL(p_post || p_base)                (d) full-vocab dose meter

and aggregates two deliverables into one JSON:

1. SURPRISAL CONTROL (pre-registration gate C4, METHODOLOGY_SAFETY_SPILLOVER
   §2.4): per-span mean base/post NLL tables keyed by (chain_id, behaviour,
   annotation_index) — the SAME keys as the extraction row_index.json — so the
   correlation between per-span base-NLL and the existing per-span displacement
   norms can be joined offline. The window means use the extraction's exact
   token window (1 preceding + first 10 execution tokens, unclipped); the
   full-sentence means are reported alongside. Plus per-behaviour mean
   ΔNLL (post − base): if the spillover selectivity ranking tracks surprisal,
   the "selectivity" is distribution shift, not safety.

2. ENTROPY BATTERY: mean H per model overall; mean ΔH (post − base) overall and
   on-span vs off-span per behaviour; mean KL(post||base) overall and
   on/off-span (chain-region tokens) — the dose meter.

Memory discipline: full-vocab logits are materialised for ONE chain at a time
(model-native bf16); NLL/H/KL are computed in fp32 over --batch-size-token
chunks and accumulated in float64 — the full-vocab fp32 tensors never exceed
one chunk per model. No activations are captured (no --layers, no hooks).

Examples
--------
    # real run (pod): R1-1.5B vs STAR-1 safety-SFT twin, base tokenizer for both
    python pt08_surprisal_entropy.py --base 1.5b --post star1-1.5b \
        --tokenizer-alias 1.5b --device auto \
        --out results/safety_posttrain/pt08_surprisal_entropy.json

    # arbitrary local checkpoints (e.g. a merged LoRA dose)
    python pt08_surprisal_entropy.py --base deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --post checkpoints/r1_1.5b_safety/dose_all/merged --tokenizer-alias 1.5b
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.config import MODELS_BY_CLI, model_tuple, provenance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pt08")

# Extraction-window parameters — MUST match extract_activations() defaults so the
# per-span window NLL covers the same tokens the displacement rows pooled over.
N_PRECEDING = 1
N_EXECUTION = 10
CLIP_WINDOW_TO_SENTENCE_END = False

#: Token-level statistics tracked by every aggregate (all fp32 per token,
#: accumulated in float64).
STAT_FIELDS = ("nll_base", "nll_post", "H_base", "H_post", "kl")


def resolve_model_arg(spec: str, dtype: str) -> tuple[str, str, str]:
    """Resolve --base/--post: a registered cli_alias (configs/config.yaml) or a
    raw HF id / local checkpoint path. Returns (model_id, short_name, dtype);
    the compute dtype always comes from --dtype (pt03 convention — the registry
    entries say float16 but the spillover scoring runs bf16)."""
    if spec in MODELS_BY_CLI:
        model_id, short, _ = model_tuple(spec)
        return model_id, short, dtype
    return spec, Path(spec).name, dtype


def pick_device(spec: str) -> str:
    if spec != "auto":
        return spec
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _Acc:
    """float64 sum/count accumulator over the per-token statistics."""

    def __init__(self) -> None:
        self.sum = {f: 0.0 for f in STAT_FIELDS}
        self.n = 0

    def add(self, stats: dict[str, np.ndarray], mask: np.ndarray) -> None:
        k = int(mask.sum())
        if k == 0:
            return
        for f in STAT_FIELDS:
            self.sum[f] += float(stats[f][mask].sum(dtype=np.float64))
        self.n += k

    def report(self) -> dict:
        if self.n == 0:
            return {"n_tokens": 0}
        m = {f: self.sum[f] / self.n for f in STAT_FIELDS}
        return {
            "n_tokens": self.n,
            "mean_nll_base": round(m["nll_base"], 6),
            "mean_nll_post": round(m["nll_post"], 6),
            "mean_delta_nll": round(m["nll_post"] - m["nll_base"], 6),
            "mean_H_base": round(m["H_base"], 6),
            "mean_H_post": round(m["H_post"], 6),
            "mean_dH": round(m["H_post"] - m["H_base"], 6),
            "mean_kl": round(m["kl"], 6),
        }


def score_chain_pair(model_base, model_post, input_ids, batch_size: int) -> dict[str, np.ndarray]:
    """Teacher-force BOTH models on one chain's input_ids (1, T) and return
    per-target-token fp32 arrays of length T-1 (index i <-> token position i+1):
    nll_base, nll_post, H_base, H_post, kl = KL(post||base).

    Full-vocab logits exist for this ONE chain only (model-native dtype); the
    fp32 log-softmax / entropy / KL work is chunked to `batch_size` token
    positions at a time so fp32 full-vocab tensors never exceed one chunk.
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        logits_base = model_base(input_ids=input_ids, use_cache=False).logits[0]
        logits_post = model_post(input_ids=input_ids, use_cache=False).logits[0]
    if logits_base.shape != logits_post.shape:
        raise ValueError(
            f"base/post logit shapes differ ({tuple(logits_base.shape)} vs "
            f"{tuple(logits_post.shape)}) — not a same-vocab model pair"
        )

    targets = input_ids[0, 1:]
    n = targets.shape[0]
    out = {f: torch.empty(n, dtype=torch.float32, device=input_ids.device)
           for f in STAT_FIELDS}

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        # logits row i predicts input_ids[i+1] == targets[i]
        lb = F.log_softmax(logits_base[s:e].to(torch.float32), dim=-1)
        lp = F.log_softmax(logits_post[s:e].to(torch.float32), dim=-1)
        tg = targets[s:e].unsqueeze(1)
        out["nll_base"][s:e] = -lb.gather(1, tg).squeeze(1)
        out["nll_post"][s:e] = -lp.gather(1, tg).squeeze(1)
        pb, pp = lb.exp(), lp.exp()
        out["H_base"][s:e] = -(pb * lb).sum(dim=-1)
        out["H_post"][s:e] = -(pp * lp).sum(dim=-1)
        out["kl"][s:e] = (pp * (lp - lb)).sum(dim=-1)
        del lb, lp, pb, pp

    del logits_base, logits_post
    return {f: out[f].cpu().numpy() for f in STAT_FIELDS}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True,
                    help="base model: cli_alias from config.yaml (e.g. '1.5b') "
                         "or an HF id / local checkpoint path")
    ap.add_argument("--post", required=True,
                    help="post-trained model: cli_alias (e.g. 'star1-1.5b') or path")
    ap.add_argument("--annotated", default="data/annotated_R1-1.5B.json")
    ap.add_argument("--tokenizer-alias", default=None, choices=list(MODELS_BY_CLI),
                    help="tokenize with ANOTHER registered model's tokenizer "
                         "(04_extract_activations convention; both models are "
                         "teacher-forced on the ONE resulting token sequence, "
                         "so input_ids are byte-identical by construction — "
                         "pass '--tokenizer-alias 1.5b' for the STAR1 pair "
                         "since STAR1's tokenizer prepends an extra BOS)")
    ap.add_argument("--behaviours", default=None,
                    help="comma list; default = all 6 annotation labels (pt03 convention)")
    ap.add_argument("--dtype", default="bfloat16",
                    help="model compute dtype (fp32 statistics regardless)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--batch-size", type=int, default=1024,
                    help="scoring chunk size in TOKEN POSITIONS (not chains): the "
                         "fp32 full-vocab log-softmax/entropy/KL buffers are "
                         "(batch_size, vocab); lower it if memory is tight "
                         "(1024 x 152k vocab fp32 ~= 0.6 GB per live tensor)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", default="results/safety_posttrain/pt08_surprisal_entropy.json")
    args = ap.parse_args(argv)

    # Heavy imports deferred so --help / arg errors don't require torch.
    import torch
    from src.activation_extraction import _sentence_to_token_positions
    from src.annotation import VALID_LABELS, load_annotated
    from src.chain_gen import load_model
    from src.text_offsets import locate_annotation_offsets

    if args.behaviours:
        behaviours = [b.strip() for b in args.behaviours.split(",")]
    else:
        behaviours = sorted(VALID_LABELS)

    device = pick_device(args.device)
    base_id, base_short, dtype = resolve_model_arg(args.base, args.dtype)
    post_id, post_short, _ = resolve_model_arg(args.post, args.dtype)
    log.info("base=%s  post=%s  device=%s dtype=%s behaviours=%s",
             base_id, post_id, device, dtype, behaviours)

    annotated = load_annotated(Path(args.annotated))
    log.info("loaded %d annotated chains from %s", len(annotated), args.annotated)

    log.info("loading BASE model: %s", base_id)
    model_base, tokenizer = load_model(base_id, dtype=dtype,
                                       device_map={"": device},
                                       cache_dir=args.cache_dir)
    log.info("loading POST model: %s", post_id)
    model_post, _ = load_model(post_id, dtype=dtype, device_map={"": device},
                               cache_dir=args.cache_dir)
    model_base.eval()
    model_post.eval()
    if args.tokenizer_alias:
        from transformers import AutoTokenizer
        tok_id = model_tuple(args.tokenizer_alias)[0]
        log.info("Tokenizer override: %s (byte-identical input_ids across "
                 "checkpoints, as in 04_extract_activations)", tok_id)
        tokenizer = AutoTokenizer.from_pretrained(tok_id, cache_dir=args.cache_dir)
    tokenizer_id = model_tuple(args.tokenizer_alias)[0] if args.tokenizer_alias else base_id

    # Aggregates: whole corpus, prompt region, chain region, any annotated span,
    # and per-behaviour on/off-span (off-span = chain tokens outside that
    # behaviour's spans).
    agg = {
        "overall": _Acc(),
        "prompt_tokens": _Acc(),
        "chain_tokens": _Acc(),
        "any_on": _Acc(),
        "any_off": _Acc(),
    }
    beh_agg = {b: {"on_span": _Acc(), "off_span": _Acc()} for b in behaviours}
    per_span_rows: list[dict] = []
    n_spans_skipped = {b: 0 for b in behaviours}
    n_chains_scored = 0

    from tqdm import tqdm
    for chain in tqdm(annotated, desc="Scoring chains (base+post)"):
        chain_text: str = chain["chain"]
        prompt_text: str = chain["prompt"]
        full_text: str = prompt_text + chain_text
        chain_offset: int = len(prompt_text)
        chain_id = chain.get("chain_id") or chain.get("task_id")
        annotations = chain.get("annotations", [])

        # Occurrence-aware offsets over the FULL annotation list (all labels) so
        # the cursor advances identically to extraction (CF-13).
        sent_offsets = locate_annotation_offsets(
            chain_text, [a.get("text", "") for a in annotations]
        )

        enc = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
        offsets: list[tuple[int, int]] = enc.pop("offset_mapping")[0].tolist()
        input_ids = enc["input_ids"].to(device)
        seq_len = input_ids.shape[1]
        if seq_len < 2:
            continue

        stats = score_chain_pair(model_base, model_post, input_ids, args.batch_size)
        n_chains_scored += 1

        # Masks over target-token positions t = 1..T-1 (array index i = t-1).
        starts = np.array([o[0] for o in offsets[1:]], dtype=np.int64)
        ends = np.array([o[1] for o in offsets[1:]], dtype=np.int64)
        real = ends > starts                      # zero-width = special tokens
        chain_mask = real & (starts >= chain_offset)
        all_mask = np.ones(seq_len - 1, dtype=bool)

        agg["overall"].add(stats, all_mask)
        agg["chain_tokens"].add(stats, chain_mask)
        agg["prompt_tokens"].add(stats, ~chain_mask)

        beh_on = {b: np.zeros(seq_len - 1, dtype=bool) for b in behaviours}
        for ann_idx, ann in enumerate(annotations):
            cat = ann["label"]
            if cat not in behaviours:
                continue
            sent_offset = sent_offsets[ann_idx]
            if sent_offset is None:
                n_spans_skipped[cat] += 1
                continue
            abs_start = chain_offset + sent_offset
            abs_end = abs_start + len(ann.get("text", ""))

            # (i) full-sentence token set: target tokens overlapping the span
            span_mask = chain_mask & (starts < abs_end) & (ends > abs_start)
            beh_on[cat] |= span_mask

            # (ii) extraction window (1 preceding + 10 execution, unclipped) —
            # the tokens the displacement rows pooled over; t=0 has no NLL.
            positions = _sentence_to_token_positions(
                full_text, abs_start, offsets, N_PRECEDING, N_EXECUTION,
                sentence_end_char=abs_end if CLIP_WINDOW_TO_SENTENCE_END else None,
            )
            widx = np.array([p - 1 for p in positions if 1 <= p < seq_len],
                            dtype=np.int64)
            if widx.size == 0 and not span_mask.any():
                n_spans_skipped[cat] += 1
                continue

            row = {"chain_id": chain_id, "behaviour": cat,
                   "annotation_index": ann_idx,
                   "n_window_tokens": int(widx.size),
                   "n_span_tokens": int(span_mask.sum())}
            if widx.size:
                mb = float(stats["nll_base"][widx].mean(dtype=np.float64))
                mp = float(stats["nll_post"][widx].mean(dtype=np.float64))
                row.update(mean_nll_base=round(mb, 5), mean_nll_post=round(mp, 5),
                           delta_nll=round(mp - mb, 5))
            if span_mask.any():
                sb = float(stats["nll_base"][span_mask].mean(dtype=np.float64))
                sp = float(stats["nll_post"][span_mask].mean(dtype=np.float64))
                row.update(
                    mean_nll_base_span=round(sb, 5),
                    mean_nll_post_span=round(sp, 5),
                    delta_nll_span=round(sp - sb, 5),
                    mean_kl_span=round(float(stats["kl"][span_mask]
                                             .mean(dtype=np.float64)), 5),
                    mean_dH_span=round(float((stats["H_post"][span_mask]
                                              - stats["H_base"][span_mask])
                                             .mean(dtype=np.float64)), 5),
                )
            per_span_rows.append(row)

        any_on = np.zeros(seq_len - 1, dtype=bool)
        for b in behaviours:
            agg_b = beh_agg[b]
            agg_b["on_span"].add(stats, beh_on[b])
            agg_b["off_span"].add(stats, chain_mask & ~beh_on[b])
            any_on |= beh_on[b]
        agg["any_on"].add(stats, any_on)
        agg["any_off"].add(stats, chain_mask & ~any_on)

    # ── Surprisal-control summary (C4): per-behaviour mean ΔNLL over spans ────
    per_behaviour: dict[str, dict] = {}
    for b in behaviours:
        rows = [r for r in per_span_rows if r["behaviour"] == b]
        win = [r for r in rows if r["n_window_tokens"] > 0]
        sen = [r for r in rows if r["n_span_tokens"] > 0]
        per_behaviour[b] = {
            "n_spans": len(rows),
            "n_spans_skipped": n_spans_skipped[b],
            "mean_nll_base_window": round(float(np.mean(
                [r["mean_nll_base"] for r in win])), 5) if win else None,
            "mean_delta_nll_window": round(float(np.mean(
                [r["delta_nll"] for r in win])), 5) if win else None,
            "mean_delta_nll_span": round(float(np.mean(
                [r["delta_nll_span"] for r in sen])), 5) if sen else None,
        }

    report = {
        "provenance": provenance(args, inputs=[args.annotated]),
        "config": {
            "base_model": base_id, "post_model": post_id,
            "base_short": base_short, "post_short": post_short,
            "tokenizer": tokenizer_id, "device": device, "dtype": dtype,
            "batch_size_tokens": args.batch_size, "behaviours": behaviours,
            "n_preceding": N_PRECEDING, "n_execution": N_EXECUTION,
            "clip_window_to_sentence_end": CLIP_WINDOW_TO_SENTENCE_END,
        },
        "corpus": {
            "n_chains": len(annotated),
            "n_chains_scored": n_chains_scored,
            "n_target_tokens": agg["overall"].n,
            "n_spans_located": len(per_span_rows),
            "n_spans_skipped": int(sum(n_spans_skipped.values())),
        },
        "surprisal_control": {
            "note": ("C4 gate: join per_span with the extraction row_index.json "
                     "displacement norms on (chain_id, behaviour, annotation_index); "
                     "window means cover the extraction's 1+10 token window, "
                     "*_span means the full sentence. If the spillover selectivity "
                     "ranking tracks mean_nll, it is distribution shift, not safety."),
            "per_behaviour": per_behaviour,
            "per_span": per_span_rows,
        },
        "entropy_battery": {
            "overall": agg["overall"].report(),
            "prompt_tokens": agg["prompt_tokens"].report(),
            "chain_tokens": agg["chain_tokens"].report(),
            "any_annotated_span": {"on_span": agg["any_on"].report(),
                                   "off_span": agg["any_off"].report()},
            "by_behaviour": {b: {"on_span": beh_agg[b]["on_span"].report(),
                                 "off_span": beh_agg[b]["off_span"].report()}
                             for b in behaviours},
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    ov = report["entropy_battery"]["overall"]
    log.info("overall (%d tokens): H_base=%.4f H_post=%.4f dH=%+.4f  KL(post||base)=%.5f",
             ov.get("n_tokens", 0), ov.get("mean_H_base", float("nan")),
             ov.get("mean_H_post", float("nan")), ov.get("mean_dH", float("nan")),
             ov.get("mean_kl", float("nan")))
    ranking = sorted(((b, d["mean_delta_nll_window"]) for b, d in per_behaviour.items()
                      if d["mean_delta_nll_window"] is not None),
                     key=lambda kv: kv[1], reverse=True)
    log.info("per-behaviour mean ΔNLL (post−base, window): %s",
             ", ".join(f"{b}={v:+.4f}" for b, v in ranking))
    log.info("wrote surprisal/entropy report -> %s", out)


if __name__ == "__main__":
    main()
