#!/usr/bin/env python3
"""
Phase 7b - Activation patching: causal layer localisation per behaviour.

Companion document Section 4.2 (revised): promoted from "Medium / appendix"
to a Paper 2 main result. Provides the causal complement to the geometric
characterisation in Paper 2.

Procedure:
  1. Load annotated chains (data/annotated_<model>.json).
  2. For each target behaviour, select donor pairs:
       positive chain   - sentence labelled with the target behaviour
       negative chain   - sentence at matched chain-position labelled DEDUCTION
  3. For each donor pair, run a forward pass on both chains up to the
     transition token; cache residuals at the saved layers; then patch the
     positive's residual at each layer L with the negative's residual at the
     same (layer, token_position), and measure behaviour-marker logprob
     shift on the next token.
  4. Aggregate across pairs to produce a per-layer effect curve per behaviour.

Outputs:
  results/patching/<model>/effect_curves_<behaviour>.json
  results/patching/<model>/effect_curves_summary.md
  results/patching/<model>/effect_curves.png

Requires GPU. Runtime: ~30 min for 4 behaviours x 20 pairs x 8 layers on
DGX Spark.

Usage:
  python 07b_activation_patching.py --behaviours backtracking --n-pairs 20
  python 07b_activation_patching.py --behaviours all
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from activation_patching import (
    select_donor_pairs,
    layer_sweep_single_pair,
    aggregate_layer_effect,
    behaviour_marker_token_ids,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


from src.annotation import TARGET_BEHAVIOURS


def load_model_and_tokenizer(model_id: str, dtype: str = "float16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Loading {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id)
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype, torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, device_map="auto")
    model.eval()
    return model, tok


def cache_residuals(model, input_ids, layers):
    """Run a forward pass and cache the residual stream at each layer in `layers`.
    Returns dict {layer: tensor of shape (T, d)}."""
    import torch
    cache = {}
    handles = []
    def make_hook(L):
        def hook(module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cache[L] = h[0].detach().clone()   # (T, d) on the model device
        return hook
    for L in layers:
        handles.append(model.model.layers[L].register_forward_hook(make_hook(L)))
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    for h in handles:
        h.remove()
    return cache




def _sentence_token_onset(tok, chain_text: str, sentence_text: str) -> int | None:
    """Return the token index in *chain_text* at which *sentence_text* begins.
    Uses the same offset-mapping logic as src/activation_extraction.py.
    Returns None if the sentence cannot be located."""
    char_off = chain_text.find(sentence_text)
    if char_off < 0:
        # Fallback: match first 40 chars (tolerate minor annotator rewording)
        prefix = sentence_text[:40].strip()
        if len(prefix) < 10:
            return None
        char_off = chain_text.find(prefix)
        if char_off < 0:
            return None
    enc = tok(chain_text, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc["offset_mapping"][0].tolist()
    for tok_idx, (s, e) in enumerate(offsets):
        if e > char_off:
            return tok_idx
    return None

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-short", default="R1-1.5B")
    parser.add_argument("--model-id",    default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--annotated", type=Path, default=None,
                        help="Path to annotated chains JSON. Default: data/annotated_<model>.json")
    parser.add_argument("--behaviours", nargs="+", default=["backtracking"],
                        help="Subset of behaviours, or 'all'.")
    parser.add_argument("--n-pairs", type=int, default=20,
                        help="Donor pairs per behaviour. Default 20.")
    parser.add_argument("--layers", nargs="+", type=int, default=[3, 7, 10, 14, 17, 21, 24, 27])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.behaviours == ["all"]:
        args.behaviours = list(TARGET_BEHAVIOURS)
    if args.annotated is None:
        args.annotated = Path(f"data/annotated_{args.model_short}.json")
    if args.out_dir is None:
        args.out_dir = Path(f"results/patching/{args.model_short}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.annotated.exists():
        logger.error(f"Annotated chains not found: {args.annotated}")
        sys.exit(1)

    with open(args.annotated) as f:
        annotations = json.load(f)
    logger.info(f"Loaded {len(annotations)} annotated chains")

    # Load model once for all behaviours
    model, tok = load_model_and_tokenizer(args.model_id, args.dtype)
    import torch
    device = next(model.parameters()).device

    summary = {}

    for beh in args.behaviours:
        logger.info(f"=== {beh} ===")
        pairs = select_donor_pairs(annotations, beh, n_pairs=args.n_pairs,
                                     random_state=args.seed)
        if not pairs:
            logger.warning(f"No donor pairs found for {beh}; skipping")
            summary[beh] = {"error": "no_donor_pairs"}
            continue
        logger.info(f"Selected {len(pairs)} donor pairs")

        results_by_pair = []
        for pair_idx, ((pos_chain, pos_ann), (neg_chain, neg_ann)) in enumerate(pairs):
            t0 = time.time()
            try:
                # Encode both chains up to the transition token
                pos_text = pos_chain.get("chain", "")
                neg_text = neg_chain.get("chain", "")
                pos_ids = tok.encode(pos_text, return_tensors="pt").to(device)
                neg_ids = tok.encode(neg_text, return_tensors="pt").to(device)

                # Behaviour onset token position: annotation file only carries
                # the sentence text (no token_start), so compute the token index
                # on the fly using the same logic as activation_extraction.py.
                pos_sentence = pos_ann.get("text", "")
                neg_sentence = neg_ann.get("text", "")
                pos_tpos = _sentence_token_onset(tok, pos_text, pos_sentence)
                neg_tpos = _sentence_token_onset(tok, neg_text, neg_sentence)
                if pos_tpos is None or neg_tpos is None:
                    logger.warning(f"  pair {pair_idx}: could not locate sentence; skipping")
                    continue

                # Cache negative-chain residuals at all layers
                neg_residuals = cache_residuals(model, neg_ids, args.layers)

                # Truncate both chains to a common minimum length to avoid index issues
                T_min = min(pos_ids.shape[1], neg_ids.shape[1], int(pos_tpos) + 1, int(neg_tpos) + 1)
                if T_min < 2:
                    continue
                pos_trunc = pos_ids[:, :T_min]
                neg_trunc = neg_ids[:, :T_min]
                neg_res_trunc = {L: v[:T_min] for L, v in neg_residuals.items()}
                tpos = T_min - 1   # always patch at the boundary token

                results = layer_sweep_single_pair(
                    model, tok, pos_trunc, neg_trunc,
                    neg_res_trunc, args.layers, tpos, beh,
                )
                # Stamp chain IDs for the record
                for r in results:
                    r.positive_chain_id = pos_chain.get("chain_id") or pos_chain.get("task_id", "")
                    r.negative_chain_id = neg_chain.get("chain_id") or neg_chain.get("task_id", "")
                results_by_pair.append(results)

                if (pair_idx + 1) % 5 == 0:
                    logger.info(f"  pair {pair_idx+1}/{len(pairs)} done ({time.time()-t0:.1f}s)")
            except Exception as e:
                logger.warning(f"  pair {pair_idx}: {type(e).__name__}: {e}")

        # Aggregate
        agg = aggregate_layer_effect(results_by_pair)
        per_beh_out = args.out_dir / f"effect_curves_{beh}.json"
        per_beh_out.write_text(json.dumps({
            "behaviour": beh,
            "n_pairs": len(results_by_pair),
            "layer_effect": agg,
            "trials": [[asdict(r) for r in trial] for trial in results_by_pair],
        }, indent=2, default=str))
        summary[beh] = agg
        logger.info(f"  wrote {per_beh_out}")

    # Cross-behaviour summary
    md_path = args.out_dir / "effect_curves_summary.md"
    write_summary(summary, args, md_path)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for beh, agg in summary.items():
            if isinstance(agg, dict) and "error" not in agg:
                Ls = sorted(agg.keys(), key=lambda x: int(x) if not isinstance(x, int) else x)
                means = [agg[L]["mean_effect"] for L in Ls]
                sems  = [agg[L]["sem_effect"]  for L in Ls]
                ax.errorbar(Ls, means, yerr=sems, marker="o", capsize=3, label=beh)
        ax.axhline(0, color="grey", linestyle="--", alpha=0.4)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Patching effect (1 = full causal influence)")
        ax.set_title(f"Activation patching - {args.model_short}")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(args.out_dir / "effect_curves.png", dpi=120)
        logger.info(f"Plot saved: {args.out_dir / 'effect_curves.png'}")
    except ImportError:
        logger.warning("matplotlib unavailable; skipping plot")

    print(f"\nResults: {args.out_dir}")


def write_summary(summary, args, path):
    lines = [
        f"# Activation patching - {args.model_short}",
        "",
        f"Layers swept: {args.layers}",
        f"Pairs per behaviour: {args.n_pairs}",
        "",
        "## Per-layer mean effect (1 = full causal influence, 0 = no effect)",
        "",
        "| Behaviour | " + " | ".join(f"L{L}" for L in args.layers) + " |",
        "|-----------|" + "|".join(["---"] * len(args.layers)) + "|",
    ]
    for beh, agg in summary.items():
        if not isinstance(agg, dict) or "error" in agg:
            lines.append(f"| {beh} | " + " | ".join(["ERR"] * len(args.layers)) + " |")
            continue
        cells = []
        for L in args.layers:
            v = agg.get(L) or agg.get(str(L))
            if v is None:
                cells.append("-")
            else:
                cells.append(f"{v['mean_effect']:.2f}+/-{v['sem_effect']:.2f}")
        lines.append(f"| {beh} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
