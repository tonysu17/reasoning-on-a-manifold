#!/usr/bin/env python3
"""
Phase 7c — Attribution patching: CAUSAL per-behaviour layer selection.

Why this script exists (METHODOLOGY §5, CONFOUNDS_AND_REMEDIATION CF-10)
-----------------------------------------------------------------------
The steering layer is currently borrowed from Huang (L27) or read off the
*descriptive* PR-trough. Neither answers the causal question: **at which layer
does intervening most change the behaviour?** This script answers it with
attribution patching (Syed et al. 2023; Nanda 2023) — the gradient×(corrupt −
clean) first-order approximation to activation patching — using a *validated,
geometry-based* behaviour metric instead of the lexical "wait/actually" proxy
(the CF-10 fix), with positions aligned across chains at the behaviour onset
(CF-10b).

It fills the slot the triangulation pipeline already reserves but that is
currently MISSING: `results/patching/<model>/pilot_effect_curves.json`
(consumed by `compute_layer_triangulation.load_phase7b_curves`). It also writes
the richer `attribution_curves.json` (per-behaviour, per-layer, with per-pair
and per-position detail).

What it does, per behaviour b:
  1. Build the behaviour metric from the steering geometry
     (`results/steering_vectors/<model>/`): the unit diff-of-means direction
     r_b (``mode=projection``) — optionally centred on mean(OFF) at the read-out
     layer, computed from the saved activations.
  2. Select donor pairs (positive: a sentence labelled b; negative: a matched
     DEDUCTION sentence) — reusing `activation_patching.select_donor_pairs`.
  3. For each pair: anchored alignment at the behaviour-onset token (± a small
     window), one clean forward+backward and one corrupt forward, then
     attribution at every layer 0..L_readout.
  4. Aggregate per-layer across pairs → the causal-effect curve.

Read-out: the metric reads the residual at the **steering layer**
(``--read-layer``, default 27 = where r_b was built and validated). Attribution
then asks which layers' activations most move that read-out; a layer downstream
of the read-out has zero effect to first order, so reading at the last layer
makes all 28 layers' curves meaningful.

⚠️ FIRST-ORDER CAVEAT. Attribution patching is a Taylor approximation; it can
mis-rank layers where the network responds non-linearly to the swap. Treat the
curve as a SCREEN and confirm the chosen layer (+ a couple of neighbours) with
the exact `brute_force_patch_effect` check (``--brute-check`` runs it on the
top-k layers per behaviour).

Requires GPU (forward + backward through the full model). Do NOT run on CPU /
without the corpus. Runtime estimate: ~2 passes per donor pair × 20 pairs × 4
behaviours — a few minutes on DGX Spark (cheaper than 07b's per-layer brute
force, which is #layers forward passes per pair).

Usage (eventual GPU run):
  python 07c_attribution_patching.py --behaviours all --n-pairs 20
  python 07c_attribution_patching.py --behaviours backtracking --brute-check 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.annotation import TARGET_BEHAVIOURS
from src.attribution_patching import (
    BehaviourMetric,
    Alignment,
    anchored_alignment,
    attribution_patching,
    make_residual_metric_fn,
    brute_force_patch_effect,
    aggregate_attribution_curves,
)
from src.activation_patching import select_donor_pairs

# The onset-token locator already lives in 07b; reuse it verbatim so extraction
# and patching place sentences identically (src/text_offsets is the canonical
# rule it follows).
from importlib import import_module
_p7b = import_module("07b_activation_patching")
_sentence_token_onset = _p7b._sentence_token_onset
cache_residuals = _p7b.cache_residuals

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── Metric construction from the steering geometry ───────────────────────────

def build_behaviour_metric(
    steering_dir: Path,
    behaviour: str,
    read_layer: int,
    activations_dir: "Path | None",
    use_center: bool,
) -> BehaviourMetric:
    """Build the validated behaviour metric for `behaviour` at `read_layer`.

    Uses the unit diff-of-means steering direction (``{beh}_single.npy``) as the
    projection axis — the same object Phase-7 steers along, so the causal-layer
    question is self-consistent with the intervention. If ``use_center`` and the
    OFF activations are available, subtracts mean(OFF) at ``read_layer`` so the
    score is the behaviour-specific deviation, not the residual's bulk norm.

    NOTE: the saved vectors are built at the canonical steering layer (L27 in
    metadata). Reading the L27 direction off the L27 residual is the consistent
    choice; ``read_layer`` should normally equal that build layer.
    """
    r = np.load(steering_dir / f"{behaviour}_single.npy").astype(np.float64)
    r = r / (np.linalg.norm(r) + 1e-12)

    center = None
    if use_center and activations_dir is not None:
        # OFF = concatenation of the OTHER behaviours' activations at read_layer
        # (the same OFF definition src/steering.py uses to build r_b).
        off_parts = []
        for other in TARGET_BEHAVIOURS:
            if other == behaviour:
                continue
            p = activations_dir / f"{other}_layer{read_layer}.npy"
            if p.exists():
                off_parts.append(np.load(p).astype(np.float64))
        if off_parts:
            center = np.concatenate(off_parts, axis=0).mean(axis=0)
            logger.info(f"  {behaviour}: centring metric on mean(OFF) "
                        f"at L{read_layer} (n_off={sum(len(x) for x in off_parts)})")
    return BehaviourMetric(direction=r, mode="projection", center=center,
                           behaviour=behaviour)


# ── One donor pair → per-layer attribution ───────────────────────────────────

def attribution_for_pair(
    model, tok, device,
    pos_chain, pos_ann, neg_chain, neg_ann,
    metric: BehaviourMetric,
    layers: list[int],
    read_layer: int,
    window: int,
):
    """Run attribution patching for one (positive, negative) donor pair.

    Clean = positive chain (the behaviour fired); corrupt = negative chain.
    Positions are aligned at the behaviour-onset token of each chain (CF-10b),
    ± `window`. Returns an ``AttributionResult`` or None if the pair is
    unusable (sentence not locatable, too short, etc.).
    """
    import torch
    pos_text = pos_chain.get("chain", "")
    neg_text = neg_chain.get("chain", "")
    pos_ids = tok.encode(pos_text, return_tensors="pt").to(device)
    neg_ids = tok.encode(neg_text, return_tensors="pt").to(device)

    pos_onset = _sentence_token_onset(tok, pos_text, pos_ann.get("text", ""))
    neg_onset = _sentence_token_onset(tok, neg_text, neg_ann.get("text", ""))
    if pos_onset is None or neg_onset is None:
        return None

    Tc, Tk = int(pos_ids.shape[1]), int(neg_ids.shape[1])
    # Read-out at the behaviour onset of the CLEAN chain (the boundary token
    # where b begins), not the last token — the onset is where the behaviour is
    # decided. Clamp into range.
    read_pos = min(max(pos_onset, 0), Tc - 1)
    if not (0 <= neg_onset < Tk):
        return None

    al = anchored_alignment(Tc, Tk, anchor_clean=read_pos,
                            anchor_corrupt=min(neg_onset, Tk - 1), window=window)
    if not al.pairs:
        return None

    metric_fn = make_residual_metric_fn(metric, read_layer=read_layer,
                                        read_position=read_pos)
    res = attribution_patching(model, pos_ids, neg_ids, metric_fn, layers,
                               alignment=al, behaviour=metric.behaviour)
    return res, pos_ids, neg_ids, al, read_pos


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-short", default="R1-1.5B")
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--annotated", type=Path, default=None,
                        help="Annotated chains JSON. Default data/annotated_<model>.json")
    parser.add_argument("--steering-dir", type=Path, default=None,
                        help="Steering vectors dir. Default results/steering_vectors/<model>")
    parser.add_argument("--activations-dir", type=Path, default=None,
                        help="Activations dir for the metric centre. "
                             "Default data/activations/<model>")
    parser.add_argument("--behaviours", nargs="+", default=["backtracking"],
                        help="Subset of behaviours, or 'all'.")
    parser.add_argument("--n-pairs", type=int, default=20)
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Layers to sweep. Default 0..read-layer (all 28).")
    parser.add_argument("--read-layer", type=int, default=27,
                        help="Layer whose residual defines the behaviour score "
                             "(= steering build layer). Default 27.")
    parser.add_argument("--window", type=int, default=2,
                        help="± token window around the behaviour onset to score.")
    parser.add_argument("--no-center", action="store_true",
                        help="Do NOT centre the metric on mean(OFF).")
    parser.add_argument("--brute-check", type=int, default=0,
                        help="If >0, also run the EXACT brute-force patch on the "
                             "top-K attribution layers per behaviour (sanity "
                             "check of the first-order estimate). Adds K×|pairs| "
                             "forward passes per behaviour.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.behaviours == ["all"]:
        args.behaviours = list(TARGET_BEHAVIOURS)
    if args.annotated is None:
        args.annotated = Path(f"data/annotated_{args.model_short}.json")
    if args.steering_dir is None:
        args.steering_dir = Path(f"results/steering_vectors/{args.model_short}")
    if args.activations_dir is None:
        args.activations_dir = Path(f"data/activations/{args.model_short}")
    if args.out_dir is None:
        args.out_dir = Path(f"results/patching/{args.model_short}")
    if args.layers is None:
        args.layers = list(range(args.read_layer + 1))   # 0..read_layer inclusive
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.annotated.exists():
        logger.error(f"Annotated chains not found: {args.annotated}")
        sys.exit(1)
    if not args.steering_dir.exists():
        logger.error(f"Steering vectors not found: {args.steering_dir}")
        sys.exit(1)

    with open(args.annotated) as f:
        annotations = json.load(f)
    logger.info(f"Loaded {len(annotations)} annotated chains")

    model, tok = _p7b.load_model_and_tokenizer(args.model_id, args.dtype)
    import torch
    device = next(model.parameters()).device

    activations_dir = args.activations_dir if args.activations_dir.exists() else None
    if activations_dir is None and not args.no_center:
        logger.warning(f"Activations dir {args.activations_dir} absent — metric "
                       f"will NOT be centred (projection on raw residual).")

    summary = {}            # for attribution_curves.json
    pilot_agg = {}          # for pilot_effect_curves.json (triangulation slot)

    for beh in args.behaviours:
        logger.info(f"=== {beh} ===")
        metric = build_behaviour_metric(
            args.steering_dir, beh, args.read_layer, activations_dir,
            use_center=not args.no_center)

        pairs = select_donor_pairs(annotations, beh, n_pairs=args.n_pairs,
                                   random_state=args.seed)
        if not pairs:
            logger.warning(f"No donor pairs for {beh}; skipping")
            summary[beh] = {"error": "no_donor_pairs"}
            continue
        logger.info(f"Selected {len(pairs)} donor pairs")

        per_pair_layer: list[dict[int, float]] = []
        trials = []
        for i, ((pos_chain, pos_ann), (neg_chain, neg_ann)) in enumerate(pairs):
            t0 = time.time()
            try:
                out = attribution_for_pair(
                    model, tok, device, pos_chain, pos_ann, neg_chain, neg_ann,
                    metric, args.layers, args.read_layer, args.window)
                if out is None:
                    logger.warning(f"  pair {i}: unusable (onset not located); skip")
                    continue
                res, _, _, al, read_pos = out
                per_pair_layer.append(dict(res.per_layer))
                trials.append({
                    "positive_chain_id": pos_chain.get("task_id", ""),
                    "negative_chain_id": neg_chain.get("task_id", ""),
                    "read_position": int(read_pos),
                    "n_aligned_positions": len(al.pairs),
                    "per_layer": {int(L): float(v) for L, v in res.per_layer.items()},
                })
                if (i + 1) % 5 == 0:
                    logger.info(f"  pair {i+1}/{len(pairs)} ({time.time()-t0:.1f}s)")
            except Exception as e:
                logger.warning(f"  pair {i}: {type(e).__name__}: {e}")

        agg = aggregate_attribution_curves(per_pair_layer)

        # Optional exact check on the top-K layers (first-order sanity).
        brute = None
        if args.brute_check > 0 and per_pair_layer:
            ranked = sorted(agg.items(),
                            key=lambda kv: kv[1]["mean_effect"], reverse=True)
            top_layers = [L for L, _ in ranked[: args.brute_check]]
            brute = run_brute_check(model, tok, device, pairs, metric,
                                    top_layers, args.read_layer, args.window)
            logger.info(f"  brute-force check on layers {top_layers}: {brute}")

        summary[beh] = {
            "behaviour": beh,
            "read_layer": args.read_layer,
            "n_pairs": len(per_pair_layer),
            "layer_effect": {int(L): v for L, v in agg.items()},
            "trials": trials,
            "brute_force_check": brute,
        }
        pilot_agg[beh] = {
            "behaviour": beh,
            "n_pairs": len(per_pair_layer),
            "layer_effect": {str(L): v for L, v in agg.items()},
        }
        per_beh_out = args.out_dir / f"attribution_curves_{beh}.json"
        per_beh_out.write_text(json.dumps(summary[beh], indent=2, default=str))
        logger.info(f"  wrote {per_beh_out}")

    # Canonical combined output.
    (args.out_dir / "attribution_curves.json").write_text(
        json.dumps(summary, indent=2, default=str))
    # Triangulation slot (the empty one CF-10 / METHODOLOGY §5 flag).
    (args.out_dir / "pilot_effect_curves.json").write_text(
        json.dumps(pilot_agg, indent=2, default=str))

    write_summary_md(summary, args, args.out_dir / "attribution_summary.md")
    _maybe_plot(summary, args)

    print(f"\nResults: {args.out_dir}")
    print(f"  attribution_curves.json     (canonical)")
    print(f"  pilot_effect_curves.json    (triangulation consumer)")


def run_brute_check(model, tok, device, pairs, metric, layers,
                    read_layer, window):
    """Exact activation-patching effect on `layers`, averaged across pairs —
    the first-order sanity check the docstring recommends."""
    import torch
    per_pair = []
    for (pos_chain, pos_ann), (neg_chain, neg_ann) in pairs:
        pos_text, neg_text = pos_chain.get("chain", ""), neg_chain.get("chain", "")
        pos_ids = tok.encode(pos_text, return_tensors="pt").to(device)
        neg_ids = tok.encode(neg_text, return_tensors="pt").to(device)
        po = _sentence_token_onset(tok, pos_text, pos_ann.get("text", ""))
        no = _sentence_token_onset(tok, neg_text, neg_ann.get("text", ""))
        if po is None or no is None:
            continue
        Tc, Tk = int(pos_ids.shape[1]), int(neg_ids.shape[1])
        read_pos = min(max(po, 0), Tc - 1)
        al = anchored_alignment(Tc, Tk, read_pos, min(no, Tk - 1), window=window)
        if not al.pairs:
            continue
        bf = brute_force_patch_effect(model, pos_ids, neg_ids, metric, layers,
                                      alignment=al, read_layer=read_layer,
                                      read_position=read_pos)
        # reduce to sum|·| over positions per layer (same reduction as attribution)
        per_pair.append({L: float(sum(abs(v) for v in bf[L].values())) for L in layers})
    return aggregate_attribution_curves(per_pair)


def write_summary_md(summary, args, path):
    lines = [
        f"# Attribution patching — {args.model_short}",
        "",
        f"Read-out layer (metric): L{args.read_layer}. Layers swept: "
        f"{args.layers[0]}..{args.layers[-1]}. Pairs/behaviour: {args.n_pairs}. "
        f"Onset window: ±{args.window}.",
        "",
        "Metric = projection of the residual onto the behaviour's unit "
        "diff-of-means steering direction (CF-10 fix; not lexical markers); "
        "positions aligned at the behaviour onset across chains (CF-10b).",
        "",
        "⚠️ First-order estimate — confirm the peak layer with `--brute-check`.",
        "",
        "## Per-layer mean attribution (higher = more causal influence)",
        "",
    ]
    # Build a compact table over a readable layer subset.
    show = sorted({0, 3, 7, 10, 14, 17, 21, 24, args.read_layer}
                  & set(args.layers))
    lines.append("| Behaviour | argmax L | " + " | ".join(f"L{L}" for L in show) + " |")
    lines.append("|---|---|" + "|".join(["---"] * len(show)) + "|")
    for beh, d in summary.items():
        if not isinstance(d, dict) or "error" in d:
            lines.append(f"| {beh} | ERR | " + " | ".join(["—"] * len(show)) + " |")
            continue
        le = d["layer_effect"]
        if not le:
            continue
        argmax_L = max(le, key=lambda L: le[L]["mean_effect"])
        cells = []
        for L in show:
            v = le.get(L) or le.get(str(L))
            cells.append(f"{v['mean_effect']:.3g}" if v else "—")
        lines.append(f"| {beh} | **{argmax_L}** | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines))


def _maybe_plot(summary, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable; skipping plot")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for beh, d in summary.items():
        if not isinstance(d, dict) or "error" in d or not d.get("layer_effect"):
            continue
        le = d["layer_effect"]
        Ls = sorted(int(L) for L in le)
        means = [le[L]["mean_effect"] for L in Ls]
        sems = [le[L]["sem_effect"] for L in Ls]
        ax.errorbar(Ls, means, yerr=sems, marker="o", ms=3, capsize=2, label=beh)
    ax.axhline(0, color="grey", ls="--", alpha=0.4)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Attribution effect (|grad·Δ|, summed over onset window)")
    ax.set_title(f"Attribution patching — {args.model_short} (read-out L{args.read_layer})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out_dir / "attribution_curves.png", dpi=120)
    logger.info(f"Plot saved: {args.out_dir / 'attribution_curves.png'}")


if __name__ == "__main__":
    main()
