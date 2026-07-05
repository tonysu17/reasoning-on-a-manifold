#!/usr/bin/env python3
"""
Phase 7e — FAITHFUL Venhoff per-behaviour steering-LAYER attribution on R1-1.5B.

What this is (and is NOT)
-------------------------
A deliberately literal reproduction of Venhoff et al.'s released layer-attribution
(arXiv:2506.18167; github.com/cvenhoff/steering-thinking-llms), wired to OUR
R1-Distill-Qwen-1.5B corpus, so the published per-behaviour layer table can be
reproduced in-stack and compared like-for-like with OUR two methods (07c
attribution patching, 07d forward-intervention sweep). It is NOT a new method —
faithfulness over cleanliness; every quirk of the original is preserved (see
``src/venhoff_attribution.py`` for the inline flags).

The algorithm (per behaviour b), exactly as the released code:

  1.  **Feature vector** ``u_ℓ = mean(b rows at ℓ) − μ_overall_ℓ`` where
      ``μ_overall_ℓ`` = mean over ALL labels' rows at ℓ (Venhoff's "overall"
      mean), re-normalised to UNIT before the dot product. Built once over all 28
      layers from ``data/activations/<model>/{label}_layer{ℓ}.npy``. [change (c)
      vs our 07c/07d, which use μ_b − μ_other-three]

  2.  **Read-out** = LM-head logits KL-with-detached-copy at position ``start-1``
      (Venhoff ``compute_kl_divergence_metric(logits[0, start-1:start].mean(0))``)
      — the value ≈ 0; its GRADIENT is the attribution signal. Read at the OUTPUT,
      never an intermediate layer. [change (a)]

  3.  **Effect** ``effect(ℓ) = | unit(u_ℓ) · ∂KL/∂a_ℓ[start-1] |`` from ONE
      forward + ONE backward per scored span (no corrupt chain, no alignment).
      [change (b)]

  4.  **Cross-label span selection (the non-obvious quirk):** the curve for b is
      scored over every annotated span that is **NOT** b (Venhoff's ``!= label``
      filter). The feature vector is u^b; the positions are the not-b spans.

  5.  Accumulate ``|u·grad|`` over the (not-b) spans, divide by the span count,
      then average over examples → a per-behaviour per-layer curve. argmax = the
      chosen layer. Expect (paper Table 2 / steering_config):
      backtracking→17, uncertainty-estimation→18, example-testing→15,
      adding-knowledge→18.

n_examples (spec §8)
--------------------
``--n-examples`` default **500** — the compute run that produced the published
layers used the full dataset (paper: "500 tasks"; the example/help command in the
original also says 500). The committed ``run.sh`` value of 50 is a viz-only
convenience (``--only_viz`` re-plots saved ``.pt``) and is NOT the scoring N — do
not infer 50. ``argparse`` default in the original is 10.

Deviations from the original (documented; see also the lib docstring)
--------------------------------------------------------------------
  * **No nnsight.** The original captures grads via ``nnsight`` (``model.trace()``,
    ``lm_head.output.save()``, ``layers[ℓ].output[0].grad``). We use plain PyTorch
    forward hooks + ``retain_grad`` + ``model(...).logits`` (spec hard-req 5),
    which yields the identical post-block residual gradient.
  * **"overall" mean source.** The original builds ``overall`` from the raw
    ``min_pos:max_pos`` thinking span; we reconstruct it as the row-mean over ALL
    labels' extracted ``*_layer{ℓ}.npy`` matrices (the same rows our steering
    vectors use), since the raw-span activations are not separately cached. This
    is inert for layer-SELECTION up to the row weighting (the attribution score
    unit-normalises u_ℓ regardless).
  * **Annotated spans.** Our annotated file stores spans as ``{"label","text"}``
    dicts (already parsed from the same ``["label"]…["end-section"]`` delimiter
    format Venhoff annotates with — see ``src/annotation.py``). We map them to
    token ``(start, end)`` with the IDENTICAL char→token rule
    (``label_positions_from_spans`` == Venhoff's ``get_label_positions``).
  * **Annotator.** Our annotations come from the project's Sonnet pass (the
    repo's canonical ``data/annotated_R1-1.5B.json``), not Venhoff's gpt-4.1.
    The attribution math is annotator-agnostic; only the spans differ.
  * **Long-chain cropping** (``--context-window``, default 1024): our chains are
    ~8k tokens, so a span deep in a chain would force an 8k-token fwd+bwd.
    Cropping to the last W tokens before the span ``end`` is EXACT for the
    read-out (autoregressive causality: logits at start-1 depend only on tokens
    < start) whenever the full prefix fits (start ≤ W), and a standard
    local-context approximation otherwise. Set <0 to use the full chain
    (byte-faithful but slow).

Outputs to ``results/venhoff_attribution/<model>/``:
  layer_effects.json   per-behaviour full per-layer curve (mean ± SEM ± n) +
                       point argmax + per-example curves + the published layer
  summary.md           per-behaviour argmax vs published 17/18/15/18 + curve table
  layer_effects.png    (optional, if matplotlib present)

Requires GPU. One fwd+bwd per scored span (cheaper than 07c's two passes; far
cheaper than 07d's (1+R) forwards/layer), but the backward runs through the
unembedding so peak memory is higher than 07c.

Usage (eventual GPU run):
  python 07e_venhoff_layer_attribution.py --behaviours all --n-examples 500
  python 07e_venhoff_layer_attribution.py --behaviours backtracking --n-examples 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.annotation import TARGET_BEHAVIOURS, VALID_LABELS
from src.venhoff_attribution import (
    overall_mean_per_layer,
    behaviour_overall_directions,
    label_positions_from_spans,
    cross_label_positions,
    venhoff_attribution_single,
    aggregate_attribution_curves,
    argmax_layer,
    PUBLISHED_LAYERS,
)

# Reuse 07b's model loader VERBATIM so the model/tokenizer match extraction.
_p7b = import_module("07b_activation_patching")
load_model_and_tokenizer = _p7b.load_model_and_tokenizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# Venhoff's ``labels`` come from the steering_config keys (the 4 behaviours with a
# published layer), in that order — NOT alphabetical (spec §8).
VENHOFF_LABELS = list(TARGET_BEHAVIOURS)   # backtracking, uncertainty-estimation,
#                                            example-testing, adding-knowledge


def _crop_to_window(input_ids, start: int, end: int, context_window):
    """Crop ``input_ids[:, :end]`` to a local window and remap ``(start, end)``.

    Venhoff forwards on ``input_ids[:, :end]`` and reads/grads at ``start-1``.
    Keeping only the last ``W`` tokens before ``end`` is EXACT for the read-out
    when the full prefix fits (``start <= W``): the logits at the remapped
    ``start-1`` depend only on tokens at positions ``< start`` (autoregressive),
    none of which are dropped. When ``start > W`` it is the standard
    local-context approximation. Returns ``(ids_local, start_local, end_local)``.

    ``context_window=None`` ⇒ no crop (byte-faithful full forward to ``end``).
    """
    end = int(min(end, int(input_ids.shape[-1])))
    if context_window is None:
        return input_ids[:, :end], int(start), end
    lo = max(0, int(start) - int(context_window))
    ids_local = input_ids[:, lo:end]
    return ids_local, int(start) - lo, end - lo


def attribute_for_example(model, tok, device, chain, *, behaviour, directions,
                          layers, context_window):
    """Venhoff attribution for ONE example chain, scored over its not-``behaviour``
    spans (the ``!= label`` quirk). Returns the per-layer ``{ℓ: |u·grad|}`` curve
    (a dict) or ``None`` if the chain has no not-``behaviour`` spans / cannot be
    located.
    """
    import torch

    text = chain.get("chain", "")
    spans = chain.get("annotations", [])
    if not text or not spans:
        return None

    # Map ALL annotated spans → token (start, end) with Venhoff's char→token rule.
    spans_by_cat = label_positions_from_spans(spans, text, tok)
    # The curve for ``behaviour`` is scored on the spans of all OTHER labels (§6c).
    positions = cross_label_positions(spans_by_cat, behaviour)
    if not positions:
        return None

    ids = tok.encode(text, return_tensors="pt").to(device)
    T = int(ids.shape[-1])

    # Crop each span's forward to a local window (exact for the read-out; see
    # _crop_to_window). Venhoff scores each span with its OWN truncated forward
    # (to ``end``), so we crop per span and accumulate the per-span effects here
    # rather than inside the lib (which assumes one shared id tensor). To stay
    # faithful AND tractable we therefore call the lib once per span with the
    # cropped ids and average the per-span curves with the same 1/len(positions)
    # divisor Venhoff uses.
    acc = {int(L): 0.0 for L in layers}
    n_scored = 0
    for (start, end) in positions:
        if not (1 <= start < T):
            # No prefix / out of range — still counts toward the divisor (Venhoff
            # divides by the full span count); contributes 0.
            n_scored += 1
            continue
        ids_local, start_l, end_l = _crop_to_window(ids, start, end, context_window)
        res = venhoff_attribution_single(
            model, ids_local, [(start_l, end_l)], directions, layers,
            behaviour=behaviour,
        )
        n_scored += 1
        if res is None:
            continue
        for L in layers:
            v = res.per_layer.get(int(L))
            if v is not None and np.isfinite(v):
                acc[int(L)] += float(v)
    if n_scored == 0:
        return None
    # Divide by the span count (Venhoff: patching_effects[ℓ] / len(label_positions)).
    return {int(L): acc[int(L)] / n_scored for L in layers}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-short", default="R1-1.5B")
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Venhoff loads R1-1.5B in bfloat16 (spec). Default bfloat16.")
    parser.add_argument("--annotated", type=Path, default=None,
                        help="Annotated chains JSON. Default data/annotated_<model>.json")
    parser.add_argument("--activations-dir", type=Path, default=None,
                        help="All-layer activations dir (for u_ℓ). "
                             "Default data/activations/<model>")
    parser.add_argument("--behaviours", nargs="+", default=["all"],
                        help="Subset of the 4 behaviours, or 'all' (default).")
    parser.add_argument("--n-examples", type=int, default=500,
                        help="Examples (chains) scored per behaviour, results[:N]. "
                             "Default 500 (the published compute setting; spec §8). "
                             "The committed run.sh uses 50 for VIZ-ONLY — that is "
                             "NOT the scoring N.")
    parser.add_argument("--n-layers", type=int, default=28,
                        help="Number of decoder layers (R1-1.5B = 28). u_ℓ and the "
                             "attribution are built over layers 0..n_layers-1.")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Explicit layers to score (overrides 0..n_layers-1). "
                             "Default: ALL layers (Venhoff scores every layer and "
                             "takes the argmax over the full curve).")
    parser.add_argument("--context-window", type=int, default=1024,
                        help="Crop each scored span's forward to the last W tokens "
                             "before the span end (tractability on ~8k-token "
                             "chains). EXACT for the read-out when start ≤ W "
                             "(autoregressive), local-context approximation "
                             "otherwise. Set <0 to use the full chain "
                             "(byte-faithful, slow).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default results/venhoff_attribution/<model>")
    parser.add_argument("--seed", type=int, default=42,
                        help="Unused for scoring (Venhoff scans results[:N] in "
                             "order, no sampling); kept for parity / reproducibility.")
    args = parser.parse_args()

    if args.behaviours == ["all"]:
        args.behaviours = list(VENHOFF_LABELS)
    if args.annotated is None:
        args.annotated = Path(f"data/annotated_{args.model_short}.json")
    if args.activations_dir is None:
        args.activations_dir = Path(f"data/activations/{args.model_short}")
    if args.out_dir is None:
        args.out_dir = Path(f"results/venhoff_attribution/{args.model_short}")
    layers = (list(args.layers) if args.layers is not None
              else list(range(args.n_layers)))
    context_window = None if args.context_window < 0 else int(args.context_window)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.annotated.exists():
        logger.error(f"Annotated chains not found: {args.annotated}")
        sys.exit(1)
    if not args.activations_dir.exists():
        logger.error(f"Activations dir not found: {args.activations_dir} "
                     f"(u_ℓ needs {{label}}_layer{{ℓ}}.npy)")
        sys.exit(1)

    with open(args.annotated) as f:
        annotations = json.load(f)
    logger.info(f"Loaded {len(annotations)} annotated chains "
                f"(scoring the first {args.n_examples} per behaviour)")

    # The "overall" mean uses ALL labels present (the 6-label framework), not just
    # the 4 target behaviours — Venhoff's overall is the whole annotated thinking
    # region across every label. Use all VALID_LABELS whose files exist.
    all_labels = sorted(VALID_LABELS)
    logger.info(f"Building overall mean over labels {all_labels} for {len(layers)} layers")
    ctx_desc = (f"ctx-window={context_window}" if context_window is not None
                else "ctx-window=full")
    logger.info(f"Layers {layers[0]}..{layers[-1]} | dtype={args.dtype} | "
                f"n-examples={args.n_examples} | {ctx_desc}")

    model, tok = load_model_and_tokenizer(args.model_id, args.dtype)
    if not getattr(tok, "is_fast", False):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
        logger.info("Reloaded FAST tokenizer (char->token offset_mapping needs a fast tokenizer).")
    device = next(model.parameters()).device

    # μ_overall_ℓ computed ONCE (behaviour-independent), reused across behaviours.
    overall_means = overall_mean_per_layer(args.activations_dir, all_labels, layers)
    if not overall_means:
        logger.error("No overall mean could be built (no activation files?). Abort.")
        sys.exit(1)

    summary = {}
    for beh in args.behaviours:
        logger.info(f"=== {beh} ===")
        directions = behaviour_overall_directions(
            args.activations_dir, beh, layers, all_labels,
            overall_means=overall_means)
        present = sorted(directions.keys())
        if not present:
            logger.warning(f"No per-layer u_ℓ for {beh}; skipping")
            summary[beh] = {"error": "no_directions"}
            continue
        if len(present) < len(layers):
            missing = sorted(set(layers) - set(present))
            logger.warning(f"  {beh}: missing u_ℓ at layers {missing}; "
                           f"scoring {len(present)} layers")

        per_example_curves: list[dict] = []
        n_used = 0
        t0 = time.time()
        for i, chain in enumerate(annotations[:args.n_examples]):
            try:
                curve = attribute_for_example(
                    model, tok, device, chain, behaviour=beh,
                    directions=directions, layers=present,
                    context_window=context_window)
            except Exception as e:               # noqa: BLE001 — keep going on bad chains
                logger.warning(f"  example {i}: {type(e).__name__}: {e}")
                continue
            if curve is None:
                continue                          # no not-beh spans / unlocatable
            per_example_curves.append(curve)
            n_used += 1
            if n_used % 25 == 0:
                logger.info(f"  {beh}: {n_used} examples scored "
                            f"({time.time()-t0:.1f}s)")

        if not per_example_curves:
            logger.warning(f"  {beh}: no scorable examples; skipping")
            summary[beh] = {"error": "no_examples"}
            continue

        # Example-averaged per-layer curve (mean ± SEM ± n; NaNs skipped — the
        # finite-mean equivalent of Venhoff's nan_to_num→nanmean).
        agg = aggregate_attribution_curves(per_example_curves)
        peak = argmax_layer(agg)
        published = PUBLISHED_LAYERS.get(beh)
        logger.info(f"  {beh}: argmax layer = {peak}  (published = {published}; "
                    f"n_examples used = {n_used})")

        summary[beh] = {
            "behaviour": beh,
            "layers": present,
            "n_examples_scored": n_used,
            "context_window": context_window,
            "argmax_layer": peak,
            "published_layer": published,
            "matches_published": (peak == published) if published is not None else None,
            "layer_effect": {int(L): agg[L] for L in agg},
            "per_example_curves": [
                {int(L): float(c[L]) for L in c} for c in per_example_curves
            ],
        }
        per_beh_out = args.out_dir / f"layer_effects_{beh}.json"
        per_beh_out.write_text(json.dumps(summary[beh], indent=2, default=str))
        logger.info(f"  wrote {per_beh_out}")

    (args.out_dir / "layer_effects.json").write_text(
        json.dumps(summary, indent=2, default=str))
    write_summary_md(summary, args, layers, args.out_dir / "summary.md")
    _maybe_plot(summary, args)

    # ── Console report: argmax per behaviour vs the published 17/18/15/18 ──
    print(f"\nResults: {args.out_dir}")
    print("  layer_effects.json   (per-behaviour curve + argmax + per-example)")
    print("  summary.md           (argmax vs published 17/18/15/18 + curve table)")
    print("\nVenhoff per-behaviour argmax vs PUBLISHED (17/18/15/18):")
    print(f"  {'behaviour':>24}  {'argmax':>6}  {'published':>9}  match")
    for beh in VENHOFF_LABELS:
        d = summary.get(beh)
        if not isinstance(d, dict) or "argmax_layer" not in d:
            err = d.get("error", "?") if isinstance(d, dict) else "not-run"
            print(f"  {beh:>24}  {'ERR':>6}  "
                  f"{str(PUBLISHED_LAYERS.get(beh)):>9}  ({err})")
            continue
        mark = "✓" if d.get("matches_published") else "✗"
        print(f"  {beh:>24}  {str(d['argmax_layer']):>6}  "
              f"{str(d.get('published_layer')):>9}  {mark}")


def write_summary_md(summary, args, layers, path):
    lines = [
        f"# Venhoff per-behaviour layer-attribution — {args.model_short}",
        "",
        "FAITHFUL reproduction of Venhoff et al. (arXiv:2506.18167) released "
        "layer-attribution code, wired to our R1-Distill-Qwen-1.5B corpus.",
        "",
        f"- Feature `u_ℓ = mean(b rows at ℓ) − μ_overall_ℓ` (overall = all-label "
        f"row mean), **unit-normed** before the dot product.",
        f"- Read-out = LM-head KL-with-detached-copy at the pre-onset token "
        f"`start-1` (value ≈ 0; its GRADIENT is the signal). Read at the OUTPUT.",
        f"- `effect(ℓ) = |unit(u_ℓ) · ∂KL/∂a_ℓ[start-1]|`, one fwd+bwd per span, "
        f"accumulated over the **not-`b`** spans (Venhoff `!= label` quirk), "
        f"÷ span count, averaged over examples.",
        f"- Layers scored: {layers[0]}..{layers[-1]} ({len(layers)}). "
        f"n_examples = {args.n_examples}. dtype = {args.dtype}. "
        f"context-window = {args.context_window}.",
        "",
        "## Per-behaviour argmax vs published (paper Table 2 / steering_config)",
        "",
        "| Behaviour | argmax ℓ | published ℓ | match | n examples |",
        "|---|---|---|---|---|",
    ]
    for beh in VENHOFF_LABELS:
        d = summary.get(beh)
        pub = PUBLISHED_LAYERS.get(beh)
        if not isinstance(d, dict) or "argmax_layer" not in d:
            err = d.get("error", "?") if isinstance(d, dict) else "not-run"
            lines.append(f"| {beh} | ERR ({err}) | {pub} | — | — |")
            continue
        mark = "✓" if d.get("matches_published") else "✗"
        lines.append(f"| {beh} | **{d['argmax_layer']}** | {pub} | {mark} | "
                     f"{d.get('n_examples_scored')} |")

    # Compact per-layer mean-|u·grad| table (≤ ~10 evenly spaced layers).
    show = sorted({int(L) for d in summary.values()
                   if isinstance(d, dict) for L in d.get("layer_effect", {})})
    if len(show) > 10:
        idx = np.linspace(0, len(show) - 1, 10).round().astype(int)
        show = [show[i] for i in sorted(set(idx))]
    if show:
        lines += ["", "## Per-layer mean |u·grad| (± SEM)", "",
                  "| Behaviour | " + " | ".join(f"L{L}" for L in show) + " |",
                  "|---|" + "|".join(["---"] * len(show)) + "|"]
        for beh in VENHOFF_LABELS:
            d = summary.get(beh)
            if not isinstance(d, dict) or "layer_effect" not in d:
                continue
            le = d["layer_effect"]
            cells = []
            for L in show:
                v = le.get(L) or le.get(str(L))
                if not isinstance(v, dict):
                    cells.append("—")
                else:
                    m = v.get("mean_effect")
                    s = v.get("sem_effect") or 0.0
                    cells.append("—" if m is None else f"{m:.3g}±{s:.2g}")
            # Mark the argmax cell.
            am = d.get("argmax_layer")
            row = []
            for L, c in zip(show, cells):
                row.append(f"**{c}**" if L == am else c)
            lines.append(f"| {beh} | " + " | ".join(row) + " |")

    lines += [
        "",
        "Full per-layer {mean_effect, sem_effect, n} and per-example curves are in "
        "`layer_effects.json` (and per-behaviour `layer_effects_<b>.json`).",
        "",
        "Deviations from the original (all documented in the script header): no "
        "nnsight (PyTorch hooks + retain_grad give the same residual gradient); "
        "`overall` reconstructed as the all-label row mean from the extracted "
        "`*_layer{ℓ}.npy`; annotations are the project's Sonnet pass, not gpt-4.1; "
        "long chains cropped to a local window (exact for the read-out when the "
        "prefix fits).",
    ]
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
    plotted = False
    for beh, d in summary.items():
        if not isinstance(d, dict) or "layer_effect" not in d:
            continue
        le = d["layer_effect"]
        Ls = sorted(int(L) for L in le)
        means, sems = [], []
        for L in Ls:
            v = le.get(L) or le.get(str(L))
            means.append(v.get("mean_effect") if isinstance(v, dict) else None)
            sems.append((v.get("sem_effect") or 0.0) if isinstance(v, dict) else 0.0)
        line = ax.errorbar(Ls, means, yerr=sems, marker="o", ms=3, capsize=2, label=beh)
        # Star the argmax; vertical line at the published layer (same colour).
        am = d.get("argmax_layer")
        if am is not None:
            ax.scatter([am], [le.get(am, le.get(str(am), {})).get("mean_effect")],
                       marker="*", s=160, zorder=5,
                       color=line[0].get_color(), edgecolor="k", linewidth=0.4)
        pub = d.get("published_layer")
        if pub is not None:
            ax.axvline(pub, color=line[0].get_color(), ls=":", alpha=0.5)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("mean |unit(u_ℓ) · ∂KL/∂a_ℓ[start-1]|")
    ax.set_title(f"Venhoff layer-attribution — {args.model_short} "
                 f"(★ = argmax; : = published)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out_dir / "layer_effects.png", dpi=120)
    logger.info(f"Plot saved: {args.out_dir / 'layer_effects.png'}")


if __name__ == "__main__":
    main()
