#!/usr/bin/env python3
"""
Phase 7d — Per-layer STEERING-EFFECT sweep: causal per-behaviour layer selection
by FORWARD-PASS INTERVENTION (the de-confounded replacement for 07c attribution
patching).

Why this script exists (METHODOLOGY §5, CONFOUNDS_AND_REMEDIATION CF-10)
-----------------------------------------------------------------------
`07c_attribution_patching.py` chose the steering layer by FIRST-ORDER attribution
patching, reading a behaviour metric at a FIXED late layer (L27). It came back
CONFOUNDED: all four behaviours ramp monotonically to L26–27 — a read-out-
proximity artefact of a *linear* gradient that cannot see amplification (see
`results/patching/R1-1.5B/attribution_summary.md`). A first-order gradient at a
fixed late read-out has no interior peak to find.

This script measures the ACTUAL non-linear per-layer steering effect by FORWARD
intervention, not a gradient, and (per an adversarial review) makes the result
*interpretable* with five de-confounding controls. Per behaviour b:

  1.  **Per-layer direction v_ℓ** — the unit diff-of-means direction for b built
      AT each layer ℓ from the already-extracted all-layer activations
      (`data/activations/<model>/{b}_layer{ℓ}.npy`; ON = b, OFF = the other
      three behaviours). Reuses `src.steering.single_direction_vector`. CPU.

  2.  **Output read-out — KL (primary) + onset-logprob (secondary).** For a
      positive donor where b fired at onset token t*, the PRIMARY read-out is
      KL(p_steered ‖ p_baseline) over the FULL next-token distribution at t*−1
      (captures probability mass moved onto *synonym* onset tokens the single-
      token read-out misses). The SECONDARY diagnostic is the teacher-forced
      log-prob of the donor's REAL onset token (CF-10a; no lexical marker list).

  3.  **Per-layer norm-matched RANDOM-DIRECTION null (MUST-FIX #1).** For each ℓ,
      besides v_ℓ, run R norm-matched random directions (default R=3,
      deterministically seeded per (b,ℓ,r)) identically. The reported, argmax-
      bearing quantity is the **de-confounded effect = Score_b(ℓ) −
      mean_r Score_random(ℓ)** — subtracts any global sensitivity of a layer to
      ANY perturbation (late layers near the logits) and isolates the behaviour-
      specific causal effect.

  4.  **Per-layer α normalization (MUST-FIX #2).** The same nominal α is NOT the
      same intervention strength across depth (residual norm grows with depth);
      the injected delta is scaled to a fixed fraction (`--delta-frac`, default
      0.1) of the median ‖h‖ at that layer over the steered prefix, applied
      IDENTICALLY to v_ℓ and the random null.

  5.  **Bootstrap the argmax → a SHORTLIST (MUST-FIX #4).** Resample donors
      (`--n-boot`, default 1000), recompute the de-confounded argmax each
      resample, and report a SHORTLIST = layers selected in ≥ `--shortlist-frac`
      of resamples OR within 1 SEM of the max (2–4 candidates), framed as a
      PRE-FILTER for Phase 7 to confirm — NOT a single load-bearing argmax.

  6.  **Sweep ℓ** over a configurable range; the default `--start-layer` skips
      ~the first 20% of depth (≈L5 for 28 layers — Venhoff "ignore early
      embedding-correlated layers", MUST-FIX #5). `--drop-embed-cos` additionally
      drops layers whose v_ℓ is highly cosine-similar to the input embeddings.

Why forward intervention + random null de-confound proximity
------------------------------------------------------------
Attribution reads at a fixed late layer and linearises, so its score is dominated
by ∂(late metric)/∂(act at ℓ), mechanically larger near the read-out. Here the
read-out is the OUTPUT (common to every ℓ) and the swap runs through the full
non-linear stack, so there is no fixed-read-out proximity term. The random-
direction null is the second line of defence: it removes any *residual* global-
sensitivity term that survives the output read-out, so the de-confounded curve is
behaviour-specific.

Caveats (also printed in the summary)
-------------------------------------
  * The logprob read-out is token-anchored; the KL read-out is the primary,
    behaviour-agnostic measure to soften this (KL is unsigned — magnitude of
    redistribution).
  * α / delta_frac and the suppression window (prefix ≤ t*−1) are choices.
  * Forward-only: ~(1 + R) forwards per swept layer per donor + one baseline pass
    for the per-layer norms. No backward pass.

Outputs to `results/patching/<model>/`:
  steering_effect_curves.json    (per-behaviour per-layer {behaviour_effect,
                                  random_null, deconfounded_effect, kl_effect,
                                  logprob_effect, sem, n} + bootstrap + per-donor)
  steering_effect_summary.md     (per-behaviour SHORTLIST + de-confounded table)
  steering_effect_curves.png     (optional, if matplotlib present)

Also writes `pilot_effect_curves.json` ONLY with --write-pilot (the de-confounded
curve, mean_effect = deconfounded_effect, feeds the triangulation consumer).

Requires GPU. Do NOT run on CPU / without the corpus. Forward-only.

Usage (eventual GPU run):
  python 07d_layer_steering_sweep.py --behaviours all --n-donors 20
  python 07d_layer_steering_sweep.py --behaviours backtracking --start-layer 5 --amplify
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

from src.annotation import TARGET_BEHAVIOURS
from src.activation_patching import select_donor_pairs
from src.layer_sweep import (
    per_layer_directions,
    layer_steering_sweep_single,
    truncate_to_window,
    aggregate_sweep_curves,
    aggregate_full_curves,
    argmax_layer,
    bootstrap_shortlist,
    embedding_cosine,
    ambient_covariance,
    covariance_matched_unit_directions,
)

# Reuse 07b's model loader + onset locator VERBATIM so extraction and steering
# place sentences identically (src/text_offsets is the canonical rule).
_p7b = import_module("07b_activation_patching")
load_model_and_tokenizer = _p7b.load_model_and_tokenizer
_sentence_token_onset = _p7b._sentence_token_onset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── Positive-donor selection ─────────────────────────────────────────────────

def select_positive_donors(annotations: list, behaviour: str, n: int,
                           seed: int) -> list:
    """Positive donors only: (chain, annotation) where the chain emits
    ``behaviour``.

    The forward-intervention read-out is the positive donor's OWN onset-token
    log-prob, so — unlike 07b/07c — no negative/corrupt chain is needed. We reuse
    ``select_donor_pairs`` (which also matches a negative) and keep the positive
    halves, so donor selection is identical to 07c's (same chains, same seed) for
    a like-for-like layer comparison.
    """
    pairs = select_donor_pairs(annotations, behaviour, n_pairs=n, random_state=seed)
    return [pos for (pos, _neg) in pairs]


# ── One donor → per-layer steering-effect curve ──────────────────────────────

def sweep_for_donor(model, tok, device, chain, ann, directions, layers,
                    alpha, amplify, span_tokens, *, behaviour, readout,
                    n_random, delta_frac, context_window=None,
                    null_directions=None):
    """Run the forward-intervention sweep for one positive donor chain.

    Locates the behaviour-onset token t* in the chain (same locator as 07b/07c),
    requires t* ≥ 1 (the read-out predicts t* from the prefix), and returns a
    ``SweepResult`` (de-confounded with the per-layer norm-matched random null)
    or None if the sentence cannot be located / is at position 0.

    Tractability on long (~8k-token) chains: when ``context_window`` (W) is set,
    the donor is cropped to the local window ``[max(0, t*−W) : t*+span_tokens]``
    via ``truncate_to_window`` and the onset is remapped into the cropped tensor
    BEFORE the sweep. The post-onset tail never affects the teacher-forced onset
    read-out (autoregressive causality ⇒ exact), and the pre-onset crop is a
    standard local-context approximation (exact when t* ≤ W, i.e. the prefix
    already fits). All read/steer indices are derived downstream from the
    remapped onset, so a single onset shift keeps them consistent. With W=None
    the full chain is used (legacy path).
    """
    text = chain.get("chain", "")
    ids = tok.encode(text, return_tensors="pt").to(device)
    onset = _sentence_token_onset(tok, text, ann.get("text", ""))
    if onset is None:
        return None
    T = int(ids.shape[1])
    if not (1 <= onset < T):
        # onset==0 has no prefix to steer/predict from; out-of-range ⇒ skip.
        return None
    if context_window is not None:
        # Crop to the local window and remap the onset; the post-onset tail is
        # dropped exactly (it never enters the onset read-out) and the pre-onset
        # side is the local-context approximation (no-op when onset ≤ W).
        ids, onset = truncate_to_window(
            ids, onset, context_window=context_window, span_tokens=span_tokens)
    res = layer_steering_sweep_single(
        model, ids, onset_pos=onset, directions=directions,
        layers=layers, alpha=alpha, amplify=amplify, span_tokens=span_tokens,
        readout=readout, n_random=n_random, delta_frac=delta_frac,
        behaviour=behaviour, null_directions=null_directions,
    )
    return res


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-short", default="R1-1.5B")
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--annotated", type=Path, default=None,
                        help="Annotated chains JSON. Default data/annotated_<model>.json")
    parser.add_argument("--activations-dir", type=Path, default=None,
                        help="All-layer activations dir. Default data/activations/<model>")
    parser.add_argument("--behaviours", nargs="+", default=["backtracking"],
                        help="Subset of behaviours, or 'all'.")
    parser.add_argument("--n-donors", type=int, default=20,
                        help="Positive donor chains per behaviour. Default 20.")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Explicit layers to sweep (overrides --start/--end-layer).")
    parser.add_argument("--start-layer", type=int, default=5,
                        help="First layer to sweep (default 5 — skip ~the first "
                             "20%% of depth, the embedding-correlated early "
                             "layers, Venhoff 'ignore early layers').")
    parser.add_argument("--end-layer", type=int, default=27,
                        help="Last layer to sweep, inclusive. Default 27.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Steering scale α (used only when --delta-frac is 0; "
                             "vectors are unit-norm). Default 1.0.")
    parser.add_argument("--delta-frac", type=float, default=0.1,
                        help="Per-layer α-normalization: the injected delta's norm "
                             "is this fraction of the median ‖h‖ at the layer over "
                             "the steered prefix (applied identically to the "
                             "behaviour vector and the random null). 0 disables "
                             "(use raw --alpha). Default 0.1.")
    parser.add_argument("--n-random", type=int, default=3,
                        help="R norm-matched random directions for the per-layer "
                             "null (de-confounded effect = behaviour − mean "
                             "random). 0 disables the null. Default 3.")
    parser.add_argument("--readout", choices=["kl", "logprob"], default="kl",
                        help="Primary read-out driving the de-confounded effect: "
                             "'kl' (full-distribution KL at onset−1, default) or "
                             "'logprob' (onset-token Δlog-prob, secondary).")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Bootstrap resamples of the donor set for the argmax "
                             "→ SHORTLIST. Default 1000.")
    parser.add_argument("--shortlist-frac", type=float, default=0.15,
                        help="A layer is shortlisted if it wins ≥ this fraction of "
                             "bootstrap resamples (or is within 1 SEM of the max). "
                             "Default 0.15.")
    parser.add_argument("--drop-embed-cos", type=float, default=None,
                        help="If set, drop swept layers whose v_ℓ has |cos| to any "
                             "input-embedding row ≥ this threshold (extra early-"
                             "layer guard). Default off.")
    parser.add_argument("--amplify", action="store_true",
                        help="Amplify (+α) instead of suppress (−α). Score sign flips.")
    parser.add_argument("--span-tokens", type=int, default=3,
                        help="Average the secondary log-prob read-out over the "
                             "first N onset-span tokens (stability). Default 3.")
    parser.add_argument("--context-window", type=int, default=1024,
                        help="Tractability on long (~8k-token) chains: crop each "
                             "donor to the local token window "
                             "[max(0, onset−W) : onset+span-tokens] and remap the "
                             "onset before the sweep (W = this value, default "
                             "1024). The post-onset tail is dropped EXACTLY "
                             "(autoregressive causality ⇒ it never affects the "
                             "teacher-forced onset read-out); the pre-onset crop "
                             "is the standard local-context approximation for "
                             "steering and is a NO-OP for short chains (onset≤W), "
                             "so results there are unchanged. Set <0 to disable "
                             "(use the full chain).")
    parser.add_argument("--write-pilot", action="store_true",
                        help="Also write pilot_effect_curves.json (feed the "
                             "de-confounded curve to the triangulation consumer "
                             "in place of the confounded 07c output).")
    parser.add_argument("--null-mode", choices=["isotropic", "covariance"],
                        default="isotropic",
                        help="Per-layer random-null distribution for the "
                             "de-confounded effect. 'isotropic' (default) = the "
                             "original norm-matched standard-normal null. "
                             "'covariance' = draw the null from the ambient "
                             "residual covariance Σ_ℓ so it shares the stream's "
                             "anisotropy (CF-10 fix: the isotropic null cannot "
                             "subtract the behaviour direction's alignment with "
                             "high-variance/near-logit directions, which inflates "
                             "the late-layer effect).")
    parser.add_argument("--cov-max-rows", type=int, default=20000,
                        help="Row subsample per layer for the ambient covariance "
                             "estimate (--null-mode covariance). Default 20000.")
    parser.add_argument("--out-suffix", default="",
                        help="Suffix appended to every output filename (e.g. "
                             "'_covmatched') so a re-run does NOT overwrite the "
                             "original isotropic-null results. Default ''.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.behaviours == ["all"]:
        args.behaviours = list(TARGET_BEHAVIOURS)
    if args.annotated is None:
        args.annotated = Path(f"data/annotated_{args.model_short}.json")
    if args.activations_dir is None:
        args.activations_dir = Path(f"data/activations/{args.model_short}")
    if args.out_dir is None:
        args.out_dir = Path(f"results/patching/{args.model_short}")
    if args.layers is None:
        args.layers = list(range(args.start_layer, args.end_layer + 1))
    # context_window < 0 disables the local-window crop (use the full chain).
    context_window = None if args.context_window < 0 else int(args.context_window)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.annotated.exists():
        logger.error(f"Annotated chains not found: {args.annotated}")
        sys.exit(1)
    if not args.activations_dir.exists():
        logger.error(f"Activations dir not found: {args.activations_dir} "
                     f"(per-layer directions need {{b}}_layer{{ℓ}}.npy)")
        sys.exit(1)

    with open(args.annotated) as f:
        annotations = json.load(f)
    logger.info(f"Loaded {len(annotations)} annotated chains")

    model, tok = load_model_and_tokenizer(args.model_id, args.dtype)
    device = next(model.parameters()).device

    mode = "amplify (+α)" if args.amplify else "suppress (−α)"
    delta_frac = None if (args.delta_frac is None or args.delta_frac <= 0) else args.delta_frac
    norm_desc = f"δ-frac={delta_frac}" if delta_frac is not None else f"α={args.alpha}"
    ctx_desc = f"ctx-window={context_window}" if context_window is not None else "ctx-window=full"
    logger.info(f"Layers {args.layers[0]}..{args.layers[-1]} | {norm_desc} | "
                f"{mode} | read-out={args.readout} (primary) | R-random={args.n_random} | "
                f"span={args.span_tokens} | {ctx_desc} | donors/beh={args.n_donors} | "
                f"n-boot={args.n_boot}")

    # Input-embedding matrix for the optional early-layer cosine guard.
    embed_matrix = None
    if args.drop_embed_cos is not None:
        try:
            embed_matrix = model.get_input_embeddings().weight.detach().float().cpu().numpy()
        except Exception as e:
            logger.warning(f"could not read input embeddings for --drop-embed-cos: {e}")

    # Covariance-matched null (CF-10): the ambient residual covariance Σ_ℓ is
    # behaviour-independent, so estimate it ONCE per swept layer and reuse across
    # behaviours. Isotropic mode leaves this empty (the sweep uses its in-built
    # standard-normal null).
    cov_by_layer = {}
    if args.null_mode == "covariance":
        logger.info(f"null-mode=covariance: estimating ambient Σ_ℓ for "
                    f"{len(args.layers)} layers (≤{args.cov_max_rows} rows/layer)")
        for L in args.layers:
            cov = ambient_covariance(
                args.activations_dir, list(TARGET_BEHAVIOURS), L,
                max_rows=args.cov_max_rows, seed=args.seed)
            if cov is not None:
                cov_by_layer[L] = cov
        logger.info(f"  Σ_ℓ ready for layers {sorted(cov_by_layer)}")

    summary = {}
    pilot_agg = {}

    for beh in args.behaviours:
        logger.info(f"=== {beh} ===")
        others = [b for b in TARGET_BEHAVIOURS if b != beh]
        directions = per_layer_directions(
            args.activations_dir, beh, args.layers, others)
        present = sorted(directions.keys())
        if not present:
            logger.warning(f"No per-layer directions for {beh}; skipping")
            summary[beh] = {"error": "no_directions"}
            continue
        if len(present) < len(args.layers):
            missing = sorted(set(args.layers) - set(present))
            logger.warning(f"  {beh}: missing directions at layers {missing} "
                           f"(activation files absent); sweeping {present}")

        # Optional embedding-cosine early-layer guard: drop v_ℓ too aligned with
        # the input embeddings (warn either way).
        embed_cos = {}
        dropped_embed = []
        if embed_matrix is not None:
            embed_cos = embedding_cosine(directions, embed_matrix)
            for L in list(present):
                if embed_cos.get(L, 0.0) >= args.drop_embed_cos:
                    dropped_embed.append(L)
            if dropped_embed:
                logger.warning(f"  {beh}: dropping embedding-aligned layers "
                               f"{dropped_embed} (|cos|≥{args.drop_embed_cos})")
                present = [L for L in present if L not in dropped_embed]
            if not present:
                logger.warning(f"  {beh}: all layers dropped by embed-cos; skipping")
                summary[beh] = {"error": "all_dropped_embed_cos"}
                continue

        # Build the per-layer covariance-matched null directions for this
        # behaviour (seeded per (behaviour, ℓ, r), matched draw-for-draw to the
        # isotropic seeds). None ⇒ the sweep uses its default isotropic null.
        null_dirs = None
        if args.null_mode == "covariance":
            null_dirs = {}
            for L in present:
                cov = cov_by_layer.get(L)
                if cov is not None:
                    null_dirs[L] = covariance_matched_unit_directions(
                        cov, beh, L, args.n_random)
            missing_cov = [L for L in present if L not in null_dirs]
            if missing_cov:
                logger.warning(f"  {beh}: no Σ_ℓ at layers {missing_cov}; "
                               f"those layers get an EMPTY null (effect=raw)")

        donors = select_positive_donors(annotations, beh, args.n_donors, args.seed)
        if not donors:
            logger.warning(f"No positive donors for {beh}; skipping")
            summary[beh] = {"error": "no_donors"}
            continue
        logger.info(f"Selected {len(donors)} positive donors")

        per_donor_scores: list[dict[int, float]] = []
        sweep_results = []
        trials = []
        for i, (chain, ann) in enumerate(donors):
            t0 = time.time()
            try:
                res = sweep_for_donor(
                    model, tok, device, chain, ann, directions, present,
                    args.alpha, args.amplify, args.span_tokens,
                    behaviour=beh, readout=args.readout,
                    n_random=args.n_random, delta_frac=delta_frac,
                    context_window=context_window, null_directions=null_dirs)
                if res is None:
                    logger.warning(f"  donor {i}: onset not locatable / no prefix; skip")
                    continue
                per_donor_scores.append(dict(res.score))
                sweep_results.append(res)
                trials.append({
                    "chain_id": chain.get("task_id", ""),
                    "onset_pos": int(res.onset_pos),
                    "baseline_M": float(res.baseline_M),
                    "n_span_tokens": int(res.n_span_tokens),
                    "deconfounded_effect": {int(L): float(v) for L, v in res.score.items()},
                    "behaviour_effect": {int(L): float(v) for L, v in res.behaviour_effect.items()},
                    "random_null": {int(L): float(v) for L, v in res.random_null.items()},
                    "kl_effect": {int(L): float(v) for L, v in res.kl_effect.items()},
                    "logprob_effect": {int(L): float(v) for L, v in res.logprob_effect.items()},
                })
                if (i + 1) % 5 == 0:
                    logger.info(f"  donor {i+1}/{len(donors)} ({time.time()-t0:.1f}s)")
            except Exception as e:
                logger.warning(f"  donor {i}: {type(e).__name__}: {e}")

        # Per-layer full record (de-confounded effect drives mean_effect/argmax).
        full = aggregate_full_curves(sweep_results)
        peak = argmax_layer(full)
        boot = bootstrap_shortlist(
            per_donor_scores, n_boot=args.n_boot,
            select_frac=args.shortlist_frac, seed=args.seed)
        shortlist = boot.get("shortlist", [])
        logger.info(f"  {beh}: point argmax = {peak} | SHORTLIST = {shortlist} "
                    f"(n_donors used = {len(per_donor_scores)})")

        summary[beh] = {
            "behaviour": beh,
            "layers": present,
            "alpha": args.alpha,
            "delta_frac": delta_frac,
            "readout": args.readout,
            "n_random": args.n_random,
            "null_mode": args.null_mode,
            "mode": "amplify" if args.amplify else "suppress",
            "span_tokens": args.span_tokens,
            "context_window": context_window,
            "n_donors": len(per_donor_scores),
            "argmax_layer": peak,
            "shortlist": shortlist,
            "bootstrap": {
                "n_boot": boot.get("n_boot"),
                "shortlist_frac": args.shortlist_frac,
                "boot_freq": boot.get("boot_freq", {}),
            },
            "embed_cos": {int(L): float(c) for L, c in embed_cos.items()},
            "dropped_embed_layers": dropped_embed,
            "layer_effect": {int(L): v for L, v in full.items()},
            "trials": trials,
        }
        pilot_agg[beh] = {
            "behaviour": beh,
            "n_pairs": len(per_donor_scores),
            # mean_effect == deconfounded_effect ⇒ drop-in for the triangulation
            # consumer (compute_layer_triangulation.load_phase7b_curves).
            "layer_effect": {str(L): v for L, v in full.items()},
            "shortlist": shortlist,
        }
        per_beh_out = args.out_dir / f"steering_effect_curves_{beh}{args.out_suffix}.json"
        per_beh_out.write_text(json.dumps(summary[beh], indent=2, default=str))
        logger.info(f"  wrote {per_beh_out}")

    (args.out_dir / f"steering_effect_curves{args.out_suffix}.json").write_text(
        json.dumps(summary, indent=2, default=str))
    if args.write_pilot:
        (args.out_dir / f"pilot_effect_curves{args.out_suffix}.json").write_text(
            json.dumps(pilot_agg, indent=2, default=str))
        logger.info("  wrote pilot_effect_curves (triangulation consumer)")

    write_summary_md(summary, args,
                     args.out_dir / f"steering_effect_summary{args.out_suffix}.md")
    _maybe_plot(summary, args)

    print(f"\nResults: {args.out_dir}")
    print(f"  steering_effect_curves.json   (per-behaviour de-confounded curves "
          f"+ bootstrap shortlist + per-donor)")
    print(f"  steering_effect_summary.md    (SHORTLIST + de-confounded table)")
    if args.write_pilot:
        print(f"  pilot_effect_curves.json      (triangulation consumer; "
              f"mean_effect = deconfounded_effect)")
    print("\nSHORTLIST per behaviour (PRE-FILTER for Phase 7 to confirm — NOT a "
          "single load-bearing argmax):")
    for beh, d in summary.items():
        if isinstance(d, dict) and "shortlist" in d:
            print(f"  {beh:>24}: {d['shortlist']}  (point argmax {d.get('argmax_layer')})")


def _cell(le, L, key):
    """Pull ``key`` for layer ``L`` from a layer_effect dict (int or str keyed)."""
    v = le.get(L)
    if v is None:
        v = le.get(str(L))
    if not isinstance(v, dict):
        return None
    return v.get(key)


def write_summary_md(summary, args, path):
    mode = "amplify (+α)" if args.amplify else "suppress (−α)"
    norm_desc = (f"per-layer δ-frac={args.delta_frac} (delta norm = δ-frac·median‖h‖)"
                 if (args.delta_frac and args.delta_frac > 0) else f"raw α={args.alpha}")
    ctx_desc = (f"each donor cropped to a local window of W={args.context_window} "
                f"pre-onset tokens (+span tail) — the post-onset tail is dropped "
                f"exactly (it never enters the teacher-forced onset read-out) and "
                f"the pre-onset crop is the standard local-context approximation, "
                f"a no-op for chains whose onset ≤ W"
                if args.context_window >= 0
                else "full chains (no local-window crop)")
    lines = [
        f"# Per-layer steering-effect sweep — {args.model_short}",
        "",
        f"Method: FORWARD-PASS intervention, de-confounded (CF-10 replacement for "
        f"07c attribution patching). Layers {args.layers[0]}..{args.layers[-1]} "
        f"(start-layer skips the embedding-correlated early band); {norm_desc}; "
        f"{mode}; **primary read-out = {args.readout}**; R={args.n_random} "
        f"norm-matched random directions; donors/behaviour={args.n_donors}; "
        f"bootstrap n={args.n_boot}; {ctx_desc}.",
        "",
        "**De-confounded effect(ℓ) = Score_b(ℓ) − mean_r Score_random(ℓ)**, where "
        "Score is the KL(p_steered‖p_baseline) at onset−1 (primary; captures "
        "redistribution to synonym onsets) — the secondary onset-token Δlog-prob "
        "is also recorded. The random-direction null subtracts any *global* "
        "sensitivity of a layer to ANY perturbation (e.g. late layers near the "
        "logits), isolating the behaviour-specific causal effect. The read-out is "
        "the fixed OUTPUT, so there is no read-out-proximity term either.",
        "",
        "**The SHORTLIST is a PRE-FILTER for Phase 7 to CONFIRM, not a single "
        "load-bearing argmax.** Layers are shortlisted if they win ≥ "
        f"{args.shortlist_frac:.0%} of bootstrap resamples OR sit within 1 SEM of "
        "the max de-confounded effect (SEM = std(ddof=1)/√n).",
        "",
        "⚠️ Caveats: the secondary log-prob read-out is token-anchored; KL is "
        "unsigned (magnitude of redistribution). α/δ-frac and the suppression "
        "window are choices.",
        "",
        "## Per-behaviour SHORTLIST (pre-filter for Phase 7)",
        "",
        "| Behaviour | SHORTLIST (candidate ℓ) | point argmax | n donors |",
        "|---|---|---|---|",
    ]
    for beh, d in summary.items():
        if not isinstance(d, dict) or "error" in d:
            lines.append(f"| {beh} | ERR ({d.get('error','?') if isinstance(d,dict) else '?'}) | — | — |")
            continue
        sl = d.get("shortlist", [])
        sl_str = ", ".join(f"**{L}**" for L in sl) if sl else "—"
        lines.append(f"| {beh} | {sl_str} | {d.get('argmax_layer')} | "
                     f"{d.get('n_donors')} |")

    lines += ["", "## De-confounded effect curve (mean ± SEM)", ""]
    show = sorted({L for d in summary.values()
                   if isinstance(d, dict) for L in d.get("layer_effect", {})}
                  | set())
    # Keep the table compact: at most ~9 evenly spaced layers.
    show = sorted(int(L) for L in show)
    if len(show) > 9:
        idx = np.linspace(0, len(show) - 1, 9).round().astype(int)
        show = [show[i] for i in sorted(set(idx))]
    if show:
        lines.append("| Behaviour | " + " | ".join(f"L{L}" for L in show) + " |")
        lines.append("|---|" + "|".join(["---"] * len(show)) + "|")
        for beh, d in summary.items():
            if not isinstance(d, dict) or "error" in d:
                continue
            le = d.get("layer_effect", {})
            if not le:
                continue
            cells = []
            for L in show:
                m = _cell(le, L, "deconfounded_effect")
                s = _cell(le, L, "sem")
                if m is None:
                    cells.append("—")
                else:
                    cells.append(f"{m:.3g}±{(s or 0):.2g}")
            lines.append(f"| {beh} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Per-layer {behaviour_effect, random_null, deconfounded_effect, kl_effect, "
        "logprob_effect, sem, n} and the bootstrap frequencies are in "
        "`steering_effect_curves.json`; per-donor breakdowns under each "
        "behaviour's `trials`.",
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
        if not isinstance(d, dict) or "error" in d or not d.get("layer_effect"):
            continue
        le = d["layer_effect"]
        Ls = sorted(int(L) for L in le)
        means = [_cell(le, L, "deconfounded_effect") for L in Ls]
        sems = [_cell(le, L, "sem") or 0.0 for L in Ls]
        line = ax.errorbar(Ls, means, yerr=sems, marker="o", ms=3, capsize=2, label=beh)
        # Mark shortlisted layers with a filled star at their de-confounded value.
        sl = set(d.get("shortlist", []))
        if sl:
            xs = [L for L in Ls if L in sl]
            ys = [_cell(le, L, "deconfounded_effect") for L in xs]
            ax.scatter(xs, ys, marker="*", s=120, zorder=5,
                       color=line[0].get_color(), edgecolor="k", linewidth=0.4)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.axhline(0, color="grey", ls="--", alpha=0.4)
    ax.set_xlabel("Steering layer ℓ")
    ax.set_ylabel(f"De-confounded effect  (Score_b − mean_r Score_random, {args.readout})")
    mode = "amplify" if args.amplify else "suppress"
    ax.set_title(f"Per-layer de-confounded steering effect — {args.model_short} "
                 f"({mode}, {args.readout}; ★ = shortlist)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out_dir / f"steering_effect_curves{args.out_suffix}.png", dpi=120)
    logger.info(f"Plot saved: {args.out_dir / f'steering_effect_curves{args.out_suffix}.png'}")


if __name__ == "__main__":
    main()
