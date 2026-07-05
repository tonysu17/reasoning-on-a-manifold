#!/usr/bin/env python3
"""
Annotation-noise band — quantify the run-to-run nondeterminism of the Sonnet
annotator on byte-identical chains, so every Phase-7 behaviour-fraction gets an
error bar.

Why this exists (REVIEW_PHASE7_2026-06-25 / critic N, METHODOLOGY_REFINEMENT §2.6)
---------------------------------------------------------------------------------
The trimmed pilot annotated the (un-steered) VANILLA generations twice — once in
the L27 batch, once in the L16 batch. The generations are byte-identical (same
tasks, no steering: identical mean_n_tokens=2184.7 and repetition_rate in both
`eval_summary.json`), yet the vanilla uncertainty-estimation sentence-fraction came
back **0.181 (L27 batch) vs 0.153 (L16 batch)** — a 0.028 absolute swing, ≈25% of
the headline steering effect (−0.114). At annotator temperature 0 this is pure
run-to-run nondeterminism of the proxy/model. A single annotation therefore has an
unstated error bar; a steering Δ smaller than that bar is not real.

This script RE-ANNOTATES the same vanilla chains K times with the SAME annotator,
SAME prompt, SAME temperature (0.0) and SAME metric the pilot used
(`src.annotation.annotate_chain` → `src.evaluation.behaviour_fraction`), then
reports, per behaviour, the spread of the corpus-mean vanilla fraction across the
re-annotations. That spread is the **annotator self-consistency band** — a LOWER
BOUND on total annotation noise (cross-annotator disagreement, Sonnet-vs-Qwen3/Nova,
is strictly larger; see METHODOLOGY_REFINEMENT §2.6, which wants the cross-annotator
fraction-RMS as the *acceptance* band and is NOT replaced by this). It is the cheap
self-noise floor the recap asked to measure before the big Phase-7 spend.

Outcome metric (byte-identical to the pilot)
--------------------------------------------
For each annotation run k and behaviour b:
  frac_k,b = mean over the N vanilla chains of behaviour_fraction(spans, b)
            = mean over chains of  (#sentences labelled b) / (#sentences)
The headline statistic is corpus-level, so the band is the spread of frac_k,b
across runs k:  band_b = std_k(frac_k,b)  (ddof=1), with the empirical min..max
range and the pairwise fraction-RMS also reported.

Cost / safety
-------------
Pure annotation API (no GPU). Bounded scope = K × N_vanilla annotations (default
8 × 10 = 80, ≈ $4–8 at the pilot's ~$0.05–0.10/chain). Checkpointed per
(k, task_id) to a resumable JSONL, so credits-out loses ≤1 annotation and a
re-run resumes. The K × N ceiling IS the $-cap (no live meter on the proxy).

Creds: CLAUDE_PROXY_URL / CLAUDE_PROXY_KEY (same proxy as phases 1/3/7;
on the cluster: `source ~/.rom_proxy_env`).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.annotation import annotate_chain, ANNOTATION_MODEL, TARGET_BEHAVIOURS
from src.evaluation import behaviour_fraction

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_vanilla_chains(eval_dir: Path) -> list[dict]:
    """The un-steered vanilla generations from a Phase-7 steering_results.json
    (method == 'vanilla'). Returns [{task_id, chain, n_tokens}]."""
    results = json.loads((eval_dir / "steering_results.json").read_text())
    van = [r for r in results if str(r.get("method", "")).lower() == "vanilla"]
    out = []
    for r in van:
        text = r.get("chain", "")
        if text.strip():
            out.append({"task_id": r.get("task_id", ""), "chain": text,
                        "n_tokens": r.get("n_tokens")})
    return out


def historical_vanilla_fractions(eval_dir: Path) -> dict[str, float] | None:
    """Recover the per-behaviour vanilla corpus-mean fraction already computed in
    a pilot eval_summary.json (summary[b]['vanilla']['0.0']['mean']). These are
    independent annotations of the SAME chains → fold in as extra runs."""
    p = eval_dir / "eval_summary.json"
    if not p.exists():
        return None
    summ = json.loads(p.read_text())
    out = {}
    for b in TARGET_BEHAVIOURS:
        try:
            out[b] = float(summ[b]["vanilla"]["0.0"]["mean"])
        except (KeyError, TypeError, ValueError):
            pass
    return out or None


def load_checkpoint(path: Path) -> dict[tuple, dict]:
    """{(k, task_id): record} from the resumable JSONL (empty if absent)."""
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[(int(rec["k"]), rec["task_id"])] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path,
                    default=Path("results/eval/R1-1.5B__L27_trim"),
                    help="Phase-7 eval dir holding the vanilla steering_results.json.")
    ap.add_argument("--historical-dirs", type=Path, nargs="*",
                    default=[Path("results/eval/R1-1.5B__L27_trim"),
                             Path("results/eval/R1-1.5B__L16_trim")],
                    help="Pilot eval dirs whose vanilla fractions are folded in as "
                         "extra (already-paid) annotation runs.")
    ap.add_argument("--k", type=int, default=8,
                    help="Number of FRESH re-annotation passes over the vanilla "
                         "chains. Cost ≈ k × n_vanilla annotations. Default 8.")
    ap.add_argument("--max-chains", type=int, default=None,
                    help="Cap the number of vanilla chains (cost control). "
                         "Default: all (~10).")
    ap.add_argument("--behaviours", nargs="+", default=["all"])
    ap.add_argument("--out-dir", type=Path, default=Path("results/eval/noise_band"))
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="Seconds to sleep between annotation calls (politeness).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Orders the chain list deterministically (annotation "
                         "itself is temp-0; the seed only fixes traversal order).")
    args = ap.parse_args()

    if "CLAUDE_PROXY_URL" not in os.environ or "CLAUDE_PROXY_KEY" not in os.environ:
        logger.error("CLAUDE_PROXY_URL / CLAUDE_PROXY_KEY not set "
                     "(on the cluster: `source ~/.rom_proxy_env`). Aborting.")
        sys.exit(1)

    behaviours = list(TARGET_BEHAVIOURS) if args.behaviours == ["all"] else args.behaviours
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / "noise_band_runs.jsonl"

    chains = load_vanilla_chains(args.eval_dir)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(chains))
    chains = [chains[i] for i in order]
    if args.max_chains is not None:
        chains = chains[: args.max_chains]
    if not chains:
        logger.error(f"No vanilla chains in {args.eval_dir}/steering_results.json")
        sys.exit(1)
    logger.info(f"Loaded {len(chains)} byte-identical vanilla chains from {args.eval_dir}")
    logger.info(f"Annotator: {ANNOTATION_MODEL} (temperature 0.0) | K={args.k} fresh "
                f"runs | ≈{args.k * len(chains)} annotations | checkpoint {ckpt_path}")

    done = load_checkpoint(ckpt_path)
    if done:
        logger.info(f"Resuming: {len(done)} (k, chain) annotations already on disk")

    # ── Re-annotate: K passes × chains, checkpointed per (k, task_id) ──────────
    t_start = time.time()
    with ckpt_path.open("a") as ck:
        for k in range(args.k):
            for chain in chains:
                key = (k, chain["task_id"])
                if key in done:
                    continue
                t0 = time.time()
                try:
                    spans, complete = annotate_chain(chain["chain"])
                except Exception as e:                       # transport / proxy error
                    logger.warning(f"  run {k} {chain['task_id']}: "
                                   f"{type(e).__name__}: {e}; skipping (resumable)")
                    continue
                fracs = {b: float(behaviour_fraction(spans, b)) for b in behaviours}
                rec = {"k": k, "task_id": chain["task_id"], "n_spans": len(spans),
                       "complete": bool(complete), "empty": len(spans) == 0,
                       "fractions": fracs}
                ck.write(json.dumps(rec) + "\n")
                ck.flush()
                done[key] = rec
                logger.info(f"  run {k+1}/{args.k} {chain['task_id']}: "
                            f"{len(spans)} spans ({time.time()-t0:.1f}s)"
                            + ("" if complete else "  [PARTIAL]"))
                if args.sleep > 0:
                    time.sleep(args.sleep)
    logger.info(f"Annotation done in {(time.time()-t_start)/60:.1f} min")

    # ── Aggregate: per-run corpus mean, then the band across runs ─────────────
    # Group the checkpoint records by run k.
    runs: dict[int, list[dict]] = {}
    for (k, _tid), rec in done.items():
        runs.setdefault(k, []).append(rec)

    historical = []                                          # (label, {b: frac})
    for hd in args.historical_dirs:
        hv = historical_vanilla_fractions(hd)
        if hv:
            historical.append((hd.name, hv))
            logger.info(f"Folded historical vanilla fractions from {hd.name}")

    report = {
        "annotator": ANNOTATION_MODEL,
        "temperature": 0.0,
        "eval_dir": str(args.eval_dir),
        "n_vanilla_chains": len(chains),
        "k_fresh_runs": args.k,
        "behaviours": {},
        "note": ("Annotator self-consistency band (Sonnet-vs-Sonnet, temp-0 "
                 "run-to-run) on byte-identical vanilla chains. LOWER BOUND on "
                 "annotation noise — cross-annotator (Sonnet-vs-Qwen3/Nova) "
                 "disagreement is strictly larger and is the METHODOLOGY_REFINEMENT "
                 "§2.6 acceptance band. Use band_std as the error bar on any single "
                 "behaviour-fraction; a steering Δ must exceed it to be real."),
    }

    for b in behaviours:
        # Per fresh run k: corpus-mean fraction over non-empty chains.
        run_means = []
        for k in sorted(runs):
            vals = [r["fractions"].get(b) for r in runs[k]
                    if not r.get("empty") and b in r["fractions"]]
            vals = [v for v in vals if v is not None]
            if vals:
                run_means.append(float(np.mean(vals)))
        hist_means = [hv[b] for (_lbl, hv) in historical if b in hv]
        all_means = run_means + hist_means

        if len(all_means) >= 1:
            arr = np.array(all_means, dtype=float)
            std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
            # Fraction-scale RMS of pairwise |diff| (the §2.6 scale, here within-annotator).
            if arr.size >= 2:
                diffs = [abs(arr[i] - arr[j])
                         for i in range(arr.size) for j in range(i + 1, arr.size)]
                pairwise_rms = float(np.sqrt(np.mean(np.square(diffs))))
            else:
                pairwise_rms = 0.0
            report["behaviours"][b] = {
                "mean_fraction": float(arr.mean()),
                "band_std": std,                              # the error bar
                "pairwise_rms": pairwise_rms,
                "min": float(arr.min()),
                "max": float(arr.max()),
                "range": float(arr.max() - arr.min()),
                "n_runs": int(arr.size),
                "n_fresh_runs": len(run_means),
                "n_historical": len(hist_means),
                "fresh_run_means": [round(x, 4) for x in run_means],
                "historical_means": [round(x, 4) for x in hist_means],
                "ci95_halfwidth": float(1.96 * std / np.sqrt(arr.size)) if arr.size >= 2 else 0.0,
            }

    (args.out_dir / "annotation_noise_band.json").write_text(
        json.dumps(report, indent=2))

    # ── Human-readable summary ────────────────────────────────────────────────
    lines = ["# Annotation-noise band — vanilla re-annotation (Sonnet self-consistency)\n",
             f"Annotator **{ANNOTATION_MODEL}**, temperature 0.0, on "
             f"{len(chains)} byte-identical vanilla chains; "
             f"{args.k} fresh runs + {len(historical)} historical pilot annotations.\n",
             "`band_std` is the run-to-run std of the corpus-mean vanilla fraction "
             "— the error bar a steering Δ must clear. This is the Sonnet-vs-Sonnet "
             "self-noise floor (a LOWER bound; cross-annotator is larger).\n",
             "| Behaviour | mean frac | band (std) | pairwise-RMS | min..max | n runs |",
             "|---|---|---|---|---|---|"]
    for b in behaviours:
        r = report["behaviours"].get(b)
        if not r:
            continue
        lines.append(
            f"| {b} | {r['mean_fraction']:.3f} | ±{r['band_std']:.3f} | "
            f"{r['pairwise_rms']:.3f} | {r['min']:.3f}..{r['max']:.3f} | {r['n_runs']} |")
    unc = report["behaviours"].get("uncertainty-estimation")
    if unc:
        lines += ["",
                  f"**Uncertainty-estimation** (the candidate headline behaviour): "
                  f"vanilla fraction {unc['mean_fraction']:.3f} ± {unc['band_std']:.3f} "
                  f"(range {unc['min']:.3f}..{unc['max']:.3f}). The pilot's headline "
                  f"steering effect was −0.114; compare against this band to judge "
                  f"whether the effect clears annotator self-noise."]
    (args.out_dir / "annotation_noise_band.md").write_text("\n".join(lines) + "\n")

    logger.info(f"Wrote {args.out_dir}/annotation_noise_band.json + .md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
