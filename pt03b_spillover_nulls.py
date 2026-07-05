"""Post-training spillover — Step 3b: the GATED analysis (pre-registered nulls).

Runs the gates of ``METHODOLOGY_SAFETY_SPILLOVER_2026-07-03.md`` over two
activation dirs produced by ``04_extract_activations.py`` (the base checkpoint
and a post-trained checkpoint, extracted on the SAME chains with a SHARED
tokenizer, e.g. ``--tokenizer-alias 1.5b``):

  1. parity check (row identity + token_start; hard-fails on tokenizer drift);
  2. per behaviour x layer: matched-n mean principal angle vs a pooled
     permutation null (excess + smoothed p), split-half within-model floors;
  3. row-paired displacement coherence vs a pairing-destroyed null;
  4. duplicate-row disclosure (raw n vs unique-span n);
  5. selectivity ranking by EXCESS angle (never raw angle).

Gates NOT implemented here (must be listed as un-run in any write-up):
surprisal control (needs per-span NLL from a forward pass) and the
annotator-swap re-ranking (needs re-extraction under the alternative
annotators' spans).

Example
-------
python pt03b_spillover_nulls.py \
    --base-acts data/activations/R1-1.5B-pilot100 \
    --post-acts data/activations/STAR1-1.5B-pilot100 \
    --out results/safety_posttrain/spillover_gated_pilot100.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.safety_posttrain import nulls as N
from src.row_provenance import duplicate_fraction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_BEHAVIOURS = ["backtracking", "uncertainty-estimation",
                      "example-testing", "adding-knowledge"]
DEFAULT_LAYERS = [12, 16]      # pre-committed in the methodology note (C9)


def _load(act_dir: Path, behaviour: str, layer: int) -> np.ndarray | None:
    f = act_dir / f"{behaviour}_layer{layer}.npy"
    return np.load(f) if f.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-acts", required=True)
    ap.add_argument("--post-acts", required=True)
    ap.add_argument("--behaviours", nargs="+", default=DEFAULT_BEHAVIOURS)
    ap.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--matched-n", type=int, default=None,
                    help="common subsample size (default: min behaviour count)")
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--n-rep", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-unpaired", action="store_true",
                    help="continue past a parity failure (angles only, paired "
                         "displacement skipped, report flagged CONTAMINATED)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base_dir, post_dir = Path(args.base_acts), Path(args.post_acts)
    report: dict = {
        "base_acts": str(base_dir), "post_acts": str(post_dir),
        "k": args.k, "n_perm": args.n_perm, "n_rep": args.n_rep, "seed": args.seed,
        "gates_not_run": ["surprisal_control", "annotator_swap"],
    }

    # 1. parity
    parity = N.parity_check(base_dir, post_dir, args.behaviours)
    report["parity"] = parity
    if not parity.get("ok"):
        log.error("PARITY FAILED: %s", json.dumps(parity)[:400])
        if not args.allow_unpaired:
            raise SystemExit("parity check failed — refusing to diff (use "
                             "--allow-unpaired to override, angles only)")
        report["CONTAMINATED"] = "parity failed; angles indicative only"

    # matched n across behaviours (C3), from the smaller side per behaviour
    counts = {}
    for b in args.behaviours:
        Xb = _load(base_dir, b, args.layers[0])
        Xp = _load(post_dir, b, args.layers[0])
        if Xb is not None and Xp is not None:
            counts[b] = min(Xb.shape[0], Xp.shape[0])
    if not counts:
        raise SystemExit("no behaviour present in both dirs")
    m = args.matched_n or min(counts.values())
    report["matched_n"] = m
    report["behaviour_counts"] = counts

    # 2-4. per behaviour x layer
    report["layers"] = {}
    for layer in args.layers:
        lrep: dict = {}
        for b in args.behaviours:
            Xb, Xp = _load(base_dir, b, layer), _load(post_dir, b, layer)
            if Xb is None or Xp is None:
                lrep[b] = {"skipped": "missing activations"}
                continue
            log.info("layer %d %s: angles vs within-model null (n=%d, m=%d)...",
                     layer, b, Xb.shape[0], min(m, Xb.shape[0], Xp.shape[0]))
            entry = N.gated_angle_report(Xb, Xp, k=args.k, m=min(m, Xb.shape[0], Xp.shape[0]),
                                         n_perm=args.n_perm, n_rep=args.n_rep,
                                         seed=args.seed)
            log.info("layer %d %s: observed %.2f vs null %.2f (excess %+.2f, p=%.4f)",
                     layer, b, entry["observed_deg"], entry["null_mean_deg"],
                     entry["excess_deg"], entry["p_within"])
            entry["dup_fraction_base"] = round(duplicate_fraction(Xb), 4)
            entry["dup_fraction_post"] = round(duplicate_fraction(Xp), 4)
            if parity.get("ok") and Xb.shape == Xp.shape:
                entry["paired"] = N.paired_displacement(Xb, Xp, n_perm=args.n_perm,
                                                        seed=args.seed)
            lrep[b] = entry
        report["layers"][str(layer)] = lrep

    # 5. selectivity by excess angle (per layer)
    report["selectivity_by_excess"] = {
        L: sorted(((b, r["excess_deg"]) for b, r in lrep.items()
                   if isinstance(r, dict) and "excess_deg" in r),
                  key=lambda t: t[1], reverse=True)
        for L, lrep in report["layers"].items()
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    log.info("gated report -> %s", out)
    for L, ranking in report["selectivity_by_excess"].items():
        log.info("layer %s selectivity (excess deg): %s", L, ranking)


if __name__ == "__main__":
    main()
