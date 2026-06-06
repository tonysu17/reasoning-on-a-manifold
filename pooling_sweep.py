#!/usr/bin/env python3
"""Pooling sensitivity sweep — does the geometry depend on how we pool a span?

Methodological caution: Phase 4 reduces each behaviour span to one vector by
MEAN-pooling [1 preceding + first n_execution] tokens. Mean discards order and
can cancel opposing directions; the LAST execution token is the only
context-complete position in a causal model. This script re-extracts the SAME
chains under each pooling mode and reports whether the conclusions move:

  * cos(single_direction[mean], single_direction[last|first])  per behaviour
    — if low, the steering direction is pooling-dependent (a real confound).
  * d_eff_70 and participation ratio per mode — does dimensionality change?

Requires the model (GPU): `pip install .[gpu]`. Extracts ONLY the steering
layer(s) by default to stay cheap. Does NOT touch the canonical (mean)
activations — writes to results/pooling_sweep/<model>/pool_<mode>/.

Usage:
  python pooling_sweep.py --model 1.5b --annotated data/annotated_R1-1.5B.json
  python pooling_sweep.py --model 1.5b --layers 27 --modes mean last first
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.activation_extraction import extract_activations, load_activations, POOLING_MODES
from src.annotation import load_annotated, TARGET_BEHAVIOURS
from src.chain_gen import load_model
from src.steering import build_steering_vectors
from src.pca import analyse_behaviour
from src.config import model_tuple, provenance, require_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="1.5b", help="cli_alias from config.yaml")
    ap.add_argument("--annotated", default="data/annotated_R1-1.5B.json")
    ap.add_argument("--layers", type=int, nargs="+", default=[27],
                    help="Layers to extract (default just the steering layer 27).")
    ap.add_argument("--modes", nargs="+", default=["mean", "last", "first"],
                    choices=list(POOLING_MODES))
    ap.add_argument("--steer-layer", type=int, default=27)
    ap.add_argument("--max-chains", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    require_file(args.annotated, "run 03_annotate_chains.py first")
    model_id, short, dtype = model_tuple(args.model)
    out = args.out or Path(f"results/pooling_sweep/{short}")
    out.mkdir(parents=True, exist_ok=True)

    annotated = load_annotated(args.annotated)
    if args.max_chains:
        annotated = annotated[:args.max_chains]
    logger.info(f"Loading {model_id} ({dtype})")
    model, tokenizer = load_model(model_id, dtype=dtype)

    # 1) Extract under each pooling mode (one save dir per mode).
    for mode in args.modes:
        mode_dir = out / f"pool_{mode}"
        if (mode_dir / "metadata.json").exists():
            logger.info(f"[{mode}] already extracted — skip")
            continue
        logger.info(f"[{mode}] extracting at layers {args.layers}")
        extract_activations(model, tokenizer, annotated, layers=args.layers,
                            save_dir=mode_dir, pooling=mode)

    # 2) Per mode: steering vectors + dimensionality at the steering layer.
    per_mode = {}
    for mode in args.modes:
        mode_dir = out / f"pool_{mode}"
        vecs = build_steering_vectors(mode_dir, layer=args.steer_layer,
                                      k_values=[1, "auto"], variance_threshold=0.70)
        dims = {}
        for beh in TARGET_BEHAVIOURS:
            try:
                mat = load_activations(mode_dir, beh, args.steer_layer)
                r = analyse_behaviour(mat)
                dims[beh] = {"n": int(mat.shape[0]), "d_eff_70": r["d_eff_70"],
                             "participation_ratio": round(r["participation_ratio"], 2)}
            except FileNotFoundError:
                dims[beh] = None
        per_mode[mode] = {"single": {b: vecs[b]["single_direction"] for b in vecs}, "dims": dims}

    # 3) Compare steering direction across modes (cos vs the mean baseline).
    base = "mean" if "mean" in per_mode else args.modes[0]
    report = {"model": short, "steer_layer": args.steer_layer, "baseline": base,
              "behaviours": {}}
    print(f"\n=== Pooling sensitivity at layer {args.steer_layer} (baseline = {base}) ===")
    others = [m for m in args.modes if m != base]
    print(f"  {'behaviour':<24}" + "".join(f"cos({base},{m})".rjust(16) for m in others)
          + "   d_eff_70 [" + "/".join(args.modes) + "]")
    for beh in TARGET_BEHAVIOURS:
        vb = per_mode[base]["single"].get(beh)
        cos_cells, rec = "", {"cos_vs_baseline": {}, "d_eff_70": {}}
        for m in others:
            vm = per_mode[m]["single"].get(beh)
            c = float(vb @ vm) if (vb is not None and vm is not None) else float("nan")
            rec["cos_vs_baseline"][m] = c
            cos_cells += f"{c:+.3f}".rjust(16)
        deff = []
        for m in args.modes:
            d = per_mode[m]["dims"].get(beh)
            rec["d_eff_70"][m] = (d or {}).get("d_eff_70")
            deff.append(str((d or {}).get("d_eff_70", "—")))
        report["behaviours"][beh] = rec
        print(f"  {beh:<24}{cos_cells}   {'/'.join(deff)}")

    (out / "pooling_sweep.json").write_text(json.dumps(report, indent=2))
    (out / "provenance.json").write_text(json.dumps(
        provenance(args, inputs=[args.annotated]), indent=2))
    print(f"\nReport -> {out/'pooling_sweep.json'}")
    print("Interpretation: cos≈1 ⇒ steering direction is pooling-robust; cos≪1 or "
          "diverging d_eff ⇒ the geometry is an artefact of the pooling choice and "
          "the choice must be justified (prefer 'last' for context-completeness).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
