#!/usr/bin/env python3
"""Pooling sensitivity sweep — does the geometry depend on how we pool a span?

Methodological caution: Phase 4 reduces each behaviour span to one vector by
MEAN-pooling [1 preceding + first n_execution] tokens. Mean discards order and
can cancel opposing directions; the LAST execution token is the only
context-complete position in a causal model. This re-extracts the SAME chains
under several pooling modes (in ONE shared forward pass) and reports whether the
conclusions move:

  * cos(single_direction[mean], single_direction[last|first]) per behaviour
    — if low, the steering direction is pooling-dependent (a real confound).
  * d_eff_70 per mode — does dimensionality change with the pooling choice?

This is a thin MANUAL trigger for the same one-pass sweep that runs
AUTOMATICALLY on every extraction when `extraction.pooling_sweep` is set in
config.yaml (see src/activation_extraction.extract_activations). It writes
<out>/pooling_sweep.json plus pool_<mode>/ activation dirs.

Requires the model (GPU): `pip install .[gpu]`. Extracts ONLY the steering
layer(s) by default to stay cheap; does NOT touch canonical mean activations.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.activation_extraction import extract_activations, POOLING_MODES
from src.annotation import load_annotated, TARGET_BEHAVIOURS
from src.chain_gen import load_model
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

    # One shared forward pass, all pooling modes (primary = modes[0]); the
    # extractor writes pool_<mode>/ dirs and out/pooling_sweep.json itself.
    extract_activations(model, tokenizer, annotated, layers=args.layers,
                        save_dir=out, pooling=args.modes[0], sweep_modes=args.modes)
    (out / "provenance.json").write_text(json.dumps(
        provenance(args, inputs=[args.annotated]), indent=2))

    report_path = out / "pooling_sweep.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        base = report["primary"]
        others = [m for m in report["modes"] if m != base]
        print(f"\n=== Pooling sensitivity at layer {report['steer_layer']} (baseline = {base}) ===")
        print(f"  {'behaviour':<24}" + "".join(f"cos({base},{m})".rjust(16) for m in others)
              + "   d_eff_70 [" + "/".join(report["modes"]) + "]")
        for beh, rec in report["behaviours"].items():
            cos_cells = "".join(f"{(rec['cos_vs_primary'].get(m) or float('nan')):+.3f}".rjust(16)
                                for m in others)
            deff = "/".join(str(rec["d_eff_70"].get(m, "—")) for m in report["modes"])
            print(f"  {beh:<24}{cos_cells}   {deff}")
        print("\ncos≈1 ⇒ steering direction is pooling-robust; cos≪1 or diverging d_eff ⇒ "
              "the geometry is an artefact of the pooling choice (prefer 'last' for "
              "context-completeness).")
    print(f"\nReport -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
