#!/usr/bin/env python3
"""Post-training spillover — Step 3: measure the generic-reasoning geometry shift.

Diffs per-behaviour residual-stream geometry between the BASE model and a
SAFETY-POST-TRAINED checkpoint, on the SAME generic (non-safety) annotated
reasoning chains. Two input modes:

  (a) --base-acts DIR --post-acts DIR
      diff two pre-extracted activation dirs (output of 04_extract_activations.py).

  (b) --base-model-id ID --post-model-id PATH --annotated FILE
      extract per-behaviour activations for both models here, then diff.

Outputs results/safety_posttrain/spillover_report.json: per-behaviour principal
angles, Δ effective-dimension, centroid/mean-direction drift, and a PH2
selectivity ranking (which reasoning types moved most).

Scientific caveat (see module docstring of src/safety_posttrain/spillover.py):
these are DESCRIPTIVE signals. The causal claim requires the size-matched
non-safety control run (pt02 on control data) diffed the same way, plus nulls —
run this for BOTH the safety and the control checkpoints and compare.

Examples
--------
    python pt03_measure_spillover.py \
        --base-model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --post-model-id checkpoints/r1_1.5b_safety/dose_all/merged \
        --annotated data/annotated_R1-1.5B.json --out results/safety_posttrain/spillover_safety.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import get_peak_layers, get_target_behaviours, provenance
from src.safety_posttrain import spillover as SP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pt03")


def _extract(model_id: str, annotated_path: str, layers: list[int],
             behaviours: list[str], save_dir: Path, dtype: str) -> dict:
    """Extract per-behaviour activations for one model into save_dir, return them."""
    from src.chain_gen import load_model
    from src.activation_extraction import extract_activations
    import json as _json

    with open(annotated_path) as f:
        annotated = _json.load(f)
    model, tokenizer = load_model(model_id, dtype=dtype)
    return extract_activations(
        model, tokenizer, annotated, layers=layers, save_dir=save_dir,
        behaviours=behaviours, pooling="mean", keep_in_memory=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-acts", help="pre-extracted BASE activation dir")
    ap.add_argument("--post-acts", help="pre-extracted POST activation dir")
    ap.add_argument("--base-model-id", help="HF id / path of the base model (mode b)")
    ap.add_argument("--post-model-id", help="HF id / path of the post-trained model (mode b)")
    ap.add_argument("--annotated", default="data/annotated_R1-1.5B.json")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--behaviours", default=None,
                    help="comma list; default = all 6 annotation labels")
    ap.add_argument("--layers", default=None,
                    help="comma list; default = config peak layers per behaviour")
    ap.add_argument("--k", type=int, default=5, help="subspace dim for principal angles")
    ap.add_argument("--out", default="results/safety_posttrain/spillover_report.json")
    args = ap.parse_args()

    # Default to ALL SIX labels (the spillover study wants the inert controls too).
    if args.behaviours:
        behaviours = [b.strip() for b in args.behaviours.split(",")]
    else:
        from src.annotation import VALID_LABELS
        behaviours = sorted(VALID_LABELS)

    peak = get_peak_layers()
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = sorted(set(int(v) for v in peak.values())) or [14, 17, 27]
    log.info("behaviours=%s  layers=%s  k=%d", behaviours, layers, args.k)

    if args.base_acts and args.post_acts:
        from src.activation_extraction import load_all_activations
        base_acts = load_all_activations(Path(args.base_acts), behaviours, layers)
        post_acts = load_all_activations(Path(args.post_acts), behaviours, layers)
    elif args.base_model_id and args.post_model_id:
        work = Path("data/activations/_spillover")
        log.info("extracting BASE activations (%s)", args.base_model_id)
        base_acts = _extract(args.base_model_id, args.annotated, layers, behaviours,
                             work / "base", args.dtype)
        log.info("extracting POST activations (%s)", args.post_model_id)
        post_acts = _extract(args.post_model_id, args.annotated, layers, behaviours,
                             work / "post", args.dtype)
    else:
        raise SystemExit("provide either --base-acts/--post-acts or "
                         "--base-model-id/--post-model-id")

    report = {"provenance": provenance(args, inputs=[args.annotated]),
              "behaviours": behaviours, "layers": layers, "k": args.k, "by_layer": {}}
    for layer in layers:
        layer_report = SP.compare_geometry(base_acts, post_acts, behaviours, layer, k=args.k)
        layer_report["selectivity_ranking"] = SP.rank_selectivity(layer_report)
        report["by_layer"][str(layer)] = layer_report
        log.info("layer %d selectivity (PH2): %s", layer,
                 layer_report["selectivity_ranking"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    log.info("wrote spillover report -> %s", out)


if __name__ == "__main__":
    main()
