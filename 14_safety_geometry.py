#!/usr/bin/env python3
"""Phase 14 — Safety-reasoning geometry (S4: post-training fingerprint).

Compares refusal-direction geometry across post-training recipes — deliberative
alignment (gpt-oss-20b) vs RLHF (Qwen-Instruct) vs reasoning-distillation
(R1-Distill) vs base — over safety activations extracted per model. See
../safety_reasoning_extension.md (S4 / H2 / H4).

Inputs (a safety extraction pass, per model short_name):
    {activations-root}/{short}/harmful_layer{L}.npy   (N, d)
    {activations-root}/{short}/harmless_layer{L}.npy   (M, d)
  — harmful/harmless rows in the SAME stimulus order across models (so CKA is
    meaningful). Produce these by extracting residual activations over a shared
    harmful/harmless stimulus set (src/safety/stimuli.py) for each model.

Output:
    {out} — JSON with fingerprint, best_layer, cross-recipe cosine, CKA,
            reasoning-effort engagement, a summary table, and a provenance stamp.

Build-now-run-later: with no activations on disk this prints an actionable error.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import MODELS_BY_CLI, provenance, backup_existing
from src.safety.geometry_analysis import build_per_model, analyze_recipes, summarise

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# deliberative-alignment, reasoning-distill (no safety), base — the recipe ladder
DEFAULT_MODELS = ["gpt-oss-20b", "1.5b", "qwen-math-1.5b"]


def main():
    p = argparse.ArgumentParser(description="Phase 14: safety-reasoning geometry (S4)")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help="config cli aliases (see configs/config.yaml)")
    p.add_argument("--activations-root", default="data/activations")
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="layers to analyse (default: infer from files)")
    p.add_argument("--out", default="results/safety/geometry.json")
    args = p.parse_args()

    model_specs: dict = {}
    for alias in args.models:
        spec = MODELS_BY_CLI.get(alias)
        if spec is None:
            logger.warning(f"unknown model alias {alias!r}; skipping")
            continue
        model_specs[spec["short_name"]] = {
            "short_name": spec["short_name"],
            "hidden_dim": spec["hidden_dim"],
        }

    per_model = build_per_model(model_specs, args.activations_root, args.layers)
    if not per_model:
        logger.error(
            "No safety activations found under "
            f"{args.activations_root}/<short>/harmful_layer*.npy. "
            "Run the safety extraction pass first (extract residual activations "
            "over a shared harmful/harmless stimulus set per model)."
        )
        sys.exit(1)

    logger.info(f"Analysing {len(per_model)} model(s): {list(per_model)}")
    result = analyze_recipes(per_model)
    result["summary"] = summarise(result)
    result["provenance"] = provenance(args=args)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    backup_existing(out)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Wrote {out}")
    logger.info("Recipe fingerprint (best layer, by Cohen's d) — H2 readout:")
    for row in result["summary"]:
        logger.info(
            f"  {row['model']:16s} layer {row['best_layer']:>3}  "
            f"d={row['cohens_d']:.2f}  auroc={row['auroc']:.2f}  "
            f"N={row['n_harmful']}/{row['n_harmless']}"
        )


if __name__ == "__main__":
    main()
