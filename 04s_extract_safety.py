#!/usr/bin/env python3
"""Phase 04s — Safety activation extraction (feeds S4).

Extracts harmful/harmless residual-stream activations over a shared stimulus set
for one model, writing the inputs that 14_safety_geometry.py consumes:
    {out-root}/{short}/harmful_layer{L}.npy
    {out-root}/{short}/harmless_layer{L}.npy   (the benign contrast)
    {out-root}/{short}/stimulus_order.json

Run the SAME --stimuli set across every model in the recipe ladder so the rows
align (required for cross-model CKA). For gpt-oss on the DGX Spark:
    python 04s_extract_safety.py --model gpt-oss-20b --stimuli strongreject_xstest.json \
        --reasoning-effort high --attn-impl eager
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import MODELS_BY_CLI
from src.model_adapters import family_of
from src.chain_gen import load_model
from src.safety.stimuli import load_stimuli
from src.safety.extraction import extract_prompt_activations, save_safety_activations

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="Phase 04s: safety activation extraction")
    p.add_argument("--model", default="gpt-oss-20b", help="config cli alias")
    p.add_argument("--stimuli", default="builtin",
                   help="JSON path of harmful/benign prompts, or 'builtin' (placeholders)")
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="layers to extract (default: all)")
    p.add_argument("--out-root", default="data/activations")
    p.add_argument("--reasoning-effort", default="high",
                   help="gpt-oss reasoning effort: low|medium|high")
    p.add_argument("--attn-impl", default=None,
                   help='e.g. "eager" for gpt-oss attention sinks off-Hopper')
    args = p.parse_args()

    spec = MODELS_BY_CLI.get(args.model)
    if spec is None:
        logger.error(f"unknown model alias {args.model!r}; choices: {list(MODELS_BY_CLI)}")
        sys.exit(1)

    family = family_of(spec["id"])
    stimuli = load_stimuli(args.stimuli)
    if args.stimuli == "builtin":
        logger.warning("using BUILTIN placeholder stimuli — supply real harmful "
                       "sets (StrongREJECT/AdvBench/XSTest) via --stimuli for science")
    logger.info(f"{len(stimuli)} stimuli; model {spec['short_name']} (family={family})")

    model, tokenizer = load_model(
        spec["id"], dtype=spec["dtype"], attn_implementation=args.attn_impl
    )
    acts, order = extract_prompt_activations(
        model, tokenizer, stimuli, args.layers,
        family=family, reasoning_effort=args.reasoning_effort,
    )

    out_dir = Path(args.out_root) / spec["short_name"]
    save_safety_activations(acts, out_dir, order)
    for label, by_layer in acts.items():
        n = next(iter(by_layer.values())).shape[0] if by_layer else 0
        logger.info(f"  {label}: {n} stimuli x {len(by_layer)} layers")
    logger.info(f"Wrote {out_dir}. Next: 14_safety_geometry.py")


if __name__ == "__main__":
    main()
