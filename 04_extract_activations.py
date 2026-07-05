#!/usr/bin/env python3
"""
Phase 4 — Activation extraction.

Re-runs annotated chains through DeepSeek-R1-Distill with residual-stream
hooks to extract per-behaviour activation matrices at every layer.
Output: data/activations/<model>/

Requirements:
  pip install .[gpu]
  Input: data/annotated_<model>.json  (from Phase 3)

Runtime: ~3–4 hours for 1000 chains × 28 layers on RTX 4090 (1.5B model)
Storage: ~500 MB per model (float32 activations)
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.activation_extraction import extract_activations
from src.annotation import load_annotated
from src.chain_gen import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Single source of truth: configs/config.yaml (keyed by each model's cli_alias).
from src.config import MODELS_BY_CLI, model_tuple
MODELS = {alias: model_tuple(alias) for alias in MODELS_BY_CLI}


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Activation extraction")
    parser.add_argument("--model", choices=list(MODELS), default="1.5b")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Specific layers to extract (default: all)")
    parser.add_argument("--4bit", action="store_true", dest="use_4bit")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: use first 20 annotated chains only")
    parser.add_argument("--max-chains", type=int, default=None,
                        help="Pilot subset: use only the first N annotated chains; "
                             "output goes to a separate -pilotN dir so a full run "
                             "is never mistaken for done")
    parser.add_argument("--tokenizer-alias", default=None, choices=list(MODELS),
                        help="Tokenize with ANOTHER registered model's tokenizer "
                             "(cross-checkpoint geometry diffs need byte-identical "
                             "input_ids; e.g. STAR1's tokenizer prepends an extra "
                             "BOS the base does not — pass '--tokenizer-alias 1.5b')")
    parser.add_argument("--model-path", default=None,
                        help="Load an arbitrary local checkpoint (e.g. a merged LoRA "
                             "dose) instead of a registered model; requires "
                             "--short-name and inherits dtype from --model")
    parser.add_argument("--short-name", default=None,
                        help="Save-dir name for --model-path (data/activations/<name>); "
                             "annotations resolve from the --model registry entry")
    args = parser.parse_args()

    model_id, short, dtype = MODELS[args.model]
    if args.model_path:
        if not args.short_name:
            parser.error("--model-path requires --short-name")
        model_id = args.model_path
        # annotations stay keyed to the registry model (same chains through the
        # fine-tuned checkpoint is the point of the spillover design)
        annotated_short = short
        short = args.short_name
    else:
        annotated_short = None
    annotated_path = Path(f"data/annotated_{annotated_short or short}.json")
    if not annotated_path.exists():
        logger.error(f"Annotations not found at {annotated_path}. Run 03_annotate_chains.py first.")
        sys.exit(1)

    dir_suffix = f"-pilot{args.max_chains}" if args.max_chains else ""
    save_dir = Path(f"data/activations/{short}{dir_suffix}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already done — but only if every target behaviour got non-zero
    # extractions, otherwise re-run (avoids silently accepting a partially
    # botched extraction).
    meta_path = save_dir / "metadata.json"
    if meta_path.exists() and not args.smoke:
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        n_ext = meta.get("n_extracted", {})
        from src.annotation import TARGET_BEHAVIOURS
        if all(n_ext.get(b, 0) > 0 for b in TARGET_BEHAVIOURS):
            logger.info(f"Activations already present at {save_dir}")
            logger.info(f"  Extracted: {n_ext}")
            return
        logger.warning(
            f"metadata.json exists but some behaviours have 0 extractions: {n_ext}. "
            f"Re-running extraction."
        )

    annotated = load_annotated(annotated_path)
    if args.smoke:
        annotated = annotated[:20]
        logger.info(f"SMOKE TEST: extracting from {len(annotated)} chains")
    elif args.max_chains:
        annotated = annotated[: args.max_chains]
        logger.info(f"PILOT SUBSET: extracting from first {len(annotated)} chains")

    logger.info(f"Loading model: {model_id}")
    model, tokenizer = load_model(model_id, dtype=dtype, use_4bit=args.use_4bit,
                                  cache_dir=args.cache_dir)
    if args.tokenizer_alias:
        from transformers import AutoTokenizer
        tok_id = MODELS[args.tokenizer_alias][0]
        logger.info(f"Tokenizer override: using {tok_id} (identical input_ids "
                    f"across checkpoints for geometry diffs)")
        tokenizer = AutoTokenizer.from_pretrained(tok_id, cache_dir=args.cache_dir)

    extract_activations(
        model=model,
        tokenizer=tokenizer,
        annotated_chains=annotated,
        layers=args.layers,
        save_dir=save_dir,
        keep_in_memory=False,   # runner ignores the return; don't hold ~6.5GB at concat
    )

    logger.info(f"Done. Activations saved to {save_dir}")
    logger.info("Next step: run  05_pca_analysis.py")


if __name__ == "__main__":
    main()
