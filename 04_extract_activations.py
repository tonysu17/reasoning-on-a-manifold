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

MODELS = {
    "1.5b": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "R1-1.5B", "float16"),
    "7b":   ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   "R1-7B",   "float16"),
    "8b":   ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B",  "R1-8B",   "float16"),
}


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Activation extraction")
    parser.add_argument("--model", choices=list(MODELS), default="1.5b")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Specific layers to extract (default: all)")
    parser.add_argument("--4bit", action="store_true", dest="use_4bit")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: use first 20 annotated chains only")
    args = parser.parse_args()

    model_id, short, dtype = MODELS[args.model]
    annotated_path = Path(f"data/annotated_{short}.json")
    if not annotated_path.exists():
        logger.error(f"Annotations not found at {annotated_path}. Run 03_annotate_chains.py first.")
        sys.exit(1)

    save_dir = Path(f"data/activations/{short}")
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

    logger.info(f"Loading model: {model_id}")
    model, tokenizer = load_model(model_id, dtype=dtype, use_4bit=args.use_4bit,
                                  cache_dir=args.cache_dir)

    extract_activations(
        model=model,
        tokenizer=tokenizer,
        annotated_chains=annotated,
        layers=args.layers,
        save_dir=save_dir,
    )

    logger.info(f"Done. Activations saved to {save_dir}")
    logger.info("Next step: run  05_pca_analysis.py")


if __name__ == "__main__":
    main()
