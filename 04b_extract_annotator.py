#!/usr/bin/env python3
"""Phase 4 (multi-annotator) — extract activations for a specific annotator's labels.

Same base model (DeepSeek-R1-Distill-Qwen-1.5B); only the annotated file + output
dir change, so each annotator yields its own activation matrices for the manifold-
replication test. Resumable: skips if all target behaviours already extracted.
"""
import argparse, json, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from src.activation_extraction import extract_activations
from src.annotation import load_annotated, TARGET_BEHAVIOURS
from src.chain_gen import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--annotated", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--behaviours", nargs="+", default=TARGET_BEHAVIOURS)
    ap.add_argument("--layers", nargs="+", type=int, default=None)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    meta = save_dir / "metadata.json"
    if meta.exists():
        m = json.load(open(meta))
        if all(m.get("n_extracted", {}).get(b, 0) > 0 for b in args.behaviours):
            logger.info(f"Already extracted at {save_dir}: {m.get('n_extracted')}"); return
        logger.warning(f"Partial metadata at {save_dir}; re-extracting.")

    annotated = load_annotated(Path(args.annotated))
    logger.info(f"Loading model {args.model_id} ({args.dtype})")
    model, tok = load_model(args.model_id, dtype=args.dtype, cache_dir=args.cache_dir)
    extract_activations(model=model, tokenizer=tok, annotated_chains=annotated,
                        layers=args.layers, save_dir=save_dir, behaviours=args.behaviours)
    logger.info(f"Done -> {save_dir}")

if __name__ == "__main__":
    main()
