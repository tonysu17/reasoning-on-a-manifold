#!/usr/bin/env python3
"""
Phase 2 — Reasoning chain generation.

Runs DeepSeek-R1-Distill on the task corpus to produce reasoning chains.
Output: data/chains_<model>.json

Requirements:
  pip install .[gpu]                     (torch, transformers, accelerate)
  GPU: ≥4 GB VRAM for 1.5B; ≥18 GB for 7B
  Input: data/tasks.json  (from Phase 1)

Runtime:  ~2–3 hours for 1000 tasks on RTX 4090 (1.5B model)
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.chain_gen import generate_chains, load_chains, load_model
from src.task_gen import load_tasks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS = {
    "1.5b": {
        "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "short": "R1-1.5B",
        "dtype": "float16",
    },
    "7b": {
        "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "short": "R1-7B",
        "dtype": "float16",
    },
    "8b": {
        "id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "short": "R1-8B",
        "dtype": "float16",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Chain generation")
    parser.add_argument("--model", choices=list(MODELS), default="1.5b",
                        help="Model size (default: 1.5b)")
    parser.add_argument("--tasks", default="data/tasks.json",
                        help="Input tasks file (default: data/tasks.json)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max new tokens per chain (default: 2048)")
    parser.add_argument("--4bit", action="store_true", dest="use_4bit",
                        help="Use 4-bit quantisation (for <8 GB VRAM)")
    parser.add_argument("--cache-dir", default=None,
                        help="HuggingFace model cache directory")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: run on the first 5 tasks only")
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        logger.error(f"Tasks not found at {tasks_path}. Run 01_generate_tasks.py first.")
        sys.exit(1)

    tasks = load_tasks(tasks_path)
    if args.smoke:
        tasks = tasks[:5]
        logger.info(f"SMOKE TEST: running on {len(tasks)} tasks only")

    model_cfg = MODELS[args.model]
    out_path = Path(f"data/chains_{model_cfg['short']}.json")

    if out_path.exists():
        existing = load_chains(out_path)
        if len(existing) >= len(tasks):
            logger.info(f"All chains already present at {out_path}")
            return
        logger.info(f"Found {len(existing)} existing chains — will resume")

    logger.info(f"Loading model: {model_cfg['id']}")
    model, tokenizer = load_model(
        model_cfg["id"],
        dtype=model_cfg["dtype"],
        use_4bit=args.use_4bit,
        cache_dir=args.cache_dir,
    )

    chains = generate_chains(
        model, tokenizer, tasks,
        max_new_tokens=args.max_tokens,
        temperature=0.0,
        save_path=out_path,
    )

    success = sum(1 for c in chains if c["n_tokens"] > 0)
    logger.info(f"Done: {success}/{len(chains)} chains generated → {out_path}")


if __name__ == "__main__":
    main()
