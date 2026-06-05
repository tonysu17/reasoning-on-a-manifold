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

from src.chain_gen import generate_chains, generate_chains_batched, load_chains, load_model
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


def _parse_seeds(seeds_arg: str) -> list[int]:
    return [int(s.strip()) for s in seeds_arg.split(",") if s.strip()]


def _load_tasks_subset(tasks: list[dict], subset_path: str) -> list[dict]:
    """Filter `tasks` to those whose id is in `subset_path` (JSON list)."""
    import json
    with open(subset_path) as f:
        wanted = set(json.load(f))
    return [t for t in tasks if t["id"] in wanted]


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Chain generation")
    parser.add_argument("--model", choices=list(MODELS), default="1.5b",
                        help="Model size (default: 1.5b)")
    parser.add_argument("--tasks", default="data/tasks_final.json",
                        help="Input tasks file (default: data/tasks_final.json)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Max new tokens per chain (default: 8192)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature; 0 = greedy (default: 0.0)")
    parser.add_argument("--seeds", type=str, default="0",
                        help="Comma-separated torch RNG seeds. With one seed "
                             "writes data/chains_{model}.json (legacy path). "
                             "With multiple seeds writes "
                             "data/chains_{model}_multiseed.json with a "
                             "`seed` field per chain (synthesis §M4.5).")
    parser.add_argument("--tasks-subset", type=str, default=None,
                        help="Path to a JSON list of task_ids to restrict to. "
                             "Required for the 100-task multi-seed run "
                             "(synthesis §M4.4).")
    parser.add_argument("--4bit", action="store_true", dest="use_4bit",
                        help="Use 4-bit quantisation (for <8 GB VRAM)")
    parser.add_argument("--cache-dir", default=None,
                        help="HuggingFace model cache directory")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: run on the first 5 tasks only")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Generation batch size. 1 (default) = original "
                             "single-sequence path, unchanged. >1 batches prompts "
                             "(left-padded) for throughput on shared GPUs; resumes "
                             "from the same checkpoint file, so already-generated "
                             "chains are preserved.")
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        # Fall back to legacy path for backward compatibility
        fallback = Path("data/tasks.json")
        if fallback.exists():
            logger.warning(f"{tasks_path} not found — falling back to {fallback}")
            tasks_path = fallback
        else:
            logger.error(f"Tasks not found at {tasks_path}. Run 04_cleanup_tasks.py first.")
            sys.exit(1)

    tasks = load_tasks(tasks_path)
    if args.tasks_subset:
        tasks = _load_tasks_subset(tasks, args.tasks_subset)
        logger.info(f"--tasks-subset: restricted to {len(tasks)} tasks")
    if args.smoke:
        tasks = tasks[:5]
        logger.info(f"SMOKE TEST: running on {len(tasks)} tasks only")

    seeds = _parse_seeds(args.seeds)
    multi_seed = len(seeds) > 1
    model_cfg = MODELS[args.model]
    if multi_seed:
        out_path = Path(f"data/chains_{model_cfg['short']}_multiseed.json")
        dedup_keys = ("task_id", "seed")
    else:
        out_path = Path(f"data/chains_{model_cfg['short']}.json")
        dedup_keys = ("task_id",)

    if out_path.exists() and not multi_seed:
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

    total = 0
    for seed in seeds:
        logger.info(f"== seed {seed}, temperature {args.temperature}, "
                    f"batch_size {args.batch_size} ==")
        if args.batch_size > 1:
            chains = generate_chains_batched(
                model, tokenizer, tasks,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                save_path=out_path,
                batch_size=args.batch_size,
                seed=seed,
                dedup_keys=dedup_keys,
            )
        else:
            chains = generate_chains(
                model, tokenizer, tasks,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                save_path=out_path,
                seed=seed,
                dedup_keys=dedup_keys,
            )
        total = len(chains)

    success = sum(1 for c in load_chains(out_path) if c["n_tokens"] > 0) if out_path.exists() else 0
    logger.info(f"Done: {success}/{total} chains generated → {out_path}")


if __name__ == "__main__":
    main()
