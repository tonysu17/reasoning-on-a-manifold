#!/usr/bin/env python3
"""
Phase 7 — Steering evaluation.

Applies steering vectors to 50 held-out tasks and measures:
  (a) Behavioural shift   — fraction of target behaviour in steered output
  (b) Saturation curves   — behaviour fraction vs. alpha
  (c) Generalisation      — across task categories

Three conditions compared for each (behaviour, alpha):
  vanilla            — unsteered baseline
  single_direction   — Venhoff-style difference-of-means vector
  manifold_projected — our manifold-projected vector

Output: results/eval/<model>/steering_results.json
         results/eval/<model>/eval_summary.json  (after re-annotation)

Requirements:
  pip install .[gpu]
  Input: data/tasks.json + results/steering_vectors/<model>/

Runtime: ~4–6 hours on RTX 4090
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.chain_gen import load_model
from src.evaluation import aggregate_results, print_summary_table, save_summary
from src.steering import load_steering_vectors
from src.steered_inference import run_steering_experiment
from src.task_gen import load_tasks

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
    parser = argparse.ArgumentParser(description="Phase 7: Steering evaluation")
    parser.add_argument("--model", choices=list(MODELS), default="1.5b")
    parser.add_argument("--n-test", type=int, default=50,
                        help="Held-out tasks for evaluation (default: 50)")
    parser.add_argument("--alpha-values", nargs="+", type=float,
                        default=[0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--4bit", action="store_true", dest="use_4bit")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--skip-annotation", action="store_true",
                        help="Skip GPT-4o re-annotation (just run generation)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 3 tasks, alpha=[0,1] only")
    args = parser.parse_args()

    model_id, short, dtype = MODELS[args.model]
    vectors_dir = Path(f"results/steering_vectors/{short}")
    if not vectors_dir.exists():
        logger.error(f"Steering vectors not found at {vectors_dir}. Run 06_build_steering.py first.")
        sys.exit(1)

    eval_dir = Path(f"results/eval/{short}")
    eval_dir.mkdir(parents=True, exist_ok=True)
    results_path = eval_dir / "steering_results.json"

    tasks = load_tasks(Path("data/tasks.json"))
    # Use last n_test tasks as held-out (they were never used in activation extraction)
    test_tasks = tasks[-args.n_test:]
    if args.smoke:
        test_tasks = test_tasks[:3]
        args.alpha_values = [0.0, 1.0]
        logger.info(f"SMOKE TEST: {len(test_tasks)} tasks, alphas={args.alpha_values}")

    logger.info(f"Loading steering vectors from {vectors_dir}")
    vectors = load_steering_vectors(vectors_dir)

    logger.info(f"Loading model: {model_id}")
    model, tokenizer = load_model(model_id, dtype=dtype, use_4bit=args.use_4bit,
                                  cache_dir=args.cache_dir)

    results = run_steering_experiment(
        model=model,
        tokenizer=tokenizer,
        tasks=test_tasks,
        steering_vectors=vectors,
        alpha_values=args.alpha_values,
        save_path=results_path,
    )

    logger.info(f"Generation complete: {len(results)} outputs → {results_path}")

    if not args.skip_annotation:
        if not os.environ.get("CLAUDE_PROXY_URL") or not os.environ.get("CLAUDE_PROXY_KEY"):
            logger.warning("CLAUDE_PROXY_URL / CLAUDE_PROXY_KEY not set — skipping re-annotation.")
        else:
            from src.annotation import annotate_chains
            ann_path = eval_dir / "annotated_steered.json"
            logger.info("Re-annotating steered outputs …")
            annotated = annotate_chains(results, save_path=ann_path)
            summary = aggregate_results(results, annotated)
            save_summary(summary, eval_dir / "eval_summary.json")
            print_summary_table(summary)

    print(f"\nResults saved → {eval_dir}")


if __name__ == "__main__":
    main()
