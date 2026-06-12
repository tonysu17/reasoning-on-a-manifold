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
  Input: data/tasks_final.json + results/steering_vectors/<model>/

Runtime: ~4–6 hours on RTX 4090
"""

import argparse
import json
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

# Single source of truth: configs/config.yaml (keyed by each model's cli_alias).
from src.config import MODELS_BY_CLI, model_tuple, provenance
MODELS = {alias: model_tuple(alias) for alias in MODELS_BY_CLI}


def main():
    parser = argparse.ArgumentParser(description="Phase 7: Steering evaluation")
    parser.add_argument("--model", choices=list(MODELS), default="1.5b")
    parser.add_argument("--n-test", type=int, default=50,
                        help="Held-out tasks for evaluation (default: 50)")
    parser.add_argument("--alpha-values", nargs="+", type=float,
                        default=[0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--4bit", action="store_true", dest="use_4bit")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Generation cap for steered chains (default: "
                             "config generation.max_new_tokens = 8192 — the "
                             "corpus cap; lower values confound α with "
                             "truncation).")
    parser.add_argument("--no-random-control", action="store_true",
                        help="Drop the norm-matched random-direction control "
                             "arm (NOT recommended: without it the manifold-"
                             "vs-single comparison has no causal baseline).")
    parser.add_argument("--skip-annotation", action="store_true",
                        help="Skip LLM re-annotation (just run generation)")
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

    # Canonical corpus is tasks_final.json (data/tasks.json is a stale May-21
    # snapshot — see data/MANIFEST.md). Reading the stale file desynchronised
    # the eval prompts from the corpus the chains/activations were built on.
    tasks = load_tasks(Path("data/tasks_final.json"))
    # Category-stratified eval set — SHARED with the vector builders (06 /
    # build_phase6), which exclude exactly these tasks' activation rows from
    # vector construction. With default-holdout vectors, Phase 7 is a true
    # out-of-sample test; with --no-holdout vectors it is an on-corpus causal
    # effect (check the vectors' metadata provenance "holdout" field).
    # Residual caveat either way: the steering LAYER choice was informed by
    # full-corpus analyses (and Huang's published layer 27).
    from src.task_gen import stratified_eval_split
    test_tasks, split_rule = stratified_eval_split(tasks, args.n_test)
    cat_counts = {}
    for t in test_tasks:
        cat_counts[t.get("category", "unknown")] = \
            cat_counts.get(t.get("category", "unknown"), 0) + 1
    logger.info(f"Eval split: {len(test_tasks)} tasks ({split_rule}), "
                f"stratified by category: {cat_counts}")
    rule = split_rule
    if args.smoke:
        test_tasks = test_tasks[:3]
        args.alpha_values = [0.0, 1.0]
        rule += " (smoke: truncated to first 3)"
        logger.info(f"SMOKE TEST: {len(test_tasks)} tasks, alphas={args.alpha_values}")
    # Persist the exact eval-task ids for provenance/reproducibility.
    # Counts recomputed from the FINAL set so smoke provenance is truthful.
    final_counts = {}
    for t in test_tasks:
        final_counts[t.get("category", "unknown")] = \
            final_counts.get(t.get("category", "unknown"), 0) + 1
    (eval_dir / "eval_task_ids.json").write_text(
        json.dumps({"task_ids": [t["id"] for t in test_tasks],
                    "category_counts": final_counts,
                    "rule": rule}, indent=2))

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
        max_new_tokens=args.max_new_tokens,   # None → config cap (8192)
        save_path=results_path,
        include_random_control=not args.no_random_control,
    )

    logger.info(f"Generation complete: {len(results)} outputs → {results_path}")
    (eval_dir / "provenance.json").write_text(json.dumps(provenance(args), indent=2))

    if not args.skip_annotation:
        if not os.environ.get("CLAUDE_PROXY_URL") or not os.environ.get("CLAUDE_PROXY_KEY"):
            logger.warning("CLAUDE_PROXY_URL / CLAUDE_PROXY_KEY not set — skipping re-annotation.")
        else:
            from src.annotation import annotate_chains
            ann_path = eval_dir / "annotated_steered.json"
            logger.info("Re-annotating steered outputs …")
            annotated = annotate_chains(
                results,
                save_path=ann_path,
                dedup_keys=("task_id", "behaviour", "method", "alpha"),
            )
            summary = aggregate_results(results, annotated)
            save_summary(summary, eval_dir / "eval_summary.json")
            print_summary_table(summary)

    print(f"\nResults saved → {eval_dir}")


if __name__ == "__main__":
    main()
