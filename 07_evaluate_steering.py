#!/usr/bin/env python3
"""
Phase 7 — Steering evaluation.

Applies steering vectors to 50 held-out tasks and measures:
  (a) Behavioural shift   — fraction of target behaviour in steered output
  (b) Saturation curves   — behaviour fraction vs. alpha
  (c) Generalisation      — across task categories

Arms compared for each (behaviour, alpha) — see src.steered_inference._build_arms:
  vanilla                  — unsteered baseline (one shared generation per task)
  single_direction         — Venhoff-style difference-of-means vector
  manifold_k{1,3,5,10}/auto — HEADLINE k-sweep (every built k, not just auto)
  random_subspace_k{k}     — equal-k random-subspace control (R replicates)
  random_direction         — norm-matched random (sanity floor)
  energy_matched_random    — random rescaled to equal injected energy (real floor)
  orthogonal_complement    — off-subspace (I-P_k)r component alone

Output: results/eval/<model>/steering_results.json
         results/eval/<model>/steering_geometry.json   (per-(beh,k) cos + retained energy)
         results/eval/<model>/generation_metrics.json  (--skip-annotation: damage only)
         results/eval/<model>/eval_summary.json         (after re-annotation)

Workflow: run with --skip-annotation first (generation-first — inspect
degenerate_rate/repetition before paying), then re-run without it to annotate
(it resumes). --annotator-model picks a NON-builder annotator to de-circularise.

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
    parser.add_argument("--steer-mode", choices=["subtract", "add"],
                        default="subtract",
                        help="Intervention sign: 'subtract' = suppression "
                             "(every run before E9.1b), 'add' = amplification "
                             "(E9.1b sign/parity test: h' = h + a(r^T h)r). "
                             "Use a dedicated --out-dir per mode.")
    parser.add_argument("--behaviours", nargs="+", default=None,
                        help="Restrict the run to these behaviours (default: all "
                             "behaviours in the vectors dir). E9.1 uses "
                             "example-testing (collapse-inflating) + backtracking "
                             "(collapse-reducing) only.")
    parser.add_argument("--k-values", nargs="+", type=int, default=None,
                        help="Restrict manifold + random-subspace arms to these "
                             "k values (default: every k the vectors dir carries). "
                             "E9.1 uses 5 only.")
    parser.add_argument("--no-random-control", action="store_true",
                        help="Drop the norm-matched random-direction SANITY-FLOOR "
                             "arm (it injects ~19x less energy than the behaviour "
                             "arm; the causal baseline is energy_matched_random).")
    parser.add_argument("--no-random-subspace", action="store_true",
                        help="Drop the random-SUBSPACE-of-equal-k control "
                             "(random_subspace_k{k}). This is the arm that "
                             "isolates 'the behaviour's PCA subspace matters' "
                             "from 'any k-dim projection + renorm' — keep it.")
    parser.add_argument("--no-energy-matched", action="store_true",
                        help="Drop the energy-matched random control "
                             "(energy_matched_random) — the real generic-"
                             "perturbation floor at equal injected energy.")
    parser.add_argument("--no-orthogonal-complement", action="store_true",
                        help="Drop the orthogonal-complement arm ((I-P_k)r "
                             "alone) that probes whether the discarded "
                             "off-subspace component is pure collateral.")
    parser.add_argument("--n-random-subspaces", type=int, default=3,
                        help="Replicates per random-subspace k (averaged "
                             "downstream); >=3 recommended (default: 3).")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="greedy generation batch size (>1 batches tasks per "
                             "arm to fill an under-utilised GPU; greedy/temp=0 only, "
                             "falls back to per-chain when temperature>0)")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Samples per (task, arm, alpha) at --temperature>0 "
                             "(default: 1 = greedy single sample). The Phase-7 "
                             "design wants ~5; each sample is keyed distinctly "
                             "(#s{j} into task_id) and pooled per cell downstream. "
                             "n_samples>1 REQUIRES --temperature>0.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 = greedy). Set "
                             ">0 (e.g. 0.7) to enable the multi-sample path. The "
                             "shared vanilla baseline is sampled the same way so "
                             "the Delta-vs-vanilla is N-vs-N.")
    parser.add_argument("--sample-seed-base", type=int, default=0,
                        help="Base torch seed for multi-sample draws; sample j "
                             "uses sample_seed_base+j (default: 0). Distinct "
                             "seeds make the N samples genuinely different yet "
                             "reproducible across reruns.")
    parser.add_argument("--skip-annotation", action="store_true",
                        help="GENERATION-FIRST: run all arms and write the "
                             "annotation-free generation metrics (degenerate_"
                             "rate / repetition_rate / mean_n_tokens) so you can "
                             "inspect collapse BEFORE paying for re-annotation. "
                             "Re-run without this flag (resumes) to annotate.")
    parser.add_argument("--annotator-model", default=None,
                        help="Annotator model id for re-annotation. Default = "
                             "src.annotation.ANNOTATION_MODEL (Sonnet 4.5). Set a "
                             "NON-builder annotator (e.g. the proxy id for "
                             "Qwen3-235B) to break the Sonnet-built-and-scores "
                             "circularity flagged in RESULTS_LEDGER §C.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 3 tasks, alpha=[0,1] only")
    parser.add_argument("--vectors-dir", default=None,
                        help="Override the steering-vectors directory (default: "
                             "results/steering_vectors/<short>). Use for a "
                             "per-layer build, e.g. "
                             "results/steering_vectors/R1-1.5B__L16, so a "
                             "layer bake-off reads the right vectors.")
    parser.add_argument("--out-dir", default=None,
                        help="Override the eval output directory (default: "
                             "results/eval/<short>). Set a per-layer dir (e.g. "
                             "results/eval/R1-1.5B__L16) so the two layers' "
                             "bake-off runs don't collide / cross-resume.")
    parser.add_argument("--max-eval-tasks", type=int, default=None,
                        help="Truncate the eval split to the first N tasks "
                             "(applied AFTER the n-test stratified split, like "
                             "--smoke's 3) so a bake-off subset stays a strict "
                             "subset of the same 50-task hold-out the vectors "
                             "excluded — no leakage, no re-split.")
    args = parser.parse_args()

    model_id, short, dtype = MODELS[args.model]
    vectors_dir = (Path(args.vectors_dir) if args.vectors_dir
                   else Path(f"results/steering_vectors/{short}"))
    if not vectors_dir.exists():
        logger.error(f"Steering vectors not found at {vectors_dir}. Run 06_build_steering.py first.")
        sys.exit(1)

    eval_dir = Path(args.out_dir) if args.out_dir else Path(f"results/eval/{short}")
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
    if args.max_eval_tasks is not None:
        test_tasks = test_tasks[:args.max_eval_tasks]
        rule += f" (subset: first {args.max_eval_tasks} of the hold-out split)"
        logger.info(f"Eval subset: first {len(test_tasks)} tasks (subset of the "
                    f"same {args.n_test}-task hold-out — no leakage)")
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
    if args.behaviours:
        unknown = [b for b in args.behaviours if b not in vectors]
        if unknown:
            logger.error(f"--behaviours not in vectors dir: {unknown} "
                         f"(available: {sorted(vectors)})")
            sys.exit(1)
        vectors = {b: vectors[b] for b in args.behaviours}
        logger.info(f"Behaviour filter: running {sorted(vectors)} only")
    if args.k_values:
        keep = set(args.k_values)
        for b, v in vectors.items():
            have = set(v["manifold_projected"])
            missing = keep - have
            if missing:
                logger.error(f"--k-values {sorted(missing)} not built for {b} "
                             f"(available: {sorted(have, key=str)})")
                sys.exit(1)
            v["manifold_projected"] = {
                k: vec for k, vec in v["manifold_projected"].items() if k in keep}
        logger.info(f"k filter: manifold/random-subspace arms at k={sorted(keep)} only")

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
        include_random_subspace=not args.no_random_subspace,
        include_energy_matched=not args.no_energy_matched,
        include_orthogonal_complement=not args.no_orthogonal_complement,
        n_random_subspaces=args.n_random_subspaces,
        geometry_path=eval_dir / "steering_geometry.json",
        n_samples=args.n_samples,
        temperature=args.temperature,
        sample_seed_base=args.sample_seed_base,
        batch_size=args.batch_size,
        steer_mode=args.steer_mode,
    )

    logger.info(f"Generation complete: {len(results)} outputs → {results_path}")
    (eval_dir / "provenance.json").write_text(json.dumps(provenance(args), indent=2))

    if args.skip_annotation:
        # GENERATION-FIRST: emit the annotation-free damage metrics now so the
        # operator can inspect degenerate_rate / repetition_rate / mean_n_tokens
        # per arm BEFORE committing to (paid) re-annotation. aggregate_results
        # with empty annotations populates exactly those cells (on-target `mean`
        # is None until annotated, by design).
        gen_summary = aggregate_results(results, [])
        save_summary(gen_summary, eval_dir / "generation_metrics.json")
        logger.info("Generation-first: wrote annotation-free damage metrics → "
                    "generation_metrics.json. Inspect degenerate_rate/"
                    "repetition_rate, then re-run WITHOUT --skip-annotation "
                    "(it resumes from the saved generations) to annotate.")
    else:
        if not os.environ.get("CLAUDE_PROXY_URL") or not os.environ.get("CLAUDE_PROXY_KEY"):
            logger.warning("CLAUDE_PROXY_URL / CLAUDE_PROXY_KEY not set — skipping re-annotation.")
        else:
            from src.annotation import annotate_chains, ANNOTATION_MODEL
            ann_path = eval_dir / "annotated_steered.json"
            annotator = args.annotator_model or ANNOTATION_MODEL
            logger.info(f"Re-annotating steered outputs with {annotator} …")
            if args.annotator_model and args.annotator_model != ANNOTATION_MODEL:
                logger.info("Using a NON-default annotator — this is the lever "
                            "that breaks the builder-scores-its-own-output "
                            "circularity (RESULTS_LEDGER §C).")
            annotated = annotate_chains(
                results,
                save_path=ann_path,
                dedup_keys=("task_id", "behaviour", "method", "alpha"),
                model=annotator,
            )
            summary = aggregate_results(results, annotated)
            save_summary(summary, eval_dir / "eval_summary.json")
            print_summary_table(summary)

    print(f"\nResults saved → {eval_dir}")


if __name__ == "__main__":
    main()
