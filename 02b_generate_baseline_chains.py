#!/usr/bin/env python3
"""
Phase 2b — Baseline (non-reasoning) chain generation.

Generates one response per task using Qwen/Qwen2.5-Math-1.5B (the empirically-
verified base model of DeepSeek-R1-Distill-Qwen-1.5B) with a Q/A scaffold
prompt — NO chat template, NO <think> mode. The baseline is the "control"
against which we measure how much reasoning behaviour (backtracking,
uncertainty, example-testing, knowledge-augmentation) is *added* by the R1
distillation process.

Output schema is IDENTICAL to 02_generate_chains.py (same fields, same
prompt+chain reconstruction) so downstream phases (annotation, activation
extraction, PCA, steering) work unchanged via --model-short QwenMath-1.5B.

Prompt format: "Question: {instruction}\n\nAnswer:"

Requires: torch, transformers, accelerate
GPU: RTX 4090 / A100 recommended; 1.5B model fits in ~4 GB VRAM.

Runtime estimate: 3-6 hours for 1000 prompts on shared cluster GPU
(baseline outputs are typically 100-500 tokens, no <think> chain).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from src.chain_gen import load_model
from src.task_gen import load_tasks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


MODELS = {
    "qwen-math-1.5b": (
        "Qwen/Qwen2.5-Math-1.5B",
        "QwenMath-1.5B",
        "float16",
    ),
    # Add others here as needed
}


def format_baseline_prompt(instruction: str) -> str:
    """Q/A scaffold for non-reasoning base models.

    Matches the typical math chain-of-thought corpus format that
    Qwen2.5-Math was pretrained on. NO chat template, NO <think>.
    """
    return f"Question: {instruction}\n\nAnswer:"


def generate_baseline_chain(
    model,
    tokenizer,
    instruction: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
) -> dict:
    """Single baseline generation."""
    import torch

    prompt = format_baseline_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = outputs[0][prompt_len:]
    chain = tokenizer.decode(new_ids, skip_special_tokens=True)

    return {
        "instruction": instruction,
        "prompt": prompt,
        "chain": chain,
        "full_text": prompt + chain,
        "n_tokens": len(new_ids),
    }


def generate_baseline_chains(
    model,
    tokenizer,
    tasks: list[dict],
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    save_path: Optional[Path] = None,
    checkpoint_every: int = 25,
    kill_after: Optional[int] = None,
) -> list[dict]:
    """Batch baseline generation with checkpointing.

    Same schema as src.chain_gen.generate_chains so downstream phases are
    schema-compatible. Resume-safe — re-running picks up from the checkpoint.
    """
    from tqdm import tqdm

    chains: list[dict] = []
    save_path = Path(save_path) if save_path else None

    if save_path and save_path.exists():
        with open(save_path) as f:
            chains = json.load(f)
        logger.info(f"Resuming from checkpoint: {len(chains)}/{len(tasks)} done")

    completed_ids = {c["task_id"] for c in chains}
    new_count = 0

    for task in tqdm(tasks, initial=len(completed_ids), total=len(tasks),
                     desc="Generating baseline chains"):
        tid = task["id"]
        if tid in completed_ids:
            continue

        try:
            result = generate_baseline_chain(model, tokenizer, task["prompt"],
                                              max_new_tokens, temperature)
            record = {
                "task_id":     tid,
                "category":    task.get("category", "unknown"),
                "instruction": task["prompt"],
                "prompt":      result["prompt"],
                "chain":       result["chain"],
                "full_text":   result["full_text"],
                "n_tokens":    result["n_tokens"],
            }
        except Exception as exc:
            logger.warning(f"Task {tid} failed: {exc}")
            record = {
                "task_id":     tid,
                "category":    task.get("category", "unknown"),
                "instruction": task["prompt"],
                "prompt":      "",
                "chain":       "",
                "full_text":   "",
                "n_tokens":    0,
                "error":       str(exc),
            }

        chains.append(record)
        new_count += 1

        if save_path and len(chains) % checkpoint_every == 0:
            _save_json(chains, save_path)
            logger.info(f"  checkpoint: {len(chains)}/{len(tasks)}")

        if kill_after and new_count >= kill_after:
            logger.info(f"  --kill-after {kill_after} reached — saving and exiting")
            if save_path:
                _save_json(chains, save_path)
            break

    if save_path:
        _save_json(chains, save_path)

    success = sum(1 for c in chains if c["n_tokens"] > 0)
    tokens  = [c["n_tokens"] for c in chains if c["n_tokens"] > 0]
    if tokens:
        import statistics
        logger.info(
            f"Done: {success}/{len(chains)} chains generated  "
            f"[mean={statistics.mean(tokens):.0f}, "
            f"min={min(tokens)}, max={max(tokens)} tokens]"
        )
    return chains


def _save_json(data, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.rename(path)


def main():
    parser = argparse.ArgumentParser(description="Phase 2b: Baseline (non-reasoning) chain generation")
    parser.add_argument("--model", choices=list(MODELS), default="qwen-math-1.5b")
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks_final.json"))
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Generation budget (default: 2048; baseline is much shorter than R1's 8192)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Greedy if 0.0 (default), else sample")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 20 tasks only, --kill-after 20")
    parser.add_argument("--kill-after", type=int, default=None,
                        help="Exit after N new chains (for smoke or partial runs)")
    args = parser.parse_args()

    model_id, short, dtype = MODELS[args.model]
    out_path = Path(f"data/chains_{short}.json")
    if args.smoke:
        out_path = Path(f"data/chains_{short}_smoke.json")
        args.kill_after = 20

    logger.info(f"Loading baseline model: {model_id}")
    model, tokenizer = load_model(model_id, dtype=dtype, cache_dir=args.cache_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"  set pad_token = eos_token")

    if not args.tasks.exists():
        logger.error(f"Tasks file not found: {args.tasks}")
        sys.exit(1)

    tasks = load_tasks(args.tasks)
    logger.info(f"Loaded {len(tasks)} tasks from {args.tasks}")
    if args.smoke:
        tasks = tasks[:20]
        logger.info(f"SMOKE TEST: {len(tasks)} tasks → {out_path}")

    generate_baseline_chains(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        save_path=out_path,
        kill_after=args.kill_after,
    )

    logger.info(f"Done. Output → {out_path}")
    logger.info("Next: annotate via 03_annotate_chains.py (uses same schema)")


if __name__ == "__main__":
    main()
