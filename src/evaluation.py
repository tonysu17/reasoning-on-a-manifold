"""
Phase 7 evaluation metrics.

After steering, we re-annotate steered outputs with GPT-4o and compare:
  (a) Behavioural shift     — fraction of target behaviour in output
  (b) Accuracy preservation — task-solving ability on math benchmarks
  (c) Saturation curves     — behaviour fraction vs. α
  (d) Generalisation        — held-out task categories

All comparisons are: vanilla vs. single_direction vs. manifold_projected.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def behaviour_fraction(annotated_chain: list[dict], target: str) -> float:
    """Fraction of sentences in *annotated_chain* classified as *target*."""
    if not annotated_chain:
        return 0.0
    n_target = sum(1 for a in annotated_chain if a["category"] == target)
    return n_target / len(annotated_chain)


def aggregate_results(
    steered_results: list[dict],
    annotated_steered: list[dict],
    target_behaviours: Optional[list[str]] = None,
) -> dict:
    """
    Aggregate behaviour fractions per (behaviour, method, alpha).

    Args:
        steered_results:   output of run_steering_experiment()
        annotated_steered: steered results re-annotated with GPT-4o
                           (same structure; each record has "annotations" list)

    Returns:
        Nested dict:
          {behaviour: {method: {alpha: {"mean": float, "std": float, "n": int}}}}
    """
    if target_behaviours is None:
        from src.activation_extraction import TARGET_BEHAVIOURS
        target_behaviours = TARGET_BEHAVIOURS

    # Index annotations by (task_id, behaviour, method, alpha)
    ann_index: dict[tuple, list] = {}
    for r in annotated_steered:
        key = (r["task_id"], r["behaviour"], r["method"], r["alpha"])
        ann_index[key] = r.get("annotations", [])

    # Compute fractions
    fractions: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for r in steered_results:
        beh = r["behaviour"]
        method = r["method"]
        alpha = r["alpha"]
        tid = r["task_id"]
        key = (tid, beh, method, alpha)
        anns = ann_index.get(key, [])
        frac = behaviour_fraction(anns, beh)
        fractions[beh][method][alpha].append(frac)

    # Summarise
    summary: dict = {}
    for beh in fractions:
        summary[beh] = {}
        for method in fractions[beh]:
            summary[beh][method] = {}
            for alpha, vals in sorted(fractions[beh][method].items()):
                arr = np.array(vals)
                summary[beh][method][alpha] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "n": len(vals),
                }
    return summary


def print_summary_table(summary: dict) -> None:
    for beh, methods in summary.items():
        print(f"\n{'='*60}")
        print(f"Behaviour: {beh}")
        print(f"{'Alpha':>6s} {'vanilla':>10s} {'single_dir':>12s} {'manifold':>12s}")
        print("─" * 44)
        alphas = sorted({a for m in methods.values() for a in m})
        for alpha in alphas:
            row = f"{alpha:>6.1f}"
            for method in ("vanilla", "single_direction", "manifold_projected"):
                cell = methods.get(method, {}).get(alpha)
                if cell:
                    row += f"  {cell['mean']:>8.3f}±{cell['std']:.3f}"
                else:
                    row += f"  {'n/a':>12s}"
            print(row)


def save_summary(summary: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Evaluation summary → {path}")


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
