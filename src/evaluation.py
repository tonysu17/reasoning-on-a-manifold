"""
Phase 7 evaluation metrics.

After steering, we re-annotate steered outputs with the annotator LLM
(Claude Sonnet via the lab proxy — src/annotation.py) and compare:
  (a) Behavioural shift     — fraction of target behaviour in output
  (b) Accuracy preservation — task-solving ability on math benchmarks
  (c) Saturation curves     — behaviour fraction vs. α
  (d) Generalisation        — held-out task categories

Comparisons: vanilla (shared baseline) vs. single_direction vs.
manifold_projected vs. random_direction (norm-matched control).
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
    n_target = sum(1 for a in annotated_chain if a["label"] == target)
    return n_target / len(annotated_chain)


#: Generated chains shorter than this many tokens count as degenerate output.
DEGENERATE_TOKEN_FLOOR = 32


def repetition_rate(text: str, n: int = 4) -> float:
    """1 − distinct/total whitespace n-grams. High values flag the repetition
    loops that destructive steering produces; too-short text scores 1.0."""
    toks = text.split()
    total = len(toks) - n + 1
    if total <= 0:
        return 1.0
    distinct = len({tuple(toks[i:i + n]) for i in range(total)})
    return 1.0 - distinct / total


def aggregate_results(
    steered_results: list[dict],
    annotated_steered: list[dict],
    target_behaviours: Optional[list[str]] = None,
) -> dict:
    """
    Aggregate behaviour fractions per (behaviour, method, alpha).

    Args:
        steered_results:   output of run_steering_experiment()
        annotated_steered: steered results re-annotated by the annotator LLM
                           (same structure; each record has "annotations" list)

    Degenerate-record handling (changed 2026-06-12):
      * A MISSING re-annotation (no record for the key) is SKIPPED and counted
        in "n_missing" — previously it silently scored 0.0, deflating whichever
        arm produced more annotation failures (which correlates with how
        destructive that arm's steering is — exactly the comparison under study).
      * A present-but-EMPTY annotation list is skipped into "n_empty" (an empty
        list on a non-empty chain is an annotator failure, not a measured
        absence of the behaviour).
      * Records with behaviour == "shared" (the hoisted vanilla baseline, one
        generation per task) expand into a vanilla data point for EVERY target
        behaviour.

    Returns:
        Nested dict:
          {behaviour: {method: {alpha: {"mean", "std", "n", "n_missing", "n_empty"}}}}
    """
    if target_behaviours is None:
        from src.annotation import TARGET_BEHAVIOURS
        target_behaviours = TARGET_BEHAVIOURS

    SHARED = "shared"  # = src.steered_inference.SHARED_BASELINE

    # Index annotations by (task_id, behaviour, method, alpha)
    ann_index: dict[tuple, list] = {}
    for r in annotated_steered:
        key = (r["task_id"], r["behaviour"], r["method"], r["alpha"])
        ann_index[key] = r.get("annotations", [])

    # Compute fractions + annotation-free generation metrics
    fractions: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    n_missing: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    n_empty: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    # Off-target damage metrics computed straight from the generated text (no
    # annotation spend): mean length, repetition, degenerate-output rate. A
    # steering arm can "reduce the behaviour" by simply wrecking generation;
    # these are the cheap controls that distinguish suppression from damage.
    gen_tokens: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    gen_rep: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    gen_degen: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for r in steered_results:
        method = r["method"]
        alpha = r["alpha"]
        key = (r["task_id"], r["behaviour"], method, alpha)
        anns = ann_index.get(key)
        n_tok = r.get("n_tokens")
        rep = repetition_rate(r.get("chain", ""))
        # Shared vanilla counts toward every behaviour's baseline series.
        targets = list(target_behaviours) if r["behaviour"] == SHARED else [r["behaviour"]]
        for beh in targets:
            if n_tok is not None:
                gen_tokens[beh][method][alpha].append(n_tok)
                gen_degen[beh][method][alpha].append(n_tok < DEGENERATE_TOKEN_FLOOR)
            gen_rep[beh][method][alpha].append(rep)
            if anns is None:
                n_missing[beh][method][alpha] += 1
            elif len(anns) == 0:
                n_empty[beh][method][alpha] += 1
            else:
                fractions[beh][method][alpha].append(behaviour_fraction(anns, beh))

    total_missing = sum(v for b in n_missing.values() for m in b.values() for v in m.values())
    total_empty = sum(v for b in n_empty.values() for m in b.values() for v in m.values())
    if total_missing or total_empty:
        logger.warning(f"aggregate_results: skipped {total_missing} missing and "
                       f"{total_empty} empty re-annotations (reported per cell as "
                       f"n_missing/n_empty — NOT scored as 0.0)")

    # Summarise over the UNION of cell keys. Iterating `fractions` alone made a
    # cell whose re-annotations ALL failed vanish from the summary entirely —
    # and total annotation failure concentrates in exactly the most destructive
    # arm the accounting exists to expose.
    def _cells(store):
        return {(b, m, a) for b, ms in store.items()
                for m, als in ms.items() for a in als}

    all_cells = (_cells(fractions) | _cells(n_missing) | _cells(n_empty)
                 | _cells(gen_tokens) | _cells(gen_rep))
    summary: dict = {}
    for beh, method, alpha in sorted(all_cells,
                                     key=lambda c: (c[0], c[1], float(c[2]))):
        vals = fractions.get(beh, {}).get(method, {}).get(alpha, [])
        cell = {
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals)) if vals else None,
            "n": len(vals),
            "n_missing": int(n_missing.get(beh, {}).get(method, {}).get(alpha, 0)),
            "n_empty": int(n_empty.get(beh, {}).get(method, {}).get(alpha, 0)),
        }
        toks = gen_tokens.get(beh, {}).get(method, {}).get(alpha, [])
        if toks:
            cell["mean_n_tokens"] = float(np.mean(toks))
        degs = gen_degen.get(beh, {}).get(method, {}).get(alpha, [])
        if degs:
            cell["degenerate_rate"] = float(np.mean(degs))
        reps = gen_rep.get(beh, {}).get(method, {}).get(alpha, [])
        if reps:
            cell["repetition_rate"] = float(np.mean(reps))
        summary.setdefault(beh, {}).setdefault(method, {})[alpha] = cell
    return summary


def print_summary_table(summary: dict) -> None:
    for beh, methods in summary.items():
        print(f"\n{'='*60}")
        print(f"Behaviour: {beh}")
        print(f"{'Alpha':>6s} {'vanilla':>10s} {'single_dir':>12s} {'manifold':>12s} {'random':>12s}")
        print("─" * 58)
        alphas = sorted({a for m in methods.values() for a in m})
        for alpha in alphas:
            row = f"{alpha:>6.1f}"
            for method in ("vanilla", "single_direction", "manifold_projected",
                           "random_direction"):
                cell = methods.get(method, {}).get(alpha)
                if cell is None:
                    row += f"  {'n/a':>12s}"
                elif cell.get("mean") is None:
                    # generated but every re-annotation failed — surface it
                    n_fail = cell.get("n_missing", 0) + cell.get("n_empty", 0)
                    row += f"  {f'FAIL×{n_fail}':>12s}"
                else:
                    row += f"  {cell['mean']:>8.3f}±{cell['std']:.3f}"
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
