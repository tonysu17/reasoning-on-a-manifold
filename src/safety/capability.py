"""The capability control for H1 (red-team F3; CONFOUNDS_AND_REMEDIATION CF-11).

A "safety object" may be nothing of the sort: harmful and benign prompts differ in
difficulty as well as in safety, so a direction that separates them could be
tracking competence, not caution (Ponkshe et al. 2505.14185: safety subspaces are
not linearly distinct from capability). ch08 makes this control *blocking* —
"no shape is credited to a recipe until the safety geometry has been reported
against a difficulty-matched capability baseline" — but until now it lived only in
prose. This module implements it, with the pass/fail rule pre-registered rather
than read off the result.

The control has two independent legs, both of which must pass:

  * **Partialling.** Estimate a capability direction from hard-vs-easy *non-safety*
    prompts, project it out of the safety activations, and recompute the safety
    separation. If the separation collapses to the hard-vs-easy-benign baseline,
    the "safety" axis was a difficulty axis (failure signature 1).
  * **Alignment.** If the refusal direction is nearly collinear with the capability
    direction, the two are the same axis wearing two labels (failure signature 2).

Difficulty itself must be operationalised model-independently and *before* the
safety activations are seen (e.g. base-model solve-rate, reference-answer length,
or rated step count); :func:`difficulty_matched_indices` then matches harmful to
benign at equal difficulty so the contrast is read at fixed competence.

Pure numpy; reuses the diff-of-means / ablation primitives from
``refusal_direction``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from src.safety.refusal_direction import (
    directional_ablation, recipe_direction_cosine, refusal_direction, separation,
)

# Pre-registered thresholds (F3). After projecting out the capability direction,
# the safety separation must RETAIN at least this fraction of its uncontrolled
# magnitude (else it was mostly capability), and the refusal and capability
# directions must not be near-collinear.
DEFAULT_RETENTION_MIN = 0.50
DEFAULT_ALIGNMENT_MAX = 0.50


def capability_direction(hard: np.ndarray, easy: np.ndarray) -> np.ndarray:
    """Unit capability direction = diff-of-means(hard, easy) over *non-safety*
    prompts graded by difficulty (hard projects higher). Same estimator as the
    refusal direction, applied to a difficulty contrast instead of a safety one."""
    return refusal_direction(np.asarray(hard, float), np.asarray(easy, float))


def partial_out(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove the capability component from every row (Arditi ablation), so the
    residual lives in the complement of the capability axis."""
    return directional_ablation(acts, direction)


def difficulty_matched_indices(
    harmful_difficulty: Sequence[float],
    benign_difficulty: Sequence[float],
    *,
    tolerance: Optional[float] = None,
) -> list[tuple]:
    """Greedy nearest-neighbour matching of harmful↔benign items on a scalar
    difficulty proxy, each benign used at most once. Returns ``[(i_harmful,
    j_benign), ...]``. Pairs further apart than *tolerance* (if given) are
    dropped, so a poorly overlapping difficulty range yields fewer, cleaner pairs.
    """
    h = np.asarray(harmful_difficulty, float)
    b = np.asarray(benign_difficulty, float)
    used = set()
    pairs = []
    for i in np.argsort(h):  # match easiest-first for stability
        diffs = np.abs(b - h[i])
        for j in used:
            diffs[j] = np.inf
        j = int(np.argmin(diffs))
        if diffs[j] == np.inf:
            break
        if tolerance is not None and diffs[j] > tolerance:
            continue
        used.add(j)
        pairs.append((int(i), j))
    return sorted(pairs)


def capability_controlled_separation(
    harmful: np.ndarray,
    harmless: np.ndarray,
    capability_dir: np.ndarray,
) -> dict:
    """Safety separation *after* projecting out the capability direction from both
    sides. Compare its ``cohens_d`` to the uncontrolled separation: a large drop
    means much of the apparent safety axis was capability."""
    h = partial_out(np.asarray(harmful, float), capability_dir)
    l = partial_out(np.asarray(harmless, float), capability_dir)
    return separation(h, l)


def capability_control(
    harmful: np.ndarray,
    harmless: np.ndarray,
    hard_nonsafety: np.ndarray,
    easy_nonsafety: np.ndarray,
    *,
    retention_min: float = DEFAULT_RETENTION_MIN,
    alignment_max: float = DEFAULT_ALIGNMENT_MAX,
) -> dict:
    """Run the full H1 capability control and return a pre-registered verdict.

    Two independent legs, both required to pass:

      * **survives_partialling** — the safety separation retains at least
        ``retention_min`` of its uncontrolled Cohen's d after the capability
        direction is projected out. A collapse means the "safety" axis was
        largely capability (failure signature 1).
      * **axis_distinct** — the refusal direction is not near-collinear with the
        capability direction (|cos| <= ``alignment_max``; failure signature 2).

    The hard-vs-easy difficulty baseline is reported for context (the magnitude of
    a pure difficulty separation) but not used as a gate, since comparing it to the
    partialled safety d across different axes is not apples-to-apples.
    """
    harmful = np.asarray(harmful, float)
    harmless = np.asarray(harmless, float)
    cap_dir = capability_direction(hard_nonsafety, easy_nonsafety)
    ref_dir = refusal_direction(harmful, harmless)

    full = separation(harmful, harmless, direction=ref_dir)
    controlled = capability_controlled_separation(harmful, harmless, cap_dir)
    baseline = separation(np.asarray(hard_nonsafety, float),
                          np.asarray(easy_nonsafety, float), direction=cap_dir)
    alignment = recipe_direction_cosine(ref_dir, cap_dir)

    d_full = abs(full["cohens_d"])
    d_ctrl = abs(controlled["cohens_d"])
    retention = (d_ctrl / d_full) if d_full > 1e-9 else 0.0
    survives = retention >= retention_min
    axis_distinct = alignment <= alignment_max
    return {
        "separation_uncontrolled": full,
        "separation_capability_controlled": controlled,
        "difficulty_baseline": baseline,
        "refusal_capability_alignment": round(alignment, 4),
        "retention": round(float(retention), 4),
        "retention_min": retention_min,
        "alignment_max": alignment_max,
        "survives_partialling": bool(survives),
        "axis_distinct_from_capability": bool(axis_distinct),
        "passed": bool(survives and axis_distinct),
    }


__all__ = [
    "DEFAULT_RETENTION_MIN", "DEFAULT_ALIGNMENT_MAX",
    "capability_direction", "partial_out", "difficulty_matched_indices",
    "capability_controlled_separation", "capability_control",
]
