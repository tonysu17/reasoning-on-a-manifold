"""
src/cbs/ablation.py — CBS steering ablation (causal experiment).

Purpose
-------
Construct v_CBS = normalise(mean(tier3) - mean(tier1)) at the steering layer
(27 by default; 17 also supported), validate it with a hard fail-stop, then
ablate via projection h' = h - alpha (v^T h) v on textbook-solvable vs
bridge-required task sets. Measures selective effects on tier-3 frequency,
tier-1 frequency, and accuracy.

Validation
----------
HARD FAIL-STOP - all three conditions must hold:
  |cos(v_cbs, v_adding_knowledge_centroid)| < 0.5
  cv_probe_accuracy_mean >= 0.7
  cv_probe_accuracy_std  <= 0.15

If any condition fails, halt and write
`results/cbs/{model}/FAILSTOP_M5.md` with the violating numbers and three
options for the user. Do not silently fall back.

Validation also includes:
* Steering-saturation curve over alpha in {0, 0.5, 1.0, 2.0}.
* Annotation reliability spot-check at strongest ablation.
* Random-vector control (v_random).

Milestone
---------
M5 (synthesis §M5). Gated on Tony's go-ahead before implementation.
"""

from __future__ import annotations

import numpy as np

from src.steered_inference import SteeredModel


def build_v_cbs(
    tier3_activations: np.ndarray,
    tier1_activations: np.ndarray,
) -> np.ndarray:
    """v = normalise(mean(tier3, axis=0) - mean(tier1, axis=0)).
    Returns unit-norm np.ndarray of shape (d,)."""
    raise NotImplementedError("Filled in at M5 (synthesis §M5.2).")


def validate_v_cbs(
    v_cbs: np.ndarray,
    v_adding_knowledge_centroid: np.ndarray,
    tier3_acts: np.ndarray,
    tier1_acts: np.ndarray,
    *,
    cv_folds: int = 5,
    seed: int = 0,
) -> dict:
    """5-fold CV probe + cosine-similarity sanity check.

    Returns:
        {
          cosine_sim_with_knowledge_centroid: float,
          cv_probe_accuracy_mean: float,
          cv_probe_accuracy_std:  float,
          passes: bool,            # all three conditions in module docstring
        }
    The caller is expected to halt and emit a FAILSTOP report on `passes=False`.
    """
    raise NotImplementedError("Filled in at M5 (synthesis §M5.2).")


class CBSAblationModel(SteeredModel):
    """Projection-style intervention.

        h' = h - alpha (v_cbs^T h) v_cbs

    At alpha = 1.0 this is full projection ablation. Implementation extends
    SteeredModel (src/steered_inference.py) - same hook discipline.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Filled in at M5 (synthesis §M5.2).")


def construct_task_sets(annotated_chains: list[dict]) -> dict:
    """Build the two cohorts:

    A = textbook-solvable: chains correct in baseline AND no tier-3 sentences.
    B = bridge-required:   chains correct in baseline AND >= 1 tier-3 sentence
                           AND removing that sentence breaks correctness
                           (spot-checked on 20).

    Targets 100 each. If either < 50, raises with a clear corpus-widening
    message (synthesis §M5.2)."""
    raise NotImplementedError("Filled in at M5 (synthesis §M5.2).")


def selectivity_ratio(delta_tier3: float, delta_tier1: float) -> float:
    """(delta tier-3 frequency) / (delta tier-1 frequency); guarded
    against zero."""
    raise NotImplementedError("Filled in at M5 (synthesis §M5.2).")
