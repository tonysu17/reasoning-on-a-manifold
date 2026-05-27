"""
src/cbs/matching.py — Matched-pair construction and verification gradient.

Purpose
-------
For the M4 exploration-doc Exp 7 brought back: pair correct-chain tier-3
sentences against incorrect-chain tier-3 sentences on the same task by
Jaccard token similarity >= 0.6, then run Wilcoxon paired tests on
per-sentence geometric statistics. Also computes the verification-gradient
probe.

Validation
----------
* Permutation on success labels collapses effect sizes.
* Match-quality sensitivity sweep at Jaccard >= {0.5, 0.6, 0.7}.
* 5-fold CV accuracy mean/std reported; probe disabled if std > 0.15.

Milestone
---------
M4 (synthesis §M4.2). Blocked from real-data runs until multi-seed chain
re-generation lands (synthesis §M4.4 / §M4.5).
"""

from __future__ import annotations

import numpy as np


def jaccard_token_similarity(a: str, b: str) -> float:
    """|tokens(a) cap tokens(b)| / |tokens(a) cup tokens(b)|, lowercased
    whitespace-tokenised."""
    raise NotImplementedError("Filled in at M4 (synthesis §M4.2).")


def build_matched_pairs(
    success_chains: list[dict],
    failure_chains: list[dict],
    *,
    cbs_tier_filter: int = 3,
    similarity_threshold: float = 0.6,
) -> list[dict]:
    """For each tier-N sentence in a correct chain, find the closest tier-N
    sentence in an incorrect chain on the same task by Jaccard >= threshold.

    Returns: list of {task_id, success_sentence_id, failure_sentence_id,
    jaccard, ...}.
    """
    raise NotImplementedError("Filled in at M4 (synthesis §M4.2).")


def paired_geometric_tests(
    pairs: list[dict],
    activations: dict[str, np.ndarray],
    layer: int,
) -> dict:
    """Wilcoxon paired on centroid distance, OOS residual, projection-onto-
    deduction-subspace."""
    raise NotImplementedError("Filled in at M4 (synthesis §M4.2).")


def verification_gradient(
    correct_acts: np.ndarray,
    incorrect_acts: np.ndarray,
    *,
    cv_folds: int = 5,
    seed: int = 0,
) -> dict:
    """5-fold CV linear probe. Returns
        {probe_weights, cv_accuracy_mean, cv_accuracy_std}.
    Cosine-similarity fields are filled by the caller (M4 runner) once it has
    v_CBS (M5) and the adding-knowledge centroid (M2)."""
    raise NotImplementedError("Filled in at M4 (synthesis §M4.2).")
