"""
src/cbs/comparison.py — Cross-model baseline replication.

Purpose
-------
Orchestration code for re-running M1-M4 against Qwen-2.5-Math-1.5B (the
empirically-verified base of R1-Distill-Qwen-1.5B; see
`results/base_model_verification/verdict.md`), plus the cross-model
comparison tables.

The distilled-vs-base contrast is the empirically loaded test of the
"distillation creates the geometry" vs "distillation reveals a pre-existing
geometry" hypotheses.

Prerequisites
-------------
Extension A pipeline on QwenMath-1.5B: Phase 2b chain generation, Phase 3
annotation, Phase 4 activation extraction.

Validation
----------
All M2 / M3 / M4 sanity checks must pass on baseline data; cross-model
differences must survive bootstrap CI. If the baseline produces too few
tier-3 sentences (< 3% rate), report as a structured null finding rather
than forcing parallel statistical tests on incommensurable sample sizes.

Milestone
---------
M6 (synthesis §M6). No smoke run - orchestration only at build time.
"""

from __future__ import annotations

import numpy as np


def cross_model_compare(r1_results: dict, base_results: dict) -> dict:
    """Per geometric statistic: report
        {value_r1, value_base, delta, bootstrap_p}.
    """
    raise NotImplementedError("Filled in at M6 (synthesis §M6.3).")


def trajectory_wasserstein(
    r1_success: np.ndarray,
    r1_failure: np.ndarray,
    base_success: np.ndarray,
    base_failure: np.ndarray,
) -> dict:
    """2-Wasserstein distance between (success | failure) trajectory-stat
    distributions, in each model. Uses the POT library."""
    raise NotImplementedError("Filled in at M6 (synthesis §M6.3).")


def cross_model_classifier(
    r1_train: np.ndarray,
    r1_labels: np.ndarray,
    base_test: np.ndarray,
    base_labels: np.ndarray,
) -> float:
    """Train logistic regression on R1 success/failure trajectory features;
    test on base; return transfer accuracy."""
    raise NotImplementedError("Filled in at M6 (synthesis §M6.3).")
