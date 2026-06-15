"""
src/predict/ — The Predictive Geometry of Reasoning.

A learned next-step predictor over chain-of-thought reasoning-STEP embeddings.
The object of study is the predictor's RESIDUAL r_t = x_{t+1} - f(x_{<=t}):
where the predictor fails localises branch/backtrack points, and per-chain
residual statistics are the candidate correctness signal (the part that
trajectory-smoothness work — SSP, arXiv:2604.18464 — and static
activation-difference probes — arXiv:2604.05655 — leave open).

Escalation ladder (each rung gated by beating the previous + nulls):
  Rung 0  predictor-free raw-trajectory curvature (src/cbs/trajectory.py)
  Rung 1  linear ridge next-step predictor, NO anti-collapse  <-- this package
  Rung 2  small JEPA predictor (SIGReg / Barlow-Twins)        [deferred, GPU]
  Rung 3  goal-conditioned rollout + causal steering           [deferred, GPU]

Design invariants (from the confound register, CONFOUNDS_AND_REMEDIATION.md):
  * chain-grouped splits everywhere (a chain never trains and tests together) — CF-2
  * read geometry only off the RESIDUAL, never off a regularised latent (no
    isotropy/rank inflation contaminating the measurement)
  * functional/relative claims only (success-vs-failure, predictor-vs-baseline)
  * step-gap is tracked per pair because the saved activations cover only the 4
    Phase-4 behaviours, so a trajectory is a sparse sub-sequence — CF-6/sparsity

Milestone: Predictive Geometry — Rung 1.
"""

from __future__ import annotations

from src.predict.trajectory_dataset import (
    StepDataset,
    build_step_datasets,
    make_supervised_pairs,
    step_shuffle_within_chain,
)
from src.predict.predictor import (
    RidgeConfig,
    ResidualResult,
    oof_residuals,
    chain_residual_features,
)
from src.predict.evaluation import (
    grouped_auc,
    align_labels,
    raw_curvature_features,
    length_features,
    raw_step_features,
    residual_auc_statistic,
)
from src.predict.nulls_predict import (
    step_shuffle_null,
    label_permutation_null,
)

__all__ = [
    "StepDataset",
    "build_step_datasets",
    "make_supervised_pairs",
    "step_shuffle_within_chain",
    "RidgeConfig",
    "ResidualResult",
    "oof_residuals",
    "chain_residual_features",
    "grouped_auc",
    "align_labels",
    "raw_curvature_features",
    "length_features",
    "raw_step_features",
    "residual_auc_statistic",
    "step_shuffle_null",
    "label_permutation_null",
]
