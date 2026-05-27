"""
src/cbs/ablation.py — CBS steering ablation (causal experiment, M5).

Purpose
-------
Construct v_CBS = normalise(mean(tier3) - mean(tier1)) at the steering layer,
validate it (hard fail-stop), ablate via projection
    h' = h - alpha * (v_cbs^T h) * v_cbs
on textbook-solvable vs bridge-required task sets. Reports the selective
effect on (a) tier-3 frequency, (b) tier-1 frequency, (c) accuracy.

Hard fail-stop (synthesis §M5.3) — all three must hold:
  |cos(v_cbs, v_adding_knowledge_centroid)| < 0.5
  cv_probe_accuracy_mean >= 0.7
  cv_probe_accuracy_std  <= 0.15

If any fails, the runner writes `results/cbs/{model}/FAILSTOP_M5.md` with the
violating numbers and three options for the user. The synthetic-activation
unit tests in src/cbs/tests/test_ablation.py exercise both the passing and
failing paths.

Milestone
---------
M5 (synthesis §M5). Build-now: code + tests; run-phase: actual generation
requires HF model + cluster GPU + Phase 7 answer-checker labels.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.steered_inference import SteeredModel

logger = logging.getLogger(__name__)


# ── v_CBS construction ─────────────────────────────────────────────────────

def build_v_cbs(
    tier3_activations: np.ndarray,
    tier1_activations: np.ndarray,
) -> np.ndarray:
    """v = normalise(mean(tier3, axis=0) - mean(tier1, axis=0))."""
    t3 = np.asarray(tier3_activations)
    t1 = np.asarray(tier1_activations)
    if t3.size == 0 or t1.size == 0:
        raise ValueError(f"non-empty tier3 / tier1 activations required; "
                         f"got shapes t3={t3.shape}, t1={t1.shape}")
    if t3.shape[1] != t1.shape[1]:
        raise ValueError(f"feature-dim mismatch: t3={t3.shape}, t1={t1.shape}")
    diff = t3.mean(axis=0) - t1.mean(axis=0)
    norm = np.linalg.norm(diff)
    if norm <= 0.0:
        raise ValueError("tier3 and tier1 means coincide; v_cbs undefined")
    return (diff / norm).astype(np.float32)


# ── v_CBS validation (hard fail-stop) ──────────────────────────────────────

FAILSTOP_COS_MAX = 0.5
FAILSTOP_PROBE_ACC_MIN = 0.7
FAILSTOP_PROBE_STD_MAX = 0.15


def validate_v_cbs(
    v_cbs: np.ndarray,
    v_adding_knowledge_centroid: np.ndarray,
    tier3_acts: np.ndarray,
    tier1_acts: np.ndarray,
    *,
    cv_folds: int = 5,
    seed: int = 0,
) -> dict:
    """Validate v_CBS against the hard fail-stop conditions.

    Returns
    -------
    {
      "cosine_sim_with_knowledge_centroid": float,
      "cv_probe_accuracy_mean": float,
      "cv_probe_accuracy_std":  float,
      "cv_accuracies":          [float, ...],
      "n_tier3":                int,
      "n_tier1":                int,
      "passes": bool,
      "failures": [str, ...],
    }

    `passes=True` iff all three conditions hold (synthesis §M5.3).
    """
    v_cbs = np.asarray(v_cbs).astype(float).reshape(-1)
    centroid = np.asarray(v_adding_knowledge_centroid).astype(float).reshape(-1)

    # Centroid should be a direction; unit-normalise to be robust.
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm
    v_cbs_norm = np.linalg.norm(v_cbs)
    v_cbs_unit = v_cbs / v_cbs_norm if v_cbs_norm > 0 else v_cbs
    cos_sim = float(v_cbs_unit @ centroid)

    # 5-fold CV linear probe (tier-3 vs tier-1 sentences).
    t3 = np.asarray(tier3_acts)
    t1 = np.asarray(tier1_acts)
    X = np.concatenate([t3, t1], axis=0)
    y = np.concatenate([np.ones(t3.shape[0]), np.zeros(t1.shape[0])])
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        actual_folds = min(cv_folds,
                           int(min(t3.shape[0], t1.shape[0])))
        if actual_folds < 2:
            cv_mean = 0.0
            cv_std = 1.0
            cv_accs: list[float] = []
        else:
            kf = StratifiedKFold(n_splits=actual_folds, shuffle=True,
                                 random_state=seed)
            cv_accs = []
            for train_idx, test_idx in kf.split(X, y):
                clf = LogisticRegression(max_iter=1000, solver="liblinear")
                clf.fit(X[train_idx], y[train_idx])
                cv_accs.append(float(clf.score(X[test_idx], y[test_idx])))
            cv_mean = float(np.mean(cv_accs))
            cv_std = float(np.std(cv_accs))
    except ImportError:
        cv_mean = 0.0
        cv_std = 1.0
        cv_accs = []

    failures: list[str] = []
    if abs(cos_sim) >= FAILSTOP_COS_MAX:
        failures.append(
            f"|cos(v_cbs, v_adding_knowledge_centroid)| = {abs(cos_sim):.3f} "
            f">= {FAILSTOP_COS_MAX}"
        )
    if cv_mean < FAILSTOP_PROBE_ACC_MIN:
        failures.append(
            f"cv_probe_accuracy_mean = {cv_mean:.3f} < {FAILSTOP_PROBE_ACC_MIN}"
        )
    if cv_std > FAILSTOP_PROBE_STD_MAX:
        failures.append(
            f"cv_probe_accuracy_std = {cv_std:.3f} > {FAILSTOP_PROBE_STD_MAX}"
        )

    return {
        "cosine_sim_with_knowledge_centroid": cos_sim,
        "cv_probe_accuracy_mean": cv_mean,
        "cv_probe_accuracy_std": cv_std,
        "cv_accuracies": cv_accs,
        "n_tier3": int(t3.shape[0]),
        "n_tier1": int(t1.shape[0]),
        "passes": len(failures) == 0,
        "failures": failures,
    }


# ── Ablation model ─────────────────────────────────────────────────────────

class CBSAblationModel(SteeredModel):
    """Projection-style intervention.

        h' = h - alpha * (v_cbs^T h) * v_cbs

    At alpha=1.0 this is full projection ablation. v_cbs must be unit-norm
    (raises otherwise). Extends `src/steered_inference.py::SteeredModel`;
    same hook discipline (hook is active only inside `generate()`).
    """

    def __init__(self, model, tokenizer, v_cbs, layer, alpha: float = 1.0):
        v = np.asarray(v_cbs, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(v))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(
                f"v_cbs must be unit-norm for projection ablation; "
                f"got ||v|| = {norm:.4f}. Call build_v_cbs(...) first."
            )
        super().__init__(model, tokenizer, vector=v, layer=int(layer),
                         alpha=float(alpha), mode="subtract")
        self.v_cbs_norm = norm


# ── Task-set construction ──────────────────────────────────────────────────

def construct_task_sets(
    annotated_chains: list[dict],
    *,
    target_per_set: int = 100,
    floor: int = 50,
    correct_field: str = "answer_correct",
) -> dict:
    """Build the two cohorts (synthesis §M5.2):

    A = textbook-solvable: chains where the model is correct in baseline
                            AND has no tier-3 sentences.
    B = bridge-required:   chains where the model is correct in baseline
                            AND has >= 1 tier-3 sentence.

    Note
    ----
    The full M5 spec calls for an extra step on set B: spot-check on 20
    that removing the tier-3 sentence breaks correctness. That step
    requires actual generation under sentence-masking and lives in the
    run-phase `12_cbs_ablation.py`. This function returns the *candidate*
    sets; the runner narrows B further.

    Raises
    ------
    `RuntimeError` if either set has fewer than `floor` candidates — the
    user is told to widen the corpus per synthesis §M5.2.
    """
    set_a: list[dict] = []
    set_b: list[dict] = []
    for chain in annotated_chains:
        if not chain.get(correct_field, False):
            continue
        spans = chain.get("annotations", []) or []
        tier3_count = sum(1 for s in spans
                          if int(s.get("cbs_tier", 0)) == 3)
        if tier3_count == 0:
            set_a.append(chain)
        elif tier3_count >= 1:
            set_b.append(chain)
    if len(set_a) < floor or len(set_b) < floor:
        raise RuntimeError(
            f"Insufficient candidates for M5 task sets: "
            f"|A|={len(set_a)} (need >= {floor}), "
            f"|B|={len(set_b)} (need >= {floor}). "
            f"Widen the chain corpus or re-run with a lower --floor, per "
            f"synthesis §M5.2."
        )
    return {
        "set_a_textbook": set_a[:target_per_set],
        "set_b_bridge": set_b[:target_per_set],
        "n_set_a": len(set_a),
        "n_set_b": len(set_b),
        "target_per_set": int(target_per_set),
        "floor": int(floor),
    }


# ── Selectivity ratio ──────────────────────────────────────────────────────

def selectivity_ratio(delta_tier3: float, delta_tier1: float) -> float:
    """(Δ tier-3 frequency) / (Δ tier-1 frequency).

    Guards against zero denominator: returns nan when |delta_tier1| is below
    a small epsilon to avoid spurious large ratios."""
    eps = 1e-6
    if abs(delta_tier1) < eps:
        return float("nan")
    return float(delta_tier3 / delta_tier1)


__all__ = [
    "FAILSTOP_COS_MAX",
    "FAILSTOP_PROBE_ACC_MIN",
    "FAILSTOP_PROBE_STD_MAX",
    "build_v_cbs",
    "validate_v_cbs",
    "CBSAblationModel",
    "construct_task_sets",
    "selectivity_ratio",
]
