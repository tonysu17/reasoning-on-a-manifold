"""
src/cbs/matching.py — Matched-pair construction and verification gradient (M4).

Purpose
-------
Brings back exploration-doc Exp 7: pair correct-chain tier-3 sentences against
incorrect-chain tier-3 sentences on the same task by Jaccard token similarity,
then run Wilcoxon paired tests on per-sentence geometric statistics. Also
computes the 5-fold CV verification-gradient probe.

Inputs
------
* Success / failure chain dicts (from Phase 7 answer-checker + multi-seed
  re-generation; see synthesis §M4.4–§M4.5).
* Per-behaviour activation matrices and the corresponding row index from
  `src/cbs/trajectory.py::build_row_index`.

Outputs
-------
* `build_matched_pairs` returns a list of pair-record dicts.
* `paired_geometric_tests` returns Wilcoxon p, Cliff's delta, bootstrap CI
  per geometric statistic.
* `verification_gradient` returns probe weights + 5-fold CV accuracy.

Validation
----------
* Permutation on success labels collapses effect sizes.
* Match-quality sensitivity sweep at Jaccard >= {0.5, 0.6, 0.7}.
* 5-fold CV accuracy mean / std reported; probe disabled if std > 0.15.

Milestone
---------
M4 (synthesis §M4.2). Real-data runs blocked on:
  (1) Phase 7 answer-checker (success/failure labels), and
  (2) multi-seed chain re-generation (chains_R1-1.5B_multiseed.json).
Code construction lives at build-now.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Token similarity ───────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(s: str) -> set[str]:
    return set(_TOKEN_RE.findall(s.lower())) if s else set()


def jaccard_token_similarity(a: str, b: str) -> float:
    """|tokens(a) ∩ tokens(b)| / |tokens(a) ∪ tokens(b)|, lowercased
    alphanumeric tokens. Returns 0.0 when either side is empty."""
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    return float(len(inter)) / float(len(union))


# ── Matched-pair construction ──────────────────────────────────────────────

def _tier_spans(chain: dict, *, tier: int) -> list[dict]:
    """All span records with a given CBS tier from a chain."""
    out = []
    for i, span in enumerate(chain.get("annotations", []) or []):
        if int(span.get("cbs_tier", 0)) == tier:
            out.append({**span, "_chain_id": chain.get("task_id", ""),
                        "_sentence_idx": i})
    return out


def build_matched_pairs(
    success_chains: list[dict],
    failure_chains: list[dict],
    *,
    cbs_tier_filter: int = 3,
    similarity_threshold: float = 0.6,
) -> list[dict]:
    """For each tier-N sentence in a success chain, find the closest tier-N
    sentence in a failure chain on the *same task* by Jaccard ≥ threshold.

    Returns a list of {task_id, success_sentence_id, failure_sentence_id,
    jaccard, success_chain_id, failure_chain_id} records, one per matched
    pair (multiple per task possible).
    """
    # Bucket failure spans by task_id.
    by_task: dict[str, list[tuple[dict, dict]]] = {}
    for fc in failure_chains:
        tid = fc.get("task_id", "")
        for span in _tier_spans(fc, tier=cbs_tier_filter):
            by_task.setdefault(tid, []).append((fc, span))

    pairs: list[dict] = []
    for sc in success_chains:
        tid = sc.get("task_id", "")
        if tid not in by_task:
            continue
        for span in _tier_spans(sc, tier=cbs_tier_filter):
            best: tuple[float, Optional[dict], Optional[dict]] = (
                -1.0, None, None,
            )
            for fc, fspan in by_task[tid]:
                j = jaccard_token_similarity(span.get("text", ""),
                                             fspan.get("text", ""))
                if j > best[0]:
                    best = (j, fc, fspan)
            if best[0] >= similarity_threshold and best[1] is not None \
                    and best[2] is not None:
                pairs.append({
                    "task_id": tid,
                    "success_chain_id": sc.get("task_id", ""),
                    "failure_chain_id": best[1].get("task_id", ""),
                    "success_sentence_id": (f"{span['_chain_id']}:"
                                            f"{span['_sentence_idx']}"),
                    "failure_sentence_id": (f"{best[2]['_chain_id']}:"
                                            f"{best[2]['_sentence_idx']}"),
                    "jaccard": best[0],
                })
    return pairs


# ── Paired geometric tests ────────────────────────────────────────────────

def paired_geometric_tests(
    pairs: list[dict],
    activations_lookup: dict[str, np.ndarray],
    layer: int,
    *,
    deduction_subspace: Optional[np.ndarray] = None,
) -> dict:
    """Wilcoxon paired on centroid distance, OOS residual, and projection-
    onto-deduction-subspace.

    Parameters
    ----------
    pairs                  : output of `build_matched_pairs`.
    activations_lookup     : {sentence_id: (d,) np.ndarray} — the runner builds
                              this from per-behaviour matrices + row_index.
    layer                  : layer index (for metadata; activations are passed
                              already at the right layer).
    deduction_subspace     : (d, k) orthonormal basis for the deduction subspace,
                              if available. Optional — when missing, the
                              projection statistic is skipped.

    Returns
    -------
    A dict per geometric statistic with:
        {
          "n_pairs": int,
          "wilcoxon_p": float,
          "cliffs_delta": float,
          "ci95": [float, float],
          "median_diff": float,
        }
    """
    from src.cbs.geometry import bootstrap_ci, cliffs_delta

    if not pairs:
        return {"layer": layer, "n_pairs": 0,
                "stats": {}, "note": "no pairs"}

    s_acts, f_acts = [], []
    for p in pairs:
        sa = activations_lookup.get(p["success_sentence_id"])
        fa = activations_lookup.get(p["failure_sentence_id"])
        if sa is None or fa is None:
            continue
        s_acts.append(sa)
        f_acts.append(fa)
    if not s_acts:
        return {"layer": layer, "n_pairs": 0,
                "stats": {}, "note": "no matched activations"}

    S = np.stack(s_acts)
    F = np.stack(f_acts)

    centroid = ((S.sum(0) + F.sum(0)) / (2.0 * len(S)))
    cd_success = np.linalg.norm(S - centroid[None, :], axis=1)
    cd_failure = np.linalg.norm(F - centroid[None, :], axis=1)

    s_norm = np.linalg.norm(S, axis=1)
    f_norm = np.linalg.norm(F, axis=1)

    stats_out: dict[str, dict] = {}

    def _wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
        try:
            from scipy.stats import wilcoxon
            _, p = wilcoxon(a, b)
            return float(p)
        except (ImportError, ValueError):
            return float("nan")

    # 1. Centroid distance
    p_cd = _wilcoxon(cd_success, cd_failure)
    delta_cd = cliffs_delta(cd_success, cd_failure)
    try:
        ci_cd = bootstrap_ci(cliffs_delta, cd_success, cd_failure,
                             n_bootstrap=500, paired=False)
    except Exception:  # noqa: BLE001
        ci_cd = (float("nan"), float("nan"))
    stats_out["centroid_distance"] = {
        "n_pairs": int(len(S)),
        "wilcoxon_p": p_cd,
        "cliffs_delta": float(delta_cd),
        "ci95": [float(ci_cd[0]), float(ci_cd[1])],
        "median_diff": float(np.median(cd_success - cd_failure)),
    }

    # 2. Deduction-subspace projection magnitude (optional)
    if deduction_subspace is not None and deduction_subspace.size > 0:
        proj_s = np.linalg.norm(S @ deduction_subspace, axis=1) / np.maximum(s_norm, 1e-12)
        proj_f = np.linalg.norm(F @ deduction_subspace, axis=1) / np.maximum(f_norm, 1e-12)
        p_proj = _wilcoxon(proj_s, proj_f)
        delta_proj = cliffs_delta(proj_s, proj_f)
        try:
            ci_proj = bootstrap_ci(cliffs_delta, proj_s, proj_f,
                                   n_bootstrap=500, paired=False)
        except Exception:  # noqa: BLE001
            ci_proj = (float("nan"), float("nan"))
        stats_out["deduction_projection"] = {
            "n_pairs": int(len(S)),
            "wilcoxon_p": p_proj,
            "cliffs_delta": float(delta_proj),
            "ci95": [float(ci_proj[0]), float(ci_proj[1])],
            "median_diff": float(np.median(proj_s - proj_f)),
        }

    return {"layer": layer, "n_pairs": int(len(S)),
            "stats": stats_out, "note": "ok"}


# ── Cross-validated linear probe (shared by M4 gradient + M5 fail-stop) ─────

def cv_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: "np.ndarray | None" = None,
    *,
    cv_folds: int = 5,
    seed: int = 0,
) -> "tuple[list[float], np.ndarray | None]":
    """CV accuracy of a logistic probe, with optional chain-aware splitting.

    If `groups` (one chain id per row) is provided, uses StratifiedGroupKFold so
    that sentences from the SAME chain never span train and test. Without this,
    multiple sentences per chain leak across folds and inflate accuracy — a real
    hazard here because tier-3 / success sentences cluster within chains. If
    `groups` is None it falls back to StratifiedKFold and logs a leakage warning.

    Returns (accuracies, mean_unit_weight) — `mean_unit_weight` is None when no
    fold could be fit (too few samples or groups).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

    X = np.asarray(X)
    y = np.asarray(y)
    if groups is not None:
        groups = np.asarray(groups)
        n_units = int(np.unique(groups).size)
    else:
        n_units = int(min(np.sum(y == 1), np.sum(y == 0)))
        logger.warning(
            "cv_probe running WITHOUT chain groups: sentences from the same "
            "chain may leak across CV folds and inflate accuracy. Pass `groups` "
            "(per-row chain ids) for a leak-free estimate."
        )
    folds = min(cv_folds, n_units)
    if folds < 2:
        return [], None

    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)

    accs: list[float] = []
    weights: list[np.ndarray] = []
    for train_idx, test_idx in split_iter:
        clf = LogisticRegression(max_iter=1000, solver="liblinear")
        clf.fit(X[train_idx], y[train_idx])
        accs.append(float(clf.score(X[test_idx], y[test_idx])))
        weights.append(clf.coef_[0])
    if not weights:
        return [], None
    mean_w = np.mean(weights, axis=0)
    norm = np.linalg.norm(mean_w)
    if norm > 0:
        mean_w = mean_w / norm
    return accs, mean_w


# ── Verification gradient (5-fold CV probe) ────────────────────────────────

def verification_gradient(
    correct_acts: np.ndarray,
    incorrect_acts: np.ndarray,
    *,
    cv_folds: int = 5,
    seed: int = 0,
    correct_groups: "np.ndarray | None" = None,
    incorrect_groups: "np.ndarray | None" = None,
) -> dict:
    """5-fold-CV linear probe with success/failure labels.

    Returns
    -------
    {
      "probe_weights": (d,) np.ndarray averaged across folds — the
                       "verification gradient" unit-normalised,
      "cv_accuracy_mean": float,
      "cv_accuracy_std":  float,
      "cv_accuracies":    list[float] per fold,
      "n_correct": int,
      "n_incorrect": int,
      "stable": bool   (True iff std <= 0.15),
    }
    """
    correct_acts = np.asarray(correct_acts)
    incorrect_acts = np.asarray(incorrect_acts)
    if correct_acts.size == 0 or incorrect_acts.size == 0:
        return {"probe_weights": np.zeros(0),
                "cv_accuracy_mean": 0.0,
                "cv_accuracy_std": 0.0,
                "cv_accuracies": [],
                "n_correct": int(correct_acts.shape[0]),
                "n_incorrect": int(incorrect_acts.shape[0]),
                "stable": False,
                "note": "empty input"}
    X = np.concatenate([correct_acts, incorrect_acts], axis=0)
    y = np.concatenate([
        np.ones(correct_acts.shape[0]),
        np.zeros(incorrect_acts.shape[0]),
    ])
    groups = None
    if correct_groups is not None and incorrect_groups is not None:
        groups = np.concatenate([np.asarray(correct_groups),
                                 np.asarray(incorrect_groups)])

    empty = {"probe_weights": np.zeros(X.shape[1]),
             "cv_accuracy_mean": 0.0,
             "cv_accuracy_std": 0.0,
             "cv_accuracies": [],
             "n_correct": int(correct_acts.shape[0]),
             "n_incorrect": int(incorrect_acts.shape[0]),
             "stable": False}
    try:
        accs, mean_w = cv_probe(X, y, groups, cv_folds=cv_folds, seed=seed)
    except ImportError:
        return {**empty, "note": "sklearn unavailable"}
    if not accs:
        return {**empty, "note": "too few samples/groups for CV"}

    std_acc = float(np.std(accs))
    return {
        "probe_weights": mean_w,
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std": std_acc,
        "cv_accuracies": accs,
        "n_correct": int(correct_acts.shape[0]),
        "n_incorrect": int(incorrect_acts.shape[0]),
        "stable": bool(std_acc <= 0.15),
        "grouped": groups is not None,
    }


__all__ = [
    "jaccard_token_similarity",
    "build_matched_pairs",
    "paired_geometric_tests",
    "verification_gradient",
    "cv_probe",
]
