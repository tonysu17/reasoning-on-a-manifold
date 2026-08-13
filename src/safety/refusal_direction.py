"""Refusal-direction geometry — the S4 "post-training fingerprint" engine.

Thesis (see ../../safety_reasoning_extension.md, §3 H2/H4): the *shape* of the
safety object in activation space is a fingerprint of the post-training recipe.
This module is the measurement engine for that claim. It:

  1. extracts the refusal direction (Arditi-style diff-of-means of harmful vs
     harmless activations) per model/recipe,
  2. measures how *sharply* it separates harmful from harmless — the per-recipe
     fingerprint (Cohen's d / AUROC / diff-of-means norm). Prediction:
     deliberative-alignment (gpt-oss) and RLHF (Qwen-Instruct) carve a sharp
     direction; reasoning-distillation (R1-Distill) leaves a weak/absent one,
  3. compares directions and subspaces ACROSS recipes — cosine + principal
     angles for same-ambient models (H4 output-convergence), and linear CKA for
     cross-architecture pairs of *different* width (gpt-oss 2880-d vs R1 1536-d),
  4. measures whether safety-direction engagement scales with gpt-oss reasoning
     effort (the within-model H2 knob).

Everything operates on numpy activation matrices ``(N, d)`` — the same per-layer
arrays produced by ``04_extract_activations`` — so it composes with the existing
pipeline and is testable on synthetic data with known geometry.

Reuses: ``src.steering.single_direction_vector`` (diff-of-means) and
``src.cbs.geometry.principal_angles`` (subspace comparison).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.steering import single_direction_vector

logger = logging.getLogger(__name__)

_EPS = 1e-12


# ── Direction extraction & application ────────────────────────────────────────

def refusal_direction(harmful: np.ndarray, harmless: np.ndarray) -> np.ndarray:
    """Unit refusal direction = diff-of-means(harmful, harmless) (Arditi 2024).

    harmful/harmless: (N, d) and (M, d) residual-stream activations at a layer.
    Returns a unit vector (d,) oriented so that harmful projects *higher*.
    """
    return single_direction_vector(
        np.asarray(harmful, dtype=float), np.asarray(harmless, dtype=float)
    )


def project(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Scalar projection of each row of *acts* onto *direction* → (N,)."""
    acts = np.asarray(acts, dtype=float)
    d = np.asarray(direction, dtype=float)
    nd = np.linalg.norm(d)
    if nd < _EPS:
        return np.zeros(acts.shape[0])
    return acts @ (d / nd)


def directional_ablation(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Project the *direction* component OUT of every row (Arditi ablation).

    acts' = acts − (acts · r̂) r̂. Used to causally test whether removing the
    refusal direction disables refusal (S3) without otherwise perturbing acts.
    """
    acts = np.asarray(acts, dtype=float)
    d = np.asarray(direction, dtype=float)
    nd = np.linalg.norm(d)
    if nd < _EPS:
        return acts.copy()
    u = d / nd
    return acts - np.outer(acts @ u, u)


# ── Per-recipe separation fingerprint ─────────────────────────────────────────

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    sp = float(np.sqrt(sp2))
    return float((a.mean() - b.mean()) / sp) if sp > _EPS else 0.0


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC of *pos* scoring higher than *neg* (Mann–Whitney form, tie-safe)."""
    from scipy.stats import rankdata

    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[:n_pos].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def separation(
    harmful: np.ndarray,
    harmless: np.ndarray,
    direction: Optional[np.ndarray] = None,
) -> dict:
    """How sharply the refusal direction separates harmful from harmless — the
    per-recipe fingerprint. If *direction* is None it is extracted from the data.

    Returns {cohens_d, auroc, dom_norm, n_harmful, n_harmless}. A large
    cohens_d/auroc ⇒ a sharp, post-training-installed refusal axis; ~0 / ~0.5 ⇒
    weak or absent (the reasoning-distillation prediction).
    """
    harmful = np.asarray(harmful, dtype=float)
    harmless = np.asarray(harmless, dtype=float)
    if direction is None:
        direction = refusal_direction(harmful, harmless)
    p_h = project(harmful, direction)
    p_l = project(harmless, direction)
    dom = harmful.mean(axis=0) - harmless.mean(axis=0)
    return {
        "cohens_d": _cohens_d(p_h, p_l),
        "auroc": _auroc(p_h, p_l),
        "dom_norm": float(np.linalg.norm(dom)),
        "n_harmful": int(harmful.shape[0]),
        "n_harmless": int(harmless.shape[0]),
    }


def recipe_fingerprint(per_recipe: dict) -> dict:
    """H2 headline table. per_recipe: {recipe: {"harmful": (N,d), "harmless":(M,d)}}
    → {recipe: separation(...)}. Compare cohens_d/auroc across recipes."""
    return {name: separation(d["harmful"], d["harmless"])
            for name, d in per_recipe.items()}


# ── Cross-recipe comparison (same ambient dim) ────────────────────────────────

def recipe_direction_cosine(dir_a: np.ndarray, dir_b: np.ndarray) -> float:
    """|cos| between two refusal directions (sign is arbitrary in diff-of-means).
    H4 output-convergence test: ~1 ⇒ the recipes share a refusal axis."""
    a = np.asarray(dir_a, dtype=float)
    b = np.asarray(dir_b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < _EPS or nb < _EPS:
        return 0.0
    return float(abs(a @ b) / (na * nb))


def cross_recipe_cosines(directions: dict) -> dict:
    """Pairwise |cos| between per-recipe refusal directions.
    directions: {recipe: (d,)} → {(recipe_a, recipe_b): cosine}."""
    names = list(directions)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[(a, b)] = recipe_direction_cosine(directions[a], directions[b])
    return out


def category_refusal_subspace(
    harmful_by_category: dict,
    harmless: np.ndarray,
    k: Optional[int] = None,
) -> np.ndarray:
    """Multi-dimensional refusal subspace (Wollschläger 'cone' analogue): the
    orthonormalised span of per-category diff-of-means (mu_category − mu_harmless).

    harmful_by_category: {category: (N_c, d)}; harmless: (M, d).
    Returns an orthonormal basis (d, r) with r = #categories (or k). Use with
    ``recipe_principal_angles`` to compare the refusal *subspace* across recipes.
    """
    harmless = np.asarray(harmless, dtype=float)
    mu_l = harmless.mean(axis=0)
    cols = []
    for cat, H in harmful_by_category.items():
        H = np.asarray(H, dtype=float)
        if H.shape[0] == 0:
            continue
        cols.append(H.mean(axis=0) - mu_l)
    if not cols:
        raise ValueError("no non-empty harmful categories")
    D = np.stack(cols, axis=1)  # (d, n_cat)
    U, _, _ = np.linalg.svd(D, full_matrices=False)
    r = U.shape[1] if k is None else max(1, min(int(k), U.shape[1]))
    return U[:, :r]


def recipe_principal_angles(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    top_k: int = 10,
) -> np.ndarray:
    """Principal angles (radians, ascending) between two refusal subspaces of the
    SAME ambient dimension (e.g. R1-Distill vs Qwen-Instruct vs Qwen-base, all
    1536-d). Inputs are QR-orthonormalised defensively. For cross-architecture
    pairs of different width, use ``linear_cka`` instead."""
    from src.cbs.geometry import principal_angles

    Qa, _ = np.linalg.qr(np.asarray(basis_a, dtype=float))
    Qb, _ = np.linalg.qr(np.asarray(basis_b, dtype=float))
    return principal_angles(Qa, Qb, top_k=top_k)


# ── Cross-architecture comparison (different ambient dim) ─────────────────────

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two activation matrices over the SAME N paired stimuli
    (rows aligned), allowing DIFFERENT widths — the tool for gpt-oss (2880-d) vs
    R1-Distill (1536-d). Invariant to orthogonal transforms and isotropic scaling;
    1.0 ⇒ representationally equivalent up to linear map, ~0 ⇒ unrelated."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"CKA needs paired rows: X has {X.shape[0]}, Y has {Y.shape[0]}")
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic = float(np.linalg.norm(X.T @ Y, ord="fro") ** 2)
    nx = float(np.linalg.norm(X.T @ X, ord="fro"))
    ny = float(np.linalg.norm(Y.T @ Y, ord="fro"))
    if nx < _EPS or ny < _EPS:
        return 0.0
    return hsic / (nx * ny)


# ── gpt-oss reasoning-effort engagement (within-model H2 knob) ────────────────

def effort_engagement(acts_by_effort: dict, direction: np.ndarray) -> dict:
    """Mean refusal-direction projection per reasoning-effort level.
    acts_by_effort: {"low"/"medium"/"high": (N, d)} → {effort: mean projection}.
    Deliberative alignment predicts engagement increases low→medium→high."""
    return {eff: float(project(acts, direction).mean())
            for eff, acts in acts_by_effort.items()}


__all__ = [
    "refusal_direction", "project", "directional_ablation",
    "separation", "recipe_fingerprint",
    "recipe_direction_cosine", "cross_recipe_cosines",
    "category_refusal_subspace", "recipe_principal_angles",
    "linear_cka", "effort_engagement",
]
