"""Synthetic data generators with KNOWN ground-truth geometry.

These are the backbone of the core regression suite: every estimator is checked
against data whose intrinsic dimension / flatness / curvature we control, so a
bias or miscalibration shows up as a numeric disagreement rather than passing
silently (which is how the curvature-confound and intrinsic-dim-bias bugs
survived 98 green CBS tests).

All generators take an explicit `seed` and are fully deterministic.
"""

from __future__ import annotations

import numpy as np

AMBIENT = 1536  # Qwen-1.5B hidden dim — the real ambient space


def flat_subspace(n: int, dim: int, ambient: int = AMBIENT, noise: float = 0.0,
                  seed: int = 0) -> np.ndarray:
    """`n` points on a perfectly FLAT linear subspace of dimension `dim`.

    Random Gaussian coordinates mapped through a random linear embedding — no
    curvature. A correct curvature diagnostic must score this ~1.0; a correct
    intrinsic-dimension estimator must recover `dim` (for small `dim`).
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, dim))
    basis = rng.standard_normal((dim, ambient))
    X = Z @ basis
    if noise > 0:
        X = X + noise * rng.standard_normal((n, ambient))
    return X


def sphere(n: int, ambient: int = AMBIENT, noise: float = 1e-3,
           seed: int = 0) -> np.ndarray:
    """`n` points on a CURVED 2-sphere (intrinsic dim 2) embedded in `ambient`.

    Locally ~2-dimensional (tangent plane) but a random sample spans 3 linear
    dimensions, so a correct curvature diagnostic must score < 1.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    X = np.zeros((n, ambient))
    X[:, :3] = v
    return X + noise * rng.standard_normal((n, ambient))


def helix(n: int, ambient: int = AMBIENT, noise: float = 1e-3,
          seed: int = 0) -> np.ndarray:
    """`n` points on a 1-D helix (intrinsic dim 1, curved) embedded in `ambient`."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 6 * np.pi, n)
    X = np.zeros((n, ambient))
    X[:, 0] = np.cos(t)
    X[:, 1] = np.sin(t)
    X[:, 2] = 0.3 * t
    return X + noise * rng.standard_normal((n, ambient))


def chained_labelled(
    n_chains: int = 40,
    per_chain: int = 12,
    target_label: str = "backtracking",
    structure: bool = True,
    ambient: int = AMBIENT,
    seed: int = 0,
):
    """Build (activations, chain_ids, labels) for null-model tests.

    If `structure=True`, sentences carrying `target_label` lie on a tight 1-D
    line (low effective dimension => high top-k variance ratio), while other
    sentences fill many dimensions. A valid structural null must then place the
    real statistic in the upper tail (small p). If `structure=False`, the target
    label is geometry-agnostic and the real value should sit inside the null.
    """
    rng = np.random.default_rng(seed)
    acts, chain_ids, labels = [], [], []
    line_dir = rng.standard_normal(ambient)
    line_dir /= np.linalg.norm(line_dir)
    other_labels = ["deduction", "initializing", "uncertainty-estimation"]
    for c in range(n_chains):
        for _ in range(per_chain):
            is_target = rng.random() < 0.4
            if is_target and structure:
                vec = rng.standard_normal() * line_dir + 0.02 * rng.standard_normal(ambient)
            else:
                vec = rng.standard_normal(ambient)
            acts.append(vec)
            chain_ids.append(f"chain{c}")
            labels.append(target_label if is_target else rng.choice(other_labels))
    return np.array(acts), np.array(chain_ids), np.array(labels)
