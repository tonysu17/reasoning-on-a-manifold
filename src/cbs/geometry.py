"""
src/cbs/geometry.py — Per-sentence geometric tests for CBS-tier and
cross-domain labels.

Purpose
-------
At each saved layer, test whether CBS tier (and, separately, the binary
cross-domain flag) correlates with four geometric quantities of the
sentence-level activation vector:

  1. centroid distance,
  2. out-of-subspace residual against the union of behaviour subspaces,
  3. local intrinsic dimension via TwoNN over k-NN,
  4. principal angles between behaviour-subspace pairs.

Each test is run twice (once under each label) so both signals are preserved
(synthesis §M2.1).

Validation
----------
Shuffle test (|Cliff's delta| < 0.05); reversal test (JT sign flip);
category-stratified rerun; smoke unit tests.

Milestone
---------
M2 (synthesis §M2).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def centroid_distance(X: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Per-row L2 distance ||x_i - mu||."""
    raise NotImplementedError("Filled in at M2 (synthesis §M2.2).")


def out_of_subspace_residual(
    X: np.ndarray,
    union_basis: np.ndarray,
) -> np.ndarray:
    """Per-row ||(I - V V^T) x||_2 / ||x||_2.

    `union_basis` is a (d, k) orthonormal basis returned by `build_union_basis`.
    """
    raise NotImplementedError("Filled in at M2 (synthesis §M2.2).")


def local_intrinsic_dim(
    X: np.ndarray,
    k: int = 20,
    estimator: str = "twoNN",
) -> np.ndarray:
    """Per-row intrinsic-dim estimate via TwoNN over k-NN.

    Reuses `src/intrinsic_dim.py::twoNN_estimate` on each row's k-NN cloud.
    """
    raise NotImplementedError("Filled in at M2 (synthesis §M2.2).")


def principal_angles(
    V_a: np.ndarray,
    V_b: np.ndarray,
    top_k: int = 10,
) -> np.ndarray:
    """Top-k principal angles (radians) between subspaces span(V_a), span(V_b),
    sorted ascending. Standard SVD-based formulation."""
    raise NotImplementedError("Filled in at M2 (synthesis §M2.2).")


def build_union_basis(
    per_behaviour_pcs: dict[str, np.ndarray],
    variance_threshold: float = 0.95,
) -> np.ndarray:
    """Concatenate per-behaviour top PCs, orthonormalise via QR, return basis
    covering `variance_threshold` of the joint variance (default 0.95).
    Sensitivity sweeps at 0.90 and 0.99 happen in the runner."""
    raise NotImplementedError("Filled in at M2 (synthesis §M2.2).")


def jonckheere_terpstra(values: np.ndarray, tiers: np.ndarray) -> dict:
    """Jonckheere-Terpstra ordinal-trend test for tiers 1->2->3.

    Returns: {statistic, p_value, trend_direction in {-1, 0, +1}}.
    """
    raise NotImplementedError("Filled in at M2 (synthesis §M2.2).")


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size: P(A>B) - P(B>A)."""
    raise NotImplementedError("Filled in at M2 (synthesis §M2.2).")


def bootstrap_ci(
    fn: Callable,
    *args,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> tuple[float, float]:
    """Generic percentile bootstrap CI for a scalar statistic. Used by the
    M2 runner for effect-size CIs."""
    raise NotImplementedError("Filled in at M2 (synthesis §M2.2).")
