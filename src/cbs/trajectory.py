"""
src/cbs/trajectory.py — Chain-as-curve trajectory module.

Purpose
-------
Build the per-chain trajectory as a first-class geometric object: an ordered
sequence of activations at a chosen layer, with per-sentence behaviour / CBS
labels and a defined arc-length parameterisation.

This is the new methodological contribution that supports the
locus-in-process commitment from the exploration doc §2.5: the trajectory,
rather than the per-sentence point, becomes the object of interest.

Validation
----------
* Synthetic helix: constant curvature within 5% (unit test).
* Straight-line trajectory: zero curvature, arc length = ||x_T - x_0||.
* Degenerate cases (T < 3): documented behaviour.
* Performance: 1000 trajectories, one layer, < 5 min on CPU.

CRITICAL - curvature formula
----------------------------
Arc-length-reparameterised discrete Frenet curvature (synthesis §M3.2 /
§12.11). NOT ||x_{t+1} - 2 x_t + x_{t-1}||. Do NOT silently substitute the
simpler form: if it looks wrong, halt and ask.

Milestone
---------
M3 (synthesis §M3); extended for group comparisons at M4 (§M4.2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.cbs.schemas import ChainTrajectory

__all__ = [
    "ChainTrajectory",
    "build_trajectory",
    "arc_length_sequence",
    "total_arc_length",
    "curvature_sequence",
    "subspace_visit_sequence",
    "cross_subspace_returns",
    "trajectory_cone_angle",
    "compare_groups",
    "per_sentence_curvature_vs_tier",
]


def build_trajectory(
    chain: dict,
    activations_dir: Path,
    layer: int,
) -> ChainTrajectory:
    """Assemble per-sentence activations into a ChainTrajectory at one layer.

    Convention: sentence-final-token activations (Venhoff). Resolved against
    the per-extraction metadata. If the existing `metadata.json` lacks
    row-to-sentence provenance, the M3 implementation must reconstruct it
    deterministically from the annotated-chains source.
    """
    raise NotImplementedError("Filled in at M3 (synthesis §M3.2).")


def arc_length_sequence(traj: ChainTrajectory) -> np.ndarray:
    """Cumulative arc length s_t = sum_{i<t} ||x_{i+1} - x_i||.

    Returns shape (T,) with s_0 = 0.
    """
    raise NotImplementedError("Filled in at M3 (synthesis §M3.2).")


def total_arc_length(traj: ChainTrajectory) -> float:
    """Final cumulative arc length s_{T-1}."""
    raise NotImplementedError("Filled in at M3 (synthesis §M3.2).")


def curvature_sequence(traj: ChainTrajectory) -> np.ndarray:
    """Arc-length-reparameterised discrete Frenet curvature.

    For each interior sentence t in [1, T-2]:
        T_left  = (x_t     - x_{t-1}) / ||x_t     - x_{t-1}||
        T_right = (x_{t+1} - x_t)     / ||x_{t+1} - x_t||
        ds      = (||x_t - x_{t-1}|| + ||x_{t+1} - x_t||) / 2
        kappa_t = ||T_right - T_left|| / ds

    Returns shape (T,) with kappa_0 = kappa_{T-1} = NaN.
    Locked formula - see module docstring.
    """
    raise NotImplementedError("Filled in at M3 (synthesis §M3.2).")


def subspace_visit_sequence(
    traj: ChainTrajectory,
    subspaces: dict[str, np.ndarray],
) -> list[str]:
    """Per t: argmax behaviour by projection magnitude
        ||V_b V_b^T x_t||_2 / ||x_t||_2.
    """
    raise NotImplementedError("Filled in at M3 (synthesis §M3.2).")


def cross_subspace_returns(
    traj: ChainTrajectory,
    subspaces: dict[str, np.ndarray],
) -> dict:
    """Returns {visit_sequence, n_transitions, return_rate, transition_matrix}.
    """
    raise NotImplementedError("Filled in at M3 (synthesis §M3.2).")


def trajectory_cone_angle(traj: ChainTrajectory) -> float:
    """Largest principal angle of the unit-normalised activation matrix.
    Diagnoses whether the trajectory stays in a low-dimensional cone."""
    raise NotImplementedError("Filled in at M3 (synthesis §M3.2).")


# M4 extensions (live here per synthesis §M4.2 "extend").

def compare_groups(
    summary,                       # pandas.DataFrame
    group_col: str,
    stat_cols: list[str],
    residualise_on: Optional[list[str]] = None,
):
    """Per stat_col: residualise on chain length T (OLS), then Wilcoxon p,
    Cliff's delta, bootstrap CI. Returns a DataFrame keyed by `stat_col`."""
    raise NotImplementedError("Filled in at M4 (synthesis §M4.2).")


def per_sentence_curvature_vs_tier(trajectories: list[ChainTrajectory]):
    """Long-format DataFrame: chain_id, sentence_idx, curvature, cbs_tier,
    cross_domain, behaviour, position. For mixed-effects regression."""
    raise NotImplementedError("Filled in at M4 (synthesis §M4.2).")
