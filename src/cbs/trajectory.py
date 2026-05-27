"""
src/cbs/trajectory.py — Chain-as-curve trajectory module (M3).

Purpose
-------
Build the per-chain trajectory as a first-class geometric object: an ordered
sequence of activations at a chosen layer, with per-sentence behaviour / CBS
labels and a defined arc-length parameterisation. Supports the
locus-in-process commitment from the exploration doc §2.5 — the trajectory,
not the per-sentence point, becomes the object of interest.

Validation
----------
* Synthetic helix: constant curvature within 5% (unit test).
* Straight-line trajectory: zero curvature, arc length = ||x_T - x_0||.
* Degenerate cases (T < 3): documented behaviour.
* Performance: 1000 trajectories, one layer, < 5 min on CPU.

CRITICAL — curvature formula
----------------------------
Arc-length-reparameterised discrete Frenet curvature
(synthesis §M3.2 / §12.11). NOT `||x_{t+1} - 2 x_t + x_{t-1}||`. The
arc-length normalisation is what makes it a real curvature.

Row-to-sentence provenance
--------------------------
`activation_extraction.py` writes per-behaviour matrices in the order
chains appear in the source annotated-chains JSON, then in span order
within each chain. `_build_row_index` reconstructs that mapping
deterministically.

Milestone
---------
M3 (synthesis §M3); extended for group comparisons at M4 (§M4.2).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from src.cbs.cohort import is_truncated
from src.cbs.schemas import ChainTrajectory

logger = logging.getLogger(__name__)


# The behaviours that Phase 4 saved activations for. Other behaviours
# (initializing, deduction) are not in the existing *_layer{N}.npy files,
# so their sentences will be skipped when building trajectories.
PHASE_4_BEHAVIOURS: tuple[str, ...] = (
    "backtracking",
    "uncertainty-estimation",
    "example-testing",
    "adding-knowledge",
)


__all__ = [
    "ChainTrajectory",
    "PHASE_4_BEHAVIOURS",
    "build_row_index",
    "load_layer_activations",
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


# ── Row-to-sentence provenance reconstruction ───────────────────────────────

def build_row_index(
    annotated_chains: Iterable[dict],
    *,
    target_behaviours: tuple[str, ...] = PHASE_4_BEHAVIOURS,
) -> dict[str, list[tuple[str, int]]]:
    """Reconstruct (chain_id, span_idx) per row index, per behaviour.

    Matches the iteration order in `src/activation_extraction.py`:
    chains are walked in source-JSON order; within each chain, spans are
    walked in array order; only spans whose label is in `target_behaviours`
    are appended to that behaviour's per-row list.

    Returns
    -------
    `{behaviour: [(chain_id, span_idx), ...]}` with one entry per row of
    the `{behaviour}_layer{N}.npy` matrix.
    """
    out: dict[str, list[tuple[str, int]]] = {b: [] for b in target_behaviours}
    for chain in annotated_chains:
        chain_id = chain.get("task_id") or chain.get("id") or ""
        for i, span in enumerate(chain.get("annotations", []) or []):
            label = span.get("label")
            if label in target_behaviours:
                out[label].append((chain_id, i))
    return out


# ── Per-layer activation loader ────────────────────────────────────────────

def load_layer_activations(
    activations_dir: Path,
    layer: int,
    *,
    behaviours: tuple[str, ...] = PHASE_4_BEHAVIOURS,
) -> dict[str, np.ndarray]:
    """Load every `{behaviour}_layer{layer}.npy` file present."""
    out: dict[str, np.ndarray] = {}
    activations_dir = Path(activations_dir)
    for b in behaviours:
        fp = activations_dir / f"{b}_layer{layer}.npy"
        if fp.exists():
            out[b] = np.load(fp)
    return out


# ── build_trajectory ───────────────────────────────────────────────────────

def build_trajectory(
    chain: dict,
    activations_dir: Path,
    layer: int,
    *,
    row_index: Optional[dict[str, list[tuple[str, int]]]] = None,
    activations: Optional[dict[str, np.ndarray]] = None,
    target_behaviours: tuple[str, ...] = PHASE_4_BEHAVIOURS,
) -> ChainTrajectory:
    """Assemble a `ChainTrajectory` at one layer.

    Sentences whose behaviour is not in `target_behaviours` (e.g.
    initializing, deduction) are skipped because Phase 4 did not save
    activations for them. The remaining sentences form a possibly-sparse
    sub-trajectory; arc length and curvature still make sense over those
    points.

    `row_index` must be the index reconstructed from the same source JSON
    used by Phase 4 (see `build_row_index`); `activations` is the per-layer
    `{behaviour: matrix}` map (see `load_layer_activations`). Both are
    optional in the convenience path — when omitted they are computed on
    the fly from `activations_dir`, but this only works correctly if
    `chain` is the only chain (or the row_index is built from the full
    corpus).
    """
    chain_id = chain.get("task_id") or chain.get("id") or ""
    if activations is None:
        activations = load_layer_activations(activations_dir, layer,
                                             behaviours=target_behaviours)
    if row_index is None:
        # Builds an index that contains only this single chain — for tests.
        row_index = build_row_index([chain], target_behaviours=target_behaviours)

    # Lookup: (chain_id, span_idx) -> (behaviour, row).
    lookup: dict[tuple[str, int], tuple[str, int]] = {}
    for beh, items in row_index.items():
        for row, key in enumerate(items):
            lookup[key] = (beh, row)

    spans = chain.get("annotations", []) or []
    sentence_ids: list[str] = []
    behaviours: list[str] = []
    cbs_tiers: list[int] = []
    cross_domain: list[Optional[bool]] = []
    rows: list[np.ndarray] = []
    for i, span in enumerate(spans):
        key = (chain_id, i)
        if key not in lookup:
            continue
        beh, row = lookup[key]
        mat = activations.get(beh)
        if mat is None or row >= mat.shape[0]:
            logger.debug("missing activation for %s:%d (%s row=%d)",
                         chain_id, i, beh, row)
            continue
        rows.append(mat[row])
        sentence_ids.append(f"{chain_id}:{i}")
        behaviours.append(span.get("label", ""))
        cbs_tiers.append(int(span.get("cbs_tier", 0)) if "cbs_tier" in span else 0)
        cd_val = span.get("cbs_cross_domain")
        cross_domain.append(bool(cd_val) if cd_val is not None else None)

    if rows:
        X = np.stack(rows).astype(np.float32, copy=False)
    else:
        d = next(iter(activations.values())).shape[1] if activations else 0
        X = np.zeros((0, d), dtype=np.float32)

    return ChainTrajectory(
        chain_id=chain_id,
        layer=int(layer),
        sentence_ids=sentence_ids,
        X=X,
        behaviours=behaviours,
        cbs_tiers=cbs_tiers,
        cross_domain=cross_domain,
        truncated=is_truncated(chain),
    )


# ── Arc length and curvature ───────────────────────────────────────────────

def arc_length_sequence(traj: ChainTrajectory) -> np.ndarray:
    """Cumulative arc length s_t = sum_{i<t} ||x_{i+1} - x_i||, shape (T,)."""
    X = traj.X
    if X is None or X.shape[0] == 0:
        return np.zeros(0)
    if X.shape[0] == 1:
        return np.zeros(1)
    diffs = np.linalg.norm(np.diff(X, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(diffs)])


def total_arc_length(traj: ChainTrajectory) -> float:
    s = arc_length_sequence(traj)
    return float(s[-1]) if s.size > 0 else 0.0


def curvature_sequence(traj: ChainTrajectory) -> np.ndarray:
    """Arc-length-reparameterised discrete Frenet curvature.

    For each interior sentence t in [1, T-2]:
        T_left  = (x_t     - x_{t-1}) / ||x_t     - x_{t-1}||
        T_right = (x_{t+1} - x_t)     / ||x_{t+1} - x_t||
        ds      = (||x_t - x_{t-1}|| + ||x_{t+1} - x_t||) / 2
        kappa_t = ||T_right - T_left|| / ds

    Returns shape (T,) with kappa_0 = kappa_{T-1} = NaN. NaN at any interior
    point t where ||x_t - x_{t-1}|| = 0 or ||x_{t+1} - x_t|| = 0 (duplicate
    consecutive activations).
    """
    X = traj.X
    T = 0 if X is None else int(X.shape[0])
    out = np.full(T, np.nan, dtype=np.float64)
    if T < 3:
        return out
    diffs = np.diff(X, axis=0)                          # (T-1, d)
    norms = np.linalg.norm(diffs, axis=1)               # (T-1,)
    for t in range(1, T - 1):
        nl = norms[t - 1]
        nr = norms[t]
        if nl <= 0.0 or nr <= 0.0:
            out[t] = np.nan
            continue
        T_left = diffs[t - 1] / nl
        T_right = diffs[t] / nr
        ds = (nl + nr) / 2.0
        out[t] = float(np.linalg.norm(T_right - T_left) / ds)
    return out


# ── Subspace visit / return dynamics ───────────────────────────────────────

def subspace_visit_sequence(
    traj: ChainTrajectory,
    subspaces: dict[str, np.ndarray],
) -> list[str]:
    """Per t: argmax behaviour by projection magnitude
        ||V_b V_b^T x_t||_2 / ||x_t||_2.

    `subspaces[b]` is the (d, k_b) orthonormal basis for behaviour b.
    Empty trajectory or no subspaces -> empty list.
    """
    X = traj.X
    if X is None or X.shape[0] == 0 or not subspaces:
        return []
    behaviour_names = list(subspaces.keys())
    proj_norms = np.zeros((X.shape[0], len(behaviour_names)))
    for j, b in enumerate(behaviour_names):
        V = subspaces[b]
        if V.shape[0] != X.shape[1]:
            raise ValueError(f"subspace dim mismatch for {b!r}: "
                             f"X.shape={X.shape}, V.shape={V.shape}")
        coef = X @ V                          # (T, k_b)
        proj_norms[:, j] = np.linalg.norm(coef, axis=1)
    x_norms = np.linalg.norm(X, axis=1)
    ratios = proj_norms / np.maximum(x_norms[:, None], 1e-12)
    argmax_idx = np.argmax(ratios, axis=1)
    return [behaviour_names[i] for i in argmax_idx]


def cross_subspace_returns(
    traj: ChainTrajectory,
    subspaces: dict[str, np.ndarray],
) -> dict:
    """Visit sequence + transition counts + return rate + transition matrix."""
    visit_seq = subspace_visit_sequence(traj, subspaces)
    if not visit_seq:
        return {
            "visit_sequence": [],
            "n_transitions": 0,
            "return_rate": 0.0,
            "transition_matrix": {},
        }

    # Transitions: count consecutive position changes.
    n_transitions = sum(1 for i in range(1, len(visit_seq))
                        if visit_seq[i] != visit_seq[i - 1])

    # Return rate over the compressed visit sequence (collapse runs).
    compressed: list[str] = []
    for v in visit_seq:
        if not compressed or compressed[-1] != v:
            compressed.append(v)
    seen: set[str] = set()
    revisits = 0
    for v in compressed:
        if v in seen:
            revisits += 1
        seen.add(v)
    return_rate = revisits / max(1, len(compressed))

    # Transition matrix on inter-state transitions.
    bnames = sorted(set(visit_seq))
    mat: dict[str, dict[str, int]] = {b: {b2: 0 for b2 in bnames} for b in bnames}
    for i in range(1, len(visit_seq)):
        a, b = visit_seq[i - 1], visit_seq[i]
        if a != b:
            mat[a][b] += 1

    return {
        "visit_sequence": visit_seq,
        "n_transitions": int(n_transitions),
        "return_rate": float(return_rate),
        "transition_matrix": mat,
    }


def trajectory_cone_angle(traj: ChainTrajectory) -> float:
    """Maximum angular deviation (radians) of any unit-normalised sentence
    activation from the trajectory's mean unit direction. Diagnoses whether
    the trajectory stays in a tight cone (small return value) or fans out
    (close to pi)."""
    X = traj.X
    if X is None or X.shape[0] < 2:
        return 0.0
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    Xu = X / safe
    mu = Xu.mean(axis=0)
    mu_norm = float(np.linalg.norm(mu))
    if mu_norm < 1e-12:
        return float(np.pi)
    mu = mu / mu_norm
    cos_angles = np.clip(Xu @ mu, -1.0, 1.0)
    angles = np.arccos(cos_angles)
    return float(np.max(angles))


# ── M4 group comparisons (live here per synthesis §M4.2 "extend") ──────────

def compare_groups(
    summary,                                # pandas.DataFrame
    group_col: str,
    stat_cols: list[str],
    residualise_on: Optional[list[str]] = None,
):
    """Per `stat_col`: residualise on chain length T (OLS), then Wilcoxon p,
    Cliff's delta, bootstrap CI. Returns a DataFrame keyed by stat_col.

    Implementation note (M4): kept here per synthesis §M4.2 "extend trajectory
    module". The runner `11_trajectory_analysis.py` calls this on the
    layer-summary parquet emitted by `10_trajectory_build.py`.
    """
    import pandas as pd
    from src.cbs.geometry import bootstrap_ci, cliffs_delta

    residualise_on = list(residualise_on or [])
    rows = []
    for stat in stat_cols:
        if stat not in summary.columns:
            continue
        y = summary[stat].astype(float).values
        g = summary[group_col].values
        # Residualise on covariates by OLS, in-place.
        y_resid = y.copy()
        if residualise_on:
            X = np.column_stack([
                np.ones_like(y, dtype=float),
                *[summary[c].astype(float).values for c in residualise_on],
            ])
            try:
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                y_resid = y - X @ beta
            except np.linalg.LinAlgError:
                pass
        uniq = sorted(set(g))
        if len(uniq) != 2:
            rows.append({"stat": stat, "group_col": group_col, "n_groups": len(uniq),
                         "note": "compare_groups expects a binary group_col"})
            continue
        ga = uniq[0]
        gb = uniq[1]
        a = y_resid[g == ga]
        b = y_resid[g == gb]
        if len(a) < 3 or len(b) < 3:
            rows.append({"stat": stat, "group_col": group_col, "note": "too few"})
            continue
        try:
            from scipy.stats import mannwhitneyu
            _, p = mannwhitneyu(a, b, alternative="two-sided")
        except ImportError:
            p = float("nan")
        delta = cliffs_delta(a, b)
        try:
            ci_lo, ci_hi = bootstrap_ci(cliffs_delta, a, b,
                                        n_bootstrap=500, paired=False)
        except Exception:  # noqa: BLE001
            ci_lo, ci_hi = float("nan"), float("nan")
        rows.append({
            "stat": stat, "group_col": group_col,
            "group_a": str(ga), "group_b": str(gb),
            "n_a": int(len(a)), "n_b": int(len(b)),
            "wilcoxon_p": float(p),
            "cliffs_delta": float(delta),
            "ci95_lo": float(ci_lo), "ci95_hi": float(ci_hi),
        })
    return pd.DataFrame(rows)


def per_sentence_curvature_vs_tier(trajectories: list[ChainTrajectory]):
    """Long-format DataFrame: chain_id, sentence_idx, curvature, cbs_tier,
    cross_domain, behaviour, position. For mixed-effects regression at M4."""
    import pandas as pd

    rows = []
    for traj in trajectories:
        if traj.X is None or traj.X.shape[0] < 3:
            continue
        kappa = curvature_sequence(traj)
        T = traj.T
        for t in range(T):
            rows.append({
                "chain_id": traj.chain_id,
                "sentence_idx": int(traj.sentence_ids[t].split(":")[-1])
                                 if traj.sentence_ids else t,
                "position_in_trajectory": t,
                "position_norm": t / max(1, T - 1),
                "curvature": float(kappa[t]) if not np.isnan(kappa[t]) else None,
                "cbs_tier": traj.cbs_tiers[t] if t < len(traj.cbs_tiers) else 0,
                "cross_domain": traj.cross_domain[t] if t < len(traj.cross_domain) else None,
                "behaviour": traj.behaviours[t] if t < len(traj.behaviours) else "",
                "truncated": traj.truncated,
            })
    return pd.DataFrame(rows)
