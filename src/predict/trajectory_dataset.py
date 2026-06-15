"""
src/predict/trajectory_dataset.py — Step datasets for next-step prediction.

Purpose
-------
Turn the existing per-chain trajectories (src/cbs/trajectory.py) into the
supervised tensors a next-step predictor consumes: ordered step embeddings per
chain, plus the (chain, original-span-index) provenance needed to (a) split by
chain (CF-2) and (b) control for the gap between retained steps. The saved
activations cover only the 4 Phase-4 behaviours, so a chain's trajectory is a
SPARSE sub-sequence; `orig_indices` records each step's original span index so
callers can measure — and optionally filter on — that gap (CF-6 / sparsity).

This module is label-agnostic: a chain-level correctness label is attached when
provided, but nothing here requires it, so the whole module is testable offline
with no API or cluster access.

Validation
----------
* make_supervised_pairs shapes + per-pair gap accounting (unit test).
* step_shuffle_within_chain only reorders within a chain, preserves the set
  of step embeddings and the chain-level fields (unit test).

Milestone
---------
Predictive Geometry — Rung 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from src.cbs.trajectory import (
    PHASE_4_BEHAVIOURS,
    build_row_index,
    build_trajectory,
    load_layer_activations,
)

__all__ = [
    "StepDataset",
    "build_step_datasets",
    "make_supervised_pairs",
    "step_shuffle_within_chain",
]


@dataclass
class StepDataset:
    """Ordered reasoning-step embeddings for one chain at one layer.

    Fields
    ------
    chain_id     : task / chain identifier (matches src.cbs trajectory chain_id)
    X            : (T, d) float32, one row per retained reasoning step, in order
    orig_indices : (T,) int, original annotation-span index of each retained
                   step. A gap > 1 between consecutive entries means non-Phase-4
                   behaviours (initializing / deduction) sat between them.
    behaviours   : length-T behaviour labels
    truncated    : chain hit the generation token cap (CF-8)
    correct      : chain-level reasoning-correctness label (None if unlabelled)
    difficulty   : task difficulty (for matching / stratified nulls)
    category     : task category (for stratification; CF-8 truncation ~ category)
    """

    chain_id: str
    X: np.ndarray
    orig_indices: np.ndarray
    behaviours: list[str] = field(default_factory=list)
    truncated: bool = False
    correct: Optional[bool] = None
    difficulty: Optional[str] = None
    category: Optional[str] = None

    @property
    def T(self) -> int:
        return int(self.X.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.X.shape[1]) if self.X.ndim == 2 else 0


def _coerce_correct(value) -> Optional[bool]:
    """Accept a bool or a {'correct': ...} record; None stays None."""
    if value is None:
        return None
    if isinstance(value, dict):
        inner = value.get("correct")
        return None if inner is None else bool(inner)
    return bool(value)


def build_step_datasets(
    chains: list[dict],
    activations_dir,
    layer: int,
    *,
    target_behaviours: tuple[str, ...] = PHASE_4_BEHAVIOURS,
    labels: Optional[dict] = None,
    difficulty_map: Optional[dict] = None,
    category_map: Optional[dict] = None,
    min_steps: int = 2,
) -> list[StepDataset]:
    """Assemble per-chain `StepDataset`s from the saved layer activations.

    The row index and the per-layer activation matrices are built ONCE and
    shared across all chains (so this is one pass over the corpus, not one load
    per chain). Chains with fewer than `min_steps` retained steps are dropped —
    a next-step pair needs at least two points.

    `labels` / `difficulty_map` / `category_map` are `chain_id -> value` dicts;
    `labels[chain_id]` may be a bool or a dict carrying a `'correct'` key.
    """
    activations_dir = Path(activations_dir)
    row_index = build_row_index(chains, target_behaviours=target_behaviours)
    activations = load_layer_activations(
        activations_dir, layer, behaviours=target_behaviours
    )

    labels = labels or {}
    difficulty_map = difficulty_map or {}
    category_map = category_map or {}

    out: list[StepDataset] = []
    for chain in chains:
        traj = build_trajectory(
            chain,
            activations_dir,
            layer,
            row_index=row_index,
            activations=activations,
            target_behaviours=target_behaviours,
        )
        if traj.T < min_steps:
            continue
        # Recover the original span index of each retained step from its
        # sentence id, which build_trajectory writes as f"{chain_id}:{idx}".
        orig = np.array(
            [int(sid.split(":")[-1]) for sid in traj.sentence_ids], dtype=int
        )
        cid = traj.chain_id
        out.append(
            StepDataset(
                chain_id=cid,
                X=np.asarray(traj.X, dtype=np.float32),
                orig_indices=orig,
                behaviours=list(traj.behaviours),
                truncated=bool(traj.truncated),
                correct=_coerce_correct(labels.get(cid)),
                difficulty=difficulty_map.get(cid),
                category=category_map.get(cid),
            )
        )
    return out


def make_supervised_pairs(
    datasets: Iterable[StepDataset],
    *,
    max_gap: Optional[int] = None,
) -> dict:
    """Build next-step prediction pairs (x_t -> x_{t+1}) over each chain.

    Returns a dict of arrays, all aligned row-for-row:
      X_hist : (n, d) the step-t embedding (predictor input)
      X_next : (n, d) the step-(t+1) embedding (target)
      groups : (n,) chain_id per pair — pass as GroupKFold groups (CF-2)
      gaps   : (n,) orig_indices[t+1] - orig_indices[t]  (>=1; >1 = skipped steps)
      pos    : (n,) position t within the retained trajectory

    `max_gap` drops pairs whose original-span gap exceeds it — an adjacency
    control for the sparse-trajectory confound (CF-6): with max_gap=1 only
    genuinely consecutive retained steps are used.
    """
    datasets = list(datasets)
    d = datasets[0].X.shape[1] if datasets else 0

    Xh: list[np.ndarray] = []
    Xn: list[np.ndarray] = []
    groups: list[str] = []
    gaps: list[int] = []
    pos: list[int] = []
    for ds in datasets:
        T = ds.T
        for t in range(T - 1):
            gap = int(ds.orig_indices[t + 1] - ds.orig_indices[t])
            if max_gap is not None and gap > max_gap:
                continue
            Xh.append(ds.X[t])
            Xn.append(ds.X[t + 1])
            groups.append(ds.chain_id)
            gaps.append(gap)
            pos.append(t)

    return {
        "X_hist": np.asarray(Xh, dtype=np.float32).reshape(-1, d),
        "X_next": np.asarray(Xn, dtype=np.float32).reshape(-1, d),
        "groups": np.asarray(groups, dtype=object),
        "gaps": np.asarray(gaps, dtype=int),
        "pos": np.asarray(pos, dtype=int),
    }


def step_shuffle_within_chain(
    datasets: Iterable[StepDataset], rng: np.random.Generator
) -> list[StepDataset]:
    """Return copies with the step ORDER permuted within each chain.

    Destroys the temporal next-step structure while preserving each chain's set
    of step embeddings, its size, and its chain-level fields — the null for
    "does the *order* of reasoning steps carry predictable structure beyond the
    static cloud of steps in a chain?". Analogous to the within-chain
    permutation in src/nulls.py, but permuting the step index, not the label.
    """
    out: list[StepDataset] = []
    for ds in datasets:
        perm = rng.permutation(ds.T)
        out.append(
            StepDataset(
                chain_id=ds.chain_id,
                X=ds.X[perm],
                orig_indices=ds.orig_indices[perm],
                behaviours=[ds.behaviours[i] for i in perm] if ds.behaviours else [],
                truncated=ds.truncated,
                correct=ds.correct,
                difficulty=ds.difficulty,
                category=ds.category,
            )
        )
    return out
