"""src/predict/collapse.py — shared collapse labels + pre-onset leak-guard.

The trajectory value track (PG §11/§12) uses COLLAPSE (loop onset) as its
judge-noise-free primary outcome. Two helpers used by every runner on that track
(24_belief_lens, 26_value_head, …), factored here so there is exactly one copy:

  collapse_labels    — chain_id -> {True=loop, False=clean}; 'ambiguous'
                       (no periodic tail but repetitive) dropped so label noise
                       cannot straddle the classes. Also returns the per-chain
                       loop onset_char (the leak-guard boundary; -1 = no loop).
  truncate_preonset  — keep only trajectory steps whose span STARTS strictly
                       before loop onset, so loop-region steps never leak into a
                       "predict collapse from pre-onset states" analysis.

Pure-CPU (loop_geometry is numpy-only); no model / API needed.
"""

from __future__ import annotations

import numpy as np

from src.loop_geometry import loop_labels_for_chain
from src.text_offsets import locate_annotation_offsets
from src.predict.trajectory_dataset import StepDataset

__all__ = ["collapse_labels", "truncate_preonset"]


def collapse_labels(chains: list[dict]) -> tuple[dict, dict, dict]:
    """chain_id -> {True=loop, False=clean} (+ onset_char map + class counts)."""
    collapse_map: dict = {}
    onset_map: dict = {}
    counts = {"loop": 0, "clean": 0, "ambiguous": 0}
    for c in chains:
        cid = c.get("task_id") or c.get("id") or ""
        info = loop_labels_for_chain(c.get("chain", ""))
        counts[info["class"]] = counts.get(info["class"], 0) + 1
        onset_map[cid] = info["onset_char"]
        if info["class"] == "loop":
            collapse_map[cid] = True
        elif info["class"] == "clean":
            collapse_map[cid] = False
    return collapse_map, onset_map, counts


def truncate_preonset(datasets, chains_by_id: dict, onset_map: dict,
                      min_steps: int = 2) -> list[StepDataset]:
    """Keep only steps whose annotation span starts strictly before loop onset.

    Clean chains (onset < 0) keep every step. Maps each retained step's original
    annotation-span index to its char offset via the canonical occurrence-aware
    locator, then compares to onset_char.
    """
    out: list[StepDataset] = []
    for ds in datasets:
        chain = chains_by_id.get(ds.chain_id)
        onset = onset_map.get(ds.chain_id, -1)
        if chain is None:
            continue
        if onset is None or onset < 0:
            keep = np.arange(ds.T)
        else:
            sentences = [a.get("text", "") for a in (chain.get("annotations") or [])]
            offsets = locate_annotation_offsets(chain.get("chain", ""), sentences)
            keep_list = []
            for t in range(ds.T):
                oi = int(ds.orig_indices[t])
                off = offsets[oi] if 0 <= oi < len(offsets) else None
                if off is not None and off < onset:
                    keep_list.append(t)
            keep = np.asarray(keep_list, dtype=int)
        if keep.size < min_steps:
            continue
        out.append(StepDataset(
            chain_id=ds.chain_id, X=ds.X[keep], orig_indices=ds.orig_indices[keep],
            behaviours=[ds.behaviours[i] for i in keep] if ds.behaviours else [],
            truncated=ds.truncated, correct=ds.correct,
            difficulty=ds.difficulty, category=ds.category,
        ))
    return out
