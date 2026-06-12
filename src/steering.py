"""
Phase 6: Steering vector construction.

Two parallel tracks for each target behaviour:

  1. Single-direction (Venhoff-style)
     r = mean(on_activations) − mean(off_activations),  normalised to unit norm.
     "off" = activations from all *other* behaviours, providing a neutral baseline.

  2. Manifold-projected (Huang-style, our method)
     Compute the single-direction vector r, then project it onto the top-k
     principal components of the behaviour's own activation subspace:
       r_proj = Σ_{i=1}^{k} (r · v_i) v_i,  normalised to unit norm.
     This anchors the steering direction inside the behaviour's natural manifold.

The key prediction: if behaviours have manifold structure, the manifold-projected
vector should yield cleaner behaviour suppression (less off-target disruption,
better saturation curve) than the single-direction vector.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

from src.annotation import TARGET_BEHAVIOURS

logger = logging.getLogger(__name__)


# ── Core vector operations ────────────────────────────────────────────────────

def single_direction_vector(
    on_activations: np.ndarray,
    off_activations: np.ndarray,
) -> np.ndarray:
    """
    Difference-of-means steering vector (Venhoff et al.).

    Args:
        on_activations:  (N_on,  hidden_dim) — activations during the behaviour
        off_activations: (N_off, hidden_dim) — activations from other behaviours

    Returns:
        Unit-norm vector of shape (hidden_dim,)
    """
    r = on_activations.mean(axis=0) - off_activations.mean(axis=0)
    norm = np.linalg.norm(r)
    if norm < 1e-10:
        logger.warning("Steering vector has near-zero norm — returning zero vector")
        return r
    return r / norm


def manifold_projected_vector(
    on_activations: np.ndarray,
    off_activations: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Manifold-projected steering vector (our method, adapting Huang et al.).

    Computes the single-direction vector, then orthogonally projects it onto
    the top-k PCA subspace of the *on* activations.

    Args:
        on_activations:  (N_on, hidden_dim)
        off_activations: (N_off, hidden_dim)
        k:               number of PCA components to project onto

    Returns:
        Unit-norm projected vector of shape (hidden_dim,)
    """
    r = single_direction_vector(on_activations, off_activations)

    n_components = min(k, on_activations.shape[0] - 1, on_activations.shape[1])
    if n_components < 1:
        logger.warning(f"Cannot project: k={k} but only {on_activations.shape[0]} samples")
        return r

    pca = PCA(n_components=n_components, svd_solver="full")  # exact + reproducible
    pca.fit(on_activations)
    V = pca.components_  # (k, hidden_dim)

    # Project r onto the subspace spanned by V
    coords = V @ r               # (k,) — coordinates in PCA space
    r_proj = coords @ V          # (hidden_dim,) — back in activation space

    norm = np.linalg.norm(r_proj)
    if norm < 1e-10:
        logger.warning("Projected vector has near-zero norm — falling back to r")
        return r
    return r_proj / norm


def auto_k(on_activations: np.ndarray, variance_threshold: float = 0.70) -> int:
    """Return the smallest k such that the top-k PCs explain >= threshold variance."""
    max_k = min(on_activations.shape[0] - 1, on_activations.shape[1], 100)
    if max_k < 1:
        return 1
    pca = PCA(n_components=max_k, svd_solver="full")  # exact + reproducible
    pca.fit(on_activations)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    idx = int(np.searchsorted(cumvar, variance_threshold))
    return min(idx + 1, max_k)


# ── Build all vectors for a model ────────────────────────────────────────────

def build_steering_vectors(
    activations_dir: Path,
    layer: int,
    behaviours: Optional[list[str]] = None,
    k_values: Optional[list] = None,
    variance_threshold: float = 0.70,
    exclude_chain_ids: Optional[set] = None,
    annotated_path: Optional[Path] = None,
) -> dict:
    """
    Build single-direction and manifold-projected steering vectors for all
    target behaviours at the specified layer.

    Args:
        activations_dir:  directory produced by activation_extraction.py
        layer:            transformer layer to use (e.g. 27 for Qwen-1.5B)
        behaviours:       which behaviours to process; defaults to all 4 targets
        k_values:         list of ints (or "auto") for manifold projection
        variance_threshold: threshold for "auto" k selection
        exclude_chain_ids: chain/task ids whose rows are dropped before the
            vectors are computed. Pass the Phase-7 eval split here (via
            src.task_gen.stratified_eval_split) so the vectors NEVER see the
            evaluation tasks' activations — that is what turns Phase 7 from
            an on-corpus effect into a true out-of-sample test. Requires row
            provenance (the row_index.json sidecar, or annotated_path for
            legacy replay); fails loud if rows can't be identified.
        annotated_path:   annotation file for legacy provenance replay.

    Returns:
        {behaviour: {
            "layer":              int,
            "single_direction":   np.ndarray (hidden_dim,),
            "manifold_projected": {k: np.ndarray},
            "n_on":               int,
            "n_off":              int,
            "auto_k":             int,
            "n_excluded":         int,   # rows dropped by the hold-out
        }}
    """
    if behaviours is None:
        behaviours = TARGET_BEHAVIOURS
    if k_values is None:
        k_values = [1, 3, 5, 10, "auto"]

    activations_dir = Path(activations_dir)
    results = {}

    chain_id_map = None
    if exclude_chain_ids:
        from src.row_provenance import chain_ids_for
        exclude_chain_ids = set(exclude_chain_ids)
        chain_id_map = chain_ids_for(activations_dir, annotated_path,
                                     list(behaviours))

    # Load all activation matrices for this layer upfront
    all_acts: dict[str, np.ndarray] = {}
    n_excluded: dict[str, int] = {}
    for beh in behaviours:
        path = activations_dir / f"{beh}_layer{layer}.npy"
        if not path.exists():
            logger.warning(f"Missing: {path}")
            continue
        X = np.load(path).astype(np.float32)
        n_excluded[beh] = 0
        if exclude_chain_ids:
            from src.row_provenance import require_aligned
            cids = require_aligned(beh, X.shape[0], chain_id_map.get(beh),
                                   context="steering hold-out")
            keep = ~np.isin(cids, list(exclude_chain_ids))
            n_excluded[beh] = int((~keep).sum())
            X = X[keep]
        all_acts[beh] = X
        logger.info(f"  {beh}: {X.shape[0]} instances loaded"
                    + (f" ({n_excluded[beh]} eval-task rows held out)"
                       if exclude_chain_ids else ""))

    for beh in behaviours:
        if beh not in all_acts:
            continue
        on_acts = all_acts[beh]
        # "off" = all other loaded behaviours concatenated
        off_parts = [v for k, v in all_acts.items() if k != beh]
        if not off_parts:
            logger.warning(f"No off-activations for {beh} — skipping")
            continue
        off_acts = np.concatenate(off_parts, axis=0)

        k_auto = auto_k(on_acts, variance_threshold)
        r_single = single_direction_vector(on_acts, off_acts)

        manifold = {}
        for k in k_values:
            k_int = k_auto if k == "auto" else int(k)
            manifold[k] = manifold_projected_vector(on_acts, off_acts, k_int)

        results[beh] = {
            "layer": layer,
            "single_direction": r_single,
            "manifold_projected": manifold,
            "n_on": len(on_acts),
            "n_off": len(off_acts),
            "auto_k": k_auto,
            "n_excluded": n_excluded.get(beh, 0),
        }
        logger.info(
            f"  {beh}: auto_k={k_auto}, n_on={len(on_acts)}, n_off={len(off_acts)}"
        )

    return results


# ── Save / load ───────────────────────────────────────────────────────────────

def save_steering_vectors(vectors: dict, save_dir: Path,
                          provenance: Optional[dict] = None) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metadata = {}
    if provenance is not None:
        metadata["_provenance"] = provenance
    for beh, data in vectors.items():
        np.save(save_dir / f"{beh}_single.npy", data["single_direction"])
        for k, vec in data["manifold_projected"].items():
            np.save(save_dir / f"{beh}_manifold_k{k}.npy", vec)
        metadata[beh] = {
            "layer": data["layer"],
            "n_on": data["n_on"],
            "n_off": data["n_off"],
            "auto_k": data["auto_k"],
            "n_excluded": data.get("n_excluded", 0),
            "k_values": list(data["manifold_projected"].keys()),
        }

    # Back up a prior metadata (a shorter re-run shouldn't clobber it) and write
    # atomically so an interrupted write can't leave a corrupt metadata.json.
    from src.config import backup_existing
    meta_path = save_dir / "metadata.json"
    backup_existing(meta_path)
    tmp = meta_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(metadata, f, indent=2)
    tmp.replace(meta_path)
    logger.info(f"Steering vectors saved → {save_dir}")


def load_steering_vectors(save_dir: Path) -> dict:
    save_dir = Path(save_dir)
    with open(save_dir / "metadata.json") as f:
        metadata = json.load(f)

    vectors = {}
    for beh, meta in metadata.items():
        if beh.startswith("_"):
            continue  # metadata keys like _provenance, not a behaviour
        vectors[beh] = {
            "layer": meta["layer"],
            "single_direction": np.load(save_dir / f"{beh}_single.npy"),
            "manifold_projected": {
                k: np.load(save_dir / f"{beh}_manifold_k{k}.npy")
                for k in meta["k_values"]
            },
            "n_on": meta["n_on"],
            "n_off": meta["n_off"],
            "auto_k": meta["auto_k"],
            "n_excluded": meta.get("n_excluded", 0),
        }
    return vectors
