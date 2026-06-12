"""Row provenance + row hygiene for the activation matrices.

Two long-standing failure modes in the analysis path are closed here:

1. **Replay-based provenance.** Activation matrices carry no row ids; every
   analysis script reconstructed per-row chain-ids by replaying the Phase-4
   iteration (same filter, same order) and silently fell back to proxy chain
   ids on a count mismatch — under which the chain-stratified permutation null
   degenerates to a no-op (p≈1.0) with only a log warning. Phase 4 now writes
   a ``row_index.json`` sidecar (one record per row, exact row order; see
   src/activation_extraction.py). `chain_ids_for` loads it, falling back to an
   occurrence-aware replay only for legacy extractions, and `require_aligned`
   turns every mismatch into a hard error instead of a proxy.

2. **Exact-duplicate rows.** First-occurrence sentence matching bound repeated
   sentences to the same token span, so 35–56% of rows in the headline
   matrices are byte-identical (CONFOUNDS_AND_REMEDIATION.md CF-13). Zero
   distances corrupt every kNN-based estimator (TwoNN, Levina–Bickel, geodesic
   graphs) and bias the permutation nulls anti-conservatively: duplicates
   concentrate within a behaviour label, so the real per-behaviour matrix is
   duplicate-rich while label-permuted resamples are duplicate-poor.
   `dedup_rows` / `duplicate_fraction` are the shared helpers the headline
   path (05, 05b, tier1) now applies before any estimator — previously only
   robustness_geometry.py deduplicated.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

ROW_INDEX_FILE = "row_index.json"


# ── Sidecar-first chain-id loading ───────────────────────────────────────────

def load_row_index(act_dir: Path) -> Optional[dict]:
    """Parsed ``row_index.json`` sidecar for an activation directory, or None."""
    p = Path(act_dir) / ROW_INDEX_FILE
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _replay_chain_ids(annotated_path: Path, behaviours) -> dict[str, list]:
    """Legacy fallback: reconstruct per-row chain ids by replaying Phase 4's
    iteration (occurrence-aware locator over the FULL annotation list, then
    filter by label) for extractions that predate the sidecar.

    Note: this mirrors Phase 4's unlocatable-sentence skip but cannot mirror
    its (in practice unreachable) empty-token-positions skip; `require_aligned`
    catches any residual mismatch loudly.
    """
    from src.text_offsets import locate_annotation_offsets

    with open(annotated_path) as f:
        chains = json.load(f)
    per_beh: dict[str, list] = {b: [] for b in behaviours}
    n_filtered = {b: 0 for b in behaviours}
    for chain in chains:
        chain_text = chain.get("chain", "")
        cid = chain.get("chain_id") or chain.get("task_id")
        annotations = chain.get("annotations", [])
        offsets = locate_annotation_offsets(
            chain_text, [a.get("text", "") for a in annotations]
        )
        for ann, off in zip(annotations, offsets):
            label = ann.get("label", "")
            if label not in per_beh:
                continue
            if off is None:
                n_filtered[label] += 1
                continue
            per_beh[label].append(cid)
    for b, n in n_filtered.items():
        if n > 0:
            logger.info(f"  replay loader filtered {n} unlocatable {b} annotations "
                        f"(Phase 4 skipped these too)")
    return per_beh


def chain_ids_for(
    act_dir: Path,
    annotated_path: Path,
    behaviours,
) -> dict[str, Optional[np.ndarray]]:
    """Per-behaviour chain-id arrays aligned to the activation-matrix rows.

    Prefers the extraction-time ``row_index.json`` sidecar (exact, by
    construction); falls back to occurrence-aware replay of the annotation
    file for legacy extractions. Returns ``{behaviour: ndarray | None}`` —
    pass results through `require_aligned` before use.
    """
    sidecar = load_row_index(act_dir)
    if sidecar is not None:
        rows = sidecar.get("rows", {})
        out: dict[str, Optional[np.ndarray]] = {}
        for b in behaviours:
            recs = rows.get(b)
            out[b] = np.array([r["chain_id"] for r in recs]) if recs else None
        logger.info(f"chain ids from sidecar {Path(act_dir) / ROW_INDEX_FILE} "
                    f"(matching={sidecar.get('sentence_matching')})")
        return out

    annotated_path = Path(annotated_path)
    if not annotated_path.exists():
        logger.warning(f"No sidecar and no annotation file {annotated_path}; "
                       f"chain ids unavailable.")
        return {b: None for b in behaviours}
    logger.warning(f"No {ROW_INDEX_FILE} in {act_dir} (legacy extraction) — "
                   f"replaying annotation iteration to reconstruct chain ids. "
                   f"Re-extract to get exact row provenance.")
    replayed = _replay_chain_ids(annotated_path, behaviours)
    return {b: (np.array(v) if v else None) for b, v in replayed.items()}


def require_aligned(
    behaviour: str,
    n_rows: int,
    chain_ids: Optional[np.ndarray],
    context: str = "",
) -> np.ndarray:
    """Hard-fail unless ``chain_ids`` exists and matches the row count.

    Replaces the silent proxy-chain / arange fallbacks: under a proxy id the
    within-chain permutation null is a NO-OP (every "chain" is single-label,
    so the null equals the real value and p≈1.0 while looking legitimate).
    """
    if chain_ids is None:
        raise RuntimeError(
            f"[{context or 'row_provenance'}] no chain ids for '{behaviour}' "
            f"(N={n_rows}). Refusing the proxy-chain fallback — it silently "
            f"neuters the chain-stratified null. Re-extract activations to get "
            f"{ROW_INDEX_FILE}, or pass the matching annotation file."
        )
    if len(chain_ids) != n_rows:
        raise RuntimeError(
            f"[{context or 'row_provenance'}] chain-id/row mismatch for "
            f"'{behaviour}': {len(chain_ids)} ids vs {n_rows} rows. The "
            f"annotation file does not correspond to this extraction (or the "
            f"extraction predates the occurrence-aware matcher). Re-extract "
            f"rather than analysing misaligned rows."
        )
    return np.asarray(chain_ids)


# ── Exact-duplicate hygiene ───────────────────────────────────────────────────

def duplicate_fraction(X: np.ndarray) -> float:
    """Fraction of rows that are exact duplicates of an earlier row."""
    if X.shape[0] == 0:
        return 0.0
    n_unique = np.unique(X, axis=0).shape[0]
    return float(1.0 - n_unique / X.shape[0])


def dedup_rows(X: np.ndarray, *aligned: np.ndarray):
    """Drop exact-duplicate rows (keep first occurrence, preserve order).

    Any *aligned* per-row arrays (chain ids, labels) are subset identically.
    Returns ``(X_unique, *aligned_unique)``.
    """
    _, keep = np.unique(X, axis=0, return_index=True)
    keep = np.sort(keep)
    return (X[keep], *[np.asarray(a)[keep] for a in aligned])


__all__ = [
    "ROW_INDEX_FILE",
    "load_row_index",
    "chain_ids_for",
    "require_aligned",
    "duplicate_fraction",
    "dedup_rows",
]
