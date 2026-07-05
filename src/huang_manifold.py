"""Huang-faithful pooled-class PCA manifold construction (arXiv:2505.22411v2).

This module is a *parallel, isolated* counterpart to ``src/steering.py``. The
two differ in exactly one substantive place — which set the PCA is fit on:

  * ``src/steering.py`` (Venhoff arm, the live build) fits PCA on the **ON
    activations only** (``pca.fit(on_activations)``), then projects the
    difference-of-means ``r = mean(ON) − mean(OFF)`` onto that ON-only basis.

  * THIS module fits PCA on the **POOLED, mean-centered** two-class set
    ``vstack(ON, OFF)`` (Huang Eq. 5: ``D_reasoning = D_redundant ∪ D_concise``)
    and projects ``r`` onto the pooled top-k (Huang Eq. 9).

Why pooling is load-bearing (Huang's claim, confirmed sound for our setting):
mean-centering the *pooled* cloud puts each class at ≈ ±½·r about the global
mean, so the pooled covariance gains a rank-1 term ∝ r·rᵀ along the
between-class axis. If the classes are separated relative to within-class
spread, that axis is a HIGH-variance direction and lands in the top-k, so
``P_M r ≈ r`` (the manifold projection preserves the direction while stripping
orthogonal ``r_other`` noise). An ON-only fit centers on the ON mean and keeps
only within-ON variance — the ON↔OFF separation axis is removed by centering and
need not appear in the top-k at all. Pooling is what guarantees the separation
axis is representable in M.

The construction is deliberately kept drop-in compatible with
``src/steering.py``: it reuses ``single_direction_vector`` for the saved
``single_direction`` arm (so the single-vector arm is byte-identical across the
two modules) and ``save_steering_vectors`` / ``load_steering_vectors`` for I/O
(construction-agnostic), so the ``__huang`` artefacts are consumable by
``07_evaluate_steering.py`` and ``src/steered_inference.py`` unchanged.

Deviations from Huang's paper, recorded (see the spec's flagged ambiguities):
  * k is a SWEEP {1,3,5,10,auto≥70%var}; ``huang_k=10`` is the faithful headline
    arm (k=10 is a fixed constant in Huang, justified post-hoc by >70% variance,
    NOT a per-layer VR threshold). ``auto_k`` is reported for completeness only.
  * No whitening anywhere (Huang mentions none) — ``U_eff`` are raw orthonormal
    eigenvectors and ``P_M = U_eff U_effᵀ``.
  * Outlier filtering (IsolationForest, Huang's direction set) is NOT applied
    here: our ``r`` reuses ``single_direction_vector`` over the full matrices so
    the single arm matches the existing build. The asymmetry Huang applies
    (filter the 100-sample direction set, NOT the 500-sample subspace set) is
    documented but not reproduced, because our ON/OFF inputs are the same matrices
    for both r and the subspace (we do not have Huang's separate 100/500 split).
  * Single ``r_steer`` per behaviour; layer indexing follows our hooking code.

A small **robustness battery** (split-half principal angles, eigengap /
Davis–Kahan, Horn parallel analysis, random-subspace null) is included because
Huang reports NO PCA stability check — these are pure numpy and live here so the
runner can call them. Method refs in the function docstrings.

Pure / GPU-free / numpy-only (sklearn used only for the PCA fit, matching the
repo standard ``svd_solver="full"``). Stub-testable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

from src.annotation import TARGET_BEHAVIOURS
# Reuse the EXACT single-direction + I/O from the Venhoff module so the single
# arm and on-disk layout are identical across constructions.
from src.steering import (
    single_direction_vector,
    save_steering_vectors,
    load_steering_vectors,
)

logger = logging.getLogger(__name__)


# ── Core Huang pooled-PCA construction ────────────────────────────────────────

def _pooled_pca(
    on_activations: np.ndarray,
    off_activations: np.ndarray,
    n_components: int,
) -> PCA:
    """Fit a full-SVD PCA on the POOLED, mean-centered two-class set.

    ``sklearn.decomposition.PCA`` always mean-centers internally (it stores
    ``pca.mean_`` and subtracts it before the SVD), so fitting on the
    concatenation centers on the **pooled grand mean** — exactly Huang Eq. 5
    (center the combined class cloud, not each class separately). Do NOT
    pre-center; that would double-center.

    ``svd_solver="full"`` is the project standard for determinism (the default
    "auto" picks randomized SVD, which is non-reproducible without a fixed
    ``random_state``); see ``src/pca.py`` and ``src/steering.py``.
    """
    pooled = np.concatenate([on_activations, off_activations], axis=0)
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(pooled)
    return pca


def pooled_auto_k(
    on_activations: np.ndarray,
    off_activations: np.ndarray,
    variance_threshold: float = 0.70,
) -> int:
    """Smallest k whose top-k POOLED PCs explain >= ``variance_threshold``.

    Huang's analogue is fit on the pooled ``D_reasoning`` set (consistency with
    the pooled subspace fit), not ON-only. Capped at 100 components like
    ``src/steering.auto_k``. NOTE the pooled spectrum differs from the ON-only
    spectrum, so this ``auto_k`` will differ from the ON-only value recorded in
    the existing ``R1-1.5B__venhoff/metadata.json``.
    """
    pooled_n = on_activations.shape[0] + off_activations.shape[0]
    max_k = min(pooled_n - 1, on_activations.shape[1], 100)
    if max_k < 1:
        return 1
    pca = _pooled_pca(on_activations, off_activations, max_k)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    idx = int(np.searchsorted(cumvar, variance_threshold))
    return min(idx + 1, max_k)


def pooled_manifold_vector(
    on_activations: np.ndarray,
    off_activations: np.ndarray,
    k: int,
) -> np.ndarray:
    """Huang-faithful manifold-projected steering vector.

    Steps (Huang Eqs. 2 → 5 → 9):
      r       = mean(ON) − mean(OFF)                      (diff-of-means, Eq. 2)
      U_eff   = top-k PCs of PCA fit on vstack(ON, OFF)   (pooled, Eq. 5; no whitening)
      r_steer = renorm( (U_eff U_effᵀ) r )                (project onto M, Eq. 9)

    The ONE substantive difference from ``manifold_projected_vector`` in
    ``src/steering.py`` is that ``U_eff`` comes from the **pooled** fit, not the
    ON-only fit.

    Guards mirror ``src/steering.py`` (load-bearing — see tests/test_steering.py):
      * empty ON/OFF → raise (via ``single_direction_vector``);
      * n_components < 1 (too few rows) → fall back to the unit ``r``;
      * non-finite / near-zero projected norm → fall back to ``r``.

    Returns a unit-norm vector of shape (hidden_dim,).
    """
    r = single_direction_vector(on_activations, off_activations)

    pooled_n = on_activations.shape[0] + off_activations.shape[0]
    n_components = min(k, pooled_n - 1, on_activations.shape[1])
    if n_components < 1:
        logger.warning(
            f"Cannot project: k={k} but pooled set has only {pooled_n} samples"
        )
        return r

    pca = _pooled_pca(on_activations, off_activations, n_components)
    U = pca.components_  # (k, hidden_dim) — rows are the top-k pooled eigenvectors

    # Project r onto span(U): r_proj = (Uᵀ U) r. We project the (already unit)
    # r; renormalising P r vs P r_unit differs only by a positive scalar.
    coords = U @ r          # (k,)
    r_proj = coords @ U     # (hidden_dim,)

    norm = np.linalg.norm(r_proj)
    if not np.isfinite(norm) or norm < 1e-10:
        logger.warning(
            "Pooled-projected vector has near-zero/non-finite norm — falling back to r"
        )
        return r
    return r_proj / norm


def energy_in_subspace(r_unit: np.ndarray, U: np.ndarray) -> float:
    """Fraction of ``r``'s squared norm captured by the subspace span(U).

    ``E = ‖U Uᵀ r‖² / ‖r‖²`` ∈ [0, 1]. For an orthonormal basis U (rows are
    orthonormal directions, as from PCA ``components_``), this equals
    ``Σ_i (uᵢ·r)² / ‖r‖²``. The diagnostic primitive behind the "how much of the
    difference-of-means lives in the top-k manifold" claim.
    """
    rn = float(np.dot(r_unit, r_unit))
    if rn < 1e-30:
        return 0.0
    coords = U @ r_unit            # (k,)
    return float(np.dot(coords, coords) / rn)


# ── Build all Huang manifold vectors for a model ──────────────────────────────

def build_huang_manifold_vectors(
    activations_dir: Path,
    layers: dict,
    behaviours: Optional[list[str]] = None,
    k_values=(1, 3, 5, 10, "auto"),
    huang_k: int = 10,
    variance_threshold: float = 0.70,
    exclude_chain_ids: Optional[set] = None,
    annotated_path: Optional[Path] = None,
) -> dict:
    """Build Huang pooled-PCA manifold vectors for all target behaviours.

    Unlike ``build_steering_vectors`` (single ``layer``) this takes a
    per-behaviour ``layers`` dict, because the Venhoff layers differ per
    behaviour (17/18/15/18). Behaviours are grouped by layer so each layer's
    four matrices are loaded once; OFF = concat of the OTHER three behaviours at
    the **same** layer.

    The hold-out machinery is carried over verbatim from ``src/steering.py``:
    rows whose chain id is in ``exclude_chain_ids`` are dropped from BOTH ON and
    OFF **before** the pooled fit, so Phase 7 stays a true out-of-sample test.

    Per-behaviour result dict matches ``build_steering_vectors`` plus diagnostic
    fields:
        {
          "layer", "single_direction", "manifold_projected": {k: vec},
          "n_on", "n_off", "auto_k", "n_excluded",
          "energy_in_manifold": {k: float},     # ‖P_k r‖²/‖r‖²  (pooled)
          "cos_single_manifold": {k: float},    # cos(manifold_k, single)
          "cumulative_variance": {k: float},    # pooled cumulative VR at k
          "huang_k": int,                       # which k is the faithful arm
        }
    """
    if behaviours is None:
        behaviours = list(TARGET_BEHAVIOURS)
    behaviours = list(behaviours)
    k_values = list(k_values)

    activations_dir = Path(activations_dir)
    results: dict = {}

    chain_id_map = None
    if exclude_chain_ids:
        from src.row_provenance import chain_ids_for
        exclude_chain_ids = set(exclude_chain_ids)
        chain_id_map = chain_ids_for(activations_dir, annotated_path, behaviours)

    # Group behaviours by layer; load each layer's matrices (with hold-out) once.
    from collections import defaultdict
    by_layer: dict = defaultdict(list)
    for b in behaviours:
        if b not in layers:
            logger.warning(f"No layer specified for {b} — skipping")
            continue
        by_layer[layers[b]].append(b)

    for L in sorted(by_layer):
        layer_behs = by_layer[L]
        all_acts: dict[str, np.ndarray] = {}
        n_excluded: dict[str, int] = {}
        # Load EVERY behaviour at this layer (not just layer_behs): a behaviour
        # built here needs the OTHER behaviours' activations AT THIS LAYER as its
        # OFF set, even though those others are built at their own layers. Loading
        # only the same-layer-group behaviours left singletons (backtracking@17,
        # example-testing@15) with no OFF, and paired groups with too small an OFF.
        for beh in behaviours:
            path = activations_dir / f"{beh}_layer{L}.npy"
            if not path.exists():
                logger.warning(f"Missing: {path}")
                continue
            X = np.load(path).astype(np.float32)
            n_excluded[beh] = 0
            if exclude_chain_ids:
                from src.row_provenance import require_aligned
                cids = require_aligned(beh, X.shape[0], chain_id_map.get(beh),
                                       context="huang hold-out")
                keep = ~np.isin(cids, list(exclude_chain_ids))
                n_excluded[beh] = int((~keep).sum())
                X = X[keep]
            all_acts[beh] = X
            logger.info(
                f"  {beh} @ L{L}: {X.shape[0]} instances loaded"
                + (f" ({n_excluded[beh]} eval-task rows held out)"
                   if exclude_chain_ids else "")
            )

        for beh in layer_behs:
            if beh not in all_acts:
                continue
            on_acts = all_acts[beh]
            off_parts = [v for kk, v in all_acts.items() if kk != beh]
            if not off_parts:
                logger.warning(f"No off-activations for {beh} @ L{L} — skipping")
                continue
            off_acts = np.concatenate(off_parts, axis=0)

            k_auto = pooled_auto_k(on_acts, off_acts, variance_threshold)
            r_single = single_direction_vector(on_acts, off_acts)

            # Fit the pooled PCA ONCE at the largest needed k, then slice
            # components_[:k] for the smaller k (the top PCs are shared). The
            # "auto" arm may need a larger k than the {1,3,5,10} sweep.
            sweep_int = sorted({(k_auto if k == "auto" else int(k))
                                for k in k_values})
            pooled_n = on_acts.shape[0] + off_acts.shape[0]
            k_max = min(max(sweep_int), pooled_n - 1, on_acts.shape[1])
            U_full = None
            cumvar_full = None
            if k_max >= 1:
                pca_full = _pooled_pca(on_acts, off_acts, k_max)
                U_full = pca_full.components_           # (k_max, d)
                cumvar_full = np.cumsum(pca_full.explained_variance_ratio_)

            manifold: dict = {}
            energy: dict = {}
            cos_single: dict = {}
            cumvar_at: dict = {}
            for k in k_values:
                k_int = k_auto if k == "auto" else int(k)
                if U_full is not None and 1 <= k_int <= U_full.shape[0]:
                    Uk = U_full[:k_int]
                    coords = Uk @ r_single
                    r_proj = coords @ Uk
                    norm = np.linalg.norm(r_proj)
                    vec = (r_proj / norm
                           if np.isfinite(norm) and norm >= 1e-10 else r_single)
                    energy[k] = energy_in_subspace(r_single, Uk)
                    cumvar_at[k] = float(cumvar_full[k_int - 1])
                else:
                    # Fallback path (too few rows): pooled_manifold_vector clamps.
                    vec = pooled_manifold_vector(on_acts, off_acts, k_int)
                    energy[k] = float("nan")
                    cumvar_at[k] = float("nan")
                manifold[k] = vec
                cos_single[k] = float(np.dot(r_single, vec))

            results[beh] = {
                "layer": L,
                "single_direction": r_single,
                "manifold_projected": manifold,
                "n_on": len(on_acts),
                "n_off": len(off_acts),
                "auto_k": k_auto,
                "n_excluded": n_excluded.get(beh, 0),
                "energy_in_manifold": energy,
                "cos_single_manifold": cos_single,
                "cumulative_variance": cumvar_at,
                "huang_k": huang_k,
            }
            logger.info(
                f"  {beh}: auto_k={k_auto}, n_on={len(on_acts)}, "
                f"n_off={len(off_acts)}, cos(single,k{huang_k})="
                f"{cos_single.get(huang_k, float('nan')):.4f}"
            )

    return results


# ── ON-only comparison (for the side-by-side diagnostic) ──────────────────────

def on_only_manifold_components(on_activations: np.ndarray, k: int) -> np.ndarray:
    """Top-k PCA components of an ON-ONLY fit (Venhoff/old construction).

    Used purely by the diagnostic to recompute the OLD ``E_only[k]`` /
    ``cos_only[k]`` columns side-by-side with the pooled ones. ``svd_solver``
    "full" matches ``src/steering.py``.
    """
    n_components = min(k, on_activations.shape[0] - 1, on_activations.shape[1])
    if n_components < 1:
        return np.zeros((0, on_activations.shape[1]))
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(on_activations)
    return pca.components_


# ── Robustness battery (Huang reports NONE — pure numpy) ───────────────────────

def pca_basis(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k PCA basis of X via SVD of the *centered* matrix.

    ``Xc = X − X.mean(0); Xc = U S Vt`` (full_matrices=False). Returns
    ``(Q, eigvals)`` where Q is (d, k) with ORTHONORMAL columns = top-k right
    singular vectors (``Vt[:k].T``), and ``eigvals`` are the PCA eigenvalues
    ``S² / (N−1)`` descending. Center, never z-score (z-scoring changes the
    subspace; keep the activation metric the steering vectors live in). Computed
    in float64.

    Refs: Björck & Golub (1973), Math. Comp. 27; Golub & Van Loan §6.4.3.
    """
    Xc = X.astype(np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    N = Xc.shape[0]
    # full_matrices=False → Vt is (min(N,d), d).
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    kk = min(k, Vt.shape[0])
    Q = Vt[:kk].T.copy()                       # (d, k), orthonormal columns
    eigvals = (S ** 2) / max(N - 1, 1)         # (min(N,d),), descending
    return Q, eigvals


def principal_angles(Q1: np.ndarray, Q2: np.ndarray) -> np.ndarray:
    """Principal (Jordan) angles between two subspaces, ASCENDING, in RADIANS.

    ``M = Q1ᵀ Q2``; singular values of M are the cosines of the principal
    angles. Requires Q1, Q2 to have orthonormal columns (true from
    ``pca_basis``). Clipped to [-1, 1] to guard fp drift.

    Refs: Björck & Golub (1973); Golub & Van Loan §6.4.3 (Grassmann distance).
    """
    M = Q1.T @ Q2
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    theta = np.arccos(s)                        # largest cos → smallest angle
    return np.sort(theta)                       # ascending


def subspace_energy(R: np.ndarray, Q: np.ndarray) -> float:
    """Fraction of R's total (centered) variance captured by span(Q).

    ``Rc = R − R.mean(0); return Σ(Rc Q)² / Σ Rc²`` ∈ [0,1]; equals k/d in
    expectation for a random Q (Q with orthonormal columns).
    """
    Rc = R.astype(np.float64)
    Rc = Rc - Rc.mean(axis=0, keepdims=True)
    denom = float(np.sum(Rc ** 2))
    if denom < 1e-30:
        return 0.0
    proj = Rc @ Q
    return float(np.sum(proj ** 2) / denom)


def _summ(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    return {
        "mean": float(a.mean()), "sd": float(a.std()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p05": float(np.percentile(a, 5)),
    }


def split_half_stability(
    X: np.ndarray,
    k: int = 10,
    chain_ids: Optional[np.ndarray] = None,
    n_boot: int = 200,
    seed: int = 42,
) -> dict:
    """Split-half / bootstrap principal-angle stability of the top-k subspace.

    For each resample, fit the top-k subspace on two halves and measure their
    principal angles. Instability ⇒ the manifold is a sampling artefact.

      * CHAIN-AWARE (preferred, ``chain_ids`` given): split the set of UNIQUE
        chain ids 50/50 and assign whole chains to each half — prevents
        same-chain sentences leaking across halves (a row-level split is the
        optimistic, chain-contaminated version; CF-2 is this project's keystone).
      * ROW-LEVEL fallback (``chain_ids`` None): permute rows, split 50/50.

    Returns summaries of max/mean principal angle (degrees), mean overlap
    ``mean(cos²)``, the per-direction 95th-pct angle (exposes a single unstable
    trailing axis), and the split mode used.

    Ref: Efron & Tibshirani (1993) for the resampling distribution of the angle.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    split_mode = "chain-aware" if chain_ids is not None else "row-level"
    if chain_ids is not None:
        uniq = np.unique(np.asarray(chain_ids))
        groups = {c: np.where(np.asarray(chain_ids) == c)[0] for c in uniq}

    max_ang, mean_ang, mean_cos2 = [], [], []
    per_dir = []  # (n_boot, k) padded with nan if a side is rank-deficient
    n_eff = n_boot
    for _ in range(n_boot):
        if chain_ids is not None:
            perm = rng.permutation(len(uniq))
            half = len(uniq) // 2
            ca, cb = uniq[perm[:half]], uniq[perm[half:]]
            idx_a = np.concatenate([groups[c] for c in ca]) if len(ca) else np.array([], int)
            idx_b = np.concatenate([groups[c] for c in cb]) if len(cb) else np.array([], int)
        else:
            perm = rng.permutation(N)
            half = N // 2
            idx_a, idx_b = perm[:half], perm[half:]
        if len(idx_a) <= k or len(idx_b) <= k:
            n_eff -= 1
            continue
        Qa = pca_basis(X[idx_a], k)[0]
        Qb = pca_basis(X[idx_b], k)[0]
        ang = principal_angles(Qa, Qb)              # radians, ascending
        deg = np.degrees(ang)
        max_ang.append(float(deg.max()))
        mean_ang.append(float(deg.mean()))
        mean_cos2.append(float(np.mean(np.cos(ang) ** 2)))
        row = np.full(k, np.nan)
        row[:len(deg)] = deg
        per_dir.append(row)

    if n_eff <= 0 or not max_ang:
        return {"split": split_mode, "n_effective": 0,
                "error": "all resamples rank-deficient for this k"}

    per_dir_arr = np.asarray(per_dir)
    per_direction_p95 = np.nanpercentile(per_dir_arr, 95, axis=0).tolist()
    return {
        "split": split_mode,
        "n_effective": int(n_eff),
        "max_angle_deg": _summ(np.asarray(max_ang)),
        "mean_angle_deg": _summ(np.asarray(mean_ang)),
        "mean_cos2": _summ(np.asarray(mean_cos2)),
        "per_direction_angle_deg_p95": per_direction_p95,
    }


def eigengap_report(X: np.ndarray, k: int = 10, ks_scan=range(1, 31)) -> dict:
    """Eigenvalue gap / scree analysis (Davis–Kahan sinΘ).

    Davis–Kahan (1970): for symmetric C and perturbation E, the sine of the
    largest principal angle between true and estimated top-k eigenspaces is
    bounded by ‖E‖ / gap, gap = λ_k − λ_{k+1}. A vanishing gap ⇒ the k-th
    direction is interchangeable with the (k+1)-th and the subspace boundary is
    arbitrary. The sampling perturbation is crudely scaled as
    ``E_norm ≈ λ₁ / sqrt(N)``.

    Refs: Davis & Kahan (1970), SIAM J. Numer. Anal. 7(1); Yu, Wang & Samworth
    (2015) for the statistical variant.
    """
    ks_scan = list(ks_scan)
    need = max(max(ks_scan) + 1, k + 1)
    _, eigvals = pca_basis(X, min(need, X.shape[0] - 1, X.shape[1]))
    N = X.shape[0]
    total = float(eigvals.sum()) if eigvals.size else 1.0

    def lam(i):  # 0-based safe accessor
        return float(eigvals[i]) if 0 <= i < len(eigvals) else 0.0

    gap_k = lam(k - 1) - lam(k)
    rel_gap_k = gap_k / lam(k - 1) if lam(k - 1) > 0 else 0.0
    E_norm = (lam(0) / np.sqrt(N)) if N > 0 else float("inf")
    sin_bound = min(1.0, E_norm / gap_k) if gap_k > 0 else 1.0
    dk_deg = float(np.degrees(np.arcsin(sin_bound)))

    gap_curve = [lam(j - 1) - lam(j) for j in ks_scan]
    rel_gaps = [(lam(j - 1) - lam(j)) / lam(j - 1) if lam(j - 1) > 0 else 0.0
                for j in ks_scan]
    argmax_gap = int(ks_scan[int(np.argmax(rel_gaps))]) if rel_gaps else k

    top = min(max(ks_scan), len(eigvals))
    cumvar = (np.cumsum(eigvals[:top]) / total).tolist()
    return {
        "k": k,
        "eigvals_top": eigvals[:top].tolist(),
        "cumvar": cumvar,
        "lambda_k": lam(k - 1),
        "lambda_k_plus_1": lam(k),
        "gap_k": float(gap_k),
        "rel_gap_k": float(rel_gap_k),
        "gap_curve": [float(g) for g in gap_curve],
        "davis_kahan_bound_deg": dk_deg,
        "argmax_gap": argmax_gap,
        "E_norm_proxy": float(E_norm),
    }


def horn_parallel_analysis(
    X: np.ndarray,
    n_perm: int = 50,
    percentile: float = 95.0,
    k_huang: int = 10,
    seed: int = 42,
    max_components: int = 60,
) -> dict:
    """Horn parallel analysis for k-selection.

    Horn (1965): build a null by permuting each feature column independently
    (destroys cross-feature covariance, preserves each column's marginal),
    recompute eigenvalues, repeat; keep components whose observed eigenvalue
    exceeds the null's upper ``percentile``. This is the SAME null family as the
    repo's tier1 feature-shuffle nulls.

    Returns ``k_horn`` (count of the leading run of observed > null), the
    observed/null leading spectra (for a scree overlay), and the ratio at
    ``k_huang``.

    Refs: Horn (1965), Psychometrika 30; Glorfeld (1995) upper-percentile variant.
    """
    rng = np.random.default_rng(seed)
    N, d = X.shape
    mc = min(max_components, N - 1, d)
    if mc < 1:
        return {"k_horn": 0, "k_huang": k_huang, "error": "too few rows/dims"}
    obs = pca_basis(X, mc)[1][:mc]

    Xf = X.astype(np.float64)
    null = np.empty((n_perm, mc), dtype=np.float64)
    for p in range(n_perm):
        Xp = np.empty_like(Xf)
        for j in range(d):
            Xp[:, j] = Xf[rng.permutation(N), j]
        null[p] = pca_basis(Xp, mc)[1][:mc]
    null_pct = np.percentile(null, percentile, axis=0)

    # Count the leading run of observed > null (stop at first failure).
    k_horn = 0
    for i in range(mc):
        if obs[i] > null_pct[i]:
            k_horn += 1
        else:
            break
    ratio = (float(obs[k_huang - 1] / null_pct[k_huang - 1])
             if 1 <= k_huang <= mc and null_pct[k_huang - 1] > 0 else float("nan"))
    return {
        "k_horn": int(k_horn),
        "k_huang": int(k_huang),
        "obs_top": obs.tolist(),
        "null_pct_top": null_pct.tolist(),
        "ratio_at_k_huang": ratio,
        "n_perm": int(n_perm),
        "percentile": float(percentile),
    }


def random_subspace_null(
    X_on: np.ndarray,
    X_off: np.ndarray,
    k: int = 10,
    n_rand: int = 200,
    seed: int = 42,
    subspace_X: Optional[np.ndarray] = None,
) -> dict:
    """Random-subspace null: does behaviour variance concentrate in the manifold?

    Builds the manifold ``Q = pca_basis(subspace_X or X_on, k)`` — pass
    ``subspace_X = vstack(ON, OFF)`` (the POOLED set) to test the manifold we
    ACTUALLY steer with (Huang Eq. 5); leave it ``None`` for the ON-only
    subspace. Measures ON and OFF energy in ``Q`` and compares ON energy against
    ``n_rand`` Haar-orthonormal random k-subspaces (QR of a Gaussian). A genuine
    low-rank manifold beats random by a wide margin (random ≈ k/d).
    ``specificity = e_on − e_off`` is the meaningful comparison; defer its
    verdict to the existing specificity analysis (sanity floor — treat a FAIL as
    a code/data bug).
    """
    rng = np.random.default_rng(seed)
    d = X_on.shape[1]
    Q = pca_basis(X_on if subspace_X is None else subspace_X, k)[0]
    e_on = subspace_energy(X_on, Q)
    e_off = subspace_energy(X_off, Q)
    e_rand = np.empty(n_rand, dtype=np.float64)
    for i in range(n_rand):
        G = rng.standard_normal((d, k))
        Qr, _ = np.linalg.qr(G)              # (d, k) orthonormal columns
        e_rand[i] = subspace_energy(X_on, Qr)
    mean_r = float(e_rand.mean())
    sd_r = float(e_rand.std())
    z_on = (e_on - mean_r) / sd_r if sd_r > 1e-12 else float("inf")
    fold = e_on / mean_r if mean_r > 1e-12 else float("inf")
    return {
        "e_on": float(e_on),
        "e_off": float(e_off),
        "specificity": float(e_on - e_off),
        "rand_on": {"mean": mean_r, "sd": sd_r,
                    "p97_5": float(np.percentile(e_rand, 97.5))},
        "baseline_k_over_d": float(k / d),
        "z_on": float(z_on),
        "fold_over_random": float(fold),
    }


# ── Rubric helpers (PASS / BORDERLINE / FAIL) ─────────────────────────────────

def grade_split_half(c1: dict) -> str:
    if not c1 or "max_angle_deg" not in c1:
        return "ERROR"
    mx = c1["max_angle_deg"]["p95"]
    mn = c1["mean_angle_deg"]["mean"]
    cz = c1["mean_cos2"]["p05"]
    if mx < 25 and mn < 15 and cz > 0.90:
        return "PASS"
    if mx > 45 or mn > 30 or cz < 0.75:
        return "FAIL"
    return "BORDERLINE"


def grade_eigengap(c2: dict) -> str:
    rg = c2.get("rel_gap_k", 0.0)
    dk = c2.get("davis_kahan_bound_deg", 90.0)
    if rg > 0.10 and dk < 20:
        return "PASS"
    if rg < 0.02 or dk > 45:
        return "FAIL"
    return "BORDERLINE"


def grade_random_null(c4: dict) -> str:
    if c4["e_on"] > c4["rand_on"]["p97_5"] and c4["fold_over_random"] > 5 and c4["z_on"] > 10:
        return "PASS"
    if c4["e_on"] <= c4["rand_on"]["p97_5"] or c4["z_on"] < 3:
        return "FAIL"
    return "BORDERLINE"


def aggregate_verdict(c1: dict, c2: dict, c4: dict) -> str:
    """PASS = check1 PASS and check4 PASS and (check2 PASS or BORDERLINE).
    FAIL = check1 FAIL or check4 FAIL. Else BORDERLINE."""
    g1, g2, g4 = grade_split_half(c1), grade_eigengap(c2), grade_random_null(c4)
    if g1 == "FAIL" or g4 == "FAIL":
        return "FAIL"
    if g1 == "PASS" and g4 == "PASS" and g2 in ("PASS", "BORDERLINE"):
        return "PASS"
    return "BORDERLINE"


__all__ = [
    "pooled_manifold_vector",
    "pooled_auto_k",
    "energy_in_subspace",
    "build_huang_manifold_vectors",
    "on_only_manifold_components",
    "save_steering_vectors",
    "load_steering_vectors",
    "single_direction_vector",
    # robustness battery
    "pca_basis",
    "principal_angles",
    "subspace_energy",
    "split_half_stability",
    "eigengap_report",
    "horn_parallel_analysis",
    "random_subspace_null",
    "grade_split_half",
    "grade_eigengap",
    "grade_random_null",
    "aggregate_verdict",
]
