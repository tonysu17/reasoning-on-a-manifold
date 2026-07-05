"""Before/after geometry diff for the post-training spillover measurement.

Pure-numpy comparators (no torch) so they are cheap to import and unit-test.
Given per-behaviour activation matrices for the BASE and the SAFETY-POST-TRAINED
model (same generic, non-safety reasoning chains, same layer), quantify how the
geometry of each behaviour moved:

- ``principal_angles``     : angles between the top-k PCA subspaces (subspace rotation);
- ``d_eff``                : participation-ratio effective dimensionality (cheap ID proxy);
- centroid L2 + mean-direction cosine : how the behaviour's centre/mean shifted.

These are the *descriptive* spillover signals (PH1 existence, PH2 selectivity).
NOTE (see memory ``reasoning_taxonomy_factorization`` and the safety extension's
F2/F9 findings): a naive principal-angle diff is necessary but NOT sufficient —
it must be backed by in-sample/out-of-sample splits, label-permutation nulls, a
size-matched non-safety control, and ultimately causal (steering/patching)
tests. This module computes the descriptive layer; the runner pairs it with the
control arm and the report flags the un-nulled metrics.
"""

from __future__ import annotations

import numpy as np


def _center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def subspace_basis(X: np.ndarray, k: int) -> np.ndarray:
    """Top-k principal directions of X as orthonormal rows (k, d)."""
    Xc = _center(X.astype(np.float64))
    # right singular vectors are the principal axes
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = max(1, min(k, Vt.shape[0]))
    return Vt[:k]


def principal_angles(A: np.ndarray, B: np.ndarray, k: int = 5) -> np.ndarray:
    """Principal angles (degrees, ascending) between the top-k subspaces of A,B."""
    Ba = subspace_basis(A, k)
    Bb = subspace_basis(B, k)
    kk = min(Ba.shape[0], Bb.shape[0])
    M = Ba[:kk] @ Bb[:kk].T
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


def d_eff(X: np.ndarray) -> float:
    """Participation-ratio effective dimensionality: (Σλ)² / Σλ².

    A cheap, dependency-free dimensionality proxy. (The full TwoNN/MLE
    estimators in ``src/intrinsic_dim.py`` can be swapped in for publication.)
    """
    Xc = _center(X.astype(np.float64))
    # eigenvalues of the covariance = singular values squared / (n-1)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    denom = float((lam ** 2).sum())
    if denom <= 0:
        return 0.0
    return float((lam.sum() ** 2) / denom)


def compare_behaviour(base_X: np.ndarray, post_X: np.ndarray, k: int = 5) -> dict:
    """Descriptive geometry diff for one behaviour at one layer."""
    out: dict = {
        "n_base": int(base_X.shape[0]),
        "n_post": int(post_X.shape[0]),
        "dim": int(base_X.shape[1]) if base_X.ndim == 2 else None,
    }
    # Subspace rotation
    try:
        angles = principal_angles(base_X, post_X, k=k)
        out["principal_angles_deg"] = [round(float(a), 3) for a in angles]
        out["mean_principal_angle_deg"] = round(float(np.mean(angles)), 3)
        out["max_principal_angle_deg"] = round(float(np.max(angles)), 3)
    except Exception as exc:  # noqa: BLE001
        out["principal_angles_deg"] = None
        out["error_angles"] = str(exc)
    # Effective dimensionality
    out["d_eff_base"] = round(d_eff(base_X), 3)
    out["d_eff_post"] = round(d_eff(post_X), 3)
    out["delta_d_eff"] = round(out["d_eff_post"] - out["d_eff_base"], 3)
    # Centroid / mean-direction shift
    mu_b = base_X.mean(axis=0)
    mu_p = post_X.mean(axis=0)
    out["centroid_l2"] = round(float(np.linalg.norm(mu_p - mu_b)), 4)
    nb, npp = np.linalg.norm(mu_b), np.linalg.norm(mu_p)
    out["mean_direction_cos"] = (
        round(float(mu_b @ mu_p / (nb * npp)), 4) if nb > 0 and npp > 0 else None
    )
    return out


def compare_geometry(
    base_acts: dict,
    post_acts: dict,
    behaviours: list[str],
    layer: int,
    k: int = 5,
) -> dict:
    """Diff per-behaviour activations at one layer.

    ``base_acts`` / ``post_acts`` are ``{behaviour: {layer: ndarray}}`` (the
    shape returned by ``src.activation_extraction.load_all_activations``).
    """
    report = {"layer": layer, "k": k, "behaviours": {}}
    for b in behaviours:
        bX = base_acts.get(b, {}).get(layer)
        pX = post_acts.get(b, {}).get(layer)
        if bX is None or pX is None:
            report["behaviours"][b] = {"skipped": "missing activations"}
            continue
        report["behaviours"][b] = compare_behaviour(np.asarray(bX), np.asarray(pX), k=k)
    return report


def rank_selectivity(report: dict) -> list[tuple[str, float]]:
    """Order behaviours by how much they moved (mean principal angle, desc).

    This is the PH2 'selectivity' readout: which reasoning types are most
    safety-entangled. Behaviours lacking a valid angle are dropped.
    """
    scored = []
    for b, r in report.get("behaviours", {}).items():
        ang = r.get("mean_principal_angle_deg")
        if ang is not None:
            scored.append((b, ang))
    return sorted(scored, key=lambda x: x[1], reverse=True)
