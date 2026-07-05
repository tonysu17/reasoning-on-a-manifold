"""
composition_check.py — READ-ONLY diagnostic (no pipeline rerun, no model, no cluster).

Question: is each of the four reasoning-behaviour directions a COMPOSITION of the
other three? Validates the thesis's own instrument (src.cbs.geometry) on the
existing layer-27 artifacts before any niche/safety application.

Three tests, each against a null:
  1. Gram matrix          — pairwise cosine of the four saved single steering vectors
                            (are the "atoms" near-orthogonal? — the gating condition
                            for linear composability).
  2. Vector reconstruction — v_b ≈ Σ_{j≠b} s_j v_j  (least squares); residual ratio
                            & weights vs a random-direction null reconstructed from
                            the same three vectors.
  3. Subspace-in-span     — top-k_auto PCA subspace of behaviour b vs the union span
                            of the other three: principal angles + the saved steering
                            vector's off-subspace residual, vs a random-direction null.

Reads:  results/steering_vectors/R1-1.5B/{beh}_single.npy
        data/activations/R1-1.5B/{beh}_layer27.npy
Writes: results/composition/composition_check.{json,md}  (additive; safe to delete)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.cbs.geometry import principal_angles, build_union_basis, out_of_subspace_residual

LAYER = 27
SEED = 42
BEHAVIOURS = ["backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"]
AUTO_K = {"backtracking": 58, "uncertainty-estimation": 71, "example-testing": 60, "adding-knowledge": 83}
VEC_DIR = ROOT / "results/steering_vectors/R1-1.5B"
ACT_DIR = ROOT / "data/activations/R1-1.5B"
OUT_DIR = ROOT / "results/composition"

rng = np.random.default_rng(SEED)


def deg(a):
    return np.round(np.degrees(a), 2)


def random_unit(n, d):
    g = rng.standard_normal((n, d))
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def main():
    # ── Load the four saved single steering vectors (the real artifacts) ──
    vecs = {b: np.load(VEC_DIR / f"{b}_single.npy").astype(np.float64) for b in BEHAVIOURS}
    for b, v in vecs.items():
        vecs[b] = v / np.linalg.norm(v)  # ensure unit
    d = vecs[BEHAVIOURS[0]].shape[0]

    report = {"layer": LAYER, "seed": SEED, "dim": d, "behaviours": BEHAVIOURS, "auto_k": AUTO_K}

    # ── Test 1: Gram matrix of the four single vectors ──
    V = np.stack([vecs[b] for b in BEHAVIOURS], axis=0)  # (4, d)
    gram = V @ V.T                                        # cosines (unit vectors)
    report["gram_cosine"] = {BEHAVIOURS[i]: {BEHAVIOURS[j]: round(float(gram[i, j]), 4)
                                             for j in range(4)} for i in range(4)}
    offdiag = gram[~np.eye(4, dtype=bool)]
    report["gram_offdiag_abs_mean"] = round(float(np.abs(offdiag).mean()), 4)
    report["gram_offdiag_abs_max"] = round(float(np.abs(offdiag).max()), 4)

    # ── Test 2: vector reconstruction v_b ≈ Σ_{j≠b} s_j v_j ──
    null_u = random_unit(5000, d)
    test2 = {}
    for i, b in enumerate(BEHAVIOURS):
        others = [BEHAVIOURS[j] for j in range(4) if j != i]
        A = np.stack([vecs[o] for o in others], axis=1)        # (d, 3)
        s, *_ = np.linalg.lstsq(A, vecs[b], rcond=None)
        recon = A @ s
        resid = float(np.linalg.norm(vecs[b] - recon) / np.linalg.norm(vecs[b]))
        # null: random unit vectors reconstructed from the SAME three
        coef_n, *_ = np.linalg.lstsq(A, null_u.T, rcond=None)   # (3, 5000)
        recon_n = (A @ coef_n).T                                # (5000, d)
        resid_n = np.linalg.norm(null_u - recon_n, axis=1) / np.linalg.norm(null_u, axis=1)
        test2[b] = {
            "weights": {o: round(float(w), 3) for o, w in zip(others, s)},
            "residual_ratio": round(resid, 4),
            "R2": round(1 - resid**2, 4),
            "null_residual_mean": round(float(resid_n.mean()), 4),
            "null_residual_sd": round(float(resid_n.std()), 4),
            "z_vs_null": round(float((resid - resid_n.mean()) / resid_n.std()), 2),
        }
    report["test2_vector_reconstruction"] = test2

    # ── Build per-behaviour top-k_auto PCA subspaces from the ON activations ──
    pcs = {}
    for b in BEHAVIOURS:
        X = np.load(ACT_DIR / f"{b}_layer{LAYER}.npy").astype(np.float64)
        k = AUTO_K[b]
        pca = PCA(n_components=k, svd_solver="full")
        pca.fit(X)
        pcs[b] = pca.components_.T  # (d, k) — columns are orthonormal PC directions
    report["activation_rows"] = {b: int(np.load(ACT_DIR / f'{b}_layer{LAYER}.npy').shape[0])
                                 for b in BEHAVIOURS}

    # ── Test 3: subspace-in-span (principal angles) + steering-vec off-span residual ──
    test3 = {}
    for i, b in enumerate(BEHAVIOURS):
        others = {BEHAVIOURS[j]: pcs[BEHAVIOURS[j]] for j in range(4) if j != i}
        B = build_union_basis(others, variance_threshold=0.999)   # (d, m) orthonormal span of the 3
        m = B.shape[1]
        Vb = pcs[b]                                               # (d, k_b)
        angles = principal_angles(B, Vb, top_k=Vb.shape[1])       # k_b angles, ascending
        cos2 = np.cos(angles) ** 2
        # saved steering vector's residual outside the 3-others span:
        resid_vec = float(out_of_subspace_residual(vecs[b][None, :], B)[0])
        # null: random unit directions' residual outside the same m-dim span:
        resid_null = out_of_subspace_residual(random_unit(5000, d), B)
        test3[b] = {
            "k_b": int(Vb.shape[1]),
            "union_span_dim_m": int(m),
            "principal_angle_min_deg": float(deg(angles[0])),
            "principal_angle_median_deg": float(deg(np.median(angles))),
            "principal_angle_max_deg": float(deg(angles[-1])),
            "frac_angles_lt_30deg": round(float(np.mean(angles < np.radians(30))), 3),
            "mean_cos2_energy_in_span": round(float(cos2.mean()), 3),
            "steervec_offspan_residual": round(resid_vec, 4),
            "steervec_offspan_null_mean": round(float(resid_null.mean()), 4),
            "steervec_offspan_null_sd": round(float(resid_null.std()), 4),
            "steervec_z_vs_null": round(float((resid_vec - resid_null.mean()) / resid_null.std()), 2),
        }
    report["test3_subspace_in_span"] = test3

    # ── Persist ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "composition_check.json").write_text(json.dumps(report, indent=2))

    # ── Console report ──
    P = print
    P("\n" + "=" * 78)
    P(f"COMPOSITION DIAGNOSTIC — R1-1.5B, layer {LAYER}, d={d}  (seed {SEED})")
    P("=" * 78)

    P("\n[1] GRAM — cosine between the four single steering vectors")
    P("    " + "".join(f"{b[:6]:>9}" for b in BEHAVIOURS))
    for i, b in enumerate(BEHAVIOURS):
        P(f"{b[:18]:>18} " + "".join(f"{gram[i,j]:>9.3f}" for j in range(4)))
    P(f"    off-diagonal |cos|: mean={report['gram_offdiag_abs_mean']}  max={report['gram_offdiag_abs_max']}")

    P("\n[2] VECTOR RECONSTRUCTION  v_b = Σ s·(other 3)   (resid 0=perfect, ~1=independent)")
    P(f"    {'behaviour':>18} {'resid':>7} {'R2':>7} {'null':>7} {'z':>7}   weights")
    for b in BEHAVIOURS:
        t = test2[b]
        w = " ".join(f"{o[:4]}={t['weights'][o]:+.2f}" for o in t["weights"])
        P(f"    {b:>18} {t['residual_ratio']:>7.3f} {t['R2']:>7.3f} "
          f"{t['null_residual_mean']:>7.3f} {t['z_vs_null']:>7.1f}   {w}")

    P("\n[3] SUBSPACE-IN-SPAN  (behaviour's k_auto PCA subspace vs union span of other 3)")
    P(f"    {'behaviour':>18} {'k_b':>4} {'m':>4} {'ang_min':>8} {'ang_med':>8} "
      f"{'cos2':>6} {'vec_res':>8} {'null':>6} {'z':>7}")
    for b in BEHAVIOURS:
        t = test3[b]
        P(f"    {b:>18} {t['k_b']:>4} {t['union_span_dim_m']:>4} "
          f"{t['principal_angle_min_deg']:>7.1f}° {t['principal_angle_median_deg']:>7.1f}° "
          f"{t['mean_cos2_energy_in_span']:>6.2f} {t['steervec_offspan_residual']:>8.3f} "
          f"{t['steervec_offspan_null_mean']:>6.3f} {t['steervec_z_vs_null']:>7.1f}")

    P("\nwrote: results/composition/composition_check.json")
    P("=" * 78)


if __name__ == "__main__":
    main()
