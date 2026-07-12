#!/usr/bin/env python3
"""pt05: recalibrated rotation test for the spillover claim (statistician review F1/F2/F5).

The executed rotation test (`matched_angle_observed` vs `within_model_null`) is
structurally biased: the observed statistic draws one subsample per model
INDEPENDENTLY (expected shared-row fraction m/N, shared rows near-identical across
models, pulling the cross-model angle DOWN), while the null uses DISJOINT pairs.
Under H0 the "excess" is therefore negative by construction — reproduced empirically
at -0.5 to -2.7 deg on base-vs-base — which both manufactured the negative excesses
reported across all 8+32 cells and could mask a small true rotation.

Redesign (H0-mean zero by construction), unit = chains:
  for j in 1..R random chain partitions into halves A/B:
    cross_j  = angle(base[A], post[B])                       (disjoint rows+chains)
    within_j = 0.5 * (angle(base[A], base[B]) + angle(post[A], post[B]))
    excess_j = cross_j - within_j
  report mean excess, percentile interval over splits, and equivalence vs 1 deg.
Angle = mean of top-k=5 principal angles between centered PCA subspaces (identical
convention to src/safety_posttrain/spillover.py).

Also: minimum-detectable-effect curve (I2) — synthetic rotation of the post
activations by theta in the plane of base PC1<->PC6, same statistic, so the null
claim can be stated as "no rotation above theta*".

Run:  python3 pt05_rotation_halfsplit.py
Out:  results/safety_posttrain/pt05_rotation_halfsplit.json
"""
import json
import time
import numpy as np
from pathlib import Path
from scipy.linalg import eigh

RNG_SEED = 42
K = 5
LAYERS = [12, 16]
BEHAVIOURS = ["backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"]
ACT = Path("data/activations")
BASE = "R1-1.5B"
ARMS = {
    "star1_full": ("STAR1-1.5B", 200),
    "safety100": ("R1-1.5B-lora-safety100", 100),
    "safety300": ("R1-1.5B-lora-safety300", 100),
    "safety1000": ("R1-1.5B-lora-safety1000", 100),
    "control1000": ("R1-1.5B-lora-control1000", 100),
}
MDE_CELLS = [("backtracking", 12), ("backtracking", 16), ("adding-knowledge", 12), ("adding-knowledge", 16)]
MDE_THETAS = [0.5, 1.0, 2.0, 5.0]
MDE_R = 50
OUT = Path("results/safety_posttrain/pt05_rotation_halfsplit.json")


def topk_basis(X: np.ndarray, k: int = K) -> np.ndarray:
    """Top-k principal directions (k, d) via covariance eigh (fast path,
    same subspace as spillover.subspace_basis)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc).astype(np.float64)
    d = C.shape[0]
    _, V = eigh(C, subset_by_index=[d - k, d - 1])
    return V.T[::-1]  # descending eigenvalue order


def mean_angle(Ba: np.ndarray, Bb: np.ndarray) -> float:
    s = np.clip(np.linalg.svd(Ba @ Bb.T, compute_uv=False), -1.0, 1.0)
    return float(np.degrees(np.arccos(s)).mean())


def chain_groups(behaviour: str) -> np.ndarray:
    ri = json.load(open(ACT / BASE / "row_index.json"))
    return np.array([r["chain_id"] for r in ri["rows"][behaviour]])


def halfsplit_excess(Xb: np.ndarray, Xp: np.ndarray, chains: np.ndarray,
                     R: int, rng: np.random.Generator) -> np.ndarray:
    uniq = np.unique(chains)
    excess = np.empty(R)
    for j in range(R):
        perm = rng.permutation(len(uniq))
        half_a = set(uniq[perm[: len(uniq) // 2]])
        mask = np.fromiter((c in half_a for c in chains), dtype=bool, count=len(chains))
        bA, bB = topk_basis(Xb[mask]), topk_basis(Xb[~mask])
        pA, pB = topk_basis(Xp[mask]), topk_basis(Xp[~mask])
        cross = mean_angle(bA, pB)
        within = 0.5 * (mean_angle(bA, bB) + mean_angle(pA, pB))
        excess[j] = cross - within
    return excess


def rotate_in_plane(X: np.ndarray, u: np.ndarray, v: np.ndarray, theta_deg: float) -> np.ndarray:
    """Rotate rows of X (about their mean) by theta in the (u, v) plane."""
    th = np.radians(theta_deg)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    cu, cv = Xc @ u, Xc @ v
    Xr = (Xc
          + np.outer(cu * (np.cos(th) - 1) - cv * np.sin(th), u)
          + np.outer(cu * np.sin(th) + cv * (np.cos(th) - 1), v))
    return Xr + mu


def summ(e: np.ndarray) -> dict:
    return {"mean_excess_deg": round(float(e.mean()), 3),
            "split_iqr95": [round(float(np.percentile(e, 2.5)), 3), round(float(np.percentile(e, 97.5)), 3)],
            "frac_above_1deg": round(float((e > 1.0).mean()), 3)}


def main():
    rng = np.random.default_rng(RNG_SEED)
    report = {"seed": RNG_SEED, "k": K, "design": "chain-partition half-split; excess = cross - within, H0-mean 0",
              "arms": {}, "mde": {}}
    for arm, (dirname, R) in ARMS.items():
        report["arms"][arm] = {}
        for beh in BEHAVIOURS:
            chains = chain_groups(beh)
            for layer in LAYERS:
                t0 = time.time()
                Xb = np.load(ACT / BASE / f"{beh}_layer{layer}.npy")
                Xp = np.load(ACT / dirname / f"{beh}_layer{layer}.npy")
                e = halfsplit_excess(Xb, Xp, chains, R, rng)
                report["arms"][arm][f"{beh}_L{layer}"] = summ(e)
                print(f"{arm} {beh} L{layer}: {report['arms'][arm][f'{beh}_L{layer}']} ({time.time()-t0:.0f}s)", flush=True)

    # MDE: synthetic rotation injected into the STAR1-post matrix
    for beh, layer in MDE_CELLS:
        chains = chain_groups(beh)
        Xb = np.load(ACT / BASE / f"{beh}_layer{layer}.npy")
        Xp = np.load(ACT / "STAR1-1.5B" / f"{beh}_layer{layer}.npy")
        basis = topk_basis(Xb, 6)
        u, v = basis[0], basis[5]
        v = v - (v @ u) * u
        v /= np.linalg.norm(v)
        cell = {}
        for th in MDE_THETAS:
            e = halfsplit_excess(Xb, rotate_in_plane(Xp.astype(np.float64), u, v, th), chains, MDE_R, rng)
            cell[str(th)] = summ(e)
            print(f"MDE {beh} L{layer} theta={th}: {cell[str(th)]}", flush=True)
        report["mde"][f"{beh}_L{layer}"] = cell

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
