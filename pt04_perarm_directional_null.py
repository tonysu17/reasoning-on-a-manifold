#!/usr/bin/env python3
"""pt04: per-arm null for the spillover DIRECTIONAL claim (the hardening declared
owed in thesis ch:safety / RESULTS_LEDGER B2).

The executed attribution analysis reported point cosines between each arm's global
mean-displacement direction and the full-SFT (STAR1) direction: safety1000 0.57 vs
control1000 0.15 (random floor ~0.03), with no per-arm uncertainty. This script adds:

  (a) PRIMARY: chain-level bootstrap (B=2000) of every arm's cosine to full-SFT,
      resampling the 986 chains with replacement, the SAME resample applied to all
      arms (the design is paired: identical chains underlie every arm). Reports 95%
      percentile CIs and the paired one-sided test Delta = cos(safety1000, full)
      - cos(control1000, full) > 0.
  (b) BIAS CHECK: all displacements share the same base rows X (D_a = Y_a - X), so
      cos(mean D_a, mean D_full) is inflated by Var(X-bar) common noise. Disjoint
      chain-half estimate: mean D_a on half A vs mean D_full on half B (independent
      X noise), averaged over R=200 random splits. An arm's alignment that survives
      disjoint halves is not a shared-noise artefact.

Statistic matches the original: displacement rows pooled across the four behaviours,
per layer (12, 16), direction = normalised mean displacement.

Run:  python3 pt04_perarm_directional_null.py
Out:  results/safety_posttrain/pt04_perarm_null.json (+ stdout summary)
"""
import json
import numpy as np
from pathlib import Path

RNG_SEED = 42
B_BOOT = 2000
R_SPLITS = 200
LAYERS = [12, 16]
BEHAVIOURS = ["backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"]
ACT = Path("data/activations")
BASE = "R1-1.5B"
ARMS = {
    "safety100": "R1-1.5B-lora-safety100",
    "safety300": "R1-1.5B-lora-safety300",
    "safety1000": "R1-1.5B-lora-safety1000",
    "control1000": "R1-1.5B-lora-control1000",
    "star1_full": "STAR1-1.5B",
}
OUT = Path("results/safety_posttrain/pt04_perarm_null.json")


def load_chain_ids(dirname: str) -> dict:
    ri = json.load(open(ACT / dirname / "row_index.json"))
    return {b: [r["chain_id"] for r in ri["rows"][b]] for b in BEHAVIOURS}


def per_chain_sums(dirname: str, layer: int, base_chain_ids: dict, chain_order: list) -> tuple:
    """Sum of (Y_arm - X_base) rows per chain, pooled across behaviours.
    Returns (S: n_chains x d, n: n_chains counts)."""
    idx = {c: i for i, c in enumerate(chain_order)}
    d = None
    S = None
    n = np.zeros(len(chain_order), dtype=np.int64)
    for b in BEHAVIOURS:
        Y = np.load(ACT / dirname / f"{b}_layer{layer}.npy").astype(np.float64)
        X = np.load(ACT / BASE / f"{b}_layer{layer}.npy").astype(np.float64)
        assert Y.shape == X.shape, f"{dirname}/{b}: shape mismatch {Y.shape} vs {X.shape}"
        D = Y - X
        if S is None:
            d = D.shape[1]
            S = np.zeros((len(chain_order), d))
        rows = np.fromiter((idx[c] for c in base_chain_ids[b]), dtype=np.int64, count=len(base_chain_ids[b]))
        np.add.at(S, rows, D)
        np.add.at(n, rows, 1)
        del Y, X, D
    return S, n


def cos(u: np.ndarray, v: np.ndarray) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def main():
    rng = np.random.default_rng(RNG_SEED)

    # row-pairing integrity: every arm's row_index must equal the base's
    base_ids = load_chain_ids(BASE)
    for arm, dirname in ARMS.items():
        ids = load_chain_ids(dirname)
        for b in BEHAVIOURS:
            assert ids[b] == base_ids[b], f"row_index mismatch: {arm}/{b}"
    chain_order = sorted({c for b in BEHAVIOURS for c in base_ids[b]})
    n_chains = len(chain_order)
    print(f"row-pairing verified across {len(ARMS)} arms; {n_chains} chains, "
          f"{sum(len(v) for v in base_ids.values())} pooled rows")

    report = {"seed": RNG_SEED, "B_boot": B_BOOT, "R_splits": R_SPLITS,
              "n_chains": n_chains, "statistic": "cos(normalised mean displacement, full-SFT dir), behaviours pooled",
              "analytic_random_floor_abscos": float(np.sqrt(2 / (np.pi * 1536))),
              "layers": {}}

    for layer in LAYERS:
        print(f"\n=== layer {layer}: loading per-chain displacement sums…")
        sums, counts = {}, None
        for arm, dirname in ARMS.items():
            S, n = per_chain_sums(dirname, layer, base_ids, chain_order)
            sums[arm] = S
            if counts is None:
                counts = n
            else:
                assert np.array_equal(counts, n), f"per-chain counts differ for {arm}"
        total = counts.sum()

        # point estimates
        dirs = {a: sums[a].sum(axis=0) / total for a in ARMS}
        point = {a: cos(dirs[a], dirs["star1_full"]) for a in ARMS if a != "star1_full"}

        # (a) paired chain bootstrap — one resample drives all arms
        boot = {a: np.empty(B_BOOT) for a in point}
        delta = np.empty(B_BOOT)
        for i in range(B_BOOT):
            w = rng.multinomial(n_chains, np.full(n_chains, 1 / n_chains)).astype(np.float64)
            wn = w @ counts
            mf = (w @ sums["star1_full"]) / wn
            for a in point:
                boot[a][i] = cos((w @ sums[a]) / wn, mf)
            delta[i] = boot["safety1000"][i] - boot["control1000"][i]
        ci = {a: [float(np.percentile(boot[a], 2.5)), float(np.percentile(boot[a], 97.5))] for a in point}
        p_delta = float((delta <= 0).mean())

        # (b) disjoint-half bias check (independent base noise between arm and full)
        half = {a: np.empty(R_SPLITS) for a in point}
        for r in range(R_SPLITS):
            perm = rng.permutation(n_chains)
            A, Bh = perm[: n_chains // 2], perm[n_chains // 2:]
            mf = sums["star1_full"][Bh].sum(axis=0) / counts[Bh].sum()
            for a in point:
                ma = sums[a][A].sum(axis=0) / counts[A].sum()
                half[a][r] = cos(ma, mf)
        halves = {a: {"mean": float(half[a].mean()), "sd": float(half[a].std())} for a in point}

        report["layers"][str(layer)] = {
            "point_cos_to_full": {a: round(v, 4) for a, v in point.items()},
            "boot_ci95": {a: [round(x, 4) for x in ci[a]] for a in point},
            "delta_safety1000_minus_control1000": {
                "point": round(point["safety1000"] - point["control1000"], 4),
                "ci95": [round(float(np.percentile(delta, 2.5)), 4), round(float(np.percentile(delta, 97.5)), 4)],
                "p_one_sided_leq0": p_delta if p_delta > 0 else f"<{1 / B_BOOT}",
            },
            "disjoint_half_cos": halves,
        }
        print(json.dumps(report["layers"][str(layer)], indent=1))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
