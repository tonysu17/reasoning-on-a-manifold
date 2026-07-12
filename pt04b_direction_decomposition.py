#!/usr/bin/env python3
"""pt04b: direction-decomposition and chain-level coherence null (review I4 + I5).

I4 — shared-component check: part of every arm's displacement-direction cosine to the
full-SFT direction could ride on a component every fine-tune shares — the base
mean-activation (DC) direction. Report cos(arm_dir, DC), and the arm<->full-SFT cosines
recomputed after projecting DC out of both, plus the DC-predicted cosine
cos(a,DC)*cos(t,DC). The recipe-specificity claim survives if the safety-vs-control gap
survives DC removal.

I5 — the original coherence null sign-flipped individual SPANS; chains are the
independence unit. Recompute the sign-flip null flipping whole CHAINS (300 draws):
null mean/max rises ~5x but the observed 0.70-0.78 coherence must still clear it.

Run:  python3 pt04b_direction_decomposition.py
Out:  results/safety_posttrain/pt04b_decomposition.json
"""
import json
import numpy as np
from pathlib import Path

RNG_SEED = 42
N_FLIP = 300
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
OUT = Path("results/safety_posttrain/pt04b_decomposition.json")


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def main():
    rng = np.random.default_rng(RNG_SEED)
    ri = json.load(open(ACT / BASE / "row_index.json"))
    chain_ids = {b: np.array([r["chain_id"] for r in ri["rows"][b]]) for b in BEHAVIOURS}

    report = {"seed": RNG_SEED, "n_flip_draws": N_FLIP, "layers": {}}
    for layer in LAYERS:
        base = {b: np.load(ACT / BASE / f"{b}_layer{layer}.npy").astype(np.float64) for b in BEHAVIOURS}
        dc = np.concatenate([base[b] for b in BEHAVIOURS]).mean(axis=0)
        dc_u = dc / np.linalg.norm(dc)

        dirs, disps = {}, {}
        for arm, dirname in ARMS.items():
            D = np.concatenate([
                np.load(ACT / dirname / f"{b}_layer{layer}.npy").astype(np.float64) - base[b]
                for b in BEHAVIOURS])
            disps[arm] = D
            dirs[arm] = D.mean(axis=0)

        # I4: DC decomposition
        t = dirs["star1_full"]
        t_perp = t - (t @ dc_u) * dc_u
        i4 = {"cos_to_dc": {a: round(cos(d, dc), 4) for a, d in dirs.items()}}
        i4["cos_to_full_raw"] = {a: round(cos(dirs[a], t), 4) for a in ARMS if a != "star1_full"}
        i4["cos_to_full_dc_removed"] = {
            a: round(cos(dirs[a] - (dirs[a] @ dc_u) * dc_u, t_perp), 4) for a in ARMS if a != "star1_full"}
        i4["dc_predicted_cos"] = {
            a: round(cos(dirs[a], dc) * cos(t, dc), 4) for a in ARMS if a != "star1_full"}

        # I5: chain-level sign-flip coherence null (per behaviour, star1 arm)
        i5 = {}
        for b in BEHAVIOURS:
            D = np.load(ACT / ARMS["star1_full"] / f"{b}_layer{layer}.npy").astype(np.float64) - base[b]
            norms = np.linalg.norm(D, axis=1)
            obs = np.linalg.norm(D.mean(axis=0)) / norms.mean()
            uniq = np.unique(chain_ids[b])
            row_chain = np.searchsorted(uniq, chain_ids[b])
            null = np.empty(N_FLIP)
            for j in range(N_FLIP):
                signs = rng.choice([-1.0, 1.0], size=len(uniq))[row_chain]
                null[j] = np.linalg.norm((D * signs[:, None]).mean(axis=0)) / norms.mean()
            i5[b] = {"observed_coherence": round(float(obs), 4),
                     "chain_null_mean": round(float(null.mean()), 4),
                     "chain_null_max": round(float(null.max()), 4),
                     "p": float((null >= obs).sum() + 1) / (N_FLIP + 1)}

        report["layers"][str(layer)] = {"i4_dc_decomposition": i4, "i5_chain_signflip": i5}
        print(f"== L{layer}")
        print(json.dumps(report["layers"][str(layer)], indent=1))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
