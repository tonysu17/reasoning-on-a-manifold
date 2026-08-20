#!/usr/bin/env python3
"""P0.4 — sealed sign predictions for PT-B1's behavioural contrasts.

Committed BEFORE any PT-B1 annotation, analysis, or report file is read.
Inputs are exclusively RQ3-era artifacts: stored adapter/seed activations
(layers 12/16) and the base activations that define behaviour axes.

Heuristic (declared): if the environment-matched, within-seed safety-minus-
control mean-activation difference has positive projection on behaviour b's
base-frame axis (mean of b rows minus mean of other-3 rows, unit norm), a
linear-readout heuristic predicts PT-B1's D_b = safety-minus-control
prevalence > 0; negative projection predicts D_b < 0. A prediction is issued
only when all three seeds agree in sign at BOTH layers; otherwise "no
prediction". This memo is auxiliary and cannot modify PT-B1's sealed
endpoints, contracts, or wording ceilings.
"""
import json, hashlib, subprocess
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ACT = ROOT / "data/activations"
BEH = ["backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"]
LAYERS = [12, 16]
SEEDS = {"42": "", "43": "-s43", "44": "-s44"}

def mean_vec(ckpt_dir, b, l):
    a = np.load(ckpt_dir / f"{b}_layer{l}.npy", mmap_mode="r")
    return np.asarray(a, dtype=np.float64).mean(axis=0), a.shape[0]

def pooled_mean(ckpt_dir, l):
    tot, n = None, 0
    for b in BEH:
        m, k = mean_vec(ckpt_dir, b, l)
        tot = m * k if tot is None else tot + m * k
        n += k
    return tot / n

out = {"per_behaviour": {}, "provenance": {}}
base = ACT / "R1-1.5B"
axes = {}
for l in LAYERS:
    means = {b: mean_vec(base, b, l) for b in BEH}
    for b in BEH:
        on = means[b][0]
        off_tot, off_n = None, 0
        for c in BEH:
            if c == b:
                continue
            m, k = means[c]
            off_tot = m * k if off_tot is None else off_tot + m * k
            off_n += k
        ax = on - off_tot / off_n
        axes[(b, l)] = ax / np.linalg.norm(ax)

diffs = {}
for s, suf in SEEDS.items():
    for l in LAYERS:
        ms = pooled_mean(ACT / f"R1-1.5B-lora-safety1000{suf}", l)
        mc = pooled_mean(ACT / f"R1-1.5B-lora-control1000{suf}", l)
        diffs[(s, l)] = ms - mc

for b in BEH:
    row = {}
    signs = []
    for l in LAYERS:
        per_seed = {}
        for s in SEEDS:
            d = diffs[(s, l)]
            proj = float(d @ axes[(b, l)])
            per_seed[s] = {"projection": proj, "cosine": float(proj / np.linalg.norm(d))}
            signs.append(np.sign(proj))
        row[f"L{l}"] = per_seed
    unanimous = len(set(signs)) == 1
    row["predicted_sign_of_D_b"] = ("positive" if signs[0] > 0 else "negative") if unanimous else "no prediction (seeds/layers disagree)"
    out["per_behaviour"][b] = row

out["provenance"] = {
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    "inputs": "stored seed activations L12/16 (six adapter arms) + base R1-1.5B behaviour files",
    "no_ptb1_output_read": True,
}
Path("results/prereg").mkdir(exist_ok=True, parents=True)
json.dump(out, open("results/prereg/ptb1_predicted_signs.json", "w"), indent=1)
for b in BEH:
    r = out["per_behaviour"][b]
    print(f"{b:24s} -> {r['predicted_sign_of_D_b']}")
    for l in LAYERS:
        c = {s: round(v['cosine'], 3) for s, v in r[f'L{l}'].items()}
        print(f"   L{l} cosines per seed: {c}")
