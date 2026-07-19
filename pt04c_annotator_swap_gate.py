#!/usr/bin/env python3
"""pt04c: annotator-swap gate (HANDOFF_ANNOTATOR_SWAP.md — the last declared gate
in thesis ch:safety).

Re-computes the STAR1 translation direction with NOVA-PRO span annotations
(activations extracted 2026-07-12 on nova spans for BOTH base and STAR1, layers
12/16, tokenizer-alias 1.5b) and asks whether the Sonnet-span directional result
reproduces under a second annotator's span boundaries.

Per span-set (nova / sonnet) and layer (12, 16), displacement rows pooled across
the four behaviours (the pt03/pt04 statistic):
  direction  = normalised mean displacement (STAR1 − base, row-paired)
  coherence  = ||mean d|| / mean ||d||   (src.safety_posttrain.nulls def)
  null       = CHAIN-level sign-flip (all of a chain's rows share a flip; the
               ~0.05 floor from pt04b, stricter than the row-level ~0.01)

GATE (per layer): cos(direction_nova, direction_sonnet) > 0.8
                  AND nova coherence clears its chain-level null (perm p < .05).

Run:  python3 pt04c_annotator_swap_gate.py
Out:  results/safety_posttrain/pt04c_annotator_swap.json (+ stdout summary)
"""
import json
import numpy as np
from pathlib import Path

RNG_SEED = 42
N_PERM = 500
LAYERS = [12, 16]
BEHAVIOURS = ["backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"]
ACT = Path("data/activations")
SPANSETS = {
    "nova": ("R1-1.5B-novaspans", "STAR1-1.5B-novaspans"),
    "sonnet": ("R1-1.5B", "STAR1-1.5B"),
}
OUT = Path("results/safety_posttrain/pt04c_annotator_swap.json")


def load_pooled(base_dir: str, post_dir: str, layer: int):
    """Row-paired displacement rows pooled across behaviours + chain ids."""
    D, chains = [], []
    for b in BEHAVIOURS:
        X = np.load(ACT / base_dir / f"{b}_layer{layer}.npy").astype(np.float32)
        Y = np.load(ACT / post_dir / f"{b}_layer{layer}.npy").astype(np.float32)
        assert X.shape == Y.shape, f"parity failure {b} L{layer}: {X.shape} vs {Y.shape}"
        rib = json.load(open(ACT / base_dir / "row_index.json"))["rows"][b]
        rip = json.load(open(ACT / post_dir / "row_index.json"))["rows"][b]
        cb = [r["chain_id"] for r in rib]
        cp = [r["chain_id"] for r in rip]
        assert cb == cp, f"row_index chain mismatch {b} L{layer}"
        assert len(cb) == X.shape[0], f"row_index/npy length mismatch {b} L{layer}"
        D.append(Y - X)
        chains += cb
    return np.concatenate(D).astype(np.float64), np.asarray(chains)


def coherence_chain_null(D: np.ndarray, chains: np.ndarray, seed: int):
    """Coherence (||mean d|| / mean ||d||) + CHAIN-level sign-flip null."""
    norms = np.linalg.norm(D, axis=1)
    mean_norm = float(norms.mean())
    coh = float(np.linalg.norm(D.mean(axis=0)) / mean_norm)
    uniq = np.unique(chains)
    rng = np.random.default_rng(seed)
    null = np.empty(N_PERM)
    for j in range(N_PERM):
        flips = dict(zip(uniq, rng.choice((-1.0, 1.0), size=uniq.size)))
        s = np.fromiter((flips[c] for c in chains), dtype=np.float64, count=len(chains))
        null[j] = np.linalg.norm((D * s[:, None]).mean(axis=0)) / mean_norm
    p = (1 + int((null >= coh).sum())) / (1 + N_PERM)
    return coh, float(null.mean()), p


def main() -> None:
    report: dict = {"n_perm": N_PERM, "seed": RNG_SEED, "layers": {}}
    for L in LAYERS:
        row: dict = {}
        dirs = {}
        for name, (bd, pd) in SPANSETS.items():
            D, chains = load_pooled(bd, pd, L)
            mu = D.mean(axis=0)
            dirs[name] = mu / np.linalg.norm(mu)
            coh, null_mean, p = coherence_chain_null(D, chains, RNG_SEED + L)
            base_scale = float(np.mean([np.linalg.norm(
                np.load(ACT / bd / f"{b}_layer{L}.npy").astype(np.float64), axis=1).mean()
                for b in BEHAVIOURS]))
            row[name] = {
                "n_rows": int(D.shape[0]), "n_chains": int(np.unique(chains).size),
                "disp_over_base_norm": round(float(np.linalg.norm(D, axis=1).mean())
                                             / base_scale, 5),
                "coherence": round(coh, 4),
                "coherence_chain_null_mean": round(null_mean, 4),
                "coherence_p_perm": round(p, 5),
            }
        gate_cos = float(dirs["nova"] @ dirs["sonnet"])
        nova_ok = row["nova"]["coherence_p_perm"] < 0.05
        row["cos_nova_vs_sonnet_direction"] = round(gate_cos, 4)
        row["gate"] = {"cos_bar": 0.8, "pass": bool(gate_cos > 0.8 and nova_ok)}
        report["layers"][str(L)] = row
        print(f"L{L}: cos(nova,sonnet)={gate_cos:+.4f} | nova coh "
              f"{row['nova']['coherence']} (null {row['nova']['coherence_chain_null_mean']}, "
              f"p={row['nova']['coherence_p_perm']}) | sonnet coh "
              f"{row['sonnet']['coherence']} | GATE "
              f"{'PASS' if row['gate']['pass'] else 'FAIL'}")
    report["verdict"] = ("PASS — recipe direction reproduces under Nova-Pro spans"
                         if all(r["gate"]["pass"] for r in report["layers"].values())
                         else "FAIL / MIXED — see per-layer rows")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print("verdict:", report["verdict"], "→", OUT)


if __name__ == "__main__":
    main()
