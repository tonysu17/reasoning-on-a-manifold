#!/usr/bin/env python3
"""pt12: the entropy-frame verdict (POSTTRAIN_RL_PRIMER.md §2 predictions).

Per arm x layer, over the frozen 993-chain corpus (row-paired with base):
  - KL(post||base) + dH            (dose + entropy spend; from pt08 JSONs)
  - translation: |mean displacement| / base norm, cos(direction, STAR1 direction)
  - contraction: dPR (participation ratio post-base), d top-5 variance share,
    total-variance ratio post/base  (behaviours pooled, centered)

Decides P-RL1 (SFT translates, RL contracts, at matched KL) and flags the frame
falsifier (RL translation-dominant like SFT).

Run:  python3 pt12_entropy_frame_verdict.py
Out:  results/safety_posttrain/pt12_verdict.json
"""
import json
from pathlib import Path

import numpy as np

BEH = ["backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"]
ACT = Path("data/activations")
RES = Path("results/safety_posttrain")
ARMS = {  # name -> (activation dir, pt08 json)
    "sft_full_safety(star1)": ("STAR1-1.5B", RES / "consol/pt08_star1.json"),
    "sft_fullft_safety": ("R1-1.5B-fullft-safety-s42", RES / "consol/pt08_fullft_safety.json"),
    "sft_fullft_control": ("R1-1.5B-fullft-control-s42", RES / "consol/pt08_fullft_control.json"),
    "sft_lora_safety": ("R1-1.5B-lora-safety1000", RES / "consol/pt08_lora_safety.json"),
    "sft_lora_control": ("R1-1.5B-lora-control1000", RES / "consol/pt08_lora_control.json"),
    "dpo_safety": ("R1-1.5B-dpo-safety", RES / "rl/pt08_R1-1.5B-dpo-safety.json"),
    "dpo_control": ("R1-1.5B-dpo-control", RES / "rl/pt08_R1-1.5B-dpo-control.json"),
    "grpo_refusal": ("R1-1.5B-grpo-refusal", RES / "rl/pt08_R1-1.5B-grpo-refusal.json"),
    "grpo_math": ("R1-1.5B-grpo-math", RES / "rl/pt08_R1-1.5B-grpo-math.json"),
}


def pooled(arm_dir: str, layer: int) -> np.ndarray:
    return np.concatenate([
        np.load(ACT / arm_dir / f"{b}_layer{layer}.npy").astype(np.float64) for b in BEH])


def spectrum_stats(X: np.ndarray) -> tuple:
    Xc = X - X.mean(axis=0, keepdims=True)
    lam = np.linalg.eigvalsh((Xc.T @ Xc) / (len(Xc) - 1))
    lam = np.clip(lam, 0, None)
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    top5 = float(np.sort(lam)[-5:].sum() / lam.sum())
    return pr, top5, float(lam.sum())


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


report = {"layers": {}}
for L in (12, 16):
    base = pooled("R1-1.5B", L)
    base_pr, base_t5, base_var = spectrum_stats(base)
    base_norm = float(np.linalg.norm(base, axis=1).mean())
    star1_dir = (pooled("STAR1-1.5B", L) - base).mean(axis=0)
    rows = {}
    for name, (adir, p8) in ARMS.items():
        post = pooled(adir, L)
        D = post - base
        o = json.load(open(p8))["entropy_battery"]["overall"]
        pr, t5, var = spectrum_stats(post)
        rows[name] = {
            "kl": round(o["mean_kl"], 5), "dH": round(o["mean_dH"], 5),
            "disp_pct": round(100 * np.linalg.norm(D.mean(axis=0)) / base_norm, 2),
            "cos_star1": round(cos(D.mean(axis=0), star1_dir), 3),
            "dPR": round(pr - base_pr, 2), "d_top5_share": round(t5 - base_t5, 4),
            "var_ratio": round(var / base_var, 4),
        }
        del post, D
    report["layers"][str(L)] = {"base_PR": round(base_pr, 2), "base_top5_share": round(base_t5, 4), "arms": rows}
    print(f"\n== L{L} (base PR {base_pr:.1f}, top5 share {base_t5:.3f})")
    hdr = f"{'arm':24s} {'KL':>8s} {'dH':>9s} {'disp%':>6s} {'cos*':>6s} {'dPR':>7s} {'dTop5':>8s} {'varR':>7s}"
    print(hdr)
    for n, r in rows.items():
        print(f"{n:24s} {r['kl']:8.5f} {r['dH']:+9.5f} {r['disp_pct']:6.2f} {r['cos_star1']:6.3f} "
              f"{r['dPR']:+7.2f} {r['d_top5_share']:+8.4f} {r['var_ratio']:7.4f}")

json.dump(report, open(RES / "pt12_verdict.json", "w"), indent=2)
print(f"\nwritten {RES / 'pt12_verdict.json'}")
