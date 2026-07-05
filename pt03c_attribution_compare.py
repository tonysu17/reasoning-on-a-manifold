"""Post-training spillover — Step 3c: ATTRIBUTION + DOSE-RESPONSE comparison.

Reads the per-arm gated reports (pt03b) plus the raw activations and answers the
two questions Rung-1 exists for:

1. ATTRIBUTION: is the global translation safety-specific, or does a size-matched
   NON-SAFETY fine-tune produce the same shift? Compared on (a) magnitude
   (mean displacement / base norm), (b) direction (cosine between arms' global
   mean-displacement directions, incl. vs the public STAR-1 checkpoint), and
   (c) rotation excess (should stay null everywhere if LoRA arms behave like STAR-1).
2. DOSE-RESPONSE: does the safety translation grow with dose {100, 300, 1000}?

Output: results/safety_posttrain/attribution_report.json (+ stdout summary).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

BASE = Path("data/activations/R1-1.5B")
ARMS = {
    "safety100": Path("data/activations/R1-1.5B-lora-safety100"),
    "safety300": Path("data/activations/R1-1.5B-lora-safety300"),
    "safety1000": Path("data/activations/R1-1.5B-lora-safety1000"),
    "control1000": Path("data/activations/R1-1.5B-lora-control1000"),
    "star1_full_sft": Path("data/activations/STAR1-1.5B"),
}
BEH = ["backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"]
LAYERS = [12, 16]


def global_direction(arm_dir: Path, layer: int) -> tuple[np.ndarray, dict]:
    """Span-weighted global mean displacement + per-behaviour magnitudes."""
    num = None
    n_tot = 0
    mags = {}
    for b in BEH:
        Xb = np.load(BASE / f"{b}_layer{layer}.npy").astype(np.float64)
        Xp = np.load(arm_dir / f"{b}_layer{layer}.npy").astype(np.float64)
        d = Xp - Xb
        mags[b] = round(float(np.linalg.norm(d, axis=1).mean()
                              / np.linalg.norm(Xb, axis=1).mean()), 5)
        s = d.sum(axis=0)
        num = s if num is None else num + s
        n_tot += d.shape[0]
    g = num / n_tot
    return g, mags


def main() -> None:
    report: dict = {"layers": {}, "gated_rotation_summary": {}}

    for L in LAYERS:
        dirs, mags = {}, {}
        for arm, path in ARMS.items():
            g, m = global_direction(path, L)
            dirs[arm] = g / np.linalg.norm(g)
            mags[arm] = {"per_behaviour_disp_over_base": m,
                         "global_disp_norm": round(float(np.linalg.norm(g)), 5)}
        cos = {f"{a}~{b}": round(float(dirs[a] @ dirs[b]), 4)
               for i, a in enumerate(ARMS) for b in list(ARMS)[i + 1:]}
        report["layers"][str(L)] = {"magnitudes": mags, "direction_cosines": cos}
        log.info("layer %d cosines: %s", L, {k: v for k, v in cos.items()
                                             if "control" in k or "star1" in k})

    # rotation excess summary from the gated reports
    for arm in ("safety100", "safety300", "safety1000", "control1000"):
        f = Path(f"results/safety_posttrain/spillover_gated_{arm}.json")
        if not f.exists():
            report["gated_rotation_summary"][arm] = "missing"
            continue
        g = json.loads(f.read_text())
        report["gated_rotation_summary"][arm] = {
            L: {b: {"excess_deg": e["excess_deg"], "p_within": e["p_within"],
                    "paired_coherence": e.get("paired", {}).get("coherence"),
                    "disp_over_base": e.get("paired", {}).get("mean_disp_over_base_norm")}
                for b, e in cells.items() if isinstance(e, dict) and "excess_deg" in e}
            for L, cells in g["layers"].items()}
        report.setdefault("parity", {})[arm] = g["parity"]["ok"]

    # dose-response: safety translation magnitude vs dose (span-weighted mean over behaviours)
    dose_curve = {}
    for L in LAYERS:
        m = report["layers"][str(L)]["magnitudes"]
        dose_curve[str(L)] = {
            arm: round(float(np.mean(list(m[arm]["per_behaviour_disp_over_base"].values()))), 5)
            for arm in ("safety100", "safety300", "safety1000", "control1000", "star1_full_sft")}
    report["dose_curve_disp_over_base"] = dose_curve

    out = Path("results/safety_posttrain/attribution_report.json")
    out.write_text(json.dumps(report, indent=1))
    log.info("attribution report -> %s", out)
    log.info("dose curve: %s", dose_curve)


if __name__ == "__main__":
    main()
