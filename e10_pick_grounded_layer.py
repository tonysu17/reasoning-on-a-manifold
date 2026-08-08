#!/usr/bin/env python
"""E10.3 grounded-layer picker — the SEALED width-gate rule (prereg Amendment 4).

Reads a main-run controls.json and prints the layer at which the learned DAS
frame is GROUNDED, by the rule sealed before the pod run:

    grounded  iff  coord_auc_source_vs_base >= 0.65
              AND  state-dependence ratio (real induce / base-donor-null induce) >= 2.0
              AND  real induce > 0
    pick = argmax coord-AUC among grounded layers.

(Thresholds calibrated on the executed backtracking run: L17 passed at
AUC 0.760 / ratio 4.3x; the ungrounded L11/L27 sat at AUC 0.357/0.211.)

Exit status 3 + "NONE" on stderr when no layer grounds — the width probe is then
SKIPPED for that behaviour. Per the amendment, that skip is itself the sealed
outcome ("no grounded causal frame found"), not a failure to run.

Usage: python e10_pick_grounded_layer.py results/das/R1-1.5B/unc_main/controls.json
"""
import json
import sys

AUC_MIN = 0.65
RATIO_MIN = 2.0


def pick(controls: dict):
    best = None
    for layer, cell in controls.items():
        lc = cell["learned"]
        auc = lc["coord_auc_source_vs_base"]
        real = lc["real_donor"]["induce_dlp"]
        null = lc["base_donor_null"]["induce_dlp"]
        ratio = (real / null) if null > 1e-9 else (float("inf") if real > 0 else 0.0)
        if auc >= AUC_MIN and ratio >= RATIO_MIN and real > 0:
            if best is None or auc > best[1]:
                best = (int(layer), auc)
    return best


if __name__ == "__main__":
    result = pick(json.load(open(sys.argv[1])))
    if result is None:
        print("NONE", file=sys.stderr)
        sys.exit(3)
    print(result[0])
