#!/usr/bin/env python3
"""Supervisor-meeting figures (2026-06-18).
Reads regenerated Gate-0.1 / Tier-0 + predictive-geometry pilot outputs and
writes PNGs to results/supervisor_meeting/. Read-only on results/; safe to re-run.

Sources:
  - results/robustness/R1-1.5B/geometry_robustness.json   (Tier-0 scorecard)
  - results/predict/R1-1.5B/corrected/compare_rung2.md     (pilot AUCs; values below)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "figure.dpi": 150, "savefig.bbox": "tight",
})
R = Path("results"); OUT = R / "supervisor_meeting"; OUT.mkdir(parents=True, exist_ok=True)
BEH = ["backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"]
SHORT = {"backtracking": "back-\ntracking", "uncertainty-estimation": "uncertainty\nestimation",
         "example-testing": "example\ntesting", "adding-knowledge": "adding\nknowledge"}
# Wong colour-blind-safe palette (matches thesis figures)
BLUE, SKY, ORANGE, GREEN, RED, PURPLE, YELLOW = (
    "#0072B2", "#56B4E9", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#F0E442")

# ============================================================ FIG 1: predictive geometry pilot
# Values transcribed from results/predict/R1-1.5B/corrected/compare_rung2.md (183 labelled chains).
layers = [14, 17, 27]
ridge = [0.542, 0.486, 0.578]
jepa  = [0.582, 0.541, 0.591]
base  = [0.549, 0.506, 0.543]   # best baseline (persistence @14/17, curvature @27)
ridge_p = [0.128, 0.421, 0.034]
jepa_p  = [0.026, 0.146, 0.024]

fig, ax = plt.subplots(figsize=(6.0, 3.4))
x = np.arange(len(layers)); w = 0.26
b1 = ax.bar(x - w, ridge, w, label="Rung-1  ridge predictor", color=BLUE)
b2 = ax.bar(x,     jepa,  w, label="Rung-2  JEPA predictor",  color=ORANGE)
b3 = ax.bar(x + w, base,  w, label="best baseline (curv./persist.)", color="0.75")
ax.axhline(0.5, ls="--", lw=0.9, color="0.3")
ax.text(len(layers)-0.55, 0.503, "chance (AUC = 0.5)", ha="right", va="bottom", fontsize=7.5, color="0.3")
# significance stars (label-permutation p < 0.05)
for xi, v, p in zip(x - w, ridge, ridge_p):
    if p < 0.05: ax.text(xi, v + 0.006, "*", ha="center", va="bottom", fontsize=13, color=BLUE)
for xi, v, p in zip(x, jepa, jepa_p):
    if p < 0.05: ax.text(xi, v + 0.006, "*", ha="center", va="bottom", fontsize=13, color=ORANGE)
ax.set_ylim(0.42, 0.64)
ax.set_xticks(x); ax.set_xticklabels([f"layer {l}" for l in layers])
ax.set_ylabel("correctness AUC (chain-grouped OOF)")
ax.legend(fontsize=7.6, frameon=False, loc="upper center", ncol=1)
ax.set_title("Predictive-geometry pilot: predictor residual → answer correctness\n"
             "* = beats label-permutation null at $p<0.05$;  modest signal, concentrated at L27",
             fontsize=8.6)
fig.savefig(OUT / "meeting_fig1_predictive_geometry.png"); plt.close(fig)
print("wrote meeting_fig1_predictive_geometry.png")

# ============================================================ FIG 2: cleaned geometry scorecard
rob = json.load(open(R / "robustness/R1-1.5B/geometry_robustness.json"))
fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.5))
xb = np.arange(len(BEH))

# Panel A — compression gap (intrinsic dim vs participation ratio vs ambient)
wA = 0.38
cdim = [rob[b]["keystone_cdim"]["full"] for b in BEH]
pr   = [rob[b]["pr_mp"]["pr"] for b in BEH]
axA.bar(xb - wA/2, cdim, wA, label="intrinsic dim (correlation dim.)", color=BLUE)
axA.bar(xb + wA/2, pr,   wA, label="participation ratio (linear)", color=SKY)
axA.axhline(1536, ls="--", lw=0.8, color="0.4")
axA.text(len(BEH)-0.5, 1536*1.15, "ambient $d=1536$", ha="right", va="bottom", fontsize=7.5, color="0.4")
axA.set_yscale("log"); axA.set_ylim(3, 4000)
axA.set_xticks(xb); axA.set_xticklabels([SHORT[b] for b in BEH], fontsize=7.6)
axA.set_ylabel("dimension (log scale)")
for xi, c in zip(xb - wA/2, cdim): axA.text(xi, c*1.10, f"{c:.1f}", ha="center", va="bottom", fontsize=7)
for xi, p in zip(xb + wA/2, pr):   axA.text(xi, p*1.10, f"{p:.0f}", ha="center", va="bottom", fontsize=7)
axA.legend(fontsize=7.3, frameon=False, loc="upper left")
axA.set_title("A.  SURVIVES: low intrinsic dim ($\\approx$6–8 $\\ll$ 1536),\nstable under chain control", fontsize=8.4)

# Panel B — curvature collapses under the chain control
wB = 0.26
full  = [rob[b]["keystone_local_global"]["full"] for b in BEH]
rsub  = [rob[b]["keystone_local_global"]["random_sub"]["mean"] for b in BEH]
strat = [rob[b]["keystone_local_global"]["chain_strat"]["mean"] for b in BEH]
axB.bar(xb - wB, full,  wB, label="full data", color=RED)
axB.bar(xb,      rsub,  wB, label="random subsample (matched $N$)", color=ORANGE)
axB.bar(xb + wB, strat, wB, label="one sentence / chain (matched $N$)", color=GREEN)
axB.axhline(1.0, ls="--", lw=0.9, color="0.3")
axB.text(len(BEH)-0.5, 1.01, "flat (ratio $=1$)", ha="right", va="bottom", fontsize=7.5, color="0.3")
axB.set_ylim(0, 1.18)
axB.set_xticks(xb); axB.set_xticklabels([SHORT[b] for b in BEH], fontsize=7.6)
axB.set_ylabel("local-to-global dim ratio\n($<1$ curved, $\\approx1$ flat)")
axB.legend(fontsize=7.3, frameon=False, loc="lower right")
axB.set_title("B.  FALLS: apparent curvature $\\to$ flat once\nwithin-chain points are removed", fontsize=8.4)
fig.suptitle("Cleaned Gate-0 geometry (deduplicated, chain-controlled): what survives vs what was a confound",
             fontsize=9.4, y=1.02)
fig.savefig(OUT / "meeting_fig2_geometry_scorecard.png"); plt.close(fig)
print("wrote meeting_fig2_geometry_scorecard.png")

# ============================================================ FIG 3: two-week workstream timeline
# Narrative timeline (Jun 5 -> Jun 18). status: done / progress / pending
rows = [
    # label, start_day, end_day, status, note
    ("Audit & confound fixes\n(CF-13..17)",            5, 13, "done",     "duplicate-row + null bugs"),
    ("Gate-0.0 re-extraction\n(clean activations)",    13, 14, "done",    "dupes 35-56% -> ~1%"),
    ("Gate-0.1 regeneration\n(PCA/triang./Tier-0)",    14, 16, "done",    "geometry numbers citable"),
    ("3-annotator robustness\n(kappa + span-F1)",      12, 15, "done",    "k=0.35-0.44; F1 0.26-0.31"),
    ("Predictive-geometry\nextension (Rung 1-2)",      15, 16, "done",    "pilot AUC 0.58-0.59 @L27"),
    ("Thesis drafting +\nrefinement pass",             12, 17, "done",    "79pp, ch01-08; reframe"),
    ("Safety / gpt-oss\nextraction (offline)",          5, 13, "progress","components built, unrun"),
    ("Full-1000 correctness\nlabels (gate)",           18, 19, "pending", "spend decision needed"),
    ("Phase 7 steering\nspecificity (cluster)",        18, 19, "pending", "needs GPU run"),
]
cmap = {"done": GREEN, "progress": ORANGE, "pending": "0.78"}
fig, ax = plt.subplots(figsize=(9.4, 4.3))
for i, (label, s, e, st, note) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.barh(y, e - s, left=s, height=0.62, color=cmap[st],
            edgecolor="white", linewidth=0.8, zorder=3)
    ax.text(s - 0.15, y, label, ha="right", va="center", fontsize=7.6)
    tx = e + 0.15 if st != "pending" else e + 0.15
    ax.text(tx, y, note, ha="left", va="center", fontsize=6.9, color="0.35", style="italic")
ax.set_xlim(0.5, 24); ax.set_ylim(-0.7, len(rows) - 0.3)
ax.set_yticks([])
# x axis = calendar days in June
ticks = [5, 8, 11, 14, 17]
ax.set_xticks(ticks); ax.set_xticklabels([f"Jun {t}" for t in ticks])
ax.axvline(18, ls="--", lw=1.0, color=BLUE, zorder=2)
ax.text(18.1, len(rows) - 0.45, "today\n(meeting)", fontsize=7, color=BLUE, va="top")
ax.grid(axis="y", visible=False)
handles = [Patch(facecolor=cmap[k], label=v) for k, v in
           [("done", "completed"), ("progress", "built / in progress"), ("pending", "pending decision")]]
ax.legend(handles=handles, fontsize=7.6, frameon=False, loc="lower left",
          bbox_to_anchor=(0.0, 0.0), ncol=1)
ax.set_title("Last two weeks at a glance — eight workstreams (5 Jun → 18 Jun 2026)", fontsize=9.6)
fig.savefig(OUT / "meeting_fig3_timeline.png"); plt.close(fig)
print("wrote meeting_fig3_timeline.png")
print("done.")
