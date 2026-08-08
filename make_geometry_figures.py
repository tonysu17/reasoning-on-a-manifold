#!/usr/bin/env python3
"""Thesis figures for the per-behaviour geometry chapter (ch07).
Reads the regenerated Gate-0.1 / Tier-0 outputs and writes clean PDFs to
thesis/figures/. Read-only on results/; safe to re-run."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "figure.dpi": 150, "savefig.bbox": "tight",
})
R = Path("results"); OUT = Path("thesis/figures"); OUT.mkdir(parents=True, exist_ok=True)
BEH = ["backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"]
SHORT = {"backtracking": "back-\ntracking", "uncertainty-estimation": "uncertainty\nestimation",
         "example-testing": "example\ntesting", "adding-knowledge": "adding\nknowledge"}
# colour-blind-safe (Wong)
C = {"backtracking": "#0072B2", "uncertainty-estimation": "#E69F00",
     "example-testing": "#009E73", "adding-knowledge": "#CC79A7"}

rob = json.load(open(R / "robustness/R1-1.5B/geometry_robustness.json"))
prof = json.load(open(R / "pca/R1-1.5B/layer_profiles.json"))

# ---------------------------------------------------------------- Fig 1: compression gap
fig, ax = plt.subplots(figsize=(5.4, 3.1))
x = np.arange(len(BEH)); w = 0.38
cdim = [rob[b]["keystone_cdim"]["full"] for b in BEH]
pr = [rob[b]["pr_mp"]["pr"] for b in BEH]
ax.bar(x - w/2, cdim, w, label="intrinsic dimension\n(correlation dim.)", color="#0072B2")
ax.bar(x + w/2, pr, w, label="participation ratio\n(linear)", color="#56B4E9")
ax.axhline(1536, ls="--", lw=0.8, color="0.4")
ax.text(len(BEH)-0.5, 1536*1.12, "ambient $d=1536$", ha="right", va="bottom", fontsize=7.5, color="0.4")
ax.set_yscale("log"); ax.set_ylim(3, 4000)
ax.set_xticks(x); ax.set_xticklabels([SHORT[b] for b in BEH], fontsize=8)
ax.set_ylabel("dimension (log scale)")
for xi, c in zip(x - w/2, cdim): ax.text(xi, c*1.08, f"{c:.1f}", ha="center", va="bottom", fontsize=7)
for xi, p in zip(x + w/2, pr): ax.text(xi, p*1.08, f"{p:.0f}", ha="center", va="bottom", fontsize=7)
ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=1)
ax.set_title("Each behaviour: intrinsic dimension $\\ll$ linear extent $\\ll$ ambient", fontsize=8.5)
fig.savefig(OUT / "fig_compression_gap.pdf"); plt.close(fig)
print("wrote fig_compression_gap.pdf")

# ---------------------------------------------------------------- Fig 2: earlier curvature-ratio chain sensitivity
fig, ax = plt.subplots(figsize=(5.4, 3.1))
w = 0.26
full = [rob[b]["keystone_local_global"]["full"] for b in BEH]
rsub = [rob[b]["keystone_local_global"]["random_sub"]["mean"] for b in BEH]
strat = [rob[b]["keystone_local_global"]["chain_strat"]["mean"] for b in BEH]
ax.bar(x - w, full, w, label="full data", color="#D55E00")
ax.bar(x, rsub, w, label="random subsample (matched $N$)", color="#E69F00")
ax.bar(x + w, strat, w, label="one sentence / chain (matched $N$)", color="#009E73")
ax.axhline(1.0, ls="--", lw=0.9, color="0.3")
ax.text(len(BEH)-0.5, 1.02, "flat baseline (ratio $=1$)", ha="right", va="bottom", fontsize=7.5, color="0.3")
ax.set_ylim(0, 1.15)
ax.set_xticks(x); ax.set_xticklabels([SHORT[b] for b in BEH], fontsize=8)
ax.set_ylabel("local-to-global dimension ratio")
ax.legend(fontsize=7.5, frameon=False, loc="lower right")
ax.set_title("Earlier ratio diagnostic is chain-sensitive", fontsize=8.5)
fig.savefig(OUT / "fig_curvature_artefact.pdf"); plt.close(fig)
print("wrote fig_curvature_artefact.pdf")

# ---------------------------------------------------------------- Fig 3: layer profile (PR + probe)
probe = json.load(open(R / "cross_layer/R1-1.5B/probe_accuracy.json"))
def probe_series(b):
    v = probe.get(b, probe.get(b.replace("-", "_")))
    if isinstance(v, dict):
        ks = sorted(v.keys(), key=lambda k: int(k))
        return [int(k) for k in ks], [v[k] if not isinstance(v[k], dict) else v[k].get("accuracy", v[k].get("acc")) for k in ks]
    return list(range(len(v))), v
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharex=True)
for b in BEH:
    L = prof[b]["layers"]; pr_curve = prof[b]["participation_ratio"]
    axes[0].plot(L, pr_curve, color=C[b], lw=1.4, label=b)
    tl = int(np.nanargmin(pr_curve)); axes[0].plot(L[tl], pr_curve[tl], "o", color=C[b], ms=4)
    lp, pa = probe_series(b)
    axes[1].plot(lp, pa, color=C[b], lw=1.4)
axes[0].set_ylabel("participation ratio"); axes[0].set_xlabel("layer")
axes[0].set_title("Geometry: compression peaks mid-network\n(● = trough)", fontsize=8)
axes[1].set_ylabel("probe accuracy (chain-grouped CV)"); axes[1].set_xlabel("layer")
axes[1].axhline(0.5, ls=":", lw=0.8, color="0.5"); axes[1].set_ylim(0.45, 0.9)
axes[1].text(1, 0.51, "chance", fontsize=7, color="0.5")
axes[1].set_title("Decodability: flat across depth", fontsize=8)
axes[0].legend(fontsize=6.5, frameon=False, loc="upper right")
fig.savefig(OUT / "fig_layer_profile.pdf"); plt.close(fig)
print("wrote fig_layer_profile.pdf")

# ---------------------------------------------------------------- Fig 4: clustering silhouette (no subtypes)
fig, ax = plt.subplots(figsize=(5.0, 2.9))
for b in BEH:
    d = json.load(open(next((R / "clustering/R1-1.5B").glob(f"{b}_layer*/summary.json"))))
    sk = d["silhouette_per_k"]; ks = sorted(int(k) for k in sk)
    ax.plot(ks, [sk[str(k)] for k in ks], "o-", color=C[b], lw=1.3, ms=3.5, label=b)
ax.axhline(0.25, ls="--", lw=0.8, color="0.4")
ax.text(8, 0.255, "weak-structure threshold", ha="right", fontsize=7, color="0.4")
ax.set_xlabel("number of clusters $k$"); ax.set_ylabel("silhouette score")
ax.set_ylim(0.08, 0.35); ax.set_title("No discrete sub-types: silhouette peaks low, at $k=2$", fontsize=8.5)
ax.legend(fontsize=6.8, frameon=False, loc="upper right")
fig.savefig(OUT / "fig_clustering_silhouette.pdf"); plt.close(fig)
print("wrote fig_clustering_silhouette.pdf")
print("ALL FIGURES WRITTEN to", OUT)
PY = None
