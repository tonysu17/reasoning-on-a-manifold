#!/usr/bin/env python3
"""Regenerate the stale headline figures (fig1, fig8, fig9) from the FRESH
corrected-run JSON. fig3/4/6/7 are unchanged and kept as-is."""
import json, glob, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import provenance, require_file

M = "R1-1.5B"
OUT = "results/supervisor_meeting"
BEHS = ["backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"]
COL = dict(zip(BEHS, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]))
SHORT = {"backtracking":"backtrack","uncertainty-estimation":"uncertainty",
         "example-testing":"example-test","adding-knowledge":"add-knowledge"}

_prof_path = f"results/pca/{M}/layer_profiles.json"
_nulls_path = f"results/pca/{M}/null_pvalues_per_layer.json"
require_file(_prof_path, "run 05_pca_analysis.py (with --with-nulls) first")
require_file(_nulls_path, "run 05_pca_analysis.py --with-nulls first")
prof = json.load(open(_prof_path))
nulls = json.load(open(_nulls_path))
os.makedirs(OUT, exist_ok=True)
json.dump(provenance(inputs=[_prof_path, _nulls_path]), open(f"{OUT}/provenance.json", "w"), indent=2)

# ---- fig1: d_eff_70 (top) + PR (bottom) across layers ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for b in BEHS:
    L = prof[b]["layers"]
    ax1.plot(L, prof[b]["d_eff_70"], "o-", ms=3, color=COL[b], label=SHORT[b])
    pr = prof[b]["participation_ratio"]
    ax2.plot(L, pr, "o-", ms=3, color=COL[b], label=SHORT[b])
    tL = L[int(np.argmin(pr))]
    ax2.scatter([tL], [min(pr)], s=90, facecolors="none", edgecolors=COL[b], linewidths=2, zorder=5)
ax1.axhline(50, ls=":", c="gray", lw=1)
ax1.text(0.3, 51, "old cap = 50 (saturated)", fontsize=8, color="gray")
ax1.set_ylabel("PCA d_eff(70%)"); ax1.set_title("Linear dimensionality — de-saturated (cap=100)")
ax1.legend(fontsize=8, ncol=4, loc="upper left"); ax1.grid(alpha=.3)
ax2.set_ylabel("participation ratio"); ax2.set_xlabel("layer")
ax2.set_title("Participation ratio (○ = PR trough = manifold-peak layer)")
ax2.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_layer_sweep.png", dpi=120); plt.close()
print("fig1 written")

# ---- fig8: 28-layer chain-stratified null significance heatmap ----
layers = sorted({int(L) for b in nulls for L in nulls[b]})
grid = np.full((len(BEHS), len(layers)), np.nan)
for i, b in enumerate(BEHS):
    for j, L in enumerate(layers):
        if str(L) in nulls[b]:
            p = nulls[b][str(L)]["p_value"]
            grid[i, j] = min(-np.log10(p + 1e-3), 3.0)
fig, ax = plt.subplots(figsize=(11, 3.2))
im = ax.imshow(grid, aspect="auto", cmap="YlGn", vmin=0, vmax=3)
ax.set_xticks(range(len(layers))); ax.set_xticklabels(layers, fontsize=7)
ax.set_yticks(range(len(BEHS))); ax.set_yticklabels([SHORT[b] for b in BEHS])
ax.set_xlabel("layer"); ax.set_title("Chain-stratified permutation null  (−log10 p, capped at 3;  green = significant)")
cb = plt.colorbar(im, ax=ax, fraction=0.025); cb.set_label("−log10 p")
# mark Bonferroni threshold cells with a dot (p < 0.05/112)
for i, b in enumerate(BEHS):
    for j, L in enumerate(layers):
        if str(L) in nulls[b] and nulls[b][str(L)]["p_value"] < 0.05/112:
            ax.text(j, i, "·", ha="center", va="center", fontsize=8, color="black")
plt.tight_layout(); plt.savefig(f"{OUT}/fig8_null_hierarchy.png", dpi=120); plt.close()
print("fig8 written")

# ---- fig9: scorecard table (fresh d_eff + 5b intrinsic/curvature at each trough layer) ----
def trough_layer(b):
    L = prof[b]["layers"]; pr = prof[b]["participation_ratio"]
    return L[int(np.argmin(pr))]
def d70_at(b, L):
    return prof[b]["d_eff_70"][prof[b]["layers"].index(L)]
def n_sig(b):
    return sum(1 for L in nulls[b] if nulls[b][L]["p_value"] < 0.01)

# 5b diagnostics: prefer fresh, else archived (deterministic point estimates)
def load_5b(layer):
    fresh = f"results/geometric/{M}/diagnostics_layer{layer}.json"
    if os.path.exists(fresh): return json.load(open(fresh))
    arch = glob.glob(f"results/_archive_run1_*/geometric/{M}/diagnostics_layer{layer}.json")
    return json.load(open(arch[0])) if arch else None
def twoNN(b, L):
    d = load_5b(L)
    if not d: return None
    for e in d["per_behaviour"][b]["intrinsic_dim"]:
        if e["estimator"] == "twoNN": return e["estimate"]
def curv(b, L, sub):
    d = load_5b(L)
    if not d: return None
    for c in d["per_behaviour"][b]["curvature"]:
        if sub in c["diagnostic"] and c.get("k", 10) == 10: return c["mean"]

REF = 17  # common reference layer for 5b diagnostics (matches prior reporting)
rows = []
for b in BEHS:
    tL = trough_layer(b)
    d70 = d70_at(b, REF); tw = twoNN(b, REF)
    geo = curv(b, REF, "geodesic"); tan = curv(b, REF, "tangent")
    comp = f"{d70/tw:.1f}x" if tw else "-"
    rows.append([SHORT[b], f"{tw:.1f}" if tw else "-", str(d70), comp,
                 f"{geo:.2f}" if geo else "-", f"{tan:.0f}°" if tan else "-",
                 f"L{tL}", f"{n_sig(b)}/28"])
cols = ["behaviour", "intrinsic\n(twoNN,L17)", "PCA d_eff70\n(L17)", "compress",
        "geo/eucl\n(L17)", "tangent\n(L17)", "PR\ntrough", "null sig\n(p<.01)"]
fig, ax = plt.subplots(figsize=(11, 2.4)); ax.axis("off")
t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.8)
for j in range(len(cols)):
    t[0, j].set_facecolor("#2c3e50"); t[0, j].get_text().set_color("white")
for i, b in enumerate(BEHS):
    t[i+1, 0].set_facecolor(COL[b]); t[i+1, 0].get_text().set_color("white")
ax.set_title("Phase-5 / 5b scorecard — manifold signature at each behaviour's PR-trough layer",
             fontsize=11, pad=10)
plt.tight_layout(); plt.savefig(f"{OUT}/fig9_scorecard.png", dpi=120); plt.close()
print("fig9 written")
print("\nScorecard rows:")
for r in rows: print("  ", r)
