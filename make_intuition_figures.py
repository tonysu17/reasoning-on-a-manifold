#!/usr/bin/env python3
"""Illustrative (synthetic) geometric-intuition figures for the thesis:
PCA directions, flat-vs-curved manifold (local-vs-global dimension), and
manifold steering as projection. Not data; for intuition. -> thesis/figures/."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 150,
                     "savefig.bbox": "tight"})
from pathlib import Path
OUT = Path("thesis/figures"); OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

# ---------------------------------------------------- PCA intuition
fig, ax = plt.subplots(figsize=(3.4, 3.2))
A = np.array([[2.2, 0.9], [0.9, 0.7]])
X = (rng.standard_normal((400, 2)) @ A) + np.array([0, 0])
ax.scatter(X[:, 0], X[:, 1], s=6, alpha=0.35, color="#0072B2", edgecolors="none")
C = np.cov(X.T); w, V = np.linalg.eigh(C); order = np.argsort(w)[::-1]; w, V = w[order], V[:, order]
mu = X.mean(0)
for i, (lab, col) in enumerate([("PC$_1$ (most variance)", "#D55E00"), ("PC$_2$", "#009E73")]):
    d = V[:, i] * np.sqrt(w[i]) * 2.3
    ax.add_patch(FancyArrowPatch(mu, mu + d, arrowstyle="-|>", mutation_scale=14, lw=2, color=col))
    ax.text(*(mu + d * 1.12), lab, color=col, fontsize=8, ha="center")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("PCA: orthogonal axes of\nmaximal variance", fontsize=9)
for s in ax.spines.values(): s.set_visible(False)
fig.savefig(OUT / "fig_pca_intuition.pdf"); plt.close(fig); print("pca_intuition")

# ---------------------------------------------------- flat vs curved (local-vs-global dim)
fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.0))
# flat: points near a straight line (1D structure in 2D)
t = np.linspace(-1, 1, 200)
flat = np.c_[t * 2.4, t * 0.9] + rng.standard_normal((200, 2)) * 0.05
axes[0].scatter(flat[:, 0], flat[:, 1], s=7, alpha=0.4, color="#0072B2", edgecolors="none")
axes[0].add_patch(plt.Circle((0, 0), 0.55, fill=False, ls="--", color="#D55E00", lw=1.3))
axes[0].annotate("local view\n= global shape\n(a line)", (0, 0), (1.0, -1.4), fontsize=7.5,
                 color="#D55E00", ha="center", arrowprops=dict(arrowstyle="->", color="#D55E00"))
axes[0].set_title("Flat subspace\nlocal dim $=$ global dim", fontsize=9)
# curved: points on a strong arc (locally 1D tangent, globally spans 2D)
th = np.linspace(-1.15, 1.15, 220)
curved = np.c_[np.sin(th) * 2.2, (np.cos(th) - 0.4) * 2.2] + rng.standard_normal((220, 2)) * 0.05
axes[1].scatter(curved[:, 0], curved[:, 1], s=7, alpha=0.4, color="#009E73", edgecolors="none")
axes[1].add_patch(plt.Circle((np.sin(0.0) * 2.2, (np.cos(0.0) - 0.4) * 2.2), 0.55, fill=False, ls="--", color="#D55E00", lw=1.3))
axes[1].annotate("local view\n= a line\n(low dim)", (0, 1.3), (1.7, 1.5), fontsize=7.5,
                 color="#D55E00", ha="center", arrowprops=dict(arrowstyle="->", color="#D55E00"))
axes[1].text(0, -1.9, "global shape\nspans the plane", fontsize=7.5, ha="center", color="0.3")
axes[1].set_title("Curved manifold\nlocal dim $<$ global dim", fontsize=9)
for ax in axes:
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-3, 3); ax.set_ylim(-2.4, 2.2)
    for s in ax.spines.values(): s.set_visible(False)
fig.savefig(OUT / "fig_flat_vs_curved.pdf"); plt.close(fig); print("flat_vs_curved")

# ---------------------------------------------------- manifold steering as projection
fig, ax = plt.subplots(figsize=(3.8, 3.3))
ang = np.deg2rad(28)
ax.add_patch(Ellipse((0, 0), 4.2, 1.5, angle=np.rad2deg(ang), color="#0072B2", alpha=0.16))
u = np.array([np.cos(ang), np.sin(ang)])              # principal direction of the subspace
v = np.array([1.6, 1.9])                               # raw diff-of-means steering vector
vproj = (v @ u) * u                                    # projection onto the top PC
ax.add_patch(FancyArrowPatch((0, 0), v, arrowstyle="-|>", mutation_scale=15, lw=2, color="#D55E00"))
ax.add_patch(FancyArrowPatch((0, 0), vproj, arrowstyle="-|>", mutation_scale=15, lw=2, color="#009E73"))
ax.plot([v[0], vproj[0]], [v[1], vproj[1]], ls=":", color="0.5", lw=1)
ax.text(*(v * 1.04), r"$\mathbf{v}_b$ (diff-of-means)", color="#D55E00", fontsize=8)
ax.text(vproj[0] + 0.1, vproj[1] - 0.35, r"$\mathbf{P}_k\,\mathbf{v}_b$ (manifold steering)", color="#009E73", fontsize=8)
ax.text(2.1, -0.95, "behaviour subspace\n(top-$k$ PCs)", color="#0072B2", fontsize=7.5, ha="center")
ax.set_aspect("equal"); ax.set_xlim(-1.0, 3.2); ax.set_ylim(-1.6, 2.6); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Manifold steering: project the raw\ndirection onto the behaviour subspace", fontsize=8.7)
for s in ax.spines.values(): s.set_visible(False)
fig.savefig(OUT / "fig_manifold_steering.pdf"); plt.close(fig); print("manifold_steering")
print("intuition figures written to", OUT)
