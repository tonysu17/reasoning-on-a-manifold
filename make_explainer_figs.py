#!/usr/bin/env python3
"""Pedagogical visuals for the supervisor presentation:
  viz1 — curvature inflates linear (PCA) dimensionality (line -> circle -> wiggly curve)
  viz2 — Swiss roll: intrinsic 2-D sheet, ambient 3-D, geodesic vs Euclidean, local tangent
  viz3 — how the intrinsic-dim estimators recover the true dimension (TwoNN + correlation dim)
All synthetic; the point is conceptual (real data is >3-D)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
import os, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import provenance

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
                     "figure.titlesize": 16, "figure.titleweight": "bold"})
OUT = "results/supervisor_meeting"
os.makedirs(OUT, exist_ok=True)
json.dump(provenance(), open(f"{OUT}/provenance.json", "w"), indent=2)
rng = np.random.default_rng(0)


def deff70(X):
    p = PCA().fit(X)
    cum = np.cumsum(p.explained_variance_ratio_)
    return int(np.searchsorted(cum, 0.70) + 1), p.explained_variance_ratio_


# ============================ VIZ 1 ============================
fig = plt.figure(figsize=(15, 8))
fig.suptitle("Curvature inflates the linear (PCA) dimensionality of a low-dimensional manifold")

# (a) straight line: 1-D, flat, in 2-D
t = np.linspace(0, 1, 400)
line = np.c_[t, np.zeros_like(t)] + rng.normal(0, 0.004, (400, 2))
# (b) circle: 1-D, curved, in 2-D
th = np.linspace(0, 2*np.pi, 400, endpoint=False)
circ = np.c_[np.cos(th), np.sin(th)]
# (c) "wiggly" 1-D curve embedded in 40-D via decaying Fourier modes
tt = np.linspace(0, 1, 600)
cols, K = [], 25
for k in range(1, K + 1):
    a = 1.0 / k**0.6
    cols += [a*np.cos(2*np.pi*k*tt), a*np.sin(2*np.pi*k*tt)]
wig = np.array(cols).T                      # (600, 50) — but still a 1-D curve in t
wig3 = PCA(n_components=3).fit_transform(wig)

manis = [("(a) straight line\nintrinsic = 1", line, line, "2-D"),
         ("(b) circle\nintrinsic = 1", circ, circ, "2-D"),
         (f"(c) wiggly curve\nintrinsic = 1", wig, wig3, "50-D")]

for i, (title, Xfull, Xshow, amb) in enumerate(manis):
    # top: the manifold
    if Xshow.shape[1] == 2:
        ax = fig.add_subplot(2, 3, i+1)
        ax.plot(Xshow[:, 0], Xshow[:, 1], ".", ms=3, color="#1f77b4")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    else:
        ax = fig.add_subplot(2, 3, i+1, projection="3d")
        ax.plot(Xshow[:, 0], Xshow[:, 1], Xshow[:, 2], ".", ms=2, color="#1f77b4")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.text2D(0.5, -0.05, "(top-3 PCA projection of a curve living in 50-D)",
                  transform=ax.transAxes, ha="center", fontsize=9, style="italic")
    ax.set_title(title)
    # bottom: PCA spectrum
    d, evr = deff70(Xfull)
    axb = fig.add_subplot(2, 3, i+4)
    n = min(12, len(evr))
    axb.bar(range(1, n+1), evr[:n]*100, color="#d62728", alpha=0.8)
    axb.axhline(0, color="k", lw=0.5)
    axb.set_xlabel("PCA component"); axb.set_ylabel("% variance")
    axb.set_title(f"PCA d_eff(70%) = {d}", fontsize=12, color="#b22222")
    axb.set_xticks(range(1, n+1, 2))

fig.text(0.5, 0.005,
         "All three are 1-D curves (one free parameter, t).  Linear PCA reports d_eff = 1 -> 2 -> "
         f"{deff70(wig)[0]} as curvature grows.  The gap between PCA dimension and the true intrinsic "
         "dimension IS the curvature.\nOur reasoning behaviours sit at the far end of this spectrum: "
         "intrinsic dim ~7-13, but PCA d_eff ~47-98.",
         ha="center", fontsize=11)
plt.tight_layout(rect=[0, 0.06, 1, 0.96])
plt.savefig(f"{OUT}/viz1_curvature_inflates_dim.png", dpi=130); plt.close()
print("viz1 written; wiggly d_eff70 =", deff70(wig)[0])


# ============================ VIZ 2 ============================
n = 2500
u = rng.uniform(0, 1, n)
t = 1.5*np.pi*(1 + 2*u)               # intrinsic coord 1 (position along roll)
h = rng.uniform(0, 21, n)             # intrinsic coord 2 (height)
X, Y, Z = t*np.cos(t), h, t*np.sin(t)

fig = plt.figure(figsize=(14, 6))
fig.suptitle("A 2-D sheet rolled through 3-D: intrinsic dim 2, but linear PCA needs 3 — and curvature warps distance")
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.scatter(X, Y, Z, c=t, cmap="viridis", s=6, alpha=0.6)
# two points one full winding apart, same height -> near in 3-D, far along the sheet
t1, t2, h0 = 2*np.pi, 4*np.pi, 10.0
P1 = np.array([t1*np.cos(t1), h0, t1*np.sin(t1)])
P2 = np.array([t2*np.cos(t2), h0, t2*np.sin(t2)])
eucl = np.linalg.norm(P1 - P2)
tp = np.linspace(t1, t2, 80)
path = np.c_[tp*np.cos(tp), np.full_like(tp, h0), tp*np.sin(tp)]
geo = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
ax.plot(*path.T, color="crimson", lw=3, label=f"geodesic (along sheet) ≈ {geo:.0f}")
ax.plot(*np.c_[P1, P2], "--", color="black", lw=2, label=f"Euclidean (chord) ≈ {eucl:.0f}")
ax.scatter(*P1, color="red", s=80); ax.scatter(*P2, color="red", s=80)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.legend(loc="upper center", fontsize=10)
ax.set_title(f"geodesic / Euclidean ≈ {geo/eucl:.1f}  (flat would be 1.0)")

ax2 = fig.add_subplot(1, 2, 2)
ax2.text(0.02, 0.98, (
    "How this maps to the three curvature diagnostics\n"
    "(measured directly on our activation manifolds):\n\n"
    "1.  Local-vs-global PCA ratio\n"
    "    A small patch of the sheet is flat -> locally 2-D.\n"
    "    Globally PCA sees 3-D.  ratio = 2/3 < 1.\n"
    "    (Ours at L17 ≈ 0.09 -> very strongly curved.)\n\n"
    "2.  Geodesic / Euclidean ratio\n"
    "    Straight-line chord cuts through empty space;\n"
    "    the true path runs ALONG the sheet and is longer.\n"
    "    ratio > 1.  (Ours ≈ 1.6-1.9.)\n\n"
    "3.  Tangent-space rotation\n"
    "    The flat tangent patch tilts as you move along the\n"
    "    roll; the angle between nearby patches grows with\n"
    "    curvature.  (Ours ≈ 57-68 degrees.)\n\n"
    "All three say the same thing three different ways:\n"
    "the manifold is genuinely curved, not a flat subspace."),
    transform=ax2.transAxes, va="top", ha="left", fontsize=11.5, family="monospace")
ax2.axis("off")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f"{OUT}/viz2_swiss_roll.png", dpi=130); plt.close()
print(f"viz2 written; eucl={eucl:.1f} geo={geo:.1f} ratio={geo/eucl:.2f}")


# ============================ VIZ 3 ============================
# Known-truth dataset: uniform 2-D disk (intrinsic dim = 2 exactly)
m = 3000
rr = np.sqrt(rng.uniform(0, 1, m)); ang = rng.uniform(0, 2*np.pi, m)
D = np.c_[rr*np.cos(ang), rr*np.sin(ang)]

# --- TwoNN ---
nn = NearestNeighbors(n_neighbors=3).fit(D)
dist, _ = nn.kneighbors(D)
mu = dist[:, 2] / (dist[:, 1] + 1e-12)
mu = mu[np.isfinite(mu) & (mu > 1)]
mu.sort()
Nmu = len(mu)
F = np.arange(1, Nmu+1) / (Nmu + 1)        # empirical CDF on the FULL set
keep = int(0.9*Nmu)                        # discard top 10% but keep original F scale
xv, yv = np.log(mu[:keep]), -np.log(1 - F[:keep])
d_twoNN = float(np.sum(xv*yv) / np.sum(xv*xv))   # slope through origin

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("How intrinsic-dimension estimators recover the true dimension (here: a 2-D disk, true dim = 2)")

ax = axes[0]
ax.scatter(D[:, 0], D[:, 1], s=4, alpha=0.4, color="#2ca02c")
p0 = D[0]
d0, idx0 = nn.kneighbors([p0])
ax.scatter(*p0, color="black", s=60, zorder=5)
for j, c in [(1, "#1f77b4"), (2, "#d62728")]:
    q = D[idx0[0, j]]
    ax.plot([p0[0], q[0]], [p0[1], q[1]], color=c, lw=2,
            label=f"r{j} = dist to NN #{j}")
ax.set_aspect("equal"); ax.legend(fontsize=10); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("TwoNN uses only the 2 nearest neighbours\nratio  mu = r2 / r1")

ax = axes[1]
ax.plot(xv, yv, ".", ms=3, color="#1f77b4", alpha=0.5)
xs = np.linspace(0, xv.max(), 50)
ax.plot(xs, d_twoNN*xs, "r-", lw=2.5)
ax.set_xlabel("log(mu)"); ax.set_ylabel("-log(1 - CDF)")
ax.set_title(f"slope of this line = intrinsic dim\nfitted ≈ {d_twoNN:.2f}  (truth = 2)")
ax.grid(alpha=0.3)

# --- correlation dimension ---
sub = D[rng.choice(m, 1200, replace=False)]
dd = pdist(sub)
radii = np.logspace(np.log10(dd.min()*1.5), np.log10(dd.max()*0.5), 25)
C = np.array([(dd < r).mean() for r in radii])
lo, hi = 6, 19
sl = np.polyfit(np.log(radii[lo:hi]), np.log(C[lo:hi]), 1)[0]
ax = axes[2]
ax.loglog(radii, C, "o-", color="#9467bd", ms=4)
ax.loglog(radii[lo:hi], C[lo:hi], "o", color="crimson", ms=6, label="fit region")
ax.set_xlabel("radius r"); ax.set_ylabel("C(r) = frac. of pairs within r")
ax.set_title(f"Correlation dim = log-log slope\nfitted ≈ {sl:.2f}  (truth = 2)")
ax.legend(fontsize=10); ax.grid(alpha=0.3, which="both")

fig.text(0.5, 0.005, "Applied to our backtracking activations (1536-D ambient) these same estimators "
         "return ~7-10, not ~1536 — the cloud is intrinsically low-dimensional.",
         ha="center", fontsize=11, style="italic")
plt.tight_layout(rect=[0, 0.04, 1, 0.93])
plt.savefig(f"{OUT}/viz3_intrinsic_estimators.png", dpi=130); plt.close()
print(f"viz3 written; TwoNN={d_twoNN:.2f} corr_dim={sl:.2f}")
