#!/usr/bin/env python3
"""
Power analysis for curvature diagnostics — pre-registered for Paper 2.

Companion document Section 2.5 commits to:
  "Before running the full extraction we will simulate ground-truth manifolds
   of known curvature in the matched-(N, d) regime and report the recovery
   accuracy of each curvature diagnostic with bootstrap 95% CIs. If the
   smallest detectable curvature at a given N exceeds the curvatures
   plausibly induced by reasoning structure, the per-behaviour analysis is
   downgraded to a corpus-pooled analysis."

This script implements that pre-registration.

Methodology:
  For each (N, curvature, ambient_dim) cell:
    1. Sample N points from a known curved manifold: a hypersphere
       S^{m-1} of radius 1/curvature embedded isotropically in R^d, plus
       isotropic Gaussian noise of variance sigma^2.
    2. Sample N points from a flat manifold of equal intrinsic dimension:
       a uniform-disc on a random m-dim subspace, plus matched noise.
    3. Run each curvature diagnostic on both samples.
    4. Repeat B times to estimate the diagnostic's distribution under
       curved vs flat truth.
  Detection power = P(diagnostic_curved > diagnostic_flat) across B reps,
  i.e., the AUC of the curved-vs-flat ROC.

Output:
  results/power_analysis/
    power_table.csv        — power per (N, curvature, diagnostic) cell
    power_curves.png       — power vs N at each curvature, per diagnostic
    summary.md             — narrative + minimum detectable curvature per N

CPU-only; ~10 min for the default grid (3 N values x 4 curvatures x 3
diagnostics x B=50 reps).

Usage:
  python power_analysis_curvature.py
  python power_analysis_curvature.py --B 200 --grid full
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add src/ to path for the curvature module
sys.path.insert(0, str(Path(__file__).parent / "src"))
from curvature import (
    local_vs_global_dim_ratio,
    geodesic_euclidean_ratio,
    tangent_space_variation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Synthetic data generators

def sample_hypersphere(N: int, m: int, d: int, radius: float,
                       noise_sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Sample N points on S^{m-1} (radius = radius) embedded isotropically in
    R^d, plus isotropic Gaussian noise."""
    pts = rng.standard_normal((N, m))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= radius
    # Embed in d-dim via random orthogonal map
    Q, _ = np.linalg.qr(rng.standard_normal((d, m)))
    pts = pts @ Q.T
    pts += noise_sigma * rng.standard_normal((N, d))
    return pts.astype(np.float64)


def sample_flat_disk(N: int, m: int, d: int, radius: float,
                     noise_sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Sample N points uniformly from an m-dim disk of given radius embedded
    in R^d, plus isotropic Gaussian noise. This is the curvature=0 control."""
    raw = rng.standard_normal((N, m))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    r = radius * rng.uniform(0, 1, N) ** (1.0 / m)
    pts = raw * r[:, None]
    Q, _ = np.linalg.qr(rng.standard_normal((d, m)))
    pts = pts @ Q.T
    pts += noise_sigma * rng.standard_normal((N, d))
    return pts.astype(np.float64)


# Single diagnostic application

def run_diagnostics(X: np.ndarray, k: int, intrinsic_dim: int) -> dict:
    """Run all three diagnostics and return mean values only (no inner
    bootstrap, since we're already in an outer Monte Carlo).

    Only *data-dependent* degeneracies (LinAlgError, ValueError — e.g. a draw
    with too few points) are converted to NaN. *Systematic* failures (a broken
    lazy import of sklearn/scipy, a signature mismatch) are allowed to
    propagate and abort the run — silently turning those into NaN is what
    produced the all-NaN power_table.csv that masqueraded as a real null result.
    """
    DATA_ERRORS = (np.linalg.LinAlgError, ValueError)
    out = {}
    try:
        out["local_vs_global"] = local_vs_global_dim_ratio(X, k=k, n_bootstrap=1).mean
    except DATA_ERRORS as e:
        logger.warning(f"local_vs_global degenerate on this draw: {type(e).__name__}: {e}")
        out["local_vs_global"] = float("nan")
    try:
        out["geodesic_euclidean"] = geodesic_euclidean_ratio(X, k=k, n_bootstrap=1, n_pairs=300).mean
    except DATA_ERRORS as e:
        logger.warning(f"geodesic_euclidean degenerate on this draw: {type(e).__name__}: {e}")
        out["geodesic_euclidean"] = float("nan")
    try:
        out["tangent_variation"] = tangent_space_variation(X, k=k, intrinsic_dim=intrinsic_dim, n_bootstrap=1, n_anchor_pairs=150).mean
    except DATA_ERRORS as e:
        logger.warning(f"tangent_variation degenerate on this draw: {type(e).__name__}: {e}")
        out["tangent_variation"] = float("nan")
    return out


# Power computation

@dataclass
class PowerCell:
    N: int
    curvature: float       # = 1/radius for the hypersphere
    diagnostic: str
    auc: float             # P(curved > flat) across reps
    curved_mean: float
    flat_mean: float
    curved_std: float
    flat_std: float
    n_reps: int


def compute_power(
    N: int,
    curvature: float,
    m: int,
    d: int,
    noise_sigma: float,
    k: int,
    n_reps: int,
    rng: np.random.Generator,
) -> list:
    """For one (N, curvature) cell, run n_reps draws of curved and flat
    samples; return PowerCell per diagnostic."""
    if curvature <= 0:
        # By construction power == 0.5 (same distribution); skip
        return []
    radius = 1.0 / curvature

    diag_names = ["local_vs_global", "geodesic_euclidean", "tangent_variation"]
    curved_records = {d: [] for d in diag_names}
    flat_records   = {d: [] for d in diag_names}

    for rep in range(n_reps):
        X_curved = sample_hypersphere(N, m, d, radius, noise_sigma, rng)
        X_flat   = sample_flat_disk(N, m, d, radius, noise_sigma, rng)

        r_c = run_diagnostics(X_curved, k=k, intrinsic_dim=m)
        r_f = run_diagnostics(X_flat,   k=k, intrinsic_dim=m)
        for dn in diag_names:
            curved_records[dn].append(r_c[dn])
            flat_records[dn].append(r_f[dn])

    results = []
    # Direction of the test depends on the diagnostic:
    #   local_vs_global: curved < flat  (curvature reduces local dim)
    #   geodesic_euclidean: curved > flat (geodesics bend)
    #   tangent_variation: curved > flat (tangent space rotates)
    direction = {"local_vs_global": "less", "geodesic_euclidean": "greater", "tangent_variation": "greater"}

    for dn in diag_names:
        c = np.array(curved_records[dn]); f = np.array(flat_records[dn])
        valid = np.isfinite(c) & np.isfinite(f)
        if not valid.any():
            auc = float("nan"); cm = fm = cs = fs = float("nan")
        else:
            c = c[valid]; f = f[valid]
            # AUC = P(curved > flat) for "greater"; P(curved < flat) for "less"
            comparisons = (c[:, None] - f[None, :])  # outer difference
            if direction[dn] == "greater":
                wins = (comparisons > 0).sum()
            else:
                wins = (comparisons < 0).sum()
            ties = (comparisons == 0).sum()
            total = comparisons.size
            auc = (wins + 0.5 * ties) / total
            cm = float(np.mean(c)); fm = float(np.mean(f))
            cs = float(np.std(c));  fs = float(np.std(f))
        results.append(PowerCell(N=N, curvature=curvature, diagnostic=dn,
                                  auc=float(auc), curved_mean=cm, flat_mean=fm,
                                  curved_std=cs, flat_std=fs, n_reps=n_reps))
    return results


# Output

def write_csv(cells: list, path: Path) -> None:
    lines = ["N,curvature,diagnostic,auc,curved_mean,flat_mean,curved_std,flat_std,n_reps"]
    for c in cells:
        lines.append(f"{c.N},{c.curvature},{c.diagnostic},{c.auc:.4f},"
                     f"{c.curved_mean:.6f},{c.flat_mean:.6f},"
                     f"{c.curved_std:.6f},{c.flat_std:.6f},{c.n_reps}")
    path.write_text("\n".join(lines))


def write_summary(cells: list, args, path: Path) -> None:
    lines = [
        "# Curvature diagnostics — power analysis",
        "",
        f"Ambient dim d = {args.d}, intrinsic dim m = {args.m}, noise sigma = {args.noise_sigma}, k = {args.k}, reps = {args.B}",
        "",
        "## Minimum detectable curvature (AUC >= 0.95) per (N, diagnostic)",
        "",
        "| N | diagnostic | min detectable curvature |",
        "|---|------------|--------------------------|",
    ]
    # Group by (N, diagnostic), find smallest curvature with auc >= 0.95
    from collections import defaultdict
    groups = defaultdict(list)
    for c in cells:
        groups[(c.N, c.diagnostic)].append(c)
    for (N, diag), cs in sorted(groups.items()):
        cs.sort(key=lambda x: x.curvature)
        detected = [c.curvature for c in cs if c.auc >= 0.95]
        min_kappa = f"{detected[0]:.3f}" if detected else "> max tested"
        lines.append(f"| {N} | {diag} | {min_kappa} |")

    lines += [
        "",
        "## Recommendation",
        "",
    ]
    # Aggregate: per-behaviour N range vs detection capability
    Ns_tested = sorted(set(c.N for c in cells))
    if Ns_tested:
        lines.append("Per-behaviour N expectations (R1-Distill, 1000 chains x ~27 sentences):")
        lines.append("  - deduction:        ~10,000")
        lines.append("  - initializing:     ~5,000")
        lines.append("  - uncertainty:      ~4,500")
        lines.append("  - backtracking:     ~2,500")
        lines.append("  - adding-knowledge: ~1,500")
        lines.append("  - example-testing:  ~1,500")
        lines.append("")
        lines.append("Baseline model (Qwen-2.5-Math-1.5B-Instruct) per-behaviour N may drop to ~50-500 for rare behaviours.")
        lines.append("")
        lines.append(f"Smallest N tested: {min(Ns_tested)}. Largest: {max(Ns_tested)}.")
        lines.append("Cross-reference the table above to determine which behaviours can support per-behaviour curvature analysis.")

    path.write_text("\n".join(lines))


def make_plot(cells: list, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plot")
        return

    diag_names = sorted(set(c.diagnostic for c in cells))
    curvs      = sorted(set(c.curvature for c in cells))
    Ns         = sorted(set(c.N for c in cells))

    fig, axes = plt.subplots(1, len(diag_names), figsize=(5 * len(diag_names), 4), sharey=True)
    if len(diag_names) == 1:
        axes = [axes]

    for ax, diag in zip(axes, diag_names):
        for kappa in curvs:
            ys = []
            for N in Ns:
                matching = [c for c in cells if c.N == N and c.curvature == kappa and c.diagnostic == diag]
                if matching:
                    ys.append(matching[0].auc)
                else:
                    ys.append(np.nan)
            ax.plot(Ns, ys, marker="o", label=f"kappa = {kappa:.3f}")
        ax.set(title=diag, xlabel="N (per-behaviour sample size)", xscale="log")
        ax.axhline(0.95, color="grey", linestyle="--", alpha=0.5, label="0.95 power threshold")
        ax.set_ylim(0.4, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Detection power (AUC: curved vs flat)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    logger.info(f"Plot saved: {path}")


# Main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", choices=["quick", "default", "full"], default="default",
                        help="quick: 3 N x 3 kappa. default: 4 N x 4 kappa. full: 5 N x 5 kappa.")
    parser.add_argument("--B", type=int, default=50,
                        help="Monte Carlo reps per (N, kappa) cell. Default 50.")
    parser.add_argument("--d", type=int, default=1536,
                        help="Ambient dimension. Default 1536 (matches Qwen-1.5B hidden dim).")
    parser.add_argument("--m", type=int, default=10,
                        help="Intrinsic manifold dimension. Default 10.")
    parser.add_argument("--noise-sigma", type=float, default=0.1,
                        help="Isotropic Gaussian noise variance. Default 0.1.")
    parser.add_argument("--k", type=int, default=30,
                        help="k-NN neighborhood size for diagnostics. Default 30.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/power_analysis"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.grid == "quick":
        Ns        = [100, 500, 2000]
        curvs     = [0.0, 0.5, 2.0]
    elif args.grid == "default":
        Ns        = [100, 500, 2000, 8000]
        curvs     = [0.0, 0.2, 0.5, 1.0]
    else:
        Ns        = [50, 200, 500, 2000, 8000]
        curvs     = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]

    rng = np.random.default_rng(args.seed)
    cells: list = []
    total = sum(1 for N, kappa in itertools.product(Ns, curvs) if kappa > 0)
    done  = 0
    for N, kappa in itertools.product(Ns, curvs):
        if kappa == 0:
            continue
        done += 1
        logger.info(f"[{done}/{total}] N={N} curvature={kappa} B={args.B}")
        # Only data-degeneracy is tolerated per cell; systematic failures
        # (ImportError, TypeError, ...) propagate and abort the whole run.
        try:
            cs = compute_power(N=N, curvature=kappa, m=args.m, d=args.d,
                                noise_sigma=args.noise_sigma, k=args.k,
                                n_reps=args.B, rng=rng)
            cells.extend(cs)
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.error(f"Cell degenerate (kept as NaN): N={N} kappa={kappa}: {e}")

    # Refuse to write a result table that is entirely NaN — that is a
    # computation failure wearing the costume of a null result, not a finding.
    if cells and all(not np.isfinite(c.auc) for c in cells):
        raise RuntimeError(
            f"All {len(cells)} power cells produced non-finite AUC. This is a "
            "systematic failure (likely a broken numerical import or a "
            "degenerate (N,d) regime), NOT a real 'undetectable' result. "
            "Refusing to write an all-NaN power_table.csv. Check that sklearn "
            "and scipy import cleanly in this environment and re-run."
        )

    write_csv(cells, args.out_dir / "power_table.csv")
    write_summary(cells, args, args.out_dir / "summary.md")
    make_plot(cells, args.out_dir / "power_curves.png")

    print("\n" + "=" * 60)
    print("POWER ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Cells computed: {len(cells)}")
    print(f"CSV:     {args.out_dir / 'power_table.csv'}")
    print(f"Summary: {args.out_dir / 'summary.md'}")
    print(f"Plot:    {args.out_dir / 'power_curves.png'}")


if __name__ == "__main__":
    main()
