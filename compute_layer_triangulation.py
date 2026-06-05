#!/usr/bin/env python3
"""
Multi-criteria layer triangulation.

Reads layer-sweep outputs from Phase 5 (participation ratio), Phase 5c (probe
accuracy), and Phase 7b-pilot (attribution patching effect). For each target
behaviour, identifies the manifold-peak layer in each curve and constructs a
'candidate layer set' as the union. This set is what downstream phases (5b deep
analysis, 6 vector construction, 7 steering eval) operate on.

GEOMETRY SIGNAL = participation ratio (PR), NOT d_eff_70.
  The manifold hypothesis predicts a *low*-dimensional curved manifold, so the
  layer where structure is strongest is the one with the LOWEST PR (variance most
  concentrated). We therefore take argmin(PR). The earlier design used
  argmax(d_eff_70), but d_eff_70 saturated at the PCA component cap, producing a
  flat curve that always triggered the fallback. PR is sample-size-robust and is
  not capped in the same way.

Pre-registered rules (see plan doc):
  1. Apply 3-point moving average to each curve before peak detection.
  2. Geometry peak  = argmin of smoothed PR curve   (lower PR = stronger manifold).
     Probe/patching = argmax of their smoothed curves (higher = stronger).
     Plateau (within 1 SD of the extremum) counts.
  3. Fallback if all curves are flat: {L18, L27} (Venhoff and Huang defaults).
  4. Candidate set per behaviour: sorted unique union of peaks, capped at 4.
  5. Bonferroni correction across 4 behaviours for null-test significance.

Inputs:
  results/pca/{model_short}/null_pvalues_per_layer.json    (Phase 5 --with-nulls)
  results/pca/{model_short}/layer_profiles.json            (Phase 5)
  results/cross_layer/{model_short}/probe_accuracy.json    (Phase 5c)
  results/patching/{model_short}/pilot_effect_curves.json  (Phase 7b --pilot, optional)

Output:
  results/triangulation/{model_short}/candidate_layers.json
    {behaviour: {pr_peak, probe_peak, patching_peak,
                 candidate_set, agreement_level}}
  results/triangulation/{model_short}/layer_sweep_curves.png   (per-behaviour figure)
  results/triangulation/{model_short}/summary.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.annotation import TARGET_BEHAVIOURS
from src.config import provenance

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Pre-registered fallback layers if all curves are flat
FALLBACK_LAYERS = [18, 27]


def smooth_curve(y, window=3):
    """3-point moving average. Handles boundaries by reflection."""
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        return y
    pad = window // 2
    y_pad = np.pad(y, pad, mode="edge")  # edge (not reflect): preserves boundary troughs
    kernel = np.ones(window) / window
    return np.convolve(y_pad, kernel, mode="valid")


def find_peak_with_plateau(layers, values, plateau_sd=1.0, minimize=False):
    """Return peak layer plus a plateau set (within 1 SD of the extremum).

    layers:  sorted list of layer indices
    values:  same length, the (smoothed) metric per layer
    plateau_sd: how many SDs from the extremum counts as plateau
    minimize: if True the 'peak' is the minimum (used for participation ratio,
              where lower = stronger low-dimensional manifold).
    """
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return None, []
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return None, []
    if minimize:
        peak_idx = int(np.nanargmin(values))
    else:
        peak_idx = int(np.nanargmax(values))
    peak_layer = layers[peak_idx]
    sd = np.nanstd(values)
    if minimize:
        threshold = values[peak_idx] + plateau_sd * sd
        plateau = [layers[i] for i in range(len(layers))
                   if finite_mask[i] and values[i] <= threshold]
    else:
        threshold = values[peak_idx] - plateau_sd * sd
        plateau = [layers[i] for i in range(len(layers))
                   if finite_mask[i] and values[i] >= threshold]
    return peak_layer, plateau


def is_curve_flat(values, threshold_sd=0.03):
    """Curve is 'flat' if its coefficient of variation (SD/|mean|) is tiny.

    The threshold is deliberately small (0.03) so that only genuinely featureless
    curves are rejected. The old 0.5 threshold flagged every real curve as flat,
    including the participation-ratio curve (which has a real ~10-15% dip at the
    manifold-peak layer), and forced the {L18, L27} fallback every time.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return True
    return (np.std(v) / (abs(np.mean(v)) + 1e-9)) < threshold_sd


def triangulate_behaviour(beh, pr_curve, probe_curve, patching_curve):
    """Apply the per-behaviour triangulation rule.

    Each curve arg is dict {layer (int): value (float)} or None if unavailable.
      pr_curve:       participation ratio (geometry) - minimised
      probe_curve:    linear-probe accuracy          - maximised
      patching_curve: attribution-patching effect    - maximised
    """
    out = {"behaviour": beh}

    def _peak(curve, minimize=False):
        if curve is None:
            return None, [], "missing"
        layers = sorted(int(k) for k in curve.keys())
        values = [curve[k] for k in layers]
        smoothed = smooth_curve(values, window=3)
        if is_curve_flat(smoothed):
            return None, [], "flat"
        peak, plateau = find_peak_with_plateau(layers, smoothed, minimize=minimize)
        return peak, plateau, "peaked"

    pr_peak,    pr_plateau,    pr_status    = _peak(pr_curve, minimize=True)
    probe_peak, probe_plateau, probe_status = _peak(probe_curve, minimize=False)
    patch_peak, patch_plateau, patch_status = _peak(patching_curve, minimize=False)

    out.update({
        "pr_peak":          pr_peak,
        "pr_plateau":       pr_plateau,
        "pr_status":        pr_status,
        "probe_peak":       probe_peak,
        "probe_plateau":    probe_plateau,
        "probe_status":     probe_status,
        "patching_peak":    patch_peak,
        "patching_plateau": patch_plateau,
        "patching_status":  patch_status,
    })

    # Candidate set: union of peaks that actually fired (status "peaked").
    fired = [(nm, L) for nm, L, st in (
                ("PR", pr_peak, pr_status),
                ("probe", probe_peak, probe_status),
                ("patching", patch_peak, patch_status))
             if st == "peaked" and L is not None]
    n_fired = len(fired)
    candidates = sorted({L for _, L in fired})
    if n_fired == 0:
        out["candidate_set"]   = sorted(FALLBACK_LAYERS)
        out["agreement_level"] = "fallback (all curves flat or missing)"
    else:
        if len(candidates) > 4:
            candidates = candidates[:4]
        out["candidate_set"] = candidates
        names = "+".join(nm for nm, _ in fired)
        if n_fired == 1:
            out["agreement_level"] = f"single-signal ({names} only; others flat/missing)"
        elif len(candidates) == 1:
            out["agreement_level"] = f"strong ({n_fired} signals [{names}] converge on one layer)"
        elif len(candidates) <= 2:
            out["agreement_level"] = f"moderate ({n_fired} signals [{names}], {len(candidates)} layers)"
        else:
            out["agreement_level"] = f"weak ({n_fired} signals [{names}], {len(candidates)} layers)"

    return out


def load_phase5_curves(pca_dir):
    """Return {behaviour: {layer: participation_ratio}}, or None if Phase 5 absent.

    Participation ratio is the geometry signal; downstream we take its argmin.
    """
    p = pca_dir / "layer_profiles.json"
    if not p.exists():
        return None
    raw = json.load(open(p))
    out = {}
    for b in raw:
        layers = raw[b].get("layers")
        pr = raw[b].get("participation_ratio")
        if layers and pr:
            out[b] = dict(zip(layers, pr))
    return out or None


def load_phase5_nulls(pca_dir):
    """Return {behaviour: {layer: p_value}}."""
    p = pca_dir / "null_pvalues_per_layer.json"
    if not p.exists():
        return None
    raw = json.load(open(p))
    return {b: {int(L): raw[b][L]["p_value"] for L in raw[b]} for b in raw}


def load_phase5c_curves(probe_dir):
    """Return {behaviour: {layer: test_acc}}."""
    p = probe_dir / "probe_accuracy.json"
    if not p.exists():
        return None
    raw = json.load(open(p))
    out = {}
    for b in raw:
        out[b] = {}
        for L, r in raw[b].items():
            if isinstance(r, dict) and "test_acc" in r:
                out[b][int(L)] = r["test_acc"]
    return out


def load_phase7b_curves(patching_dir):
    """Return {behaviour: {layer: mean_effect}}."""
    p = patching_dir / "pilot_effect_curves.json"
    if not p.exists():
        return None
    raw = json.load(open(p))
    out = {}
    for b in raw:
        out[b] = {int(L): v["mean_effect"]
                  for L, v in raw[b].get("layer_effect", {}).items()}
    return out


def render_summary_md(triangulation, paths_used):
    lines = [
        "# Layer triangulation summary",
        "",
        "Geometry signal = **participation ratio** (argmin: lower PR = stronger "
        "low-dimensional manifold). Probe accuracy and patching effect use argmax.",
        "",
        "## Inputs",
        "",
    ]
    for label, p in paths_used.items():
        lines.append(f"- {label}: `{p}` ({'found' if p.exists() else 'MISSING'})")
    lines += ["", "## Per-behaviour layer peaks", "",
              "| Behaviour | PR trough | Probe peak | Patching peak | Candidate set | Agreement |",
              "|---|---|---|---|---|---|"]
    for b, t in triangulation.items():
        pr = t.get("pr_peak")       if t.get("pr_status")       == "peaked" else f"({t['pr_status']})"
        pb = t.get("probe_peak")    if t.get("probe_status")    == "peaked" else f"({t['probe_status']})"
        pa = t.get("patching_peak") if t.get("patching_status") == "peaked" else f"({t['patching_status']})"
        cs = ", ".join(str(x) for x in t["candidate_set"])
        lines.append(f"| {b} | {pr} | {pb} | {pa} | {cs} | {t['agreement_level']} |")
    lines += [
        "",
        "## Methodological commitments (pre-registered)",
        "",
        "1. 3-point moving average applied to each curve before peak detection.",
        "2. Geometry peak = argmin of smoothed participation-ratio curve.",
        "   Probe/patching peak = argmax of their smoothed curves. Plateau = within 1 SD.",
        "3. Fallback to {L18, L27} if all curves are flat (CV < 0.03).",
        "4. Candidate set capped at 4 layers per behaviour.",
        "5. Bonferroni correction across 4 behaviours for null-test significance.",
    ]
    return "\n".join(lines)


def plot_curves(triangulation, pr, probe, patching, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable; skipping plot")
        return

    behs = [b for b in TARGET_BEHAVIOURS if b in triangulation]
    n = len(behs)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, b in zip(axes, behs):
        ax.set_title(b)
        # Participation ratio on the primary (left) axis
        if pr and b in pr:
            layers = sorted(pr[b].keys())
            vals = [pr[b][L] for L in layers]
            ax.plot(layers, vals, "o-", color="goldenrod", label="participation ratio")
            ax.set_ylabel("PR (lower = stronger)", color="goldenrod")
            ax.tick_params(axis="y", labelcolor="goldenrod")
        # Probe accuracy + patching on a twin (right) axis
        ax2 = ax.twinx()
        plotted_right = False
        if probe and b in probe:
            layers = sorted(probe[b].keys())
            vals = [probe[b][L] for L in layers]
            ax2.plot(layers, vals, "s-", color="steelblue", label="probe acc")
            plotted_right = True
        if patching and b in patching:
            layers = sorted(patching[b].keys())
            vals = [patching[b][L] for L in layers]
            ax2.plot(layers, vals, "^-", color="firebrick", label="patching effect")
            plotted_right = True
        if plotted_right:
            ax2.set_ylabel("probe acc / patching effect")
        # Mark candidate set
        t = triangulation[b]
        for L in t.get("candidate_set", []):
            ax.axvline(L, color="gray", linestyle="--", alpha=0.4)
        # Combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Layer")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    logger.info(f"Plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-short", default="R1-1.5B")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    pca_dir      = Path(f"results/pca/{args.model_short}")
    probe_dir    = Path(f"results/cross_layer/{args.model_short}")
    patching_dir = Path(f"results/patching/{args.model_short}")
    if args.out_dir is None:
        args.out_dir = Path(f"results/triangulation/{args.model_short}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths_used = {
        "Phase 5 (PR curves)":         pca_dir / "layer_profiles.json",
        "Phase 5 (null p-values)":     pca_dir / "null_pvalues_per_layer.json",
        "Phase 5c (probe accuracy)":   probe_dir / "probe_accuracy.json",
        "Phase 7b-pilot (patching)":   patching_dir / "pilot_effect_curves.json",
    }

    (args.out_dir / "provenance.json").write_text(json.dumps(
        provenance(args, inputs=[str(p) for p in paths_used.values()]), indent=2))

    pr_curves       = load_phase5_curves(pca_dir)
    probe_curves    = load_phase5c_curves(probe_dir)
    patching_curves = load_phase7b_curves(patching_dir)

    if not any([pr_curves, probe_curves, patching_curves]):
        logger.error("No input curves found. Run Phase 5 (--with-nulls), 5c, or 7b-pilot first.")
        sys.exit(1)

    triangulation = {}
    for beh in TARGET_BEHAVIOURS:
        pr = pr_curves.get(beh)       if pr_curves       else None
        pb = probe_curves.get(beh)    if probe_curves    else None
        pa = patching_curves.get(beh) if patching_curves else None
        triangulation[beh] = triangulate_behaviour(beh, pr, pb, pa)

    with open(args.out_dir / "candidate_layers.json", "w") as f:
        json.dump(triangulation, f, indent=2)
    md = render_summary_md(triangulation, paths_used)
    (args.out_dir / "summary.md").write_text(md)
    plot_curves(triangulation, pr_curves, probe_curves, patching_curves,
                args.out_dir / "layer_sweep_curves.png")

    print(f"\nTriangulation results -> {args.out_dir}")
    for b, t in triangulation.items():
        print(f"  {b:<28s}  candidate: {t['candidate_set']}  ({t['agreement_level']})")


if __name__ == "__main__":
    main()
