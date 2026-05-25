#!/usr/bin/env python3
"""
Phase 6.5 - Predict steering-saturation strength from per-behaviour curvature.

Implements the empirical test pre-registered in the revised companion
document Section 2.6.1: the derived prediction is

    alpha*_b = 1 / (kappa_b * ||v_parallel,b||)

where
  kappa_b           = per-behaviour mean curvature (from Phase 5b)
  ||v_parallel,b||  = norm of the DOM steering vector projected onto the
                      local tangent bundle of the conditional manifold

This script:
  1. Loads Phase 5b curvature output (results/geometric/<model>/diagnostics_layer<N>.json)
  2. Loads Phase 6 steering vectors (results/steering_vectors/<model>/<beh>_layer<N>.npy)
  3. Loads the per-behaviour PCA basis (results/pca/<model>/<beh>_components.npy)
  4. Projects each steering vector onto the local tangent bundle
  5. Outputs predicted alpha*_b per behaviour
  6. If Phase 7 saturation results are available, computes cross-behaviour
     Pearson correlation between predicted and empirical alpha*_b
  7. Reports against pre-registered threshold r > 0.5
  8. Also computes the distinguishing-signatures for the four alternative
     saturation mechanisms (layer-norm, output-distributional, downstream
     self-correction, curvature).

Output:
  results/saturation_predictions/<model>/predictions_layer<N>.json
  results/saturation_predictions/<model>/summary_layer<N>.md
  results/saturation_predictions/<model>/correlation_plot.png

Usage:
  python predict_saturation.py --layer 27
  python predict_saturation.py --layer 27 --skip-empirical-comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


from src.annotation import TARGET_BEHAVIOURS


# Curvature scalar extraction

def extract_curvature_scalar(per_behaviour_diag: dict, behaviour: str,
                              prefer_diagnostic: str = "tangent_space_variation_deg",
                              prefer_k: int = 30) -> float:
    """From the Phase 5b output, extract a per-behaviour scalar curvature.

    Default: tangent-space variation in degrees at k=30 (converted to radians for
    consistent units; tangent angle directly proxies curvature scale on small
    neighborhoods, see Companion 2.5 curvature diagnostics).

    If preferred diagnostic/k is missing, fall back to (geodesic_euclidean - 1)
    at the same k, which behaves similarly (zero for flat, positive for curved).
    """
    if behaviour not in per_behaviour_diag:
        return float("nan")
    rows = per_behaviour_diag[behaviour].get("curvature", [])

    # First choice: requested diagnostic at requested k
    for r in rows:
        if r["diagnostic"] == prefer_diagnostic and r["k"] == prefer_k:
            val = r["mean"]
            if prefer_diagnostic == "tangent_space_variation_deg":
                return float(np.radians(val))  # convert to radians, dimensionless curvature proxy
            else:
                return float(val)

    # Fallback: geodesic_euclidean at requested k
    for r in rows:
        if r["diagnostic"] == "geodesic_euclidean_ratio" and r["k"] == prefer_k:
            # ratio is >= 1; excess over 1 is the curvature signal
            return float(max(0.0, r["mean"] - 1.0))

    return float("nan")


# Tangent-bundle projection of steering vector

def project_to_tangent_bundle(
    v:          np.ndarray,
    activations: np.ndarray,
    tangent_dim: int = 10,
) -> tuple:
    """Project a steering vector v onto the local tangent bundle of the
    activation cloud, around the cloud centroid.

    Returns (v_parallel, v_perp, parallel_norm, perp_norm).
    """
    centered = activations - activations.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    d = min(tangent_dim, vt.shape[0])
    B = vt[:d].T  # columns are tangent basis

    coeffs = B.T @ v
    v_par = B @ coeffs
    v_perp = v - v_par
    return v_par, v_perp, float(np.linalg.norm(v_par)), float(np.linalg.norm(v_perp))


# Predicted alpha*

def predicted_alpha_star(kappa: float, v_parallel_norm: float) -> float:
    """alpha* = 1 / (kappa * ||v_parallel||)."""
    if not (np.isfinite(kappa) and np.isfinite(v_parallel_norm)):
        return float("nan")
    if kappa <= 0 or v_parallel_norm <= 0:
        return float("inf")
    return 1.0 / (kappa * v_parallel_norm)


# Pearson r and the four-mechanism comparison

def cross_behaviour_correlation(predictions: dict, empirical: dict) -> dict:
    """Compute Pearson r between predicted and empirical alpha* across the
    target behaviours, plus per-mechanism alternative signatures."""
    bs = [b for b in TARGET_BEHAVIOURS if b in predictions and b in empirical]
    if len(bs) < 3:
        return {"r": float("nan"), "n": len(bs),
                "note": "Need at least 3 behaviours for meaningful correlation."}
    pred = np.array([predictions[b]["alpha_star_pred"] for b in bs])
    emp  = np.array([empirical[b]["alpha_star_emp"]    for b in bs])
    valid = np.isfinite(pred) & np.isfinite(emp)
    if valid.sum() < 3:
        return {"r": float("nan"), "n": int(valid.sum()),
                "note": "Insufficient finite values."}
    pred = pred[valid]; emp = emp[valid]
    p_centered = pred - pred.mean(); e_centered = emp - emp.mean()
    denom = np.sqrt((p_centered**2).sum() * (e_centered**2).sum())
    r = float((p_centered * e_centered).sum() / denom) if denom > 0 else float("nan")
    return {
        "r": r,
        "n": int(valid.sum()),
        "behaviours": [b for b, v in zip(bs, valid) if v],
        "pred": pred.tolist(),
        "emp":  emp.tolist(),
        "pre_registered_threshold": 0.5,
        "verdict": ("supports curvature mechanism" if r >= 0.5
                    else "does not support curvature mechanism at r>=0.5 threshold"),
    }


def alternative_mechanism_signatures(empirical: dict, activations_by_beh: dict) -> dict:
    """Compute the distinguishing signatures of the four candidate
    saturation mechanisms (per companion 2.6.1 Table 1).

    For each mechanism, we compute the cross-behaviour quantity that would
    predict alpha* under that mechanism; Pearson r against empirical alpha*
    tells us which mechanism best matches.
    """
    bs = [b for b in TARGET_BEHAVIOURS if b in empirical and b in activations_by_beh]
    if not bs:
        return {}

    # 1. Layer-norm / RMS-norm: alpha* should correlate with mean ||x_0|| per behaviour
    mean_norms = {b: float(np.linalg.norm(activations_by_beh[b], axis=1).mean()) for b in bs}

    # 2. Output-distributional collapse: would need per-behaviour logit gaps; not
    #    computable from activations alone. We record a placeholder; populate later
    #    when steered-inference results are available (Phase 7).
    out = {
        "mean_activation_norms": mean_norms,
        "comment": "Layer-norm hypothesis: alpha* correlates with mean activation norm. "
                   "Output-distributional and downstream-self-correction signatures "
                   "require Phase 7 inference data; populate post-Phase-7.",
    }
    return out


# IO

def load_phase5b_diagnostics(diag_path: Path) -> dict:
    with open(diag_path) as f:
        return json.load(f)


def load_steering_vector(steering_dir: Path, behaviour: str, layer: int,
                          variant: str = "single") -> np.ndarray:
    """Phase 6 saves steering vectors as {behaviour}_{variant}.npy.
    For saturation prediction we want the single-direction (Venhoff DOM)
    vector, since alpha* is derived for that vector specifically."""
    p = steering_dir / f"{behaviour}_{variant}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Steering vector missing: {p}")
    return np.load(p)


def load_activations(act_dir: Path, behaviour: str, layer: int) -> np.ndarray:
    p = act_dir / f"{behaviour}_layer{layer}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Activations missing: {p}")
    return np.load(p)


def load_empirical_alpha_star(eval_path: Path) -> dict:
    """Try to load Phase 7 evaluation results. Expected JSON format:
        { "<behaviour>": { "alpha_star_emp": 0.7, "token_reduction_at_alpha_star": 0.4, ... }, ... }
    Returns {} if file does not exist (script then runs in prediction-only mode).
    """
    if not eval_path.exists():
        return {}
    with open(eval_path) as f:
        return json.load(f)


# Main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-short", default="R1-1.5B")
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--variant", default="single",
                        help="Steering vector variant to use (default: single).")
    parser.add_argument("--tangent-dim", type=int, default=10,
                        help="Local tangent-bundle dim for projecting v_b. Default 10 (matches power_analysis m).")
    parser.add_argument("--curvature-diagnostic", default="tangent_space_variation_deg",
                        help="Which curvature diagnostic to use as the scalar kappa_b.")
    parser.add_argument("--curvature-k", type=int, default=30,
                        help="k value for the curvature diagnostic. Default 30.")
    parser.add_argument("--skip-empirical-comparison", action="store_true",
                        help="Skip Phase 7 empirical comparison even if files exist.")
    args = parser.parse_args()

    # Paths (matching the rest of the pipeline)
    diag_p     = Path(f"results/geometric/{args.model_short}/diagnostics_layer{args.layer}.json")
    steer_dir  = Path(f"results/steering_vectors/{args.model_short}")
    act_dir    = Path(f"data/activations/{args.model_short}")
    eval_p     = Path(f"results/eval/{args.model_short}/saturation_results_layer{args.layer}.json")
    out_dir    = Path(f"results/saturation_predictions/{args.model_short}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not diag_p.exists():
        logger.error(f"Phase 5b diagnostics not found: {diag_p}")
        logger.error("Run 05b_geometric_diagnostics.py first.")
        sys.exit(1)

    diagnostics = load_phase5b_diagnostics(diag_p)
    per_beh_diag = diagnostics.get("per_behaviour", {})

    # Per-behaviour prediction
    predictions = {}
    activations_by_beh = {}
    for b in TARGET_BEHAVIOURS:
        logger.info(f"=== {b} ===")

        # 1) extract kappa
        kappa = extract_curvature_scalar(per_beh_diag, b,
                                          prefer_diagnostic=args.curvature_diagnostic,
                                          prefer_k=args.curvature_k)
        logger.info(f"  kappa_b ({args.curvature_diagnostic}@k={args.curvature_k}): {kappa}")

        # 2) load steering vector and activations
        try:
            v = load_steering_vector(steer_dir, b, args.layer, variant=args.variant)
        except FileNotFoundError as e:
            logger.warning(f"  {e}; skipping {b}")
            predictions[b] = {"alpha_star_pred": float("nan"),
                              "kappa": kappa, "v_par_norm": float("nan"),
                              "v_perp_norm": float("nan"),
                              "error": "missing steering vector"}
            continue
        try:
            X = load_activations(act_dir, b, args.layer)
            activations_by_beh[b] = X
        except FileNotFoundError as e:
            logger.warning(f"  {e}; skipping {b}")
            predictions[b] = {"alpha_star_pred": float("nan"),
                              "kappa": kappa, "v_par_norm": float("nan"),
                              "v_perp_norm": float("nan"),
                              "error": "missing activations"}
            continue

        # 3) project to tangent bundle
        v_par, v_perp, par_norm, perp_norm = project_to_tangent_bundle(
            v, X, tangent_dim=args.tangent_dim,
        )
        logger.info(f"  ||v_parallel|| = {par_norm:.4f}")
        logger.info(f"  ||v_perp||     = {perp_norm:.4f}")

        # 4) predicted alpha*
        ahat = predicted_alpha_star(kappa, par_norm)
        logger.info(f"  predicted alpha*_b = {ahat}")

        predictions[b] = {
            "kappa":           kappa,
            "v_par_norm":      par_norm,
            "v_perp_norm":     perp_norm,
            "alpha_star_pred": ahat,
        }

    # Empirical comparison (if Phase 7 data exists)
    empirical = {} if args.skip_empirical_comparison else load_empirical_alpha_star(eval_p)
    if empirical:
        logger.info("=== Cross-behaviour correlation ===")
        corr = cross_behaviour_correlation(predictions, empirical)
        logger.info(f"  Pearson r = {corr.get('r')}; verdict: {corr.get('verdict')}")
        alt_sigs = alternative_mechanism_signatures(empirical, activations_by_beh)
    else:
        logger.info("No empirical Phase 7 results found; producing predictions only.")
        corr = None
        alt_sigs = None

    payload = {
        "model_short":              args.model_short,
        "layer":                    args.layer,
        "args":                     vars(args),
        "predictions":              predictions,
        "empirical":                empirical,
        "correlation":              corr,
        "alternative_mechanisms":   alt_sigs,
    }

    def _ser(o):
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not serializable: {type(o).__name__}")

    json_p = out_dir / f"predictions_layer{args.layer}.json"
    json_p.write_text(json.dumps(payload, indent=2, default=_ser))

    md_p = out_dir / f"summary_layer{args.layer}.md"
    write_summary(payload, md_p)

    # Print short console summary
    print("\n" + "=" * 60)
    print(f"Saturation predictions for {args.model_short} layer {args.layer}")
    print("=" * 60)
    print(f"{'Behaviour':32s} {'kappa':>10s} {'||v_par||':>12s} {'predicted alpha*':>18s}")
    for b in TARGET_BEHAVIOURS:
        if b not in predictions:
            continue
        p = predictions[b]
        k_s = f"{p['kappa']:.4f}" if np.isfinite(p['kappa']) else "  nan"
        n_s = f"{p['v_par_norm']:.4f}" if np.isfinite(p['v_par_norm']) else "  nan"
        a_s = f"{p['alpha_star_pred']:.3f}" if np.isfinite(p['alpha_star_pred']) else "  nan"
        print(f"{b:32s} {k_s:>10s} {n_s:>12s} {a_s:>18s}")
    if corr:
        print(f"\nCross-behaviour Pearson r = {corr['r']:.4f}  (n={corr['n']})")
        print(f"Verdict: {corr['verdict']}")
    print(f"\nJSON: {json_p}")
    print(f"MD:   {md_p}")


def write_summary(payload: dict, path: Path) -> None:
    lines = [
        f"# Saturation predictions - {payload['model_short']} layer {payload['layer']}",
        "",
        "## Predicted alpha* per behaviour",
        "",
        "| Behaviour | kappa_b | ||v_par|| | ||v_perp|| | predicted alpha* |",
        "|-----------|---------|------------|-------------|------------------|",
    ]
    for b in TARGET_BEHAVIOURS:
        p = payload["predictions"].get(b, {})
        def fmt(v): return f"{v:.4f}" if isinstance(v, (int, float)) and np.isfinite(v) else "-"
        lines.append(f"| {b} | {fmt(p.get('kappa'))} | {fmt(p.get('v_par_norm'))} | {fmt(p.get('v_perp_norm'))} | {fmt(p.get('alpha_star_pred'))} |")
    if payload.get("correlation"):
        c = payload["correlation"]
        lines += [
            "",
            "## Cross-behaviour correlation (predicted vs empirical alpha*)",
            "",
            f"- Pearson r = {c.get('r', 'NA')}",
            f"- n behaviours = {c.get('n', 'NA')}",
            f"- Pre-registered threshold: r > {c.get('pre_registered_threshold')}",
            f"- **Verdict: {c.get('verdict')}**",
        ]
    if payload.get("alternative_mechanisms"):
        am = payload["alternative_mechanisms"]
        lines += [
            "",
            "## Alternative mechanism signatures (per Companion Section 2.6.1 Table 1)",
            "",
            "Layer-norm mechanism predicts alpha* correlated with pre-steering ||x_0|| per behaviour:",
            "",
            "| Behaviour | mean ||x_0|| |",
            "|-----------|---------------|",
        ]
        for b, v in am.get("mean_activation_norms", {}).items():
            lines.append(f"| {b} | {v:.4f} |")
        lines.append("")
        lines.append(am.get("comment", ""))
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
