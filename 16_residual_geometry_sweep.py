#!/usr/bin/env python3
"""
16_residual_geometry_sweep.py — Label-free characterization of predictive structure.

For each layer, fit the Rung-1 delta predictor (chain-grouped OOF) and describe
the geometry of the RESIDUAL cloud — no correctness labels needed. Answers:
  * How predictable is the step-to-step displacement? (variance-explained R^2,
    residual/persistence ratio) — and does a learned map beat persistence anywhere?
  * Does the residual (the UNpredictable part) have low-dimensional structure?
    (correlation dimension + participation ratio of the residual cloud vs the raw
    displacement cloud)
  * Is there positional structure to surprise? (mean residual-direction churn,
    mean residual-norm slope across the trajectory)
This is descriptive Movement-1 material and tells us which layer (if any) to
focus the correctness gate on. Runs entirely on CPU on the existing mean-pooled
activations.

  python3 16_residual_geometry_sweep.py --layers 5,11,14,17,20,23,27
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import backup_existing, provenance  # noqa: E402
from src.intrinsic_dim import correlation_dimension_estimate  # noqa: E402
from src.nulls import participation_ratio  # noqa: E402
from src.predict.trajectory_dataset import build_step_datasets, make_supervised_pairs  # noqa: E402
from src.predict.predictor import RidgeConfig, oof_residuals, chain_residual_features  # noqa: E402


def _id(X, subsample):
    try:
        r = correlation_dimension_estimate(X, subsample=min(subsample, X.shape[0]))
        return float(r.estimate)
    except Exception as e:  # noqa: BLE001
        return float("nan")


def _layer_metrics(datasets, subsample):
    cfg = RidgeConfig(target="delta")
    pairs = make_supervised_pairs(datasets)
    res = oof_residuals(pairs, cfg)
    delta = (pairs["X_next"] - pairs["X_hist"]).astype(np.float64)
    resid = res.residuals
    rn = np.linalg.norm(resid, axis=1)
    dn = np.linalg.norm(delta, axis=1)
    var_d = float(delta.var(axis=0).sum())
    var_r = float(resid.var(axis=0).sum())

    pairs1 = make_supervised_pairs(datasets, max_gap=1)
    ratio1 = float("nan")
    if pairs1["X_hist"].shape[0] > 50:
        res1 = oof_residuals(pairs1, cfg)
        rn1 = np.linalg.norm(res1.residuals, axis=1).mean()
        dn1 = np.linalg.norm(pairs1["X_next"] - pairs1["X_hist"], axis=1).mean()
        ratio1 = float(rn1 / dn1) if dn1 else float("nan")

    feats, _, names = chain_residual_features(res)
    churn = float(feats[:, names.index("resid_dir_churn")].mean())
    slope = float(feats[:, names.index("resid_slope")].mean())

    return {
        "n_pairs": int(pairs["X_hist"].shape[0]),
        "n_chains": int(np.unique(pairs["groups"]).size),
        "residual_to_persistence_ratio": float(rn.mean() / dn.mean()) if dn.mean() else float("nan"),
        "residual_to_persistence_ratio_gap1": ratio1,
        "displacement_variance_explained_r2": float(1.0 - var_r / var_d) if var_d else float("nan"),
        "resid_corr_dim": _id(resid, subsample),
        "delta_corr_dim": _id(delta, subsample),
        "resid_participation_ratio": float(participation_ratio(resid)),
        "delta_participation_ratio": float(participation_ratio(delta)),
        "mean_resid_dir_churn": churn,
        "mean_resid_norm_slope": slope,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", default="data/activations/R1-1.5B")
    ap.add_argument("--chains", default="data/annotated_R1-1.5B.json")
    ap.add_argument("--layers", default="5,11,14,17,20,23,27")
    ap.add_argument("--subsample", type=int, default=3000)
    ap.add_argument("--max-chains", type=int, default=None, help="smoke: cap chains")
    ap.add_argument("--out", default="results/predict/R1-1.5B")
    args = ap.parse_args()

    chains = json.load(open(args.chains))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for layer in [int(x) for x in args.layers.split(",")]:
        ds_all = build_step_datasets(chains, args.activations, layer)
        if args.max_chains:
            ds_all = ds_all[: args.max_chains]
        m = {"layer": layer, **_layer_metrics(ds_all, args.subsample)}
        results.append(m)
        print(f"layer {layer:2d}: R2={m['displacement_variance_explained_r2']:+.3f} "
              f"ratio={m['residual_to_persistence_ratio']:.3f} "
              f"resid_ID={m['resid_corr_dim']:.2f} (delta_ID={m['delta_corr_dim']:.2f}) "
              f"PR={m['resid_participation_ratio']:.1f} churn={m['mean_resid_dir_churn']:.3f}")

    payload = {"results": results, "provenance": provenance(args, inputs=[args.chains])}
    out_json = out_dir / "residual_geometry_sweep.json"
    backup_existing(out_json)
    json.dump(payload, open(out_json, "w"), indent=2, default=str)

    lines = ["# Residual-geometry sweep (label-free)", "",
             "| layer | disp. R² | resid/persist | resid/persist (gap1) | resid ID | delta ID | resid PR | dir-churn | norm-slope |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in results:
        lines.append(f"| {m['layer']} | {m['displacement_variance_explained_r2']:+.3f} | "
                     f"{m['residual_to_persistence_ratio']:.3f} | {m['residual_to_persistence_ratio_gap1']:.3f} | "
                     f"{m['resid_corr_dim']:.2f} | {m['delta_corr_dim']:.2f} | "
                     f"{m['resid_participation_ratio']:.1f} | {m['mean_resid_dir_churn']:.3f} | "
                     f"{m['mean_resid_norm_slope']:+.4f} |")
    lines += ["", "R² > 0 ⇒ learned predictor beats persistence at that layer; "
              "resid ID < delta ID ⇒ the unpredictable part is lower-dimensional than the raw step."]
    (out_dir / "residual_geometry_sweep.md").write_text("\n".join(lines))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
