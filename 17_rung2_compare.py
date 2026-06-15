#!/usr/bin/env python3
"""
17_rung2_compare.py — Rung-1 (ridge) vs Rung-2 (JEPA) head-to-head on correctness.

For the chosen layer(s), compute chain-grouped correctness AUC from the residual
geometry of BOTH predictors and the same baselines (raw curvature, persistence,
length), each with a label-permutation null (cheap; operates on precomputed
per-chain features). Rung 2 earns its place only if its residual AUC beats Rung 1,
beats the baselines, and the label-permutation null rejects.

The expensive within-chain step-shuffle null is NOT rerun here (it would retrain
the JEPA hundreds of times); 15_predict_gate.py already runs it for the ridge
Rung-1 signal. Use a small layer set for the JEPA pass (CPU training).

  python3 17_rung2_compare.py --labels data/correctness_R1-1.5B_pilot.json --layers 14,27
  python3 17_rung2_compare.py --smoke 80 --layers 14 --jepa-epochs 5   # plumbing check
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import backup_existing, provenance  # noqa: E402
from src.predict.trajectory_dataset import build_step_datasets, make_supervised_pairs  # noqa: E402
from src.predict.predictor import RidgeConfig, oof_residuals, chain_residual_features  # noqa: E402
from src.predict.jepa import JEPAConfig, oof_residuals_jepa  # noqa: E402
from src.predict.evaluation import (  # noqa: E402
    align_labels, grouped_auc, raw_curvature_features, raw_step_features, length_features,
)
from src.predict.nulls_predict import label_permutation_null  # noqa: E402


def _label_map(labels: dict, min_confidence: str | None) -> dict:
    order = {"low": 0, "medium": 1, "high": 2}
    floor = order.get(min_confidence, -1)
    out = {}
    for cid, rec in labels.items():
        if rec.get("correct") is None:
            continue
        if min_confidence and order.get(str(rec.get("confidence")).lower(), -1) < floor:
            continue
        out[cid] = bool(rec["correct"])
    return out


def _auc(features, cids, label_map):
    return grouped_auc(features, align_labels(cids, label_map), cids)


def _resid_auc(res, label_map, strata_map, n_resamples):
    feats, cids, _ = chain_residual_features(res)
    auc = _auc(feats, cids, label_map)
    rn = float(np.linalg.norm(res.residuals, axis=1).mean())
    strata = (np.array([strata_map.get(c) for c in cids], dtype=object)
              if strata_map else None)
    lp = label_permutation_null(feats, cids, align_labels(cids, label_map),
                                strata=strata, n_resamples=n_resamples)
    return auc, rn, {"real": lp.real_value, "null_mean": lp.null_mean, "p_value": lp.p_value}


def _compare_layer(chains, activations, layer, label_map, strata_map, args) -> dict:
    ds_all = build_step_datasets(chains, activations, layer)
    labelled = [d for d in ds_all if d.chain_id in label_map]
    n_pos = sum(1 for d in labelled if label_map[d.chain_id])
    out = {"layer": layer, "n_labelled_chains": len(labelled),
           "n_correct": n_pos, "n_incorrect": len(labelled) - n_pos,
           "n_train_chains": len(ds_all)}
    if len(labelled) < 10 or n_pos < 3 or (len(labelled) - n_pos) < 3:
        out["status"] = "insufficient labelled chains (need >=10, >=3/class)"
        return out

    # Train the label-agnostic predictor on ALL chains (chain-grouped OOF);
    # correctness AUC is evaluated only on the labelled subset via align_labels
    # (unlabelled chains -> NaN label -> dropped by grouped_auc). Restricting
    # training to the labelled subset underpowers it (see residual_geometry_sweep).
    pairs = make_supervised_pairs(ds_all, max_gap=args.max_gap)
    persist = float(np.linalg.norm(pairs["X_next"] - pairs["X_hist"], axis=1).mean())

    rres = oof_residuals(pairs, RidgeConfig(target="delta"))
    ridge_auc, ridge_rn, ridge_lp = _resid_auc(rres, label_map, strata_map, args.label_resamples)

    jcfg = JEPAConfig(target="delta", anti_collapse=args.jepa_anti_collapse,
                      epochs=args.jepa_epochs, hidden_dim=args.jepa_hidden, n_splits=5)
    jres = oof_residuals_jepa(pairs, jcfg)
    jepa_auc, jepa_rn, jepa_lp = _resid_auc(jres, label_map, strata_map, args.label_resamples)

    cf, ccids, _ = raw_curvature_features(ds_all)
    sf, scids, _ = raw_step_features(ds_all)
    lf, lcids, _ = length_features(ds_all)
    base = {"curvature": _auc(cf, ccids, label_map)["auc_oof"],
            "persistence": _auc(sf, scids, label_map)["auc_oof"],
            "length": _auc(lf, lcids, label_map)["auc_oof"]}

    out.update({
        "persistence_norm": persist,
        "ridge": {"auc": ridge_auc, "resid_persist_ratio": ridge_rn / persist if persist else float("nan"),
                  "label_perm": ridge_lp},
        "jepa": {"auc": jepa_auc, "resid_persist_ratio": jepa_rn / persist if persist else float("nan"),
                 "label_perm": jepa_lp, "anti_collapse": args.jepa_anti_collapse, "epochs": args.jepa_epochs},
        "baselines_auc": base,
    })
    r1 = ridge_auc["auc_oof"]
    r2 = jepa_auc["auc_oof"]
    best_base = max(v for v in base.values() if np.isfinite(v)) if base else float("nan")
    out["rung2_beats"] = bool(np.isfinite(r2) and r2 > max(r1, best_base) and jepa_lp["p_value"] < 0.05)
    out["rung1_beats"] = bool(np.isfinite(r1) and r1 > best_base and ridge_lp["p_value"] < 0.05)
    return out


def _write_md(path: Path, results, smoke):
    L = ["# Rung-1 (ridge) vs Rung-2 (JEPA) — correctness AUC", ""]
    if smoke:
        L.append("**SMOKE — synthetic random labels; AUCs should be ~0.5.**\n")
    for r in results:
        L.append(f"## layer {r['layer']}")
        if "ridge" not in r:
            L.append(f"- {r.get('status')}\n"); continue
        L.append(f"- labelled: {r['n_labelled_chains']} (correct {r['n_correct']}/incorrect {r['n_incorrect']})")
        L.append("")
        L.append("| predictor | AUC (oof) | fold mean±std | resid/persist | label-perm p |")
        L.append("|---|---|---|---|---|")
        for name in ("ridge", "jepa"):
            a = r[name]["auc"]
            L.append(f"| {name} | {a['auc_oof']:.3f} | {a['auc_fold_mean']:.3f}±{a['auc_fold_std']:.3f} | "
                     f"{r[name]['resid_persist_ratio']:.3f} | {r[name]['label_perm']['p_value']:.3f} |")
        b = r["baselines_auc"]
        L.append(f"| _baseline_ curvature | {b['curvature']:.3f} | | | |")
        L.append(f"| _baseline_ persistence | {b['persistence']:.3f} | | | |")
        L.append(f"| _baseline_ length | {b['length']:.3f} | | | |")
        L.append("")
        L.append(f"- **Rung-1 beats baselines+null: {r['rung1_beats']}; Rung-2 beats Rung-1+baselines+null: {r['rung2_beats']}**\n")
    path.write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", default="data/activations/R1-1.5B")
    ap.add_argument("--chains", default="data/annotated_R1-1.5B.json")
    ap.add_argument("--tasks", default="data/tasks_final.json")
    ap.add_argument("--labels", default="data/correctness_R1-1.5B_pilot.json")
    ap.add_argument("--layers", default="14,27")
    ap.add_argument("--max-gap", type=int, default=None)
    ap.add_argument("--min-confidence", choices=["low", "medium", "high"], default=None)
    ap.add_argument("--label-resamples", type=int, default=500)
    ap.add_argument("--jepa-anti-collapse", choices=["none", "barlow", "sigreg"], default="barlow")
    ap.add_argument("--jepa-epochs", type=int, default=30)
    ap.add_argument("--jepa-hidden", type=int, default=512)
    ap.add_argument("--out", default="results/predict/R1-1.5B")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    chains = json.load(open(args.chains))
    tasks = json.load(open(args.tasks))
    difficulty = {t["id"]: t.get("difficulty") for t in tasks}

    if args.smoke:
        rng = random.Random(42)
        label_map = {c["task_id"]: bool(rng.getrandbits(1)) for c in chains[: args.smoke]}
        smoke = True
    else:
        label_map = _label_map(json.load(open(args.labels)), args.min_confidence)
        smoke = False
    strata_map = {cid: difficulty.get(cid) for cid in label_map}
    print(f"{'SMOKE ' if smoke else ''}{len(label_map)} labels ({sum(label_map.values())} correct)")

    results = []
    for layer in [int(x) for x in args.layers.split(",")]:
        print(f"\n=== layer {layer} ===")
        r = _compare_layer(chains, args.activations, layer, label_map, strata_map, args)
        results.append(r)
        if "ridge" in r:
            print(f"  ridge AUC={r['ridge']['auc']['auc_oof']:.3f} (lp p={r['ridge']['label_perm']['p_value']:.3f}) "
                  f"ratio={r['ridge']['resid_persist_ratio']:.3f}")
            print(f"  jepa  AUC={r['jepa']['auc']['auc_oof']:.3f} (lp p={r['jepa']['label_perm']['p_value']:.3f}) "
                  f"ratio={r['jepa']['resid_persist_ratio']:.3f}")
            print(f"  baselines {r['baselines_auc']}")
            print(f"  rung1_beats={r['rung1_beats']} rung2_beats={r['rung2_beats']}")
        else:
            print(f"  {r.get('status')}")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if smoke else "rung2"
    oj = out_dir / f"compare_{tag}.json"
    backup_existing(oj)
    json.dump({"results": results, "smoke": smoke,
               "provenance": provenance(args, inputs=[args.chains] + ([] if smoke else [args.labels]))},
              open(oj, "w"), indent=2, default=str)
    _write_md(out_dir / f"compare_{tag}.md", results, smoke)
    print(f"\nwrote {oj}")


if __name__ == "__main__":
    main()
