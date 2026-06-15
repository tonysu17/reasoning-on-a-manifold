#!/usr/bin/env python3
"""
15_predict_gate.py — Rung-0 / Rung-1 correctness gate (local, CPU, no API).

For each layer, on the difficulty-labelled chains:
  * Rung 0  raw-trajectory curvature           (predictor-free; SSP-style)
  * persistence  ||x_{t+1}-x_t|| step features  (the trivial-predictor baseline)
  * length / gap / truncation                   (CF-8 control the contrast must beat)
  * Rung 1  learned-predictor RESIDUAL geometry (delta-mode ridge, chain-grouped)
predicting chain-level correctness via a chain-grouped logistic probe (ROC-AUC),
then two nulls on the Rung-1 signal: label permutation (within difficulty strata)
and within-chain step-order shuffle.

Gate logic: Rung 1 is interesting only if its AUC beats Rung 0, persistence, and
length, AND both nulls reject. All claims are functional/relative and chain-grouped
(CF-2); geometry is read off the residual only (no anti-collapse contaminant).

  python3 15_predict_gate.py --labels data/correctness_R1-1.5B_pilot.json --layers 14,27
  python3 15_predict_gate.py --smoke 80          # synthetic labels, validates plumbing
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import backup_existing, provenance  # noqa: E402
from src.predict.trajectory_dataset import build_step_datasets, make_supervised_pairs  # noqa: E402
from src.predict.predictor import RidgeConfig, oof_residuals, chain_residual_features  # noqa: E402
from src.predict.evaluation import (  # noqa: E402
    align_labels, grouped_auc, raw_curvature_features, raw_step_features,
    length_features, residual_auc_statistic,
)
from src.predict.nulls_predict import label_permutation_null, step_shuffle_null  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _label_map(labels: dict, min_confidence: str | None) -> dict:
    """chain_id -> bool, dropping uncertain (None) and below-confidence records."""
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


def _auc_for(features, chain_ids, label_map):
    labels = align_labels(chain_ids, label_map)
    return grouped_auc(features, labels, chain_ids)


def _gate_layer(chains, activations_dir, layer, label_map, strata_map, args) -> dict:
    ds_all = build_step_datasets(chains, activations_dir, layer)
    datasets = [d for d in ds_all if d.chain_id in label_map]
    n_pos = sum(1 for d in datasets if label_map[d.chain_id])
    res_blocks: dict = {"layer": layer, "n_labelled_chains": len(datasets),
                        "n_correct": n_pos, "n_incorrect": len(datasets) - n_pos}
    if len(datasets) < 10 or n_pos < 3 or (len(datasets) - n_pos) < 3:
        res_blocks["status"] = "insufficient labelled chains (need >=10, >=3/class)"
        return res_blocks

    cfg = RidgeConfig(target="delta")
    # Train the label-agnostic predictor on ALL chains (chain-grouped OOF);
    # correctness AUC is evaluated only on the labelled subset (align_labels ->
    # NaN for unlabelled, dropped by grouped_auc). Restricting predictor training
    # to the labelled subset underpowers it badly (see residual_geometry_sweep).
    pairs = make_supervised_pairs(ds_all, max_gap=args.max_gap)
    res = oof_residuals(pairs, cfg)
    rn = float(np.linalg.norm(res.residuals, axis=1).mean())
    step = float(np.linalg.norm(pairs["X_next"] - pairs["X_hist"], axis=1).mean())
    res_blocks["residual_to_persistence_ratio"] = rn / step if step else float("nan")

    rf, rcids, _ = chain_residual_features(res)
    cf, ccids, _ = raw_curvature_features(ds_all)
    sf, scids, _ = raw_step_features(ds_all)
    lf, lcids, _ = length_features(ds_all)
    res_blocks["auc"] = {
        "rung1_residual": _auc_for(rf, rcids, label_map),
        "rung0_curvature": _auc_for(cf, ccids, label_map),
        "persistence_step": _auc_for(sf, scids, label_map),
        "length_gap_trunc": _auc_for(lf, lcids, label_map),
    }

    # nulls on the Rung-1 residual signal
    strata = (np.array([strata_map.get(c) for c in rcids], dtype=object)
              if strata_map else None)
    labels_aligned = align_labels(rcids, label_map)
    lp = label_permutation_null(rf, rcids, labels_aligned, strata=strata,
                                n_resamples=args.label_resamples, seed=42)
    def _stat(dsets):
        return residual_auc_statistic(dsets, label_map, ridge_cfg=cfg, max_gap=args.max_gap)
    ss = step_shuffle_null(ds_all, _stat, n_resamples=args.shuffle_resamples, seed=42)
    res_blocks["nulls"] = {
        "label_permutation": {"real": lp.real_value, "null_mean": lp.null_mean,
                              "p_value": lp.p_value, "n": lp.n_resamples},
        "step_shuffle": {"real": ss.real_value, "null_mean": ss.null_mean,
                         "p_value": ss.p_value, "n": ss.n_resamples},
    }

    a = res_blocks["auc"]
    r1 = a["rung1_residual"]["auc_oof"]
    beats = (np.isfinite(r1)
             and r1 > max(a["rung0_curvature"]["auc_oof"],
                          a["persistence_step"]["auc_oof"],
                          a["length_gap_trunc"]["auc_oof"])
             and lp.p_value < 0.05 and ss.p_value < 0.05)
    res_blocks["status"] = "rung1_beats_baselines_and_nulls" if beats else "no_rung1_advantage"
    return res_blocks


def _write_md(path: Path, results: list[dict], smoke: bool) -> None:
    lines = ["# Predictive-geometry gate", ""]
    if smoke:
        lines.append("**SMOKE RUN — synthetic random labels; AUCs should be ~0.5.**\n")
    for r in results:
        lines.append(f"## layer {r['layer']}")
        if "auc" not in r:
            lines.append(f"- {r.get('status')}\n")
            continue
        lines.append(f"- labelled chains: {r['n_labelled_chains']} "
                     f"(correct {r['n_correct']} / incorrect {r['n_incorrect']})")
        lines.append(f"- residual/persistence ratio: {r['residual_to_persistence_ratio']:.3f} "
                     f"(>1 ⇒ learned predictor worse than repeating the step)")
        lines.append("")
        lines.append("| feature set | AUC (oof) | fold mean±std | stable | n_pos/n |")
        lines.append("|---|---|---|---|---|")
        for name, a in r["auc"].items():
            lines.append(f"| {name} | {a['auc_oof']:.3f} | "
                         f"{a['auc_fold_mean']:.3f}±{a['auc_fold_std']:.3f} | "
                         f"{a['stable']} | {a['n_pos']}/{a['n']} |")
        nl = r["nulls"]
        lines.append("")
        lines.append(f"- label-permutation null: real {nl['label_permutation']['real']:.3f} "
                     f"vs null {nl['label_permutation']['null_mean']:.3f}, "
                     f"p={nl['label_permutation']['p_value']:.3f}")
        lines.append(f"- step-shuffle null: real {nl['step_shuffle']['real']:.3f} "
                     f"vs null {nl['step_shuffle']['null_mean']:.3f}, "
                     f"p={nl['step_shuffle']['p_value']:.3f}")
        lines.append(f"- **verdict: {r['status']}**\n")
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", default="data/activations/R1-1.5B")
    ap.add_argument("--chains", default="data/annotated_R1-1.5B.json")
    ap.add_argument("--tasks", default="data/tasks_final.json")
    ap.add_argument("--labels", default="data/correctness_R1-1.5B_pilot.json")
    ap.add_argument("--layers", default="14,27")
    ap.add_argument("--max-gap", type=int, default=None,
                    help="restrict to pairs whose original-span gap <= this (e.g. 1)")
    ap.add_argument("--min-confidence", choices=["low", "medium", "high"], default=None)
    ap.add_argument("--label-resamples", type=int, default=500)
    ap.add_argument("--shuffle-resamples", type=int, default=100)
    ap.add_argument("--out", default="results/predict/R1-1.5B")
    ap.add_argument("--smoke", type=int, default=0,
                    help="N: synthesize random labels for N chains (validates plumbing)")
    args = ap.parse_args()

    chains = json.load(open(args.chains))
    tasks = json.load(open(args.tasks))
    difficulty = {t["id"]: t.get("difficulty") for t in tasks}

    if args.smoke:
        rng = random.Random(42)
        sub = chains[: args.smoke]
        label_map = {c["task_id"]: bool(rng.getrandbits(1)) for c in sub}
        smoke = True
    else:
        from src.predict.labels import load_correctness_labels
        labels = load_correctness_labels(args.labels)
        label_map = _label_map(labels, args.min_confidence)
        smoke = False
    strata_map = {cid: difficulty.get(cid) for cid in label_map}
    print(f"{'SMOKE: ' if smoke else ''}{len(label_map)} usable labels "
          f"({sum(label_map.values())} correct)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for layer in [int(x) for x in args.layers.split(",")]:
        print(f"\n=== layer {layer} ===")
        r = _gate_layer(chains, args.activations, layer, label_map, strata_map, args)
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "auc"}, indent=2, default=str))
        if "auc" in r:
            for name, a in r["auc"].items():
                print(f"  {name:18s} AUC={a['auc_oof']:.3f} (n_pos {a['n_pos']}/{a['n']})")

    tag = "smoke" if smoke else "pilot"
    out_json = out_dir / f"gate_{tag}.json"
    backup_existing(out_json)
    payload = {"results": results, "smoke": smoke,
               "provenance": provenance(args, inputs=[args.chains] +
                                        ([] if smoke else [args.labels]))}
    json.dump(payload, open(out_json, "w"), indent=2, default=str)
    _write_md(out_dir / f"gate_{tag}.md", results, smoke)
    print(f"\nwrote {out_json} and gate_{tag}.md")


if __name__ == "__main__":
    main()
