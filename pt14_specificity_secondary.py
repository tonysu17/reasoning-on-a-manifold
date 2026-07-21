#!/usr/bin/env python3
"""pt14: behaviour-specificity null on a SECOND annotator (ledger §F3).

The behaviour-specificity verdict (2/4: backtracking + uncertainty concentrate
their variance beyond a within-chain label-permutation null; example-testing +
adding-knowledge do not) is still single-annotator (Sonnet). R2.2 replicated
intrinsic dimension + the curvature negative 3-way, but never the specificity
null. Nova-span activations exist locally for all four behaviours at L12/L16, so
the Nova arm can run now (CPU); Qwen has no local span activations and needs a
re-extraction (flagged, not run).

Runs the EXACT primary instrument (``src.nulls.full_null_hierarchy`` with
``top_k_variance_ratio``, chain-stratified within-chain label permutation, the
CF-13 row dedup, n_resamples=500, seed 42, tail='upper') for Sonnet and Nova at
matched layers, so "specific" means the same thing for both. A behaviour is
specific iff its chain-stratified p < 0.05.

Run:  python3 pt14_specificity_secondary.py
Out:  results/robustness/specificity_secondary_annotator.json
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from src.nulls import full_null_hierarchy, top_k_variance_ratio
from src.row_provenance import chain_ids_for, dedup_rows, duplicate_fraction

BEHAVIOURS = ["backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"]
LAYERS = [12, 16]
K = 10
N_RESAMPLES = 500
SEED = 42
ALPHA = 0.05

ANNOTATORS = {
    "Sonnet": ("data/activations/R1-1.5B", "data/annotated_R1-1.5B.json"),
    "Nova":   ("data/activations/R1-1.5B-novaspans", "data/annotated_R1-1.5B__nova-pro.json"),
}
OUT = Path("results/robustness/specificity_secondary_annotator.json")


def build_layer_matrix(act_dir: Path, cids: dict, layer: int):
    parts_X, parts_c, parts_l = [], [], []
    for b in BEHAVIOURS:
        X = np.load(act_dir / f"{b}_layer{layer}.npy")
        c = np.asarray(cids[b])
        assert len(c) == X.shape[0], f"{b} L{layer}: {len(c)} cids vs {X.shape[0]} rows"
        parts_X.append(X)
        parts_c.append(c)
        parts_l.append(np.array([b] * X.shape[0]))
    X_all = np.vstack(parts_X)
    chain_all = np.concatenate(parts_c)
    label_all = np.concatenate(parts_l)
    dup = duplicate_fraction(X_all)
    if dup > 0:
        X_all, chain_all, label_all = dedup_rows(X_all, chain_all, label_all)
    return X_all, chain_all, label_all, dup


def main():
    report = {"instrument": "chain-stratified within-chain label-permutation null on top-10 variance ratio",
              "k": K, "n_resamples": N_RESAMPLES, "seed": SEED, "alpha": ALPHA,
              "annotators": {}}
    for name, (ad, ann) in ANNOTATORS.items():
        ad, ann = Path(ad), Path(ann)
        cids = chain_ids_for(ad, ann, BEHAVIOURS)
        report["annotators"][name] = {}
        for L in LAYERS:
            X, chains, labels, dup = build_layer_matrix(ad, cids, L)
            cell = {"dup_frac": round(dup, 4), "n_total": int(X.shape[0]), "behaviours": {}}
            for b in BEHAVIOURS:
                nh = full_null_hierarchy(
                    X, chains, labels, b,
                    statistic_fn=top_k_variance_ratio, statistic_name="top10_var_ratio",
                    n_resamples=N_RESAMPLES, random_state=SEED, tail="upper")
                cs = asdict(nh["chain_strat"])
                p = cs.get("p_value")
                cell["behaviours"][b] = {
                    "real": round(cs.get("real_value", float("nan")), 4),
                    "null_mean": round(cs.get("null_mean", float("nan")), 4),
                    "p_value": p,
                    "specific": bool(p is not None and p < ALPHA),
                }
            report["annotators"][name][f"L{L}"] = cell
            spec = [b for b in BEHAVIOURS if cell["behaviours"][b]["specific"]]
            print(f"{name} L{L}: specific = {spec}")

    # cross-annotator agreement on the specificity call, per behaviour-layer
    agree = {}
    for L in LAYERS:
        for b in BEHAVIOURS:
            s = report["annotators"]["Sonnet"][f"L{L}"]["behaviours"][b]["specific"]
            n = report["annotators"]["Nova"][f"L{L}"]["behaviours"][b]["specific"]
            agree[f"{b}_L{L}"] = {"Sonnet": s, "Nova": n, "agree": s == n}
    report["cross_annotator"] = agree
    n_agree = sum(1 for v in agree.values() if v["agree"])
    report["summary"] = {
        "cells": len(agree), "agree": n_agree,
        "note": "Qwen arm needs span re-extraction (no local qwenspans); Nova is the 2nd annotator here.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)
    print(f"\ncross-annotator specificity agreement: {n_agree}/{len(agree)} cells")
    print("written", OUT)


if __name__ == "__main__":
    main()
