#!/usr/bin/env python3
"""Cross-annotator comparison: inter-annotator agreement + manifold replication.

(1) Agreement: character-level label agreement + Cohen's kappa across the three
    annotators (Sonnet, Qwen3-235B, Nova-Pro), pairwise, on the 6-label taxonomy
    and on the 4 target behaviours.
(2) Replication: per-annotator intrinsic dim (corr-dim), curvature (geodesic), and
    PR-trough layer at matched layers, from each annotator's robustness JSON.

Outputs: results/robustness/cross_annotator_comparison.{json,md}
"""
import json, itertools, sys
from pathlib import Path
import numpy as np

ANNOTATORS = {
    "Sonnet-4.5":  ("R1-1.5B",              "data/annotated_R1-1.5B.json"),
    "Qwen3-235B":  ("R1-1.5B__qwen3-235b",  "data/annotated_R1-1.5B__qwen3-235b.json"),
    "Nova-Pro":    ("R1-1.5B__nova-pro",    "data/annotated_R1-1.5B__nova-pro.json"),
}
LABELS = ["O", "initializing", "deduction", "adding-knowledge", "example-testing",
          "uncertainty-estimation", "backtracking"]
TARGET = {"backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"}
L2I = {l: i for i, l in enumerate(LABELS)}
OUT = Path("results/robustness"); OUT.mkdir(parents=True, exist_ok=True)

from src.text_offsets import find_sentence_offset as _off  # single source of truth

def char_labels(chain):
    """Per-character label-code array for one chain (0=O)."""
    ct = chain.get("chain", ""); arr = np.zeros(len(ct), dtype=np.int8)
    for a in chain.get("annotations", []):
        lb = a.get("label", ""); txt = a.get("text", "")
        if lb not in L2I: continue
        o = _off(ct, txt)
        if o is None: continue
        arr[o:o + len(txt)] = L2I[lb]
    return arr

def kappa_from_cm(cm):
    n = cm.sum()
    if n == 0: return float("nan")
    po = np.trace(cm) / n
    pe = ((cm.sum(0) / n) * (cm.sum(1) / n)).sum()
    return float((po - pe) / (1 - pe)) if (1 - pe) > 1e-9 else float("nan")

def main():
    # load all three, index by task_id
    data = {}
    for name, (short, path) in ANNOTATORS.items():
        if not Path(path).exists():
            print(f"WARN: {path} missing; skipping {name}"); continue
        data[name] = {c["task_id"]: c for c in json.load(open(path))}
    names = list(data)
    common = set.intersection(*[set(d) for d in data.values()]) if names else set()
    print(f"annotators: {names}; common chains: {len(common)}")

    # pairwise confusion matrices (6-label) accumulated over characters
    K = len(LABELS)
    cms = {pair: np.zeros((K, K), dtype=np.int64) for pair in itertools.combinations(names, 2)}
    for tid in common:
        cl = {nm: char_labels(data[nm][tid]) for nm in names}
        m = min(len(a) for a in cl.values())
        for pair in cms:
            a, b = cl[pair[0]][:m], cl[pair[1]][:m]
            np.add.at(cms[pair], (a, b), 1)

    res = {"n_common_chains": len(common), "agreement": {}, "label_distributions": {}}
    # label distributions per annotator
    for nm in names:
        cc = {}
        for tid in common:
            for an in data[nm][tid].get("annotations", []):
                cc[an["label"]] = cc.get(an["label"], 0) + 1
        tot = sum(cc.values()) or 1
        res["label_distributions"][nm] = {k: round(100 * v / tot, 1) for k, v in sorted(cc.items(), key=lambda x: -x[1])}

    tgt_idx = [L2I[l] for l in TARGET]
    for pair, cm in cms.items():
        key = " vs ".join(pair)
        labeled = cm.copy(); labeled[0, 0] = 0  # ignore O-O for "labeled" agreement
        n_lab = labeled.sum()
        # collapse to target-vs-other for the target kappa
        tcm = np.zeros((2, 2), dtype=np.int64)
        for i in range(K):
            for j in range(K):
                ti = 1 if i in tgt_idx else 0; tj = 1 if j in tgt_idx else 0
                tcm[ti, tj] += cm[i, j]
        res["agreement"][key] = {
            "kappa_6label": round(kappa_from_cm(cm), 3),
            "kappa_target_vs_other": round(kappa_from_cm(tcm), 3),
            "char_agreement_overall": round(float(np.trace(cm) / cm.sum()), 3),
            "char_agreement_labeled": round(float(np.trace(labeled) / n_lab), 3) if n_lab else None,
        }

    # ---- manifold replication ----
    rep = {}
    for nm, (short, _) in ANNOTATORS.items():
        if nm not in names: continue
        rp = Path(f"results/robustness/{short}/geometry_robustness.json")
        lp = Path(f"results/pca/{short}/layer_profiles.json")
        entry = {}
        if rp.exists():
            g = json.load(open(rp))
            for b, r in g.items():
                entry[b] = {"cdim": round(r["keystone_cdim"]["full"], 2),
                            "geo": round(r["keystone_geodesic"]["full"], 2),
                            "chainstrat_cdim": round(r["keystone_cdim"]["chain_strat"]["mean"], 2)}
        if lp.exists():
            prof = json.load(open(lp))
            for b in prof:
                if prof[b].get("participation_ratio"):
                    tl = prof[b]["layers"][int(np.argmin(prof[b]["participation_ratio"]))]
                    entry.setdefault(b, {})["pr_trough_layer"] = int(tl)
        rep[nm] = entry
    res["manifold_replication"] = rep

    json.dump(res, open(OUT / "cross_annotator_comparison.json", "w"), indent=2)

    # markdown
    L = ["# Cross-annotator comparison", "",
         f"Common chains: {len(common)}", "", "## Label distribution (% of spans, common chains)", "",
         "| Label | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    alllabels = sorted({k for nm in names for k in res["label_distributions"][nm]})
    for lab in alllabels:
        L.append(f"| {lab} | " + " | ".join(f"{res['label_distributions'][nm].get(lab,0)}%" for nm in names) + " |")
    L += ["", "## Inter-annotator agreement (character-level)", "",
          "| Pair | kappa (6-label) | kappa (target vs other) | agree (labeled) | agree (overall) |",
          "|---|---|---|---|---|"]
    for pair, a in res["agreement"].items():
        L.append(f"| {pair} | {a['kappa_6label']} | {a['kappa_target_vs_other']} | {a['char_agreement_labeled']} | {a['char_agreement_overall']} |")
    L += ["", "## Manifold replication (intrinsic dim / curvature at matched trough layers)", "",
          "| Behaviour | " + " | ".join(f"{nm} cdim" for nm in names) + " | " + " | ".join(f"{nm} geo" for nm in names) + " |",
          "|---|" + "---|" * (2 * len(names))]
    for b in ["backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"]:
        row = [b]
        for nm in names: row.append(str(rep.get(nm, {}).get(b, {}).get("cdim", "—")))
        for nm in names: row.append(str(rep.get(nm, {}).get(b, {}).get("geo", "—")))
        L.append("| " + " | ".join(row) + " |")
    L += ["", "**Read:** if cdim and geo are similar across annotators despite differing label distributions,",
          "the manifold result is annotator-robust (the headline external-robustness claim)."]
    (OUT / "cross_annotator_comparison.md").write_text("\n".join(L))
    print("Saved -> results/robustness/cross_annotator_comparison.{json,md}")

if __name__ == "__main__":
    main()
