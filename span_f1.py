#!/usr/bin/env python3
"""Span-level F1 between the three annotators (Sonnet, Qwen3-235B, Nova-Pro).

Complements the character-level Cohen's kappa of compare_annotators.py. Char-level
kappa is harsh: a one-token boundary disagreement on an otherwise-agreed span counts
as disagreement at every off-by-one character. Span-F1 scores whole labelled spans,
so it separates "did they find the same behaviour in the same place" (overlap match)
from "did they agree on the exact boundary" (exact match).

A span is one labelled annotation, located occurrence-aware (same matcher as Phase 4)
as a [start, end) character range with a label. For an ordered pair we greedily match
each span in A to an unmatched same-label span in B under a criterion, and report
F1 = 2*TP / (|A| + |B|) accumulated over the common chains. F1 is symmetric, so the
pair is reported once. Three criteria: exact boundary, IoU >= 0.5, any overlap.

Outputs: results/robustness/span_f1.{json,md}  (does not touch the Gate-0.1 dirs).
Read-only on the annotation JSONs; safe to run alongside a Phase-5 regeneration.
"""
import json, itertools, sys
from pathlib import Path

sys.path.insert(0, ".")
from src.text_offsets import locate_annotation_offsets as _locate

ANNOTATORS = {
    "Sonnet-4.5": "data/annotated_R1-1.5B.json",
    "Qwen3-235B": "data/annotated_R1-1.5B__qwen3-235b.json",
    "Nova-Pro":   "data/annotated_R1-1.5B__nova-pro.json",
}
LABELSET = {"initializing", "deduction", "adding-knowledge", "example-testing",
            "uncertainty-estimation", "backtracking"}
TARGET = ["backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"]
OUT = Path("results/robustness"); OUT.mkdir(parents=True, exist_ok=True)


def spans(chain):
    """Occurrence-aware list of (label, start, end) for one chain."""
    ct = chain.get("chain", ""); anns = chain.get("annotations", [])
    offs = _locate(ct, [a.get("text", "") for a in anns])
    out = []
    for a, o in zip(anns, offs):
        lb, txt = a.get("label", ""), a.get("text", "")
        if lb in LABELSET and o is not None and txt:
            out.append((lb, o, o + len(txt)))
    return out


def iou(s1, s2):
    a, b = max(s1[1], s2[1]), min(s1[2], s2[2])
    inter = max(0, b - a)
    union = (s1[2] - s1[1]) + (s2[2] - s2[1]) - inter
    return (inter / union) if union > 0 else 0.0, inter


def matches(A, B, crit):
    """Greedy same-label match count + per-label TP under a criterion."""
    usedB, tp, per = set(), 0, {}
    for sa in A:
        best_j, best_score = -1, 0.0
        for j, sb in enumerate(B):
            if j in usedB or sb[0] != sa[0]:
                continue
            ov, inter = iou(sa, sb)
            if inter <= 0:
                continue
            if crit == "exact":
                ok, score = (sa[1] == sb[1] and sa[2] == sb[2]), 1.0
            elif crit == "iou":
                ok, score = ov >= 0.5, ov
            else:  # any overlap
                ok, score = True, ov
            if ok and score > best_score:
                best_score, best_j = max(score, 1e-9), j
        if best_j >= 0:
            usedB.add(best_j); tp += 1; per[sa[0]] = per.get(sa[0], 0) + 1
    return tp, per


def main():
    data, counts = {}, {}
    for name, path in ANNOTATORS.items():
        if not Path(path).exists():
            print(f"WARN: {path} missing; skipping {name}"); continue
        data[name] = {c["task_id"]: c for c in json.load(open(path))}
    names = list(data)
    common = sorted(set.intersection(*[set(d) for d in data.values()])) if names else []
    print(f"annotators: {names}; common chains: {len(common)}")

    # pre-extract spans per annotator over the common chains
    sp = {nm: {tid: spans(data[nm][tid]) for tid in common} for nm in names}
    for nm in names:
        counts[nm] = sum(len(v) for v in sp[nm].values())

    res = {"n_common_chains": len(common), "span_counts": counts, "pairs": {}}
    CRITS = ["exact", "iou", "overlap"]
    for a, b in itertools.combinations(names, 2):
        key = f"{a} vs {b}"
        nA = sum(len(sp[a][t]) for t in common)
        nB = sum(len(sp[b][t]) for t in common)
        entry = {"n_spans": {a: nA, b: nB}, "f1": {}, "per_label_f1_iou50": {}}
        for crit in CRITS:
            tp = sum(matches(sp[a][t], sp[b][t], crit)[0] for t in common)
            entry["f1"][crit] = round(2 * tp / (nA + nB), 3) if (nA + nB) else None
        # per-label F1 at IoU>=0.5
        per_tp, cntA, cntB = {}, {}, {}
        for t in common:
            _, pl = matches(sp[a][t], sp[b][t], "iou")
            for k, v in pl.items():
                per_tp[k] = per_tp.get(k, 0) + v
            for lb, _, _ in sp[a][t]:
                cntA[lb] = cntA.get(lb, 0) + 1
            for lb, _, _ in sp[b][t]:
                cntB[lb] = cntB.get(lb, 0) + 1
        for lb in sorted(LABELSET):
            den = cntA.get(lb, 0) + cntB.get(lb, 0)
            entry["per_label_f1_iou50"][lb] = round(2 * per_tp.get(lb, 0) / den, 3) if den else None
        res["pairs"][key] = entry

    json.dump(res, open(OUT / "span_f1.json", "w"), indent=2)

    L = ["# Span-level F1 between annotators", "",
         f"Common chains: {len(common)}. A span = one labelled annotation, located",
         "occurrence-aware. F1 = 2*TP/(|A|+|B|) over greedy same-label matches; symmetric.",
         "Complements the character-level Cohen's kappa in `cross_annotator_comparison.md`.",
         "", "## Span counts (labelled spans on the common chains)", "",
         "| " + " | ".join(names) + " |", "|" + "---|" * len(names)]
    L.append("| " + " | ".join(str(counts[nm]) for nm in names) + " |")
    L += ["", "## Span-F1 by matching criterion", "",
          "| Pair | exact boundary | IoU ≥ 0.5 | any overlap |", "|---|---|---|---|"]
    for key, e in res["pairs"].items():
        f = e["f1"]
        L.append(f"| {key} | {f['exact']} | {f['iou']} | {f['overlap']} |")
    L += ["", "## Per-label span-F1 (IoU ≥ 0.5)", "",
          "| Pair | " + " | ".join(TARGET) + " |", "|---|" + "---|" * len(TARGET)]
    for key, e in res["pairs"].items():
        pl = e["per_label_f1_iou50"]
        L.append(f"| {key} | " + " | ".join(str(pl.get(lb, "—")) for lb in TARGET) + " |")
    L += ["", "**Read:** span-F1 at IoU≥0.5 should sit well above the character-level kappa,",
          "since it forgives boundary jitter and scores the agreed-upon spans; the gap between",
          "the *exact* and *overlap* columns is the size of the boundary-disagreement effect that",
          "makes the character metric harsh. Per-label F1 localises where the annotators diverge."]
    (OUT / "span_f1.md").write_text("\n".join(L))
    print("Saved -> results/robustness/span_f1.{json,md}")


if __name__ == "__main__":
    main()
