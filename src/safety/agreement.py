"""Inter-annotator agreement for the multi-annotator DSR pass (red-team F4).

The 6-label behaviour scheme already sits at fair-to-moderate agreement
(kappa 0.436 / 0.350 / 0.345; see ``compare_annotators.py`` and
CONFOUNDS_AND_REMEDIATION CF-7). The four DSR labels add *harder* distinctions
(``safe_complete`` vs ``comply``; ``adjudication``), so their reliability cannot
be assumed — it must be measured before any geometry is computed on them. This
module is the measurement.

Two design choices mirror the rest of the project:

  * **Character-level comparison.** DSR labels are non-exclusive (a sentence may
    be both ``spec_citation`` and ``adjudication``), and each judge splits the
    chain into its own sentences, so there is no shared sentence index to align
    on. We project every judge's spans onto the chain's characters via the
    occurrence-aware ``src.text_offsets.find_sentence_offset`` (the CF-13 fix)
    and compute agreement *per label* over characters — exactly the metric
    ``compare_annotators.py`` uses for the behaviour scheme, lifted to the
    multi-label case.
  * **Per-label gates.** F4's pre-registered thresholds (kappa < 0.4 → that
    label's geometry is uninterpretable; 0.4–0.6 → must replicate across
    annotators; >= 0.6 → citable) are encoded in :func:`gate_for_kappa` so the
    runner stamps each label with its gate rather than leaving the judgement to
    prose.

Pure numpy; no model or network calls.
"""

from __future__ import annotations

import itertools
from typing import Callable, Optional, Sequence

import numpy as np

# F4 pre-registered reliability gates (kappa thresholds). Landis & Koch bands.
KAPPA_DEAD = 0.40       # below: geometry on this label is uninterpretable
KAPPA_CITABLE = 0.60    # at/above: citable; between: must replicate across annotators


def gate_for_kappa(kappa: float) -> str:
    """Map a kappa to its pre-registered DSR gate (F4).

    Returns one of ``"uninterpretable"`` (< 0.40), ``"replicate"`` (0.40–0.60),
    ``"citable"`` (>= 0.60). NaN (no labelled characters) → ``"uninterpretable"``.
    """
    if kappa != kappa:  # NaN
        return "uninterpretable"
    if kappa < KAPPA_DEAD:
        return "uninterpretable"
    if kappa < KAPPA_CITABLE:
        return "replicate"
    return "citable"


# ── Kappa primitives ──────────────────────────────────────────────────────────

def kappa_from_confusion(cm: np.ndarray) -> float:
    """Cohen's kappa from a square confusion matrix (verbatim with
    ``compare_annotators.kappa_from_cm`` so the two reports agree)."""
    cm = np.asarray(cm, dtype=float)
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = np.trace(cm) / n
    pe = ((cm.sum(0) / n) * (cm.sum(1) / n)).sum()
    return float((po - pe) / (1 - pe)) if (1 - pe) > 1e-9 else float("nan")


def cohen_kappa_binary(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's kappa between two binary (0/1) label arrays of equal length."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    if a.shape != b.shape:
        raise ValueError(f"length mismatch: {a.shape} vs {b.shape}")
    cm = np.zeros((2, 2), dtype=np.int64)
    np.add.at(cm, (a, b), 1)
    return kappa_from_confusion(cm)


def fleiss_kappa(table: np.ndarray) -> float:
    """Fleiss' kappa for N items rated by a fixed number of raters into k
    categories. ``table`` is (N, k) of per-item category counts; every row must
    sum to the same number of raters. Returns NaN if undefined.
    """
    table = np.asarray(table, dtype=float)
    if table.ndim != 2:
        raise ValueError("table must be (N_items, k_categories)")
    n_items = table.shape[0]
    if n_items == 0:
        return float("nan")
    raters = table.sum(axis=1)
    if not np.allclose(raters, raters[0]) or raters[0] < 2:
        # Unequal rater counts (e.g. a judge skipped an item) or <2 raters:
        # Fleiss is undefined. Caller should fall back to pairwise Cohen.
        return float("nan")
    n_raters = raters[0]
    p_j = table.sum(axis=0) / (n_items * n_raters)        # category marginals
    P_i = (np.square(table).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_e = float(np.square(p_j).sum())
    if (1 - P_e) <= 1e-9:
        return float("nan")
    return float((P_bar - P_e) / (1 - P_e))


# ── Span → character projection (multi-label) ────────────────────────────────

def char_label_array(
    chain_text: str,
    annotations: Sequence[dict],
    label: str,
    *,
    locate_fn: Optional[Callable[[str, str], Optional[int]]] = None,
) -> np.ndarray:
    """Binary per-character array: 1 where some span carrying *label* covers the
    character, else 0.

    ``annotations`` are DSR spans ``{"text": str, "dsr_labels": [str, ...]}``.
    The locator defaults to the occurrence-aware ``find_sentence_offset`` so a
    sentence that recurs verbatim is not collapsed onto its first occurrence
    (CF-13).
    """
    if locate_fn is None:
        from src.text_offsets import find_sentence_offset as locate_fn  # type: ignore
    arr = np.zeros(len(chain_text), dtype=np.int8)
    for a in annotations:
        if label not in a.get("dsr_labels", []):
            continue
        txt = a.get("text", "")
        if not txt:
            continue
        off = locate_fn(chain_text, txt)
        if off is None:
            continue
        arr[off:off + len(txt)] = 1
    return arr


def span_f1(
    chain_text: str,
    ref_annotations: Sequence[dict],
    hyp_annotations: Sequence[dict],
    label: str,
    *,
    locate_fn: Optional[Callable[[str, str], Optional[int]]] = None,
) -> float:
    """Character-level F1 for *label* between two annotators (F4 asked for span-F1
    alongside kappa, since kappa is harsh on boundary disagreements). Symmetric
    up to the ref/hyp swap of precision and recall; F1 itself is symmetric."""
    ref = char_label_array(chain_text, ref_annotations, label, locate_fn=locate_fn)
    hyp = char_label_array(chain_text, hyp_annotations, label, locate_fn=locate_fn)
    tp = float(np.sum((ref == 1) & (hyp == 1)))
    fp = float(np.sum((ref == 0) & (hyp == 1)))
    fn = float(np.sum((ref == 1) & (hyp == 0)))
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


# ── DSR multi-annotator agreement ────────────────────────────────────────────

def dsr_label_agreement(
    per_chain_judge_spans: Sequence[dict],
    labels: Sequence[str],
    *,
    locate_fn: Optional[Callable[[str, str], Optional[int]]] = None,
) -> dict:
    """Per-label agreement across >= 2 judges, accumulated over characters of all
    chains.

    ``per_chain_judge_spans`` is a list of records, one per chain::

        {"chain": <chain text>, "judges": {judge_name: [DSR span, ...], ...}}

    Returns ``{label: {"pairwise": {pair: cohen_kappa},
                       "fleiss": kappa_or_nan, "span_f1": {pair: f1},
                       "prevalence": {judge: fraction_chars_positive},
                       "kappa": headline, "gate": gate, "n_chars": int}}``
    where the headline ``kappa`` is Fleiss when defined (equal judges on every
    chain) else the mean pairwise Cohen.
    """
    if not per_chain_judge_spans:
        return {lab: {"kappa": float("nan"), "gate": "uninterpretable",
                      "pairwise": {}, "fleiss": float("nan"),
                      "span_f1": {}, "prevalence": {}, "n_chars": 0}
                for lab in labels}

    judge_names = sorted({j for rec in per_chain_judge_spans for j in rec["judges"]})
    out: dict = {}
    for label in labels:
        # Concatenate per-character binary arrays across all chains, per judge.
        per_judge_chars: dict[str, list] = {j: [] for j in judge_names}
        present = True
        for rec in per_chain_judge_spans:
            ct = rec["chain"]
            for j in judge_names:
                spans = rec["judges"].get(j)
                if spans is None:
                    present = False
                    per_judge_chars[j].append(np.zeros(len(ct), dtype=np.int8))
                else:
                    per_judge_chars[j].append(
                        char_label_array(ct, spans, label, locate_fn=locate_fn))
        cat = {j: (np.concatenate(v) if v else np.zeros(0, dtype=np.int8))
               for j, v in per_judge_chars.items()}
        n_chars = int(len(next(iter(cat.values()))) if cat else 0)

        pairwise = {}
        f1s = {}
        for a, b in itertools.combinations(judge_names, 2):
            pairwise[f"{a}|{b}"] = round(cohen_kappa_binary(cat[a], cat[b]), 4)
        # span-F1 averaged per pair over chains (boundary-tolerant companion)
        for a, b in itertools.combinations(judge_names, 2):
            vals = [span_f1(rec["chain"], rec["judges"].get(a, []),
                            rec["judges"].get(b, []), label, locate_fn=locate_fn)
                    for rec in per_chain_judge_spans
                    if a in rec["judges"] and b in rec["judges"]]
            f1s[f"{a}|{b}"] = round(float(np.mean(vals)), 4) if vals else float("nan")

        # Fleiss over characters (only when every judge labelled every chain).
        fleiss = float("nan")
        if present and len(judge_names) >= 2 and n_chars > 0:
            J = len(judge_names)
            stacked = np.vstack([cat[j] for j in judge_names])  # (J, n_chars)
            pos = stacked.sum(axis=0)                            # per-char positives
            tbl = np.stack([J - pos, pos], axis=1)              # (n_chars, 2)
            fleiss = fleiss_kappa(tbl)

        prevalence = {j: round(float(cat[j].mean()), 4) if n_chars else 0.0
                      for j in judge_names}

        if fleiss == fleiss:  # not NaN
            headline = fleiss
        else:
            vals = [v for v in pairwise.values() if v == v]
            headline = float(np.mean(vals)) if vals else float("nan")

        out[label] = {
            "kappa": round(headline, 4) if headline == headline else float("nan"),
            "gate": gate_for_kappa(headline),
            "pairwise": pairwise,
            "fleiss": round(fleiss, 4) if fleiss == fleiss else float("nan"),
            "span_f1": f1s,
            "prevalence": prevalence,
            "n_chars": n_chars,
        }
    return out


__all__ = [
    "KAPPA_DEAD", "KAPPA_CITABLE", "gate_for_kappa",
    "kappa_from_confusion", "cohen_kappa_binary", "fleiss_kappa",
    "char_label_array", "span_f1", "dsr_label_agreement",
]
