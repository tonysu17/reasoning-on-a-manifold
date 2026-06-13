"""Tests for src/safety/agreement.py — DSR inter-annotator agreement (F4).

Hand-computable kappa fixtures so a bug surfaces as a numeric disagreement, plus
the F4 gate mapping and the multi-label char-projection.
"""

from __future__ import annotations

import numpy as np

from src.safety.agreement import (
    KAPPA_CITABLE, KAPPA_DEAD, char_label_array, cohen_kappa_binary,
    dsr_label_agreement, fleiss_kappa, gate_for_kappa, kappa_from_confusion, span_f1,
)


def test_cohen_kappa_perfect_with_variance_is_one():
    a = [1, 0, 1, 0, 1, 0]
    assert cohen_kappa_binary(a, a) == 1.0


def test_cohen_kappa_independent_near_zero():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 2000)
    b = rng.integers(0, 2, 2000)
    assert abs(cohen_kappa_binary(a, b)) < 0.1


def test_cohen_kappa_partial_known_value():
    # 2x2 cm: agree 8/10, disagree 2/10, balanced marginals → kappa computable.
    a = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    b = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    k = cohen_kappa_binary(a, b)
    assert 0.55 < k < 0.65  # po=0.8, pe=0.5 → 0.6


def test_kappa_from_confusion_matches_helper():
    cm = np.array([[40, 10], [10, 40]])
    assert abs(kappa_from_confusion(cm) - 0.6) < 1e-9


def test_fleiss_perfect_agreement_is_one():
    # 2 items, 3 raters, 2 categories; both items unanimous.
    table = np.array([[3, 0], [0, 3]])
    assert abs(fleiss_kappa(table) - 1.0) < 1e-9


def test_fleiss_unequal_raters_is_nan():
    table = np.array([[3, 0], [0, 2]])  # second item only 2 raters
    assert np.isnan(fleiss_kappa(table))


def test_gate_thresholds():
    assert gate_for_kappa(0.39) == "uninterpretable"
    assert gate_for_kappa(KAPPA_DEAD) == "replicate"
    assert gate_for_kappa(0.55) == "replicate"
    assert gate_for_kappa(KAPPA_CITABLE) == "citable"
    assert gate_for_kappa(float("nan")) == "uninterpretable"


def test_char_label_array_marks_span():
    chain = "Wait. The policy forbids it. Done."
    anns = [{"text": "The policy forbids it.", "dsr_labels": ["spec_citation"]}]
    arr = char_label_array(chain, anns, "spec_citation")
    assert arr.sum() == len("The policy forbids it.")
    # the other label is absent
    assert char_label_array(chain, anns, "adjudication").sum() == 0


def test_span_f1_identical_is_one_disjoint_is_zero():
    chain = "The policy forbids it. I will comply anyway."
    a = [{"text": "The policy forbids it.", "dsr_labels": ["spec_citation"]}]
    assert span_f1(chain, a, a, "spec_citation") == 1.0
    b = [{"text": "I will comply anyway.", "dsr_labels": ["spec_citation"]}]
    assert span_f1(chain, a, b, "spec_citation") == 0.0


def test_dsr_label_agreement_consensus_label_is_citable():
    chain = "The policy says no. So I refuse now."
    span = {"text": "The policy says no.", "dsr_labels": ["spec_citation"]}
    recs = [{"chain": chain, "judges": {"A": [span], "B": [span], "C": [span]}}]
    out = dsr_label_agreement(recs, ["spec_citation", "adjudication"])
    assert out["spec_citation"]["gate"] == "citable"
    assert out["spec_citation"]["kappa"] == 1.0
    # adjudication never marked → degenerate (no positives) → uninterpretable
    assert out["adjudication"]["gate"] == "uninterpretable"


def test_dsr_label_agreement_divergent_label_lower_kappa():
    chain = "The policy says no. So I refuse now. Maybe though it is fine."
    s_spec = {"text": "The policy says no.", "dsr_labels": ["adjudication"]}
    s_other = {"text": "Maybe though it is fine.", "dsr_labels": ["adjudication"]}
    recs = [{"chain": chain, "judges": {
        "A": [s_spec], "B": [s_other], "C": [s_spec]}}]
    out = dsr_label_agreement(recs, ["adjudication"])
    # A and C agree, B differs → not perfect
    assert out["adjudication"]["kappa"] < 1.0
    assert "A|B" in out["adjudication"]["pairwise"]
