"""Tests for src/safety/cot_extraction.py — the F1 CoT-spanning pooling core.

The pooling is exercised torch-free via a char-level offset mapping and a fake
``pool_fn``, so the span→token-position→vector logic is checked without a GPU.
This is the keystone fix: pool per DSR span of the generated chain, not the
prompt token.
"""

from __future__ import annotations

import numpy as np

from src.safety.cot_extraction import pool_dsr_spans, split_by_stim_label


def _char_offsets(text):
    """One 'token' per character → offsets [(0,1),(1,2),...]."""
    return [(i, i + 1) for i in range(len(text))]


def _count_pool(layer, positions):
    """Fake pooling: return the number of pooled positions as a 1-vector."""
    return np.array([float(len(positions))], dtype=np.float32)


def _record(labels):
    prompt = "Q "
    chain = "Wait. The policy forbids it."
    return {
        "task_id": "t1",
        "prompt": prompt,
        "chain": chain,
        "label": "harmful",
        "dsr_consensus": {"spans": [
            {"text": "The policy forbids it.", "dsr_labels": labels, "decision_type": None},
        ]},
    }


def test_pool_single_label_span():
    rec = _record(["spec_citation"])
    full = rec["prompt"] + rec["chain"]
    acc, rows = pool_dsr_spans(rec, _char_offsets(full), _count_pool, [0],
                               seq_len=len(full))
    assert len(acc["spec_citation"][0]) == 1
    # 1 preceding + 10 execution tokens pooled
    assert acc["spec_citation"][0][0][0] == 11.0
    assert rows["spec_citation"][0]["stim_label"] == "harmful"
    assert rows["spec_citation"][0]["chain_id"] == "t1"
    # untouched labels stay empty
    assert acc["adjudication"][0] == []


def test_pool_multi_label_span_contributes_to_each():
    rec = _record(["spec_citation", "adjudication"])
    full = rec["prompt"] + rec["chain"]
    acc, rows = pool_dsr_spans(rec, _char_offsets(full), _count_pool, [0],
                               seq_len=len(full))
    assert len(acc["spec_citation"][0]) == 1
    assert len(acc["adjudication"][0]) == 1


def test_pool_skips_span_not_in_chain():
    rec = _record(["spec_citation"])
    rec["dsr_consensus"]["spans"] = [
        {"text": "this text is absent from the chain", "dsr_labels": ["spec_citation"]}]
    full = rec["prompt"] + rec["chain"]
    acc, _ = pool_dsr_spans(rec, _char_offsets(full), _count_pool, [0], seq_len=len(full))
    assert acc["spec_citation"][0] == []


def test_split_by_stim_label():
    mat = np.array([[1.0], [2.0], [3.0]])
    meta = [{"stim_label": "harmful"}, {"stim_label": "benign"}, {"stim_label": "harmful"}]
    h, l = split_by_stim_label(mat, meta)
    assert h.shape[0] == 2 and l.shape[0] == 1
