"""Tests for src/safety/annotate.py — multi-annotator DSR pass (F4).

All judge calls are mocked; no network. Covers JSON parsing + repair, retry/fail
accounting, character-level consensus aggregation (majority + ties), and the
resumable batch runner.
"""

from __future__ import annotations

import json

from src.safety.annotate import (
    Judge, aggregate_dsr, agreement_report, annotate_chain_dsr,
    annotate_chains_dsr, parse_dsr_response,
)

SPAN = {"text": "The policy says no.", "dsr_labels": ["spec_citation"], "decision_type": None}


# ── parsing ───────────────────────────────────────────────────────────────────

def test_parse_plain_json_array():
    out = parse_dsr_response(json.dumps([SPAN]))
    assert len(out) == 1
    assert out[0]["dsr_labels"] == ["spec_citation"]


def test_parse_code_fence():
    out = parse_dsr_response("```json\n" + json.dumps([SPAN]) + "\n```")
    assert len(out) == 1


def test_parse_prose_wrapped_array():
    out = parse_dsr_response("Here are the labels:\n" + json.dumps([SPAN]) + "\nDone.")
    assert len(out) == 1


def test_parse_malformed_returns_empty():
    assert parse_dsr_response("not json at all") == []
    assert parse_dsr_response("") == []


def test_parse_drops_unknown_label_keeps_valid():
    item = {"text": "x", "dsr_labels": ["bogus", "spec_citation"], "decision_type": None}
    out = parse_dsr_response(json.dumps([item]))
    assert out[0]["dsr_labels"] == ["spec_citation"]


def test_parse_decision_type_adds_decision_label():
    item = {"text": "I refuse.", "dsr_labels": [], "decision_type": "refuse"}
    out = parse_dsr_response(json.dumps([item]))
    assert "decision" in out[0]["dsr_labels"]
    assert out[0]["decision_type"] == "refuse"


def test_parse_invalid_decision_type_nulled():
    item = {"text": "x", "dsr_labels": ["decision"], "decision_type": "maybe"}
    out = parse_dsr_response(json.dumps([item]))
    assert out[0]["decision_type"] is None


# ── single-chain, multi-judge ─────────────────────────────────────────────────

def _canned(spans):
    return lambda judge, prompt: json.dumps(spans)


def test_annotate_chain_all_judges_complete():
    judges = [Judge("A", "m"), Judge("B", "m"), Judge("C", "m")]
    res = annotate_chain_dsr("The policy says no.", judges, call_fn=_canned([SPAN]))
    assert res["complete"] is True
    assert set(res["judges"]) == {"A", "B", "C"}


def test_annotate_chain_failing_judge_marks_incomplete():
    judges = [Judge("A", "m"), Judge("B", "m")]

    def call_fn(judge, prompt):
        if judge.name == "B":
            raise RuntimeError("proxy down")
        return json.dumps([SPAN])

    res = annotate_chain_dsr("The policy says no.", judges, call_fn=call_fn, max_retries=1)
    assert res["complete"] is False
    assert res["judges"]["B"] == []
    assert res["judges"]["A"]


# ── consensus aggregation ─────────────────────────────────────────────────────

def test_aggregate_majority_present():
    chain = "The policy says no. I will comply now."
    spec = {"text": "The policy says no.", "dsr_labels": ["spec_citation"], "decision_type": None}
    dec = {"text": "I will comply now.", "dsr_labels": ["decision"], "decision_type": "comply"}
    by_judge = {
        "A": [spec, dec],
        "B": [spec, dec],
        "C": [{"text": "I will comply now.", "dsr_labels": ["decision"], "decision_type": "comply"}],
    }
    out = aggregate_dsr(chain, by_judge)
    assert any("spec_citation" in s["dsr_labels"] for s in out["spans"])
    dec_spans = [s for s in out["spans"] if "decision" in s["dsr_labels"]]
    assert dec_spans and dec_spans[0]["decision_type"] == "comply"


def test_aggregate_tie_is_unresolved_not_present():
    chain = "Weighing whether to help here."
    span = {"text": "Weighing whether to help here.", "dsr_labels": ["adjudication"], "decision_type": None}
    by_judge = {"A": [span], "B": []}  # 1 of 2 → tie
    out = aggregate_dsr(chain, by_judge)
    assert out["n_unresolved_chars"] > 0
    assert all("adjudication" not in s["dsr_labels"] for s in out["spans"])


# ── batch runner ──────────────────────────────────────────────────────────────

def test_annotate_chains_dsr_writes_and_resumes(tmp_path):
    chains = [
        {"task_id": "t1", "chain": "The policy says no."},
        {"task_id": "t2", "chain": "I will comply now."},
    ]
    judges = [Judge("A", "m"), Judge("B", "m"), Judge("C", "m")]
    out = tmp_path / "dsr.json"
    ann = annotate_chains_dsr(chains, judges, save_path=out, call_fn=_canned([SPAN]))
    assert len(ann) == 2
    assert all(a["dsr_complete"] for a in ann)
    assert out.exists()
    assert all("dsr_consensus" in a for a in ann)

    # resume: nothing new to do, same length, file still valid
    def _boom(judge, prompt):
        raise AssertionError("should not be called on resume")

    ann2 = annotate_chains_dsr(chains, judges, save_path=out, call_fn=_boom)
    assert len(ann2) == 2


def test_agreement_report_has_all_labels(tmp_path):
    chains = [{"task_id": "t1", "chain": "The policy says no."}]
    judges = [Judge("A", "m"), Judge("B", "m"), Judge("C", "m")]
    ann = annotate_chains_dsr(chains, judges, call_fn=_canned([SPAN]))
    rep = agreement_report(ann)
    assert set(rep) == {"spec_citation", "adjudication", "decision", "harm_recognition"}
    assert rep["spec_citation"]["gate"] in {"citable", "replicate", "uninterpretable"}
