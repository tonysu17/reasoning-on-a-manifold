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


def test_parse_normalises_decision_colon_type_shorthand():
    # Judges imitating the v2 prompt's example notation emit 'decision:refuse'
    # as a single label (caught live in the P1v2 launch, 2026-07-20).
    item = {"text": "We must refuse.", "dsr_labels": ["spec_citation", "decision:refuse"]}
    out = parse_dsr_response(json.dumps([item]))
    assert out[0]["dsr_labels"] == ["decision", "spec_citation"]
    assert out[0]["decision_type"] == "refuse"


def test_parse_decision_shorthand_defers_to_explicit_type():
    item = {"text": "x", "dsr_labels": ["decision:comply"], "decision_type": "refuse"}
    out = parse_dsr_response(json.dumps([item]))
    assert out[0]["dsr_labels"] == ["decision"]
    assert out[0]["decision_type"] == "refuse"


def test_parse_decision_shorthand_bad_type_still_dropped():
    item = {"text": "x", "dsr_labels": ["decision:maybe"]}
    out = parse_dsr_response(json.dumps([item]))
    assert out[0]["dsr_labels"] == []
    assert out[0]["decision_type"] is None


# ── Chunking (29 s proxy-gateway cap) ─────────────────────────────────────────

def test_chunk_short_text_single_chunk():
    from src.safety.annotate import chunk_chain_text
    assert chunk_chain_text("short. text.", max_chars=100) == ["short. text."]


def test_chunk_verbatim_slices_and_size_cap():
    from src.safety.annotate import chunk_chain_text
    text = " ".join(f"Sentence number {i} ends here." for i in range(60))
    chunks = chunk_chain_text(text, max_chars=200)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert "".join(chunks) == text          # exact reconstruction
    assert all(c in text for c in chunks)   # verbatim slices


def test_chunk_boundary_free_window_hard_cuts():
    from src.safety.annotate import chunk_chain_text
    text = "x" * 500  # no sentence boundary anywhere
    chunks = chunk_chain_text(text, max_chars=200)
    assert "".join(chunks) == text
    assert all(len(c) <= 200 for c in chunks)


def test_chunked_annotation_concatenates_and_passes_context():
    from src.safety.annotate import annotate_chain_dsr
    text = " ".join(f"Sentence {i} is right here." for i in range(40))
    prompts = []

    def fake(judge, prompt):
        prompts.append(prompt)
        # echo one span per call so every chunk parses
        return json.dumps([{"text": f"chunk-{len(prompts)}", "dsr_labels": ["spec_citation"]}])

    res = annotate_chain_dsr(text, [Judge("A", "m")], call_fn=fake, chunk_chars=200)
    assert res["n_chunks"] > 1
    assert res["complete"]
    assert len(res["judges"]["A"]) == res["n_chunks"]  # spans concatenated
    # later chunks carry unlabelled context from their predecessor
    assert "REFERENCE ONLY" not in prompts[0]
    assert all("REFERENCE ONLY" in p for p in prompts[1:])


def test_backoff_503_much_longer_than_default():
    from src.safety.annotate import _backoff_seconds
    assert _backoff_seconds(0, "503 Server Error: Service Unavailable") == 20.0
    assert _backoff_seconds(2, "503 Server Error") == 60.0
    assert _backoff_seconds(2, "some other error") == 4.0


def test_topup_recalls_only_failed_judge(tmp_path):
    from src.safety.annotate import annotate_chains_dsr
    chains = [{"task_id": "t1", "chain": "The policy says no. We must refuse."}]
    judges = [Judge("A", "m"), Judge("B", "m")]
    out = tmp_path / "dsr.json"

    span = [{"text": "The policy says no.", "dsr_labels": ["spec_citation"]}]

    # pass 1: A succeeds, B fails all retries -> incomplete record with flags
    def b_down(judge, prompt):
        if judge.name == "B":
            raise RuntimeError("connection reset")
        return json.dumps(span)

    ann = annotate_chains_dsr(chains, judges, save_path=out, call_fn=b_down)
    assert not ann[0]["dsr_complete"]
    assert ann[0]["dsr_judges_ok"] == {"A": True, "B": False}

    # pass 2: proxy healthy — only B may be called
    calls = []

    def healthy(judge, prompt):
        calls.append(judge.name)
        return json.dumps(span)

    ann2 = annotate_chains_dsr(chains, judges, save_path=out, call_fn=healthy)
    assert calls and set(calls) == {"B"}          # A's spans kept, never re-billed
    assert ann2[0]["dsr_complete"]
    assert ann2[0]["dsr_judges_ok"] == {"A": True, "B": True}
    assert set(ann2[0]["dsr_per_judge"]) == {"A", "B"}

    # pass 3: nothing left to do — no calls at all
    def boom(judge, prompt):
        raise AssertionError("complete record must not be re-annotated")

    ann3 = annotate_chains_dsr(chains, judges, save_path=out, call_fn=boom)
    assert ann3[0]["dsr_complete"]


def test_legacy_incomplete_record_fully_rerun(tmp_path):
    from src.safety.annotate import annotate_chains_dsr
    chains = [{"task_id": "t1", "chain": "The policy says no."}]
    judges = [Judge("A", "m")]
    out = tmp_path / "dsr.json"
    # legacy incomplete record: no dsr_judges_ok field
    out.write_text(json.dumps([{"task_id": "t1", "chain": "The policy says no.",
                                "dsr_per_judge": {"A": []}, "dsr_complete": False}]))
    calls = []

    def healthy(judge, prompt):
        calls.append(judge.name)
        return json.dumps([{"text": "The policy says no.",
                            "dsr_labels": ["spec_citation"]}])

    ann = annotate_chains_dsr(chains, judges, save_path=out, call_fn=healthy)
    assert calls == ["A"]                          # discarded and re-run in full
    assert ann[0]["dsr_complete"]


def test_chunked_annotation_one_failed_chunk_marks_incomplete():
    from src.safety.annotate import annotate_chain_dsr
    text = " ".join(f"Sentence {i} is right here." for i in range(40))
    calls = {"n": 0}

    def flaky(judge, prompt):
        calls["n"] += 1
        if calls["n"] <= 3:  # first chunk fails all 3 retries
            raise RuntimeError("boom")
        return json.dumps([{"text": "ok", "dsr_labels": []}])

    res = annotate_chain_dsr(text, [Judge("A", "m")], call_fn=flaky, chunk_chars=200)
    assert not res["complete"]              # a lost chunk is an incomplete judge
    assert len(res["judges"]["A"]) == res["n_chunks"] - 1  # other chunks kept


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
