"""
Pre-flight crash/resume-safety tests for Phase-3 / Phase-7 annotation.

The annotation step is the only paid ($) consumer in Phase 7. Credits may run
out mid-run, the AWS proxy may 503/timeout, or a JSON body may be truncated.
The contract these tests pin down:

  * A credit/HTTP/parse failure must NEVER crash the batch loop and must lose
    at most the single in-flight chain.
  * `annotate_chains` resumes from its checkpoint file: fully-complete chains
    (by ``dedup_keys``) are skipped, partial chains (annotation_complete=False)
    are retried.
  * The Phase-7 dedup key ``(task_id, behaviour, method, alpha)`` keeps distinct
    steered variants of the SAME task_id as separate records (a key of just
    ``task_id`` would collapse them — a silent, citation-poisoning bug).
  * Saves are atomic (``.tmp`` → ``os.replace``) so a crash during write can't
    truncate the checkpoint.

NO real API calls — the proxy is mocked at the ``_proxy_call`` boundary (and in
one case at ``requests.post``) so the transport, retry, and parse paths are all
exercised without network access.
"""

import json
from pathlib import Path

import pytest

import src.annotation as ann
from src.annotation import (
    annotate_chain,
    annotate_chains,
    chunk_chain,
    merge_chunk_annotations,
    _annotate_single,
)


# ── helpers ──────────────────────────────────────────────────────────────────

# A canned, well-formed Venhoff-format response (two short spans).
GOOD_RESPONSE = (
    '["initializing"]Let me restate the problem.["end-section"]'
    '["deduction"]Therefore x = 2.["end-section"]'
)


def _phase7_chain(task_id, behaviour, method, alpha, text="Some reasoning. More."):
    """A steered-chain record shaped like src/steered_inference.py emits."""
    return {
        "task_id": task_id,
        "behaviour": behaviour,
        "method": method,
        "alpha": alpha,
        "chain": text,
    }


class _ProxyStub:
    """Records call count and returns/raises a scripted sequence.

    Patched in over ``_proxy_call`` so retries, parsing and the batch loop all
    run for real; only the network is replaced.
    """

    def __init__(self, behaviour):
        self.calls = 0
        self.behaviour = behaviour  # callable(call_index) -> str OR raises

    def __call__(self, prompt, proxy_url=None, proxy_key=None,
                 max_tokens=8192, temperature=0.0, model=ann.ANNOTATION_MODEL):
        self.calls += 1
        return self.behaviour(self.calls, prompt)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Make retry backoff and the batch rate-limiter instant in all tests."""
    monkeypatch.setattr(ann.time, "sleep", lambda *_: None)


@pytest.fixture
def always_good(monkeypatch):
    stub = _ProxyStub(lambda i, prompt: GOOD_RESPONSE)
    monkeypatch.setattr(ann, "_proxy_call", stub)
    return stub


# ── 1. atomic save + basic happy path ─────────────────────────────────────────

def test_save_json_is_atomic_and_leaves_no_tmp(tmp_path, always_good):
    out = tmp_path / "annotated.json"
    chains = [_phase7_chain("t1", "backtracking", "single_direction", 8.0)]
    annotate_chains(chains, save_path=out, dedup_keys=("task_id",))

    assert out.exists()
    # No stray .tmp left behind (atomic replace cleaned up).
    assert not (tmp_path / "annotated.tmp").exists()
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0]["annotation_complete"] is True
    assert data[0]["annotations"]  # non-empty
    # Original record fields preserved.
    assert data[0]["task_id"] == "t1"
    assert data[0]["behaviour"] == "backtracking"


# ── 2. Phase-7 dedup key keeps steered variants separate ──────────────────────

def test_phase7_dedup_keys_do_not_collapse_same_task_id(tmp_path, always_good):
    """Four steered variants share task_id 't1' but differ in (behaviour, method,
    alpha). With the Phase-7 key all four must be annotated as DISTINCT records.
    """
    key = ("task_id", "behaviour", "method", "alpha")
    chains = [
        _phase7_chain("t1", "backtracking", "single_direction", 8.0),
        _phase7_chain("t1", "backtracking", "manifold_k3", 8.0),
        _phase7_chain("t1", "uncertainty-estimation", "single_direction", 8.0),
        _phase7_chain("t1", "backtracking", "single_direction", 16.0),
    ]
    out = tmp_path / "ann.json"
    result = annotate_chains(chains, save_path=out, dedup_keys=key)

    assert len(result) == 4, "Phase-7 key collapsed distinct steered variants!"
    seen = {(r["task_id"], r["behaviour"], r["method"], r["alpha"]) for r in result}
    assert len(seen) == 4
    # The proxy was hit once per distinct variant (no accidental skipping).
    assert always_good.calls == 4


def test_wrong_key_task_id_only_DOES_collapse(tmp_path, always_good):
    """Negative control: prove the collapse bug is real with the WRONG key, so
    the test above is actually load-bearing. With dedup_keys=('task_id',) the
    four same-task variants collapse to one annotated record.
    """
    chains = [
        _phase7_chain("t1", "backtracking", "single_direction", 8.0),
        _phase7_chain("t1", "backtracking", "manifold_k3", 8.0),
        _phase7_chain("t1", "uncertainty-estimation", "single_direction", 8.0),
        _phase7_chain("t1", "backtracking", "single_direction", 16.0),
    ]
    out = tmp_path / "ann.json"
    result = annotate_chains(chains, save_path=out, dedup_keys=("task_id",))
    # All four iterate, but the LAST write per key wins / earlier get skipped on
    # the 2nd..4th iterations because the first is appended immediately and its
    # key is in done_ids. Net: fewer than 4 unique keys are stored.
    seen = {r["task_id"] for r in result}
    assert len(seen) == 1  # collapsed — this is the bug the right key avoids


# ── 3. resume: complete chains skipped, proxy not re-hit ──────────────────────

def test_resume_skips_completed_chains(tmp_path, monkeypatch):
    key = ("task_id", "behaviour", "method", "alpha")
    chains = [
        _phase7_chain("t1", "backtracking", "single_direction", 8.0),
        _phase7_chain("t2", "backtracking", "single_direction", 8.0),
    ]
    out = tmp_path / "ann.json"

    stub1 = _ProxyStub(lambda i, prompt: GOOD_RESPONSE)
    monkeypatch.setattr(ann, "_proxy_call", stub1)
    annotate_chains(chains, save_path=out, dedup_keys=key)
    assert stub1.calls == 2

    # Re-run: everything already complete → proxy must NOT be hit again.
    stub2 = _ProxyStub(lambda i, prompt: GOOD_RESPONSE)
    monkeypatch.setattr(ann, "_proxy_call", stub2)
    result = annotate_chains(chains, save_path=out, dedup_keys=key)
    assert stub2.calls == 0, "resume re-annotated already-complete chains"
    assert len(result) == 2


# ── 4. kill_after interruption then resume from exactly where it stopped ──────

def test_kill_after_then_resume(tmp_path, monkeypatch):
    key = ("task_id", "behaviour", "method", "alpha")
    chains = [_phase7_chain(f"t{i}", "backtracking", "single_direction", 8.0)
              for i in range(5)]
    out = tmp_path / "ann.json"

    stub1 = _ProxyStub(lambda i, prompt: GOOD_RESPONSE)
    monkeypatch.setattr(ann, "_proxy_call", stub1)
    annotate_chains(chains, save_path=out, dedup_keys=key, kill_after=2)
    saved = json.loads(out.read_text())
    assert len(saved) == 2, "kill_after did not stop+save after 2 new chains"
    assert stub1.calls == 2

    # Resume: only the remaining 3 are annotated.
    stub2 = _ProxyStub(lambda i, prompt: GOOD_RESPONSE)
    monkeypatch.setattr(ann, "_proxy_call", stub2)
    result = annotate_chains(chains, save_path=out, dedup_keys=key)
    assert stub2.calls == 3, "resume re-did work or skipped remaining chains"
    assert len(result) == 5
    assert len({(r["task_id"]) for r in result}) == 5


# ── 5. CRASH SAFETY: a mid-loop exception must not crash the batch ────────────

def test_proxy_503_does_not_crash_loop_and_saves_progress(tmp_path, monkeypatch):
    """Credit-exhausted / 503 surfaces as an exception from the transport. After
    retries it must degrade to annotation_complete=False for that ONE chain; all
    other chains must still be annotated and the whole file saved.

    The failure is keyed on the chain's TEXT (carried into the prompt), so ALL
    retries of the bad chain fail (not just the first attempt) — the chain must
    end up genuinely incomplete, not accidentally succeed on retry.
    """
    key = ("task_id", "behaviour", "method", "alpha")
    chains = [
        _phase7_chain("t0", "backtracking", "single_direction", 8.0, "alpha reasoning."),
        _phase7_chain("t1", "backtracking", "single_direction", 8.0, "beta reasoning."),
        _phase7_chain("t2", "backtracking", "single_direction", 8.0, "CREDIT_DEAD here."),
        _phase7_chain("t3", "backtracking", "single_direction", 8.0, "delta reasoning."),
    ]
    out = tmp_path / "ann.json"

    def behaviour(i, prompt):
        if "CREDIT_DEAD" in prompt:
            raise RuntimeError("HTTP 503: credits exhausted")
        return GOOD_RESPONSE

    monkeypatch.setattr(ann, "_proxy_call", _ProxyStub(behaviour))
    result = annotate_chains(chains, save_path=out, dedup_keys=key)

    assert len(result) == 4, "a failing chain dropped the run's other chains"
    by_task = {r["task_id"]: r for r in result}
    assert by_task["t0"]["annotation_complete"] is True
    assert by_task["t3"]["annotation_complete"] is True
    assert by_task["t2"]["annotation_complete"] is False  # the failed one
    assert by_task["t2"]["annotations"] == []
    # Everything-so-far is on disk (atomic, complete file).
    on_disk = json.loads(out.read_text())
    assert len(on_disk) == 4

    # Resume with credits restored: only the incomplete chain is retried.
    good = _ProxyStub(lambda i, prompt: GOOD_RESPONSE)
    monkeypatch.setattr(ann, "_proxy_call", good)
    result2 = annotate_chains(chains, save_path=out, dedup_keys=key)
    assert good.calls == 1, "resume should retry exactly the 1 failed chain"
    assert all(r["annotation_complete"] for r in result2)


def test_annotate_chain_raise_does_not_crash_loop(tmp_path, monkeypatch):
    """Hard case from the brief: mock annotate_chain itself to RAISE on the 3rd
    chain (e.g. an unexpected KeyError / TypeError deeper in the stack that the
    retry wrapper does not catch). The batch loop must catch-save-continue, not
    propagate. Chains 1–2 must be on disk and a re-run must resume.
    """
    key = ("task_id", "behaviour", "method", "alpha")
    chains = [_phase7_chain(f"t{i}", "backtracking", "single_direction", 8.0)
              for i in range(4)]
    out = tmp_path / "ann.json"

    call = {"n": 0}
    real_done = []

    def fake_annotate_chain(chain_text, **kwargs):
        call["n"] += 1
        if call["n"] == 3:
            raise ValueError("boom: malformed response object")
        return [{"label": "deduction", "text": "ok"}], True

    monkeypatch.setattr(ann, "annotate_chain", fake_annotate_chain)
    result = annotate_chains(chains, save_path=out, dedup_keys=key)

    # The loop survived the raise.
    assert len(result) == 4
    on_disk = json.loads(out.read_text())
    assert len(on_disk) == 4
    # The crashing chain is recorded as incomplete (so a re-run retries it).
    incomplete = [r for r in on_disk if not r["annotation_complete"]]
    assert len(incomplete) == 1

    # Re-run with a now-healthy annotate_chain: only the 1 incomplete chain is
    # retried, and it completes.
    def healthy(chain_text, **kwargs):
        return [{"label": "deduction", "text": "ok"}], True

    monkeypatch.setattr(ann, "annotate_chain", healthy)
    result2 = annotate_chains(chains, save_path=out, dedup_keys=key)
    assert all(r["annotation_complete"] for r in result2)
    assert len(result2) == 4


def test_missing_chain_key_does_not_crash_loop(tmp_path, always_good):
    """A malformed record missing the 'chain' field must not KeyError-crash the
    whole batch — it should be recorded incomplete and the loop continue.
    """
    key = ("task_id", "behaviour", "method", "alpha")
    chains = [
        _phase7_chain("t0", "backtracking", "single_direction", 8.0),
        {  # malformed: no "chain"
            "task_id": "t1", "behaviour": "backtracking",
            "method": "single_direction", "alpha": 8.0,
        },
        _phase7_chain("t2", "backtracking", "single_direction", 8.0),
    ]
    out = tmp_path / "ann.json"
    result = annotate_chains(chains, save_path=out, dedup_keys=key)
    assert len(result) == 3
    by_task = {r["task_id"]: r for r in result}
    assert by_task["t0"]["annotation_complete"] is True
    assert by_task["t2"]["annotation_complete"] is True
    assert by_task["t1"]["annotation_complete"] is False


def test_missing_dedup_key_does_not_crash_loop(tmp_path, always_good):
    """A record missing a dedup key (e.g. no 'alpha') must not KeyError-crash the
    batch when building the resume key.
    """
    key = ("task_id", "behaviour", "method", "alpha")
    chains = [
        _phase7_chain("t0", "backtracking", "single_direction", 8.0),
        {  # missing 'alpha'
            "task_id": "t1", "behaviour": "backtracking",
            "method": "single_direction", "chain": "Reasoning here.",
        },
    ]
    out = tmp_path / "ann.json"
    # Must not raise.
    result = annotate_chains(chains, save_path=out, dedup_keys=key)
    assert len(result) == 2


# ── 6. partial-retry on resume ────────────────────────────────────────────────

def test_partial_record_is_retried_on_resume(tmp_path, monkeypatch):
    """A pre-existing checkpoint with annotation_complete=False must be retried
    (and replaced) on resume, not skipped.
    """
    key = ("task_id", "behaviour", "method", "alpha")
    out = tmp_path / "ann.json"
    # Seed a checkpoint: t0 complete, t1 partial.
    seed = [
        {"task_id": "t0", "behaviour": "backtracking",
         "method": "single_direction", "alpha": 8.0,
         "chain": "x", "annotations": [{"label": "deduction", "text": "x"}],
         "annotation_complete": True},
        {"task_id": "t1", "behaviour": "backtracking",
         "method": "single_direction", "alpha": 8.0,
         "chain": "Reasoning here.", "annotations": [], "annotation_complete": False},
    ]
    out.write_text(json.dumps(seed))

    chains = [
        _phase7_chain("t0", "backtracking", "single_direction", 8.0, "x"),
        _phase7_chain("t1", "backtracking", "single_direction", 8.0, "Reasoning here."),
    ]
    stub = _ProxyStub(lambda i, prompt: GOOD_RESPONSE)
    monkeypatch.setattr(ann, "_proxy_call", stub)
    result = annotate_chains(chains, save_path=out, dedup_keys=key)

    # Only t1 retried (t0 already complete).
    assert stub.calls == 1
    by_task = {r["task_id"]: r for r in result}
    assert by_task["t1"]["annotation_complete"] is True
    assert by_task["t1"]["annotations"]
    # No duplicate t1 record.
    assert len(result) == 2


def test_backward_compat_legacy_record_without_complete_flag(tmp_path, monkeypatch):
    """Legacy records (written before annotation_complete existed) with non-empty
    annotations are treated as complete and skipped on resume.
    """
    key = ("task_id",)
    out = tmp_path / "ann.json"
    seed = [{"task_id": "t0", "chain": "x",
             "annotations": [{"label": "deduction", "text": "x"}]}]  # no complete flag
    out.write_text(json.dumps(seed))

    chains = [_phase7_chain("t0", "backtracking", "single_direction", 8.0, "x")]
    stub = _ProxyStub(lambda i, prompt: GOOD_RESPONSE)
    monkeypatch.setattr(ann, "_proxy_call", stub)
    result = annotate_chains(chains, save_path=out, dedup_keys=key)
    assert stub.calls == 0, "legacy complete record was re-annotated"
    assert len(result) == 1


def test_legacy_record_with_empty_annotations_is_retried(tmp_path, monkeypatch):
    """Legacy record with EMPTY annotations and no flag is treated as incomplete
    → retried.
    """
    key = ("task_id",)
    out = tmp_path / "ann.json"
    seed = [{"task_id": "t0", "chain": "Reasoning here.", "annotations": []}]
    out.write_text(json.dumps(seed))

    chains = [_phase7_chain("t0", "backtracking", "single_direction", 8.0,
                            "Reasoning here.")]
    stub = _ProxyStub(lambda i, prompt: GOOD_RESPONSE)
    monkeypatch.setattr(ann, "_proxy_call", stub)
    result = annotate_chains(chains, save_path=out, dedup_keys=key)
    assert stub.calls == 1
    assert len(result) == 1
    assert result[0]["annotation_complete"] is True


# ── 7. env-var absence handling at the transport boundary ─────────────────────

def test_proxy_call_missing_env_raises_keyerror(monkeypatch):
    """With no proxy_url override and CLAUDE_PROXY_URL unset, _proxy_call raises
    KeyError. (This is the documented behaviour; the batch loop must absorb it —
    see test_proxy_503... — but at the transport level it surfaces.)
    """
    monkeypatch.delenv("CLAUDE_PROXY_URL", raising=False)
    monkeypatch.delenv("CLAUDE_PROXY_KEY", raising=False)
    with pytest.raises(KeyError):
        ann._proxy_call("hello")


def test_missing_env_is_absorbed_by_annotate_single(monkeypatch):
    """A missing env var inside the retry loop is caught like any other
    exception and yields [] (no crash) — so a misconfigured run degrades to
    'all incomplete' rather than aborting.
    """
    monkeypatch.delenv("CLAUDE_PROXY_URL", raising=False)
    monkeypatch.delenv("CLAUDE_PROXY_KEY", raising=False)
    # Patch sleep so the retry backoff doesn't slow the test.
    monkeypatch.setattr(ann.time, "sleep", lambda *_: None)
    spans = _annotate_single("some text", max_retries=2)
    assert spans == []


# ── 8. transport / parse paths via mocked requests.post ───────────────────────

class _FakeResp:
    def __init__(self, status, json_body=None, raise_json=False):
        self.status_code = status
        self._json_body = json_body
        self._raise_json = raise_json

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        if self._raise_json:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._json_body


def test_requests_post_credit_error_is_retried_then_empty(monkeypatch):
    """At the real transport boundary: a 429/503 raises in raise_for_status,
    gets retried max_retries times, then _annotate_single returns []."""
    calls = {"n": 0}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        return _FakeResp(503)

    monkeypatch.setattr(ann.requests, "post", fake_post)
    monkeypatch.setattr(ann.time, "sleep", lambda *_: None)
    spans = _annotate_single("text", max_retries=3,
                             proxy_url="http://x", proxy_key="k")
    assert spans == []
    assert calls["n"] == 3  # retried exactly max_retries times


def test_requests_post_bad_json_is_retried_then_empty(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp(200, raise_json=True)

    monkeypatch.setattr(ann.requests, "post", fake_post)
    monkeypatch.setattr(ann.time, "sleep", lambda *_: None)
    spans = _annotate_single("text", max_retries=2,
                             proxy_url="http://x", proxy_key="k")
    assert spans == []


def test_requests_post_anthropic_list_shape_parses(monkeypatch):
    body = {"content": [{"type": "text", "text": GOOD_RESPONSE}]}

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp(200, json_body=body)

    monkeypatch.setattr(ann.requests, "post", fake_post)
    spans = _annotate_single("text", proxy_url="http://x", proxy_key="k")
    assert [s["label"] for s in spans] == ["initializing", "deduction"]


def test_model_param_threads_through_to_proxy(monkeypatch):
    """A non-default annotator model passed to annotate_chains must reach
    _proxy_call (the lever that breaks builder-scores-own-output circularity).
    """
    seen = {}

    def fake_proxy(prompt, proxy_url=None, proxy_key=None,
                   max_tokens=8192, temperature=0.0, model=ann.ANNOTATION_MODEL):
        seen["model"] = model
        return GOOD_RESPONSE

    monkeypatch.setattr(ann, "_proxy_call", fake_proxy)
    chains = [_phase7_chain("t1", "backtracking", "single_direction", 8.0)]
    annotate_chains(chains, dedup_keys=("task_id",),
                    model="eu.anthropic.claude-other-model")
    assert seen["model"] == "eu.anthropic.claude-other-model"


# ── 9. chunking: offset/overlap-merge logic (+ CF-18 note) ────────────────────

def test_chunk_chain_splits_long_text():
    paras = ["Paragraph %d content here with several words." % i for i in range(40)]
    text = "\n\n".join(paras)
    chunks = chunk_chain(text, target_tokens=50, overlap_tokens=10)
    assert len(chunks) > 1
    # Every paragraph is preserved across the chunk set (nothing dropped).
    joined = " ".join(chunks)
    for i in range(40):
        assert ("Paragraph %d " % i) in joined


def test_chunk_chain_overlap_is_dropped_when_paragraph_exceeds_overlap_budget():
    """FINDING (BUG, documented not fixed): chunk_chain's overlap loop breaks on
    the FIRST paragraph whose own token-estimate exceeds ``overlap_tokens``, so
    when individual paragraphs are larger than the overlap budget NO overlap is
    carried between chunks at all. With the production defaults
    (CHUNK_OVERLAP_TOKENS=100 ≈ 400 chars) any reasoning paragraph longer than
    ~400 chars — extremely common — produces ZERO overlap, defeating the
    seam-artefact protection that merge_chunk_annotations + _CONTINUATION_PREFIX
    were built around. This test PINS the current behaviour.
    """
    # Each paragraph (~11 tokens) is larger than overlap_tokens=10.
    paras = ["Paragraph %d content here with several words." % i for i in range(40)]
    chunks = chunk_chain("\n\n".join(paras), target_tokens=50, overlap_tokens=10)
    overlaps = []
    for a, b in zip(chunks, chunks[1:]):
        tail_para = a.split("\n\n")[-1]
        overlaps.append(tail_para in b)
    assert not any(overlaps), (
        "overlap unexpectedly present — the chunk_chain overlap bug may have "
        "been fixed; if so, update merge tests and this assertion deliberately"
    )


def test_merge_discards_overlap_spans_from_later_chunks():
    chunk_texts = ["A. B. C. OVERLAP_TAIL.", "OVERLAP_TAIL. D. E."]
    chunk_anns = [
        [{"label": "deduction", "text": "A."},
         {"label": "deduction", "text": "OVERLAP_TAIL."}],
        [{"label": "deduction", "text": "OVERLAP_TAIL."},  # dup from overlap
         {"label": "deduction", "text": "D."}],
    ]
    merged = merge_chunk_annotations(chunk_texts, chunk_anns, overlap_tokens=8)
    texts = [s["text"] for s in merged]
    # OVERLAP_TAIL kept once (from chunk 0), not duplicated from chunk 1.
    assert texts.count("OVERLAP_TAIL.") == 1
    assert "D." in texts


def test_partial_chunk_failure_marks_chain_incomplete(monkeypatch):
    """If one chunk of a long chain fails after retries, annotate_chain returns
    complete=False so the chain is retried on resume (not silently truncated).

    Note: annotate_chain calls chunk_chain() with no args, so it uses the real
    default CHUNK_TARGET_TOKENS (1000). We therefore build a genuinely long
    (>1200-token) chain so it actually splits under production settings, rather
    than monkeypatching the module global (which can't affect the def-time
    default-arg binding).
    """
    # ~50 paragraphs of ~60 tokens each ≈ 3000 tokens → multiple chunks.
    big_para = ("This is a fairly long reasoning paragraph that contains many "
                "words so that its token estimate is comfortably large enough "
                "to matter for chunk sizing in the annotation pipeline. ")
    paras = [f"P{i}: {big_para}" for i in range(50)]
    # One sentinel paragraph that will fail when it reaches the proxy.
    paras[30] = "P30_SENTINEL: CREDIT_DEAD mid chain " + big_para
    long_text = "\n\n".join(paras)
    assert ann._estimate_tokens(long_text) > ann.CHUNK_THRESHOLD_TOKENS

    def behaviour(i, prompt):
        if "CREDIT_DEAD" in prompt:
            raise RuntimeError("503 mid-chunk")
        return GOOD_RESPONSE

    monkeypatch.setattr(ann, "_proxy_call", _ProxyStub(behaviour))
    spans, complete = annotate_chain(long_text, max_retries=1)
    assert complete is False, "partial-chunk failure must mark chain incomplete"
    # The surviving chunks still contributed spans (partial, not empty).
    assert spans, "expected partial annotations from the chunks that succeeded"


def test_cf18_overlap_dedup_can_delete_genuine_recurrence():
    """CF-18 documentation test (NOT a fix): merge_chunk_annotations drops a span
    from chunk N+1 purely because its text also appears verbatim in the overlap
    tail of chunk N. A genuinely-recurring SHORT sentence (e.g. 'Wait.') that
    legitimately appears in both the overlap region AND later in chunk N+1 can be
    silently deleted. This test PINS the current (lossy) behaviour so a future
    fix is a deliberate, reviewed change.
    """
    chunk_texts = ["... Wait.", "Wait. Now something else. Wait."]
    chunk_anns = [
        [{"label": "backtracking", "text": "Wait."}],
        [
            {"label": "backtracking", "text": "Wait."},        # overlap dup (should drop)
            {"label": "deduction", "text": "Now something else."},
            {"label": "backtracking", "text": "Wait."},        # GENUINE recurrence
        ],
    ]
    merged = merge_chunk_annotations(chunk_texts, chunk_anns, overlap_tokens=4)
    texts = [s["text"] for s in merged]
    # Current behaviour: BOTH "Wait." spans in chunk 1 are dropped (text-in-region
    # check is not positional), so the genuine recurrence is lost.
    assert texts.count("Wait.") == 1  # documents the CF-18 over-deletion
