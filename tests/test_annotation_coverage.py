"""Regression tests for the 2026-08-11 Phase-2 annotation coverage failure.

The halted run produced a checkpoint that passed every structural and
provenance check — unique keys, exact schema, six permitted labels, every span
source-faithful and in order — while five of fifteen rows silently dropped
material reasoning and post-``</think>`` text was annotated in four rows and
omitted in five.  All fifteen were nevertheless ``annotation_complete: true``.

Each fixture below reproduces one real defect shape from that checkpoint, so a
regression cannot pass by satisfying the schema alone.  No test performs
network I/O; the transport is mocked or asserted never to be reached.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from src import annotation
from src.annotation_budget import (AnnotationAttemptGuard, AnnotationAttemptLimitError,
                                   AnnotationCostLimitError, AnnotationRateCard,
                                   SCHEMA_VERSION)
from src.annotation_coverage import (COVERAGE_RULE_VERSION, annotation_region,
                                     normalise, region_source_text,
                                     row_is_coverage_complete, validate_coverage)
from src.annotation_quarantine import (QuarantineError, apply_quarantine,
                                       plan_quarantine)


def span(label, text):
    return {"label": label, "text": text}


# ── the annotation region ────────────────────────────────────────────────────

def test_region_ends_at_think_close_and_excludes_the_response():
    chain = "Let me start. I deduce X. </think> The answer is X because of Y."
    region = annotation_region(chain)
    assert "</think>" not in region.text
    assert "The answer is X" not in region.text
    assert region.text == "Let me start. I deduce X."
    assert [e.kind for e in region.exclusions] == ["post_think_response"]


def test_region_is_the_whole_window_when_no_think_marker_is_present():
    chain = "Let me start. I deduce X."
    assert annotation_region(chain).text == chain


def test_short_trailing_final_answer_block_is_excluded():
    trailing = "I deduce X. **Final Answer** The value is \\boxed{7}."
    region = annotation_region(trailing)
    assert region.text == "I deduce X."
    assert [e.kind for e in region.exclusions] == ["final_answer_suffix"]


def test_an_oversized_final_answer_block_stays_in_scope():
    """The exclusion fails safe: too-long trailing text is never deleted.

    A long block after the marker is material, so it remains in the region and
    surfaces as a coverage gap if unannotated — rather than being silently
    dropped, which is the failure class this module exists to prevent.
    """
    # Non-repetitive filler: a repeated sentence would (correctly) trigger the
    # rule -4 degenerate cut instead of exercising the Final-Answer clause.
    long_tail = "**Final Answer** " + " ".join(
        f"and then I reconsider aspect number {i} of the whole thing." for i in range(12)
    )
    chain = f"I deduce X. {long_tail}"
    region = annotation_region(chain)
    assert "and then I reconsider" in region.text
    assert not region.exclusions

    report = validate_coverage(chain, [span("deduction", "I deduce X.")])
    assert not report.complete


def test_matching_folds_typographic_unicode():
    """Observed live 2026-08-11: the annotator echoes ``Let’s`` as ``Let's``.
    A faithful echo must never be rejected for typography."""
    chain = "Let’s set P at (0, 0, 0). Then—clearly—Q follows."
    spans = [span("deduction", "Let's set P at (0, 0, 0). Then-clearly-Q follows.")]
    report = validate_coverage(chain, spans)
    assert report.complete, report.reasons


def test_echoed_continuation_prefix_is_stripped_not_penalised():
    """Observed live 2026-08-11 (SPAT_126): the annotator returned the
    continuation-prefix instruction itself as a labelled span."""
    from src.annotation_coverage import CONTINUATION_PREFIX
    chain = "Real reasoning sentence one. Real reasoning sentence two."
    spans = [
        span("initializing", CONTINUATION_PREFIX + "Real reasoning sentence one."),
        span("deduction", "Real reasoning sentence two."),
    ]
    report = validate_coverage(chain, spans)
    assert report.complete, report.reasons
    # A pure-prefix span is ignored entirely rather than counted against the row.
    pure = [span("initializing", CONTINUATION_PREFIX)] + [
        span("deduction", "Real reasoning sentence one. Real reasoning sentence two.")
    ]
    assert validate_coverage(chain, pure).complete


def test_degenerate_cycling_is_truncated_after_five_occurrences():
    """Rule -4: 'Wait, no,' x100 keeps its first five occurrences only."""
    loop = "Wait, no, 40 divided by 2 is 20."
    chain = "Let me compute. " + " ".join([loop] * 40) + " </think> answer"
    region = annotation_region(chain)
    assert region.text.count("40 divided by 2") == 5
    kinds = [e.kind for e in region.exclusions]
    assert "degenerate_repetition" in kinds
    # The request the annotator sees is truncated identically.
    assert normalise(region_source_text(chain)) == region.text


def test_four_adjacent_repeats_stay_in_scope():
    """MATH_119's four adjacent occurrences are genuine reasoning, not
    degeneracy — below the six-repeat trigger, nothing is cut."""
    loop = "But wait, the relation is defined as a plus b."
    chain = " ".join([loop] * 4)
    region = annotation_region(chain)
    assert region.text.count("relation is defined") == 4
    assert not region.exclusions


def test_interleaved_repetition_is_never_cut():
    """MATH_117's sentence recurs 31 times WITH reasoning in between — only
    adjacent cycling triggers, so interleaved repetition survives whole."""
    unit = ("Wait, but hold on, is that always the case? "
            "Consider the next candidate value and check divisibility again. ")
    chain = (unit + "Some fresh deduction step here. ") * 31
    region = annotation_region(chain)
    assert region.text.count("is that always the case?") == 31
    assert not any(e.kind == "degenerate_repetition" for e in region.exclusions)


def test_a_row_failing_only_on_a_degenerate_tail_is_now_complete():
    """The LATE_118 shape: full annotation up to loop onset used to fail with
    a 2,700-char gap; under rule -4 the loop tail is out of region."""
    head = "The bartender pulled out a gun for the hiccups."
    chain = head + " " + " ".join(["Wait, no, wait, no."] * 60)
    spans = [span("deduction", head)] + [span("backtracking", "Wait, no, wait, no.")] * 5
    report = validate_coverage(chain, spans)
    assert report.complete, report.reasons


def test_region_ignores_whitespace_reflow():
    assert (annotation_region("A  b.\n\n C   d.").text
            == annotation_region("A b. C d.").text)


# ── the five known coverage defects ──────────────────────────────────────────

def test_omitted_adjacent_duplicate_is_incomplete():
    """MATH_119: a looped sentence occurs four times, annotated only three.

    Collapsing the fourth occurrence changes the numerator AND the denominator
    of ``behaviour_fraction``, which divides by the number of RETURNED spans.
    """
    loop = "But wait, in the graph, the relation is defined as a plus b."
    chain = " ".join([loop] * 4)
    three_of_four = [span("adding-knowledge", loop)] * 3

    report = validate_coverage(chain, three_of_four)
    assert not report.complete
    assert len(report.gaps) == 1
    assert loop in report.gaps[0].text

    all_four = [span("adding-knowledge", loop)] * 4
    assert validate_coverage(chain, all_four).complete


def test_repeated_reasoning_is_retained_as_separate_observations():
    """Duplicate spans must map to distinct source positions, never dedupe."""
    sentence = "Wait, maybe I need a different approach."
    chain = " ".join([sentence] * 4)
    spans = [span("backtracking", sentence)] * 4

    report = validate_coverage(chain, spans)
    assert report.complete
    assert report.matched_spans == 4
    assert report.n_spans == 4
    assert not report.gaps
    assert report.coverage_fraction > 0.98

    # A dedup implementation would return the sentence once and leave three
    # occurrences uncovered. That must NOT be complete.
    deduped = validate_coverage(chain, [span("backtracking", sentence)])
    assert not deduped.complete
    assert deduped.matched_spans == 1


def test_large_internal_omission_is_incomplete():
    """MATH_124: a 1,404-character block of algebra dropped mid-chain."""
    head = "Let me check symmetry."
    dropped = ("(3a + 5b) - (3b + 5a) is congruent to 0 mod 8, which simplifies "
               "to -2a + 2b congruent to 0 mod 8, so b is congruent to a mod 4. "
               "But this is not necessarily true for all a and b.")
    tail = "So the relation is not symmetric."
    chain = f"{head} {dropped} {tail}"

    report = validate_coverage(
        chain, [span("deduction", head), span("deduction", tail)]
    )
    assert not report.complete
    assert len(report.gaps) == 1
    assert report.gaps[0].n_chars > 100
    assert any("gap" in reason for reason in report.reasons)


def test_unique_omitted_reasoning_is_incomplete():
    """SPAT_120: an omission that is NOT a repeat of anything annotated."""
    kept_a = "Town A is at the origin."
    unique = ("Which is the same as sqrt of 14500 over 9, so the distance from "
              "D to C uses the coordinates 64 and 48.")
    kept_b = "So the radius follows."
    chain = f"{kept_a} {unique} {kept_b}"

    report = validate_coverage(
        chain, [span("deduction", kept_a), span("deduction", kept_b)]
    )
    assert not report.complete
    assert unique in report.gaps[0].text
    # The omission is unique: it appears nowhere among the returned spans.
    assert not any(unique in s["text"] for s in
                   [span("deduction", kept_a), span("deduction", kept_b)])


def test_inconsistent_post_think_inclusion_is_resolved_deterministically():
    """Four of nine eligible rows annotated the response; five did not.

    Under the frozen region rule the outcome no longer depends on which the
    annotator chose: response spans are outside the region, and a row that
    covers only the reasoning is complete.
    """
    chain = "I deduce X. </think> The answer is X, because Y holds throughout."
    reasoning_only = [span("deduction", "I deduce X.")]
    with_response = reasoning_only + [
        span("deduction", "The answer is X, because Y holds throughout.")
    ]

    assert validate_coverage(chain, reasoning_only).complete

    included = validate_coverage(chain, with_response)
    assert not included.complete
    assert included.outside_region_spans == (1,)
    assert any("excluded material" in reason for reason in included.reasons)


def test_span_out_of_source_order_is_not_a_match():
    chain = "First step. Second step."
    report = validate_coverage(
        chain, [span("deduction", "Second step."), span("deduction", "First step.")]
    )
    assert not report.complete
    assert report.out_of_order_spans == (1,)


def test_fabricated_span_is_unmatched():
    report = validate_coverage(
        "Real reasoning here.", [span("deduction", "Text the model invented.")]
    )
    assert not report.complete
    assert report.unmatched_spans == (0,)


def test_region_token_estimate_excludes_the_response():
    """The per-1k denominator must shrink with the region, or rates deflate."""
    chain = "I deduce X. </think> " + ("padding words here. " * 200)
    region = annotation_region(chain)
    assert region.estimated_tokens < len(chain) // 4 // 10


# ── the request must contain only the region ─────────────────────────────────

def test_region_source_text_preserves_paragraphs_and_agrees_with_the_scored_region():
    """The request slice is raw (the chunker splits on blank lines), but it must
    cut in the same place as the normalised region used for scoring."""
    chain = "Para one.\n\nPara two.\n\n</think>\n\nThe answer is 7."
    raw = region_source_text(chain)
    assert "\n\n" in raw, "paragraph structure must survive for the chunker"
    assert "</think>" not in raw and "The answer is 7." not in raw
    assert normalise(raw) == annotation_region(chain).text


def test_the_annotator_is_never_sent_excluded_material(tmp_path, monkeypatch):
    """Regression for the 2026-08-11 smoke run.

    Validating a region the REQUEST still contained made the annotator label
    post-`</think>` text and then be rejected for it — deterministically, on
    every one of the ~30% of corpus rows whose window contains the marker.
    """
    chain = "I deduce X.\n\n</think>\n\nThe answer is X, because Y holds."
    seen = []

    def fake(prompt, **kwargs):
        seen.append(prompt)
        return annotation.ProxyText('["deduction"]I deduce X.["end-section"]',
                                    usage_cost=0.001, remaining_quota=100.0)

    monkeypatch.setattr(annotation, "_proxy_call", fake)
    monkeypatch.setattr(annotation.time, "sleep", lambda *_: None)
    result = annotation.annotate_chains(
        [{"task_id": "T1", "chain": chain}], save_path=tmp_path / "a.json",
        dedup_keys=("task_id",),
    )
    assert len(seen) == 1
    assert "</think>" not in seen[0]
    assert "The answer is X" not in seen[0], "response text reached the annotator"
    assert result[0]["annotation_coverage_complete"] is True


def test_a_record_with_no_chain_is_incomplete_not_silently_empty():
    """A missing 'chain' must raise into the handler, never annotate ''."""
    result = annotation.annotate_chains(
        [{"task_id": "T1"}], dedup_keys=("task_id",)
    )
    assert result[0]["annotation_complete"] is False
    assert result[0]["annotation_coverage_complete"] is False


# ── completeness semantics ───────────────────────────────────────────────────

def test_schema_valid_but_coverage_incomplete_row_is_not_complete():
    """The exact failure mode: parsed spans, valid labels, partial coverage."""
    chain = "Step one here. A dropped middle section of genuine reasoning. Step two."
    spans = [span("deduction", "Step one here."), span("deduction", "Step two.")]
    for s in spans:                      # schema and label checks all pass
        assert s["label"] in annotation.VALID_LABELS
    record = {
        "task_id": "T1", "chain": chain, "annotations": spans,
        "annotation_complete": True,     # transport succeeded
    }
    assert not row_is_coverage_complete(record)


def test_pre_validator_rows_without_a_verdict_are_not_trusted():
    """Rows written before coverage validation existed must be re-eligible."""
    record = {"task_id": "T1", "chain": "A.", "annotations": [span("deduction", "A.")],
              "annotation_complete": True}
    assert not row_is_coverage_complete(record)


def test_a_verdict_from_a_superseded_rule_version_does_not_count():
    record = {
        "task_id": "T1", "chain": "A.", "annotations": [span("deduction", "A.")],
        "annotation_complete": True,
        "annotation_coverage": {"rule_version": "some-older-rule", "complete": True},
    }
    assert not row_is_coverage_complete(record)


# ── batch + resume behaviour ─────────────────────────────────────────────────

def _mock_proxy(monkeypatch, responses):
    """Serve canned annotated text, one entry per call."""
    queue = list(responses)

    def fake(prompt, **kwargs):
        return annotation.ProxyText(queue.pop(0), usage_cost=0.001,
                                    remaining_quota=100.0)

    monkeypatch.setattr(annotation, "_proxy_call", fake)
    monkeypatch.setattr(annotation.time, "sleep", lambda *_: None)
    return queue


def test_batch_marks_an_incomplete_response_unresolved(tmp_path, monkeypatch):
    chain = "Step one here. A dropped middle section of genuine reasoning. Step two."
    _mock_proxy(monkeypatch, ['["deduction"]Step one here.["end-section"]'
                              '["deduction"]Step two.["end-section"]'])
    out = tmp_path / "ann.json"
    result = annotation.annotate_chains(
        [{"task_id": "T1", "chain": chain}], save_path=out, dedup_keys=("task_id",)
    )
    assert result[0]["annotation_complete"] is True      # transport succeeded
    assert result[0]["annotation_coverage_complete"] is False
    assert result[0]["annotation_coverage"]["rule_version"] == COVERAGE_RULE_VERSION
    assert result[0]["annotation_coverage"]["n_gaps"] == 1


def test_resume_retries_coverage_incomplete_rows_and_skips_clean_ones(
    tmp_path, monkeypatch
):
    """The five defective rows must NOT be skipped as complete on resume."""
    clean = "All of this text is annotated."
    defective = "Kept sentence one. A dropped middle section of real reasoning. Kept two."
    chains = [{"task_id": "CLEAN", "chain": clean},
              {"task_id": "DEFECT", "chain": defective}]
    out = tmp_path / "ann.json"

    _mock_proxy(monkeypatch, [
        '["deduction"]All of this text is annotated.["end-section"]',
        '["deduction"]Kept sentence one.["end-section"]'
        '["deduction"]Kept two.["end-section"]',
    ])
    first = annotation.annotate_chains(chains, save_path=out, dedup_keys=("task_id",))
    assert [r["annotation_coverage_complete"] for r in first] == [True, False]

    # Model the 2026-08-11 V2 rows: written before the request digest existed,
    # so the deterministic-failure seal cannot apply and a resume MUST retry.
    saved = json.loads(out.read_text())
    for row in saved:
        row.pop("annotation_request_sha256", None)
    out.write_text(json.dumps(saved))

    # Resume: only DEFECT may be re-requested, and it now returns full coverage.
    calls = []

    def fake(prompt, **kwargs):
        calls.append(prompt)
        return annotation.ProxyText(
            '["deduction"]Kept sentence one.["end-section"]'
            '["deduction"]A dropped middle section of real reasoning.["end-section"]'
            '["deduction"]Kept two.["end-section"]',
            usage_cost=0.001, remaining_quota=100.0)

    monkeypatch.setattr(annotation, "_proxy_call", fake)
    second = annotation.annotate_chains(chains, save_path=out, dedup_keys=("task_id",))

    assert len(calls) == 1, "the clean row must not be re-requested"
    assert "Kept sentence one" in calls[0]
    by_id = {r["task_id"]: r for r in second}
    assert by_id["DEFECT"]["annotation_coverage_complete"] is True
    assert by_id["CLEAN"]["annotation_coverage_complete"] is True


def test_coverage_failure_gets_one_bounded_retry_then_seals(tmp_path, monkeypatch):
    """The proxy is not perfectly deterministic at temperature 0 (observed in
    the 2026-08-11 pilot), so a coverage-failed row earns exactly
    MAX_COVERAGE_ATTEMPTS attempts per request version, then seals."""
    chain = "Kept one. A dropped middle section of real reasoning. Kept two."
    out = tmp_path / "ann.json"
    partial = ('["deduction"]Kept one.["end-section"]'
               '["deduction"]Kept two.["end-section"]')
    _mock_proxy(monkeypatch, [partial])
    first = annotation.annotate_chains(
        [{"task_id": "T1", "chain": chain}], save_path=out, dedup_keys=("task_id",))
    assert first[0]["annotation_coverage_complete"] is False
    assert first[0]["annotation_coverage_attempts"] == 1

    # Attempt 2 IS allowed — and can succeed thanks to nondeterminism.
    calls = []

    def second_try(prompt, **kwargs):
        calls.append(prompt)
        return annotation.ProxyText(partial, usage_cost=0.001, remaining_quota=99.0)

    monkeypatch.setattr(annotation, "_proxy_call", second_try)
    second = annotation.annotate_chains(
        [{"task_id": "T1", "chain": chain}], save_path=out, dedup_keys=("task_id",))
    assert len(calls) == 1
    assert second[0]["annotation_coverage_attempts"] == 2
    assert second[0]["annotation_coverage_complete"] is False

    # Attempt 3 is NOT: the row is sealed, no call made.
    def explode(*args, **kwargs):
        raise AssertionError("re-requested a sealed coverage failure")

    monkeypatch.setattr(annotation, "_proxy_call", explode)
    third = annotation.annotate_chains(
        [{"task_id": "T1", "chain": chain}], save_path=out, dedup_keys=("task_id",))
    assert len(third) == 1
    assert third[0]["annotation_coverage_complete"] is False


def test_a_changed_request_reopens_a_sealed_coverage_failure(tmp_path, monkeypatch):
    """Editing the source (or the region rule changing the request) invalidates
    the stored verdict, so the row IS re-requested."""
    chain = "Kept one. A dropped middle section of real reasoning. Kept two."
    out = tmp_path / "ann.json"
    _mock_proxy(monkeypatch, ['["deduction"]Kept one.["end-section"]'
                              '["deduction"]Kept two.["end-section"]'])
    annotation.annotate_chains(
        [{"task_id": "T1", "chain": chain}], save_path=out, dedup_keys=("task_id",))

    grown = chain + " And one more closing thought."
    calls = []

    def fake(prompt, **kwargs):
        calls.append(prompt)
        return annotation.ProxyText(
            '["deduction"]Kept one.["end-section"]'
            '["deduction"]A dropped middle section of real reasoning.["end-section"]'
            '["deduction"]Kept two.["end-section"]'
            '["deduction"]And one more closing thought.["end-section"]',
            usage_cost=0.001, remaining_quota=100.0)

    monkeypatch.setattr(annotation, "_proxy_call", fake)
    second = annotation.annotate_chains(
        [{"task_id": "T1", "chain": grown}], save_path=out, dedup_keys=("task_id",))
    assert len(calls) == 1
    assert second[0]["annotation_coverage_complete"] is True


def test_transport_failures_are_still_retried_on_resume(tmp_path, monkeypatch):
    """Sealing applies ONLY to coverage failures: a transport-incomplete row
    (annotation_complete=False) is nondeterministic and must be retried."""
    chain = "All of this text is annotated."
    out = tmp_path / "ann.json"

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic 503")

    monkeypatch.setattr(annotation, "_proxy_call", fail)
    monkeypatch.setattr(annotation.time, "sleep", lambda *_: None)
    first = annotation.annotate_chains(
        [{"task_id": "T1", "chain": chain}], save_path=out, dedup_keys=("task_id",))
    assert first[0]["annotation_complete"] is False

    _mock_proxy(monkeypatch, ['["deduction"]All of this text is annotated.["end-section"]'])
    second = annotation.annotate_chains(
        [{"task_id": "T1", "chain": chain}], save_path=out, dedup_keys=("task_id",))
    assert second[0]["annotation_coverage_complete"] is True


def test_a_completed_row_is_never_requested_twice(tmp_path, monkeypatch):
    chain = "All of this text is annotated."
    _mock_proxy(monkeypatch, ['["deduction"]All of this text is annotated.["end-section"]'])
    out = tmp_path / "ann.json"
    chains = [{"task_id": "T1", "chain": chain}]
    annotation.annotate_chains(chains, save_path=out, dedup_keys=("task_id",))

    def explode(*args, **kwargs):
        raise AssertionError("duplicate call for an already-complete row")

    monkeypatch.setattr(annotation, "_proxy_call", explode)
    again = annotation.annotate_chains(chains, save_path=out, dedup_keys=("task_id",))
    assert len(again) == 1


# ── quarantine / migration ───────────────────────────────────────────────────

def _checkpoint(tmp_path):
    rows = [
        {"task_id": "CLEAN", "chain": "All of this text is annotated.",
         "annotations": [span("deduction", "All of this text is annotated.")],
         "annotation_complete": True},
        {"task_id": "DEFECT",
         "chain": "Kept one. A dropped middle section of real reasoning. Kept two.",
         "annotations": [span("deduction", "Kept one."), span("deduction", "Kept two.")],
         "annotation_complete": True},
    ]
    path = tmp_path / "base.json"
    path.write_text(json.dumps(rows, indent=2))
    return path


def test_quarantine_plan_is_side_effect_free_and_requires_approval(tmp_path):
    path = _checkpoint(tmp_path)
    before = path.read_bytes()

    plan = plan_quarantine(path)
    assert path.read_bytes() == before, "planning must not touch the checkpoint"
    assert [r["task_id"] for r in plan.quarantined] == ["DEFECT"]
    assert [r["task_id"] for r in plan.retained] == ["CLEAN"]

    with pytest.raises(QuarantineError, match="approval"):
        apply_quarantine(plan, tmp_path / "quarantine", approved=False)
    assert path.read_bytes() == before


def test_quarantine_preserves_originals_and_records_why(tmp_path):
    path = _checkpoint(tmp_path)
    plan = plan_quarantine(path)
    manifest = apply_quarantine(plan, tmp_path / "quarantine", approved=True)

    kept = json.loads(path.read_text())
    assert [r["task_id"] for r in kept] == ["CLEAN"]

    preserved = json.loads((tmp_path / "quarantine" / "base.v1.json").read_text())
    assert [r["task_id"] for r in preserved] == ["DEFECT"]
    # The ORIGINAL spans survive verbatim, with the reason attached.
    assert preserved[0]["annotations"] == [
        span("deduction", "Kept one."), span("deduction", "Kept two.")
    ]
    assert preserved[0]["annotation_coverage"]["reasons"]
    assert manifest["checkpoint_sha256_before"] != manifest["checkpoint_sha256_after"]
    assert manifest["coverage_rule_version"] == COVERAGE_RULE_VERSION
    assert manifest["n_quarantined"] == 1


def test_quarantine_refuses_if_the_checkpoint_changed_since_planning(tmp_path):
    path = _checkpoint(tmp_path)
    plan = plan_quarantine(path)
    path.write_text(json.dumps([{"task_id": "OTHER"}]))
    with pytest.raises(QuarantineError, match="changed since"):
        apply_quarantine(plan, tmp_path / "quarantine", approved=True)


def test_quarantine_versions_successive_migrations(tmp_path):
    path = _checkpoint(tmp_path)
    apply_quarantine(plan_quarantine(path), tmp_path / "q", approved=True)
    path.write_text(json.dumps([
        {"task_id": "DEFECT2", "chain": "Kept one. A dropped middle section here. Kept two.",
         "annotations": [span("deduction", "Kept one.")], "annotation_complete": True},
    ], indent=2))
    apply_quarantine(plan_quarantine(path), tmp_path / "q", approved=True)
    assert (tmp_path / "q" / "base.v1.json").exists()
    assert (tmp_path / "q" / "base.v2.json").exists()


# ── the per-call cost bound ──────────────────────────────────────────────────

RATE_CARD = AnnotationRateCard(input_usd_per_mtok=3.0, output_usd_per_mtok=15.0,
                               tokens_per_char_upper=0.4)


def test_the_superseded_4000_token_allowance_is_not_bounded_by_five_cents():
    """Why the run stopped: 4,000 output tokens alone prices above $0.05."""
    worst = RATE_CARD.worst_case_cost_usd(annotation.max_prompt_chars(), 4000)
    assert worst > 0.05


def test_the_corrected_allowance_is_bounded_by_five_cents():
    worst = RATE_CARD.worst_case_cost_usd(annotation.max_prompt_chars(), 2800)
    assert worst <= 0.05


def _guard_manifest(tmp_path, *, max_output_tokens, max_cost=0.05):
    source = tmp_path / "source.json"
    source.write_text('[{"chain":"bounded"}]')
    code = tmp_path / "code.py"
    code.write_text("# bound\n")
    doc = {
        "schema": SCHEMA_VERSION, "status": "authorised",
        "model": annotation.ANNOTATION_MODEL, "annotation_window_tokens": 3000,
        "coverage_rule_version": COVERAGE_RULE_VERSION,
        "rate_card": {"input_usd_per_mtok": 3.0, "output_usd_per_mtok": 15.0,
                      "tokens_per_char_upper": 0.4},
        "guards": {
            "planned_initial_requests": 2, "global_retry_capacity": 1,
            "max_total_attempts": 3, "max_attempts_per_scope": 2,
            "approved_spend_ceiling_usd": 10.0,
            "max_cost_per_attempt_usd": max_cost,
            "remaining_quota_floor_usd": 5.0,
            "max_output_tokens": max_output_tokens,
            "max_prompt_chars": annotation.max_prompt_chars(),
            "prior_committed_spend_usd": 0.0,
        },
        "source_artifacts": [{"path": "source.json",
                              "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
        "bound_artifacts": [{"path": "code.py",
                             "sha256": hashlib.sha256(code.read_bytes()).hexdigest()}],
    }
    manifest = tmp_path / "guard.json"
    manifest.write_text(json.dumps(doc, sort_keys=True))
    return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def _load(tmp_path, manifest, digest, max_output_tokens):
    return AnnotationAttemptGuard.from_manifest(
        manifest, digest, tmp_path / "journal.jsonl", root=tmp_path,
        expected_model=annotation.ANNOTATION_MODEL,
        expected_annotation_window_tokens=3000,
        expected_bound_paths=("code.py",),
        expected_max_output_tokens=max_output_tokens,
        expected_max_prompt_chars=annotation.max_prompt_chars(),
        expected_coverage_rule_version=COVERAGE_RULE_VERSION,
    )


def test_manifest_refuses_an_allowance_whose_worst_case_exceeds_authorisation(tmp_path):
    manifest, digest = _guard_manifest(tmp_path, max_output_tokens=4000)
    with pytest.raises(AnnotationAttemptLimitError, match="worst-case per-call cost"):
        _load(tmp_path, manifest, digest, 4000)


def test_manifest_refuses_a_per_call_ceiling_above_the_hard_project_limit(tmp_path):
    manifest, digest = _guard_manifest(tmp_path, max_output_tokens=2800, max_cost=0.50)
    with pytest.raises(AnnotationAttemptLimitError, match="hard project limit"):
        _load(tmp_path, manifest, digest, 2800)


def test_an_over_budget_call_is_refused_before_any_network_io(tmp_path, monkeypatch):
    """The decisive property: refusal happens with zero HTTP requests made."""
    manifest, digest = _guard_manifest(tmp_path, max_output_tokens=2800)
    guard = _load(tmp_path, manifest, digest, 2800)

    posted = []
    monkeypatch.setattr(annotation.requests, "post",
                        lambda *a, **k: posted.append(a) or (_ for _ in ()).throw(
                            AssertionError("network I/O attempted")))
    monkeypatch.setattr(annotation.time, "sleep", lambda *_: None)

    with pytest.raises(AnnotationCostLimitError, match="output allowance"):
        annotation._annotate_single(
            "chain", max_retries=1, max_tokens=8192, attempt_guard=guard,
            attempt_scope=hashlib.sha256(b"s").hexdigest(),
        )
    assert posted == []
    # And no attempt was consumed: a refused call never happened.
    assert guard.summary()["attempts_reserved"] == 0


def test_an_oversized_prompt_is_refused_before_any_network_io(tmp_path, monkeypatch):
    manifest, digest = _guard_manifest(tmp_path, max_output_tokens=2800)
    guard = _load(tmp_path, manifest, digest, 2800)
    monkeypatch.setattr(annotation.requests, "post",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("network I/O attempted")))
    with pytest.raises(AnnotationCostLimitError, match="prompt of"):
        guard.assert_attempt_cost_bound("x" * (annotation.max_prompt_chars() + 1), 2800)


def test_the_chunk_plan_never_builds_a_prompt_over_the_declared_bound():
    """max_prompt_chars must bound what chunk_chain actually produces."""
    limit = annotation.max_prompt_chars()
    envelope = len(annotation._PROMPT_TEMPLATE.format(thinking_process=""))
    envelope += len(annotation._CONTINUATION_PREFIX)
    for chain in ("word " * 20_000, "x" * 60_000, "para.\n\n" * 5_000):
        for chunk in annotation.chunk_chain(chain):
            assert len(chunk) + envelope <= limit


def test_prior_committed_spend_carries_across_a_manifest_replacement(tmp_path):
    """Replacing the manifest starts a fresh journal; spend must not reset."""
    manifest, digest = _guard_manifest(tmp_path, max_output_tokens=2800)
    doc = json.loads(manifest.read_text())
    doc["guards"]["prior_committed_spend_usd"] = 1.052535
    manifest.write_text(json.dumps(doc, sort_keys=True))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    guard = _load(tmp_path, manifest, digest, 2800)
    assert guard.summary()["committed_cost_usd"] == pytest.approx(1.052535)


# ── anchor on the frozen evidence ────────────────────────────────────────────

#: The live checkpoint is migrated by quarantine on an approved resume, so
#: the anchor reads the preserved read-only evidence copy of the halted run.
FROZEN = ".codex/out/PH2_ANNOTATION_V2_STOPPED_CHECKPOINT_2026-08-11.json"
FROZEN_SHA = "4273ff37ed62881c3aeb9aeed5ada525bcfe28796687400923626ea0887efb61"


@pytest.mark.skipif(not __import__("pathlib").Path(FROZEN).is_file(),
                    reason="frozen 2026-08-11 checkpoint not present")
def test_frozen_checkpoint_reproduces_the_audited_defects():
    """Anchor the validator on the real halted run rather than fixtures alone.

    The five gap rows are exactly those named in the 2026-08-11 audit; the two
    extra rows are the post-``</think>`` inconsistency the audit reported
    separately.  All fifteen were marked ``annotation_complete: true``.
    """
    from pathlib import Path
    raw = Path(FROZEN).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FROZEN_SHA, "frozen evidence altered"
    rows = json.loads(raw)

    assert all(r["annotation_complete"] for r in rows)
    gapped, outside = set(), set()
    for row in rows:
        report = validate_coverage(row["chain"], row["annotations"])
        if report.gaps:
            gapped.add(row["task_id"])
        if report.outside_region_spans:
            outside.add(row["task_id"])

    assert gapped == {"MATH_119", "MATH_121", "MATH_124", "MATH_126", "SPAT_120"}
    assert outside == {"MATH_120", "MATH_126", "SPAT_119", "SPAT_120"}
    assert not any(row_is_coverage_complete(r) for r in rows)
