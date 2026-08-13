"""Deterministic annotation-region definition and semantic coverage validation.

Phase-2 QA 2026-08-11 found that ``annotation_complete`` only asserted that
every API chunk returned *at least one* parsed span.  Fifteen retained rows all
passed structural/provenance checks while five of them silently dropped
material reasoning, including a fourth occurrence of a looped sentence that had
been annotated three times (MATH_119).  Because ``behaviour_fraction`` divides
by the number of *returned* spans, a discretionary omission moves the estimand.

This module supplies the two things that were missing:

1.  A deterministic **annotation region** — the annotator never decides
    row-by-row whether a tail is in scope (:func:`annotation_region`).
2.  A **coverage validator** that requires every returned span to be
    source-faithful, in order, and jointly exhaustive over that region, with
    repeated reasoning kept as separate observations (:func:`validate_coverage`).

The region rule is versioned.  Callers that persist results must record
:data:`COVERAGE_RULE_VERSION` so a later rule change is a migration, never a
silent reinterpretation of existing rows.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

#: Bump when the region definition or the completeness criterion changes.
#: Persisted next to every coverage verdict; a mismatch forces re-validation.
#: -4 (2026-08-11, owner decision "Go with B"): the region is truncated at the
#: onset of DEGENERATE repetition — a cycle of <= 12 whitespace tokens repeated
#: >= 6 times consecutively keeps its first 5 occurrences and excludes the
#: rest.  Rationale: the annotator provably refuses to echo "Wait, no," x100
#: blocks (46/165 rows), and that refusal correlates with loop-heaviness
#: (median 4-gram repetition 0.766 vs 0.175), i.e. differential missingness on
#: the primary backtracking endpoint across steering arms.  Loop behaviour
#: itself stays measured full-chain by the existing looped/repetition-rate
#: endpoints (where E9 measured it).  Interleaved repetition — the same
#: sentence recurring with reasoning in between (MATH_117's 31 occurrences,
#: MATH_119's 4) — does NOT trigger; only adjacent cycling does.
#: -3 (2026-08-11): spacing around punctuation is canonicalised — R1-1.5B
#: chains contain "word,word" run-ons that the annotator echoes as "word, word";
#: both sides now normalise to the latter.
#: -2 (2026-08-11, mid-run correction): normalisation folds unicode
#: (NFKC + typographic quotes/dashes) — the annotator echoes ``’`` as ``'``,
#: which exact matching rejected as unfaithful; and spans consisting of the
#: echoed continuation-prefix instruction are excluded rather than unmatched.
COVERAGE_RULE_VERSION = "ph2-annotation-coverage-4"

#: Closing marker of the model's reasoning region.
THINK_CLOSE = "</think>"

#: A trailing "**Final Answer** ..." block is the model's answer restatement,
#: not reasoning.  It is excluded ONLY when it terminates the reasoning region
#: (never mid-chain), so the exclusion cannot swallow reasoning that continues
#: after it.
_FINAL_ANSWER_RE = re.compile(r"\*\*\s*Final Answer\s*\*\*")

#: Upper bound on an excludable Final-Answer suffix.  The exclusion exists to
#: drop a short answer restatement, so anything longer is treated as material
#: that must still be annotated.  This fails safe: an unusually long trailing
#: block stays IN scope and shows up as a coverage gap, rather than being
#: silently deleted — the exact failure class this module exists to prevent.
#: The two occurrences in the frozen 2026-08-11 checkpoint are 187 and 144
#: characters.
MAX_FINAL_ANSWER_SUFFIX_CHARS = 400

#: An uncovered run shorter than this (after stripping) is treated as
#: delimiter/punctuation slack rather than omitted reasoning.  Calibrated on
#: the 2026-08-11 retained checkpoint: the ten clean rows produce no uncovered
#: run at or above this length, and every known defect is far above it.
MIN_MATERIAL_GAP_CHARS = 25

_WS_RE = re.compile(r"\s+")

#: Degenerate-repetition truncation (rule -4).  A cycle of at most
#: MAX_CYCLE_TOKENS whitespace tokens repeated at least DEGENERATE_MIN_REPEATS
#: times consecutively marks degenerate output; the region keeps the first
#: DEGENERATE_KEEP_REPEATS occurrences and ends there.  Calibrated so genuine
#: repeated reasoning survives: the frozen checkpoint's MATH_119 (4 adjacent
#: occurrences, all annotation-worthy) stays in scope, while the live run's
#: LATE_118/126 and CAUS_121 ("Wait, no," cycled 100+ times) are cut.
MAX_CYCLE_TOKENS = 12
DEGENERATE_MIN_REPEATS = 6
DEGENERATE_KEEP_REPEATS = 5


def degenerate_cut(text: str) -> int | None:
    """Offset where degenerate cycling begins to be excluded, else ``None``.

    Scans whitespace tokens for the first position where a cycle of
    1..MAX_CYCLE_TOKENS tokens repeats >= DEGENERATE_MIN_REPEATS times back to
    back.  Returns the character offset of occurrence KEEP+1 — everything
    before it (the first KEEP occurrences included) stays in the region.
    Deterministic in the source text alone; never depends on annotator output.
    """
    tokens = []
    offsets = []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group(0))
        offsets.append(m.start())
    n = len(tokens)
    for i in range(n):
        for g in range(1, MAX_CYCLE_TOKENS + 1):
            if i + g * DEGENERATE_MIN_REPEATS > n:
                break
            if tokens[i] != tokens[i + g]:
                continue
            cycle = tokens[i:i + g]
            repeats = 1
            j = i + g
            while j + g <= n and tokens[j:j + g] == cycle:
                repeats += 1
                j += g
            if repeats >= DEGENERATE_MIN_REPEATS:
                return offsets[i + g * DEGENERATE_KEEP_REPEATS]
    return None


#: Canonicalise spacing after sentence punctuation (see rule -3 note above).
_PUNCT_SPACING_RE = re.compile(r"\s*([,.;:!?])\s*")

#: Canonical continuation-prefix instruction prepended to chunks 2+ by
#: ``src.annotation`` (which imports it from here — single source of truth).
#: The annotator sometimes echoes it as a labelled span (observed SPAT_126,
#: 2026-08-11); that echo is deterministic markup, not chain content, so the
#: validator strips it from spans rather than failing the row for it.
CONTINUATION_PREFIX = (
    "This is a continuation of a reasoning chain. "
    "Earlier portions have already been processed; label only the sentences in this excerpt.\n\n"
)

#: Typographic variants the annotator provably flattens when echoing (observed
#: 2026-08-11: ``Let’s`` returned as ``Let's``).  Folded on BOTH sides before
#: matching, so a faithful echo is never rejected for typography while a model
#: that drops or rewrites words still fails.
_UNICODE_FOLDS = str.maketrans({
    "’": "'", "‘": "'",          # curly single quotes
    "“": '"', "”": '"',          # curly double quotes
    "−": "-", "–": "-", "—": "-",  # minus, en-, em-dash
    " ": " ",                          # no-break space
})


def normalise(text: str) -> str:
    """Deterministic matching space: NFKC, typographic folds, whitespace.

    Idempotent.  All region offsets and all span matching in this module are
    defined in this normalised space, so a model that re-wraps lines or
    flattens curly quotes is still scored as source-faithful while a model
    that drops words is not.
    """
    text = unicodedata.normalize("NFKC", text).translate(_UNICODE_FOLDS)
    text = _PUNCT_SPACING_RE.sub(lambda m: m.group(1) + " ", text)
    return _WS_RE.sub(" ", text).strip()


@dataclass(frozen=True)
class Exclusion:
    """A span of the window deliberately outside the annotation region."""

    kind: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class AnnotationRegion:
    """The deterministic scientific annotation region of one chain window."""

    rule_version: str
    normalised_window: str
    region_text: str
    exclusions: tuple[Exclusion, ...] = ()

    @property
    def text(self) -> str:
        return self.region_text

    @property
    def n_chars(self) -> int:
        return len(self.region_text)

    @property
    def estimated_tokens(self) -> int:
        """Token estimate over the region actually annotated.

        Must be used as the per-1k denominator: pairing a region-restricted
        numerator with a whole-window denominator understates every rate.
        Matches ``src.annotation``'s 4-chars-per-token convention.
        """
        return self.n_chars // 4


def annotation_region(
    window_text: str,
    *,
    include_post_think: bool = False,
    exclude_final_answer_suffix: bool = True,
) -> AnnotationRegion:
    """Return the deterministic annotation region of ``window_text``.

    The default rule (draft Amendment A5, pending owner approval):

    * The region starts at the beginning of the A4 window.
    * The region ends at the first :data:`THINK_CLOSE` marker.  The marker and
      everything after it — the model's user-facing response — are excluded.
      When the window contains no marker, the whole window is reasoning.
    * A trailing ``**Final Answer**`` block inside the reasoning region is
      excluded as an answer restatement.

    ``include_post_think=True`` selects the opposite convention (response text
    inside the region) for callers that must reproduce it.  Either way the
    choice is made once, by configuration, never per row by the annotator.
    """
    normalised = normalise(window_text)
    raw_region, raw_exclusions = _raw_region(
        window_text,
        include_post_think=include_post_think,
        exclude_final_answer_suffix=exclude_final_answer_suffix,
    )
    # The scored region is DERIVED from the request slice, so the two agree by
    # construction — all cuts happen once, in raw space, in _raw_region. (An
    # earlier revision computed the degenerate cut independently in each space
    # and diverged on 108/3,600 corpus rows where the punctuation fold changes
    # tokenisation.)
    region_text = normalise(raw_region)
    exclusions = tuple(
        Exclusion(kind, len(region_text), len(normalised), normalise(chunk))
        for kind, chunk in raw_exclusions
    )
    return AnnotationRegion(
        rule_version=COVERAGE_RULE_VERSION,
        normalised_window=normalised,
        region_text=region_text,
        exclusions=exclusions,
    )


def region_source_text(
    window_text: str,
    *,
    include_post_think: bool = False,
    exclude_final_answer_suffix: bool = True,
) -> str:
    """The annotation region as a slice of the ORIGINAL text.

    :func:`annotation_region` works in normalised space, which is right for
    scoring but wrong for the request: the chunker splits on ``\\n\\n``, so
    sending it normalised text would collapse every paragraph boundary.

    This applies the same cuts to the raw string, preserving formatting. It is
    what must actually be sent to the annotator — excluded material the model
    never sees is material it cannot mislabel. Validating a region the request
    still contained would reject ~30% of this corpus on arrival, forever.
    """
    region, _ = _raw_region(
        window_text,
        include_post_think=include_post_think,
        exclude_final_answer_suffix=exclude_final_answer_suffix,
    )
    return region


def _raw_region(
    window_text: str,
    *,
    include_post_think: bool,
    exclude_final_answer_suffix: bool,
) -> tuple[str, list[tuple[str, str]]]:
    """All region cuts, applied once, in raw space.

    Returns the raw region slice plus (kind, excluded_text) pairs. Both the
    request builder and the scorer consume this single result, which is what
    guarantees the annotator is scored exactly on what it was sent.
    """
    exclusions: list[tuple[str, str]] = []
    cut = len(window_text)
    if not include_post_think:
        marker = window_text.find(THINK_CLOSE)
        if marker >= 0:
            exclusions.append(("post_think_response", window_text[marker:cut]))
            cut = marker
    degen = degenerate_cut(window_text[:cut])
    if degen is not None:
        exclusions.append(("degenerate_repetition", window_text[degen:cut]))
        cut = degen
    if exclude_final_answer_suffix:
        matches = list(_FINAL_ANSWER_RE.finditer(window_text, 0, cut))
        if matches:
            start = matches[-1].start()
            if len(normalise(window_text[start:cut])) <= MAX_FINAL_ANSWER_SUFFIX_CHARS:
                exclusions.append(("final_answer_suffix", window_text[start:cut]))
                cut = start
    return window_text[:cut].rstrip(), exclusions


@dataclass(frozen=True)
class Gap:
    """Region text no returned span accounts for."""

    start: int
    end: int
    text: str

    @property
    def n_chars(self) -> int:
        return len(self.text.strip())


@dataclass(frozen=True)
class CoverageReport:
    """Verdict on whether returned spans exhaust the annotation region."""

    rule_version: str
    complete: bool
    region_chars: int
    covered_chars: int
    n_spans: int
    matched_spans: int
    unmatched_spans: tuple[int, ...] = ()
    out_of_order_spans: tuple[int, ...] = ()
    outside_region_spans: tuple[int, ...] = ()
    gaps: tuple[Gap, ...] = ()
    reasons: tuple[str, ...] = field(default=())

    @property
    def coverage_fraction(self) -> float:
        return self.covered_chars / self.region_chars if self.region_chars else 1.0

    def to_dict(self) -> dict:
        """JSON-serialisable summary for persistence in the checkpoint."""
        return {
            "rule_version": self.rule_version,
            "complete": self.complete,
            "region_chars": self.region_chars,
            "covered_chars": self.covered_chars,
            "coverage_fraction": round(self.coverage_fraction, 6),
            "n_spans": self.n_spans,
            "matched_spans": self.matched_spans,
            "unmatched_spans": list(self.unmatched_spans),
            "out_of_order_spans": list(self.out_of_order_spans),
            "outside_region_spans": list(self.outside_region_spans),
            "n_gaps": len(self.gaps),
            "largest_gap_chars": max((g.n_chars for g in self.gaps), default=0),
            "gap_previews": [g.text.strip()[:120] for g in self.gaps[:5]],
            "reasons": list(self.reasons),
        }


def validate_coverage(
    window_text: str,
    spans: list[dict],
    *,
    include_post_think: bool = False,
    exclude_final_answer_suffix: bool = True,
    min_material_gap_chars: int = MIN_MATERIAL_GAP_CHARS,
) -> CoverageReport:
    """Check that ``spans`` exhaust the annotation region of ``window_text``.

    Matching is a strict forward scan in normalised space.  The cursor only
    moves forward, so the *k*-th occurrence of a repeated sentence matches the
    *k*-th position in the source: genuine repeated reasoning is retained as
    separate observations and is never deduplicated.  A span that exists in the
    region but only *before* the cursor is an ordering violation, not a match;
    a span that exists nowhere in the region is unmatched.

    A report is ``complete`` only when every span matched in order and no
    uncovered run of at least ``min_material_gap_chars`` remains.  Schema
    validity alone is never sufficient.
    """
    region = annotation_region(
        window_text,
        include_post_think=include_post_think,
        exclude_final_answer_suffix=exclude_final_answer_suffix,
    )
    text = region.text
    cursor = 0
    intervals: list[tuple[int, int]] = []
    unmatched: list[int] = []
    out_of_order: list[int] = []
    outside: list[int] = []

    prefix_needle = normalise(CONTINUATION_PREFIX)
    for index, span in enumerate(spans):
        needle = normalise(str(span.get("text", "")))
        if needle.startswith(prefix_needle):
            # Echoed continuation-prefix instruction: deterministic markup,
            # never chain content. Strip it; an empty remainder is simply
            # ignored rather than counted against the row.
            needle = needle[len(prefix_needle):].lstrip()
            if not needle:
                continue
        if not needle:
            unmatched.append(index)
            continue
        hit = text.find(needle, cursor)
        if hit >= 0:
            intervals.append((hit, hit + len(needle)))
            cursor = hit + len(needle)
            continue
        if text.find(needle) >= 0:
            out_of_order.append(index)
        elif region.normalised_window.find(needle) >= 0:
            # Source-faithful, but drawn from excluded material (e.g. the
            # post-</think> response). Under the frozen region rule this span
            # does not belong to the estimand.
            outside.append(index)
        else:
            unmatched.append(index)

    covered = 0
    gaps: list[Gap] = []
    position = 0
    for start, stop in intervals:
        if start > position:
            chunk = text[position:start]
            if len(chunk.strip()) >= min_material_gap_chars:
                gaps.append(Gap(position, start, chunk))
        covered += stop - max(start, position)
        position = max(position, stop)
    if position < len(text):
        chunk = text[position:]
        if len(chunk.strip()) >= min_material_gap_chars:
            gaps.append(Gap(position, len(text), chunk))

    reasons: list[str] = []
    if unmatched:
        reasons.append(f"{len(unmatched)} span(s) not found in the annotation region")
    if out_of_order:
        reasons.append(f"{len(out_of_order)} span(s) out of source order")
    if outside:
        reasons.append(f"{len(outside)} span(s) drawn from excluded material")
    if gaps:
        reasons.append(
            f"{len(gaps)} unexplained gap(s), largest "
            f"{max(g.n_chars for g in gaps)} chars"
        )

    return CoverageReport(
        rule_version=COVERAGE_RULE_VERSION,
        complete=not reasons,
        region_chars=len(text),
        covered_chars=covered,
        n_spans=len(spans),
        matched_spans=len(intervals),
        unmatched_spans=tuple(unmatched),
        out_of_order_spans=tuple(out_of_order),
        outside_region_spans=tuple(outside),
        gaps=tuple(gaps),
        reasons=tuple(reasons),
    )


def row_coverage_excludes(record: dict) -> bool:
    """Analysis-side predicate: should this row be withheld from estimation?

    True when the row CARRIES a coverage verdict and that verdict is stale or
    incomplete.  Records with no verdict at all (legacy E8/Phase-7 corpora,
    written before coverage validation existed) return False — their historical
    inclusion semantics are unchanged, so this predicate is safe inside shared
    extractors like ``src.delta_floor.per_task_fraction``.  Phase-2 rows always
    carry verdicts, so for them this is exactly "not coverage-complete".
    """
    verdict = record.get("annotation_coverage")
    if not isinstance(verdict, dict):
        return False
    if verdict.get("rule_version") != COVERAGE_RULE_VERSION:
        return True
    return not verdict.get("complete", False)


def row_is_coverage_complete(record: dict) -> bool:
    """Resume/analysis predicate for one annotated record.

    A row counts as coverage-complete only when the transport completed *and*
    a coverage verdict from the current rule version says the region was
    exhausted.  Rows written before coverage validation existed carry no
    verdict and are therefore NOT complete — they are eligible for
    reannotation rather than silently trusted.
    """
    if not record.get("annotation_complete", False):
        return False
    verdict = record.get("annotation_coverage")
    if not isinstance(verdict, dict):
        return bool(record.get("annotation_coverage_complete", False)) and (
            record.get("annotation_coverage_rule_version") == COVERAGE_RULE_VERSION
        )
    if verdict.get("rule_version") != COVERAGE_RULE_VERSION:
        return False
    return bool(verdict.get("complete", False))
