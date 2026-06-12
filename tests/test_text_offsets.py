"""Tests for the canonical sentence-offset locator (AUDIT.md §5 dedup) and the
occurrence-aware whole-chain locator (CONFOUNDS_AND_REMEDIATION.md CF-13).

Locks in (a) the matching behaviour, (b) that the remaining alias sites resolve
to the SAME function object so they can't drift, and (c) the char→token mapper
in activation_extraction (previously untested).
"""

import importlib

from src.text_offsets import find_sentence_offset as canon
from src.text_offsets import locate_annotation_offsets


def test_exact_match():
    assert canon("hello world", "world") == 6


def test_prefix_fallback_for_long_sentence():
    # Sentence > 40 chars whose first 40 appear in the chain but whose full text
    # does not (the annotator truncated the tail) -> match on the 40-char prefix.
    sentence = "A" * 45 + "_realtail_that_differs"
    chain = "junk_" + "A" * 45 + "_WRONGTAIL"
    assert canon(chain, sentence) == 5


def test_returns_none_when_absent():
    assert canon("abc", "z" * 20) is None


def test_short_sentence_no_false_prefix_match():
    # prefix rule requires >= 10 chars; a short miss returns None.
    assert canon("abcdef", "xyz") is None


def test_start_cursor_finds_later_occurrence():
    chain = "Wait, let me reconsider. Some math. Wait, let me reconsider. More."
    first = canon(chain, "Wait, let me reconsider.")
    assert first == 0
    second = canon(chain, "Wait, let me reconsider.", start=first + 1)
    assert second == chain.index("Wait, let me reconsider.", 1)


def test_start_cursor_falls_back_to_full_search():
    # Sentence occurs only BEFORE the cursor -> from-the-start fallback keeps
    # it locatable (out-of-order annotations must not become unlocatable).
    chain = "alpha beta gamma"
    assert canon(chain, "alpha", start=10) == 0


# ── locate_annotation_offsets (occurrence-aware whole-chain rule) ────────────

def test_locate_repeats_bind_to_successive_occurrences():
    s = "Wait, that seems wrong, let me check."
    chain = f"Setup. {s} Some deduction here. {s} Conclusion."
    offsets = locate_annotation_offsets(chain, ["Setup.", s, "Some deduction here.", s])
    assert offsets[0] == 0
    assert offsets[1] == chain.index(s)                      # first occurrence
    assert offsets[3] == chain.index(s, offsets[1] + 1)      # SECOND occurrence
    assert offsets[1] != offsets[3]


def test_locate_unique_sentences_match_per_sentence_rule():
    chain = "one fish. two fish. red fish."
    sents = ["one fish.", "two fish.", "red fish."]
    assert locate_annotation_offsets(chain, sents) == [0, 10, 20]
    # identical result to the legacy per-sentence rule when nothing repeats
    assert [canon(chain, s) for s in sents] == [0, 10, 20]


def test_locate_none_pattern_matches_per_sentence_rule():
    # The skip/keep pattern is what keeps legacy row reconstruction aligned:
    # absent stays absent, out-of-order falls back to a full search.
    chain = "alpha beta gamma"
    sents = ["beta", "totally absent sentence xyz", "alpha"]
    got = locate_annotation_offsets(chain, sents)
    assert got[0] == 6
    assert got[1] is None
    assert got[2] == 0


def test_locate_triple_repeat():
    s = "I should double check this."
    block = s + " filler one. "
    chain = block * 3
    assert locate_annotation_offsets(chain, [s, s, s]) == [0, len(block), 2 * len(block)]


# ── alias sites can't drift ──────────────────────────────────────────────────

def test_alias_sites_are_the_same_object():
    # 05/05b/robustness_geometry now route through src.row_provenance (which
    # calls locate_annotation_offsets); these two still alias the per-sentence
    # canon directly.
    sites = {
        "src.activation_extraction": "_find_sentence_offset",
        "compare_annotators": "_off",
    }
    for mod_name, attr in sites.items():
        mod = importlib.import_module(mod_name)
        assert getattr(mod, attr) is canon, f"{mod_name}.{attr} drifted from canonical"


def test_extraction_uses_occurrence_aware_locator():
    mod = importlib.import_module("src.activation_extraction")
    assert getattr(mod, "locate_annotation_offsets") is locate_annotation_offsets


# ── char→token mapper (previously untested) ──────────────────────────────────

def _mapper():
    from src.activation_extraction import _sentence_to_token_positions
    return _sentence_to_token_positions


def test_token_positions_basic_window():
    f = _mapper()
    # 10 tokens of width 4: [(0,4),(4,8),...,(36,40)]
    offsets = [(i * 4, i * 4 + 4) for i in range(10)]
    # sentence starts at char 12 -> onset token 3; 1 preceding + 3 execution
    assert f("x" * 40, 12, offsets, n_preceding=1, n_execution=3) == [2, 3, 4, 5]


def test_token_positions_onset_rule_mid_token():
    f = _mapper()
    offsets = [(0, 4), (4, 8), (8, 12)]
    # char 5 falls INSIDE token 1 (4..8): first token whose end exceeds 5 is 1
    assert f("x" * 12, 5, offsets, n_preceding=0, n_execution=1) == [1]


def test_token_positions_clips_preceding_at_zero():
    f = _mapper()
    offsets = [(0, 4), (4, 8), (8, 12)]
    # onset at token 0; 2 preceding requested but none exist
    assert f("x" * 12, 0, offsets, n_preceding=2, n_execution=2) == [0, 1]


def test_token_positions_clips_execution_at_sequence_end():
    f = _mapper()
    offsets = [(0, 4), (4, 8), (8, 12)]
    # onset token 2 (last); 10 execution requested, only 1 available
    assert f("x" * 12, 8, offsets, n_preceding=0, n_execution=10) == [2]


def test_token_positions_beyond_text_returns_empty():
    f = _mapper()
    offsets = [(0, 4), (4, 8)]
    # sentence start char beyond every token end -> unlocatable
    assert f("x" * 8, 99, offsets, n_preceding=1, n_execution=5) == []
