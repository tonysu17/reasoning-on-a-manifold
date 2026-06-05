"""Tests for the canonical sentence-offset locator (AUDIT.md §5 dedup).

Locks in (a) the matching behaviour, and (b) that every former copy site now
resolves to the SAME function object, so the four duplicates can't drift and
silently misalign chain-id arrays with activation rows.
"""

import importlib

from src.text_offsets import find_sentence_offset as canon


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


def test_all_former_copies_are_the_same_object():
    sites = {
        "src.activation_extraction": "_find_sentence_offset",
        "05_pca_analysis": "_find_sentence_offset",
        "05b_geometric_diagnostics": "_find_sentence_offset",
        "compare_annotators": "_off",
        "robustness_geometry": "_off",
    }
    for mod_name, attr in sites.items():
        mod = importlib.import_module(mod_name)
        assert getattr(mod, attr) is canon, f"{mod_name}.{attr} drifted from canonical"
