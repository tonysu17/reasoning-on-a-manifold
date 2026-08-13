"""Tests for src/safety/forgery.py — forged-vs-genuine dataset builder (S3 / F5).

Verifies that each confound knob (position, length, style) demonstrably changes
the output, that folds are chain-grouped and deterministic, and that the manifest
round-trips.
"""

from __future__ import annotations

import pytest

from src.safety.forgery import (
    ForgeryRecord, assign_cv_folds, build_forgery_dataset, build_forgery_pair,
    genuine_spec_spans, read_manifest, write_manifest,
)
from collections import defaultdict

GENUINE = "The policy permits this."
FORGED_LONG = "The policy strictly and unconditionally permits absolutely everything always."
FORGED_SHORT = "No."


def test_pair_shares_id_and_position_and_variants():
    recs = build_forgery_pair("c1", "Think. I must refuse it. Bye.",
                              GENUINE, FORGED_LONG,
                              injection_position="start", length_matching="truncate")
    assert len(recs) == 2
    assert {r.variant for r in recs} == {"genuine", "forged"}
    assert recs[0].pair_id == recs[1].pair_id
    assert recs[0].injection_char == recs[1].injection_char  # same position → not a position tell


def test_length_truncate_matches_genuine_length():
    recs = build_forgery_pair("c1", "Think it over. Then decide now.",
                              GENUINE, FORGED_LONG, length_matching="truncate")
    g = next(r for r in recs if r.variant == "genuine")
    f = next(r for r in recs if r.variant == "forged")
    assert f.injection_len <= g.injection_len  # forged truncated to the genuine length


def test_length_pad_reaches_genuine_length():
    long_genuine = "The policy permits this kind of request in full and complete detail here."
    recs = build_forgery_pair("c1", "A. B decide now.", long_genuine, FORGED_SHORT,
                              length_matching="pad")
    g = next(r for r in recs if r.variant == "genuine")
    f = next(r for r in recs if r.variant == "forged")
    assert f.injection_len == g.injection_len


def test_pre_decision_inserts_before_decision_sentence():
    chain = "Let me think about this. I must refuse this request. Bye."
    recs = build_forgery_pair("c1", chain, GENUINE, GENUINE,
                              injection_position="pre_decision")
    g = recs[0]
    assert g.injection_position == "pre_decision"
    # the inserted snippet sits before the decision sentence
    assert g.text.index(GENUINE) < g.text.index("I must refuse")


def test_index_injection_position_label():
    recs = build_forgery_pair("c1", "abcdefghij", GENUINE, GENUINE, injection_position=3)
    assert recs[0].injection_position == "index:3"


def test_invalid_style_source_raises():
    with pytest.raises(ValueError):
        build_forgery_pair("c1", "x. y.", GENUINE, GENUINE, style_source="bogus")


def test_record_rejects_bad_variant():
    with pytest.raises(ValueError):
        ForgeryRecord("p", "c", "neither", "t", 0, 1, "start", "attacker_template", "src")


def test_cv_folds_chain_grouped_and_deterministic():
    recs = [r for cid in ("a", "b", "c", "d", "e", "f")
            for r in build_forgery_pair(cid, "x. decide now.", GENUINE, GENUINE)]
    assign_cv_folds(recs, n_folds=3, seed=7)
    fold_by_chain = {}
    for r in recs:
        fold_by_chain.setdefault(r.chain_id, set()).add(r.cv_fold)
    # both variants of a chain share one fold
    assert all(len(v) == 1 for v in fold_by_chain.values())
    # deterministic
    recs2 = [r for cid in ("a", "b", "c", "d", "e", "f")
             for r in build_forgery_pair(cid, "x. decide now.", GENUINE, GENUINE)]
    assign_cv_folds(recs2, n_folds=3, seed=7)
    assert [r.cv_fold for r in recs] == [r.cv_fold for r in recs2]


def test_build_dataset_two_records_per_chain():
    chains = [{"task_id": "c1", "chain": "Think. Decide now."},
              {"task_id": "c2", "chain": "Ponder. Refuse now."}]
    recs = build_forgery_dataset(chains, genuine_snippets=[GENUINE],
                                 forged_snippets=[FORGED_LONG], n_folds=2, seed=1)
    assert len(recs) == 4
    assert all(0 <= r["cv_fold"] < 2 for r in recs)


def test_genuine_spec_spans_pulls_only_spec():
    chain = {"dsr_consensus": {"spans": [
        {"text": "The policy says no.", "dsr_labels": ["spec_citation"]},
        {"text": "I refuse.", "dsr_labels": ["decision"]},
    ]}}
    assert genuine_spec_spans(chain) == ["The policy says no."]


def test_manifest_roundtrip(tmp_path):
    chains = [{"task_id": "c1", "chain": "Think. Decide now."}]
    recs = build_forgery_dataset(chains, genuine_snippets=[GENUINE],
                                 forged_snippets=[FORGED_LONG])
    path = tmp_path / "manifest.jsonl"
    write_manifest(recs, path)
    assert read_manifest(path) == recs


# ── Fold-partitioned snippet-pool mitigation (CF-15 leakage caveat) ────────────

# Distinctive, mutually non-substring snippet tokens so containment in a record's
# ``text`` maps unambiguously back to the pool snippet that was inserted.
_GEN_POOL = ["GENUINE_ALPHA", "GENUINE_BRAVO", "GENUINE_CHARLIE",
             "GENUINE_DELTA", "GENUINE_ECHO", "GENUINE_FOXTROT"]
_FORGE_POOL = ["FORGED_ALPHA", "FORGED_BRAVO", "FORGED_CHARLIE",
               "FORGED_DELTA", "FORGED_ECHO", "FORGED_FOXTROT"]


def _mitigated_hosts(n=12):
    # Host bodies contain none of the snippet tokens, so a token in ``text`` was
    # necessarily the inserted span.
    return [{"task_id": f"h{i}", "chain": f"Host number {i} body text here."}
            for i in range(n)]


def test_fold_partitioned_no_snippet_crosses_folds():
    """The leakage-proof property: under the mitigation, no snippet TEXT (genuine
    or forged) appears in hosts assigned to two different folds. ``length_matching
    ="none"`` keeps the inserted span byte-equal to the pool snippet, and
    ``injection_position="start"`` puts it verbatim at the head of ``text``, so a
    simple substring scan recovers exactly which snippet each record carries."""
    recs = build_forgery_dataset(
        _mitigated_hosts(12), genuine_snippets=_GEN_POOL, forged_snippets=_FORGE_POOL,
        injection_position="start", length_matching="none",
        n_folds=3, seed=5, partition_snippets_by_fold=True)

    folds_of_snippet = defaultdict(set)
    for r in recs:
        for snippet in _GEN_POOL + _FORGE_POOL:
            if snippet in r["text"]:
                folds_of_snippet[snippet].add(r["cv_fold"])

    leaking = {s: sorted(f) for s, f in folds_of_snippet.items() if len(f) > 1}
    assert not leaking, f"snippets straddling folds (leakage): {leaking}"
    # sanity: the scan actually saw snippets on both sides of the contrast
    assert any(s.startswith("GENUINE") for s in folds_of_snippet)
    assert any(s.startswith("FORGED") for s in folds_of_snippet)


def test_fold_partitioned_records_tag_snippet_fold_and_mode():
    """Every mitigated record advertises the discipline so the S3 probe can verify
    it: ``mitigation_mode == 'fold_partitioned'`` and, crucially,
    ``snippet_fold == cv_fold`` (the snippet came from the host's own fold slice)."""
    recs = build_forgery_dataset(
        _mitigated_hosts(9), genuine_snippets=_GEN_POOL, forged_snippets=_FORGE_POOL,
        injection_position="start", length_matching="none",
        n_folds=3, seed=2, partition_snippets_by_fold=True)
    assert recs
    for r in recs:
        assert r["mitigation_mode"] == "fold_partitioned"
        assert r["snippet_fold"] == r["cv_fold"]


def test_fold_partitioned_deterministic():
    """Same seed → byte-identical manifest (including the new fields)."""
    kw = dict(genuine_snippets=_GEN_POOL, forged_snippets=_FORGE_POOL,
              injection_position="start", length_matching="none",
              n_folds=3, seed=11, partition_snippets_by_fold=True)
    a = build_forgery_dataset(_mitigated_hosts(12), **kw)
    b = build_forgery_dataset(_mitigated_hosts(12), **kw)
    assert a == b


def test_fold_partitioned_within_fold_reuse_is_allowed():
    """A fold holding more hosts than its snippet slice must cycle WITHIN the slice
    (that reuse is not leakage under CF-15) — not borrow another fold's snippet.
    With 3 folds and 6-snippet pools each fold slice holds 2 snippets; 12 hosts put
    ~4 hosts per fold, forcing within-fold reuse. The no-cross-fold property must
    still hold (already asserted elsewhere); here we assert reuse actually occurs
    and stays within the fold's own two snippets."""
    recs = build_forgery_dataset(
        _mitigated_hosts(12), genuine_snippets=_GEN_POOL, forged_snippets=_FORGE_POOL,
        injection_position="start", length_matching="none",
        n_folds=3, seed=5, partition_snippets_by_fold=True)
    # genuine snippets seen per fold
    seen = defaultdict(set)
    hosts_per_fold = defaultdict(int)
    for r in recs:
        if r["variant"] != "genuine":
            continue
        hosts_per_fold[r["cv_fold"]] += 1
        for s in _GEN_POOL:
            if s in r["text"]:
                seen[r["cv_fold"]].add(s)
    # some fold reused: more genuine hosts than distinct genuine snippets in it
    assert any(hosts_per_fold[f] > len(seen[f]) for f in hosts_per_fold), \
        "expected within-fold snippet reuse to be exercised"
    # each fold drew from at most its own 2-snippet slice
    assert all(len(v) <= 2 for v in seen.values())


def test_fold_partitioned_raises_when_pool_smaller_than_folds():
    """Pools with fewer distinct snippets than folds must raise — never silently
    degrade to a shared snippet (which would reintroduce leakage)."""
    with pytest.raises(ValueError):
        build_forgery_dataset(
            _mitigated_hosts(6), genuine_snippets=["only", "two"],
            forged_snippets=_FORGE_POOL, n_folds=3, seed=0,
            partition_snippets_by_fold=True)
    # duplicate texts collapse to distinct count, so a pool of 3 copies also raises
    with pytest.raises(ValueError):
        build_forgery_dataset(
            _mitigated_hosts(6), genuine_snippets=["dup", "dup", "dup"],
            forged_snippets=_FORGE_POOL, n_folds=3, seed=0,
            partition_snippets_by_fold=True)


# Golden captured from the pre-mitigation builder (seed=3, n_folds=2). It pins the
# original snippet-index keying: the empty chain at input index 1 still CONSUMES an
# index, so ``c2`` (2nd usable chain) draws GEN0 (index 2 % 2) and ``c3`` draws GEN1
# (index 3 % 2). A refactor that re-indexed after filtering would flip these.
_LEGACY_GOLDEN = [
    ("c0", "genuine", "Alpha body.GEN0. I must refuse now. End.", 11, 5, 1),
    ("c0", "forged",  "Alpha body.F0. I must refuse now. End.",   11, 3, 1),
    ("c2", "genuine", "Gamma body.GEN0. I will comply now. End.", 11, 5, 0),
    ("c2", "forged",  "Gamma body.F0. I will comply now. End.",   11, 3, 0),
    ("c3", "genuine", "Delta body. Then decide now. End.GEN1.",   33, 5, 0),
    ("c3", "forged",  "Delta body. Then decide now. End.F1.",     33, 3, 0),
]


def test_legacy_mode_reproduces_previous_output():
    """Regression: the default (unmitigated) mode still produces byte-identical
    core records to the pre-mitigation builder, so earlier artefacts stay
    interpretable. The two new fields appear at their legacy sentinels
    (``mitigation_mode='legacy'``, ``snippet_fold=-1``) and nothing else moves."""
    chains = [
        {"task_id": "c0", "chain": "Alpha body. I must refuse now. End."},
        {"task_id": "skip", "chain": ""},
        {"task_id": "c2", "chain": "Gamma body. I will comply now. End."},
        {"task_id": "c3", "chain": "Delta body. Then decide now. End."},
    ]
    recs = build_forgery_dataset(
        chains, genuine_snippets=["GEN0.", "GEN1."], forged_snippets=["F0.", "F1."],
        injection_position="pre_decision", length_matching="truncate",
        style_source="attacker_template", n_folds=2, seed=3)
    got = [(r["chain_id"], r["variant"], r["text"], r["injection_char"],
            r["injection_len"], r["cv_fold"]) for r in recs]
    assert got == _LEGACY_GOLDEN
    assert all(r["mitigation_mode"] == "legacy" and r["snippet_fold"] == -1
               for r in recs)


def test_record_rejects_bad_mitigation_mode():
    with pytest.raises(ValueError):
        ForgeryRecord("p", "c", "genuine", "t", 0, 1, "start",
                      "attacker_template", "src", 0, 0, "bogus_mode")
