"""Unit tests for src/cbs/matching.py (M4)."""

from __future__ import annotations

import numpy as np
import pytest

from src.cbs import matching as M


def test_matching_module_importable() -> None:
    from src.cbs import matching  # noqa: F401


# ── Jaccard ─────────────────────────────────────────────────────────────

def test_jaccard_basic() -> None:
    assert M.jaccard_token_similarity("apply Fermat", "Fermat is great") > 0
    assert M.jaccard_token_similarity("apple", "banana") == 0.0
    assert M.jaccard_token_similarity("", "anything") == 0.0
    assert M.jaccard_token_similarity("apply Fermat", "apply Fermat") == 1.0


def test_jaccard_is_symmetric_and_lowercased() -> None:
    s1 = "Apply Fermat's Little Theorem"
    s2 = "fermat is the key here"
    a = M.jaccard_token_similarity(s1, s2)
    b = M.jaccard_token_similarity(s2, s1)
    assert a == b
    assert a > 0


# ── Matched-pair construction ───────────────────────────────────────────

def _toy_chain(task_id: str, sentences: list[tuple[int, str]]) -> dict:
    """sentences: list of (tier, text); tiers 0 are non-targeted spans."""
    spans = []
    for tier, text in sentences:
        span = {"label": "adding-knowledge", "text": text}
        if tier > 0:
            span["cbs_tier"] = tier
        spans.append(span)
    return {"task_id": task_id, "annotations": spans}


def test_matched_pair_construction() -> None:
    success = _toy_chain("TASK1", [(0, "intro"),
                                    (3, "apply Fermat little theorem here"),
                                    (3, "this completes the proof")])
    failure = _toy_chain("TASK1", [(0, "intro"),
                                    (3, "apply Fermat little theorem now"),
                                    (3, "but this fails to terminate")])
    pairs = M.build_matched_pairs([success], [failure],
                                  cbs_tier_filter=3,
                                  similarity_threshold=0.5)
    # First success tier-3 should match first failure tier-3.
    assert len(pairs) >= 1
    first = pairs[0]
    assert first["task_id"] == "TASK1"
    assert first["jaccard"] >= 0.5
    assert "TASK1" in first["success_sentence_id"]


def test_matched_pair_skips_below_threshold() -> None:
    success = _toy_chain("TASK1", [(3, "apply Fermat")])
    failure = _toy_chain("TASK1", [(3, "nothing in common at all")])
    pairs = M.build_matched_pairs([success], [failure],
                                  cbs_tier_filter=3,
                                  similarity_threshold=0.6)
    assert len(pairs) == 0


def test_matched_pair_requires_same_task_id() -> None:
    success = _toy_chain("TASK1", [(3, "apply Fermat")])
    failure = _toy_chain("TASK2", [(3, "apply Fermat")])
    pairs = M.build_matched_pairs([success], [failure],
                                  cbs_tier_filter=3,
                                  similarity_threshold=0.5)
    assert pairs == []


# ── Verification gradient ───────────────────────────────────────────────

def test_verification_gradient_cv_accuracy() -> None:
    rng = np.random.default_rng(0)
    d = 32
    correct = rng.normal(loc=+1.0, scale=0.5, size=(50, d))
    incorrect = rng.normal(loc=-1.0, scale=0.5, size=(50, d))
    out = M.verification_gradient(correct, incorrect,
                                  cv_folds=5, seed=0)
    assert out["cv_accuracy_mean"] > 0.85
    assert out["cv_accuracy_std"] < 0.15
    assert out["stable"] is True
    assert out["probe_weights"].shape == (d,)
    # Normalised
    assert abs(np.linalg.norm(out["probe_weights"]) - 1.0) < 1e-6


def test_verification_gradient_no_separation_low_accuracy() -> None:
    rng = np.random.default_rng(1)
    d = 16
    correct = rng.normal(size=(50, d))
    incorrect = rng.normal(size=(50, d))
    out = M.verification_gradient(correct, incorrect, cv_folds=5, seed=0)
    # Random labels should give ~50% accuracy.
    assert 0.3 < out["cv_accuracy_mean"] < 0.7


def test_verification_gradient_handles_empty() -> None:
    d = 5
    out = M.verification_gradient(np.zeros((0, d)), np.zeros((0, d)))
    assert out["cv_accuracy_mean"] == 0.0
    assert out["stable"] is False


# ── Paired geometric tests ──────────────────────────────────────────────

def test_paired_geometric_tests_empty_pairs() -> None:
    out = M.paired_geometric_tests([], {}, layer=17)
    assert out["n_pairs"] == 0


def test_paired_geometric_tests_runs() -> None:
    rng = np.random.default_rng(2)
    d = 16
    pairs = [
        {"task_id": "T1",
         "success_sentence_id": "T1:1", "failure_sentence_id": "T1:2",
         "success_chain_id": "T1", "failure_chain_id": "T1", "jaccard": 0.8},
        {"task_id": "T2",
         "success_sentence_id": "T2:1", "failure_sentence_id": "T2:2",
         "success_chain_id": "T2", "failure_chain_id": "T2", "jaccard": 0.7},
    ]
    activations = {
        "T1:1": rng.normal(size=d), "T1:2": rng.normal(size=d),
        "T2:1": rng.normal(size=d), "T2:2": rng.normal(size=d),
    }
    out = M.paired_geometric_tests(pairs, activations, layer=17)
    assert out["n_pairs"] == 2
    assert "centroid_distance" in out["stats"]
