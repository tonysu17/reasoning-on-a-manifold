"""Unit tests for src/cbs/annotation.py (M1).

The tests exercise the CBS annotation pipeline with a mock Sonnet client to
avoid network calls. The mock returns canned JSON; the production annotator
parses that JSON and returns CBSResult / extends chain dicts.
"""

from __future__ import annotations

import json
from typing import Optional

import pytest

from src.cbs import annotation
from src.cbs.annotation import (
    annotate_chains_cbs,
    annotate_sentence_cbs,
    annotate_task_domain,
    CBS_PROMPT_TEMPLATE,
    cohen_kappa_three_tier,
    PLACEHOLDER_ANCHOR_BLOCK,
    TASK_DOMAIN_PROMPT,
)
from src.cbs.schemas import CBSResult


# --- Mock Sonnet client --------------------------------------------------------

class MockSonnetClient:
    """Returns canned responses. Records prompts/seeds for assertions."""

    def __init__(
        self,
        *,
        responses: Optional[list[str]] = None,
        by_seed: Optional[dict[int, str]] = None,
        on_cbs: Optional[callable] = None,
        on_domain: Optional[callable] = None,
        default: Optional[str] = None,
    ):
        self.responses = list(responses or [])
        self.by_seed = by_seed or {}
        self.on_cbs = on_cbs
        self.on_domain = on_domain
        self.default = default
        self.calls: list[dict] = []

    def complete(self, prompt: str, *, seed: int = 0, temperature: float = 0.0) -> str:
        self.calls.append({"prompt": prompt, "seed": seed, "temperature": temperature})
        if self.on_domain is not None and "Classify the home" in prompt:
            return self.on_domain(prompt, seed)
        if self.on_cbs is not None:
            return self.on_cbs(prompt, seed)
        if seed in self.by_seed:
            return self.by_seed[seed]
        if self.responses:
            return self.responses.pop(0)
        if self.default is not None:
            return self.default
        return json.dumps({
            "tier": 1,
            "knowledge_domain": "algebra",
            "cross_domain": False,
            "rationale": "default mock response",
            "confidence": "low",
        })


def _canned_cbs(tier: int, kd: str = "algebra", cd: bool = False) -> str:
    return json.dumps({
        "tier": tier,
        "knowledge_domain": kd,
        "cross_domain": cd,
        "rationale": "test",
        "confidence": "medium",
    })


# --- Module-level smoke -------------------------------------------------------

def test_annotation_module_importable() -> None:
    from src.cbs import annotation as _a  # noqa: F401


def test_cbsresult_validates_fields() -> None:
    with pytest.raises(ValueError):
        CBSResult(tier=4, knowledge_domain="algebra", cross_domain=False,
                  rationale="x", confidence="high")
    with pytest.raises(ValueError):
        CBSResult(tier=1, knowledge_domain="bogus", cross_domain=False,
                  rationale="x", confidence="high")
    with pytest.raises(ValueError):
        CBSResult(tier=1, knowledge_domain="algebra", cross_domain=False,
                  rationale="x", confidence="medium-high")
    ok = CBSResult(tier=2, knowledge_domain="algebra", cross_domain=False,
                   rationale="x", confidence="medium")
    assert ok.tier == 2


# --- CBS prompt template -------------------------------------------------------

def test_cbs_prompt_template_renders() -> None:
    rendered = CBS_PROMPT_TEMPLATE.format(
        anchor_block=PLACEHOLDER_ANCHOR_BLOCK,
        task_domain="algebra",
        task_prompt="Solve x^2 = 4",
        prev_sentences="- prev1\n- prev2",
        behaviour="adding-knowledge",
        sentence="Recall that x^2 = a means x = +/- sqrt(a).",
    )
    assert "algebra" in rendered
    assert "Solve x^2 = 4" in rendered
    assert "adding-knowledge" in rendered
    assert "PILOT-ONLY PLACEHOLDER" in rendered
    # Literal JSON braces preserved
    assert '"tier": 1 | 2 | 3' in rendered
    assert '"knowledge_domain"' in rendered


# --- annotate_task_domain -----------------------------------------------------

def test_annotate_task_domain_returns_valid_domain() -> None:
    client = MockSonnetClient(
        on_domain=lambda p, s: json.dumps({"domain": "geometry"}),
    )
    task = {"task_id": "T1", "instruction": "Find the area of a triangle ..."}
    domain = annotate_task_domain(task, client)
    assert domain == "geometry"
    assert len(client.calls) == 1


def test_annotate_task_domain_invalid_falls_back_to_other() -> None:
    client = MockSonnetClient(
        on_domain=lambda p, s: json.dumps({"domain": "not-a-real-domain"}),
    )
    domain = annotate_task_domain({"task_id": "T1", "instruction": "..."}, client)
    assert domain == "other"


def test_annotate_task_domain_malformed_falls_back_to_other() -> None:
    client = MockSonnetClient(on_domain=lambda p, s: "not json at all")
    domain = annotate_task_domain({"task_id": "T1", "instruction": "..."}, client)
    assert domain == "other"


# --- annotate_sentence_cbs ----------------------------------------------------

def test_annotate_sentence_returns_cbsresult() -> None:
    client = MockSonnetClient(on_cbs=lambda p, s: _canned_cbs(3, "number_theory", True))
    result = annotate_sentence_cbs(
        sentence="Apply Fermat's little theorem.",
        behaviour="adding-knowledge",
        task_domain="geometry",
        task_prompt="Find a triangle satisfying ...",
        prev_sentences=["prev1", "prev2", "prev3", "prev4"],
        client=client,
        seed=0,
    )
    assert isinstance(result, CBSResult)
    assert result.tier == 3
    assert result.knowledge_domain == "number_theory"
    assert result.cross_domain is True
    # Sanity: prompt embedded the sentence and task domain
    last = client.calls[-1]["prompt"]
    assert "Fermat" in last
    assert "geometry" in last
    # Only the last 3 prev sentences should be included
    assert "prev2" in last
    assert "prev3" in last
    assert "prev4" in last
    assert "prev1" not in last


def test_annotate_sentence_dual_seed_passes_through_to_client() -> None:
    client = MockSonnetClient(
        by_seed={
            0: _canned_cbs(1, "algebra", False),
            1: _canned_cbs(2, "algebra", False),
        },
    )
    r0 = annotate_sentence_cbs(
        sentence="x", behaviour="deduction", task_domain="algebra",
        task_prompt="p", prev_sentences=[], client=client, seed=0,
    )
    r1 = annotate_sentence_cbs(
        sentence="x", behaviour="deduction", task_domain="algebra",
        task_prompt="p", prev_sentences=[], client=client, seed=1,
    )
    assert r0.tier == 1
    assert r1.tier == 2
    # The runner uses seed -> temperature mapping for dual-seed kappa.
    seeds_called = sorted(c["seed"] for c in client.calls)
    assert seeds_called == [0, 1]


def test_annotate_sentence_raises_on_malformed_json() -> None:
    client = MockSonnetClient(on_cbs=lambda p, s: "garbage not json")
    with pytest.raises(ValueError):
        annotate_sentence_cbs(
            sentence="x", behaviour="deduction", task_domain="algebra",
            task_prompt="p", prev_sentences=[], client=client, seed=0,
        )


def test_annotate_sentence_raises_on_invalid_tier() -> None:
    client = MockSonnetClient(on_cbs=lambda p, s: _canned_cbs(99))
    with pytest.raises(ValueError):
        annotate_sentence_cbs(
            sentence="x", behaviour="deduction", task_domain="algebra",
            task_prompt="p", prev_sentences=[], client=client, seed=0,
        )


# --- annotate_chains_cbs ------------------------------------------------------

def _toy_chain(task_id: str = "T1") -> dict:
    return {
        "task_id": task_id,
        "category": "algebra",
        "instruction": "Solve x^2 = 4",
        "annotations": [
            {"label": "initializing", "text": "Initializing thought."},
            {"label": "adding-knowledge", "text": "Recall sqrt(4) = +/- 2."},
            {"label": "deduction", "text": "Therefore x is +/- 2."},
            {"label": "uncertainty-estimation", "text": "I am not sure."},
        ],
    }


def test_annotate_chains_cbs_adds_fields_to_targeted_behaviours() -> None:
    client = MockSonnetClient(
        on_domain=lambda p, s: json.dumps({"domain": "algebra"}),
        on_cbs=lambda p, s: _canned_cbs(2, "algebra", False),
    )
    out = annotate_chains_cbs([_toy_chain()], client, seed=0)
    assert len(out) == 1
    spans = out[0]["annotations"]
    # The two targeted spans got CBS fields
    assert "cbs_tier" in spans[1]
    assert spans[1]["cbs_tier"] == 2
    assert "cbs_tier" in spans[2]
    # The two non-targeted spans did NOT get CBS fields
    assert "cbs_tier" not in spans[0]
    assert "cbs_tier" not in spans[3]


def test_annotate_chains_cbs_uses_task_domain_cache() -> None:
    client = MockSonnetClient(
        on_domain=lambda p, s: json.dumps({"domain": "algebra"}),
        on_cbs=lambda p, s: _canned_cbs(1, "algebra", False),
    )
    chains = [_toy_chain("T1"), _toy_chain("T1")]      # same task_id twice
    annotate_chains_cbs(chains, client, seed=0)
    domain_calls = [c for c in client.calls if "Classify the home" in c["prompt"]]
    assert len(domain_calls) == 1, (
        f"task_domain should be cached per task_id; got {len(domain_calls)} calls"
    )


def test_annotate_chains_cbs_handles_per_sentence_failure_gracefully() -> None:
    # Use a text-based failure trigger so the test is deterministic under
    # within-chain ThreadPoolExecutor parallelism (max_workers > 1).

    def on_cbs(prompt: str, seed: int) -> str:
        if "Therefore x is" in prompt:
            return "bad json"
        return _canned_cbs(1, "algebra", False)

    client = MockSonnetClient(
        on_domain=lambda p, s: json.dumps({"domain": "algebra"}),
        on_cbs=on_cbs,
    )
    out = annotate_chains_cbs([_toy_chain()], client, seed=0)
    spans = out[0]["annotations"]
    # adding-knowledge succeeded
    assert "cbs_tier" in spans[1]
    # deduction failed -> cbs fields absent but span preserved
    assert "cbs_tier" not in spans[2]
    assert spans[2]["text"] == "Therefore x is +/- 2."


# --- Cohen's kappa ------------------------------------------------------------

def test_cohen_kappa_perfect_agreement_is_one() -> None:
    a = [1, 2, 3, 1, 2, 3]
    b = [1, 2, 3, 1, 2, 3]
    assert abs(cohen_kappa_three_tier(a, b) - 1.0) < 1e-9


def test_cohen_kappa_random_agreement_near_zero() -> None:
    a = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    b = [3, 3, 3, 1, 1, 1, 2, 2, 2]
    # Strong disagreement -> negative or near-zero kappa
    k = cohen_kappa_three_tier(a, b)
    assert k < 0.1, f"expected near-zero kappa for disagreement, got {k}"


def test_cohen_kappa_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError):
        cohen_kappa_three_tier([1, 2], [1, 2, 3])


# --- Task-domain prompt template ----------------------------------------------

def test_task_domain_prompt_template_renders() -> None:
    rendered = TASK_DOMAIN_PROMPT.format(task_prompt="Solve x^2 = 4")
    assert "Solve x^2 = 4" in rendered
    assert "algebra" in rendered
    assert "geometry" in rendered
