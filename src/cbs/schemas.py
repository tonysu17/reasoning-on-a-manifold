"""
src/cbs/schemas.py — Data classes and TypedDicts for the CBS pipeline.

Purpose
-------
Single source of truth for the data structures that flow through the CBS
extension. Locked at P0.3; downstream milestones may extend but should not
rename fields.

Validation
----------
Schema-level only; downstream modules type-check at boundaries.

Milestone
---------
P0.3 scaffolding (synthesis §M1.3 / §P0.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, TypedDict


# Allowed task / knowledge domain labels.
TASK_DOMAINS: tuple[str, ...] = (
    "algebra",
    "geometry",
    "number_theory",
    "combinatorics",
    "calculus",
    "probability",
    "logic",
    "other",
)

# CBS tier ordering, used wherever ordinal comparisons happen.
CBS_TIERS: tuple[int, ...] = (1, 2, 3)

CONFIDENCE_VALUES: tuple[str, ...] = ("high", "medium", "low")


@dataclass
class CBSResult:
    """Output of `annotate_sentence_cbs`.

    Fields
    ------
    tier             : 1 retrieval | 2 recombination | 3 novel application
    knowledge_domain : one of TASK_DOMAINS
    cross_domain     : True iff knowledge_domain != task home domain
    rationale        : one-sentence justification (<= 30 words)
    confidence       : "high" | "medium" | "low"
    """

    tier: int
    knowledge_domain: str
    cross_domain: bool
    rationale: str
    confidence: Literal["high", "medium", "low"]

    def __post_init__(self) -> None:
        if self.tier not in CBS_TIERS:
            raise ValueError(f"tier must be in {CBS_TIERS}, got {self.tier!r}")
        if self.knowledge_domain not in TASK_DOMAINS:
            raise ValueError(
                f"knowledge_domain must be in {TASK_DOMAINS}, "
                f"got {self.knowledge_domain!r}"
            )
        if self.confidence not in CONFIDENCE_VALUES:
            raise ValueError(
                f"confidence must be in {CONFIDENCE_VALUES}, "
                f"got {self.confidence!r}"
            )

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "knowledge_domain": self.knowledge_domain,
            "cross_domain": self.cross_domain,
            "rationale": self.rationale,
            "confidence": self.confidence,
        }


class CBSAnnotatedSentence(TypedDict, total=False):
    """Phase 3 sentence record extended with CBS fields.

    Phase 3 fields preserved as-is; CBS fields prefixed `cbs_`. `total=False`
    lets CBS fields be absent on sentences whose behaviour is not in
    {adding-knowledge, deduction} (we only annotate those).
    """

    sentence: str
    behaviour: str
    chain_id: str
    sentence_idx: int

    cbs_tier: int
    cbs_knowledge_domain: str
    cbs_cross_domain: bool
    cbs_rationale: str
    cbs_confidence: str


@dataclass
class ChainTrajectory:
    """Per-chain trajectory at one layer.

    Fields
    ------
    chain_id      : task / chain identifier
    layer         : residual-stream layer index
    sentence_ids  : length-T list of sentence identifiers (e.g. f"{chain_id}:{idx}")
    X             : (T, d) activation matrix, sentence-final-token convention
    behaviours    : length-T behaviour labels
    cbs_tiers     : length-T CBS tier (0 if not annotated)
    cross_domain  : length-T cross-domain flag (None if not annotated)
    truncated     : carried through from P0.4 truncation cohort
    """

    chain_id: str
    layer: int
    sentence_ids: list[str] = field(default_factory=list)
    X: Any = None                          # numpy.ndarray of shape (T, d)
    behaviours: list[str] = field(default_factory=list)
    cbs_tiers: list[int] = field(default_factory=list)
    cross_domain: list[Optional[bool]] = field(default_factory=list)
    truncated: bool = False

    @property
    def T(self) -> int:
        return 0 if self.X is None else int(self.X.shape[0])

    @property
    def hidden_dim(self) -> int:
        return 0 if self.X is None else int(self.X.shape[1])
