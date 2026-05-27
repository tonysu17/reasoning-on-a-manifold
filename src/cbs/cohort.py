"""
src/cbs/cohort.py — P0.4 truncation-cohort helpers.

Implements the single source of truth for the `truncated: bool` flag that
every M2 / M3 / M4 runner uses to stratify analyses per the P0.4 decision.

Tony's decision (results/cbs/R1-1.5B/truncation_policy_decision.md):
  (b) stratify by truncated:bool.

Synthesis-plan reference: §P0.4.
"""

from __future__ import annotations

# Phase-2 generation cap (see configs/config.yaml and synthesis §P0.4).
DEFAULT_MAX_NEW_TOKENS: int = 8192


def is_truncated(chain: dict, *,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> bool:
    """A chain is "truncated" iff it hit the token cap AND has no closing
    `</think>` tag in its chain text. Matches the
    `chains_R1-1.5B.md` cross-tabulation rule."""
    n_tokens = int(chain.get("n_tokens", 0))
    chain_text = chain.get("chain", "") or ""
    return n_tokens >= max_new_tokens and not chain_text.rstrip().endswith("</think>")


__all__ = ["DEFAULT_MAX_NEW_TOKENS", "is_truncated"]
