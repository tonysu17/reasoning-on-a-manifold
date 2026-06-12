"""Canonical sentence→char-offset locator (single source of truth).

This pure-string helper was previously copy-pasted into at least four places
(05_pca_analysis, 05b_geometric_diagnostics, compare_annotators,
robustness_geometry) plus src/activation_extraction — each claiming to
"replicate activation_extraction exactly". Any drift silently misaligns the
chain-id / row arrays with the extracted activation rows. Import from here so
there is exactly one implementation.

Occurrence-aware matching (2026-06-12). The original `str.find`-based rule
mapped every repeat of a sentence to its FIRST occurrence. R1 chains loop and
repeat verbatim — especially truncated ones and backtracking phrases — so
repeated annotations all bound to the same token span, producing 35–56%
exact-duplicate activation rows (zero-distance neighbours in every kNN-based
estimator the geometry claims rest on). `locate_annotation_offsets` fixes this
with a forward cursor over the chain's ordered annotations: each sentence is
searched from just past the previous match first, so the i-th repeat binds to
the i-th occurrence. Out-of-order annotations fall back to a from-the-start
search (the pre-fix behaviour), so nothing that matched before fails now.

Deliberately dependency-free (no numpy/torch/annotation imports) so the
lightweight CPU analysis scripts can import it without pulling the extraction
or API stack. See AUDIT.md §5 and CONFOUNDS_AND_REMEDIATION.md CF-13.
"""

from __future__ import annotations

from typing import Optional

#: Minimum usable prefix length for the truncated-sentence fallback.
_PREFIX_LEN = 40
_PREFIX_MIN = 10


def _find_from(chain_text: str, needle: str, start: int) -> Optional[int]:
    """Exact search from *start*, falling back to a from-the-start search.

    The fallback keeps pre-fix behaviour for annotations that appear out of
    text order (the cursor never makes a previously-locatable sentence
    unlocatable)."""
    idx = chain_text.find(needle, start)
    if idx >= 0:
        return idx
    if start > 0:
        idx = chain_text.find(needle)
        if idx >= 0:
            return idx
    return None


def find_sentence_offset(chain_text: str, sentence_text: str,
                         start: int = 0) -> Optional[int]:
    """Character offset of *sentence_text* inside *chain_text*.

    Searches from *start* first (occurrence-aware callers pass a cursor),
    falling back to a whole-string search, then to matching the first 40
    characters (handles minor annotator truncation). Returns None if
    untraceable. This is the exact rule the Phase-4 activation extractor uses
    to place sentences, so callers that map rows back to sentences MUST use
    this same function — or, for whole chains, `locate_annotation_offsets`.
    """
    idx = _find_from(chain_text, sentence_text, start)
    if idx is not None:
        return idx
    prefix = sentence_text[:_PREFIX_LEN].strip()
    if len(prefix) >= _PREFIX_MIN:
        idx = _find_from(chain_text, prefix, start)
        if idx is not None:
            return idx
    return None


def locate_annotation_offsets(chain_text: str,
                              sentences: list[str]) -> list[Optional[int]]:
    """Occurrence-aware char offsets for a chain's ordered annotation sentences.

    For each sentence (in annotation order, which is chain order), search from
    just past the previous match so verbatim repeats bind to successive
    occurrences instead of collapsing onto the first. Returns one offset (or
    None) per input sentence; the None/skip pattern is what keeps analysis-side
    row reconstruction aligned with extraction, so extraction and every loader
    must call THIS function over the chain's full annotation list (all labels,
    not just target behaviours) to advance the cursor identically.
    """
    offsets: list[Optional[int]] = []
    cursor = 0
    for sent in sentences:
        idx = find_sentence_offset(chain_text, sent, start=cursor)
        offsets.append(idx)
        if idx is not None:
            # +1 (not +len) so overlapping/nested annotation boundaries can
            # still match while guaranteeing strict forward progress for
            # identical repeats. Never move the cursor backwards on an
            # out-of-order fallback match.
            cursor = max(cursor, idx + 1)
    return offsets


__all__ = ["find_sentence_offset", "locate_annotation_offsets"]
