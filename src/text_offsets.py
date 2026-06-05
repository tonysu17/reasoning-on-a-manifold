"""Canonical sentence→char-offset locator (single source of truth).

This pure-string helper was previously copy-pasted into at least four places
(05_pca_analysis, 05b_geometric_diagnostics, compare_annotators,
robustness_geometry) plus src/activation_extraction — each claiming to
"replicate activation_extraction exactly". Any drift silently misaligns the
chain-id / row arrays with the extracted activation rows. Import from here so
there is exactly one implementation.

Deliberately dependency-free (no numpy/torch/annotation imports) so the
lightweight CPU analysis scripts can import it without pulling the extraction
or API stack. See AUDIT.md §5.
"""

from __future__ import annotations

from typing import Optional


def find_sentence_offset(chain_text: str, sentence_text: str) -> Optional[int]:
    """Character offset of *sentence_text* inside *chain_text*.

    Falls back to matching the first 40 characters (handles minor annotator
    truncation). Returns None if untraceable. This is the exact rule the
    Phase-4 activation extractor uses to place sentences, so callers that map
    rows back to sentences MUST use this same function.
    """
    idx = chain_text.find(sentence_text)
    if idx >= 0:
        return idx
    prefix = sentence_text[:40].strip()
    if len(prefix) >= 10:
        idx = chain_text.find(prefix)
        if idx >= 0:
            return idx
    return None


__all__ = ["find_sentence_offset"]
