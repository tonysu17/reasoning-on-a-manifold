"""Forged-vs-genuine policy-deliberation dataset builder (S3 scaffold; red-team F5).

The "rigorous Lucky Coin" (safety_reasoning_extension.md §S3, ch08
§safety-forgery) asks whether a probe on residual activations can tell *forged*
policy deliberation (an attacker's fabricated spec reasoning, appended in the
model's own style) from *genuine* policy deliberation. The headline is a null:
a probe at chance is the Lucky Coin failure made quantitative.

For that null to mean "the model cannot tell provenance" rather than "we built a
sloppy contrast", the forged and genuine examples must differ in provenance and
*nothing else the probe could cheat on*. This module is pure-text machinery that
constructs matched pairs with every confound F5 named exposed as a knob:

  * ``injection_position`` — forged and genuine spans are inserted at the SAME
    ordinal position, so the probe cannot separate them by *where in the chain*
    they sit (position confound).
  * ``length_matching`` — the inserted spans are matched in length, so the probe
    cannot separate them by token count (length confound).
  * ``style_source`` — ``"model_paraphrase"`` uses pre-generated model-style text
    for the forged span (supplied by the caller; no live model call here), so the
    probe cannot separate them by register/word-choice (style confound). The
    naive ``"attacker_template"`` is retained as the *uncontrolled* baseline to
    quantify how much of any separation was mere style.

The manifest is chain-grouped: every record carries the host ``chain_id`` and a
``cv_fold`` assigned by grouping on that id, so the cross-layer probe
(``05c_cross_layer_probing``) can hold whole chains out and cannot memorise a
chain it has already seen (the CF-15 leakage discipline).

CAVEAT (probe validity): :func:`build_forgery_dataset` draws genuine/forged
snippets cyclically from the supplied pools, so a given snippet can appear in both
a train-fold host and a test-fold host. A probe could then separate variants by
memorising *snippet identity* rather than provenance. Folds group only the HOST
chain. For a clean S3 probe the run-time caller should partition the snippet pools
by fold as well (draw genuine and forged snippets disjointly across folds), or
pass one snippet per host. This is left to the run-time step because the genuine
snippets are the models' own spec spans, which do not exist until the chains are
generated and DSR-annotated.

Fully offline and deterministic given ``seed``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

INJECTION_POSITIONS = ("start", "middle", "pre_decision")
LENGTH_STRATEGIES = ("none", "truncate", "pad")
STYLE_SOURCES = ("attacker_template", "model_paraphrase")
VARIANTS = ("genuine", "forged")

_PAD_FILLER = " Furthermore, this consideration applies in the present case."


@dataclass
class ForgeryRecord:
    pair_id: str
    chain_id: str
    variant: str               # "genuine" | "forged"
    text: str                  # host chain with the span inserted
    injection_char: int        # char index of the insertion in ``text``
    injection_len: int         # length of the inserted span (chars)
    injection_position: str    # the requested position label or "index:N"
    style_source: str          # which confound regime the forged span came from
    snippet_source: str        # provenance tag of the inserted span
    cv_fold: int = -1          # chain-grouped fold (assigned later)

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")


# ── Insertion mechanics ───────────────────────────────────────────────────────

def _snap_to_boundary(text: str, idx: int) -> int:
    """Move *idx* to the nearest sentence/paragraph boundary at or after it, so an
    insertion never splits a word. Falls back to the raw index near the end."""
    idx = max(0, min(idx, len(text)))
    if idx == 0 or idx == len(text):
        return idx
    nl = text.find("\n", idx)
    dot = text.find(". ", idx)
    cands = [c + (2 if c == dot else 1) for c in (nl, dot) if c != -1]
    return min(cands) if cands else idx


def _resolve_injection_char(chain_text: str, injection_position) -> tuple[int, str]:
    """Map an ``injection_position`` knob to a char index + a label for the
    manifest. ``pre_decision`` inserts just before the first decision-like
    sentence (reusing the heuristic decision detector); absent one, it falls back
    to the end of the chain."""
    if isinstance(injection_position, int):
        return _snap_to_boundary(chain_text, injection_position), f"index:{injection_position}"
    if injection_position == "start":
        return 0, "start"  # already a boundary; no forward snap
    if injection_position == "middle":
        return _snap_to_boundary(chain_text, len(chain_text) // 2), "middle"
    if injection_position == "pre_decision":
        from src.safety.deliberation import decision_type
        # ``_iter_sentences`` partitions the text exactly, so ``cursor`` is always a
        # sentence boundary — insert there directly (snapping forward would land
        # the span *inside* the decision sentence, defeating "pre-decision").
        cursor = 0
        for chunk in _iter_sentences(chain_text):
            if decision_type(chunk.strip()):
                return cursor, "pre_decision"
            cursor += len(chunk)
        return len(chain_text), "pre_decision"  # no decision found → end
    raise ValueError(f"unknown injection_position {injection_position!r}")


def _iter_sentences(text: str):
    """Yield rough sentences (split on '. ' and newlines, keeping separators) so
    char offsets stay exact when summed."""
    buf = []
    i = 0
    while i < len(text):
        buf.append(text[i])
        if text[i] == "\n" or (text[i] == "." and i + 1 < len(text) and text[i + 1] == " "):
            yield "".join(buf)
            buf = []
        i += 1
    if buf:
        yield "".join(buf)


def _length_match(snippet: str, target_len: int, strategy: str) -> str:
    """Match *snippet* to ``target_len`` characters per *strategy* (truncate at a
    word boundary, or pad with neutral filler). ``none`` returns it unchanged."""
    if strategy == "none" or target_len <= 0:
        return snippet
    if strategy == "truncate":
        if len(snippet) <= target_len:
            return snippet
        cut = snippet.rfind(" ", 0, target_len)
        return snippet[:cut] if cut > 0 else snippet[:target_len]
    if strategy == "pad":
        out = snippet
        while len(out) < target_len:
            out += _PAD_FILLER
        return out[:max(target_len, len(snippet))] if len(snippet) < target_len else snippet
    raise ValueError(f"unknown length strategy {strategy!r}")


def build_forgery_pair(
    chain_id: str,
    chain_text: str,
    genuine_snippet: str,
    forged_snippet: str,
    *,
    injection_position="pre_decision",
    length_matching: str = "truncate",
    style_source: str = "attacker_template",
    snippet_source_genuine: str = "model_spec_span",
    snippet_source_forged: str = "attacker",
) -> list[ForgeryRecord]:
    """One matched (genuine, forged) pair for a host chain.

    Both spans are inserted at the SAME char index and matched in length, so the
    only systematic difference is provenance. Returns two records sharing a
    ``pair_id``.
    """
    if style_source not in STYLE_SOURCES:
        raise ValueError(f"style_source must be one of {STYLE_SOURCES}")
    if length_matching not in LENGTH_STRATEGIES:
        raise ValueError(f"length_matching must be one of {LENGTH_STRATEGIES}")

    char_idx, pos_label = _resolve_injection_char(chain_text, injection_position)

    # Match the forged span's length to the genuine one (the genuine span is the
    # reference, since it is the model's real reasoning).
    target = len(genuine_snippet)
    forged = _length_match(forged_snippet, target, length_matching)
    genuine = genuine_snippet

    def _mk(variant: str, span: str, src: str) -> ForgeryRecord:
        text = chain_text[:char_idx] + span + chain_text[char_idx:]
        return ForgeryRecord(
            pair_id=f"{chain_id}::pair",
            chain_id=chain_id,
            variant=variant,
            text=text,
            injection_char=char_idx,
            injection_len=len(span),
            injection_position=pos_label,
            style_source=style_source,
            snippet_source=src,
        )

    return [
        _mk("genuine", genuine, snippet_source_genuine),
        _mk("forged", forged, snippet_source_forged),
    ]


# ── Dataset assembly + chain-grouped folds ────────────────────────────────────

def assign_cv_folds(records: Sequence[ForgeryRecord], n_folds: int, seed: int) -> None:
    """Assign ``cv_fold`` in place, grouping by ``chain_id`` so a chain's records
    never straddle the train/test split (CF-15). Deterministic given *seed*."""
    chain_ids = sorted({r.chain_id for r in records})
    # Seeded Fisher–Yates over the unique chain ids (no Math.random equivalent
    # needed; a simple LCG keeps it dependency-free and reproducible).
    order = list(chain_ids)
    state = (seed * 2654435761 + 1) & 0xFFFFFFFF
    for i in range(len(order) - 1, 0, -1):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        j = state % (i + 1)
        order[i], order[j] = order[j], order[i]
    fold_of = {cid: (k % n_folds) for k, cid in enumerate(order)}
    for r in records:
        r.cv_fold = fold_of[r.chain_id]


def build_forgery_dataset(
    chains: Sequence[dict],
    *,
    genuine_snippets: Sequence[str],
    forged_snippets: Sequence[str],
    injection_position="pre_decision",
    length_matching: str = "truncate",
    style_source: str = "attacker_template",
    n_folds: int = 5,
    seed: int = 0,
) -> list[dict]:
    """Build a full forged-vs-genuine manifest from host chains.

    ``chains`` are records with ``task_id`` (or ``chain_id``) and ``chain`` text.
    ``genuine_snippets`` / ``forged_snippets`` are pools the builder draws from
    cyclically (so the caller controls provenance: genuine = real model spec
    spans; forged = attacker text, or model-paraphrased text for the style
    control). Returns a list of manifest dicts with chain-grouped ``cv_fold``.
    """
    if not genuine_snippets or not forged_snippets:
        raise ValueError("need at least one genuine and one forged snippet")
    records: list[ForgeryRecord] = []
    for i, chain in enumerate(chains):
        cid = str(chain.get("chain_id", chain.get("task_id", i)))
        text = chain.get("chain", "")
        if not text:
            continue
        g = genuine_snippets[i % len(genuine_snippets)]
        f = forged_snippets[i % len(forged_snippets)]
        records.extend(build_forgery_pair(
            cid, text, g, f,
            injection_position=injection_position,
            length_matching=length_matching,
            style_source=style_source,
        ))
    assign_cv_folds(records, n_folds=n_folds, seed=seed)
    return [asdict(r) for r in records]


def genuine_spec_spans(dsr_annotated_chain: dict) -> list[str]:
    """Pull the model's own ``spec_citation`` span texts from a DSR-annotated chain
    (the consensus written by ``annotate_chains_dsr``). These are the genuine
    policy-deliberation snippets to contrast against forgeries."""
    consensus = dsr_annotated_chain.get("dsr_consensus", {})
    return [s["text"] for s in consensus.get("spans", [])
            if "spec_citation" in s.get("dsr_labels", [])]


def write_manifest(records: Sequence[dict], path) -> None:
    """Write the manifest as JSONL (one record per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(path)


def read_manifest(path) -> list[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


__all__ = [
    "INJECTION_POSITIONS", "LENGTH_STRATEGIES", "STYLE_SOURCES", "VARIANTS",
    "ForgeryRecord", "build_forgery_pair", "build_forgery_dataset",
    "assign_cv_folds", "genuine_spec_spans", "write_manifest", "read_manifest",
]
