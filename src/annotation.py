"""
Phase 3: Sentence-level behavioural annotation via Claude Sonnet 4.5 (AWS proxy).

Uses the verbatim prompt from Venhoff et al. arXiv:2506.18167 Appendix A.
API transport identical to Phase 1: Claude proxy via CLAUDE_PROXY_URL /
CLAUDE_PROXY_KEY.

Note on annotator choice:
    Venhoff used GPT-4o for annotation. GPT-4o-2024-11-20 is not available
    on the AWS proxy used in this project. Claude Sonnet 4.5 is used instead
    and noted as a deviation in the methods section.

Label names (Venhoff taxonomy, hyphenated lowercase):
    initializing, deduction, adding-knowledge,
    example-testing, uncertainty-estimation, backtracking

Output format per span:  ["label"]sentence text["end-section"]

Environment:
    CLAUDE_PROXY_URL  — proxy endpoint (same as Phase 1)
    CLAUDE_PROXY_KEY  — proxy API key  (same as Phase 1)
"""

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests

from src.annotation_budget import (AnnotationAttemptGuard, AnnotationAttemptLimitError,
                                   AnnotationRetryLimitError)
from src.annotation_coverage import (CONTINUATION_PREFIX, COVERAGE_RULE_VERSION,
                                     annotation_region,
                                     region_source_text, row_is_coverage_complete,
                                     validate_coverage)

# Token threshold above which chunking is applied.
# Two ceilings bind here:
#   * the 29-s API-Gateway hard limit @ ~80 tok/s output (~2,300 output tokens);
#   * the $0.05 provable per-call cost bound (2026-08-11 correction). Because
#     the bound prices the WHOLE prompt plus the WHOLE output allowance, a
#     smaller chunk buys output headroom. 800 keeps the worst-case call at
#     roughly $0.047 against the frozen Sonnet 4.5 rate card while leaving the
#     label echo (~1.35x input) comfortably inside the output allowance.
CHUNK_THRESHOLD_TOKENS = 800
CHUNK_TARGET_TOKENS    = 800
# CF-18: production re-annotation uses non-overlapping chunks. The former
# overlap merge could delete genuine repeated sentences (precisely the
# backtracking/uncertainty phenomena being measured). Continuation context is
# supplied by _CONTINUATION_PREFIX; every source segment is annotated once.
CHUNK_OVERLAP_TOKENS   = 0

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

ANNOTATION_MODEL = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"

VALID_LABELS = frozenset({
    "initializing",
    "deduction",
    "adding-knowledge",
    "example-testing",
    "uncertainty-estimation",
    "backtracking",
})


class AnnotationParseError(ValueError):
    """A response cannot be safely scored under the frozen label schema."""


class ProxyText(str):
    """Response text carrying sanitized billing telemetry from the proxy."""

    def __new__(cls, value: str, *, usage_cost=None, remaining_quota=None):
        obj = super().__new__(cls, value)
        obj.usage_cost = usage_cost
        obj.remaining_quota = remaining_quota
        return obj

# The 4 behaviours we care about (distinct to thinking models).
# Single source of truth — imported by extraction/PCA/steering/eval/patching.
# All downstream filenames are built as {behaviour}_layer{N}.npy with the
# raw hyphenated lowercase name (no .lower() or transforms).
TARGET_BEHAVIOURS = [
    "backtracking",
    "uncertainty-estimation",
    "example-testing",
    "adding-knowledge",
]

# Expected sentence fractions from Venhoff et al. Figure 2 (R1-Distill models).
VENHOFF_FRACTIONS = {
    "deduction":              0.52,
    "adding-knowledge":       0.15,
    "uncertainty-estimation": 0.09,
    "initializing":           0.07,
    "example-testing":        0.06,
    "backtracking":           0.04,
}

# ── Prompt (Venhoff et al. arXiv:2506.18167 Appendix A — verbatim) ───────────
#
# Single user message; chain text substituted for {thinking_process}.
# No system message — matches the paper's protocol exactly.

_PROMPT_TEMPLATE = """\
Please split the following reasoning chain of an LLM into \
annotated parts using labels and the following format ["label\
"]...["end-section"]. A sentence should be split into multiple \
parts if it incorporates multiple behaviours indicated by the \
labels.

Available labels:
0. initializing -> The model is rephrasing the given task and \
states initial thoughts.
1. deduction -> The model is performing a deduction step based on \
its current approach and assumptions.
2. adding-knowledge -> The model is enriching the current approach \
with recalled facts.
3. example-testing -> The model generates examples to test its \
current approach.
4. uncertainty-estimation -> The model is stating its own \
uncertainty.
5. backtracking -> The model decides to change its approach.

The reasoning chain to analyze:
{thinking_process}

Answer only with the annotated text. Only use the labels outlined \
above. If there is a tail that has no annotation leave it out. \
Annotate every part of the chain in order; do not skip or summarise \
any text.\
"""

#: A5R8 (owner decision "Go with A", 2026-08-11): one sentence appended to the
#: otherwise-verbatim Venhoff Appendix-A prompt. The original's "leave it out"
#: clause licenses omission, and Sonnet generalises it to whole prose blocks —
#: 30-37% of rows failed exhaustive coverage with missingness correlated with
#: reflective (backtracking/uncertainty) content. A deviation from
#: Venhoff-verbatim, recorded alongside the A3 annotator-model deviation.
#: Versioned so resume seals reopen when the prompt changes: the request digest
#: hashes this string with the region text.
ANNOTATION_PROMPT_VERSION = "venhoff-appendixA+exhaustive-1"

#: Coverage attempts allowed per request version before a row seals unresolved.
#: 2 balances the observed proxy nondeterminism (a second identical request can
#: pass) against paying repeatedly for rows the annotator reliably refuses.
MAX_COVERAGE_ATTEMPTS = 2

# Continuation prefix prepended to chunks 2+ to prevent seam artefacts.
# Without this, Sonnet labels the first sentence of each continuation chunk
# as "initializing" because it looks like a fresh response.
# Canonical text lives in src.annotation_coverage (the validator must strip
# annotator echoes of this exact instruction, so both sides share one string).
_CONTINUATION_PREFIX = CONTINUATION_PREFIX


# ── Chunking helpers ──────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Fast token estimate: ~4 chars per token (good enough for chunking)."""
    return len(text) // 4


def chunk_chain(
    text: str,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """
    Split a chain primarily on \\n\\n paragraph boundaries.

    Oversized single paragraphs are split at a late sentence/whitespace
    boundary so a request cannot silently exceed the proxy budget. Production
    Phase-2 uses zero overlap (CF-18); bounded overlap remains supported for
    legacy callers.
    """
    if target_tokens <= 0 or overlap_tokens < 0:
        raise ValueError("chunk token budgets require target>0 and overlap>=0")
    target_chars = target_tokens * 4

    def split_oversized(unit: str) -> list[str]:
        pieces: list[str] = []
        remaining = unit
        while len(remaining) > target_chars:
            window = remaining[:target_chars + 1]
            minimum = target_chars // 2
            sentence_cuts = [
                m.end()
                for m in re.finditer(r"[.!?](?:[\"')\]]*)\s+", window)
                if m.end() >= minimum
            ]
            if sentence_cuts:
                cut = sentence_cuts[-1]
            else:
                whitespace_cuts = [
                    m.start() for m in re.finditer(r"\s+", window)
                    if m.start() >= minimum
                ]
                cut = whitespace_cuts[-1] if whitespace_cuts else target_chars
            piece = remaining[:cut].rstrip()
            if not piece:  # defensive progress guarantee
                piece, cut = remaining[:target_chars], target_chars
            pieces.append(piece)
            remaining = remaining[cut:].lstrip()
        if remaining:
            pieces.append(remaining)
        return pieces

    units: list[str] = []
    for paragraph in text.split("\n\n"):
        units.extend(split_oversized(paragraph))

    base_chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        candidate = "\n\n".join([*current, unit])
        if current and _estimate_tokens(candidate) > target_tokens:
            base_chunks.append("\n\n".join(current))
            current = [unit]
        else:
            current.append(unit)

    if current:
        base_chunks.append("\n\n".join(current))

    if overlap_tokens == 0 or len(base_chunks) <= 1:
        return base_chunks

    overlap_chars = overlap_tokens * 4
    chunks = [base_chunks[0]]
    for previous, current_chunk in zip(base_chunks, base_chunks[1:]):
        suffix = previous[-overlap_chars:]
        boundary = re.search(r"[.!?](?:[\"')\]]*)\s+", suffix)
        if boundary:
            suffix = suffix[boundary.end():]
        chunks.append(f"{suffix}\n\n{current_chunk}" if suffix else current_chunk)

    return chunks


def max_prompt_chars(
    threshold_tokens: int = CHUNK_THRESHOLD_TOKENS,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> int:
    """Largest prompt, in characters, this chunk plan can hand to the proxy.

    ``chunk_chain`` flushes *before* appending a unit that would exceed the
    target, and ``split_oversized`` guarantees every unit is itself within the
    target, so a chunk never exceeds ``target_tokens`` under the 4-chars-per-
    token convention.  Chains at or below ``threshold_tokens`` bypass chunking
    entirely, so that path bounds the prompt instead.  The continuation prefix
    is counted because chunks 2+ carry it.

    This is the input side of the provable per-call cost bound; the guard
    refuses before network I/O if a real prompt ever exceeds it.
    """
    # _estimate_tokens is len//4, and both the bypass test and the chunker's
    # flush test compare that estimate to the token budget — so a body of up to
    # budget*4 + 3 characters still passes (integer-division remainder). The
    # 2026-08-11 full-run launch was refused on exactly this: a 4,262-char
    # prompt against a 4,260 bound derived without the +3.
    body_chars = max(int(threshold_tokens), int(target_tokens) + int(overlap_tokens)) * 4 + 3
    envelope = len(_PROMPT_TEMPLATE.format(thinking_process="")) + len(_CONTINUATION_PREFIX)
    return body_chars + envelope


def annotation_initial_request_count(text: str) -> int:
    """Return the deterministic no-retry call count for one annotation row."""
    if not text.strip():
        return 0
    if _estimate_tokens(text) <= CHUNK_THRESHOLD_TOKENS:
        return 1
    return len(chunk_chain(text))


def _attempt_scope(base_scope: str, chunk_index: int, chunk_text: str) -> str:
    """Derive a content-bound scope without persisting text in the journal."""
    payload = (
        f"{base_scope}|chunk={chunk_index}|"
        f"sha256={hashlib.sha256(chunk_text.encode()).hexdigest()}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def merge_chunk_annotations(
    chunk_texts: list[str],
    chunk_annotations: list[list[dict]],
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[dict]:
    """
    Merge per-chunk annotation lists.

    Production Phase-2 has no overlap, so this is lossless concatenation. For
    legacy non-zero-overlap callers, only a *leading run* of duplicated spans
    is removed. Once a new span appears, later identical text is retained.
    """
    if len(chunk_texts) == 1:
        return chunk_annotations[0]

    if overlap_tokens <= 0:
        return [span for annotations in chunk_annotations for span in annotations]

    merged: list[dict] = list(chunk_annotations[0])

    for i in range(1, len(chunk_texts)):
        prev_text = chunk_texts[i - 1]
        # Identify the overlap region: tail ~overlap_tokens chars of prev chunk
        overlap_chars = overlap_tokens * 4
        overlap_region = prev_text[-overlap_chars:]

        keep = []
        in_overlap_prefix = True
        for span in chunk_annotations[i]:
            if in_overlap_prefix and span["text"] in overlap_region:
                continue
            in_overlap_prefix = False
            keep.append(span)
        merged.extend(keep)

    return merged


# ── Proxy call (identical pattern to Phase 1 task_gen.py) ────────────────────

def _extract_text(payload: dict) -> str:
    """Universal text extractor across proxy model families.

    The AWS proxy returns different response shapes per provider:
      - Anthropic models  -> content is a LIST of blocks: [{"type":"text","text":...}]
      - Qwen/GLM/Nova/...  -> content is a plain STRING
      - empty/failed gen   -> content is None  (e.g. gpt-oss returned 0 output tokens)
    Returns "" when no usable text is present so the retry loop can react.
    """
    content = payload.get("content")
    if isinstance(content, list):
        return "".join(
            block.get("text", "") or ""
            for block in content
            if isinstance(block, dict) and block.get("type", "text") == "text"
        )
    if isinstance(content, str):
        return content
    return ""


def _proxy_accounting(payload: dict) -> tuple[object, object]:
    """Extract the lab proxy's cost and remaining-budget telemetry.

    Cost is under ``usage.cost``.  Remaining quota is a proxy wrapper field at
    ``metadata.remaining_quota.remaining_budget`` (not ``usage``); retain the
    older flat usage key only as a compatibility fallback.
    """
    usage = payload.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    remaining = metadata.get("remaining_quota")
    remaining = remaining if isinstance(remaining, dict) else {}
    quota = remaining.get("remaining_budget")
    if quota is None:
        quota = usage.get("remaining_quota")
    return usage.get("cost"), quota


def _proxy_call(
    prompt: str,
    proxy_url: Optional[str] = None,
    proxy_key: Optional[str] = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    model: str = ANNOTATION_MODEL,
) -> str:
    url = proxy_url or os.environ["CLAUDE_PROXY_URL"]
    key = proxy_key or os.environ["CLAUDE_PROXY_KEY"]

    resp = requests.post(
        url,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        headers={"X-Api-Key": key, "Content-Type": "application/json"},
        timeout=120,
    )
    resp.raise_for_status()
    payload = resp.json()
    usage_cost, remaining_quota = _proxy_accounting(payload)
    if usage_cost is not None or remaining_quota is not None:
        # Auditable per-call cost trail (codex P5 calibration ask, 2026-08-09):
        # never credentials or prompt text — cost + quota numbers only.
        logger.info(f"proxy usage: cost={usage_cost} "
                    f"remaining_quota={remaining_quota}")
    text = _extract_text(payload)
    if not text:
        raise RuntimeError(f"empty response from model {model}")
    return ProxyText(
        text,
        usage_cost=usage_cost,
        remaining_quota=remaining_quota,
    )


# ── Parsing ───────────────────────────────────────────────────────────────────

_SPAN_RE = re.compile(
    r'\["?([^"\]]+)"?\]'       # ["label"] or [label]
    r'(.*?)'                   # content (non-greedy)
    r'\["?end-section"?\]',    # ["end-section"] or [end-section]
    re.DOTALL,
)

# Map numeric-only labels to their names (from the prompt's numbered list)
_NUMERIC_LABELS = {
    "0": "initializing",
    "1": "deduction",
    "2": "adding-knowledge",
    "3": "example-testing",
    "4": "uncertainty-estimation",
    "5": "backtracking",
}


def _normalise_label(raw: str) -> str:
    """
    Normalise a raw label string to a valid Venhoff label.

    Handles all variants the model produces:
      - "backtracking"              → "backtracking"          (clean)
      - "0. initializing"           → "initializing"          (period prefix)
      - "1-deduction"               → "deduction"             (hyphen prefix)
      - "4-uncertainty-estimation"  → "uncertainty-estimation"(hyphen prefix)
      - "1"                         → "deduction"             (bare number)
      - "4"                         → "uncertainty-estimation"(bare number)
    """
    label = raw.strip().lower()
    # Strip "N. " prefix (e.g. "0. initializing" → "initializing")
    label = re.sub(r'^\d+\.\s*', '', label)
    # Strip "N-" prefix (e.g. "1-deduction" → "deduction")
    label = re.sub(r'^\d+-', '', label)
    # Map bare digit to label name (e.g. "4" → "uncertainty-estimation")
    if label in _NUMERIC_LABELS:
        label = _NUMERIC_LABELS[label]
    return label


def parse_annotation_response(text: str) -> list[dict]:
    """
    Parse Claude output in Venhoff delimiter format.

    Input:  '["backtracking"]Wait, that is wrong.["end-section"]...'
    Output: [{"label": "backtracking", "text": "Wait, that is wrong."}, ...]

    Unknown labels invalidate the response. They are retried within the fixed
    attempt budget and become unresolved if no schema-valid response arrives;
    they are never silently coerced to ``deduction`` (CF-18).
    """
    spans = []
    for m in _SPAN_RE.finditer(text):
        label = _normalise_label(m.group(1))
        content = m.group(2).strip()
        if not content:
            continue
        if label not in VALID_LABELS:
            raise AnnotationParseError(
                f"unknown annotation label {m.group(1).strip()!r}"
            )
        spans.append({"label": label, "text": content})
    return spans


# ── Single-chain annotation ───────────────────────────────────────────────────

def annotate_chain(
    chain_text: str,
    max_retries: int = 3,
    proxy_url: Optional[str] = None,
    proxy_key: Optional[str] = None,
    model: str = ANNOTATION_MODEL,
    max_tokens: Optional[int] = None,
    shrink_on_retry: bool = False,
    attempt_guard: Optional[AnnotationAttemptGuard] = None,
    attempt_scope: Optional[str] = None,
) -> tuple[list[dict], bool]:
    """
    Annotate a single chain.

    Automatically chunks chains above CHUNK_THRESHOLD_TOKENS to stay within
    the AWS API Gateway 29-second hard timeout. Chunks are split primarily on
    paragraph boundaries, with oversized paragraphs safely subdivided. The
    Phase-2 path is non-overlapping to close CF-18.

    ``max_tokens`` caps the per-call output budget (None → the historical 8192
    default, byte-identical corpus behaviour). ``shrink_on_retry`` halves that
    budget (floor 1024) when a retry follows a 504/timeout-class failure — the
    Phase-2 proxy rule (29-s API-Gateway limit).

    Returns:
        (spans, complete)
        spans    — list of {"label": str, "text": str} dicts; [] on total failure
        complete — True iff every chunk succeeded (safe to mark chain as done)
                   False if any chunk failed (chain has partial annotations;
                   must be retried on next resume)
    """
    if not chain_text.strip():
        return [], True  # empty chain → trivially complete

    estimated_tokens = _estimate_tokens(chain_text)
    if attempt_guard is not None and attempt_scope is None:
        attempt_scope = hashlib.sha256(chain_text.encode()).hexdigest()

    if estimated_tokens <= CHUNK_THRESHOLD_TOKENS:
        # Short chain — single request
        spans = _annotate_single(chain_text, max_retries, proxy_url, proxy_key, model=model,
                                 max_tokens=max_tokens, shrink_on_retry=shrink_on_retry,
                                 attempt_guard=attempt_guard,
                                 attempt_scope=(_attempt_scope(attempt_scope, 0, chain_text)
                                                if attempt_scope else None))
        return spans, bool(spans)

    # Long chain — split into chunks and annotate each
    logger.info(f"  Chain ~{estimated_tokens} tokens — chunking for annotation")
    chunks = chunk_chain(chain_text)
    logger.info(f"  Split into {len(chunks)} chunks")

    chunk_annotations: list[list[dict]] = []
    any_failed = False
    for i, chunk in enumerate(chunks):
        prefix = _CONTINUATION_PREFIX if i > 0 else ""
        anns = _annotate_single(
            chunk, max_retries, proxy_url, proxy_key, prefix=prefix, model=model,
            max_tokens=max_tokens, shrink_on_retry=shrink_on_retry,
            attempt_guard=attempt_guard,
            attempt_scope=(_attempt_scope(attempt_scope, i, chunk)
                           if attempt_scope else None),
        )
        if not anns:
            logger.warning(f"  Chunk {i+1}/{len(chunks)} failed — returning partial")
            any_failed = True
        chunk_annotations.append(anns)

    spans = merge_chunk_annotations(chunks, chunk_annotations)
    return spans, not any_failed


#: Substrings identifying a timeout-class transport failure (the AWS
#: API-Gateway 29-s hard limit surfaces as 504 / 502 or a requests timeout).
_TIMEOUT_MARKERS = ("504", "502", "timed out", "timeout")


def _annotate_single(
    chain_text: str,
    max_retries: int = 3,
    proxy_url: Optional[str] = None,
    proxy_key: Optional[str] = None,
    prefix: str = "",
    model: str = ANNOTATION_MODEL,
    max_tokens: Optional[int] = None,
    shrink_on_retry: bool = False,
    attempt_guard: Optional[AnnotationAttemptGuard] = None,
    attempt_scope: Optional[str] = None,
) -> list[dict]:
    """Annotate a single chunk with retries. Returns [] on failure.

    With ``shrink_on_retry``, a 504/timeout-class failure halves the output
    budget (floor 1024) before the next attempt — the Phase-2 rule for the
    29-second proxy ceiling. Defaults leave the corpus pipeline unchanged.
    """
    prompt = _PROMPT_TEMPLATE.format(thinking_process=prefix + chain_text)
    budget = 8192 if max_tokens is None else int(max_tokens)

    for attempt in range(max_retries):
        # Reserve before network I/O. Limit errors deliberately sit outside
        # the retry-catching block so they stop the run rather than becoming
        # another retry or an "incomplete" row.
        reservation = None
        outcome_recorded = False
        if attempt_guard is not None:
            if attempt_scope is None:
                raise AnnotationAttemptLimitError(
                    "attempt guard requires a deterministic attempt scope"
                )
            # Cost bound FIRST: a call whose theoretical maximum charge exceeds
            # the authorised per-call ceiling must never reach the network, and
            # must not consume an attempt reservation either (it never happened).
            attempt_guard.assert_attempt_cost_bound(prompt, budget)
            try:
                reservation = attempt_guard.reserve(attempt_scope)
            except AnnotationRetryLimitError as exc:
                # The current chunk is terminally unresolved under its frozen
                # allowance. Do not let it block first attempts on untouched
                # chunks; importantly, no network call was made here.
                logger.error(f"  Annotation retry allowance exhausted: {exc}")
                return []
        try:
            text = _proxy_call(prompt, proxy_url=proxy_url, proxy_key=proxy_key,
                               model=model, max_tokens=budget)
            if attempt_guard is not None and reservation is not None:
                attempt_guard.record_response(
                    reservation["reservation_id"],
                    getattr(text, "usage_cost", None),
                    getattr(text, "remaining_quota", None),
                )
                outcome_recorded = True
            spans = parse_annotation_response(text)
            if spans:
                return spans
            logger.warning(f"  Attempt {attempt+1}: parsed 0 spans, retrying")
        except AnnotationAttemptLimitError:
            raise
        except Exception as exc:
            if (attempt_guard is not None and reservation is not None
                    and not outcome_recorded):
                attempt_guard.record_transport_failure(
                    reservation["reservation_id"], type(exc).__name__
                )
            logger.warning(f"  Attempt {attempt+1}/{max_retries} failed: {exc}")
            if shrink_on_retry and any(m in str(exc).lower() for m in _TIMEOUT_MARKERS):
                budget = max(1024, budget // 2)
                logger.warning(f"  timeout-class failure — output budget halved "
                               f"to {budget} for the next attempt")
            time.sleep(2 ** attempt)

    logger.error("  Annotation failed after all retries")
    return []


# ── Batch annotation with checkpointing ──────────────────────────────────────

def annotate_chains(
    chains: list[dict],
    save_path: Optional[Path] = None,
    checkpoint_every: int = 1,
    proxy_url: Optional[str] = None,
    proxy_key: Optional[str] = None,
    kill_after: Optional[int] = None,
    dedup_keys: tuple = ("task_id",),
    model: str = ANNOTATION_MODEL,
    max_tokens: Optional[int] = None,
    shrink_on_retry: bool = False,
    max_retries: int = 3,
    attempt_guard: Optional[AnnotationAttemptGuard] = None,
    coverage_validation: bool = True,
    include_post_think: bool = False,
) -> list[dict]:
    """
    Annotate all chains sequentially with checkpointing.

    Safe to interrupt and resume — re-running retries any chain that did not
    complete all its chunks (annotation_complete=False). The default
    checkpoint_every=1 writes the results file after EVERY chain, so an
    interruption (e.g. API credits exhausted mid-run) loses at most the single
    in-flight chain; re-run to resume from exactly where it stopped.

    Args:
        kill_after:  if set, exit after annotating this many NEW chains.
                     Used for resume-logic smoke tests.
        dedup_keys:  tuple of dict-keys that identifies a chain for resume.
                     Default ("task_id",) is correct for Phase 3 where each
                     task has one chain. For Phase 7 — which generates many
                     steered chains per task_id (one per behaviour, method,
                     alpha) — pass ("task_id", "behaviour", "method", "alpha")
                     so each steered variant is annotated separately.

    Returns list of dicts: original chain fields + "annotations" +
    "annotation_complete" (True iff all chunks succeeded).
    """
    from tqdm import tqdm

    annotated: list[dict] = []
    save_path = Path(save_path) if save_path else None

    if save_path and save_path.exists():
        from src.config import backup_existing
        backup_existing(save_path)  # snapshot prior annotations before this run touches them
        with open(save_path) as f:
            annotated = json.load(f)
        n_complete = sum(1 for a in annotated if a.get("annotation_complete", False))
        n_partial  = len(annotated) - n_complete
        logger.info(
            f"Resuming annotation from checkpoint: {len(annotated)}/{len(chains)} "
            f"({n_complete} complete, {n_partial} partial — will retry partial)"
        )

    # Backward-compatibility: records written before annotation_complete was added
    # are treated as complete if they have non-empty annotations.
    #
    # With coverage validation on, transport success is NECESSARY BUT NOT
    # SUFFICIENT: a row must also carry a current-rule coverage verdict saying
    # the annotation region was exhausted. Rows written before the validator
    # existed carry no verdict, so they are eligible for reannotation instead of
    # being skipped as complete — this is what makes the five known defective
    # rows resumable rather than silently trusted.
    def _is_complete(record: dict) -> bool:
        if coverage_validation:
            return row_is_coverage_complete(record)
        if "annotation_complete" in record:
            return bool(record["annotation_complete"])
        return bool(record.get("annotations"))  # legacy: non-empty → assume complete

    # Only skip chains that are fully complete.
    # Partial chains (some chunks failed) must be retried.
    # Use .get() (not c[k]) so a record missing a dedup key (e.g. a malformed
    # steered record with no "alpha") yields a None-padded key instead of a
    # KeyError that would crash the whole batch (mirrors chain_gen.py). Such a
    # record can never match a fully-formed key, so it is simply re-processed.
    def _chain_key(c: dict) -> tuple:
        return tuple(c.get(k) for k in dedup_keys)
    done_ids = {_chain_key(a) for a in annotated if _is_complete(a)}

    # Coverage-failure memory. The proxy is NOT perfectly deterministic at
    # temperature 0 (observed 2026-08-11: repeated identical pilot requests
    # produced differing span sets), so one re-request of a coverage-failed
    # row can genuinely succeed. Allow up to MAX_COVERAGE_ATTEMPTS attempts
    # per request version, then seal — still unresolved, still counted — until
    # the request text, the prompt version, or the coverage rule changes.
    def _is_deterministic_coverage_failure(record: dict) -> bool:
        if not coverage_validation:
            return False
        verdict = record.get("annotation_coverage")
        return (
            bool(record.get("annotation_complete"))          # transport succeeded
            and isinstance(verdict, dict)
            and verdict.get("rule_version") == COVERAGE_RULE_VERSION
            and not verdict.get("complete", False)
            and isinstance(record.get("annotation_request_sha256"), str)
        )

    sealed_failures = {
        _chain_key(a): a for a in annotated
        if not _is_complete(a) and _is_deterministic_coverage_failure(a)
    }

    # Remove partial-completion records so they will be re-processed.
    annotated = [a for a in annotated if _is_complete(a)]

    # Always iterate all chains; rely on done_ids to skip fully-completed ones.
    new_count = 0
    for chain in tqdm(chains, initial=len(done_ids), total=len(chains),
                      desc="Annotating chains"):
        if _chain_key(chain) in done_ids:
            continue

        # Crash safety: annotate_chain already absorbs transport/parse errors
        # inside its retry loop (→ empty annotations, complete=False). This
        # outer guard is the backstop for everything else — a missing "chain"
        # field (KeyError), a credit/HTTP error that somehow escapes the retry
        # loop, or any unexpected exception deeper in the stack. A single bad
        # record must NEVER abort a multi-thousand-chain Phase-7 run: catch it,
        # record the chain as incomplete (so a re-run retries it), and continue.
        # Everything annotated so far is already on disk via checkpoint_every=1.
        coverage_attempts = 1  # default when the request cannot even be built
        try:
            # Send ONLY the annotation region. Excluded material the annotator
            # never sees cannot be mislabelled, so a row is not rejected for
            # spans it was invited to produce. The scope digest binds the text
            # actually sent, so changing the request is correctly a new scope.
            # Indexed, not .get(): a record with no "chain" must still raise
            # into the handler below and be recorded incomplete, never be
            # silently annotated as an empty string and marked complete.
            request_text = (
                region_source_text(chain["chain"],
                                   include_post_think=include_post_think)
                if coverage_validation else chain["chain"]
            )
            # The digest binds the prompt version too: amending the prompt
            # invalidates every seal, so rows that failed under the old prompt
            # are re-requested rather than staying sealed against a request
            # that no longer exists.
            request_sha = hashlib.sha256(
                f"{ANNOTATION_PROMPT_VERSION}|{request_text}".encode()
            ).hexdigest()
            prior = sealed_failures.get(_chain_key(chain))
            if prior is not None and prior["annotation_request_sha256"] == request_sha:
                attempts = int(prior.get("annotation_coverage_attempts", 1))
                if attempts >= MAX_COVERAGE_ATTEMPTS:
                    # Same request has failed coverage MAX times. Retain the
                    # stored verdict; unresolved, counted, no call made.
                    logger.info(
                        f"  {_chain_key(chain)}: coverage failed "
                        f"{attempts}x for this request — sealed unresolved"
                    )
                    annotated.append(prior)
                    continue
                coverage_attempts = attempts + 1
            else:
                coverage_attempts = 1
            scope_payload = json.dumps(
                {"dedup_key": _chain_key(chain), "chain_sha256": request_sha},
                sort_keys=True, separators=(",", ":"), default=str,
            )
            anns, complete = annotate_chain(
                request_text,
                max_retries=max_retries,
                proxy_url=proxy_url,
                proxy_key=proxy_key,
                model=model,
                max_tokens=max_tokens,
                shrink_on_retry=shrink_on_retry,
                attempt_guard=attempt_guard,
                attempt_scope=hashlib.sha256(scope_payload.encode()).hexdigest(),
            )
        except KeyboardInterrupt:
            # Operator Ctrl-C: save what we have, then let it propagate so the
            # run actually stops (don't swallow an intentional interrupt).
            if save_path:
                _save_json(annotated, save_path)
            raise
        except AnnotationAttemptLimitError:
            # A budget refusal is a stage-level stop condition. Continuing
            # would only churn through rows while no further call is allowed.
            if save_path:
                _save_json(annotated, save_path)
            raise
        except Exception as exc:
            logger.error(
                f"  Unhandled error annotating chain {_chain_key(chain)} — "
                f"recording incomplete and continuing: {exc!r}"
            )
            anns, complete = [], False
        record = {**chain, "annotations": anns, "annotation_complete": complete}
        if coverage_validation:
            record["annotation_coverage_attempts"] = coverage_attempts
            try:
                record["annotation_request_sha256"] = hashlib.sha256(
                    (ANNOTATION_PROMPT_VERSION + "|" + region_source_text(
                        chain["chain"], include_post_think=include_post_think))
                    .encode()).hexdigest()
            except KeyError:
                pass  # malformed record (no "chain"): incomplete, always retried
            region = annotation_region(chain.get("chain", ""),
                                       include_post_think=include_post_think)
            report = validate_coverage(chain.get("chain", ""), anns,
                                       include_post_think=include_post_think)
            record["annotation_coverage"] = report.to_dict()
            record["annotation_coverage_complete"] = bool(complete and report.complete)
            record["annotation_coverage_rule_version"] = COVERAGE_RULE_VERSION
            # Per-1k denominators must count only what was actually in scope.
            record["annotated_region_tokens"] = region.estimated_tokens
            if not report.complete:
                logger.warning(
                    f"  coverage incomplete for {_chain_key(chain)}: "
                    f"{'; '.join(report.reasons)}"
                )
        annotated.append(record)
        new_count += 1

        if save_path and len(annotated) % checkpoint_every == 0:
            _save_json(annotated, save_path)
            if len(annotated) % 25 == 0:  # saved every chain; throttle the log line
                logger.info(f"  checkpoint: {len(annotated)}/{len(chains)}")

        time.sleep(0.3)  # rate-limit headroom

        if kill_after and new_count >= kill_after:
            logger.info(f"  --kill-after {kill_after} reached — saving and exiting")
            if save_path:
                _save_json(annotated, save_path)
            break

    if save_path:
        _save_json(annotated, save_path)

    _log_summary(annotated)
    return annotated


# ── Analysis helpers ──────────────────────────────────────────────────────────

def behaviour_counts(annotated: list[dict]) -> dict[str, int]:
    from collections import Counter
    counts: Counter = Counter()
    for chain in annotated:
        for ann in chain.get("annotations", []):
            counts[ann["label"]] += 1
    return dict(counts)


def _log_summary(annotated: list[dict]) -> None:
    counts = behaviour_counts(annotated)
    total = sum(counts.values())
    logger.info(f"Annotation summary — {len(annotated)} chains, {total} sentences:")
    order = ["backtracking", "uncertainty-estimation", "example-testing",
             "adding-knowledge", "initializing", "deduction"]
    for label in order:
        n = counts.get(label, 0)
        frac = n / total if total else 0
        logger.info(f"  {label:<28s} {n:>5d}  ({frac:>5.1%})")


def load_annotated(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _save_json(data, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.rename(path)
