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

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests

# Token threshold above which chunking is applied
# 29s API Gateway hard limit @ ~80 tok/s output = ~2300 tokens max per chunk
CHUNK_THRESHOLD_TOKENS = 1800
CHUNK_TARGET_TOKENS    = 1500
CHUNK_OVERLAP_TOKENS   = 150

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
above. If there is a tail that has no annotation leave it out.\
"""

# Continuation prefix prepended to chunks 2+ to prevent seam artefacts.
# Without this, Sonnet labels the first sentence of each continuation chunk
# as "initializing" because it looks like a fresh response.
_CONTINUATION_PREFIX = (
    "This is a continuation of a reasoning chain. "
    "Earlier portions have already been processed; label only the sentences in this excerpt.\n\n"
)


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
    Split a chain on \\n\\n paragraph boundaries.

    Targets ~target_tokens per chunk with ~overlap_tokens of overlap between
    consecutive chunks.  The overlap region is discarded from chunk N+1's
    labels at merge time (see merge_chunk_annotations).
    """
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)
        if current_tokens + para_tokens > target_tokens and current:
            chunks.append("\n\n".join(current))
            # Build overlap from the tail of the current chunk
            overlap: list[str] = []
            overlap_count = 0
            for p in reversed(current):
                pt = _estimate_tokens(p)
                if overlap_count + pt > overlap_tokens:
                    break
                overlap.insert(0, p)
                overlap_count += pt
            current = overlap
            current_tokens = overlap_count
        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def merge_chunk_annotations(
    chunk_texts: list[str],
    chunk_annotations: list[list[dict]],
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[dict]:
    """
    Merge per-chunk annotation lists, discarding overlap spans from chunk N+1.

    For each chunk after the first, any leading spans whose text appears in
    the overlap region (tail of the previous chunk) are dropped — the first
    chunk's labels take precedence for those sentences.
    """
    if len(chunk_texts) == 1:
        return chunk_annotations[0]

    merged: list[dict] = list(chunk_annotations[0])

    for i in range(1, len(chunk_texts)):
        prev_text = chunk_texts[i - 1]
        # Identify the overlap region: tail ~overlap_tokens chars of prev chunk
        overlap_chars = overlap_tokens * 4
        overlap_region = prev_text[-overlap_chars:]

        keep = []
        for span in chunk_annotations[i]:
            # Drop span if its text appears verbatim in the overlap region
            if span["text"] in overlap_region:
                continue
            keep.append(span)
        merged.extend(keep)

    return merged


# ── Proxy call (identical pattern to Phase 1 task_gen.py) ────────────────────

def _proxy_call(
    prompt: str,
    proxy_url: Optional[str] = None,
    proxy_key: Optional[str] = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
) -> str:
    url = proxy_url or os.environ["CLAUDE_PROXY_URL"]
    key = proxy_key or os.environ["CLAUDE_PROXY_KEY"]

    resp = requests.post(
        url,
        json={
            "model": ANNOTATION_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        headers={"X-Api-Key": key, "Content-Type": "application/json"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


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

    Unknown labels fall back to "deduction" with a warning.
    """
    spans = []
    for m in _SPAN_RE.finditer(text):
        label = _normalise_label(m.group(1))
        content = m.group(2).strip()
        if not content:
            continue
        if label not in VALID_LABELS:
            logger.warning(f"  Unknown label '{m.group(1).strip()}' → deduction")
            label = "deduction"
        spans.append({"label": label, "text": content})
    return spans


# ── Single-chain annotation ───────────────────────────────────────────────────

def annotate_chain(
    chain_text: str,
    max_retries: int = 3,
    proxy_url: Optional[str] = None,
    proxy_key: Optional[str] = None,
) -> tuple[list[dict], bool]:
    """
    Annotate a single chain.

    Automatically chunks chains above CHUNK_THRESHOLD_TOKENS to stay within
    the AWS API Gateway 29-second hard timeout.  Chunks are split on paragraph
    boundaries with overlap; overlap spans are discarded at merge time.

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

    if estimated_tokens <= CHUNK_THRESHOLD_TOKENS:
        # Short chain — single request
        spans = _annotate_single(chain_text, max_retries, proxy_url, proxy_key)
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
            chunk, max_retries, proxy_url, proxy_key, prefix=prefix
        )
        if not anns:
            logger.warning(f"  Chunk {i+1}/{len(chunks)} failed — returning partial")
            any_failed = True
        chunk_annotations.append(anns)

    spans = merge_chunk_annotations(chunks, chunk_annotations)
    return spans, not any_failed


def _annotate_single(
    chain_text: str,
    max_retries: int = 3,
    proxy_url: Optional[str] = None,
    proxy_key: Optional[str] = None,
    prefix: str = "",
) -> list[dict]:
    """Annotate a single chunk with retries. Returns [] on failure."""
    prompt = _PROMPT_TEMPLATE.format(thinking_process=prefix + chain_text)

    for attempt in range(max_retries):
        try:
            text = _proxy_call(prompt, proxy_url=proxy_url, proxy_key=proxy_key)
            spans = parse_annotation_response(text)
            if spans:
                return spans
            logger.warning(f"  Attempt {attempt+1}: parsed 0 spans, retrying")
        except Exception as exc:
            logger.warning(f"  Attempt {attempt+1}/{max_retries} failed: {exc}")
            time.sleep(2 ** attempt)

    logger.error("  Annotation failed after all retries")
    return []


# ── Batch annotation with checkpointing ──────────────────────────────────────

def annotate_chains(
    chains: list[dict],
    save_path: Optional[Path] = None,
    checkpoint_every: int = 25,
    proxy_url: Optional[str] = None,
    proxy_key: Optional[str] = None,
    kill_after: Optional[int] = None,
) -> list[dict]:
    """
    Annotate all chains sequentially with checkpointing.

    Safe to interrupt and resume — re-running retries any chain that did not
    complete all its chunks (annotation_complete=False).

    Args:
        kill_after: if set, exit after annotating this many NEW chains.
                    Used for resume-logic smoke tests.

    Returns list of dicts: original chain fields + "annotations" +
    "annotation_complete" (True iff all chunks succeeded).
    """
    from tqdm import tqdm

    annotated: list[dict] = []
    save_path = Path(save_path) if save_path else None

    if save_path and save_path.exists():
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
    def _is_complete(record: dict) -> bool:
        if "annotation_complete" in record:
            return bool(record["annotation_complete"])
        return bool(record.get("annotations"))  # legacy: non-empty → assume complete

    # Only skip chains that are fully complete.
    # Partial chains (some chunks failed) must be retried.
    done_ids = {a["task_id"] for a in annotated if _is_complete(a)}

    # Remove partial-completion records so they will be re-processed.
    annotated = [a for a in annotated if _is_complete(a)]

    # Always iterate all chains; rely on done_ids to skip fully-completed ones.
    new_count = 0
    for chain in tqdm(chains, initial=len(done_ids), total=len(chains),
                      desc="Annotating chains"):
        if chain["task_id"] in done_ids:
            continue

        anns, complete = annotate_chain(
            chain["chain"],
            proxy_url=proxy_url,
            proxy_key=proxy_key,
        )
        annotated.append({**chain, "annotations": anns, "annotation_complete": complete})
        new_count += 1

        if save_path and len(annotated) % checkpoint_every == 0:
            _save_json(annotated, save_path)
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
