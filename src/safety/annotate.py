"""Multi-annotator DSR annotation pass (red-team F4; closes the gap noted in
``safety_reasoning_extension.md`` §"Remaining": the multi-annotator LLM-judge DSR
pass was unbuilt).

``deliberation.py`` supplies the DSR schema and a single judge prompt. The
*labels are the dependent variable* for every safety-geometry claim, and a single
LLM judge is a single point of failure — the same lesson the behaviour-scheme
robustness arm learned (CF-7). So this module runs N >= 3 configurable judges over
the analysis-channel chain, aggregates to a consensus, and reports per-label
agreement so the F4 gates can be applied before any geometry is computed.

Design decisions, all mirroring the existing pipeline:

  * **Transport** identical to ``src.annotation._proxy_call`` (AWS Bedrock proxy
    via ``CLAUDE_PROXY_URL`` / ``CLAUDE_PROXY_KEY``; per-judge ``model`` id;
    retry/backoff; tolerant text extraction). Never hardcoded to OpenAI.
  * **Aggregation is character-level**, like ``compare_annotators.py``: judges
    segment the chain differently and DSR labels are non-exclusive, so we project
    each judge's spans onto characters, take a per-character majority per label,
    and re-segment the consensus into maximal constant-label-set runs. This needs
    no fragile cross-judge sentence alignment.
  * **Checkpointing / atomic writes / provenance** follow ``annotate_chains`` and
    ``14_safety_geometry`` so the runner is resumable and its output is stamped.

Offline-testable: ``annotate_chain_dsr`` takes a ``call_fn(judge, prompt) -> str``
that defaults to the proxy but is replaced by a fake in tests.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from src.safety.agreement import dsr_label_agreement
from src.safety.deliberation import (
    DECISION_TYPES, DSR_LABELS, build_dsr_judge_prompt,
)

logger = logging.getLogger(__name__)

# Default judge panel — three distinct model families on the lab proxy, chosen to
# decorrelate annotator error the way the behaviour-scheme robustness arm does
# (Sonnet / Qwen3 / Nova). Model ids are proxy aliases; override via --judges.
DEFAULT_JUDGES = (
    ("Sonnet-4.5", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"),
    ("Qwen3-235B", "qwen.qwen3-235b-a22b-2507-v1:0"),
    ("Nova-Pro", "amazon.nova-pro-v1:0"),
)


@dataclass
class Judge:
    """One annotator: a display name and the proxy model id to call."""
    name: str
    model: str


def default_judges() -> list[Judge]:
    return [Judge(n, m) for n, m in DEFAULT_JUDGES]


# ── Response parsing (JSON with repair) ───────────────────────────────────────

def _extract_json_array(text: str) -> Optional[list]:
    """Best-effort recovery of a JSON array from a judge response.

    Tolerates markdown code fences and leading/trailing prose. Returns the parsed
    list or None.
    """
    if not text:
        return None
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    body = fence.group(1) if fence else text
    # Try a direct parse first, then fall back to the outermost [ ... ].
    for candidate in (body, text):
        candidate = candidate.strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and "sentences" in obj:
                return obj["sentences"]
        except (json.JSONDecodeError, TypeError):
            pass
    start = body.find("[")
    end = body.rfind("]")
    if 0 <= start < end:
        try:
            obj = json.loads(body[start:end + 1])
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def parse_dsr_response(text: str) -> list[dict]:
    """Parse a judge response into validated DSR spans.

    Each span is ``{"text": str, "dsr_labels": [valid labels], "decision_type":
    one of DECISION_TYPES or None}``. Unknown labels are dropped (with a warning);
    a ``decision_type`` is only kept when ``decision`` is among the labels.
    Returns ``[]`` on unrecoverable output so the retry loop reacts.
    """
    raw = _extract_json_array(text)
    if raw is None:
        return []
    spans: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        txt = str(item.get("text", "")).strip()
        if not txt:
            continue
        labels = item.get("dsr_labels", []) or []
        if isinstance(labels, str):
            labels = [labels]
        clean = []
        for lab in labels:
            lab = str(lab).strip().lower()
            if lab in DSR_LABELS:
                clean.append(lab)
                continue
            # Judges sometimes imitate the prompt's example notation and emit
            # 'decision:refuse' as a single label — normalise it rather than
            # dropping the decision signal (caught live in the P1v2 launch).
            m = re.match(r"^decision[:=/\s]+([a-z_]+)$", lab)
            if m and m.group(1) in DECISION_TYPES:
                clean.append("decision")
                if not item.get("decision_type"):
                    item["decision_type"] = m.group(1)
                continue
            logger.warning(f"  dropping unknown DSR label {lab!r}")
        dt = item.get("decision_type")
        if dt is not None:
            dt = str(dt).strip().lower()
            if dt not in DECISION_TYPES:
                dt = None
        if dt is not None and "decision" not in clean:
            clean.append("decision")
        spans.append({
            "text": txt,
            "dsr_labels": sorted(set(clean)),
            "decision_type": dt if "decision" in clean else None,
        })
    return spans


# ── Proxy transport (mirrors src.annotation._proxy_call) ──────────────────────

def _proxy_call(
    prompt: str,
    model: str,
    *,
    proxy_url: Optional[str] = None,
    proxy_key: Optional[str] = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
) -> str:
    import requests  # local import: keeps the module importable without network deps

    from src.annotation import _extract_text  # reuse the tolerant extractor

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
    text = _extract_text(resp.json())
    if not text:
        raise RuntimeError(f"empty response from judge model {model}")
    return text


def _default_call_fn(judge: Judge, prompt: str) -> str:
    return _proxy_call(prompt, judge.model)


# ── Single-chain, multi-judge annotation ──────────────────────────────────────

# The lab proxy fronts the models with AWS API Gateway, which hard-terminates
# every call at 29 s. The judge echoes each sentence back as JSON, so output
# length tracks chain length and a long chain can NEVER finish inside the
# window — retries are doomed by construction (caught live in P1v2, 2026-07-20:
# 24 judge failures, concentrated on the slowest judge). Fix: annotate the
# chain in sentence-boundary chunks small enough that each call fits the
# window, and concatenate the per-chunk spans. Chunks are verbatim slices of
# the chain, so ``aggregate_dsr``'s substring locating is unaffected.
CHUNK_MAX_CHARS = 2500
_CONTEXT_TAIL_CHARS = 300


def _backoff_seconds(attempt: int, exc_str: str) -> float:
    """Retry delay. Capacity errors (503/Service Unavailable) get tens of
    seconds — the P1v2 run showed three sub-minute retries burn straight
    through an outage blip; everything else keeps the short exponential."""
    if "503" in exc_str or "Service Unavailable" in exc_str:
        return 20.0 * (attempt + 1)
    return float(2 ** attempt)


def chunk_chain_text(chain_text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """Split ``chain_text`` into verbatim slices of at most ``max_chars``,
    cutting at the last sentence-ish boundary (``.!?`` or newline followed by
    whitespace) inside each window; a boundary-free window is hard-cut. The
    chunks concatenate back to the original text exactly."""
    if len(chain_text) <= max_chars:
        return [chain_text]
    chunks: list[str] = []
    start = 0
    n = len(chain_text)
    while n - start > max_chars:
        window = chain_text[start:start + max_chars]
        last = None
        for last in re.finditer(r"[.!?\n]\s", window):
            pass
        cut = start + (last.end() if last else max_chars)
        chunks.append(chain_text[start:cut])
        start = cut
    chunks.append(chain_text[start:])
    return [c for c in chunks if c.strip()]


def annotate_chain_dsr(
    chain_text: str,
    judges: Sequence[Judge],
    *,
    policy_excerpt: Optional[str] = None,
    call_fn: Callable[[Judge, str], str] = _default_call_fn,
    max_retries: int = 3,
    chunk_chars: Optional[int] = CHUNK_MAX_CHARS,
) -> dict:
    """Annotate one chain with every judge, chunking long chains to fit the
    proxy's 29 s gateway window.

    Returns ``{"judges": {judge_name: [span, ...]}, "judges_ok": {judge_name:
    bool}, "complete": bool, "n_chunks": int}`` where a judge is ok iff it
    returned a non-empty parse for every chunk and ``complete`` is True iff
    every judge is ok. Each chunk after the first carries the tail of its
    predecessor as unlabelled context so chain-scope rules (D2) keep their
    footing.
    """
    chunks = chunk_chain_text(chain_text, chunk_chars) if chunk_chars else [chain_text]
    by_judge: dict[str, list] = {}
    judges_ok: dict[str, bool] = {}
    complete = True
    for judge in judges:
        spans: list[dict] = []
        judge_ok = True
        for ci, chunk in enumerate(chunks):
            context_tail = chunks[ci - 1][-_CONTEXT_TAIL_CHARS:] if ci else None
            prompt = build_dsr_judge_prompt(
                chunk, policy_excerpt=policy_excerpt, context_tail=context_tail,
            )
            chunk_spans: list[dict] = []
            for attempt in range(max_retries):
                try:
                    text = call_fn(judge, prompt)
                    chunk_spans = parse_dsr_response(text)
                    if chunk_spans:
                        break
                    logger.warning(
                        f"  {judge.name} chunk {ci+1}/{len(chunks)} "
                        f"attempt {attempt+1}: 0 spans parsed")
                except Exception as exc:  # noqa: BLE001 — log and retry, mirror annotation.py
                    logger.warning(
                        f"  {judge.name} chunk {ci+1}/{len(chunks)} "
                        f"attempt {attempt+1}/{max_retries}: {exc}")
                    time.sleep(_backoff_seconds(attempt, str(exc)))
            if not chunk_spans:
                judge_ok = False
            spans.extend(chunk_spans)
        if not judge_ok:
            complete = False
            logger.error(
                f"  {judge.name}: incomplete after {max_retries} retries "
                f"({len(spans)} spans over {len(chunks)} chunks)")
        by_judge[judge.name] = spans
        judges_ok[judge.name] = judge_ok
    return {"judges": by_judge, "judges_ok": judges_ok,
            "complete": complete, "n_chunks": len(chunks)}


# ── Consensus aggregation (character-level majority) ──────────────────────────

def _decision_consensus(
    chain_text: str, by_judge: dict, locate_fn,
) -> "np.ndarray":
    """Per-character majority decision_type code over judges (0 = none)."""
    import numpy as np

    codes = {dt: i + 1 for i, dt in enumerate(DECISION_TYPES)}
    n = len(chain_text)
    votes = np.zeros((n, len(DECISION_TYPES) + 1), dtype=np.int32)  # votes[char, code]
    arange = np.arange(n)
    for spans in by_judge.values():
        per_char = np.zeros(n, dtype=np.int32)
        for a in spans:
            dt = a.get("decision_type")
            if dt not in codes:
                continue
            off = locate_fn(chain_text, a.get("text", ""))
            if off is None:
                continue
            per_char[off:off + len(a["text"])] = codes[dt]
        np.add.at(votes, (arange, per_char), 1)
    # Majority non-none code per char (set only when it is the strict majority).
    n_judges = max(1, len(by_judge))
    nonnull = votes[:, 1:]
    best = nonnull.argmax(axis=1) + 1
    best_count = nonnull.max(axis=1)
    out = np.where(best_count * 2 > n_judges, best, 0).astype(np.int32)
    return out


def aggregate_dsr(chain_text: str, by_judge: dict, *, locate_fn=None) -> dict:
    """Consensus DSR annotation from per-judge spans (character-level majority).

    A label is consensus-present at a character when strictly more than half the
    judges mark it there; characters where exactly half mark it (possible only
    with an even panel) are recorded as ``n_unresolved_chars`` and treated as
    absent. The consensus is re-segmented into maximal runs of constant
    label-set; ``decision_type`` is the per-run majority.

    Returns ``{"spans": [{text, dsr_labels, decision_type}], "n_unresolved_chars":
    int, "n_judges": int}``.
    """
    import numpy as np

    from src.safety.agreement import char_label_array

    if locate_fn is None:
        from src.text_offsets import find_sentence_offset as locate_fn  # type: ignore

    n = len(chain_text)
    n_judges = len(by_judge)
    if n == 0 or n_judges == 0:
        return {"spans": [], "n_unresolved_chars": 0, "n_judges": n_judges}

    # Per-label per-char vote counts → consensus presence + tie tracking.
    label_present: dict[str, np.ndarray] = {}
    unresolved = np.zeros(n, dtype=bool)
    for label in DSR_LABELS:
        counts = np.zeros(n, dtype=np.int32)
        for spans in by_judge.values():
            counts += char_label_array(chain_text, spans, label, locate_fn=locate_fn)
        label_present[label] = counts * 2 > n_judges
        unresolved |= (counts * 2 == n_judges) & (counts > 0)

    decision_code = _decision_consensus(chain_text, by_judge, locate_fn)
    decode = {i + 1: dt for i, dt in enumerate(DECISION_TYPES)}

    # Re-segment: a "key" per character = (frozenset of present labels, decision).
    def char_key(c: int):
        labs = frozenset(lab for lab in DSR_LABELS if label_present[lab][c])
        dec = int(decision_code[c]) if "decision" in labs else 0
        return (labs, dec)

    spans: list[dict] = []
    start = 0
    while start < n:
        key = char_key(start)
        end = start + 1
        while end < n and char_key(end) == key:
            end += 1
        labs, dec = key
        if labs:  # skip unlabelled regions
            spans.append({
                "text": chain_text[start:end],
                "dsr_labels": sorted(labs),
                "decision_type": decode.get(dec),
            })
        start = end

    return {
        "spans": spans,
        "n_unresolved_chars": int(unresolved.sum()),
        "n_judges": n_judges,
    }


# ── Batch annotation with checkpointing (mirrors annotate_chains) ─────────────

def annotate_chains_dsr(
    chains: Sequence[dict],
    judges: Sequence[Judge],
    *,
    save_path: Optional[Path] = None,
    policy_excerpt: Optional[str] = None,
    call_fn: Callable[[Judge, str], str] = _default_call_fn,
    checkpoint_every: int = 25,
    kill_after: Optional[int] = None,
    dedup_keys: tuple = ("task_id",),
    chunk_chars: Optional[int] = CHUNK_MAX_CHARS,
) -> list[dict]:
    """Annotate every chain with the judge panel, with resume + atomic checkpoint.

    Each output record is the original chain plus ``dsr_per_judge`` (raw per-judge
    spans), ``dsr_consensus`` (aggregated spans + unresolved count),
    ``dsr_judges_ok`` (per-judge success) and ``dsr_complete`` (all judges
    parsed). Resume skips records already complete; incomplete records that
    carry ``dsr_judges_ok`` get a judge-targeted top-up (only the failed judges
    are re-called — the successful judges' spans are kept, not re-billed);
    legacy incomplete records without the flags are fully re-run.
    """
    annotated: list[dict] = []
    save_path = Path(save_path) if save_path else None

    if save_path and save_path.exists():
        annotated = json.loads(save_path.read_text())
        n_done = sum(1 for a in annotated if a.get("dsr_complete"))
        logger.info(f"Resuming DSR annotation: {len(annotated)} records ({n_done} complete)")

    def _key(c: dict) -> tuple:
        return tuple(c.get(k) for k in dedup_keys)

    done = {_key(a) for a in annotated if a.get("dsr_complete")}
    topup = {_key(a): a for a in annotated
             if not a.get("dsr_complete") and isinstance(a.get("dsr_judges_ok"), dict)}
    annotated = [a for a in annotated if a.get("dsr_complete")]

    new_count = 0
    for chain in chains:
        if _key(chain) in done:
            continue
        chain_text = chain.get("chain", "")
        prior = topup.get(_key(chain))
        if prior is not None:
            failed = [j for j in judges
                      if not prior["dsr_judges_ok"].get(j.name, False)]
            if failed:
                logger.info(f"  top-up {_key(chain)}: re-calling only "
                            f"{[j.name for j in failed]}")
            result = annotate_chain_dsr(
                chain_text, failed,
                policy_excerpt=policy_excerpt, call_fn=call_fn,
                chunk_chars=chunk_chars,
            )
            by_judge = {
                **{n: s for n, s in prior.get("dsr_per_judge", {}).items()
                   if prior["dsr_judges_ok"].get(n)},
                **result["judges"],
            }
            judges_ok = {**prior["dsr_judges_ok"], **result["judges_ok"]}
            complete = all(judges_ok.get(j.name, False) for j in judges)
            n_chunks = result["n_chunks"] if failed else prior.get("dsr_n_chunks", 1)
        else:
            result = annotate_chain_dsr(
                chain_text, judges,
                policy_excerpt=policy_excerpt, call_fn=call_fn,
                chunk_chars=chunk_chars,
            )
            by_judge = result["judges"]
            judges_ok = result["judges_ok"]
            complete = result["complete"]
            n_chunks = result["n_chunks"]
        consensus = aggregate_dsr(chain_text, by_judge)
        annotated.append({
            **chain,
            "dsr_per_judge": by_judge,
            "dsr_consensus": consensus,
            "dsr_judges_ok": judges_ok,
            "dsr_complete": complete,
            "dsr_n_chunks": n_chunks,
        })
        new_count += 1
        if save_path and len(annotated) % checkpoint_every == 0:
            _save_json(annotated, save_path)
        if kill_after and new_count >= kill_after:
            break

    if save_path:
        _save_json(annotated, save_path)
    return annotated


def agreement_report(
    annotated: Sequence[dict], *, locate_fn=None,
) -> dict:
    """Per-label agreement across judges over all annotated chains (F4 report).

    Wraps :func:`dsr_label_agreement`, reading the ``dsr_per_judge`` block written
    by :func:`annotate_chains_dsr`.
    """
    recs = [{"chain": a.get("chain", ""), "judges": a.get("dsr_per_judge", {})}
            for a in annotated if a.get("dsr_per_judge")]
    return dsr_label_agreement(recs, DSR_LABELS, locate_fn=locate_fn)


def _save_json(data, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.rename(path)


__all__ = [
    "Judge", "default_judges", "DEFAULT_JUDGES",
    "parse_dsr_response", "annotate_chain_dsr", "aggregate_dsr",
    "annotate_chains_dsr", "agreement_report",
]
