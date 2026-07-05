#!/usr/bin/env python3
"""Principled straggler recovery (one pass; re-run to keep going).

Two failure modes were diagnosed in the 6-shard run's stragglers:
  (a) transient Bedrock 503 / empty-response on NORMAL content (server throttling
      from heavy same-day Sonnet usage) — fixable by patient, paced retries;
  (b) genuinely DEGENERATE steered chains that looped into repeated text — a chunk
      of pure repetition has 0 reasoning sentences, so the annotator returns
      nothing; no retry can fix it, and that is the CORRECT answer (0 spans there).

Policy per chain (chunked):
  • retry each chunk patiently (paced, exponential backoff);
  • chain COMPLETE if every chunk returned spans (clean recovery — case a); OR
  • chain COMPLETE (partial_accepted) if the ONLY chunks that failed are themselves
    degenerate/repetitive (case b) — we keep the real spans from the good chunks
    and correctly treat the looped chunks as 0 reasoning;
  • else leave INCOMPLETE (a normal chunk still failing = transient not yet cleared;
    a re-run retries it).

This never DROPS a chain (avoids selective-drop bias) and never fabricates spans
for real reasoning (a normal failing chunk blocks completion until it truly succeeds).
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from src.annotation import (chunk_chain, _annotate_single, merge_chunk_annotations,  # noqa: E402
                            _estimate_tokens, _CONTINUATION_PREFIX, CHUNK_THRESHOLD_TOKENS)

EVAL = Path("results/eval/R1-1.5B__E1")
DEDUP = ("task_id", "behaviour", "method", "alpha")
MODEL = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
RETRIES = 5
# Smaller chunks keep each annotation's OUTPUT under the proxy's 29s API-Gateway
# limit (~2300 output tokens). Dense chains failed at the default ~1200; 500 halves it.
TARGET_TOKENS = int(os.environ.get("RECOVER_TARGET_TOKENS", "500"))


def key(r):
    return tuple(r.get(k) for k in DEDUP)


def is_degenerate(text: str) -> bool:
    """A chunk is degenerate if it is mostly repeated lines (a generation loop)."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 6:
        return False
    dup_frac = 1 - len(set(lines)) / len(lines)
    return dup_frac > 0.4


def annotate_patient(chain_text: str):
    """Return (spans, n_fail_normal, n_fail_degenerate, n_chunks)."""
    if _estimate_tokens(chain_text) <= TARGET_TOKENS:
        chunks = [chain_text]
    else:
        chunks = chunk_chain(chain_text, target_tokens=TARGET_TOKENS)
    chunk_anns, fail_normal, fail_degen = [], 0, 0
    for i, ch in enumerate(chunks):
        if is_degenerate(ch):
            # pure repetition → 0 reasoning sentences; skip the call (no retry waste)
            chunk_anns.append([])
            fail_degen += 1
            continue
        prefix = _CONTINUATION_PREFIX if i > 0 else ""
        anns = _annotate_single(ch, max_retries=RETRIES, prefix=prefix, model=MODEL)
        if not anns:
            fail_normal += 1       # a NORMAL chunk that failed = transient (retry later)
            time.sleep(3)          # extra cooldown after a hard chunk
        chunk_anns.append(anns)
        time.sleep(1.5)            # pace to respect Bedrock rate limits
    spans = merge_chunk_annotations(chunks, chunk_anns)
    return spans, fail_normal, fail_degen, len(chunks)


def main():
    ann = json.loads((EVAL / "annotated_steered.json").read_text())
    complete = {key(r) for r in ann if r.get("annotation_complete")}
    allc = json.loads((EVAL / "steering_results.json").read_text())
    strag = [c for c in allc if key(c) not in complete]
    print(f"{len(strag)} stragglers this pass", flush=True)

    recovered = []
    for r in strag:
        spans, fn, fd, nc = annotate_patient(r["chain"])
        tid = f"{r['task_id']} {r['behaviour']} {r['method']}"
        if fn == 0 and fd == 0:
            recovered.append({**r, "annotations": spans, "annotation_complete": True})
            print(f"  OK            {tid} spans={len(spans)}", flush=True)
        elif fn == 0 and fd > 0:   # only degenerate chunks failed → accept partial
            recovered.append({**r, "annotations": spans, "annotation_complete": True,
                              "partial_accepted": True, "degenerate_chunks": fd, "n_chunks": nc})
            print(f"  PARTIAL(degen) {tid} spans={len(spans)} degen_chunks={fd}/{nc}", flush=True)
        else:
            print(f"  STILL-FAIL    {tid} spans={len(spans)} normal_fail={fn} degen_fail={fd}/{nc}", flush=True)

    seen, final = set(), []
    for r in ann + recovered:
        if not r.get("annotation_complete"):
            continue
        k = key(r)
        if k in seen:
            continue
        seen.add(k)
        final.append(r)
    tmp = EVAL / "annotated_steered.json.tmp"
    tmp.write_text(json.dumps(final, indent=2, ensure_ascii=False))
    tmp.replace(EVAL / "annotated_steered.json")
    print(f"annotated_steered.json: {len(final)} complete | {len(allc) - len(final)} remain", flush=True)


if __name__ == "__main__":
    main()
