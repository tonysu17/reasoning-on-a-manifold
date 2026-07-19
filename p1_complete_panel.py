#!/usr/bin/env python3
"""Complete the DSR 3-judge panel: re-annotate only the (chain, judge) pairs that
failed (empty spans), CHUNKING long chains so every proxy request finishes under the
API gateway's 29 s limit. Merges results back into dsr_annotated.json and recomputes
dsr_agreement.json. Idempotent: re-runnable until every judge on every chain is filled.

Root cause the chunking fixes: Qwen3-235B cannot emit a complete annotation for a long
chain within 29 s (enough output tokens to close the JSON exceeds the gateway timeout).
Chunking keeps each request small (short prompt + bounded output) so it always fits.
"""
import json, os, sys, time
import requests

WT = "/Users/tonysu/Documents/Reasoning on a Manifold/rom-safety-worktree"
sys.path.insert(0, WT)
from src.safety.deliberation import build_dsr_judge_prompt          # noqa: E402
from src.safety.annotate import parse_dsr_response, aggregate_dsr, agreement_report, DEFAULT_JUDGES  # noqa: E402
from src.annotation import _extract_text                            # noqa: E402

ANNOT = "/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold/results/safety/dsr_annotated.json"
AGREE = "/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold/results/safety/dsr_agreement.json"
NAME2MODEL = {n: m for n, m in DEFAULT_JUDGES}

URL = os.environ["CLAUDE_PROXY_URL"]; KEY = os.environ["CLAUDE_PROXY_KEY"]
CHUNK_CHARS = 4500     # keeps prompt small enough that even Qwen closes the JSON < 29 s
REQ_TIMEOUT = 28       # client-side, just under the 29 s gateway limit → fail fast, retry
MAX_TOK = 2048
MAX_RETRY = 5


def call_judge(model, prompt):
    """One proxy request, bounded < 29 s. Returns text or raises."""
    r = requests.post(URL, json={"model": model, "messages": [{"role": "user", "content": prompt}],
                                 "max_tokens": MAX_TOK, "temperature": 0.0},
                      headers={"X-Api-Key": KEY, "Content-Type": "application/json"},
                      timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return _extract_text(r.json())


def chunk_text(text, n=CHUNK_CHARS):
    """Split at line boundaries into <= n-char segments (never mid-line)."""
    if len(text) <= n:
        return [text]
    out, cur = [], ""
    for line in text.splitlines(keepends=True):
        if cur and len(cur) + len(line) > n:
            out.append(cur); cur = ""
        # a single over-long line: hard-split it
        while len(line) > n:
            out.append(line[:n]); line = line[n:]
        cur += line
    if cur:
        out.append(cur)
    return out


def annotate_chunked(chain, model):
    """Annotate a chain with one judge, chunking so each request fits < 29 s.
    Returns (spans, ok). spans carry their 'text'; downstream locates them in the
    full chain, so concatenating chunk spans reconstructs the annotation."""
    spans = []
    for seg in chunk_text(chain):
        prompt = build_dsr_judge_prompt(seg)
        got = None
        for a in range(MAX_RETRY):
            try:
                txt = call_judge(model, prompt)
                s = parse_dsr_response(txt)
                if s or txt.strip():   # empty-but-valid (no spans in this segment) is OK
                    got = s; break
            except Exception as e:
                sys.stderr.write(f"    seg retry {a+1}/{MAX_RETRY}: {type(e).__name__} {str(e)[:50]}\n")
                time.sleep(min(2 ** a, 8))
        if got is None:
            return spans, False    # a segment permanently failed
        spans.extend(got)
    return spans, True


def main():
    recs = json.load(open(ANNOT))
    # find failed (record_idx, judge_name) pairs
    todo = [(i, jn) for i, r in enumerate(recs)
            for jn in r.get("dsr_per_judge", {}) if not r["dsr_per_judge"][jn]]
    print(f"failed (chain,judge) pairs to complete: {len(todo)}", flush=True)
    from collections import Counter
    print("  by judge:", dict(Counter(jn for _, jn in todo)), flush=True)

    fixed = 0
    for k, (i, jn) in enumerate(todo):
        r = recs[i]; chain = r.get("chain", "")
        model = NAME2MODEL.get(jn, jn)
        t = time.time()
        spans, ok = annotate_chunked(chain, model)
        if ok:
            r["dsr_per_judge"][jn] = spans
            fixed += 1
            print(f"  [{k+1}/{len(todo)}] chain#{i} {jn}: {len(spans)} spans "
                  f"({len(chain)}c, {len(chunk_text(chain))} chunks, {time.time()-t:.0f}s)", flush=True)
        else:
            print(f"  [{k+1}/{len(todo)}] chain#{i} {jn}: STILL FAILED", flush=True)

    # recompute per-record consensus + completeness, then agreement
    for r in recs:
        pj = r["dsr_per_judge"]
        r["dsr_complete"] = all(bool(v) for v in pj.values())
        try:
            r["dsr_consensus"] = aggregate_dsr(r.get("chain", ""), pj)
        except Exception as e:
            sys.stderr.write(f"consensus recompute failed on a record: {e}\n")
    json.dump(recs, open(ANNOT, "w"))
    report = agreement_report(recs)
    json.dump(report, open(AGREE, "w"), indent=2)

    n_complete = sum(1 for r in recs if r.get("dsr_complete"))
    print(f"\nFIXED {fixed}/{len(todo)} | chains with full 3-judge panel: {n_complete}/{len(recs)}", flush=True)
    agr = report.get("agreement", report)
    for lab, v in (agr.items() if isinstance(agr, dict) else []):
        if isinstance(v, dict) and "kappa" in v:
            print(f"  {lab:18s} kappa={v['kappa']:.3f}  {v.get('gate','')}", flush=True)


if __name__ == "__main__":
    main()
