#!/usr/bin/env python3
"""Post-training spillover — Step 1c: OFF-POLICY non-safety control dataset.

The pt01b control (``data/control_generic_sft.json``) is built from the
model's OWN chains, so its SFT arm is on-policy: any geometric shift the
safety arm shows over it could be "training on off-policy data moves the
geometry", not "safety content moves the geometry". This builder adds the
missing cell — completions NOT written by R1-1.5B:

    ``data/control_offpolicy_sft.json`` — --n instruction->completion pairs
    from **``meta-math/MetaMathQA``** (395K math instruction rows; solutions
    written by GPT-3.5-Turbo during MetaMath's augmentation, i.e. genuinely
    off-policy for R1-1.5B and non-safety in content). Completion lengths are
    quantile-matched to the STAR-1 completion word-length distribution using
    pt01b's machinery (``approx_tokens`` / ``sentence_truncate`` / the
    ``ref_lens[int(i/n*len)]`` target formula), so the arm differs from the
    safety arm in CONTENT and POLICY-ORIGIN but not in size/shape.

One departure from pt01b, forced by the source: pt01b's own chains are ~2k
tokens so every target is reachable by truncation; an external pool is not
uniformly long, so targets are greedily assigned to the shortest pool item
that still covers them (rank/quantile assignment preserved, truncate-down
only) and any shortfall at the long tail is logged.

Record schema is exactly what pt02/pt02c consume (prompt / reasoning / answer
/ id / category / label / source); MetaMathQA's trailing "The answer is: ..."
sentence becomes the ``answer`` field, the solution body the ``reasoning``.

Run on the machine with internet (pod):
    python pt01c_build_offpolicy_control.py --n 1000 --seed 42 \
        --out data/control_offpolicy_sft.json
``--mock`` builds from deterministic synthetic text without network (tests).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

# Reuse pt01b's length-matching machinery verbatim (same proxy, same truncation).
from pt01b_build_star1_and_control import approx_tokens, sentence_truncate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("pt01c")

ANSWER_MARKER = "The answer is"
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# ── Source rows -> (prompt, body, answer) ────────────────────────────────────

def split_body_answer(response: str) -> tuple[str, str]:
    """Split a solution into (reasoning body, final-answer sentence).

    MetaMathQA solutions end with "The answer is: X"; that sentence becomes
    the ``answer`` field. Fallback: last sentence as the answer; single
    sentence gets a neutral closing (pt01b-style) so ``answer`` is never empty.
    """
    text = response.strip()
    idx = text.rfind(ANSWER_MARKER)
    if idx > 0:
        body, ans = text[:idx].strip(), text[idx:].strip()
        if body and ans:
            return body, ans
    sents = _SENT_SPLIT_RE.split(text)
    if len(sents) >= 2:
        return " ".join(sents[:-1]).strip(), sents[-1].strip()
    return text, "That is the full solution to the problem."


# ── Candidate pools ──────────────────────────────────────────────────────────

def real_pool(n: int, pool_factor: int, rng: random.Random,
              dataset_id: str = "meta-math/MetaMathQA") -> list[dict]:
    """Sample a deduplicated candidate pool from the HF dataset (network)."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    pool, seen = [], set()
    for i in idx:
        row = ds[i]
        q, resp = (row.get("query") or "").strip(), (row.get("response") or "").strip()
        if not q or not resp or q in seen:
            continue
        seen.add(q)
        body, ans = split_body_answer(resp)
        pool.append({"prompt": q, "body": body, "answer": ans,
                     "category": row.get("type", "math"),
                     "_len": approx_tokens(body) + approx_tokens(ans)})
        if len(pool) >= pool_factor * n:
            break
    log.info("%s: pooled %d candidates (of %d rows)", dataset_id, len(pool), len(ds))
    return pool


_MOCK_WORDS = ("we compute the value of the expression then simplify each term "
               "collect the factors check the remainder against the modulus and "
               "substitute back into the original equation to verify the result").split()


def mock_pool(n: int, ref_lens: list[int], rng: random.Random,
              pool_factor: int = 3) -> list[dict]:
    """Deterministic offline pool mirroring real_pool's output shape.

    Candidate lengths sweep the reference distribution with headroom so the
    quantile matching path (assignment + truncation) is exercised for real.
    """
    m = pool_factor * n
    pool = []
    for i in range(m):
        target = ref_lens[int(i / m * len(ref_lens))]
        length = int(target * (1.15 + 0.25 * rng.random())) + 15
        words, sents = 0, []
        while words < length:
            k = rng.randint(6, 12)
            sent = " ".join(rng.choice(_MOCK_WORDS) for _ in range(k)).capitalize() + "."
            words += k
            sents.append(sent)
        body = " ".join(sents)
        ans = f"The answer is: {rng.randint(0, 999)}."
        pool.append({"prompt": f"Mock math problem {i}: evaluate the expression E_{i}.",
                     "body": body, "answer": ans, "category": "mock_math",
                     "_len": approx_tokens(body) + approx_tokens(ans)})
    rng.shuffle(pool)
    return pool


# ── Quantile matching (pt01b's formula, truncate-down assignment) ────────────

def reference_lens(star1_path) -> list[int]:
    """Sorted STAR-1 completion word lengths (same expression as pt01b)."""
    recs = json.loads(Path(star1_path).read_text())
    return sorted(approx_tokens(r["reasoning"]) + approx_tokens(r["answer"]) for r in recs)


def quantile_match(pool: list[dict], ref_lens: list[int], n: int,
                   rng: random.Random, source: str) -> list[dict]:
    """Pick n pool items and truncate their completions to the reference quantiles.

    Target for rank i is pt01b's ``ref_lens[int(i/n*len(ref_lens))]``. Each
    (ascending) target gets the shortest unused candidate that covers it —
    minimal truncation; when the pool's long tail runs out the longest
    leftover is used as-is (logged as shortfall).
    """
    if len(pool) < n:
        raise SystemExit(f"pool has {len(pool)} candidates < --n {n}")
    targets = [ref_lens[int(i / n * len(ref_lens))] for i in range(n)]  # ascending
    pool_sorted = sorted(pool, key=lambda r: r["_len"])
    chosen, skipped, j, shortfall = [], [], 0, 0
    for t in targets:
        while j < len(pool_sorted) and pool_sorted[j]["_len"] < t:
            skipped.append(pool_sorted[j])
            j += 1
        if j < len(pool_sorted):
            chosen.append((t, pool_sorted[j]))
            j += 1
        else:  # tail exhausted: longest leftover, taken whole
            chosen.append((t, skipped.pop()))
            shortfall += 1
    if shortfall:
        log.warning("%d/%d targets exceed the pool's tail — matched as-is "
                    "(raise --pool-factor to tighten)", shortfall, n)

    records = []
    for t, item in chosen:
        ans_words = approx_tokens(item["answer"])
        reasoning = sentence_truncate(item["body"], max_words=max(40, t - ans_words))
        records.append({
            "category": item["category"],
            "label": "control_offpolicy",
            "prompt": item["prompt"],
            "reasoning": reasoning,
            "answer": item["answer"],
            "source": source,
        })
    # de-sort so pt02-style head slices (--dose) aren't length-biased
    rng.shuffle(records)
    for i, r in enumerate(records):
        r["id"] = f"offpolicy_{i:05d}"
    return records


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/control_offpolicy_sft.json")
    ap.add_argument("--star1", default="data/safety_star1_sft.json",
                    help="STAR-1 JSON (pt01b output) providing the reference lengths")
    ap.add_argument("--dataset", default="meta-math/MetaMathQA")
    ap.add_argument("--pool-factor", type=int, default=30,
                    help="candidate pool size = pool-factor * n")
    ap.add_argument("--mock", action="store_true",
                    help="synthetic offline pool (tests); no network")
    args = ap.parse_args(argv)

    ref = reference_lens(args.star1)
    rng = random.Random(args.seed)
    if args.mock:
        pool = mock_pool(args.n, ref, rng)
        source = "mock (synthetic off-policy control)"
    else:
        pool = real_pool(args.n, args.pool_factor, rng, dataset_id=args.dataset)
        source = f"{args.dataset} (off-policy, non-safety, length-matched)"
    records = quantile_match(pool, ref, args.n, rng, source)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=1, ensure_ascii=False))

    lens = sorted(approx_tokens(r["reasoning"]) + approx_tokens(r["answer"]) for r in records)
    for name, xs in (("reference", ref), (str(out), lens)):
        log.info("%s: n=%d, completion words p50=%d p90=%d max=%d",
                 name, len(xs), xs[len(xs) // 2], xs[int(len(xs) * 0.9)], xs[-1])


if __name__ == "__main__":
    main()
