"""Post-training spillover — Step 1b ($0 data): STAR-1 safety set + size-matched control.

Two datasets in the pt01/pt02 record schema (prompt / reasoning / answer / label
/ id / category / source), with zero API cost:

1. ``data/safety_star1_sft.json`` — the actual ``UCSC-VLAA/STAR-1`` dataset
   (arXiv:2504.01903), the very 1K deliberative-safety examples that produced
   the ``STAR1-R1-Distill-1.5B`` checkpoint the Rung-0 pilot compares against.
   Using the true recipe data means the Rung-1 LoRA arm reproduces the public
   intervention rather than approximating it, and the Rung-0 / Rung-1 pair
   become two views of the SAME post-training step (public checkpoint vs owned
   LoRA). Fields: ``question`` -> prompt; ``response`` is split on the
   ``<think>...</think>`` boundary into reasoning / answer.

2. ``data/control_generic_sft.json`` — the size-matched NON-SAFETY control the
   spillover design pre-registers: the model's OWN generic reasoning chains
   (``data/chains_R1-1.5B.json``), same number of examples, completions
   truncated at whole-sentence boundaries to match the STAR-1 completion
   token-length distribution (quantile-matched), so "safety SFT" is isolated
   from "any SFT of this size/shape". Truncation keeps the head of the chain;
   answers for truncated chains are a neutral closing sentence, so the control
   teaches format+domain, not new content.

Run:  python pt01b_build_star1_and_control.py
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

SEED = 42
STAR1_OUT = Path("data/safety_star1_sft.json")
CONTROL_OUT = Path("data/control_generic_sft.json")
CHAINS = Path("data/chains_R1-1.5B.json")

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def approx_tokens(text: str) -> int:
    # cheap whitespace proxy; only used for LENGTH MATCHING, not for training
    return max(1, len(text.split()))


def build_star1() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("UCSC-VLAA/STAR-1", split="train")
    records = []
    n_unsplit = 0
    for i, row in enumerate(ds):
        resp = row["response"]
        m = THINK_RE.search(resp)
        if m:
            reasoning = m.group(1).strip()
            answer = resp[m.end():].strip()
        else:
            n_unsplit += 1
            reasoning, answer = "", resp.strip()
        if not answer:            # think-only rows: keep the deliberation as the answer
            reasoning, answer = "", resp.strip()
        records.append({
            "id": f"star1_{i:05d}",
            "category": row.get("category", ""),
            "label": "safety",
            "prompt": row["question"],
            "reasoning": reasoning,
            "answer": answer,
            "source": "UCSC-VLAA/STAR-1",
        })
    log.info("STAR-1: %d records (%d without a <think> split)", len(records), n_unsplit)
    return records


def sentence_truncate(text: str, max_words: int) -> str:
    """Cut at the last sentence boundary within the word budget."""
    words = text.split()
    if len(words) <= max_words:
        return text
    head = " ".join(words[:max_words])
    cut = max(head.rfind(". "), head.rfind(".\n"), head.rfind("! "), head.rfind("? "))
    return head[: cut + 1] if cut > 40 else head


def build_control(star1: list[dict]) -> list[dict]:
    chains = json.loads(CHAINS.read_text())
    rng = random.Random(SEED)
    rng.shuffle(chains)
    n = len(star1)

    # target completion lengths: quantile-match the STAR-1 completion budget
    star1_lens = sorted(approx_tokens(r["reasoning"]) + approx_tokens(r["answer"]) for r in star1)

    records = []
    for i, ch in enumerate(chains[:n]):
        target = star1_lens[int(i / n * len(star1_lens))]
        # strip the chat scaffold off the stored prompt; keep the bare instruction
        instruction = ch.get("instruction") or ch["prompt"]
        reasoning = sentence_truncate(ch["chain"].strip(), max_words=max(40, target - 20))
        records.append({
            "id": f"control_{i:05d}",
            "category": ch.get("category", ""),
            "label": "control",
            "prompt": instruction,
            "reasoning": reasoning,
            "answer": "Having reasoned it through above, that is my analysis of the problem.",
            "source": "chains_R1-1.5B (own generic chains, length-matched)",
        })
    log.info("control: %d records (quantile length-matched to STAR-1)", len(records))
    return records


def main() -> None:
    star1 = build_star1()
    control = build_control(star1)

    for path, recs in ((STAR1_OUT, star1), (CONTROL_OUT, control)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(recs, indent=1, ensure_ascii=False))
        lens = sorted(approx_tokens(r["reasoning"]) + approx_tokens(r["answer"]) for r in recs)
        log.info("%s: n=%d, completion words p50=%d p90=%d max=%d",
                 path, len(recs), lens[len(lens)//2], lens[int(len(lens)*0.9)], lens[-1])


if __name__ == "__main__":
    main()
