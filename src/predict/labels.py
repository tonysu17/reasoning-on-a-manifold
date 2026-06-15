"""
src/predict/labels.py — Reasoning-correctness labels via an LLM judge.

The tasks are open-ended (proofs, lateral-thinking, creative) so there is no
exact-match grader; correctness is obtained by judging each chain's final answer
with Claude Sonnet on the same AWS Bedrock proxy used for annotation
(src/annotation._proxy_call — same CLAUDE_PROXY_URL / CLAUDE_PROXY_KEY transport).

Outputs a {chain_id: record} map where record.correct is True / False / None
(None = the judge was uncertain or the task is not objectively gradable, e.g. a
truncated chain with no final answer — those are EXCLUDED downstream, not coerced).

The judge is a noisy instrument on open-ended proofs (cf. the κ=0.35–0.44 seen on
behaviour annotation); `confidence` is recorded so a confidence floor can be
applied, and `truncated` is carried so the correctness↔truncation confound (CF-8)
can be controlled in the matched-pair analysis.

Milestone: Predictive Geometry — Rung-1 labels (gates the real R0/R1 result).
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Optional

from src.annotation import ANNOTATION_MODEL, _proxy_call
from src.cbs.cohort import is_truncated

logger = logging.getLogger(__name__)

__all__ = [
    "select_balanced_pilot",
    "judge_chain",
    "generate_correctness_labels",
    "load_correctness_labels",
    "build_judge_prompt",
    "parse_verdict",
]

_JUDGE_PROMPT = """\
You are grading whether an AI model's reasoning correctly solves a task.

TASK:
{instruction}

THE MODEL'S REASONING AND FINAL ANSWER:
{answer}

Judge ONLY whether the model's FINAL answer / conclusion is correct and adequately
justified for the task. Ignore style, length, and the path taken. Use "uncertain"
if the reasoning was cut off before a final answer, or if the task has no
objectively correct answer and the proposed solution is not clearly valid.

Respond with ONLY a JSON object and nothing else:
{{"verdict": "correct|incorrect|uncertain", "confidence": "high|medium|low", "rationale": "<=25 words"}}
"""


def _cap_text(text: str, max_chars: int = 16000) -> str:
    """Keep head + tail (where the conclusion lives) if the chain is very long."""
    if len(text) <= max_chars:
        return text
    head = max_chars // 3
    tail = max_chars - head
    return text[:head] + "\n\n[... middle of reasoning elided ...]\n\n" + text[-tail:]


def build_judge_prompt(chain: dict, max_chars: int = 16000) -> str:
    instruction = (chain.get("instruction") or chain.get("prompt") or "").strip()
    answer = (chain.get("full_text") or chain.get("chain") or "").strip()
    return _JUDGE_PROMPT.format(
        instruction=_cap_text(instruction, 4000), answer=_cap_text(answer, max_chars)
    )


_VERDICT_MAP = {"correct": True, "incorrect": False, "uncertain": None}


def parse_verdict(text: str) -> dict:
    """Extract {verdict, correct, confidence, rationale} from the judge's reply.

    Robust to surrounding prose / code fences: parses the first balanced JSON
    object, and falls back to a keyword scan. Returns correct=None (uncertain)
    when no verdict can be read, so an unparseable reply never fakes a label.
    """
    verdict, confidence, rationale = None, None, ""
    obj = None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    if isinstance(obj, dict):
        verdict = str(obj.get("verdict", "")).strip().lower() or None
        confidence = str(obj.get("confidence", "")).strip().lower() or None
        rationale = str(obj.get("rationale", "")).strip()
    if verdict not in _VERDICT_MAP:
        low = text.lower()
        if "incorrect" in low:
            verdict = "incorrect"
        elif "correct" in low:
            verdict = "correct"
        else:
            verdict = "uncertain"
    return {
        "verdict": verdict,
        "correct": _VERDICT_MAP.get(verdict),
        "confidence": confidence,
        "rationale": rationale,
    }


def select_balanced_pilot(
    chains: list[dict],
    tasks: list[dict],
    *,
    n: int = 200,
    prefer_complete: bool = True,
    seed: int = 42,
):
    """Pick ~n chains balanced across (category, difficulty) strata.

    Returns (selected_chains, meta) where meta[chain_id] =
    {difficulty, category, truncated}. With `prefer_complete`, non-truncated
    chains are taken first within each stratum (a truncated chain has no gradable
    final answer), falling back to truncated only to fill a thin stratum.
    """
    difficulty = {t["id"]: t.get("difficulty") for t in tasks}
    rng = random.Random(seed)

    strata: dict[tuple, list[dict]] = {}
    for c in chains:
        key = (c.get("category"), difficulty.get(c.get("task_id")))
        strata.setdefault(key, []).append(c)

    keys = sorted(strata.keys(), key=lambda k: (str(k[0]), str(k[1])))
    per = max(1, n // max(1, len(keys)))

    chosen: list[dict] = []
    for key in keys:
        items = list(strata[key])
        rng.shuffle(items)
        if prefer_complete:
            complete = [c for c in items if not is_truncated(c)]
            trunc = [c for c in items if is_truncated(c)]
            pick = complete[:per]
            if len(pick) < per:
                pick += trunc[: per - len(pick)]
        else:
            pick = items[:per]
        chosen.extend(pick)

    rng.shuffle(chosen)
    chosen = chosen[:n]
    meta = {
        c["task_id"]: {
            "difficulty": difficulty.get(c.get("task_id")),
            "category": c.get("category"),
            "truncated": bool(is_truncated(c)),
        }
        for c in chosen
    }
    return chosen, meta


def judge_chain(
    chain: dict,
    *,
    max_retries: int = 3,
    proxy_url: Optional[str] = None,
    proxy_key: Optional[str] = None,
    model: str = ANNOTATION_MODEL,
) -> dict:
    """Judge one chain's correctness. Returns parse_verdict(...) + raw reply.

    On total API failure returns an uncertain (correct=None) record so a failed
    call is never silently scored as correct/incorrect.
    """
    prompt = build_judge_prompt(chain)
    for attempt in range(max_retries):
        try:
            text = _proxy_call(
                prompt, proxy_url=proxy_url, proxy_key=proxy_key,
                max_tokens=300, temperature=0.0, model=model,
            )
            out = parse_verdict(text)
            out["raw"] = text
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"  judge attempt {attempt+1}/{max_retries}: {exc}")
            time.sleep(2 ** attempt)
    return {"verdict": "uncertain", "correct": None, "confidence": None,
            "rationale": "API failure", "raw": ""}


def _save(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.rename(path)


def generate_correctness_labels(
    chains: list[dict],
    save_path: Optional[Path] = None,
    *,
    meta: Optional[dict] = None,
    checkpoint_every: int = 25,
    max_chains: Optional[int] = None,
    proxy_url: Optional[str] = None,
    proxy_key: Optional[str] = None,
    model: str = ANNOTATION_MODEL,
) -> dict:
    """Judge each chain, with resume + checkpointing (mirrors annotate_chains).

    Returns {chain_id: {verdict, correct, confidence, rationale, ...meta}}.
    Safe to interrupt: re-running skips chains already judged in `save_path`.
    """
    from tqdm import tqdm

    meta = meta or {}
    labels: dict = {}
    save_path = Path(save_path) if save_path else None
    if save_path and save_path.exists():
        from src.config import backup_existing
        backup_existing(save_path)
        with open(save_path) as f:
            labels = json.load(f)
        logger.info(f"Resuming judge from checkpoint: {len(labels)} already done")

    new_count = 0
    for chain in tqdm(chains, desc="Judging correctness"):
        cid = chain.get("task_id")
        if cid in labels:
            continue
        rec = judge_chain(chain, proxy_url=proxy_url, proxy_key=proxy_key, model=model)
        rec.pop("raw", None)  # keep the file small; rationale is retained
        if cid in meta:
            rec.update(meta[cid])
        labels[cid] = rec
        new_count += 1
        if save_path and new_count % checkpoint_every == 0:
            _save(labels, save_path)
            logger.info(f"  checkpoint: {len(labels)} judged")
        time.sleep(0.3)  # rate-limit headroom
        if max_chains and new_count >= max_chains:
            break

    if save_path:
        _save(labels, save_path)
    return labels


def load_correctness_labels(path) -> dict:
    with open(path) as f:
        return json.load(f)
