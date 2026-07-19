"""Rule-based rewards + answer extraction for the DPO/GRPO post-training arms.

Shared by ``pt09_build_dpo_pairs.py`` (self-consistency / correctness labelling),
``pt11_train_grpo.py`` (GRPO reward functions), and their tests. Pure Python +
regex only — **no torch** — so the reward logic is unit-testable on CPU and cheap
to import.

Two reward families (primer §3, rung R4):

* **math** — extract ``\\boxed{...}`` from a completion and exact-match it against
  a reference answer (normalised). Judge-free, verifiable.
* **refusal-format** — a rule-based rubric: on harmful-tagged prompts, reward
  refusal-*style* completions; on benign prompts, reward non-refusal. This
  rewards the *shape* of a refusal (hence "refusal-format"), not a semantic
  safety judgement — crude but judge-free, exactly as the primer's R4(i) asks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


# ── \boxed{...} extraction ───────────────────────────────────────────────────

def extract_boxed(text: Optional[str]) -> Optional[str]:
    r"""Return the content of the LAST ``\boxed{...}`` in *text*, brace-balanced.

    Handles nested braces (e.g. ``\boxed{\frac{1}{2}}``). Returns ``None`` when
    there is no (balanced) ``\boxed{...}``.
    """
    if not text:
        return None
    idx = text.rfind(r"\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j].strip()
    return None  # unbalanced braces


def normalize_answer(s) -> Optional[str]:
    r"""Normalise a math answer for exact-match comparison.

    Unwraps a ``\boxed{...}`` if the whole string is one, strips a handful of
    LaTeX spacing/format tokens, drops thousands-commas and a trailing period,
    and lowercases. Deliberately conservative — it is a *match* helper, not a
    CAS.
    """
    if s is None:
        return None
    s = str(s).strip()
    boxed = extract_boxed(s)
    if boxed is not None:
        s = boxed
    for tok in (r"\left", r"\right", r"\,", r"\!", r"\;", r"\ ", "$", "`", " "):
        s = s.replace(tok, "")
    s = s.replace(r"\text{", "").replace(r"\mathrm{", "")
    s = s.replace("{", "").replace("}", "")
    s = s.strip().rstrip(".").lower()
    s = s.replace(",", "")  # 1,000 -> 1000
    return s


def answers_match(a, b) -> bool:
    """True iff two answers are equal after :func:`normalize_answer`."""
    na, nb = normalize_answer(a), normalize_answer(b)
    if not na or not nb:
        return False
    return na == nb


# ── Reward: math (verifiable, boxed exact-match) ─────────────────────────────

def math_reward(completion: str, reference) -> float:
    """1.0 if the completion's boxed answer matches *reference*, else 0.0.

    A completion with no ``\\boxed{...}`` scores 0.0 (it never produced a
    checkable answer).
    """
    got = extract_boxed(completion)
    if got is None:
        return 0.0
    return 1.0 if answers_match(got, reference) else 0.0


# ── Reward: refusal-format (rule-based rubric) ───────────────────────────────

# Regexes that mark a refusal / safe-completion *style*. Matched against the
# answer span (after ``</think>``) when present, else the whole completion.
REFUSAL_PATTERNS = [
    r"\bi\s+can(?:no|')?t\b",
    r"\bi\s+cannot\b",
    r"\bi(?:'m| am)\s+unable\b",
    r"\bi\s+won'?t\b",
    r"\bi\s+will\s+not\b",
    r"can(?:no|')?t\s+help",
    r"can(?:no|')?t\s+assist",
    r"can(?:no|')?t\s+provide",
    r"cannot\s+comply",
    r"can(?:no|')?t\s+comply",
    r"cannot\s+fulfil",
    r"\bunable\s+to\b",
    r"\bi\s+must\s+decline\b",
    r"\bi\s+have\s+to\s+decline\b",
    r"\bnot\s+able\s+to\s+help\b",
    r"against\s+my\s+(?:guidelines|policy|principles|programming)",
    r"\bi(?:'m| am)\s+sorry\b",
    r"\bi\s+apolog",
    r"\bnot\s+appropriate\b",
    r"\bi\s+refuse\b",
    r"seek\s+(?:professional|help|support)",
    r"\bhelpline\b",
    r"\b988\b",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal(text: Optional[str]) -> bool:
    """True if *text* reads like a refusal / safe deflection.

    Scored on the answer span after ``</think>`` if one exists (so a model that
    *reasons about* refusing but then complies is not counted as a refusal).
    """
    if not text:
        return False
    answer = text.split("</think>")[-1] if "</think>" in text else text
    return bool(_REFUSAL_RE.search(answer))


def refusal_format_reward(completion: str, harmful) -> float:
    """Rule-based refusal reward.

    * harmful prompt  -> 1.0 if the completion refuses, else 0.0
    * benign prompt   -> 1.0 if the completion does NOT refuse, else 0.0
    """
    want_refusal = bool(harmful)
    return 1.0 if (is_refusal(completion) == want_refusal) else 0.0


# ── Tolerant reference-answer loader ─────────────────────────────────────────

_REF_KEYS = ("answer", "reference", "reference_answer", "solution",
             "final", "final_answer", "boxed", "gold", "target")
_ID_KEYS = ("id", "task_id", "taskid", "qid")


def _first(d: dict, keys) -> Optional[object]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def load_reference_answers(path) -> dict:
    """Load a reference-answers file into ``{task_id: answer_str}``, tolerantly.

    Accepts any of:
      * ``{"MATH_000": "12", ...}``                       (id -> answer)
      * ``{"MATH_000": {"answer": "12", ...}, ...}``       (id -> record)
      * ``[{"id": "MATH_000", "answer": "12"}, ...]``      (list of records)

    Record answer keys tried (in order): answer, reference, reference_answer,
    solution, final, final_answer, boxed, gold, target. Id keys: id, task_id,
    taskid, qid. Unknown shapes contribute nothing (never raises on a row).
    """
    data = json.loads(Path(path).read_text())
    out: dict[str, str] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            ans = _first(v, _REF_KEYS) if isinstance(v, dict) else v
            if ans is not None:
                out[str(k)] = str(ans)
    elif isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            rid = _first(row, _ID_KEYS)
            ans = _first(row, _REF_KEYS)
            if rid is not None and ans is not None:
                out[str(rid)] = str(ans)
    return out


__all__ = [
    "extract_boxed", "normalize_answer", "answers_match",
    "math_reward", "refusal_format_reward", "is_refusal",
    "REFUSAL_PATTERNS", "load_reference_answers",
]
