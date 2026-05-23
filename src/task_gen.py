"""
Phase 1: Reasoning task generation via the Claude proxy.

Generates 1000 diverse tasks (100 per category × 10 categories) designed to
elicit extended multi-step reasoning chains from DeepSeek-R1-Distill.

Environment variables required:
    CLAUDE_PROXY_URL  — proxy endpoint URL
    CLAUDE_PROXY_KEY  — proxy API key

Cost estimate: ~$2 equivalent from your proxy budget.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

CATEGORIES: dict[str, str] = {
    "mathematical_logic": "Problems requiring formal logic, proofs, and mathematical reasoning",
    "spatial_reasoning": "Tasks involving spatial relationships, geometry, and visualisation",
    "verbal_logic": "Syllogisms, verbal analogies, and language-based reasoning",
    "pattern_recognition": "Identifying and continuing abstract sequences or patterns",
    "lateral_thinking": "Problems requiring creative, non-linear approaches",
    "causal_reasoning": "Cause-and-effect relationships and causal inference",
    "probabilistic_thinking": "Uncertainty, probability, and statistical reasoning",
    "systems_thinking": "Complex systems, interdependencies, and emergent behaviour",
    "creative_problem_solving": "Open-ended problems requiring novel approaches",
    "scientific_reasoning": "Hypothesis formation, experimental design, evidence evaluation",
}

_MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"

_SYSTEM = (
    "You are a research assistant generating diverse reasoning tasks for AI research. "
    "Return only valid JSON — no markdown, no commentary."
)

_USER = """Generate {n} diverse, challenging reasoning tasks for the category: {category}.
Description: {description}

Requirements:
- Each task requires at least 3–5 reasoning steps (not answerable in one sentence)
- Self-contained (no external resources, links, or images needed)
- Mix of moderate and hard difficulty
- Do NOT include the answer, solution hints, or worked examples

Return a JSON array where every element has exactly these keys:
  "id":         "{prefix}_{start:03d}", "{prefix}_{start1:03d}", ... (sequential from {start})
  "prompt":     the full task text as a clear, complete question or problem statement
  "difficulty": "moderate" or "hard"

Return ONLY the JSON array. No other text."""

# Used when existing tasks / blocklist context is provided to prevent repetition.
_USER_WITH_CONTEXT = """Generate {n} diverse, challenging reasoning tasks for the category: {category}.
Description: {description}

Requirements:
- Each task requires at least 3–5 reasoning steps (not answerable in one sentence)
- Self-contained (no external resources, links, or images needed)
- Mix of moderate and hard difficulty
- Do NOT include the answer, solution hints, or worked examples
- CRITICALLY: Generate entirely new, original scenarios. Do NOT produce any variation, \
rephrasing, or lightly disguised version of the following existing/blocked tasks:

{context}

Return a JSON array where every element has exactly these keys:
  "id":         "{prefix}_{start:03d}", "{prefix}_{start1:03d}", ... (sequential from {start})
  "prompt":     the full task text as a clear, complete question or problem statement
  "difficulty": "moderate" or "hard"

Return ONLY the JSON array. No other text."""


def _proxy_call(
    messages: list[dict],
    temperature: float = 0.8,
    max_tokens: int = 4096,
    proxy_url: Optional[str] = None,
    proxy_key: Optional[str] = None,
) -> str:
    """
    Call the Claude proxy and return the response text.
    Reads CLAUDE_PROXY_URL / CLAUDE_PROXY_KEY from env if not passed explicitly.
    """
    url = proxy_url or os.environ["CLAUDE_PROXY_URL"]
    key = proxy_key or os.environ["CLAUDE_PROXY_KEY"]

    resp = requests.post(
        url,
        json={
            "model": _MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        headers={"X-Api-Key": key, "Content-Type": "application/json"},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"]


def _call_api(
    category: str,
    description: str,
    prefix: str,
    start: int,
    n: int,
    proxy_url: Optional[str] = None,
    proxy_key: Optional[str] = None,
    context_summaries: Optional[list[str]] = None,
) -> list[dict]:
    if context_summaries:
        context_str = "\n".join(f"- {s}" for s in context_summaries)
        prompt = _USER_WITH_CONTEXT.format(
            n=n,
            category=category,
            description=description,
            context=context_str,
            prefix=prefix,
            start=start,
            start1=start + 1,
        )
    else:
        prompt = _USER.format(
            n=n,
            category=category,
            description=description,
            prefix=prefix,
            start=start,
            start1=start + 1,
        )
    for attempt in range(3):
        try:
            text = _proxy_call(
                messages=[
                    {"role": "user", "content": _SYSTEM + "\n\n" + prompt},
                ],
                max_tokens=2048,  # increased from 1024 to avoid truncation
                proxy_url=proxy_url,
                proxy_key=proxy_key,
            )
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            batch = json.loads(text)
            if not isinstance(batch, list):
                raise ValueError("Expected a JSON array")
            for i, task in enumerate(batch):
                task["id"] = f"{prefix}_{start + i:03d}"
                task["category"] = category
            return batch
        except Exception as exc:
            logger.warning(f"  Attempt {attempt + 1}/3 failed: {exc}")
            time.sleep(2 ** attempt)
    logger.error(f"  All attempts failed for batch starting at {start}")
    return []


def generate_tasks(
    categories: Optional[dict[str, str]] = None,
    n_per_category: int = 100,
    batch_size: int = 5,
    save_path: Optional[Path] = None,
    proxy_url: Optional[str] = None,
    proxy_key: Optional[str] = None,
) -> list[dict]:
    """
    Generate reasoning tasks for all categories.

    Reads CLAUDE_PROXY_URL and CLAUDE_PROXY_KEY from environment if not
    passed explicitly.

    Args:
        categories:      {name: description} mapping. Defaults to 10 standard categories.
        n_per_category:  Tasks per category (default 100 → 1000 total).
        batch_size:      Tasks per API call (≤25 avoids truncation).
        save_path:       If given, save the full task list as JSON here.

    Returns:
        List of task dicts: {id, prompt, difficulty, category}
    """
    if categories is None:
        categories = CATEGORIES

    all_tasks: list[dict] = []

    for cat_name, cat_desc in categories.items():
        logger.info(f"[{cat_name}] generating {n_per_category} tasks …")
        prefix = cat_name[:4].upper()
        cat_tasks: list[dict] = []
        n_batches = -(-n_per_category // batch_size)

        for b in range(n_batches):
            start = b * batch_size
            n_this = min(batch_size, n_per_category - start)
            # Pass tasks already generated in this category as context so the
            # model does not repeat the same classic problems across batches.
            context = [t["prompt"][:120] for t in cat_tasks] if cat_tasks else None
            batch = _call_api(cat_name, cat_desc, prefix, start, n_this,
                              proxy_url, proxy_key,
                              context_summaries=context)
            cat_tasks.extend(batch)
            time.sleep(0.5)

        cat_tasks = cat_tasks[:n_per_category]
        logger.info(f"  → {len(cat_tasks)} tasks")
        all_tasks.extend(cat_tasks)

    logger.info(f"Total tasks generated: {len(all_tasks)}")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(all_tasks, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved → {save_path}")

    return all_tasks


def load_tasks(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def task_summary(tasks: list[dict]) -> None:
    from collections import Counter
    cats = Counter(t["category"] for t in tasks)
    diffs = Counter(t.get("difficulty", "unknown") for t in tasks)
    print(f"Total tasks: {len(tasks)}")
    print("By category:")
    for cat, n in sorted(cats.items()):
        print(f"  {cat:<30s} {n}")
    print("By difficulty:")
    for diff, n in sorted(diffs.items()):
        print(f"  {diff:<15s} {n}")
