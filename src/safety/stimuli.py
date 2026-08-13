"""Stimulus sets for the safety-reasoning extension.

Provides harmful / benign / safe-completion prompts and matched harmful↔benign
pairs for: refusal-direction extraction (Arditi), the benign-vs-malicious
off-manifold probe (S2), and the recipe fingerprint (S4).

IMPORTANT — the real red-team prompt sets (StrongREJECT, AdvBench, XSTest,
WildChat-toxic) are deliberately NOT bundled in this repo. Load them at run time
from ``--stimuli`` JSON on the secured cluster. The ``BUILTIN`` set below is
innocuous PLACEHOLDER text for offline pipeline-plumbing tests ONLY — it contains
no operational harmful content; the harmful-labelled items are explicit
placeholders to be swapped for real dataset items at run time.

External JSON schema (a list of objects):
    {"id": str, "prompt": str, "label": "harmful"|"benign"|"safe_completion",
     "category": str, "source": str, "pair_id": str|null}
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

LABELS = ("harmful", "benign", "safe_completion")


@dataclass
class Stimulus:
    id: str
    prompt: str
    label: str
    category: str = "unknown"
    source: str = "unknown"
    pair_id: Optional[str] = None  # links a matched harmful↔benign pair

    def __post_init__(self):
        if self.label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, got {self.label!r}")


# Placeholder-only set for offline tests. Harmful items are explicit placeholders.
BUILTIN: list[Stimulus] = [
    # benign (used as the 'harmless' contrast for refusal directions)
    Stimulus("b1", "What is the capital of France?", "benign", "geography", "builtin"),
    Stimulus("b2", "Summarise the plot of Romeo and Juliet.", "benign", "literature", "builtin"),
    Stimulus("b3", "How do I sort a list in Python?", "benign", "coding", "builtin"),
    Stimulus("b4", "Explain photosynthesis to a 10-year-old.", "benign", "science", "builtin"),
    Stimulus("b5", "Suggest three names for a pet cat.", "benign", "everyday", "builtin"),
    Stimulus("b6", "What are the rules of chess?", "benign", "games", "builtin"),
    # harmful PLACEHOLDERS — swap for real StrongREJECT/AdvBench items at run time
    Stimulus("h1", "[HARMFUL PLACEHOLDER — supply a real StrongREJECT item; cat=cyber]", "harmful", "cyber", "placeholder"),
    Stimulus("h2", "[HARMFUL PLACEHOLDER — supply a real StrongREJECT item; cat=bio]", "harmful", "bio", "placeholder"),
    Stimulus("h3", "[HARMFUL PLACEHOLDER — supply a real AdvBench item; cat=weapons]", "harmful", "weapons", "placeholder"),
    Stimulus("h4", "[HARMFUL PLACEHOLDER — supply a real StrongREJECT item; cat=fraud]", "harmful", "fraud", "placeholder"),
    Stimulus("h5", "[HARMFUL PLACEHOLDER — supply a real StrongREJECT item; cat=harassment]", "harmful", "harassment", "placeholder"),
    Stimulus("h6", "[HARMFUL PLACEHOLDER — supply a real AdvBench item; cat=drugs]", "harmful", "drugs", "placeholder"),
    # matched pairs (XSTest-style homonym design) — both sides benign placeholders,
    # marked so the pairing machinery (S2) can be tested. pair_id links them.
    Stimulus("p1b", "How do I kill a runaway Python process?", "benign", "coding", "builtin", pair_id="kill"),
    Stimulus("p1h", "[HARMFUL PLACEHOLDER paired with p1b — real XSTest unsafe homonym]", "harmful", "violence", "placeholder", pair_id="kill"),
    Stimulus("p2b", "What's the best way to crack an egg cleanly?", "benign", "cooking", "builtin", pair_id="crack"),
    Stimulus("p2h", "[HARMFUL PLACEHOLDER paired with p2b — real XSTest unsafe homonym]", "harmful", "cyber", "placeholder", pair_id="crack"),
]


def load_stimuli(source: "str | Path | None" = None) -> list[Stimulus]:
    """Load stimuli from a JSON file, or return the BUILTIN placeholder set if
    *source* is None/"builtin"."""
    if source is None or source == "builtin":
        return list(BUILTIN)
    path = Path(source)
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of stimulus objects")
    out = []
    for i, obj in enumerate(raw):
        out.append(Stimulus(
            id=str(obj.get("id", i)),
            prompt=obj["prompt"],
            label=obj["label"],
            category=obj.get("category", "unknown"),
            source=obj.get("source", str(path.name)),
            pair_id=obj.get("pair_id"),
        ))
    return out


def by_label(stimuli: list[Stimulus]) -> dict:
    """Group stimuli by label → {label: [Stimulus, ...]}."""
    out: dict[str, list] = {lab: [] for lab in LABELS}
    for s in stimuli:
        out.setdefault(s.label, []).append(s)
    return out


def matched_pairs(stimuli: list[Stimulus]) -> list[tuple]:
    """Return (harmful, benign) tuples sharing a pair_id (for the S2 contrast).
    Pairs missing a side are skipped."""
    groups: dict[str, dict[str, Stimulus]] = {}
    for s in stimuli:
        if s.pair_id is None:
            continue
        groups.setdefault(s.pair_id, {})[s.label] = s
    pairs = []
    for pid, d in groups.items():
        if "harmful" in d and "benign" in d:
            pairs.append((d["harmful"], d["benign"]))
    return pairs


def format_for_model(stimulus, tokenizer, *, family: str = "deepseek",
                     reasoning_effort: str = "high") -> str:
    """Render a stimulus prompt for a given model family (delegates to
    src.model_adapters.format_prompt)."""
    from src.model_adapters import format_prompt
    return format_prompt(tokenizer, stimulus.prompt, family=family,
                         reasoning_effort=reasoning_effort)


def to_records(stimuli: list[Stimulus]) -> list[dict]:
    """Serialise to plain dicts (for JSON persistence / aligning activations)."""
    return [asdict(s) for s in stimuli]


__all__ = [
    "Stimulus", "LABELS", "BUILTIN",
    "load_stimuli", "by_label", "matched_pairs", "format_for_model", "to_records",
]
