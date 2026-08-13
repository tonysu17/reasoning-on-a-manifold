"""Safety activation extraction — produce the harmful/harmless residual-stream
matrices that ``14_safety_geometry`` consumes.

For each stimulus: format the prompt for the model family, run one forward pass,
capture residual-stream activations at every requested layer, and pool at the
chosen token position (default: the last prompt token — the Arditi "decision"
position where refusal is determined). Group by label and save as
``{out}/harmful_layer{L}.npy`` / ``harmless_layer{L}.npy`` with rows in stimulus
order, so cross-model CKA is aligned (same prompts, same row order).

Reuses ``src.hooks.ActivationCache`` and ``src.model_adapters`` so it handles
DeepSeek, gpt-oss (harmony), and base models uniformly. Build-now-run-later:
the pure functions are unit-tested with a fake model; the runner is
``04s_extract_safety.py``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Stimulus label -> on-disk file prefix. "benign" is the harmless contrast for
# the refusal direction, so it is written as "harmless" (what the geometry layer
# expects).
DEFAULT_LABEL_ALIASES = {"benign": "harmless"}


def extract_prompt_activations(
    model,
    tokenizer,
    stimuli,
    layers: Optional[list] = None,
    *,
    family: str = "deepseek",
    reasoning_effort: str = "high",
    position: str = "last",
    pool_last_k: int = 1,
):
    """Extract per-stimulus residual activations, grouped by label.

    Returns ``(acts_by_label, order)`` where
    ``acts_by_label = {label: {layer: (N_label, d)}}`` (rows in stimulus order)
    and ``order = {label: [stimulus_id, ...]}`` for provenance/alignment.

    position="last" pools the final ``pool_last_k`` prompt tokens; otherwise pass
    an integer token index as a string.
    """
    import torch

    from src.hooks import ActivationCache
    from src.model_adapters import format_prompt, locate_decoder_layers

    if layers is None:
        layers = list(range(len(locate_decoder_layers(model))))

    acc: dict = {}
    order: dict = {}

    for s in stimuli:
        prompt = format_prompt(tokenizer, s.prompt, family=family,
                              reasoning_effort=reasoning_effort)
        enc = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in enc.items()}
        seq_len = inputs["input_ids"].shape[1]

        if position == "last":
            positions = list(range(max(0, seq_len - pool_last_k), seq_len))
        else:
            positions = [int(position)]
        positions = [p for p in positions if 0 <= p < seq_len]
        if not positions:
            logger.warning(f"stimulus {s.id}: no valid token positions; skipping")
            continue

        with ActivationCache(model, layers=layers) as cache:
            with torch.no_grad():
                model(**inputs)
            by_layer = acc.setdefault(s.label, {L: [] for L in layers})
            for L in layers:
                vec = cache.mean_at_positions(L, positions, batch_idx=0)
                by_layer[L].append(vec.numpy().astype(np.float32))
        order.setdefault(s.label, []).append(s.id)

    out = {
        label: {L: np.stack(vecs) for L, vecs in by_layer.items() if vecs}
        for label, by_layer in acc.items()
    }
    return out, order


def save_safety_activations(
    acts_by_label: dict,
    out_dir,
    order: Optional[dict] = None,
    label_aliases: Optional[dict] = None,
) -> None:
    """Write ``{prefix}_layer{L}.npy`` per label (benign→harmless by default) plus
    a ``stimulus_order.json`` provenance file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aliases = DEFAULT_LABEL_ALIASES if label_aliases is None else label_aliases
    for label, by_layer in acts_by_label.items():
        prefix = aliases.get(label, label)
        for L, mat in by_layer.items():
            np.save(out_dir / f"{prefix}_layer{L}.npy", mat)
    if order is not None:
        with open(out_dir / "stimulus_order.json", "w") as f:
            json.dump(order, f, indent=2)


__all__ = ["extract_prompt_activations", "save_safety_activations", "DEFAULT_LABEL_ALIASES"]
