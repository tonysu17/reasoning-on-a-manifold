"""
Phase 4: Per-behaviour activation extraction.

For each annotated chain:
  1. Reconstruct the full text (prompt + chain) — identical to what was fed to
     the model during generation.
  2. Run a single forward pass with hooks to capture residual-stream activations
     at every layer.
  3. Map each annotated sentence to its token positions (one preceding token +
     first N execution tokens), following Venhoff et al.
  4. Mean-pool activations across those positions.
  5. Accumulate into per-behaviour, per-layer matrices, flushed to shard files
     every `flush_every` chains to bound peak memory, then concatenated and saved
     as .npy with a `complete` integrity flag in metadata.

Output layout (in activations_dir/):
    {behaviour}_layer{n}.npy          float32 array (N_instances, hidden_dim)
    metadata.json                     includes `complete` + `saved_layers`

Requires: torch, transformers  (pip install .[gpu])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from src.annotation import TARGET_BEHAVIOURS

logger = logging.getLogger(__name__)


# ── Token-position mapping ────────────────────────────────────────────────────

def _sentence_to_token_positions(
    full_text: str,
    sentence_start_char: int,
    offsets: list[tuple[int, int]],
    n_preceding: int,
    n_execution: int,
) -> list[int]:
    """
    Map a sentence's char offset to token positions in the full sequence.

    Returns a list of token indices: [preceding …] + [execution …]
    Returns [] if the sentence cannot be located.
    """
    # Find first token whose end exceeds the sentence start
    onset = None
    for tok_idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end > sentence_start_char:
            onset = tok_idx
            break
    if onset is None:
        return []

    seq_len = len(offsets)
    positions: list[int] = []
    for offset in range(n_preceding, 0, -1):
        p = onset - offset
        if p >= 0:
            positions.append(p)
    for offset in range(min(n_execution, seq_len - onset)):
        positions.append(onset + offset)
    return positions


def _find_sentence_offset(chain_text: str, sentence_text: str) -> Optional[int]:
    """
    Find the character offset of *sentence_text* inside *chain_text*.
    Falls back to matching the first 40 characters (handles minor GPT truncation).
    """
    idx = chain_text.find(sentence_text)
    if idx >= 0:
        return idx
    prefix = sentence_text[:40].strip()
    if len(prefix) >= 10:
        idx = chain_text.find(prefix)
        if idx >= 0:
            return idx
    return None


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_activations(
    model,
    tokenizer,
    annotated_chains: list[dict],
    layers: Optional[list[int]],
    save_dir: Path,
    behaviours: Optional[list[str]] = None,
    n_preceding: int = 1,
    n_execution: int = 10,
    max_chains: Optional[int] = None,
    flush_every: int = 100,
    keep_in_memory: bool = True,
) -> dict[str, dict[int, np.ndarray]]:
    """
    Extract per-behaviour activation matrices across all annotated chains.

    Args:
        model / tokenizer:  loaded DeepSeek-R1-Distill model
        annotated_chains:   output of annotate_chains()
        layers:             which transformer layers to extract; None = all
        save_dir:           directory for .npy output files
        behaviours:         subset of TARGET_BEHAVIOURS to extract
        n_preceding:        tokens before behaviour onset to include
        n_execution:        first N tokens of the behaviour to include
        max_chains:         process at most this many chains (debugging)
        flush_every:        flush accumulators to shard files every N chains to
                            bound peak memory (prevents the OOM-during-write that
                            previously truncated a behaviour to 1/28 layers)
        keep_in_memory:     if False, free each final array after saving and
                            return only structure/counts (callers that only need
                            the .npy files on disk should pass False)

    Returns:
        {behaviour: {layer_idx: ndarray(N_instances, hidden_dim)}}
    """
    import torch
    from src.hooks import ActivationCache

    if behaviours is None:
        behaviours = TARGET_BEHAVIOURS
    if layers is None:
        layers = list(range(len(model.model.layers)))

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Accumulators: behaviour → layer → list of float32 vectors. To bound peak
    # memory (an all-at-once dump previously OOM-killed extraction mid-write,
    # truncating a behaviour to 1/28 layers), accumulators are flushed to
    # per-(behaviour, layer) shard files every `flush_every` chains, then
    # concatenated at the end.
    acc: dict[str, dict[int, list]] = {b: {l: [] for l in layers} for b in behaviours}
    n_extracted = {b: 0 for b in behaviours}
    n_skipped = {b: 0 for b in behaviours}

    shard_dir = save_dir / "_shards"
    if shard_dir.exists():
        import shutil
        shutil.rmtree(shard_dir)          # avoid contaminating with a prior run's shards
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = {b: {l: 0 for l in layers} for b in behaviours}

    def _flush() -> None:
        """Write current in-RAM vectors to shard files and free the RAM."""
        for b in behaviours:
            for l in layers:
                vecs = acc[b][l]
                if not vecs:
                    continue
                np.save(shard_dir / f"{b}_layer{l}.part{shard_idx[b][l]:05d}.npy", np.stack(vecs))
                shard_idx[b][l] += 1
                acc[b][l] = []

    chains = annotated_chains[:max_chains] if max_chains else annotated_chains

    for chain_i, chain in enumerate(tqdm(chains, desc="Extracting activations")):
        annotations = chain.get("annotations", [])
        if not annotations:
            continue

        chain_text: str = chain["chain"]
        prompt_text: str = chain["prompt"]
        full_text: str = prompt_text + chain_text
        chain_offset: int = len(prompt_text)

        # Tokenise once per chain (with offset mapping for char→token alignment)
        enc = tokenizer(
            full_text,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets: list[tuple[int, int]] = enc.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(model.device) for k, v in enc.items()}
        seq_len = inputs["input_ids"].shape[1]

        with ActivationCache(model, layers=layers) as cache:
            with torch.no_grad():
                model(**inputs)

            for ann in annotations:
                cat = ann["label"]
                if cat not in behaviours:
                    continue

                # Find sentence position in chain_text
                sent_offset = _find_sentence_offset(chain_text, ann["text"])
                if sent_offset is None:
                    n_skipped[cat] += 1
                    continue

                # Convert chain-relative char offset → full-text char offset
                abs_offset = chain_offset + sent_offset

                positions = _sentence_to_token_positions(
                    full_text, abs_offset, offsets, n_preceding, n_execution
                )
                # Filter to valid range
                positions = [p for p in positions if 0 <= p < seq_len]
                if not positions:
                    n_skipped[cat] += 1
                    continue

                for layer_idx in layers:
                    vec = cache.mean_at_positions(layer_idx, positions, batch_idx=0)
                    acc[cat][layer_idx].append(vec.numpy().astype(np.float32))

                n_extracted[cat] += 1

        if (chain_i + 1) % flush_every == 0:
            _flush()

    _flush()  # final partial batch

    # ── Save: concatenate shards per (behaviour, layer) ──────────────────
    # Peak memory here is bounded to ONE (behaviour, layer) array at a time, not
    # the whole corpus, so the final write cannot OOM the way it did before.
    results: dict[str, dict[int, np.ndarray]] = {}
    saved_layers: dict[str, list[int]] = {b: [] for b in behaviours}
    logger.info("Extraction summary:")

    for beh in behaviours:
        results[beh] = {}
        logger.info(f"  {beh}: {n_extracted[beh]} extracted, {n_skipped[beh]} skipped")
        for layer_idx in layers:
            shards = sorted(shard_dir.glob(f"{beh}_layer{layer_idx}.part*.npy"))
            if shards:
                mat = np.concatenate([np.load(s) for s in shards], axis=0)
                np.save(save_dir / f"{beh}_layer{layer_idx}.npy", mat)
                if keep_in_memory:
                    results[beh][layer_idx] = mat
                saved_layers[beh].append(layer_idx)
                for s in shards:
                    s.unlink()
                del mat
            else:
                logger.warning(f"    Layer {layer_idx}: no vectors!")

    try:
        shard_dir.rmdir()
    except OSError:
        pass

    # Integrity marker: written LAST, only after every expected (behaviour, layer)
    # file is on disk. `complete` lets downstream refuse to run on a partial set
    # instead of silently using it (the failure that truncated a behaviour).
    complete = all(
        len(saved_layers[b]) == len(layers) for b in behaviours if n_extracted[b] > 0
    )
    with open(save_dir / "metadata.json", "w") as f:
        json.dump({
            "behaviours": behaviours,
            "layers": layers,
            "n_preceding": n_preceding,
            "n_execution": n_execution,
            "n_extracted": n_extracted,
            "n_skipped": n_skipped,
            "saved_layers": saved_layers,
            "complete": complete,
        }, f, indent=2)

    if not complete:
        missing = {b: len(saved_layers[b]) for b in behaviours
                   if n_extracted[b] > 0 and len(saved_layers[b]) != len(layers)}
        logger.error(f"Extraction INCOMPLETE — behaviours missing layers "
                     f"(have/expected {len(layers)}): {missing}. metadata.complete=False.")

    return results


def verify_extraction_complete(
    save_dir: Path,
    behaviours: Optional[list[str]] = None,
    layers: Optional[list[int]] = None,
) -> tuple[bool, list[str]]:
    """
    Validate that an activation extraction finished cleanly.

    Returns (ok, problems). `ok` is True only if metadata.json exists, its
    `complete` flag is True, and every expected {behaviour}_layer{n}.npy file is
    present for behaviours that had instances. Callers use this to refuse to build
    geometry on a partially written set — the failure mode that truncated a
    behaviour to 1/28 layers after an OOM mid-write and then propagated silently.
    """
    save_dir = Path(save_dir)
    meta_path = save_dir / "metadata.json"
    if not meta_path.exists():
        return False, ["metadata.json missing (extraction did not finish saving)"]
    with open(meta_path) as f:
        meta = json.load(f)
    if behaviours is None:
        behaviours = meta.get("behaviours", [])
    if layers is None:
        layers = meta.get("layers", [])
    problems: list[str] = []
    n_extracted = meta.get("n_extracted", {})
    for beh in behaviours:
        if n_extracted.get(beh, 0) == 0:
            continue  # legitimately empty behaviour — nothing to check
        for layer in layers:
            if not (save_dir / f"{beh}_layer{layer}.npy").exists():
                problems.append(f"{beh}_layer{layer}.npy missing")
    if not meta.get("complete", False):
        problems.append("metadata.complete is not True")
    return (len(problems) == 0), problems


def load_activations(
    save_dir: Path,
    behaviour: str,
    layer: int,
) -> np.ndarray:
    path = Path(save_dir) / f"{behaviour}_layer{layer}.npy"
    return np.load(path)


def load_all_activations(
    save_dir: Path,
    behaviours: Optional[list[str]] = None,
    layers: Optional[list[int]] = None,
) -> dict[str, dict[int, np.ndarray]]:
    save_dir = Path(save_dir)
    meta_path = save_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if behaviours is None:
            behaviours = meta["behaviours"]
        if layers is None:
            layers = meta["layers"]
    else:
        if behaviours is None:
            behaviours = TARGET_BEHAVIOURS
        if layers is None:
            raise ValueError("No metadata.json found; pass layers explicitly")

    results: dict[str, dict[int, np.ndarray]] = {}
    for beh in behaviours:
        results[beh] = {}
        for layer in layers:
            path = save_dir / f"{beh}_layer{layer}.npy"
            if path.exists():
                results[beh][layer] = np.load(path)
    return results
