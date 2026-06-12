"""
Phase 4: Per-behaviour activation extraction.

For each annotated chain:
  1. Reconstruct the full text (prompt + chain) — identical to what was fed to
     the model during generation.
  2. Run a single forward pass with hooks to capture residual-stream activations
     at every layer.
  3. Map each annotated sentence to its token positions (one preceding token +
     first N execution tokens), following Venhoff et al.
  4. Pool activations across those positions (configurable: mean / last /
     first / max — see _pool() and config.yaml extraction.pooling). An optional
     sweep (config.yaml extraction.pooling_sweep) pools several modes from the
     SAME forward pass for a mean-vs-last sensitivity comparison.
  5. Accumulate into per-behaviour, per-layer matrices, flushed to shard files
     every `flush_every` chains to bound peak memory, then concatenated and
     saved as .npy with a `complete` integrity flag in metadata.

Output layout (in activations_dir/):
    {behaviour}_layer{n}.npy          float32 array (N_instances, hidden_dim)
    metadata.json                     includes `complete` + `saved_layers`
    pool_<mode>/                      extra pooling-sweep modes (if enabled)
    pooling_sweep.json                mean-vs-last comparison (if sweep enabled)

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
    sentence_end_char: "int | None" = None,
) -> list[int]:
    """
    Map a sentence's char offset to token positions in the full sequence.

    Returns a list of token indices: [preceding …] + [execution …]
    Returns [] if the sentence cannot be located.

    If *sentence_end_char* is given, execution tokens that START at or beyond
    it are excluded: 15.8% of target sentences are shorter than the 1+10
    window, so the unclipped window pools tokens from the NEXT sentence —
    a measured onset/surface-lexis bias (see config
    extraction.clip_window_to_sentence_end; default off to preserve
    comparability with existing extractions).
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
        p = onset + offset
        if (sentence_end_char is not None and offset > 0
                and offsets[p][0] >= sentence_end_char):
            break  # token starts past the sentence end — next sentence's text
        positions.append(p)
    return positions


# Canonical implementation lives in src/text_offsets.py (single source of truth);
# re-exported here under the historical private name for backward compatibility.
# Extraction itself uses the occurrence-aware whole-chain locator so verbatim
# repeats bind to successive occurrences (the first-occurrence rule produced
# 35–56% exact-duplicate rows; see CONFOUNDS_AND_REMEDIATION.md CF-13).
from src.text_offsets import find_sentence_offset as _find_sentence_offset
from src.text_offsets import locate_annotation_offsets

#: Bump when the sentence→span matching rule changes; recorded in metadata and
#: the row-provenance sidecar so analyses can refuse mixed-rule data.
SENTENCE_MATCHING_VERSION = "occurrence_aware_v1"


# ── Pooling over a behaviour span's token positions ──────────────────────────

POOLING_MODES = ("mean", "last", "first", "max")


def _pool(acts, mode: str):
    """Pool a (n_positions, hidden) activation slice into a (hidden,) vector.

    positions are ordered [preceding ...] + [execution ...] (increasing token
    index), so acts[-1] is the LAST execution token and acts[0] the first.

      mean  — average over positions (order-invariant; smears the within-span
              trajectory and can cancel opposing directions).
      last  — last execution token: in a causal transformer this is the only
              position that has attended over the whole span (most
              context-complete), and it is the residual the model decodes from.
      first — first position (onset; mostly the lexical marker).
      max   — element-wise max (robustness check).
    """
    if mode == "mean":
        return acts.mean(dim=0)
    if mode == "last":
        return acts[-1]
    if mode == "first":
        return acts[0]
    if mode == "max":
        return acts.max(dim=0).values
    raise ValueError(f"unknown pooling mode {mode!r}; expected one of {POOLING_MODES}")


# ── Main extraction ───────────────────────────────────────────────────────────

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
    pooling: Optional[str] = None,
    sweep_modes: Optional[list[str]] = None,
    clip_to_sentence_end: Optional[bool] = None,
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
        clip_to_sentence_end: stop the execution window at the sentence end
                            (config extraction.clip_window_to_sentence_end;
                            default False — 15.8% of target sentences are
                            shorter than the 1+10 window)
        flush_every:        flush accumulators to shard files every N chains to
                            bound peak memory (prevents the OOM-during-write
                            that killed the 2026-06-12 full-corpus run after
                            7/230 files and, earlier, truncated a behaviour to
                            1/28 layers)
        keep_in_memory:     if False, free each final array after saving and
                            return only structure/counts (runners that ignore
                            the return value should pass False — keeping the
                            full primary-mode matrices in RAM at concat time
                            re-adds ~6.5 GB of peak memory for nothing)

    Returns:
        {behaviour: {layer_idx: ndarray(N_instances, hidden_dim)}}
    """
    import torch
    from src.hooks import ActivationCache

    if behaviours is None:
        behaviours = TARGET_BEHAVIOURS
    if layers is None:
        layers = list(range(len(model.model.layers)))
    # Pooling strategy: explicit arg > config.yaml (extraction.pooling) > "mean".
    if pooling is None:
        try:
            from src.config import load_config
            pooling = load_config().get("extraction", {}).get("pooling", "mean")
        except Exception:
            pooling = "mean"
    if pooling not in POOLING_MODES:
        raise ValueError(f"unknown pooling {pooling!r}; expected one of {POOLING_MODES}")
    logger.info(f"Pooling mode: {pooling}")

    # Optional pooling SWEEP — pool several modes from the SAME forward pass.
    if sweep_modes is None:
        try:
            from src.config import load_config
            sweep_modes = load_config().get("extraction", {}).get("pooling_sweep") or None
        except Exception:
            sweep_modes = None
    if sweep_modes:
        bad = [m for m in sweep_modes if m not in POOLING_MODES]
        if bad:
            raise ValueError(f"unknown pooling_sweep modes {bad}; expected {POOLING_MODES}")
    extra_modes = [m for m in (sweep_modes or []) if m != pooling]
    if extra_modes:
        logger.info(f"Pooling sweep: also extracting {extra_modes} (one shared forward pass)")

    # Window clipping: explicit arg > config extraction.clip_window_to_sentence_end > False.
    if clip_to_sentence_end is None:
        try:
            from src.config import load_config
            clip_to_sentence_end = bool(load_config().get("extraction", {})
                                        .get("clip_window_to_sentence_end", False))
        except Exception:
            clip_to_sentence_end = False
    if clip_to_sentence_end:
        logger.info("Window clipping ON: execution tokens stop at sentence end")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Pooling modes to materialise: the primary (-> save_dir) plus any sweep
    # extras (-> save_dir/pool_<mode>/). All share the SAME forward pass.
    streams = [pooling] + extra_modes

    def _stream_dir(mode):
        return save_dir if mode == pooling else save_dir / f"pool_{mode}"

    # Accumulators: stream -> behaviour -> layer -> list of float32 vectors. To
    # bound peak memory, accumulators are flushed to per-(stream, behaviour,
    # layer) shard files every `flush_every` chains and concatenated at the end.
    acc: dict[str, dict] = {
        m: {b: {l: [] for l in layers} for b in behaviours} for m in streams
    }
    n_extracted = {b: 0 for b in behaviours}
    n_skipped = {b: 0 for b in behaviours}
    # Row-provenance sidecar: one record per accepted row, in exact row order
    # (identical across streams — they share span selection). Tiny (dicts), so
    # it is NOT sharded; see src/row_provenance.py for the loader.
    row_index: dict[str, list[dict]] = {b: [] for b in behaviours}

    shard_dir = save_dir / "_shards"
    if shard_dir.exists():
        import shutil
        shutil.rmtree(shard_dir)          # avoid contaminating with a prior run's shards
    shard_dir.mkdir(parents=True, exist_ok=True)

    def _shard_sub(mode):
        return shard_dir if mode == pooling else shard_dir / f"pool_{mode}"

    shard_idx = {m: {b: {l: 0 for l in layers} for b in behaviours} for m in streams}

    def _flush() -> None:
        """Write current in-RAM vectors to shard files and free the RAM."""
        for m in streams:
            sd = _shard_sub(m)
            sd.mkdir(parents=True, exist_ok=True)
            for b in behaviours:
                for l in layers:
                    vecs = acc[m][b][l]
                    if not vecs:
                        continue
                    np.save(sd / f"{b}_layer{l}.part{shard_idx[m][b][l]:05d}.npy", np.stack(vecs))
                    shard_idx[m][b][l] += 1
                    acc[m][b][l] = []

    chains = annotated_chains[:max_chains] if max_chains else annotated_chains

    for chain_i, chain in enumerate(tqdm(chains, desc="Extracting activations")):
        annotations = chain.get("annotations", [])
        if not annotations:
            continue

        chain_text: str = chain["chain"]
        prompt_text: str = chain["prompt"]
        full_text: str = prompt_text + chain_text
        chain_offset: int = len(prompt_text)
        chain_id = chain.get("chain_id") or chain.get("task_id")

        # Locate ALL annotations up front (occurrence-aware): the cursor must
        # advance over every annotation — including non-target labels — so
        # repeats bind to successive occurrences deterministically (CF-13).
        sent_offsets = locate_annotation_offsets(
            chain_text, [a.get("text", "") for a in annotations]
        )

        # Tokenise once per chain (with offset mapping for char->token alignment)
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

            for ann_idx, ann in enumerate(annotations):
                cat = ann["label"]
                if cat not in behaviours:
                    continue

                # Occurrence-aware sentence position in chain_text
                sent_offset = sent_offsets[ann_idx]
                if sent_offset is None:
                    n_skipped[cat] += 1
                    continue

                # Convert chain-relative char offset -> full-text char offset
                abs_offset = chain_offset + sent_offset
                abs_end = (abs_offset + len(ann.get("text", ""))
                           if clip_to_sentence_end else None)

                positions = _sentence_to_token_positions(
                    full_text, abs_offset, offsets, n_preceding, n_execution,
                    sentence_end_char=abs_end,
                )
                # Filter to valid range
                positions = [p for p in positions if 0 <= p < seq_len]
                if not positions:
                    n_skipped[cat] += 1
                    continue

                for layer_idx in layers:
                    sl = cache[layer_idx][0, positions, :]   # (n_positions, hidden)
                    for m in streams:   # primary + sweep extras, same forward pass
                        acc[m][cat][layer_idx].append(_pool(sl, m).numpy().astype(np.float32))

                row_index[cat].append({
                    "chain_id": chain_id,
                    "annotation_index": ann_idx,
                    "char_offset": sent_offset,
                    "token_start": positions[0],
                    "n_positions": len(positions),
                })
                n_extracted[cat] += 1

        if (chain_i + 1) % flush_every == 0:
            _flush()

    _flush()  # final partial batch

    # ── Save: concatenate shards per (stream, behaviour, layer) ───────────
    # Peak memory here is bounded to ONE (behaviour, layer) array at a time, not
    # the whole corpus, so the final write cannot OOM the way it did before.
    results: dict[str, dict[int, np.ndarray]] = {b: {} for b in behaviours}
    logger.info("Extraction summary:")
    for beh in behaviours:
        logger.info(f"  {beh}: {n_extracted[beh]} extracted, {n_skipped[beh]} skipped")

    sidecar = {
        "version": 1,
        "sentence_matching": SENTENCE_MATCHING_VERSION,
        "pooling": pooling,
        "clip_window_to_sentence_end": clip_to_sentence_end,
        "rows": row_index,
    }

    for m in streams:
        tdir = _stream_dir(m)
        tdir.mkdir(parents=True, exist_ok=True)
        sd = _shard_sub(m)
        saved_layers: dict[str, list[int]] = {b: [] for b in behaviours}
        for beh in behaviours:
            for layer_idx in layers:
                shards = sorted(sd.glob(f"{beh}_layer{layer_idx}.part*.npy"))
                if not shards:
                    if m == pooling:
                        logger.warning(f"    {beh} layer {layer_idx}: no vectors!")
                    continue
                mat = np.concatenate([np.load(s) for s in shards], axis=0)
                np.save(tdir / f"{beh}_layer{layer_idx}.npy", mat)
                if m == pooling and keep_in_memory:
                    results[beh][layer_idx] = mat
                saved_layers[beh].append(layer_idx)
                for s in shards:
                    s.unlink()
                del mat
        # Integrity marker: True only when every behaviour with instances has all
        # its layers on disk. Written LAST so downstream (verify_extraction_complete)
        # can refuse a partial set instead of silently using it.
        complete = all(len(saved_layers[b]) == len(layers)
                       for b in behaviours if n_extracted[b] > 0)
        with open(tdir / "metadata.json", "w") as f:
            json.dump({
                "behaviours": behaviours, "layers": layers,
                "n_preceding": n_preceding, "n_execution": n_execution,
                "pooling": m,
                "pooling_sweep": sweep_modes if m == pooling else None,
                "n_extracted": n_extracted, "n_skipped": n_skipped,
                "saved_layers": saved_layers, "complete": complete,
                "sentence_matching": SENTENCE_MATCHING_VERSION,
                "clip_window_to_sentence_end": clip_to_sentence_end,
            }, f, indent=2)
        # Row-provenance sidecar (same rows/order for every stream; only the
        # pooling tag differs).
        with open(tdir / "row_index.json", "w") as f:
            json.dump({**sidecar, "pooling": m}, f, indent=2)
        if m == pooling and not complete:
            missing = {b: len(saved_layers[b]) for b in behaviours
                       if n_extracted[b] > 0 and len(saved_layers[b]) != len(layers)}
            logger.error(f"Extraction INCOMPLETE — behaviours missing layers "
                         f"(have/expected {len(layers)}): {missing}. metadata.complete=False.")

    import shutil
    shutil.rmtree(shard_dir, ignore_errors=True)   # remove the now-empty shard tree

    if extra_modes:
        try:
            _write_pooling_sweep_report(save_dir, primary=pooling, modes=streams,
                                        layers=layers, behaviours=behaviours)
        except Exception as e:  # report is a convenience; never fail extraction over it
            logger.warning(f"pooling-sweep report skipped: {type(e).__name__}: {e}")

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


def _write_pooling_sweep_report(save_dir: Path, primary: str, modes: list,
                                layers: list, behaviours: list) -> dict:
    """After a sweep extraction, compare pooling modes at the steering layer:
    cos(primary_single_direction, mode_single_direction) per behaviour (does the
    steering direction survive the pooling choice?) and d_eff_70 per mode (does
    the dimensionality?). Writes save_dir/pooling_sweep.json. The primary mode's
    activations live in save_dir; each extra mode in save_dir/pool_<mode>/."""
    from src.steering import build_steering_vectors
    from src.pca import analyse_behaviour

    steer_layer = 27 if 27 in layers else max(layers)

    def _dir(mode):
        return save_dir if mode == primary else save_dir / f"pool_{mode}"

    singles, dims = {}, {}
    for mode in modes:
        vecs = build_steering_vectors(_dir(mode), layer=steer_layer,
                                      k_values=["auto"], variance_threshold=0.70)
        singles[mode] = {b: vecs[b]["single_direction"] for b in vecs}
        dims[mode] = {}
        for beh in behaviours:
            p = _dir(mode) / f"{beh}_layer{steer_layer}.npy"
            if p.exists():
                r = analyse_behaviour(np.load(p))
                dims[mode][beh] = {"n": int(r["n_samples"]), "d_eff_70": r["d_eff_70"]}

    report = {"steer_layer": steer_layer, "primary": primary, "modes": modes,
              "behaviours": {}}
    for beh in behaviours:
        vp = singles.get(primary, {}).get(beh)
        rec = {"cos_vs_primary": {}, "d_eff_70": {}}
        for mode in modes:
            vm = singles.get(mode, {}).get(beh)
            rec["cos_vs_primary"][mode] = (float(vp @ vm)
                                           if vp is not None and vm is not None else None)
            rec["d_eff_70"][mode] = dims.get(mode, {}).get(beh, {}).get("d_eff_70")
        report["behaviours"][beh] = rec

    (save_dir / "pooling_sweep.json").write_text(json.dumps(report, indent=2))
    logger.info(f"Pooling-sweep report -> {save_dir/'pooling_sweep.json'} "
                f"(compared {modes} at layer {steer_layer})")
    return report


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
