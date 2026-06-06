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
     first / max — see _pool() and config.yaml extraction.pooling).
  5. Accumulate into per-behaviour, per-layer matrices and save as .npy files.

Output layout (in activations_dir/):
    {behaviour}_layer{n}.npy          float32 array (N_instances, hidden_dim)
    extraction_metadata.json

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


# Canonical implementation lives in src/text_offsets.py (single source of truth);
# re-exported here under the historical private name for backward compatibility.
from src.text_offsets import find_sentence_offset as _find_sentence_offset


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
    # Default "mean" keeps existing extractions (incl. the in-flight multi-
    # annotator arms) unchanged; set extraction.pooling: "last" in config to
    # switch without touching the runners. See _pool() and pooling_sweep.py.
    if pooling is None:
        try:
            from src.config import load_config
            pooling = load_config().get("extraction", {}).get("pooling", "mean")
        except Exception:
            pooling = "mean"
    if pooling not in POOLING_MODES:
        raise ValueError(f"unknown pooling {pooling!r}; expected one of {POOLING_MODES}")
    logger.info(f"Pooling mode: {pooling}")

    # Optional pooling SWEEP — pool several modes from the SAME forward pass and
    # auto-emit a mean-vs-last sensitivity comparison. Triggered by an explicit
    # sweep_modes arg, else config extraction.pooling_sweep. Because the forward
    # pass is shared, the only extra cost is the (cheap) repeated pooling + a few
    # more .npy files — so it runs "automatically on every re-extraction" once
    # extraction.pooling_sweep is set in config, with no extra GPU passes.
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

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Accumulators: behaviour → layer → list of float32 vectors
    acc: dict[str, dict[int, list]] = {b: {l: [] for l in layers} for b in behaviours}
    # Side accumulators for the extra sweep modes (same structure, per mode).
    acc_sweep: dict[str, dict] = {
        m: {b: {l: [] for l in layers} for b in behaviours} for m in extra_modes
    }
    n_extracted = {b: 0 for b in behaviours}
    n_skipped = {b: 0 for b in behaviours}

    chains = annotated_chains[:max_chains] if max_chains else annotated_chains

    for chain in tqdm(chains, desc="Extracting activations"):
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
                    sl = cache[layer_idx][0, positions, :]   # (n_positions, hidden)
                    acc[cat][layer_idx].append(_pool(sl, pooling).numpy().astype(np.float32))
                    for m in extra_modes:   # same span, other pooling modes (sweep)
                        acc_sweep[m][cat][layer_idx].append(
                            _pool(sl, m).numpy().astype(np.float32))

                n_extracted[cat] += 1

    # ── Save ────────────────────────────────────────────────────────────
    results: dict[str, dict[int, np.ndarray]] = {}
    logger.info("Extraction summary:")

    for beh in behaviours:
        results[beh] = {}
        logger.info(f"  {beh}: {n_extracted[beh]} extracted, {n_skipped[beh]} skipped")
        for layer_idx in layers:
            vecs = acc[beh][layer_idx]
            if vecs:
                mat = np.stack(vecs)  # (N, hidden_dim)
                np.save(save_dir / f"{beh}_layer{layer_idx}.npy", mat)
                results[beh][layer_idx] = mat
            else:
                logger.warning(f"    Layer {layer_idx}: no vectors!")

    with open(save_dir / "metadata.json", "w") as f:
        json.dump({
            "behaviours": behaviours,
            "layers": layers,
            "n_preceding": n_preceding,
            "n_execution": n_execution,
            "pooling": pooling,
            "pooling_sweep": sweep_modes,
            "n_extracted": n_extracted,
            "n_skipped": n_skipped,
        }, f, indent=2)

    # ── Sweep: save extra-mode activations to pool_<mode>/ + comparison report ──
    for m in extra_modes:
        mdir = save_dir / f"pool_{m}"
        mdir.mkdir(parents=True, exist_ok=True)
        for beh in behaviours:
            for layer_idx in layers:
                vs = acc_sweep[m][beh][layer_idx]
                if vs:
                    np.save(mdir / f"{beh}_layer{layer_idx}.npy", np.stack(vs))
        with open(mdir / "metadata.json", "w") as f:
            json.dump({"behaviours": behaviours, "layers": layers,
                       "n_preceding": n_preceding, "n_execution": n_execution,
                       "pooling": m, "n_extracted": n_extracted,
                       "n_skipped": n_skipped}, f, indent=2)
    if extra_modes:
        try:
            _write_pooling_sweep_report(save_dir, primary=pooling,
                                        modes=[pooling] + extra_modes, layers=layers,
                                        behaviours=behaviours)
        except Exception as e:  # report is a convenience; never fail extraction over it
            logger.warning(f"pooling-sweep report skipped: {type(e).__name__}: {e}")

    return results


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
