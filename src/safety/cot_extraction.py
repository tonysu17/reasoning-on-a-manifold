"""CoT-spanning safety extraction — the object H1/H2/H3 actually need (red-team F1).

The existing ``src.safety.extraction.extract_prompt_activations`` runs a
prompt-only forward pass and pools the last *prompt* token. That is the Arditi
refusal-*output* probe — exactly the territory the niche concedes is settled
(safety_reasoning_extension.md §1). It cannot test any claim about safety
*reasoning across the chain*, because no chain is generated and no DSR span is
read. This module is the corrected path:

  1. generate the analysis-channel chain at a chosen ``reasoning_effort``
     (:func:`generate_safety_chain`);
  2. take its DSR consensus spans (from ``annotate_chains_dsr``);
  3. pool residual-stream activations *per DSR-labelled span* using the SAME
     pooling as Phase 4 — one preceding token plus the first ``n_execution``
     tokens, mean-pooled, located with the occurrence-aware offset mapper
     (:func:`src.activation_extraction._sentence_to_token_positions` and
     ``src.text_offsets.find_sentence_offset``).

The pooling is factored into a torch-free :func:`pool_dsr_spans` that takes a
``pool_fn(layer, positions) -> vector`` callback, so the span→token→vector logic
is unit-tested without a GPU; :func:`extract_dsr_span_activations` wires the real
``ActivationCache`` into it. Build-now-run-later, like the rest of ``src/safety``.

Output is keyed by DSR label (a span may contribute to several labels, since the
labels are non-exclusive), with a parallel row-meta list carrying the host
``chain_id`` and the stimulus label (harmful/benign) so downstream can split or
hold out by chain (CF-14 provenance).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import numpy as np

from src.activation_extraction import _sentence_to_token_positions
from src.safety.deliberation import DSR_LABELS
from src.text_offsets import find_sentence_offset

logger = logging.getLogger(__name__)


def generate_safety_chain(
    model,
    tokenizer,
    prompt_text: str,
    *,
    family: str = "gpt_oss",
    reasoning_effort: str = "high",
    max_new_tokens: int = 8192,
) -> str:
    """Generate one chain and return its reasoning (analysis-channel) text.

    Mirrors ``src.chain_gen``: format for the family, generate, decode with
    ``skip_special_tokens=False`` so harmony channel markers survive, then split
    out the analysis channel. The answer (``final`` channel) is discarded here;
    callers that need it should use the chain-generation pipeline directly.
    """
    import torch  # local import: keep the module importable offline

    from src.model_adapters import format_prompt, split_reasoning_final

    prompt = format_prompt(tokenizer, prompt_text, family=family,
                           reasoning_effort=reasoning_effort)
    enc = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    gen = out[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(gen, skip_special_tokens=False)
    reasoning, _final = split_reasoning_final(decoded, family=family)
    return reasoning


def pool_dsr_spans(
    chain_record: dict,
    offsets: Sequence[tuple],
    pool_fn: Callable[[int, list], np.ndarray],
    layers: Sequence[int],
    *,
    n_preceding: int = 1,
    n_execution: int = 10,
    seq_len: Optional[int] = None,
) -> tuple[dict, list]:
    """Pool one chain's DSR spans into per-label vectors (torch-free core).

    ``chain_record`` carries ``prompt``, ``chain``, a stimulus ``label``, and
    ``dsr_consensus.spans`` (each ``{text, dsr_labels, ...}``). ``offsets`` is the
    tokenizer offset mapping for ``prompt + chain``. ``pool_fn(layer, positions)``
    returns the mean-pooled vector at those token positions for that layer.

    Returns ``(acc, rows)`` where ``acc[label][layer]`` is a list of vectors and
    ``rows[label]`` is a parallel list of ``{chain_id, stim_label, text}`` meta.
    A span contributes one pooled vector to every DSR label it carries.
    """
    prompt_text = chain_record.get("prompt", "")
    chain_text = chain_record.get("chain", "")
    chain_offset = len(prompt_text)
    full_len = seq_len if seq_len is not None else len(offsets)
    cid = str(chain_record.get("chain_id", chain_record.get("task_id", "?")))
    stim_label = chain_record.get("label", "unknown")
    spans = chain_record.get("dsr_consensus", {}).get("spans", [])

    acc: dict = {lab: {L: [] for L in layers} for lab in DSR_LABELS}
    rows: dict = {lab: [] for lab in DSR_LABELS}

    for span in spans:
        labels = [l for l in span.get("dsr_labels", []) if l in DSR_LABELS]
        if not labels:
            continue
        sent_off = find_sentence_offset(chain_text, span.get("text", ""))
        if sent_off is None:
            continue
        abs_off = chain_offset + sent_off
        positions = _sentence_to_token_positions(
            prompt_text + chain_text, abs_off, list(offsets), n_preceding, n_execution)
        positions = [p for p in positions if 0 <= p < full_len]
        if not positions:
            continue
        per_layer = {L: pool_fn(L, positions) for L in layers}
        for lab in labels:
            for L in layers:
                acc[lab][L].append(per_layer[L])
            rows[lab].append({"chain_id": cid, "stim_label": stim_label,
                              "text": span.get("text", "")})
    return acc, rows


def extract_dsr_span_activations(
    model,
    tokenizer,
    chains: Sequence[dict],
    layers: Optional[Sequence[int]] = None,
    *,
    n_preceding: int = 1,
    n_execution: int = 10,
    max_chains: Optional[int] = None,
) -> tuple[dict, dict]:
    """Pool per-DSR-label residual activations across DSR-annotated chains.

    Returns ``(acts, row_meta)`` where ``acts[label][layer]`` is an ``(N, d)``
    matrix and ``row_meta[label]`` is the parallel list of row provenance. The
    caller can split each label's matrix into harmful/harmless by the
    ``stim_label`` in ``row_meta`` (H1: is the DSR object separable, and separable
    from a difficulty-matched capability baseline — see ``src.safety.capability``).
    """
    import torch

    from src.hooks import ActivationCache
    from src.model_adapters import locate_decoder_layers

    if layers is None:
        layers = list(range(len(locate_decoder_layers(model))))
    layers = list(layers)

    acc: dict = {lab: {L: [] for L in layers} for lab in DSR_LABELS}
    rows: dict = {lab: [] for lab in DSR_LABELS}

    use = chains[:max_chains] if max_chains else chains
    for chain in use:
        full_text = chain.get("prompt", "") + chain.get("chain", "")
        enc = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(model.device) for k, v in enc.items()}
        seq_len = inputs["input_ids"].shape[1]
        with ActivationCache(model, layers=layers) as cache:
            with torch.no_grad():
                model(**inputs)

            def _pool(L, positions, _cache=cache):
                return _cache.mean_at_positions(L, positions, batch_idx=0).numpy().astype(np.float32)

            c_acc, c_rows = pool_dsr_spans(
                chain, offsets, _pool, layers,
                n_preceding=n_preceding, n_execution=n_execution, seq_len=seq_len)
        for lab in DSR_LABELS:
            for L in layers:
                acc[lab][L].extend(c_acc[lab][L])
            rows[lab].extend(c_rows[lab])

    out = {lab: {L: (np.stack(v) if v else np.empty((0, 0), dtype=np.float32))
                 for L, v in by_layer.items()}
           for lab, by_layer in acc.items()}
    return out, rows


def split_by_stim_label(
    matrix: np.ndarray, row_meta: Sequence[dict], harmful="harmful", harmless="benign",
) -> tuple[np.ndarray, np.ndarray]:
    """Split a per-label matrix into (harmful_rows, harmless_rows) using row meta."""
    h_idx = [i for i, r in enumerate(row_meta) if r.get("stim_label") == harmful]
    l_idx = [i for i, r in enumerate(row_meta) if r.get("stim_label") == harmless]
    h = matrix[h_idx] if h_idx else np.empty((0, matrix.shape[1] if matrix.ndim == 2 else 0))
    l = matrix[l_idx] if l_idx else np.empty((0, matrix.shape[1] if matrix.ndim == 2 else 0))
    return h, l


__all__ = [
    "generate_safety_chain", "pool_dsr_spans", "extract_dsr_span_activations",
    "split_by_stim_label",
]
