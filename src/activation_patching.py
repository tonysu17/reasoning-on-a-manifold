"""
Activation patching for causal layer localisation of reasoning behaviours.

Companion document Section 4.2 (revised): promoted from "Medium / appendix"
to a Paper 2 main result.

Question
--------
For a behaviour-labelled sentence in a reasoning chain, which layer(s) in
the network are causally responsible for the behaviour being emitted?
The mech-interp gold-standard answer is activation patching (ROME-style):
replace the residual stream at a chosen layer with a counterfactual, and
measure how much the behaviour-relevant downstream output changes.

Method
------
Donor pair construction. For each target behaviour b, we pair a chain
where the model emitted b at sentence i (positive donor) with a chain where
the model did NOT emit b at the matched position (negative donor). The
positive's residual stream at layer L gets patched with the negative's
residual at the same layer-and-token-position, then forward pass continues.

Behavioural metric. Two complementary measurements:
  (a) Behavioural-shift via re-annotation: ask the same annotator (GPT-4o)
      whether the patched continuation still exhibits behaviour b. Effect
      size: change in P(b in continuation).
  (b) Direct logit-difference: the token-level next-token logit shift on
      hesitation/transition tokens characteristic of b (e.g., "wait",
      "however" for backtracking). Cheap; doesn't require API calls.

Layer scan. We run the patch independently at each of the saved layers
{3, 7, 10, 14, 17, 21, 24, 27} for Qwen-1.5B and report the layer-wise
effect curve. The layer at which the effect saturates is the minimum
causal depth for behaviour b.

This module provides the patching primitives. The runner is
07b_activation_patching.py.

Requires GPU (forward passes through full model).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PatchResult:
    """One patching trial's outcome."""
    target_behaviour: str
    layer:            int
    positive_chain_id:  str
    negative_chain_id:  str
    token_position:     int
    logit_diff_baseline: float        # P(b-tokens | unpatched positive) - P(b-tokens | negative)
    logit_diff_patched:  float        # P(b-tokens | patched positive)   - P(b-tokens | negative)
    relative_effect:     float        # 1 - (patched - negative) / (baseline - negative)
    n_b_tokens:          int          # number of behaviour-marker tokens scored


# Behaviour-marker token sets (cheap proxy for behaviour onset)

BEHAVIOUR_MARKER_TOKENS = {
    # Words that typically signal each behaviour in DeepSeek-R1 chain outputs.
    # These are first-pass markers; replace with model-specific token IDs at runtime.
    # Keys MUST match the canonical hyphen-lowercase labels emitted by src/annotation.py.
    "backtracking":           ["wait", "actually", "no", "hmm", "alternatively"],
    "uncertainty-estimation": ["maybe", "perhaps", "possibly", "might", "unsure", "guess"],
    "example-testing":        ["test", "example", "try", "consider", "case", "instance"],
    "adding-knowledge":       ["recall", "know", "formula", "fact", "definition", "theorem"],
}


def behaviour_marker_token_ids(tokenizer, behaviour: str) -> "list[int]":
    """Resolve behaviour-marker words to token IDs.

    Returns a list of leading-token IDs for each marker word, with
    leading-space variants included where the tokenizer makes that
    distinction (most BPE/Sentencepiece tokenizers do)."""
    words = BEHAVIOUR_MARKER_TOKENS.get(behaviour, [])
    ids = set()
    for w in words:
        for v in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            t = tokenizer.encode(v, add_special_tokens=False)
            if t:
                ids.add(t[0])
    return sorted(ids)


# Patching primitive

@contextmanager
def patched_residual(model, layer_idx: int, position: int, donor_vec):
    """Context manager that temporarily patches the residual stream at
    (layer_idx, position) on the next forward pass with `donor_vec`.

    The model must be a HuggingFace Qwen2-style architecture
    (model.model.layers[L] hooked at the output).
    """
    import torch
    block = model.model.layers[layer_idx]
    handle = None

    def _hook(module, _input, output):
        # Qwen2 block output is a tuple (hidden, ...) on HF; first element is residual
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[..., position, :] = donor_vec.to(hidden.dtype).to(hidden.device)
            return (hidden,) + output[1:]
        hidden = output.clone()
        hidden[..., position, :] = donor_vec.to(hidden.dtype).to(hidden.device)
        return hidden

    handle = block.register_forward_hook(_hook)
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


# Behaviour score: aggregate next-token probability over behaviour-marker tokens

def behaviour_marker_logprob(model, tokenizer, input_ids, behaviour: str) -> float:
    """Run a forward pass and return sum log-prob of behaviour-marker tokens
    at the NEXT token position."""
    import torch
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[..., -1, :]  # (B, V)
        logprobs = torch.log_softmax(logits, dim=-1)             # (B, V)
    marker_ids = behaviour_marker_token_ids(tokenizer, behaviour)
    if not marker_ids:
        return float("nan")
    marker_lp = logprobs[..., marker_ids]                       # (B, K)
    # Sum over markers (logsumexp gives joint event probability of "any marker fires")
    summed = torch.logsumexp(marker_lp, dim=-1)                 # (B,)
    return float(summed.mean().item())


# Single patching trial

def run_one_patch(
    model,
    tokenizer,
    positive_input_ids,
    negative_input_ids,
    positive_donor_residual,    # (d,) tensor — residual at (layer, token_position) of NEGATIVE chain
    layer:           int,
    token_position:  int,
    target_behaviour: str,
) -> PatchResult:
    """Compute baseline and patched behaviour-marker logprobs for one trial."""
    import torch

    # Baseline: unpatched positive
    base_pos = behaviour_marker_logprob(model, tokenizer, positive_input_ids, target_behaviour)
    base_neg = behaviour_marker_logprob(model, tokenizer, negative_input_ids, target_behaviour)

    # Patched: positive with residual at (layer, token_position) replaced by negative's residual
    with patched_residual(model, layer, token_position, positive_donor_residual):
        patched = behaviour_marker_logprob(model, tokenizer, positive_input_ids, target_behaviour)

    # Effect: how much of the (positive - negative) gap is closed by the patch?
    # relative_effect = 1.0 means patched matches negative (full effect)
    # relative_effect = 0.0 means patched matches positive (no effect)
    eps = 1e-8
    denom = (base_pos - base_neg) if abs(base_pos - base_neg) > eps else eps
    relative_effect = float(1.0 - (patched - base_neg) / denom)

    return PatchResult(
        target_behaviour=target_behaviour,
        layer=layer,
        positive_chain_id="",
        negative_chain_id="",
        token_position=int(token_position),
        logit_diff_baseline=float(base_pos - base_neg),
        logit_diff_patched=float(patched - base_neg),
        relative_effect=relative_effect,
        n_b_tokens=len(behaviour_marker_token_ids(tokenizer, target_behaviour)),
    )


# Layer sweep over a single donor pair

def layer_sweep_single_pair(
    model,
    tokenizer,
    positive_input_ids,
    negative_input_ids,
    negative_residuals_per_layer: dict,   # {layer_idx: (T, d) tensor of residuals from negative's forward}
    layers: Sequence[int],
    token_position: int,
    target_behaviour: str,
) -> "list[PatchResult]":
    """Return one PatchResult per layer in `layers`."""
    out = []
    for L in layers:
        donor = negative_residuals_per_layer[L][token_position]
        r = run_one_patch(
            model, tokenizer,
            positive_input_ids, negative_input_ids,
            positive_donor_residual=donor,
            layer=L, token_position=token_position,
            target_behaviour=target_behaviour,
        )
        out.append(r)
    return out


# Donor-pair selection

def select_donor_pairs(
    annotations: list,
    target_behaviour: str,
    n_pairs: int = 20,
    random_state: int = 42,
) -> list:
    """From a list of annotated chains, select n_pairs of (positive_chain,
    negative_chain) where the positive chain emits target_behaviour at
    sentence i, and the negative chain has a labelled sentence at a matched
    position but with a DIFFERENT label (preferably 'DEDUCTION', the most
    common neutral label).

    Each chain item is expected to have:
      {"task_id": str, "chain": str, "prompt": str,
       "annotations": [{"label": str, "text": str}, ...]}

    NOTE: token_start / token_end are NOT in the annotated file. They must be
    computed on the fly by the caller (e.g. via the same offset-mapping logic
    used in src/activation_extraction.py:_sentence_to_token_positions).
    """
    rng = np.random.default_rng(random_state)
    positives = []
    negatives = []
    for ch in annotations:
        for ann in ch.get("annotations", []):
            lbl = ann.get("label", "")
            if lbl == target_behaviour:
                positives.append((ch, ann))
            elif lbl == "deduction":
                negatives.append((ch, ann))
    if not positives or not negatives:
        return []

    rng.shuffle(positives)
    rng.shuffle(negatives)
    n = min(n_pairs, len(positives), len(negatives))
    return list(zip(positives[:n], negatives[:n]))


# Aggregate effect curve

def aggregate_layer_effect(results_by_pair: list) -> dict:
    """results_by_pair is a list-of-lists: outer index = pair, inner = layer.
    Returns {layer: {'mean_effect': ..., 'sem_effect': ..., 'n': ...}} averaged
    across pairs.
    """
    if not results_by_pair:
        return {}
    layers = sorted({r.layer for trial in results_by_pair for r in trial})
    out = {}
    for L in layers:
        effects = [r.relative_effect for trial in results_by_pair for r in trial if r.layer == L]
        effects = np.array(effects)
        finite = np.isfinite(effects)
        if not finite.any():
            out[L] = {"mean_effect": float("nan"), "sem_effect": float("nan"), "n": 0}
            continue
        e = effects[finite]
        out[L] = {
            "mean_effect": float(e.mean()),
            "sem_effect":  float(e.std() / max(1, np.sqrt(e.size - 1))),
            "n":           int(e.size),
        }
    return out
