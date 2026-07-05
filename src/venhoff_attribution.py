"""
FAITHFUL reimplementation of Venhoff et al.'s per-behaviour steering-LAYER
attribution (arXiv:2506.18167; github.com/cvenhoff/steering-thinking-llms).

Why this module exists (and how it differs from 07c / 07d)
----------------------------------------------------------
`src/attribution_patching.py` (07c) and `src/layer_sweep.py` (07d) are OUR two
per-behaviour layer-selection methods. Neither is Venhoff's. This module is a
third, deliberately literal port of the *released* Venhoff attribution code so
the published R1-1.5B layer table (backtracking→17, uncertainty-estimation→18,
example-testing→15, adding-knowledge→18) can be reproduced in our stack and
compared like-for-like. Faithfulness over cleanliness — the apparent quirks of
the original (the detached-copy KL, the ``start-1``-only read-out, the cross-
label ``!= label`` span selection, the ``μ_b − μ_overall`` feature, the unit
re-normalisation) are reproduced exactly and flagged inline.

Venhoff's attribution, in one line:

    effect(ℓ) = | unit(u_ℓ) · ∂KL/∂a_ℓ |   read at the pre-onset token (start-1),

where ``u_ℓ = mean(b activations at ℓ) − mean(ALL-behaviour activations at ℓ)``
(the "overall" mean), re-normalised to UNIT before the dot product, and
``∂KL/∂a_ℓ`` is the gradient of a KL-with-detached-copy sensitivity scalar read
at the LM-head logits, position ``start-1``, w.r.t. the residual at layer ℓ.
One forward + one backward per scored span; accumulate ``|u·grad|`` over the
behaviour's spans, divide by the number of spans, average over examples →
a per-behaviour per-layer curve whose argmax is the chosen layer.

Three load-bearing differences vs OUR 07c (all small, all here):

  (a) Read-out = LM-head logits KL at ``start-1`` (Venhoff's
      ``logits[0, start-1:start].mean(dim=0)``), NOT a projection of the L27
      residual onto ``r_b``. Read at the OUTPUT, never an intermediate layer —
      that is what removes 07c's read-out-proximity artefact (the monotone ramp
      to L26-27). See :func:`lm_head_kl_metric_fn`.
  (b) Gradient projected onto the UNIT behaviour direction ``u_ℓ``, NOT onto
      ``(act_corrupt − act_clean)``. No corrupt chain, no alignment — Venhoff
      scores a SINGLE chain. See :func:`venhoff_attribution_single`.
  (c) ``u_ℓ = μ_b − μ_overall`` (the overall = mean over ALL labels' spans), NOT
      ``μ_b − μ_other-three``. See :func:`behaviour_overall_directions`.

Stub-testability
----------------
Like ``src/attribution_patching.py`` and ``src/layer_sweep.py``, everything here
talks to the model ONLY through ``model.model.layers`` (the decoder blocks; via
the shared :func:`iter_residual_layers`) and the model's forward output, whose
``.logits`` are the LM-head read-out (via the shared :func:`_logits_of`). A tiny
``nn.Module`` stub that, BY CONSTRUCTION, makes a MIDDLE layer the most causal
for the read-out satisfies the contract and is exercised in
``tests/test_venhoff_attribution.py`` with no GPU and no real model. Nothing
here loads a model or touches a GPU; the runner
(``07e_venhoff_layer_attribution.py``) wires it to R1-1.5B.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

# Reuse — do NOT re-define — the shared thin-model contract and aggregation, so
# this module stays consistent with 07c/07d (same access path, same output shape
# the triangulation consumer reads).
from src.attribution_patching import (
    AttributionResult,
    aggregate_attribution_curves,
    iter_residual_layers,
    _residual_of,
)
from src.layer_sweep import _logits_of, argmax_layer

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# 1. The metric — VERBATIM from Venhoff (DO NOT "fix" it)  [spec §1]
# ════════════════════════════════════════════════════════════════════════════


def compute_kl_divergence_metric(logits):
    """Compute KL divergence between predicted distribution and detached version

    VERBATIM from ``vector-layer-attribution/analyze_layer_effects.py``. This is
    the attribution-patching *sensitivity* trick, not a meaningful divergence:

      * Both arguments are ``log_softmax`` outputs; the second is ``.detach()``ed.
        ``F.kl_div(input, target)`` expects ``input`` = log-probs and ``target``
        = probs (unless ``log_target=True``, which is NOT set). Passing log-probs
        for ``target`` is technically an API misuse — but it is EXACTLY what the
        original does, and the resulting gradient is what DEFINES their
        attribution. Do not switch to true cross-entropy, swap the arg order, or
        change ``reduction``.
      * At the evaluation point the two distributions are numerically identical,
        so the VALUE ≈ 0. The point is the GRADIENT: ``∂KL/∂logits`` backpropped
        through the network gives ``∂L/∂a_ℓ`` — the per-layer residual gradient
        the attribution formula contracts against ``u_ℓ``.
      * ``reduction='batchmean'`` divides the summed KL by the batch size; with a
        single pooled vector (the ``start-1`` read-out, batch size 1) it is just
        the sum over the vocab dim.
    """
    import torch.nn.functional as F

    probs = F.log_softmax(logits, dim=-1)
    detached_probs = F.log_softmax(logits.detach(), dim=-1)
    return F.kl_div(probs, detached_probs, reduction='batchmean')


# ════════════════════════════════════════════════════════════════════════════
# 2. Per-layer feature vector  u_ℓ = μ_b − μ_overall  (unit at scoring)  [§3]
# ════════════════════════════════════════════════════════════════════════════


def overall_mean_per_layer(
    activations_dir,
    all_labels: Sequence[str],
    layers: Sequence[int],
) -> dict[int, np.ndarray]:
    """The "overall" mean ``μ_overall_ℓ`` = mean over ALL labels' activation rows
    at each layer, from ``{activations_dir}/{label}_layer{ℓ}.npy``.

    This is Venhoff's ``mean_vectors["overall"]`` term — the mean activation over
    the entire annotated thinking region across ALL behaviours, the term
    subtracted to form every feature vector (§3a). Built by *pooling rows* of the
    per-label activation matrices: concatenating the rows of every label file at
    layer ℓ and taking their mean is the row-weighted mean over all annotated
    spans, which matches the spirit of Venhoff's overall pool (the union of every
    label's spans). Float64. Layers with no label files are skipped (logged).

    NB on the "overall" pool: the original builds ``overall`` from the raw
    ``min_pos:max_pos`` thinking span (no ``-1``/``+10`` adjustment) while the
    per-label pool uses ``start-1 : min(end-1, start+10)``. We do not have the
    raw thinking-span activations cached separately; the all-label row mean is
    the faithful reconstruction available from the extracted ``*_layer{ℓ}.npy``
    matrices (the same rows the steering vectors are built from). Documented as a
    deviation in the runner.
    """
    activations_dir = Path(activations_dir)
    out: dict[int, np.ndarray] = {}
    for L in layers:
        parts = []
        for lab in all_labels:
            p = activations_dir / f"{lab}_layer{L}.npy"
            if p.exists():
                parts.append(np.load(p).astype(np.float64))
        if not parts:
            logger.warning(f"  overall: no label activations at L{L}; skip")
            continue
        rows = np.concatenate(parts, axis=0)
        out[int(L)] = rows.mean(axis=0)
    return out


def behaviour_overall_directions(
    activations_dir,
    behaviour: str,
    layers: Sequence[int],
    all_labels: Sequence[str],
    *,
    overall_means: Optional[dict[int, np.ndarray]] = None,
) -> dict[int, np.ndarray]:
    """Venhoff's per-layer feature vector ``u_ℓ`` for ``behaviour`` (UNIT-normed).

    ``u_ℓ = mean(behaviour rows at ℓ) − μ_overall_ℓ`` where ``μ_overall_ℓ`` is
    the mean over ALL labels' rows (§3a), then **unit-normalised per layer**
    (§3b.2: the attribution score uses ``u_ℓ / ‖u_ℓ‖``). This is the operative
    direction for the attribution dot product.

    Distinction from ``src.layer_sweep.per_layer_directions`` (07c/07d): that
    builds ``μ_b − μ_other-three`` (OFF = the OTHER behaviours). Here OFF is the
    GLOBAL all-label mean — Venhoff's ``overall``. The two differ whenever the
    behaviour itself is a non-negligible fraction of the corpus (it is included
    in ``overall`` but not in ``other-three``), and whenever ``initializing`` /
    ``deduction`` rows are present (they are in ``overall`` but never in the
    4-behaviour ``other-three``). This is change (c).

    The storage-time rescale-to-overall-norm (Venhoff §3b.1) is INERT for the
    attribution score — the score re-normalises ``u_ℓ`` to unit norm anyway — so
    it is intentionally omitted here (it matters only for actual steering
    generation, which this module does not do). Pure / CPU. Returns
    ``{ℓ: (d,) unit vector}`` for every ℓ whose behaviour file exists.

    ``overall_means`` may be passed (e.g. computed once via
    :func:`overall_mean_per_layer` and reused across behaviours) to avoid
    re-reading every label file per behaviour.
    """
    activations_dir = Path(activations_dir)
    if overall_means is None:
        overall_means = overall_mean_per_layer(activations_dir, all_labels, layers)
    out: dict[int, np.ndarray] = {}
    for L in layers:
        on_path = activations_dir / f"{behaviour}_layer{L}.npy"
        if not on_path.exists():
            logger.warning(f"  {behaviour}: missing {on_path}; skip layer {L}")
            continue
        if L not in overall_means:
            logger.warning(f"  {behaviour}: no overall mean at L{L}; skip")
            continue
        on = np.load(on_path).astype(np.float64)
        if on.shape[0] == 0:
            logger.warning(f"  {behaviour}: empty activations at L{L}; skip")
            continue
        u = on.mean(axis=0) - overall_means[L]          # μ_b − μ_overall  (§3a)
        norm = float(np.linalg.norm(u))
        if not np.isfinite(norm) or norm < 1e-12:
            logger.warning(f"  {behaviour}: near-zero/non-finite u at L{L}; skip")
            continue
        out[int(L)] = (u / norm).astype(np.float64)     # UNIT (§3b.2)
    return out


# ════════════════════════════════════════════════════════════════════════════
# 3. Annotated spans → token (start, end) positions  [spec §6b, VERBATIM logic]
# ════════════════════════════════════════════════════════════════════════════


def get_char_to_token_map(text: str, tokenizer) -> dict:
    """Map character positions → token positions (VERBATIM Venhoff §6b).

    Uses the HF fast-tokenizer ``offset_mapping`` from ``encode_plus`` to assign
    every character in ``[start, end)`` of a token to that token's index.
    Requires a fast tokenizer (DeepSeek-R1-Distill-Qwen-1.5B ships one).
    """
    token_offsets = tokenizer(text, return_offsets_mapping=True)['offset_mapping']
    char_to_token: dict[int, int] = {}
    for token_idx, (start, end) in enumerate(token_offsets):
        for char_pos in range(start, end):
            char_to_token[char_pos] = token_idx
    return char_to_token


def get_label_positions(annotated_thinking: str, response_text: str, tokenizer) -> dict:
    """Parse annotated spans → ``{label: [(token_start, token_end), ...]}``.

    VERBATIM logic from Venhoff ``utils.get_label_positions`` (§6b):

      * regex ``\\["(\\S+?)"\\](.*?)\\["end-section"\\]`` (DOTALL) captures
        ``group(1)=label`` (non-greedy, no whitespace) and ``group(2)=span text``;
      * the span text (stripped) is located in the RAW ``response_text`` via
        ``str.find`` (first occurrence);
      * ``token_start = char_to_token[text_pos]``;
        ``token_end = char_to_token[text_pos + len(text) - 1] + 1`` (exclusive,
        expanded to cover the final token);
      * degenerate / unlocatable spans are skipped.

    Note our annotated file stores spans as a list of ``{"label","text"}`` dicts
    (``src/annotation.py``), already parsed from this same delimiter format. The
    runner reconstructs the ``["label"]…["end-section"]`` string from that list
    so this verbatim parser drives the position mapping (keeping the operative
    Venhoff path), but :func:`label_positions_from_spans` is the direct,
    list-based equivalent for the runner's convenience.
    """
    label_positions: dict[str, list] = {}
    pattern = r'\["(\S+?)"\](.*?)\["end-section"\]'
    matches = list(re.finditer(pattern, annotated_thinking, re.DOTALL))
    char_to_token = get_char_to_token_map(response_text, tokenizer)
    for match in matches:
        label = match.group(1).strip()
        text = match.group(2).strip()
        if not text:
            continue
        text_pos = response_text.find(text)
        if text_pos >= 0:
            token_start = char_to_token.get(text_pos, None)
            token_end = char_to_token.get(text_pos + len(text) - 1, None)
            if token_end is not None:
                token_end += 1
            if token_start is None or token_end is None or token_start >= token_end:
                continue
            if label not in label_positions:
                label_positions[label] = []
            label_positions[label].append((token_start, token_end))
    return label_positions


def label_positions_from_spans(spans: Sequence[dict], response_text: str, tokenizer) -> dict:
    """List-based equivalent of :func:`get_label_positions` for OUR annotated
    format (``spans`` = ``[{"label","text"}, ...]`` from ``src/annotation.py``).

    Applies the IDENTICAL char→token mapping and ``start-1``-aware bounds as the
    verbatim parser (``token_end = char_to_token[text_pos+len(text)-1] + 1``;
    first-occurrence ``str.find``; skip degenerate). Returns
    ``{label: [(token_start, token_end), ...]}``. This avoids re-serialising the
    spans into a delimiter string and re-parsing them, while producing the same
    positions Venhoff's parser would.
    """
    char_to_token = get_char_to_token_map(response_text, tokenizer)
    label_positions: dict[str, list] = {}
    for span in spans:
        label = str(span.get("label", "")).strip()
        text = str(span.get("text", "")).strip()
        if not text:
            continue
        text_pos = response_text.find(text)
        if text_pos < 0:
            continue
        token_start = char_to_token.get(text_pos, None)
        token_end = char_to_token.get(text_pos + len(text) - 1, None)
        if token_end is not None:
            token_end += 1
        if token_start is None or token_end is None or token_start >= token_end:
            continue
        label_positions.setdefault(label, []).append((token_start, token_end))
    return label_positions


def cross_label_positions(label_positions: dict, label: str) -> list:
    """The CROSS-LABEL ``!= label`` span set for behaviour ``label`` (spec §6c).

    Venhoff's driver scores the curve for behaviour ``c`` over every annotated
    span that is **not** ``c``::

        positions = [s for cat in spans_by_cat if cat != label for s in spans_by_cat[cat]]

    i.e. "how much does feature ``c``'s direction influence the prediction at the
    ONSET of OTHER behaviours' segments." Reproduced exactly (it is non-obvious
    and easy to get wrong by scoring the label's own spans).
    """
    out = []
    for cat, spans in label_positions.items():
        if cat != label:
            out.extend(spans)
    return out


# ════════════════════════════════════════════════════════════════════════════
# 4. Differentiable read-out: LM-head KL at start-1  [spec §1, §2 — change (a)]
# ════════════════════════════════════════════════════════════════════════════

#: Key under which the runner / gather step stashes the in-graph LM-head logits
#: so the metric_fn can read them from the same ``captured`` dict the residuals
#: live in (no second hook needed; backward through the logits reaches every
#: captured residual). Distinct sentinel so it never collides with a layer index.
LOGITS_KEY = "__logits__"


def lm_head_kl_metric_fn(start: int) -> Callable[[dict], "object"]:
    """Build Venhoff's differentiable read-out ``metric_fn(captured) -> scalar``.

    The returned callable reads the in-graph LM-head logits stashed at
    ``captured[LOGITS_KEY]`` (shape ``(1, T, vocab)``) and returns

        compute_kl_divergence_metric(logits[0, start-1:start].mean(dim=0))

    EXACTLY as the original (§2): the logits at position ``start-1`` only (the
    slice ``start-1:start`` is length 1; ``.mean(dim=0)`` reduces that length-1
    axis to a ``(vocab,)`` vector — "the token that PREDICTS the behaviour
    onset"). The value ≈ 0; the gradient through it (after ``.backward()``) is
    the attribution signal. ``start`` must be ≥ 1 (a prefix is needed).

    The read-out is at the OUTPUT (logits), never an intermediate layer — this is
    change (a) that removes 07c's read-out-proximity artefact. Because the scored
    tensor is the captured logits, the backward pass propagates a gradient into
    EVERY captured residual layer that feeds them.
    """
    if start < 1:
        raise ValueError(f"start must be >= 1 (need a prefix), got {start}")

    def metric_fn(captured: dict):
        logits = _logits_of_captured(captured)          # (1, T, vocab), in graph
        # logits[0, start-1:start].mean(dim=0) -> (vocab,)   [Venhoff §2 verbatim]
        return compute_kl_divergence_metric(logits[0, start - 1:start].mean(dim=0))

    metric_fn.read_position = start - 1
    return metric_fn


def _logits_of_captured(captured: dict):
    """Pull the in-graph logits from the captured dict (stashed at LOGITS_KEY)."""
    if LOGITS_KEY not in captured:
        raise KeyError(
            f"captured dict is missing {LOGITS_KEY!r}; the LM-head logits must be "
            f"stashed in-graph before the metric_fn runs (see "
            f"gather_residuals_and_logits)."
        )
    return captured[LOGITS_KEY]


# ════════════════════════════════════════════════════════════════════════════
# 5. Gradient capture: one fwd+bwd, per-layer residual grads + in-graph logits
# ════════════════════════════════════════════════════════════════════════════


def gather_residuals_and_logits(model, input_ids, layers: Sequence[int]):
    """Single forward pass; capture each layer's residual (with ``retain_grad``)
    AND the LM-head logits, both IN-GRAPH.

    The same forward-hook + ``retain_grad`` machinery as
    ``src.attribution_patching._gather_residuals(requires_grad=True)``, plus the
    forward output's ``.logits`` stashed at ``captured[LOGITS_KEY]`` so the
    Venhoff KL read-out can be applied in-graph and a single ``.backward()``
    populates ``.grad`` on every captured residual. Returns
    ``(captured, forward_output)`` where ``captured[L]`` is layer ``L``'s
    residual tensor (batch dim kept) and ``captured[LOGITS_KEY]`` is the logits.

    This replaces nnsight (the original used ``model.trace()`` /
    ``lm_head.output.save()`` / ``layers[ℓ].output[0].grad``); we use plain
    PyTorch forward hooks + ``retain_grad`` + ``model(...).logits``, which gives
    the identical gradient (post-block residual stream) without the nnsight
    dependency — spec hard-requirement 5.
    """
    blocks = iter_residual_layers(model)
    captured: dict = {}
    handles = []

    def make_hook(L):
        def hook(_module, _inp, output):
            h = _residual_of(output)
            h.retain_grad()
            captured[L] = h
        return hook

    for L in layers:
        handles.append(blocks[L].register_forward_hook(make_hook(L)))
    try:
        out = model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()
    captured[LOGITS_KEY] = _logits_of(out)              # in-graph logits
    return captured, out


# ════════════════════════════════════════════════════════════════════════════
# 6. The attribution:  effect(ℓ) = | unit(u_ℓ) · grad_ℓ[start-1] |   [§4]
# ════════════════════════════════════════════════════════════════════════════


def venhoff_attribution_single(
    model,
    input_ids,
    positions: Sequence[tuple],
    directions: dict[int, np.ndarray],
    layers: Sequence[int],
    *,
    behaviour: str = "",
) -> "AttributionResult | None":
    """Venhoff per-layer attribution for ONE chain over its (cross-label) spans.

    For each span ``(start, end)`` in ``positions``:
      * forward on ``input_ids[:, :end]`` (truncated to the span end, §5) with the
        residuals + LM-head logits captured in-graph;
      * ``value = compute_kl_divergence_metric(logits[0, start-1:start].mean(0))``
        (§1, §2) and ``value.backward()`` — ONE forward + ONE backward per span;
      * for every layer ℓ::

            uℓ   = directions[ℓ] / ‖directions[ℓ]‖              # unit  (§3b.2)
            g    = grad_ℓ[0, start-1:start]                     # (1, d)  (§4)
            eff  = | einsum('d,sd->s', uℓ, g).mean() |          # single pos → mean is id
            acc[ℓ] += eff

    then ``acc[ℓ] /= len(positions)`` (§4). The result's ``per_layer`` is that
    per-span-averaged ``|u·grad|`` curve; ``per_position`` carries the raw
    ``u·grad`` (signed) at ``start-1`` for each span. Returns ``None`` if
    ``positions`` is empty (Venhoff's ``if len(label_positions) == 0: return
    None``).

    Reuses the shared :class:`AttributionResult` shape so
    :func:`aggregate_attribution_curves` / :func:`argmax_layer` consume it
    directly. Single chain only — NO corrupt chain, NO alignment (change (b)).
    """
    import torch

    positions = [(int(s), int(e)) for (s, e) in positions]
    if len(positions) == 0:                              # Venhoff: return None
        return None

    layers = [int(L) for L in layers if int(L) in directions]

    # Pre-unit-normalise the per-layer feature vectors once (they are reused for
    # every span). ``directions`` may already be unit (from
    # behaviour_overall_directions); re-normalising is idempotent and matches the
    # original's ``feature_activation[ℓ] / feature_activation[ℓ].norm()`` (§3b.2).
    unit_dirs: dict[int, np.ndarray] = {}
    for L in layers:
        v = np.asarray(directions[L], dtype=np.float64)
        n = float(np.linalg.norm(v))
        unit_dirs[L] = v / n if (np.isfinite(n) and n > 1e-12) else v

    acc: dict[int, float] = {L: 0.0 for L in layers}
    per_position: dict[int, dict[int, float]] = {L: {} for L in layers}

    for (start, end) in positions:
        T = int(input_ids.shape[-1])
        if start < 1 or start >= T:
            # No prefix to predict the onset, or out of range: this span cannot
            # be scored. (Venhoff's data construction keeps start >= 1; guard
            # defensively and skip — it still counts toward len(positions) in the
            # original's divisor, so we DON'T re-count: see note below.)
            continue
        end = min(end, T)
        if end < start + 1:
            # The forward must include at least up to ``start`` so the read-out
            # at start-1 predicts the onset token at ``start``.
            end = min(start + 1, T)

        ids_slice = input_ids[:, :end]                  # truncate to span end (§5)

        model.zero_grad(set_to_none=True)
        captured, _ = gather_residuals_and_logits(model, ids_slice, layers)
        metric_fn = lm_head_kl_metric_fn(start)         # KL at start-1  (§1,§2)
        value = metric_fn(captured)
        value.backward()

        for L in layers:
            h = captured[L]
            g = h.grad
            if g is None:
                # Metric did not depend on this layer's activation (downstream of
                # the read-out is impossible here — the read-out is the logits —
                # but guard anyway). Contributes 0 to this span.
                per_position[L][start] = 0.0
                continue
            # grad_ℓ[0, start-1:start] -> (1, d)   [the SAME single position as KL]
            grad_slice = g[0, start - 1:start]          # (1, d)
            u = torch.as_tensor(unit_dirs[L], dtype=grad_slice.dtype,
                                 device=grad_slice.device)
            # einsum('d,sd->s', uℓ, grad) -> (s=1,)  then .mean() (identity) .abs()
            dotted = torch.einsum('d,sd->s', u, grad_slice)
            signed = float(dotted.mean().item())
            per_position[L][start] = signed
            acc[L] += abs(signed)

        # Free graph references for this span before the next forward.
        del captured, value
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n = len(positions)                                  # Venhoff divides by len(positions)
    per_layer = {L: (acc[L] / n if n else float("nan")) for L in layers}

    return AttributionResult(
        behaviour=behaviour,
        layers=list(layers),
        per_position=per_position,
        per_layer=per_layer,
        positions=[s for (s, _e) in positions],
    )


# ════════════════════════════════════════════════════════════════════════════
# 7. Aggregation across examples → per-layer curve, argmax  [§4, §9]
# ════════════════════════════════════════════════════════════════════════════
#
# Venhoff reduces the per-example curves with ``np.nanmean`` (NaN→0 first). Our
# shared :func:`aggregate_attribution_curves` already does the finite-mean /
# SEM / n reduction over a list of ``{layer: score}`` dicts (it skips NaNs),
# which is the same point estimate; ``argmax_layer`` (from layer_sweep) then
# selects the chosen layer from the aggregated ``mean_effect`` curve. We re-
# export both so the runner imports everything Venhoff-related from one place.

__all__ = [
    "compute_kl_divergence_metric",
    "overall_mean_per_layer",
    "behaviour_overall_directions",
    "get_char_to_token_map",
    "get_label_positions",
    "label_positions_from_spans",
    "cross_label_positions",
    "lm_head_kl_metric_fn",
    "gather_residuals_and_logits",
    "venhoff_attribution_single",
    "aggregate_attribution_curves",
    "argmax_layer",
    "iter_residual_layers",
    "AttributionResult",
    "LOGITS_KEY",
    "PUBLISHED_LAYERS",
]


#: Venhoff's published R1-1.5B per-behaviour ``vector_layer`` (spec §7; paper
#: Table 2) — the argmax-attribution layer the reproduction should land near.
PUBLISHED_LAYERS = {
    "backtracking": 17,
    "uncertainty-estimation": 18,
    "example-testing": 15,
    "adding-knowledge": 18,
}
