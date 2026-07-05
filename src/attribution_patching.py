"""
Attribution patching for *causal* per-behaviour layer selection.

Why this module exists (METHODOLOGY §5, CONFOUNDS_AND_REMEDIATION CF-10)
-----------------------------------------------------------------------
The steering layer is currently chosen either by borrowing Huang's published
L27 or by the **descriptive** PR-trough — neither is a *causal* criterion. The
honest question is: *at which layer does intervening most change the
behaviour?* That is an activation-patching question.

`src/activation_patching.py` already answers it by brute force (patch the
residual at one layer, re-run the forward pass, measure the shift), but two
things make its numbers untrustworthy:

  CF-10a  **the metric is a lexical proxy.** It scores a behaviour by the
          next-token log-prob mass on hand-picked marker words ("wait",
          "actually", "maybe", …). That conflates the *behaviour* with its
          *surface lexis*: a chain can backtrack without "wait", or say "wait"
          without backtracking, and the marker lists differ in size/frequency
          across behaviours so the scores are not comparable.
  CF-10b  **no positional alignment.** It patches token position *i* of one
          chain with position *i* of another, but position *i* is a different
          point in the reasoning across two non-aligned chains.

This module fixes both, and adds **attribution patching** (Syed et al. 2023,
*Attribution Patching Outperforms Automated Circuit Discovery*; Nanda 2023,
"Attribution Patching") — the first-order Taylor approximation to activation
patching that estimates the effect of patching *every* layer (and position)
from a **single** clean forward+backward pass instead of one forward pass per
(layer, position):

    effect(L, t) ≈ grad_clean[L, t] · (act_corrupt[L, t] − act_clean[L, t])

where ``grad_clean[L, t] = ∂ metric / ∂ act_clean[L, t]`` is read off the
backward pass of the metric on the clean run. The exact (brute-force) patch
replaces ``act_clean[L,t]`` with ``act_corrupt[L,t]`` and re-runs; attribution
patching is its linearisation about the clean activation, so it is cheap but
**first-order**: it is most accurate when the corrupt−clean gap is small and
the metric is locally linear in the residual, and it can mis-rank layers where
the network responds non-linearly to the swap. Treat the attribution curve as a
*screen* and confirm the chosen layer (and a couple of neighbours) with the
brute-force `activation_patching` reference (`brute_force_patch_effect`).

The two pieces are deliberately separable:

  * `BehaviourMetric` — the validated, geometry-based behaviour-effect metric.
    Pure: it consumes activations (a residual-stream tensor) and returns a
    scalar, so it can be the `metric_fn` of either the attribution or the
    brute-force patcher, and it is differentiable (it is a linear projection),
    which is exactly what attribution patching needs.
  * `attribution_patching` — the gradient×(corrupt−clean) estimator, written
    against a thin `LayeredModel` protocol (anything exposing `.model.layers`,
    the same indexing as `src/steered_inference.py`). That protocol is what
    lets the math be unit-tested on a tiny stub `nn.Module` with random tensors,
    with no GPU and no real model.

Nothing here loads a model or touches a GPU. The runner
(`07c_attribution_patching.py`) wires it to the real R1-1.5B.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# 1. Behaviour-effect metric (CF-10 fix)
# ════════════════════════════════════════════════════════════════════════════
#
# Interface (the contract the task asks for):
#
#     behaviour_effect(activations, behaviour, layer) -> float
#
# `activations` is the residual stream at `layer` for ONE position (shape (d,))
# or a (T, d) / (B, T, d) block; the metric reduces it to a scalar "how much
# this looks like `behaviour`" score. It is implemented as the projection of
# the residual onto the behaviour's steering geometry, because that geometry is
# *already the operational definition of the behaviour direction we steer along*
# (src/steering.py): the single diff-of-means direction r_b, or its restriction
# to the behaviour's own top-k PCA subspace. Scoring the same object we steer
# makes the causal-layer question ("where does moving along r_b matter?")
# self-consistent with the steering intervention, which the lexical proxy was
# not.
#
# Why this beats the lexical "wait/actually" proxy (CF-10a):
#   • Behaviour-defined, not lexis-defined. r_b = mean(ON_b) − mean(OFF) is
#     computed from activations the *annotator* labelled `behaviour`, so the
#     score tracks the behaviour representation, not a guess at its vocabulary.
#   • Commensurable across behaviours. Every behaviour is scored by the SAME
#     functional form (a unit-vector projection in the same units), so a layer
#     curve for backtracking and one for adding-knowledge can be compared. The
#     marker-token sets had different sizes and base rates, so their logprob
#     sums were not on a common scale.
#   • Differentiable & linear in the residual — exactly what makes the
#     attribution-patching gradient exact for this metric (the only
#     approximation left is the network's own non-linearity between layers,
#     which is the honest thing attribution patching approximates).
#   • Subspace option. With `mode="subspace"` the score is the norm of the
#     residual's projection onto the behaviour's top-k PCA subspace (the same
#     subspace the manifold-projected vector lives in), so a behaviour with no
#     single dominant axis is still scored by *its own geometry* rather than a
#     marker list.


@dataclass
class BehaviourMetric:
    """Geometry-based behaviour-effect metric — the CF-10 replacement for the
    lexical marker proxy.

    Parameters
    ----------
    direction : (d,) array
        Unit-norm steering direction r_b for the behaviour (``*_single.npy`` or
        a manifold ``*_manifold_k*.npy`` from ``results/steering_vectors/``).
        Used when ``mode="projection"``.
    subspace : (k, d) array, optional
        Orthonormal basis V of the behaviour's top-k PCA subspace, rows = basis
        vectors. Used when ``mode="subspace"``. (For a manifold vector built by
        ``src/steering.py`` the natural basis is that behaviour's PCA
        components; the runner can supply them, otherwise ``mode="projection"``
        with the manifold vector is the cheap proxy.)
    mode : {"projection", "subspace"}
        ``projection`` — signed scalar  s = rᵀ h   (how far along r_b).
        ``subspace``   — magnitude      s = ‖Vᵀ h‖ (how much energy in the
        behaviour's subspace; non-negative).
    center : (d,) array, optional
        Mean activation to subtract before projecting (e.g. mean(OFF) or the
        global mean). Recommended: it removes the component every residual
        shares so the score reflects the behaviour-specific deviation, not the
        residual's overall magnitude. Defaults to zeros.

    Notes
    -----
    The metric is a *pure function of activations*. It never sees tokens, so it
    cannot be fooled by surface lexis (CF-10a). It is linear (projection) /
    piecewise-smooth (subspace norm) in ``h``, so it is differentiable for
    attribution patching.
    """

    direction: Optional[np.ndarray] = None
    subspace: Optional[np.ndarray] = None
    mode: str = "projection"
    center: Optional[np.ndarray] = None
    behaviour: str = ""

    def __post_init__(self) -> None:
        if self.mode not in ("projection", "subspace"):
            raise ValueError(f"mode must be 'projection'|'subspace', got {self.mode!r}")
        if self.mode == "projection" and self.direction is None:
            raise ValueError("mode='projection' requires `direction`")
        if self.mode == "subspace" and self.subspace is None:
            raise ValueError("mode='subspace' requires `subspace`")

    # ---- numpy path (for the validated metric / brute-force reference) ----

    def _center_np(self, h: np.ndarray) -> np.ndarray:
        if self.center is None:
            return h
        return h - np.asarray(self.center, dtype=h.dtype)

    def score_np(self, h: np.ndarray) -> float:
        """Scalar behaviour score for a single residual vector ``h`` (shape (d,))."""
        h = np.asarray(h, dtype=np.float64)
        h = self._center_np(h)
        if self.mode == "projection":
            r = np.asarray(self.direction, dtype=np.float64)
            return float(h @ r)
        V = np.asarray(self.subspace, dtype=np.float64)        # (k, d)
        return float(np.linalg.norm(V @ h))

    # ---- torch path (differentiable; used inside attribution patching) ----

    def score_torch(self, h):
        """Differentiable scalar score for a torch residual ``h`` (shape (d,)).

        Mirrors :meth:`score_np` but stays in the autograd graph so the metric's
        gradient w.r.t. the residual can be read off a backward pass. Keeps the
        computation in float32 for numerical parity with the model dtype path.
        """
        import torch
        if self.center is not None:
            c = torch.as_tensor(self.center, dtype=h.dtype, device=h.device)
            h = h - c
        if self.mode == "projection":
            r = torch.as_tensor(self.direction, dtype=h.dtype, device=h.device)
            return torch.dot(h, r)
        V = torch.as_tensor(self.subspace, dtype=h.dtype, device=h.device)  # (k, d)
        return torch.linalg.vector_norm(V @ h)


def behaviour_effect(
    activations: np.ndarray,
    behaviour: str,
    layer: int,
    metric: "BehaviourMetric | dict[tuple[str, int], BehaviourMetric] | dict[str, BehaviourMetric]",
) -> float:
    """Validated behaviour-effect score — the clean interface the spec asks for.

    ``behaviour_effect(activations, behaviour, layer) -> float``

    The geometry (a :class:`BehaviourMetric`) is passed in rather than loaded
    here so this stays a pure, GPU-free, unit-testable function. ``metric`` may
    be:
      * a single :class:`BehaviourMetric` (used as-is), or
      * a dict keyed by ``(behaviour, layer)`` or by ``behaviour`` — the right
        one is selected, since the steering direction is layer-specific.

    ``activations`` may be a single residual ``(d,)`` or a block ``(T, d)`` /
    ``(B, T, d)``; a block is reduced to a scalar by **mean over positions**
    (NOT over the hidden dim) — see :func:`aligned_position_scores` for why a
    naive cross-chain position-by-position comparison is invalid and how to
    align first.
    """
    m = _select_metric(metric, behaviour, layer)
    a = np.asarray(activations, dtype=np.float64)
    if a.ndim == 1:
        return m.score_np(a)
    # Reduce over all leading (batch/position) axes, scoring each residual.
    flat = a.reshape(-1, a.shape[-1])
    return float(np.mean([m.score_np(row) for row in flat]))


def _select_metric(metric, behaviour: str, layer: int) -> "BehaviourMetric":
    if isinstance(metric, BehaviourMetric):
        return metric
    if isinstance(metric, dict):
        if (behaviour, layer) in metric:
            return metric[(behaviour, layer)]
        if behaviour in metric:
            return metric[behaviour]
        raise KeyError(
            f"No BehaviourMetric for behaviour={behaviour!r} layer={layer} "
            f"(have keys: {list(metric)[:6]}…)"
        )
    raise TypeError(f"metric must be BehaviourMetric or dict, got {type(metric)}")


# ════════════════════════════════════════════════════════════════════════════
# 2. Positional alignment across chains (CF-10b)
# ════════════════════════════════════════════════════════════════════════════
#
# Patching/attribution compares "the same place" in two chains. Token index t
# is NOT the same place across two chains of different length / pacing. We offer
# two alignment strategies, both returning index pairs into the two chains:
#
#   • anchored:    align on a known behaviour-onset anchor in each chain (the
#                  annotation's sentence-onset token, computed upstream). This
#                  is the right default for the donor-pair design: positive and
#                  negative are aligned at the boundary token where the
#                  behaviour does / does not begin, and a window around it is
#                  compared. (This generalises the old "always patch the last
#                  token" hack into "patch the behaviour-onset token and a
#                  symmetric window".)
#   • proportional: map fractional position p∈[0,1] of one chain to the nearest
#                  index in the other (round(p · (T-1))). Cheap, anchor-free,
#                  for whole-chain curves when no anchor is available.


@dataclass
class Alignment:
    """A correspondence between positions of a clean and a corrupt sequence.

    ``pairs[i] = (t_clean, t_corrupt)`` means "clean position ``t_clean``
    corresponds to corrupt position ``t_corrupt``". Patching/attribution should
    only ever compare residuals at corresponding positions.
    """
    pairs: list[tuple[int, int]] = field(default_factory=list)
    strategy: str = ""

    def clean_positions(self) -> list[int]:
        return [a for a, _ in self.pairs]

    def corrupt_positions(self) -> list[int]:
        return [b for _, b in self.pairs]


def anchored_alignment(
    T_clean: int,
    T_corrupt: int,
    anchor_clean: int,
    anchor_corrupt: int,
    window: int = 0,
) -> Alignment:
    """Align two sequences at a known anchor token, ± a symmetric window.

    With ``window=0`` this is the single anchored pair (the behaviour-onset
    boundary). With ``window=w`` it adds the ``w`` positions on each side,
    clamped to both sequences, giving 2w+1 corresponding pairs that stay locked
    in step. This is the positionally-honest replacement for the old
    "patch position i across chains" / "always patch the last token" rules.
    """
    if not (0 <= anchor_clean < T_clean):
        raise ValueError(f"anchor_clean {anchor_clean} out of range [0,{T_clean})")
    if not (0 <= anchor_corrupt < T_corrupt):
        raise ValueError(f"anchor_corrupt {anchor_corrupt} out of range [0,{T_corrupt})")
    pairs: list[tuple[int, int]] = []
    for d in range(-window, window + 1):
        tc, tk = anchor_clean + d, anchor_corrupt + d
        if 0 <= tc < T_clean and 0 <= tk < T_corrupt:
            pairs.append((tc, tk))
    return Alignment(pairs=pairs, strategy="anchored")


def proportional_alignment(
    T_clean: int,
    T_corrupt: int,
    positions_clean: Optional[Sequence[int]] = None,
) -> Alignment:
    """Map clean positions to nearest-fractional corrupt positions.

    For each clean index ``t`` (default: all of them), the corresponding corrupt
    index is ``round(t/(T_clean-1) · (T_corrupt-1))``. Anchor-free; use when no
    behaviour-onset anchor is available and a whole-chain curve is wanted.
    """
    if T_clean < 1 or T_corrupt < 1:
        raise ValueError("both sequences need length >= 1")
    if positions_clean is None:
        positions_clean = range(T_clean)
    denom = max(T_clean - 1, 1)
    pairs = []
    for t in positions_clean:
        if not (0 <= t < T_clean):
            raise ValueError(f"clean position {t} out of range")
        frac = t / denom
        tk = int(round(frac * (T_corrupt - 1)))
        pairs.append((int(t), tk))
    return Alignment(pairs=pairs, strategy="proportional")


def aligned_position_scores(
    acts_clean: np.ndarray,
    acts_corrupt: np.ndarray,
    behaviour: str,
    layer: int,
    metric: "BehaviourMetric | dict",
    alignment: Alignment,
) -> list[tuple[float, float]]:
    """Score corresponding positions of two chains under the validated metric.

    Returns ``[(score_clean_t, score_corrupt_t'), …]`` over the alignment's
    pairs. This is the positionally-honest comparison: a behaviour score at
    clean position ``t`` is only ever compared to the corrupt score at its
    aligned partner ``t'`` — never naively at the same index. ``acts_*`` are
    ``(T, d)`` residual blocks at ``layer``.
    """
    m = _select_metric(metric, behaviour, layer)
    out = []
    for tc, tk in alignment.pairs:
        out.append((m.score_np(acts_clean[tc]), m.score_np(acts_corrupt[tk])))
    return out


# ════════════════════════════════════════════════════════════════════════════
# 3. Thin model interface (so the math is stub-testable, no GPU)
# ════════════════════════════════════════════════════════════════════════════
#
# Both the attribution patcher and the brute-force reference talk to the model
# ONLY through `iter_residual_layers(model)` (the decoder blocks) and a metric
# applied to a chosen layer's residual at chosen positions. Anything exposing
# `.model.layers` as an indexable sequence of nn.Modules whose forward returns
# either a tensor or a tuple `(hidden, …)` satisfies the contract — the real HF
# Qwen2 model, and the tiny `StubLayeredModel` in the tests.


def iter_residual_layers(model):
    """Return the indexable sequence of residual-stream (decoder) blocks.

    Same access path as ``src/steered_inference.py`` (``model.model.layers``)
    and ``src/hooks.py``; falls back to the project's robust locator for
    non-Qwen architectures.
    """
    try:
        return model.model.layers
    except AttributeError:
        from src.model_adapters import locate_decoder_layers
        return locate_decoder_layers(model)


def _residual_of(output):
    """The residual tensor from a decoder block's output (tuple or bare)."""
    return output[0] if isinstance(output, tuple) else output


# ════════════════════════════════════════════════════════════════════════════
# 4. Attribution patching:  grad_clean · (act_corrupt − act_clean)
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class AttributionResult:
    """Per-(layer, position) attribution-patching estimate for one clean/corrupt
    pair, plus the per-layer reduction the layer sweep consumes."""
    behaviour: str
    layers: list[int]
    #: {layer: {position: attribution}} — signed effect estimate.
    per_position: dict[int, dict[int, float]]
    #: {layer: summed |attribution| over scored positions} — the layer score.
    per_layer: dict[int, float]
    #: positions (clean-chain indices) that were scored.
    positions: list[int]


def _gather_residuals(
    model,
    input_ids,
    layers: Sequence[int],
    *,
    requires_grad: bool,
):
    """Single forward pass; capture each layer's residual via a forward hook.

    Returns ``(captured, forward_output)`` where ``captured[L]`` is the residual
    tensor for layer ``L`` (batch dim kept). When ``requires_grad`` is True the
    captured tensors are retained in the graph (``retain_grad``) so a later
    ``backward`` populates ``.grad`` — that is how we read the metric's gradient
    w.r.t. each layer's clean activation in ONE backward pass.
    """
    blocks = iter_residual_layers(model)
    captured: dict[int, "object"] = {}
    handles = []

    def make_hook(L):
        def hook(_module, _inp, output):
            h = _residual_of(output)
            if requires_grad:
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
    return captured, out


def make_residual_metric_fn(metric: BehaviourMetric, read_layer: int,
                            read_position: int):
    """Build a differentiable ``metric_fn`` for :func:`attribution_patching`.

    The returned callable has signature ``metric_fn(captured) -> scalar`` where
    ``captured`` is the ``{layer: residual_tensor}`` dict that
    :func:`attribution_patching` captures on the clean forward pass. It scores
    ``metric.score_torch(captured[read_layer][0, read_position])`` — i.e. the
    behaviour's geometry read off the **read-out** layer/position. Because the
    scored residual is one of the captured tensors, the backward pass propagates
    a gradient into *every* swept layer that feeds it (all layers ≤ ``read_layer``
    in a causal transformer), with no second hook needed.

    The scored ``read_layer`` is the layer whose residual *defines* the
    behaviour score (typically the steering layer where ``metric``'s direction
    was built); attribution then asks which layers' activations most move that
    score. Effects from layers downstream of ``read_layer`` are zero to first
    order, so the sweep is normally ``layers ≤ read_layer``.
    """
    def metric_fn(captured: dict):
        h = captured[read_layer]              # (1, T, d), in graph
        return metric.score_torch(h[0, read_position])
    metric_fn.read_layer = read_layer
    metric_fn.read_position = read_position
    return metric_fn


def attribution_patching(
    model,
    clean_inputs,
    corrupt_inputs,
    metric_fn: Callable[[dict], "object"],
    layers: Sequence[int],
    *,
    alignment: Optional[Alignment] = None,
    behaviour: str = "",
) -> AttributionResult:
    """First-order (gradient) approximation to activation patching.

    For each layer ``L`` and aligned position ``t``::

        effect(L, t) ≈ grad_clean[L, t] · (act_corrupt[L, t'] − act_clean[L, t])

    where ``grad_clean[L, t] = ∂ metric_fn(clean) / ∂ act_clean[L, t]`` comes
    from a single backward pass on the clean run, ``act_clean`` / ``act_corrupt``
    come from one forward pass each, and ``(t, t')`` are corresponding positions
    under ``alignment`` (CF-10b). This estimates, in two passes total, what the
    brute-force patcher gets in one forward pass *per (layer, position)*.

    Parameters
    ----------
    model
        Anything satisfying the ``.model.layers`` contract (real model or stub).
    clean_inputs, corrupt_inputs
        ``input_ids`` tensors (shape ``(1, T)``). "Clean" = the run whose metric
        we differentiate (the donor *positive* chain); "corrupt" = the
        counterfactual whose activations we would swap in (the *negative* chain).
    metric_fn
        Callable mapping the captured ``{layer: residual}`` dict to a scalar
        torch tensor differentiable w.r.t. those residuals. Build it with
        :func:`make_residual_metric_fn` from a :class:`BehaviourMetric`.
    layers
        Residual-stream layer indices to estimate (``model.model.layers[L]``).
        Must include the metric's read-out layer so the metric is in the graph.
    alignment
        Position correspondence (CF-10b). If ``None``, every clean position is
        paired with the same corrupt index (identity) — only valid when the two
        sequences are already aligned (e.g. equal length, shared prefix); a
        warning is logged.
    behaviour
        Label, carried into the result for bookkeeping.

    Notes
    -----
    **First-order caveat.** This is a Taylor expansion of the metric about the
    clean activation. It is exact only in the limit of a small corrupt−clean
    gap and a metric that is locally linear in the residual. Our metric *is*
    linear in its own layer's residual, but the network between a swept layer
    and the read-out is not, so the estimate can mis-rank layers where the swap
    drives a large non-linear downstream change. Verify the top layers with
    :func:`brute_force_patch_effect`.
    """
    import torch

    if alignment is None:
        Tc = int(clean_inputs.shape[-1])
        alignment = Alignment(pairs=[(t, t) for t in range(Tc)], strategy="identity")
        logger.warning("attribution_patching: no alignment given — assuming "
                       "position-identity across chains (CF-10b); pass an "
                       "Alignment for non-aligned chains.")

    layers = list(layers)

    # --- clean: forward + backward to get residuals AND their gradients ---
    model.zero_grad(set_to_none=True)
    clean_acts, _ = _gather_residuals(model, clean_inputs, layers,
                                      requires_grad=True)
    metric_val = metric_fn(clean_acts)
    metric_val.backward()

    # --- corrupt: forward only, no grad needed ---
    with torch.no_grad():
        corrupt_acts, _ = _gather_residuals(model, corrupt_inputs, layers,
                                            requires_grad=False)

    per_position: dict[int, dict[int, float]] = {}
    per_layer: dict[int, float] = {}
    scored_positions = [tc for tc, _ in alignment.pairs]

    for L in layers:
        g = clean_acts[L].grad           # (1, T, d)
        a_clean = clean_acts[L].detach() # (1, T, d)
        a_corr = corrupt_acts[L].detach()
        if g is None:
            # Metric did not depend on this layer's activation (e.g. the layer
            # is downstream of the read-out). Effect is exactly zero.
            per_position[L] = {int(tc): 0.0 for tc, _ in alignment.pairs}
            per_layer[L] = 0.0
            continue
        pos_map: dict[int, float] = {}
        for tc, tk in alignment.pairs:
            delta = a_corr[0, tk] - a_clean[0, tc]      # (d,)
            eff = float(torch.dot(g[0, tc], delta).item())
            pos_map[int(tc)] = eff
        per_position[L] = pos_map
        per_layer[L] = float(sum(abs(v) for v in pos_map.values()))

    return AttributionResult(
        behaviour=behaviour,
        layers=layers,
        per_position=per_position,
        per_layer=per_layer,
        positions=[int(t) for t in scored_positions],
    )


# ════════════════════════════════════════════════════════════════════════════
# 5. Brute-force activation-patching reference (the ground truth tests check)
# ════════════════════════════════════════════════════════════════════════════


def _patched_forward_score(
    model,
    clean_inputs,
    donor_vec,
    layer: int,
    position: int,
    metric: BehaviourMetric,
    read_layer: int,
    read_position: int,
):
    """Run the clean inputs with layer/position residual replaced by ``donor_vec``;
    return the metric read off ``read_layer``/``read_position``.

    When ``layer == read_layer`` the read must see the *patched* value, so the
    patch hook itself captures the residual it returns (a second read hook on
    the same module would race the patch hook). Otherwise a separate read hook
    on the downstream read-out layer captures the propagated residual.
    """
    import torch
    blocks = iter_residual_layers(model)
    captured = {}
    same_layer = (layer == read_layer)

    def patch_hook(_m, _i, output):
        h = _residual_of(output)
        h = h.clone()
        h[0, position] = torch.as_tensor(donor_vec, dtype=h.dtype, device=h.device)
        if same_layer:
            captured["resid"] = h
        if isinstance(output, tuple):
            return (h,) + tuple(output[1:])
        return h

    def read_hook(_m, _i, output):
        captured["resid"] = _residual_of(output)

    handles = [blocks[layer].register_forward_hook(patch_hook)]
    if not same_layer:
        handles.append(blocks[read_layer].register_forward_hook(read_hook))
    try:
        with torch.no_grad():
            model(input_ids=clean_inputs)
    finally:
        for h in handles:
            h.remove()
    h = captured["resid"][0, read_position]
    return metric.score_np(h.detach().cpu().numpy())


def brute_force_patch_effect(
    model,
    clean_inputs,
    corrupt_inputs,
    metric: BehaviourMetric,
    layers: Sequence[int],
    *,
    alignment: Optional[Alignment] = None,
    read_layer: Optional[int] = None,
    read_position: Optional[int] = None,
) -> dict[int, dict[int, float]]:
    """Exact activation-patching effect for each (layer, aligned position).

    For each layer ``L`` and aligned pair ``(t, t')``: replace the clean
    residual at ``(L, t)`` with the corrupt residual at ``(L, t')``, re-run, and
    record ``metric(patched) − metric(clean)`` read at ``(read_layer,
    read_position)``. This is the quantity attribution patching approximates;
    the tests assert the two agree to first order on a stub. Returns
    ``{layer: {position: Δmetric}}``.

    ``read_layer``/``read_position`` MUST match the metric read-out used to
    build the attribution ``metric_fn`` (see :func:`make_residual_metric_fn`),
    or the two estimators are measuring different scalars. They default to
    ``max(layers)`` / last position.

    O(#layers × #positions) forward passes — that is exactly the cost
    attribution patching avoids, and why this is a *check on a few layers*, not
    the production estimator.
    """
    import torch
    Tc = int(clean_inputs.shape[-1])
    if read_position is None:
        read_position = Tc - 1
    if alignment is None:
        alignment = Alignment(pairs=[(t, t) for t in range(Tc)], strategy="identity")
    layers = list(layers)
    # Read-out defaults to the deepest swept layer so it is downstream of every
    # patched layer (an upstream patch can propagate to it; the reverse cannot).
    if read_layer is None:
        read_layer = max(layers)

    # corrupt residuals (forward only)
    with torch.no_grad():
        corrupt_acts, _ = _gather_residuals(model, corrupt_inputs, layers,
                                             requires_grad=False)

    # clean baseline score, read at (read_layer, read_position)
    read_layers = layers if read_layer in layers else layers + [read_layer]
    with torch.no_grad():
        clean_acts, _ = _gather_residuals(model, clean_inputs, read_layers,
                                          requires_grad=False)
    base = metric.score_np(
        clean_acts[read_layer][0, read_position].detach().cpu().numpy()
    )

    out: dict[int, dict[int, float]] = {}
    for L in layers:
        pos_map = {}
        for tc, tk in alignment.pairs:
            donor = corrupt_acts[L][0, tk]
            patched = _patched_forward_score(
                model, clean_inputs, donor, L, tc, metric,
                read_layer=read_layer, read_position=read_position,
            )
            pos_map[int(tc)] = float(patched - base)
        out[L] = pos_map
    return out


# ════════════════════════════════════════════════════════════════════════════
# 6. Aggregation across donor pairs → per-layer causal curve
# ════════════════════════════════════════════════════════════════════════════


def aggregate_attribution_curves(per_pair: list[dict[int, float]]) -> dict[int, dict]:
    """Average per-layer attribution scores across donor pairs.

    ``per_pair`` is a list of ``{layer: score}`` (one dict per donor pair,
    e.g. ``AttributionResult.per_layer``). Returns
    ``{layer: {"mean_effect", "sem_effect", "n"}}`` — the same shape the
    triangulation consumer reads from ``pilot_effect_curves.json``.
    """
    if not per_pair:
        return {}
    layers = sorted({L for d in per_pair for L in d})
    out: dict[int, dict] = {}
    for L in layers:
        vals = np.array([d[L] for d in per_pair if L in d], dtype=np.float64)
        finite = np.isfinite(vals)
        if not finite.any():
            out[L] = {"mean_effect": float("nan"), "sem_effect": float("nan"), "n": 0}
            continue
        v = vals[finite]
        out[L] = {
            "mean_effect": float(v.mean()),
            "sem_effect": float(v.std() / max(1.0, np.sqrt(v.size - 1))),
            "n": int(v.size),
        }
    return out
