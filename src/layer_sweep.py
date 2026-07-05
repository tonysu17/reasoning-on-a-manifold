"""
Forward-intervention per-layer STEERING-EFFECT sweep — the de-confounded
replacement for first-order attribution patching (CONFOUNDS_AND_REMEDIATION
CF-10, METHODOLOGY §5).

Why this module exists
----------------------
`src/attribution_patching.py` + `07c_attribution_patching.py` pick the
per-behaviour steering layer by **first-order attribution patching**: a single
backward pass gives ``grad_clean · (act_corrupt − act_clean)`` at every layer,
read against a behaviour metric at a FIXED late read-out layer (L27). Run on
R1-1.5B it came back **confounded**: all four behaviours ramp monotonically to
L26–27 (see ``results/patching/R1-1.5B/attribution_summary.md``). That is a
*read-out-proximity artefact* of a **linear** estimator — patching a layer
adjacent to the L27 read-out moves the L27 metric mechanically, and the
gradient cannot see that an early-layer perturbation **amplifies** as it flows
through the (non-linear) network. So the attribution curve has no interior peak
and gives no per-behaviour signal.

The fix here measures the **actual non-linear steering effect by forward-pass
intervention**, not a gradient:

  1.  **Per-layer direction** ``v_ℓ`` — the unit diff-of-means direction for the
      behaviour built AT layer ℓ from the already-extracted all-layer
      activations (ON = b, OFF = the other behaviours). Reuses
      ``src.steering.single_direction_vector``. (CPU; all 28 layers exist.)

  2.  **Output read-out — KL (primary), onset-logprob (secondary).** For a
      positive donor chain where b fired at onset token ``t*``, the PRIMARY
      read-out is ``KL(p_steered ‖ p_baseline)`` over the **full next-token
      distribution** at position ``t*−1`` (bigger = more redistribution = more
      causal effect). This captures probability mass moved onto *synonym* onset
      tokens that a single-token read-out misses. The SECONDARY diagnostic is the
      teacher-forced log-prob of the donor's **actual** onset token(s) (the
      CF-10a real-token measure). Both read at the OUTPUT (logits).

  3.  **Intervention with a per-layer norm-matched RANDOM null.** For each layer
      ℓ a forward pass adds ``−α·(rᵀh)·r`` at ℓ over positions ``≤ t*−1``
      (suppress; the projective form of ``src/steered_inference.py``), then
      recompute the read-out. In addition, ``R`` **norm-matched random**
      directions (deterministically seeded per ``(behaviour,layer,r)``) are run
      identically (same positions, same per-layer effective α), and their effect
      averaged into ``Score_random(ℓ)``. The reported, argmax-bearing quantity is
      the **de-confounded effect = Score_b(ℓ) − mean_r Score_random(ℓ)** — this
      subtracts off any *global* sensitivity of a layer to ANY perturbation (late
      layers near the logits can be sensitive to everything) and isolates the
      behaviour-specific causal effect.

  4.  **Per-layer α normalization.** The same nominal α is NOT the same
      intervention strength across depth — the residual-stream norm grows with
      depth. So the added delta is scaled to have norm a fixed fraction
      (``delta_frac``, default 0.1) of the **median ‖h‖ at that layer over the
      steered prefix positions**, collected in one baseline forward pass and
      applied IDENTICALLY to the behaviour direction and the random null (so the
      two are compared at equal physical strength).

  5.  **Bootstrap the argmax → a SHORTLIST.** Donors are resampled (default
      1000); the de-confounded argmax is recomputed each resample; the runner
      reports the bootstrap distribution and a **SHORTLIST** of 2–4 candidate
      layers (selected in ≥ a fraction of resamples, or within 1 SEM of the max),
      framed as a *pre-filter for Phase 7 to confirm*, NOT a single load-bearing
      argmax.

  6.  **Sweep ℓ** over a configurable range (default skips ~the first 20% of
      depth — embedding-correlated early layers, Venhoff "ignore early layers";
      ≈ L5 for 28 layers); optionally drops layers whose ``v_ℓ`` is highly
      cosine-similar to the input embedding matrix.

Why forward intervention de-confounds proximity
------------------------------------------------
Attribution reads the metric at a fixed late layer and linearises about the
clean activation, so its score is dominated by ``∂(L27 metric)/∂(act at ℓ)`` —
mechanically larger for ℓ near L27. This module reads at the OUTPUT and runs the
swap through the full non-linear stack, so the raw ``Score_b(ℓ)`` reflects how
much a real ℓ-perturbation changes the model's *output* probability of the
behaviour — which can peak in the interior (early perturbations grow). The read-
out is common across all swept ℓ (it never moves), so there is no fixed-read-out
proximity term. The **random-direction null** is the second line of defence: it
exposes and removes any *residual* proximity / global-sensitivity term that
survives the output read-out (e.g. a late layer whose logits move for ANY delta),
so the de-confounded effect is behaviour-specific.

Caveats (documented, surfaced in the runner summary)
----------------------------------------------------
  * The log-prob read-out is still **token-anchored**: it scores the
    probability of the donor's specific onset token(s), not the abstract
    behaviour. The KL read-out is the primary, behaviour-agnostic measure
    precisely to soften this, but KL is unsigned (magnitude of redistribution).
  * ``α``/``delta_frac`` and the suppression window are choices; defaults
    ``delta_frac=0.1`` and prefix ``≤ t*−1``. ``--amplify`` flips the sign.
  * Cost: ~``(1 + R)`` forwards per swept layer per donor (behaviour + R random),
    plus one baseline forward for the per-layer norms. Forward-only, no backward.

Stub-testability
----------------
Like ``src/attribution_patching.py``, everything talks to the model ONLY through
``model.model.layers`` (the decoder blocks) and the model's forward returning
logits (``output.logits`` or a bare ``(B, T, V)`` tensor). A tiny ``nn.Module``
stub that, BY CONSTRUCTION, makes a MIDDLE layer the most causal satisfies the
contract and is exercised in ``tests/test_layer_sweep.py`` with no GPU and no
real model. Nothing here loads a model or touches a GPU; the runner
(``07d_layer_steering_sweep.py``) wires it to R1-1.5B.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# 1. Thin model interface (shared contract with src/attribution_patching.py)
# ════════════════════════════════════════════════════════════════════════════


def iter_residual_layers(model):
    """Indexable sequence of residual-stream (decoder) blocks.

    Same access path as ``src/steered_inference.py`` (``model.model.layers``)
    and ``src/attribution_patching.py``; falls back to the project locator for
    non-Qwen architectures.
    """
    try:
        return model.model.layers
    except AttributeError:
        from src.model_adapters import locate_decoder_layers
        return locate_decoder_layers(model)


def _residual_of(output):
    """Residual tensor from a decoder block's output (tuple or bare tensor)."""
    return output[0] if isinstance(output, tuple) else output


def _logits_of(output):
    """Logits tensor from a model forward output.

    Handles HF ``CausalLMOutput`` (``.logits``), a plain dict (``["logits"]``),
    and a bare ``(B, T, V)`` tensor (the stub) — the same defensive shape
    handling the steering hook uses for decoder outputs.
    """
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    return output


def truncate_to_window(
    input_ids,
    onset_pos: int,
    *,
    context_window: int,
    span_tokens: int = 3,
):
    """Crop a donor's ``input_ids`` to a local window around the behaviour onset.

    Returns ``(ids_local, onset_local)`` where ``ids_local`` keeps only the token
    window ``[lo : onset_pos + span_tokens]`` with ``lo = max(0, onset_pos −
    context_window)``, and ``onset_local = onset_pos − lo`` is the onset remapped
    into the cropped tensor. Every position-derived index in the sweep
    (read-out at ``onset−1``, steering prefix ``≤ onset−1``, the onset span
    ``[onset, onset+span_tokens)``) is computed downstream from ``onset_local``
    and ``ids_local``, so a single consistent shift of the onset is sufficient —
    no other index needs separate remapping.

    Why this is sound — and where the approximation lives
    -----------------------------------------------------
    The sweep's read-out is teacher-forced at ``onset_pos`` (and the next few
    span tokens), read from the logits at ``onset_pos − 1``. By autoregressive
    causality those logits depend ONLY on tokens at positions ``< onset_pos``:
    tokens AT or AFTER the onset never enter the onset read-out. Dropping the
    tail beyond ``onset_pos + span_tokens`` is therefore **exact** for the
    read-out (the extra ``span_tokens`` are retained only so the secondary
    onset-span log-prob still has its targets in-tensor; the KL read-out at
    ``onset−1`` needs none of them).

    The PRE-onset crop (keeping only the last ``context_window`` tokens before
    the onset) is a **tractability approximation**: it assumes the steering
    decision and the model's onset prediction are governed by the recent local
    context, so distant earlier tokens contribute negligibly to both the
    baseline distribution and the steered delta at ``onset−1``. This is the
    standard local-context approximation used for steering long chains (cf. the
    pre-onset signal that forms in the immediate context, arXiv:2507.12638). It
    is EXACT whenever the full prefix already fits — i.e. ``onset_pos ≤
    context_window`` (then ``lo == 0`` and only the always-exact tail is
    dropped); ``context_window`` is configurable to trade fidelity for cost.

    Parameters
    ----------
    input_ids : LongTensor ``(1, T)``
        The donor's full token ids.
    onset_pos : int
        Behaviour-onset token index in ``input_ids`` (``1 ≤ onset_pos < T``).
    context_window : int
        Number of pre-onset tokens to keep (``W``). ``lo = max(0, onset_pos − W)``.
    span_tokens : int
        Onset-span length the secondary read-out averages over; the upper crop
        bound is ``onset_pos + span_tokens`` so those targets survive. Default 3.

    Returns
    -------
    (ids_local, onset_local) : (LongTensor ``(1, T')``, int)
        Cropped ids (a view/narrow of ``input_ids``) and the remapped onset, with
        ``T' = min(onset_pos + span_tokens, T) − lo`` and ``onset_local =
        onset_pos − lo``. If the window already covers the whole prefix
        (``onset_pos ≤ context_window``) ``lo == 0`` and ``onset_local ==
        onset_pos`` (no pre-onset truncation — exactness for short chains).
    """
    if context_window < 0:
        raise ValueError(f"context_window must be >= 0, got {context_window}")
    T = int(input_ids.shape[-1])
    lo = max(0, int(onset_pos) - int(context_window))
    hi = min(int(onset_pos) + int(span_tokens), T)      # exclusive upper bound
    ids_local = input_ids[:, lo:hi]
    onset_local = int(onset_pos) - lo
    return ids_local, onset_local


# ════════════════════════════════════════════════════════════════════════════
# 2. Per-layer steering directions v_ℓ (CPU; reuses src.steering)
# ════════════════════════════════════════════════════════════════════════════


def per_layer_directions(
    activations_dir,
    behaviour: str,
    layers: Sequence[int],
    other_behaviours: Sequence[str],
) -> dict[int, np.ndarray]:
    """Unit diff-of-means direction for ``behaviour`` at each layer in ``layers``.

    Built from the already-extracted all-layer activations
    ``{activations_dir}/{b}_layer{ℓ}.npy``: ON = ``behaviour``, OFF = the
    concatenation of every behaviour in ``other_behaviours``. This is the SAME
    construction ``src/steering.py`` uses for the canonical L27 vector, applied
    per layer so the intervention at layer ℓ steers along *that layer's* own
    behaviour direction. Returns ``{ℓ: (d,) unit vector}`` for every ℓ whose ON
    file exists; layers with missing files are skipped (logged).

    Pure / CPU: no model, no GPU.
    """
    from pathlib import Path

    from src.steering import single_direction_vector

    activations_dir = Path(activations_dir)
    out: dict[int, np.ndarray] = {}
    for L in layers:
        on_path = activations_dir / f"{behaviour}_layer{L}.npy"
        if not on_path.exists():
            logger.warning(f"  {behaviour}: missing {on_path}; skip layer {L}")
            continue
        on = np.load(on_path).astype(np.float64)
        off_parts = []
        for other in other_behaviours:
            p = activations_dir / f"{other}_layer{L}.npy"
            if p.exists():
                off_parts.append(np.load(p).astype(np.float64))
        if not off_parts:
            logger.warning(f"  {behaviour}: no OFF activations at L{L}; skip")
            continue
        off = np.concatenate(off_parts, axis=0)
        v = single_direction_vector(on, off)            # already unit-norm
        out[int(L)] = v.astype(np.float64)
    return out


# ════════════════════════════════════════════════════════════════════════════
# 3. Output read-out: full next-token logprobs → KL (primary) + onset logprob
# ════════════════════════════════════════════════════════════════════════════


def _next_token_logprobs(model, input_ids, *, steer: "SteerSpec | None" = None):
    """Full next-token log-prob matrix ``(T, V)`` at the OUTPUT (optionally steered).

    One forward pass (no backward); upcasts fp16 logits to fp32 for a stable
    softmax and keeps fp32/fp64 as-is so the float64 stubs are exact. ``[t]`` is
    ``log p(· | x_{≤t})`` — the distribution that predicts token ``t+1``. With
    ``steer`` set, the residual-stream hook is applied for the duration of the
    pass. Returns a torch tensor on the model's device.
    """
    import torch

    handle = None
    if steer is not None:
        handle = _register_steer_hook(model, steer)
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids)
        logits = _logits_of(out)                        # (1, T, V)
        lg = logits[0]
        if lg.dtype not in (torch.float32, torch.float64):
            lg = lg.float()
        return torch.log_softmax(lg, dim=-1)            # (T, V)
    finally:
        if handle is not None:
            handle.remove()


def onset_token_logprob(
    model,
    input_ids,
    onset_pos: int,
    *,
    span_tokens: int = 3,
    steer: "SteerSpec | None" = None,
) -> float:
    """Mean teacher-forced log-prob of the behaviour-span tokens at the OUTPUT.

    For the donor's own ``input_ids`` (shape ``(1, T)``), reads the model's
    next-token log-probabilities and returns the mean of
    ``log p(x_t | x_{<t})`` over ``t`` in ``[onset_pos, onset_pos+span_tokens)``
    (clamped to the sequence) — i.e. how much the model, given the real prefix,
    wants to emit the actual behaviour-onset token(s). The prediction of token
    ``t`` is read from logits at position ``t−1`` (standard causal-LM shift), so
    ``onset_pos`` must be ≥ 1.

    With ``steer`` set, a forward hook adds ``±α·(rᵀh)·r`` at ``steer.layer``
    over positions ``≤ steer.max_pos`` before the read-out is taken — the
    intervention whose effect on this output measure defines the steering score.

    This is the SECONDARY diagnostic read-out (token-anchored). The PRIMARY,
    behaviour-agnostic read-out is ``onset_kl`` below. Returns a Python float
    (log-prob, ≤ 0).
    """
    T = int(input_ids.shape[-1])
    if onset_pos < 1:
        raise ValueError(f"onset_pos must be >= 1 (need a prefix), got {onset_pos}")
    if onset_pos >= T:
        raise ValueError(f"onset_pos {onset_pos} out of range for T={T}")
    last = min(onset_pos + span_tokens, T)              # exclusive
    targets = list(range(onset_pos, last))              # token positions to score

    logprobs = _next_token_logprobs(model, input_ids, steer=steer)
    vals = []
    for t in targets:
        tok = int(input_ids[0, t].item())
        vals.append(float(logprobs[t - 1, tok].item()))           # predict t from t-1
    return float(np.mean(vals)) if vals else float("nan")


def onset_kl(
    model,
    input_ids,
    onset_pos: int,
    *,
    baseline_logprobs=None,
    steer: "SteerSpec | None" = None,
) -> float:
    """``KL(p_steered ‖ p_baseline)`` over the full next-token distribution at
    ``onset_pos−1`` — the PRIMARY, behaviour-agnostic steering-effect read-out.

    Reads the WHOLE next-token distribution at the position that predicts the
    onset (``onset_pos−1``, causal-LM shift), with and without the steer, and
    returns the forward KL ``Σ_v p_steered(v)·(log p_steered(v) − log p_base(v))``
    in nats. Unlike the single-token log-prob read-out, this captures probability
    mass redistributed onto *synonym* onset tokens (a behaviour realisable by
    several near-synonymous first tokens). Bigger KL = more causal effect.

    ``baseline_logprobs`` (the ``(T, V)`` matrix from an unsteered pass) may be
    passed to avoid recomputing the baseline for every layer/null. ``steer`` must
    be set (the steered distribution); if it is ``None`` the KL is 0 by
    definition. Returns a Python float ≥ 0.
    """
    import torch

    T = int(input_ids.shape[-1])
    if onset_pos < 1:
        raise ValueError(f"onset_pos must be >= 1 (need a prefix), got {onset_pos}")
    if onset_pos >= T:
        raise ValueError(f"onset_pos {onset_pos} out of range for T={T}")
    pos = onset_pos - 1                                  # predicts the onset token

    if baseline_logprobs is None:
        baseline_logprobs = _next_token_logprobs(model, input_ids, steer=None)
    if steer is None:
        return 0.0
    steered_logprobs = _next_token_logprobs(model, input_ids, steer=steer)

    lp_s = steered_logprobs[pos]                         # (V,)
    lp_b = baseline_logprobs[pos]                        # (V,)
    p_s = torch.exp(lp_s)
    kl = float((p_s * (lp_s - lp_b)).sum().item())
    # Numerical floor: tiny negative KL from rounding → clamp at 0.
    return max(0.0, kl)


# ════════════════════════════════════════════════════════════════════════════
# 4. The steering intervention hook (projective form from steered_inference)
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class SteerSpec:
    """One per-layer steering intervention to apply during a forward pass.

    Parameters
    ----------
    layer : int
        Decoder block index to hook (``model.model.layers[layer]``).
    direction : (d,) array
        Unit-norm direction ``v_ℓ`` for the behaviour at this layer (or a
        norm-matched RANDOM direction for the null).
    alpha : float
        Scale α. The hook adds ``sign·α·(rᵀh)·r`` (projective, Huang Eq. 3) —
        used ONLY when ``target_delta_norm`` is ``None``.
    sign : float
        ``-1.0`` to suppress (default), ``+1.0`` to amplify.
    max_pos : int | None
        Apply only at positions ``≤ max_pos`` (the prefix that predicts the
        onset). ``None`` = every position. For the sweep this is ``onset−1``.
    target_delta_norm : float | None
        Per-layer α-normalization. When set, the projective delta is rescaled so
        EACH steered position's delta has norm equal to this value (a fixed
        fraction ``delta_frac`` of the median ‖h‖ at this layer). This makes "the
        same α" the same *physical* intervention strength across depth, and is
        applied IDENTICALLY to the behaviour direction and the random null so the
        two are matched. ``None`` = use raw ``alpha`` (legacy).
    """

    layer: int
    direction: np.ndarray
    alpha: float = 1.0
    sign: float = -1.0
    max_pos: Optional[int] = None
    target_delta_norm: Optional[float] = None


def _register_steer_hook(model, spec: SteerSpec):
    """Register the residual-stream steering hook for ``spec``; return its handle.

    Mirrors ``src/steered_inference.SteeredModel._hook_fn`` (projective
    h − α·(rᵀh)·r in float32, restored to the layer dtype), restricted to
    positions ``≤ spec.max_pos`` so the prefix is steered while the scored onset
    token(s) are not directly overwritten — the change at the onset logit is
    purely the *downstream* (non-linear) consequence of the prefix steer.

    If ``spec.target_delta_norm`` is set, the projective delta is RESCALED so each
    steered position's delta has that target norm — the per-layer
    α-normalization. Because the rescale is direction-agnostic, the behaviour
    vector and the random null are injected at equal physical strength.
    """
    import torch

    blocks = iter_residual_layers(model)
    alpha = float(spec.alpha) * float(spec.sign)
    sign = float(spec.sign)
    max_pos = spec.max_pos
    target = spec.target_delta_norm
    dir_np = np.asarray(spec.direction)

    def hook(_module, _inp, output):
        is_tuple = isinstance(output, tuple)
        hidden = _residual_of(output)
        # Upcast for numerical stability (fp16 → fp32), but never DOWNcast: a
        # float64 stub stays float64 so the math tests are exact. Mirrors
        # src/steered_inference.SteeredModel but dtype-preserving.
        work_dtype = hidden.dtype if hidden.dtype in (torch.float32, torch.float64) else torch.float32
        h = hidden.to(work_dtype)                           # (B, T, d)
        r = torch.as_tensor(dir_np, dtype=work_dtype, device=h.device)
        proj = torch.einsum("bsd,d->bs", h, r).unsqueeze(-1)   # (B, T, 1)
        # Position mask (steer only the prefix ≤ max_pos).
        T = h.shape[1]
        if max_pos is not None:
            mask = torch.zeros(1, T, 1, dtype=work_dtype, device=h.device)
            hi = min(int(max_pos) + 1, T)
            if hi > 0:
                mask[:, :hi, :] = 1.0
        else:
            mask = torch.ones(1, T, 1, dtype=work_dtype, device=h.device)

        if target is not None:
            # Per-layer α-normalization: rescale the projective delta so EACH
            # steered position's delta has norm == target. r is unit, so the
            # projective delta norm at a position is |proj|; dividing it out and
            # multiplying by target gives a fixed-norm step ±target·r per
            # position. Direction-agnostic ⇒ behaviour and random null are
            # injected at identical physical strength.
            proj_mag = proj.abs().clamp_min(1e-12)          # (B, T, 1)
            unit_proj = proj / proj_mag                     # ±1 per position
            delta = sign * float(target) * unit_proj * r.view(1, 1, -1)  # (B, T, d)
        else:
            delta = alpha * proj * r.view(1, 1, -1)         # (B, T, d) — legacy

        delta = delta * mask
        h = h + delta
        h = h.to(hidden.dtype)
        return ((h,) + tuple(output[1:])) if is_tuple else h

    return blocks[spec.layer].register_forward_hook(hook)


# ════════════════════════════════════════════════════════════════════════════
# 4b. Per-layer residual norms (for α-normalization) + random-direction null
# ════════════════════════════════════════════════════════════════════════════


def collect_layer_norms(
    model,
    input_ids,
    layers: Sequence[int],
    *,
    max_pos: Optional[int] = None,
) -> dict[int, float]:
    """Median residual-stream norm ‖h‖ at each layer over the steered positions.

    One baseline forward pass with read-only hooks on each requested decoder
    block; for layer ℓ returns the median over positions ``≤ max_pos`` (the same
    prefix the steer touches) of the per-position L2 norm of that block's output
    residual. These medians scale the per-layer α so the injected delta is a
    fixed fraction (``delta_frac``) of ‖h‖ at depth ℓ — the residual norm grows
    with depth, so a constant nominal α is NOT a constant intervention. Returns
    ``{ℓ: median_norm}`` (CPU floats). One forward, no backward.
    """
    import torch

    blocks = iter_residual_layers(model)
    layers = [int(L) for L in layers]
    caps: dict[int, float] = {}
    handles = []

    def make_hook(L):
        def hook(_m, _i, output):
            hidden = _residual_of(output)
            h = hidden[0]                                   # (T, d)
            if h.dtype not in (torch.float32, torch.float64):
                h = h.float()
            T = h.shape[0]
            hi = T if max_pos is None else min(int(max_pos) + 1, T)
            if hi <= 0:
                caps[L] = float("nan")
                return
            norms = torch.linalg.vector_norm(h[:hi], dim=-1)   # (hi,)
            caps[L] = float(norms.median().item())
        return hook

    for L in layers:
        handles.append(blocks[L].register_forward_hook(make_hook(L)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for hd in handles:
            hd.remove()
    return caps


def random_unit_direction(d: int, behaviour: str, layer: int, r: int) -> np.ndarray:
    """Deterministic norm-1 random direction, seeded per ``(behaviour,layer,r)``.

    The per-layer norm-matched RANDOM null. The seed is a stable hash of
    ``(behaviour, layer, r)`` so the null is reproducible across runs and a given
    ``(behaviour, layer, r)`` always draws the SAME direction (auditable). Drawn
    isotropically (standard normal, normalized). Pure / CPU.
    """
    import hashlib

    key = f"{behaviour}|{layer}|{r}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**32)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(int(d))
    n = np.linalg.norm(v)
    if n < 1e-12:                                           # degenerate; retry once
        v = rng.standard_normal(int(d))
        n = np.linalg.norm(v)
    return (v / n).astype(np.float64)


def ambient_covariance(
    activations_dir,
    behaviours: Sequence[str],
    layer: int,
    *,
    max_rows: int = 20000,
    shrinkage: float = 1e-3,
    seed: int = 0,
) -> "np.ndarray | None":
    """Ambient residual-stream covariance Σ_ℓ at ``layer`` (CF-10 de-confounder).

    Estimated from a row-subsample of the concatenated all-behaviour activations
    ``{b}_layer{ℓ}.npy`` (the same files ``per_layer_directions`` reads), mmap'd so
    no whole file is materialised: up to ``max_rows`` rows are drawn proportionally
    across the behaviour files. Diagonal shrinkage ``shrinkage·tr(Σ)/d·I`` is added
    for positive-definiteness. Returns ``(d, d)`` float64, or ``None`` if no files.

    Why ambient (all-behaviour) and not behaviour-conditional: the confound the
    isotropic null misses is that a behaviour diff-of-means direction aligns with
    the residual stream's HIGH-VARIANCE directions, whose downstream leverage grows
    toward the logits (a spurious late-layer peak). A null drawn from Σ_ℓ shares
    that anisotropy, so subtracting it removes the generic high-variance-leverage
    term and isolates the behaviour-specific effect. Pure / CPU.
    """
    from pathlib import Path

    activations_dir = Path(activations_dir)
    arrays = []
    for b in behaviours:
        p = activations_dir / f"{b}_layer{layer}.npy"
        if p.exists():
            arrays.append(np.load(p, mmap_mode="r"))
    if not arrays:
        return None
    rng = np.random.default_rng(int(seed) + int(layer))
    total = int(sum(a.shape[0] for a in arrays))
    if total == 0:
        return None
    take = min(int(max_rows), total)
    rows = []
    for a in arrays:
        n = int(a.shape[0])
        if n == 0:
            continue
        k = max(1, int(round(take * n / total)))
        k = min(k, n)
        idx = np.sort(rng.choice(n, size=k, replace=False))
        rows.append(np.asarray(a[idx], dtype=np.float64))
    X = np.concatenate(rows, axis=0)
    X -= X.mean(axis=0, keepdims=True)
    d = int(X.shape[1])
    cov = (X.T @ X) / max(1, X.shape[0] - 1)
    tr = float(np.trace(cov))
    if tr > 0:
        cov = cov + (float(shrinkage) * tr / d) * np.eye(d)
    return cov


def covariance_matched_unit_directions(
    cov: np.ndarray, behaviour: str, layer: int, n_random: int,
) -> list[np.ndarray]:
    """``n_random`` unit directions ~ N(0, Σ), seeded per ``(behaviour,layer,r)``.

    The COVARIANCE-MATCHED null (the de-confounded replacement for the isotropic
    ``random_unit_direction``). Each draw is ``r = L z / ‖L z‖`` where ``Σ = L Lᵀ``
    (Cholesky, with escalating jitter then an eigen fallback for PD safety) and
    ``z ~ N(0, I)``. The seed scheme is byte-identical to ``random_unit_direction``
    so the two nulls are matched draw-for-draw and auditable. Because Σ carries the
    residual stream's anisotropy, these directions share the behaviour direction's
    alignment with high-variance directions — at the sweep's fixed injection norm
    their downstream leverage matches, so ``Score_b − mean_r Score_cov-random``
    subtracts the late-layer high-variance-leverage artefact the isotropic null
    leaves behind. Pure / CPU.
    """
    import hashlib

    cov = np.asarray(cov, dtype=np.float64)
    d = int(cov.shape[0])
    base = float(np.trace(cov)) / max(1, d)
    L = None
    jit = 0.0
    for _ in range(6):
        try:
            L = np.linalg.cholesky(cov + (jit * np.eye(d) if jit else 0.0))
            break
        except np.linalg.LinAlgError:
            jit = (base * 1e-6) if jit == 0.0 else (jit * 10.0)
    if L is None:                                            # eigen fallback
        w, V = np.linalg.eigh(cov)
        w = np.clip(w, 0.0, None)
        L = V * np.sqrt(w)
    out: list[np.ndarray] = []
    for r in range(int(n_random)):
        key = f"{behaviour}|{layer}|{r}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**32)
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(d)
        v = L @ z
        n = float(np.linalg.norm(v))
        if n < 1e-12:                                        # degenerate; retry once
            z = rng.standard_normal(d)
            v = L @ z
            n = float(np.linalg.norm(v))
        out.append((v / (n or 1.0)).astype(np.float64))
    return out


def embedding_cosine(directions: dict[int, np.ndarray], embed_matrix) -> dict[int, float]:
    """Max |cos| of each ``v_ℓ`` to any input-embedding row — early-layer flag.

    ``v_ℓ`` built at an embedding-correlated early layer aligns with token-
    embedding directions; a high value warns the layer is reading lexical/input
    structure rather than an abstract behaviour. Returns ``{ℓ: max_abs_cos}``.
    Embedding rows and ``v_ℓ`` are L2-normalized. ``embed_matrix`` is
    ``(vocab, d)`` (torch or numpy). Pure / CPU.
    """
    E = np.asarray(embed_matrix, dtype=np.float64)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    out: dict[int, float] = {}
    for L, v in directions.items():
        vv = np.asarray(v, dtype=np.float64)
        vv = vv / (np.linalg.norm(vv) + 1e-12)
        out[int(L)] = float(np.abs(En @ vv).max())
    return out


# ════════════════════════════════════════════════════════════════════════════
# 5. Per-donor layer sweep:  de-confounded effect = Score_b − mean_r Score_rand
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class SweepResult:
    """Per-layer steering-effect scores for ONE donor chain.

    ``score`` is the **de-confounded effect in the primary read-out** (the field
    the argmax/aggregation/bootstrap consume) — for the default ``readout``
    selection it is ``behaviour_effect − mean_r random_effect``. The richer
    per-read-out / per-component breakdown is carried alongside so the runner can
    emit the full ``{behaviour_effect, random_null, deconfounded_effect,
    kl_effect, logprob_effect}`` record.
    """

    behaviour: str
    layers: list[int]
    #: {layer: de-confounded effect in the PRIMARY read-out} — argmax-bearing
    score: dict[int, float]
    baseline_M: float
    #: {layer: M_steered}  (secondary logprob read-out, behaviour direction)
    steered_M: dict[int, float]
    onset_pos: int
    n_span_tokens: int
    #: which read-out drives ``score`` ("kl" or "logprob")
    readout: str = "logprob"
    #: {layer: behaviour-direction effect in the primary read-out}
    behaviour_effect: dict[int, float] = field(default_factory=dict)
    #: {layer: mean_r random-null effect in the primary read-out}
    random_null: dict[int, float] = field(default_factory=dict)
    #: {layer: de-confounded KL effect} (KL_b − mean_r KL_random)
    kl_effect: dict[int, float] = field(default_factory=dict)
    #: {layer: de-confounded logprob effect} (Δlogp_b − mean_r Δlogp_random)
    logprob_effect: dict[int, float] = field(default_factory=dict)
    #: number of random-null directions used per layer
    n_random: int = 0


def layer_steering_sweep_single(
    model,
    input_ids,
    onset_pos: int,
    directions: dict[int, np.ndarray],
    *,
    layers: Optional[Sequence[int]] = None,
    alpha: float = 1.0,
    amplify: bool = False,
    span_tokens: int = 3,
    readout: str = "logprob",
    n_random: int = 0,
    delta_frac: Optional[float] = None,
    behaviour: str = "",
    null_directions: Optional[dict[int, list[np.ndarray]]] = None,
) -> SweepResult:
    """Forward-intervention steering-effect curve over layers for one donor.

    For each layer ℓ in ``layers`` (default: the keys of ``directions``):
      * the baseline output distribution is read ONCE (no steer);
      * a forward pass with the behaviour direction ``−α·v_ℓ`` (or ``+α`` if
        ``amplify``) added over the prefix positions ``≤ onset_pos−1`` gives the
        steered read-outs (KL over the full next-token distribution at
        ``onset_pos−1``, and the secondary onset-token Δlog-prob);
      * if ``n_random > 0``, ``n_random`` **norm-matched random** directions
        (deterministically seeded per ``(behaviour, ℓ, r)``) are run identically
        and averaged into the per-layer random null;
      * ``score[ℓ]`` = the **de-confounded effect** in the ``readout`` read-out
        = behaviour effect − mean random-null effect.

    Per-layer α-normalization (``delta_frac``): when set, the injected delta is
    rescaled to a fixed fraction of the median ‖h‖ at that layer (collected in one
    baseline pass), applied IDENTICALLY to the behaviour vector and the random
    null. When ``None`` the raw ``alpha`` is used (legacy / exact-math stub path).

    ``argmax`` over the returned de-confounded ``score`` is the candidate causal
    steering layer; the read-out is the fixed OUTPUT (never a steered layer), so
    there is no fixed-read-out proximity term, and the random-null subtraction
    removes any residual global-sensitivity term — the curve can peak in the
    interior. ~``(1 + n_random)`` forwards per layer (no backward).
    """
    import torch

    if readout not in ("kl", "logprob"):
        raise ValueError(f"readout must be 'kl' or 'logprob', got {readout!r}")
    if layers is None:
        layers = sorted(directions.keys())
    layers = [int(L) for L in layers if int(L) in directions]
    sign = +1.0 if amplify else -1.0
    max_pos = onset_pos - 1                              # steer the predicting prefix
    n_random = max(0, int(n_random))

    T = int(input_ids.shape[-1])
    n_span = min(onset_pos + span_tokens, T) - onset_pos

    # Baseline read-outs (computed once; shared by every layer + null).
    base_logprobs = _next_token_logprobs(model, input_ids, steer=None)
    baseline = onset_token_logprob(
        model, input_ids, onset_pos, span_tokens=span_tokens, steer=None
    )

    # Per-layer α-normalization targets (one baseline pass over the prefix).
    target_norms: dict[int, Optional[float]] = {L: None for L in layers}
    if delta_frac is not None:
        med = collect_layer_norms(model, input_ids, layers, max_pos=max_pos)
        for L in layers:
            mn = med.get(L, float("nan"))
            target_norms[L] = (float(delta_frac) * mn) if np.isfinite(mn) else None

    d = int(np.asarray(directions[layers[0]]).shape[-1]) if layers else 0

    def _effects_for(direction, L):
        """(kl_effect, logp_effect, steered_M) for one direction at layer L."""
        spec = SteerSpec(
            layer=L, direction=direction, alpha=alpha, sign=sign,
            max_pos=max_pos, target_delta_norm=target_norms[L],
        )
        kl = onset_kl(
            model, input_ids, onset_pos,
            baseline_logprobs=base_logprobs, steer=spec,
        )
        m = onset_token_logprob(
            model, input_ids, onset_pos, span_tokens=span_tokens, steer=spec
        )
        return float(kl), float(baseline - m), float(m)

    behaviour_eff: dict[int, float] = {}
    random_null: dict[int, float] = {}
    kl_effect: dict[int, float] = {}
    logprob_effect: dict[int, float] = {}
    steered_M: dict[int, float] = {}
    score: dict[int, float] = {}

    for L in layers:
        kl_b, logp_b, m_b = _effects_for(directions[L], L)
        steered_M[L] = m_b
        prim_b = kl_b if readout == "kl" else logp_b

        # Per-layer fixed-norm random null (averaged over the random draws).
        # ``null_directions`` (e.g. covariance-matched, CF-10) overrides the
        # default isotropic ``random_unit_direction``; the injection itself is
        # direction-agnostic at fixed norm, so only the draw distribution changes.
        if null_directions is not None:
            draws = list(null_directions.get(L, []))
        elif n_random > 0 and d > 0:
            draws = [random_unit_direction(d, behaviour, L, r) for r in range(n_random)]
        else:
            draws = []
        if draws:
            kl_rs, logp_rs, prim_rs = [], [], []
            for rv in draws:
                kl_r, logp_r, _ = _effects_for(rv, L)
                kl_rs.append(kl_r)
                logp_rs.append(logp_r)
                prim_rs.append(kl_r if readout == "kl" else logp_r)
            kl_null = float(np.mean(kl_rs))
            logp_null = float(np.mean(logp_rs))
            prim_null = float(np.mean(prim_rs))
        else:
            kl_null = logp_null = prim_null = 0.0

        behaviour_eff[L] = prim_b
        random_null[L] = prim_null
        kl_effect[L] = kl_b - kl_null
        logprob_effect[L] = logp_b - logp_null
        score[L] = prim_b - prim_null

    return SweepResult(
        behaviour=behaviour,
        layers=layers,
        score=score,
        baseline_M=baseline,
        steered_M=steered_M,
        onset_pos=int(onset_pos),
        n_span_tokens=int(n_span),
        readout=readout,
        behaviour_effect=behaviour_eff,
        random_null=random_null,
        kl_effect=kl_effect,
        logprob_effect=logprob_effect,
        n_random=n_random,
    )


# ════════════════════════════════════════════════════════════════════════════
# 6. Aggregation across donors → per-layer mean / sem curve
# ════════════════════════════════════════════════════════════════════════════


def _sem(v: np.ndarray) -> float:
    """Standard error of the mean: ``std(ddof=1)/sqrt(n)`` (0 for n<2)."""
    n = int(v.size)
    if n < 2:
        return 0.0
    return float(np.std(v, ddof=1) / np.sqrt(n))


def aggregate_sweep_curves(per_donor: list[dict[int, float]]) -> dict[int, dict]:
    """Average per-layer steering-effect scores across donors.

    ``per_donor`` is a list of ``{layer: score}`` (one dict per donor, e.g.
    ``SweepResult.score`` — the de-confounded effect). Returns ``{layer:
    {"mean_effect", "sem_effect", "n"}}`` — the SAME shape
    ``compute_layer_triangulation.load_phase7b_curves`` reads from
    ``pilot_effect_curves.json``. SEM is the standard ``std(ddof=1)/sqrt(n)``.
    """
    if not per_donor:
        return {}
    layers = sorted({L for d in per_donor for L in d})
    out: dict[int, dict] = {}
    for L in layers:
        vals = np.array([d[L] for d in per_donor if L in d], dtype=np.float64)
        finite = np.isfinite(vals)
        if not finite.any():
            out[L] = {"mean_effect": float("nan"), "sem_effect": float("nan"), "n": 0}
            continue
        v = vals[finite]
        out[L] = {
            "mean_effect": float(v.mean()),
            "sem_effect": _sem(v),
            "n": int(v.size),
        }
    return out


def aggregate_full_curves(per_donor: list["SweepResult"]) -> dict[int, dict]:
    """Per-layer aggregate of every effect component across donor ``SweepResult``s.

    Returns ``{layer: {behaviour_effect, random_null, deconfounded_effect,
    kl_effect, logprob_effect, mean_effect, sem, n}}`` — the per-layer record the
    runner writes to ``steering_effect_curves.json``. ``mean_effect`` mirrors
    ``deconfounded_effect`` (the primary read-out's de-confounded mean), so the
    file stays drop-in for the triangulation consumer. ``sem`` is the SEM of the
    de-confounded effect (``std(ddof=1)/sqrt(n)``).
    """
    if not per_donor:
        return {}
    layers = sorted({L for res in per_donor for L in res.layers})
    out: dict[int, dict] = {}

    def _mean(getter, L):
        vals = np.array(
            [getter(res)[L] for res in per_donor if L in getter(res)],
            dtype=np.float64,
        )
        vals = vals[np.isfinite(vals)]
        return (float(vals.mean()) if vals.size else float("nan")), vals

    for L in layers:
        beh_m, _ = _mean(lambda r: r.behaviour_effect, L)
        null_m, _ = _mean(lambda r: r.random_null, L)
        kl_m, _ = _mean(lambda r: r.kl_effect, L)
        logp_m, _ = _mean(lambda r: r.logprob_effect, L)
        dec_m, dec_vals = _mean(lambda r: r.score, L)       # de-confounded primary
        out[L] = {
            "behaviour_effect": beh_m,
            "random_null": null_m,
            "deconfounded_effect": dec_m,
            "kl_effect": kl_m,
            "logprob_effect": logp_m,
            "mean_effect": dec_m,                           # alias for consumers
            "sem": _sem(dec_vals),
            "sem_effect": _sem(dec_vals),                   # alias
            "n": int(dec_vals.size),
        }
    return out


def argmax_layer(curve: dict[int, dict]) -> "int | None":
    """Layer with the largest finite ``mean_effect`` in an aggregated curve."""
    best_L, best_v = None, -np.inf
    for L, d in curve.items():
        m = d.get("mean_effect", float("nan"))
        if np.isfinite(m) and m > best_v:
            best_L, best_v = int(L), float(m)
    return best_L


# ════════════════════════════════════════════════════════════════════════════
# 7. Bootstrap the argmax over donors → a SHORTLIST (pre-filter for Phase 7)
# ════════════════════════════════════════════════════════════════════════════


def bootstrap_shortlist(
    per_donor: list[dict[int, float]],
    *,
    n_boot: int = 1000,
    select_frac: float = 0.15,
    seed: int = 0,
) -> dict:
    """Bootstrap the de-confounded argmax over donors → a candidate SHORTLIST.

    ``per_donor`` is a list of ``{layer: deconfounded_score}`` (one per donor,
    i.e. ``SweepResult.score``). Resamples donors WITH REPLACEMENT ``n_boot``
    times; on each resample averages the per-layer score and records the argmax.
    Returns a dict with:

      * ``argmax``        — the point-estimate argmax on the full sample;
      * ``boot_freq``     — ``{layer: fraction of resamples it won the argmax}``;
      * ``shortlist``     — layers selected in ≥ ``select_frac`` of resamples OR
                            within 1 SEM of the max mean (the union), sorted by
                            descending mean — the 2–4 candidate layers to hand to
                            Phase 7 to confirm (NOT a single load-bearing argmax);
      * ``mean``/``sem``  — per-layer mean and SEM (``std(ddof=1)/sqrt(n)``) of
                            the de-confounded score.

    Framing: this is a PRE-FILTER. The bootstrap exposes how unstable the single
    argmax is; the shortlist is the robust object Phase 7 should test.
    """
    agg = aggregate_sweep_curves(per_donor)
    finite_layers = [L for L, d in agg.items() if np.isfinite(d["mean_effect"])]
    point_argmax = argmax_layer(agg)
    if not per_donor or not finite_layers:
        return {
            "argmax": point_argmax, "boot_freq": {}, "shortlist": [],
            "mean": {L: agg[L]["mean_effect"] for L in agg},
            "sem": {L: agg[L]["sem_effect"] for L in agg},
            "n_boot": int(n_boot), "n_donors": len(per_donor),
        }

    layers = sorted(finite_layers)
    # Matrix of per-donor scores (NaN where a donor lacks a layer) for fast resample.
    n_donors = len(per_donor)
    mat = np.full((n_donors, len(layers)), np.nan, dtype=np.float64)
    for i, dct in enumerate(per_donor):
        for j, L in enumerate(layers):
            if L in dct:
                mat[i, j] = dct[L]

    rng = np.random.default_rng(seed)
    win_counts = np.zeros(len(layers), dtype=np.int64)
    n_boot = max(1, int(n_boot))
    for _ in range(n_boot):
        idx = rng.integers(0, n_donors, size=n_donors)
        sample = mat[idx]                                   # (n_donors, n_layers)
        with np.errstate(invalid="ignore"):
            means = np.nanmean(sample, axis=0)              # (n_layers,)
        if np.all(np.isnan(means)):
            continue
        win_counts[int(np.nanargmax(means))] += 1

    boot_freq = {int(L): float(win_counts[j] / n_boot) for j, L in enumerate(layers)}

    # SHORTLIST: union of (won ≥ select_frac of resamples) and (within 1 SEM of max).
    means = {L: agg[L]["mean_effect"] for L in layers}
    sems = {L: agg[L]["sem_effect"] for L in layers}
    max_L = max(means, key=means.get)
    thresh = means[max_L] - sems[max_L]                     # within 1 SEM of the max
    shortlist = {
        L for L in layers
        if boot_freq[L] >= select_frac or means[L] >= thresh
    }
    shortlist = sorted(shortlist, key=lambda L: means[L], reverse=True)

    return {
        "argmax": point_argmax,
        "boot_freq": boot_freq,
        "shortlist": [int(L) for L in shortlist],
        "mean": {int(L): float(means[L]) for L in layers},
        "sem": {int(L): float(sems[L]) for L in layers},
        "n_boot": int(n_boot),
        "n_donors": int(n_donors),
    }
