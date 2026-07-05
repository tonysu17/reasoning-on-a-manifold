"""Synthetic / stub tests for src/layer_sweep.py — the de-confounded,
forward-intervention per-layer STEERING-EFFECT method that replaces first-order
attribution patching (CONFOUNDS_AND_REMEDIATION CF-10, METHODOLOGY §5).

No GPU, no real model. Tiny ``nn.Module`` stubs exposing ``.model.layers`` (same
access path as src/steered_inference.py) and a ``forward`` returning logits
stand in for R1-1.5B. All stubs run in float64 (``.double()``) so these are
exact math-correctness tests, not float16-rounding tests.

THE CRITICAL TEST (``test_interior_peak_*``)
--------------------------------------------
A stub is constructed so that, BY CONSTRUCTION, intervening at a MIDDLE layer
produces the largest change in the OUTPUT read-out while the LAST layer produces
~zero change. The method's ``argmax`` must recover the MIDDLE layer — it must NOT
monotonically ramp to the last layer. This is exactly the failure mode the
confounded attribution-patching curve showed (a monotone ramp to L26–27, a
read-out-proximity artefact of a *linear, fixed-late-read-out* estimator). The
forward-intervention estimator reads at the OUTPUT and runs the swap through the
full non-linear stack, so it captures the amplified middle-layer effect and
recovers the interior peak. A proximity-confounded proxy (read the change at a
fixed late layer) ranks the last layer top on the same stub and FAILS — that
contrast is asserted in ``test_old_proximity_logic_would_fail_this_stub``.

Why a pure-gain stub CANNOT show an interior peak (and what does)
----------------------------------------------------------------
Projective steering ``h − α·(rᵀh)·r`` is *scale-covariant*: in any network whose
per-layer map is linear in the steered direction, the delivered output change is
INDEPENDENT of the injection layer (the upstream gains that shrink ``rᵀh`` cancel
the downstream gains that re-grow the delta — the product telescopes). So a
diagonal/gain stub gives a flat curve, not a peak. An interior peak needs two
genuinely non-telescoping ingredients, both real transformer phenomena:

  * **Regeneration**: the steered coordinate is rebuilt downstream from a source
    coordinate (``e0 ← decay·e0 + leak·e1``). Suppressing ``e0`` early lets later
    blocks refill it, so an *earlier* injection has a *smaller* surviving effect.
  * **Latched read-out**: the read-out path snapshots the steered coordinate at
    ONE layer (``e_read += amp·e0`` at the planted layer) and then ignores it.
    Suppressing ``e0`` *after* that layer cannot move the read-out at all.

Together: the effect ramps up over layers ``0..planted`` (less refill the closer
to the latch) and drops to ~0 after ``planted`` (the latch already fired) — a
unique interior maximum at ``planted``. This is the de-confounded curve shape;
the first-order gradient at a fixed late read-out sees none of it.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from src.layer_sweep import (
    SteerSpec,
    SweepResult,
    onset_token_logprob,
    onset_kl,
    layer_steering_sweep_single,
    truncate_to_window,
    aggregate_sweep_curves,
    aggregate_full_curves,
    argmax_layer,
    bootstrap_shortlist,
    collect_layer_norms,
    random_unit_direction,
    embedding_cosine,
    iter_residual_layers,
    per_layer_directions,
    _register_steer_hook,
    _next_token_logprobs,
)


# ════════════════════════════════════════════════════════════════════════════
# Stub A — interior-peak model: regeneration + latched read-out (see module doc)
# ════════════════════════════════════════════════════════════════════════════
#
# Coordinates (per position, processed independently across positions):
#   e0  — the STEERED signal.  Block update:  e0 ← DECAY·e0 + LEAK·e1
#         (regenerates from the constant source e1 → earlier suppression refills)
#   e1  — constant source (untouched).
#   e2  — the READ-OUT accumulator.  At the latch layer only:  e2 ← e2 + AMP·e0
#         (snapshots e0 once; the scored token's unembed row reads e2).
# Steering direction r = e0 (a fixed coordinate), so (rᵀh)·r touches e0 only.


class _RegenLatchBlock(nn.Module):
    """One block of the regeneration+latch stub (float64).

    ``is_latch`` marks the single layer whose read-out path snapshots ``e0`` into
    the read-out coordinate ``e2`` — the planted causal layer. Returns a tuple
    ``(h, None)`` like an HF Qwen2 decoder block.
    """

    def __init__(self, decay, leak, amp, is_latch):
        super().__init__()
        self.decay = float(decay)
        self.leak = float(leak)
        self.amp = float(amp)
        self.is_latch = bool(is_latch)

    def forward(self, hidden, **_):
        h = hidden.clone()
        e0 = self.decay * hidden[..., 0] + self.leak * hidden[..., 1]   # regenerate
        h[..., 0] = e0
        if self.is_latch:
            h[..., 2] = hidden[..., 2] + self.amp * e0                  # latch read-out
        return (h, None)


class _RegenLatchInner(nn.Module):
    def __init__(self, d, vocab, n_layers, latch_layer, decay, leak, amp,
                 read_tok, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.d = d
        self.read_tok = read_tok
        # Deterministic embedding: every token starts with e0=e1=1, e2=0, rest 0.
        emb = torch.zeros(vocab, d, dtype=torch.float64)
        emb[:, 0] = 1.0
        emb[:, 1] = 1.0
        self.embed = nn.Embedding(vocab, d)
        with torch.no_grad():
            self.embed.weight.copy_(emb)
        self.layers = nn.ModuleList([
            _RegenLatchBlock(decay, leak, amp, is_latch=(i == latch_layer))
            for i in range(n_layers)
        ])
        # Unembed: the scored token reads e2; other rows are small random noise so
        # log_softmax is well-defined and the scored logit is exactly e2.
        U = torch.randn(vocab, d, generator=g, dtype=torch.float64) * 0.1
        U[read_tok] = 0.0
        U[read_tok, 2] = 1.0
        self.unembed = nn.Linear(d, vocab, bias=False)
        with torch.no_grad():
            self.unembed.weight.copy_(U)

    def forward(self, input_ids):
        h = self.embed(input_ids)            # (B, T, d)
        for blk in self.layers:
            h = blk(h)[0]
        return self.unembed(h)               # (B, T, vocab) logits


class RegenLatchStub(nn.Module):
    """Interior-peak stub. ``argmax`` of the steering-effect curve == ``peak_layer``.

    Hook semantics matter here. The sweep registers a forward hook on
    ``model.model.layers[ℓ]``, so "steering at layer ℓ" perturbs that block's
    OUTPUT = the INPUT to block ℓ+1 (exactly ``src/steered_inference.SteeredModel``).
    The read-out latch reads ``e0`` *inside* a block from that block's input, so a
    latch placed at block ``peak_layer+1`` is driven by the output of block
    ``peak_layer`` — i.e. steering layer ``peak_layer`` has the maximal effect.
    The stub therefore puts the latch at ``peak_layer + 1`` internally and
    exposes ``peak_layer`` as the planted causal layer the method must recover.

    The scored "onset" token is ``read_tok`` (place it at the onset position in
    the donor ids); the steering direction is ``e0`` (one-hot on coordinate 0).
    The curve ramps up to ``peak_layer`` (less downstream refill the closer to
    the latch) and is ~0 afterwards (the latch already fired).
    """

    def __init__(self, d=8, vocab=16, n_layers=8, peak_layer=4,
                 decay=0.5, leak=0.5, amp=6.0, read_tok=5, seed=1):
        super().__init__()
        latch_layer = peak_layer + 1          # hook-on-output ⇒ peak is one before
        if not (0 <= latch_layer < n_layers):
            raise ValueError(
                f"peak_layer={peak_layer} needs a latch at {latch_layer} < n_layers={n_layers}"
            )
        self.model = _RegenLatchInner(d, vocab, n_layers, latch_layer,
                                      decay, leak, amp, read_tok, seed)
        self.d = d
        self.vocab = vocab
        self.n_layers = n_layers
        self.peak_layer = peak_layer
        self.latch_layer = latch_layer
        self.read_tok = read_tok

    def forward(self, input_ids=None, **_):
        return self.model(input_ids)

    @property
    def e0_direction(self):
        """Unit steering direction = coordinate 0 (the steered signal)."""
        v = np.zeros(self.d)
        v[0] = 1.0
        return v


# ════════════════════════════════════════════════════════════════════════════
# Stub B — plain gain stub (linear, scale-covariant) for plumbing / no-op tests
# ════════════════════════════════════════════════════════════════════════════


class _GainBlock(nn.Module):
    """Residual update h ← gain · h (diagonal, float64). HF-style tuple output."""

    def __init__(self, gain):
        super().__init__()
        self.gain = float(gain)

    def forward(self, hidden, **_):
        return (hidden * self.gain, None)


class _GainInner(nn.Module):
    def __init__(self, d, vocab, gains, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(vocab, d)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(vocab, d, generator=g, dtype=torch.float64) * 0.3)
        self.layers = nn.ModuleList([_GainBlock(gv) for gv in gains])
        self.unembed = nn.Linear(d, vocab, bias=False)
        with torch.no_grad():
            self.unembed.weight.copy_(torch.randn(vocab, d, generator=g, dtype=torch.float64) * 0.3)

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for blk in self.layers:
            h = blk(h)[0]
        return self.unembed(h)


class GainStubModel(nn.Module):
    """Linear gain stub exposing ``.model.layers``; float64."""

    def __init__(self, d=8, vocab=16, gains=(1, 1, 1, 1), seed=0):
        super().__init__()
        self.model = _GainInner(d, vocab, gains, seed)
        self.d = d
        self.vocab = vocab
        self.n_layers = len(gains)

    def forward(self, input_ids=None, **_):
        return self.model(input_ids)


# ════════════════════════════════════════════════════════════════════════════
# Stub C — GLOBALLY-sensitive (isotropic) latch: responds to ANY direction equally
# ════════════════════════════════════════════════════════════════════════════
#
# Purpose: the per-layer norm-matched RANDOM-DIRECTION null must subtract to ~0 on
# a layer that is sensitive to the *magnitude* of any perturbation, not to the
# behaviour direction specifically (e.g. a late layer near the logits). This stub
# plants exactly such a layer.
#
# Mechanism (no cross-position mixing; processed per position). With per-layer
# α-normalization (``delta_frac``), the projective hook delivers an EXACT
# fixed-norm step ``±target·r`` at EACH steered position for ANY unit direction r
# (the per-position projection magnitude is divided out). So at the probe column:
#   * The probe column is onset−1, whose token embeds to a FIXED constant vector
#     ``c`` (preserved by the identity blocks). Its baseline residual is exactly
#     ``c``; after steering it is ``c ± target·r``.
#   * The latch block (at sensitive_layer+1, hook-on-output convention) overwrites
#     the read coordinate with ``amp·‖h − c‖`` — the distance from the KNOWN
#     baseline ``c``. At the probe column this is ``amp·‖±target·r‖ = amp·target``
#     for EVERY direction (the cross term cancels because we subtract the exact
#     baseline). Baseline (no steer): ``‖c − c‖ = 0`` → 0. Hence behaviour_effect
#     == random_effect at this layer and the de-confounded effect is exactly 0.


class _IsoBlock(nn.Module):
    """Identity, except the latch writes read_coord = amp·‖h − c‖ (float64).

    Subtracting the known baseline ``c`` before the norm makes the read-out depend
    ONLY on the perturbation magnitude (radially symmetric about ``c``), so a
    fixed-norm step in ANY direction moves it identically.
    """

    def __init__(self, read_coord, amp, is_latch, baseline_c):
        super().__init__()
        self.read_coord = int(read_coord)
        self.amp = float(amp)
        self.is_latch = bool(is_latch)
        self.register_buffer("baseline_c", baseline_c)

    def forward(self, hidden, **_):
        if not self.is_latch:
            return (hidden, None)
        h = hidden.clone()
        c = self.baseline_c.to(hidden.dtype)
        dist = torch.linalg.vector_norm(hidden - c, dim=-1)     # ‖h − c‖ per position
        h[..., self.read_coord] = self.amp * dist               # overwrite (iso)
        return (h, None)


class _IsoInner(nn.Module):
    def __init__(self, d, vocab, n_layers, latch_layer, amp, read_tok, probe_tok,
                 read_coord, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.d = d
        # probe_tok → fixed constant c (the probe baseline); other tokens → noise.
        c = torch.ones(d, dtype=torch.float64) * 0.5
        emb = torch.randn(vocab, d, generator=g, dtype=torch.float64) * 0.2
        emb[probe_tok] = c
        self.embed = nn.Embedding(vocab, d)
        with torch.no_grad():
            self.embed.weight.copy_(emb)
        self.layers = nn.ModuleList([
            _IsoBlock(read_coord, amp, is_latch=(i == latch_layer), baseline_c=c.clone())
            for i in range(n_layers)
        ])
        # ONLY read_tok's logit reads the (latch-overwritten) read coordinate; every
        # other logit is identically 0. So the full next-token distribution depends
        # ONLY on the direction-blind read logit (amp·‖h−c‖) → the KL read-out is
        # EXACTLY equal for the behaviour vector and every random direction, making
        # the de-confounded effect exactly 0 (not merely small). Any noise in the
        # other rows would let the off-read logits drift with the steered residual
        # and break the exact cancellation.
        U = torch.zeros(vocab, d, dtype=torch.float64)
        U[read_tok, read_coord] = 1.0
        self.unembed = nn.Linear(d, vocab, bias=False)
        with torch.no_grad():
            self.unembed.weight.copy_(U)

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for blk in self.layers:
            h = blk(h)[0]
        return self.unembed(h)


class IsoNormStub(nn.Module):
    """Globally-sensitive stub: steering ``sensitive_layer`` with ANY norm-matched
    direction moves the read logit by the SAME amount → de-confounded effect == 0.

    Place ``probe_tok`` at the onset−1 column (the steered probe) and ``read_tok``
    at the onset column. Requires ``delta_frac`` (per-layer α-normalization) and a
    single steered span so the delivered delta has a fixed norm for every
    direction.
    """

    def __init__(self, d=8, vocab=16, n_layers=8, sensitive_layer=4, amp=5.0,
                 read_tok=5, probe_tok=6, seed=1):
        super().__init__()
        read_coord = d - 1
        latch_layer = sensitive_layer + 1            # hook-on-output ⇒ one before
        if not (0 <= latch_layer < n_layers):
            raise ValueError(
                f"sensitive_layer={sensitive_layer} needs a latch at {latch_layer}"
                f" < n_layers={n_layers}")
        self.model = _IsoInner(d, vocab, n_layers, latch_layer, amp, read_tok,
                               probe_tok, read_coord, seed)
        self.d = d
        self.vocab = vocab
        self.n_layers = n_layers
        self.sensitive_layer = sensitive_layer
        self.read_tok = read_tok
        self.probe_tok = probe_tok

    def forward(self, input_ids=None, **_):
        return self.model(input_ids)


def _double(model):
    """Cast a stub to float64 and eval mode."""
    return model.double().eval()


def _ids(T, vocab, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (1, T), generator=g)


def _unit(d, seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def _ids_with_onset(read_tok, onset, T, vocab, seed):
    """Random ids of length T with ``read_tok`` planted at position ``onset``."""
    ids = _ids(T, vocab, seed)
    ids[0, onset] = read_tok
    return ids


# ════════════════════════════════════════════════════════════════════════════
# 1. THE CRITICAL TEST — interior peak, not a monotone ramp to the last layer
# ════════════════════════════════════════════════════════════════════════════


def test_interior_peak_argmax_recovers_middle_layer():
    """THE WHOLE POINT. On a stub where a MIDDLE layer is most causal for the
    OUTPUT read-out (regeneration + latch), ``argmax`` of the forward-intervention
    steering-effect curve == that middle layer, and is NOT the last layer."""
    d, vocab, n_layers, planted = 8, 16, 8, 4
    read_tok = 5
    model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=n_layers,
                                   peak_layer=planted, read_tok=read_tok, seed=1))
    layers = list(range(n_layers))
    v = model.e0_direction
    directions = {L: v for L in layers}
    onset = 4
    ids = _ids_with_onset(read_tok, onset, T=6, vocab=vocab, seed=11)

    res = layer_steering_sweep_single(
        model, ids, onset_pos=onset, directions=directions,
        layers=layers, alpha=1.0, span_tokens=1,
    )
    curve = aggregate_sweep_curves([res.score])
    am = argmax_layer(curve)

    assert am == planted, (
        f"expected interior peak at planted layer {planted}, got {am}; "
        f"scores={ {L: round(res.score[L], 5) for L in layers} }"
    )
    # Explicitly NOT the last layer (the proximity-confounded answer).
    assert am != n_layers - 1
    # The planted layer's effect dwarfs the last layer's (latch already fired).
    assert abs(res.score[planted]) > 10 * abs(res.score[n_layers - 1])


def test_interior_peak_planted_at_several_layers():
    """The peak tracks wherever the causal layer actually is (3, 4, or 5) — the
    method is not biased toward any fixed interior layer."""
    d, vocab, n_layers, read_tok = 8, 16, 8, 5
    for planted in (3, 4, 5):
        model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=n_layers,
                                       peak_layer=planted, read_tok=read_tok, seed=2))
        layers = list(range(n_layers))
        v = model.e0_direction
        directions = {L: v for L in layers}
        onset = 4
        ids = _ids_with_onset(read_tok, onset, T=6, vocab=vocab, seed=12 + planted)
        res = layer_steering_sweep_single(
            model, ids, onset_pos=onset, directions=directions,
            layers=layers, alpha=1.0, span_tokens=1,
        )
        am = argmax_layer(aggregate_sweep_curves([res.score]))
        assert am == planted, f"planted={planted}: argmax={am}, scores={res.score}"


def test_old_proximity_logic_would_fail_this_stub():
    """THE DISCRIMINATING CONTRAST — old logic fails, new logic passes, on ONE
    stub.

    Emulate the OLD attribution design faithfully: read a fixed-direction metric
    at a FIXED LATE INTERMEDIATE layer ``L_read`` (= the L27 read-out), and score
    each layer by how much steering it changes that metric. Because the steered
    coordinate is reconstructed downstream (regeneration), a perturbation at
    layer ``L`` reaches ``L_read`` attenuated by ``decay**(L_read−L)`` — larger
    the closer ``L`` is to ``L_read``. So the OLD estimator ramps MONOTONICALLY
    to ``L_read`` (argmax = last layer), exactly the read-out-proximity artefact
    documented in results/patching/.../attribution_summary.md (all behaviours
    ramp to L26–27).

    The NEW forward-intervention estimator reads at the OUTPUT (never a fixed
    intermediate layer), so it has no proximity term and recovers the planted
    interior layer instead. The two argmaxes therefore DISAGREE — which is the
    whole point of the re-design.
    """
    d, vocab, n_layers, planted, read_tok = 8, 16, 8, 4, 5
    model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=n_layers,
                                   peak_layer=planted, read_tok=read_tok, seed=3))
    layers = list(range(n_layers))
    v = model.e0_direction
    directions = {L: v for L in layers}
    onset = 4
    ids = _ids_with_onset(read_tok, onset, T=6, vocab=vocab, seed=21)
    blocks = iter_residual_layers(model)
    L_read = n_layers - 1                       # the fixed late read-out (≈ L27)
    cap = {}

    def read_hook(_m, _i, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach().clone()

    def old_metric(steer_L=None):
        """Projection of resid[L_read, onset−1] onto v — the OLD fixed-late metric."""
        handles = []
        if steer_L is not None:
            handles.append(_register_steer_hook(
                model, SteerSpec(layer=steer_L, direction=v, alpha=1.0,
                                 sign=-1.0, max_pos=onset - 1)))
        handles.append(blocks[L_read].register_forward_hook(read_hook))
        with torch.no_grad():
            model(input_ids=ids)
        for h in handles:
            h.remove()
        return float((cap["h"][0, onset - 1] @ torch.as_tensor(v)).item())

    base_metric = old_metric(None)
    old_score = {L: abs(old_metric(L) - base_metric) for L in layers}
    old_argmax = max(old_score, key=old_score.get)

    res = layer_steering_sweep_single(
        model, ids, onset_pos=onset, directions=directions,
        layers=layers, alpha=1.0, span_tokens=1,
    )
    new_argmax = argmax_layer(aggregate_sweep_curves([res.score]))

    # OLD (fixed-late read-out) ramps to the last layer — the proximity artefact.
    assert old_argmax == L_read == n_layers - 1, \
        f"old fixed-late metric should ramp to L_read; got {old_argmax} ({old_score})"
    # And it IS monotone non-decreasing toward L_read (the ramp, not a peak).
    old_curve = [old_score[L] for L in layers]
    assert old_curve == sorted(old_curve), f"old metric not a monotone ramp: {old_curve}"
    # NEW (output read-out) recovers the interior layer — no proximity term.
    assert new_argmax == planted, f"new method should pick planted; got {new_argmax}"
    assert new_argmax != old_argmax            # the estimators disagree — the point


def test_interior_peak_holds_across_several_donors():
    """The interior peak survives aggregation over multiple donor chains."""
    d, vocab, n_layers, planted, read_tok = 8, 16, 8, 3, 5
    model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=n_layers,
                                   peak_layer=planted, read_tok=read_tok, seed=4))
    layers = list(range(n_layers))
    v = model.e0_direction
    directions = {L: v for L in layers}
    onset = 4

    per_donor = []
    for s in range(6):
        ids = _ids_with_onset(read_tok, onset, T=7, vocab=vocab, seed=100 + s)
        res = layer_steering_sweep_single(
            model, ids, onset_pos=onset, directions=directions,
            layers=layers, alpha=1.0, span_tokens=1,
        )
        per_donor.append(res.score)
    curve = aggregate_sweep_curves(per_donor)
    assert argmax_layer(curve) == planted
    assert argmax_layer(curve) != n_layers - 1
    assert curve[planted]["n"] == 6


def test_sweep_skips_early_layers_when_range_restricted():
    """The Venhoff 'ignore early layers' option: restricting ``layers`` to start
    above the embedding-correlated band still returns a valid curve over exactly
    the requested layers (and finds the planted peak if it is in range)."""
    d, vocab, n_layers, planted, read_tok = 8, 16, 8, 5, 5
    model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=n_layers,
                                   peak_layer=planted, read_tok=read_tok, seed=5))
    v = model.e0_direction
    directions = {L: v for L in range(n_layers)}
    onset = 4
    ids = _ids_with_onset(read_tok, onset, T=6, vocab=vocab, seed=33)
    layers = list(range(2, n_layers))          # skip L0, L1
    res = layer_steering_sweep_single(
        model, ids, onset_pos=onset, directions=directions,
        layers=layers, alpha=1.0, span_tokens=1,
    )
    assert set(res.score.keys()) == set(layers)
    assert 0 not in res.score and 1 not in res.score
    assert argmax_layer(aggregate_sweep_curves([res.score])) == planted


# ════════════════════════════════════════════════════════════════════════════
# 2. Monotone control — recovers a planted peak at the LAST swept layer
# ════════════════════════════════════════════════════════════════════════════


def test_recovers_planted_layer_when_latch_is_last():
    """If the read-out latch is the LAST block, regeneration makes the effect ramp
    monotonically to the deepest steerable layer — and the method recovers THAT
    layer (it is not biased toward the interior; it finds a genuine late peak
    too). With hook-on-output semantics, a latch on the last block is driven by
    steering the second-to-last layer, so the peak is at n_layers−2."""
    d, vocab, n_layers, read_tok = 8, 16, 8, 5
    planted = n_layers - 2                       # latch lands on the LAST block
    model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=n_layers,
                                   peak_layer=planted, read_tok=read_tok, seed=6))
    assert model.latch_layer == n_layers - 1
    layers = list(range(n_layers))
    v = model.e0_direction
    directions = {L: v for L in layers}
    onset = 4
    ids = _ids_with_onset(read_tok, onset, T=6, vocab=vocab, seed=41)
    res = layer_steering_sweep_single(
        model, ids, onset_pos=onset, directions=directions,
        layers=layers, alpha=1.0, span_tokens=1,
    )
    curve = aggregate_sweep_curves([res.score])
    # Monotone non-decreasing ramp over layers 0..planted, peaking at planted.
    means = [curve[L]["mean_effect"] for L in range(planted + 1)]
    assert means == sorted(means), f"expected monotone ramp, got {means}"
    assert argmax_layer(curve) == planted == n_layers - 2


# ════════════════════════════════════════════════════════════════════════════
# 3. Suppression sign is correct (and --amplify flips it)
# ════════════════════════════════════════════════════════════════════════════


def test_suppression_lowers_onset_logprob_so_score_is_positive():
    """Steering AWAY from the behaviour direction lowers the onset token's
    log-prob → Score = baseline − steered > 0. The read token's logit IS e0 here
    (regeneration coordinate read directly), and the donor's e0 starts positive,
    so subtracting along e0 cuts the logit → positive suppression score."""
    d, vocab, read_tok = 8, 16, 5
    # peak_layer=0 ⇒ latch on block 1, driven by steering layer 0. The donor's e0
    # starts positive and the read token's logit is +e2 ∝ +e0, so subtracting
    # along e0 over the onset−1 prefix lowers the onset token's logprob.
    model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=4, peak_layer=0,
                                   amp=4.0, read_tok=read_tok, seed=7))
    v = model.e0_direction
    onset = 2
    ids = _ids_with_onset(read_tok, onset, T=4, vocab=vocab, seed=51)

    base = onset_token_logprob(model, ids, onset, span_tokens=1, steer=None)
    spec_sup = SteerSpec(layer=0, direction=v, alpha=1.0, sign=-1.0, max_pos=onset - 1)
    steered = onset_token_logprob(model, ids, onset, span_tokens=1, steer=spec_sup)
    assert base - steered > 0, f"suppression should lower logprob: base={base} steered={steered}"

    # Amplify (sign +1) raises it → negative suppression score.
    spec_amp = SteerSpec(layer=0, direction=v, alpha=1.0, sign=+1.0, max_pos=onset - 1)
    amp = onset_token_logprob(model, ids, onset, span_tokens=1, steer=spec_amp)
    assert base - amp < 0


def test_amplify_flag_flips_sweep_sign():
    """The --amplify path (sign +1) yields scores of the opposite sign to the
    default suppress path on the same donor/direction."""
    d, vocab, read_tok = 8, 16, 5
    model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=6, peak_layer=3,
                                   read_tok=read_tok, seed=8))
    layers = list(range(6))
    v = model.e0_direction
    directions = {L: v for L in layers}
    onset = 4
    ids = _ids_with_onset(read_tok, onset, T=6, vocab=vocab, seed=61)

    sup = layer_steering_sweep_single(
        model, ids, onset_pos=onset, directions=directions, layers=layers,
        alpha=1.0, amplify=False, span_tokens=1,
    )
    amp = layer_steering_sweep_single(
        model, ids, onset_pos=onset, directions=directions, layers=layers,
        alpha=1.0, amplify=True, span_tokens=1,
    )
    # Only the layers with a meaningful (non-negligible) suppress effect must flip
    # sign: post-latch layers are numerically ~0 and the nonlinear log-softmax
    # makes their tiny suppress/amplify deltas not perfectly antisymmetric.
    peak = max(abs(s) for s in sup.score.values())
    checked = 0
    for L in layers:
        if abs(sup.score[L]) > 0.01 * peak:
            assert np.sign(sup.score[L]) != np.sign(amp.score[L]), (
                f"layer {L}: suppress {sup.score[L]} vs amplify {amp.score[L]} "
                "should have opposite sign"
            )
            checked += 1
    assert checked >= 2, "expected at least two layers with a clear, sign-flipping effect"


# ════════════════════════════════════════════════════════════════════════════
# 4. α = 0 is a no-op (score ≡ 0)
# ════════════════════════════════════════════════════════════════════════════


def test_alpha_zero_is_noop():
    """At α=0 the hook adds nothing → steered M == baseline M → every score 0."""
    d, vocab, read_tok = 8, 16, 5
    model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=7, peak_layer=3,
                                   read_tok=read_tok, seed=9))
    layers = list(range(7))
    v = model.e0_direction
    directions = {L: v for L in layers}
    onset = 4
    ids = _ids_with_onset(read_tok, onset, T=6, vocab=vocab, seed=71)

    res = layer_steering_sweep_single(
        model, ids, onset_pos=onset, directions=directions, layers=layers,
        alpha=0.0, span_tokens=1,
    )
    for L in layers:
        assert res.score[L] == pytest.approx(0.0, abs=1e-9), (
            f"layer {L} score {res.score[L]} != 0 at alpha=0"
        )


# ════════════════════════════════════════════════════════════════════════════
# 5. Read-out & plumbing
# ════════════════════════════════════════════════════════════════════════════


def test_onset_logprob_matches_manual_log_softmax():
    """The read-out is exactly the teacher-forced log-prob of the onset token,
    read from logits at position onset−1 (causal-LM shift)."""
    d, vocab = 6, 12
    model = _double(GainStubModel(d=d, vocab=vocab, gains=[1.0, 1.0, 1.0], seed=18))
    ids = _ids(T=5, vocab=vocab, seed=61)
    onset = 3
    got = onset_token_logprob(model, ids, onset, span_tokens=1, steer=None)
    with torch.no_grad():
        logits = model(input_ids=ids)
    lp = torch.log_softmax(logits[0].double(), dim=-1)
    want = float(lp[onset - 1, int(ids[0, onset].item())].item())
    assert got == pytest.approx(want, abs=1e-9)


def test_onset_logprob_averages_over_span():
    """span_tokens>1 averages the log-probs of the first up-to-N onset-span
    tokens (each predicted from its preceding position)."""
    d, vocab = 6, 12
    model = _double(GainStubModel(d=d, vocab=vocab, gains=[1.0, 1.0, 1.0], seed=19))
    ids = _ids(T=8, vocab=vocab, seed=71)
    onset, span = 3, 3
    got = onset_token_logprob(model, ids, onset, span_tokens=span, steer=None)
    with torch.no_grad():
        logits = model(input_ids=ids)
    lp = torch.log_softmax(logits[0].double(), dim=-1)
    manual = np.mean([
        float(lp[t - 1, int(ids[0, t].item())].item())
        for t in range(onset, onset + span)
    ])
    assert got == pytest.approx(float(manual), abs=1e-9)


def test_onset_logprob_span_clamps_at_sequence_end():
    """A span running past the sequence end is clamped, not an error."""
    d, vocab = 6, 12
    model = _double(GainStubModel(d=d, vocab=vocab, gains=[1.0, 1.0], seed=24))
    ids = _ids(T=4, vocab=vocab, seed=84)
    onset = 3                               # only token 3 exists; span 3 clamps to it
    got = onset_token_logprob(model, ids, onset, span_tokens=3, steer=None)
    with torch.no_grad():
        lp = torch.log_softmax(model(input_ids=ids)[0].double(), dim=-1)
    want = float(lp[onset - 1, int(ids[0, onset].item())].item())
    assert got == pytest.approx(want, abs=1e-9)


def test_onset_logprob_requires_prefix():
    model = _double(GainStubModel(d=6, vocab=12, gains=[1.0, 1.0], seed=20))
    ids = _ids(T=4, vocab=12, seed=81)
    with pytest.raises(ValueError):
        onset_token_logprob(model, ids, onset_pos=0, span_tokens=1)


def test_steer_hook_masks_to_prefix_positions():
    """The max_pos window steers only positions ≤ max_pos. With a stub that has
    NO cross-position mixing, steering positions strictly BELOW the read-out
    position (max_pos = onset−2) cannot reach the onset read-out (read from
    position onset−1) → score exactly 0. This pins the masking semantics: only
    the steered positions matter, and they propagate within their own column."""
    d, vocab, read_tok = 8, 16, 5
    model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=5, peak_layer=2,
                                   read_tok=read_tok, seed=22))
    layers = list(range(5))
    v = model.e0_direction
    onset = 4
    ids = _ids_with_onset(read_tok, onset, T=6, vocab=vocab, seed=91)
    # Steer only up to onset−2 (exclude the onset−1 column that feeds the read-out).
    base = onset_token_logprob(model, ids, onset, span_tokens=1, steer=None)
    for L in layers:
        spec = SteerSpec(layer=L, direction=v, alpha=2.0, sign=-1.0, max_pos=onset - 2)
        m = onset_token_logprob(model, ids, onset, span_tokens=1, steer=spec)
        assert (base - m) == pytest.approx(0.0, abs=1e-9), (
            f"layer {L}: steering below the read-out column must not move it "
            f"(no cross-position mixing); got Δ={base - m}"
        )


def test_iter_residual_layers_finds_stub_layers():
    model = GainStubModel(d=4, vocab=8, gains=[1.0, 1.0, 1.0])
    assert len(iter_residual_layers(model)) == 3


def test_aggregate_curve_shape_matches_consumer():
    """aggregate_sweep_curves emits {layer: {mean_effect, sem_effect, n}} — the
    shape compute_layer_triangulation.load_phase7b_curves reads."""
    per_donor = [{2: 1.0, 3: 2.0, 4: 3.0}, {2: 3.0, 3: 4.0, 4: 5.0}]
    agg = aggregate_sweep_curves(per_donor)
    assert set(agg.keys()) == {2, 3, 4}
    assert agg[2]["mean_effect"] == pytest.approx(2.0)
    assert agg[2]["n"] == 2
    assert "sem_effect" in agg[2]
    assert agg[4]["mean_effect"] == pytest.approx(4.0)


def test_aggregate_handles_nan_and_missing():
    per_donor = [{2: 1.0}, {2: float("nan"), 3: 2.0}, {3: 4.0}]
    agg = aggregate_sweep_curves(per_donor)
    assert agg[2]["n"] == 1 and agg[2]["mean_effect"] == pytest.approx(1.0)
    assert agg[3]["n"] == 2 and agg[3]["mean_effect"] == pytest.approx(3.0)


def test_argmax_layer_ignores_nan():
    curve = {2: {"mean_effect": float("nan")}, 3: {"mean_effect": 1.0},
             4: {"mean_effect": 0.5}}
    assert argmax_layer(curve) == 3


def test_empty_aggregate_is_empty():
    assert aggregate_sweep_curves([]) == {}
    assert argmax_layer({}) is None


def test_per_layer_directions_builds_unit_diff_of_means(tmp_path):
    """per_layer_directions reproduces src.steering.single_direction_vector at
    each layer from the on/off .npy files (ON = behaviour, OFF = others)."""
    from src.steering import single_direction_vector

    rng = np.random.default_rng(101)
    d = 5
    for L in (2, 3):
        np.save(tmp_path / f"b_layer{L}.npy", rng.standard_normal((7, d)))
        np.save(tmp_path / f"x_layer{L}.npy", rng.standard_normal((4, d)))
        np.save(tmp_path / f"y_layer{L}.npy", rng.standard_normal((3, d)))
    dirs = per_layer_directions(tmp_path, "b", layers=[2, 3], other_behaviours=["x", "y"])
    assert set(dirs) == {2, 3}
    for L in (2, 3):
        on = np.load(tmp_path / f"b_layer{L}.npy").astype(np.float64)
        off = np.concatenate([
            np.load(tmp_path / f"x_layer{L}.npy").astype(np.float64),
            np.load(tmp_path / f"y_layer{L}.npy").astype(np.float64),
        ])
        want = single_direction_vector(on, off)
        assert np.allclose(dirs[L], want)
        assert dirs[L].shape == (d,)
        assert np.linalg.norm(dirs[L]) == pytest.approx(1.0, abs=1e-9)


def test_per_layer_directions_skips_missing_layer(tmp_path):
    rng = np.random.default_rng(202)
    d = 4
    np.save(tmp_path / "b_layer2.npy", rng.standard_normal((5, d)))
    np.save(tmp_path / "x_layer2.npy", rng.standard_normal((5, d)))
    # Layer 3 ON file is absent → skipped.
    dirs = per_layer_directions(tmp_path, "b", layers=[2, 3], other_behaviours=["x"])
    assert set(dirs) == {2}


# ════════════════════════════════════════════════════════════════════════════
# 6. MUST-FIX #1 — per-layer norm-matched RANDOM-DIRECTION null
# ════════════════════════════════════════════════════════════════════════════


def test_random_null_zeroes_a_globally_sensitive_layer():
    """The de-confounding test. On a stub layer that is sensitive to the MAGNITUDE
    of ANY norm-matched perturbation (not to the behaviour direction), the
    behaviour effect and the random-null effect are equal → the de-confounded
    effect (= behaviour − mean random) is ~0. This is exactly the residual
    proximity/global-sensitivity term the random null is meant to remove."""
    d, vocab, n_layers, sens = 8, 16, 8, 4
    read_tok, probe_tok = 5, 6
    model = _double(IsoNormStub(d=d, vocab=vocab, n_layers=n_layers,
                                sensitive_layer=sens, read_tok=read_tok,
                                probe_tok=probe_tok, seed=1))
    layers = list(range(n_layers))
    # Behaviour direction can be ANY unit vector — the iso layer treats it like the
    # random ones. Use a generic (non-axis) direction to make the point.
    v = _unit(d, seed=99)
    directions = {L: v for L in layers}
    onset = 4
    ids = _ids_with_onset(read_tok, onset, T=6, vocab=vocab, seed=303)
    ids[0, onset - 1] = probe_tok                     # probe column (fixed baseline c)

    res = layer_steering_sweep_single(
        model, ids, onset_pos=onset, directions=directions, layers=layers,
        alpha=1.0, span_tokens=1, readout="kl", n_random=3,
        delta_frac=0.1, behaviour="iso",
    )
    # At the planted globally-sensitive layer the behaviour and random KL effects
    # coincide → de-confounded ~0, even though the RAW behaviour effect is large.
    assert res.behaviour_effect[sens] > 1e-6, "iso layer should have a real RAW effect"
    assert res.random_null[sens] == pytest.approx(res.behaviour_effect[sens], rel=1e-6), \
        f"random null should match behaviour at iso layer: {res.random_null[sens]} vs {res.behaviour_effect[sens]}"
    assert res.score[sens] == pytest.approx(0.0, abs=1e-9), \
        f"de-confounded effect at the globally-sensitive layer should be ~0, got {res.score[sens]}"
    assert res.kl_effect[sens] == pytest.approx(0.0, abs=1e-9)


def test_random_null_preserves_a_behaviour_specific_layer():
    """The complement: on a behaviour-SELECTIVE layer (RegenLatch e0-latch), random
    directions miss the latched coordinate, so the random null is ~0 and the
    de-confounded effect retains the genuine behaviour effect (does not zero it)."""
    d, vocab, n_layers, planted, read_tok = 8, 16, 8, 4, 5
    model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=n_layers,
                                   peak_layer=planted, read_tok=read_tok, seed=3))
    layers = list(range(n_layers))
    v = model.e0_direction                            # axis-aligned behaviour dir
    directions = {L: v for L in layers}
    onset = 4
    ids = _ids_with_onset(read_tok, onset, T=6, vocab=vocab, seed=304)

    res = layer_steering_sweep_single(
        model, ids, onset_pos=onset, directions=directions, layers=layers,
        alpha=1.0, span_tokens=1, readout="kl", n_random=3,
        delta_frac=0.1, behaviour="backtracking",
    )
    # Random directions barely touch the e0 latch → null ≪ behaviour effect, so
    # the de-confounded effect stays clearly positive and the argmax is still the
    # planted layer.
    assert res.behaviour_effect[planted] > 0
    assert res.random_null[planted] < 0.5 * res.behaviour_effect[planted]
    assert res.score[planted] > 0
    assert argmax_layer(aggregate_sweep_curves([res.score])) == planted


def test_random_unit_direction_is_deterministic_and_unit():
    """The null directions are reproducible per (behaviour, layer, r) and unit-norm;
    different (behaviour, layer, r) give different directions."""
    a = random_unit_direction(16, "backtracking", 5, 0)
    b = random_unit_direction(16, "backtracking", 5, 0)
    assert np.allclose(a, b)                          # deterministic
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-12)
    assert not np.allclose(a, random_unit_direction(16, "backtracking", 5, 1))
    assert not np.allclose(a, random_unit_direction(16, "backtracking", 6, 0))
    assert not np.allclose(a, random_unit_direction(16, "uncertainty", 5, 0))


# ════════════════════════════════════════════════════════════════════════════
# 6b. CF-10 fix — COVARIANCE-MATCHED null (anisotropy-aware de-confounder)
# ════════════════════════════════════════════════════════════════════════════


def test_ambient_covariance_recovers_planted_anisotropy(tmp_path):
    """ambient_covariance reads the {b}_layer{ℓ}.npy files and returns a symmetric
    PSD (d,d) covariance whose diagonal reflects the planted per-dim variance."""
    from src.layer_sweep import ambient_covariance

    rng = np.random.default_rng(7)
    d, L = 6, 9
    scales = np.array([10.0, 5.0, 1.0, 1.0, 0.5, 0.2])     # strongly anisotropic
    for b, n in (("backtracking", 400), ("uncertainty-estimation", 300)):
        X = rng.standard_normal((n, d)) * scales
        np.save(tmp_path / f"{b}_layer{L}.npy", X.astype(np.float32))
    cov = ambient_covariance(
        tmp_path, ["backtracking", "uncertainty-estimation"], L,
        max_rows=10000, seed=0)
    assert cov is not None and cov.shape == (d, d)
    assert np.allclose(cov, cov.T, atol=1e-8)              # symmetric
    assert np.linalg.eigvalsh(cov).min() > 0               # PD after shrinkage
    diag = np.diag(cov)
    assert np.argmax(diag) == 0 and np.argmin(diag) == d - 1   # tracks `scales`
    assert diag[0] > 10 * diag[-1]                         # anisotropy preserved


def test_ambient_covariance_missing_files_returns_none(tmp_path):
    from src.layer_sweep import ambient_covariance
    assert ambient_covariance(tmp_path, ["backtracking"], 3) is None


def test_covariance_matched_directions_unit_deterministic_and_anisotropic():
    """Covariance-matched null draws are unit-norm, reproducible per (behaviour,
    layer, r), and — unlike the ISOTROPIC null — concentrate along the high-
    variance eigen-axis of Σ (the whole point of the CF-10 fix)."""
    from src.layer_sweep import covariance_matched_unit_directions

    d = 8
    var = np.array([100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])   # axis-0 dominates
    cov = np.diag(var)

    a = covariance_matched_unit_directions(cov, "backtracking", 5, 1)[0]
    b = covariance_matched_unit_directions(cov, "backtracking", 5, 1)[0]
    assert np.allclose(a, b)                                       # deterministic
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-9)       # unit
    c = covariance_matched_unit_directions(cov, "uncertainty", 5, 1)[0]
    assert not np.allclose(a, c)                                   # behaviour-seeded

    # Anisotropy: covariance-matched draws load heavily on axis-0; isotropic don't.
    cov_draws = covariance_matched_unit_directions(cov, "b", 5, 200)
    iso_draws = [random_unit_direction(d, "b", 5, r) for r in range(200)]
    cov_axis0 = float(np.mean([abs(v[0]) for v in cov_draws]))
    iso_axis0 = float(np.mean([abs(v[0]) for v in iso_draws]))
    assert cov_axis0 > 2 * iso_axis0, (cov_axis0, iso_axis0)
    assert cov_axis0 > 0.8           # nearly aligned with the dominant axis


def test_null_directions_override_reproduces_default_isotropic_path():
    """Regression: feeding the sweep null_directions built from the SAME isotropic
    seeds reproduces the default in-sweep null exactly — proving the new override
    plumbing is a no-op when given matching draws (so covariance mode changes ONLY
    the draw distribution, nothing else)."""
    d, vocab, n_layers, planted, read_tok = 8, 16, 8, 4, 5
    layers = list(range(n_layers))
    onset = 4
    ids = _ids_with_onset(read_tok, onset, T=6, vocab=vocab, seed=304)

    def run(null_directions):
        model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=n_layers,
                                       peak_layer=planted, read_tok=read_tok, seed=3))
        directions = {L: model.e0_direction for L in layers}
        return layer_steering_sweep_single(
            model, ids, onset_pos=onset, directions=directions, layers=layers,
            alpha=1.0, span_tokens=1, readout="kl", n_random=3,
            delta_frac=0.1, behaviour="backtracking",
            null_directions=null_directions)

    default = run(None)
    explicit = run({L: [random_unit_direction(d, "backtracking", L, r)
                        for r in range(3)] for L in layers})
    for L in layers:
        assert explicit.score[L] == pytest.approx(default.score[L], rel=1e-9, abs=1e-12)
        assert explicit.random_null[L] == pytest.approx(default.random_null[L], rel=1e-9, abs=1e-12)


# ════════════════════════════════════════════════════════════════════════════
# 7. MUST-FIX #2 — per-layer α-normalization (fixed-fraction delta norm)
# ════════════════════════════════════════════════════════════════════════════


def test_collect_layer_norms_matches_manual_median():
    """collect_layer_norms returns the median ‖h‖ over the steered prefix at each
    block output — verified against a manual per-position-norm median."""
    d, vocab = 8, 16
    model = _double(GainStubModel(d=d, vocab=vocab, gains=[1.5, 2.0, 0.5], seed=31))
    ids = _ids(T=6, vocab=vocab, seed=131)
    max_pos = 3
    norms = collect_layer_norms(model, ids, layers=[0, 1, 2], max_pos=max_pos)
    # Recompute manually: capture each block output, take per-position L2 norm over
    # positions ≤ max_pos, median.
    blocks = iter_residual_layers(model)
    caps = {}
    handles = [blocks[L].register_forward_hook(
        (lambda LL: (lambda _m, _i, o: caps.__setitem__(LL, (o[0] if isinstance(o, tuple) else o).detach())))(L))
        for L in (0, 1, 2)]
    with torch.no_grad():
        model(input_ids=ids)
    for h in handles:
        h.remove()
    for L in (0, 1, 2):
        per_pos = torch.linalg.vector_norm(caps[L][0, :max_pos + 1], dim=-1)
        assert norms[L] == pytest.approx(float(per_pos.median().item()), abs=1e-9)


def test_delta_frac_sets_delta_norm_to_fraction_of_h():
    """With delta_frac set and a single steered position, the injected delta has
    norm == delta_frac · ‖h‖ at the probe (independent of the direction). Measured
    by the change in the probe residual at the hooked layer's output."""
    d, vocab = 8, 16
    gains = [1.0, 1.0, 1.0, 1.0]
    model = _double(GainStubModel(d=d, vocab=vocab, gains=gains, seed=32))
    ids = _ids(T=5, vocab=vocab, seed=132)
    onset = 3
    max_pos = onset - 1                                # single steered column (=2)
    L = 1
    # Baseline residual at the hooked layer's output, probe column.
    blocks = iter_residual_layers(model)
    cap = {}
    h0 = blocks[L].register_forward_hook(
        lambda _m, _i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach().clone()))
    with torch.no_grad():
        model(input_ids=ids)
    h0.remove()
    base_probe = cap["h"][0, max_pos].clone()
    h_norm = float(torch.linalg.vector_norm(base_probe).item())
    delta_frac = 0.1

    # Now capture the steered residual at the same layer/column. The steer hook
    # must register BEFORE the capture hook so the capture sees the steered output
    # (forward hooks fire in registration order; a returned value replaces the
    # output for later hooks).
    v = _unit(d, seed=77)
    target = delta_frac * h_norm
    spec = SteerSpec(layer=L, direction=v, alpha=1.0, sign=-1.0,
                     max_pos=max_pos, target_delta_norm=target)
    cap2 = {}
    handle = _register_steer_hook(model, spec)
    h1 = blocks[L].register_forward_hook(
        lambda _m, _i, o: cap2.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach().clone()))
    with torch.no_grad():
        model(input_ids=ids)
    h1.remove()
    handle.remove()
    delta = cap2["h"][0, max_pos] - base_probe
    assert float(torch.linalg.vector_norm(delta).item()) == pytest.approx(target, abs=1e-9), \
        "delta norm should equal delta_frac · ‖h‖ at the probe"


def test_delta_frac_norm_matches_behaviour_and_random_equally():
    """The α-normalization is direction-agnostic: a behaviour direction and a
    random direction injected with the SAME target_delta_norm deliver deltas of
    the SAME norm (the precondition for a fair random null)."""
    d, vocab = 8, 16
    model = _double(GainStubModel(d=d, vocab=vocab, gains=[1.0, 1.0, 1.0], seed=33))
    ids = _ids(T=5, vocab=vocab, seed=133)
    onset, L = 3, 1
    max_pos = onset - 1
    blocks = iter_residual_layers(model)

    def probe_delta(direction, target):
        base, steer = {}, {}
        hb = blocks[L].register_forward_hook(
            lambda _m, _i, o: base.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach().clone()))
        with torch.no_grad():
            model(input_ids=ids)
        hb.remove()
        spec = SteerSpec(layer=L, direction=direction, alpha=1.0, sign=-1.0,
                         max_pos=max_pos, target_delta_norm=target)
        # Steer hook BEFORE capture so the capture observes the steered output.
        handle = _register_steer_hook(model, spec)
        hs = blocks[L].register_forward_hook(
            lambda _m, _i, o: steer.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach().clone()))
        with torch.no_grad():
            model(input_ids=ids)
        hs.remove()
        handle.remove()
        return float(torch.linalg.vector_norm(steer["h"][0, max_pos] - base["h"][0, max_pos]).item())

    target = 0.37
    nb = probe_delta(_unit(d, 1), target)
    nr = probe_delta(random_unit_direction(d, "b", L, 0), target)
    assert nb == pytest.approx(target, abs=1e-9)
    assert nr == pytest.approx(target, abs=1e-9)
    assert nb == pytest.approx(nr, abs=1e-9)


# ════════════════════════════════════════════════════════════════════════════
# 8. MUST-FIX #3 — KL read-out (primary)
# ════════════════════════════════════════════════════════════════════════════


def test_onset_kl_matches_manual_kl():
    """onset_kl == Σ_v p_steered(v)·(log p_steered − log p_base) at onset−1."""
    d, vocab, read_tok = 8, 16, 5
    model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=6, peak_layer=2,
                                   read_tok=read_tok, seed=7))
    v = model.e0_direction
    onset = 3
    ids = _ids_with_onset(read_tok, onset, T=6, vocab=vocab, seed=141)
    spec = SteerSpec(layer=2, direction=v, alpha=1.0, sign=-1.0, max_pos=onset - 1)

    base_lp = _next_token_logprobs(model, ids, steer=None)
    steer_lp = _next_token_logprobs(model, ids, steer=spec)
    pos = onset - 1
    p_s = torch.exp(steer_lp[pos])
    manual = float((p_s * (steer_lp[pos] - base_lp[pos])).sum().item())

    got = onset_kl(model, ids, onset, baseline_logprobs=base_lp, steer=spec)
    assert got == pytest.approx(max(0.0, manual), abs=1e-9)
    assert got >= 0.0


def test_onset_kl_zero_when_no_steer():
    d, vocab, read_tok = 8, 16, 5
    model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=5, peak_layer=2,
                                   read_tok=read_tok, seed=8))
    ids = _ids_with_onset(read_tok, 3, T=5, vocab=vocab, seed=142)
    assert onset_kl(model, ids, 3, steer=None) == 0.0


def test_kl_readout_recovers_interior_peak_on_stub():
    """The KL primary read-out works on the interior-peak stub: argmax of the
    de-confounded KL curve recovers the planted middle layer (KL catches the
    redistribution the single-token read-out sees here too)."""
    d, vocab, n_layers, planted, read_tok = 8, 16, 8, 4, 5
    model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=n_layers,
                                   peak_layer=planted, read_tok=read_tok, seed=2))
    layers = list(range(n_layers))
    v = model.e0_direction
    directions = {L: v for L in layers}
    onset = 4
    ids = _ids_with_onset(read_tok, onset, T=6, vocab=vocab, seed=143)
    res = layer_steering_sweep_single(
        model, ids, onset_pos=onset, directions=directions, layers=layers,
        alpha=1.0, span_tokens=1, readout="kl", n_random=0,
    )
    # With n_random=0 the de-confounded score == raw KL effect.
    assert res.readout == "kl"
    assert argmax_layer(aggregate_sweep_curves([res.score])) == planted
    assert all(res.score[L] >= -1e-9 for L in layers)          # KL ≥ 0
    # logprob_effect is carried as the SECONDARY diagnostic.
    assert set(res.logprob_effect) == set(layers)


# ════════════════════════════════════════════════════════════════════════════
# 9. MUST-FIX #4 — bootstrap the argmax → a SHORTLIST
# ════════════════════════════════════════════════════════════════════════════


def test_bootstrap_shortlist_returns_multiple_layers_on_flat_noisy_input():
    """On a FLAT/noisy de-confounded curve (no single dominant layer), the
    bootstrap shortlist returns MULTIPLE candidate layers — not one load-bearing
    argmax — because the argmax jumps around under resampling."""
    rng = np.random.default_rng(7)
    layers = [5, 6, 7, 8]
    # Per-donor scores: all layers same mean (0), pure noise → argmax is unstable.
    per_donor = [{L: float(rng.normal(0.0, 1.0)) for L in layers} for _ in range(40)]
    out = bootstrap_shortlist(per_donor, n_boot=500, select_frac=0.15, seed=0)
    assert len(out["shortlist"]) >= 2, f"flat curve should shortlist ≥2: {out}"
    # No single layer dominates the bootstrap.
    assert max(out["boot_freq"].values()) < 0.9
    assert sum(out["boot_freq"].values()) == pytest.approx(1.0, abs=1e-9)


def test_bootstrap_shortlist_concentrates_on_clear_winner():
    """When ONE layer clearly dominates, the bootstrap concentrates on it (it wins
    the vast majority of resamples) while still returning a small shortlist."""
    rng = np.random.default_rng(8)
    layers = [5, 6, 7, 8]
    per_donor = []
    for _ in range(40):
        d = {L: float(rng.normal(0.0, 0.3)) for L in layers}
        d[7] = float(rng.normal(5.0, 0.3))            # layer 7 dominates
        per_donor.append(d)
    out = bootstrap_shortlist(per_donor, n_boot=500, select_frac=0.15, seed=1)
    assert out["argmax"] == 7
    assert out["boot_freq"][7] > 0.9
    assert 7 in out["shortlist"]
    assert len(out["shortlist"]) <= 3


def test_bootstrap_shortlist_within_one_sem_union():
    """The shortlist includes layers within 1 SEM of the max mean even if they
    rarely win the bootstrap argmax (the 'within 1 SEM' arm of the union)."""
    # Two near-tied top layers (6, 7) within 1 SEM of each other, and a clear loser
    # (5). With a high select_frac, the bootstrap-frequency arm alone would NOT
    # shortlist the non-winning top layer — the 1-SEM arm must pull it in.
    rng = np.random.default_rng(9)
    per_donor = []
    for _ in range(40):
        per_donor.append({
            5: float(rng.normal(0.0, 0.3)),
            6: float(rng.normal(2.00, 0.3)),
            7: float(rng.normal(2.02, 0.3)),       # ~tied with 6; gap ≪ 1 SEM
        })
    agg = aggregate_sweep_curves(per_donor)
    # Sanity: the two top layers really are within 1 SEM of the max.
    top = max((6, 7), key=lambda L: agg[L]["mean_effect"])
    other = 6 if top == 7 else 7
    assert agg[other]["mean_effect"] >= agg[top]["mean_effect"] - agg[top]["sem_effect"]
    out = bootstrap_shortlist(per_donor, n_boot=800, select_frac=0.95, seed=2)
    assert 6 in out["shortlist"] and 7 in out["shortlist"], out
    assert 5 not in out["shortlist"]


def test_bootstrap_shortlist_empty_input():
    out = bootstrap_shortlist([], n_boot=100)
    assert out["shortlist"] == [] and out["argmax"] is None


# ════════════════════════════════════════════════════════════════════════════
# 10. MUST-FIX #5 — early-layer exclusion (embedding cosine)
# ════════════════════════════════════════════════════════════════════════════


def test_embedding_cosine_flags_embedding_aligned_direction():
    """embedding_cosine reports max |cos| of v_ℓ to any embedding row: an
    embedding-aligned direction scores ~1, an orthogonal one scores low."""
    rng = np.random.default_rng(11)
    d, vocab = 6, 10
    E = rng.standard_normal((vocab, d))
    aligned = E[3] / np.linalg.norm(E[3])              # exactly an embedding row dir
    # A direction in the null space of E (orthogonal to all rows) when vocab<d would
    # be exact; here just use a low-overlap random direction and assert < aligned.
    other = _unit(d, seed=123)
    cos = embedding_cosine({2: aligned, 3: other}, E)
    assert cos[2] == pytest.approx(1.0, abs=1e-9)
    assert cos[3] < cos[2]


# ════════════════════════════════════════════════════════════════════════════
# 11. Full per-layer record + de-confounded aggregation (runner output shape)
# ════════════════════════════════════════════════════════════════════════════


def test_aggregate_full_curves_emits_all_components():
    """aggregate_full_curves emits the per-layer record the runner writes:
    {behaviour_effect, random_null, deconfounded_effect, kl_effect,
    logprob_effect, mean_effect(=deconfounded), sem, n}."""
    d, vocab, n_layers, planted, read_tok = 8, 16, 8, 4, 5
    model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=n_layers,
                                   peak_layer=planted, read_tok=read_tok, seed=4))
    layers = list(range(n_layers))
    v = model.e0_direction
    directions = {L: v for L in layers}
    onset = 4
    per_donor = []
    for s in range(4):
        ids = _ids_with_onset(read_tok, onset, T=6, vocab=vocab, seed=200 + s)
        per_donor.append(layer_steering_sweep_single(
            model, ids, onset_pos=onset, directions=directions, layers=layers,
            alpha=1.0, span_tokens=1, readout="kl", n_random=2,
            delta_frac=0.1, behaviour="backtracking"))
    full = aggregate_full_curves(per_donor)
    for L in layers:
        rec = full[L]
        for k in ("behaviour_effect", "random_null", "deconfounded_effect",
                  "kl_effect", "logprob_effect", "mean_effect", "sem", "n"):
            assert k in rec, f"missing {k} at layer {L}"
        # mean_effect mirrors the de-confounded effect (drop-in for triangulation).
        assert rec["mean_effect"] == pytest.approx(rec["deconfounded_effect"])
        assert rec["n"] == 4
    # The de-confounded argmax (via mean_effect) is the planted layer.
    assert argmax_layer(full) == planted


def test_aggregate_full_curves_sem_is_standard():
    """aggregate_full_curves SEM == std(ddof=1)/sqrt(n) of the de-confounded score."""
    # Hand-built SweepResults with known scores per layer.
    def mk(score):
        return SweepResult(
            behaviour="b", layers=sorted(score), score=dict(score),
            baseline_M=0.0, steered_M={}, onset_pos=2, n_span_tokens=1,
            readout="kl", behaviour_effect=dict(score), random_null={L: 0.0 for L in score},
            kl_effect=dict(score), logprob_effect=dict(score), n_random=2)
    per_donor = [mk({5: 1.0}), mk({5: 2.0}), mk({5: 3.0}), mk({5: 4.0})]
    full = aggregate_full_curves(per_donor)
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    assert full[5]["sem"] == pytest.approx(float(np.std(vals, ddof=1) / np.sqrt(4)))
    assert full[5]["mean_effect"] == pytest.approx(2.5)


def test_sweep_score_is_deconfounded_with_random_null():
    """End-to-end: SweepResult.score == behaviour_effect − random_null per layer
    (the de-confounded effect), in the selected read-out."""
    d, vocab, n_layers, planted, read_tok = 8, 16, 8, 3, 5
    model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=n_layers,
                                   peak_layer=planted, read_tok=read_tok, seed=6))
    layers = list(range(n_layers))
    v = model.e0_direction
    directions = {L: v for L in layers}
    onset = 4
    ids = _ids_with_onset(read_tok, onset, T=6, vocab=vocab, seed=205)
    res = layer_steering_sweep_single(
        model, ids, onset_pos=onset, directions=directions, layers=layers,
        alpha=1.0, span_tokens=1, readout="kl", n_random=3,
        delta_frac=0.1, behaviour="backtracking")
    for L in layers:
        assert res.score[L] == pytest.approx(
            res.behaviour_effect[L] - res.random_null[L], abs=1e-12)


def test_backward_compatible_logprob_default_unchanged():
    """The legacy default (readout='logprob', n_random=0, delta_frac=None) keeps
    score == M_baseline − M_steered along the behaviour direction — so the
    existing interior-peak tests are exercising the unchanged path."""
    d, vocab, n_layers, planted, read_tok = 8, 16, 8, 4, 5
    model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=n_layers,
                                   peak_layer=planted, read_tok=read_tok, seed=1))
    layers = list(range(n_layers))
    v = model.e0_direction
    directions = {L: v for L in layers}
    onset = 4
    ids = _ids_with_onset(read_tok, onset, T=6, vocab=vocab, seed=11)
    res = layer_steering_sweep_single(
        model, ids, onset_pos=onset, directions=directions, layers=layers,
        alpha=1.0, span_tokens=1)                      # all defaults
    assert res.readout == "logprob" and res.n_random == 0
    base = onset_token_logprob(model, ids, onset, span_tokens=1, steer=None)
    for L in layers:
        spec = SteerSpec(layer=L, direction=v, alpha=1.0, sign=-1.0, max_pos=onset - 1)
        m = onset_token_logprob(model, ids, onset, span_tokens=1, steer=spec)
        assert res.score[L] == pytest.approx(base - m, abs=1e-12)


# ════════════════════════════════════════════════════════════════════════════
# 12. Local-window truncation for long chains (truncate_to_window)
# ════════════════════════════════════════════════════════════════════════════
#
# Tractability on ~8k-token chains: crop each donor to [max(0, onset−W) :
# onset+span] and remap the onset, WITHOUT changing the result for short chains
# (onset ≤ W) — see truncate_to_window's docstring. The RegenLatchStub processes
# each position independently (no cross-position mixing), so the sweep's onset
# read-out at onset−1 is a pure function of the kept onset-prefix columns; a crop
# that preserves the (remapped) onset column therefore reproduces the per-layer
# scores. These tests pin (a) the short-chain no-op, (b) the index/shape remap,
# and (c) the de-confounded-argmax invariance under a crop that covers the causal
# window.


def test_truncate_to_window_noop_when_onset_fits():
    """(a) When the whole prefix already fits the window (onset < W) and the span
    tail does not extend past T, the crop is a TENSOR-level no-op: identical ids,
    identical onset, and the per-layer sweep scores are byte-identical to the
    untruncated run on the stub."""
    d, vocab, n_layers, planted, read_tok = 8, 16, 8, 4, 5
    model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=n_layers,
                                   peak_layer=planted, read_tok=read_tok, seed=1))
    layers = list(range(n_layers))
    v = model.e0_direction
    directions = {L: v for L in layers}
    onset, span, T = 4, 1, 5          # onset+span == T ⇒ tail bound is T (kept whole)
    ids = _ids_with_onset(read_tok, onset, T=T, vocab=vocab, seed=11)

    W = 1024                          # onset (4) ≪ W ⇒ lo == 0
    ids_local, onset_local = truncate_to_window(
        ids, onset, context_window=W, span_tokens=span)
    # Tensor-level no-op: same onset, same ids (lo==0 and hi==T).
    assert onset_local == onset
    assert ids_local.shape == ids.shape
    assert torch.equal(ids_local, ids)

    # And the per-layer scores match byte-for-byte (it is literally the same input).
    full = layer_steering_sweep_single(
        model, ids, onset_pos=onset, directions=directions, layers=layers,
        alpha=1.0, span_tokens=span, readout="kl", n_random=2,
        delta_frac=0.1, behaviour="backtracking")
    crop = layer_steering_sweep_single(
        model, ids_local, onset_pos=onset_local, directions=directions, layers=layers,
        alpha=1.0, span_tokens=span, readout="kl", n_random=2,
        delta_frac=0.1, behaviour="backtracking")
    for L in layers:
        assert crop.score[L] == full.score[L]                       # exact equality
        assert crop.behaviour_effect[L] == full.behaviour_effect[L]
        assert crop.random_null[L] == full.random_null[L]
        assert crop.kl_effect[L] == full.kl_effect[L]
    assert crop.onset_pos == full.onset_pos


def test_truncate_to_window_noop_scores_even_when_tail_trimmed():
    """(a, sharper) Even when the post-onset TAIL is trimmed (onset < W but T is
    long after the onset), the crop leaves every per-layer score unchanged to
    floating-point precision — the dropped tail is past onset+span and, by
    autoregressive causality, never enters the onset read-out. (Exact equality is
    only guaranteed when the tensor itself is unchanged, as in the no-op test
    above; here the tensor LENGTH changes, so the log-softmax is reduced over a
    different-sized row count and the result agrees to rounding, not bit-for-bit
    — this is the mathematically-exact causal guarantee realised in float64.)"""
    d, vocab, n_layers, planted, read_tok = 8, 16, 8, 4, 5
    model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=n_layers,
                                   peak_layer=planted, read_tok=read_tok, seed=2))
    layers = list(range(n_layers))
    v = model.e0_direction
    directions = {L: v for L in layers}
    onset, span = 4, 1
    # Long tail AFTER the onset (T=40), but onset (4) < W ⇒ no pre-onset crop.
    ids = _ids_with_onset(read_tok, onset, T=40, vocab=vocab, seed=12)

    W = 1024
    ids_local, onset_local = truncate_to_window(
        ids, onset, context_window=W, span_tokens=span)
    assert onset_local == onset                       # prefix untouched
    assert int(ids_local.shape[-1]) == onset + span   # tail trimmed to onset+span
    assert int(ids_local.shape[-1]) < int(ids.shape[-1])

    full = layer_steering_sweep_single(
        model, ids, onset_pos=onset, directions=directions, layers=layers,
        alpha=1.0, span_tokens=span, readout="kl", n_random=2,
        delta_frac=0.1, behaviour="backtracking")
    crop = layer_steering_sweep_single(
        model, ids_local, onset_pos=onset_local, directions=directions, layers=layers,
        alpha=1.0, span_tokens=span, readout="kl", n_random=2,
        delta_frac=0.1, behaviour="backtracking")
    for L in layers:
        assert crop.score[L] == pytest.approx(full.score[L], rel=1e-9, abs=1e-12), (
            f"layer {L}: trimming the post-onset tail must not change the score "
            f"(crop {crop.score[L]} vs full {full.score[L]})")


def test_truncate_to_window_remaps_onset_and_bounds_length():
    """(b) A LONG synthetic sequence is cropped to length ≤ W + span_tokens, and
    the remapped onset points at the SAME token id as the original onset."""
    vocab, read_tok = 64, 5
    T, onset, W, span = 4000, 3200, 256, 3          # onset ≫ W ⇒ real pre-onset crop
    ids = _ids_with_onset(read_tok, onset, T=T, vocab=vocab, seed=777)
    # Mark the onset−1 token distinctively too, to check the prefix alignment.
    sentinel = 41
    ids[0, onset - 1] = sentinel

    ids_local, onset_local = truncate_to_window(
        ids, onset, context_window=W, span_tokens=span)

    # Length is bounded by W + span (here exactly W + span: lo=onset−W, hi=onset+span).
    assert int(ids_local.shape[-1]) <= W + span
    assert int(ids_local.shape[-1]) == W + span
    # The remapped onset lands on the SAME token, and onset_local == W (lo=onset−W).
    assert onset_local == W
    assert int(ids_local[0, onset_local].item()) == read_tok
    assert int(ids_local[0, onset_local].item()) == int(ids[0, onset].item())
    # The onset−1 (read-out) column is preserved and correctly shifted.
    assert int(ids_local[0, onset_local - 1].item()) == sentinel
    assert int(ids_local[0, onset_local - 1].item()) == int(ids[0, onset - 1].item())


def test_truncate_to_window_argmax_unchanged_when_window_covers_causal_span():
    """(c) On the de-confounded stub, cropping a long donor to a window W that
    COVERS the causal span leaves the de-confounded argmax (the candidate causal
    layer) unchanged — and the cropped tensor is short (≤ W + span)."""
    d, vocab, n_layers, planted, read_tok = 8, 16, 8, 4, 5
    model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=n_layers,
                                   peak_layer=planted, read_tok=read_tok, seed=3))
    layers = list(range(n_layers))
    v = model.e0_direction
    directions = {L: v for L in layers}
    # A LONG chain: deep onset with a long pre-onset prefix that WILL be cropped.
    T, onset, span = 600, 500, 1
    ids = _ids_with_onset(read_tok, onset, T=T, vocab=vocab, seed=313)

    # Full-chain reference argmax (de-confounded).
    full = layer_steering_sweep_single(
        model, ids, onset_pos=onset, directions=directions, layers=layers,
        alpha=1.0, span_tokens=span, readout="kl", n_random=3,
        delta_frac=0.1, behaviour="backtracking")
    full_argmax = argmax_layer(aggregate_sweep_curves([full.score]))
    assert full_argmax == planted

    # The RegenLatch causal window is the onset−1 column itself (no cross-position
    # mixing); any W ≥ 1 covers it. Use a modest W ≪ onset so the prefix is really
    # cropped, then confirm the de-confounded argmax is unchanged.
    W = 64
    ids_local, onset_local = truncate_to_window(
        ids, onset, context_window=W, span_tokens=span)
    assert int(ids_local.shape[-1]) <= W + span
    assert onset_local == W                                   # lo = onset − W
    crop = layer_steering_sweep_single(
        model, ids_local, onset_pos=onset_local, directions=directions, layers=layers,
        alpha=1.0, span_tokens=span, readout="kl", n_random=3,
        delta_frac=0.1, behaviour="backtracking")
    crop_argmax = argmax_layer(aggregate_sweep_curves([crop.score]))
    assert crop_argmax == full_argmax == planted, (
        f"crop argmax {crop_argmax} should equal full argmax {full_argmax}")
    # Because the seed of the random null is keyed on (behaviour, layer, r) — NOT
    # on sequence length — and the onset column is identical, the de-confounded
    # scores match exactly here too (RegenLatch has no cross-position mixing).
    for L in layers:
        assert crop.score[L] == pytest.approx(full.score[L], abs=1e-12)


def test_truncate_to_window_via_sweep_for_donor_long_chain():
    """End-to-end through the runner's per-donor entry point: a stub 'donor' with a
    long chain is cropped by sweep_for_donor's --context-window path, recovering
    the planted layer on a short tensor. Exercises the onset relocation + remap
    exactly as 07d wires it (no GPU, no real model)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_p7d", str(Path(__file__).resolve().parents[1] / "07d_layer_steering_sweep.py"))
    p7d = importlib.util.module_from_spec(spec)
    # 07d imports the model loader at module import time; that import is light
    # (no model is constructed until called), so loading the module is safe here.
    spec.loader.exec_module(p7d)

    d, vocab, n_layers, planted, read_tok = 8, 16, 8, 4, 5
    model = _double(RegenLatchStub(d=d, vocab=vocab, n_layers=n_layers,
                                   peak_layer=planted, read_tok=read_tok, seed=4))
    layers = list(range(n_layers))
    v = model.e0_direction
    directions = {L: v for L in layers}
    T, onset, span, W = 500, 400, 1, 32

    # A fake tokenizer whose encode() returns a fixed long ids tensor with the
    # read token at the onset, and a fake onset locator returning `onset`. We
    # monkeypatch 07d's module-level `_sentence_token_onset` to that locator.
    ids_full = _ids_with_onset(read_tok, onset, T=T, vocab=vocab, seed=414)

    class _FakeTok:
        def encode(self, _text, return_tensors=None):
            return ids_full.clone()

    p7d._sentence_token_onset = lambda _tok, _text, _ann: onset
    chain = {"chain": "irrelevant text", "task_id": "t0"}
    ann = {"text": "behaviour sentence"}

    res = p7d.sweep_for_donor(
        model, _FakeTok(), torch.device("cpu"), chain, ann, directions, layers,
        1.0, False, span, behaviour="backtracking", readout="kl",
        n_random=3, delta_frac=0.1, context_window=W)
    assert res is not None
    # The cropped tensor is short and the onset was remapped to W (lo = onset − W).
    assert res.onset_pos == W
    assert argmax_layer(aggregate_sweep_curves([res.score])) == planted
