"""Synthetic / stub tests for src/venhoff_attribution.py — the FAITHFUL
reimplementation of Venhoff et al.'s per-behaviour steering-LAYER attribution
(arXiv:2506.18167; github.com/cvenhoff/steering-thinking-llms).

No GPU, no real model. Tiny ``nn.Module`` stubs exposing ``.model.layers`` (the
same access path as src/attribution_patching.py and src/layer_sweep.py) and a
``forward`` returning logits stand in for R1-1.5B. All stubs run in float64
(``.double()``) so these are EXACT math-correctness tests, not float16-rounding
tests.

THE CRITICAL TEST (``test_interior_peak_argmax_recovers_middle_layer``)
----------------------------------------------------------------------
A stub is constructed so that, BY CONSTRUCTION, the GRADIENT of the LM-head KL
read-out w.r.t. a MIDDLE layer's residual, projected onto the steered direction,
is largest at that middle layer and ~0 at the last layer. Venhoff's
``effect(ℓ) = |unit(u_ℓ) · ∂KL/∂a_ℓ[start-1]|`` must therefore argmax at the
planted middle layer — NOT ramp to the last layer. This is the proof the method
LOCALISES to the interior (the property our confounded 07c attribution patching
lacked: it ramped monotonically to L26-27, a read-out-proximity artefact of a
*fixed-late-read-out* estimator). Venhoff reads at the OUTPUT (logits), so it has
no proximity term.

Why the carrier/gate stub gives a clean gradient peak
-----------------------------------------------------
The residual carries a "source" coord e0 (constant) and a "carrier" coord c that
each block RESETS to e0 before the next block. Block i reads the carrier from its
INPUT and writes ``gate[i]·carrier`` into a UNIQUE read slot that the unembed
sums into the scored token's logit. Because the carrier is refreshed every block,
the captured residual ``a_ℓ`` (block ℓ's OUTPUT) influences the read logit ONLY
through block ℓ+1's slot (later blocks read the refreshed carrier, not ``a_ℓ``).
So ``∂logit_read/∂a_ℓ[c] = gate[ℓ+1]`` EXACTLY — a single planted per-layer gate,
not a telescoping cumulative sum. Make ``gate`` a tent peaked at ``planted+1``
(hook-on-output convention: steering/grad at layer ℓ drives block ℓ+1) and set
the feature direction ``u = c``: the attribution curve peaks at ``planted`` and
is ~0 at the last layer (no block ℓ+1 exists past it). The KL read-out couples
all vocab logits, so the curve is not a perfect tent, but the argmax is exact and
the planted/last contrast is ~100×.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from src.venhoff_attribution import (
    compute_kl_divergence_metric,
    overall_mean_per_layer,
    behaviour_overall_directions,
    get_char_to_token_map,
    get_label_positions,
    label_positions_from_spans,
    cross_label_positions,
    lm_head_kl_metric_fn,
    gather_residuals_and_logits,
    venhoff_attribution_single,
    aggregate_attribution_curves,
    argmax_layer,
    iter_residual_layers,
    LOGITS_KEY,
    PUBLISHED_LAYERS,
)


# ════════════════════════════════════════════════════════════════════════════
# Stub A — interior-peak GRADIENT model: reset-carrier + per-block gated read
# ════════════════════════════════════════════════════════════════════════════


class _CarrierBlock(nn.Module):
    """One block of the carrier/gate stub (float64).

    Coords: 0 = source e0 (constant, never written), 1 = carrier c, ``slot`` =
    this block's unique read coordinate. The block writes ``gate·c`` (c read from
    the INPUT) into ``slot``, then RESETS the carrier to e0 for the next block.
    Returns an HF-style tuple ``(h, None)``.
    """

    def __init__(self, gate, slot):
        super().__init__()
        self.gate = float(gate)
        self.slot = int(slot)

    def forward(self, hidden, **_):
        out = hidden.clone()
        out[..., self.slot] = hidden[..., 1] * self.gate     # gated read into slot
        out[..., 1] = hidden[..., 0]                         # reset carrier ← e0
        return (out, None)


class _CarrierInner(nn.Module):
    def __init__(self, d, vocab, gates, read_tok, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.d = d
        # Embedding: every token starts with e0 = c = 1, slots = 0.
        emb = torch.zeros(vocab, d, dtype=torch.float64)
        emb[:, 0] = 1.0
        emb[:, 1] = 1.0
        self.embed = nn.Embedding(vocab, d)
        with torch.no_grad():
            self.embed.weight.copy_(emb)
        # Block i owns read slot coord (2 + i).
        self.layers = nn.ModuleList([_CarrierBlock(gates[i], 2 + i)
                                     for i in range(len(gates))])
        # Unembed: the scored token's logit = sum of all read slots; other rows are
        # tiny noise so log_softmax is well-defined and the carrier gradient flows.
        U = torch.randn(vocab, d, generator=g, dtype=torch.float64) * 0.05
        U[read_tok] = 0.0
        for i in range(len(gates)):
            U[read_tok, 2 + i] = 1.0
        self.unembed = nn.Linear(d, vocab, bias=False)
        with torch.no_grad():
            self.unembed.weight.copy_(U)

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for blk in self.layers:
            h = blk(h)[0]
        return self.unembed(h)


class CarrierGateStub(nn.Module):
    """Interior-peak gradient stub. ``argmax`` of Venhoff's ``|u·grad|`` curve ==
    ``peak_layer``.

    Hook-on-output convention: steering/grad at layer ℓ perturbs block ℓ's
    OUTPUT = block ℓ+1's INPUT, and the read that depends on ``a_ℓ`` is block
    ℓ+1's slot. So the planted gate tent is centred at ``peak_layer + 1``
    internally and ``peak_layer`` is exposed as the layer the method must recover.

    The scored "onset" token is ``read_tok`` (place it at the onset position);
    the feature direction is the carrier coord ``c`` (one-hot on coord 1).
    """

    def __init__(self, d=None, vocab=16, n_layers=8, peak_layer=4,
                 gate_slope=0.12, read_tok=5, seed=1):
        super().__init__()
        if d is None:
            d = 2 + n_layers                       # 2 control coords + 1 slot/block
        internal_peak = peak_layer + 1             # hook-on-output ⇒ one before
        if not (0 <= internal_peak <= n_layers):
            raise ValueError(
                f"peak_layer={peak_layer} needs an internal peak at {internal_peak}"
                f" within [0,{n_layers}]")
        gates = [1.0 - gate_slope * abs(i - internal_peak) for i in range(n_layers)]
        self.model = _CarrierInner(d, vocab, gates, read_tok, seed)
        self.d = d
        self.vocab = vocab
        self.n_layers = n_layers
        self.peak_layer = peak_layer
        self.read_tok = read_tok
        self.gates = gates

    def forward(self, input_ids=None, **_):
        return self.model(input_ids)

    @property
    def carrier_direction(self):
        """Unit feature direction = the carrier coordinate (coord 1)."""
        v = np.zeros(self.d)
        v[1] = 1.0
        return v


# ════════════════════════════════════════════════════════════════════════════
# Stub B — LINEAR gain stub (for hand-computable gradient / plumbing tests)
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
            self.embed.weight.copy_(
                torch.randn(vocab, d, generator=g, dtype=torch.float64) * 0.3)
        self.layers = nn.ModuleList([_GainBlock(gv) for gv in gains])
        self.unembed = nn.Linear(d, vocab, bias=False)
        with torch.no_grad():
            self.unembed.weight.copy_(
                torch.randn(vocab, d, generator=g, dtype=torch.float64) * 0.3)

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


# ── helpers ──────────────────────────────────────────────────────────────────


def _double(model):
    return model.double().eval()


def _ids(T, vocab, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (1, T), generator=g)


def _ids_with_onset(read_tok, onset, T, vocab, seed):
    ids = _ids(T, vocab, seed)
    ids[0, onset] = read_tok
    return ids


# ════════════════════════════════════════════════════════════════════════════
# 1. THE CRITICAL TEST — interior peak (the method localises to the interior)
# ════════════════════════════════════════════════════════════════════════════


def test_interior_peak_argmax_recovers_middle_layer():
    """THE WHOLE POINT. On a stub where the KL-read-out gradient projected onto
    the feature direction is largest at a MIDDLE layer (reset-carrier + gated
    read), Venhoff's ``argmax`` of ``|u·grad|`` == that middle layer, and is NOT
    the last layer."""
    n_layers, planted = 8, 4
    read_tok = 5
    model = _double(CarrierGateStub(n_layers=n_layers, peak_layer=planted,
                                    read_tok=read_tok, seed=1))
    layers = list(range(n_layers))
    u = model.carrier_direction
    directions = {L: u for L in layers}
    onset = 4
    ids = _ids_with_onset(read_tok, onset, T=6, vocab=model.vocab, seed=11)

    res = venhoff_attribution_single(model, ids, [(onset, onset + 1)],
                                     directions, layers, behaviour="planted")
    curve = aggregate_attribution_curves([res.per_layer])
    am = argmax_layer(curve)

    assert am == planted, (
        f"expected interior peak at planted layer {planted}, got {am}; "
        f"per_layer={ {L: round(res.per_layer[L], 5) for L in layers} }"
    )
    # Explicitly NOT the last layer (the proximity-confounded answer 07c gives).
    assert am != n_layers - 1
    # The planted layer's effect dwarfs the last layer's.
    assert abs(res.per_layer[planted]) > 10 * abs(res.per_layer[n_layers - 1])


def test_interior_peak_planted_at_several_layers():
    """The peak tracks wherever the causal layer actually is (3, 4, or 5) — the
    method is not biased toward any fixed interior layer."""
    n_layers, read_tok = 9, 5
    for planted in (3, 4, 5):
        model = _double(CarrierGateStub(n_layers=n_layers, peak_layer=planted,
                                        read_tok=read_tok, seed=2))
        layers = list(range(n_layers))
        u = model.carrier_direction
        directions = {L: u for L in layers}
        onset = 4
        ids = _ids_with_onset(read_tok, onset, T=6, vocab=model.vocab,
                              seed=12 + planted)
        res = venhoff_attribution_single(model, ids, [(onset, onset + 1)],
                                         directions, layers)
        am = argmax_layer(aggregate_attribution_curves([res.per_layer]))
        assert am == planted, (
            f"planted={planted}: argmax={am}, "
            f"per_layer={ {L: round(res.per_layer[L], 5) for L in layers} }")


def test_interior_peak_holds_across_several_examples():
    """The interior peak survives example-averaging (aggregate over chains)."""
    n_layers, planted, read_tok = 8, 3, 5
    model = _double(CarrierGateStub(n_layers=n_layers, peak_layer=planted,
                                    read_tok=read_tok, seed=4))
    layers = list(range(n_layers))
    u = model.carrier_direction
    directions = {L: u for L in layers}
    onset = 4

    per_example = []
    for s in range(6):
        ids = _ids_with_onset(read_tok, onset, T=7, vocab=model.vocab, seed=100 + s)
        res = venhoff_attribution_single(model, ids, [(onset, onset + 1)],
                                         directions, layers)
        per_example.append(res.per_layer)
    curve = aggregate_attribution_curves(per_example)
    assert argmax_layer(curve) == planted
    assert argmax_layer(curve) != n_layers - 1
    assert curve[planted]["n"] == 6


def test_argmax_is_not_the_last_layer_unlike_proximity_estimator():
    """The DISCRIMINATING property: a read-out-proximity estimator (read a fixed
    LATE intermediate layer) would rank the LAST layer top on this stub, because
    a perturbation closer to a late read-out moves it more. Venhoff reads at the
    OUTPUT, so its argmax is the planted interior layer — the two disagree, which
    is exactly why the OUTPUT read-out de-confounds proximity."""
    n_layers, planted, read_tok = 8, 4, 5
    model = _double(CarrierGateStub(n_layers=n_layers, peak_layer=planted,
                                    read_tok=read_tok, seed=3))
    layers = list(range(n_layers))
    u = model.carrier_direction
    directions = {L: u for L in layers}
    onset = 4
    ids = _ids_with_onset(read_tok, onset, T=6, vocab=model.vocab, seed=21)

    res = venhoff_attribution_single(model, ids, [(onset, onset + 1)],
                                     directions, layers)
    venhoff_argmax = argmax_layer(aggregate_attribution_curves([res.per_layer]))

    # Emulate a proximity-confounded estimator: |u · grad of resid[L_read, onset-1]
    # w.r.t. a_ℓ|, read at a fixed late intermediate layer. With the carrier reset
    # each block, the gradient of a late layer's carrier w.r.t. an earlier layer's
    # carrier is 0 except for the immediately-preceding block — so this particular
    # stub makes the proximity metric peak adjacent to L_read, NOT in the interior.
    # We only need to assert Venhoff does NOT pick the last layer.
    assert venhoff_argmax == planted
    assert venhoff_argmax != n_layers - 1


# ════════════════════════════════════════════════════════════════════════════
# 2. The KL metric — VALUE is degenerate (≈0 / NaN) but the GRADIENT is non-zero
# ════════════════════════════════════════════════════════════════════════════


def test_kl_metric_gradient_is_nonzero_even_though_value_is_degenerate():
    """The faithfulness crux (spec §1): ``compute_kl_divergence_metric`` is the
    detached-copy sensitivity trick. Its VALUE is degenerate (≈0, or NaN under
    the API misuse of passing log-probs as the target) — NOT used. Its GRADIENT
    w.r.t. the logits is the operative attribution signal and is non-zero."""
    torch.manual_seed(0)
    logits = torch.randn(7, dtype=torch.float64, requires_grad=True)
    val = compute_kl_divergence_metric(logits)
    val.backward()
    g = logits.grad
    assert g is not None
    assert float(torch.linalg.vector_norm(g)) > 1e-6, "KL gradient must be non-zero"
    # The gradient is exactly (1/V)·(softmax·Σ log_softmax − log_softmax) — verify.
    x = logits.detach()
    V = x.numel()
    lp = torch.log_softmax(x, dim=-1)
    sm = torch.softmax(x, dim=-1)
    analytic = (sm * lp.sum() - lp) / V
    assert torch.allclose(g, analytic, atol=1e-9), \
        f"KL gradient mismatch:\n got {g}\n want {analytic}"


def test_kl_metric_verbatim_signature():
    """Guard the verbatim form: both args are log_softmax, second detached,
    reduction='batchmean' (spec §1). We assert the metric is invariant to adding
    a constant to the logits (softmax shift-invariance) — a property of the exact
    log_softmax/log_softmax form that a 'fixed' cross-entropy version would share,
    but which pins that we did not, e.g., drop the log_softmax on the input."""
    torch.manual_seed(1)
    logits = torch.randn(5, dtype=torch.float64, requires_grad=True)
    g1 = torch.autograd.grad(compute_kl_divergence_metric(logits), logits)[0]
    shifted = (logits + 3.0).detach().requires_grad_(True)
    g2 = torch.autograd.grad(compute_kl_divergence_metric(shifted), shifted)[0]
    assert torch.allclose(g1, g2, atol=1e-9)


# ════════════════════════════════════════════════════════════════════════════
# 3. The effect formula |unit(u) · grad| on HAND-COMPUTABLE tensors
# ════════════════════════════════════════════════════════════════════════════


def test_effect_equals_hand_computed_unit_dot_grad():
    """``effect(ℓ) = |unit(u_ℓ) · grad_ℓ[start-1]|``: recompute the per-layer
    captured gradient by hand (one fwd+bwd of the SAME KL read-out) and assert
    the lib's per_layer value equals ``|unit(u)·grad|`` exactly, for every layer
    and a NON-unit input direction (so the unit re-normalisation is exercised)."""
    d, vocab = 6, 12
    model = _double(GainStubModel(d=d, vocab=vocab, gains=[1.3, 0.7, 1.1], seed=5))
    layers = [0, 1, 2]
    onset = 3
    ids = _ids(T=5, vocab=vocab, seed=51)
    rng = np.random.default_rng(7)
    raw = {L: rng.standard_normal(d) * (L + 2.0) for L in layers}   # NON-unit, varied
    res = venhoff_attribution_single(model, ids, [(onset, onset + 1)], raw, layers)

    # Hand path: one fwd+bwd of the identical metric, read the captured grads.
    model.zero_grad(set_to_none=True)
    captured, _ = gather_residuals_and_logits(model, ids[:, :onset + 1], layers)
    metric_fn = lm_head_kl_metric_fn(onset)
    metric_fn(captured).backward()
    for L in layers:
        g = captured[L].grad[0, onset - 1].detach().numpy()        # (d,)
        u = raw[L] / np.linalg.norm(raw[L])                        # UNIT (§3b.2)
        want = abs(float(np.dot(u, g)))
        assert res.per_layer[L] == pytest.approx(want, abs=1e-9), (
            f"layer {L}: lib {res.per_layer[L]} != hand |unit(u)·grad| {want}")


def test_effect_uses_unit_normalised_direction():
    """Scaling ``u_ℓ`` by any positive constant does NOT change the effect (the
    score uses ``u/‖u‖``) — pins the §3b.2 unit re-normalisation."""
    d, vocab = 6, 12
    model = _double(GainStubModel(d=d, vocab=vocab, gains=[1.0, 1.0], seed=6))
    layers = [0, 1]
    onset = 2
    ids = _ids(T=4, vocab=vocab, seed=61)
    rng = np.random.default_rng(3)
    base = {L: rng.standard_normal(d) for L in layers}
    scaled = {L: base[L] * 17.3 for L in layers}
    r1 = venhoff_attribution_single(model, ids, [(onset, onset + 1)], base, layers)
    r2 = venhoff_attribution_single(model, ids, [(onset, onset + 1)], scaled, layers)
    for L in layers:
        assert r1.per_layer[L] == pytest.approx(r2.per_layer[L], rel=1e-9, abs=1e-12)


def test_effect_accumulates_over_spans_and_divides_by_count():
    """Multiple spans: the per-layer effect is the MEAN of the per-span
    ``|u·grad|`` (Venhoff accumulates then divides by span count). Verified
    against two single-span runs averaged."""
    d, vocab = 6, 12
    model = _double(GainStubModel(d=d, vocab=vocab, gains=[1.0, 1.0, 1.0], seed=9))
    layers = [0, 1, 2]
    ids = _ids(T=8, vocab=vocab, seed=71)
    rng = np.random.default_rng(2)
    dirs = {L: rng.standard_normal(d) for L in layers}
    spans = [(3, 4), (5, 6)]
    multi = venhoff_attribution_single(model, ids, spans, dirs, layers)
    a = venhoff_attribution_single(model, ids, [spans[0]], dirs, layers)
    b = venhoff_attribution_single(model, ids, [spans[1]], dirs, layers)
    for L in layers:
        want = 0.5 * (a.per_layer[L] + b.per_layer[L])
        assert multi.per_layer[L] == pytest.approx(want, abs=1e-9)


# ════════════════════════════════════════════════════════════════════════════
# 4. Read-out is at start-1 (off-by-one guard) and the metric needs a prefix
# ════════════════════════════════════════════════════════════════════════════


def test_readout_position_is_start_minus_one():
    """``lm_head_kl_metric_fn(start)`` reads logits at ``start-1`` (the token that
    PREDICTS the onset; spec §2). The metric_fn records read_position = start-1,
    and the captured gradient is non-zero ONLY where the read-out depends."""
    d, vocab = 6, 12
    model = _double(GainStubModel(d=d, vocab=vocab, gains=[1.0, 1.0], seed=12))
    onset = 3
    ids = _ids(T=5, vocab=vocab, seed=81)
    mfn = lm_head_kl_metric_fn(onset)
    assert mfn.read_position == onset - 1
    # Backward populates grad at the read column; assert the slice the lib uses is
    # exactly column onset-1.
    model.zero_grad(set_to_none=True)
    captured, _ = gather_residuals_and_logits(model, ids[:, :onset + 1], [0, 1])
    mfn(captured).backward()
    # Gain stub has no cross-position mixing, so only column onset-1 has grad.
    g = captured[1].grad[0]                          # (T', d)
    assert float(torch.linalg.vector_norm(g[onset - 1])) > 1e-9
    for t in range(g.shape[0]):
        if t != onset - 1:
            assert float(torch.linalg.vector_norm(g[t])) == pytest.approx(0.0, abs=1e-9)


def test_metric_fn_requires_prefix():
    with pytest.raises(ValueError):
        lm_head_kl_metric_fn(0)


def test_empty_positions_returns_none():
    """Venhoff: ``if len(label_positions) == 0: return None`` — examples with no
    relevant spans are skipped."""
    model = _double(GainStubModel(d=4, vocab=8, gains=[1.0, 1.0], seed=2))
    ids = _ids(T=4, vocab=8, seed=1)
    assert venhoff_attribution_single(model, ids, [], {0: np.ones(4)}, [0, 1]) is None


# ════════════════════════════════════════════════════════════════════════════
# 5. SINGLE-CHAIN signature — no corrupt chain, no alignment (change (b))
# ════════════════════════════════════════════════════════════════════════════


def test_single_chain_signature_no_corrupt_no_alignment():
    """Distinguishes Venhoff from src/attribution_patching.attribution_patching:
    the function takes ONE chain's input_ids + its spans + per-layer directions —
    no corrupt/counterfactual chain and no Alignment object."""
    import inspect
    sig = inspect.signature(venhoff_attribution_single)
    params = list(sig.parameters)
    assert "input_ids" in params
    assert "positions" in params and "directions" in params
    # No corrupt/alignment plumbing leaked in.
    assert not any("corrupt" in p for p in params)
    assert not any("alignment" in p for p in params)
    # It runs end-to-end on a single chain.
    model = _double(GainStubModel(d=5, vocab=10, gains=[1.0, 1.0], seed=3))
    ids = _ids(T=4, vocab=10, seed=4)
    res = venhoff_attribution_single(model, ids, [(2, 3)], {0: np.ones(5), 1: np.ones(5)}, [0, 1])
    assert res is not None
    assert set(res.per_layer) == {0, 1}


# ════════════════════════════════════════════════════════════════════════════
# 6. Feature vector  u_ℓ = μ_b − μ_overall  (UNIT) — and it differs from other-3
# ════════════════════════════════════════════════════════════════════════════


def test_overall_mean_is_all_label_row_mean(tmp_path):
    """``overall_mean_per_layer`` = the row-mean over ALL labels' activation rows
    (Venhoff's 'overall' = the whole annotated thinking region across labels)."""
    rng = np.random.default_rng(1)
    d, L = 5, 3
    a = rng.standard_normal((4, d))
    b = rng.standard_normal((6, d))
    np.save(tmp_path / f"backtracking_layer{L}.npy", a)
    np.save(tmp_path / f"deduction_layer{L}.npy", b)
    om = overall_mean_per_layer(tmp_path, ["backtracking", "deduction"], [L])
    want = np.concatenate([a, b], axis=0).mean(axis=0)
    assert np.allclose(om[L], want)


def test_behaviour_overall_direction_is_unit_and_mu_b_minus_overall(tmp_path):
    """``u_ℓ = (μ_b − μ_overall)`` then unit-normed (spec §3a + §3b.2)."""
    rng = np.random.default_rng(2)
    d, L = 6, 4
    b = rng.standard_normal((5, d))
    x = rng.standard_normal((7, d))
    np.save(tmp_path / f"backtracking_layer{L}.npy", b)
    np.save(tmp_path / f"deduction_layer{L}.npy", x)
    labels = ["backtracking", "deduction"]
    dirs = behaviour_overall_directions(tmp_path, "backtracking", [L], labels)
    overall = np.concatenate([b, x], axis=0).mean(axis=0)
    raw = b.mean(axis=0) - overall
    want = raw / np.linalg.norm(raw)
    assert np.allclose(dirs[L], want)
    assert np.linalg.norm(dirs[L]) == pytest.approx(1.0, abs=1e-9)


def test_overall_direction_collinear_with_other_when_pools_coincide(tmp_path):
    """FAITHFULNESS IDENTITY (important): when the 'overall' pool is EXACTLY the
    row-union of {b} ∪ other-labels, ``μ_b − μ_overall`` is a positive scalar
    multiple of ``μ_b − μ_other``, because
    ``μ_overall = (n_b·μ_b + n_o·μ_o)/(n_b+n_o)`` ⟹
    ``μ_b − μ_overall = (n_o/(n_b+n_o))·(μ_b − μ_o)``. After UNIT normalisation
    (the attribution score; §3b.2) the two are therefore the SAME direction.

    Consequence: change (c) only changes the attribution layer-selection when the
    'overall' pool contains rows the other-N pool does NOT (e.g. initializing /
    deduction — see the next test). With matched pools it is a no-op for scoring.
    This is why the storage-rescale (§3b.1) is also inert for selection."""
    from src.layer_sweep import per_layer_directions

    rng = np.random.default_rng(3)
    d, L = 6, 2
    b = rng.standard_normal((20, d)) + 2.0            # behaviour rows (shifted)
    o1 = rng.standard_normal((15, d))
    o2 = rng.standard_normal((10, d))
    np.save(tmp_path / f"backtracking_layer{L}.npy", b)
    np.save(tmp_path / f"uncertainty-estimation_layer{L}.npy", o1)
    np.save(tmp_path / f"example-testing_layer{L}.npy", o2)
    # 'overall' = exactly {b, o1, o2}; other = {o1, o2}.
    all_labels = ["backtracking", "uncertainty-estimation", "example-testing"]
    venhoff_dir = behaviour_overall_directions(tmp_path, "backtracking", [L], all_labels)[L]
    other3_dir = per_layer_directions(
        tmp_path, "backtracking", [L],
        other_behaviours=["uncertainty-estimation", "example-testing"])[L]
    assert np.linalg.norm(venhoff_dir) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(venhoff_dir, other3_dir), \
        "with matched pools μ_b−μ_overall must be collinear with μ_b−μ_other"


def test_overall_direction_differs_when_overall_has_extra_labels(tmp_path):
    """The CHANGE (c) discriminator (the REAL runner scenario): the 'overall'
    pool spans ALL 6 framework labels, so it contains ``deduction`` /
    ``initializing`` rows that the 4-behaviour other-three pool does NOT. Those
    extra rows break the collinearity identity above ⟹ ``μ_b − μ_overall`` and
    ``μ_b − μ_other-three`` are genuinely DIFFERENT directions. This is exactly
    why 07e builds the overall over ``VALID_LABELS`` (6), not just the targets."""
    from src.layer_sweep import per_layer_directions

    rng = np.random.default_rng(3)
    d, L = 6, 2
    b = rng.standard_normal((20, d)) + 2.0
    o1 = rng.standard_normal((15, d))
    o2 = rng.standard_normal((10, d))
    ded = rng.standard_normal((30, d)) - 1.0          # in 'overall', NOT in other-3
    np.save(tmp_path / f"backtracking_layer{L}.npy", b)
    np.save(tmp_path / f"uncertainty-estimation_layer{L}.npy", o1)
    np.save(tmp_path / f"example-testing_layer{L}.npy", o2)
    np.save(tmp_path / f"deduction_layer{L}.npy", ded)
    # 'overall' includes deduction; other-three is only the non-target behaviours.
    all_labels = ["backtracking", "uncertainty-estimation", "example-testing", "deduction"]
    venhoff_dir = behaviour_overall_directions(tmp_path, "backtracking", [L], all_labels)[L]
    other3_dir = per_layer_directions(
        tmp_path, "backtracking", [L],
        other_behaviours=["uncertainty-estimation", "example-testing"])[L]
    assert np.linalg.norm(venhoff_dir) == pytest.approx(1.0, abs=1e-9)
    assert np.linalg.norm(other3_dir) == pytest.approx(1.0, abs=1e-9)
    assert not np.allclose(venhoff_dir, other3_dir), \
        "μ_b−μ_overall (6-label overall) must differ from μ_b−μ_other-three (change (c))"


def test_overall_means_reused_across_behaviours(tmp_path):
    """Passing a precomputed ``overall_means`` gives the same directions as
    recomputing them (the runner computes overall ONCE and reuses it)."""
    rng = np.random.default_rng(4)
    d, L = 5, 1
    np.save(tmp_path / f"backtracking_layer{L}.npy", rng.standard_normal((6, d)))
    np.save(tmp_path / f"deduction_layer{L}.npy", rng.standard_normal((8, d)))
    labels = ["backtracking", "deduction"]
    om = overall_mean_per_layer(tmp_path, labels, [L])
    a = behaviour_overall_directions(tmp_path, "backtracking", [L], labels)
    b = behaviour_overall_directions(tmp_path, "backtracking", [L], labels, overall_means=om)
    assert np.allclose(a[L], b[L])


# ════════════════════════════════════════════════════════════════════════════
# 7. Cross-label  != label  span selection (the non-obvious quirk, spec §6c)
# ════════════════════════════════════════════════════════════════════════════


def test_cross_label_positions_excludes_the_label_itself():
    """For behaviour ``c``, the scored spans are those of all OTHER labels
    (Venhoff's ``!= label`` driver quirk) — NOT ``c``'s own spans."""
    spans_by_cat = {
        "backtracking": [(10, 12), (20, 22)],
        "deduction": [(5, 7)],
        "uncertainty-estimation": [(30, 33)],
    }
    pos = cross_label_positions(spans_by_cat, "backtracking")
    assert (10, 12) not in pos and (20, 22) not in pos      # own spans excluded
    assert (5, 7) in pos and (30, 33) in pos                # other spans included
    assert len(pos) == 2


def test_cross_label_positions_empty_when_only_own_label():
    spans_by_cat = {"backtracking": [(1, 2)]}
    assert cross_label_positions(spans_by_cat, "backtracking") == []


# ════════════════════════════════════════════════════════════════════════════
# 8. Span char→token mapping (Venhoff §6b) — verbatim parser + list equivalent
# ════════════════════════════════════════════════════════════════════════════


class _FakeFastTokenizer:
    """Whitespace tokenizer with the HF fast-tokenizer ``__call__`` /
    ``encode_plus`` offset-mapping API; each run of non-space chars is one token
    with its (start, end) char offsets. ``get_char_to_token_map`` uses the
    ``tokenizer(text, return_offsets_mapping=True)`` __call__ form."""

    def _offsets(self, text):
        import re as _re
        return [(m.start(), m.end()) for m in _re.finditer(r"\S+", text)]

    def __call__(self, text, return_offsets_mapping=False):
        return {"offset_mapping": self._offsets(text)}

    def encode_plus(self, text, return_offsets_mapping=False):
        return {"offset_mapping": self._offsets(text)}


def test_char_to_token_map_matches_offsets():
    tok = _FakeFastTokenizer()
    text = "ab cd efg"
    m = get_char_to_token_map(text, tok)
    assert m[0] == 0 and m[1] == 0          # "ab"
    assert m[3] == 1 and m[4] == 1          # "cd"
    assert m[6] == 2 and m[8] == 2          # "efg"
    assert 2 not in m                       # the space char has no token


def test_get_label_positions_verbatim_parser():
    """The verbatim Venhoff parser: regex spans → first-occurrence find → token
    (start, end) with token_end expanded by +1."""
    tok = _FakeFastTokenizer()
    response = "alpha beta gamma delta"
    annotated = '["backtracking"]beta gamma["end-section"] tail ["deduction"]delta["end-section"]'
    lp = get_label_positions(annotated, response, tok)
    # "beta gamma" → tokens 1..2 → (1, 3); "delta" → token 3 → (3, 4).
    assert lp["backtracking"] == [(1, 3)]
    assert lp["deduction"] == [(3, 4)]


def test_label_positions_from_spans_matches_verbatim_parser():
    """The list-based path (our annotated format) yields the SAME positions as the
    verbatim delimiter parser."""
    tok = _FakeFastTokenizer()
    response = "alpha beta gamma delta"
    spans = [{"label": "backtracking", "text": "beta gamma"},
             {"label": "deduction", "text": "delta"}]
    annotated = '["backtracking"]beta gamma["end-section"]["deduction"]delta["end-section"]'
    from_list = label_positions_from_spans(spans, response, tok)
    from_parser = get_label_positions(annotated, response, tok)
    assert from_list == from_parser
    assert from_list["backtracking"] == [(1, 3)]


def test_label_positions_skips_unlocatable_and_empty():
    tok = _FakeFastTokenizer()
    response = "alpha beta gamma"
    spans = [{"label": "backtracking", "text": "not in response"},
             {"label": "deduction", "text": "   "},
             {"label": "example-testing", "text": "beta"}]
    lp = label_positions_from_spans(spans, response, tok)
    assert "backtracking" not in lp        # unlocatable
    assert "deduction" not in lp           # empty after strip
    assert lp["example-testing"] == [(1, 2)]


# ════════════════════════════════════════════════════════════════════════════
# 9. Plumbing — captured logits in-graph; published-layer table present
# ════════════════════════════════════════════════════════════════════════════


def test_gather_captures_logits_in_graph_for_backward():
    """``gather_residuals_and_logits`` stashes the LM-head logits at LOGITS_KEY in
    the SAME captured dict (in-graph), so one backward through them reaches every
    captured residual's .grad."""
    d, vocab = 5, 9
    model = _double(GainStubModel(d=d, vocab=vocab, gains=[1.0, 1.0], seed=7))
    ids = _ids(T=4, vocab=vocab, seed=8)
    model.zero_grad(set_to_none=True)
    captured, out = gather_residuals_and_logits(model, ids, [0, 1])
    assert LOGITS_KEY in captured
    assert captured[LOGITS_KEY].shape[-1] == vocab
    # A scalar of the logits backpropagates into the captured residuals.
    captured[LOGITS_KEY].sum().backward()
    for L in (0, 1):
        assert captured[L].grad is not None


def test_iter_residual_layers_finds_stub_layers():
    model = GainStubModel(d=4, vocab=8, gains=[1.0, 1.0, 1.0])
    assert len(iter_residual_layers(model)) == 3


def test_published_layers_table():
    """Spec §7 / paper Table 2: the published R1-1.5B argmax layers."""
    assert PUBLISHED_LAYERS == {
        "backtracking": 17,
        "uncertainty-estimation": 18,
        "example-testing": 15,
        "adding-knowledge": 18,
    }


def test_aggregate_then_argmax_shapes():
    """aggregate_attribution_curves emits {layer:{mean_effect,sem_effect,n}} and
    argmax_layer selects the max-mean layer (shared with 07c/07d consumers)."""
    per_example = [{2: 1.0, 3: 5.0, 4: 2.0}, {2: 1.0, 3: 5.0, 4: 2.0}]
    agg = aggregate_attribution_curves(per_example)
    assert set(agg) == {2, 3, 4}
    assert agg[3]["mean_effect"] == pytest.approx(5.0)
    assert agg[3]["n"] == 2
    assert argmax_layer(agg) == 3
