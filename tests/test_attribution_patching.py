"""Synthetic / stub tests for src/attribution_patching.py.

No GPU, no real model. A tiny `StubLayeredModel` (an nn.Module exposing
`.model.layers`, the same access path as src/steered_inference.py) stands in for
R1-1.5B on random tensors. Two things are pinned:

  1. **The behaviour metric (CF-10 fix)** — projection / subspace scores match a
     hand-computed reference; the metric is token-blind and commensurable; the
     positional-alignment helpers map positions correctly and the cross-chain
     comparison only ever uses aligned pairs.

  2. **The attribution math** — on a LINEAR stub, attribution patching
     (grad·(corrupt−clean)) equals the brute-force activation-patching reference
     EXACTLY (to numerical tolerance), because the metric is linear in the
     patched layer's residual when the network between layer and read-out is
     linear. On a NON-LINEAR stub it is only first-order accurate — the test
     asserts it stays close but is *not* exact, which is the documented caveat
     (and the reason a brute-force check on a few layers is advised).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from src.attribution_patching import (
    BehaviourMetric,
    behaviour_effect,
    Alignment,
    anchored_alignment,
    proportional_alignment,
    aligned_position_scores,
    attribution_patching,
    make_residual_metric_fn,
    brute_force_patch_effect,
    aggregate_attribution_curves,
    iter_residual_layers,
)


# ════════════════════════════════════════════════════════════════════════════
# Stub model: nn.Module exposing `.model.layers` (Qwen2-style tuple output)
# ════════════════════════════════════════════════════════════════════════════


class _StubBlock(nn.Module):
    """One decoder block. Residual update h ← act(W h + b).

    With ``nonlinear=False`` the update is affine (linear), so the whole stack
    is linear in any layer's residual and attribution patching is exact. With
    ``nonlinear=True`` a tanh is inserted, making the stack non-linear so
    attribution is only first-order accurate.
    """

    def __init__(self, d: int, seed: int, nonlinear: bool):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # Moderate weights: large enough that activations are O(0.1) (so the
        # linear-exactness test is non-vacuous and the tanh bends the
        # upstream→read-out path enough to give the first-order caveat test a
        # clear gap), small enough to stay out of tanh saturation/chaos.
        self.W = nn.Parameter(torch.randn(d, d, generator=g) * (0.5 / d ** 0.5))
        self.b = nn.Parameter(torch.randn(d, generator=g) * 0.15)
        self.nonlinear = nonlinear

    def forward(self, hidden, **_):
        h = hidden @ self.W.T + self.b
        if self.nonlinear:
            h = torch.tanh(h)
        # Mimic HF Qwen2: decoder block returns a tuple (hidden, ...extras).
        return (h, None)


class _Inner(nn.Module):
    def __init__(self, d, n_layers, vocab, seed, nonlinear):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(vocab, d)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(vocab, d, generator=g) * 0.5)
        self.layers = nn.ModuleList(
            [_StubBlock(d, seed + 100 + i, nonlinear) for i in range(n_layers)]
        )

    def forward(self, input_ids):
        h = self.embed(input_ids)            # (B, T, d)
        for blk in self.layers:
            h = blk(h)[0]
        return h


class StubLayeredModel(nn.Module):
    """Minimal stand-in exposing ``.model.layers`` like a HF causal LM.

    ``forward(input_ids=...)`` runs the embedding + every block so the registered
    forward hooks fire exactly as they would on the real model.
    """

    def __init__(self, d=8, n_layers=5, vocab=20, seed=0, nonlinear=False):
        super().__init__()
        self.model = _Inner(d, n_layers, vocab, seed, nonlinear)
        self.d = d
        self.n_layers = n_layers
        self.vocab = vocab

    def forward(self, input_ids=None, **_):
        return self.model(input_ids)


def _ids(T, vocab, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (1, T), generator=g)


def _unit(d, seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


# ════════════════════════════════════════════════════════════════════════════
# 1. Behaviour metric (CF-10 fix)
# ════════════════════════════════════════════════════════════════════════════


def test_projection_metric_matches_dot_product():
    r = _unit(8, 1)
    m = BehaviourMetric(direction=r, mode="projection", behaviour="backtracking")
    h = np.random.default_rng(2).standard_normal(8)
    assert m.score_np(h) == pytest.approx(float(h @ r))


def test_projection_metric_centering_subtracts_first():
    r = _unit(6, 3)
    c = np.arange(6, dtype=float)
    m = BehaviourMetric(direction=r, mode="projection", center=c)
    h = np.random.default_rng(4).standard_normal(6)
    assert m.score_np(h) == pytest.approx(float((h - c) @ r))


def test_subspace_metric_is_projection_norm():
    # Orthonormal 2-D basis in R^5.
    Q, _ = np.linalg.qr(np.random.default_rng(5).standard_normal((5, 2)))
    V = Q.T  # (2, 5), orthonormal rows
    m = BehaviourMetric(subspace=V, mode="subspace")
    h = np.random.default_rng(6).standard_normal(5)
    assert m.score_np(h) == pytest.approx(float(np.linalg.norm(V @ h)))


def test_subspace_score_is_zero_off_subspace():
    # A vector orthogonal to the subspace scores 0; one inside scores its norm.
    # Full QR of a (6,6) matrix gives a complete orthonormal basis of R^6.
    Q, _ = np.linalg.qr(np.random.default_rng(7).standard_normal((6, 6)))
    V = Q[:, :2].T                      # rows span a 2-D subspace
    perp = Q[:, 2]                      # orthogonal to that subspace
    m = BehaviourMetric(subspace=V, mode="subspace")
    assert m.score_np(perp) == pytest.approx(0.0, abs=1e-10)
    inside = V[0]
    assert m.score_np(inside) == pytest.approx(1.0, abs=1e-8)


def test_behaviour_effect_token_blind_cf10():
    """The CF-10 point: the score is a pure function of activations — it never
    sees tokens, so identical activations give an identical score regardless of
    any (absent) lexical content."""
    r = _unit(8, 8)
    m = BehaviourMetric(direction=r, mode="projection")
    h = np.random.default_rng(9).standard_normal(8)
    s1 = behaviour_effect(h, "backtracking", layer=27, metric=m)
    s2 = behaviour_effect(h.copy(), "backtracking", layer=27, metric=m)
    assert s1 == s2 == pytest.approx(float(h @ r))


def test_behaviour_effect_block_reduces_over_positions():
    r = _unit(4, 10)
    m = BehaviourMetric(direction=r, mode="projection")
    block = np.random.default_rng(11).standard_normal((3, 4))   # (T, d)
    got = behaviour_effect(block, "b", 5, metric=m)
    want = float(np.mean([row @ r for row in block]))
    assert got == pytest.approx(want)


def test_behaviour_effect_selects_layer_specific_metric():
    """The direction is layer-specific; behaviour_effect must pick the right one
    from a {(behaviour, layer): metric} dict."""
    r10 = _unit(5, 1)
    r27 = _unit(5, 2)
    metrics = {
        ("backtracking", 10): BehaviourMetric(direction=r10, mode="projection"),
        ("backtracking", 27): BehaviourMetric(direction=r27, mode="projection"),
    }
    h = np.random.default_rng(3).standard_normal(5)
    assert behaviour_effect(h, "backtracking", 10, metrics) == pytest.approx(float(h @ r10))
    assert behaviour_effect(h, "backtracking", 27, metrics) == pytest.approx(float(h @ r27))


def test_behaviour_effect_commensurable_across_behaviours():
    """Two behaviours scored by the SAME functional form (unit-vector
    projection) are on a common scale — the property the lexical marker proxy
    lacked (marker sets differed in size/base-rate)."""
    h = np.random.default_rng(12).standard_normal(8)
    m_a = BehaviourMetric(direction=_unit(8, 21), mode="projection")
    m_b = BehaviourMetric(direction=_unit(8, 22), mode="projection")
    sa = behaviour_effect(h, "a", 1, m_a)
    sb = behaviour_effect(h, "b", 1, m_b)
    # Both are bounded by |h| (Cauchy–Schwarz with a unit direction): same units.
    nh = float(np.linalg.norm(h))
    assert abs(sa) <= nh + 1e-9 and abs(sb) <= nh + 1e-9


# ════════════════════════════════════════════════════════════════════════════
# 2. Positional alignment (CF-10b)
# ════════════════════════════════════════════════════════════════════════════


def test_anchored_alignment_single_pair():
    al = anchored_alignment(T_clean=10, T_corrupt=7, anchor_clean=6,
                            anchor_corrupt=3, window=0)
    assert al.pairs == [(6, 3)]
    assert al.strategy == "anchored"


def test_anchored_alignment_window_stays_in_step_and_clamps():
    al = anchored_alignment(T_clean=10, T_corrupt=10, anchor_clean=1,
                            anchor_corrupt=5, window=2)
    # d ranges -2..2 but clean anchor=1 clamps the -2 offset (would be -1).
    assert al.pairs == [(0, 4), (1, 5), (2, 6), (3, 7)]
    # Each pair keeps anchor_corrupt - anchor_clean = 4 offset.
    assert all((tk - tc) == 4 for tc, tk in al.pairs)


def test_anchored_alignment_rejects_out_of_range_anchor():
    with pytest.raises(ValueError):
        anchored_alignment(5, 5, anchor_clean=5, anchor_corrupt=0)


def test_proportional_alignment_maps_endpoints_and_midpoint():
    al = proportional_alignment(T_clean=5, T_corrupt=11)
    d = dict(al.pairs)
    assert d[0] == 0          # start -> start
    assert d[4] == 10         # end -> end
    assert d[2] == 5          # midpoint -> midpoint
    assert al.strategy == "proportional"


def test_aligned_scores_compare_only_aligned_partners():
    """Position i of clean is compared to its aligned partner i' of corrupt —
    never naively to corrupt position i. We make the two chains' position-i
    residuals deliberately different and check the alignment, not the index,
    drives the pairing."""
    r = _unit(4, 30)
    m = BehaviourMetric(direction=r, mode="projection")
    clean = np.random.default_rng(31).standard_normal((6, 4))
    corrupt = np.random.default_rng(32).standard_normal((4, 4))
    al = anchored_alignment(6, 4, anchor_clean=5, anchor_corrupt=3, window=1)
    out = aligned_position_scores(clean, corrupt, "b", 1, m, al)
    # Hand-check each returned pair uses (tc, tk) from the alignment.
    for (tc, tk), (sc, sk) in zip(al.pairs, out):
        assert sc == pytest.approx(float(clean[tc] @ r))
        assert sk == pytest.approx(float(corrupt[tk] @ r))


# ════════════════════════════════════════════════════════════════════════════
# 3. Attribution math vs brute force — EXACT on a linear stub
# ════════════════════════════════════════════════════════════════════════════


def _run_pair(model, metric, layers, clean, corrupt, read_layer, read_pos,
              alignment=None):
    metric_fn = make_residual_metric_fn(metric, read_layer=read_layer,
                                        read_position=read_pos)
    attr = attribution_patching(model, clean, corrupt, metric_fn, layers,
                                alignment=alignment, behaviour="backtracking")
    brute = brute_force_patch_effect(model, clean, corrupt, metric, layers,
                                     alignment=alignment, read_layer=read_layer,
                                     read_position=read_pos)
    return attr, brute


def test_attribution_equals_brute_force_on_linear_stub():
    """The core math identity. On a linear stub, for every (layer, position):
        grad_clean · (act_corrupt − act_clean)  ==  metric(patched) − metric(clean)
    exactly (the Taylor remainder is zero for a linear map)."""
    d, T = 8, 6
    model = StubLayeredModel(d=d, n_layers=5, vocab=20, seed=1, nonlinear=False)
    model.eval()
    metric = BehaviourMetric(direction=_unit(d, 7), mode="projection")
    layers = [0, 1, 2, 3, 4]
    read_layer, read_pos = 4, T - 1
    clean = _ids(T, 20, 11)
    corrupt = _ids(T, 20, 12)

    attr, brute = _run_pair(model, metric, layers, clean, corrupt,
                            read_layer, read_pos)

    for L in layers:
        for t in attr.per_position[L]:
            assert attr.per_position[L][t] == pytest.approx(brute[L][t], abs=1e-5), \
                f"layer {L} pos {t}: attr {attr.per_position[L][t]} vs brute {brute[L][t]}"


def test_attribution_approximates_brute_force_subspace_metric():
    """The subspace metric ‖V h‖ is a NORM — non-linear in h — so unlike the
    projection metric, attribution is only FIRST-ORDER for it even on a linear
    stub (the Taylor remainder of a norm is nonzero for a finite swap). Assert
    the estimates track brute force (correct sign-ish, same scale), not that
    they're exact. (The projection metric is the one with the exact identity;
    use it when an exact necessity check matters.)"""
    d, T = 8, 5
    model = StubLayeredModel(d=d, n_layers=4, vocab=16, seed=3, nonlinear=False)
    model.eval()
    Q, _ = np.linalg.qr(np.random.default_rng(40).standard_normal((d, d)))
    metric = BehaviourMetric(subspace=Q[:, :3].T, mode="subspace")
    layers = [0, 1, 2, 3]
    read_layer, read_pos = 3, T - 1
    clean = _ids(T, 16, 21)
    corrupt = _ids(T, 16, 22)

    attr, brute = _run_pair(model, metric, layers, clean, corrupt,
                            read_layer, read_pos)
    av = np.array([attr.per_position[L][t] for L in layers for t in attr.per_position[L]])
    bv = np.array([brute[L][t] for L in layers for t in brute[L]])
    # Same order of magnitude (a usable screen), and positively correlated.
    assert np.max(np.abs(av)) > 0          # non-vacuous
    assert np.max(np.abs(av - bv)) < 3.0 * (np.max(np.abs(bv)) + 1e-9)
    if np.std(av) > 1e-9 and np.std(bv) > 1e-9:
        assert np.corrcoef(av, bv)[0, 1] > 0.5


def test_attribution_exact_at_readout_layer_itself():
    """Patching the read-out layer at the read-out position: even with a
    non-linear stack the *metric* is read directly off that layer's residual,
    so attribution at (read_layer, read_pos) is exact regardless of upstream
    non-linearity (the metric is linear in its own layer's residual)."""
    d, T = 6, 4
    model = StubLayeredModel(d=d, n_layers=4, vocab=12, seed=5, nonlinear=True)
    model.eval()
    metric = BehaviourMetric(direction=_unit(d, 9), mode="projection")
    layers = [0, 1, 2, 3]
    read_layer, read_pos = 3, T - 1
    clean = _ids(T, 12, 31)
    corrupt = _ids(T, 12, 32)
    attr, brute = _run_pair(model, metric, layers, clean, corrupt,
                            read_layer, read_pos)
    # At the read-out layer/position the swap is linear in the metric → exact.
    assert attr.per_position[read_layer][read_pos] == pytest.approx(
        brute[read_layer][read_pos], abs=1e-5)


def test_attribution_respects_alignment_in_both_estimators():
    """With a non-identity alignment, both estimators must use the SAME aligned
    corrupt positions, so they still agree exactly on a linear stub."""
    d, Tc, Tk = 8, 6, 9
    model = StubLayeredModel(d=d, n_layers=4, vocab=20, seed=7, nonlinear=False)
    model.eval()
    metric = BehaviourMetric(direction=_unit(d, 13), mode="projection")
    layers = [0, 1, 2, 3]
    read_layer, read_pos = 3, Tc - 1
    clean = _ids(Tc, 20, 41)
    corrupt = _ids(Tk, 20, 42)
    al = anchored_alignment(Tc, Tk, anchor_clean=Tc - 1, anchor_corrupt=Tk - 1,
                            window=2)
    attr, brute = _run_pair(model, metric, layers, clean, corrupt,
                            read_layer, read_pos, alignment=al)
    for L in layers:
        for t in attr.per_position[L]:
            assert attr.per_position[L][t] == pytest.approx(brute[L][t], abs=1e-5)


# ════════════════════════════════════════════════════════════════════════════
# 4. First-order caveat: approximate (not exact) on a non-linear stub
# ════════════════════════════════════════════════════════════════════════════


def test_attribution_is_first_order_approximate_on_nonlinear_stub():
    """On a non-linear stack, an UPSTREAM patch reaches the read-out through a
    tanh, so attribution ≠ brute force exactly. It should stay in the right
    ballpark (correlated, same order of magnitude) but show a nonzero gap — this
    is the documented reason to confirm the chosen layer with a brute-force
    check."""
    d, T = 8, 6
    model = StubLayeredModel(d=d, n_layers=5, vocab=20, seed=2, nonlinear=True)
    model.eval()
    metric = BehaviourMetric(direction=_unit(d, 17), mode="projection")
    layers = [0, 1, 2, 3, 4]
    read_layer, read_pos = 4, T - 1
    clean = _ids(T, 20, 51)
    corrupt = _ids(T, 20, 52)
    attr, brute = _run_pair(model, metric, layers, clean, corrupt,
                            read_layer, read_pos)

    # Collect the most-upstream layer's estimates (longest non-linear path).
    a0 = np.array([attr.per_position[0][t] for t in attr.per_position[0]])
    b0 = np.array([brute[0][t] for t in brute[0]])
    # Not exact (some gap exists somewhere across the upstream layers)...
    all_attr = np.array([attr.per_position[L][t]
                         for L in layers for t in attr.per_position[L]])
    all_brute = np.array([brute[L][t] for L in layers for t in brute[L]])
    max_gap = np.max(np.abs(all_attr - all_brute))
    assert max_gap > 1e-4, "expected a first-order gap on the non-linear stub"
    # ...but the same order of magnitude (a usable screen, not noise).
    denom = np.max(np.abs(all_brute)) + 1e-9
    assert max_gap < 5.0 * denom


# ════════════════════════════════════════════════════════════════════════════
# 5. Per-layer reduction, downstream-zero, and aggregation
# ════════════════════════════════════════════════════════════════════════════


def test_per_layer_is_sum_abs_over_positions():
    d, T = 6, 4
    model = StubLayeredModel(d=d, n_layers=4, vocab=12, seed=4, nonlinear=False)
    model.eval()
    metric = BehaviourMetric(direction=_unit(d, 19), mode="projection")
    layers = [0, 1, 2, 3]
    metric_fn = make_residual_metric_fn(metric, read_layer=3, read_position=T - 1)
    clean, corrupt = _ids(T, 12, 61), _ids(T, 12, 62)
    attr = attribution_patching(model, clean, corrupt, metric_fn, layers)
    for L in layers:
        want = sum(abs(v) for v in attr.per_position[L].values())
        assert attr.per_layer[L] == pytest.approx(want)


def test_downstream_of_readout_is_zero():
    """A layer downstream of the read-out cannot influence it → zero gradient →
    zero attribution, exactly."""
    d, T = 6, 4
    model = StubLayeredModel(d=d, n_layers=5, vocab=12, seed=6, nonlinear=True)
    model.eval()
    metric = BehaviourMetric(direction=_unit(d, 23), mode="projection")
    # Read out at layer 2; sweep includes downstream layers 3, 4.
    layers = [0, 1, 2, 3, 4]
    read_layer = 2
    metric_fn = make_residual_metric_fn(metric, read_layer=read_layer,
                                        read_position=T - 1)
    clean, corrupt = _ids(T, 12, 71), _ids(T, 12, 72)
    attr = attribution_patching(model, clean, corrupt, metric_fn, layers)
    assert attr.per_layer[3] == 0.0
    assert attr.per_layer[4] == 0.0
    # And an upstream layer is (generically) nonzero.
    assert attr.per_layer[0] != 0.0


def test_aggregate_attribution_curves_shape_matches_consumer():
    """aggregate_attribution_curves emits {layer: {mean_effect, sem_effect, n}},
    the exact shape compute_layer_triangulation.load_phase7b_curves reads."""
    per_pair = [
        {0: 1.0, 1: 2.0, 27: 3.0},
        {0: 3.0, 1: 4.0, 27: 5.0},
        {0: 2.0, 1: 3.0, 27: 4.0},
    ]
    agg = aggregate_attribution_curves(per_pair)
    assert set(agg.keys()) == {0, 1, 27}
    assert agg[0]["mean_effect"] == pytest.approx(2.0)
    assert agg[0]["n"] == 3
    assert "sem_effect" in agg[0]
    # mean_effect is the field the triangulation consumer pulls.
    assert agg[27]["mean_effect"] == pytest.approx(4.0)


def test_aggregate_handles_nan_and_missing_layers():
    per_pair = [{0: 1.0}, {0: float("nan"), 1: 2.0}, {1: 4.0}]
    agg = aggregate_attribution_curves(per_pair)
    assert agg[0]["n"] == 1 and agg[0]["mean_effect"] == pytest.approx(1.0)
    assert agg[1]["n"] == 2 and agg[1]["mean_effect"] == pytest.approx(3.0)


def test_iter_residual_layers_finds_stub_layers():
    model = StubLayeredModel(d=4, n_layers=3)
    assert len(iter_residual_layers(model)) == 3


def test_empty_aggregate_is_empty():
    assert aggregate_attribution_curves([]) == {}
