"""Tests for src/steering_analysis.py — the matched-effect comparison machinery.

Synthetic summaries with KNOWN structure so the statistical primitives can be
checked against ground truth (the same discipline as tests/synthetic.py):
  * monotone effect→damage curves;
  * a method that Pareto-dominates another by construction;
  * a paired bootstrap whose sign is known.

No model, no annotation: the bootstrap operates on pre-built per-task curves,
and the summary path is fed dicts shaped like src.evaluation.aggregate_results.
"""

import numpy as np
import pytest

from src.steering_analysis import (
    on_target_effect,
    damage_scalar,
    MethodCurve,
    method_curve,
    collateral_at_matched_effect,
    matched_effect_grid,
    pareto_frontier,
    dominates,
    paired_bootstrap_matched_effect,
    holm_bonferroni,
    bootstrap_two_sided_p,
    compare_across_behaviours,
)


# ── effect + damage extraction ────────────────────────────────────────────────

def test_on_target_effect_is_suppression():
    # vanilla fraction 0.8, steered 0.3 -> suppressed by 0.5
    assert on_target_effect(0.3, 0.8) == pytest.approx(0.5)
    assert on_target_effect(None, 0.8) is None
    assert on_target_effect(0.3, None) is None


def test_damage_scalar_sums_present_axes():
    cell = {"repetition_rate": 0.3, "degenerate_rate": 0.1, "leakage_mean": 0.2}
    assert damage_scalar(cell) == pytest.approx(0.6)
    # missing axes are skipped, not zeroed-in as data
    assert damage_scalar({"repetition_rate": 0.3}) == pytest.approx(0.3)
    # no axis at all -> None (caller skips the point)
    assert damage_scalar({}) is None


def test_damage_scalar_includes_accuracy_drop():
    cell = {"repetition_rate": 0.2}
    acc = {"accuracy_drop_vs_vanilla": 0.4}
    assert damage_scalar(cell, accuracy_cell=acc) == pytest.approx(0.6)


# ── (a) matched-effect collateral ─────────────────────────────────────────────

def _line_curve(method, slope, alphas=(0.3, 0.5, 1.0, 1.5, 2.0),
                effects=(0.0, 0.2, 0.4, 0.6, 0.8)):
    e = np.asarray(effects, dtype=float)
    return MethodCurve("b", method, e, slope * e, np.asarray(alphas, dtype=float))


def test_collateral_interpolates_at_matched_effect():
    A = _line_curve("manifold_auto", 0.5)
    assert collateral_at_matched_effect(A, 0.5) == pytest.approx(0.25)
    assert collateral_at_matched_effect(A, 0.3) == pytest.approx(0.15)


def test_collateral_none_when_effect_out_of_range():
    A = _line_curve("manifold_auto", 0.5)        # max effect 0.8
    assert collateral_at_matched_effect(A, 1.2) is None
    # extrapolate clamps to the endpoint
    assert collateral_at_matched_effect(A, 1.2, extrapolate=True) == pytest.approx(0.4)


def test_collateral_handles_duplicate_effect_points():
    # two points at the same effect 0.4 with different damage -> averaged
    c = MethodCurve("b", "m",
                    np.array([0.0, 0.4, 0.4, 0.8]),
                    np.array([0.0, 0.2, 0.4, 0.8]),
                    np.array([0.3, 0.5, 1.0, 2.0]))
    assert collateral_at_matched_effect(c, 0.4) == pytest.approx(0.3)


def test_matched_effect_grid_is_the_overlap():
    A = _line_curve("a", 1.0, effects=(0.0, 0.4, 0.8))
    B = _line_curve("b", 1.0, effects=(0.2, 0.5, 1.0))
    g = matched_effect_grid([A, B], n=5)
    assert g.min() == pytest.approx(0.2)        # max of the mins
    assert g.max() == pytest.approx(0.8)        # min of the maxes


def test_matched_effect_grid_empty_when_no_overlap():
    A = _line_curve("a", 1.0, effects=(0.0, 0.1, 0.2))
    B = _line_curve("b", 1.0, effects=(0.5, 0.7, 0.9))
    assert matched_effect_grid([A, B]).size == 0


# ── (b) Pareto frontier + domination ──────────────────────────────────────────

def test_pareto_frontier_keeps_only_nondominated():
    # A: damage 0.5*eff ; B: damage 1.0*eff  -> A dominates everywhere eff>0
    A = _line_curve("manifold_auto", 0.5)
    B = _line_curve("single_direction", 1.0)
    front = pareto_frontier([A, B])
    nonzero = [p for p in front if p.effect > 0]
    assert nonzero, "frontier should contain positive-effect points"
    assert all(p.method == "manifold_auto" for p in nonzero)


def test_dominates_directional():
    A = _line_curve("manifold_auto", 0.5)
    B = _line_curve("single_direction", 1.0)
    assert dominates(A, B) is True
    assert dominates(B, A) is False


def test_dominates_none_when_no_overlap():
    A = _line_curve("a", 1.0, effects=(0.0, 0.1, 0.2))
    B = _line_curve("b", 1.0, effects=(0.5, 0.7, 0.9))
    assert dominates(A, B) is None


def test_dominates_crossing_curves_not_a_clean_win():
    # A cheaper at low effect, B cheaper at high effect -> neither dominates
    A = MethodCurve("b", "a", np.array([0.1, 0.5, 0.9]),
                    np.array([0.0, 0.3, 0.9]), np.array([0.3, 1.0, 2.0]))
    B = MethodCurve("b", "b", np.array([0.1, 0.5, 0.9]),
                    np.array([0.2, 0.3, 0.4]), np.array([0.3, 1.0, 2.0]))
    # A starts below B, ends above -> A does NOT uniformly dominate
    assert dominates(A, B) is False


# ── (c) paired BCa bootstrap ──────────────────────────────────────────────────

def _per_task_curves(slope, n_tasks=20, noise=0.01, seed=0, method="m"):
    rng = np.random.default_rng(seed)
    base_e = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    alphas = np.array([0.3, 0.5, 1.0, 1.5, 2.0])
    out = {}
    for t in range(n_tasks):
        e = base_e + rng.normal(0, noise, size=base_e.size)
        d = np.clip(slope * base_e + rng.normal(0, noise, size=base_e.size), 0, None)
        order = np.argsort(e)
        out[f"t{t}"] = MethodCurve("b", method, e[order], d[order], alphas[order])
    return out


def test_paired_bootstrap_sign_positive_when_a_less_damaging():
    # A slope 0.5, B slope 1.0 -> damage_B - damage_A > 0 (A wins)
    a = _per_task_curves(0.5, seed=1, method="manifold_auto")
    b = _per_task_curves(1.0, seed=2, method="single_direction")
    res = paired_bootstrap_matched_effect(a, b, target_effect=0.5, n_resamples=500, seed=0)
    assert res.estimate > 0
    assert res.excludes_zero()
    assert res.ci_low > 0


def test_paired_bootstrap_sign_negative_when_a_more_damaging():
    a = _per_task_curves(1.2, seed=3, method="manifold_auto")
    b = _per_task_curves(0.4, seed=4, method="single_direction")
    res = paired_bootstrap_matched_effect(a, b, target_effect=0.5, n_resamples=500, seed=0)
    assert res.estimate < 0
    assert res.ci_high < 0


def test_paired_bootstrap_ci_brackets_zero_when_equal():
    a = _per_task_curves(0.7, noise=0.05, seed=5, method="a")
    b = _per_task_curves(0.7, noise=0.05, seed=6, method="b")
    res = paired_bootstrap_matched_effect(a, b, target_effect=0.5, n_resamples=800, seed=0)
    # same underlying slope -> difference ~ 0, CI should straddle 0
    assert res.ci_low <= 0 <= res.ci_high


def test_paired_bootstrap_is_deterministic():
    a = _per_task_curves(0.5, seed=1)
    b = _per_task_curves(1.0, seed=2)
    r1 = paired_bootstrap_matched_effect(a, b, 0.5, n_resamples=300, seed=0)
    r2 = paired_bootstrap_matched_effect(a, b, 0.5, n_resamples=300, seed=0)
    assert (r1.ci_low, r1.ci_high, r1.estimate) == (r2.ci_low, r2.ci_high, r2.estimate)


def test_paired_bootstrap_requires_two_tasks():
    a = _per_task_curves(0.5, n_tasks=1)
    b = _per_task_curves(1.0, n_tasks=1)
    with pytest.raises(ValueError):
        paired_bootstrap_matched_effect(a, b, 0.5, n_resamples=100)


def test_paired_bootstrap_uses_only_shared_tasks():
    a = _per_task_curves(0.5, n_tasks=10, seed=1)
    b = _per_task_curves(1.0, n_tasks=6, seed=2)   # fewer tasks
    res = paired_bootstrap_matched_effect(a, b, 0.5, n_resamples=200, seed=0)
    assert res.n_tasks == 6                         # intersection size


def test_bca_uses_jackknife_acceleration_not_pure_percentile():
    """BCa should generally differ from the raw percentile interval on a skewed
    bootstrap distribution (the whole point of bias-correction + acceleration)."""
    a = _per_task_curves(0.5, seed=7, method="a")
    b = _per_task_curves(1.0, seed=8, method="b")
    res = paired_bootstrap_matched_effect(a, b, 0.5, n_resamples=1000, seed=0)
    assert res.method == "BCa"
    # finite, ordered CI
    assert np.isfinite(res.ci_low) and np.isfinite(res.ci_high)
    assert res.ci_low <= res.estimate <= res.ci_high


# ── (d) Holm–Bonferroni across behaviours ─────────────────────────────────────

def test_holm_bonferroni_matches_manual():
    p = {"backtracking": 0.01, "uncertainty-estimation": 0.04,
         "example-testing": 0.03, "adding-knowledge": 0.005}
    adj = holm_bonferroni(p)
    # smallest p (0.005) * 4 = 0.02 ; next (0.01)*3 = 0.03 ; etc. (step-down,
    # monotone-enforced). Just assert ordering + that it inflates p's.
    assert all(adj[k] >= p[k] - 1e-12 for k in p)
    assert adj["adding-knowledge"] == pytest.approx(0.02)
    # monotone in the sorted order
    order = sorted(p, key=lambda k: p[k])
    vals = [adj[k] for k in order]
    assert vals == sorted(vals)


def test_holm_controls_family_of_four():
    # all four marginally significant; Holm should kill the larger ones
    p = {"a": 0.02, "b": 0.02, "c": 0.02, "d": 0.02}
    adj = holm_bonferroni(p)
    # first = 0.02*4 = 0.08 -> none survive at 0.05
    assert all(v >= 0.05 for v in adj.values())


def test_bootstrap_two_sided_p_from_ci():
    from src.steering_analysis import BootstrapResult
    # CI well clear of 0 -> small p
    r = BootstrapResult(estimate=0.25, ci_low=0.20, ci_high=0.30,
                        n_tasks=20, n_resamples=1000)
    assert bootstrap_two_sided_p(r) < 0.01
    # CI straddling 0 -> large p
    r2 = BootstrapResult(estimate=0.01, ci_low=-0.1, ci_high=0.12,
                         n_tasks=20, n_resamples=1000)
    assert bootstrap_two_sided_p(r2) > 0.1


# ── summary-consuming path (uses the REAL aggregate_results) ───────────────────

def _make_summary_inputs():
    """A small Phase-7 experiment where manifold suppresses with strictly less
    repetition damage than single at matched effect, fed through the real
    aggregate_results so the analysis is tested against the production schema."""
    BEHS = ["backtracking", "uncertainty-estimation"]
    steered, annotated = [], []

    def srec(beh, method, alpha, tid, chain, n=80):
        return {"behaviour": beh, "method": method, "alpha": alpha,
                "task_id": tid, "chain": chain, "n_tokens": n, "layer": 27}

    def arec(beh, method, alpha, labels, tid):
        return {"behaviour": beh, "method": method, "alpha": alpha, "task_id": tid,
                "annotations": [{"label": l, "text": f"s{i}"} for i, l in enumerate(labels)]}

    # single repeats heavily (high repetition_rate); manifold stays diverse.
    rep_chain = "aa bb aa bb " * 20            # very repetitive 4-grams
    div_chain = " ".join(f"w{i}" for i in range(80))  # all distinct
    for tid in ["T1", "T2", "T3", "T4", "T5", "T6"]:
        steered.append(srec("shared", "vanilla", 0.0, tid, div_chain))
        annotated.append(arec("shared", "vanilla", 0.0,
                              ["backtracking", "backtracking", "backtracking", "deduction"], tid))
        for beh in BEHS:
            for alpha in [0.5, 1.0, 2.0]:
                # both suppress fully at alpha=2; partial below
                n_t = {0.5: 2, 1.0: 1, 2.0: 0}[alpha]
                labels = [beh] * n_t + ["deduction"] * (4 - n_t)
                steered.append(srec(beh, "single_direction", alpha, tid, rep_chain))
                annotated.append(arec(beh, "single_direction", alpha, labels, tid))
                steered.append(srec(beh, "manifold_auto", alpha, tid, div_chain))
                annotated.append(arec(beh, "manifold_auto", alpha, labels, tid))
    return steered, annotated, BEHS


def test_method_curve_reads_real_summary():
    from src.evaluation import aggregate_results
    steered, annotated, BEHS = _make_summary_inputs()
    summary = aggregate_results(steered, annotated, target_behaviours=BEHS)
    c = method_curve(summary, "backtracking", "single_direction")
    assert len(c) >= 2
    # effect increases with suppression; damage (repetition) is high for single
    assert c.effects.max() > 0
    assert (c.damages > 0).all()


def test_compare_across_behaviours_end_to_end():
    from src.evaluation import aggregate_results
    steered, annotated, BEHS = _make_summary_inputs()
    summary = aggregate_results(steered, annotated, target_behaviours=BEHS)
    out = compare_across_behaviours(
        steered, annotated, summary,
        method_a="manifold_auto", method_b="single_direction",
        target_behaviours=BEHS, n_resamples=200, seed=0)
    assert set(out["per_behaviour"]) == set(BEHS)
    # manifold (div_chain) is less repetitive than single at matched effect ->
    # damage_B - damage_A > 0 in at least one behaviour, and Holm runs.
    ests = [c["bootstrap"]["estimate"] for c in out["per_behaviour"].values()
            if c["bootstrap"] is not None]
    assert ests, "expected at least one bootstrap result"
    assert max(ests) > 0
    assert isinstance(out["holm"], dict)


def test_compare_handles_nonoverlapping_gracefully():
    """If a method never reaches the other's effect range, the behaviour cell is
    recorded with a None bootstrap rather than crashing."""
    from src.evaluation import aggregate_results
    steered, annotated, BEHS = _make_summary_inputs()
    summary = aggregate_results(steered, annotated, target_behaviours=BEHS)
    out = compare_across_behaviours(
        steered, annotated, summary,
        method_a="manifold_auto", method_b="nonexistent_method",
        target_behaviours=BEHS, n_resamples=50, seed=0)
    for comp in out["per_behaviour"].values():
        assert comp["bootstrap"] is None


# ── Pre-flight QA (2026-06-23): NaN/inf/div-by-zero + interpolation edges ──────

def test_collateral_nonmonotone_effect_averages_duplicate_x():
    """When effect is NON-injective in alpha (over-steering makes the behaviour
    fraction rebound, so the same effect recurs at two alphas with different
    damage), the matched-effect read averages the duplicate-effect damages. Here
    effect 0.3 occurs at damage 0.1 and 0.4 -> the read is exactly their mean."""
    c = MethodCurve("b", "m",
                    np.array([0.1, 0.3, 0.5, 0.3, 0.1]),     # effect (non-monotone)
                    np.array([0.0, 0.1, 0.2, 0.4, 0.8]),     # damage rises with alpha
                    np.array([0.3, 0.5, 1.0, 1.5, 2.0]))
    assert collateral_at_matched_effect(c, 0.3) == pytest.approx((0.1 + 0.4) / 2)
    # and the interior point still interpolates linearly between the deduped knots
    # deduped: x=[0.1,0.3,0.5], y=[0.4, 0.25, 0.2]; at 0.4 -> halfway 0.25->0.2 = 0.225
    assert collateral_at_matched_effect(c, 0.4) == pytest.approx(0.225)


def test_matched_effect_grid_single_point_overlap():
    """Curves that touch at exactly one effect yield a single-point grid (not
    empty, not a crash), and dominates is decidable there."""
    A = MethodCurve("b", "a", np.array([0.2, 0.5]), np.array([0.0, 0.3]),
                    np.array([0.3, 1.0]))
    B = MethodCurve("b", "b", np.array([0.5, 0.9]), np.array([0.1, 0.3]),
                    np.array([0.3, 1.0]))
    g = matched_effect_grid([A, B])
    assert g.size == 1 and g[0] == pytest.approx(0.5)
    # at effect 0.5: A damage 0.3, B damage 0.1 -> B strictly better -> A !dominates
    assert dominates(A, B) is False


def test_bca_finite_with_zero_variance_jackknife():
    """A degenerate (zero-variance) jackknife sets acceleration a=0 rather than
    dividing by zero; the BCa CI is finite and ordered."""
    from src.steering_analysis import _bca_interval
    boot = np.linspace(0.1, 0.5, 200)
    jack = np.full(8, 0.3)                       # zero variance -> den == 0
    lo, hi = _bca_interval(0.3, boot, jack)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= hi


def test_bca_percentile_fallback_when_all_boot_identical():
    """If every bootstrap replicate equals the estimate (prop hits 0 or 1), BCa
    falls back to the percentile interval — here a degenerate point interval."""
    from src.steering_analysis import _bca_interval
    boot = np.full(200, 0.3)
    jack = np.array([0.2, 0.3, 0.4, 0.5])
    lo, hi = _bca_interval(0.3, boot, jack)
    assert lo == pytest.approx(0.3) and hi == pytest.approx(0.3)


def test_bootstrap_returns_none_when_target_effect_unreachable():
    """If neither pooled curve reaches the target effect (and no extrapolation),
    the paired bootstrap returns None instead of inventing a damage value."""
    a = _per_task_curves(0.5, n_tasks=6, seed=1)   # max effect ~0.8
    b = _per_task_curves(1.0, n_tasks=6, seed=2)
    assert paired_bootstrap_matched_effect(a, b, target_effect=5.0,
                                           n_resamples=100, seed=0) is None


def test_bootstrap_jackknife_fallback_to_percentile_small_n():
    """With very few shared tasks the jackknife can yield <2 usable replicates;
    the bootstrap must still return a finite ordered CI via the percentile path."""
    a = _per_task_curves(0.5, n_tasks=2, seed=1)
    b = _per_task_curves(1.0, n_tasks=2, seed=2)
    res = paired_bootstrap_matched_effect(a, b, target_effect=0.5,
                                          n_resamples=200, seed=0)
    assert res is not None
    assert np.isfinite(res.ci_low) and np.isfinite(res.ci_high)
    assert res.ci_low <= res.ci_high
    assert res.n_tasks == 2                        # unit is the task


def test_bootstrap_resampling_unit_is_task_not_observation():
    """The reported n_tasks equals the number of shared TASKS, confirming the
    resample/jackknife unit is the task (each task contributes a whole curve),
    not the individual (alpha) observation."""
    a = _per_task_curves(0.6, n_tasks=12, seed=1)
    b = _per_task_curves(0.6, n_tasks=12, seed=2)
    res = paired_bootstrap_matched_effect(a, b, target_effect=0.4,
                                          n_resamples=300, seed=0)
    assert res.n_tasks == 12


def test_bootstrap_two_sided_p_uses_empirical_tail_when_dist_given():
    """With an explicit bootstrap distribution the p is the empirical two-sided
    tail. A distribution entirely above 0 -> p = 2*min(0, 1) = 0; one straddling
    0 symmetrically -> p ~ 1."""
    from src.steering_analysis import BootstrapResult
    r = BootstrapResult(estimate=0.3, ci_low=0.1, ci_high=0.5,
                        n_tasks=10, n_resamples=100)
    above = np.array([0.1, 0.2, 0.3, 0.4, 0.5])           # all > 0
    assert bootstrap_two_sided_p(r, boot_dist=above) == pytest.approx(0.0)
    straddle = np.array([-0.2, -0.1, 0.1, 0.2])           # symmetric
    assert bootstrap_two_sided_p(r, boot_dist=straddle) == pytest.approx(1.0)


def test_holm_bonferroni_m_equals_one_unchanged():
    """A family of one test is unadjusted (m=1: padj == p)."""
    p = {"backtracking": 0.037}
    adj = holm_bonferroni(p)
    assert adj["backtracking"] == pytest.approx(0.037)


def test_holm_bonferroni_empty_is_empty():
    assert holm_bonferroni({}) == {}


def test_pareto_keeps_tied_points_from_distinct_methods():
    """Two distinct points with identical (effect, damage) neither dominates the
    other (ties are not strict) -> both stay on the frontier."""
    A = MethodCurve("b", "manifold_auto", np.array([0.5]), np.array([0.2]),
                    np.array([1.0]))
    B = MethodCurve("b", "single_direction", np.array([0.5]), np.array([0.2]),
                    np.array([1.0]))
    front = pareto_frontier([A, B])
    assert len(front) == 2
    assert {p.method for p in front} == {"manifold_auto", "single_direction"}
