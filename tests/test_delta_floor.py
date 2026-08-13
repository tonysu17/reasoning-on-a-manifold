"""Tests for src/delta_floor.py — the Phase-7 Δ_floor headline analysis.

Covers: arm→floor pairing, per-task fraction pooling (samples/reps/suffixes),
the paired BCa bootstrap, the three corroborator tests (sign, McNemar exact,
transitions), the fraction-RMS noise band, the PASS gate, end-to-end headline on
a planted signal vs a null, and the vanilla-cancels-in-Δ_floor invariant.
"""
import numpy as np
import pytest

from src.delta_floor import (
    floor_for_arm, per_task_fraction, paired_bootstrap_mean, empirical_two_sided_p,
    matched_pair_transitions, sign_test, mcnemar_exact,
    noise_band_fraction_rms, delta_floor_cell, delta_floor_headline,
    DeltaFloorCell,
)

SHARED = "shared"


# ── synthetic record builders ─────────────────────────────────────────────────

def _sent(labels):
    return [{"label": l} for l in labels]


def _labels_at_fraction(behaviour, frac, L=10, other="other"):
    n = int(round(frac * L))
    return [behaviour] * n + [other] * (L - n)


def _records(behaviour, method, alpha, fracs, *, prefix="t", reps=None, samples=None,
             rec_behaviour=None, L=10):
    """Build paired (steered, annotated) records, one base task per entry in *fracs*.

    reps/samples (lists) optionally fan each base task into multiple records with
    ``#rs{rep}``/``#s{j}`` suffixes (to test pooling). *rec_behaviour* overrides the
    record's behaviour field (vanilla uses "shared")."""
    rb = rec_behaviour or behaviour
    steered, annotated = [], []
    for i, frac in enumerate(fracs):
        bt = f"{prefix}{i}"
        variants = []
        if reps is not None:
            variants = [(f"{bt}#rs{r}", r) for r in reps]
        elif samples is not None:
            variants = [(f"{bt}#s{j}", None) for j in samples]
        else:
            variants = [(bt, None)]
        # If fanned, vary the fraction across variants so pooling has something to average.
        fr_list = (frac if isinstance(frac, (list, tuple)) else [frac] * len(variants))
        for (tid, rep), fr in zip(variants, fr_list):
            rec = {"task_id": tid, "base_task_id": bt, "behaviour": rb,
                   "method": method, "alpha": alpha, "chain": "w " * 60, "n_tokens": 120,
                   "subspace_replicate": rep}
            steered.append(rec)
            annotated.append({**rec, "annotations": _sent(_labels_at_fraction(behaviour, fr, L))})
    return steered, annotated


def _scenario(behaviour, n_tasks, van_frac, arm_frac, floor_frac, seed=0,
              arm="single_direction", floor="energy_matched_random", alpha=1.0):
    """Full (steered, annotated) for vanilla+arm+floor at deterministic jittered fractions."""
    rng = np.random.default_rng(seed)
    def jit(base):
        return [float(np.clip(base + d, 0, 1)) for d in rng.uniform(-0.05, 0.05, n_tasks)]
    s, a = [], []
    for meth, fr, rb in [("vanilla", jit(van_frac), SHARED),
                         (arm, jit(arm_frac), behaviour),
                         (floor, jit(floor_frac), behaviour)]:
        al = 0.0 if meth == "vanilla" else alpha
        ss, aa = _records(behaviour, meth, al, fr, rec_behaviour=rb)
        s += ss; a += aa
    return s, a


# ── floor_for_arm ─────────────────────────────────────────────────────────────

def test_floor_for_arm_single():
    assert floor_for_arm("single_direction") == "energy_matched_random"


@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_floor_for_arm_manifold_k(k):
    assert floor_for_arm(f"manifold_k{k}") == f"random_subspace_k{k}"


def test_floor_for_arm_manifold_auto():
    assert floor_for_arm("manifold_auto") == "random_subspace_kauto"


@pytest.mark.parametrize("m", ["vanilla", "energy_matched_random",
                               "random_subspace_k3", "orthogonal_complement", "junk"])
def test_floor_for_arm_non_arm_is_none(m):
    assert floor_for_arm(m) is None


# ── per_task_fraction ─────────────────────────────────────────────────────────

def test_per_task_fraction_basic():
    s, a = _records("backtracking", "single_direction", 1.0, [0.2, 0.4, 0.6])
    fr = per_task_fraction(s, a, "backtracking", "single_direction", 1.0)
    assert set(fr) == {"t0", "t1", "t2"}
    assert fr["t0"] == pytest.approx(0.2)
    assert fr["t2"] == pytest.approx(0.6)


def test_per_task_fraction_pools_samples():
    # one base task, three samples at 0.2/0.4/0.6 → mean 0.4
    s, a = _records("backtracking", "single_direction", 1.0,
                    [[0.2, 0.4, 0.6]], samples=[0, 1, 2])
    fr = per_task_fraction(s, a, "backtracking", "single_direction", 1.0)
    assert set(fr) == {"t0"}
    assert fr["t0"] == pytest.approx(0.4)


def test_per_task_fraction_pools_subspace_reps():
    s, a = _records("backtracking", "random_subspace_k3", 1.0,
                    [[0.3, 0.5, 0.7]], reps=[0, 1, 2])
    fr = per_task_fraction(s, a, "backtracking", "random_subspace_k3", 1.0)
    assert fr["t0"] == pytest.approx(0.5)


def test_per_task_fraction_strips_suffix_without_base_field():
    s, a = _records("backtracking", "single_direction", 1.0, [[0.4, 0.6]], samples=[0, 1])
    for r in s + a:
        r.pop("base_task_id")  # force suffix-stripping fallback
    fr = per_task_fraction(s, a, "backtracking", "single_direction", 1.0)
    assert set(fr) == {"t0"} and fr["t0"] == pytest.approx(0.5)


def test_per_task_fraction_skips_missing_and_empty():
    s, a = _records("backtracking", "single_direction", 1.0, [0.2, 0.4, 0.6])
    a[0]["annotations"] = []          # empty → skip
    del a[1]                          # missing → skip
    fr = per_task_fraction(s, a, "backtracking", "single_direction", 1.0)
    assert set(fr) == {"t2"}


def test_per_task_fraction_skips_partial_nonempty_annotation():
    s, a = _records("backtracking", "single_direction", 1.0, [0.2, 0.4])
    a[0]["annotation_complete"] = False
    assert a[0]["annotations"]  # partial payload exists but is not admissible
    fr = per_task_fraction(s, a, "backtracking", "single_direction", 1.0)
    assert set(fr) == {"t1"}


def test_per_task_fraction_alpha_filter():
    s, a = _records("backtracking", "single_direction", 1.0, [0.5])
    s2, a2 = _records("backtracking", "single_direction", 0.5, [0.1], prefix="u")
    fr = per_task_fraction(s + s2, a + a2, "backtracking", "single_direction", 1.0)
    assert set(fr) == {"t0"}


def test_per_task_fraction_vanilla_matches_shared_behaviour():
    s, a = _records("backtracking", "vanilla", 0.0, [0.5, 0.5], rec_behaviour=SHARED)
    fr = per_task_fraction(s, a, "backtracking", "vanilla", 0.0)
    assert set(fr) == {"t0", "t1"} and fr["t0"] == pytest.approx(0.5)


# ── paired_bootstrap_mean ─────────────────────────────────────────────────────

def test_bootstrap_estimate_is_sample_mean():
    vals = [0.1, 0.2, 0.3, 0.4]
    res = paired_bootstrap_mean(vals, n_resamples=500, seed=1)
    assert res.estimate == pytest.approx(np.mean(vals))
    assert res.n_tasks == 4


def test_bootstrap_clear_positive_excludes_zero():
    vals = list(0.25 + 0.05 * np.sin(np.arange(20)))  # mean ~0.25, bounded variance
    res = paired_bootstrap_mean(vals, n_resamples=2000, seed=2)
    assert res.estimate > 0
    assert res.excludes_zero()
    assert res.ci_low > 0


def test_bootstrap_symmetric_noise_includes_zero():
    vals = list(0.2 * np.sin(np.arange(1, 31)))  # mean ~0
    res = paired_bootstrap_mean(vals, n_resamples=2000, seed=3)
    assert not res.excludes_zero()
    assert res.ci_low < 0 < res.ci_high


def test_bootstrap_constant_degenerate_fallback():
    res = paired_bootstrap_mean([0.3] * 8, n_resamples=500, seed=4)
    assert res.estimate == pytest.approx(0.3)
    assert res.ci_low == pytest.approx(0.3) and res.ci_high == pytest.approx(0.3)


def test_bootstrap_needs_two_tasks():
    with pytest.raises(ValueError):
        paired_bootstrap_mean([0.3], n_resamples=100)


def test_bootstrap_return_distribution():
    res, dist = paired_bootstrap_mean([0.1, 0.2, 0.3, 0.4], n_resamples=500,
                                      seed=1, return_distribution=True)
    assert dist.shape[0] == res.n_resamples
    assert res.estimate == pytest.approx(np.mean([0.1, 0.2, 0.3, 0.4]))


# ── matched_pair_transitions ──────────────────────────────────────────────────

def test_transitions_counts():
    arm = {"a": 0.1, "b": 0.5, "c": 0.3, "d": 0.4}
    flr = {"a": 0.5, "b": 0.1, "c": 0.3, "e": 0.9}  # 'd' only in arm, 'e' only in floor
    t = matched_pair_transitions(arm, flr, margin=0.0)
    assert t["improved"] == 1   # a: floor−arm = +0.4
    assert t["degraded"] == 1   # b: −0.4
    assert t["preserved"] == 1  # c: 0
    assert t["unresolved"] == 2  # d, e
    assert t["n_resolved"] == 3


def test_transitions_margin_creates_preserved():
    arm = {"a": 0.30, "b": 0.10}
    flr = {"a": 0.34, "b": 0.50}
    t = matched_pair_transitions(arm, flr, margin=0.05)
    assert t["preserved"] == 1   # a: +0.04 within margin
    assert t["improved"] == 1    # b: +0.40


# ── sign_test ─────────────────────────────────────────────────────────────────

def test_sign_test_all_positive_significant():
    res = sign_test([0.1] * 10)
    assert res["n_pos"] == 10 and res["n_neg"] == 0
    assert res["p_value"] < 0.01


def test_sign_test_balanced_not_significant():
    res = sign_test([0.1, 0.1, 0.1, -0.1, -0.1, -0.1])
    assert res["n_pos"] == 3 and res["n_neg"] == 3
    assert res["p_value"] == pytest.approx(1.0)


def test_sign_test_drops_ties_within_margin():
    res = sign_test([0.2, 0.2, 0.01, -0.01], margin=0.05)
    assert res["n_used"] == 2 and res["n_pos"] == 2 and res["n_neg"] == 0


def test_sign_test_empty():
    assert sign_test([]) ["p_value"] == 1.0


# ── mcnemar_exact ─────────────────────────────────────────────────────────────

def test_mcnemar_all_discordant_one_way_significant():
    arm = [True] * 10
    flr = [False] * 10
    res = mcnemar_exact(arm, flr)
    assert res["arm_only"] == 10 and res["floor_only"] == 0
    assert res["p_value"] < 0.01


def test_mcnemar_balanced_discordant_not_significant():
    arm = [True, True, False, False]
    flr = [False, False, True, True]
    res = mcnemar_exact(arm, flr)
    assert res["arm_only"] == 2 and res["floor_only"] == 2
    assert res["p_value"] == pytest.approx(1.0)


def test_mcnemar_no_discordant_pairs():
    res = mcnemar_exact([True, True, False], [True, True, False])
    assert res["n_discordant"] == 0 and res["p_value"] == 1.0


def test_mcnemar_length_mismatch_raises():
    with pytest.raises(ValueError):
        mcnemar_exact([True, False], [True])


def test_mcnemar_distinct_from_sign_test():
    # Sign test on the continuous diff and McNemar on suppress-vs-vanilla can
    # disagree — they are genuinely different statistics.
    # arm always beats floor slightly (sign → all positive), but both suppress
    # vs vanilla on every task (McNemar → no discordant pairs).
    diffs = [0.05] * 6
    assert sign_test(diffs)["p_value"] < 0.05
    mc = mcnemar_exact([True] * 6, [True] * 6)
    assert mc["p_value"] == 1.0


# ── noise_band_fraction_rms ───────────────────────────────────────────────────

def test_noise_band_rms_value():
    band = noise_band_fraction_rms([0.1, 0.2], [0.2, 0.4])
    assert band == pytest.approx(np.sqrt((0.01 + 0.04) / 2))


def test_noise_band_zero_when_identical():
    assert noise_band_fraction_rms([0.5, 0.3], [0.5, 0.3]) == 0.0


def test_noise_band_mismatch_raises():
    with pytest.raises(ValueError):
        noise_band_fraction_rms([0.1, 0.2], [0.1])
    with pytest.raises(ValueError):
        noise_band_fraction_rms([], [])


# ── DeltaFloorCell.passes() gate logic ────────────────────────────────────────

def _cell(delta, ci_low, ci_high, holm_p, band):
    from src.steering_analysis import BootstrapResult
    b = BootstrapResult(estimate=delta, ci_low=ci_low, ci_high=ci_high,
                        n_tasks=20, n_resamples=2000)
    return DeltaFloorCell(behaviour="b", arm="single_direction", floor="energy_matched_random",
                          alpha=1.0, n_tasks=20, delta_floor=delta, bootstrap=b,
                          raw_p=0.001, holm_p=holm_p, band_b=band)


def test_passes_true_when_sig_and_band_cleared():
    assert _cell(0.30, 0.20, 0.40, holm_p=0.01, band=0.05).passes() is True


def test_passes_false_when_band_not_cleared():
    assert _cell(0.04, 0.01, 0.07, holm_p=0.01, band=0.05).passes() is False


def test_passes_false_when_ci_includes_zero():
    assert _cell(0.30, -0.01, 0.61, holm_p=0.20, band=0.05).passes() is False


def test_passes_none_when_band_absent_preliminary():
    assert _cell(0.30, 0.20, 0.40, holm_p=0.01, band=None).passes() is None


def test_passes_none_when_no_bootstrap():
    c = DeltaFloorCell("b", "single_direction", "energy_matched_random", 1.0,
                       n_tasks=1, delta_floor=None, bootstrap=None, raw_p=None)
    assert c.passes() is None


# ── delta_floor_cell end-to-end ───────────────────────────────────────────────

def test_delta_floor_cell_positive_signal():
    # arm suppresses to 0.2, floor barely (0.45); vanilla 0.5 → Δ_floor ≈ 0.25 > 0
    s, a = _scenario("backtracking", 20, van_frac=0.5, arm_frac=0.2, floor_frac=0.45, seed=7)
    cell = delta_floor_cell(s, a, "backtracking", "single_direction", 1.0, band_b=0.05)
    assert cell.n_tasks == 20
    assert cell.delta_floor == pytest.approx(0.25, abs=0.05)
    assert cell.bootstrap.excludes_zero()
    assert cell.suppression_arm > cell.suppression_floor  # arm suppresses more
    assert cell.sign["n_pos"] > cell.sign["n_neg"]


def test_delta_floor_cell_null_signal():
    # arm ≈ floor → Δ_floor ≈ 0, CI includes 0
    s, a = _scenario("backtracking", 24, van_frac=0.5, arm_frac=0.3, floor_frac=0.3, seed=11)
    cell = delta_floor_cell(s, a, "backtracking", "single_direction", 1.0, band_b=0.05)
    assert abs(cell.delta_floor) < 0.05
    assert not cell.bootstrap.excludes_zero()
    # A standalone cell has no Holm correction → not adjudicable for a PASS.
    assert cell.passes() is None


def test_delta_floor_cell_mcnemar_when_floor_does_not_suppress():
    # floor ≈ vanilla (no suppression), arm suppresses → McNemar all arm-only
    s, a = _scenario("backtracking", 16, van_frac=0.5, arm_frac=0.15, floor_frac=0.5, seed=13)
    cell = delta_floor_cell(s, a, "backtracking", "single_direction", 1.0, band_b=0.02)
    assert cell.mcnemar["arm_only"] >= 14
    assert cell.mcnemar["floor_only"] == 0
    assert cell.mcnemar["p_value"] < 0.01


def test_delta_floor_cell_insufficient_tasks():
    s, a = _scenario("backtracking", 1, van_frac=0.5, arm_frac=0.2, floor_frac=0.45, seed=1)
    cell = delta_floor_cell(s, a, "backtracking", "single_direction", 1.0)
    assert cell.status == "insufficient-tasks"
    assert cell.bootstrap is None and cell.passes() is None


def test_delta_floor_invariant_to_vanilla():
    # Δ_floor = floor_frac − arm_frac, so the vanilla fraction must NOT change it.
    s1, a1 = _scenario("backtracking", 18, van_frac=0.5, arm_frac=0.2, floor_frac=0.45, seed=5)
    s2, a2 = _scenario("backtracking", 18, van_frac=0.9, arm_frac=0.2, floor_frac=0.45, seed=5)
    c1 = delta_floor_cell(s1, a1, "backtracking", "single_direction", 1.0)
    c2 = delta_floor_cell(s2, a2, "backtracking", "single_direction", 1.0)
    assert c1.delta_floor == pytest.approx(c2.delta_floor)
    assert c1.bootstrap.estimate == pytest.approx(c2.bootstrap.estimate)
    # but the reported suppression context DOES move with vanilla
    assert c2.suppression_arm > c1.suppression_arm


def test_delta_floor_cell_unknown_arm_raises():
    s, a = _scenario("backtracking", 4, 0.5, 0.2, 0.45)
    with pytest.raises(ValueError):
        delta_floor_cell(s, a, "backtracking", "vanilla", 1.0)


# ── delta_floor_headline (multi-behaviour, Holm, band gate) ───────────────────

def _two_behaviour_signal(seed=21):
    s, a = [], []
    for beh in ("backtracking", "uncertainty-estimation"):
        ss, aa = _scenario(beh, 20, van_frac=0.5, arm_frac=0.2, floor_frac=0.46, seed=seed)
        s += ss; a += aa
    return s, a


def test_headline_planted_signal_passes_with_band():
    s, a = _two_behaviour_signal()
    out = delta_floor_headline(
        s, a, arms=["single_direction"],
        alpha_star={"backtracking": 1.0, "uncertainty-estimation": 1.0},
        behaviours=["backtracking", "uncertainty-estimation"],
        band_b={"backtracking": 0.05, "uncertainty-estimation": 0.05})
    assert out["n_pass"] == 2
    for key, cell in out["cells"].items():
        assert cell["delta_floor"] > 0
        assert cell["holm_p"] is not None and cell["holm_p"] < 0.05
        assert cell["passes"] is True


def test_headline_null_does_not_pass():
    # arm ≈ floor for both behaviours → Holm-corrected, banded, still no pass.
    s, a = [], []
    for beh in ("backtracking", "uncertainty-estimation"):
        ss, aa = _scenario(beh, 22, van_frac=0.5, arm_frac=0.3, floor_frac=0.3, seed=31)
        s += ss; a += aa
    out = delta_floor_headline(
        s, a, arms=["single_direction"],
        alpha_star={"backtracking": 1.0, "uncertainty-estimation": 1.0},
        behaviours=["backtracking", "uncertainty-estimation"],
        band_b={"backtracking": 0.05, "uncertainty-estimation": 0.05})
    assert out["n_pass"] == 0
    for cell in out["cells"].values():
        assert cell["passes"] is False


def test_headline_band_ungated_is_preliminary_not_pass():
    s, a = _two_behaviour_signal()
    out = delta_floor_headline(
        s, a, arms=["single_direction"],
        alpha_star={"backtracking": 1.0, "uncertainty-estimation": 1.0},
        behaviours=["backtracking", "uncertainty-estimation"],
        band_b=None)  # no non-builder annotator → cannot pass
    assert out["n_pass"] == 0
    assert out["n_preliminary"] == 2
    for cell in out["cells"].values():
        assert cell["passes"] is None and cell["bootstrap"]["excludes_zero"]


def test_headline_holm_applied_across_family():
    s, a = _two_behaviour_signal()
    out = delta_floor_headline(
        s, a, arms=["single_direction"],
        alpha_star={"backtracking": 1.0, "uncertainty-estimation": 1.0},
        behaviours=["backtracking", "uncertainty-estimation"],
        band_b={"backtracking": 0.05, "uncertainty-estimation": 0.05})
    for key, cell in out["cells"].items():
        assert cell["holm_p"] >= cell["raw_p"]  # Holm only inflates p
    assert len(out["holm"]) == 2


def test_headline_serializable():
    import json
    s, a = _two_behaviour_signal()
    out = delta_floor_headline(
        s, a, arms=["single_direction"],
        alpha_star={"backtracking": 1.0, "uncertainty-estimation": 1.0},
        behaviours=["backtracking", "uncertainty-estimation"],
        band_b={"backtracking": 0.05, "uncertainty-estimation": 0.05})
    json.dumps(out)  # must not raise (no tuple keys, no numpy scalars leaking)


# ── oracle-audit regression tests (M1 empirical-p, M2 negative Δ, edge cases) ──

def test_passes_false_for_significant_negative_delta():
    # arm suppresses LESS than its floor, significantly → the Δ_floor>0 clause
    # must reject (the scientifically dangerous direction the sign guard protects).
    c = _cell(-0.30, -0.50, -0.10, holm_p=0.001, band=0.05)
    assert c.bootstrap.excludes_zero()
    assert c.passes() is False


def test_passes_false_when_delta_equals_band():
    # strict '>' band gate: Δ_floor exactly at the band does NOT clear it
    assert _cell(0.05, 0.02, 0.08, holm_p=0.01, band=0.05).passes() is False


def test_empirical_p_floored_not_zero():
    boot = np.full(2000, 0.3)            # all positive → nothing on the null side
    p = empirical_two_sided_p(boot, estimate=0.3)
    assert p == pytest.approx(2 / 2001)
    assert p > 0                          # never exactly 0 → Holm cannot collapse


def test_empirical_p_null_centered_is_one():
    boot = np.linspace(-1.0, 1.0, 2001)  # symmetric about 0
    assert empirical_two_sided_p(boot, estimate=0.0) == pytest.approx(1.0, abs=0.01)


def test_empirical_p_empty_dist():
    assert empirical_two_sided_p([], estimate=0.3) == 1.0


def test_delta_floor_cell_raw_p_never_zero():
    # strong signal used to underflow raw_p to 0.0 (M1); now floored at 1/(B+1).
    s, a = _scenario("backtracking", 20, van_frac=0.5, arm_frac=0.15, floor_frac=0.5, seed=9)
    cell = delta_floor_cell(s, a, "backtracking", "single_direction", 1.0, n_resamples=2000)
    assert cell.raw_p >= 1 / 2001
    assert cell.raw_p > 0


def test_delta_floor_cell_missing_vanilla_still_computes():
    s, a = _scenario("backtracking", 12, van_frac=0.5, arm_frac=0.2, floor_frac=0.45, seed=3)
    s = [r for r in s if r["method"] != "vanilla"]
    a = [r for r in a if r["method"] != "vanilla"]
    cell = delta_floor_cell(s, a, "backtracking", "single_direction", 1.0)
    assert cell.delta_floor is not None and cell.bootstrap is not None
    assert cell.mcnemar is None and cell.suppression_arm is None  # context gone, Δ intact


def test_noise_band_rejects_nan():
    with pytest.raises(ValueError):
        noise_band_fraction_rms([0.1, float("nan")], [0.2, 0.3])


def test_per_task_fraction_excludes_coverage_incomplete_rows():
    """2026-08-13: a schema-valid row whose coverage verdict is incomplete (or
    stale) is unresolved for estimation; legacy rows without any verdict keep
    their historical inclusion."""
    from src.annotation_coverage import COVERAGE_RULE_VERSION
    from src.delta_floor import per_task_fraction

    def gen_row(tid):
        return {"task_id": tid, "behaviour": "backtracking",
                "method": "transported_raw_suppress", "alpha": 1.0}

    spans = [{"label": "backtracking", "text": "Wait."},
             {"label": "deduction", "text": "So."}]
    ann = [
        {**gen_row("T_ok"), "annotations": spans, "annotation_complete": True,
         "annotation_coverage": {"rule_version": COVERAGE_RULE_VERSION,
                                 "complete": True}},
        {**gen_row("T_gap"), "annotations": spans, "annotation_complete": True,
         "annotation_coverage": {"rule_version": COVERAGE_RULE_VERSION,
                                 "complete": False}},
        {**gen_row("T_stale"), "annotations": spans, "annotation_complete": True,
         "annotation_coverage": {"rule_version": "some-old-rule",
                                 "complete": True}},
        {**gen_row("T_legacy"), "annotations": spans, "annotation_complete": True},
    ]
    gen = [gen_row(t) for t in ("T_ok", "T_gap", "T_stale", "T_legacy")]
    fr = per_task_fraction(gen, ann, "backtracking",
                           "transported_raw_suppress", 1.0)
    assert set(fr) == {"T_ok", "T_legacy"}, fr
    assert fr["T_ok"] == 0.5
