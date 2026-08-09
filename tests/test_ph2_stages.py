"""Pre-spend checks for the Phase-2 stage library (src/ph2_stages.py).

Everything here runs WITHOUT torch/transformers/proxy — it locks the sealed
design facts (arms, sites, doses, floors, thresholds) and the decision
arithmetic before any pod or API spend.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import ph2_executor as px          # noqa: E402
import src.ph2_stages as st        # noqa: E402


# ── layer convention (the E10.1 one-block-off trap) ──────────────────────────

def test_layer_convention_locked():
    # hs[17] = output of model.model.layers[16]; 23_p2_redesign.py executed the
    # DAS direction with SteeredModel layer 16 — the battery must match.
    assert st.HS_SITE == 17
    assert st.machinery_layer(st.HS_SITE) == 16
    assert st.HS_NEIGHBOURS == (16, 18)
    with pytest.raises(ValueError):
        st.machinery_layer(0)     # hs[0] is the embedding — never a hook site


# ── the sealed battery ───────────────────────────────────────────────────────

def test_battery_is_thirteen_cells():
    arms = st.battery_arms()
    assert len(arms) == st.N_BATTERY_CELLS == 13
    methods = [a["method"] for a in arms]
    assert len(set(methods)) == 13          # no duplicate labels
    # every ± family has exactly both signs
    for fam in ("transported_raw", "transported_norm", "transported_whitened",
                "refit"):
        assert f"{fam}_induce" in methods and f"{fam}_suppress" in methods
    for single in ("vanilla", "sham_frame", "random_orthogonal",
                   "count_matched_floor", "energy_matched_floor"):
        assert single in methods
    # floors/controls run suppression-oriented (primary is suppression-oriented)
    by = {a["method"]: a for a in arms}
    for m in ("sham_frame", "random_orthogonal", "count_matched_floor",
              "energy_matched_floor"):
        assert by[m]["sign"] == "suppress"


def test_floor_pairing():
    # every active arm and every non-floor control contrasts against the
    # energy-matched floor; vanilla and the floors themselves pair to nothing
    assert st.floor_for_battery_arm("transported_raw_suppress") == "energy_matched_floor"
    assert st.floor_for_battery_arm("transported_whitened_induce") == "energy_matched_floor"
    assert st.floor_for_battery_arm("refit_induce") == "energy_matched_floor"
    assert st.floor_for_battery_arm("sham_frame") == "energy_matched_floor"
    assert st.floor_for_battery_arm("vanilla") is None
    assert st.floor_for_battery_arm("energy_matched_floor") is None
    assert st.SECONDARY_FLOOR == "sham_frame"   # §5: "vs sham and energy-matched"


def test_cell_enumeration_counts_and_resume():
    tasks = [{"id": f"T{i:03d}", "prompt": "p"} for i in range(100)]
    cells = st.enumerate_battery_cells(tasks)
    assert len(cells) == 13 * 100 * 3           # full grid before aliasing
    keys = [c["key"] for c in cells]
    assert len(set(keys)) == len(keys)          # resume keys unique
    # resume: marking half of one role's cells done removes exactly those
    done = {k for k in keys if k[0] == "star1" and k[2] < "T050"}
    remaining = st.enumerate_battery_cells(tasks, done=done)
    assert len(remaining) == len(cells) - len(done)

    inj = st.enumerate_injection_cells(tasks)
    assert len(inj) == 3 * 100                  # f∈{0.25,0.5,0.75}, base only
    assert {c["role"] for c in inj} == {"base"}
    assert sorted({c["fraction"] for c in inj}) == [0.25, 0.5, 0.75]


# ── frames, corrections, mixing ──────────────────────────────────────────────

def test_random_orthonormal_frame_properties():
    U = st.random_orthonormal_frame(64, 2, "seedA")
    assert U.shape == (64, 2)
    np.testing.assert_allclose(U.T @ U, np.eye(2), atol=1e-5)
    U2 = st.random_orthonormal_frame(64, 2, "seedA")
    np.testing.assert_array_equal(U, U2)        # deterministic per key
    U3 = st.random_orthonormal_frame(64, 2, "seedB")
    assert not np.allclose(U, U3)               # distinct across keys


def test_class_means_norm_gain_whitening():
    rng = np.random.default_rng(0)
    U = st.random_orthonormal_frame(16, 2, "cm")
    H_on = rng.normal(2.0, 0.5, size=(50, 16))
    H_off = rng.normal(-1.0, 0.5, size=(50, 16))
    cm = st.class_mean_coords(H_on, H_off, U)
    np.testing.assert_allclose(cm["c_on"], (H_on @ U).mean(axis=0))
    np.testing.assert_allclose(cm["c_off"], (H_off @ U).mean(axis=0))
    # doubling the states doubles the norm gain vs the originals
    assert st.norm_gain(H_on, 2.0 * H_on) == pytest.approx(2.0)
    assert st.norm_gain(H_on, H_on) == pytest.approx(1.0)
    stds = st.whitening_stats(H_on)
    assert stds.shape == (16,) and np.all(stds > 0)
    # identical stds ⇒ whitened transport returns the same subspace
    W = st.whitened_frame(U, stds, stds)
    np.testing.assert_allclose(np.abs(W.T @ U), np.eye(2), atol=1e-5)


def test_clamp_targets_signs_and_gain():
    cm = {"c_on": np.array([1.0, 2.0]), "c_off": np.array([-3.0, 0.5])}
    np.testing.assert_allclose(
        st.clamp_targets_for_arm("transported_raw", "induce", cm), [1.0, 2.0])
    np.testing.assert_allclose(
        st.clamp_targets_for_arm("transported_raw", "suppress", cm), [-3.0, 0.5])
    np.testing.assert_allclose(
        st.clamp_targets_for_arm("transported_norm", "induce", cm, gain=1.5),
        [1.5, 3.0])


def test_injection_mixing_endpoints():
    # f=0 → pure frame (the battery's transported arm); f=1 → pure sham
    assert st.mixing_weights(0.0) == {"frame": 1.0, "sham": 0.0}
    assert st.mixing_weights(1.0) == {"frame": 0.0, "sham": 1.0}
    w = st.mixing_weights(0.5)
    assert w["frame"] == pytest.approx(0.5) and w["sham"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        st.mixing_weights(1.2)


# ── chain endpoints + damage gate (sealed thresholds) ────────────────────────

def test_boxed_answer_extraction():
    assert st.boxed_answer(r"so the answer is \boxed{42}") == "42"
    assert st.boxed_answer(r"\boxed{a}\ then \boxed{ b\, c }") == "b\\, c"
    assert st.boxed_answer(r"\boxed{\frac{1}{2}}") == "\\frac{1}{2}"
    assert st.boxed_answer("no box here") is None


def test_damage_gate_thresholds():
    ok = [{"looped": False, "truncated": False, "boxed_present": True,
           "boxed_correct": True}] * 10
    # loop excess just over 10pp trips the gate
    loopy = [{**r, "looped": i < 2} for i, r in enumerate(ok)]  # 20% vs 0%
    g = st.damage_gate(loopy, ok, ok)
    assert not g["damage_ok"] and "loop excess" in g["reasons"][0]
    # boxed drop over 15pp trips it
    wrong = [{**r, "boxed_correct": i >= 2} for i, r in enumerate(ok)]  # 80% vs 100%
    g2 = st.damage_gate(wrong, ok, ok)
    assert not g2["damage_ok"] and any("boxed drop" in r for r in g2["reasons"])
    # within thresholds passes
    mild = [{**r, "looped": i < 1} for i, r in enumerate(ok)]   # 10% vs 0% — at, not over
    assert st.damage_gate(mild, ok, ok)["damage_ok"]


# ── attenuation guards + primary test ────────────────────────────────────────

def test_attenuation_guard():
    assert st.attenuation_fraction(0.10, 0.05) == pytest.approx(0.5)
    assert st.attenuation_fraction(0.0, 0.05) is None       # unstable denominator
    assert st.attenuation_fraction(-0.02, 0.05) is None     # wrong sign
    assert st.attenuation_fraction(None, 0.05) is None


def test_paired_attenuation_test_recovers_known_effect():
    rng = np.random.default_rng(1)
    base = {f"T{i}": 0.10 + rng.normal(0, 0.01) for i in range(60)}
    tgt = {t: v * 0.4 + rng.normal(0, 0.01) for t, v in base.items()}  # f ≈ 0.6
    cell = st.paired_attenuation_test(base, tgt, n_resamples=500, seed=0)
    assert cell["status"] == "ok" and cell["n_shared_tasks"] == 60
    assert cell["paired_diff"]["excludes_zero"]
    assert 0.4 < cell["attenuation"] < 0.8
    assert cell["attenuation_ci"] is not None
    # f≈0.6 ≥ margin 0.5 ⇒ NOT retained under the sealed rule
    assert st.raw_retained(cell) is False
    # near-zero attenuation IS retained
    tgt2 = {t: v + rng.normal(0, 0.005) for t, v in base.items()}
    cell2 = st.paired_attenuation_test(base, tgt2, n_resamples=500, seed=0)
    assert st.raw_retained(cell2) is True


def test_paired_attenuation_insufficient_tasks():
    assert st.paired_attenuation_test({"A": 1.0}, {"A": 0.5})["status"] == \
        "insufficient-tasks"


# ── injection power gate ─────────────────────────────────────────────────────

def test_injection_power_separates_strong_from_null():
    rng = np.random.default_rng(2)
    f0 = {f"T{i}": 0.10 + rng.normal(0, 0.02) for i in range(80)}
    # strong attenuation at f=0.5: half the effect gone → power should be high
    fx = {t: v * 0.5 + rng.normal(0, 0.02) for t, v in f0.items()}
    strong = st.injection_detection_power(f0, fx, n_resamples=200,
                                          n_power_reps=60, seed=0)
    # pure noise → power should be low
    fnull = {t: v + rng.normal(0, 0.02) for t, v in f0.items()}
    null = st.injection_detection_power(f0, fnull, n_resamples=200,
                                        n_power_reps=60, seed=0)
    assert strong["power"] > 0.8 >= st.INJECTION_PASS_POWER - 0.001
    assert null["power"] < 0.5
    assert strong["passes"] and not null["passes"]


# ── record schema feeds the AUTHORITATIVE pooling path unchanged ─────────────

def _fake_gen_and_ann(role, methods, tasks, frac_by_method):
    """Synthetic battery records + annotations: each chain gets 10 sentences,
    round(frac*10) of them backtracking."""
    gen, ann = [], []
    for m in methods:
        for t in tasks:
            rec = st.battery_record(
                role, m, {"id": t}, {"chain": "x. " * 10, "n_tokens": 100},
                sign=None, frame_name=None)
            gen.append(rec)
            n_bt = round(frac_by_method[m] * 10)
            spans = ([{"label": "backtracking", "text": "x."}] * n_bt
                     + [{"label": "deduction", "text": "x."}] * (10 - n_bt))
            ann.append({**rec, "annotations": spans})
    return gen, ann


def test_battery_record_schema_flows_through_delta_floor():
    """The red-team F-item: Phase-2 pooling MUST run through
    src.delta_floor.per_task_fraction / delta_floor_cell unchanged."""
    from src.delta_floor import delta_floor_cell, per_task_fraction
    tasks = [f"T{i:03d}" for i in range(20)]
    gen, ann = _fake_gen_and_ann(
        "star1", ["transported_raw_suppress", "energy_matched_floor"], tasks,
        {"transported_raw_suppress": 0.1, "energy_matched_floor": 0.4})
    fr = per_task_fraction(gen, ann, "backtracking", "transported_raw_suppress",
                           st.CLAMP_GAIN)
    assert len(fr) == 20 and all(v == pytest.approx(0.1) for v in fr.values())
    cell = delta_floor_cell(gen, ann, "backtracking", "transported_raw_suppress",
                            st.CLAMP_GAIN, floor="energy_matched_floor",
                            n_resamples=200, seed=0)
    assert cell.n_tasks == 20
    assert cell.delta_floor == pytest.approx(0.3)   # floor 0.4 − arm 0.1
    assert cell.bootstrap is not None


def test_vanilla_record_uses_shared_key():
    rec = st.battery_record("base", "vanilla", {"id": "T1"},
                            {"chain": "c", "n_tokens": 5}, None, None)
    assert rec["behaviour"] == "shared" and rec["alpha"] == 0.0
    steered = st.battery_record("base", "sham_frame", {"id": "T1"},
                                {"chain": "c", "n_tokens": 5}, "suppress", "sham0")
    assert steered["behaviour"] == "backtracking"
    assert steered["alpha"] == st.CLAMP_GAIN
    assert steered["layer"] == 16 and steered["hs_site"] == 17


def test_missing_annotations_stay_unresolved_through_pooling():
    from src.delta_floor import per_task_fraction
    tasks = ["A", "B", "C"]
    gen, ann = _fake_gen_and_ann("base", ["sham_frame"], tasks, {"sham_frame": 0.2})
    ann[0]["annotations"] = []          # task A: empty → must vanish, not zero
    fr = per_task_fraction(gen, ann, "backtracking", "sham_frame", st.CLAMP_GAIN)
    assert set(fr) == {"B", "C"}        # A excluded pairwise, never coerced to 0


# ── gate + alignment arithmetic ──────────────────────────────────────────────

def test_target_gate_verdict_p95():
    rng = np.random.default_rng(3)
    nulls = [{"coord_auc": float(rng.uniform(0.45, 0.60)),
              "state_dependence": float(rng.normal(0, 0.5))} for _ in range(40)]
    good = {"coord_auc": 0.95, "state_dependence": 5.0}
    v = st.target_gate_verdict(good, nulls)
    assert v["gate_pass"] and v["n_null"] == 40
    # beating only ONE metric must fail the gate (BOTH required)
    lop = {"coord_auc": 0.95, "state_dependence": -1.0}
    assert not st.target_gate_verdict(lop, nulls)["gate_pass"]
    # continuity thresholds reported but never adjudicated
    assert v["base_absolute_thresholds_continuity"] == {"coord_auc": 0.65,
                                                        "state_dependence": 2.0}


def test_refit_alignment_vs_null():
    d = 32
    U = st.random_orthonormal_frame(d, 2, "align_base")
    nulls = [st.random_orthonormal_frame(d, 2, f"null{i}") for i in range(20)]
    same = st.refit_alignment(U, U.copy(), nulls)
    assert same["refit_aligned"] and not same["refit_misaligned_beyond_null"]
    assert same["alignment_mean_cos"] == pytest.approx(1.0, abs=1e-6)
    # a random frame should not read as aligned
    rand = st.refit_alignment(U, st.random_orthonormal_frame(d, 2, "other"), nulls)
    assert not rand["refit_aligned"]


# ── annotation glue budget + dedup keys ──────────────────────────────────────

def test_proxy_chunk_budget_within_29s():
    b = st.proxy_chunk_budget_ok()
    assert b["ok"], f"chunk budget would breach the 29-s proxy ceiling: {b}"
    assert b["worst_output_tokens_est"] <= 2300


def test_annotation_dedup_keys_separate_cells():
    # two arms on the same task must not collide in resume/dedup space
    a = st.battery_record("star1", "sham_frame", {"id": "T1"},
                          {"chain": "c", "n_tokens": 5}, "suppress", "sham0")
    b = st.battery_record("star1", "transported_raw_suppress", {"id": "T1"},
                          {"chain": "c", "n_tokens": 5}, "suppress", "base")
    key = lambda r: tuple(r.get(k) for k in st.ANNOTATION_DEDUP_KEYS)  # noqa: E731
    assert key(a) != key(b)
    c = st.battery_record("deepscaler", "sham_frame", {"id": "T1"},
                          {"chain": "c", "n_tokens": 5}, "suppress", "sham0")
    assert key(a) != key(c)             # same arm, different model role


# ── A2 adjunct table (estimation only) ───────────────────────────────────────

def test_a2_adjunct_paired_estimation():
    tasks = [f"T{i}" for i in range(30)]

    def rows(role, bt_frac, n_tokens):
        out = []
        for t in tasks:
            n_bt = round(bt_frac * 10)
            spans = ([{"label": "backtracking", "text": "x."}] * n_bt
                     + [{"label": "deduction", "text": "x."}] * (10 - n_bt))
            out.append({"task_id": t, "chain": r"ok \boxed{1}", "n_tokens": n_tokens,
                        "method": "vanilla", "annotations": spans})
        return out

    table = st.a2_adjunct_table(
        {"base": rows("base", 0.2, 1000),
         "star1": rows("star1", 0.1, 800),
         "deepscaler": rows("deepscaler", 0.3, 1200)}, cap=8192,
        n_resamples=200)
    assert table["estimation_only"] is True
    s = table["endpoints"]["star1"]
    assert s["prev_backtracking"]["diff_mean"] == pytest.approx(-0.1, abs=1e-9)
    assert s["length"]["diff_mean"] == pytest.approx(-200.0)
    d = table["endpoints"]["deepscaler"]
    assert d["prev_backtracking"]["diff_mean"] == pytest.approx(0.1, abs=1e-9)
    # bt per 1k: base 2/1000tok = 2.0; deepscaler 3/1200 = 2.5 → diff 0.5
    assert d["bt_per_1k"]["diff_mean"] == pytest.approx(0.5, abs=1e-6)


def test_a2_adjunct_unresolved_counted_not_zeroed():
    tasks = [f"T{i}" for i in range(10)]

    def rows(missing_first):
        out = []
        for i, t in enumerate(tasks):
            anns = None if (missing_first and i == 0) else \
                [{"label": "deduction", "text": "x."}] * 10
            out.append({"task_id": t, "chain": "c", "n_tokens": 100,
                        "method": "vanilla", "annotations": anns})
        return out

    table = st.a2_adjunct_table(
        {"base": rows(False), "star1": rows(True), "deepscaler": rows(False)},
        cap=8192, n_resamples=100)
    cell = table["endpoints"]["star1"]["prev_backtracking"]
    assert cell["n"] == 9 and cell["n_unresolved"] == 1


# ── executor wiring: refusal, aliasing, verdict assembly ─────────────────────

def test_spend_stages_refuse_without_authorisation():
    for stage in px.SPEND_STAGES:
        with pytest.raises(SystemExit, match="requires --authorised"):
            px._authorise_guard(stage, authorised=False)
    # non-spend stages pass the guard freely
    px._authorise_guard("discovery_extract", authorised=False)


def test_base_role_aliases_duplicate_families():
    """On base, norm gain = 1 / whitening ratio = 1 / refit == builder frame,
    so those families must ALIAS transported_raw (never regenerate)."""
    disc = {"base": {"norm_gain_vs_base": 1.0,
                     "class_means_base_frame": {"c_on": [0.0, 0.0],
                                                "c_off": [0.0, 0.0]}}}
    for fam in ("transported_norm", "transported_whitened", "refit"):
        for sign in ("induce", "suppress"):
            arm = {"method": f"{fam}_{sign}", "family": fam, "sign": sign}
            terms, _fname, alias = px._arm_terms("base", arm, disc,
                                                 Path("/nonexistent"))
            assert terms is None and alias == f"transported_raw_{sign}"


def test_decide_outcome_full_matrix_still_holds():
    """The verdict assembly's boolean vocabulary matches decide_outcome's."""
    o = dict(sensitivity_adequate=True, damage_ok=True, gate_pass=True,
             raw_retained=True, corrected_gate_pass=True, corrected_retained=False,
             refit_pass=True, refit_aligned=True, refit_misaligned_beyond_null=False,
             refit_recovered=True, repr_signal_above_null=True)
    assert px.decide_outcome(o) == "retained"
    o2 = {**o, "sensitivity_adequate": False}
    assert px.decide_outcome(o2) == "downgraded"


def test_annotation_504_shrink_floor():
    """The Phase-2 proxy rule: output budget halves on timeout-class retries
    but never below 1024 (and defaults leave the corpus path untouched)."""
    import inspect
    from src import annotation
    sig = inspect.signature(annotation.annotate_chain)
    assert sig.parameters["max_tokens"].default is None
    assert sig.parameters["shrink_on_retry"].default is False
    sig2 = inspect.signature(annotation.annotate_chains)
    assert sig2.parameters["max_tokens"].default is None
    # the halving arithmetic: 2048 → 1024 → floor
    budget = 2048
    for _ in range(3):
        budget = max(1024, budget // 2)
    assert budget == 1024


def test_battery_planned_generation_count():
    """§11 envelope: with base aliasing, planned generations = 7×100 (base)
    + 13×100×2 (targets) + 3×100 (injection) = 3,600 ≤ the 3,900–4,200 seal."""
    n_base_real = 13 - 6            # 3 aliased families × 2 signs collapse
    total = n_base_real * 100 + 13 * 100 * 2 + 3 * 100
    assert total == 3600
    assert total <= 4200


def test_provenance_frame_source_records_convention():
    p = px.build_provenance({}, stage="target_gates")
    fs = p["frame_source"]
    assert fs["hs_site"] == 17 and fs["width"] == 2
    assert "layers[K-1]" in fs["layer_convention"]
    assert fs["path"].endswith("width/frame_k2.npy")
    if (Path(px.ROOT) / fs["path"]).exists():
        assert len(fs["sha256"]) == 64


# ── analyse-stage DRY RUN (plumbing check: synthetic artifacts → verdicts) ───

def _spans(n_bt: int, n_total: int = 10):
    return ([{"label": "backtracking", "text": "x."}] * n_bt
            + [{"label": "deduction", "text": "x."}] * (n_total - n_bt))


def _mk_role_files(root: Path, role: str, tasks, bt_by_method: dict,
                   n_tokens: int = 1000):
    """Battery + annotation shards for one role. bt_by_method maps method →
    per-task backtracking sentence COUNT function f(i) out of 10."""
    gen, ann = [], []
    for m, f in bt_by_method.items():
        for i, t in enumerate(tasks):
            rec = st.battery_record(
                role, m, {"id": t},
                {"chain": r"x. " * 30 + r"\boxed{7}", "n_tokens": n_tokens},
                sign=None, frame_name=None)
            gen.append(rec)
            ann.append({**rec, "annotations": _spans(f(i))})
    (root / "battery").mkdir(parents=True, exist_ok=True)
    (root / "annotation").mkdir(parents=True, exist_ok=True)
    (root / "battery" / f"{role}.json").write_text(json.dumps(gen))
    (root / "annotation" / f"{role}.json").write_text(json.dumps(ann))


def test_stage_analyse_dry_run_end_to_end(tmp_path, monkeypatch):
    """Full plumbing dry-run: synthetic battery/injection/annotation/gates/refit
    artifacts through the REAL stage_analyse to five-outcome verdicts.

    Construction: base effect Δ≈0.30; STAR1 keeps ≈93% of it (retained-shaped,
    gates pass, refit aligned); DeepScaleR loses ≈93% (gate fails, refit fails,
    no representational signal → disabled-shaped). Injection f=0.5 halves the
    base effect with high power → sensitivity adequate.
    """
    tasks = [f"T{i:03d}" for i in range(12)]
    root = tmp_path / "ph2"
    monkeypatch.setattr(px, "OUT", root)
    monkeypatch.setattr(px, "PROV_DIR", root / "provenance")
    monkeypatch.setattr(st, "B_BOOT", 200)
    # shrink the injection power estimate (2000×500 default is a pod-scale run)
    real_power = st.injection_detection_power
    monkeypatch.setattr(
        st, "injection_detection_power",
        lambda f0, fx, **kw: real_power(f0, fx, n_resamples=200,
                                        n_power_reps=40,
                                        seed=kw.get("seed", 0)))

    j = lambda i: i % 3 - 1              # −1/0/+1 per-task jitter  # noqa: E731
    # base: floor 5/10, raw 2/10 → Δ≈0.30; sham 4/10 (secondary floor)
    _mk_role_files(root, "base", tasks, {
        "vanilla": lambda i: 4,
        "transported_raw_suppress": lambda i: 2 + j(i),
        "transported_raw_induce": lambda i: 7,
        "sham_frame": lambda i: 4 + j(i),
        "random_orthogonal": lambda i: 4,
        "count_matched_floor": lambda i: 4,
        "energy_matched_floor": lambda i: 5 + j(i),
    })
    # STAR1: floor 5, raw 2.2 → Δ≈0.28 (≈7% attenuation → retained-shaped)
    _mk_role_files(root, "star1", tasks, {
        "vanilla": lambda i: 4,
        "transported_raw_suppress": lambda i: 2 + (1 if i % 4 == 0 else 0) + j(i),
        "transported_raw_induce": lambda i: 7,
        "transported_norm_suppress": lambda i: 2 + j(i),
        "transported_norm_induce": lambda i: 7,
        "transported_whitened_suppress": lambda i: 2 + j(i),
        "transported_whitened_induce": lambda i: 7,
        "refit_suppress": lambda i: 2 + j(i),
        "refit_induce": lambda i: 7,
        "sham_frame": lambda i: 4 + j(i),
        "random_orthogonal": lambda i: 4,
        "count_matched_floor": lambda i: 4,
        "energy_matched_floor": lambda i: 5 + j(i),
    })
    # DeepScaleR: floor 5, raw 4.8 → Δ≈0.02 (≈93% attenuation); refit dead too
    _mk_role_files(root, "deepscaler", tasks, {
        "vanilla": lambda i: 4,
        "transported_raw_suppress": lambda i: 5 + j(i) - (1 if i % 6 == 0 else 0),
        "transported_raw_induce": lambda i: 4,
        "transported_norm_suppress": lambda i: 5 + j(i),
        "transported_norm_induce": lambda i: 4,
        "transported_whitened_suppress": lambda i: 5 + j(i),
        "transported_whitened_induce": lambda i: 4,
        "refit_suppress": lambda i: 5 + j(i),
        "refit_induce": lambda i: 4,
        "sham_frame": lambda i: 5 + j(i),
        "random_orthogonal": lambda i: 5,
        "count_matched_floor": lambda i: 5,
        "energy_matched_floor": lambda i: 5 + j(i),
    })
    # injection shard: f=0.25 ≈ 85% of the base effect, f=0.5 ≈ half, f=0.75 ≈ none
    inj_gen, inj_ann = [], []
    for f, raw in ((0.25, lambda i: 2 + j(i)), (0.5, lambda i: 4),
                   (0.75, lambda i: 5 + j(i))):
        m = st.injection_method_label(f)
        for i, t in enumerate(tasks):
            rec = st.battery_record("base", m, {"id": t},
                                    {"chain": "x. " * 30, "n_tokens": 1000},
                                    "suppress", "mixed", extra={"fraction": f})
            inj_gen.append(rec)
            inj_ann.append({**rec, "annotations": _spans(raw(i))})
    (root / "injection").mkdir(parents=True)
    (root / "injection" / "base_injection.json").write_text(json.dumps(inj_gen))
    (root / "annotation" / "base_injection.json").write_text(json.dumps(inj_ann))
    # gates + refit artifacts (STAR1 grounded+aligned; DeepScaleR neither)
    (root / "gates").mkdir(parents=True)
    (root / "gates" / "target_gates.json").write_text(json.dumps(
        {"star1": {"gate_pass": True, "complete": True},
         "deepscaler": {"gate_pass": False, "complete": True}}))
    (root / "refit").mkdir(parents=True)
    (root / "refit" / "refit_report.json").write_text(json.dumps(
        {"star1": {"widths": {"2": {"alignment_vs_base": {
            "refit_aligned": True, "refit_misaligned_beyond_null": False}}}},
         "deepscaler": {"widths": {"2": {"alignment_vs_base": {
             "refit_aligned": False, "refit_misaligned_beyond_null": True}}}}}))

    px.stage_analyse(authorised=False)

    out = json.loads((root / "analysis" / "ph2_analysis.json").read_text())
    assert out["sensitivity"]["sensitivity_adequate"] is True
    assert out["verdicts"]["star1"]["outcome"] == "retained"
    assert out["verdicts"]["deepscaler"]["outcome"] == "disabled"
    # primary family: Holm over exactly the two checkpoints
    assert set(out["primary_attenuation"]) == {"star1", "deepscaler"}
    for role in ("star1", "deepscaler"):
        assert "holm_p" in out["primary_attenuation"][role]
    att = out["primary_attenuation"]["deepscaler"]["attenuation"]
    assert att is not None and att > 0.8
    # A2 adjunct present with all six endpoint groups per contrast
    a2 = out["a2_adjunct"]["endpoints"]
    assert set(a2) == {"star1", "deepscaler"}
    assert {"prev_backtracking", "bt_per_1k", "boxed", "length", "looped",
            "truncated"} <= set(a2["star1"])
    # caveat travels with every verdict (A3)
    assert "builder-annotator" in out["verdicts"]["star1"]["wording_caveat"]
    # provenance sidecar written with analyse stage + provisional status
    prov = json.loads((root / "provenance" / "analyse.json").read_text())
    assert prov["stage"] == "analyse"
    assert prov["empirical_evidence_status"] == "provisional"
