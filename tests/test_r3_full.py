"""
Tests for the FULL R3 strategy-entropy harness (32_r3_strategy.py full-* stages).

CPU-only, seconds: task construction (golds re-derived INDEPENDENTLY per
template — brute-force sums, lattice DP, exact linear solve, float roots),
classifier freeze guard, strategy-entropy estimator, resume-safe batch
builder ((cell, seed) homogeneity under resume holes), atomic checkpoint
writes, and a synthetic end-to-end full-analyse (planted strategy
distributions → known entropies, frontier + P-R2.1-reframed verdicts).
No model is loaded anywhere here (the GPU path is exercised on the pod via
--smoke, which reuses the pilot's pattern).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import types
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

#: sha256 of json.dumps({"FAMILIES":…, "SPACES":…}, sort_keys=True) at seal
#: time (pilot prereg, R3_PILOT_PREREG.md). The lexical classifier is FROZEN:
#: if this test fails you changed the sealed instrument — revert, or amend the
#: prereg explicitly before touching it.
CLASSIFIER_SHA256 = "268ec687a196146854247db2cf002054ce3fee685dd8fdef1fa7c169711d0c1d"


def _runner():
    """Import 32_r3_strategy.py (digit-leading name needs spec loading)."""
    path = Path(__file__).resolve().parents[1] / "32_r3_strategy.py"
    spec = importlib.util.spec_from_file_location("r3_strategy_runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def r3():
    return _runner()


@pytest.fixture(scope="module")
def full_tasks(r3):
    return r3.build_full_tasks()


# ── full task set ─────────────────────────────────────────────────────────────

def test_full_tasks_scale_and_strata(r3, full_tasks):
    assert 50 <= len(full_tasks) <= 80          # prereg: "~50–80 tasks"
    assert len(full_tasks) == 64
    by_tpl = {}
    for t in full_tasks:
        by_tpl.setdefault(t["template"], []).append(t)
    assert sorted(by_tpl) == [f"T{i}" for i in range(1, 9)]
    for tpl, ts in by_tpl.items():
        assert len(ts) == 8
        assert sum(t["difficulty"] == "easy" for t in ts) == 4
        assert sum(t["difficulty"] == "hard" for t in ts) == 4
        # declared strategy spaces UNCHANGED from the sealed pilot
        assert all(t["strategy_space"] == r3.SPACES[tpl] for t in ts)
    ids = [t["task_id"] for t in full_tasks]
    assert len(set(ids)) == len(ids)
    assert all(t["prompt"].endswith(r3.SUFFIX) for t in full_tasks)
    assert all(t["gold"].lstrip("-").isdigit() for t in full_tasks)


def test_full_golds_independent(full_tasks):
    """Re-derive every gold with an INDEPENDENT method per template."""
    def paths(m, n):
        grid = [[1] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                grid[i][j] = grid[i - 1][j] + grid[i][j - 1]
        return grid[m][n]

    def tilings(n):                              # DP, not the Fib closed form
        a, b = 1, 1                              # f(0)=1, f(1)=1
        for _ in range(n - 1):
            a, b = b, a + b
        return b

    for t in full_tasks:
        p, g = t["params"], int(t["gold"])
        tpl = t["template"]
        if tpl == "T1":
            assert tilings(p["n"]) == g
        elif tpl == "T2":
            assert sum(k * (k + 1) for k in range(1, p["n"] + 1)) == g
        elif tpl == "T3":
            assert paths(p["m"], p["n"]) == g
        elif tpl == "T4":
            det = p["a"] * p["e"] - p["b"] * p["d"]
            assert det != 0
            x = Fraction(p["c"] * p["e"] - p["b"] * p["f"], det)
            y = Fraction(p["a"] * p["f"] - p["c"] * p["d"], det)
            assert x == p["x"] and y == p["y"]   # derived constants consistent
            assert x * y == g
        elif tpl == "T5":
            assert sum(k * k for k in range(1, p["n"] + 1)) == g
        elif tpl == "T6":
            assert (p["a"] ** p["b"]) % 10 == g
        elif tpl == "T7":
            assert 6 ** p["n"] - 5 ** p["n"] == g
        elif tpl == "T8":
            disc = p["p"] ** 2 - 4 * p["q"]
            assert disc > 0                      # real, distinct roots
            r_ = (p["p"] + math.sqrt(disc)) / 2
            s_ = (p["p"] - math.sqrt(disc)) / 2
            assert abs(r_ ** 2 + s_ ** 2 - g) < 1e-6


def test_full_tasks_disjoint_from_pilot(r3, full_tasks):
    pilot = {t["prompt"] for t in r3.build_tasks()}
    assert not any(t["prompt"] in pilot for t in full_tasks)


def test_sign_formatting(full_tasks):
    """Negative coefficients render with a real minus, never '+ -'."""
    for t in full_tasks:
        assert "+ -" not in t["prompt"] and "− -" not in t["prompt"]
        assert "--" not in t["prompt"]


def test_stage_full_tasks_writes(r3, tmp_path):
    args = types.SimpleNamespace(full_tasks_file=str(tmp_path / "tasks.json"))
    r3.stage_full_tasks(args)
    tasks = json.loads((tmp_path / "tasks.json").read_text())
    assert len(tasks) == 64 and tasks[0]["strategy_space"]


# ── classifier freeze ─────────────────────────────────────────────────────────

def test_classifier_frozen(r3):
    blob = json.dumps({"FAMILIES": r3.FAMILIES, "SPACES": r3.SPACES},
                      sort_keys=True)
    assert hashlib.sha256(blob.encode()).hexdigest() == CLASSIFIER_SHA256, (
        "FAMILIES/SPACES changed — the lexical classifier was SEALED with the "
        "pilot prereg and must not drift into the full run")


# ── strategy entropy ──────────────────────────────────────────────────────────

def test_strategy_entropy_known_values(r3):
    H = r3.strategy_entropy
    assert H(["formula"] * 8) == 0.0
    assert H(["formula"] * 4 + ["telescoping"] * 4) == pytest.approx(1.0)
    assert H(["a", "b", "c", "d"]) == pytest.approx(2.0)
    # unclassified excluded by default …
    assert H(["formula", "formula", "unclassified"]) == 0.0
    # … and <2 labelled → None (a single label is not a distribution)
    assert H(["formula", "unclassified", "unclassified"]) is None
    assert H([]) is None
    # sensitivity variant counts unclassified as its own label
    assert H(["formula", "unclassified"], include_unclassified=True) == \
        pytest.approx(1.0)


# ── batch builder (resume safety + E9.1 homogeneity contract) ─────────────────

def test_build_batches_resume_and_homogeneity(r3, full_tasks):
    tasks = full_tasks[:5]
    cells = r3.build_cells([0.6, 1.2], [1.0], "subtract")
    assert len(cells) == 3
    done = {(tasks[0]["task_id"], "vanilla_T0.6", 0),
            (tasks[2]["task_id"], "pump_subtract_a1", 1)}
    batches = r3._build_full_batches(tasks, cells, [0, 1], done, batch_size=2)
    total = sum(len(b) for _, _, b in batches)
    assert total == len(tasks) * len(cells) * 2 - len(done)
    for cell, seed, batch in batches:
        assert 1 <= len(batch) <= 2
        # a batch never mixes cells or seeds (the pilot could straddle on
        # resume; the full builder groups per (cell, seed) explicitly)
        assert all((t["task_id"], cell["cell"], seed) not in done for t in batch)
    # rerun with everything done → nothing to do
    all_done = {(t["task_id"], c["cell"], s)
                for t in tasks for c in cells for s in (0, 1)}
    assert r3._build_full_batches(tasks, cells, [0, 1], all_done, 2) == []


def test_build_cells_default_grid(r3):
    cells = r3.build_cells(r3.FULL_TEMPS, r3.FULL_ALPHAS, "subtract")
    names = [c["cell"] for c in cells]
    assert len(cells) == 7 and "vanilla_T0.6" in names
    for c in cells:
        if c["method"] == "single_direction":
            assert c["temperature"] == r3.PUMP_T and c["mode"] == "subtract"
            assert c["behaviour"] == "backtracking"


def test_atomic_write(r3, tmp_path):
    path = tmp_path / "gen.json"
    r3._atomic_write_json(path, [{"a": 1}])
    assert json.loads(path.read_text()) == [{"a": 1}]
    r3._atomic_write_json(path, [{"a": 1}, {"a": 2}])
    assert len(json.loads(path.read_text())) == 2
    assert not path.with_suffix(".json.tmp").exists()


# ── matched-entropy gap (P-R2.1 reframed) ─────────────────────────────────────

def _pts(levels_x_y):
    return [{"level": l, "cell": f"c{l}", "strategy_entropy": x, "y": y}
            for l, x, y in levels_x_y]


def test_matched_gap_supported_direction(r3):
    pump = _pts([(0.0, 0.2, 0.9), (1.0, 0.8, 0.8)])
    thermo = _pts([(0.6, 0.1, 0.7), (1.2, 0.9, 0.3)])
    out = r3._matched_entropy_gap(pump, thermo)
    assert out["mean_gap_pump_minus_thermo"] > 0
    assert "SUPPORTED" in out["verdict"]


def test_matched_gap_no_overlap_is_kill(r3):
    pump = _pts([(0.0, 0.0, 0.9), (1.0, 0.2, 0.8)])
    thermo = _pts([(0.6, 0.5, 0.7), (1.2, 0.9, 0.3)])
    out = r3._matched_entropy_gap(pump, thermo)
    assert "status" in out and "overlap" in out["status"]


def test_matched_gap_needs_two_points(r3):
    out = r3._matched_entropy_gap(_pts([(0, 0.1, 0.5)]),
                                  _pts([(0, 0.1, 0.5), (1, 0.5, 0.4)]))
    assert "status" in out


# ── full-analyse end-to-end on synthetic rows ─────────────────────────────────

#: one high-signal keyword per strategy family (from the sealed FAMILIES lists)
KW = {"recursion": "recurrence", "pattern": "pattern", "casework": "casework",
      "formula": "formula", "telescoping": "telescoping",
      "induction": "induction", "substitution": "substitution",
      "elimination": "eliminate", "matrix": "matrix", "modular": "modulo",
      "complement": "complement", "vieta": "vieta", "identity": "expand",
      "explicit_roots": "quadratic formula"}


def _chain(strategy, ans):
    kw = KW[strategy]
    return (f"Let me solve this. I will use {kw}. Applying {kw} carefully "
            f"gives the result.\nThe answer is \\boxed{{{ans}}}")


def test_full_analyse_end_to_end(r3, full_tasks, tmp_path):
    tasks = full_tasks[:8] + full_tasks[32:40]   # 8 easy + 8 hard
    cells = r3.build_cells([0.6, 1.2], [1.0], "subtract")
    rows = []
    for c in cells:
        for t in tasks:
            space = t["strategy_space"]
            for s in range(4):
                # anchor: pure strategy (H=0); other cells: 2-way even (H=1)
                strat = space[0] if c["cell"] == "vanilla_T0.6" \
                    else space[s % 2]
                ans = t["gold"] if s < 3 else str(int(t["gold"]) + 1)
                rows.append({"task_id": t["task_id"], "template": t["template"],
                             "difficulty": t["difficulty"], "gold": t["gold"],
                             "cell": c["cell"], "method": c["method"],
                             "behaviour": c["behaviour"], "mode": c["mode"],
                             "layer": c["layer"], "alpha": c["alpha"],
                             "temperature": c["temperature"], "sample": s,
                             "seed": 1, "max_new": 6144,
                             "chain": _chain(strat, ans), "n_tokens": 40})
    (tmp_path / "full_gen.json").write_text(json.dumps(rows))
    args = types.SimpleNamespace(out=str(tmp_path), full_tasks_file="unused")
    r3.stage_full_analyse(args)

    rep = json.loads((tmp_path / "full_report.json").read_text())
    assert (tmp_path / "FULL_REPORT.md").exists()
    assert rep["n_rows"] == len(rows)

    anchor = rep["cells"]["vanilla_T0.6"]
    hot = rep["cells"]["vanilla_T1.2"]
    pump = rep["cells"]["pump_subtract_a1"]
    # planted distributions: anchor pure (H=0), others 2-way even (H=1 bit)
    assert anchor["strategy_entropy"] == pytest.approx(0.0)
    assert hot["strategy_entropy"] == pytest.approx(1.0)
    assert pump["strategy_entropy"] == pytest.approx(1.0)
    # planted 3/4 correct everywhere; every chain parses + is classified
    for m in (anchor, hot, pump):
        assert m["accuracy"] == pytest.approx(0.75)
        assert m["parse_rate"] == 1.0 and m["coverage_answered"] == 1.0
        assert m["collapse"] == 0.0 and m["cap_hit"] == 0.0
    assert anchor["multi_strategy_frac"] == 0.0
    assert hot["multi_strategy_frac"] == 1.0

    # frontier: pump = anchor + α-cell, thermostat = both vanilla cells
    assert [p["cell"] for p in rep["pump_frontier"]] == \
        ["vanilla_T0.6", "pump_subtract_a1"]
    assert [p["cell"] for p in rep["thermo_frontier"]] == \
        ["vanilla_T0.6", "vanilla_T1.2"]
    assert "verdict" in rep["P_R2_1_reframed"] or \
        "status" in rep["P_R2_1_reframed"]

    # difficulty strata present for both strata of every cell
    for name in rep["cells"]:
        assert set(rep["difficulty_strata"][name]) == {"easy", "hard"}
        for m in rep["difficulty_strata"][name].values():
            assert m["accuracy"] == pytest.approx(0.75)

    # CF-U cross-tab covers every planted strategy with sane counts
    xt = rep["strategy_correctness_xtab"]
    assert all(st["n"] > 0 and 0 <= st["accuracy"] <= 1
               for tpl in xt.values() for st in tpl.values())


def test_full_analyse_handles_unparsed_and_unclassified(r3, full_tasks, tmp_path):
    """Chains with no \\boxed and no keywords must not crash the analysis and
    must count as incorrect / unclassified."""
    t = full_tasks[0]
    rows = [{"task_id": t["task_id"], "template": t["template"],
             "difficulty": t["difficulty"], "gold": t["gold"],
             "cell": "vanilla_T0.6", "method": "vanilla", "behaviour": "shared",
             "mode": None, "layer": None, "alpha": 0.0, "temperature": 0.6,
             "sample": s, "seed": 1, "max_new": 6144,
             "chain": "I ran out of ideas entirely and just stopped",
             "n_tokens": 9999} for s in range(3)]
    (tmp_path / "full_gen.json").write_text(json.dumps(rows))
    args = types.SimpleNamespace(out=str(tmp_path), full_tasks_file="unused")
    r3.stage_full_analyse(args)
    rep = json.loads((tmp_path / "full_report.json").read_text())
    m = rep["cells"]["vanilla_T0.6"]
    assert m["accuracy"] == 0.0 and m["parse_rate"] == 0.0
    assert m["strategy_entropy"] is None          # nothing labelled
    assert m["cap_hit"] == 1.0                    # 9999 ≥ max_new
