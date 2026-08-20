"""Offline tests for the PT-B1 executor (no torch, no network, no spend)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import ptb1_executor as X


# ── arm table + gate bands ───────────────────────────────────────────────────

def test_arm_table_is_the_sealed_six():
    assert len(X.ARMS) == 6
    assert set(X.SAFETY_ARMS) == {f"safety1000-s{s}" for s in (42, 43, 44)}
    assert set(X.CONTROL_ARMS) == {f"control1000-s{s}" for s in (42, 43, 44)}
    for arm, (data, seed, stored) in X.ARMS.items():
        assert (X.ROOT / data).exists(), data
        assert (X.ROOT / stored / "row_index.json").exists(), stored
        assert str(seed) in arm
        assert X.opposite_arm(X.opposite_arm(arm)) == arm
        assert X.opposite_arm(arm) in X.ARMS
    safety_data = {X.ARMS[a][0] for a in X.SAFETY_ARMS}
    control_data = {X.ARMS[a][0] for a in X.CONTROL_ARMS}
    assert safety_data == {"data/safety_star1_sft.json"}
    assert control_data == {"data/control_generic_sft.json"}


@pytest.mark.parametrize("same,opp,cov,verdict", [
    ({"12": 0.99, "16": 0.985}, {"12": 0.1, "16": 0.2}, 1.0, "PASS"),
    ({"12": 0.99, "16": 0.97}, {"12": 0.1, "16": 0.2}, 1.0, "AMENDED"),
    ({"12": 0.96, "16": 0.955}, {"12": 0.1, "16": 0.2}, 1.0, "AMENDED"),
    ({"12": 0.99, "16": 0.94}, {"12": 0.1, "16": 0.2}, 1.0, "STOP"),
    ({"12": 0.99, "16": 0.99}, {"12": 0.6, "16": 0.2}, 1.0, "STOP"),
    ({"12": 0.99, "16": 0.99}, {"12": 0.1, "16": 0.2}, 0.90, "STOP"),
])
def test_gate_verdict_bands(same, opp, cov, verdict):
    assert X.gate_verdict(same, opp, cov)["verdict"] == verdict


# ── row-index helpers ────────────────────────────────────────────────────────

def _fake_act_dir(tmp_path: Path, rows: dict, dim: int = 4) -> Path:
    d = tmp_path
    ri = {"version": "x", "rows": {b: [{"chain_id": c, "annotation_index": a}
                                       for c, a in keys]
                                   for b, keys in rows.items()}}
    (d / "row_index.json").write_text(json.dumps(ri))
    rng = np.random.default_rng(0)
    for b, keys in rows.items():
        for layer in X.GATE_LAYERS:
            np.save(d / f"{b}_layer{layer}.npy",
                    rng.normal(size=(len(keys), dim)).astype(np.float32))
    return d


def test_row_key_helpers_roundtrip(tmp_path):
    rows = {b: [("MATH_001", 0), ("MATH_002", 3)] for b in X.BEHAVIOURS}
    d = _fake_act_dir(tmp_path, rows)
    keys = X.load_row_keys(d)
    assert keys["backtracking"] == [("MATH_001", 0), ("MATH_002", 3)]
    pos = X.key_positions(keys)
    assert pos[("backtracking", "MATH_002", 3)] == 1
    wanted = [("backtracking", "MATH_002", 3), ("example-testing", "MATH_001", 0)]
    mat = X.gather_rows(d, X.GATE_LAYERS[0], wanted)
    ref_bt = np.load(d / f"backtracking_layer{X.GATE_LAYERS[0]}.npy")
    ref_ex = np.load(d / f"example-testing_layer{X.GATE_LAYERS[0]}.npy")
    assert np.allclose(mat[0], ref_bt[1]) and np.allclose(mat[1], ref_ex[0])


# ── estimand: class merge equals mean over resolved seed rows ────────────────

def test_class_cell_merge_equals_seed_mean():
    cells = {
        "safety1000-s42": {"T1": {"values": [0.10], "n_missing": 0}},
        "safety1000-s43": {"T1": {"values": [0.30], "n_missing": 0}},
        "safety1000-s44": {"T1": {"values": [], "n_missing": 1}},
    }
    merged = X.class_cell(cells, X.SAFETY_ARMS)
    assert merged["T1"]["n_missing"] == 1
    assert np.isclose(np.mean(merged["T1"]["values"]), 0.20)


# ── analyse stage end-to-end on synthetic shards ─────────────────────────────

def _ann_row(task, frac_bt, n_sent=10, complete=True):
    from src.annotation_coverage import COVERAGE_RULE_VERSION
    n_bt = round(frac_bt * n_sent)
    anns = ([{"label": "backtracking", "text": "w"}] * n_bt
            + [{"label": "other", "text": "w"}] * (n_sent - n_bt))
    return {"method": "vanilla", "behaviour": "shared", "alpha": 0.0,
            "task_id": task, "base_task_id": task, "chain": "x", "n_tokens": 100,
            "annotations": anns if complete else None,
            "annotation_complete": complete,
            "annotation_coverage": {"rule_version": COVERAGE_RULE_VERSION,
                                    "complete": complete},
            "annotation_coverage_complete": complete,
            "annotated_region_tokens": 1000}


def _gen_row(task, n_tokens, chain="alpha beta gamma delta " * 40):
    return {"method": "vanilla", "task_id": task, "chain": chain,
            "n_tokens": n_tokens}


def test_stage_analyse_synthetic(tmp_path, monkeypatch):
    tasks = [f"T{i}" for i in range(8)]
    ann_dir = tmp_path / "annotation"; ann_dir.mkdir(parents=True)
    bat_dir = tmp_path / "battery"; bat_dir.mkdir(parents=True)
    # Safety arms: bt fraction 0.2; control arms: 0.5 -> diff -0.3 exactly.
    for arm in X.ARMS:
        frac = 0.2 if arm.startswith("safety") else 0.5
        (ann_dir / f"{arm}.json").write_text(json.dumps(
            [_ann_row(t, frac) for t in tasks]))
        n_tok = 1000 if arm.startswith("safety") else 2000
        (bat_dir / f"{arm}.json").write_text(json.dumps(
            [_gen_row(t, n_tok) for t in tasks]))
    monkeypatch.setattr(X, "OUT", tmp_path)
    monkeypatch.setattr(X, "write_provenance",
                        lambda *a, **k: None)
    X.stage_analyse(authorised=False)

    out = json.loads((tmp_path / "analysis" / "ptb1_analysis.json").read_text())
    bt = out["primary"]["backtracking"]
    assert np.isclose(bt["complete_case"]["diff_mean"], -0.3)
    assert bt["complete_case"]["n_pairs"] == 8
    # Zero-variance diffs: CI collapses to the point, p floors at 2/(B+1).
    assert bt["complete_case"]["raw_p"] < 0.05
    assert bt["missingness"]["label"].startswith("sign-robust")
    assert bt["verdict"].startswith("resolved nonzero")
    assert np.isclose(out["full_chain"]["length"]["diff_mean"], -1000.0)
    assert out["damage_context_rule"]["tripped"] is True
    grid = out["secondary_per_seed_grid"]["backtracking"]
    assert grid["n_negative"] == 9 and grid["n_positive"] == 0
    report = (tmp_path / "analysis" / "PTB1_REPORT.md").read_text()
    assert "safety-versus-control" in report.lower() or "safety" in report


def test_stage_analyse_unresolved_rows_never_zero_filled(tmp_path, monkeypatch):
    tasks = [f"T{i}" for i in range(4)]
    ann_dir = tmp_path / "annotation"; ann_dir.mkdir(parents=True)
    bat_dir = tmp_path / "battery"; bat_dir.mkdir(parents=True)
    for arm in X.ARMS:
        rows = []
        for i, t in enumerate(tasks):
            # T0 unresolved in EVERY safety arm -> must drop from the pair set,
            # not score 0.
            complete = not (arm.startswith("safety") and i == 0)
            rows.append(_ann_row(t, 0.4, complete=complete))
        (ann_dir / f"{arm}.json").write_text(json.dumps(rows))
        (bat_dir / f"{arm}.json").write_text(json.dumps(
            [_gen_row(t, 1500) for t in tasks]))
    monkeypatch.setattr(X, "OUT", tmp_path)
    monkeypatch.setattr(X, "write_provenance", lambda *a, **k: None)
    X.stage_analyse(authorised=False)
    out = json.loads((tmp_path / "analysis" / "ptb1_analysis.json").read_text())
    bt = out["primary"]["backtracking"]
    assert bt["complete_case"]["n_pairs"] == 3          # T0 pairwise-deleted
    assert np.isclose(bt["complete_case"]["diff_mean"], 0.0)
    assert bt["missingness"]["manski"]["n_tasks_universe"] == 4


# ── battery-record compatibility with the authoritative extractor ────────────

def test_vanilla_rows_feed_per_task_fraction():
    from src.delta_floor import per_task_fraction
    gen = [{"method": "vanilla", "behaviour": "shared", "alpha": 0.0,
            "task_id": "T1", "base_task_id": "T1", "chain": "c", "n_tokens": 5}]
    ann = [_ann_row("T1", 0.3)]
    ann[0]["behaviour"] = "shared"
    out = per_task_fraction(gen, ann, "backtracking", "vanilla", 0.0)
    assert np.isclose(out["T1"], 0.3)


# ── prereg + amendment are committed and hash-stable inputs exist ────────────

def test_sealed_documents_exist():
    assert X.PREREG.exists() and X.AMENDMENT_1.exists()
    text = X.PREREG.read_text()
    for needle in ("--epochs 5", "--lr 1e-5", "--grad-accum 32",
                   "$15", "$60", "$75", "0.98", "0.95"):
        assert needle in text, needle
