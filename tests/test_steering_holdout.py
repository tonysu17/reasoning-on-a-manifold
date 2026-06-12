"""Tests for the true-hold-out steering build: the Phase-7 eval tasks' rows
must never touch the vectors, and 06/07 must agree on the split."""

import json

import numpy as np
import pytest

from src.steering import build_steering_vectors
from src.task_gen import stratified_eval_split


def _make_act_dir(tmp_path, layer=27, d=16):
    """Two behaviours, 3 chains each; chain c_eval is the eval task."""
    act_dir = tmp_path / "acts"
    act_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = {}
    for i, beh in enumerate(["backtracking", "uncertainty-estimation"]):
        # 6 rows per behaviour: chains c0, c0, c1, c1, c_eval, c_eval
        X = rng.standard_normal((6, d)).astype(np.float32)
        # give the eval rows an extreme offset so exclusion visibly moves the mean
        X[4:] += 100.0 * (i + 1)
        np.save(act_dir / f"{beh}_layer{layer}.npy", X)
        rows[beh] = [{"chain_id": c, "annotation_index": j, "char_offset": 0,
                      "token_start": 0, "n_positions": 11}
                     for j, c in enumerate(["c0", "c0", "c1", "c1", "c_eval", "c_eval"])]
    (act_dir / "row_index.json").write_text(json.dumps({
        "version": 1, "sentence_matching": "occurrence_aware_v1",
        "pooling": "mean", "rows": rows,
    }))
    return act_dir


def test_holdout_excludes_eval_rows(tmp_path):
    act_dir = _make_act_dir(tmp_path)
    with_holdout = build_steering_vectors(
        act_dir, layer=27, behaviours=["backtracking", "uncertainty-estimation"],
        k_values=[1], exclude_chain_ids={"c_eval"})
    without = build_steering_vectors(
        act_dir, layer=27, behaviours=["backtracking", "uncertainty-estimation"],
        k_values=[1])
    for beh in with_holdout:
        assert with_holdout[beh]["n_on"] == 4
        assert with_holdout[beh]["n_excluded"] == 2
        assert without[beh]["n_on"] == 6
        assert without[beh]["n_excluded"] == 0
        # the extreme eval rows must change the direction when included
        cos = float(with_holdout[beh]["single_direction"]
                    @ without[beh]["single_direction"])
        assert cos < 0.999


def test_holdout_fails_loud_without_provenance(tmp_path):
    act_dir = _make_act_dir(tmp_path)
    (act_dir / "row_index.json").unlink()  # no sidecar, no annotated file
    with pytest.raises(RuntimeError, match="proxy-chain|chain ids"):
        build_steering_vectors(act_dir, layer=27,
                               behaviours=["backtracking"],
                               k_values=[1], exclude_chain_ids={"c_eval"},
                               annotated_path=tmp_path / "missing.json")


def test_split_is_shared_and_stratified():
    tasks = [{"id": f"{cat}_{i:03d}", "category": cat, "prompt": "p"}
             for cat in ["math", "spatial", "verbal"] for i in range(10)]
    test_tasks, rule = stratified_eval_split(tasks, n_test=6)
    assert len(test_tasks) == 6                      # 2 per category
    cats = {t["category"] for t in test_tasks}
    assert cats == {"math", "spatial", "verbal"}     # never single-category
    # deterministic: same call → same ids (06 and 07 must agree exactly)
    again, _ = stratified_eval_split(tasks, n_test=6)
    assert [t["id"] for t in again] == [t["id"] for t in test_tasks]
    # takes the LAST per category
    assert {t["id"] for t in test_tasks} == {
        "math_008", "math_009", "spatial_008", "spatial_009",
        "verbal_008", "verbal_009"}
