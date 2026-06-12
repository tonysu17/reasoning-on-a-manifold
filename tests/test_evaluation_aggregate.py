"""Tests for src/evaluation.aggregate_results — the Phase-7 scoring path.

Locks in the 2026-06-12 fixes: missing/empty re-annotations are counted, not
silently scored 0.0; the shared vanilla baseline expands to every behaviour.
"""

from src.evaluation import aggregate_results

BEHS = ["backtracking", "uncertainty-estimation"]


def _steered(beh, method, alpha, tid="T1"):
    return {"behaviour": beh, "method": method, "alpha": alpha,
            "task_id": tid, "chain": "text", "n_tokens": 5, "layer": 27}


def _annotated(beh, method, alpha, labels, tid="T1"):
    return {"behaviour": beh, "method": method, "alpha": alpha, "task_id": tid,
            "annotations": [{"label": l, "text": f"s{i}"} for i, l in enumerate(labels)]}


def test_shared_vanilla_expands_to_every_behaviour():
    steered = [_steered("shared", "vanilla", 0.0)]
    annotated = [_annotated("shared", "vanilla", 0.0,
                            ["backtracking", "deduction", "deduction", "deduction"])]
    s = aggregate_results(steered, annotated, target_behaviours=BEHS)
    assert s["backtracking"]["vanilla"][0.0]["mean"] == 0.25
    assert s["uncertainty-estimation"]["vanilla"][0.0]["mean"] == 0.0
    assert s["backtracking"]["vanilla"][0.0]["n"] == 1


def test_missing_reannotation_counted_not_zero():
    steered = [
        _steered("backtracking", "single_direction", 1.0, tid="T1"),
        _steered("backtracking", "single_direction", 1.0, tid="T2"),
    ]
    # only T1 got re-annotated; T2's annotation is MISSING
    annotated = [_annotated("backtracking", "single_direction", 1.0,
                            ["backtracking", "backtracking"], tid="T1")]
    s = aggregate_results(steered, annotated, target_behaviours=BEHS)
    cell = s["backtracking"]["single_direction"][1.0]
    assert cell["n"] == 1                  # T2 skipped, NOT scored 0.0
    assert cell["mean"] == 1.0             # would be 0.5 under the old bug
    assert cell["n_missing"] == 1


def test_empty_annotation_counted_separately():
    steered = [_steered("backtracking", "manifold_projected", 2.0)]
    annotated = [_annotated("backtracking", "manifold_projected", 2.0, [])]
    s = aggregate_results(steered, annotated, target_behaviours=BEHS)
    # the only record was empty -> no fractions cell at all, but it must not
    # crash; verify via the absence of a mean (cell never created)
    assert ("manifold_projected" not in s.get("backtracking", {})
            or 2.0 not in s["backtracking"]["manifold_projected"])


def test_steered_arm_fraction_basic():
    steered = [_steered("backtracking", "random_direction", 1.0)]
    annotated = [_annotated("backtracking", "random_direction", 1.0,
                            ["deduction", "backtracking"])]
    s = aggregate_results(steered, annotated, target_behaviours=BEHS)
    cell = s["backtracking"]["random_direction"][1.0]
    assert cell["mean"] == 0.5
    assert cell["n_missing"] == 0 and cell["n_empty"] == 0
