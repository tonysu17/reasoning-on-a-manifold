"""Tests for the two Phase-7 metrics added 2026-06-21 (plan item E1):

  * off-target leakage   — folded into aggregate_results' per-cell summary
  * task-accuracy preservation — aggregate_accuracy(), consumes an externally
    supplied correctness map (no model / no judge here, just the aggregation).

Uses small synthetic annotation/correctness lists — no model, no API.
"""

import pytest

from src.evaluation import aggregate_accuracy, aggregate_results

# Full four-behaviour basis so "the other three" is exercised.
BEHS = ["backtracking", "uncertainty-estimation", "example-testing",
        "adding-knowledge"]


def _steered(beh, method, alpha, tid="T1", chain="some generated text here ok",
             n_tokens=64):
    return {"behaviour": beh, "method": method, "alpha": alpha,
            "task_id": tid, "chain": chain, "n_tokens": n_tokens, "layer": 27}


def _annotated(beh, method, alpha, labels, tid="T1"):
    return {"behaviour": beh, "method": method, "alpha": alpha, "task_id": tid,
            "annotations": [{"label": l, "text": f"s{i}"}
                            for i, l in enumerate(labels)]}


# --------------------------------------------------------------------------
# Off-target leakage
# --------------------------------------------------------------------------

def test_leakage_reports_the_other_three_behaviours():
    # Steering backtracking; the chain also contains the other targets.
    steered = [_steered("backtracking", "single_direction", 1.0)]
    labels = ["backtracking", "backtracking",          # on-target = 2/8
              "uncertainty-estimation",                # 1/8
              "example-testing", "example-testing",    # 2/8
              "adding-knowledge",                      # 1/8
              "deduction", "deduction"]                # off-basis, ignored
    annotated = [_annotated("backtracking", "single_direction", 1.0, labels)]
    s = aggregate_results(steered, annotated, target_behaviours=BEHS)
    cell = s["backtracking"]["single_direction"][1.0]

    assert cell["mean"] == 2 / 8                        # on-target unchanged
    leak = cell["leakage_by_behaviour"]
    # exactly the OTHER three targets, never the steered behaviour itself
    assert set(leak) == {"uncertainty-estimation", "example-testing",
                         "adding-knowledge"}
    assert "backtracking" not in leak
    assert leak["uncertainty-estimation"] == 1 / 8
    assert leak["example-testing"] == 2 / 8
    assert leak["adding-knowledge"] == 1 / 8
    # leakage_mean = mean over the other three of their fractions
    assert cell["leakage_mean"] == pytest.approx((1 / 8 + 2 / 8 + 1 / 8) / 3)


def test_leakage_averages_over_chains():
    # Two tasks; uncertainty-estimation fraction is 0.5 then 0.0 -> mean 0.25.
    steered = [
        _steered("backtracking", "manifold_projected", 1.0, tid="T1"),
        _steered("backtracking", "manifold_projected", 1.0, tid="T2"),
    ]
    annotated = [
        _annotated("backtracking", "manifold_projected", 1.0,
                   ["backtracking", "uncertainty-estimation"], tid="T1"),
        _annotated("backtracking", "manifold_projected", 1.0,
                   ["backtracking", "backtracking"], tid="T2"),
    ]
    s = aggregate_results(steered, annotated, target_behaviours=BEHS)
    cell = s["backtracking"]["manifold_projected"][1.0]
    assert cell["n"] == 2
    assert cell["leakage_by_behaviour"]["uncertainty-estimation"] == 0.25
    assert cell["leakage_by_behaviour"]["example-testing"] == 0.0


def test_leakage_none_when_no_scored_chain():
    # Empty annotation -> on-target mean is None; leakage must mirror that.
    steered = [_steered("backtracking", "single_direction", 2.0)]
    annotated = [_annotated("backtracking", "single_direction", 2.0, [])]
    s = aggregate_results(steered, annotated, target_behaviours=BEHS)
    cell = s["backtracking"]["single_direction"][2.0]
    assert cell["mean"] is None
    assert cell["leakage_mean"] is None
    assert cell["leakage_by_behaviour"] == {}


def test_leakage_for_shared_vanilla_uses_remaining_targets():
    # Shared baseline expands to every behaviour; for the backtracking series
    # its leakage is the other three, for the uncertainty series it is the
    # remaining three (which now INCLUDES backtracking).
    steered = [_steered("shared", "vanilla", 0.0)]
    annotated = [_annotated("shared", "vanilla", 0.0,
                            ["backtracking", "uncertainty-estimation",
                             "example-testing", "adding-knowledge"])]
    s = aggregate_results(steered, annotated, target_behaviours=BEHS)

    bt = s["backtracking"]["vanilla"][0.0]
    assert bt["mean"] == 0.25                           # 1/4 backtracking
    assert set(bt["leakage_by_behaviour"]) == {
        "uncertainty-estimation", "example-testing", "adding-knowledge"}
    assert all(v == 0.25 for v in bt["leakage_by_behaviour"].values())

    ue = s["uncertainty-estimation"]["vanilla"][0.0]
    assert "backtracking" in ue["leakage_by_behaviour"]
    assert "uncertainty-estimation" not in ue["leakage_by_behaviour"]


def test_leakage_does_not_break_aggregate_schema():
    # The pre-existing fields must still be present and correct alongside leakage.
    steered = [_steered("backtracking", "random_direction", 1.0)]
    annotated = [_annotated("backtracking", "random_direction", 1.0,
                            ["deduction", "backtracking"])]
    s = aggregate_results(steered, annotated, target_behaviours=BEHS)
    cell = s["backtracking"]["random_direction"][1.0]
    assert cell["mean"] == 0.5
    assert cell["n_missing"] == 0 and cell["n_empty"] == 0
    assert "leakage_mean" in cell and "leakage_by_behaviour" in cell
    # no off-basis "deduction" leaks into the leakage dict
    assert "deduction" not in cell["leakage_by_behaviour"]


# --------------------------------------------------------------------------
# Task-accuracy preservation
# --------------------------------------------------------------------------

def test_accuracy_basic_aggregation():
    steered = [
        _steered("backtracking", "single_direction", 1.0, tid="T1"),
        _steered("backtracking", "single_direction", 1.0, tid="T2"),
        _steered("backtracking", "single_direction", 1.0, tid="T3"),
    ]
    correctness = {
        ("T1", "backtracking", "single_direction", 1.0): True,
        ("T2", "backtracking", "single_direction", 1.0): True,
        ("T3", "backtracking", "single_direction", 1.0): False,
    }
    s = aggregate_accuracy(steered, correctness, target_behaviours=BEHS)
    cell = s["backtracking"]["single_direction"][1.0]
    assert cell["n"] == 3
    assert cell["n_correct"] == 2
    assert cell["accuracy"] == pytest.approx(2 / 3)
    assert cell["n_missing"] == 0


def test_accuracy_missing_label_skipped_and_counted():
    steered = [
        _steered("backtracking", "single_direction", 1.0, tid="T1"),
        _steered("backtracking", "single_direction", 1.0, tid="T2"),
    ]
    # T2 has no correctness label -> skipped, counted, NOT scored wrong.
    correctness = {("T1", "backtracking", "single_direction", 1.0): True}
    s = aggregate_accuracy(steered, correctness, target_behaviours=BEHS)
    cell = s["backtracking"]["single_direction"][1.0]
    assert cell["n"] == 1                  # T2 skipped
    assert cell["accuracy"] == 1.0         # would be 0.5 if missing scored wrong
    assert cell["n_missing"] == 1


def test_accuracy_all_missing_cell_survives_with_none():
    steered = [_steered("backtracking", "single_direction", 3.0)]
    s = aggregate_accuracy(steered, correctness={}, target_behaviours=BEHS)
    cell = s["backtracking"]["single_direction"][3.0]
    assert cell["accuracy"] is None
    assert cell["n"] == 0 and cell["n_missing"] == 1
    assert cell["accuracy_drop_vs_vanilla"] is None


def test_accuracy_drop_vs_explicit_vanilla_map():
    steered = [
        _steered("backtracking", "single_direction", 1.0, tid="T1"),
        _steered("backtracking", "single_direction", 1.0, tid="T2"),
    ]
    correctness = {
        ("T1", "backtracking", "single_direction", 1.0): True,
        ("T2", "backtracking", "single_direction", 1.0): False,   # cell acc 0.5
    }
    vanilla = {"T1": True, "T2": True}                              # vanilla 1.0
    s = aggregate_accuracy(steered, correctness, vanilla_correctness=vanilla,
                           target_behaviours=BEHS)
    cell = s["backtracking"]["single_direction"][1.0]
    assert cell["accuracy"] == 0.5
    assert cell["accuracy_drop_vs_vanilla"] == pytest.approx(0.5)


def test_accuracy_drop_is_paired_on_shared_tasks_only():
    # Cell scored T1,T2,T3 but vanilla only has T1,T2 -> drop uses {T1,T2}.
    steered = [
        _steered("backtracking", "single_direction", 1.0, tid="T1"),
        _steered("backtracking", "single_direction", 1.0, tid="T2"),
        _steered("backtracking", "single_direction", 1.0, tid="T3"),
    ]
    correctness = {
        ("T1", "backtracking", "single_direction", 1.0): True,
        ("T2", "backtracking", "single_direction", 1.0): False,
        ("T3", "backtracking", "single_direction", 1.0): False,
    }
    vanilla = {"T1": True, "T2": True}  # no T3 baseline
    s = aggregate_accuracy(steered, correctness, vanilla_correctness=vanilla,
                           target_behaviours=BEHS)
    cell = s["backtracking"]["single_direction"][1.0]
    assert cell["accuracy"] == pytest.approx(1 / 3)        # over all 3 tasks
    # drop is the PAIRED comparison on {T1,T2}: vanilla 1.0 vs cell 0.5
    assert cell["accuracy_drop_vs_vanilla"] == pytest.approx(0.5)


def test_accuracy_vanilla_baseline_reconstructed_from_shared_cells():
    # No explicit vanilla map: the shared-vanilla cell supplies the per-task
    # baseline and expands to every behaviour.
    steered = [
        _steered("shared", "vanilla", 0.0, tid="T1"),
        _steered("backtracking", "single_direction", 1.0, tid="T1"),
    ]
    correctness = {
        ("T1", "shared", "vanilla", 0.0): True,                    # baseline
        ("T1", "backtracking", "single_direction", 1.0): False,    # steered
    }
    s = aggregate_accuracy(steered, correctness, target_behaviours=BEHS)
    # shared vanilla expanded to every behaviour's vanilla series
    assert s["backtracking"]["vanilla"][0.0]["accuracy"] == 1.0
    assert s["adding-knowledge"]["vanilla"][0.0]["accuracy"] == 1.0
    # steered cell drops vs the reconstructed T1 baseline
    sd = s["backtracking"]["single_direction"][1.0]
    assert sd["accuracy"] == 0.0
    assert sd["accuracy_drop_vs_vanilla"] == pytest.approx(1.0)
    # the vanilla cell is its own baseline -> zero drop
    assert s["backtracking"]["vanilla"][0.0]["accuracy_drop_vs_vanilla"] == 0.0


# --------------------------------------------------------------------------
# Pre-flight QA (2026-06-23): accuracy reconstruction + pairing edges.
# --------------------------------------------------------------------------

def test_accuracy_vanilla_reconstructed_from_per_behaviour_vanilla_cells():
    """Reconstruct path with NON-shared vanilla records (method=='vanilla',
    behaviour=='backtracking'): the per-task baseline is taken from those cells
    and the vanilla cell drops zero against itself."""
    steered = [
        _steered("backtracking", "vanilla", 0.0, tid="T1"),
        _steered("backtracking", "single_direction", 1.0, tid="T1"),
    ]
    correctness = {
        ("T1", "backtracking", "vanilla", 0.0): True,
        ("T1", "backtracking", "single_direction", 1.0): False,
    }
    s = aggregate_accuracy(steered, correctness, target_behaviours=BEHS)
    assert s["backtracking"]["vanilla"][0.0]["accuracy_drop_vs_vanilla"] == 0.0
    assert s["backtracking"]["single_direction"][1.0]["accuracy_drop_vs_vanilla"] == pytest.approx(1.0)


def test_accuracy_drop_none_when_no_vanilla_anywhere():
    """A steered cell with NO vanilla map and NO vanilla cells to reconstruct from
    must report accuracy but a None drop (no paired baseline) — not a 0 or crash."""
    steered = [
        _steered("backtracking", "single_direction", 1.0, tid="T1"),
        _steered("backtracking", "single_direction", 1.0, tid="T2"),
    ]
    correctness = {
        ("T1", "backtracking", "single_direction", 1.0): True,
        ("T2", "backtracking", "single_direction", 1.0): False,
    }
    s = aggregate_accuracy(steered, correctness, target_behaviours=BEHS)
    cell = s["backtracking"]["single_direction"][1.0]
    assert cell["accuracy"] == pytest.approx(0.5)
    assert cell["accuracy_drop_vs_vanilla"] is None


def test_accuracy_drop_skips_unpaired_steered_tasks():
    """Steered cell scored on {T1,T2,T3}; vanilla only on {T1} -> the paired drop
    is computed on {T1} alone, NOT the mean-of-means over all three."""
    steered = [_steered("backtracking", "single_direction", 1.0, tid=t)
               for t in ("T1", "T2", "T3")]
    correctness = {
        ("T1", "backtracking", "single_direction", 1.0): False,
        ("T2", "backtracking", "single_direction", 1.0): True,
        ("T3", "backtracking", "single_direction", 1.0): True,
    }
    vanilla = {"T1": True}            # only T1 has a baseline
    s = aggregate_accuracy(steered, correctness, vanilla_correctness=vanilla,
                           target_behaviours=BEHS)
    cell = s["backtracking"]["single_direction"][1.0]
    assert cell["accuracy"] == pytest.approx(2 / 3)           # over all 3
    # paired drop on {T1} only: vanilla 1.0 - cell 0.0 = 1.0 (NOT 1.0 - 2/3)
    assert cell["accuracy_drop_vs_vanilla"] == pytest.approx(1.0)


def test_accuracy_shared_vanilla_missing_label_counts_all_behaviours():
    """A MISSING shared-vanilla correctness label increments n_missing for every
    expanded behaviour (each behaviour's vanilla series lacks that task), and is
    never scored as wrong."""
    steered = [_steered("shared", "vanilla", 0.0, tid="T1")]
    s = aggregate_accuracy(steered, correctness={}, target_behaviours=BEHS)
    for beh in BEHS:
        cell = s[beh]["vanilla"][0.0]
        assert cell["accuracy"] is None
        assert cell["n"] == 0 and cell["n_missing"] == 1
