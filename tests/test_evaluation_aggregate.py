"""Tests for src/evaluation.aggregate_results — the Phase-7 scoring path.

Locks in the 2026-06-12 fixes: missing/empty re-annotations are counted, not
silently scored 0.0; the shared vanilla baseline expands to every behaviour.
"""

from src.evaluation import aggregate_results

BEHS = ["backtracking", "uncertainty-estimation"]


def _steered(beh, method, alpha, tid="T1", chain="some generated text here ok",
             n_tokens=64):
    return {"behaviour": beh, "method": method, "alpha": alpha,
            "task_id": tid, "chain": chain, "n_tokens": n_tokens, "layer": 27}


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


def test_empty_annotation_cell_survives_with_counts():
    """A cell whose re-annotations ALL failed must still appear in the summary
    (n=0 + counts + generation metrics) — total annotation failure concentrates
    in exactly the most-destructive arm the accounting exists to expose."""
    steered = [_steered("backtracking", "manifold_projected", 2.0)]
    annotated = [_annotated("backtracking", "manifold_projected", 2.0, [])]
    s = aggregate_results(steered, annotated, target_behaviours=BEHS)
    cell = s["backtracking"]["manifold_projected"][2.0]
    assert cell["mean"] is None and cell["std"] is None
    assert cell["n"] == 0
    assert cell["n_empty"] == 1 and cell["n_missing"] == 0
    # generation metrics computed from the chain itself, no annotation needed
    assert cell["mean_n_tokens"] == 64
    assert 0.0 <= cell["repetition_rate"] <= 1.0
    assert cell["degenerate_rate"] == 0.0


def test_all_missing_cell_survives():
    steered = [_steered("backtracking", "single_direction", 3.0)]
    annotated = []  # nothing came back from the annotator
    s = aggregate_results(steered, annotated, target_behaviours=BEHS)
    cell = s["backtracking"]["single_direction"][3.0]
    assert cell["mean"] is None and cell["n"] == 0
    assert cell["n_missing"] == 1
    assert cell["mean_n_tokens"] == 64


def test_degenerate_output_flagged():
    steered = [_steered("backtracking", "single_direction", 2.0,
                        chain="a a a a a a a a", n_tokens=8)]
    annotated = [_annotated("backtracking", "single_direction", 2.0,
                            ["backtracking"])]
    s = aggregate_results(steered, annotated, target_behaviours=BEHS)
    cell = s["backtracking"]["single_direction"][2.0]
    assert cell["degenerate_rate"] == 1.0          # n_tokens < 32
    # "a a a a a a a a": 5 four-grams, 1 distinct -> 1 - 1/5
    assert abs(cell["repetition_rate"] - 0.8) < 1e-9


def test_steered_arm_fraction_basic():
    steered = [_steered("backtracking", "random_direction", 1.0)]
    annotated = [_annotated("backtracking", "random_direction", 1.0,
                            ["deduction", "backtracking"])]
    s = aggregate_results(steered, annotated, target_behaviours=BEHS)
    cell = s["backtracking"]["random_direction"][1.0]
    assert cell["mean"] == 0.5
    assert cell["n_missing"] == 0 and cell["n_empty"] == 0


# --------------------------------------------------------------------------
# Pre-flight QA (2026-06-23): edges that gate the real Phase-7 run.
# --------------------------------------------------------------------------

def test_mixed_n_tokens_in_cell_averages_only_present():
    """A cell with one n_tokens present and one None must average mean_n_tokens /
    degenerate_rate over ONLY the present record (no None→0 contamination), while
    still scoring both annotations on-target."""
    steered = [
        _steered("backtracking", "single_direction", 1.0, tid="T1", n_tokens=64),
        _steered("backtracking", "single_direction", 1.0, tid="T2", n_tokens=None),
    ]
    annotated = [
        _annotated("backtracking", "single_direction", 1.0, ["backtracking"], tid="T1"),
        _annotated("backtracking", "single_direction", 1.0, ["backtracking"], tid="T2"),
    ]
    s = aggregate_results(steered, annotated, target_behaviours=BEHS)
    cell = s["backtracking"]["single_direction"][1.0]
    assert cell["n"] == 2                       # both scored on-target
    assert cell["mean_n_tokens"] == 64.0        # only the present token count
    assert cell["degenerate_rate"] == 0.0       # over the one present (64 >= 32)


def test_cell_survives_on_gen_rep_alone():
    """All n_tokens None AND empty annotation: only repetition (always computed
    from the chain) keeps the cell in the union. It must still appear with n=0,
    counts, and a repetition_rate — but NO mean_n_tokens/degenerate_rate keys."""
    steered = [_steered("backtracking", "manifold_projected", 2.0, n_tokens=None)]
    annotated = [_annotated("backtracking", "manifold_projected", 2.0, [])]
    s = aggregate_results(steered, annotated, target_behaviours=BEHS)
    cell = s["backtracking"]["manifold_projected"][2.0]
    assert cell["n"] == 0 and cell["n_empty"] == 1
    assert cell["mean"] is None
    assert "repetition_rate" in cell
    assert "mean_n_tokens" not in cell and "degenerate_rate" not in cell


def test_repetition_rate_short_text_is_one():
    """The documented <n-gram edge: text shorter than the 4-gram window scores
    1.0. degenerate_rate (token floor) is the independent terse-output signal."""
    from src.evaluation import repetition_rate
    assert repetition_rate("a b c") == 1.0      # 3 tokens < n=4
    assert repetition_rate("a b c d") == 0.0    # exactly the window, all distinct
    assert repetition_rate("") == 1.0


def test_leakage_mean_safe_when_all_others_absent():
    """Steering backtracking on a chain containing ONLY the steered behaviour and
    off-basis labels: leakage_by_behaviour collapses each other target to 0.0
    (present-but-zero), leakage_mean = 0.0 — never a div-by-zero/NaN."""
    steered = [_steered("backtracking", "single_direction", 1.0)]
    annotated = [_annotated("backtracking", "single_direction", 1.0,
                            ["backtracking", "deduction", "deduction"])]
    s = aggregate_results(steered, annotated, target_behaviours=BEHS)
    cell = s["backtracking"]["single_direction"][1.0]
    assert cell["leakage_by_behaviour"]["uncertainty-estimation"] == 0.0
    assert cell["leakage_mean"] == 0.0


def test_union_key_covers_all_methods_for_a_failed_alpha():
    """At one alpha, single fully fails annotation while manifold succeeds. Both
    cells must appear (the union of keys), so a fully-destructive arm cannot
    vanish from the table by producing only annotation failures."""
    steered = [
        _steered("backtracking", "single_direction", 2.0, tid="T1"),
        _steered("backtracking", "manifold_projected", 2.0, tid="T1"),
    ]
    annotated = [
        _annotated("backtracking", "single_direction", 2.0, [], tid="T1"),  # failed
        _annotated("backtracking", "manifold_projected", 2.0, ["backtracking"], tid="T1"),
    ]
    s = aggregate_results(steered, annotated, target_behaviours=BEHS)
    assert s["backtracking"]["single_direction"][2.0]["mean"] is None
    assert s["backtracking"]["single_direction"][2.0]["n_empty"] == 1
    assert s["backtracking"]["manifold_projected"][2.0]["mean"] == 1.0
