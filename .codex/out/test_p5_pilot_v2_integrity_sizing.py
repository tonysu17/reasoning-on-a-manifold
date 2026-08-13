from __future__ import annotations

import json
from pathlib import Path

import p5_pilot_v2_integrity_sizing as qa


HERE = Path(__file__).resolve().parent
REPORT = HERE / "P5_PILOT_V2_1_INTEGRITY_AND_BLINDED_SIZING_2026-08-09.json"


def load_report():
    return json.loads(REPORT.read_text())


def test_integrity_report_covers_exact_primary_run():
    report = load_report()
    assert report["integrity"]["status"] == "pass"
    assert report["integrity"]["observed_assignments"] == 176
    assert report["integrity"]["unique_assignment_ids"] == 176
    assert report["integrity"]["same_sonnet_repeat_rows"] == 0
    assert report["requests_and_budget"]["attempts_reserved"] == 385
    assert report["requests_and_budget"]["attempts_completed"] == 385
    assert report["requests_and_budget"]["retries"] == 0


def test_all_execution_guards_remained_inside_bounds():
    budget = load_report()["requests_and_budget"]
    assert budget["all_guards_respected"] is True
    assert budget["attempts_reserved"] <= budget["request_ceiling"]
    assert budget["maximum_observed_call_cost_usd"] <= 0.025
    assert budget["minimum_observed_remaining_quota_usd"] >= budget["quota_floor_usd"]
    assert budget["conservative_authorized_cost_commitment_usd"] <= budget[
        "spend_ceiling_usd"
    ]


def test_blinded_report_contains_no_arm_labelled_effects():
    report = load_report()
    assert report["blinding"]["arm_labelled_effect_differences_computed"] is False
    assert report["blinding"]["arm_labelled_effect_differences_reported"] is False
    rendered_sizing = json.dumps(report["blinded_sizing"], sort_keys=True)
    for role in (
        "base_r1",
        "public_star1",
        "owned_fullft_safety_s42",
        "owned_fullft_control_s42",
    ):
        assert role not in rendered_sizing


def test_safety_missingness_blocks_powered_size_freeze():
    report = load_report()
    assert report["pilot_gate"]["status"] == "not_ready_for_powered_size_freeze"
    assert report["scoring"]["overall_parse_rate"] < 0.98
    assert report["blinded_sizing"]["safety"]["harmful_refusal"][
        "pair_completion_rate"
    ] < 0.90
    assert report["blinded_sizing"]["safety"]["benign_compliance"][
        "pair_completion_rate"
    ] < 0.90


def test_sizing_formula_is_deterministic():
    curve = qa.required_n(1.0, 0.05, 1.0)
    assert curve["required_complete_prompt_pairs"] == 3140
    assert curve["required_attempted_prompts_at_observed_pair_completion"] == 3140
