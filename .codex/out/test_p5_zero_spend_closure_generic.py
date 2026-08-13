from __future__ import annotations

import json
from pathlib import Path

import pytest

import p5_preflight as p5
import p5_zero_spend_closure_generic as closure


HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE / "p5_runs" / "p5-pilot-20260808"
SAFETY = HERE / "P5_SAFETY_SONNET_PATH_IMMUTABLE_DISPOSITION_2026-08-09.json"
GENERIC = HERE / "P5_GENERIC_PILOT_ARM_BLINDED_QA_SIZING_2026-08-09.json"
SKELETON = HERE / "P5_POWERED_GENERIC_PROTOCOL_MANIFEST_SKELETON_2026-08-09.json"


def load(path):
    document = json.loads(path.read_text())
    assert document["document_sha256"] == closure.internal_hash(document)
    return document


def test_all_frozen_source_hashes_reverify():
    sources = closure.verify_sources(RUN_ROOT)
    assert sources["generations.jsonl"]["sha256"] == closure.EXPECTED_HASHES["generations.jsonl"]
    assert sources["behaviour_annotations_v2.jsonl"]["sha256"] == closure.EXPECTED_HASHES["behaviour_annotations_v2.jsonl"]


def test_safety_path_is_formally_closed_without_promoting_missingness():
    document = load(SAFETY)
    assert document["status"] == "immutable_closed_no_further_sonnet_safety_scoring"
    assert document["proxy_model_or_pod_calls_made_by_closure"] == 0
    assert document["v2_1"]["parsed_success"] == 37
    assert document["v2_1"]["unresolved"] == 59
    assert document["v2_2_validation"]["parser_evidence_complete_chunks"] == 90
    assert document["v2_2_validation"]["network_success_chunks"] == 90
    assert document["v2_2_validation"]["network_success_rate"] == pytest.approx(90 / 104)
    assert document["v2_2_validation"]["gate_status"] == "fail"
    assert document["closure"]["v2_2_1_recovery_status"] == "superseded_unexecuted_closed"
    assert document["closure"]["held_out_status"] == "not_executed_closed"
    assert "remain missing or unresolved" in document["closure"]["missingness_rule"]


def test_generic_qa_uses_task_as_independent_unit_and_preserves_blinding():
    document = load(GENERIC)
    assert document["design_grain"]["independent_unit"] == "task/prompt"
    assert document["design_grain"]["independent_pilot_tasks"] == 20
    assert document["annotation_qa"]["planned_rows"] == 80
    assert document["annotation_qa"]["parsed_rows"] == 78
    assert document["annotation_qa"]["unresolved_rows"] == 2
    assert document["annotation_qa"]["row_parse_rate"] == pytest.approx(0.975)
    assert document["annotation_qa"]["gate_pass"] is False
    assert document["blinding"]["arm_labelled_means_computed_or_reported"] is False
    assert document["blinding"]["signed_arm_contrasts_computed_or_reported"] is False
    blinded_sections = json.dumps(
        {
            "endpoint_distributions": document["endpoint_distributions"],
            "sizing": document["predeclared_sizing"],
        },
        sort_keys=True,
    )
    for role in ("base_r1", "public_star1", "owned_fullft_control_s42", "owned_fullft_safety_s42"):
        assert role not in blinded_sections


def test_generic_missingness_pairing_and_cap_are_exact():
    document = load(GENERIC)
    assert document["annotation_qa"]["task_cluster_complete_checkpoint_count_distribution"] == {"3": 2, "4": 18}
    assert document["annotation_qa"]["anonymized_checkpoint_completion_counts_sorted"] == [19, 19, 20, 20]
    for endpoint in closure.TARGET_ENDPOINTS:
        pairing = document["unsigned_paired_design_inputs"][endpoint]
        assert pairing["contrast_family_A"]["complete_task_pairs"] == 18
        assert pairing["contrast_family_B"]["complete_task_pairs"] == 20
    cap = document["cap_and_truncation"]
    assert cap["length_cap_hits"] == 34
    assert cap["length_cap_rate"] == pytest.approx(34 / 80)
    assert cap["five_percent_trigger_exceeded"] is True


def test_endpoint_descriptions_are_bounded_and_clustered():
    document = load(GENERIC)
    for endpoint in closure.TARGET_ENDPOINTS:
        description = document["endpoint_distributions"][endpoint]
        pooled = description["pooled_checkpoint_rows_descriptive_only_not_independent"]
        clusters = description["task_cluster_means"]
        assert pooled["n"] == 78
        assert clusters["n"] == 20
        assert 0 <= pooled["minimum"] <= pooled["maximum"] <= 1
        assert pooled["sample_variance"] >= 0


def test_sizing_is_conservative_paired_and_category_balanced():
    document = load(GENERIC)
    sizing = document["predeclared_sizing"]
    assert sizing["independent_N"] == "number of tasks with complete pairs, not checkpoint rows"
    assert sizing["effect_grid_absolute_fraction_points"] == [0.02, 0.03, 0.05, 0.075, 0.1]
    assert sizing["primary_hypothesis_count_for_bound"] == 8
    assert all(row["required_category_balanced_tasks"] % 10 == 0 for row in sizing["curve_cells"])
    check = closure.conservative_required_tasks(0.25, 0.05, 0.80, 0.05, 0.90)
    assert check["required_complete_tasks"] > 2
    assert check["required_attempted_tasks"] >= check["required_complete_tasks"]


def test_powered_skeleton_is_non_executable_and_blocks_on_phase2_prefix_and_N():
    document = load(SKELETON)
    assert document["status"] == "draft_non_executable_owner_decisions_required"
    assert document["execution_authorized"] is False
    assert document["proxy_model_or_pod_calls_authorized"] == 0
    phase2 = document["phase2_shared_vanilla_condition"]
    assert phase2["required_phase2_manifest_file_sha256"] is None
    assert phase2["read_only_no_regeneration"] is True
    assert document["analytic_prefix_decision"]["primary_prefix_generated_tokens"] is None
    assert document["analytic_prefix_decision"]["recommended_choice"] == 4096
    assert document["sample_size_freeze"]["selected_tasks"] is None
    assert document["generic_annotation_validation"]["minimum_row_parse_rate"] == 0.98
    assert document["generic_annotation_validation"]["missing_stays_missing"] is True


def test_fixed_outputs_contain_no_scientific_or_arm_effect_claim():
    generic = load(GENERIC)
    assert generic["status"] == "arm_blinded_pilot_resource_record_not_scientific_result"
    rendered = json.dumps(generic).lower()
    assert "arm a improved" not in rendered
    assert "arm b improved" not in rendered
    assert "scientific finding" in rendered  # only in the explicit negation/caveat

