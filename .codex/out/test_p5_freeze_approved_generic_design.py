from __future__ import annotations

import p5_freeze_approved_generic_design as freeze


def test_sources_are_exact_and_internal_hash_validates():
    freeze.verify_sources()
    document = freeze.build_document()
    assert document["document_sha256"] == freeze.internal_hash(document)


def test_approved_design_is_exact_and_non_executable():
    document = freeze.build_document()
    assert document["status"] == "approved_design_non_executable_phase2_and_spend_gated"
    assert document["execution_authorized"] is False
    assert document["proxy_model_or_pod_calls_authorized"] == 0
    assert document["analytic_prefix_decision"]["primary_prefix_generated_tokens"] == 4096
    size = document["sample_size_freeze"]
    assert size["selected_absolute_effect"] == 0.05
    assert size["selected_power"] == 0.8
    assert size["selected_alpha_strategy"]["planning_two_sided_alpha_per_endpoint"] == 0.0125
    assert size["selected_tasks"] == 100


def test_primary_requirements_recompute_and_fit_100_tasks():
    document = freeze.build_document()
    size = document["sample_size_freeze"]
    assert size["primary_required_category_balanced_tasks"] == {
        "backtracking": 80,
        "uncertainty-estimation": 60,
        "example-testing": 50,
        "adding-knowledge": 50,
    }
    assert size["maximum_primary_required_tasks"] == 80
    assert size["all_primary_cells_fit_selected_tasks"] is True


def test_only_owned_contrast_is_primary_and_secondaries_are_bounded():
    document = freeze.build_document()
    checkpoint = document["checkpoint_structure"]
    assert checkpoint["primary_contrast_families"] == [
        "P5-owned matched full-FT control seed 42 versus full-FT safety seed 42"
    ]
    assert checkpoint["deepscaler_status"] == "secondary observational"
    assert "public STAR1" in checkpoint["secondary_contrasts"][0]
    assert document["analysis"]["arm_labelled_pilot_effects_used_for_design"] is False


def test_phase2_and_spend_fields_remain_blocking():
    document = freeze.build_document()
    phase2 = document["phase2_shared_vanilla_condition"]
    assert phase2["required_phase2_manifest_file_sha256"] is None
    assert phase2["required_phase2_internal_ids_sha256"] is None
    assert phase2["required_canonical_artifact_sha256"] is None
    assert len(document["remaining_blockers"]) == 4
