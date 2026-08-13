from __future__ import annotations

import json

import p5_safety_human_validation_preflight as preflight


def test_frozen_sources_and_selection_hashes_reproduce():
    private, public = preflight.reproduce_selection()
    spec = json.loads(preflight.SPEC.read_text())
    assert len(private) == 24
    assert preflight.sha256_json(private) == spec["expected"]["private_rows_sha256"]
    assert preflight.sha256_json(public) == spec["expected"]["public_rows_sha256"]


def test_selection_is_balanced_without_outcome_selection():
    private, _ = preflight.reproduce_selection()
    assert sorted({row["pair_id"] for row in private}) == [
        "p5sp_cyber_01",
        "p5sp_drugs_01",
        "p5sp_fraud_01",
        "p5sp_privacy_01",
        "p5sp_violence_01",
        "p5sp_weapons_01",
    ]
    assert sum(row["stratum"] == "harmful" for row in private) == 12
    assert sum(row["stratum"] == "benign" for row in private) == 12
    assert sorted(
        sum(row["checkpoint_role"] == role for row in private)
        for role in {
            "base_r1",
            "public_star1",
            "owned_fullft_safety_s42",
            "owned_fullft_control_s42",
        }
    ) == [6, 6, 6, 6]


def test_public_selection_does_not_leak_checkpoint_or_labels():
    _, public = preflight.reproduce_selection()
    for row in public:
        assert not preflight.HIDDEN_FROM_HUMANS.intersection(row)
        assert "checkpoint" not in " ".join(row).lower()


def test_dry_report_is_non_executable_and_arm_blind():
    report = preflight.build_dry_report()
    assert report["status"] == "pass_zero_spend_non_executable"
    assert report["network_model_api_pod_calls"] == 0
    assert report["human_annotations"] == 0
    assert report["selection"]["n_rows"] == 24
    assert report["selection"]["total_generated_tokens"] == 38391
    assert report["selection"]["length_cap_hits"] == 6
    assert report["selection"]["checkpoint_role_counts_private_only"] == [6, 6, 6, 6]
    assert report["blinding"]["public_rows_contain_checkpoint_identity"] is False
    assert report["blinding"]["arm_labelled_effect_estimation_authorized"] is False
