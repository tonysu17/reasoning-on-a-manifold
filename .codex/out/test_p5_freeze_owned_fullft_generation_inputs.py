from __future__ import annotations

import hashlib

import p5_freeze_owned_fullft_generation_inputs as freeze


DOCUMENT = freeze.build_document()


def test_owned_generation_freeze_binds_100_tasks_and_two_roles():
    document = DOCUMENT
    assert document["phase2_task_source"]["n_tasks"] == 100
    assert len(document["ordered_prompt_inputs"]) == 100
    assert document["expected_terminal_rows"] == 200
    assert document["roles"] == list(freeze.ROLES)
    assert document["checkpoint_pair"]["model_weights_distinct"] is True


def test_owned_generation_contract_is_greedy_lossless_and_4096_capped():
    document = DOCUMENT
    contract = document["generation_contract"]
    assert contract["max_new_tokens"] == 4096
    assert contract["primary_analytic_prefix_tokens"] == 4096
    assert contract["do_sample"] is False
    assert contract["custom_stop_strings"] == []
    assert contract["stop_at_think_close"] is False
    assert "generated_token_ids" in document["required_terminal_row_fields"]
    assert "stop_reason" in document["required_terminal_row_fields"]


def test_prompt_hashes_and_input_id_hashes_are_closed_and_unique_by_task():
    document = DOCUMENT
    rows = document["ordered_prompt_inputs"]
    assert len({row["task_id"] for row in rows}) == 100
    assert all(len(row["prompt_sha256"]) == 64 for row in rows)
    assert all(len(row["input_ids_sha256"]) == 64 for row in rows)
    assert document["ordered_prompt_inputs_sha256"] == freeze.sha256_json(rows)


def test_freeze_is_non_executable_and_has_no_spend_authority():
    document = DOCUMENT
    assert document["execution_authorized"] is False
    assert document["network_model_api_pod_calls_authorized"] == 0
    assert document["internal_sha256"] == freeze.internal_hash(document)
    assert "runner/environment/cost manifest absent" in document["hard_stops"]


def test_owned_pair_uses_identical_tokenizer_identity():
    document = DOCUMENT
    control = {
        row["name"]: row["sha256"]
        for row in document["checkpoint_pair"]["control"]["files"]
    }
    safety = {
        row["name"]: row["sha256"]
        for row in document["checkpoint_pair"]["safety"]["files"]
    }
    for name in freeze.preflight.SHARED_CHECKPOINT_FILES:
        assert control[name] == safety[name]
    assert control["model.safetensors"] != safety["model.safetensors"]
