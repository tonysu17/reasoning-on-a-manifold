from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import p5_powered_generic_preflight as preflight


def task_manifest() -> dict:
    tasks = []
    for category_index in range(10):
        category = f"category_{category_index}"
        for offset in range(10):
            tasks.append({
                "id": f"C{category_index}_{117 + offset:03d}",
                "prompt": f"prompt {category_index} {offset}",
                "category": category,
            })
    ids = [task["id"] for task in tasks]
    counts = {f"category_{index}": 10 for index in range(10)}
    return {
        "n": 100,
        "id_start": 102,
        "ids_sha256": preflight.ph2_ids_sha256(ids),
        "per_category": counts,
        "generator": {"id_start": 117},
        "tasks": tasks,
    }


def write_manifest(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "phase2_task_manifest.json"
    path.write_text(json.dumps(document))
    return path


def checkpoint(tmp_path: Path, name: str, model: bytes) -> Path:
    path = tmp_path / name
    path.mkdir()
    shared = {
        "chat_template.jinja": b"template",
        "config.json": b"config",
        "generation_config.json": b"generation",
        "tokenizer.json": b"tokenizer",
        "tokenizer_config.json": b"tokenizer-config",
    }
    for filename, content in shared.items():
        (path / filename).write_bytes(content)
    (path / "model.safetensors").write_bytes(model)
    (path / "training_summary.json").write_text(name)
    return path


def test_phase2_manifest_binds_prompt_bytes_not_only_ids(tmp_path):
    document = task_manifest()
    first = preflight.validate_phase2_manifest(write_manifest(tmp_path, document))
    changed = copy.deepcopy(document)
    changed["tasks"][0]["prompt"] += " changed"
    second_path = tmp_path / "changed.json"
    second_path.write_text(json.dumps(changed))
    second = preflight.validate_phase2_manifest(second_path)
    assert first["ids_sha256"] == second["ids_sha256"]
    assert first["ordered_task_content_sha256"] != second["ordered_task_content_sha256"]
    assert first["actual_dynamic_id_start"] == 117
    assert first["static_id_start_floor"] == 102


def test_phase2_manifest_refuses_bad_ids_and_category_balance(tmp_path):
    document = task_manifest()
    document["tasks"][0]["id"] = document["tasks"][1]["id"]
    document["ids_sha256"] = preflight.ph2_ids_sha256(
        [task["id"] for task in document["tasks"]]
    )
    with pytest.raises(preflight.PreflightError, match="not unique"):
        preflight.validate_phase2_manifest(write_manifest(tmp_path, document))

    document = task_manifest()
    document["tasks"][0]["category"] = "category_1"
    with pytest.raises(preflight.PreflightError, match="category balance"):
        preflight.validate_phase2_manifest(write_manifest(tmp_path, document))


def test_owned_checkpoint_pair_requires_shared_identity_and_distinct_weights(tmp_path):
    control = checkpoint(tmp_path, "control", b"control-weights")
    safety = checkpoint(tmp_path, "safety", b"safety-weights")
    report = preflight.validate_checkpoint_pair(control, safety)
    assert report["model_weights_distinct"] is True
    assert len(report["shared_tokenizer_template_config_sha256"]) == 64

    (safety / "tokenizer.json").write_bytes(b"different")
    with pytest.raises(preflight.PreflightError, match="drift"):
        preflight.validate_checkpoint_pair(control, safety)


def test_owned_checkpoint_pair_refuses_identical_weights(tmp_path):
    control = checkpoint(tmp_path, "control", b"same")
    safety = checkpoint(tmp_path, "safety", b"same")
    with pytest.raises(preflight.PreflightError, match="byte-identical"):
        preflight.validate_checkpoint_pair(control, safety)


def test_fixed_sources_and_current_report_are_non_executable():
    sources = preflight.verify_fixed_sources()
    assert sources[str(preflight.DESIGN.relative_to(preflight.ROOT))] == (
        preflight.EXPECTED[preflight.DESIGN]
    )
    report = preflight.build_report()
    assert report["execution_authorized"] is False
    assert report["network_model_api_pod_calls_made"] == 0
    assert report["status"] == "blocked_non_executable"
    assert report["phase2_task_manifest"]["n_tasks"] == 100
    assert report["owned_checkpoint_pair"]["model_weights_distinct"] is True
    assert report["internal_sha256"] == preflight.internal_hash(report)


def test_required_generation_fields_include_lossless_ids_and_stop_lineage():
    assert preflight.REQUIRED_GENERATION_FIELDS == {
        "generated_token_ids",
        "stop_reason",
        "raw_max_new_tokens",
        "generation_config_sha256",
    }
