"""Pre-flight tests for the generation-only Venhoff operator bridge."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "07f_venhoff_fixed_bridge.py"
VECTORS = (
    ROOT
    / "results/steering_vectors/R1-1.5B__venhoff_bridge_hybrid"
)
EXPECTED_UNIT_VECTOR_SHA256 = {
    "backtracking": "10b12bdd2e90afafee4198e13156d30e585cde41f29993dca2ca91d7c221b2ca",
    "uncertainty-estimation": "1462474e842167ed00b3cf89ab537f5dc7066273dc9503d320bf186986b371c6",
    "example-testing": "e943cd417ff650df5eea05366fae33fe7bdf355de9213310770138168a43a3bb",
    "adding-knowledge": "b57a6301a392994008615fc55d4e39a899e4864e76520c25fba5cbbfdd800a5b",
}


def _load_runner():
    spec = importlib.util.spec_from_file_location("venhoff_fixed_bridge", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_user_vectors_match_published_layers_and_write_norms():
    bridge = _load_runner()
    behaviours = list(bridge.DEFAULT_BEHAVIOURS)
    vectors, contract = bridge.load_user_directions(
        VECTORS,
        behaviours,
        scale_convention="published-norm",
    )

    assert bridge.DEFAULT_VECTORS_DIR == Path(
        "results/steering_vectors/R1-1.5B__venhoff_bridge_hybrid"
    )
    assert behaviours == [
        "backtracking",
        "uncertainty-estimation",
        "example-testing",
        "adding-knowledge",
    ]
    assert contract["kind"] == "user_direction_published_norm"
    assert set(vectors) == set(behaviours)
    assert contract["scale"]["values"] == {
        behaviour: bridge.RELEASED_VECTOR_NORMS[behaviour]
        for behaviour in behaviours
    }
    for behaviour in behaviours:
        assert vectors[behaviour]["layer"] == bridge.VENHOFF_LAYERS[behaviour]
        assert np.linalg.norm(vectors[behaviour]["vector"]) == pytest.approx(
            bridge.RELEASED_VECTOR_NORMS[behaviour], rel=1e-6
        )
        assert contract["unit_vectors"][behaviour]["sha256"] == (
            EXPECTED_UNIT_VECTOR_SHA256[behaviour]
        )
        assert len(
            contract["unit_vectors"][behaviour][
                "derived_scaled_vector_sha256"
            ]
        ) == 64


def test_hybrid_asset_preserves_source_arrays_and_provenance_boundary():
    metadata = json.loads((VECTORS / "metadata.json").read_text())
    sources = {
        "backtracking": ROOT
        / "results/steering_vectors/R1-1.5B__E1_pooled/backtracking_single.npy",
        "example-testing": ROOT
        / "results/steering_vectors/R1-1.5B__E1_pooled/example-testing_single.npy",
        "uncertainty-estimation": ROOT
        / (
            "results/steering_vectors/R1-1.5B__venhoff_bridge_sixlabel/"
            "uncertainty-estimation_single.npy"
        ),
        "adding-knowledge": ROOT
        / (
            "results/steering_vectors/R1-1.5B__venhoff_bridge_sixlabel/"
            "adding-knowledge_single.npy"
        ),
    }

    for behaviour, source in sources.items():
        hybrid = VECTORS / f"{behaviour}_single.npy"
        assert hybrid.read_bytes() == source.read_bytes()
        assert np.array_equal(np.load(hybrid), np.load(source))
        assert metadata[behaviour]["byte_identical_to_source"] is True
        assert metadata[behaviour]["array_equal_to_source"] is True

    contracts = metadata["_provenance"]["source_contracts"]
    assert "unresolved provenance" in contracts["current_E1_pooled"][
        "provenance_status"
    ]
    assert contracts["reconstructed_exact_layer"]["source_contract_status"] == (
        "mixed clip-window vintages"
    )


def test_released_vector_loader_reproduces_normalization(tmp_path):
    bridge = _load_runner()
    n_layers, hidden = 19, 4
    overall = torch.zeros(n_layers, hidden)
    overall[:, :2] = torch.tensor([3.0, 4.0])  # norm 5 at every layer
    means = {"overall": {"mean": overall}}
    for behaviour, layer in bridge.VENHOFF_LAYERS.items():
        behaviour_mean = overall.clone()
        behaviour_mean[layer, 2] += 2.0
        means[behaviour] = {"mean": behaviour_mean}
    path = tmp_path / "synthetic_means.pt"
    torch.save(means, path)

    vectors, contract = bridge.load_released_vectors(
        path,
        list(bridge.DEFAULT_BEHAVIOURS),
        allow_unverified=True,
    )

    assert contract["kind"] == "released_vector"
    assert contract["official_sha256_match"] is False
    for behaviour in vectors:
        assert np.linalg.norm(vectors[behaviour]["vector"]) == pytest.approx(5.0)


def test_resume_rejects_a_different_run_contract(tmp_path):
    bridge = _load_runner()
    path = tmp_path / "steering_results.json"
    path.write_text(json.dumps([{
        "behaviour": "backtracking",
        "method": "venhoff_constant_subtract",
        "alpha": 1.0,
        "task_id": "t0",
        "vector_source": "user_direction_published_norm",
        "run_contract_sha256": "a" * 64,
    }]))

    with pytest.raises(ValueError, match="different run contracts|mix run contracts"):
        bridge._load_existing_results(
            path, "user_direction_published_norm", "b" * 64
        )


def test_task_record_tracks_think_close_and_token_cap():
    bridge = _load_runner()
    record = bridge._task_record(
        {"id": "t0", "prompt": "Solve it", "category": "math"},
        {
            "prompt": "formatted prompt",
            "chain": "reasoning</think>answer",
            "full_text": "formatted promptreasoning</think>answer",
            "n_tokens": 1000,
        },
        max_new_tokens=1000,
        run_contract_sha256="c" * 64,
        behaviour="shared",
        method="vanilla",
        alpha=0.0,
    )
    assert record["closed_think"] is True
    assert record["hit_token_cap"] is True
    assert record["instruction"] == "Solve it"
    assert record["prompt"] == "formatted prompt"
    assert record["full_text"] == "formatted promptreasoning</think>answer"


def _make_baseline_reuse_source(tmp_path, bridge):
    source_dir = tmp_path / "prior_bridge"
    source_dir.mkdir(parents=True)
    task_ids = [f"task-{i:02d}" for i in range(50)]
    tasks = {
        task_id: {
            "id": task_id,
            "prompt": f"Instruction {i}",
            "category": f"category-{i // 5}",
        }
        for i, task_id in enumerate(task_ids)
    }
    model_contract = {
        "public_id": bridge.MODEL_ID,
        "revision": bridge.MODEL_REVISION,
        "local_snapshot": "/immutable/model/snapshot",
        "model_weight_sha256": bridge.MODEL_WEIGHT_SHA256,
    }
    eval_source_sha = "e" * 64
    tasks_source_sha = "f" * 64
    source_contract = {
        "model": model_contract,
        "dtype": "bfloat16",
        "use_4bit": False,
        "max_new_tokens": 1000,
        "temperature": 0.0,
        "evaluation_ids_source": {"sha256": eval_source_sha},
        "tasks_source": {"sha256": tasks_source_sha},
        "evaluation_task_ids": task_ids,
    }
    source_contract_sha = bridge._json_sha256(source_contract)
    provenance = {
        "run_contract": source_contract,
        "run_contract_sha256": source_contract_sha,
    }
    (source_dir / "provenance.json").write_text(json.dumps(provenance))
    (source_dir / "eval_task_ids.json").write_text(json.dumps({
        "task_ids": task_ids,
        "run_contract_sha256": source_contract_sha,
    }))

    rows = []
    expected_formatted_prompts = {}
    for task_id in task_ids:
        task = tasks[task_id]
        formatted_prompt = f"<chat>{task['prompt']}</chat>"
        expected_formatted_prompts[task_id] = formatted_prompt
        chain = "reasoning answer"
        rows.append({
            "task_id": task_id,
            "base_task_id": task_id,
            "category": task["category"],
            "instruction": task["prompt"],
            "prompt": formatted_prompt,
            "chain": chain,
            "full_text": formatted_prompt + chain,
            "n_tokens": 2,
            "closed_think": False,
            "hit_token_cap": False,
            "temperature": 0.0,
            "seed": 42,
            "run_contract_sha256": source_contract_sha,
            "behaviour": "shared",
            "method": "vanilla",
            "alpha": 0.0,
            "layer": None,
            "mode": "none",
            "vector_source": None,
            "vector_norm": 0.0,
            "write_norm": 0.0,
            "mean_abs_displacement": 0.0,
        })
    # A source may already have begun a steered arm. It must be ignored rather
    # than copied into the newly contracted destination.
    rows.append({
        "task_id": task_ids[0],
        "behaviour": "backtracking",
        "method": "venhoff_constant_subtract",
        "alpha": 1.0,
    })
    (source_dir / "steering_results.json").write_text(json.dumps(rows))
    return {
        "source_dir": source_dir,
        "task_ids": task_ids,
        "tasks": tasks,
        "model_contract": model_contract,
        "eval_source_sha": eval_source_sha,
        "tasks_source_sha": tasks_source_sha,
        "expected_formatted_prompts": expected_formatted_prompts,
        "prompt_format_contract": {
            "formatted_prompts_sha256": bridge._json_sha256(
                expected_formatted_prompts
            ),
            "synthetic_test_contract": True,
        },
    }


def _prepare_synthetic_reuse(bridge, fixture):
    return bridge._prepare_baseline_reuse(
        fixture["source_dir"],
        model_contract=fixture["model_contract"],
        dtype="bfloat16",
        use_4bit=False,
        max_new_tokens=1000,
        temperature=0.0,
        evaluation_task_ids=fixture["task_ids"],
        evaluation_ids_source_sha256=fixture["eval_source_sha"],
        tasks_source_sha256=fixture["tasks_source_sha"],
        tasks_by_id=fixture["tasks"],
        expected_formatted_prompts=fixture["expected_formatted_prompts"],
        prompt_format_contract=fixture["prompt_format_contract"],
    )


def test_baseline_reuse_copies_only_validated_shared_rows(tmp_path):
    bridge = _load_runner()
    fixture = _make_baseline_reuse_source(tmp_path, bridge)

    reusable, manifest = _prepare_synthetic_reuse(bridge, fixture)

    assert len(reusable) == 50
    assert manifest["selected_shared_vanilla_count"] == 50
    assert manifest["ignored_nonbaseline_count"] == 1
    assert manifest["foreign_arms_used"] is False
    assert manifest["ignored_nonbaseline_arms"] == {
        "backtracking::venhoff_constant_subtract::alpha=1.0": 1
    }
    assert all(row["behaviour"] == "shared" for row in reusable)
    assert all("derived_vector_sha256" not in row for row in reusable)
    assert all("baseline_reuse" in row for row in reusable)

    destination_contract_sha = "d" * 64
    installed, changed = bridge._install_reused_baselines(
        [], reusable, destination_contract_sha
    )
    assert changed is True
    assert len(installed) == 50
    assert {row["run_contract_sha256"] for row in installed} == {
        destination_contract_sha
    }
    resumed, changed = bridge._install_reused_baselines(
        installed, reusable, destination_contract_sha
    )
    assert resumed == installed
    assert changed is False


def test_baseline_reuse_rejects_protocol_drift(tmp_path):
    bridge = _load_runner()
    fixture = _make_baseline_reuse_source(tmp_path, bridge)
    source_dir = fixture["source_dir"]
    provenance = json.loads((source_dir / "provenance.json").read_text())
    provenance["run_contract"]["dtype"] = "float16"
    revised_sha = bridge._json_sha256(provenance["run_contract"])
    provenance["run_contract_sha256"] = revised_sha
    (source_dir / "provenance.json").write_text(json.dumps(provenance))
    eval_ids = json.loads((source_dir / "eval_task_ids.json").read_text())
    eval_ids["run_contract_sha256"] = revised_sha
    (source_dir / "eval_task_ids.json").write_text(json.dumps(eval_ids))
    rows = json.loads((source_dir / "steering_results.json").read_text())
    for row in rows:
        if row.get("behaviour") == "shared":
            row["run_contract_sha256"] = revised_sha
    (source_dir / "steering_results.json").write_text(json.dumps(rows))

    with pytest.raises(ValueError, match="source dtype mismatch"):
        _prepare_synthetic_reuse(bridge, fixture)


def test_baseline_reuse_rejects_incomplete_or_corrupt_content(tmp_path):
    bridge = _load_runner()
    fixture = _make_baseline_reuse_source(tmp_path, bridge)
    results_path = fixture["source_dir"] / "steering_results.json"
    rows = json.loads(results_path.read_text())
    rows[7]["full_text"] = "not prompt plus chain"
    results_path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match=r"full_text is not prompt \+ chain"):
        _prepare_synthetic_reuse(bridge, fixture)

    fixture = _make_baseline_reuse_source(tmp_path / "second", bridge)
    results_path = fixture["source_dir"] / "steering_results.json"
    rows = json.loads(results_path.read_text())
    del rows[12]
    results_path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="exactly 50 shared/vanilla"):
        _prepare_synthetic_reuse(bridge, fixture)

    fixture = _make_baseline_reuse_source(tmp_path / "third", bridge)
    results_path = fixture["source_dir"] / "steering_results.json"
    rows = json.loads(results_path.read_text())
    rows[8]["prompt"] += " tampered"
    rows[8]["full_text"] = rows[8]["prompt"] + rows[8]["chain"]
    results_path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="pinned tokenizer/chat-template"):
        _prepare_synthetic_reuse(bridge, fixture)


def test_baseline_reuse_resume_rejects_altered_destination(tmp_path):
    bridge = _load_runner()
    fixture = _make_baseline_reuse_source(tmp_path, bridge)
    reusable, _ = _prepare_synthetic_reuse(bridge, fixture)
    installed, _ = bridge._install_reused_baselines([], reusable, "a" * 64)
    installed[0]["chain"] += " changed"

    with pytest.raises(ValueError, match="altered reused baselines"):
        bridge._install_reused_baselines(installed, reusable, "a" * 64)
