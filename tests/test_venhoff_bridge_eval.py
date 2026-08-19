"""API-free tests for the all-four Venhoff bridge annotation/evaluation stage."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from src.annotation import TARGET_BEHAVIOURS
from src.annotation_budget import (
    AnnotationAttemptGuard,
    AnnotationAttemptLimitError,
    AnnotationCostLimitError,
)
from src.annotation_coverage import COVERAGE_RULE_VERSION
from src.venhoff_bridge_eval import (
    ABANDONED_SIXLABEL_VECTOR_METADATA_SHA256,
    ANNOTATION_MAX_OUTPUT_TOKENS,
    ANNOTATION_WINDOW_TOKENS,
    BRIDGE_METHOD,
    CORRECTED_FOUR_TARGET_METRIC,
    EXPECTED_EVAL_IDS_SHA256,
    EXPECTED_LAYERS,
    EXPECTED_HYBRID_VECTOR_DIRNAME,
    EXPECTED_MODEL_ID,
    EXPECTED_MODEL_REVISION,
    EXPECTED_MODEL_WEIGHT_SHA256,
    EXPECTED_PROMPT_PREFIX,
    EXPECTED_PROMPT_SUFFIX,
    EXPECTED_RECORDS,
    EXPECTED_TASKS_SHA256,
    EXPECTED_TASK_IDS,
    EXPECTED_SCALED_VECTOR_SHA256,
    EXPECTED_PUBLISHED_SCALE_SOURCE,
    EXPECTED_UNIT_VECTOR_SHA256,
    EXPECTED_VECTOR_NORMS,
    EXPECTED_VECTOR_DESCRIPTION,
    EXPECTED_VECTOR_METADATA_SHA256,
    RELEASED_BUG_COMPATIBLE_METRIC,
    SPEND_CEILING_USD,
    VENHOFF_ANNOTATION_REGION_POLICY,
    _assert_bridge_guard_policy,
    analyse_scored_records,
    audit_released_qwen15_results,
    prepare_annotation_manifest,
    released_annotation_spans,
    released_bug_compatible_token_label_counts,
    score_records,
    token_label_counts,
    load_qwen_tokenizer,
    validate_generation_contract,
    venhoff_reasoning_slice,
)


class WordOffsetTokenizer:
    """Minimal fast-tokenizer-shaped fixture: one token per non-space word."""

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert return_offsets_mapping is True
        return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}

    def encode_plus(self, text, *, return_offsets_mapping):
        assert return_offsets_mapping is True
        return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}


def _json_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode()
    ).hexdigest()


def _write_generation(root):
    eval_dir = root / "results" / "eval" / "bridge"
    eval_dir.mkdir(parents=True)
    task_ids = list(EXPECTED_TASK_IDS)
    repository_root = Path(__file__).resolve().parents[1]
    all_tasks = json.loads((repository_root / "data" / "tasks_final.json").read_text())
    tasks = {task["id"]: task for task in all_tasks}
    canonical_split = json.loads(
        (repository_root / "results" / "eval" / "R1-1.5B__E1"
         / "eval_task_ids.json").read_text()
    )
    hybrid_dir = (
        repository_root / "results" / "steering_vectors"
        / EXPECTED_HYBRID_VECTOR_DIRNAME
    )
    hybrid_metadata_path = hybrid_dir / "metadata.json"
    hybrid_metadata = json.loads(hybrid_metadata_path.read_text())
    run_contract = {
        "model": {
            "public_id": EXPECTED_MODEL_ID,
            "revision": EXPECTED_MODEL_REVISION,
            "local_snapshot": "/not/loaded",
            "model_weight_sha256": EXPECTED_MODEL_WEIGHT_SHA256,
        },
        "dtype": "bfloat16",
        "use_4bit": False,
        "behaviours": list(TARGET_BEHAVIOURS),
        "layers": EXPECTED_LAYERS,
        "mode": "constant_subtract",
        "operator": "h <- h - alpha * v",
        "alpha": 1.0,
        "max_new_tokens": 1000,
        "temperature": 0.0,
        "annotation": "not_run_by_this_runner",
        "arms": ["shared vanilla baseline", BRIDGE_METHOD],
        "vector_source": {
            "kind": "user_direction_published_norm",
            "description": EXPECTED_VECTOR_DESCRIPTION,
            "vectors_dir": str(hybrid_dir.resolve()),
            "metadata": {
                "path": str(hybrid_metadata_path.resolve()),
                "sha256": EXPECTED_VECTOR_METADATA_SHA256,
                "build_provenance": hybrid_metadata["_provenance"],
                "provenance_status": hybrid_metadata["_provenance"][
                    "provenance_status"
                ],
            },
            "unit_vectors": {
                behaviour: {
                    "path": str(
                        (hybrid_dir / f"{behaviour}_single.npy").resolve()
                    ),
                    "sha256": EXPECTED_UNIT_VECTOR_SHA256[behaviour],
                    "derived_scaled_vector_sha256": (
                        EXPECTED_SCALED_VECTOR_SHA256[behaviour]
                    ),
                }
                for behaviour in TARGET_BEHAVIOURS
            },
            "scale": {
                "convention": "published-norm",
                "values": EXPECTED_VECTOR_NORMS,
                "source": EXPECTED_PUBLISHED_SCALE_SOURCE,
            },
        },
        "evaluation_ids_source": {"sha256": EXPECTED_EVAL_IDS_SHA256},
        "tasks_source": {"sha256": EXPECTED_TASKS_SHA256},
        "evaluation_split": canonical_split["rule"],
        "evaluation_task_ids": task_ids,
    }
    run_sha = _json_sha(run_contract)

    def common(task_id):
        task = tasks[task_id]
        prompt = EXPECTED_PROMPT_PREFIX + task["prompt"] + EXPECTED_PROMPT_SUFFIX
        chain = "One short thought.</think> Answer."
        return {
            "run_contract_sha256": run_sha,
            "instruction": task["prompt"],
            "category": task["category"],
            "prompt": prompt,
            "chain": chain,
            "full_text": prompt + chain,
            "temperature": 0.0,
            "seed": 42,
        }
    rows = [
        {**common(task_id), "task_id": task_id, "base_task_id": task_id,
         "behaviour": "shared", "method": "vanilla", "alpha": 0.0,
         "mode": "none", "layer": None, "vector_source": None,
         "vector_norm": 0.0, "write_norm": 0.0}
        for task_id in task_ids
    ]
    for behaviour in TARGET_BEHAVIOURS:
        rows.extend(
            {**common(task_id), "task_id": task_id, "base_task_id": task_id,
             "behaviour": behaviour, "method": BRIDGE_METHOD, "alpha": 1.0,
             "mode": "constant_subtract", "layer": EXPECTED_LAYERS[behaviour],
             "vector_source": "user_direction_published_norm",
             "source_vector_sha256": EXPECTED_UNIT_VECTOR_SHA256[behaviour],
             "derived_vector_sha256": EXPECTED_SCALED_VECTOR_SHA256[behaviour],
             "vector_norm": EXPECTED_VECTOR_NORMS[behaviour],
             "write_norm": EXPECTED_VECTOR_NORMS[behaviour]}
            for task_id in task_ids
        )
    (eval_dir / "steering_results.json").write_text(json.dumps(rows))
    (eval_dir / "provenance.json").write_text(json.dumps({
        "run_contract": run_contract, "run_contract_sha256": run_sha,
    }))
    (eval_dir / "eval_task_ids.json").write_text(json.dumps({
        "task_ids": task_ids,
        "category_counts": canonical_split["category_counts"],
        "rule": canonical_split["rule"],
        "run_contract_sha256": run_sha,
    }))
    return eval_dir, rows


def _reseal_generation_contract(eval_dir):
    provenance_path = eval_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    changed_sha = _json_sha(provenance["run_contract"])
    provenance["run_contract_sha256"] = changed_sha
    provenance_path.write_text(json.dumps(provenance))
    rows = json.loads((eval_dir / "steering_results.json").read_text())
    for row in rows:
        row["run_contract_sha256"] = changed_sha
    (eval_dir / "steering_results.json").write_text(json.dumps(rows))
    split = json.loads((eval_dir / "eval_task_ids.json").read_text())
    split["run_contract_sha256"] = changed_sha
    (eval_dir / "eval_task_ids.json").write_text(json.dumps(split))


def _covered(row, spans):
    return {
        **row,
        "annotations": spans,
        "annotation_complete": True,
        "annotation_coverage": {
            "rule_version": COVERAGE_RULE_VERSION,
            "complete": True,
        },
        "annotation_coverage_complete": True,
        "annotation_coverage_rule_version": COVERAGE_RULE_VERSION,
        "annotation_region_policy": VENHOFF_ANNOTATION_REGION_POLICY,
    }


def test_generation_contract_requires_shared_once_plus_all_four(tmp_path):
    eval_dir, _ = _write_generation(tmp_path)
    rows, _, split = validate_generation_contract(eval_dir)
    assert len(rows) == EXPECTED_RECORDS == 250
    assert rows[:50][0]["behaviour"] == "shared"
    assert [rows[50 + i * 50]["behaviour"] for i in range(4)] == list(TARGET_BEHAVIOURS)
    assert len(split["task_ids"]) == 50

    raw = json.loads((eval_dir / "steering_results.json").read_text())
    raw.pop()
    (eval_dir / "steering_results.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="exactly 250"):
        validate_generation_contract(eval_dir)


def test_generation_contract_rejects_noncanonical_prompt_and_direction(tmp_path):
    eval_dir, _ = _write_generation(tmp_path)
    rows = json.loads((eval_dir / "steering_results.json").read_text())
    rows[0]["instruction"] = "A substituted task."
    rows[0]["prompt"] = EXPECTED_PROMPT_PREFIX + rows[0]["instruction"] + EXPECTED_PROMPT_SUFFIX
    rows[0]["full_text"] = rows[0]["prompt"] + rows[0]["chain"]
    (eval_dir / "steering_results.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="instruction differs from corpus"):
        validate_generation_contract(eval_dir)

    eval_dir, _ = _write_generation(tmp_path / "empty")
    rows = json.loads((eval_dir / "steering_results.json").read_text())
    rows[0]["chain"] = "   </think> Answer."
    rows[0]["full_text"] = rows[0]["prompt"] + rows[0]["chain"]
    (eval_dir / "steering_results.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="empty Venhoff reasoning region"):
        validate_generation_contract(eval_dir)

    eval_dir, _ = _write_generation(tmp_path / "second")
    provenance_path = eval_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    contract = provenance["run_contract"]
    contract["vector_source"]["unit_vectors"]["backtracking"]["sha256"] = "0" * 64
    changed_sha = _json_sha(contract)
    provenance["run_contract_sha256"] = changed_sha
    provenance_path.write_text(json.dumps(provenance))
    rows = json.loads((eval_dir / "steering_results.json").read_text())
    for row in rows:
        row["run_contract_sha256"] = changed_sha
    (eval_dir / "steering_results.json").write_text(json.dumps(rows))
    split = json.loads((eval_dir / "eval_task_ids.json").read_text())
    split["run_contract_sha256"] = changed_sha
    (eval_dir / "eval_task_ids.json").write_text(json.dumps(split))
    with pytest.raises(ValueError, match="direction hash differs for backtracking"):
        validate_generation_contract(eval_dir)


def test_generation_contract_rejects_abandoned_sixlabel_metadata(tmp_path):
    eval_dir, _ = _write_generation(tmp_path)
    provenance_path = eval_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["run_contract"]["vector_source"]["metadata"]["sha256"] = (
        ABANDONED_SIXLABEL_VECTOR_METADATA_SHA256
    )
    provenance_path.write_text(json.dumps(provenance))
    _reseal_generation_contract(eval_dir)

    with pytest.raises(ValueError, match="not the audited final hybrid asset"):
        validate_generation_contract(eval_dir)


def test_generation_contract_rejects_tampered_per_behaviour_source(tmp_path):
    eval_dir, _ = _write_generation(tmp_path)
    provenance_path = eval_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    source_contracts = provenance["run_contract"]["vector_source"]["metadata"][
        "build_provenance"
    ]["source_contracts"]
    source_contracts["current_E1_pooled"]["behaviours"] = ["example-testing"]
    source_contracts["reconstructed_exact_layer"]["behaviours"] = [
        "backtracking",
        "uncertainty-estimation",
        "adding-knowledge",
    ]
    provenance_path.write_text(json.dumps(provenance))
    _reseal_generation_contract(eval_dir)

    with pytest.raises(ValueError, match="per-behaviour source interpretation"):
        validate_generation_contract(eval_dir)


def test_tokenizer_override_must_have_the_pinned_tokenizer_json_hash(tmp_path):
    tokenizer_dir = tmp_path / "foreign-tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}")
    provenance = {
        "run_contract": {
            "model": {
                "revision": EXPECTED_MODEL_REVISION,
                "local_snapshot": str(tokenizer_dir),
            }
        }
    }
    with pytest.raises(ValueError, match="not the audited Qwen-1.5B tokenizer"):
        load_qwen_tokenizer(provenance, tokenizer_dir)


def test_prepare_is_api_free_and_bakes_in_actual_spend_guard(tmp_path, monkeypatch):
    eval_dir, _ = _write_generation(tmp_path)
    (tmp_path / "runner.py").write_text("# bound fixture\n")

    def network_forbidden(*args, **kwargs):
        raise AssertionError("prepare must never call the annotation API")

    monkeypatch.setattr("src.annotation.requests.post", network_forbidden)
    prepared = prepare_annotation_manifest(
        eval_dir, root=tmp_path, bound_paths=("runner.py",)
    )
    assert prepared["record_count"] == 250
    assert prepared["planned_initial_requests"] == 250
    manifest = json.loads((eval_dir / "annotation_guard_manifest.json").read_text())
    assert manifest["guards"]["approved_spend_ceiling_usd"] == SPEND_CEILING_USD == 15.0
    assert manifest["guards"]["max_cost_per_attempt_usd"] == 0.05
    assert manifest["guards"]["max_attempts_per_scope"] == 2

    guard = AnnotationAttemptGuard.from_manifest(
        eval_dir / "annotation_guard_manifest.json",
        prepared["manifest_sha256"],
        eval_dir / "attempts-test.jsonl",
        root=tmp_path,
        expected_model=manifest["model"],
        expected_annotation_window_tokens=ANNOTATION_WINDOW_TOKENS,
        expected_bound_paths=("runner.py",),
        expected_max_output_tokens=ANNOTATION_MAX_OUTPUT_TOKENS,
        expected_max_prompt_chars=manifest["guards"]["max_prompt_chars"],
        expected_coverage_rule_version=COVERAGE_RULE_VERSION,
    )
    assert guard.policy.approved_spend_ceiling_usd == 15.0
    assert guard.summary()["committed_cost_usd"] == 0.0

    # Idempotent preparation returns the same sealed manifest; it does not
    # rotate the budget contract.
    again = prepare_annotation_manifest(
        eval_dir, root=tmp_path, bound_paths=("runner.py",)
    )
    assert again["manifest_sha256"] == prepared["manifest_sha256"]


def test_executing_code_rejects_a_hash_valid_manifest_widened_above_15(tmp_path):
    eval_dir, _ = _write_generation(tmp_path)
    (tmp_path / "runner.py").write_text("# bound fixture\n")
    prepared = prepare_annotation_manifest(
        eval_dir, root=tmp_path, bound_paths=("runner.py",)
    )
    manifest_path = eval_dir / "annotation_guard_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["guards"]["approved_spend_ceiling_usd"] = 20.0
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    guard = AnnotationAttemptGuard.from_manifest(
        manifest_path, digest, eval_dir / "widened.jsonl", root=tmp_path,
        expected_model=manifest["model"],
        expected_annotation_window_tokens=ANNOTATION_WINDOW_TOKENS,
        expected_bound_paths=("runner.py",),
        expected_max_output_tokens=ANNOTATION_MAX_OUTPUT_TOKENS,
        expected_max_prompt_chars=manifest["guards"]["max_prompt_chars"],
        expected_coverage_rule_version=COVERAGE_RULE_VERSION,
    )
    with pytest.raises(AnnotationAttemptLimitError, match="exactly USD 15"):
        _assert_bridge_guard_policy(guard)


def test_fresh_eval_directory_cannot_reset_repository_global_15_budget(tmp_path):
    eval_one, _ = _write_generation(tmp_path)
    (tmp_path / "runner.py").write_text("# bound fixture\n")
    first = prepare_annotation_manifest(
        eval_one, root=tmp_path, bound_paths=("runner.py",)
    )
    manifest_one = json.loads((eval_one / "annotation_guard_manifest.json").read_text())
    shared_journal = Path(first["journal_path"])
    guard_one = AnnotationAttemptGuard.from_manifest(
        eval_one / "annotation_guard_manifest.json",
        first["manifest_sha256"], shared_journal, root=tmp_path,
        expected_model=manifest_one["model"],
        expected_annotation_window_tokens=ANNOTATION_WINDOW_TOKENS,
        expected_bound_paths=("runner.py",),
        expected_max_output_tokens=ANNOTATION_MAX_OUTPUT_TOKENS,
        expected_max_prompt_chars=manifest_one["guards"]["max_prompt_chars"],
        expected_coverage_rule_version=COVERAGE_RULE_VERSION,
    )
    guard_one.reserve(hashlib.sha256(b"first-global-scope").hexdigest())

    eval_two = tmp_path / "results" / "eval" / "bridge-copy"
    eval_two.mkdir(parents=True)
    for name in ("steering_results.json", "provenance.json", "eval_task_ids.json"):
        shutil.copy2(eval_one / name, eval_two / name)
    second = prepare_annotation_manifest(
        eval_two, root=tmp_path, bound_paths=("runner.py",)
    )
    assert second["journal_path"] == first["journal_path"]
    assert second["manifest_sha256"] != first["manifest_sha256"]
    manifest_two = json.loads((eval_two / "annotation_guard_manifest.json").read_text())
    guard_two = AnnotationAttemptGuard.from_manifest(
        eval_two / "annotation_guard_manifest.json",
        second["manifest_sha256"], shared_journal, root=tmp_path,
        expected_model=manifest_two["model"],
        expected_annotation_window_tokens=ANNOTATION_WINDOW_TOKENS,
        expected_bound_paths=("runner.py",),
        expected_max_output_tokens=ANNOTATION_MAX_OUTPUT_TOKENS,
        expected_max_prompt_chars=manifest_two["guards"]["max_prompt_chars"],
        expected_coverage_rule_version=COVERAGE_RULE_VERSION,
    )
    with pytest.raises(AnnotationAttemptLimitError, match="foreign annotation"):
        guard_two.summary()


def test_journal_reservations_cannot_commit_more_than_15(tmp_path):
    eval_dir, _ = _write_generation(tmp_path)
    (tmp_path / "runner.py").write_text("# bound fixture\n")
    prepared = prepare_annotation_manifest(
        eval_dir, root=tmp_path, bound_paths=("runner.py",)
    )
    manifest = json.loads((eval_dir / "annotation_guard_manifest.json").read_text())
    guard = AnnotationAttemptGuard.from_manifest(
        eval_dir / "annotation_guard_manifest.json",
        prepared["manifest_sha256"], eval_dir / "ceiling.jsonl", root=tmp_path,
        expected_model=manifest["model"],
        expected_annotation_window_tokens=ANNOTATION_WINDOW_TOKENS,
        expected_bound_paths=("runner.py",),
        expected_max_output_tokens=ANNOTATION_MAX_OUTPUT_TOKENS,
        expected_max_prompt_chars=manifest["guards"]["max_prompt_chars"],
        expected_coverage_rule_version=COVERAGE_RULE_VERSION,
    )
    scopes = [hashlib.sha256(f"scope-{i}".encode()).hexdigest() for i in range(250)]
    for scope in scopes:
        guard.reserve(scope)  # 250 x conservative $0.05 = $12.50
    for scope in scopes[:49]:
        guard.reserve(scope)  # 299 conservative reservations commit ~$14.95
    assert guard.summary()["committed_cost_usd"] == pytest.approx(14.95)
    # Floating-point accumulation may conservatively refuse the nominal 300th
    # $0.05 reservation.  Either way, the journal can never cross USD 15.
    with pytest.raises(AnnotationCostLimitError, match="spend ceiling"):
        guard.reserve(scopes[49])
    assert guard.summary()["committed_cost_usd"] <= 15.0


def test_token_fraction_uses_four_target_denominator_and_six_label_sensitivity():
    spans = [
        {"label": "initializing", "text": "Setup."},
        {"label": "deduction", "text": "Deduce."},
        {"label": "backtracking", "text": "Wait."},
        {"label": "uncertainty-estimation", "text": "Maybe."},
        {"label": "example-testing", "text": "Example."},
        {"label": "adding-knowledge", "text": "Fact."},
    ]
    chain = "Setup. Deduce. Wait. Maybe. Example. Fact.</think> final answer"
    out = token_label_counts(chain, spans, WordOffsetTokenizer())
    assert out["four_target_token_total"] == 4
    assert out["six_label_token_total"] == 6
    assert out["venhoff_four_target_token_fraction"]["backtracking"] == 0.25
    assert out["six_label_token_fraction"]["backtracking"] == pytest.approx(1 / 6)
    assert out["six_label_sentence_fraction"]["backtracking"] == pytest.approx(1 / 6)


def test_reasoning_slice_matches_released_pre_close_rule_without_extra_cuts():
    chain = (
        "prefix generated text <think>\nReason.\n\n"
        "**Final Answer** still reasoning here.\n"
        "repeat repeat repeat repeat repeat\n</think> answer"
    )
    reasoning, start, end = venhoff_reasoning_slice(chain)
    # ``chain`` is the generation suffix: the real opening marker is in the
    # prompt, so a generated/nested marker remains part of Venhoff's reasoning.
    assert reasoning.startswith("prefix generated text <think>\nReason.")
    assert reasoning.count("<think>") == 1
    assert "**Final Answer** still reasoning here." in reasoning
    assert "repeat repeat repeat repeat repeat" in reasoning
    assert chain[start:end] == reasoning
    assert "answer" not in reasoning

    empty, empty_start, empty_end = venhoff_reasoning_slice("   </think> answer")
    assert empty == ""
    assert empty_start == empty_end == 3


def test_released_bug_compatible_sensitivity_uses_first_find_and_silent_skip():
    response = "<think>Wait. Then continue.</think>"
    spans = [
        {"label": "backtracking", "text": "Wait."},
        {"label": "adding-knowledge", "text": "Invented text."},
    ]
    out = released_bug_compatible_token_label_counts(
        response, spans, WordOffsetTokenizer()
    )
    assert out["released_bug_compatible_skipped_by_reason"] == {
        "text_not_found": 1
    }
    assert out[RELEASED_BUG_COMPATIBLE_METRIC]["backtracking"] == 1.0
    assert out[RELEASED_BUG_COMPATIBLE_METRIC]["adding-knowledge"] == 0.0

    # The released regex ignores tolerant-schema variants; this behaviour is
    # replayed only when the raw public annotation is available.
    raw = (
        '["backtracking"]Wait.["end-section"] '
        '[4. uncertainty-estimation]Maybe.["end-section"]'
    )
    assert released_annotation_spans(raw) == [
        {"label": "backtracking", "text": "Wait."}
    ]


def test_released_audit_helper_locks_stored_fraction_arithmetic():
    annotation = '["backtracking"]Wait.["end-section"]'
    expected = {label: float(label == "backtracking") for label in TARGET_BEHAVIOURS}
    record = {
        "response": "<think>Wait.</think>",
        "annotated_response": annotation,
        "label_fractions": expected,
    }
    results = {
        target: [{arm: dict(record) for arm in ("original", "positive", "negative")}]
        for target in TARGET_BEHAVIOURS
    }
    audit = audit_released_qwen15_results(results, WordOffsetTokenizer())
    assert audit == {
        "records": 12,
        "label_fraction_comparisons": 48,
        "max_absolute_difference": 0.0,
        "mean_absolute_difference": 0.0,
        "exact_comparisons": 48,
        "released_silently_skipped_spans": 0,
    }


def test_all_600_public_qwen15_fractions_replay_exactly_when_assets_exist():
    """Local primary-source audit; skips cleanly when the public clone is absent."""
    results_path = Path(
        "/tmp/venhoff-repo.del2cM/steering/results/vars/"
        "steering_evaluation_results_deepseek-r1-distill-qwen-1.5b.json"
    )
    tokenizer_path = (
        Path.home() / ".cache" / "huggingface" / "hub"
        / "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"
        / "snapshots" / EXPECTED_MODEL_REVISION
    )
    if not results_path.is_file() or not (tokenizer_path / "tokenizer.json").is_file():
        pytest.skip("released Qwen-1.5B audit asset/tokenizer not installed")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, use_fast=True
    )
    audit = audit_released_qwen15_results(
        json.loads(results_path.read_text()), tokenizer
    )
    assert audit["records"] == 600
    assert audit["label_fraction_comparisons"] == 2400
    assert audit["max_absolute_difference"] == 0.0
    assert audit["exact_comparisons"] == 2400
    assert audit["released_silently_skipped_spans"] == 6779


def test_valid_zero_target_denominator_is_distinct_from_missing_annotation():
    spans = [
        {"label": "initializing", "text": "Setup."},
        {"label": "deduction", "text": "Deduce."},
    ]
    out = token_label_counts("Setup. Deduce.</think>", spans, WordOffsetTokenizer())
    assert out["four_target_denominator_zero"] is True
    assert all(value == 0.0
               for value in out["venhoff_four_target_token_fraction"].values())
    assert out["six_label_token_fraction"]["backtracking"] == 0.0


def test_missing_annotation_record_remains_unresolved_not_zero(tmp_path):
    eval_dir, generation = _write_generation(tmp_path)
    spans = [{"label": "backtracking", "text": "One short thought."}]
    # Deliberately omit the final generated record from the annotation checkpoint.
    annotated = [_covered(row, spans) for row in generation[:-1]]
    scored = score_records(generation, annotated, WordOffsetTokenizer())
    assert len(scored) == 250
    assert scored[-1]["score_status"] == "unresolved"
    assert scored[-1]["unresolved_reason"] == "missing_annotation_record"
    assert "venhoff_four_target_token_fraction" not in scored[-1]


def _resolved_score(task_id, behaviour, method, value):
    metrics = {
        name: {target: (value if target == behaviour or behaviour == "shared" else 0.0)
               for target in TARGET_BEHAVIOURS}
        for name in (
            CORRECTED_FOUR_TARGET_METRIC,
            RELEASED_BUG_COMPATIBLE_METRIC,
            "six_label_token_fraction",
            "six_label_sentence_fraction",
        )
    }
    return {
        "task_id": task_id,
        "base_task_id": task_id,
        "behaviour": behaviour,
        "method": method,
        "alpha": 0.0 if method == "vanilla" else 1.0,
        "score_status": "resolved",
        **metrics,
    }


def test_paired_analysis_bootstraps_tasks_and_drops_unresolved_pair():
    scored = []
    for i in range(50):
        task = f"T{i:02d}"
        scored.append(_resolved_score(task, "shared", "vanilla", 0.40))
        for behaviour in TARGET_BEHAVIOURS:
            scored.append(_resolved_score(task, behaviour, BRIDGE_METHOD, 0.20))
    # One missing steered row is unresolved; it must reduce N, not enter as 0.
    target = TARGET_BEHAVIOURS[-1]
    missing = next(
        row for row in scored
        if row["task_id"] == "T49" and row["behaviour"] == target
    )
    missing.clear()
    missing.update({
        "task_id": "T49", "behaviour": target, "method": BRIDGE_METHOD,
        "alpha": 1.0, "score_status": "unresolved",
        "unresolved_reason": "missing_annotation_record",
    })

    report = analyse_scored_records(scored, n_resamples=500, seed=3)
    primary = report["metrics"]["venhoff_four_target_token_fraction"]["cells"]
    assert primary[TARGET_BEHAVIOURS[0]]["n_resolved_pairs"] == 50
    assert primary[target]["n_resolved_pairs"] == 49
    assert primary[target]["n_unresolved_pairs"] == 1
    assert primary[target]["difference"] == pytest.approx(0.20)
    assert primary[target]["bootstrap"]["resample_unit"] == "task"
    assert primary[target]["bootstrap"]["method"] == (
        "percentile (BCa degenerate fallback)"
    )
    assert primary[target]["bootstrap"]["ci_low"] > 0
    assert primary[target]["complete_case_ci_excludes_zero"] is True
    assert primary[target]["confirmatory_eligible"] is False
    assert primary[target]["holm_input_p"] == 1.0
    assert primary[target]["confirmatory_ci_excludes_zero"] is None
    assert primary[target]["sign_flip_rejects_holm_0_05"] is None
    bounds = primary[target]["missing_pair_worst_case_bounds"]
    assert bounds["lower"] == pytest.approx((49 * 0.20 - 1) / 50)
    assert bounds["upper"] == pytest.approx((49 * 0.20 + 1) / 50)


def test_two_of_fifty_pairs_can_never_be_called_significant():
    scored = []
    target = TARGET_BEHAVIOURS[0]
    for i in range(50):
        task = f"T{i:02d}"
        scored.append(_resolved_score(task, "shared", "vanilla", 1.0))
        for behaviour in TARGET_BEHAVIOURS:
            row = _resolved_score(task, behaviour, BRIDGE_METHOD, 0.0)
            if behaviour == target and i >= 2:
                for metric in (
                    CORRECTED_FOUR_TARGET_METRIC,
                    RELEASED_BUG_COMPATIBLE_METRIC,
                    "six_label_token_fraction",
                    "six_label_sentence_fraction",
                ):
                    row.pop(metric)
                row["score_status"] = "unresolved"
                row["unresolved_reason"] = "missing_annotation_record"
            scored.append(row)

    report = analyse_scored_records(scored, n_resamples=250, seed=9)
    cell = report["metrics"][CORRECTED_FOUR_TARGET_METRIC]["cells"][target]
    assert cell["n_resolved_pairs"] == 2
    assert cell["difference"] == 1.0
    assert cell["complete_case_ci_excludes_zero"] is True
    assert cell["confirmatory_eligible"] is False
    assert cell["holm_input_p"] == 1.0
    assert cell["holm_p"] == 1.0
    assert cell["confirmatory_ci_excludes_zero"] is None
    assert cell["sign_flip_rejects_uncorrected_0_05"] is None
    assert cell["sign_flip_rejects_holm_0_05"] is None
    assert cell["missing_pair_worst_case_bounds"]["exclude_zero"] is False
