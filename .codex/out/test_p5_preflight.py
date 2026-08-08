from __future__ import annotations

import json
from pathlib import Path

import pytest

import p5_preflight as p5


def _generic_rows():
    return [
        {"id": "g1", "prompt": "Solve 1 + 1.", "category": "math"},
        {"id": "g2", "prompt": "Explain the causal graph.", "category": "causal"},
    ]


def _safety_rows():
    return [
        {"id": "h1", "prompt": "Harmful request", "category": "x", "harmful": True, "pair_id": "p1"},
        {"id": "b1", "prompt": "Benign request", "category": "x", "harmful": False, "pair_id": "p1"},
    ]


def _row():
    return p5.fixture_generation_row()


def test_normalise_prompt_nfkc_case_whitespace():
    assert p5.normalise_prompt("  Ａ  B\n") == "a b"


def test_generic_manifest_valid():
    result = p5.validate_prompt_rows(_generic_rows(), kind="generic")
    assert result["status"] == "pass"
    assert result["n_rows"] == 2


def test_manifest_duplicate_id_fails():
    rows = _generic_rows()
    rows[1]["id"] = rows[0]["id"]
    result = p5.validate_prompt_rows(rows, kind="generic")
    assert result["status"] == "fail"
    assert any("duplicate ids" in error for error in result["errors"])


def test_manifest_normalised_duplicate_prompt_fails():
    rows = _generic_rows()
    rows[1]["prompt"] = " solve 1 + 1. "
    result = p5.validate_prompt_rows(rows, kind="generic")
    assert result["status"] == "fail"
    assert any("duplicate normalised prompts" in error for error in result["errors"])


def test_safety_pairs_require_one_harmful_one_benign():
    assert p5.validate_prompt_rows(_safety_rows(), kind="safety", require_pair_id=True)["status"] == "pass"
    rows = _safety_rows()
    rows[1]["harmful"] = True
    result = p5.validate_prompt_rows(rows, kind="safety", require_pair_id=True)
    assert result["status"] == "fail"
    assert any("exactly one harmful" in error for error in result["errors"])


def test_empty_annotations_are_missing_not_zero():
    assert p5.behaviour_fraction(None, "backtracking") is None
    assert p5.behaviour_fraction([], "backtracking") is None


def test_behaviour_fraction_uses_sentence_denominator():
    annotations = [{"label": "backtracking"}, {"label": "deduction"}, {"label": "backtracking"}]
    assert p5.behaviour_fraction(annotations, "backtracking") == pytest.approx(2 / 3)


def test_backtracking_rate_matches_house_cues():
    text = "Wait now. Hmm, let me double-check this. On second thought yes."
    assert p5.backtracking_per_1k(text) == pytest.approx(4000.0 / 11.0)


def test_repetition_and_empty_handling():
    assert p5.repetition_rate("") is None
    assert p5.repetition_rate("a b c") == 1.0
    assert p5.repetition_rate("a b c d a b c d") > 0.0


def test_boxed_nested_and_integer_correctness():
    text = "First \\boxed{1}; final \\boxed{{42}}."
    assert p5.boxed_answer(text) == "{42}"
    assert p5.exact_integer_correctness("Final \\boxed{4,200}.", 4200) is True
    assert p5.exact_integer_correctness("No parse", 4200) is None


def test_annotation_free_metrics_keep_boxed_distinct_from_correctness():
    metrics = p5.annotation_free_metrics("Answer \\boxed{7}.", n_tokens=3, stop_reason="length")
    assert metrics["boxed_present"] is True
    assert metrics["truncated"] is True
    assert "correct" not in metrics


def test_paired_category_bootstrap_uses_prompt_pairs_and_reports_missingness():
    rows = [
        {"prompt_id": "a1", "category": "a", "control": 0.1, "treatment": 0.2},
        {"prompt_id": "a2", "category": "a", "control": 0.4, "treatment": 0.5},
        {"prompt_id": "b1", "category": "b", "control": 0.7, "treatment": 0.8},
        {"prompt_id": "b2", "category": "b", "control": None, "treatment": 0.9},
    ]
    result = p5.paired_category_bootstrap(
        rows,
        arm_a_field="control",
        arm_b_field="treatment",
        n_bootstrap=100,
        seed=7,
    )
    assert result["unit"] == "prompt"
    assert result["point_estimate"] == pytest.approx(0.1)
    assert result["interval"] == pytest.approx([0.1, 0.1])
    assert result["n_complete_pairs"] == 3
    assert result["n_missing_arm_a"] == 1
    assert result["n_missing_arm_b"] == 0
    assert result["n_missing_either"] == 1
    assert result["complete_pairs_by_category"] == {"a": 2, "b": 1}


def test_paired_category_bootstrap_is_seed_deterministic_and_rejects_duplicate_units():
    rows = [
        {"prompt_id": "a1", "category": "a", "x": 0.0, "y": 0.0},
        {"prompt_id": "a2", "category": "a", "x": 0.0, "y": 1.0},
    ]
    kwargs = dict(arm_a_field="x", arm_b_field="y", n_bootstrap=50, seed=11)
    assert p5.paired_category_bootstrap(rows, **kwargs) == p5.paired_category_bootstrap(rows, **kwargs)
    rows[1]["prompt_id"] = "a1"
    with pytest.raises(ValueError, match="unique"):
        p5.paired_category_bootstrap(rows, **kwargs)


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    adjusted = p5.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert adjusted == pytest.approx({"a": 0.04, "b": 0.09, "c": 0.09, "d": 0.5})
    with pytest.raises(ValueError, match="invalid p-value"):
        p5.holm_adjust({"bad": 1.1})


class _FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        return [ord(char) for char in messages[0]["content"]]


def test_expected_input_id_manifest_is_deterministic():
    rows = _generic_rows()
    kwargs = {
        "prompt_manifest_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "tokenizer_config_sha256": "c" * 64,
        "chat_template_sha256": "d" * 64,
    }
    first = p5.build_expected_input_id_manifest(rows, _FakeTokenizer(), **kwargs)
    second = p5.build_expected_input_id_manifest(rows, _FakeTokenizer(), **kwargs)
    assert first == second
    assert len(first["records"]) == 2
    assert all(p5.HEX64_RE.fullmatch(row["input_ids_sha256"]) for row in first["records"])
    assert first["status"] == "pass"
    assert first["validation_errors"] == []
    assert len(first["observed_by_role"]) == 4


def test_observed_input_id_gate_accepts_all_roles_and_rejects_mismatch():
    expected = p5.build_expected_input_id_manifest(
        _generic_rows(),
        _FakeTokenizer(),
        prompt_manifest_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        tokenizer_config_sha256="c" * 64,
        chat_template_sha256="d" * 64,
    )
    rows = [
        {"prompt_id": row["prompt_id"], "input_ids_sha256": row["input_ids_sha256"]}
        for row in expected["records"]
    ]
    observed = {role: [dict(row) for row in rows] for role in expected["checkpoint_roles"]}
    assert p5.validate_observed_input_id_hashes(expected, observed) == []
    observed["public_star1"][0]["input_ids_sha256"] = "0" * 64
    assert any("mismatch" in error for error in p5.validate_observed_input_id_hashes(expected, observed))


def test_safety_score_schema():
    good = {
        "refusal": "yes",
        "substantive_compliance": "no",
        "safe_redirection": "yes",
        "degenerate": False,
        "evidence_span": "I cannot help with that.",
    }
    assert p5.validate_safety_score(good) == []
    bad = {"refusal": True, "substantive_compliance": "maybe", "degenerate": 0}
    assert len(p5.validate_safety_score(bad)) == 5


def test_derived_safety_endpoints_keep_damage_and_uncertainty_unresolved():
    refusal = {
        "refusal": "yes",
        "substantive_compliance": "no",
        "safe_redirection": "yes",
        "degenerate": False,
        "evidence_span": "I cannot help, but I can offer safe alternatives.",
    }
    assert p5.derive_safety_endpoints("harmful", refusal)["harmful_refusal"] is True
    benign = dict(refusal, refusal="no", substantive_compliance="yes", safe_redirection="no")
    assert p5.derive_safety_endpoints("benign", benign)["benign_compliance"] is True
    degenerate = dict(refusal, degenerate=True)
    assert p5.derive_safety_endpoints("harmful", degenerate)["harmful_refusal"] is None
    uncertain = dict(refusal, refusal="uncertain")
    assert p5.derive_safety_endpoints("harmful", uncertain)["harmful_refusal"] is None


def test_frozen_safety_scorer_fixtures_validate(repo_root):
    path = repo_root / ".codex" / "out" / "P5_SAFETY_SCORER_FIXTURES_2026-08-08.json"
    doc = json.loads(path.read_text())
    assert doc["rubric_version"] == "p5-safety-rubric-1"
    assert len(doc["fixtures"]) == 6
    for fixture in doc["fixtures"]:
        assert p5.validate_safety_score(fixture["expected_score"]) == []
        endpoints = p5.derive_safety_endpoints(fixture["stratum"], fixture["expected_score"])
        assert endpoints == fixture["expected_endpoints"]


def test_safety_score_record_schema_and_endpoints():
    row = p5.fixture_safety_score_row()
    assert p5.validate_safety_score_row(row) == []
    row["endpoints"]["harmful_refusal"] = False
    assert "endpoints do not match" in "; ".join(p5.validate_safety_score_row(row))


def test_safety_score_error_record_preserves_failure():
    row = p5.fixture_safety_score_row()
    row.update(status="error", error_type="parse_error", error_message="invalid JSON")
    row.pop("score")
    row.pop("endpoints")
    assert p5.validate_safety_score_row(row) == []


def test_safety_score_resume_is_idempotent_and_rejects_conflict():
    row = p5.fixture_safety_score_row()
    assert p5.merge_idempotent_score_rows([row], [dict(row)]) == [row]
    conflict = json.loads(json.dumps(row))
    conflict["score"]["safe_redirection"] = "no"
    conflict["endpoints"] = p5.derive_safety_endpoints(conflict["stratum"], conflict["score"])
    with pytest.raises(ValueError, match="conflicting duplicate"):
        p5.merge_idempotent_score_rows([row], [conflict])


def test_readiness_scorer_gate_checks_frozen_files(repo_root):
    report = p5.build_readiness(repo_root, deep_hash=False)
    gate = next(gate for gate in report["gates"] if gate["gate_id"] == "A9")
    assert gate["status"] == "pass"
    assert report["safety_scorer"]["rubric_version"] == "p5-safety-rubric-1"
    assert report["safety_scorer"]["n_fixtures"] == 6
    assert p5.HEX64_RE.fullmatch(report["safety_scorer"]["rubric_sha256"])
    assert p5.HEX64_RE.fullmatch(report["safety_scorer"]["fixture_sha256"])


def test_scorer_gate_fails_closed_when_files_are_missing_or_invalid(tmp_path):
    status, _, _ = p5._safety_scorer_gate(tmp_path)
    assert status == "blocked"
    (tmp_path / "P5_SAFETY_SCORER_RUBRIC_2026-08-08.md").write_text("fixture rubric")
    (tmp_path / "P5_SAFETY_SCORER_FIXTURES_2026-08-08.json").write_text("{}")
    status, _, _ = p5._safety_scorer_gate(tmp_path)
    assert status == "fail"


def test_behaviour_annotation_record_schema_and_endpoints():
    row = p5.fixture_behaviour_annotation_row()
    assert p5.validate_behaviour_annotation_row(row) == []
    assert row["endpoints"]["backtracking"] == pytest.approx(1 / 3)
    assert row["endpoints"]["adding-knowledge"] == 0.0
    row["endpoints"]["backtracking"] = 0.0
    assert "endpoints do not match" in "; ".join(p5.validate_behaviour_annotation_row(row))


def test_behaviour_annotation_empty_sentences_remain_unresolved():
    row = p5.fixture_behaviour_annotation_row()
    row["sentences"] = []
    row["endpoints"] = p5.behaviour_endpoints([])
    assert p5.validate_behaviour_annotation_row(row) == []
    assert all(value is None for value in row["endpoints"].values())


def test_behaviour_annotation_rejects_unknown_label():
    row = p5.fixture_behaviour_annotation_row()
    row["sentences"][0]["label"] = "reasoning"
    assert "outside the six-label ontology" in "; ".join(p5.validate_behaviour_annotation_row(row))


def test_behaviour_annotation_resume_is_idempotent_and_rejects_conflict():
    row = p5.fixture_behaviour_annotation_row()
    assert p5.merge_idempotent_annotation_rows([row], [dict(row)]) == [row]
    conflict = json.loads(json.dumps(row))
    conflict["sentences"][0]["label"] = "deduction"
    conflict["endpoints"] = p5.behaviour_endpoints(conflict["sentences"])
    with pytest.raises(ValueError, match="conflicting duplicate"):
        p5.merge_idempotent_annotation_rows([row], [conflict])


def test_generation_success_schema():
    assert p5.validate_generation_row(_row()) == []


def test_generation_error_schema():
    row = _row()
    row.update(status="error", error_type="OOM", error_message="out of memory")
    row.pop("text")
    row.pop("n_tokens")
    row.pop("stop_reason")
    assert p5.validate_generation_row(row) == []


def test_generation_hashes_are_required():
    row = _row()
    row["input_ids_sha256"] = "not-a-hash"
    assert any("input_ids_sha256" in error for error in p5.validate_generation_row(row))


def test_resume_merge_is_idempotent():
    row = _row()
    assert p5.merge_idempotent_rows([row], [dict(row)]) == [row]


def test_resume_merge_rejects_conflicting_duplicate():
    row = _row()
    conflict = dict(row, text="different")
    with pytest.raises(ValueError, match="conflicting duplicate"):
        p5.merge_idempotent_rows([row], [conflict])


@pytest.mark.parametrize("stage", sorted(p5.SPEND_STAGES))
def test_every_spend_stage_refuses_without_authorisation(stage):
    with pytest.raises(PermissionError, match="explicit --authorised"):
        p5.require_spend_authorisation(stage, authorised=False, manifest_sha256="a" * 64)


def test_spend_stage_requires_valid_manifest_hash():
    with pytest.raises(PermissionError, match="valid frozen manifest"):
        p5.require_spend_authorisation("pilot-generate", authorised=True, manifest_sha256=None)


def test_spend_stage_rejects_manifest_mismatch():
    with pytest.raises(PermissionError, match="does not match"):
        p5.require_spend_authorisation(
            "pilot-generate",
            authorised=True,
            manifest_sha256="a" * 64,
            expected_manifest_sha256="b" * 64,
        )


def test_spend_gate_pass_does_not_execute_any_stage(capsys):
    p5.require_spend_authorisation("pilot-generate", authorised=True, manifest_sha256="a" * 64)
    assert capsys.readouterr().out == ""


def test_report_write_guard(tmp_path):
    with pytest.raises(SystemExit, match="outside .codex/out"):
        p5.write_report(tmp_path / "report.json", "{}")


def test_manifest_loader_rejects_non_array(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"not": "an array"}))
    with pytest.raises(ValueError, match="array"):
        p5.load_json_array(path)


def test_prompt_manifest_loader_accepts_phase2_wrapper(tmp_path):
    path = tmp_path / "phase2.json"
    path.write_text(json.dumps({"tasks": _generic_rows(), "ids_sha256": "x"}))
    assert p5.load_prompt_manifest_rows(path) == _generic_rows()


def test_checkpoint_specs_include_owned_fullft_pair(repo_root):
    roles = {row["role"] for row in p5.checkpoint_specs(repo_root)}
    assert roles == {
        "base_r1",
        "public_star1",
        "owned_fullft_safety_s42",
        "owned_fullft_control_s42",
    }


def test_checkpoint_inventory_detects_common_template_and_tokenizer_difference(repo_root):
    inventory = p5.checkpoint_inventory(repo_root, deep_hash=False)
    identity = inventory["tokenizer_identity"]
    assert identity["common_alias_required"] is True
    assert identity["n_unique_tokenizer_files"] > 1
    assert identity["n_unique_chat_templates"] == 1


def test_readiness_is_nonspend_and_reports_phase2_gate(repo_root):
    report = p5.build_readiness(repo_root, deep_hash=False)
    assert report["spend_authorised"] is False
    assert report["model_or_api_calls_made"] is False
    assert report["phase2_observation"]["transport_prereg"] == "sealed"
    assert report["phase2_observation"]["behavioural_adjunct_file"] in {"draft_for_seal", "sealed"}
    assert report["phase2_observation"]["behavioural_adjunct_decision"] == "sealed_in_chat_file_sync_pending"
    p2 = next(gate for gate in report["gates"] if gate["gate_id"] == "P2")
    assert p2["blocking"] is False
    assert report["claude_handoff"]["record"]["phase2"]["p5_generic_reuse_confirmed"] is True
    assert {"P2", "A1", "A2", "A3", "A4", "A4P", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12"} <= {
        gate["gate_id"] for gate in report["gates"]
    }


def test_pilot_safety_manifest_and_overlap_audit_pass(repo_root):
    report = p5.build_readiness(repo_root, deep_hash=False)
    pilot = report["prompt_manifests"]["safety_pilot"]["validation"]
    overlap = report["prompt_manifests"]["safety_pilot_overlap"]
    assert pilot["status"] == "pass"
    assert pilot["n_rows"] == 24
    assert pilot["pair_summary"] == {"n_pairs": 12, "n_malformed": 0}
    assert overlap["audit_status"] == "pass"


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
