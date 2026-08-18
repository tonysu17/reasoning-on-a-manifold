from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

import jspace_1p5b_atomic_orderops as R


REPO = Path(__file__).resolve().parents[1]


class JointTokenizer:
    """Tiny offset tokenizer used only to test boundary mechanics."""

    bos_token_id = 99

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        assert not add_special_tokens
        ids = list(range(10, 10 + len(text)))
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


class MergedBoundaryTokenizer:
    """The first target token begins in prompt whitespace and overlaps target."""

    bos_token_id = None

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        assert not add_special_tokens
        if return_offsets_mapping:
            return {"input_ids": [4, 8], "offset_mapping": [(0, 7), (7, len(text))]}
        return {"input_ids": [4, 123]}


class SynonymTokenizer:
    all_special_ids = [999]

    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": self.mapping.get(text, [1000, 1001])}


def test_pins_and_frozen_atomic_wrapper_are_byte_exact():
    path = REPO / R.ATOMIC_TASK_FILE
    assert hashlib.sha256(path.read_bytes()).hexdigest() == R.ATOMIC_TASK_SHA256
    frozen = json.loads(path.read_text())
    assert frozen["wrapper"] == R.ATOMIC_WRAPPER
    assert "\\n" not in frozen["wrapper"]
    assert frozen["wrapper"].count("\n") == 2
    assert len(frozen["items"]) == 81


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "zero"), (19, "nineteen"), (20, "twenty"),
        (24, "twenty-four"), (100, "one hundred"),
        (123, "one hundred twenty-three"), (999, "nine hundred ninety-nine"),
    ],
)
def test_number_to_cardinal(value, expected):
    assert R.number_to_cardinal(value) == expected


def test_sealed_operation_synonyms_are_exact_and_prefixed_at_tokenization():
    assert R.OPERATION_SYNONYMS == {
        "addition": ("+", "plus", "add", "addition"),
        "subtraction": ("-", "minus", "subtract", "subtraction"),
        "multiplication": ("*", "×", "times", "multiply", "multiplication"),
        "division": ("/", "÷", "divide", "division"),
        "mod": ("%", "mod", "modulo", "remainder"),
        "squared": ("^2", "**2", "square", "squared"),
    }
    tokenizer = SynonymTokenizer({" +": [7], " plus": [8], " add": [999]})
    assert R.eligible_synonym_ids(tokenizer, "addition", 100) == {"+": 7, "plus": 8}


def test_joint_locator_passes_only_predecessor_prefix_and_bos_is_asymmetric():
    tokenizer = JointTokenizer()
    base = R.locate_joint_readout(tokenizer, "abc ", "de", force_bos=False)
    distill = R.locate_joint_readout(tokenizer, "abc ", "de", force_bos=True)
    assert base["first_target_token_index_no_special"] == 4
    assert base["predecessor_token_index_no_special"] == 3
    assert base["causal_prefix_token_ids"] == [10, 11, 12, 13]
    assert distill["causal_prefix_token_ids"] == [99, 10, 11, 12, 13]
    assert all(token_id not in base["causal_prefix_token_ids"] for token_id in [14, 15])


def test_joint_locator_handles_token_that_starts_in_prompt_and_overlaps_target():
    located = R.locate_joint_readout(MergedBoundaryTokenizer(), "12345678", "9", force_bos=False)
    assert located["first_target_token_index_no_special"] == 1
    assert located["causal_prefix_token_ids"] == [4]
    assert located["predecessor_token_id"] == 4


def test_deterministic_exact_rank_ties_use_ascending_token_id():
    logits = np.asarray([[3.0, 2.0, 3.0, 3.0, 1.0]])
    ranks = R.deterministic_exact_ranks(logits, [0, 1, 2, 3, 4])
    assert ranks.tolist() == [[1, 4, 2, 3, 5]]


def test_semantic_rank_is_minimum_over_synonyms():
    logits = np.asarray([[0.0, 5.0, 4.0, 3.0]])
    bank = R.semantic_rank_bank_numpy(
        logits, ["x", "y"], {"x": {"a": 3, "b": 1}, "y": {"c": 2}}
    )
    assert bank.tolist() == [[1, 2]]


def test_mid_layer_persistence_at_25_uses_exact_registered_layers():
    ranks = np.full((2, 27), 100, dtype=np.int32)
    ranks[0, 9:18] = 25
    ranks[1, [9, 11, 13, 15, 17]] = 1
    got = R.per_item_layer_persistence_at_25(ranks, R.SOURCE_BANDS["mid"])
    assert got.tolist() == [1.0, 5 / 9]


def test_bootstrap_and_checkpoint_swap_are_paired_and_deterministic():
    first = np.asarray([1.0, 0.0, 1.0, 0.0])
    second = np.asarray([0.0, 0.0, 1.0, 0.0])
    a = R.paired_bootstrap_mean_difference(first, second, n_bootstrap=200, seed=7)
    b = R.paired_bootstrap_mean_difference(first, second, n_bootstrap=200, seed=7)
    assert a == b
    assert a["estimate"] == 0.25
    swap_a = R.checkpoint_vector_swap_test(first - second, n_swaps=999, seed=8)
    swap_b = R.checkpoint_vector_swap_test(first - second, n_swaps=999, seed=8)
    assert swap_a == swap_b
    assert 0 < swap_a["two_sided_p"] <= 1


def test_whole_pair_permutation_preserves_duplicate_rows_and_is_synchronized():
    # Three prompts, two layers, four semantic keys.  Duplicate pair [0,2]
    # occurs twice and must remain duplicated under every row permutation.
    pairs = np.asarray([[0, 2], [0, 2], [1, 3]], dtype=np.int64)
    bank = np.full((3, 2, 4), 100, dtype=np.int32)
    bank[:, :, 0] = 1
    result = R.synchronized_whole_pair_permutation_test(
        {"a": bank, "b": bank.copy()}, pairs,
        layers=[0, 1], n_permutations=99, seed=9, chunk_size=17,
    )
    assert result["duplicates_preserved"] is True
    assert result["whole_label_pairs_preserved"] is True
    assert result["banks"]["a"] == result["banks"]["b"]


def test_numeric_assay_gate_uses_same_complete_cases_for_observed_and_null():
    pairs = np.asarray([[0, 2], [1, 3], [0, 2]], dtype=np.int64)
    base = np.full((3, 27, 4), 100, dtype=np.int32)
    distill = np.full_like(base, 100)
    for item, key in enumerate(pairs[:, 0]):
        base[item, :, key] = 1
        distill[item, :, key] = 1
    result = R.synchronized_numeric_assay_gate(
        base, distill, pairs, layers=R.SOURCE_BANDS["mid"], n_permutations=99, seed=10,
    )
    assert result["n_items"] == 3
    assert result["observed"] == 1.0
    assert result["population"].startswith("paired technically complete")


def test_typo_union_null_is_synchronized_and_uses_registered_threshold():
    labels = np.asarray([0, 1, 2], dtype=np.int64)
    bank = np.full((3, 2, 3), 100, dtype=np.int32)
    for item, key in enumerate(labels):
        bank[item, :, key] = 1
    result = R.synchronized_typo_union_null(
        {"base": bank, "distill": bank.copy()}, labels,
        n_permutations=99, seed=11, chunk_size=17,
    )
    assert result["assignments_synchronized_across_checkpoints"] is True
    assert result["cells"]["base"] == result["cells"]["distill"]
    assert result["cells"]["base"]["any_layer_union_pass_at_25"] == 1.0


def test_orderops_target_aliases_and_cot_matcher_are_frozen():
    assert R.orderops_target_aliases("24") == ("24", "twenty-four")
    assert R.orderops_target_aliases("twenty-four") == ("24", "twenty-four")
    assert R.orderops_target_aliases("Atlantic") == ("atlantic",)
    scored = R.score_orderops_cot(
        "work\r\nTherefore, the final answer is twenty-four.", ["24", "twenty-four"]
    )
    assert scored["bounded_final_line_primary"] is True
    assert scored["bounded_anywhere_sensitivity"] is True
    assert R.score_orderops_cot("twenty-fourth", ["twenty-four"])[
        "bounded_final_line_primary"
    ] is False


def test_teacher_forced_joint_target_positions_have_no_manual_bos():
    tokenized = R.joint_target_tokenization(JointTokenizer(), "abc ", "de")
    assert tokenized["target_token_positions"] == [4, 5]
    assert tokenized["target_token_ids"] == [14, 15]
    assert tokenized["manual_bos"] is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('  "Twenty-four."\nexplanation', "twenty-four"),
        ("`BRAZIL!`", "brazil"),
        ('"Eight".', "eight"),
        ("'Eight'!", "eight"),
        ("  Au  ", "au"),
    ],
)
def test_atomic_normalization_is_first_line_exact(raw, expected):
    assert R.normalize_atomic_answer(raw) == expected


def test_atomic_scoring_has_exact_whole_term_and_substring_sensitivities():
    exact = R.score_atomic_answer("Eight.\nignored", ["8", "eight"])
    assert exact["exact_primary"] is True
    assert R.score_atomic_answer("Eight", ["8"])["exact_primary"] is False
    embedded = R.score_atomic_answer("The answer is eight", ["8", "eight"])
    assert embedded["exact_primary"] is False
    assert embedded["whole_term_sensitivity"] is True
    assert embedded["substring_sensitivity"] is True


def test_token_cap_forces_behavioral_matches_incorrect():
    atomic = R.force_incorrect_on_token_cap(
        {"exact_primary": True, "whole_term_sensitivity": True},
        reached_token_cap=True,
    )
    assert atomic["exact_primary"] is False
    assert atomic["whole_term_sensitivity"] is False
    assert atomic["forced_incorrect_reason"] == "generation_token_cap"
    uncapped = R.force_incorrect_on_token_cap(
        {"bounded_final_line_primary": True}, reached_token_cap=False
    )
    assert uncapped == {"bounded_final_line_primary": True}


def test_analyse_cli_requires_the_full_seal_commit_argument(tmp_path):
    with pytest.raises(SystemExit):
        R.build_parser().parse_args(["analyse", "--run-dir", str(tmp_path)])


def test_holm_adjusts_the_two_coprimaries():
    assert R.holm_adjust_two(0.01, 0.04) == pytest.approx([0.02, 0.04])
    assert R.holm_adjust_two(0.6, 0.2) == pytest.approx([0.6, 0.4])


def test_output_root_accepts_required_absolute_suffix_and_rejects_broad_path(tmp_path):
    good = tmp_path / "results" / "jspace_r1_pilot" / "followups" / "atomic_orderops"
    assert R.validate_output_root(REPO, good) == good.resolve()
    with pytest.raises(RuntimeError):
        R.validate_output_root(REPO, tmp_path)


def test_mutable_cache_rejects_an_ancestor_of_the_sealed_checkout(tmp_path):
    cache_parent = tmp_path / "cache-parent"
    repo = cache_parent / "sealed-checkout"
    outroot = tmp_path / "evidence-output"
    repo.mkdir(parents=True)
    outroot.mkdir()
    with pytest.raises(RuntimeError, match="outside the sealed checkout"):
        R.validate_mutable_cache_dir(repo, cache_parent, outroot)


def test_preflight_source_cannot_load_or_forward_a_model():
    source = inspect.getsource(R.preflight)
    assert "_load_hf_cell" not in source
    assert "forward(" not in source
    assert "tokenizer_only_orderops_preflight" in source


def test_sealed_test_receipt_is_stable_across_pytest_timings(monkeypatch):
    timings = iter(("0.07", "0.19"))

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0], 0,
            stdout=f"{R.EXPECTED_PURE_TEST_COUNT} passed in {next(timings)}s\n",
            stderr="",
        )

    monkeypatch.setattr(R.subprocess, "run", fake_run)
    first = R.run_sealed_pure_tests(REPO)
    second = R.run_sealed_pure_tests(REPO)
    assert first == second
    assert "stdout" not in first
    assert set(first) == {
        "test_file_sha256", "fixed_selection", "collected_count", "passed_count", "status"
    }
    assert first["passed_count"] == R.EXPECTED_PURE_TEST_COUNT


def test_execution_device_fingerprint_is_stable_and_binds_dtype():
    import torch

    first = R.execution_device_fingerprint(torch)
    second = R.execution_device_fingerprint(torch)
    assert first == second
    assert {
        "backend", "device", "device_index", "physical_device_name",
        "requested_dtype", "forward_dtype", "observed_model_dtype",
        "torch_version", "platform", "platform_machine",
        "accelerator_runtime_version", "accelerator_compute_capability",
        "bf16_execution_supported", "bf16_support_basis", "bf16_probe_digest",
        "bf16_probe_result_float32", "mps_available",
    } <= set(first)
    assert first["device"] in {"cpu", "cuda", "mps"}
    assert first["backend"] == first["device"]
    assert first["requested_dtype"] == "torch.bfloat16"
    assert first["forward_dtype"] == "torch.bfloat16"
    assert first["observed_model_dtype"] is None
    if first["device"] in {"cuda", "mps"}:
        assert first["bf16_execution_supported"] is True
    else:
        assert first["bf16_execution_supported"] is False
    assert len(first["bf16_probe_result_float32"]) == 8
    probe = {
        "schema": "rom-bf16-probe-v1",
        "a": [1, 2, 3, 4, 5, 6, 7, 8],
        "b": [8, 7, 6, 5, 4, 3, 2, 1],
        "expression": "a*b+a-b",
        "result_float32": first["bf16_probe_result_float32"],
    }
    assert first["bf16_probe_digest"] == R.sha256_bytes(
        R.canonical_json_bytes(probe)
    )


def test_observed_model_dtype_is_bound_from_both_verified_weight_headers():
    raw = {"backend": "mps", "observed_model_dtype": None}
    receipts = {
        cell: {"model_weight_dtypes": ["BF16"]} for cell in R.MODEL_CELLS
    }
    bound = R.bind_observed_model_dtype(
        raw, receipts, require_weight_headers=True
    )
    assert bound["observed_model_dtype"] == "torch.bfloat16"
    bad = dict(receipts)
    bad["distill"] = {"model_weight_dtypes": ["F32"]}
    with pytest.raises(RuntimeError, match="weight dtype drift"):
        R.bind_observed_model_dtype(raw, bad, require_weight_headers=True)


def test_resume_tree_rejects_symlinked_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    target = tmp_path / "outside.json"
    target.write_text("{}")
    (run_dir / "partial.json").symlink_to(target)
    with pytest.raises(RuntimeError, match="symlink"):
        R.reject_symlinks_beneath(run_dir)


def test_lens_verification_rejects_a_symlink_even_when_target_bytes_match(
    tmp_path, monkeypatch
):
    target = tmp_path / "lens.pt"
    target.write_bytes(b"sealed-lens-bytes")
    linked = tmp_path / "linked-lens.pt"
    linked.symlink_to(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    monkeypatch.setattr(R, "MODEL_CELLS", {
        "base": {"lens_size_bytes": target.stat().st_size, "lens_sha256": digest},
        "distill": {"lens_size_bytes": target.stat().st_size, "lens_sha256": digest},
    })
    with pytest.raises(RuntimeError, match="missing pinned base lens"):
        R.verify_lens_files({"base": linked, "distill": target})


def test_completion_marker_is_idempotent_and_rejects_mutated_listed_output(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "INCOMPLETE").write_text("in progress\n")
    evidence = run_dir / "evidence.json"
    R.atomic_write_json(evidence, {"estimate": 0.125})
    stage = "synthetic-complete"
    R.atomic_write_json(run_dir / "manifest.json", {
        "schema_version": R.SCHEMA_VERSION,
        "stage": stage,
    })
    output_record = {
        "sha256": R.sha256_file(evidence),
        "size_bytes": evidence.stat().st_size,
    }
    R.atomic_write_json(run_dir / "derivation_manifest.json", {
        "schema_version": R.SCHEMA_VERSION,
        "stage": stage,
        "created_utc": "2026-08-18T00:00:00Z",
        "inputs": {},
        "outputs": {"evidence.json": output_record},
    })
    monkeypatch.setitem(
        R.COMPLETION_STAGE_FILES, stage,
        ("manifest.json", "derivation_manifest.json"),
    )

    def validate_synthetic_derivation(repo_arg, run_arg, path, *, stage):
        assert repo_arg == repo
        assert run_arg == run_dir
        payload = json.loads(path.read_text())
        assert payload["stage"] == stage
        return payload

    monkeypatch.setattr(R, "validate_derivation_manifest", validate_synthetic_derivation)
    R.finalize_stage(
        repo, run_dir,
        stage=stage,
        marker_name="COMPLETE",
        manifest_name="manifest.json",
        derivation_name="derivation_manifest.json",
    )
    marker_bytes = (run_dir / "COMPLETE").read_bytes()
    first = R.validate_completion_marker(repo, run_dir, "COMPLETE", stage)
    second = R.validate_completion_marker(repo, run_dir, "COMPLETE", stage)
    assert first == second
    assert (run_dir / "COMPLETE").read_bytes() == marker_bytes
    assert not (run_dir / "INCOMPLETE").exists()

    marker = json.loads(marker_bytes)
    marker["manifest"]["path"] = str(tmp_path / "outside.json")
    R.atomic_write_json(run_dir / "COMPLETE", marker)
    with pytest.raises(RuntimeError, match="path drift"):
        R.validate_completion_marker(repo, run_dir, "COMPLETE", stage)

    R.atomic_write_bytes(run_dir / "COMPLETE", marker_bytes)
    R.atomic_write_json(evidence, {"estimate": 0.25})
    with pytest.raises(RuntimeError, match="evidence hash mismatch"):
        R.validate_completion_marker(repo, run_dir, "COMPLETE", stage)


def test_derivation_validator_recomputes_stage_identity_inputs_and_outputs(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    run_dir = tmp_path / "run"
    repo.mkdir()
    run_dir.mkdir()
    evidence = run_dir / "evidence.json"
    R.atomic_write_json(evidence, {"estimate": 1})
    expected_inputs = {
        "sealed-input": {"sha256": "a" * 64, "size_bytes": 7},
    }
    monkeypatch.setattr(R, "stage_derivation_extra_inputs", lambda *_: [])
    monkeypatch.setattr(
        R, "derivation_input_table", lambda repo_arg, extra: expected_inputs
    )
    path = run_dir / "derivation_manifest.json"
    payload = {
        "schema_version": R.SCHEMA_VERSION,
        "stage": "atomic",
        "created_utc": "2026-08-18T00:00:00Z",
        "inputs": expected_inputs,
        "outputs": R.derivation_output_table(run_dir, path.name, stage="atomic"),
    }
    R.atomic_write_json(path, payload)
    assert R.validate_derivation_manifest(
        repo, run_dir, path, stage="atomic"
    ) == payload

    corrupted = dict(payload)
    corrupted["stage"] = "orderops-analysis"
    R.atomic_write_json(path, corrupted)
    with pytest.raises(RuntimeError, match="identity drift"):
        R.validate_derivation_manifest(repo, run_dir, path, stage="atomic")


def test_rank_and_analysis_completion_authorities_remain_independently_valid(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    run_dir = tmp_path / "run"
    repo.mkdir()
    run_dir.mkdir()
    monkeypatch.setattr(R, "stage_derivation_extra_inputs", lambda *_: [])
    monkeypatch.setattr(R, "derivation_input_table", lambda *_: {})

    R.atomic_write_bytes(run_dir / "INCOMPLETE", b"rank in progress\n")
    R.atomic_write_json(run_dir / "manifest.json", {
        "schema_version": R.SCHEMA_VERSION,
        "stage": "orderops-rank-extraction",
    })
    R.atomic_write_json(run_dir / "rank-evidence.json", {"rank": 1})
    rank_derivation = R.write_derivation_manifest(
        repo, run_dir, stage="orderops-rank-extraction", extra_inputs=[],
        filename="rank_derivation_manifest.json",
    )
    R.finalize_stage(
        repo, run_dir, stage="orderops-rank-extraction",
        marker_name="RANKS_COMPLETE", manifest_name="manifest.json",
        derivation_name=rank_derivation.name,
    )

    R.atomic_write_bytes(run_dir / "ANALYSIS_INCOMPLETE", b"analysis in progress\n")
    R.atomic_write_json(run_dir / "report.json", {
        "schema_version": R.SCHEMA_VERSION,
        "stage": "orderops-analysis",
        "estimate": 0.25,
    })
    R.atomic_write_bytes(run_dir / "REPORT.md", b"analysis\n")
    analysis_derivation = R.write_derivation_manifest(
        repo, run_dir, stage="orderops-analysis", extra_inputs=[],
    )
    R.finalize_stage(
        repo, run_dir, stage="orderops-analysis",
        marker_name="ANALYSIS_COMPLETE", manifest_name="report.json",
        derivation_name=analysis_derivation.name,
        incomplete_name="ANALYSIS_INCOMPLETE",
    )

    R.validate_completion_marker(
        repo, run_dir, "RANKS_COMPLETE", "orderops-rank-extraction"
    )
    R.validate_completion_marker(
        repo, run_dir, "ANALYSIS_COMPLETE", "orderops-analysis"
    )


def test_campaign_authority_is_shared_and_detects_preflight_mutation(tmp_path):
    campaign_parent = tmp_path / "campaign-root"
    campaign_parent.mkdir()
    campaign_dir = campaign_parent / "campaign-20260818T000000Z-aaaaaaaaaaaa"
    campaign_dir.mkdir()
    seal = {"git_commit": "a" * 40, "git_dirty": False}
    preflight = {
        "seal": seal,
        "device_fingerprint": {"device": "cuda", "observed_model_dtype": "torch.bfloat16"},
    }
    reference = R.write_campaign_authority(
        campaign_dir, seal=seal, sealed_preflight=preflight
    )
    assert R.validate_campaign_authority(
        campaign_dir, seal=seal, sealed_preflight=preflight
    ) == reference
    R.atomic_write_json(campaign_dir / "preflight.json", {
        **preflight,
        "device_fingerprint": {"device": "mps", "observed_model_dtype": "torch.bfloat16"},
    })
    with pytest.raises(RuntimeError, match="differs from current sealed preflight"):
        R.validate_campaign_authority(
            campaign_dir, seal=seal, sealed_preflight=preflight
        )


def test_campaign_authority_rejects_a_relocated_sibling_copy(tmp_path):
    campaign_parent = tmp_path / "campaign-root"
    campaign_parent.mkdir()
    campaign_dir = campaign_parent / "campaign-20260818T000000Z-aaaaaaaaaaaa"
    campaign_dir.mkdir()
    seal = {"git_commit": "a" * 40, "git_dirty": False}
    preflight = {
        "seal": seal,
        "device_fingerprint": {
            "device": "cuda", "observed_model_dtype": "torch.bfloat16"
        },
    }
    R.write_campaign_authority(
        campaign_dir, seal=seal, sealed_preflight=preflight
    )
    copied = campaign_parent / "campaign-20260818T000001Z-bbbbbbbbbbbb"
    shutil.copytree(campaign_dir, copied)
    with pytest.raises(RuntimeError, match="one unique selection"):
        R.validate_campaign_authority(
            copied, seal=seal, sealed_preflight=preflight
        )


def test_stage_selection_allows_only_one_fresh_run_per_campaign_arm(tmp_path):
    outroot = tmp_path / "out"
    run_dir = R.new_run_dir(outroot, "atomic")
    R.atomic_write_json(run_dir / "seal.json", {"git_commit": "a" * 40})
    R.atomic_write_json(run_dir / "preflight.json", {"status": "preflight-pass"})
    R.atomic_write_json(run_dir / "campaign_authority.json", {
        "campaign_selection_sha256": "b" * 64,
    })
    selected = R.write_stage_selection(run_dir, "atomic")
    assert R.validate_stage_selection(run_dir, "atomic") == selected
    with pytest.raises(RuntimeError, match="already selected"):
        R.new_run_dir(outroot, "atomic")

    corrupted = dict(selected)
    corrupted["run_id"] = "atomic-relocated"
    R.atomic_write_json(R.stage_selection_path(run_dir), corrupted)
    with pytest.raises(RuntimeError, match="authority drift"):
        R.validate_stage_selection(run_dir, "atomic")


def test_orderops_annotation_exclusion_is_separate_and_result_blind():
    assert R.ORDEROPS_LEXICAL_EXCLUSIONS == {
        "nested-sub-add-mult", "add-add-add", "square-mult"
    }
    assert R.ORDEROPS_ANNOTATION_EXCLUSIONS == {"mult-div-mult"}
    assert R.ORDEROPS_PRIMARY_EXCLUSIONS == (
        R.ORDEROPS_LEXICAL_EXCLUSIONS | R.ORDEROPS_ANNOTATION_EXCLUSIONS
    )


def test_pair_pattern_vocabulary_does_not_turn_an_unestablished_did_into_a_negative():
    lock = json.loads((REPO / R.INPUT_LOCK).read_text())
    vocabulary = lock["orderops_pair_pattern_vocabulary"]
    assert vocabulary["recurrence_without_established_transport_specificity"] == (
        "material-recurrence-transport-specificity-not-established"
    )
    source = inspect.getsource(R.analyse_orderops_bundle)
    assert "material-recurrence-not-transport-specific" not in source
    assert vocabulary["recurrence_without_established_transport_specificity"] in source


@pytest.mark.parametrize(
    ("ids", "eos", "lf", "cap", "valid"),
    [
        (list(range(15)), True, False, False, True),
        (list(range(16)), True, False, False, False),
        (list(range(8)), False, True, False, True),
        (list(range(16)), False, False, True, True),
        (list(range(15)), False, False, True, False),
    ],
)
def test_generation_stop_state_cannot_hide_a_token_cap(ids, eos, lf, cap, valid):
    if valid:
        R.validate_generation_stop_state(
            ids, stopped_on_eos=eos, stopped_on_lf=lf,
            reached_token_cap=cap, max_new_tokens=16, context="test",
        )
    else:
        with pytest.raises(RuntimeError):
            R.validate_generation_stop_state(
                ids, stopped_on_eos=eos, stopped_on_lf=lf,
                reached_token_cap=cap, max_new_tokens=16, context="test",
            )


def test_atomic_final_manifest_accepts_json_roundtrip_of_registered_model_tuples(
    tmp_path, monkeypatch
):
    records = [{"cell": "base", "task_index": 0}]
    report = {"co_primary": {}}
    monkeypatch.setattr(R, "atomic_report_markdown", lambda _: "report\n")
    generations = tmp_path / "generations.jsonl"
    R.atomic_write_bytes(generations, R.jsonl_bytes(records))
    R.atomic_write_json(tmp_path / "report.json", report)
    R.atomic_write_bytes(tmp_path / "REPORT.md", b"report\n")
    manifest = {
        "schema_version": R.SCHEMA_VERSION,
        "stage": "atomic",
        "models": R.MODEL_CELLS,
        "generation_jsonl_sha256": R.sha256_file(generations),
        "report_sha256": R.sha256_file(tmp_path / "report.json"),
    }
    R.atomic_write_json(tmp_path / "manifest.json", manifest)
    R.validate_atomic_final_bundle(
        tmp_path, records=records, report=report, manifest=manifest
    )


def test_rank_partial_digest_and_schema_reject_corruption(tmp_path):
    binding = {
        "schema_version": R.SCHEMA_VERSION,
        "seal_sha256": "a" * 64,
        "preflight_sha256": "b" * 64,
        "campaign_authority_sha256": "c" * 64,
        "stage_selection_sha256": "d" * 64,
        "run_identity_sha256": "e" * 64,
        "device_fingerprint": {"device": "cpu"},
    }
    locator = {"causal_prefix_token_ids": [1], "predecessor_token_id": 1}
    label_pairs = np.asarray([[0, 1]], dtype=np.int32)
    arrays = {
        "schema_version": np.asarray(R.SCHEMA_VERSION),
        "stage": np.asarray("orderops-rank-partial"),
        "cell": np.asarray("base"),
        "completed_item_names": np.asarray(["x"], dtype="U64"),
        "semantic_keys": np.asarray(["3", "addition"], dtype="U32"),
        "source_layers": np.asarray(R.SOURCE_LAYERS, dtype=np.int32),
        "label_pairs": label_pairs,
        "execution_binding_json": np.asarray(
            json.dumps(binding, sort_keys=True, separators=(",", ":"))
        ),
        "locators_json": np.asarray(json.dumps(
            [{"name": "x", **locator}], sort_keys=True, separators=(",", ":")
        )),
    }
    for arm in R.ORDEROPS_READOUT_ARMS:
        arrays[f"ranks__{arm}"] = np.ones((1, 27, 2), dtype=np.int32)
        arrays[f"top25__{arm}"] = np.tile(
            np.arange(25, dtype=np.int32), (1, 27, 1)
        )
        arrays[f"boundary_ties__{arm}"] = np.asarray(0, dtype=np.int64)
    arrays["payload_sha256"] = np.asarray(R.array_bundle_digest(arrays))
    arrays["completed_prefix_digest"] = np.asarray(
        R.partial_completed_prefix_digest(
            stage="orderops-rank:base", binding=binding,
            ordered_identifiers=["x"],
            payload_sha256=str(arrays["payload_sha256"].item()),
        )
    )
    path = tmp_path / "partial.npz"
    R.atomic_savez(path, **arrays)
    R.validate_rank_partial(
        path, cell="base", expected_names=["x"], semantic_keys=["3", "addition"],
        expected_label_pairs=label_pairs,
        ranked_vocab_size=100, binding=binding, expected_locators={"x": locator},
    )
    corrupt = R.load_npz_no_pickle(path)
    corrupt["ranks__fp32_j"][0, 0, 0] = 101
    R.atomic_savez(path, **corrupt)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        R.validate_rank_partial(
            path, cell="base", expected_names=["x"],
            semantic_keys=["3", "addition"], expected_label_pairs=label_pairs,
            ranked_vocab_size=100,
            binding=binding, expected_locators={"x": locator},
        )


def test_full_rank_payload_validator_checks_top25_uniqueness():
    n_items, n_keys, n_typo = 2, 3, 3
    arrays = {}
    for arm in R.ORDEROPS_READOUT_ARMS:
        arrays[arm] = np.ones((n_items, 27, n_keys), dtype=np.int32)
        arrays[f"top25__{arm}"] = np.tile(
            np.arange(25, dtype=np.int32), (n_items, 27, 1)
        )
        arrays[f"boundary_ties__{arm}"] = np.asarray(0, dtype=np.int64)
    arrays.update({
        "label_pairs": np.asarray([[0, 1], [2, 1]], dtype=np.int32),
        "item_names": np.asarray(["a", "b"], dtype="U64"),
        "semantic_keys": np.asarray(["3", "addition", "4"], dtype="U32"),
        "typo_fp32_j": np.ones((n_typo, 27, n_typo), dtype=np.int32),
        "typo_top25__fp32_j": np.tile(
            np.arange(25, dtype=np.int32), (n_typo, 27, 1)
        ),
        "typo_boundary_ties__fp32_j": np.asarray(0, dtype=np.int64),
        "typo_label_indices": np.arange(n_typo, dtype=np.int32),
        "typo_item_names": np.asarray(["t0", "t1", "t2"], dtype="U64"),
        "typo_semantic_keys": np.asarray(["x", "y", "z"], dtype="U64"),
    })
    kwargs = dict(
        cell="base", expected_names=["a", "b"],
        semantic_keys=["3", "addition", "4"],
        expected_label_pairs=np.asarray([[0, 1], [2, 1]], dtype=np.int32),
        typo_names=["t0", "t1", "t2"], typo_keys=["x", "y", "z"],
        expected_typo_labels=np.arange(n_typo, dtype=np.int32),
        ranked_vocab_size=100,
    )
    R.validate_orderops_rank_arrays(arrays, **kwargs)
    arrays["top25__fp32_j"][0, 0, 1] = arrays["top25__fp32_j"][0, 0, 0]
    with pytest.raises(RuntimeError, match="duplicate ID"):
        R.validate_orderops_rank_arrays(arrays, **kwargs)


def test_atomic_supportive_bootstrap_recomputes_common_subset_per_draw():
    hb1 = np.ones(30, dtype=np.int8)
    hb2 = np.ones(30, dtype=np.int8)
    hd1 = np.ones(30, dtype=np.int8)
    hd2 = np.asarray([1] * 24 + [0] * 6, dtype=np.int8)
    mb = np.asarray([1, 0] * 15, dtype=np.int8)
    md = np.asarray([0, 1] * 15, dtype=np.int8)
    first = R.atomic_supportive_bootstrap(
        hb1, hb2, hd1, hd2, mb, md, n_bootstrap=200, seed=77
    )
    second = R.atomic_supportive_bootstrap(
        hb1, hb2, hd1, hd2, mb, md, n_bootstrap=200, seed=77
    )
    assert first == second
    common = first["common_known_composite"]
    assert common["n_items"] == 24
    assert common["derived_subset_recomputed_within_each_draw"] is True
    assert len(common["distill_minus_base_risk_difference_percentile_95_ci"]) == 2
    for cell in ("base", "distill"):
        assert set(first["atomic_state"][cell]) == {
            "both", "hop1_only", "hop2_only", "neither"
        }
        assert all(
            "percentile_95_ci" in value
            for value in first["atomic_state"][cell].values()
        )
    paired = first["atomic_state"]["paired_distill_minus_base"]
    assert set(paired) == {"both", "hop1_only", "hop2_only", "neither"}
    assert paired["both"]["estimate"] == pytest.approx(-0.2)
    assert paired["hop1_only"]["estimate"] == pytest.approx(0.2)
    assert all(len(value["percentile_95_ci"]) == 2 for value in paired.values())


def test_atomic_supportive_bootstrap_withholds_conditional_intervals_on_empty_draws(
    monkeypatch,
):
    class EmptyCommonKnownGenerator:
        def integers(self, low, high=None, size=None):
            assert low == 0
            assert high == 40
            return np.full(size, 20, dtype=np.int64)

    monkeypatch.setattr(
        R.np.random, "Generator", lambda bit_generator: EmptyCommonKnownGenerator()
    )
    both = np.asarray([1] * 20 + [0] * 20, dtype=np.int8)
    composite_base = np.asarray([1, 0] * 20, dtype=np.int8)
    composite_distill = 1 - composite_base
    result = R.atomic_supportive_bootstrap(
        both, both, both, both, composite_base, composite_distill,
        n_bootstrap=3, seed=123,
    )["common_known_composite"]
    assert result["shown"] is True
    assert result["n_items"] == 20
    assert result["empty_recomputed_common_known_draws"] == 3
    assert result["conditional_intervals_available"] is False
    assert result["base_composite_rate_percentile_95_ci"] is None
    assert result["distill_composite_rate_percentile_95_ci"] is None
    assert result["distill_minus_base_risk_difference_percentile_95_ci"] is None
    assert result["empty_draw_rule"].startswith("withhold all conditional intervals")


def test_orderops_stage_uses_verified_local_tokenizers_and_exact_frozen_eligibility():
    source = inspect.getsource(R.run_orderops_stage)
    assert "local_files_only=True" in source
    assert 'eligibility != tokenizer_preflight["eligibility"]' in source
    assert 'typo_eligibility != tokenizer_preflight["typo_eligibility"]' in source
    assert "tokenizer=tokenizers[cell_name]" in source
    assert 'checkpoint["cell_manifest"] != expected_cell_manifest' in source
    assert "revision=spec" not in source


def test_main_does_not_make_keyboard_interrupt_resumable_and_analysis_binds_rank_marker():
    source = inspect.getsource(R.main)
    assert "except Exception as exc:" in source
    assert "except BaseException as exc:" not in source
    assert 'run_dir / "RANKS_COMPLETE"' in source
