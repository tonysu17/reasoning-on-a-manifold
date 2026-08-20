from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "venhoff_corpus_latent_analysis",
    ROOT / "venhoff_corpus_latent_analysis.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CharacterTokenizer:
    """Minimal tokenizer whose token offsets are one token per character."""

    def __call__(self, text, **_kwargs):
        assert isinstance(text, str)
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_released_and_occurrence_aware_locators_diverge_on_repeated_text():
    response = "USER repeats target. <think>target. bridge target.</think>"
    thinking = "target. bridge target."
    annotation = (
        '["deduction"]target.["end-section"] '
        '["backtracking"]target.["end-section"]'
    )
    segments = MODULE.parse_segments(annotation)
    released = MODULE.released_char_locations(response, segments)
    corrected, boundary = MODULE.occurrence_aware_char_locations(
        response, thinking, segments
    )

    assert released[0]["start_char"] == released[1]["start_char"]
    assert corrected[0]["start_char"] > response.index("<think>")
    assert corrected[1]["start_char"] > corrected[0]["start_char"]
    assert boundary["thinking_boundary_status"] == "exact_thinking_process"


def test_released_token_pool_bounds_match_author_code_slice():
    mapping = MODULE.char_to_token_map([(i, i + 1) for i in range(30)])
    locations = MODULE.attach_token_bounds(
        [
            {
                "annotation_index": 0,
                "label": "deduction",
                "text": "abcdefghijklmno",
                "start_char": 5,
                "end_char": 20,
                "resolution": "test",
            }
        ],
        mapping,
    )
    location = locations[0]
    assert location["token_start"] == 5
    assert location["token_end"] == 20
    assert (location["pool_start"], location["pool_stop"]) == (4, 15)


def test_static_audit_compares_released_counts_to_official():
    record = {
        "full_response": "P <think>alpha beta</think>",
        "thinking_process": "alpha beta",
        "annotated_thinking": '["deduction"]alpha beta["end-section"]',
    }
    official = {"deduction": {"count": 1}, "overall": {"count": 1}}
    audit = MODULE.static_audit([record], CharacterTokenizer(), official)
    assert audit["released_counts_exactly_match_official"] is True
    assert audit["released_counts"]["deduction"] == 1
    assert audit["released_counts"]["overall"] == 1


def test_alignment_reports_squared_norm_and_random_reference():
    basis = np.eye(4)
    direction = np.array([1.0, 1.0, 0.0, 0.0])
    old_k = MODULE.K_VALUES
    MODULE.K_VALUES = (1, 2, 3)
    try:
        result = MODULE.alignment(direction, basis)
    finally:
        MODULE.K_VALUES = old_k
    assert np.isclose(result["by_k"]["1"]["captured_squared_norm"], 0.5)
    assert np.isclose(result["by_k"]["2"]["retained_norm"], 1.0)
    assert np.isclose(
        result["random_isotropic_expected_captured_squared_norm"]["1"], 0.25
    )


def test_pca_cloud_keeps_estimands_separate():
    # Symmetric rows make the cloud exactly centred.  Per-axis variances are
    # proportional to 9, 4, 1, so the first two axes exceed 70% together.
    rows = []
    for axis, scale in enumerate((3.0, 2.0, 1.0)):
        vector = np.zeros(3, dtype=np.float32)
        vector[axis] = scale
        rows.extend([vector, -vector])
    components, eigenvalues, mean, metrics = MODULE.pca_cloud([np.stack(rows)])
    assert components.shape == (3, 3)
    assert eigenvalues.shape == (3,)
    assert np.allclose(mean, 0.0)
    assert metrics["d_eff70_variance_threshold_dimension"] == 2
    expected_pr = (9 + 4 + 1) ** 2 / (9**2 + 4**2 + 1**2)
    assert np.isclose(metrics["participation_ratio"], expected_pr)
    # Top-ten is undefined for hidden dimensions below ten; production hidden
    # size is 1536, so use a padded version to exercise that estimand.
    padded = np.pad(np.stack(rows), ((0, 0), (0, 7)))
    _, _, _, padded_metrics = MODULE.pca_cloud([padded])
    assert np.isclose(padded_metrics["fixed_top10_variance_concentration"], 1.0)


def test_basis_validation_is_sign_invariant():
    basis = np.eye(10)
    sign_flipped = basis.copy()
    sign_flipped[0] *= -1
    result = MODULE.basis_validation(basis, sign_flipped)
    assert np.isclose(result["3"]["minimum_principal_cosine"], 1.0)
    assert np.isclose(result["10"]["maximum_principal_angle_degrees"], 0.0)


def test_failed_fidelity_gate_unlicenses_local_cloud_claim():
    status = MODULE.analysis_disposition({"high_fidelity_gate": False})
    assert status["strict_high_fidelity_gate"] is False
    assert status["venhoff_cloud_pca_claim_licensed"] is False
    assert status["actual_disposition"] == "unlicensed conditional local-runtime sensitivity"
    assert "not a reproduced Venhoff PCA result" in status["claim_boundary"]


def test_official_runtime_pin_audit_separates_runtime_from_hf_revision(tmp_path):
    environment = tmp_path / "environment.yaml"
    environment.write_text(
        "\n".join(
            [
                "- torch==2.5.1",
                "- transformers==4.47.1",
                "- nnsight==0.4.5",
                "- nvidia-cuda-runtime-cu12==12.4.127",
            ]
        )
    )
    result = MODULE.verify_official_runtime_pins(environment)
    assert result["cuda_stack_pinned"] is True
    assert result["hf_model_revision_pinned"] is False
    assert result["pinned_packages"]["transformers"] == "4.47.1"
