import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_duplicate_provenance.py"


def _load_auditor():
    spec = importlib.util.spec_from_file_location(
        "audit_duplicate_provenance", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _record(
    behaviour,
    chain_id,
    annotation_index,
    *,
    char_offset=0,
    token_start=0,
    n_positions=1,
    source_index=0,
):
    return {
        "behaviour": behaviour,
        "chain_id": chain_id,
        "annotation_index": annotation_index,
        "char_offset": char_offset,
        "token_start": token_start,
        "n_positions": n_positions,
        "source_index": source_index,
    }


def test_auditor_classifies_cross_label_duplicate_causes_without_vectors():
    auditor = _load_auditor()
    rows = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [4.0, 4.0, 4.0],
            [4.0, 4.0, 4.0],
            [1.0, 3.0, 2.0],
            [1.0, 3.0, 2.0],
            [9.0, 8.0, 7.0],
        ]
    )
    records = [
        _record("backtracking", "c1", 1, char_offset=10, token_start=2),
        _record(
            "uncertainty-estimation",
            "c1",
            1,
            char_offset=10,
            token_start=2,
        ),
        _record("example-testing", "c2", 1, char_offset=10),
        _record("adding-knowledge", "c2", 2, char_offset=20),
        _record("backtracking", "c3", 1, char_offset=10),
        _record("uncertainty-estimation", "c3", 2, char_offset=20),
        _record("example-testing", "c4", 1, char_offset=10),
        _record("adding-knowledge", "c4", 2, char_offset=20),
        _record("backtracking", "c5", 1),
    ]
    report = auditor.audit_exact_vector_provenance(
        rows,
        records,
        input_hashes={"fixture": "a" * 64},
    )
    summary = report["summary"]
    assert summary["exact_vector_duplicate_groups"] == 4
    assert summary["cross_label_groups"] == 4
    assert summary["cross_chain_groups"] == 0
    assert summary["classifications"] == {
        "same_location_multilabel": 1,
        "shared_zero_vector": 1,
        "shared_constant_vector": 1,
        "same_chain_distinct_location_cross_label_exact_match": 1,
    }
    assert {
        group["classification"] for group in report["groups"]
    } == set(summary["classifications"])
    for group in report["groups"]:
        assert set(group) == {
            "group_index",
            "vector_sha256",
            "size",
            "labels",
            "chain_ids",
            "cross_label",
            "cross_chain",
            "zero_vector",
            "constant_vector",
            "same_location_ignoring_label",
            "same_annotation_location_ignoring_label",
            "same_extraction_window",
            "classification",
            "occurrences",
        }
        assert len(group["vector_sha256"]) == 64
        assert "vector" not in group
        for occurrence in group["occurrences"]:
            assert set(occurrence) == {
                "occurrence_sha256",
                "behaviour",
                "chain_id",
                "annotation_index",
                "char_offset",
                "token_start",
                "n_positions",
                "source_index",
            }


def test_auditor_collapses_repeated_occurrences_before_vector_grouping():
    auditor = _load_auditor()
    rows = np.asarray([[1.0, 2.0], [8.0, 9.0], [3.0, 4.0]])
    records = [
        _record("backtracking", "c1", 0, source_index=0),
        _record("backtracking", "c1", 0, source_index=1),
        _record("uncertainty-estimation", "c2", 0, source_index=0),
    ]
    report = auditor.audit_exact_vector_provenance(
        rows,
        records,
        input_hashes={"fixture": "b" * 64},
    )
    assert report["summary"]["raw_rows"] == 3
    assert report["summary"]["rows_after_occurrence_collapse"] == 2
    assert report["summary"]["occurrence_duplicate_groups"] == 1
    assert report["summary"]["occurrence_duplicate_rows"] == 1
    assert report["summary"]["exact_vector_duplicate_groups"] == 0


def test_auditor_marks_cross_chain_exact_match():
    auditor = _load_auditor()
    rows = np.asarray([[1.0, 2.0], [1.0, 2.0]])
    records = [
        _record("backtracking", "c1", 0),
        _record("backtracking", "c2", 0),
    ]
    report = auditor.audit_exact_vector_provenance(
        rows,
        records,
        input_hashes={"fixture": "c" * 64},
    )
    assert report["summary"]["cross_chain_groups"] == 1
    assert report["summary"]["cross_label_groups"] == 0
    assert report["groups"][0]["classification"] == "cross_chain_exact_match"


def test_auditor_distinguishes_annotation_records_for_one_extraction_window():
    auditor = _load_auditor()
    rows = np.asarray([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
    records = [
        _record(
            "backtracking",
            "c1",
            7,
            char_offset=120,
            token_start=33,
            n_positions=11,
        ),
        _record(
            "uncertainty-estimation",
            "c1",
            19,
            char_offset=120,
            token_start=33,
            n_positions=11,
        ),
        _record("backtracking", "c1", 20, char_offset=180, token_start=48),
    ]
    report = auditor.audit_exact_vector_provenance(
        rows,
        records,
        input_hashes={"fixture": "e" * 64},
        minimum_target_chains=1,
    )
    group = report["groups"][0]
    assert group["same_extraction_window"] is True
    assert group["same_annotation_location_ignoring_label"] is False
    assert group["classification"] == (
        "same_extraction_window_multilabel_annotation_alias"
    )
    assert report["root_cause_classification"] == (
        "same_extraction_window_reused_across_annotation_aliases"
    )


def test_drop_all_conflicted_window_feasibility_preserves_target_chains():
    auditor = _load_auditor()
    rows = np.asarray(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ]
    )
    records = [
        _record("backtracking", "c1", 0, char_offset=10, token_start=2),
        _record(
            "uncertainty-estimation",
            "c1",
            1,
            char_offset=10,
            token_start=2,
        ),
        _record("backtracking", "c1", 2, char_offset=20, token_start=4),
        _record("adding-knowledge", "c1", 3, char_offset=30, token_start=6),
        _record("backtracking", "c2", 0, char_offset=10, token_start=2),
        _record(
            "uncertainty-estimation",
            "c2",
            1,
            char_offset=20,
            token_start=4,
        ),
    ]
    report = auditor.audit_exact_vector_provenance(
        rows,
        records,
        input_hashes={"fixture": "f" * 64},
        minimum_target_chains=1,
    )
    feasibility = report["remedy_feasibility"]
    assert feasibility["proposed_rule"] == (
        "drop_all_members_of_same_extraction_window_cross_label_groups"
    )
    assert feasibility["applicable_groups"] == 1
    assert feasibility["rows_removed_by_proposed_rule"] == 2
    assert feasibility["target_chains_before"] == 2
    assert feasibility["target_chains_after"] == 2
    assert feasibility["target_chains_lost"] == 0
    assert feasibility["all_chains_lost"] == 0
    assert feasibility["mixed_chain_fraction_after"] == 1.0
    assert feasibility["sample_minimum_met"] is True
    assert feasibility["mixed_chain_fraction_gate_met"] is True
    assert feasibility["positive"] is True


def test_audit_artifacts_are_atomic_and_contain_no_raw_vectors(tmp_path):
    auditor = _load_auditor()
    report = auditor.audit_exact_vector_provenance(
        np.asarray([[1.0, 2.0], [1.0, 2.0]]),
        [
            _record("backtracking", "c1", 0),
            _record("uncertainty-estimation", "c1", 0),
        ],
        input_hashes={"fixture": "d" * 64},
    )
    json_path = tmp_path / "audit.json"
    markdown_path = tmp_path / "audit.md"
    auditor.write_audit_artifacts(
        json_path, markdown_path, report
    )
    assert json.loads(json_path.read_text()) == report
    markdown = markdown_path.read_text()
    assert report["root_cause_classification"] in markdown
    assert "Remedy feasibility" in markdown
    assert f"Positive: `{str(report['remedy_feasibility']['positive']).upper()}`" in markdown
    assert "raw activation vectors" not in markdown.casefold()
    assert not list(tmp_path.glob(".*.tmp"))
