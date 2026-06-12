"""Tests for src/row_provenance.py — sidecar-first chain ids, fail-loud
alignment (replacing the proxy-chain fallback that neutered the null), and
exact-duplicate hygiene (CF-13)."""

import json

import numpy as np
import pytest

from src.row_provenance import (
    ROW_INDEX_FILE,
    chain_ids_for,
    dedup_rows,
    duplicate_fraction,
    load_row_index,
    require_aligned,
)


def _write_sidecar(act_dir, rows):
    act_dir.mkdir(parents=True, exist_ok=True)
    (act_dir / ROW_INDEX_FILE).write_text(json.dumps({
        "version": 1,
        "sentence_matching": "occurrence_aware_v1",
        "pooling": "mean",
        "rows": rows,
    }))


def test_sidecar_preferred_over_annotation_replay(tmp_path):
    act_dir = tmp_path / "acts"
    _write_sidecar(act_dir, {
        "backtracking": [
            {"chain_id": "c1", "annotation_index": 0, "char_offset": 0,
             "token_start": 1, "n_positions": 11},
            {"chain_id": "c2", "annotation_index": 3, "char_offset": 40,
             "token_start": 9, "n_positions": 11},
        ],
    })
    # annotation file deliberately ABSENT: sidecar must be sufficient
    out = chain_ids_for(act_dir, tmp_path / "missing.json", ["backtracking", "uncertainty-estimation"])
    assert list(out["backtracking"]) == ["c1", "c2"]
    assert out["uncertainty-estimation"] is None


def test_replay_fallback_is_occurrence_aware(tmp_path):
    act_dir = tmp_path / "acts_no_sidecar"
    act_dir.mkdir()
    s = "Wait, I need to reconsider this step."
    annotated = [{
        "chain_id": "chainA",
        "chain": f"Intro. {s} middle text. {s} end.",
        "annotations": [
            {"label": "backtracking", "text": s},
            {"label": "deduction", "text": "middle text."},
            {"label": "backtracking", "text": s},
        ],
    }]
    p = tmp_path / "annotated.json"
    p.write_text(json.dumps(annotated))
    out = chain_ids_for(act_dir, p, ["backtracking"])
    # both repeats located (occurrence-aware) -> two rows, same chain
    assert list(out["backtracking"]) == ["chainA", "chainA"]


def test_load_row_index_absent(tmp_path):
    assert load_row_index(tmp_path) is None


def test_require_aligned_passes_and_returns_array():
    ids = np.array(["a", "b", "c"])
    assert list(require_aligned("backtracking", 3, ids)) == ["a", "b", "c"]


def test_require_aligned_raises_on_none():
    with pytest.raises(RuntimeError, match="proxy-chain"):
        require_aligned("backtracking", 5, None, context="test")


def test_require_aligned_raises_on_length_mismatch():
    with pytest.raises(RuntimeError, match="mismatch"):
        require_aligned("backtracking", 5, np.array(["a", "b"]), context="test")


def test_duplicate_fraction_and_dedup_rows():
    X = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [1.0, 2.0]])
    cids = np.array(["c1", "c2", "c3", "c4"])
    labels = np.array(["b", "b", "u", "b"])
    assert duplicate_fraction(X) == pytest.approx(0.5)
    Xu, cu, lu = dedup_rows(X, cids, labels)
    # keep FIRST occurrence, preserve original order
    assert Xu.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert list(cu) == ["c1", "c3"]
    assert list(lu) == ["b", "u"]


def test_dedup_rows_no_duplicates_is_identity():
    X = np.array([[1.0], [2.0], [3.0]])
    (Xu,) = dedup_rows(X)
    assert Xu.tolist() == X.tolist()
    assert duplicate_fraction(X) == 0.0
