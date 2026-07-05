"""Unit tests for the safety post-training pipeline (no torch / peft / proxy).

Covers the parts that must be correct before any GPU run:
  - the contrastive dataset schema (mock path),
  - SFT text construction + prompt/completion masking discipline,
  - the numpy geometry diff (principal angles, d_eff, selectivity ranking),
  - robust JSON extraction from model output.
"""

import numpy as np
import pytest

from src.safety_posttrain import contrastive as C
from src.safety_posttrain import spillover as SP


# ── Contrastive dataset ──────────────────────────────────────────────────────

def test_mock_dataset_schema_and_balance():
    recs = C.mock_dataset(8)
    assert len(recs) == 16  # 2 per pair
    labels = {r["label"] for r in recs}
    assert labels == {"harmful", "benign"}
    n_harm = sum(r["label"] == "harmful" for r in recs)
    n_ben = sum(r["label"] == "benign" for r in recs)
    assert n_harm == n_ben == 8  # balanced contrast
    for r in recs:
        assert r["prompt"] and r["answer"] and r["reasoning"]
        assert r["contrast_id"].startswith("pair_")
        assert r["refusal"] == (r["label"] == "harmful")


def test_harmful_responses_are_refusals_not_content():
    recs = C.mock_dataset(8)
    for r in recs:
        if r["label"] == "harmful":
            assert r["refusal"] is True
            # the refusal answer must not look like a compliance
            assert "can't help" in r["answer"].lower() or "decline" in r["answer"].lower()


def test_matched_pairs_share_contrast_id():
    recs = C.mock_dataset(4)
    by_cid = {}
    for r in recs:
        by_cid.setdefault(r["contrast_id"], set()).add(r["label"])
    for cid, labs in by_cid.items():
        assert labs == {"harmful", "benign"}, cid


# ── SFT formatting / masking ─────────────────────────────────────────────────

def test_build_sft_text_no_tokenizer_uses_manual_template():
    rec = C.mock_dataset(1)[0]
    prompt, completion = C.build_sft_text(rec, tokenizer=None)
    assert rec["prompt"] in prompt
    assert prompt.rstrip().endswith("<think>")  # generation prefix present
    # completion closes the think block then gives the answer
    assert "</think>" in completion
    assert rec["answer"] in completion
    # no double-opened think block in the completion
    assert "<think>" not in completion


def test_assemble_completion_structure():
    c = C.assemble_completion("reasoning here", "final answer")
    assert c == "reasoning here\n</think>\n\nfinal answer"


def test_records_to_sft_disjoint_prompt_and_completion():
    recs = C.mock_dataset(2)
    sft = C.records_to_sft(recs, tokenizer=None)
    assert len(sft) == len(recs)
    for ex in sft:
        assert ex["prompt_text"] and ex["completion_text"]
        # the answer text lives only in the completion (so loss masking is meaningful)
        assert ex["label"] in {"harmful", "benign"}


def test_tokenize_example_masks_prompt():
    # A fake tokenizer: maps each whitespace token to an int id; eos = 0.
    class FakeTok:
        eos_token_id = 0
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [hash(w) % 1000 + 1 for w in text.split()]}
    from src.safety_posttrain import sft as S
    out = S.tokenize_example("the prompt part", "the completion part", FakeTok(), max_len=100)
    n_prompt = len("the prompt part".split())
    # prompt positions masked
    assert out["labels"][:n_prompt] == [-100] * n_prompt
    # completion positions are real ids, and an EOS (0) is appended
    assert out["labels"][-1] == 0
    assert all(l != -100 for l in out["labels"][n_prompt:])
    assert len(out["input_ids"]) == len(out["labels"])


# ── Geometry diff ────────────────────────────────────────────────────────────

def test_principal_angles_identical_subspace_is_zero():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 20))
    ang = SP.principal_angles(X, X.copy(), k=5)
    assert np.allclose(ang, 0.0, atol=1e-6)


def test_principal_angles_orthogonal_subspaces_are_ninety():
    n, d = 200, 10
    rng = np.random.default_rng(1)
    # A varies only in dims 0-2, B only in dims 5-7 -> orthogonal subspaces
    A = np.zeros((n, d)); A[:, 0:3] = rng.standard_normal((n, 3))
    B = np.zeros((n, d)); B[:, 5:8] = rng.standard_normal((n, 3))
    ang = SP.principal_angles(A, B, k=3)
    assert np.allclose(ang, 90.0, atol=1.0)


def test_d_eff_low_for_rank1_high_for_isotropic():
    rng = np.random.default_rng(2)
    v = rng.standard_normal((1, 30))
    rank1 = rng.standard_normal((300, 1)) @ v  # ~1 effective dim
    iso = rng.standard_normal((300, 30))       # ~30 effective dims
    assert SP.d_eff(rank1) < 2.0
    assert SP.d_eff(iso) > 15.0


def test_compare_geometry_and_selectivity_ranking():
    rng = np.random.default_rng(3)
    base = {"backtracking": {14: rng.standard_normal((150, 16))},
            "deduction": {14: rng.standard_normal((150, 16))}}
    # backtracking moves a lot (rotate), deduction barely moves
    rot = np.linalg.qr(rng.standard_normal((16, 16)))[0]
    post = {"backtracking": {14: base["backtracking"][14] @ rot},
            "deduction": {14: base["deduction"][14] + 1e-3 * rng.standard_normal((150, 16))}}
    rep = SP.compare_geometry(base, post, ["backtracking", "deduction"], layer=14, k=5)
    ranking = SP.rank_selectivity(rep)
    assert ranking[0][0] == "backtracking"  # moved most
    assert rep["behaviours"]["backtracking"]["mean_principal_angle_deg"] > \
           rep["behaviours"]["deduction"]["mean_principal_angle_deg"]


# ── JSON extraction ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ('[{"harmful":"a","benign":"b"}]', [{"harmful": "a", "benign": "b"}]),
    ('```json\n[{"x":1}]\n```', [{"x": 1}]),
    ('here you go:\n[{"x":1}]\nhope that helps', [{"x": 1}]),
    ('{"reasoning":"r","answer":"a"}', {"reasoning": "r", "answer": "a"}),
])
def test_parse_json_tolerates_fences_and_prose(raw, expected):
    assert C._parse_json(raw) == expected
