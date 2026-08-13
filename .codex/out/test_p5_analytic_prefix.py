from __future__ import annotations

import copy

import pytest

import p5_analytic_prefix as prefix


TOK = "tokenizer-hash"
CFG = "tokenizer-config-hash"


def decoder(ids):
    return " ".join(str(token) for token in ids)


def row(ids, stop_reason="eos", raw_cap=8192):
    return {
        "checkpoint_role": "phase2_base_r1_shared",
        "task_id": "task-001",
        "generation_config_sha256": "generation-config-hash",
        "generated_token_ids": ids,
        "n_tokens": len(ids),
        "stop_reason": stop_reason,
        "raw_max_new_tokens": raw_cap,
        "text": decoder(ids),
    }


def test_eos_completion_at_or_before_prefix_is_preserved_exactly():
    source = row([10, 20, 30])
    result = prefix.make_prefix_record(source, decoder, tokenizer_sha256=TOK, tokenizer_config_sha256=CFG)
    assert result["prefix_token_ids"] == [10, 20, 30]
    assert result["prefix_n_tokens"] == 3
    assert result["prefix_stop_reason"] == "eos"
    assert result["prefix_text"] == source["text"]


def test_long_source_is_cut_on_original_id_boundary_not_text():
    source = row(list(range(5000)), stop_reason="other")
    result = prefix.make_prefix_record(source, decoder, tokenizer_sha256=TOK, tokenizer_config_sha256=CFG)
    assert result["prefix_n_tokens"] == 4096
    assert result["prefix_token_ids"] == list(range(4096))
    assert result["prefix_stop_reason"] == "analytic_prefix_cap"
    assert result["source_n_tokens"] == 5000


def test_length_stop_must_match_raw_cap():
    valid = row(list(range(20)), stop_reason="length", raw_cap=20)
    assert prefix.validate_generated_ids(valid) == list(range(20))
    invalid = row(list(range(20)), stop_reason="length", raw_cap=21)
    with pytest.raises(prefix.PrefixError):
        prefix.validate_generated_ids(invalid)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("generated_token_ids"),
        lambda value: value.update(n_tokens=999),
        lambda value: value.update(generated_token_ids=[1, True]),
        lambda value: value.update(generated_token_ids=[1, -1]),
        lambda value: value.update(generated_token_ids=[1, prefix.EOS_TOKEN_ID]),
        lambda value: value.update(stop_reason="unknown"),
        lambda value: value.update(raw_max_new_tokens=None),
    ],
)
def test_invalid_source_rows_fail_closed(mutation):
    source = row([1, 2])
    mutation(source)
    with pytest.raises(prefix.PrefixError):
        prefix.validate_generated_ids(source)


def test_complete_decode_mismatch_and_source_hash_mismatch_fail():
    source = row([1, 2])
    broken = copy.deepcopy(source)
    broken["text"] = "different"
    with pytest.raises(prefix.PrefixError):
        prefix.make_prefix_record(broken, decoder, tokenizer_sha256=TOK, tokenizer_config_sha256=CFG)
    with pytest.raises(prefix.PrefixError):
        prefix.make_prefix_record(
            source,
            decoder,
            tokenizer_sha256=TOK,
            tokenizer_config_sha256=CFG,
            source_row_sha256="wrong",
        )


def test_conflicting_duplicate_prefix_rows_fail():
    source = row([1, 2])
    first = prefix.make_prefix_record(source, decoder, tokenizer_sha256=TOK, tokenizer_config_sha256=CFG)
    second = copy.deepcopy(first)
    second["prefix_text"] = "changed"
    with pytest.raises(prefix.PrefixError):
        prefix.validate_unique_prefix_rows([first, second])
