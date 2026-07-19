"""Fast CPU end-to-end tests for pt08_surprisal_entropy.py (surprisal + entropy).

No GPU, no downloads: a tiny randomly-initialised Qwen2 pair (hidden 32,
2 layers) with a toy whitespace WordLevel fast tokenizer (offset mapping works,
vocab matched to the model) is saved to disk and run through the script's real
main() with --device cpu, exactly as a pod run would go through it. Checks:

  - the output JSON has the pre-registered structure (config / corpus /
    surprisal_control / entropy_battery) and the per-span join keys,
  - KL(post||base) >= 0 everywhere and > 0 for genuinely different weights,
  - predictive entropies are positive,
  - base == post (same checkpoint twice) gives KL ~ 0, dNLL ~ 0, dH ~ 0.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

import pt08_surprisal_entropy as pt08

# ── Toy corpus (plain alphanumeric words: whitespace pre-tokenizer keeps each
#    word a single token so char offsets are exact) ──────────────────────────

PROMPT_1 = "please solve the puzzle now "
CHAIN_1 = ("Okay so I start by testing the small case first and it works "
           "Wait that was wrong I go back and try the other route "
           "so the answer is four")
SPAN_1 = "Wait that was wrong I go back and try the other route"  # backtracking

PROMPT_2 = "please count the steps here "
CHAIN_2 = ("First I list every step in order "
           "maybe I am not sure this even holds "
           "still the total comes to nine")
SPAN_2 = "maybe I am not sure this even holds"  # uncertainty-estimation


def _toy_tokenizer() -> PreTrainedTokenizerFast:
    words = sorted(set((PROMPT_1 + CHAIN_1 + " " + PROMPT_2 + CHAIN_2).split()))
    vocab = {"<unk>": 0, "<pad>": 1}
    for w in words:
        vocab[w] = len(vocab)
    tok = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok,
                                   unk_token="<unk>", pad_token="<pad>")


def _tiny_qwen2(vocab_size: int, seed: int) -> Qwen2ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, tie_word_embeddings=False,
        pad_token_id=1,
    )
    return Qwen2ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """Two tiny checkpoints (different seeds) + tokenizer + fake annotations."""
    root = tmp_path_factory.mktemp("pt08")
    tokenizer = _toy_tokenizer()
    v = tokenizer.vocab_size

    dirs = {}
    for name, seed in (("base", 0), ("post", 1)):
        d = root / name
        _tiny_qwen2(v, seed).save_pretrained(d)
        tokenizer.save_pretrained(d)
        dirs[name] = str(d)

    annotated = [
        {"task_id": "TOY_000", "prompt": PROMPT_1, "chain": CHAIN_1,
         "annotations": [{"label": "backtracking", "text": SPAN_1}]},
        {"task_id": "TOY_001", "prompt": PROMPT_2, "chain": CHAIN_2,
         "annotations": [{"label": "uncertainty-estimation", "text": SPAN_2}]},
    ]
    ann_path = root / "annotated_toy.json"
    ann_path.write_text(json.dumps(annotated))
    return {"root": root, "dirs": dirs, "annotated": str(ann_path)}


def _run(setup, base: str, post: str, out_name: str) -> dict:
    out = setup["root"] / out_name
    pt08.main([
        "--base", base, "--post", post,
        "--annotated", setup["annotated"],
        "--device", "cpu", "--dtype", "float32", "--batch-size", "8",
        "--out", str(out),
    ])
    with open(out) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def report(setup) -> dict:
    return _run(setup, setup["dirs"]["base"], setup["dirs"]["post"], "diff.json")


# ── Structure ────────────────────────────────────────────────────────────────

def test_report_has_registered_structure(report):
    for key in ("provenance", "config", "corpus", "surprisal_control",
                "entropy_battery"):
        assert key in report, key
    assert report["corpus"]["n_chains"] == 2
    assert report["corpus"]["n_chains_scored"] == 2
    assert report["corpus"]["n_target_tokens"] > 0
    assert report["corpus"]["n_spans_located"] == 2
    assert report["corpus"]["n_spans_skipped"] == 0
    battery = report["entropy_battery"]
    for key in ("overall", "prompt_tokens", "chain_tokens",
                "any_annotated_span", "by_behaviour"):
        assert key in battery, key


def test_per_span_table_has_join_keys(report):
    rows = report["surprisal_control"]["per_span"]
    assert len(rows) == 2
    assert {r["behaviour"] for r in rows} == {"backtracking",
                                              "uncertainty-estimation"}
    for r in rows:
        for key in ("chain_id", "behaviour", "annotation_index",
                    "mean_nll_base", "mean_nll_post", "delta_nll",
                    "n_window_tokens", "n_span_tokens",
                    "mean_nll_base_span", "mean_nll_post_span", "mean_kl_span",
                    "mean_dH_span"):
            assert key in r, key
        assert r["chain_id"] in {"TOY_000", "TOY_001"}
        assert r["annotation_index"] == 0
        # window is capped at 1 preceding + 10 execution tokens
        assert 0 < r["n_window_tokens"] <= 11
        assert r["n_span_tokens"] > 0


def test_per_behaviour_delta_nll_reported(report):
    pb = report["surprisal_control"]["per_behaviour"]
    for b in ("backtracking", "uncertainty-estimation"):
        assert pb[b]["n_spans"] == 1
        assert pb[b]["mean_delta_nll_window"] is not None
        assert pb[b]["mean_nll_base_window"] > 0


# ── Numerics (different weights) ─────────────────────────────────────────────

def test_entropy_positive_and_kl_nonnegative(report):
    ov = report["entropy_battery"]["overall"]
    assert ov["mean_H_base"] > 0
    assert ov["mean_H_post"] > 0
    assert ov["mean_kl"] >= 0
    # genuinely different random weights => strictly positive dose
    assert ov["mean_kl"] > 1e-4
    for r in report["surprisal_control"]["per_span"]:
        assert r["mean_kl_span"] >= -1e-6


def test_on_off_span_split_covers_chain_tokens(report):
    battery = report["entropy_battery"]
    chain_n = battery["chain_tokens"]["n_tokens"]
    for b in ("backtracking", "uncertainty-estimation"):
        on = battery["by_behaviour"][b]["on_span"]
        off = battery["by_behaviour"][b]["off_span"]
        assert on["n_tokens"] > 0
        assert on["n_tokens"] + off["n_tokens"] == chain_n
        assert on["mean_kl"] >= 0 and off["mean_kl"] >= 0
    any_split = battery["any_annotated_span"]
    assert (any_split["on_span"]["n_tokens"]
            + any_split["off_span"]["n_tokens"]) == chain_n


# ── base == post null ────────────────────────────────────────────────────────

def test_identical_models_give_zero_kl_and_zero_deltas(setup):
    rep = _run(setup, setup["dirs"]["base"], setup["dirs"]["base"], "same.json")
    ov = rep["entropy_battery"]["overall"]
    assert abs(ov["mean_kl"]) < 1e-5
    assert abs(ov["mean_delta_nll"]) < 1e-5
    assert abs(ov["mean_dH"]) < 1e-5
    assert ov["mean_H_base"] > 0  # entropy itself is not degenerate
    for r in rep["surprisal_control"]["per_span"]:
        assert abs(r["delta_nll"]) < 1e-4
        assert abs(r["mean_kl_span"]) < 1e-5
