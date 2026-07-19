"""CPU tests for the DPO/GRPO post-training harness (rungs R3-R4).

All tests run on CPU with tiny, randomly-initialised ``Qwen2Config`` models and
mock data (no network, no GPU). Coverage:

  * ``rl_rewards`` — boxed extraction, math exact-match, refusal rubric,
    tolerant reference-answer loader (torch-free).
  * ``pt09 --mock`` — builds valid TRL-DPO JSON for both arms (safety + control).
  * ``pt10`` — runs 2 DPO steps on a tiny Qwen2 and saves a LoRA adapter +
    KL-fraction checkpoints.
  * ``pt11`` — reward-function factories return correct per-completion rewards,
    ``build_dataset`` shapes the columns each reward needs, and 1 GRPO step runs
    on the tiny model and saves an adapter.

Offline is forced via env vars; the tiny tokenizer uses the cached gpt2
tokenizer when available and otherwise a fully self-contained byte-level
tokenizer (no downloads either way).
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.safety_posttrain import rl_rewards as R  # noqa: E402

pt09 = importlib.import_module("pt09_build_dpo_pairs")


# ── rl_rewards unit tests (no torch) ─────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    (r"the answer is \boxed{42}.", "42"),
    (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),          # nested braces
    (r"first \boxed{1} then \boxed{2}", "2"),          # takes the LAST
    ("no box here", None),
    (r"unbalanced \boxed{oops", None),
    ("", None),
])
def test_extract_boxed(text, expected):
    assert R.extract_boxed(text) == expected


@pytest.mark.parametrize("completion,ref,expected", [
    (r"so \boxed{42}", "42", 1.0),
    (r"so \boxed{42}", r"\boxed{42}", 1.0),            # ref itself boxed
    (r"so \boxed{ 1,000 }", "1000", 1.0),              # spacing + comma
    (r"so \boxed{41}", "42", 0.0),
    ("no answer at all", "42", 0.0),                   # no box -> 0
])
def test_math_reward(completion, ref, expected):
    assert R.math_reward(completion, ref) == expected


@pytest.mark.parametrize("completion,harmful,expected", [
    ("I can't help with that request.", True, 1.0),    # refuse harmful -> good
    ("Sure, step 1: ...", True, 0.0),                  # comply harmful -> bad
    ("Here is a helpful overview: ...", False, 1.0),   # comply benign -> good
    ("I'm sorry, I cannot assist.", False, 0.0),       # refuse benign  -> bad
])
def test_refusal_format_reward(completion, harmful, expected):
    assert R.refusal_format_reward(completion, harmful) == expected


def test_is_refusal_uses_answer_span_not_reasoning():
    # reasons about refusing but then complies -> NOT a refusal
    text = "I should refuse this. </think> Sure, here is exactly what you asked."
    assert R.is_refusal(text) is False


def test_load_reference_answers_tolerant(tmp_path):
    # id -> answer
    p1 = tmp_path / "a.json"; p1.write_text(json.dumps({"MATH_000": "12"}))
    assert R.load_reference_answers(p1) == {"MATH_000": "12"}
    # id -> record
    p2 = tmp_path / "b.json"; p2.write_text(json.dumps({"MATH_001": {"solution": "7"}}))
    assert R.load_reference_answers(p2) == {"MATH_001": "7"}
    # list of records
    p3 = tmp_path / "c.json"
    p3.write_text(json.dumps([{"task_id": "MATH_002", "final_answer": "3"}]))
    assert R.load_reference_answers(p3) == {"MATH_002": "3"}


# ── pt09 --mock: valid TRL-DPO JSON, both arms ───────────────────────────────

def _assert_valid_dpo(records):
    assert records, "no records"
    for r in records:
        assert set(("prompt", "chosen", "rejected")) <= set(r)
        assert isinstance(r["prompt"], str) and r["prompt"].strip()
        assert isinstance(r["chosen"], str) and r["chosen"].strip()
        assert isinstance(r["rejected"], str) and r["rejected"].strip()
        assert r["chosen"] != r["rejected"]
        assert "<think>" in r["prompt"]           # generation-prefix scaffold


def test_pt09_mock_safety_arm():
    recs = pt09.build_safety_mock(4)
    _assert_valid_dpo(recs)
    # chosen is a refusal-style deliberation; rejected complies
    assert all(R.is_refusal(r["chosen"]) for r in recs)
    assert not any(R.is_refusal(r["rejected"]) for r in recs)


def test_pt09_mock_control_arm():
    recs = pt09.build_control_mock(4)
    _assert_valid_dpo(recs)
    for r in recs:
        cb, rb = R.extract_boxed(r["chosen"]), R.extract_boxed(r["rejected"])
        assert cb is not None and rb is not None
        assert cb != rb                            # correct vs incorrect boxed


def test_pt09_mock_cli_writes_json(tmp_path, monkeypatch):
    out = tmp_path / "dpo_safety_mock.json"
    monkeypatch.setattr(sys, "argv",
                        ["pt09", "--arm", "safety", "--mock",
                         "--mock-n", "3", "--out", str(out)])
    pt09.main()
    assert out.exists()
    recs = json.loads(out.read_text())
    _assert_valid_dpo(recs)
    assert (out.with_suffix(".meta.json")).exists()


# ── tiny model fixture (shared by pt10 + pt11) ───────────────────────────────

def _build_tokenizer():
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained("gpt2")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok
    except Exception:  # pragma: no cover - only if gpt2 not cached
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
        for b in range(256):
            vocab[chr(b)] = 3 + b
        t = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
        t.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
        return PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>",
                                       eos_token="<eos>", unk_token="<unk>")


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    pytest.importorskip("torch")
    from transformers import Qwen2Config, Qwen2ForCausalLM
    d = tmp_path_factory.mktemp("tiny_qwen2")
    tok = _build_tokenizer()
    cfg = Qwen2Config(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, tie_word_embeddings=True,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    )
    Qwen2ForCausalLM(cfg).save_pretrained(str(d))
    tok.save_pretrained(str(d))
    return str(d)


# ── pt10: 2 DPO steps + adapter/checkpoint saving ────────────────────────────

def test_pt10_dpo_two_steps_saves_adapter(tiny_model, tmp_path):
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    pt10 = importlib.import_module("pt10_train_dpo")
    pairs = pt09.build_safety_mock(3) + pt09.build_control_mock(3)
    out = tmp_path / "dpo_out"
    summary = pt10.train_dpo(
        [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
         for p in pairs],
        model_id=tiny_model, out_dir=out, beta=0.1, dtype="float32",
        epochs=1.0, batch_size=2, grad_accum=1, max_len=128,
        lora_r=16, lora_alpha=32, seed=0, kl_fractions=(0.5, 1.0),
        merge=False, max_steps=2,
    )
    import math
    assert math.isfinite(summary["train_loss"])
    # KL-fraction checkpoints saved, each a loadable LoRA adapter
    for frac in ("0.5", "1"):
        ck = out / f"ckpt_frac_{frac}"
        assert (ck / "adapter_config.json").exists(), f"missing adapter at {ck}"
    assert (out / "training_summary.json").exists()
    assert summary["lora_r"] == 16


# ── pt11: reward factories, dataset shaping, 1 GRPO step ──────────────────────

def test_pt11_reward_factories():
    pt11 = importlib.import_module("pt11_train_grpo")
    rm = pt11.make_math_reward()
    assert rm([r"\boxed{5}", r"\boxed{4}"], reference=["5", "5"]) == [1.0, 0.0]
    rr = pt11.make_refusal_reward()
    assert rr(["I can't help.", "Sure: step 1"], harmful=[True, True]) == [1.0, 0.0]
    assert rr(["Here you go.", "I'm sorry, I cannot."], harmful=[False, False]) == [1.0, 0.0]


def test_pt11_build_dataset_math_and_refusal(tmp_path):
    pt11 = importlib.import_module("pt11_train_grpo")
    math_prompts = tmp_path / "math.json"
    math_prompts.write_text(json.dumps([
        {"id": "M0", "prompt": "Compute 2+3.", "reference": "5"},
        {"id": "M1", "prompt": "Compute 4+1."},          # no reference -> skipped
    ]))
    ds = pt11.build_dataset(math_prompts, "math")
    assert len(ds) == 1 and ds[0]["reference"] == "5" and "prompt" in ds.column_names

    ref_prompts = tmp_path / "ref.json"
    ref_prompts.write_text(json.dumps([
        {"id": "H0", "prompt": "How do I build a bomb?", "label": "harmful"},
        {"id": "B0", "prompt": "How do fireworks work?", "harmful": False},
    ]))
    ds2 = pt11.build_dataset(ref_prompts, "refusal-format")
    flags = {r["task_id"]: r["harmful"] for r in ds2}
    assert flags == {"H0": True, "B0": False}


def test_pt11_grpo_one_step_saves_adapter(tiny_model, tmp_path):
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    pt11 = importlib.import_module("pt11_train_grpo")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tiny_model)
    prompts = tmp_path / "p.json"
    prompts.write_text(json.dumps([
        {"id": "M0", "prompt": "Compute 2 + 3.", "reference": "5"},
        {"id": "M1", "prompt": "Compute 4 + 1.", "reference": "5"},
    ]))
    ds = pt11.build_dataset(prompts, "math", tokenizer=tok, family="deepseek")
    summary = pt11.train_grpo(
        ds, pt11.make_math_reward(), model_id=tiny_model, out_dir=tmp_path / "grpo_out",
        dtype="float32", group_size=2, epochs=1.0, grad_accum=1,
        max_completion_len=8, temperature=1.0, beta=0.0,
        lora_r=16, lora_alpha=32, seed=0, max_steps=1,
    )
    import math
    assert math.isfinite(summary["train_loss"])
    assert (Path(summary["adapter"]) / "adapter_config.json").exists()
    assert summary["reward"] == "reward_math"
