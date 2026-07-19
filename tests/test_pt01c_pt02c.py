"""Tests for pt01c (off-policy control builder) and pt02c (full-parameter SFT).

No network and no real models:
  - pt01c runs in --mock mode against a synthetic STAR-1 reference file and
    must produce schema-valid records whose completion word-length quantiles
    track the reference;
  - pt02c's training machinery (sft.train_full) runs 2 optimizer steps on CPU
    with a tiny randomly-initialised Qwen2 model and must save a checkpoint
    loadable the way 04_extract_activations.py --model-path loads it.
"""

import json
import random

import pytest

import pt01c_build_offpolicy_control as pt01c
from src.safety_posttrain import contrastive as C


# ── pt01c: mock builder ──────────────────────────────────────────────────────

def _write_synth_star1(path, n=60, seed=0):
    """A STAR-1-shaped reference whose completion lengths sweep ~60-500 words."""
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        n_words = 60 + int(440 * i / (n - 1)) + rng.randint(-10, 10)
        recs.append({
            "id": f"star1_{i:05d}",
            "label": "safety",
            "prompt": f"reference prompt {i}",
            "reasoning": " ".join(f"w{j}" for j in range(max(20, n_words - 8))),
            "answer": "final answer sentence of eight words total here",
        })
    path.write_text(json.dumps(recs))
    return recs


@pytest.fixture
def built(tmp_path):
    ref_path = tmp_path / "star1_ref.json"
    out_path = tmp_path / "control_offpolicy.json"
    _write_synth_star1(ref_path)
    pt01c.main(["--mock", "--n", "50", "--seed", "1",
                "--star1", str(ref_path), "--out", str(out_path)])
    return pt01c.reference_lens(ref_path), json.loads(out_path.read_text())


def test_mock_output_schema_is_pt02_valid(built):
    _, recs = built
    assert len(recs) == 50
    ids = {r["id"] for r in recs}
    assert len(ids) == 50 and all(i.startswith("offpolicy_") for i in ids)
    for r in recs:
        for field in ("prompt", "reasoning", "answer", "category", "label", "source"):
            assert r.get(field), f"missing/empty {field}"
        assert r["label"] == "control_offpolicy"
    # pt02's ingestion path: answer present + SFT text builds cleanly
    sft = C.records_to_sft(recs, tokenizer=None)
    assert len(sft) == len(recs)
    for ex, r in zip(sft, recs):
        assert "</think>" in ex["completion_text"]
        assert r["answer"] in ex["completion_text"]
        assert "<think>" not in ex["completion_text"]


def test_mock_output_lengths_match_reference_quantiles(built):
    ref_lens, recs = built
    out_lens = sorted(pt01c.approx_tokens(r["reasoning"]) + pt01c.approx_tokens(r["answer"])
                      for r in recs)
    for q in (0.25, 0.50, 0.75, 0.90):
        ref_q = ref_lens[int(q * len(ref_lens))]
        out_q = out_lens[int(q * len(out_lens))]
        tol = max(25, 0.2 * ref_q)  # sentence-boundary truncation granularity
        assert abs(out_q - ref_q) <= tol, f"p{int(q*100)}: {out_q} vs ref {ref_q}"


def test_mock_answers_carry_final_answer_sentence(built):
    _, recs = built
    assert all(r["answer"].startswith("The answer is") for r in recs)


def test_split_body_answer_fallbacks():
    body, ans = pt01c.split_body_answer("We add 2 and 2. The answer is: 4.")
    assert body == "We add 2 and 2." and ans == "The answer is: 4."
    body, ans = pt01c.split_body_answer("First step here. Second step done.")
    assert body == "First step here." and ans == "Second step done."
    body, ans = pt01c.split_body_answer("Only one sentence")
    assert body == "Only one sentence" and ans  # neutral non-empty closing


# ── pt02c: full-parameter SFT on a tiny CPU model ────────────────────────────

def _tiny_qwen2(tmp_path):
    """Save a random tiny Qwen2 + real (word-level) fast tokenizer to disk."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    vocab = {f"tok{i}": i for i in range(61)}
    vocab.update({"<unk>": 61, "<pad>": 62, "<eos>": 63})
    tok = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")

    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=512)
    model = Qwen2ForCausalLM(cfg)
    d = tmp_path / "tiny_qwen2"
    model.save_pretrained(d)
    tokenizer.save_pretrained(str(d))
    return d, model


def test_train_full_two_steps_cpu_saves_loadable_checkpoint(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    from src.safety_posttrain import sft as S

    tiny_dir, init_model = _tiny_qwen2(tmp_path)
    records = C.mock_dataset(2)                      # 4 records
    sft_examples = C.records_to_sft(records, tokenizer=None)
    assert len(sft_examples) == 4

    out_dir = tmp_path / "fullft_ckpt"
    result = S.train_full(
        sft_examples, str(tiny_dir), out_dir,
        dtype="float32", epochs=1.0, lr=1e-2,
        batch_size=2, grad_accum=1, max_len=256,
        seed=0, max_steps=2, device_map="cpu",
    )
    assert result["steps"] == 2
    assert result["n_examples"] == 4
    assert result["train_loss"] == result["train_loss"]  # finite, not NaN
    if not torch.cuda.is_available():
        assert result["optim"] == "adafactor"        # the declared CPU fallback

    # loadable exactly like 04_extract_activations.py --model-path does
    assert (out_dir / "config.json").exists()
    reloaded = AutoModelForCausalLM.from_pretrained(str(out_dir))
    p0 = dict(init_model.named_parameters())
    trained = dict(reloaded.named_parameters())
    name = "model.layers.0.self_attn.q_proj.weight"
    assert not torch.equal(p0[name], trained[name]), "weights did not update"
    logits = reloaded(input_ids=torch.tensor([[1, 2, 3]])).logits
    assert torch.isfinite(logits).all()
