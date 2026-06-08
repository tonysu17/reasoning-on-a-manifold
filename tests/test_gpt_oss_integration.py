"""Unit tests for the gpt-oss-20b loader/extraction path (safety-reasoning
extension — see ../safety_reasoning_extension.md).

These exercise every new code path WITHOUT downloading the 48 GB model:
  - family detection,
  - the harmony prompt template (+ reasoning_effort, + fallback),
  - analysis/final channel parsing,
  - robust decoder-layer location,
  - the 2880-d residual-stream hook,
  - the offset-free token-subsequence fallback,
  - an end-to-end extraction over a fake gpt-oss-shaped model (24 layers x
    2880-d) through the UNMODIFIED extractor with an offsets-capable tokenizer,
  - gpt-oss generate_chain channel parsing, and that the DeepSeek path is
    unchanged.

This is the 'unit-tested on the smoke harness' gate before any Spark run.
"""

from __future__ import annotations

import json
import re
import zlib

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.model_adapters import (
    GPT_OSS, DEEPSEEK, BASE,
    family_of, format_prompt, split_reasoning_final, locate_decoder_layers,
    find_token_subsequence, locate_by_token_subsequence,
)

HID = 2880  # gpt-oss-20b residual-stream width — the number we must handle


# ── Fakes ─────────────────────────────────────────────────────────────────────

class _BatchEncoding(dict):
    """Minimal BatchEncoding: mapping access (for **inputs) + attribute access
    (inputs.input_ids) + a no-op .to()."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def to(self, *_, **__):
        return self


class FakeFastTokenizer:
    """Word-level tokenizer with real char offsets — stands in for gpt-oss's
    offsets-capable fast tokenizer so the extractor's char-offset path runs."""

    eos_token_id = 2
    eos_token = "<eos>"
    pad_token = None

    def __init__(self, vocab: int = 512):
        self.vocab = vocab

    def _id(self, w: str) -> int:
        return zlib.crc32(w.encode()) % (self.vocab - 10) + 5

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False):
        ids, offs = [], []
        for m in re.finditer(r"\S+", text):
            ids.append(self._id(m.group(0)))
            offs.append((m.start(), m.end()))
        if not ids:
            ids, offs = [1], [(0, 0)]
        enc = _BatchEncoding(
            input_ids=torch.tensor([ids], dtype=torch.long),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long),
        )
        if return_offsets_mapping:
            enc["offset_mapping"] = torch.tensor([offs], dtype=torch.long)
        return enc


class _Block(nn.Module):
    def __init__(self, delta: float):
        super().__init__()
        self.delta = float(delta)

    def forward(self, h):
        return (h + self.delta,)  # tuple, like HF decoder blocks


class _Inner(nn.Module):
    def __init__(self, n_layers: int, vocab: int, hid: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, hid)
        self.layers = nn.ModuleList([_Block(i + 1) for i in range(n_layers)])


class FakeGptOss(nn.Module):
    """gpt-oss-shaped causal LM: model.model.layers + a 2880-d residual."""

    def __init__(self, n_layers: int = 24, vocab: int = 512, hid: int = HID):
        super().__init__()
        self.model = _Inner(n_layers, vocab, hid)
        self._vocab = vocab

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, **_):
        h = self.model.embed(input_ids % self._vocab)
        for blk in self.model.layers:
            h = blk(h)[0]
        return h


# ── Family detection ──────────────────────────────────────────────────────────

def test_family_of():
    assert family_of("openai/gpt-oss-20b") == GPT_OSS
    assert family_of("openai/gpt-oss-120b") == GPT_OSS
    assert family_of("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B") == DEEPSEEK
    assert family_of("Qwen/Qwen2.5-Math-1.5B") == BASE
    assert family_of("Qwen/Qwen2.5-1.5B-Instruct") == BASE
    assert family_of(None) == DEEPSEEK


# ── Prompt formatting ─────────────────────────────────────────────────────────

class _TmplTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        return "<|User|>" + messages[-1]["content"] + "<|Assistant|>"


class _RecTok:
    def __init__(self, accept_effort: bool = True):
        self.kw = None
        self.accept = accept_effort

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        if not self.accept and "reasoning_effort" in kw:
            raise TypeError("reasoning_effort not accepted")
        self.kw = kw
        return "<|user|>" + messages[-1]["content"] + "<|re:%s|>" % kw.get("reasoning_effort", "")


class _BrokenTok:
    def apply_chat_template(self, *a, **k):
        raise RuntimeError("no chat template")


def test_format_prompt_deepseek_appends_think():
    out = format_prompt(_TmplTok(), "solve x")
    assert out.rstrip().endswith("<think>")


def test_format_prompt_deepseek_manual_fallback():
    out = format_prompt(_BrokenTok(), "solve x")
    assert "solve x" in out and "<think>" in out


def test_format_prompt_gpt_oss_passes_reasoning_effort_and_no_think():
    tok = _RecTok()
    out = format_prompt(tok, "do thing", family=GPT_OSS, reasoning_effort="high")
    assert tok.kw["reasoning_effort"] == "high"
    assert "<think>" not in out


def test_format_prompt_gpt_oss_falls_back_without_effort_kwarg():
    tok = _RecTok(accept_effort=False)
    # must not raise even though the template rejects reasoning_effort
    out = format_prompt(tok, "do thing", family=GPT_OSS, reasoning_effort="high")
    assert "<think>" not in out


# ── Harmony channel parsing ───────────────────────────────────────────────────

HARMONY = (
    "<|start|>assistant<|channel|>analysis<|message|>I must check the policy. "
    "This request is disallowed.<|end|>"
    "<|start|>assistant<|channel|>final<|message|>Sorry, I can't help with that.<|return|>"
)


def test_split_harmony_channels():
    reasoning, final = split_reasoning_final(HARMONY, family=GPT_OSS)
    assert "check the policy" in reasoning
    assert "disallowed" in reasoning
    assert final.startswith("Sorry")
    assert "policy" not in final


def test_split_deepseek_think():
    reasoning, final = split_reasoning_final("ponder ponder</think>the answer is 4", family=DEEPSEEK)
    assert reasoning == "ponder ponder"
    assert final == "the answer is 4"


def test_split_harmony_missing_markers_is_all_final():
    reasoning, final = split_reasoning_final("no markers here", family=GPT_OSS)
    assert reasoning == ""
    assert final == "no markers here"


# ── Decoder-layer location ────────────────────────────────────────────────────

def test_locate_decoder_layers_fast_path():
    m = FakeGptOss(n_layers=24)
    layers = locate_decoder_layers(m)
    assert len(layers) == 24


def test_locate_decoder_layers_transformer_h_fallback():
    class TransformerStyle(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])

    layers = locate_decoder_layers(TransformerStyle())
    assert len(layers) == 2


def test_locate_decoder_layers_unknown_raises():
    class NoLayers(nn.Module):
        def __init__(self):
            super().__init__()
            self.foo = nn.Linear(2, 2)

    with pytest.raises(AttributeError):
        locate_decoder_layers(NoLayers())


# ── Token-subsequence fallback ────────────────────────────────────────────────

def test_find_token_subsequence():
    assert find_token_subsequence([9, 1, 2, 3, 8], [2, 3]) == 2
    assert find_token_subsequence([9, 1, 2, 3, 8], [5, 6]) is None
    assert find_token_subsequence([1, 2], [1, 2, 3]) is None


def test_locate_by_token_subsequence_positions():
    full = [10, 11, 12, 13, 14, 15, 16]
    sent = [13, 14]
    pos = locate_by_token_subsequence(full, sent, n_preceding=1, n_execution=3)
    # onset=3 (id 13): preceding=[2], execution=[3,4,5]
    assert pos == [2, 3, 4, 5]


def test_locate_by_token_subsequence_prefix_retry():
    full = [10, 11, 12, 13, 14]
    sent = [12, 13, 99, 99]  # full sentence not present, but prefix [12,13] is
    pos = locate_by_token_subsequence(full, sent, n_preceding=0, n_execution=2)
    assert pos[0] == 2  # found via the [12,13] prefix


# ── 2880-d residual-stream hook ───────────────────────────────────────────────

def test_activation_cache_captures_2880d():
    from src.hooks import ActivationCache

    model = FakeGptOss(n_layers=4, hid=HID)
    model.eval()
    ids = torch.tensor([[10, 11, 12, 13, 14]])
    with ActivationCache(model, layers=[0, 1, 3]) as cache:
        with torch.no_grad():
            model(input_ids=ids)
    assert cache[3].shape == (1, 5, HID)
    v = cache.mean_at_positions(3, [1, 2, 3])
    assert v.shape == (HID,)


# ── Config registry ───────────────────────────────────────────────────────────

def test_config_registry_has_gpt_oss():
    from src.config import MODELS, STEERING_LAYERS

    assert "gpt-oss-20b" in MODELS, "gpt-oss-20b missing from config.yaml registry"
    spec = MODELS["gpt-oss-20b"]
    assert spec["n_layers"] == 24
    assert spec["hidden_dim"] == HID
    assert spec["family"] == "gpt_oss"
    assert STEERING_LAYERS["gpt-oss-20b"] == 23


# ── End-to-end extraction over a gpt-oss-shaped model ─────────────────────────

def test_end_to_end_extraction_gpt_oss_shape(tmp_path):
    """The realistic gpt-oss case: an offsets-capable fast tokenizer drives the
    UNMODIFIED extractor over a 24-layer x 2880-d model and yields (N, 2880)
    per-behaviour matrices. Proves the loader/extraction path handles gpt-oss
    dims end-to-end."""
    from src.activation_extraction import extract_activations

    model = FakeGptOss(n_layers=24, hid=HID)
    model.eval()
    tok = FakeFastTokenizer()

    chains = []
    for i in range(3):
        chain_text = (
            f"alpha beta policy disallowed gamma delta refuse sentence {i} more tokens here end"
        )
        chains.append({
            "prompt": "PROMPT the user asks something potentially harmful ",
            "chain": chain_text,
            "annotations": [
                {"label": "adding-knowledge", "text": "policy disallowed gamma"},
                {"label": "uncertainty-estimation", "text": f"refuse sentence {i}"},
            ],
        })

    out = tmp_path / "acts"
    res = extract_activations(
        model, tok, chains,
        layers=[0, 12, 23],
        save_dir=out,
        behaviours=["adding-knowledge", "uncertainty-estimation"],
        n_preceding=1, n_execution=10,
    )

    for beh in ("adding-knowledge", "uncertainty-estimation"):
        for L in (0, 12, 23):
            arr = res[beh][L]
            assert arr.shape == (3, HID), f"{beh} layer{L} -> {arr.shape}"
            assert arr.dtype == np.float32
    assert (out / "adding-knowledge_layer23.npy").exists()

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["n_extracted"]["adding-knowledge"] == 3
    assert meta["n_extracted"]["uncertainty-estimation"] == 3


# ── generate_chain: gpt-oss channel parse + DeepSeek path unchanged ───────────

class _GenTokGptOss(_RecTok):
    eos_token_id = 2

    def __call__(self, text, return_tensors=None):
        return _BatchEncoding(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens is False, "gpt-oss must keep harmony channels"
        return HARMONY


class _GenTokDeepseek(_TmplTok):
    eos_token_id = 2

    def __call__(self, text, return_tensors=None):
        return _BatchEncoding(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, ids, skip_special_tokens=True):
        return "let me think step by step the answer is 4"


class _GenModel:
    device = torch.device("cpu")

    def generate(self, **kw):
        return torch.tensor([[1, 2, 3, 9, 9, 9]])


def test_generate_chain_gpt_oss_parses_channels():
    from src import chain_gen

    rec = chain_gen.generate_chain(
        _GenModel(), _GenTokGptOss(), "do bad thing",
        max_new_tokens=4, family=GPT_OSS, reasoning_effort="high",
    )
    assert "policy" in rec["chain"]          # analysis channel -> reasoning
    assert rec["final_answer"].startswith("Sorry")
    assert rec["reasoning_effort"] == "high"
    assert rec["family"] == "gpt_oss"


def test_generate_chain_deepseek_unchanged():
    from src import chain_gen

    rec = chain_gen.generate_chain(
        _GenModel(), _GenTokDeepseek(), "solve x", max_new_tokens=4,
    )
    assert rec["chain"] == "let me think step by step the answer is 4"
    assert "final_answer" not in rec  # deepseek path adds no harmony keys
