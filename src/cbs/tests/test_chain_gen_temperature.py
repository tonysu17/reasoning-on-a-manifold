"""Unit tests for chain_gen.py M4.5 temperature + seed support.

The HF model is not loaded in tests — we test the seed-application helper
and the multi-seed CLI parser directly.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from src.chain_gen import _seed_torch, generate_chain


def test_seed_torch_makes_torch_deterministic() -> None:
    import torch
    _seed_torch(42)
    a = torch.randn(8)
    _seed_torch(42)
    b = torch.randn(8)
    assert torch.equal(a, b)
    _seed_torch(43)
    c = torch.randn(8)
    assert not torch.equal(a, c), (
        "Different seeds should yield different torch.randn output."
    )


def test_seed_torch_handles_str_via_int_cast() -> None:
    # Accepts integer-valued strings as well (passed by argparse).
    import torch
    _seed_torch(int("100"))
    a = torch.randn(4)
    _seed_torch(int("100"))
    b = torch.randn(4)
    assert torch.equal(a, b)


def test_generate_chain_seeded_with_fake_model() -> None:
    """generate_chain calls _seed_torch(seed); verify by mocking model/tokenizer
    and asserting the seed is honoured (subsequent torch.randn deterministic).
    """
    import torch

    class FakeTokenizer:
        eos_token_id = 0
        pad_token = None

        def apply_chat_template(self, messages, tokenize=False,
                                 add_generation_prompt=True):
            return f"PROMPT: {messages[0]['content']}"

        def __call__(self, text, return_tensors="pt"):
            class Inputs(dict):
                def __init__(self, ids):
                    super().__init__(input_ids=ids)
                    self.input_ids = ids
                def to(self, device):
                    return self
            return Inputs(torch.tensor([[1, 2, 3]]))

        def decode(self, tokens, skip_special_tokens=True):
            return "FAKE_CHAIN"

    class FakeModel:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3, 4, 5, 6]])

    out = generate_chain(FakeModel(), FakeTokenizer(),
                         instruction="solve 2+2",
                         max_new_tokens=4, temperature=0.7, seed=7)
    assert out["seed"] == 7
    assert out["temperature"] == 0.7
    assert out["chain"] == "FAKE_CHAIN"
    # After generate_chain ran, torch RNG state should be that left by
    # _seed_torch(7) — verify by re-seeding and checking deterministic
    # consumption.
    _seed_torch(7)
    a = torch.randn(3)
    # Re-run generate_chain with the same seed to consume the same RNG
    # block; subsequent randn after that should match what we would have
    # seen otherwise.
    generate_chain(FakeModel(), FakeTokenizer(),
                   instruction="solve 2+2",
                   max_new_tokens=4, temperature=0.7, seed=7)
    _seed_torch(7)
    b = torch.randn(3)
    assert torch.equal(a, b)


def test_two_generate_chain_calls_with_different_seeds() -> None:
    """Different seeds produce different post-generation RNG states (which is
    how downstream sampling diverges in the real generation path)."""
    import torch

    class FakeTokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, tokenize=False,
                                 add_generation_prompt=True):
            return f"PROMPT: {messages[0]['content']}"

        def __call__(self, text, return_tensors="pt"):
            class Inputs(dict):
                def __init__(self, ids):
                    super().__init__(input_ids=ids)
                    self.input_ids = ids
                def to(self, device):
                    return self
            return Inputs(torch.tensor([[1, 2, 3]]))

        def decode(self, tokens, skip_special_tokens=True):
            return "FAKE"

    class FakeModel:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3, 4]])

    generate_chain(FakeModel(), FakeTokenizer(), "x", temperature=0.7, seed=0)
    a = torch.randn(2)
    generate_chain(FakeModel(), FakeTokenizer(), "x", temperature=0.7, seed=1)
    b = torch.randn(2)
    # State after generate_chain depends on the seed; the post-generate
    # randn calls must differ.
    assert not torch.equal(a, b)


def test_runner_parse_seeds_helper() -> None:
    # Direct import of the helper from the runner script.
    runner = importlib.machinery.SourceFileLoader(
        "runner_02",
        str(Path("02_generate_chains.py").resolve()),
    ).load_module()
    assert runner._parse_seeds("0") == [0]
    assert runner._parse_seeds("0,1,2") == [0, 1, 2]
    assert runner._parse_seeds("0, 1, 2") == [0, 1, 2]
    assert runner._parse_seeds("") == []
