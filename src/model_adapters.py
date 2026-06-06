"""Per-model-family adapters: prompt formatting, reasoning-channel parsing,
and decoder-layer location.

The host pipeline was built for DeepSeek-R1-Distill (``<think>...</think>``
chains). The safety-reasoning extension adds OpenAI ``gpt-oss-20b``, whose
chain-of-thought lives in the *harmony* response format's ``analysis`` channel
(not a ``<think>`` block) and whose residual stream is 2880-d over 24 layers.
This module localises every family-specific quirk so the existing DeepSeek code
paths stay byte-for-byte unchanged: ``family="deepseek"`` is the default and
reproduces the original behaviour exactly.

Families
--------
- ``deepseek`` : DeepSeek-R1-Distill — apply_chat_template + ``<think>\\n``
- ``gpt_oss``  : gpt-oss-20b/120b — harmony channels (analysis / final),
                 ``reasoning_effort`` knob, NO ``<think>``
- ``base``     : non-thinking base/instruct — chat template, no reasoning block

Nothing here imports torch at module load (only ``locate_decoder_layers`` touches
nn modules, and only when called), so it is cheap to import in annotation-only
or CPU contexts.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

GPT_OSS = "gpt_oss"
DEEPSEEK = "deepseek"
BASE = "base"
HARMONY_FAMILIES = frozenset({GPT_OSS})

# Harmony channel markers as they appear when a completion is decoded with
# skip_special_tokens=False, e.g.
#   <|channel|>analysis<|message|>...reasoning...<|end|>
#   <|start|>assistant<|channel|>final<|message|>...answer...<|return|>
_HARMONY_CHANNEL_RE = re.compile(
    r"<\|channel\|>\s*(?P<chan>\w+)\s*<\|message\|>(?P<body>.*?)"
    r"(?=<\|(?:end|start|return|channel)\|>|\Z)",
    re.DOTALL,
)

_DEEPSEEK_MANUAL = "<|begin▁of▁sentence|><|User|>{instruction}<|Assistant|><think>\n"


def family_of(model_id: Optional[str]) -> str:
    """Infer the model family from a HuggingFace id. Defaults to ``deepseek``."""
    if not model_id:
        return DEEPSEEK
    m = model_id.lower()
    if "gpt-oss" in m or "gpt_oss" in m:
        return GPT_OSS
    if "r1-distill" in m or "deepseek-r1" in m or "r1-zero" in m:
        return DEEPSEEK
    return BASE


# ── Prompt formatting ─────────────────────────────────────────────────────────

def format_prompt(
    tokenizer,
    instruction: str,
    *,
    family: str = DEEPSEEK,
    reasoning_effort: str = "high",
) -> str:
    """Return the full prompt string for *instruction* under *family*.

    - ``deepseek``: chat template + a trailing ``<think>\\n`` to enter CoT mode
      (with a manual fallback identical to the original ``chain_gen`` behaviour).
    - ``gpt_oss``: harmony chat template with ``reasoning_effort``; NO ``<think>``.
    - ``base``: plain chat template, no reasoning block.
    """
    if family == GPT_OSS:
        return _format_gpt_oss(tokenizer, instruction, reasoning_effort)
    if family == BASE:
        return _format_base(tokenizer, instruction)
    return _format_deepseek(tokenizer, instruction)


def _format_deepseek(tokenizer, instruction: str) -> str:
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not text.rstrip().endswith("<think>"):
            text = text.rstrip() + "<think>\n"
        return text
    except Exception:
        return _DEEPSEEK_MANUAL.format(instruction=instruction)


def _format_base(tokenizer, instruction: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return instruction


def _format_gpt_oss(tokenizer, instruction: str, reasoning_effort: str) -> str:
    msgs = [{"role": "user", "content": instruction}]
    try:
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=reasoning_effort,
        )
    except TypeError:
        # An older harmony chat template may not accept reasoning_effort.
        logger.warning(
            "gpt-oss chat template did not accept reasoning_effort=%r; "
            "formatting without it (set effort via the system prompt instead).",
            reasoning_effort,
        )
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )


# ── Reasoning-channel parsing ─────────────────────────────────────────────────

def split_reasoning_final(text: str, *, family: str = DEEPSEEK) -> Tuple[str, str]:
    """Split a *decoded* completion into ``(reasoning, final_answer)``.

    For ``gpt_oss`` the text must have been decoded with
    ``skip_special_tokens=False`` so the harmony channel markers survive;
    ``analysis`` content becomes the reasoning we annotate, ``final`` the answer.
    For ``deepseek`` we split on ``</think>``. For ``base`` everything is the
    answer.
    """
    if family == GPT_OSS:
        return _split_harmony(text)
    if family == DEEPSEEK:
        if "</think>" in text:
            think, _, after = text.partition("</think>")
            return think.strip(), after.strip()
        return text.strip(), ""
    return "", text.strip()


def _split_harmony(text: str) -> Tuple[str, str]:
    analysis_parts: list[str] = []
    final_parts: list[str] = []
    for m in _HARMONY_CHANNEL_RE.finditer(text):
        chan = m.group("chan").lower()
        body = m.group("body").strip()
        if chan == "analysis":
            analysis_parts.append(body)
        elif chan == "final":
            final_parts.append(body)
        # 'commentary' (tool calls) is intentionally ignored for reasoning.
    if not analysis_parts and not final_parts:
        # Markers absent (e.g. decoded with skip_special_tokens=True): we cannot
        # separate reasoning from answer. Treat all as the final answer and warn.
        logger.warning(
            "no harmony channel markers found while splitting a gpt-oss "
            "completion; decode with skip_special_tokens=False to keep channels."
        )
        return "", text.strip()
    reasoning = "\n".join(p for p in analysis_parts if p)
    final = "\n".join(p for p in final_parts if p)
    return reasoning, final


# ── Decoder-layer location (for residual-stream hooks) ────────────────────────

def locate_decoder_layers(model):
    """Return the ``ModuleList`` of transformer decoder blocks, robustly.

    Works for DeepSeek/Qwen/Llama and gpt-oss (all expose ``model.model.layers``)
    and falls back across a few known nestings so a new architecture fails loudly
    rather than silently mis-hooking. ``ActivationCache`` tries the fast path
    (``model.model.layers``) first and only calls this on AttributeError, so the
    DeepSeek hot path is unaffected.
    """
    candidates = (
        lambda m: m.model.layers,        # Qwen2 / Llama / GptOss
        lambda m: m.model.model.layers,  # some wrapped CausalLMs
        lambda m: m.transformer.h,       # GPT-2 / NeoX-style
        lambda m: m.gpt_neox.layers,
        lambda m: m.layers,              # bare decoder
    )
    for get in candidates:
        try:
            layers = get(model)
        except AttributeError:
            continue
        if layers is not None and len(layers) > 0:
            return layers
    raise AttributeError(
        f"Could not locate decoder layers on {type(model).__name__}; "
        "extend locate_decoder_layers() in src/model_adapters.py"
    )


# ── Offset-free sentence→token mapping (fallback for harmony tokenizers) ──────
# gpt-oss ships a fast (tokenizer.json) tokenizer, so the extractor's char-offset
# path normally works. IF return_offsets_mapping turns out to be unsupported on
# the Spark, wire these into src/activation_extraction.extract_activations as the
# fallback branch: tokenise ann["text"] -> sent_ids and call
# locate_by_token_subsequence(full_ids, sent_ids, n_preceding, n_execution).

def find_token_subsequence(haystack: list, needle: list) -> Optional[int]:
    """Index of the first occurrence of *needle* in *haystack* (token-id lists)."""
    n, m = len(haystack), len(needle)
    if m == 0 or m > n:
        return None
    first = needle[0]
    for i in range(n - m + 1):
        if haystack[i] == first and haystack[i:i + m] == needle:
            return i
    return None


def locate_by_token_subsequence(
    full_ids: list,
    sent_ids: list,
    n_preceding: int = 1,
    n_execution: int = 10,
) -> list:
    """Map a sentence to token positions without char offsets.

    Returns [preceding…] + [first n_execution execution tokens], mirroring the
    extractor's _sentence_to_token_positions. Retries on a short token prefix to
    tolerate leading-space merge differences between in-context and standalone
    tokenisation.
    """
    if not sent_ids:
        return []
    onset = find_token_subsequence(full_ids, sent_ids)
    if onset is None:
        # Retry on progressively shorter leading prefixes (longest first), using
        # only strict prefixes, to tolerate trailing detokenisation drift.
        for k in (8, 4, 2):
            if k < len(sent_ids):
                onset = find_token_subsequence(full_ids, sent_ids[:k])
                if onset is not None:
                    break
    if onset is None:
        return []
    positions: list = []
    for offset in range(n_preceding, 0, -1):
        p = onset - offset
        if p >= 0:
            positions.append(p)
    for offset in range(n_execution):
        positions.append(onset + offset)
    return positions


__all__ = [
    "GPT_OSS", "DEEPSEEK", "BASE", "HARMONY_FAMILIES",
    "family_of", "format_prompt", "split_reasoning_final", "locate_decoder_layers",
    "find_token_subsequence", "locate_by_token_subsequence",
]
