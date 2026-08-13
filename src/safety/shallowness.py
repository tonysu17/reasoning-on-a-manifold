"""Early-token safety shallowness (Qi et al. 2024, "Safety Alignment Should Be
Made More Than Just a Few Tokens Deep").

Qi's finding: classical safety alignment shifts the generative distribution almost
entirely over the FIRST FEW output tokens (KL concentrated in ~5 tokens). For the
recipe contrast this is a discriminator: an RLHF/shallow safety object should show
a high early-token KL fraction (a first-token reflex), whereas deliberative
alignment — which reasons over the policy across the CoT — should spread safety
signal deeper.

The pure functions here operate on per-position next-token log-probabilities so
they are testable offline; ``collect_per_token_logprobs`` (runner-side) teacher-
forces a continuation through a model to produce them.
"""

from __future__ import annotations

import numpy as np


def per_token_kl(logp_a: np.ndarray, logp_b: np.ndarray) -> np.ndarray:
    """KL(p_a ‖ p_b) at each position. Inputs are (T, V) LOG-probabilities
    (rows over the vocab). Returns (T,) non-negative KL per token."""
    logp_a = np.asarray(logp_a, dtype=float)
    logp_b = np.asarray(logp_b, dtype=float)
    if logp_a.shape != logp_b.shape:
        raise ValueError(f"shape mismatch: {logp_a.shape} vs {logp_b.shape}")
    p_a = np.exp(logp_a)
    kl = np.sum(p_a * (logp_a - logp_b), axis=-1)
    return np.maximum(kl, 0.0)


def shallowness_ratio(kl_per_token: np.ndarray, k: int = 5) -> float:
    """Fraction of total KL contained in the first *k* tokens. ~1 ⇒ shallow,
    first-token reflex (Qi); ~k/T ⇒ uniformly deep. 0.0 if total KL is 0."""
    kl = np.asarray(kl_per_token, dtype=float)
    total = float(kl.sum())
    if total <= 0:
        return 0.0
    return float(kl[:k].sum() / total)


def cumulative_kl_fraction(kl_per_token: np.ndarray) -> np.ndarray:
    """Cumulative fraction of KL vs token position (for the depth curve).
    Monotone non-decreasing, ending at 1.0 (or all-zeros if total KL is 0)."""
    kl = np.asarray(kl_per_token, dtype=float)
    total = float(kl.sum())
    if total <= 0:
        return np.zeros_like(kl)
    return np.cumsum(kl) / total


def depth_50_token(kl_per_token: np.ndarray) -> int:
    """Index of the token at which cumulative KL first reaches 50% (the 'safety
    depth'). Low ⇒ shallow. Returns 0 for all-zero input."""
    frac = cumulative_kl_fraction(kl_per_token)
    if frac.size == 0 or frac[-1] == 0:
        return 0
    return int(np.searchsorted(frac, 0.5))


def collect_per_token_logprobs(model, tokenizer, prompt: str, continuation: str):
    """Runner-side: teacher-force *continuation* after *prompt* and return the
    model's per-position next-token log-probabilities over the continuation span,
    shape (T_continuation, vocab). Compare two models' arrays with ``per_token_kl``.
    """
    import torch
    import torch.nn.functional as F

    full = prompt + continuation
    enc_full = tokenizer(full, return_tensors="pt").to(model.device)
    enc_prompt = tokenizer(prompt, return_tensors="pt")
    p_len = enc_prompt["input_ids"].shape[1]
    with torch.no_grad():
        logits = model(**enc_full).logits[0]  # (seq, vocab)
    # next-token distribution that PREDICTS each continuation token sits at the
    # position just before it: positions [p_len-1 : seq-1].
    span = logits[p_len - 1: -1]
    logprobs = F.log_softmax(span.float(), dim=-1)
    return logprobs.cpu().numpy()


__all__ = [
    "per_token_kl", "shallowness_ratio", "cumulative_kl_fraction",
    "depth_50_token", "collect_per_token_logprobs",
]
