"""
E9.0 loop-geometry library (COLLAPSE_AND_ENTROPY.md §5).

Pure numpy — no torch at import time, so the detect/analyse stages and the
test-suite run on CPU-only machines. Three ingredient families:

1. **Loop detection** (text level): find the periodic verbatim tail that the
   collapsed chains exhibit (two-paraphrase oscillation repeated to the token
   cap). Word-level periodicity with a mismatch tolerance, NOT a bare n-gram
   count, so the onset (where the chain *entered* the loop) is recoverable —
   the quantity E9.0b's precedence test needs.
2. **Windowed state geometry**: participation ratio (spectral effective
   dimension) and mean pairwise cosine (token uniformity, Dong et al. 2021)
   over sliding windows of residual-stream states.
3. **Probe/selection helpers**: char→token selection for in-loop /
   out-of-loop classes with a guard band, and the analytic cosine null for
   comparing the learned loop direction against steering vectors
   (random unit vectors in R^d have cos ~ N(0, 1/d)).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np

WORD_RE = re.compile(r"\S+")

#: Defaults chosen against the E8 failure mode: observed loops are 30–80-word
#: two-paraphrase cycles sustained for thousands of words; 150 words / 3 cycles
#: is far below any real tail but above chance periodicity in prose.
DEFAULT_MAX_PERIOD = 400
DEFAULT_TOL = 0.10
DEFAULT_MIN_TAIL_WORDS = 150
DEFAULT_MIN_CYCLES = 3.0


# ── Loop detection ────────────────────────────────────────────────────────────

def word_spans(text: str) -> list[tuple[int, int]]:
    """Char (start, end) span of every whitespace-delimited word."""
    return [m.span() for m in WORD_RE.finditer(text)]


@dataclass
class LoopInfo:
    has_loop: bool
    onset_word: int = -1          # index into the word sequence
    period: int = 0               # in words
    tail_words: int = 0           # length of the periodic tail
    mismatch: float = 1.0         # mismatch rate inside the accepted tail

    def to_dict(self) -> dict:
        return asdict(self)


def detect_loop_tail(
    words: Sequence[str],
    max_period: int = DEFAULT_MAX_PERIOD,
    tol: float = DEFAULT_TOL,
    min_tail_words: int = DEFAULT_MIN_TAIL_WORDS,
    min_cycles: float = DEFAULT_MIN_CYCLES,
) -> LoopInfo:
    """
    Detect a periodic verbatim tail: the longest suffix in which each word
    matches the word one period earlier, with overall mismatch rate <= tol.

    For every candidate period p we take eq[i] = (w[i+p] == w[i]) and find the
    longest suffix of eq whose cumulative mismatch rate stays <= tol, with the
    final 2p comparisons required to be near-exact (the chain must actually
    END inside the loop — this is a *tail* detector by design; E8 collapse is
    loop-to-cap). Smallest period wins ties, so multiples of the true period
    do not displace it. The winning onset is then REFINED forward: an overall
    rate <= tol lets the tail bleed ~tol*L words back into clean prose, so we
    trim the suffix until it opens with a locally clean run (planted-tail
    tests pin this bias).
    """
    n = len(words)
    if n < min_tail_words:
        return LoopInfo(False)

    vocab: dict[str, int] = {}
    ids = np.fromiter((vocab.setdefault(w, len(vocab)) for w in words),
                      dtype=np.int64, count=n)

    best = LoopInfo(False)
    best_eq: Optional[np.ndarray] = None
    p_hi = min(max_period, n // 2)
    for p in range(1, p_hi + 1):
        cand, eq = _tail_for_period(ids, p, tol, min_tail_words, min_cycles)
        if cand and cand.tail_words > best.tail_words:   # ascending p ⇒ ties → smallest
            best, best_eq = cand, eq

    if best.has_loop:
        # canonicalize: realized noise can hand a multiple of the true period a
        # marginally longer tail — the smallest divisor with a comparable tail
        # is the true period (planted-noisy-tail test pins this).
        for d in sorted(_proper_divisors(best.period)):
            cand, eq = _tail_for_period(ids, d, tol, min_tail_words, min_cycles)
            if cand and cand.tail_words >= best.tail_words - 2 * best.period:
                best, best_eq = cand, eq
                break
        best = _refine_onset(best, best_eq, n, tol)
    return best


def _tail_for_period(
    ids: np.ndarray, p: int, tol: float, min_tail_words: int, min_cycles: float,
) -> tuple[Optional[LoopInfo], Optional[np.ndarray]]:
    """Longest tolerant, end-anchored periodic tail at a single period p."""
    n = ids.size
    eq = ids[p:] == ids[:-p]
    m = eq.size
    if m == 0:
        return None, None
    mis_rev = np.cumsum((~eq)[::-1])
    k = np.arange(1, m + 1)
    hits = np.nonzero(mis_rev <= tol * k)[0]
    if hits.size == 0:
        return None, None
    L = int(hits[-1]) + 1                         # longest tolerant suffix of eq
    tail = L + p                                  # words covered by the loop
    if tail < max(min_tail_words, min_cycles * p):
        return None, None
    # End anchor: the loop must reach the END of the chain. Its only job is to
    # reject mid-chain repetition that resolved (mismatch ≈ 1 there), so it is
    # deliberately looser (2*tol) than the tail criterion — at exactly tol, an
    # unlucky noise draw in a 2p-window rejects the TRUE period and hands the
    # detection to one of its multiples.
    anchor = eq[-2 * p:] if m >= 2 * p else eq
    if (~anchor).mean() > 2 * tol:
        return None, None
    return LoopInfo(True, onset_word=n - tail, period=p, tail_words=tail,
                    mismatch=float(mis_rev[L - 1] / L)), eq


def _proper_divisors(p: int) -> list[int]:
    return [d for d in range(1, p) if p % d == 0]


def _refine_onset(info: LoopInfo, eq: np.ndarray, n: int, tol: float) -> LoopInfo:
    """Trim the accepted suffix forward until it opens with a locally clean run.

    The overall-rate criterion admits ~tol*L junk comparisons, which all pile
    up at the FRONT of the suffix (the end is anchored) — biasing the onset
    early. Advance the start to the first position from which a window of
    max(p, 20) comparisons is near-exact.
    """
    p = info.period
    L = info.tail_words - p                      # comparisons in the suffix
    suffix = eq[eq.size - L:]
    w = int(min(max(p, 20), L))
    if w >= 2:
        clean = np.convolve((~suffix).astype(np.float64),
                            np.ones(w) / w, mode="valid")   # mismatch rate per window
        opens = np.nonzero((clean <= tol) & suffix[: clean.size])[0]
        j = int(opens[0]) if opens.size else 0
    else:
        j = 0
    if j > 0:
        L2 = L - j
        info = LoopInfo(True, onset_word=n - (L2 + p), period=p,
                        tail_words=L2 + p,
                        mismatch=float((~suffix[j:]).mean()))
    return info


def loop_labels_for_chain(
    chain_text: str,
    clean_rep_max: float = 0.30,
    **detect_kwargs,
) -> dict:
    """
    Full per-chain labelling: loop tail + 4-gram repetition + class.

    class: 'loop' (periodic tail found) / 'clean' (no tail AND low 4-gram
    repetition) / 'ambiguous' (no tail but repetitive — excluded from the
    probe so label noise cannot straddle the classes).
    """
    from src.evaluation import repetition_rate  # local import: keeps numpy-only path

    spans = word_spans(chain_text)
    words = [chain_text[s:e] for s, e in spans]
    info = detect_loop_tail(words, **detect_kwargs)
    rep4 = repetition_rate(chain_text)
    if info.has_loop:
        cls = "loop"
        onset_char = spans[info.onset_word][0]
    else:
        cls = "clean" if rep4 <= clean_rep_max else "ambiguous"
        onset_char = -1
    return {
        "class": cls,
        "rep4": rep4,
        "onset_char": onset_char,          # char offset into chain_text
        "n_words": len(words),
        **info.to_dict(),
    }


# ── Windowed state geometry ───────────────────────────────────────────────────

def participation_ratio(X: np.ndarray) -> float:
    """PR = (Σλ)² / Σλ² of the covariance spectrum — spectral effective dim.

    Computed from the smaller of the two Gram matrices without an SVD. The
    covariance and the centred Gram $G = X_cX_c^\\top$ share nonzero
    eigenvalues, and PR is scale-invariant, so
    ``PR = tr(G)² / ||G||_F²`` is *exactly* the SVD value (λ_i = s_i²) at a
    fraction of the cost: for a 128×1536 window the Gram is 128×128, turning
    ~380 per-chain SVDs into cheap matmuls (the CPU bottleneck that idled the
    GPU during extraction).
    """
    n = X.shape[0]
    if n < 3:
        return float("nan")
    Xc = (X - X.mean(axis=0, keepdims=True)).astype(np.float64)
    G = Xc @ Xc.T if n <= Xc.shape[1] else Xc.T @ Xc     # nonzero-spectrum equal
    tr = np.trace(G)
    denom = float((G * G).sum())                         # Σλ² = ||G||_F²
    if tr <= 0 or denom <= 0:
        return float("nan")
    return float(tr * tr / denom)


def mean_pairwise_cosine(X: np.ndarray) -> float:
    """Mean off-diagonal cosine similarity — token uniformity (≈1 = collapsed)."""
    n = X.shape[0]
    if n < 2:
        return float("nan")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = (X / norms).astype(np.float64)
    G = Xn @ Xn.T
    return float((G.sum() - np.trace(G)) / (n * (n - 1)))


def windowed_state_metrics(
    X: np.ndarray,
    window: int = 128,
    stride: int = 64,
) -> dict:
    """PR + uniformity over sliding windows of the (T, d) state sequence.

    Returns centers (row index of each window centre), pr, unif — each (W,).
    """
    T = X.shape[0]
    centers, prs, unifs = [], [], []
    for start in range(0, max(T - window + 1, 0), stride):
        W = X[start:start + window]
        centers.append(start + window // 2)
        prs.append(participation_ratio(W))
        unifs.append(mean_pairwise_cosine(W))
    return {
        "centers": np.asarray(centers, dtype=np.int64),
        "pr": np.asarray(prs, dtype=np.float64),
        "unif": np.asarray(unifs, dtype=np.float64),
    }


# ── Probe/selection helpers ───────────────────────────────────────────────────

def select_class_token_indices(
    offsets: Sequence[tuple[int, int]],
    gen_start_char: int,
    onset_char_abs: int,
    guard_char_abs: int,
    per_class: int,
    rng: np.random.Generator,
) -> dict:
    """
    Token indices (into the FULL sequence) for the two probe classes.

    in-loop  = generated tokens starting at/after the loop onset;
    out-loop = generated tokens strictly before the guard boundary (the guard
               band [guard, onset) is excluded — near-boundary labels are
               noisy in both directions).
    For chains without a loop pass onset_char_abs < 0: everything generated is
    out-loop. Zero-width offsets (specials) are excluded. Each class is
    subsampled to at most per_class without replacement.
    """
    starts = np.asarray([s for s, _ in offsets], dtype=np.int64)
    ends = np.asarray([e for _, e in offsets], dtype=np.int64)
    gen = (starts >= gen_start_char) & (ends > starts)
    if onset_char_abs >= 0:
        in_mask = gen & (starts >= onset_char_abs)
        out_mask = gen & (starts < guard_char_abs)
    else:
        in_mask = np.zeros_like(gen)
        out_mask = gen

    def _sample(mask: np.ndarray) -> np.ndarray:
        idx = np.nonzero(mask)[0]
        if idx.size > per_class:
            idx = rng.choice(idx, size=per_class, replace=False)
        return np.sort(idx)

    return {"in": _sample(in_mask), "out": _sample(out_mask)}


def cosine_null_sigma(d: int) -> float:
    """Std of the cosine between two independent random unit vectors in R^d."""
    return 1.0 / np.sqrt(d)


def cosine_report(v: np.ndarray, w: np.ndarray) -> dict:
    """Signed cosine + |cos| z-score against the random-direction null."""
    v = v / np.linalg.norm(v)
    w = w / np.linalg.norm(w)
    c = float(v @ w)
    sigma = cosine_null_sigma(v.shape[0])
    return {"cos": c, "abs_cos": abs(c), "z_vs_random": abs(c) / sigma}
