"""
Tests for E9.0 loop geometry (src/loop_geometry.py + 18_loop_geometry.py).

Synthetic-first discipline (METHODOLOGY house rule): every estimator is
validated on planted structure before touching real data — the detector must
recover a planted periodic tail's onset/period, PR must fall under planted
rank contraction, and the probe must recover a planted separating direction
under chain-grouped CV.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.loop_geometry import (
    cosine_null_sigma,
    cosine_report,
    detect_loop_tail,
    loop_labels_for_chain,
    mean_pairwise_cosine,
    participation_ratio,
    select_class_token_indices,
    windowed_state_metrics,
    word_spans,
)

RNG = np.random.default_rng(0)


def _runner():
    """Import 18_loop_geometry.py (digit-leading name needs spec loading)."""
    path = Path(__file__).resolve().parents[1] / "18_loop_geometry.py"
    spec = importlib.util.spec_from_file_location("loop_geometry_runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rand_words(n: int, vocab: int = 500, rng=RNG) -> list[str]:
    return [f"w{v}" for v in rng.integers(0, vocab, size=n)]


# ── detector ──────────────────────────────────────────────────────────────────

class TestDetectLoopTail:
    def test_no_loop_in_random_prose(self):
        info = detect_loop_tail(_rand_words(1500))
        assert not info.has_loop

    def test_planted_periodic_tail(self):
        rng = np.random.default_rng(1)
        base = _rand_words(800, rng=rng)
        cycle = _rand_words(40, vocab=400, rng=rng)
        words = base + cycle * 12                      # 480-word tail, period 40
        info = detect_loop_tail(words)
        assert info.has_loop
        assert info.period == 40
        assert abs(info.onset_word - 800) <= 45        # within one period

    def test_noisy_tail_still_detected(self):
        rng = np.random.default_rng(2)
        base = _rand_words(600, rng=rng)
        cycle = _rand_words(30, rng=rng)
        tail = cycle * 15
        corrupt = rng.choice(len(tail), size=int(0.04 * len(tail)), replace=False)
        for i in corrupt:
            tail[i] = "XNOISEX"
        info = detect_loop_tail(base + tail)
        assert info.has_loop
        assert info.period == 30
        assert abs(info.onset_word - 600) <= 60

    def test_short_repeat_not_flagged(self):
        words = _rand_words(900) + _rand_words(35, vocab=50) * 2   # 70-word tail
        info = detect_loop_tail(words)
        assert not info.has_loop

    def test_period_one(self):
        words = _rand_words(400) + ["again"] * 300
        info = detect_loop_tail(words)
        assert info.has_loop
        assert info.period == 1
        assert abs(info.onset_word - 400) <= 5

    def test_loop_must_reach_the_end(self):
        """Mid-chain repetition that RESOLVES is not a loop-to-cap tail."""
        rng = np.random.default_rng(3)
        cycle = _rand_words(30, rng=rng)
        words = _rand_words(300, rng=rng) + cycle * 8 + _rand_words(600, rng=rng)
        info = detect_loop_tail(words)
        assert not info.has_loop


class TestLoopLabels:
    def test_labels_and_onset_char(self):
        rng = np.random.default_rng(4)
        base = _rand_words(500, rng=rng)
        cycle = _rand_words(25, rng=rng)
        text = " ".join(base + cycle * 12)
        lab = loop_labels_for_chain(text)
        assert lab["class"] == "loop"
        spans = word_spans(text)
        assert lab["onset_char"] == spans[lab["onset_word"]][0]

    def test_clean_class(self):
        lab = loop_labels_for_chain(" ".join(_rand_words(1200)))
        assert lab["class"] == "clean"
        assert lab["onset_char"] == -1


# ── state geometry ────────────────────────────────────────────────────────────

class TestStateMetrics:
    def test_pr_isotropic_vs_lowrank(self):
        rng = np.random.default_rng(5)
        iso = rng.standard_normal((400, 30))
        low = rng.standard_normal((400, 3)) @ rng.standard_normal((3, 30))
        assert participation_ratio(iso) > 20
        assert participation_ratio(low) < 5

    def test_pr_guard(self):
        assert np.isnan(participation_ratio(np.zeros((2, 8))))

    def test_pr_matches_svd_reference(self):
        """Gram-trick PR is exactly the SVD spectrum value (n<d and n>d)."""
        def _svd_pr(X):
            Xc = (X - X.mean(0)).astype(np.float64)
            lam = np.linalg.svd(Xc, compute_uv=False) ** 2
            return float(lam.sum() ** 2 / (lam ** 2).sum())
        rng = np.random.default_rng(21)
        for shape in [(128, 1536), (400, 30), (50, 50)]:
            X = rng.standard_normal(shape)
            assert participation_ratio(X) == pytest.approx(_svd_pr(X), rel=1e-9)

    def test_uniformity(self):
        rng = np.random.default_rng(6)
        same = np.tile(rng.standard_normal(16), (50, 1))
        rand = rng.standard_normal((200, 16))
        assert mean_pairwise_cosine(same) == pytest.approx(1.0, abs=1e-6)
        assert abs(mean_pairwise_cosine(rand)) < 0.1

    def test_windowed_rank_contraction(self):
        rng = np.random.default_rng(7)
        full = rng.standard_normal((300, 30))
        collapsed = rng.standard_normal((300, 2)) @ rng.standard_normal((2, 30))
        X = np.vstack([full, collapsed])
        m = windowed_state_metrics(X, window=100, stride=50)
        first = m["pr"][m["centers"] < 250]
        last = m["pr"][m["centers"] > 350]
        assert last.mean() < first.mean() * 0.5
        assert len(m["centers"]) == len(m["pr"]) == len(m["unif"])


# ── selection + cosine null ───────────────────────────────────────────────────

class TestSelection:
    def test_classes_respect_boundaries(self):
        offsets = [(i * 5, i * 5 + 4) for i in range(200)]   # 200 tokens, 5 chars each
        rng = np.random.default_rng(8)
        sel = select_class_token_indices(offsets, gen_start_char=100,
                                         onset_char_abs=600, guard_char_abs=500,
                                         per_class=50, rng=rng)
        starts = np.array([o[0] for o in offsets])
        assert (starts[sel["in"]] >= 600).all()
        assert (starts[sel["out"]] >= 100).all()
        assert (starts[sel["out"]] < 500).all()          # guard band excluded
        assert len(sel["in"]) <= 50 and len(sel["out"]) <= 50

    def test_no_loop_chain_all_out(self):
        offsets = [(i * 5, i * 5 + 4) for i in range(100)]
        sel = select_class_token_indices(offsets, 100, -1, -1, 30,
                                         np.random.default_rng(9))
        assert len(sel["in"]) == 0 and len(sel["out"]) == 30

    def test_specials_excluded(self):
        offsets = [(0, 0)] * 5 + [(i * 5, i * 5 + 4) for i in range(2, 50)]
        sel = select_class_token_indices(offsets, 0, -1, -1, 100,
                                         np.random.default_rng(10))
        assert 0 not in sel["out"][:5].tolist() or (np.array(sel["out"]) >= 5).all()


class TestCosine:
    def test_null_sigma_and_report(self):
        d = 1536
        assert cosine_null_sigma(d) == pytest.approx(1 / np.sqrt(d))
        v = np.zeros(d); v[0] = 2.0
        w = np.zeros(d); w[1] = 3.0
        r = cosine_report(v, w)
        assert r["cos"] == pytest.approx(0.0, abs=1e-9)
        r2 = cosine_report(v, v * 0.5)
        assert r2["cos"] == pytest.approx(1.0)
        assert r2["z_vs_random"] == pytest.approx(np.sqrt(d))


# ── runner analysis paths on synthetic shards ─────────────────────────────────

def _synthetic_shards(n_loop=20, n_clean=20, d=32, per_class=40, seed=11):
    """Loop chains: in-loop tokens shifted +3 along a planted direction, and a
    PR series that contracts before the (known) onset. Clean chains: flat."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d); v /= np.linalg.norm(v)
    shards, L = [], 16
    for i in range(n_loop + n_clean):
        is_loop = i < n_loop
        n_gen = 4000
        onset = rng.integers(1500, 2500) if is_loop else -1
        centers = np.arange(64, n_gen - 64, 64)
        base_pr = 20 + rng.standard_normal(len(centers))
        if is_loop:
            drop = np.clip((centers - (onset - 600)) / 600.0, 0, 1) * 14
            pr = base_pr - drop
            unif = 0.2 + drop / 20
        else:
            pr, unif = base_pr, np.full(len(centers), 0.2)
        Xout = rng.standard_normal((per_class, d))
        Xin = (rng.standard_normal((per_class, d)) + 3.0 * v) if is_loop \
            else np.zeros((0, d))
        shards.append({
            "task_id": f"SYN_{i:03d}",
            f"Xin_{L}": Xin.astype(np.float16),
            f"Xout_{L}": Xout.astype(np.float16),
            f"pr_{L}": pr, f"unif_{L}": unif, "centers": centers,
            "onset_tok_gen": np.int64(onset), "n_gen_tokens": np.int64(n_gen),
            "is_loop": np.int64(is_loop),
        })
    return shards, v, L


class TestRunnerAnalysis:
    def test_probe_recovers_planted_direction(self):
        mod = _runner()
        shards, v, L = _synthetic_shards()
        res = mod._probe_layer(shards, L, seed=0)
        assert res["token_auc_grouped_oof"] > 0.95
        assert res["chain_auc"] > 0.95
        assert res["chain_mw_p"] < 1e-4
        assert abs(float(res["probe_dir"] @ v)) > 0.8
        assert abs(float(res["meandiff_dir"] @ v)) > 0.9

    def test_probe_null_on_shuffled_shift(self):
        """No planted shift ⇒ chain-level discrimination collapses to chance."""
        mod = _runner()
        shards, v, L = _synthetic_shards(seed=12)
        rng = np.random.default_rng(13)
        for s in shards:                       # remove the signal
            if s[f"Xin_{L}"].shape[0]:
                s[f"Xin_{L}"] = rng.standard_normal(s[f"Xin_{L}"].shape).astype(np.float16)
        res = mod._probe_layer(shards, L, seed=0)
        assert res["chain_auc"] < 0.75

    def test_precedence_detects_planted_contraction(self):
        mod = _runner()
        shards, _, L = _synthetic_shards()
        res = mod._precedence(shards, L, pre=512, base_lo=128, base_hi=640)
        assert res["pr"]["delta_loop_mean"] < -3        # PR fell before onset
        assert abs(res["pr"]["delta_clean_mean"]) < 1.5
        assert res["pr"]["wilcoxon_p_loop"] < 1e-3
        assert res["pr"]["mw_p_loop_vs_clean"] < 1e-3
        assert res["unif"]["mw_p_loop_vs_clean"] < 1e-3
        curve = res["aligned_curve"]
        assert len(curve["pr"]) == len(curve["bin_lo"])
