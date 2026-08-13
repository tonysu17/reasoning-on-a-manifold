"""Tests for the S4 safety pipeline: stimulus loaders, the DSR annotation schema,
early-token shallowness (Qi), the recipe-analysis core, and the runner wiring.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.safety import stimuli, deliberation, shallowness
from src.safety.geometry_analysis import (
    analyze_recipes, summarise, build_per_model, load_safety_activations,
)


# ── Stimuli ───────────────────────────────────────────────────────────────────

def test_builtin_stimuli_have_harmful_and_benign():
    s = stimuli.load_stimuli()
    labels = {x.label for x in s}
    assert "harmful" in labels and "benign" in labels
    grouped = stimuli.by_label(s)
    assert len(grouped["harmful"]) >= 3
    assert len(grouped["benign"]) >= 3


def test_matched_pairs_have_both_sides():
    pairs = stimuli.matched_pairs(stimuli.load_stimuli())
    assert len(pairs) == 2  # 'kill' and 'crack'
    for harmful, benign in pairs:
        assert harmful.label == "harmful"
        assert benign.label == "benign"
        assert harmful.pair_id == benign.pair_id


def test_load_stimuli_from_json(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([
        {"id": "x1", "prompt": "hi", "label": "benign", "category": "test"},
        {"id": "x2", "prompt": "[placeholder]", "label": "harmful", "category": "cyber"},
    ]))
    s = stimuli.load_stimuli(p)
    assert len(s) == 2
    assert s[0].label == "benign" and s[1].label == "harmful"


def test_stimulus_rejects_bad_label():
    with pytest.raises(ValueError):
        stimuli.Stimulus("z", "p", "not_a_label")


def test_format_for_model_delegates_to_adapter():
    class _Tok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
            return "<|u|>" + messages[-1]["content"] + "<|a|>"

    s = stimuli.Stimulus("b", "What is 2+2?", "benign")
    out = stimuli.format_for_model(s, _Tok(), family="deepseek")
    assert out.rstrip().endswith("<think>")  # deepseek formatting applied


# ── DSR annotation schema ─────────────────────────────────────────────────────

def test_classify_spec_citation():
    assert "spec_citation" in deliberation.classify_sentence("This violates our content policy.")
    assert "spec_citation" in deliberation.classify_sentence("Our guidelines prohibit this.")


def test_classify_harm_recognition():
    assert "harm_recognition" in deliberation.classify_sentence("This request is harmful and illegal.")


def test_classify_decision_refuse():
    labels = deliberation.classify_sentence("I must refuse to help with this.")
    assert "decision" in labels
    assert deliberation.decision_type("I must refuse to help with this.") == "refuse"


def test_classify_adjudication_not_decision():
    labels = deliberation.classify_sentence("However, I should weigh whether to comply.")
    assert "adjudication" in labels
    assert "decision" not in labels


def test_decision_type_safe_complete_precedence():
    s = "I can provide a high-level overview without operational details."
    assert deliberation.decision_type(s) == "safe_complete"


def test_policy_citation_rate():
    ann = deliberation.annotate_dsr([
        "This violates policy.", "The sky is blue.", "Our guidelines prohibit this.",
    ])
    assert deliberation.policy_citation_rate(ann) == pytest.approx(2 / 3)
    assert deliberation.policy_citation_rate([]) == 0.0


def test_knows_but_complies():
    ann = deliberation.annotate_dsr(["This is harmful.", "Sure, here's how."])
    assert deliberation.knows_but_complies(ann, complied=True) is True
    assert deliberation.knows_but_complies(ann, complied=False) is False


def test_final_decision_is_last():
    ann = deliberation.annotate_dsr([
        "I must refuse.",
        "Actually I can provide a general overview without operational details.",
    ])
    assert deliberation.final_decision(ann) == "safe_complete"


def test_build_dsr_judge_prompt_contains_chain():
    prompt = deliberation.build_dsr_judge_prompt("some CoT text", policy_excerpt="POLICY")
    assert "some CoT text" in prompt and "POLICY" in prompt and "dsr_labels" in prompt


# ── Shallowness (early-token KL) ──────────────────────────────────────────────

def test_per_token_kl_zero_for_identical():
    V = 4
    logp = np.log(np.full((3, V), 1.0 / V))
    kl = shallowness.per_token_kl(logp, logp)
    assert np.allclose(kl, 0.0, atol=1e-9)


def test_per_token_kl_positive_for_different():
    p_a = np.array([[0.7, 0.2, 0.1]])
    p_b = np.array([[0.1, 0.2, 0.7]])
    kl = shallowness.per_token_kl(np.log(p_a), np.log(p_b))
    assert kl.shape == (1,)
    assert kl[0] > 0.0


def test_shallowness_ratio_concentrated_vs_uniform():
    concentrated = np.array([10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    uniform = np.ones(10)
    assert shallowness.shallowness_ratio(concentrated, k=5) == pytest.approx(1.0)
    assert shallowness.shallowness_ratio(uniform, k=5) == pytest.approx(0.5)
    assert shallowness.shallowness_ratio(np.zeros(10)) == 0.0


def test_cumulative_kl_fraction_monotone_ends_at_one():
    kl = np.array([1.0, 2.0, 3.0, 4.0])
    frac = shallowness.cumulative_kl_fraction(kl)
    assert np.all(np.diff(frac) >= 0)
    assert frac[-1] == pytest.approx(1.0)


def test_depth_50_token():
    # 50% of total (10) reached at index 3 (cumsum 1,3,6,10 -> >=5 at idx 2? )
    kl = np.array([1.0, 2.0, 3.0, 4.0])  # cumfrac .1,.3,.6,1.0 -> 0.5 first at idx 2
    assert shallowness.depth_50_token(kl) == 2
    assert shallowness.depth_50_token(np.zeros(5)) == 0


# ── Recipe-analysis core ──────────────────────────────────────────────────────

def _layer_acts(n, d, sep, axis, seed):
    rng = np.random.default_rng(seed)
    u = np.zeros(d)
    u[axis] = 1.0
    return {"harmful": rng.standard_normal((n, d)) + sep * u,
            "harmless": rng.standard_normal((n, d))}


def test_analyze_recipes_fingerprint_best_layer_and_cosine():
    per_model = {
        "DA":   {"hidden_dim": 8, "layers": {3: _layer_acts(120, 8, 1.0, 0, 1),
                                             5: _layer_acts(120, 8, 5.0, 0, 2)}},
        "rlhf": {"hidden_dim": 8, "layers": {3: _layer_acts(120, 8, 1.0, 0, 3),
                                             5: _layer_acts(120, 8, 2.0, 0, 4)}},
    }
    res = analyze_recipes(per_model)
    assert res["best_layer"]["DA"] == 5          # strongest separation
    assert (res["fingerprint"]["DA"][5]["cohens_d"]
            > res["fingerprint"]["rlhf"][5]["cohens_d"])   # H2 ordering
    # both refusal directions lie along axis 0 -> high cosine (H4 output-conv.)
    assert res["cosine"]["DA|rlhf"][5] > 0.5
    rows = summarise(res)
    assert rows[0]["model"] == "DA"              # sorted by cohens_d


def test_analyze_recipes_cka_across_dims():
    rng = np.random.default_rng(10)
    N, d1, d2 = 80, 8, 12
    H = rng.standard_normal((N, d1))
    Q, _ = np.linalg.qr(rng.standard_normal((d2, d1)))   # (d2,d1) orthonormal cols
    A = Q.T                                              # (d1,d2) orthonormal rows
    per_model = {
        "A": {"hidden_dim": 8,  "layers": {0: {"harmful": H,     "harmless": rng.standard_normal((N, d1))}}},
        "B": {"hidden_dim": 12, "layers": {0: {"harmful": H @ A, "harmless": rng.standard_normal((N, d2))}}},
    }
    res = analyze_recipes(per_model)
    assert "A|B" not in res["cosine"]            # different dim -> no cosine
    assert res["cka"]["A|B"][0] > 0.99           # but CKA recovers the linear map


def test_build_per_model_from_npy(tmp_path):
    root = tmp_path / "acts"
    md = root / "R1-1.5B"
    md.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for L in (3, 5):
        np.save(md / f"harmful_layer{L}.npy", rng.standard_normal((20, 8)))
        np.save(md / f"harmless_layer{L}.npy", rng.standard_normal((20, 8)))
    loaded = load_safety_activations(md)              # infers layers
    assert set(loaded) == {3, 5}
    pm = build_per_model({"R1-1.5B": {"short_name": "R1-1.5B", "hidden_dim": 8}}, root)
    assert pm["R1-1.5B"]["hidden_dim"] == 8
    assert set(pm["R1-1.5B"]["layers"]) == {3, 5}


# ── Runner wiring ─────────────────────────────────────────────────────────────

def test_runner_module_imports_and_has_main():
    import importlib
    mod = importlib.import_module("14_safety_geometry")
    assert hasattr(mod, "main")
    assert "gpt-oss-20b" in mod.DEFAULT_MODELS


def test_extract_runner_imports():
    import importlib
    mod = importlib.import_module("04s_extract_safety")
    assert hasattr(mod, "main")


# ── Safety extraction pass (end-to-end over a fake model) ─────────────────────

def test_extract_and_save_safety_activations_roundtrip(tmp_path):
    import zlib
    import torch
    import torch.nn as nn
    from src.safety.extraction import extract_prompt_activations, save_safety_activations
    from src.safety.stimuli import Stimulus

    HID = 16

    class _Blk(nn.Module):
        def __init__(self, delta):
            super().__init__()
            self.delta = float(delta)

        def forward(self, h):
            return (h + self.delta,)

    class _Inner(nn.Module):
        def __init__(self, nl, vocab, hid):
            super().__init__()
            self.embed = nn.Embedding(vocab, hid)
            self.layers = nn.ModuleList([_Blk(i + 1) for i in range(nl)])

    class _Model(nn.Module):
        def __init__(self, nl=6, vocab=64, hid=HID):
            super().__init__()
            self.model = _Inner(nl, vocab, hid)
            self._v = vocab

        @property
        def device(self):
            return torch.device("cpu")

        def forward(self, input_ids=None, attention_mask=None, **_):
            h = self.model.embed(input_ids % self._v)
            for b in self.model.layers:
                h = b(h)[0]
            return h

    class _Tok:
        eos_token_id = 0

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, **kw):
            return msgs[-1]["content"]

        def __call__(self, text, return_tensors=None):
            ids = [zlib.crc32(w.encode()) % 50 + 1 for w in text.split()] or [1]
            t = torch.tensor([ids], dtype=torch.long)
            return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    stims = [
        Stimulus("h1", "alpha beta gamma", "harmful"),
        Stimulus("h2", "delta epsilon zeta", "harmful"),
        Stimulus("b1", "one two three", "benign"),
        Stimulus("b2", "four five six", "benign"),
    ]
    acts, order = extract_prompt_activations(_Model(), _Tok(), stims, layers=[0, 3, 5], family="base")
    assert acts["harmful"][5].shape == (2, HID)
    assert acts["benign"][0].shape == (2, HID)
    assert order["harmful"] == ["h1", "h2"]

    out = tmp_path / "R1-1.5B"
    save_safety_activations(acts, out, order)          # benign -> harmless on disk
    assert (out / "harmful_layer5.npy").exists()
    assert (out / "harmless_layer5.npy").exists()       # alias applied
    assert (out / "stimulus_order.json").exists()

    loaded = load_safety_activations(out)               # the loader the runner uses
    assert set(loaded) == {0, 3, 5}
    assert loaded[5]["harmful"].shape == (2, HID)
    assert loaded[5]["harmless"].shape == (2, HID)
