"""Stub tests for the Phase-7 steering GENERATION ENGINE (no GPU, no real model).

Complements ``tests/test_steered_inference_arms.py`` (which covers the pure vector
arm construction). This file exercises the model-dependent paths of
``src/steered_inference.py`` with synthetic ``nn.Module`` stubs so the real GPU/API
run is verified to be free of runtime errors and arithmetically correct first:

  * the hook math ``h ∓ α·(rᵀh)·r`` — subtract / add / measure modes;
  * the transformers-5.x decoder-layer output being a TUPLE vs a BARE TENSOR
    (a bare-tensor ``output[0]`` would silently index the batch);
  * fp16 / bf16 upcast-then-restore-to-layer-dtype;
  * the hook is removed in ``finally`` (no lingering hooks after generate);
  * the energy probe records |rᵀh| per position on the UNPERTURBED stream and
    does not itself perturb;
  * ``measure_mean_abs_proj`` → ``energy_matched_scale`` calibration delivers
    matched energy;
  * ``run_steering_experiment``: shared-vanilla baseline (no double-count, α=0
    covered), checkpoint/resume (no lost/duplicated work, legacy per-behaviour
    vanilla dropped), atomic write;
  * MULTI-SAMPLE: N samples per (task, arm, α) at temperature>0 keyed distinctly
    (resume + aggregation treat them as N pooled draws); greedy-multisample is
    rejected.

The stubs implement only the surface ``SteeredModel`` / ``run_steering_experiment``
actually touch: ``model.model.layers[L]`` (hook target), ``model.parameters()``
(device/dtype), ``model.device``, ``model.generate(...)``, and a tokenizer with
``__call__`` (returns a dict-like ``BatchEncoding`` supporting ``**`` + ``.to``),
``.decode``, ``.eos_token_id``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

import src.chain_gen as cg
from src.steered_inference import (
    SteeredModel,
    SHARED_BASELINE,
    effective_task_id,
    measure_mean_abs_proj,
    energy_matched_scale,
    random_direction_like,
    run_steering_experiment,
    default_max_new_tokens,
)


# ── Stubs ───────────────────────────────────────────────────────────────────


class _StubLayer(nn.Module):
    """A no-op decoder layer that returns a BARE tensor (transformers-5.x form)."""

    def forward(self, x):  # noqa: D401
        return x


class _StubInner(nn.Module):
    def __init__(self, n_layers: int, hidden: int):
        super().__init__()
        self.layers = nn.ModuleList([_StubLayer() for _ in range(n_layers)])


class StubModel(nn.Module):
    """Minimal causal-LM stub: drives the hooked layer on each generated step.

    ``generate`` honours ``max_new_tokens`` and emits a token id per step (greedy:
    deterministic; sampled: drawn from a per-step random logit so distinct seeds
    diverge). It runs every layer once per step so the registered forward hook
    fires on the streamed positions, exactly like the real decode loop.
    """

    def __init__(self, n_layers: int = 4, hidden: int = 8, vocab: int = 50,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.model = _StubInner(n_layers, hidden)
        self.embed = nn.Embedding(vocab, hidden)
        self.vocab = vocab
        self.hidden = hidden
        self.n_generate_calls = 0
        if dtype != torch.float32:
            self.to(dtype)

    @property
    def device(self):
        return next(self.parameters()).device

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=16,
                 do_sample=False, temperature=1.0, pad_token_id=0, **kw):
        self.n_generate_calls += 1
        bsz = input_ids.shape[0]
        cur = input_ids
        new = []
        for t in range(max_new_tokens):
            h = self.embed(cur)
            for lyr in self.model.layers:
                h = lyr(h)  # fire the hook
            if do_sample:
                logits = torch.randn(bsz, self.vocab)
                probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = torch.full((bsz, 1), (t % (self.vocab - 2)) + 1, dtype=torch.long)
            new.append(nxt)
            cur = torch.cat([cur, nxt], dim=1)
        return torch.cat([input_ids] + new, dim=1)


class _BatchEncoding(dict):
    """dict subclass that supports ``**unpacking`` AND ``.input_ids`` / ``.to``."""

    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, _device):
        return self


class StubTokenizer:
    eos_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, text, return_tensors=None, padding=False):
        return _BatchEncoding(input_ids=torch.tensor([[1, 2, 3, 4]]))

    def decode(self, ids, skip_special_tokens=True):
        # Reflect the actual token ids so distinct sampled sequences decode to
        # distinct strings (a real tokenizer does; a constant "word "*n would
        # collapse different draws to one and defeat the distinctness tests).
        seq = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return " ".join(f"w{int(i)}" for i in seq)


@pytest.fixture(autouse=True)
def _patch_format_prompt(monkeypatch):
    """SteeredModel.generate / generate_chain call format_prompt(tokenizer, instr).
    Stub it to the identity so no real chat template is needed."""
    monkeypatch.setattr(cg, "format_prompt", lambda tok, instr, **kw: instr)


@pytest.fixture(autouse=True)
def _cap_default_tokens(monkeypatch):
    """Cap the default generation length during stub tests so the synthetic
    decode loop is fast. Any test that passes max_new_tokens explicitly overrides
    this; only the default (None) path is affected. (The real default is 8192.)"""
    monkeypatch.setattr("src.steered_inference.default_max_new_tokens", lambda: 6)


@pytest.fixture
def model_tok():
    return StubModel(), StubTokenizer()


def _unit(d, seed):
    v = np.random.default_rng(seed).standard_normal(d)
    return v / np.linalg.norm(v)


# ── Hook math: subtract / add / measure ───────────────────────────────────────


def test_hook_subtract_ablates_r_component_bare_tensor():
    m = StubModel(hidden=4)
    r = np.array([1.0, 0.0, 0.0, 0.0])
    sm = SteeredModel(m, StubTokenizer(), r, layer=0, alpha=1.0, mode="subtract")
    h = torch.tensor([[[3.0, 1.0, 1.0, 1.0], [-2.0, 5.0, 0.0, 0.0]]])
    out = sm._hook_fn(None, None, h)            # bare-tensor branch
    assert not isinstance(out, tuple)
    assert torch.allclose(out[..., 0], torch.zeros(1, 2), atol=1e-6)  # dim0 removed
    assert torch.allclose(out[..., 1:], h[..., 1:])                   # rest intact


def test_hook_add_amplifies_r_component():
    m = StubModel(hidden=4)
    r = np.array([1.0, 0.0, 0.0, 0.0])
    sm = SteeredModel(m, StubTokenizer(), r, layer=0, alpha=0.5, mode="add")
    h = torch.tensor([[[3.0, 1.0, 1.0, 1.0], [-2.0, 5.0, 0.0, 0.0]]])
    out = sm._hook_fn(None, None, h.clone())
    # add: dim0 -> h0 + 0.5*h0 = 1.5*h0
    assert torch.allclose(out[..., 0], torch.tensor([[4.5, -3.0]]), atol=1e-6)


def test_hook_energy_scale_multiplies_delta():
    """energy_scale rescales the perturbation (the energy-matched arm path)."""
    m = StubModel(hidden=4)
    r = np.array([1.0, 0.0, 0.0, 0.0])
    sm = SteeredModel(m, StubTokenizer(), r, layer=0, alpha=1.0, mode="subtract",
                      energy_scale=0.25)
    h = torch.tensor([[[4.0, 9.0, 0.0, 0.0]]])
    out = sm._hook_fn(None, None, h.clone())
    # delta on dim0 = alpha*scale*proj = 1*0.25*4 = 1 -> 4-1 = 3
    assert torch.allclose(out[..., 0], torch.tensor([[3.0]]), atol=1e-6)


def test_hook_constant_subtract_applies_same_vector_at_every_position():
    """Venhoff's write is h <- h - alpha*scale*r, independent of r^T h.

    Deliberately give the three positions positive, negative, and zero
    projections onto r. A projective ablation would produce three different
    deltas (and no delta at the zero-projection position); the constant-vector
    intervention must deliver the same delta to all three.
    """
    m = StubModel(hidden=4)
    r = np.array([1.0, 0.0, 0.0, 0.0])
    sm = SteeredModel(
        m, StubTokenizer(), r, layer=0, alpha=0.5,
        mode="constant_subtract", energy_scale=2.0,
    )
    h = torch.tensor([[[3.0, 1.0, 1.0, 1.0],
                       [-2.0, 5.0, 0.0, 0.0],
                       [0.0, 7.0, 8.0, 9.0]]])
    out = sm._hook_fn(None, None, h.clone())

    expected = h.clone()
    expected[..., 0] -= 1.0  # alpha * energy_scale = 0.5 * 2.0
    assert torch.allclose(out, expected, atol=1e-6)
    assert sm._disp_count == 3
    assert sm.mean_abs_displacement() == pytest.approx(1.0)


def test_hook_constant_subtract_preserves_tuple_extras_and_layer_dtype():
    """The new operator must retain the existing hook's ABI and dtype contract."""
    m = StubModel(hidden=3)
    r = np.array([0.0, 1.0, 0.0])
    sm = SteeredModel(
        m, StubTokenizer(), r, layer=0, alpha=0.25,
        mode="constant_subtract", energy_scale=2.0,
    )
    h = torch.tensor([[[4.0, 2.0, 8.0]]], dtype=torch.float16)
    extra = object()
    out = sm._hook_fn(None, None, (h.clone(), extra))

    assert isinstance(out, tuple) and out[1] is extra
    assert out[0].dtype == torch.float16
    assert torch.allclose(
        out[0].float(), torch.tensor([[[4.0, 1.5, 8.0]]]), atol=1e-3,
    )


def test_hook_constant_subtract_alpha_zero_is_identity():
    """The shared-baseline limit of the fixed operator must be exactly inert."""
    m = StubModel(hidden=4)
    r = _unit(4, 91)
    sm = SteeredModel(
        m, StubTokenizer(), r, layer=0, alpha=0.0,
        mode="constant_subtract", energy_scale=70.868286,
    )
    h = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    out = sm._hook_fn(None, None, h.clone())

    assert torch.equal(out, h)
    assert sm._disp_count == 6
    assert sm.mean_abs_displacement() == 0.0


def test_measure_mode_does_not_perturb_but_records_energy():
    m = StubModel(hidden=4)
    r = np.array([1.0, 0.0, 0.0, 0.0])
    sm = SteeredModel(m, StubTokenizer(), r, layer=0, alpha=0.0, mode="measure")
    h = torch.tensor([[[3.0, 1.0, 1.0, 1.0], [-2.0, 5.0, 0.0, 0.0]]])
    out = sm._hook_fn(None, None, h.clone())
    assert torch.allclose(out, h)                       # unperturbed
    # energy counts POSITIONS (proj is (b,s,1)); |3|+|-2| over 2 positions
    assert sm._abs_proj_count == 2
    assert abs(sm._abs_proj_sum - 5.0) < 1e-6
    assert abs(sm.mean_abs_projection() - 2.5) < 1e-6


# ── tuple-vs-bare-tensor output (transformers version skew) ───────────────────


def test_hook_tuple_output_steers_hidden_preserves_extras():
    m = StubModel(hidden=4)
    r = np.array([1.0, 0.0, 0.0, 0.0])
    sm = SteeredModel(m, StubTokenizer(), r, layer=0, alpha=1.0, mode="subtract")
    h = torch.tensor([[[3.0, 1.0, 1.0, 1.0], [-2.0, 5.0, 0.0, 0.0]]])
    extra = torch.randn(1, 2, 4)
    out = sm._hook_fn(None, None, (h.clone(), extra, "kv-cache"))
    assert isinstance(out, tuple) and len(out) == 3
    assert torch.allclose(out[0][..., 0], torch.zeros(1, 2), atol=1e-6)
    assert out[1] is extra and out[2] == "kv-cache"     # extras passed through


def test_hook_bare_tensor_not_indexed_as_batch():
    """A bare-tensor output[0] would index batch elem 0 (a (seq,hidden) slice).
    The hook must operate on the full (batch,seq,hidden) tensor instead."""
    m = StubModel(hidden=3)
    r = np.array([1.0, 0.0, 0.0])
    sm = SteeredModel(m, StubTokenizer(), r, layer=0, alpha=1.0, mode="subtract")
    h = torch.ones(2, 4, 3)                              # batch=2
    out = sm._hook_fn(None, None, h)
    assert out.shape == (2, 4, 3)                        # both batch rows steered
    assert torch.allclose(out[..., 0], torch.zeros(2, 4), atol=1e-6)


# ── fp16 / bf16 upcast then restore ───────────────────────────────────────────


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hook_preserves_low_precision_dtype(dtype):
    m = StubModel(hidden=4)
    r = np.array([1.0, 0.0, 0.0, 0.0])
    sm = SteeredModel(m, StubTokenizer(), r, layer=0, alpha=1.0, mode="subtract")
    h = torch.tensor([[[4.0, 9.0, 0.0, 0.0]]]).to(dtype)
    out = sm._hook_fn(None, None, h)
    assert out.dtype == dtype                            # restored to layer dtype
    assert torch.allclose(out[..., 0].float(), torch.zeros(1, 1), atol=1e-2)
    # r is stored float32 so the projection is computed in full precision
    assert sm._r.dtype == torch.float32


# ── generate(): hook lifecycle + outputs ──────────────────────────────────────


def test_generate_removes_hook_in_finally(model_tok):
    m, tok = model_tok
    sm = SteeredModel(m, tok, _unit(m.hidden, 0), layer=2, alpha=1.0, mode="subtract")
    sm.generate("hello", max_new_tokens=4)
    assert len(m.model.layers[2]._forward_hooks) == 0    # no lingering hook


def test_generate_removes_hook_even_on_error(model_tok):
    m, tok = model_tok

    def boom(*a, **k):
        raise RuntimeError("CUDA OOM (simulated)")

    m.generate = boom
    sm = SteeredModel(m, tok, _unit(m.hidden, 0), layer=2, alpha=1.0, mode="subtract")
    with pytest.raises(RuntimeError):
        sm.generate("hello", max_new_tokens=4)
    assert len(m.model.layers[2]._forward_hooks) == 0    # finally still removed it


def test_generate_returns_expected_fields(model_tok):
    m, tok = model_tok
    sm = SteeredModel(m, tok, _unit(m.hidden, 1), layer=2, alpha=1.5, mode="subtract",
                      energy_scale=2.0)
    out = sm.generate("hi", max_new_tokens=5)
    assert out["n_tokens"] == 5
    assert out["alpha"] == 1.5 and out["mode"] == "subtract" and out["layer"] == 2
    assert out["energy_scale"] == 2.0
    assert out["mean_abs_proj"] is not None and out["mean_abs_proj"] >= 0.0


def test_generate_seed_makes_sampling_reproducible_and_distinct(model_tok):
    m, tok = model_tok
    sm = SteeredModel(m, tok, _unit(m.hidden, 2), layer=2, alpha=1.0, mode="subtract")
    a = sm.generate("hi", max_new_tokens=8, temperature=0.8, seed=0)
    b = sm.generate("hi", max_new_tokens=8, temperature=0.8, seed=0)
    c = sm.generate("hi", max_new_tokens=8, temperature=0.8, seed=1)
    assert a["chain"] == b["chain"]          # same seed -> identical
    assert a["chain"] != c["chain"]          # different seed -> different draw


def test_generate_default_max_new_tokens_used_when_none(model_tok, monkeypatch):
    m, tok = model_tok
    captured = {}
    orig = m.generate

    def spy(*a, **k):
        captured["mnt"] = k.get("max_new_tokens")
        return orig(*a, **k)

    m.generate = spy
    monkeypatch.setattr("src.steered_inference.default_max_new_tokens", lambda: 4)
    sm = SteeredModel(m, tok, _unit(m.hidden, 0), layer=2, alpha=1.0, mode="subtract")
    sm.generate("hi", max_new_tokens=None)
    assert captured["mnt"] == 4


# ── energy-match calibration end-to-end ───────────────────────────────────────


def test_measure_mean_abs_proj_pools_over_tasks(model_tok):
    m, tok = model_tok
    v = _unit(m.hidden, 3)
    tasks = [{"prompt": "a"}, {"prompt": "b"}]
    e = measure_mean_abs_proj(m, tok, v, layer=2, tasks=tasks, max_new_tokens=8)
    assert e > 0.0


def test_energy_matched_scale_delivers_matched_energy(model_tok):
    """After calibration, scale·|r_rand·h| should equal E_single (the whole point)."""
    m, tok = model_tok
    single = _unit(m.hidden, 4)
    rand = random_direction_like(single, "em|backtracking|L2")
    tasks = [{"prompt": "a"}, {"prompt": "b"}]
    e_beh = measure_mean_abs_proj(m, tok, single, 2, tasks, max_new_tokens=8)
    e_rand = measure_mean_abs_proj(m, tok, rand, 2, tasks, max_new_tokens=8)
    scale = energy_matched_scale(e_beh, e_rand)
    em = SteeredModel(m, tok, rand, 2, alpha=1.0, mode="subtract", energy_scale=scale)
    out = em.generate("a", max_new_tokens=8)
    # mean_abs_proj is the UNSCALED |r·h|; delivered energy = scale * that.
    delivered = scale * out["mean_abs_proj"]
    assert delivered == pytest.approx(e_beh, rel=0.2)


# ── effective_task_id keying ──────────────────────────────────────────────────


def test_effective_task_id_composition():
    assert effective_task_id("t0") == "t0"                       # back-compat
    assert effective_task_id("t0", sample=0) == "t0"             # sample 0 = bare
    assert effective_task_id("t0", sample=2) == "t0#s2"
    assert effective_task_id("t0", replicate=1) == "t0#rs1"
    assert effective_task_id("t0", replicate=1, sample=2) == "t0#rs1#s2"


# ── run_steering_experiment: shared baseline + no double-count ────────────────


def _patch_vanilla(monkeypatch):
    """Patch generate_chain so the vanilla baseline runs without a real model,
    while still firing the model so n_generate_calls is meaningful."""
    def fake(model, tokenizer, prompt, max_new_tokens, temperature=0.0, seed=0, **k):
        model.generate(input_ids=torch.tensor([[1, 2, 3]]), max_new_tokens=3,
                        do_sample=(temperature > 0), temperature=temperature or 1.0)
        return {"chain": f"vanilla(seed={seed})", "n_tokens": 3}
    monkeypatch.setattr(cg, "generate_chain", fake)


def _vecs(hidden, seed=0):
    mk = lambda s: _unit(hidden, s)
    return {"backtracking": {
        "layer": 2,
        "single_direction": mk(seed),
        "manifold_projected": {1: mk(seed + 1), 3: mk(seed + 2), "auto": mk(seed + 3)},
        "auto_k": 3,
    }}


def test_run_shared_vanilla_once_per_task_alpha0_covered(monkeypatch, tmp_path):
    _patch_vanilla(monkeypatch)
    m, tok = StubModel(), StubTokenizer()
    tasks = [{"id": "t0", "prompt": "p0"}, {"id": "t1", "prompt": "p1"}]
    res = run_steering_experiment(
        m, tok, tasks, _vecs(m.hidden), alpha_values=[0.0, 1.0],
        save_path=tmp_path / "r.json",
        include_random_control=False, include_random_subspace=False,
        include_energy_matched=False, include_orthogonal_complement=False,
    )
    vanilla = [r for r in res if r["method"] == "vanilla"]
    assert len(vanilla) == 2                                  # one per task
    assert {r["behaviour"] for r in vanilla} == {SHARED_BASELINE}
    assert {r["alpha"] for r in vanilla} == {0.0}            # α=0 == shared vanilla
    # No steered arm was generated at α=0 (only α>0 runs):
    assert all(r["alpha"] > 0 for r in res if r["method"] != "vanilla")
    # 2 vanilla + 4 arms × 1 steered-α × 2 tasks = 10
    assert len(res) == 10


def test_run_no_duplicate_keys(monkeypatch, tmp_path):
    _patch_vanilla(monkeypatch)
    m, tok = StubModel(), StubTokenizer()
    tasks = [{"id": "t0", "prompt": "p0"}, {"id": "t1", "prompt": "p1"}]
    res = run_steering_experiment(
        m, tok, tasks, _vecs(m.hidden), alpha_values=[1.0, 2.0],
        save_path=tmp_path / "r.json", include_random_control=True,
        include_random_subspace=True, include_energy_matched=False,
        include_orthogonal_complement=True, n_random_subspaces=2,
    )
    keys = [(r["behaviour"], r["method"], r["alpha"], r["task_id"],
             r.get("subspace_replicate")) for r in res]
    assert len(keys) == len(set(keys))                       # every record unique


# ── checkpoint / resume ───────────────────────────────────────────────────────


def test_resume_no_lost_or_duplicated_work(monkeypatch, tmp_path):
    _patch_vanilla(monkeypatch)
    m, tok = StubModel(), StubTokenizer()
    tasks = [{"id": "t0", "prompt": "p0"}, {"id": "t1", "prompt": "p1"}]
    vecs = _vecs(m.hidden)
    save = tmp_path / "r.json"
    kw = dict(include_random_control=False, include_random_subspace=False,
              include_energy_matched=False, include_orthogonal_complement=False)

    full = run_steering_experiment(m, tok, tasks, vecs, [1.0],
                                   save_path=tmp_path / "full.json", **kw)
    full_keys = {(r["behaviour"], r["method"], r["alpha"], r["task_id"],
                  r.get("subspace_replicate")) for r in full}

    # Write a PARTIAL checkpoint (drop everything except vanilla + single_direction)
    partial = [r for r in full if r["method"] in ("vanilla", "single_direction")]
    save.write_text(json.dumps(partial))

    resumed = run_steering_experiment(m, tok, tasks, vecs, [1.0], save_path=save, **kw)
    resumed_keys = {(r["behaviour"], r["method"], r["alpha"], r["task_id"],
                     r.get("subspace_replicate")) for r in resumed}
    assert resumed_keys == full_keys                         # nothing lost
    assert len(resumed) == len(resumed_keys)                 # nothing duplicated


def test_resume_drops_legacy_per_behaviour_vanilla(monkeypatch, tmp_path):
    _patch_vanilla(monkeypatch)
    m, tok = StubModel(), StubTokenizer()
    tasks = [{"id": "t0", "prompt": "p0"}]
    vecs = _vecs(m.hidden)
    save = tmp_path / "r.json"
    kw = dict(include_random_control=False, include_random_subspace=False,
              include_energy_matched=False, include_orthogonal_complement=False)
    # Seed a legacy per-behaviour vanilla record (pre-hoist schema).
    legacy = [{"behaviour": "backtracking", "method": "vanilla", "alpha": 0.0,
               "task_id": "t0", "chain": "legacy", "n_tokens": 3, "layer": None}]
    save.write_text(json.dumps(legacy))
    res = run_steering_experiment(m, tok, tasks, vecs, [1.0], save_path=save, **kw)
    # The only vanilla left must be the SHARED baseline; the legacy one is gone.
    assert all(not (r["method"] == "vanilla" and r["behaviour"] != SHARED_BASELINE)
               for r in res)


def test_resume_skips_already_done_generations(monkeypatch, tmp_path):
    _patch_vanilla(monkeypatch)
    m, tok = StubModel(), StubTokenizer()
    tasks = [{"id": "t0", "prompt": "p0"}, {"id": "t1", "prompt": "p1"}]
    vecs = _vecs(m.hidden)
    save = tmp_path / "r.json"
    kw = dict(include_random_control=False, include_random_subspace=False,
              include_energy_matched=False, include_orthogonal_complement=False)
    run_steering_experiment(m, tok, tasks, vecs, [1.0], save_path=save, **kw)
    calls_after_first = m.n_generate_calls
    # A second identical run must regenerate NOTHING.
    run_steering_experiment(m, tok, tasks, vecs, [1.0], save_path=save, **kw)
    assert m.n_generate_calls == calls_after_first


def test_atomic_write_leaves_no_tmp(monkeypatch, tmp_path):
    _patch_vanilla(monkeypatch)
    m, tok = StubModel(), StubTokenizer()
    tasks = [{"id": "t0", "prompt": "p0"}]
    save = tmp_path / "r.json"
    run_steering_experiment(m, tok, tasks, _vecs(m.hidden), [1.0], save_path=save,
                            include_random_control=False, include_random_subspace=False,
                            include_energy_matched=False, include_orthogonal_complement=False)
    assert save.exists()
    assert not save.with_suffix(".tmp").exists()             # tmp renamed away


# ── multi-sample ──────────────────────────────────────────────────────────────


def test_greedy_multisample_is_rejected(monkeypatch, tmp_path):
    _patch_vanilla(monkeypatch)
    m, tok = StubModel(), StubTokenizer()
    tasks = [{"id": "t0", "prompt": "p0"}]
    with pytest.raises(ValueError, match="temperature>0"):
        run_steering_experiment(m, tok, tasks, _vecs(m.hidden), [1.0],
                                save_path=tmp_path / "r.json",
                                n_samples=5, temperature=0.0)


def test_multisample_produces_n_distinct_draws_per_cell(monkeypatch, tmp_path):
    _patch_vanilla(monkeypatch)
    m, tok = StubModel(), StubTokenizer()
    tasks = [{"id": "t0", "prompt": "p0"}, {"id": "t1", "prompt": "p1"}]
    res = run_steering_experiment(
        m, tok, tasks, _vecs(m.hidden), [1.0], save_path=tmp_path / "r.json",
        include_random_control=False, include_random_subspace=False,
        include_energy_matched=False, include_orthogonal_complement=False,
        n_samples=3, temperature=0.7,
    )
    # single_direction at α=1.0: 2 tasks × 3 samples = 6 distinct records.
    sd = [r for r in res if r["method"] == "single_direction" and r["alpha"] == 1.0]
    assert len(sd) == 6
    # task ids carry the sample suffix; base_task_id keeps the true task.
    assert {r["task_id"] for r in sd} == {
        "t0", "t0#s1", "t0#s2", "t1", "t1#s1", "t1#s2"}
    assert {r["base_task_id"] for r in sd} == {"t0", "t1"}
    assert {r["sample"] for r in sd} == {0, 1, 2}
    # Per (task, arm, α) the 3 sampled chains are genuinely different (distinct seeds).
    for base in ("t0", "t1"):
        chains = {r["chain"] for r in sd if r["base_task_id"] == base}
        assert len(chains) == 3


def test_multisample_vanilla_baseline_also_sampled(monkeypatch, tmp_path):
    _patch_vanilla(monkeypatch)
    m, tok = StubModel(), StubTokenizer()
    tasks = [{"id": "t0", "prompt": "p0"}, {"id": "t1", "prompt": "p1"}]
    res = run_steering_experiment(
        m, tok, tasks, _vecs(m.hidden), [1.0], save_path=tmp_path / "r.json",
        include_random_control=False, include_random_subspace=False,
        include_energy_matched=False, include_orthogonal_complement=False,
        n_samples=3, temperature=0.7,
    )
    vanilla = [r for r in res if r["method"] == "vanilla"]
    # N-vs-N: 2 tasks × 3 samples = 6 vanilla baseline draws (not 2).
    assert len(vanilla) == 6
    assert {r["task_id"] for r in vanilla} == {
        "t0", "t0#s1", "t0#s2", "t1", "t1#s1", "t1#s2"}
    assert {r["seed"] for r in vanilla} == {0, 1, 2}


def test_multisample_keys_match_aggregation_index(monkeypatch, tmp_path):
    """Each sample's (task_id, behaviour, method, alpha) tuple is unique — exactly
    the key aggregate_results' ann_index + annotate_chains dedup use — so samples
    are annotated separately and pooled into one cell, with no collision."""
    _patch_vanilla(monkeypatch)
    m, tok = StubModel(), StubTokenizer()
    tasks = [{"id": "t0", "prompt": "p0"}]
    res = run_steering_experiment(
        m, tok, tasks, _vecs(m.hidden), [1.0], save_path=tmp_path / "r.json",
        include_random_control=False, include_random_subspace=False,
        include_energy_matched=False, include_orthogonal_complement=False,
        n_samples=4, temperature=0.7,
    )
    ann_keys = [(r["task_id"], r["behaviour"], r["method"], r["alpha"]) for r in res]
    assert len(ann_keys) == len(set(ann_keys))               # no annotation collision


def test_multisample_resume_keys_distinct_samples(monkeypatch, tmp_path):
    """Interrupt a multi-sample run after a subset of samples; resume must fill in
    exactly the missing samples (no lost/duplicated sample)."""
    _patch_vanilla(monkeypatch)
    m, tok = StubModel(), StubTokenizer()
    tasks = [{"id": "t0", "prompt": "p0"}]
    vecs = _vecs(m.hidden)
    save = tmp_path / "r.json"
    kw = dict(include_random_control=False, include_random_subspace=False,
              include_energy_matched=False, include_orthogonal_complement=False,
              n_samples=3, temperature=0.7)
    full = run_steering_experiment(m, tok, tasks, vecs, [1.0],
                                   save_path=tmp_path / "full.json", **kw)
    full_keys = {(r["task_id"], r["behaviour"], r["method"], r["alpha"]) for r in full}
    # Drop sample #s2 of every cell to simulate an interruption mid-sample-loop.
    partial = [r for r in full if not str(r["task_id"]).endswith("#s2")]
    save.write_text(json.dumps(partial))
    resumed = run_steering_experiment(m, tok, tasks, vecs, [1.0], save_path=save, **kw)
    resumed_keys = {(r["task_id"], r["behaviour"], r["method"], r["alpha"])
                    for r in resumed}
    assert resumed_keys == full_keys
    assert len(resumed) == len(resumed_keys)


def test_single_sample_default_preserves_legacy_keys(monkeypatch, tmp_path):
    """n_samples=1 must keep the bare task_id (no #s suffix) so pre-multisample
    checkpoints resume unchanged."""
    _patch_vanilla(monkeypatch)
    m, tok = StubModel(), StubTokenizer()
    tasks = [{"id": "t0", "prompt": "p0"}]
    res = run_steering_experiment(
        m, tok, tasks, _vecs(m.hidden), [1.0], save_path=tmp_path / "r.json",
        include_random_control=False, include_random_subspace=False,
        include_energy_matched=False, include_orthogonal_complement=False)
    assert all("#s" not in str(r["task_id"]) for r in res)   # no sample suffix


def test_upgrade_single_to_multisample_on_resume(monkeypatch, tmp_path):
    """Realistic operator flow: run greedy 1-sample first, then resume asking for
    N samples at temperature>0. The bare-task_id records become sample 0; resume
    must add ONLY the new samples (#s1..#s{N-1}), not regenerate sample 0."""
    m, tok = StubModel(), StubTokenizer()
    tasks = [{"id": "t0", "prompt": "p0"}]
    vecs = _vecs(m.hidden)
    save = tmp_path / "r.json"
    kw = dict(include_random_control=False, include_random_subspace=False,
              include_energy_matched=False, include_orthogonal_complement=False)
    # 1) greedy single-sample run
    _patch_vanilla(monkeypatch)
    first = run_steering_experiment(m, tok, tasks, vecs, [1.0], save_path=save, **kw)
    assert all("#s" not in str(r["task_id"]) for r in first)
    calls_after_first = m.n_generate_calls
    # 2) resume with n_samples=3 at temperature>0
    resumed = run_steering_experiment(m, tok, tasks, vecs, [1.0], save_path=save,
                                      n_samples=3, temperature=0.7, **kw)
    # sample-0 records (bare ids) are reused, not regenerated → strictly fewer new
    # generations than a fresh 3-sample run would need.
    sd = [r for r in resumed if r["method"] == "single_direction"]
    assert {r["task_id"] for r in sd} == {"t0", "t0#s1", "t0#s2"}  # 0 reused + 2 new
    assert m.n_generate_calls > calls_after_first                  # only new samples ran
