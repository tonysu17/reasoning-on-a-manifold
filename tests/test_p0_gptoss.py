"""CPU-only smoke tests for the P0 gpt-oss-20b chain generator
(``p0_generate_gptoss_chains.py``) — the H1 pilot's P0 stage.

Proves, WITHOUT downloading the 20 B model or calling any API:
  * pool construction — arm counts, unique ids, XSTest-style pair preservation
    (F13), determinism, and the external / builtin capability loaders;
  * placeholder guard — the BUILTIN harmful placeholders are detected;
  * output schema — a fake gpt-oss-shaped model + harmony tokenizer drive the
    generation path end-to-end and yield chains carrying the analysis channel
    plus per-arm metadata (arm / label / source / difficulty);
  * verification logic — the P0 verdict passes on a good run and flags each
    failure mode (empty channel, wrong effort, missing arm, short composition).

The generation fakes mirror tests/test_gpt_oss_integration.py so the real harmony
wiring in src.chain_gen is exercised with no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

# The runner script lives at the repo root (like 02_/04s_), not under src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import p0_generate_gptoss_chains as p0  # noqa: E402


# ── Synthetic stimuli fixture ─────────────────────────────────────────────────

def _write_stimuli(tmp_path: Path, n_harmful=8, n_benign=8, n_pairs=3) -> Path:
    """A harmless synthetic StrongREJECT+XSTest-shaped stimuli JSON. The 'harmful'
    items are innocuous strings — this only exercises the loader / sampler."""
    items = []
    for k in range(n_pairs):
        items.append({"id": f"ph{k}", "prompt": f"harmful-ish request {k}",
                      "label": "harmful", "category": "cat", "source": "syn",
                      "pair_id": f"pair{k}"})
        items.append({"id": f"pb{k}", "prompt": f"benign homonym {k}",
                      "label": "benign", "category": "cat", "source": "syn",
                      "pair_id": f"pair{k}"})
    for k in range(n_harmful - n_pairs):
        items.append({"id": f"h{k}", "prompt": f"harmful-ish standalone {k}",
                      "label": "harmful", "category": "cat", "source": "syn"})
    for k in range(n_benign - n_pairs):
        items.append({"id": f"b{k}", "prompt": f"benign standalone {k}",
                      "label": "benign", "category": "cat", "source": "syn"})
    p = tmp_path / "stimuli.json"
    p.write_text(json.dumps(items))
    return p


# ── Pool construction ─────────────────────────────────────────────────────────

def test_build_pool_arm_counts_and_unique_ids(tmp_path):
    src = _write_stimuli(tmp_path, n_harmful=8, n_benign=8, n_pairs=3)
    pool = p0.build_p0_pool(src, "builtin", n_harmful=6, n_benign=6, n_capability=5, seed=1)
    counts = p0.pool_arm_counts(pool)
    assert counts == {"harmful": 6, "benign": 6, "capability": 5}
    ids = [item["id"] for item in pool]
    assert len(ids) == len(set(ids)), "pool ids must be unique"
    assert all(item["id"].startswith(item["arm"] + "::") for item in pool)


def test_build_pool_preserves_matched_pairs(tmp_path):
    # All harmful are paired: every sampled harmful pair_id must have its benign.
    src = _write_stimuli(tmp_path, n_harmful=3, n_benign=8, n_pairs=3)
    pool = p0.build_p0_pool(src, "builtin", n_harmful=3, n_benign=6, n_capability=2, seed=7)
    harmful_pids = {i["pair_id"] for i in pool if i["arm"] == "harmful" and i["pair_id"]}
    benign_pids = {i["pair_id"] for i in pool if i["arm"] == "benign" and i["pair_id"]}
    assert harmful_pids and harmful_pids.issubset(benign_pids), \
        "each sampled harmful pair must keep its benign partner (F13)"


def test_build_pool_is_deterministic(tmp_path):
    src = _write_stimuli(tmp_path)
    a = p0.build_p0_pool(src, "builtin", n_harmful=5, n_benign=5, n_capability=4, seed=42)
    b = p0.build_p0_pool(src, "builtin", n_harmful=5, n_benign=5, n_capability=4, seed=42)
    assert [i["id"] for i in a] == [i["id"] for i in b]


def test_capability_arm_carries_difficulty(tmp_path):
    src = _write_stimuli(tmp_path)
    pool = p0.build_p0_pool(src, "builtin", n_harmful=4, n_benign=4, n_capability=6, seed=3)
    cap = [i for i in pool if i["arm"] == "capability"]
    assert len(cap) == 6
    assert all(i["difficulty"] is not None for i in cap), \
        "F3 difficulty proxy must be attached to every capability item"
    assert all(i["label"] == "capability_control" for i in cap)


def test_external_capability_loader(tmp_path):
    ext = tmp_path / "cap.json"
    ext.write_text(json.dumps([
        {"id": "x1", "prompt": "prove something hard", "difficulty": 5, "category": "math"},
        {"id": "x2", "prompt": "another hard one", "difficulty": 4, "category": "cs"},
    ]))
    src = _write_stimuli(tmp_path)
    pool = p0.build_p0_pool(src, ext, n_harmful=2, n_benign=2, n_capability=2, seed=0)
    cap = [i for i in pool if i["arm"] == "capability"]
    assert {i["id"] for i in cap} == {"capability::x1", "capability::x2"}
    assert all(i["source"] == str(ext) for i in cap)


def test_builtin_capability_set_is_large_enough_for_default():
    # The bundled difficulty anchor must supply the plan's default 20.
    assert len(p0.CAPABILITY_BUILTIN) >= p0.N_CAPABILITY
    assert all("difficulty" in r and r["difficulty"] for r in p0.CAPABILITY_BUILTIN)


def test_placeholder_stimuli_detected():
    from src.safety.stimuli import load_stimuli
    assert p0._is_placeholder_stimuli(load_stimuli("builtin")) is True


# ── Generation path (fake gpt-oss model + harmony tokenizer) ──────────────────

_HARMONY_TMPL = (
    "<|start|>assistant<|channel|>analysis<|message|>{cot}<|end|>"
    "<|start|>assistant<|channel|>final<|message|>{ans}<|return|>"
)


class _Enc(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def to(self, *_, **__):
        return self


class _FakeHarmonyTokenizer:
    eos_token_id = 199999
    eos_token = "<eos>"
    pad_token = None

    def __init__(self):
        self._last = ""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, reasoning_effort=None, **kw):
        self._last = messages[-1]["content"]
        return f"<|user|>{self._last}<|assistant|>"

    def __call__(self, text, return_tensors=None):
        return _Enc(input_ids=torch.tensor([[1, 2, 3, 4]]))

    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens is False, "gpt-oss must keep harmony channels"
        # Harmful-ish prompts get a policy-citing CoT; benign get a plain one.
        cot = ("I must check the policy. This request looks disallowed."
               if "harmful" in self._last else
               "Let me work through this step by step and compute the answer.")
        return _HARMONY_TMPL.format(cot=cot, ans="Here is my response.")


class _FakeModel:
    device = torch.device("cpu")

    def generate(self, **kw):
        # prompt was 4 tokens; append a few 'generated' ids
        return torch.tensor([[1, 2, 3, 4, 9, 9, 9, 9]])


def _small_pool(tmp_path):
    src = _write_stimuli(tmp_path, n_harmful=4, n_benign=6, n_pairs=2)
    return p0.build_p0_pool(src, "builtin", n_harmful=3, n_benign=3,
                            n_capability=2, seed=5)


def test_generate_p0_chains_schema(tmp_path):
    pool = _small_pool(tmp_path)
    out = tmp_path / "chains.json"
    chains = p0.generate_p0_chains(
        _FakeModel(), _FakeHarmonyTokenizer(), pool,
        reasoning_effort="medium", max_new_tokens=8, save_path=out)

    assert len(chains) == len(pool)
    for rec in chains:
        # harmony analysis channel captured (F1)
        assert rec["chain"].strip(), "analysis channel must be non-empty"
        assert rec["final_answer"].startswith("Here is")
        assert rec["family"] == "gpt_oss"
        assert rec["reasoning_effort"] == "medium"
        # per-arm metadata merged back
        assert rec["arm"] in p0.ARMS
        assert "label" in rec and "source" in rec and "difficulty" in rec
    # persisted with the enriched fields
    saved = json.loads(out.read_text())
    assert all("arm" in r for r in saved)


# ── Verification logic ────────────────────────────────────────────────────────

def _good_run(tmp_path):
    pool = p0.build_p0_pool(
        _write_stimuli(tmp_path, n_harmful=10, n_benign=10, n_pairs=3),
        "builtin", n_harmful=6, n_benign=6, n_capability=4, seed=11)
    chains = p0.generate_p0_chains(
        _FakeModel(), _FakeHarmonyTokenizer(), pool,
        reasoning_effort="medium", max_new_tokens=8)
    return pool, chains


def test_verification_passes_on_good_run(tmp_path):
    pool, chains = _good_run(tmp_path)
    verdict = p0.run_p0_verification(
        chains, pool, effort="medium", n_harmful=6, n_benign=6, n_capability=4)
    assert verdict["passed"] is True
    for name, c in verdict["checks"].items():
        if c["hard"]:
            assert c["passed"], f"hard check {name} should pass: {c['detail']}"


def test_verification_flags_empty_channel(tmp_path):
    pool, chains = _good_run(tmp_path)
    chains[0]["chain"] = "   "  # analysis channel lost
    verdict = p0.run_p0_verification(
        chains, pool, effort="medium", n_harmful=6, n_benign=6, n_capability=4)
    assert verdict["checks"]["analysis_channel_captured"]["passed"] is False
    assert verdict["passed"] is False


def test_verification_flags_wrong_effort(tmp_path):
    pool, chains = _good_run(tmp_path)
    chains[2]["reasoning_effort"] = "high"
    verdict = p0.run_p0_verification(
        chains, pool, effort="medium", n_harmful=6, n_benign=6, n_capability=4)
    assert verdict["checks"]["effort_uniform"]["passed"] is False
    assert verdict["passed"] is False


def test_verification_flags_missing_arm(tmp_path):
    pool, chains = _good_run(tmp_path)
    chains[1].pop("arm")
    verdict = p0.run_p0_verification(
        chains, pool, effort="medium", n_harmful=6, n_benign=6, n_capability=4)
    assert verdict["checks"]["arm_metadata"]["passed"] is False
    assert verdict["passed"] is False


def test_verification_flags_generation_shortfall(tmp_path):
    pool, chains = _good_run(tmp_path)
    for c in chains[:5]:            # kill >5% of the run
        c["n_tokens"] = 0
        c["error"] = "boom"
    verdict = p0.run_p0_verification(
        chains, pool, effort="medium", n_harmful=6, n_benign=6, n_capability=4)
    assert verdict["checks"]["generated"]["passed"] is False
    assert verdict["passed"] is False


def test_verification_composition_soft_band(tmp_path):
    # A short external set (all arms populated, total in [90,110]) still passes
    # composition via the soft band even without the exact 40/40/20.
    pool = p0.build_p0_pool(
        _write_stimuli(tmp_path, n_harmful=45, n_benign=45, n_pairs=5),
        "builtin", n_harmful=40, n_benign=40, n_capability=20, seed=2)
    verdict = p0.run_p0_verification(
        [], pool, effort="medium")  # composition-only (dry-run shape)
    assert verdict["checks"]["composition"]["passed"] is True
    assert verdict["checks"]["composition"]["detail"]["exact_match"] is True
    assert verdict["metrics"]["total"] == 100
