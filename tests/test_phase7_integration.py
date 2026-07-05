"""Phase-7 PRE-FLIGHT integration test — STUBBED end-to-end dry run.

Goal: prove `07_evaluate_steering.py` runs the WHOLE pipeline end-to-end with NO
GPU and NO API spend, so the real RTX-4090 run does not die on a wiring/
integration error after hours of generation. Unit tests (test_steered_inference_
arms.py, test_evaluation_*.py) cover the pure pieces; this test wires them
together exactly as the runner does:

    load vectors  →  load_model (STUB)  →  run_steering_experiment
        →  _build_arms (every arm)  →  SteeredModel.generate (real fwd-hook)
        →  energy-matched calibration (measure mode)  →  checkpoint JSON
        →  aggregate_results  →  generation_metrics.json (--skip-annotation)
        →  [annotate path] annotate_chains (MOCK) → aggregate → eval_summary.json

The stub is a real ``torch.nn.Module`` whose ``model.layers[L]`` is a genuine
submodule, so ``register_forward_hook`` fires on every generated token and the
einsum projection / energy accumulation / mean_abs_proj all run for real. Only
the heavyweight transformer + tokenizer download is replaced.

Run:  pytest tests/test_phase7_integration.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
REAL_VECTORS = REPO / "results" / "steering_vectors" / "R1-1.5B"
# The runner HARDCODES results/eval/<short_name>/ as its output dir (no override
# flag), so every test writes there and run_steering_experiment would RESUME from
# a prior test's steering_results.json. Isolate per-test by wiping it.
EVAL_DIR_R1 = REPO / "results" / "eval" / "R1-1.5B"


def _wipe_eval_dir():
    import shutil
    shutil.rmtree(EVAL_DIR_R1, ignore_errors=True)
    # Prune the results/eval parent if it is now empty so the suite leaves the
    # working tree exactly as it found it (results/eval/ does not normally exist).
    parent = EVAL_DIR_R1.parent
    try:
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        pass


@pytest.fixture(autouse=True)
def _clean_eval_dir():
    """Remove the runner's hardcoded eval dir before and after each test so the
    checkpoint/resume logic can't leak state between tests (and so the suite
    leaves no artifacts behind in the repo working tree)."""
    _wipe_eval_dir()
    yield
    _wipe_eval_dir()


# ───────────────────────── Tiny stub model + tokenizer ────────────────────────
# Faithful to the operations SteeredModel / generate_chain perform:
#   * next(model.parameters()).device / .dtype       (SteeredModel.__init__)
#   * model.model.layers[layer].register_forward_hook (hook install)
#   * model.generate(**inputs, max_new_tokens=...)    (returns (1, prompt+gen) ids)
#   * model.device                                    (tokenize .to(device))
# The decoder layer is a real nn.Module returning (batch, seq, hidden) so the
# forward hook receives the same shape the real model produces.

class _StubLayer(nn.Module):
    """A decoder layer that returns a (batch, seq, hidden) tensor.

    Returns a 1-tuple (like transformers <5 decoder layers) so the runner's
    tuple-vs-bare-tensor handling is exercised on the tuple branch.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        # A trivial trainable param so next(model.parameters()) has device/dtype.
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, hidden_states):
        # Deterministic, non-degenerate hidden states with a clear projection
        # onto any unit vector (values O(1)). Shape (batch, seq, hidden).
        return (hidden_states * self.scale,)


class _StubInner(nn.Module):
    def __init__(self, hidden: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(_StubLayer(hidden) for _ in range(n_layers))


class StubModel(nn.Module):
    """Minimal CausalLM stub with a real generate() loop that drives the hook.

    generate() emits ``max_new_tokens`` ids but stops early at a synthetic EOS so
    the smoke run is milliseconds. On every step it pushes a (1, seq, hidden)
    tensor through ``self.model.layers[L]`` so a registered forward hook fires
    exactly as in real generation (accumulating _abs_proj_sum / count).
    """

    #: emit this many real tokens then EOS — keeps the dry run instant but >32
    #: so chains are NOT all flagged degenerate (DEGENERATE_TOKEN_FLOOR = 32).
    GEN_TOKENS = 40

    def __init__(self, hidden: int = 16, n_layers: int = 4, eos_token_id: int = 2):
        super().__init__()
        self.model = _StubInner(hidden, n_layers)
        self.hidden = hidden
        self.eos_token_id = eos_token_id
        self.config = type("Cfg", (), {"num_hidden_layers": n_layers})()

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=16,
                 do_sample=False, temperature=1.0, pad_token_id=None, **kw):
        device = self.device
        seq = input_ids
        n_emit = min(int(max_new_tokens), self.GEN_TOKENS)
        for step in range(n_emit):
            # Push hidden states through EVERY layer in sequence (as a real
            # transformer does) so the steering hook fires regardless of WHICH
            # layer index the runner registered it on (self.layer). Driving only
            # one hardcoded layer would silently miss the hook for other layers —
            # exactly the integration bug this test must surface.
            h = torch.arange(1, seq.shape[1] + 1, dtype=torch.float32, device=device)
            h = h.view(1, -1, 1).repeat(1, 1, self.hidden) * 0.01
            for layer in self.model.layers:
                h = layer(h)[0]  # layers return a 1-tuple; feed hidden forward
            nxt = self.eos_token_id if step == n_emit - 1 else (100 + step)
            seq = torch.cat([seq, torch.tensor([[nxt]], device=device)], dim=1)
        return seq


class StubTokenizer:
    """Just enough tokenizer surface for format_prompt / tokenize / decode.

    No ``apply_chat_template`` → the deepseek adapter falls back to the manual
    ``<|begin▁of▁sentence|>…<think>`` template (str.format only), so no real
    tokenizer files are needed. ``__call__`` returns an object exposing
    ``.input_ids`` and ``.to(device)`` like a transformers BatchEncoding.
    """

    eos_token_id = 2
    pad_token = "<pad>"
    eos_token = "</s>"

    class _Enc:
        def __init__(self, ids):
            self.input_ids = ids

        def to(self, device):
            self.input_ids = self.input_ids.to(device)
            return self

        def __getitem__(self, k):
            return {"input_ids": self.input_ids}[k]

        def keys(self):
            return ["input_ids"]

        def __iter__(self):
            return iter(["input_ids"])

    def __call__(self, text, return_tensors=None, **kw):
        # Deterministic fake tokenisation: 1 id per word, clamped to a few.
        n = max(1, min(len(text.split()), 24))
        ids = torch.arange(3, 3 + n, dtype=torch.long).unsqueeze(0)
        return self._Enc(ids)

    def decode(self, ids, skip_special_tokens=True):
        # Return a multi-sentence chain so re-annotation has something to split.
        n = int(ids.shape[0]) if hasattr(ids, "shape") else len(ids)
        return (" ".join(f"step {i} reasoning here." for i in range(n)))[:400]

    # format_prompt's deepseek branch calls apply_chat_template inside try/except;
    # raising forces the manual-template fallback (no jinja template needed).
    def apply_chat_template(self, *a, **k):
        raise RuntimeError("stub: no chat template — use manual fallback")


def _fake_load_model(model_id, dtype="float16", use_4bit=False, cache_dir=None,
                     **kw):
    return StubModel(hidden=16, n_layers=4), StubTokenizer()


# ───────────────────────── Synthetic vector set on disk ───────────────────────

def _write_synthetic_vectors(dirpath: Path, behaviours, hidden=16, layer=1,
                             seed=0):
    """Write a vectors dir shaped exactly like save_steering_vectors output:
    {beh}_single.npy, {beh}_manifold_k{1,3,5,10,auto}.npy + metadata.json.

    layer=1 is a VALID index into the 4-layer stub (0..3). auto_k is recorded so
    the random-subspace k for the 'auto' arm resolves (it reads vecs['auto_k']).
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    meta = {}
    for i, beh in enumerate(behaviours):
        single = rng.standard_normal(hidden).astype(np.float32)
        single /= np.linalg.norm(single)
        np.save(dirpath / f"{beh}_single.npy", single)
        ks = [1, 3, 5, 10, "auto"]
        for k in ks:
            v = rng.standard_normal(hidden).astype(np.float32)
            v /= np.linalg.norm(v)
            np.save(dirpath / f"{beh}_manifold_k{k}.npy", v)
        meta[beh] = {"layer": layer, "n_on": 100, "n_off": 200,
                     "auto_k": 7, "n_excluded": 5, "k_values": ks}
    (dirpath / "metadata.json").write_text(json.dumps(meta, indent=2))


# ───────────────────────── Runner import + arg/IO plumbing ─────────────────────

def _load_runner():
    """Import 07_evaluate_steering.py as a module (numeric filename → importlib)."""
    path = REPO / "07_evaluate_steering.py"
    spec = importlib.util.spec_from_file_location("phase7_runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_main(runner, argv, monkeypatch, tmp_path, vectors_subdir="R1-1.5B"):
    """Invoke runner.main() under argv, with the model loader stubbed and the
    repo CWD active so results/ + data/ resolve. Returns the eval dir."""
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(sys, "argv", ["07_evaluate_steering.py", *argv])
    # Stub the loader the runner imported by-name (`from src.chain_gen import
    # load_model`), AND the source symbol, so every reference resolves to the
    # stub regardless of how downstream code grabs it.
    monkeypatch.setattr(runner, "load_model", _fake_load_model)
    import src.chain_gen as cg
    monkeypatch.setattr(cg, "load_model", _fake_load_model)
    runner.main()
    short = vectors_subdir
    return REPO / "results" / "eval" / short


# Default arms (norm-matched random / random-subspace / energy-matched /
# orthogonal-complement) all ON, with replicates trimmed to 1 so the synthetic
# dry run stays fast while still exercising the replicate-folding code path.
_SMOKE_ARGS = ["--smoke", "--skip-annotation", "--n-random-subspaces", "1"]


# ════════════════════════════════ TESTS ═══════════════════════════════════════

def test_end_to_end_generation_first_writes_metrics(monkeypatch, tmp_path):
    """THE key deliverable: full --smoke --skip-annotation pipeline on SYNTHETIC
    vectors. Asserts vectors→arms→generate→aggregate→generation_metrics.json all
    fire and the expected per-(beh, method, α) cells + damage metrics appear."""
    runner = _load_runner()

    behaviours = ["backtracking", "uncertainty-estimation",
                  "example-testing", "adding-knowledge"]
    synth = tmp_path / "steer" / "STUB-1.5B"
    _write_synthetic_vectors(synth, behaviours, hidden=16, layer=1)

    # Point the runner's vectors_dir + eval_dir at the synthetic set without
    # editing the runner: it builds Path(f"results/steering_vectors/{short}").
    # We instead monkeypatch load_steering_vectors to read our synthetic dir and
    # redirect the eval-dir short code to a scratch area.
    from src import steering as steering_mod
    real_load = steering_mod.load_steering_vectors
    monkeypatch.setattr(runner, "load_steering_vectors",
                        lambda d: real_load(synth))

    eval_dir = REPO / "results" / "eval" / "1.5b"  # cli_alias short? -> no: short_name
    # The runner derives `short` from config (R1-1.5B). Compute it the same way.
    short = runner.MODELS["1.5b"][1]
    out_dir = REPO / "results" / "eval" / short

    # Make the vectors-dir existence check pass (runner checks the REAL path
    # before calling our patched loader).
    (REPO / "results" / "steering_vectors" / short).mkdir(parents=True, exist_ok=True)

    eval_dir = _run_main(runner, _SMOKE_ARGS, monkeypatch, tmp_path,
                         vectors_subdir=short)

    # ── eval_task_ids.json: deterministic, category-stratified, smoke-truncated ─
    ids = json.loads((eval_dir / "eval_task_ids.json").read_text())
    assert len(ids["task_ids"]) == 3, "smoke must truncate to 3 tasks"
    assert "smoke" in ids["rule"]

    # ── generation_metrics.json: the skip-annotation deliverable ───────────────
    gm_path = eval_dir / "generation_metrics.json"
    assert gm_path.exists(), "generation_metrics.json not written"
    summary = json.loads(gm_path.read_text())

    # Every target behaviour present.
    assert set(summary) == set(behaviours), \
        f"behaviours missing from summary: {set(behaviours) - set(summary)}"

    # Each behaviour must carry the full arm set at α=1.0 (smoke α=[0,1]).
    expected_methods = {
        "vanilla", "single_direction",
        "manifold_k1", "manifold_k3", "manifold_k5", "manifold_k10",
        "manifold_auto",
        "random_subspace_k1", "random_subspace_k3", "random_subspace_k5",
        "random_subspace_k10", "random_subspace_kauto",
        "random_direction", "energy_matched_random", "orthogonal_complement",
    }
    for beh in behaviours:
        methods = set(summary[beh])
        assert expected_methods <= methods, \
            f"{beh}: missing arms {expected_methods - methods}"
        # vanilla appears at α=0.0 (shared baseline, expanded per behaviour).
        assert "0.0" in summary[beh]["vanilla"]
        # A steered arm appears at α=1.0 with DAMAGE metrics (no annotation yet).
        cell = summary[beh]["single_direction"]["1.0"]
        assert cell["mean"] is None, "on-target mean must be None pre-annotation"
        assert "degenerate_rate" in cell
        assert "repetition_rate" in cell
        assert "mean_n_tokens" in cell
        assert cell["mean_n_tokens"] > 0
        # Chains are 40 tokens > floor(32) so they are NOT all degenerate.
        assert cell["degenerate_rate"] == 0.0

    # ── steering_results.json checkpoint exists and is non-trivial ─────────────
    results = json.loads((eval_dir / "steering_results.json").read_text())
    assert len(results) > 0
    # Shared vanilla baseline present, hoisted under behaviour="shared".
    assert any(r["behaviour"] == "shared" and r["method"] == "vanilla"
               for r in results)
    # Random-subspace replicate task-id folding (#rs<rep>) is present.
    assert any(str(r["task_id"]).find("#rs") >= 0 for r in results), \
        "random-subspace replicate task-id folding not exercised"
    # mean_abs_proj populated by the real forward hook on steered arms.
    steered = [r for r in results if r["method"] == "single_direction"]
    assert steered and all(s.get("mean_abs_proj") is not None for s in steered), \
        "forward hook did not record projection energy (integration broke)"

    # ── steering_geometry.json: per-(beh,k) bounds written ─────────────────────
    geom = json.loads((eval_dir / "steering_geometry.json").read_text())
    assert set(geom) == set(behaviours)
    for beh in behaviours:
        for k in ("1", "3", "5", "10", "auto"):
            assert k in geom[beh], f"{beh}: geometry missing k={k}"
            assert "cos_single_manifold" in geom[beh][k]
            assert "retained_energy" in geom[beh][k]
        # Energy-matched scale recorded after model calibration.
        assert "energy_matched_random_scale" in geom[beh]

    # ── provenance stamped ─────────────────────────────────────────────────────
    prov = json.loads((eval_dir / "provenance.json").read_text())
    assert "args" in prov and prov["args"]["smoke"] is True
    assert prov["args"]["skip_annotation"] is True


def test_end_to_end_on_real_vectors(monkeypatch, tmp_path):
    """Same pipeline but pointed at the REAL 1536-d vectors on disk, confirming
    the actual artifacts load + wire (shape, dtype, k-keys, auto_k) end to end.

    Uses a stub whose hidden dim is read from the real vectors so the einsum
    is dimensionally honest with the shipped vectors."""
    if not (REAL_VECTORS / "metadata.json").exists():
        pytest.skip("real steering vectors not present")
    runner = _load_runner()

    # Hidden dim from the real vectors → make the stub match (1536).
    real_single = np.load(REAL_VECTORS / "backtracking_single.npy")
    hidden = int(real_single.shape[0])

    def _real_dim_load_model(model_id, **kw):
        return StubModel(hidden=hidden, n_layers=4), StubTokenizer()

    monkeypatch.setattr(runner, "load_model", _real_dim_load_model)
    import src.chain_gen as cg
    monkeypatch.setattr(cg, "load_model", _real_dim_load_model)

    # The REAL vectors record layer=27 (out of range for a 4-layer stub). Remap
    # the loaded vectors' layer into the stub's range without touching files.
    from src import steering as steering_mod
    real_load = steering_mod.load_steering_vectors

    def _remapped_load(d):
        vecs = real_load(REAL_VECTORS)
        for beh in vecs:
            vecs[beh]["layer"] = 1  # valid index into the 4-layer stub
        return vecs

    monkeypatch.setattr(runner, "load_steering_vectors", _remapped_load)

    short = runner.MODELS["1.5b"][1]
    (REPO / "results" / "steering_vectors" / short).mkdir(parents=True, exist_ok=True)

    monkeypatch.chdir(REPO)
    monkeypatch.setattr(sys, "argv",
                        ["07_evaluate_steering.py", *_SMOKE_ARGS])
    runner.main()

    eval_dir = REPO / "results" / "eval" / short
    summary = json.loads((eval_dir / "generation_metrics.json").read_text())
    # All four real behaviours present with the manifold k-sweep arms.
    for beh in ("backtracking", "uncertainty-estimation",
                "example-testing", "adding-knowledge"):
        assert beh in summary
        # k="auto" manifold arm is labelled manifold_auto (manifold_method_label);
        # the equal-k random-subspace control for it is random_subspace_kauto.
        assert "manifold_auto" in summary[beh]
        assert "manifold_k1" in summary[beh]
        assert "single_direction" in summary[beh]


def test_annotate_path_wiring_with_mocked_annotator(monkeypatch, tmp_path):
    """Exercise the POST-GENERATION (no --skip-annotation) branch with a MOCKED
    annotate_chains, confirming aggregate + eval_summary.json wiring is correct
    without any API spend. This is the second half of the runner the smoke
    --skip-annotation path does not touch."""
    runner = _load_runner()

    behaviours = ["backtracking", "uncertainty-estimation",
                  "example-testing", "adding-knowledge"]
    synth = tmp_path / "steer"
    _write_synthetic_vectors(synth, behaviours, hidden=16, layer=1)

    from src import steering as steering_mod
    real_load = steering_mod.load_steering_vectors
    monkeypatch.setattr(runner, "load_steering_vectors",
                        lambda d: real_load(synth))

    # Provide proxy creds so the runner takes the annotate branch (it gates on
    # env presence), but the actual annotate_chains is mocked → no network call.
    monkeypatch.setenv("CLAUDE_PROXY_URL", "http://stub.invalid")
    monkeypatch.setenv("CLAUDE_PROXY_KEY", "stub-key")

    captured = {}

    def _mock_annotate_chains(results, save_path=None, dedup_keys=None,
                              model=None, **kw):
        # Echo every generated record back with a fake annotation list so
        # aggregate_results has on-target fractions to compute.
        captured["dedup_keys"] = dedup_keys
        captured["model"] = model
        out = []
        for r in results:
            out.append({**r, "annotations": [
                {"label": r["behaviour"] if r["behaviour"] != "shared"
                 else "deduction", "text": "x"},
                {"label": "deduction", "text": "y"},
            ], "annotation_complete": True})
        if save_path:
            Path(save_path).write_text(json.dumps(out))
        return out

    # annotate_chains is imported INSIDE main() via `from src.annotation import
    # annotate_chains` → patch the source module symbol.
    import src.annotation as ann_mod
    monkeypatch.setattr(ann_mod, "annotate_chains", _mock_annotate_chains)

    short = runner.MODELS["1.5b"][1]
    (REPO / "results" / "steering_vectors" / short).mkdir(parents=True, exist_ok=True)

    # NO --skip-annotation → annotate branch. Use a non-default annotator id to
    # exercise the de-circularisation knob too.
    argv = ["--smoke", "--n-random-subspaces", "1",
            "--annotator-model", "stub-non-builder-annotator"]
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(sys, "argv", ["07_evaluate_steering.py", *argv])
    monkeypatch.setattr(runner, "load_model", _fake_load_model)
    import src.chain_gen as cg
    monkeypatch.setattr(cg, "load_model", _fake_load_model)

    runner.main()

    eval_dir = REPO / "results" / "eval" / short
    # The annotator knob was threaded through.
    assert captured["model"] == "stub-non-builder-annotator"
    assert captured["dedup_keys"] == ("task_id", "behaviour", "method", "alpha")

    # eval_summary.json written (the annotated aggregation), with on-target means.
    summ = json.loads((eval_dir / "eval_summary.json").read_text())
    assert set(summ) == set(behaviours)
    cell = summ["backtracking"]["single_direction"]["1.0"]
    assert cell["mean"] is not None, "annotated cell must have an on-target mean"
    assert cell["n"] >= 1
    # Vanilla baseline expanded into every behaviour at α=0.
    assert summ["backtracking"]["vanilla"]["0.0"]["mean"] is not None


def test_smoke_is_cheap_three_tasks_two_alphas(monkeypatch, tmp_path):
    """Confirm --smoke is genuinely cheap: 3 tasks, α ∈ {0,1} only, so a real
    `--smoke --skip-annotation` is a safe $0 first GPU test (generation only,
    no annotation call)."""
    runner = _load_runner()

    # Capture exactly what run_steering_experiment is asked to do.
    seen = {}
    import src.steered_inference as si
    real_run = si.run_steering_experiment

    def _spy_run(*, tasks, alpha_values, **kw):
        seen["n_tasks"] = len(tasks)
        seen["alpha_values"] = list(alpha_values)
        return real_run(tasks=tasks, alpha_values=alpha_values, **kw)

    behaviours = ["backtracking"]
    synth = tmp_path / "steer"
    _write_synthetic_vectors(synth, behaviours, hidden=16, layer=1)
    from src import steering as steering_mod
    real_load = steering_mod.load_steering_vectors
    monkeypatch.setattr(runner, "load_steering_vectors",
                        lambda d: real_load(synth))
    monkeypatch.setattr(runner, "run_steering_experiment", _spy_run)

    short = runner.MODELS["1.5b"][1]
    (REPO / "results" / "steering_vectors" / short).mkdir(parents=True, exist_ok=True)
    _run_main(runner, _SMOKE_ARGS, monkeypatch, tmp_path, vectors_subdir=short)

    assert seen["n_tasks"] == 3, "smoke must run exactly 3 tasks"
    assert seen["alpha_values"] == [0.0, 1.0], "smoke must use α=[0,1] only"
    # No eval_summary.json (annotation skipped) → confirms $0 (no API branch).
    assert not (REPO / "results" / "eval" / short / "eval_summary.json").exists()
