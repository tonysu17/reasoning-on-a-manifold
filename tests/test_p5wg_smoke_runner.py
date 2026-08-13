"""Offline pre-spend tests for the P5 WildGuard smoke runner v1.1.

Everything here runs on the Mac with zero network, zero GPU, zero spend.
Covers the Codex review repairs: pinned-local BF16 loader interception,
cumulative job clock, stage chaining, stale-row rejection, truncation
boundaries, completion capture + diagnostic linkage, wrapper/lock binding.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import p5wg_smoke_runner as R  # noqa: E402

MANIFEST = ROOT / "results/p5_wildguard/P5WG_SMOKE_MANIFEST_2026-08-09.json"
MANIFEST_SHA = R.sha256_file(MANIFEST)


def bound_doc():
    return R.bind(MANIFEST, MANIFEST_SHA)


def start_job(out_dir: Path, ago_s: float = 0.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "JOB_START").write_text(str(time.time() - ago_s))


# ── binding ──────────────────────────────────────────────────────────────────

def test_bind_accepts_the_real_quad():
    doc = bound_doc()
    b = doc["binding"]
    assert b["runner_sha256"] == R.sha256_file(ROOT / "p5wg_smoke_runner.py")
    assert b["wrapper_sha256"] == R.sha256_file(ROOT / "runpod_p5wg_smoke.sh")
    assert b["deps_lock_sha256"] == R.sha256_file(
        ROOT / "p5wg_requirements.lock")
    assert doc["internal_sha256"] == R.manifest_internal_sha256(doc)


def test_bind_refuses_wrong_manifest_hash():
    with pytest.raises(R.SmokeRefusal, match="hash mismatch"):
        R.bind(MANIFEST, "0" * 64)


def test_bind_refuses_edited_manifest(tmp_path):
    doc = json.loads(MANIFEST.read_text())
    doc["guards"]["hard_cost_ceiling_usd"] = 999
    p = tmp_path / "m.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(R.SmokeRefusal, match="internal_sha256"):
        R.bind(p, R.sha256_file(p))


@pytest.mark.parametrize("field", ["runner_sha256", "wrapper_sha256",
                                   "deps_lock_sha256"])
def test_bind_refuses_component_drift(tmp_path, field):
    doc = json.loads(MANIFEST.read_text())
    doc["binding"][field] = "f" * 64
    doc["internal_sha256"] = R.manifest_internal_sha256(doc)
    p = tmp_path / "m.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(R.SmokeRefusal, match="authorization void"):
        R.bind(p, R.sha256_file(p))


# ── spend gate ───────────────────────────────────────────────────────────────

def test_spend_stages_refuse_without_authorised_flag():
    doc = bound_doc()
    for stage in ("stage_weights", "run"):
        with pytest.raises(R.SmokeRefusal, match="--authorised"):
            R.require_authorised(doc, False, stage)


def test_spend_stages_refuse_on_a_pending_manifest():
    doc = bound_doc()
    pending = json.loads(json.dumps(doc))
    pending["authorization"] = {"status": "pending", "authorised_by": None}
    with pytest.raises(R.SmokeRefusal, match="pending"):
        R.require_authorised(pending, True, "stage_weights")


def test_authorised_manifest_plus_flag_passes_gate():
    doc = bound_doc()
    authorised = json.loads(json.dumps(doc))
    authorised["authorization"] = {"status": "authorised",
                                   "authorised_by": "test"}
    R.require_authorised(authorised, True, "stage_weights")  # no raise
    with pytest.raises(R.SmokeRefusal):
        R.require_authorised(authorised, False, "stage_weights")


def test_cli_spend_stage_exits_3_and_writes_failed_marker(tmp_path):
    rc = R.main(["--manifest", str(MANIFEST), "--manifest-sha256", MANIFEST_SHA,
                 "--stage", "stage_weights", "--out-dir", str(tmp_path)])
    assert rc == 3
    assert (tmp_path / "SMOKE_FAILED.marker").exists()
    assert not (tmp_path / "staging.json").exists()


# ── cumulative job clock (P0-2) ──────────────────────────────────────────────

def test_spend_stage_refuses_without_job_start(tmp_path):
    with pytest.raises(R.SmokeRefusal, match="JOB_START missing"):
        R.read_job_start(tmp_path)


def test_job_start_garbage_and_future_refuse(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "JOB_START").write_text("not-a-number")
    with pytest.raises(R.SmokeRefusal, match="unreadable"):
        R.read_job_start(tmp_path)
    (tmp_path / "JOB_START").write_text(str(time.time() + 9999))
    with pytest.raises(R.SmokeRefusal, match="future"):
        R.read_job_start(tmp_path)


def test_clock_is_cumulative_across_stages(tmp_path, monkeypatch):
    """Two individually short stages cannot exceed the shared deadline in
    aggregate: the SAME JOB_START feeds both, so time spent in stage one is
    charged against stage two."""
    doc = bound_doc()
    budget = doc["guards"]["max_job_duration_s"]
    start_job(tmp_path, ago_s=budget - 50)  # 50 s left after "setup+stage 1"
    t0 = R.read_job_start(tmp_path)
    R.Guards(doc, t0).check("stage one")  # inside budget → passes
    real_now = time.time()
    monkeypatch.setattr(R.time, "time", lambda: real_now + 60)  # +60 s later
    with pytest.raises(R.SmokeRefusal, match="guard trip"):
        R.Guards(doc, t0).check("stage two")  # same t0 → aggregate exceeded


def test_expired_clock_refuses_everywhere(tmp_path):
    doc = bound_doc()
    start_job(tmp_path, ago_s=doc["guards"]["max_job_duration_s"] + 10)
    with pytest.raises(R.SmokeRefusal, match="guard trip"):
        R.Guards(doc, R.read_job_start(tmp_path)).check("any stage")


# ── stage chaining (P1-3) ────────────────────────────────────────────────────

def test_preflight_writes_chained_self_hashed_report(tmp_path):
    doc = bound_doc()
    body = R.stage_preflight(doc, tmp_path, MANIFEST_SHA)
    assert body["ok"] and body["n_items"] == 8
    rep = R.load_report(tmp_path, "preflight", doc, MANIFEST_SHA,
                        predecessor=None)
    assert rep["binding"] == R.execution_binding(doc, MANIFEST_SHA)
    assert rep["predecessor_sha256"] is None


def test_stage_weights_refuses_without_preflight(tmp_path):
    doc = json.loads(json.dumps(bound_doc()))
    doc["authorization"] = {"status": "authorised", "authorised_by": "test"}
    start_job(tmp_path)
    with pytest.raises(R.SmokeRefusal, match="preflight.json"):
        R.stage_weights(doc, tmp_path, MANIFEST_SHA, authorised=True)


def test_run_refuses_without_staging(tmp_path):
    doc = json.loads(json.dumps(bound_doc()))
    doc["authorization"] = {"status": "authorised", "authorised_by": "test"}
    start_job(tmp_path)
    R.stage_preflight(doc, tmp_path, MANIFEST_SHA)
    with pytest.raises(R.SmokeRefusal, match="staging.json"):
        R.stage_run(doc, tmp_path, MANIFEST_SHA, authorised=True)


def test_verify_refuses_without_run(tmp_path):
    doc = bound_doc()
    start_job(tmp_path)
    R.stage_preflight(doc, tmp_path, MANIFEST_SHA)
    R.save_report(tmp_path, "staging", doc, MANIFEST_SHA,
                  {"ok": True}, predecessor=tmp_path / "preflight.json")
    with pytest.raises(R.SmokeRefusal, match="run.json"):
        R.stage_verify(doc, tmp_path, MANIFEST, MANIFEST_SHA)


def test_tampered_report_breaks_the_chain(tmp_path):
    doc = bound_doc()
    R.stage_preflight(doc, tmp_path, MANIFEST_SHA)
    rep = json.loads((tmp_path / "preflight.json").read_text())
    rep["body"]["n_items"] = 9
    (tmp_path / "preflight.json").write_text(json.dumps(rep))
    with pytest.raises(R.SmokeRefusal, match="self-hash"):
        R.load_report(tmp_path, "preflight", doc, MANIFEST_SHA,
                      predecessor=None)


def test_foreign_binding_report_is_rejected(tmp_path):
    doc = bound_doc()
    R.stage_preflight(doc, tmp_path, MANIFEST_SHA)
    rep = json.loads((tmp_path / "preflight.json").read_text())
    rep["binding"]["manifest_file_sha256"] = "0" * 64
    rep["report_sha256"] = R.sha256_canonical(
        {k: v for k, v in rep.items() if k != "report_sha256"})
    (tmp_path / "preflight.json").write_text(json.dumps(rep))
    with pytest.raises(R.SmokeRefusal, match="foreign report"):
        R.load_report(tmp_path, "preflight", doc, MANIFEST_SHA,
                      predecessor=None)


# ── frozen smoke identity ────────────────────────────────────────────────────

def test_smoke_items_match_frozen_contract_and_are_arm_blind():
    doc = bound_doc()
    items = R.load_smoke_items(doc)
    assert len(items) == 8
    assert doc["smoke"]["public_items_sha256"] == (
        "24a33f952943a3edab4fefc7e553cbf7ed6964ae28137a58c64c7140aa44f95e")
    blob = json.dumps(items)
    for tok in R.IDENTITY_TOKENS:
        assert tok not in blob


# ── raw output contract ──────────────────────────────────────────────────────

VALID = {"prompt_harmfulness": "harmful", "response_harmfulness": "unharmful",
         "response_refusal": "refusal", "is_parsing_error": False}


def test_raw_validation_official_semantics():
    assert R.validate_raw(VALID)["status"] == "parsed"
    unresolved = R.validate_raw(dict(VALID, is_parsing_error=True))
    assert unresolved["status"] == "unresolved_parse_error"
    assert unresolved["raw"]["response_refusal"] is None
    with pytest.raises(R.SmokeRefusal):
        R.validate_raw(dict(VALID, response_refusal="uncertain"))
    with pytest.raises(R.SmokeRefusal):
        R.validate_raw(dict(VALID, extra_key=1))
    with pytest.raises(R.SmokeRefusal):
        R.validate_raw({k: VALID[k] for k in list(VALID)[:3]})


def test_official_na_none_fields_are_preserved_not_refused():
    na = R.validate_raw(dict(VALID, response_refusal=None))
    assert na["status"] == "parsed_partial_na"
    assert na["raw"]["response_refusal"] is None
    with pytest.raises(R.SmokeRefusal):
        R.validate_raw(dict(VALID, prompt_harmfulness=None))


# ── rows: persistence, resume, conflicts, staleness ──────────────────────────

def expected_map():
    doc = bound_doc()
    return doc, R.expected_row_invariants(doc, MANIFEST_SHA,
                                          R.load_smoke_items(doc))


def make_row(exp_one: dict, **over) -> dict:
    row = {**exp_one,
           "raw": dict(VALID), "status": "parsed", "dtype": "torch.bfloat16",
           "wildguard_input_tokens": 1200, "wildguard_context_limit": 32768,
           "generation_allowance": R.GENERATION_ALLOWANCE,
           "truncated_would_occur": False,
           "completion_sha256": R.sha256_text("COMPLETION"),
           "started_utc": "2026-08-09T00:00:00Z",
           "finished_utc": "2026-08-09T00:00:05Z", "wall_s": 5.0}
    row.update(over)
    row["row_sha256"] = R.row_sha256(row)
    return row


def test_atomic_write_then_byte_identical_resume(tmp_path):
    _, exp = expected_map()
    blind_id = sorted(exp)[0]
    p = R.persist_row(tmp_path, make_row(exp[blind_id]), exp)
    before = p.read_bytes()
    resumed = make_row(exp[blind_id], wall_s=9.9,
                       finished_utc="2026-08-09T01:00:00Z")
    R.persist_row(tmp_path, resumed, exp)
    assert p.read_bytes() == before


def test_conflicting_row_refused_not_overwritten(tmp_path):
    _, exp = expected_map()
    blind_id = sorted(exp)[0]
    p = R.persist_row(tmp_path, make_row(exp[blind_id]), exp)
    before = p.read_bytes()
    conflicting = make_row(exp[blind_id],
                           raw=dict(VALID, response_refusal="compliance"))
    with pytest.raises(R.SmokeRefusal, match="conflict"):
        R.persist_row(tmp_path, conflicting, exp)
    assert p.read_bytes() == before


def test_stale_self_consistent_row_is_rejected(tmp_path):
    """A row whose self-hash is valid but whose frozen-item invariant differs
    (e.g. from an older generations file) must NOT survive resume."""
    _, exp = expected_map()
    blind_id = sorted(exp)[0]
    stale = make_row(exp[blind_id])
    stale["generation_record_sha256"] = "a" * 64
    stale["row_sha256"] = R.row_sha256(stale)  # self-consistent again
    (tmp_path / f"{blind_id}.json").write_text(json.dumps(stale))
    with pytest.raises(R.SmokeRefusal, match="stale/foreign"):
        R.existing_valid_rows(tmp_path, exp)


def test_unknown_blind_id_and_stray_files_are_rejected(tmp_path):
    _, exp = expected_map()
    with pytest.raises(R.SmokeRefusal, match="not in the frozen"):
        R.persist_row(tmp_path, make_row(dict(
            list(exp.values())[0], blind_id="p5wg-ffffffffffffffffffff")), exp)
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(R.SmokeRefusal, match="unexpected file"):
        R.existing_valid_rows(tmp_path, exp)
    (tmp_path / "notes.txt").unlink()
    (tmp_path / "p5wg-x.json.tmp").write_text("x")
    with pytest.raises(R.SmokeRefusal, match="temporary file"):
        R.existing_valid_rows(tmp_path, exp)


def test_identity_token_and_schema_deviation_refused(tmp_path):
    _, exp = expected_map()
    blind_id = sorted(exp)[0]
    with pytest.raises(R.SmokeRefusal, match="row keys"):
        row = make_row(exp[blind_id])
        row["checkpoint_role"] = "base_r1"
        row["row_sha256"] = R.row_sha256(row)
        R.persist_row(tmp_path, row, exp)
    with pytest.raises(R.SmokeRefusal, match="identity"):
        R.persist_row(tmp_path,
                      make_row(exp[blind_id],
                               dtype="bf16 owned_fullft_safety_s42"), exp)


# ── loader interception (P0-1) ───────────────────────────────────────────────

class _FakeParam:
    def __init__(self, dtype):
        self.dtype = dtype

    def is_floating_point(self):
        return True


def _install_fake_stack(monkeypatch, local_dir, captured):
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = object()

    class FakeModel:
        def __init__(self, path):
            self.config = types.SimpleNamespace(_name_or_path=str(path))

        def to(self, device):
            captured["device"] = device
            return self

        def parameters(self):
            return [_FakeParam(fake_torch.bfloat16)]

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kw):
            captured["model_path"] = path
            captured["model_kwargs"] = kw
            return FakeModel(path)

    class FakeAutoTok:
        @staticmethod
        def from_pretrained(path, **kw):
            captured["tok_path"] = path
            captured["tok_kwargs"] = kw
            return object()

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoModelForCausalLM = FakeAutoModel
    fake_tf.AutoTokenizer = FakeAutoTok

    fake_ww = types.ModuleType("wildguard.wildguard")
    fake_ww.WILDGUARD_INPUT_FORMAT = "U:{prompt}|A:{response}"
    fake_ww.load_hf_model = lambda name, device: (_ for _ in ()).throw(
        AssertionError("upstream loader must be patched away"))
    fake_ww.load_tokenizer = lambda name, use_fast: (_ for _ in ()).throw(
        AssertionError("upstream loader must be patched away"))

    class FakeWildGuardHF:
        def __init__(self, batch_size, device, ephemeral_model):
            # mimic upstream: resolve loaders from the module namespace
            self.model = fake_ww.load_hf_model("allenai/wildguard", device)
            self.tokenizer = fake_ww.load_tokenizer("allenai/wildguard",
                                                    use_fast=False)

    fake_ww.WildGuardHF = FakeWildGuardHF
    fake_wg = types.ModuleType("wildguard")
    fake_wg.wildguard = fake_ww
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "wildguard", fake_wg)
    monkeypatch.setitem(sys.modules, "wildguard.wildguard", fake_ww)
    return fake_torch


def test_loader_uses_exact_local_path_bf16_and_local_files_only(
        tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text("{}")
    captured: dict = {}
    fake_torch = _install_fake_stack(monkeypatch, tmp_path, captured)
    wg, info = R.load_patched_classifier(str(tmp_path), device="cuda")
    assert captured["model_path"] == str(tmp_path)
    assert captured["model_kwargs"]["local_files_only"] is True
    assert captured["model_kwargs"]["torch_dtype"] is fake_torch.bfloat16
    assert captured["tok_path"] == str(tmp_path)
    assert captured["tok_kwargs"]["local_files_only"] is True
    assert captured["tok_kwargs"]["use_fast"] is False
    assert captured["device"] == "cuda"
    assert info["resolved_local_path"] == str(tmp_path)
    assert info["dtype"] == "torch.bfloat16"


def test_loader_refuses_non_bf16_parameters(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    captured: dict = {}
    fake_torch = _install_fake_stack(monkeypatch, tmp_path, captured)
    ww = sys.modules["wildguard.wildguard"]
    orig_cls = ww.WildGuardHF

    class FP32Model:
        def __init__(self):
            self.config = types.SimpleNamespace(_name_or_path=str(tmp_path))

        def parameters(self):
            return [_FakeParam("torch.float32")]

    class FP32WildGuardHF(orig_cls):
        def __init__(self, batch_size, device, ephemeral_model):
            self.model = FP32Model()
            self.tokenizer = object()

    ww.WildGuardHF = FP32WildGuardHF
    with pytest.raises(R.SmokeRefusal, match="non-BF16"):
        R.load_patched_classifier(str(tmp_path))


# ── truncation measurement (P1-4) ────────────────────────────────────────────

class _FakeTok:
    def __init__(self, n, model_max):
        self.n = n
        self.model_max_length = model_max

    def __call__(self, text, **kw):
        assert kw.get("truncation") is False
        return {"input_ids": list(range(self.n))}


def _fake_ww_format(monkeypatch):
    fake_ww = types.ModuleType("wildguard.wildguard")
    fake_ww.WILDGUARD_INPUT_FORMAT = "U:{prompt}|A:{response}"
    fake_wg = types.ModuleType("wildguard")
    fake_wg.wildguard = fake_ww
    monkeypatch.setitem(sys.modules, "wildguard", fake_wg)
    monkeypatch.setitem(sys.modules, "wildguard.wildguard", fake_ww)


def test_truncation_boundaries(monkeypatch):
    _fake_ww_format(monkeypatch)
    wg = types.SimpleNamespace(tokenizer=_FakeTok(1000, 32768))
    m = R.measure_input(wg, "p", "r")
    assert m["truncated_would_occur"] is False
    assert m["wildguard_context_limit"] == 32768
    wg = types.SimpleNamespace(tokenizer=_FakeTok(40000, 32768))
    m = R.measure_input(wg, "p", "r")
    assert m["truncated_would_occur"] is True
    # sentinel model_max_length (no practical limit)
    wg = types.SimpleNamespace(tokenizer=_FakeTok(40000, int(1e30)))
    m = R.measure_input(wg, "p", "r")
    assert m["wildguard_context_limit"] is None
    assert m["truncated_would_occur"] is False


# ── completion capture + diagnostic linkage (P1-5) ───────────────────────────

class _FakeClassifier:
    def __init__(self, decoded="Harmful request: yes\nResponse refusal: "
                               "yes\nHarmful response: no"):
        self.decoded = decoded

    def parse_model_generation_to_output(self, text):
        return {"parsed_from": text}

    def classify(self, items):
        self.parse_model_generation_to_output(self.decoded)
        return [dict(VALID)]


def test_classify_one_captures_decoded_completion():
    wg = _FakeClassifier()
    original = wg.parse_model_generation_to_output
    raw, completion = R.classify_one(wg, {"prompt": "p", "response": "r"})
    assert raw == VALID
    assert completion == wg.decoded
    assert wg.parse_model_generation_to_output == original  # restored


def make_diag(blind_id, completion="COMPLETION", **over):
    rec = {"schema_version": R.DIAG_SCHEMA, "blind_id": blind_id,
           "decoded_completion": completion,
           "completion_sha256": R.sha256_text(completion),
           "manifest_file_sha256": MANIFEST_SHA,
           "runner_version": R.RUNNER_VERSION,
           "utc": "2026-08-09T00:00:00Z"}
    rec.update(over)
    rec["diag_sha256"] = R.sha256_canonical(
        {k: v for k, v in rec.items() if k != "diag_sha256"})
    return rec


def test_diag_linkage_roundtrip_and_breakage(tmp_path):
    _, exp = expected_map()
    blind_id = sorted(exp)[0]
    rows_dir, diag_dir = tmp_path / "rows", tmp_path / "diag"
    row = make_row(exp[blind_id])
    R.persist_row(rows_dir, row, exp)
    R.persist_diag(diag_dir, make_diag(blind_id))
    rows = R.existing_valid_rows(rows_dir, exp)
    R.check_diag_linkage(rows, diag_dir)  # no raise
    # break the link
    (diag_dir / f"{blind_id}.json").write_text(
        json.dumps(make_diag(blind_id, completion="DIFFERENT")))
    with pytest.raises(R.SmokeRefusal, match="hash link broken"):
        R.check_diag_linkage(rows, diag_dir)


def test_input_too_long_rows_have_no_diag(tmp_path):
    _, exp = expected_map()
    blind_id = sorted(exp)[0]
    rows_dir, diag_dir = tmp_path / "rows", tmp_path / "diag"
    row = make_row(exp[blind_id], status="unresolved_input_too_long",
                   raw={"prompt_harmfulness": None,
                        "response_harmfulness": None,
                        "response_refusal": None, "is_parsing_error": False},
                   completion_sha256=None, truncated_would_occur=True)
    R.persist_row(rows_dir, row, exp)
    rows = R.existing_valid_rows(rows_dir, exp)
    R.check_diag_linkage(rows, diag_dir)  # absent diag dir is correct here
    R.persist_diag(diag_dir, make_diag(blind_id))
    with pytest.raises(R.SmokeRefusal, match="input-too-long"):
        R.check_diag_linkage(rows, diag_dir)


def test_parse_error_row_still_links_diag(tmp_path):
    _, exp = expected_map()
    blind_id = sorted(exp)[0]
    rows_dir, diag_dir = tmp_path / "rows", tmp_path / "diag"
    completion = "GARBLED OUTPUT"
    row = make_row(exp[blind_id], status="unresolved_parse_error",
                   raw={"prompt_harmfulness": None,
                        "response_harmfulness": None,
                        "response_refusal": None, "is_parsing_error": True},
                   completion_sha256=R.sha256_text(completion))
    R.persist_row(rows_dir, row, exp)
    R.persist_diag(diag_dir, make_diag(blind_id, completion=completion))
    R.check_diag_linkage(R.existing_valid_rows(rows_dir, exp), diag_dir)


def test_diag_identity_token_refused(tmp_path):
    with pytest.raises(R.SmokeRefusal, match="identity"):
        R.persist_diag(tmp_path, make_diag(
            "p5wg-08fdaa94c11f1eb259d8",
            completion="text mentioning owned_fullft_safety_s42"))


# ── verify never aggregates ──────────────────────────────────────────────────

def test_runner_contains_no_aggregation_machinery():
    src = (ROOT / "p5wg_smoke_runner.py").read_text()
    for banned in ("mean(", "groupby", "effect_size", "stratum_rate",
                   "import numpy", "import statistics"):
        assert banned not in src
