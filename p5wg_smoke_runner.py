#!/usr/bin/env python3
"""P5 WildGuard 8-row smoke — hash-bound, spend-gated pod runner.

Version: p5wg-smoke-runner/1.1.0 (2026-08-09)

v1.1 implements the Codex review repairs
(`.codex/out/CODEX_TO_CLAUDE_P5WG_SMOKE_REVIEW_2026-08-09.md`):
  P0-1  model + tokenizer load from the VERIFIED local snapshot with
        local_files_only=True and torch_dtype=bfloat16 AT CONSTRUCTION —
        the official wildguard loader is patched at module level (it would
        otherwise re-resolve "allenai/wildguard" in FP32); the package is
        therefore official code with a PATCHED LOADER, not byte-identical
        upstream execution.  Post-load assertions refuse non-BF16 params or
        a model resolved outside the snapshot.
  P0-2  ONE cumulative job clock: the wrapper writes JOB_START before
        apt/pip; every stage (staging, run, verify) enforces the same
        deadline + dollar ceiling cumulatively.  This bounds JOB runtime;
        billed pod lifetime still ends at console termination after the
        verified pull (operator boundary, stated in the manifest).
  P1-3  chained, self-hashed stage reports (preflight -> staging -> run ->
        verify) carrying runner/wrapper/manifest/frozen-input hashes and the
        predecessor report hash; resume/verify re-derive every row's
        invariants from the CURRENT frozen items and reject stale or foreign
        rows even when self-hash-consistent.
  P1-4  classifier-input truncation is measured per row (untruncated
        WildGuard-token count vs tokenizer limit + 128-token generation
        allowance); an over-limit row persists as unresolved_input_too_long
        and is NEVER silently classified truncated.
  P1-5  the decoded classifier completion is captured pre-parse and
        persisted as an arm-blind diagnostic record, linked from the public
        row by SHA-256; raw fields stay official; parse errors stay missing.
  P1-6  the wrapper and the exact dependency lock are hash-bound into the
        manifest; all installs use locked wheels only.

Binding model (no circular hash): Tony authorizes the exact
(runner_sha256, manifest file sha256) pair; the manifest additionally pins
the wrapper and dependency-lock hashes.  Every invocation verifies all of
manifest-file hash, manifest internal hash, runner self-hash, wrapper hash,
and lock hash.  Any edit anywhere voids the authorization.

Spend gate: stage_weights and run require BOTH --authorised and
manifest["authorization"]["status"] == "authorised".  This file ships with
the manifest saying "pending", so the runner as returned cannot spend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUNNER_VERSION = "p5wg-smoke-runner/1.1.0"
BACKEND = "wildguard-hf-1.0.1-local-bf16-loader-patch-batch1"
GENERATION_ALLOWANCE = 128  # max_new_tokens in the pinned WildGuardHF source
ROOT = Path(__file__).resolve().parent
CODEX_OUT = ROOT / ".codex" / "out"

ALLOWED_PROMPT_HARMFULNESS = {"harmful", "unharmful"}
ALLOWED_RESPONSE_HARMFULNESS = {"harmful", "unharmful"}
ALLOWED_RESPONSE_REFUSAL = {"refusal", "compliance"}
RAW_KEYS = {"prompt_harmfulness", "response_harmfulness", "response_refusal",
            "is_parsing_error"}

ROW_SCHEMA = "p5wg-smoke-row-2"
ROW_KEYS = {"schema_version", "blind_id", "generation_record_sha256",
            "prompt_sha256", "response_sha256", "raw", "status",
            "model_revision", "backend", "dtype", "runner_version",
            "manifest_file_sha256", "wildguard_input_tokens",
            "wildguard_context_limit", "generation_allowance",
            "truncated_would_occur", "completion_sha256",
            "started_utc", "finished_utc", "wall_s", "row_sha256"}
# Invariants re-derived from the CURRENT frozen items on resume/verify.
ROW_INVARIANTS = ("schema_version", "blind_id", "generation_record_sha256",
                  "prompt_sha256", "response_sha256", "model_revision",
                  "backend", "runner_version", "manifest_file_sha256")
ROW_TIMING = {"started_utc", "finished_utc", "wall_s", "row_sha256"}

DIAG_SCHEMA = "p5wg-smoke-diag-1"
DIAG_KEYS = {"schema_version", "blind_id", "decoded_completion",
             "completion_sha256", "manifest_file_sha256", "runner_version",
             "utc", "diag_sha256"}

IDENTITY_TOKENS = ("checkpoint_role", "base_r1", "public_star1",
                   "owned_fullft_safety_s42", "owned_fullft_control_s42")

VLLM_STUB_SOURCE = '''"""Inert vllm stub installed by p5wg_smoke_runner (HF backend only)."""


class _Refuse:
    def __init__(self, *a, **k):
        raise RuntimeError("vllm path is disabled in the P5 WildGuard smoke")


class LLM(_Refuse):
    pass


class SamplingParams(_Refuse):
    pass


def destroy_model_parallel(*a, **k):
    raise RuntimeError("vllm path is disabled in the P5 WildGuard smoke")
'''


class SmokeRefusal(RuntimeError):
    """Any contract violation → refuse loudly, never degrade silently."""


# ── hashing ──────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_canonical(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def manifest_internal_sha256(doc: dict) -> str:
    return sha256_canonical({k: v for k, v in doc.items()
                             if k != "internal_sha256"})


# ── binding ──────────────────────────────────────────────────────────────────

def bind(manifest_path: Path, manifest_sha256: str,
         runner_path: Path | None = None) -> dict:
    """Verify the (runner, wrapper, lock, manifest) quad; return the manifest."""
    if not manifest_path.exists():
        raise SmokeRefusal(f"manifest missing: {manifest_path}")
    actual = sha256_file(manifest_path)
    if actual != manifest_sha256:
        raise SmokeRefusal(
            f"manifest file hash mismatch: expected {manifest_sha256}, "
            f"got {actual} — authorization void")
    doc = json.loads(manifest_path.read_text())
    if doc.get("internal_sha256") != manifest_internal_sha256(doc):
        raise SmokeRefusal("manifest internal_sha256 inconsistent")
    b = doc["binding"]
    me = sha256_file(runner_path or Path(__file__).resolve())
    if me != b["runner_sha256"]:
        raise SmokeRefusal(f"runner hash {me} != manifest-bound "
                           f"{b['runner_sha256']} — authorization void")
    for key, label in (("wrapper", "wrapper"), ("deps_lock", "deps lock")):
        p = ROOT / b[f"{key}_path"]
        if not p.exists():
            raise SmokeRefusal(f"{label} missing: {p}")
        got = sha256_file(p)
        if got != b[f"{key}_sha256"]:
            raise SmokeRefusal(f"{label} hash {got} != manifest-bound "
                               f"{b[f'{key}_sha256']} — authorization void")
    return doc


def require_authorised(doc: dict, authorised_flag: bool, stage: str) -> None:
    if not authorised_flag:
        raise SmokeRefusal(f"stage {stage} spends; pass --authorised only "
                           "after Tony authorizes the exact hash pair")
    if doc.get("authorization", {}).get("status") != "authorised":
        raise SmokeRefusal(
            f"stage {stage}: manifest authorization.status is "
            f"{doc.get('authorization', {}).get('status')!r}, not 'authorised' "
            "— Tony must issue an authorized manifest revision (new hash pair)")


def execution_binding(doc: dict, manifest_sha256: str) -> dict:
    return {"runner_version": RUNNER_VERSION,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "wrapper_sha256": doc["binding"]["wrapper_sha256"],
            "deps_lock_sha256": doc["binding"]["deps_lock_sha256"],
            "manifest_file_sha256": manifest_sha256,
            "manifest_internal_sha256": doc["internal_sha256"],
            "generations_sha256": doc["frozen_inputs"]["generations"]["sha256"],
            "safety_prompt_manifest_sha256":
                doc["frozen_inputs"]["safety_prompt_manifest"]["sha256"],
            "public_items_sha256": doc["smoke"]["public_items_sha256"]}


# ── single cumulative job clock ──────────────────────────────────────────────

def read_job_start(out_dir: Path) -> float:
    """JOB_START is written ONCE by the wrapper before apt/pip; every stage
    shares it. Missing/garbled/future values refuse."""
    p = out_dir / "JOB_START"
    if not p.exists():
        raise SmokeRefusal("JOB_START missing — stages must run under the "
                           "wrapper's single job clock")
    try:
        t0 = float(p.read_text().strip())
    except ValueError:
        raise SmokeRefusal("JOB_START unreadable")
    if t0 > time.time() + 120:
        raise SmokeRefusal("JOB_START is in the future")
    return t0


class Guards:
    """Cumulative wall-clock + dollar ceiling from the shared JOB_START."""

    def __init__(self, doc: dict, t0: float):
        g = doc["guards"]
        self.t0 = t0
        self.max_wall_s = float(g["max_job_duration_s"])
        self.rate = float(g["assumed_rate_usd_per_hour"])
        self.max_usd = float(g["hard_cost_ceiling_usd"])

    def check(self, where: str) -> None:
        elapsed = time.time() - self.t0
        cost = elapsed / 3600.0 * self.rate
        if elapsed > self.max_wall_s:
            raise SmokeRefusal(f"guard trip at {where}: job elapsed "
                               f"{elapsed:.0f}s > max {self.max_wall_s:.0f}s")
        if cost > self.max_usd:
            raise SmokeRefusal(f"guard trip at {where}: est job cost "
                               f"${cost:.2f} > ceiling ${self.max_usd:.2f}")


# ── chained, self-hashed stage reports ───────────────────────────────────────

def save_report(out_dir: Path, stage: str, doc: dict, manifest_sha256: str,
                body: dict, predecessor: Path | None) -> Path:
    rep = {"stage": stage,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "binding": execution_binding(doc, manifest_sha256),
           "predecessor_sha256": sha256_file(predecessor) if predecessor else None,
           "body": body}
    rep["report_sha256"] = sha256_canonical(
        {k: v for k, v in rep.items() if k != "report_sha256"})
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stage}.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(rep, sort_keys=True, indent=1, ensure_ascii=False))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def load_report(out_dir: Path, stage: str, doc: dict, manifest_sha256: str,
                predecessor: Path | None) -> dict:
    path = out_dir / f"{stage}.json"
    if not path.exists():
        raise SmokeRefusal(f"required predecessor report missing: {path.name} "
                           "— stages must run in order")
    rep = json.loads(path.read_text())
    if rep.get("report_sha256") != sha256_canonical(
            {k: v for k, v in rep.items() if k != "report_sha256"}):
        raise SmokeRefusal(f"{path.name}: report self-hash inconsistent")
    if rep.get("stage") != stage:
        raise SmokeRefusal(f"{path.name}: stage mismatch")
    if rep.get("binding") != execution_binding(doc, manifest_sha256):
        raise SmokeRefusal(f"{path.name}: execution binding differs from the "
                           "current runner/wrapper/manifest — foreign report")
    want_pred = sha256_file(predecessor) if predecessor else None
    if rep.get("predecessor_sha256") != want_pred:
        raise SmokeRefusal(f"{path.name}: predecessor hash chain broken")
    return rep


# ── frozen smoke rows (via Codex's dry protocol, read-only) ──────────────────

def load_smoke_items(doc: dict) -> list[dict]:
    sys.path.insert(0, str(CODEX_OUT))
    try:
        import p5_wildguard_local_protocol as protocol  # noqa: PLC0415
    finally:
        sys.path.pop(0)
    private = protocol.build_private_inputs()
    smoke_public = protocol.public_model_items(protocol.smoke_selection(private))
    frozen = doc["smoke"]
    if sha256_canonical(smoke_public) != frozen["public_items_sha256"]:
        raise SmokeRefusal("smoke public-items hash != frozen manifest value")
    ids = sorted(r["blind_id"] for r in smoke_public)
    if ids != sorted(frozen["blind_ids"]) or len(ids) != 8:
        raise SmokeRefusal("smoke blind-id set != frozen manifest value")
    for row in smoke_public:
        for key in row:
            if any(tok in key for tok in IDENTITY_TOKENS):
                raise SmokeRefusal("identity token leaked into public item key")
    return smoke_public


def expected_row_invariants(doc: dict, manifest_sha256: str,
                            items: list[dict]) -> dict[str, dict]:
    """Per blind_id, the row fields that are fully determined by the frozen
    item + the bound execution — used to reject stale/foreign rows."""
    return {item["blind_id"]: {
        "schema_version": ROW_SCHEMA,
        "blind_id": item["blind_id"],
        "generation_record_sha256": item["generation_record_sha256"],
        "prompt_sha256": sha256_text(item["prompt"]),
        "response_sha256": sha256_text(item["response"]),
        "model_revision": doc["model"]["model_revision"],
        "backend": BACKEND,
        "runner_version": RUNNER_VERSION,
        "manifest_file_sha256": manifest_sha256,
    } for item in items}


# ── raw output validation (official wildguard==1.0.1 semantics) ──────────────

def validate_raw(raw: dict) -> dict:
    """The pinned parser can return response_* fields as None with
    is_parsing_error=False (its explicit "N/A" branches); that None IS the
    official raw value and persists verbatim (status parsed_partial_na).
    prompt_harmfulness has no N/A branch, so None there refuses."""
    if set(raw) != RAW_KEYS:
        raise SmokeRefusal(f"WildGuard raw keys {sorted(raw)} differ from the "
                           f"frozen contract {sorted(RAW_KEYS)}")
    if not isinstance(raw["is_parsing_error"], bool):
        raise SmokeRefusal("is_parsing_error must be boolean")
    if raw["is_parsing_error"]:
        return {"status": "unresolved_parse_error",
                "raw": {"prompt_harmfulness": None,
                        "response_harmfulness": None,
                        "response_refusal": None,
                        "is_parsing_error": True}}
    if raw["prompt_harmfulness"] not in ALLOWED_PROMPT_HARMFULNESS:
        raise SmokeRefusal("invalid prompt_harmfulness")
    for key, allowed in (("response_harmfulness", ALLOWED_RESPONSE_HARMFULNESS),
                         ("response_refusal", ALLOWED_RESPONSE_REFUSAL)):
        if raw[key] is not None and raw[key] not in allowed:
            raise SmokeRefusal(f"invalid {key}")
    complete = all(raw[k] is not None for k in
                   ("response_harmfulness", "response_refusal"))
    return {"status": "parsed" if complete else "parsed_partial_na",
            "raw": dict(raw)}


# ── atomic persistence / resume / conflict refusal ───────────────────────────

def row_sha256(row: dict) -> str:
    return sha256_canonical({k: v for k, v in row.items() if k != "row_sha256"})


def _atomic_write(path: Path, blob: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _validate_row(row: dict, expected: dict[str, dict]) -> None:
    if set(row) != ROW_KEYS:
        raise SmokeRefusal(f"row keys {sorted(set(row) ^ ROW_KEYS)} deviate "
                           "from frozen row schema")
    if row["row_sha256"] != row_sha256(row):
        raise SmokeRefusal("row self-hash inconsistent")
    exp = expected.get(row["blind_id"])
    if exp is None:
        raise SmokeRefusal(f"row {row['blind_id']} is not in the frozen "
                           "smoke selection")
    for key, want in exp.items():
        if row[key] != want:
            raise SmokeRefusal(
                f"stale/foreign row {row['blind_id']}: {key} = {row[key]!r} "
                f"but the current frozen item requires {want!r}")
    blob = json.dumps(row, sort_keys=True, ensure_ascii=False)
    for tok in IDENTITY_TOKENS:
        if tok in blob:
            raise SmokeRefusal("checkpoint identity token in public row")


def persist_row(rows_dir: Path, row: dict, expected: dict[str, dict]) -> Path:
    _validate_row(row, expected)
    rows_dir.mkdir(parents=True, exist_ok=True)
    final = rows_dir / f"{row['blind_id']}.json"
    if final.exists():
        existing = json.loads(final.read_text())
        stable = ROW_KEYS - ROW_TIMING
        if {k: existing.get(k) for k in stable} == {k: row[k] for k in stable}:
            return final  # byte-identical result → resume accepts, no rewrite
        raise SmokeRefusal(f"conflict: {final.name} exists with different "
                           "content — refusing to overwrite")
    _atomic_write(final, json.dumps(row, sort_keys=True, indent=1,
                                    ensure_ascii=False))
    return final


def existing_valid_rows(rows_dir: Path,
                        expected: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not rows_dir.exists():
        return out
    for p in sorted(rows_dir.iterdir()):
        if p.name.endswith(".tmp"):
            raise SmokeRefusal(f"stray temporary file in rows dir: {p.name}")
        if not (p.name.startswith("p5wg-") and p.name.endswith(".json")):
            raise SmokeRefusal(f"unexpected file in rows dir: {p.name}")
        row = json.loads(p.read_text())
        try:
            _validate_row(row, expected)
        except SmokeRefusal as e:
            raise SmokeRefusal(f"conflict resuming over {p.name}: {e}")
        if row["blind_id"] in out:
            raise SmokeRefusal(f"duplicate blind_id {row['blind_id']}")
        out[row["blind_id"]] = row
    return out


def persist_diag(diag_dir: Path, rec: dict) -> Path:
    if set(rec) != DIAG_KEYS:
        raise SmokeRefusal("diagnostic record keys deviate from schema")
    if rec["diag_sha256"] != sha256_canonical(
            {k: v for k, v in rec.items() if k != "diag_sha256"}):
        raise SmokeRefusal("diagnostic record self-hash inconsistent")
    if rec["completion_sha256"] != sha256_text(rec["decoded_completion"]):
        raise SmokeRefusal("diagnostic completion hash mismatch")
    for tok in IDENTITY_TOKENS:
        if tok in json.dumps(rec, ensure_ascii=False):
            raise SmokeRefusal("identity token in diagnostic record")
    diag_dir.mkdir(parents=True, exist_ok=True)
    final = diag_dir / f"{rec['blind_id']}.json"
    if final.exists():
        existing = json.loads(final.read_text())
        stable = DIAG_KEYS - {"utc", "diag_sha256"}
        if {k: existing.get(k) for k in stable} == {k: rec[k] for k in stable}:
            return final
        raise SmokeRefusal(f"conflict: {final.name} exists with different "
                           "diagnostic content")
    _atomic_write(final, json.dumps(rec, sort_keys=True, indent=1,
                                    ensure_ascii=False))
    return final


def check_diag_linkage(rows: dict[str, dict], diag_dir: Path) -> None:
    """Every classified row links its diag record by completion hash; an
    input-too-long row has neither."""
    for blind_id, row in rows.items():
        diag_path = diag_dir / f"{blind_id}.json"
        if row["status"] == "unresolved_input_too_long":
            if row["completion_sha256"] is not None or diag_path.exists():
                raise SmokeRefusal(f"{blind_id}: input-too-long row must have "
                                   "no completion/diagnostic")
            continue
        if row["completion_sha256"] is None or not diag_path.exists():
            raise SmokeRefusal(f"{blind_id}: classified row missing "
                               "diagnostic linkage")
        rec = json.loads(diag_path.read_text())
        if rec["diag_sha256"] != sha256_canonical(
                {k: v for k, v in rec.items() if k != "diag_sha256"}):
            raise SmokeRefusal(f"{blind_id}: diagnostic self-hash inconsistent")
        if rec["completion_sha256"] != row["completion_sha256"] or \
                sha256_text(rec["decoded_completion"]) != row["completion_sha256"]:
            raise SmokeRefusal(f"{blind_id}: diagnostic/row hash link broken")


# ── patched official loader (P0-1) ───────────────────────────────────────────

def load_patched_classifier(local_snapshot_path: str, device: str = "cuda"):
    """Official WildGuardHF with its module-level loader patched to load the
    VERIFIED local snapshot, BF16 at construction, local_files_only=True.
    Returns (classifier, load_info)."""
    import torch  # noqa: PLC0415
    from transformers import (AutoModelForCausalLM,  # noqa: PLC0415
                              AutoTokenizer)
    import wildguard.wildguard as W  # noqa: PLC0415

    local = str(local_snapshot_path)

    def patched_load_hf_model(name, device_):
        model = AutoModelForCausalLM.from_pretrained(
            local, torch_dtype=torch.bfloat16, local_files_only=True)
        return model.to(device_)

    def patched_load_tokenizer(name, use_fast):
        return AutoTokenizer.from_pretrained(
            local, use_fast=use_fast, local_files_only=True)

    W.load_hf_model = patched_load_hf_model
    W.load_tokenizer = patched_load_tokenizer
    wg = W.WildGuardHF(batch_size=1, device=device, ephemeral_model=False)

    bad = [p.dtype for p in wg.model.parameters()
           if p.is_floating_point() and p.dtype != torch.bfloat16]
    if bad:
        raise SmokeRefusal(f"non-BF16 floating parameters after load: "
                           f"{sorted(set(map(str, bad)))}")
    resolved = str(getattr(wg.model.config, "_name_or_path", ""))
    if resolved != local:
        raise SmokeRefusal(f"model resolved outside the verified snapshot: "
                           f"{resolved!r} != {local!r}")
    info = {"resolved_local_path": resolved,
            "dtype": "torch.bfloat16",
            "config_sha256": sha256_file(Path(local) / "config.json"),
            "tokenizer_file_sha256": {
                name: sha256_file(Path(local) / name)
                for name in ("tokenizer.model", "tokenizer_config.json",
                             "special_tokens_map.json")
                if (Path(local) / name).exists()}}
    return wg, info


def measure_input(wg, prompt: str, response: str) -> dict:
    """Untruncated WildGuard-token measurement of the exact official
    formatted input (P1-4)."""
    import wildguard.wildguard as W  # noqa: PLC0415
    formatted = W.WILDGUARD_INPUT_FORMAT.format(prompt=prompt,
                                                response=response)
    ids = wg.tokenizer(formatted, add_special_tokens=True,
                       truncation=False)["input_ids"]
    n = len(ids)
    raw_limit = int(wg.tokenizer.model_max_length)
    limit = raw_limit if raw_limit < 10**9 else None  # sentinel = unlimited
    would_truncate = limit is not None and n > limit
    return {"wildguard_input_tokens": n,
            "wildguard_context_limit": limit,
            "generation_allowance": GENERATION_ALLOWANCE,
            "truncated_would_occur": would_truncate}


def classify_one(wg, item: dict) -> tuple[dict, str]:
    """Classify one item, capturing the decoded completion pre-parse (P1-5)."""
    captured: dict = {}
    original_parse = wg.parse_model_generation_to_output

    def capturing_parse(output_text):
        captured["text"] = output_text
        return original_parse(output_text)

    wg.parse_model_generation_to_output = capturing_parse
    try:
        results = wg.classify([{"prompt": item["prompt"],
                                "response": item["response"]}])
    finally:
        wg.parse_model_generation_to_output = original_parse
    if len(results) != 1:
        raise SmokeRefusal("classifier returned != 1 result for 1 item")
    if "text" not in captured:
        raise SmokeRefusal("decoded completion was not captured")
    return {k: results[0][k] for k in RAW_KEYS}, captured["text"]


# ── environment fingerprint ──────────────────────────────────────────────────

def environment_report() -> dict:
    rep = {"python": sys.version.split()[0],
           "platform": sys.platform,
           "runpod_image_tag": os.environ.get("RUNPOD_IMAGE",
                                              os.environ.get("TEMPLATE_IMAGE")),
           "runner_version": RUNNER_VERSION}
    for mod in ("torch", "transformers", "tokenizers", "safetensors",
                "wildguard", "huggingface_hub", "tqdm"):
        try:
            m = __import__(mod)
            rep[mod] = getattr(m, "__version__", "present")
        except Exception:
            rep[mod] = None
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            rep["gpu"] = torch.cuda.get_device_name(0)
            rep["cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        rep["driver"] = smi.stdout.strip() or None
    except Exception:
        rep["driver"] = None
    rep["fingerprint_sha256"] = sha256_canonical(rep)
    return rep


# ── stages ───────────────────────────────────────────────────────────────────

def stage_preflight(doc: dict, out_dir: Path, manifest_sha256: str) -> dict:
    items = load_smoke_items(doc)
    body = {"ok": True, "n_items": len(items), "spend": None}
    save_report(out_dir, "preflight", doc, manifest_sha256, body,
                predecessor=None)
    return body


def stage_weights(doc: dict, out_dir: Path, manifest_sha256: str,
                  authorised: bool) -> dict:
    require_authorised(doc, authorised, "stage_weights")
    guards = Guards(doc, read_job_start(out_dir))
    load_report(out_dir, "preflight", doc, manifest_sha256, predecessor=None)
    guards.check("stage_weights entry")

    model = doc["model"]
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    local = Path(snapshot_download(
        model["model_id"], revision=model["model_revision"],
        allow_patterns=model["allow_patterns"]))
    guards.check("post-download")
    staged, mismatches = [], []
    for spec in model["snapshot_files"]:
        p = local / spec["rfilename"]
        if not p.exists():
            mismatches.append(f"missing {spec['rfilename']}")
            continue
        digest = sha256_file(p)
        size = p.stat().st_size
        if spec.get("sha256") and digest != spec["sha256"]:
            mismatches.append(f"sha256 mismatch {spec['rfilename']}")
        if spec.get("size") and size != spec["size"]:
            mismatches.append(f"size mismatch {spec['rfilename']}")
        staged.append({"rfilename": spec["rfilename"], "size": size,
                       "sha256": digest})
    if mismatches:
        raise SmokeRefusal("staged snapshot fails pinned manifest: "
                           + "; ".join(mismatches))
    guards.check("post-verify")

    # Locked wheels were installed by the wrapper; write the inert vllm stub
    # and prove the official package imports without vllm.
    import sysconfig  # noqa: PLC0415
    site = Path(sysconfig.get_paths()["purelib"])
    (site / "vllm.py").write_text(VLLM_STUB_SOURCE)
    import wildguard  # noqa: F401,PLC0415

    body = {"ok": True,
            "local_snapshot_path": str(local),
            "local_snapshot_manifest": staged,
            "local_snapshot_manifest_sha256": sha256_canonical(staged),
            "vllm_stub_sha256": sha256_text(VLLM_STUB_SOURCE),
            "environment": environment_report()}
    save_report(out_dir, "staging", doc, manifest_sha256,
                body, predecessor=out_dir / "preflight.json")
    return body


def stage_run(doc: dict, out_dir: Path, manifest_sha256: str,
              authorised: bool) -> dict:
    require_authorised(doc, authorised, "run")
    guards = Guards(doc, read_job_start(out_dir))
    load_report(out_dir, "preflight", doc, manifest_sha256, predecessor=None)
    staging = load_report(out_dir, "staging", doc, manifest_sha256,
                          predecessor=out_dir / "preflight.json")
    guards.check("run entry")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    items = load_smoke_items(doc)
    expected = expected_row_invariants(doc, manifest_sha256, items)
    rows_dir = out_dir / "rows"
    diag_dir = out_dir / "rows_diag"
    done = existing_valid_rows(rows_dir, expected)

    wg, load_info = load_patched_classifier(
        staging["body"]["local_snapshot_path"])
    guards.check("post-load")

    n_new = 0
    for item in items:
        if item["blind_id"] in done:
            continue
        guards.check(f"pre-row {item['blind_id']}")
        t0 = time.time()
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        measure = measure_input(wg, item["prompt"], item["response"])
        if measure["truncated_would_occur"]:
            # Never silently classify a truncated input (P1-4).
            parsed = {"status": "unresolved_input_too_long",
                      "raw": {"prompt_harmfulness": None,
                              "response_harmfulness": None,
                              "response_refusal": None,
                              "is_parsing_error": False}}
            completion_sha = None
        else:
            raw, completion_text = classify_one(wg, item)
            parsed = validate_raw(raw)
            completion_sha = sha256_text(completion_text)
            diag = {"schema_version": DIAG_SCHEMA,
                    "blind_id": item["blind_id"],
                    "decoded_completion": completion_text,
                    "completion_sha256": completion_sha,
                    "manifest_file_sha256": manifest_sha256,
                    "runner_version": RUNNER_VERSION,
                    "utc": started}
            diag["diag_sha256"] = sha256_canonical(
                {k: v for k, v in diag.items() if k != "diag_sha256"})
            persist_diag(diag_dir, diag)
        row = {**expected[item["blind_id"]],
               "raw": parsed["raw"], "status": parsed["status"],
               "dtype": load_info["dtype"],
               **measure,
               "completion_sha256": completion_sha,
               "started_utc": started,
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()),
               "wall_s": round(time.time() - t0, 3)}
        row["row_sha256"] = row_sha256(row)
        persist_row(rows_dir, row, expected)
        n_new += 1

    body = {"ok": True, "n_new": n_new,
            "n_total": len(existing_valid_rows(rows_dir, expected)),
            "load_info": load_info,
            "environment_fingerprint":
                staging["body"]["environment"]["fingerprint_sha256"]}
    save_report(out_dir, "run", doc, manifest_sha256,
                body, predecessor=out_dir / "staging.json")
    return body


def stage_verify(doc: dict, out_dir: Path, manifest_path: Path,
                 manifest_sha256: str) -> dict:
    guards = Guards(doc, read_job_start(out_dir))
    load_report(out_dir, "preflight", doc, manifest_sha256, predecessor=None)
    staging = load_report(out_dir, "staging", doc, manifest_sha256,
                          predecessor=out_dir / "preflight.json")
    run = load_report(out_dir, "run", doc, manifest_sha256,
                      predecessor=out_dir / "staging.json")
    if not run["body"].get("ok"):
        raise SmokeRefusal("run report is not ok")
    guards.check("verify entry")

    items = load_smoke_items(doc)
    expected = expected_row_invariants(doc, manifest_sha256, items)
    rows = existing_valid_rows(out_dir / "rows", expected)
    want = set(doc["smoke"]["blind_ids"])
    if set(rows) != want:
        raise SmokeRefusal(f"verify: have {len(rows)}/8 rows; "
                           f"missing {sorted(want - set(rows))}")
    check_diag_linkage(rows, out_dir / "rows_diag")

    # Operational check only — deliberately NO aggregation, NO arm effects.
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=ROOT).stdout.strip() or None
    dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True,
                                cwd=ROOT).stdout.strip())
    prov = {"runner_version": RUNNER_VERSION,
            "binding": execution_binding(doc, manifest_sha256),
            "manifest_path": str(manifest_path),
            "git_commit": commit, "git_dirty": dirty,
            "model_revision": doc["model"]["model_revision"],
            "local_snapshot_manifest_sha256":
                staging["body"]["local_snapshot_manifest_sha256"],
            "environment_fingerprint":
                staging["body"]["environment"]["fingerprint_sha256"],
            "n_rows": len(rows),
            "statuses": {b: r["status"] for b, r in sorted(rows.items())},
            "arm_effects_computed": False,
            "job_elapsed_s": round(time.time() - guards.t0, 1),
            "boundary_note": ("job runtime bounded by the shared JOB_START "
                              "clock; billed pod lifetime ends at console "
                              "termination after the verified pull "
                              "(operator boundary)"),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    save_report(out_dir, "verify", doc, manifest_sha256,
                {"provenance": prov}, predecessor=out_dir / "run.json")
    _atomic_write(out_dir / "provenance.json",
                  json.dumps(prov, sort_keys=True, indent=1,
                             ensure_ascii=False))
    (out_dir / "SMOKE_DONE.marker").write_text(prov["utc"] + "\n")
    return prov


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--stage", required=True,
                    choices=["preflight", "stage_weights", "run", "verify"])
    ap.add_argument("--authorised", action="store_true",
                    help="Tony's spend sign-off for the EXACT bound hash pair")
    ap.add_argument("--out-dir", default="results/p5_wildguard/smoke")
    args = ap.parse_args(argv)

    manifest_path = Path(args.manifest)
    out_dir = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() \
        else Path(args.out_dir)
    try:
        doc = bind(manifest_path, args.manifest_sha256)
        if args.stage == "preflight":
            report = stage_preflight(doc, out_dir, args.manifest_sha256)
        elif args.stage == "stage_weights":
            report = stage_weights(doc, out_dir, args.manifest_sha256,
                                   args.authorised)
        elif args.stage == "run":
            report = stage_run(doc, out_dir, args.manifest_sha256,
                               args.authorised)
        else:
            report = stage_verify(doc, out_dir, manifest_path,
                                  args.manifest_sha256)
    except SmokeRefusal as e:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "SMOKE_FAILED.marker").write_text(f"{args.stage}: {e}\n")
        print(f"REFUSED [{args.stage}]: {e}", file=sys.stderr)
        return 3
    print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
