#!/usr/bin/env python3
"""ph2 — Phase-2 causal-transport executor (PHASE2_TRANSPORT_PREREG_2026-08-08.md).

Stage graph with resume markers, the five-outcome decision logic, the missing-annotation
policy, and the provenance payload are IMPLEMENTED (unit-tested in
tests/test_ph2_executor.py). Model-touching stages are SPEC'D STUBS pending the pre-spend
checks (integrated plan Priority 3) — each stub's docstring is its build contract.

Nothing here spends: generation/annotation stages refuse to run until
results/prereg/phase2_task_manifest.json exists AND --authorised is passed (Tony's launch
sign-off flag, recorded into provenance).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "results/ph2"
MARKERS = OUT / "markers"
PREREG = ROOT / "results/prereg/PHASE2_TRANSPORT_PREREG_2026-08-08.md"
MANIFEST = ROOT / "results/prereg/phase2_task_manifest.json"

STAGES = ["manifest_verify", "discovery_extract", "target_gates", "refit",
          "generate_battery", "injection_recovery", "annotate", "analyse"]

PROVENANCE_KEYS = [
    "git_commit", "git_dirty", "prereg_sha256", "manifest_sha256", "checkpoint_revisions",
    "tokenizer_gate", "frame_source", "counts", "seeds", "annotator", "achieved_mde",
    "authorised", "amended",
]


# ── decision logic (prereg §8, incl. A-amendment hierarchy) ──────────────────

def decide_outcome(o: dict) -> str:
    """Five-outcome hierarchy. `o` carries booleans produced by the analyse stage:

    sensitivity_adequate, damage_ok, gate_pass (target-null grounding gate on transported
    frame), raw_retained (uncorrected effect excludes attenuation >= sealed margin),
    corrected_gate_pass / corrected_retained (predeclared norm/gain/whitening arms),
    refit_pass, refit_aligned (within null-calibrated bound), refit_misaligned_beyond_null,
    refit_recovered, repr_signal_above_null.

    First-satisfied row wins; decoupled precedes disabled whenever representational signal
    remains; inadequate sensitivity downgrades the vocabulary before any row is read.
    """
    if not o.get("sensitivity_adequate", False):
        return "downgraded"          # report bound / "not disabled" language only
    if not o.get("damage_ok", True):
        return "damage_stop"         # damage gate: no primary interpretation for this arm
    if o.get("gate_pass") and o.get("refit_aligned") and o.get("raw_retained"):
        return "retained"
    if (not o.get("raw_retained") and o.get("corrected_gate_pass")
            and o.get("refit_aligned") and o.get("corrected_retained")):
        return "rescaled"
    if ((not o.get("gate_pass") or not o.get("raw_retained"))
            and o.get("refit_pass") and o.get("refit_misaligned_beyond_null")
            and o.get("refit_recovered")):
        return "rotated"
    if (o.get("repr_signal_above_null")
            and not (o.get("raw_retained") or o.get("corrected_retained")
                     or o.get("refit_recovered"))):
        return "decoupled"
    if (not o.get("repr_signal_above_null") and not o.get("refit_pass")
            and not o.get("raw_retained")):
        return "disabled"
    return "inconclusive"


# ── missing-annotation policy (prereg gate 2: unresolved, never zero) ────────

def merge_annotations(rows: list[dict]) -> dict[str, dict]:
    """Per-task merge; a task with missing/empty annotation payload is status='unresolved'
    and MUST be excluded pairwise by downstream stats (with counts reported) — it is never
    coerced to a zero rate."""
    out: dict[str, dict] = {}
    for r in rows:
        tid = r["task_id"]
        ann = r.get("annotations")
        if not ann:
            out[tid] = {"status": "unresolved"}
            continue
        n_all = len(ann)
        counts: dict[str, int] = {}
        for a in ann:
            labs = a.get("labels", a.get("label", []))
            if isinstance(labs, str):
                labs = [labs]
            for lab in labs or []:
                counts[lab] = counts.get(lab, 0) + 1
        out[tid] = {"status": "ok", "n_sentences": n_all, "label_counts": counts}
    return out


# ── provenance ───────────────────────────────────────────────────────────────

def build_provenance(extra: dict | None = None) -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True, cwd=ROOT).stdout.strip())
    p = {k: None for k in PROVENANCE_KEYS}
    p.update({
        "git_commit": commit, "git_dirty": dirty,
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest()[:16]
        if PREREG.exists() else None,
        "manifest_sha256": (json.loads(MANIFEST.read_text())["ids_sha256"]
                            if MANIFEST.exists() else None),
        "amended": ["A1"],
    })
    p.update(extra or {})
    return p


# ── stage framework (resume via markers) ─────────────────────────────────────

def stage_done(name: str) -> bool:
    return (MARKERS / f"{name}.done").exists()


def mark_done(name: str) -> None:
    MARKERS.mkdir(parents=True, exist_ok=True)
    (MARKERS / f"{name}.done").write_text("done")


def run_stage(name: str, authorised: bool) -> None:
    if stage_done(name):
        print(f"[skip] {name} (marker present)")
        return
    fn = globals()[f"stage_{name}"]
    fn(authorised)
    mark_done(name)


def stage_manifest_verify(authorised: bool) -> None:
    import ph2_manifest
    doc = json.loads(MANIFEST.read_text())
    ph2_manifest.verify_disjoint(doc["tasks"], ph2_manifest.build_exclusions())
    assert ph2_manifest.manifest_hash(doc["tasks"]) == doc["ids_sha256"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "provenance.json").write_text(json.dumps(build_provenance(
        {"authorised": authorised}), indent=1))


def stage_discovery_extract(authorised):
    raise NotImplementedError(
        "SPEC: teacher-force annotated-corpus spans through STAR1 + DeepScaleR on "
        "byte-identical R1-tokenizer ids (--tokenizer-alias 1.5b); layers {16,17,18}; "
        "outputs to data/activations/{model}-ph2disc/; pod stage.")


def stage_target_gates(authorised):
    raise NotImplementedError(
        "SPEC: 20 sham (same builder budget) + 20 random-orthogonal frames per target; "
        "coord-AUC + state-dependence vs pooled p95 target-null; L16/L18 neighbours; "
        "base absolute thresholds reported for continuity only.")


def stage_refit(authorised):
    raise NotImplementedError(
        "SPEC: 20_das_backtracking.py sealed recipe, L17 only, width {1,2}, pairs from "
        "discovery spans, M5 — no target-specific tuning; alignment vs random-orthogonal null.")


def stage_generate_battery(authorised):
    raise NotImplementedError(
        "SPEC: resumable; arms = transported-raw(±)/norm(±)/whitened(±), refit(±), sham, "
        "rand-orth, count-floor, energy-floor, vanilla × 100 manifest tasks × 3 models; "
        "E8 sealed generation settings; seed 20260808; REFUSES without manifest+authorised.")


def stage_injection_recovery(authorised):
    raise NotImplementedError(
        "SPEC: base model only; frame⊕sham mixing at f∈{0.25,0.5,0.75}×100 tasks; endpoints "
        "reuse main-battery arms; evaluated pre-outcome; pass=80% power at f=0.5.")


def stage_annotate(authorised):
    raise NotImplementedError(
        "SPEC: Nova-Pro (non-builder), house all-behaviour schema, missing rows unresolved "
        "(merge_annotations), resume-safe sharding; Sonnet duplicate on ≤20% diagnostic only.")


def stage_analyse(authorised):
    raise NotImplementedError(
        "SPEC: authoritative pooling via src.delta_floor; paired task-level bootstrap "
        "B=10,000 seed 20260808; Holm over 2 checkpoints; damage gates; decide_outcome(); "
        "adjunct A2 estimation family on vanilla arms.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--authorised", action="store_true",
                    help="Tony's launch sign-off (recorded in provenance)")
    a = ap.parse_args()
    if not MANIFEST.exists():
        raise SystemExit("No task manifest — run ph2_manifest.py --generate first (A1).")
    if a.stage in ("generate_battery", "injection_recovery", "annotate") and not a.authorised:
        raise SystemExit(f"stage {a.stage} spends — requires --authorised (Tony).")
    run_stage(a.stage, a.authorised)
