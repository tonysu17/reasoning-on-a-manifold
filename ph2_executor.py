#!/usr/bin/env python3
"""ph2 — Phase-2 causal-transport executor (PHASE2_TRANSPORT_PREREG_2026-08-08.md).

Stage graph with resume markers, the five-outcome decision logic, the missing-
annotation policy, the provenance payload, and ALL EIGHT STAGES are implemented
(pre-spend logic unit-tested in tests/test_ph2_executor.py +
tests/test_ph2_stages.py). Model-touching stages lazy-import torch/transformers
and are pod stages; pure stages (annotate glue, analyse) run anywhere.

Nothing here spends: generation/annotation stages refuse to run until
results/prereg/phase2_task_manifest.json exists AND --authorised is passed
(Tony's launch sign-off flag, recorded into provenance). Stage order (§ = prereg):

    manifest_verify      local   provenance gate on the sealed A1 manifest
    discovery_extract    pod     pair-state extraction per role at hs{16,17,18}
                                 (byte-identical R1-tokenizer ids; sealed E10.1
                                 pairs) + norm/whitening/class-mean stats  §3/§4
    target_gates         pod     20 sham (builder-budget) + 20 random-orthogonal
                                 frames per target; coord-AUC + state-dependence
                                 vs pooled p95; L16/L18 neighbours          §4.2
    refit                pod     sealed E10.2 recipe, hs17, widths {1,2}, M5 §4.3
    generate_battery     pod     13-cell battery × manifest tasks × 3 roles at
                                 the E8 sealed generation settings          §4.4
    injection_recovery   pod     base only, frame⊕sham at f∈{.25,.5,.75}     §6
    annotate             local   Sonnet 4.5 via lab proxy (A3), 29-s chunking,
                                 resume-safe, unresolved-never-zero
    analyse              local   src.delta_floor authoritative pooling, paired
                                 B=10k bootstrap, Holm over 2 checkpoints,
                                 damage gates, decide_outcome, A2 adjunct

Provenance follows the codex field spec
(.codex/out/EVIDENCE_MANIFEST_AND_CLAIM_FIELD_SPEC_2026-08-08.md): the 13
compatibility keys keep their exact names/semantics; the rom-result-provenance-v1
companion fields ride alongside.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results/ph2"
MARKERS = OUT / "markers"
PROV_DIR = OUT / "provenance"
PREREG = ROOT / "results/prereg/PHASE2_TRANSPORT_PREREG_2026-08-08.md"
ADJUNCT = ROOT / "results/prereg/PHASE2_ADJUNCT_AMENDMENT_2026-08-08.md"
MANIFEST = ROOT / "results/prereg/phase2_task_manifest.json"
PAIRS = ROOT / "results/das/R1-1.5B/main/pairs.json"   # sealed E10.1 pairs (R1 ids)

STAGES = ["manifest_verify", "discovery_extract", "target_gates", "refit",
          "generate_battery", "injection_recovery", "annotate", "analyse"]
SPEND_STAGES = ("generate_battery", "injection_recovery", "annotate")

PROVENANCE_KEYS = [
    "git_commit", "git_dirty", "prereg_sha256", "manifest_sha256", "checkpoint_revisions",
    "tokenizer_gate", "frame_source", "counts", "seeds", "annotator", "achieved_mde",
    "authorised", "amended",
]

AMENDED = ["A1", "A2", "A3", "A4"]  # sealed amendment lineage (protocol markers)

#: Annotation output budget (owner decision, Tony 2026-08-09 second revision:
#: "2-4k tokens"). NOTE this ceiling is NOT the cost lever — billing follows
#: tokens actually generated (~1.6k/call chunk echo); cost is cut by the A4
#: annotation WINDOW (src.ph2_stages.ANNOTATION_WINDOW_TOKENS). 4,000 keeps
#: echo headroom, and the first 504-shrink (→2,000) still clears the ~1,620
#: worst-case echo. The 29-s proxy ceiling is held by chunking throughout.
ANNOTATION_MAX_TOKENS = 4000

#: Sonnet is the builder annotator (labels, frames, and E8 verdicts derive from
#: Sonnet annotations) — every behavioural verdict carries this qualifier (A3).
BUILDER_ANNOTATOR_CAVEAT = (
    "builder-annotator scored (Sonnet 4.5, Amendment A3): behavioural-rate "
    "endpoints re-acquire the builder-annotator circularity caveat; geometric "
    "calls are annotator-robust (pt04c cos 0.999; pt14 8/8) but rates have no "
    "in-phase swap test")


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


# ── provenance (compat keys + rom-result-provenance-v1 companions) ───────────

def _sha256_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _repo_rel(path: Path) -> str:
    """Repo-relative when inside ROOT; absolute otherwise (test trees)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_provenance(extra: dict | None = None, stage: str | None = None,
                     run_id: str | None = None,
                     input_paths: list[Path] | None = None,
                     evidence_status: str = "current resource record") -> dict:
    """The full sidecar: the 13 compatibility keys with their EXACT existing
    semantics (`prereg_sha256` stays the 16-char prefix; `manifest_sha256`
    stays the manifest document's ids_sha256, NOT a file hash) plus the
    thesis-grade companion fields from the codex field spec. `extra` overrides
    win last."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=ROOT).stdout.strip() or None
    dirty_out = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, cwd=ROOT).stdout
    dirty_paths = [ln[3:] for ln in dirty_out.splitlines() if ln.strip()]
    dirty = bool(dirty_paths)
    prereg_full = _sha256_file(PREREG)
    manifest_doc = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None

    from src.ph2_stages import (CHECKPOINTS, RUN_SEED, B_BOOT, DAS_BUILD,
                                HS_SITE, HS_NEIGHBOURS, FRAME_PATH, FRAME_WIDTH)
    unresolved: list[str] = []
    if commit is None:
        unresolved.append("git_commit unavailable")
    if dirty:
        unresolved.append("working tree dirty at execution")
    if manifest_doc is None:
        unresolved.append("task manifest absent")

    p = {k: None for k in PROVENANCE_KEYS}
    p.update({
        # ── compatibility keys (exact legacy semantics) ──────────────────────
        "git_commit": commit, "git_dirty": dirty,
        "prereg_sha256": prereg_full[:16] if prereg_full else None,
        "manifest_sha256": manifest_doc["ids_sha256"] if manifest_doc else None,
        "checkpoint_revisions": {r: dict(c) for r, c in CHECKPOINTS.items()},
        "seeds": {"run": RUN_SEED, "bootstrap": RUN_SEED,
                  "manifest_draw": RUN_SEED, "das_builder": DAS_BUILD["seed"]},
        "amended": list(AMENDED),
        # ── companion fields (rom-result-provenance-v1) ──────────────────────
        "schema_version": "rom-result-provenance-v1",
        "run_id": run_id or f"ph2-{time.strftime('%Y%m%d-%H%M%S')}",
        "stage": stage,
        "dirty_paths": dirty_paths,
        "preregistration_path": str(PREREG.relative_to(ROOT)),
        "prereg_sha256_full": prereg_full,
        "adjunct_prereg_sha256_full": _sha256_file(ADJUNCT),
        "manifest_path": str(MANIFEST.relative_to(ROOT)),
        "manifest_ids_sha256": manifest_doc["ids_sha256"] if manifest_doc else None,
        "manifest_file_sha256": _sha256_file(MANIFEST),
        "input_manifest": {_repo_rel(q): _sha256_file(q)
                           for q in (input_paths or []) if q.exists()},
        "frame_source": {
            "path": str(FRAME_PATH.relative_to(ROOT)),
            "sha256": _sha256_file(FRAME_PATH),
            "builder": "21_das_width.py (E10.2 sealed recipe)",
            "behaviour": "backtracking", "width": FRAME_WIDTH,
            "hs_site": HS_SITE, "hs_neighbours": list(HS_NEIGHBOURS),
            "layer_convention": ("hs[K] = output of model.model.layers[K-1]; "
                                 "steering/extraction 'layer L' = hs[L+1]; DAS "
                                 "resid_pre 'layer L' = hs[L]"),
        },
        "scientific_unit": "task",
        "observation_unit": "generated response (sentence rows nested)",
        "pairing_key": "task_id",
        "missingness_policy": "missing_or_empty_is_unresolved_never_zero",
        "argv": list(sys.argv),
        "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {"hostname": platform.node(), "system": platform.system(),
                 "machine": platform.machine(), "cpus": os.cpu_count()},
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "empirical_evidence_status": evidence_status,
        "provenance_status": ("resolved" if not unresolved
                              else "unresolved provenance"),
        "provenance_unresolved_reasons": unresolved,
        "protocol_markers": [f"amended:{a}" for a in AMENDED],
        "authorisation_record": None,
        "annotator": None,
    })
    p.update(extra or {})
    return p


def _package_versions() -> dict:
    out = {}
    for mod in ("numpy", "scipy", "torch", "transformers"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def write_provenance(stage: str, payload: dict, suffix: str = "") -> Path:
    PROV_DIR.mkdir(parents=True, exist_ok=True)
    path = PROV_DIR / f"{stage}{('_' + suffix) if suffix else ''}.json"
    payload = dict(payload)
    payload["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps(payload, indent=1, default=str))
    return path


def _atomic_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str))
    tmp.rename(path)


def _require(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise SystemExit(f"missing input {path} — run stage '{produced_by}' first")
    return path


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _authorise_guard(stage: str, authorised: bool) -> None:
    """Spend refusal, enforced INSIDE each spending stage (not only the CLI)."""
    if not MANIFEST.exists():
        raise SystemExit("No task manifest — run ph2_manifest.py --generate first (A1).")
    if stage in SPEND_STAGES and not authorised:
        raise SystemExit(f"stage {stage} spends — requires --authorised (Tony).")


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


def manifest_tasks() -> list[dict]:
    return json.loads(MANIFEST.read_text())["tasks"]


# ── stage 1: manifest verify ─────────────────────────────────────────────────

def stage_manifest_verify(authorised: bool) -> None:
    import ph2_manifest
    doc = json.loads(MANIFEST.read_text())
    ph2_manifest.verify_disjoint(doc["tasks"], ph2_manifest.build_exclusions())
    assert ph2_manifest.manifest_hash(doc["tasks"]) == doc["ids_sha256"]
    OUT.mkdir(parents=True, exist_ok=True)
    prov = build_provenance({"authorised": authorised}, stage="manifest_verify",
                            input_paths=[MANIFEST, PREREG, ADJUNCT])
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    write_provenance("manifest_verify", prov)


# ── stage 2: discovery extraction (pod) ──────────────────────────────────────

def stage_discovery_extract(authorised: bool) -> None:
    """Teacher-forced pair-state extraction per role at hs{16,17,18} (§3/§4).

    The discovery set = the sealed E10.1 onset-anchored pairs
    (results/das/R1-1.5B/main/pairs.json — contexts are R1-tokenizer token-id
    lists built from the annotated corpus, so replaying them through STAR1 /
    DeepScaleR is byte-identical-input teacher forcing by construction, C1).
    Per role it saves the source/base prediction-position states at each hs
    site (npz) and derives the §4 discovery statistics: per-model norm gain,
    diagonal whitening stds, and frame class-mean clamp targets c_on/c_off.
    Span labels transfer with the text; no new annotation.

    Pod stage (~minutes/role on a 4090). Resume: per-role substep markers.
    """
    import numpy as np
    from src.ph2_stages import (ALL_ROLES, HS_ALL, HS_SITE, FRAME_PATH,
                                DAS_BUILD, class_mean_coords, norm_gain,
                                whitening_stats, load_checkpoint)
    _require(PAIRS, "E10.1 (sealed pairs are a committed artifact)")
    pairs = json.loads(PAIRS.read_text())[:DAS_BUILD["n_pairs"]]
    das = _load_module("das20", "20_das_backtracking.py")
    disc_dir = OUT / "discovery"
    disc_dir.mkdir(parents=True, exist_ok=True)

    class _Cfg:  # sealed builder budget carries the batch/window constants
        bs = DAS_BUILD["bs"]; window = DAS_BUILD["window"]

    stats: dict = {}
    stats_path = disc_dir / "discovery_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
    U_base = np.load(FRAME_PATH)

    for role in ALL_ROLES:
        marker = MARKERS / f"discovery_{role}.done"
        npz_path = disc_dir / f"{role}_pair_states.npz"
        if marker.exists() and npz_path.exists():
            print(f"[skip] discovery {role}")
            continue
        tok, model, resolved = load_checkpoint(role)
        device = next(model.parameters()).device
        arrays: dict = {}
        for hs in HS_ALL:  # DAS helpers take the hs index directly (resid_pre)
            batches = das.make_batches(pairs, tok, model, hs, device,
                                       _Cfg.bs, _Cfg.window)
            hb = np.concatenate([b["hb"].cpu().numpy() for b in batches])
            hsrc = np.concatenate([b["hs"].cpu().numpy() for b in batches])
            arrays[f"h_base_hs{hs}"] = hb
            arrays[f"h_source_hs{hs}"] = hsrc
        np.savez_compressed(npz_path, **arrays)
        H_on = arrays[f"h_source_hs{HS_SITE}"]
        H_off = arrays[f"h_base_hs{HS_SITE}"]
        pooled = np.concatenate([H_on, H_off])
        cm = class_mean_coords(H_on, H_off, U_base)
        stats[role] = {
            "checkpoint": resolved,
            "n_pairs": len(pairs),
            "mean_l2_hs17": float(np.mean(np.linalg.norm(pooled, axis=1))),
            "whitening_std_path": f"{role}_whitening_std.npy",
            "class_means_base_frame": {"c_on": cm["c_on"].tolist(),
                                       "c_off": cm["c_off"].tolist()},
        }
        np.save(disc_dir / f"{role}_whitening_std.npy", whitening_stats(pooled))
        _atomic_json(stats, stats_path)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("done")
        del model

    base_states = np.load(disc_dir / "base_pair_states.npz")
    base_pooled = np.concatenate([base_states[f"h_source_hs{HS_SITE}"],
                                  base_states[f"h_base_hs{HS_SITE}"]])
    for role in ALL_ROLES:
        if role == "base":
            stats[role]["norm_gain_vs_base"] = 1.0
            continue
        tgt = np.load(disc_dir / f"{role}_pair_states.npz")
        tgt_pooled = np.concatenate([tgt[f"h_source_hs{HS_SITE}"],
                                     tgt[f"h_base_hs{HS_SITE}"]])
        stats[role]["norm_gain_vs_base"] = norm_gain(base_pooled, tgt_pooled)
    _atomic_json(stats, stats_path)
    write_provenance("discovery_extract", build_provenance(
        {"authorised": authorised,
         "counts": {r: stats[r]["n_pairs"] for r in stats}},
        stage="discovery_extract", input_paths=[PAIRS, FRAME_PATH]))


# ── stage 3: target grounding gates (pod) ────────────────────────────────────

def stage_target_gates(authorised: bool) -> None:
    """§4.2: per target, 20 sham frames (same builder/search budget, trained on
    shuffled pairs — the Makelov-style budget-matched null) + 20 random-
    orthogonal frames (same width/norm); gate = transported frame > pooled
    40-frame null p95 on BOTH coord-AUC and state-dependence, at hs17 with
    hs16/hs18 neighbours reported. Base absolute thresholds (0.65/2.0) reported
    for continuity only. Also trains sham #0 on BASE (the battery's sham arm +
    the injection-recovery mixing partner need a base-built sham).

    Pod stage; ~20 sham trainings × 2 targets + 1 base at the sealed budget
    (~few min each on a 4090). Every trained sham checkpoints to
    results/ph2/frames/ so interrupts resume frame-by-frame.
    """
    import numpy as np
    import torch
    from src.ph2_stages import (TARGET_ROLES, HS_ALL, HS_SITE, FRAME_PATH,
                                FRAME_WIDTH, N_NULL_FRAMES, DAS_BUILD,
                                random_orthonormal_frame, target_gate_verdict,
                                load_checkpoint)
    _require(PAIRS, "E10.1 (sealed pairs)")
    pairs = json.loads(PAIRS.read_text())[:DAS_BUILD["n_pairs"]]
    width_mod = _load_module("das21", "21_das_width.py")
    frames_dir = OUT / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    gates_path = OUT / "gates" / "target_gates.json"
    gates: dict = json.loads(gates_path.read_text()) if gates_path.exists() else {}

    class _Cfg:
        bs = DAS_BUILD["bs"]; window = DAS_BUILD["window"]
        epochs = DAS_BUILD["epochs"]; lr = DAS_BUILD["lr"]

    U_base = np.load(FRAME_PATH)
    d = U_base.shape[0]

    roles_and_counts = [("base", 1)] + [(r, N_NULL_FRAMES) for r in TARGET_ROLES]
    for role, n_sham in roles_and_counts:
        if role in gates and gates[role].get("complete"):
            print(f"[skip] gates {role}")
            continue
        tok, model, resolved = load_checkpoint(role)
        device = next(model.parameters()).device

        # sham frames: sealed builder budget, shuffled pairs, seeds 0..n-1
        shams = []
        for i in range(n_sham):
            path = frames_dir / f"sham_{role}_{i}.npy"
            if path.exists():
                shams.append(np.load(path))
                continue
            cfg = _Cfg(); cfg.seed = DAS_BUILD["seed"] + i
            U, _curve = width_mod.train_frame(model, tok, pairs, HS_SITE,
                                              FRAME_WIDTH, cfg, device,
                                              shuffle_pairs=True)
            U = U.cpu().numpy()
            np.save(path, U)
            shams.append(U)

        randorth = []
        for i in range(N_NULL_FRAMES):
            path = frames_dir / f"randorth_{role}_{i}.npy"
            if not path.exists():
                np.save(path, random_orthonormal_frame(
                    d, FRAME_WIDTH, f"ph2_randorth|{role}|rep{i}"))
            randorth.append(np.load(path))

        cell: dict = {"checkpoint": resolved, "sites": {}}
        if role != "base":            # the gate itself is a TARGET verdict
            cfg = _Cfg(); cfg.seed = DAS_BUILD["seed"]
            for hs in HS_ALL:
                trans = width_mod.grounding(
                    model, tok, pairs, hs,
                    torch.tensor(U_base, dtype=torch.float32, device=device),
                    cfg, device)
                nulls = []
                for U in (shams + randorth):
                    nulls.append(width_mod.grounding(
                        model, tok, pairs, hs,
                        torch.tensor(U, dtype=torch.float32, device=device),
                        cfg, device))
                cell["sites"][f"hs{hs}"] = {
                    "transported": trans,
                    "verdict": target_gate_verdict(trans, nulls),
                }
            cell["gate_pass"] = cell["sites"][f"hs{HS_SITE}"]["verdict"]["gate_pass"]
        cell["n_sham"] = n_sham
        cell["n_randorth"] = N_NULL_FRAMES
        cell["complete"] = True
        gates[role] = cell
        _atomic_json(gates, gates_path)
        del model

    write_provenance("target_gates", build_provenance(
        {"authorised": authorised,
         "counts": {r: {"sham": gates[r]["n_sham"], "randorth": gates[r]["n_randorth"]}
                    for r in gates}},
        stage="target_gates", input_paths=[PAIRS, FRAME_PATH]))


# ── stage 4: re-fit (pod) ────────────────────────────────────────────────────

def stage_refit(authorised: bool) -> None:
    """§4.3: re-fit target frames on the discovery pairs under the sealed
    E10.2 recipe (hs17 only, widths {1,2}, sealed optimizer/steps/seed, M5 —
    no target-specific tuning). Alignment vs the transported frame is
    adjudicated against the random-orthogonal null (refit_aligned /
    refit_misaligned_beyond_null feed decide_outcome). Also derives each
    re-fit frame's own clamp targets from the role's discovery states.
    """
    import numpy as np
    from src.ph2_stages import (TARGET_ROLES, HS_SITE, FRAME_PATH, REFIT_WIDTHS,
                                FRAME_WIDTH, N_NULL_FRAMES, DAS_BUILD,
                                class_mean_coords, refit_alignment,
                                load_checkpoint)
    _require(PAIRS, "E10.1 (sealed pairs)")
    disc_dir = OUT / "discovery"
    frames_dir = OUT / "frames"
    pairs = json.loads(PAIRS.read_text())[:DAS_BUILD["n_pairs"]]
    width_mod = _load_module("das21", "21_das_width.py")

    class _Cfg:
        bs = DAS_BUILD["bs"]; window = DAS_BUILD["window"]
        epochs = DAS_BUILD["epochs"]; lr = DAS_BUILD["lr"]
        seed = DAS_BUILD["seed"]

    U_base = np.load(FRAME_PATH)
    report_path = OUT / "refit" / "refit_report.json"
    report: dict = json.loads(report_path.read_text()) if report_path.exists() else {}

    for role in TARGET_ROLES:
        if role in report and report[role].get("complete"):
            print(f"[skip] refit {role}")
            continue
        _require(disc_dir / f"{role}_pair_states.npz", "discovery_extract")
        tok, model, resolved = load_checkpoint(role)
        device = next(model.parameters()).device
        cell: dict = {"checkpoint": resolved, "widths": {}}
        for k in REFIT_WIDTHS:
            fpath = frames_dir / f"refit_{role}_k{k}.npy"
            if fpath.exists():
                U = np.load(fpath)
            else:
                cfg = _Cfg()
                U_t, curve = width_mod.train_frame(model, tok, pairs, HS_SITE,
                                                   k, cfg, device)
                U = U_t.cpu().numpy()
                np.save(fpath, U)
            states = np.load(disc_dir / f"{role}_pair_states.npz")
            cm = class_mean_coords(states[f"h_source_hs{HS_SITE}"],
                                   states[f"h_base_hs{HS_SITE}"], U)
            nulls = [np.load(frames_dir / f"randorth_{role}_{i}.npy")
                     for i in range(N_NULL_FRAMES)
                     if (frames_dir / f"randorth_{role}_{i}.npy").exists()]
            align = (refit_alignment(U_base, U, nulls) if k == FRAME_WIDTH and nulls
                     else None)
            cell["widths"][str(k)] = {
                "frame_path": str(fpath.relative_to(ROOT)),
                "class_means": {kk: v.tolist() for kk, v in cm.items()},
                "alignment_vs_base": align,
            }
        cell["complete"] = True
        report[role] = cell
        _atomic_json(report, report_path)
        del model

    write_provenance("refit", build_provenance(
        {"authorised": authorised}, stage="refit",
        input_paths=[PAIRS, FRAME_PATH]))


# ── stage 5: generation battery (pod; SPENDS) ────────────────────────────────

def _arm_terms(role: str, arm: dict, disc: dict, frames_dir: Path):
    """Resolve one battery arm to frame-clamp terms + metadata. Returns
    (terms, frame_name, alias_of): alias_of != None means this cell is a
    base-role duplicate of transported_raw (norm gain = 1, whitening ratio = 1,
    refit == builder frame on base) and must NOT be generated again — the
    analysis reads the aliased cells instead (recorded in provenance)."""
    import numpy as np
    from src.ph2_stages import (FRAME_PATH, HS_SITE, CLAMP_GAIN, FRAME_WIDTH,
                                random_orthonormal_frame, whitened_frame,
                                class_mean_coords, clamp_targets_for_arm)
    method, family, sign = arm["method"], arm["family"], arm["sign"]
    if family == "vanilla":
        return [], None, None
    if role == "base" and family in ("transported_norm", "transported_whitened",
                                     "refit"):
        return None, None, f"transported_raw_{sign}"

    U_base = np.load(FRAME_PATH)
    stats = disc[role]
    cm_base = {k: np.array(v) for k, v in
               stats["class_means_base_frame"].items()}
    disc_dir = OUT / "discovery"

    def own_class_means(U):
        states = np.load(disc_dir / f"{role}_pair_states.npz")
        return class_mean_coords(states[f"h_source_hs{HS_SITE}"],
                                 states[f"h_base_hs{HS_SITE}"], U)

    if family == "transported_raw":
        c = clamp_targets_for_arm(family, sign, cm_base)
        return [{"U": U_base, "c": c, "beta": CLAMP_GAIN}], "base", None
    if family == "transported_norm":
        gain = stats["norm_gain_vs_base"]
        c = clamp_targets_for_arm(family, sign, cm_base, gain=gain)
        return [{"U": U_base, "c": c, "beta": CLAMP_GAIN}], "base", None
    if family == "transported_whitened":
        base_std = np.load(disc_dir / "base_whitening_std.npy")
        tgt_std = np.load(disc_dir / f"{role}_whitening_std.npy")
        U_w = whitened_frame(U_base, base_std, tgt_std)
        c = clamp_targets_for_arm(family, sign, own_class_means(U_w))
        return [{"U": U_w, "c": c, "beta": CLAMP_GAIN}], "base_whitened", None
    if family == "refit":
        U = np.load(frames_dir / f"refit_{role}_k{FRAME_WIDTH}.npy")
        c = clamp_targets_for_arm(family, sign, own_class_means(U))
        return [{"U": U, "c": c, "beta": CLAMP_GAIN}], "refit", None
    if method == "sham_frame":
        U = np.load(frames_dir / f"sham_{role}_0.npy")
        c = clamp_targets_for_arm(family, "suppress", own_class_means(U))
        return [{"U": U, "c": c, "beta": CLAMP_GAIN}], "sham0", None
    if method == "random_orthogonal":
        U = np.load(frames_dir / f"randorth_{role}_0.npy")
        c = clamp_targets_for_arm(family, "suppress", own_class_means(U))
        return [{"U": U, "c": c, "beta": CLAMP_GAIN}], "randorth0", None
    if method == "count_matched_floor":
        U = random_orthonormal_frame(U_base.shape[0], FRAME_WIDTH,
                                     f"ph2_count_floor|{role}")
        c = clamp_targets_for_arm(family, "suppress", own_class_means(U))
        return [{"U": U, "c": c, "beta": CLAMP_GAIN}], "count_floor", None
    if method == "energy_matched_floor":
        U = random_orthonormal_frame(U_base.shape[0], FRAME_WIDTH,
                                     f"ph2_energy_floor|{role}")
        c = clamp_targets_for_arm(family, "suppress", own_class_means(U))
        # energy_scale calibrated at generation time against transported_raw
        return [{"U": U, "c": c, "beta": CLAMP_GAIN, "energy_scale": None}], \
            "energy_floor", None
    raise ValueError(f"unknown arm {method}")


def stage_generate_battery(authorised: bool) -> None:
    """§4.4: the sealed 13-cell battery × the 100-task manifest × 3 roles at
    the E8 sealed generation settings (greedy, config 8192 cap, one sample per
    task per arm, run seed 20260808 — inert under greedy but recorded).

    REFUSES without the manifest + --authorised. Resume: per-role atomic
    checkpoint files keyed (role, method, task_id); a killed run loses at most
    one batch. On the BASE role the norm/whitened/refit families are exact
    duplicates of transported_raw (gain 1 / ratio 1 / builder frame) and are
    ALIASED, not regenerated (recorded in provenance.counts.aliased; ~600
    generations saved). The base vanilla rows on the manifest are additionally
    exported as the immutable shared-vanilla artifact for P5
    (P5_PHASE2_SHARED_VANILLA_CONTRACT_2026-08-08.md): E8 cap 8192, greedy,
    seed 20260808, base-tokenizer ids, no custom stop, no </think> stop.

    The energy-matched floor's energy_scale is calibrated per role on 5
    calibration tasks (mean |Δ(Uᵀh)| of transported_raw_suppress ÷ floor's),
    the frame-clamp analogue of E8's energy_matched_scale; the identity/reload
    gate (zero-term hook == plain generate) runs before any cell.
    """
    _authorise_guard("generate_battery", authorised)
    from src.ph2_stages import (ALL_ROLES, battery_arms, battery_cell_key,
                                RUN_SEED, HS_SITE, frame_clamp_model,
                                measure_clamp_displacement, load_checkpoint,
                                identity_reload_gate, battery_record)
    from src.steered_inference import default_max_new_tokens
    disc_path = _require(OUT / "discovery" / "discovery_stats.json",
                         "discovery_extract")
    frames_dir = OUT / "frames"
    disc = json.loads(disc_path.read_text())
    tasks = manifest_tasks()
    cap = default_max_new_tokens()
    bat_dir = OUT / "battery"
    bat_dir.mkdir(parents=True, exist_ok=True)

    counts = {"aliased": {}, "generated": {}}
    for role in ALL_ROLES:
        out_path = bat_dir / f"{role}.json"
        results = json.loads(out_path.read_text()) if out_path.exists() else []
        done = {battery_cell_key(role, r["method"], r["task_id"]) for r in results}
        pending_arms = []
        aliases = {}
        for arm in battery_arms():
            resolved = _arm_terms(role, arm, disc, frames_dir)
            if resolved[2] is not None:            # base-role alias — skip
                aliases[arm["method"]] = resolved[2]
                continue
            todo = [t for t in tasks
                    if battery_cell_key(role, arm["method"], t["id"]) not in done]
            if todo:
                pending_arms.append((arm, resolved, todo))
        counts["aliased"][role] = aliases
        if not pending_arms:
            print(f"[skip] battery {role} (complete)")
            continue

        tok, model, resolved_ckpt = load_checkpoint(role)
        gate = identity_reload_gate(model, tok, tasks, HS_SITE)
        if not gate["pass"]:
            raise SystemExit(f"identity/reload gate FAILED for {role}: {gate}")

        # calibrate the energy floor against transported_raw_suppress
        calib = tasks[:5]
        raw_terms = None
        for arm, (terms, _f, _a), _todo in pending_arms:
            if arm["method"] == "transported_raw_suppress":
                raw_terms = terms
        if raw_terms is None:  # fully-generated raw arm — rebuild for calibration
            for arm in battery_arms():
                if arm["method"] == "transported_raw_suppress":
                    raw_terms = _arm_terms(role, arm, disc, frames_dir)[0]
        e_active = measure_clamp_displacement(model, tok, raw_terms, calib)

        for arm, (terms, frame_name, _alias), todo in pending_arms:
            if arm["method"] == "vanilla":
                engine = frame_clamp_model(model, tok, [], HS_SITE)
            else:
                for t in terms:
                    if t.get("energy_scale") is None:
                        e_floor = measure_clamp_displacement(
                            model, tok, [{**t, "energy_scale": 1.0}], calib)
                        t["energy_scale"] = (e_active / e_floor) if e_floor > 0 else 1.0
                engine = frame_clamp_model(model, tok, terms, HS_SITE)
            for task in todo:
                gen = engine.generate(task["prompt"], max_new_tokens=cap,
                                      temperature=0.0, seed=RUN_SEED)
                gen["mean_abs_displacement"] = engine.mean_abs_displacement()
                results.append(battery_record(
                    role, arm["method"], task, gen, arm["sign"], frame_name,
                    extra={"energy_scale": (terms[0].get("energy_scale", 1.0)
                                            if terms else None)}))
                done.add(battery_cell_key(role, arm["method"], task["id"]))
                _atomic_json(results, out_path)
        counts["generated"][role] = len(results)
        del model

    # shared-vanilla artifact for P5 (immutable input; codex contract)
    base_path = bat_dir / "base.json"
    if base_path.exists():
        vanilla = [r for r in json.loads(base_path.read_text())
                   if r["method"] == "vanilla"]
        if vanilla:
            shared_path = bat_dir / "base_vanilla_shared.json"
            _atomic_json(vanilla, shared_path)
            (bat_dir / "base_vanilla_shared.sha256").write_text(
                hashlib.sha256(shared_path.read_bytes()).hexdigest())

    write_provenance("generate_battery", build_provenance(
        {"authorised": authorised, "counts": counts,
         "authorisation_record": "Tony launch sign-off via --authorised "
                                 "(prereg §12; chat record)",
         "generation_config": {"do_sample": False, "temperature": 0.0,
                               "max_new_tokens": cap, "seed": RUN_SEED,
                               "samples_per_cell": 1,
                               "protocol": "E8 sealed settings"}},
        stage="generate_battery", input_paths=[MANIFEST, disc_path]))


# ── stage 6: injection recovery (pod; SPENDS) ────────────────────────────────

def stage_injection_recovery(authorised: bool) -> None:
    """§6 (mandatory, pre-outcome): BASE model only — the transported frame
    mixed with the base-built sham at f∈{0.25,0.5,0.75} × the full manifest
    battery. The perturbation interpolates: (1−f)·frame-clamp ⊕ f·sham-clamp,
    so the f=0/f=1 endpoints ARE the main battery's transported_raw_suppress /
    sham_frame arms (never regenerated). Evaluated BEFORE any target-model
    outcome is inspected; pass = ≥80% detection of f=0.5 at one-sided α=.05
    under the executed annotation pipeline (analyse stage computes the power).
    """
    _authorise_guard("injection_recovery", authorised)
    import numpy as np
    from src.ph2_stages import (battery_cell_key, enumerate_injection_cells,
                                mixing_weights, RUN_SEED, HS_SITE, CLAMP_GAIN,
                                FRAME_PATH, class_mean_coords,
                                clamp_targets_for_arm, frame_clamp_model,
                                load_checkpoint, battery_record)
    from src.steered_inference import default_max_new_tokens
    disc_path = _require(OUT / "discovery" / "discovery_stats.json",
                         "discovery_extract")
    sham_path = _require(OUT / "frames" / "sham_base_0.npy", "target_gates")
    disc = json.loads(disc_path.read_text())
    tasks = manifest_tasks()
    cap = default_max_new_tokens()
    out_path = OUT / "injection" / "base_injection.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {battery_cell_key("base", r["method"], r["task_id"]) for r in results}
    cells = enumerate_injection_cells(tasks, done)
    if not cells:
        print("[skip] injection_recovery (complete)")
        return

    U_base = np.load(FRAME_PATH)
    U_sham = np.load(sham_path)
    states = np.load(OUT / "discovery" / "base_pair_states.npz")
    cm_frame = {k: np.array(v) for k, v in
                disc["base"]["class_means_base_frame"].items()}
    cm_sham = class_mean_coords(states[f"h_source_hs{HS_SITE}"],
                                states[f"h_base_hs{HS_SITE}"], U_sham)
    c_frame = clamp_targets_for_arm("transported_raw", "suppress", cm_frame)
    c_sham = clamp_targets_for_arm("control", "suppress", cm_sham)

    tok, model, _resolved = load_checkpoint("base")
    engines = {}
    for f in sorted({c["fraction"] for c in cells}):
        w = mixing_weights(f)
        engines[f] = frame_clamp_model(model, tok, [
            {"U": U_base, "c": c_frame, "beta": CLAMP_GAIN, "weight": w["frame"]},
            {"U": U_sham, "c": c_sham, "beta": CLAMP_GAIN, "weight": w["sham"]},
        ], HS_SITE)
    for cell in cells:
        engine = engines[cell["fraction"]]
        gen = engine.generate(cell["task"]["prompt"], max_new_tokens=cap,
                              temperature=0.0, seed=RUN_SEED)
        gen["mean_abs_displacement"] = engine.mean_abs_displacement()
        results.append(battery_record(
            "base", cell["method"], cell["task"], gen, "suppress", "mixed",
            extra={"fraction": cell["fraction"]}))
        _atomic_json(results, out_path)

    write_provenance("injection_recovery", build_provenance(
        {"authorised": authorised,
         "counts": {"generated": len(results), "planned": 3 * len(tasks)},
         "authorisation_record": "Tony launch sign-off via --authorised"},
        stage="injection_recovery", input_paths=[MANIFEST, disc_path, sham_path]))


# ── stage 7: annotation (local; SPENDS) ──────────────────────────────────────

def stage_annotate(authorised: bool) -> None:
    """Amendment A3 + A4: ALL Phase-2 behavioural verdicts + the A2 adjunct
    are annotated by Sonnet 4.5 via the lab proxy — the corpus pipeline
    (src.annotation) — over the A4 ANNOTATION WINDOW: each chain's
    paragraph-aligned first ~3,000 tokens (owner cost decision 2026-08-09;
    ~$275 vs ~$582 full-chain), uniform across arms/models. The full chain is
    preserved on every row (``chain_full``) so damage gates, boxed
    correctness, length, and truncation stay full-chain downstream. The 29-s
    AWS API-Gateway hard timeout is held by chunking (verified pre-spend by
    proxy_chunk_budget_ok); output budget 4,000 with halve-on-504 (→2,000,
    still above the worst-case echo). Sequential (≤2-concurrency etiquette
    trivially satisfied); resume-safe per-shard checkpointing after every
    chain; missing/empty rows stay unresolved (merge_annotations) — never
    zero. The builder-annotator caveat travels in provenance and in every
    verdict sentence.
    """
    _authorise_guard("annotate", authorised)
    if not (os.environ.get("CLAUDE_PROXY_URL") and os.environ.get("CLAUDE_PROXY_KEY")):
        raise SystemExit("CLAUDE_PROXY_URL/KEY not set (source ~/.zshrc) — "
                         "annotation is proxy-gated.")
    from src.annotation import annotate_chains, ANNOTATION_MODEL
    from src.ph2_stages import (ALL_ROLES, ANNOTATION_DEDUP_KEYS,
                                ANNOTATION_WINDOW_TOKENS, annotation_window,
                                proxy_chunk_budget_ok)
    budget = proxy_chunk_budget_ok()
    if not budget["ok"]:
        raise SystemExit(f"29-s chunk budget violated: {budget}")

    ann_dir = OUT / "annotation"
    ann_dir.mkdir(parents=True, exist_ok=True)
    shards = [(OUT / "battery" / f"{role}.json", ann_dir / f"{role}.json")
              for role in ALL_ROLES]
    shards.append((OUT / "injection" / "base_injection.json",
                   ann_dir / "base_injection.json"))
    status: dict = {}
    for src_path, dst_path in shards:
        if not src_path.exists():
            print(f"[warn] {src_path.name} absent — skipping shard")
            continue
        rows = []
        n_windowed = 0
        for r in json.loads(src_path.read_text()):
            win, est, truncated = annotation_window(r["chain"])
            n_windowed += int(truncated)
            rows.append({**r, "chain": win, "chain_full": r["chain"],
                         "annotated_tokens": est,
                         "annotation_window_tokens": ANNOTATION_WINDOW_TOKENS,
                         "annotation_window_truncated": truncated})
        annotated = annotate_chains(
            rows, save_path=dst_path, dedup_keys=ANNOTATION_DEDUP_KEYS,
            model=ANNOTATION_MODEL, max_tokens=ANNOTATION_MAX_TOKENS,
            shrink_on_retry=True)
        merged = merge_annotations(annotated)
        status[dst_path.name] = {
            "n_rows": len(annotated),
            "n_windowed": n_windowed,
            "n_unresolved": sum(1 for v in merged.values()
                                if v["status"] == "unresolved"),
        }
    _atomic_json(status, ann_dir / "annotation_status.json")

    write_provenance("annotate", build_provenance(
        {"authorised": authorised, "counts": status,
         "authorisation_record": "Tony launch sign-off via --authorised",
         "annotator": {
             "provider": "lab AWS proxy", "model": ANNOTATION_MODEL,
             "amendment": "A3 (Sonnet-only; Nova clause superseded)",
             "caveat": BUILDER_ANNOTATOR_CAVEAT,
             "prompt": "Venhoff appendix-A house schema (src.annotation)",
             "max_tokens": ANNOTATION_MAX_TOKENS,
             "annotation_window_tokens": ANNOTATION_WINDOW_TOKENS,
             "window_rationale": "A4 owner cost decision 2026-08-09: annotate "
                                 "the paragraph-aligned first ~3k tokens; "
                                 "generation cap SEALED at 8192, untouched",
             "retry": "<=3, backoff, halve budget on 504/timeout",
             "timeout_rule": "29-s AWS API-Gateway hard limit; chunked calls",
         }},
        stage="annotate", input_paths=[p for p, _ in shards if p.exists()]))


# ── stage 8: analysis (local, pure) ──────────────────────────────────────────

def stage_analyse(authorised: bool) -> None:
    """The full §5–§8 read-out, all pooling through the AUTHORITATIVE path
    (src.delta_floor.per_task_fraction / delta_floor_cell — the red-team F-item):

      1. per-role Δ_floor cells for every arm vs the energy-matched floor
         (primary) and the sham frame (secondary named control), B=10,000,
         seed 20260808;
      2. injection-recovery detection power (gate §6, computed BEFORE target
         outcomes are read — order enforced here by computing and writing the
         sensitivity block first);
      3. the primary paired attenuation test per target, Holm over the two
         checkpoints (§7);
      4. damage gates per active arm (sealed thresholds);
      5. decide_outcome per target (five-outcome hierarchy + downgrades);
      6. the A2 adjunct estimation table on the three vanilla arms.
    """
    import numpy as np  # noqa: F401  (transitively required by delta_floor)
    from src.delta_floor import delta_floor_cell, per_task_fraction
    from src.steering_analysis import holm_bonferroni
    from src.steered_inference import default_max_new_tokens
    from src.ph2_stages import (ALL_ROLES, TARGET_ROLES, battery_arms,
                                floor_for_battery_arm, SECONDARY_FLOOR,
                                BEHAVIOUR, CLAMP_GAIN, B_BOOT, RUN_SEED,
                                INJECTION_FRACTIONS, injection_method_label,
                                injection_detection_power, chain_stats,
                                damage_gate, paired_attenuation_test,
                                raw_retained, a2_adjunct_table)

    ann_dir = OUT / "annotation"
    bat_dir = OUT / "battery"
    cap = default_max_new_tokens()
    aliased = {}
    prov_bat = PROV_DIR / "generate_battery.json"
    if prov_bat.exists():
        aliased = (json.loads(prov_bat.read_text()).get("counts", {})
                   .get("aliased", {}))

    def load_pair(role):
        gen = json.loads(_require(bat_dir / f"{role}.json",
                                  "generate_battery").read_text())
        ann = json.loads(_require(ann_dir / f"{role}.json",
                                  "annotate").read_text())
        return gen, ann

    def arm_fractions(role, gen, ann, method):
        """Per-task sentence fractions via the AUTHORITATIVE extractor, with
        base-role aliasing resolved (aliased families read transported_raw)."""
        method = aliased.get(role, {}).get(method, method)
        if method == "vanilla":
            return per_task_fraction(gen, ann, BEHAVIOUR, "vanilla", 0.0)
        return per_task_fraction(gen, ann, BEHAVIOUR, method, CLAMP_GAIN)

    analysis: dict = {"amended": AMENDED, "caveat": BUILDER_ANNOTATOR_CAVEAT,
                      "pooling": "src.delta_floor.per_task_fraction "
                                 "(authoritative; replicates pooled per task)"}

    # ── 2 first: sensitivity gate BEFORE any target outcome is read ──────────
    inj_gen = json.loads(_require(OUT / "injection" / "base_injection.json",
                                  "injection_recovery").read_text())
    inj_ann = json.loads(_require(ann_dir / "base_injection.json",
                                  "annotate").read_text())
    base_gen, base_ann = load_pair("base")
    # Δ_floor per task at each f: floor − arm where floor = the battery's
    # energy-matched floor on base (shared across fractions).
    floor_fr = arm_fractions("base", base_gen, base_ann, "energy_matched_floor")
    f0_fr = arm_fractions("base", base_gen, base_ann, "transported_raw_suppress")
    d_f0 = {t: floor_fr[t] - f0_fr[t] for t in set(floor_fr) & set(f0_fr)}
    sensitivity: dict = {"fractions": {}}
    for f in INJECTION_FRACTIONS:
        fx_fr = per_task_fraction(inj_gen, inj_ann, BEHAVIOUR,
                                  injection_method_label(f), CLAMP_GAIN)
        d_fx = {t: floor_fr[t] - fx_fr[t] for t in set(floor_fr) & set(fx_fr)}
        sensitivity["fractions"][f"{f:.2f}"] = injection_detection_power(
            d_f0, d_fx, seed=RUN_SEED)
    gate_cell = sensitivity["fractions"].get("0.50", {})
    sensitivity["sensitivity_adequate"] = bool(gate_cell.get("passes", False))
    analysis["sensitivity"] = sensitivity
    _atomic_json(analysis, OUT / "analysis" / "ph2_analysis.json")  # pre-outcome write

    # ── 1. Δ_floor cells per role/arm ────────────────────────────────────────
    cells: dict = {}
    stats_by_role: dict = {}
    for role in ALL_ROLES:
        gen, ann = load_pair(role)
        stats_by_role[role] = (gen, ann)
        role_cells = {}
        for arm in battery_arms():
            method = arm["method"]
            floor = floor_for_battery_arm(method)
            if floor is None:
                continue
            eff_method = aliased.get(role, {}).get(method, method)
            eff_floor = aliased.get(role, {}).get(floor, floor)
            cell = delta_floor_cell(gen, ann, BEHAVIOUR, eff_method, CLAMP_GAIN,
                                    floor=eff_floor, n_resamples=B_BOOT,
                                    seed=RUN_SEED)
            sham_cell = delta_floor_cell(gen, ann, BEHAVIOUR, eff_method,
                                         CLAMP_GAIN, floor=SECONDARY_FLOOR,
                                         n_resamples=B_BOOT, seed=RUN_SEED + 1)
            role_cells[method] = {
                "vs_energy_floor": _cell_summary(cell),
                "vs_sham": _cell_summary(sham_cell),
            }
        cells[role] = role_cells
    analysis["delta_floor_cells"] = cells

    # ── 4. damage gates ──────────────────────────────────────────────────────
    damage: dict = {}
    for role in ALL_ROLES:
        gen, _ann = stats_by_role[role]
        by_method: dict = {}
        for r in gen:
            by_method.setdefault(r["method"], []).append(
                chain_stats(r["chain"], r["n_tokens"], cap,
                            r.get("expected_answer")))
        floor_rows = by_method.get("energy_matched_floor", [])
        van_rows = by_method.get("vanilla", [])
        damage[role] = {
            m: damage_gate(rows, floor_rows, van_rows)
            for m, rows in by_method.items()
            if m not in ("vanilla", "energy_matched_floor")}
    analysis["damage"] = damage

    # ── 3. primary attenuation, Holm over the two checkpoints ────────────────
    primary: dict = {}
    d_base = {t: floor_fr[t] - f0_fr[t] for t in set(floor_fr) & set(f0_fr)}
    for role in TARGET_ROLES:
        gen, ann = stats_by_role[role]
        t_floor = arm_fractions(role, gen, ann, "energy_matched_floor")
        t_raw = arm_fractions(role, gen, ann, "transported_raw_suppress")
        d_tgt = {t: t_floor[t] - t_raw[t] for t in set(t_floor) & set(t_raw)}
        primary[role] = paired_attenuation_test(d_base, d_tgt)
    holm = holm_bonferroni({r: primary[r]["raw_p"] for r in primary
                            if primary[r].get("raw_p") is not None})
    for role, padj in holm.items():
        primary[role]["holm_p"] = float(padj)
    analysis["primary_attenuation"] = primary

    # ── 5. verdicts ──────────────────────────────────────────────────────────
    gates_doc = {}
    gp = OUT / "gates" / "target_gates.json"
    if gp.exists():
        gates_doc = json.loads(gp.read_text())
    refit_doc = {}
    rp = OUT / "refit" / "refit_report.json"
    if rp.exists():
        refit_doc = json.loads(rp.read_text())

    verdicts: dict = {}
    for role in TARGET_ROLES:
        rc = cells[role]

        def _passes(method, key="vs_energy_floor"):
            c = rc.get(method, {}).get(key, {})
            b = c.get("bootstrap") or {}
            return bool(c.get("delta_floor") is not None
                        and c["delta_floor"] > 0 and b.get("excludes_zero"))

        att = primary[role]
        align = ((refit_doc.get(role, {}).get("widths", {})
                  .get("2", {}) or {}).get("alignment_vs_base") or {})
        gate_pass = bool(gates_doc.get(role, {}).get("gate_pass", False))
        corrected_pass = (_passes("transported_norm_suppress")
                          or _passes("transported_whitened_suppress"))
        o = {
            "sensitivity_adequate": sensitivity["sensitivity_adequate"],
            "damage_ok": bool(damage.get(role, {})
                              .get("transported_raw_suppress", {})
                              .get("damage_ok", True)),
            "gate_pass": gate_pass,
            "raw_retained": bool(_passes("transported_raw_suppress")
                                 and raw_retained(att)),
            "corrected_gate_pass": gate_pass,
            "corrected_retained": corrected_pass,
            "refit_pass": _passes("refit_suppress"),
            "refit_aligned": bool(align.get("refit_aligned", False)),
            "refit_misaligned_beyond_null":
                bool(align.get("refit_misaligned_beyond_null", False)),
            "refit_recovered": _passes("refit_suppress"),
            "repr_signal_above_null": bool(gate_pass
                                           or _passes("refit_suppress")),
        }
        verdicts[role] = {"inputs": o, "outcome": decide_outcome(o),
                          "wording_caveat": BUILDER_ANNOTATOR_CAVEAT}
    analysis["verdicts"] = verdicts

    # ── 6. A2 adjunct (estimation only) ─────────────────────────────────────
    # A4: prevalence/per-1k read the WINDOWED annotations (with annotated_tokens
    # as the per-1k denominator); chain-level endpoints (boxed, rep4, length,
    # truncation) must see the FULL chain — restore it from chain_full.
    ann_vanilla = {}
    for role in ALL_ROLES:
        _gen, ann = stats_by_role[role]
        ann_vanilla[role] = [
            {**r, "chain": r.get("chain_full", r["chain"])}
            for r in ann if r["method"] == "vanilla"]
    analysis["a2_adjunct"] = a2_adjunct_table(ann_vanilla, cap)

    _atomic_json(analysis, OUT / "analysis" / "ph2_analysis.json")
    write_provenance("analyse", build_provenance(
        {"authorised": authorised,
         "achieved_mde": sensitivity,
         "annotator": {"model": "sonnet-4.5 (A3)",
                       "caveat": BUILDER_ANNOTATOR_CAVEAT},
         "analysis_config": {
             "pooling": "src.delta_floor.per_task_fraction",
             "bootstrap": {"B": B_BOOT, "seed": RUN_SEED, "kind": "paired BCa"},
             "multiplicity": "Holm over {STAR1, DeepScaleR} primary cells",
             "alpha": 0.05,
         }},
        stage="analyse", evidence_status="provisional",
        input_paths=[bat_dir / f"{r}.json" for r in ALL_ROLES]))
    print(json.dumps({r: v["outcome"] for r, v in verdicts.items()}, indent=1))


def _cell_summary(cell) -> dict:
    b = cell.bootstrap
    return {"delta_floor": cell.delta_floor, "n_tasks": cell.n_tasks,
            "raw_p": cell.raw_p, "status": cell.status,
            "bootstrap": None if b is None else {
                "estimate": b.estimate, "ci_low": b.ci_low, "ci_high": b.ci_high,
                "excludes_zero": b.excludes_zero()},
            "transitions": None if cell.transitions is None else
            {k: v for k, v in cell.transitions.items() if k != "diffs"}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--authorised", action="store_true",
                    help="Tony's launch sign-off (recorded in provenance)")
    a = ap.parse_args()
    if not MANIFEST.exists():
        raise SystemExit("No task manifest — run ph2_manifest.py --generate first (A1).")
    if a.stage in SPEND_STAGES and not a.authorised:
        raise SystemExit(f"stage {a.stage} spends — requires --authorised (Tony).")
    run_stage(a.stage, a.authorised)
