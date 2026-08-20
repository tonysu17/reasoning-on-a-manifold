#!/usr/bin/env python3
"""PT-B1 pod-side driver — preflight, then per arm: train -> gate -> generate.

Sealed authority: results/prereg/PTB1_SAFETY_CONTROL_BEHAVIOUR_PREREG_2026-08-20.md
(+ Amendment 1 for the preflight operationalisation). Single source of truth
for the arm table and gate bands is ptb1_executor.py.

    python ptb1_pod_arm.py preflight
    python ptb1_pod_arm.py arm <arm-name>

Fail-closed: every failure writes results/ptb1/status/FAILED_<stage>.json and
exits non-zero; nothing is deleted on failure. The merged checkpoint is
deleted only after its arm completes; the small LoRA adapter files are
preserved under results/ptb1/adapters/<arm>/ with hashes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from ptb1_executor import (ARMS, BEHAVIOURS, GATE_LAYERS, MIN_ROW_COVERAGE,
                           OUT, PTB1_SEED, ROOT, _atomic_json, gate_verdict,
                           key_positions, load_row_keys, opposite_arm, sha256,
                           write_provenance)

STATUS = OUT / "status"
GATE_DIR = OUT / "identity_gate"
BAT_DIR = OUT / "battery"
ADAPTER_DIR = OUT / "adapters"
CKPT_ROOT = ROOT / "checkpoints"

#: Verbatim pinned training recipe (prereg §2; spark_seedrep_remote.sh).
TRAIN_ARGS = ["--dose", "all", "--merge", "--epochs", "5", "--lr", "1e-5",
              "--batch-size", "4", "--grad-accum", "32", "--max-len", "4096"]


def fail(stage: str, reason: str) -> "NoReturn":
    _atomic_json({"stage": stage, "reason": reason}, STATUS / f"FAILED_{stage}.json")
    print(f"FAILED {stage}: {reason}", file=sys.stderr)
    raise SystemExit(1)


def done_marker(stage: str, payload: dict) -> None:
    _atomic_json(payload, STATUS / f"DONE_{stage}.json")


# ── preflight (Amendment 1) ──────────────────────────────────────────────────

def preflight() -> None:
    import ph2_manifest
    from ph2_executor import manifest_tasks
    from src.ph2_stages import load_checkpoint

    doc = json.loads((ROOT / "results/prereg/phase2_task_manifest.json").read_text())
    if ph2_manifest.manifest_hash(doc["tasks"]) != doc["ids_sha256"]:
        fail("preflight", "task manifest ids_sha256 does not re-derive")

    stored = [r for r in json.loads(
        (ROOT / "results/ph2/battery/base.json").read_text())
        if r["method"] == "vanilla"]
    stored.sort(key=lambda r: r["n_tokens"])
    probes = stored[:3]
    tasks = {t["id"]: t for t in manifest_tasks()}

    tok, model, resolved = load_checkpoint("base")
    from src.ph2_stages import frame_clamp_model, HS_SITE
    engine = frame_clamp_model(model, tok, [], HS_SITE)
    cases = []
    for row in probes:
        task = tasks[row["task_id"]]
        gen = engine.generate(task["prompt"], max_new_tokens=row["n_tokens"])
        cases.append({"task_id": row["task_id"],
                      "stored_n_tokens": row["n_tokens"],
                      "identical": gen["chain"] == row["chain"]})
    n_ident = sum(1 for c in cases if c["identical"])
    report = {"manifest_ids_sha256": doc["ids_sha256"], "cases": cases,
              "n_identical": n_ident, "resolved_checkpoint": resolved,
              "rule": "3/3 full verify; 1-2/3 proceed with env-divergence "
                      "disclosure; 0/3 STOP (Amendment 1)"}
    _atomic_json(report, STATUS / "preflight.json")
    del model
    if n_ident == 0:
        fail("preflight", "0/3 chain reproductions identical — prompt drift "
                          "cannot be excluded")
    done_marker("preflight", report)
    print(json.dumps({"n_identical": n_ident}, indent=1))


# ── per-arm pipeline ─────────────────────────────────────────────────────────

def train_arm(arm: str) -> Path:
    data_file, seed, _stored = ARMS[arm]
    ckpt = CKPT_ROOT / f"ptb1_{arm}"
    merged = None
    for dose_dir in ("dose_all", "dose_1000"):
        cand = ckpt / dose_dir / "merged"
        if (cand / "config.json").exists():
            merged = cand
    if merged is None:
        cmd = [sys.executable, "-u", str(ROOT / "pt02_train_safety_lora.py"),
               "--data", str(ROOT / data_file), *TRAIN_ARGS,
               "--seed", str(seed), "--out-dir", str(ckpt)]
        print("TRAIN:", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            fail(f"train_{arm}", f"pt02 exited {rc}")
        for dose_dir in ("dose_all", "dose_1000"):
            cand = ckpt / dose_dir / "merged"
            if (cand / "config.json").exists():
                merged = cand
        if merged is None:
            fail(f"train_{arm}", f"no merged checkpoint under {ckpt}")

    # Preserve the small LoRA adapter files + summary, with hashes.
    keep = ADAPTER_DIR / arm
    keep.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for pattern in ("adapter_*.json", "adapter_*.safetensors",
                    "adapter_*.bin", "training_summary.json"):
        for src in list(merged.parent.glob(pattern)) + list(ckpt.glob(pattern)):
            dst = keep / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            hashes[src.name] = sha256(dst)
    _atomic_json({"arm": arm, "data": data_file, "seed": seed,
                  "train_args": TRAIN_ARGS, "adapter_sha256": hashes},
                 keep / "ADAPTER_MANIFEST.json")
    return merged


def identity_gate(arm: str, merged: Path):
    """Sealed G1/G2 gate; returns (verdict_dict, tok, model) on PASS/AMENDED."""
    import numpy as np
    from src.activation_extraction import extract_activations
    from src.ph2_stages import load_checkpoint

    ref = np.load(GATE_DIR / "reference.npz")
    ref_keys = [tuple(k) for k in
                json.loads((GATE_DIR / "reference_keys.json").read_text())]
    chains = json.loads((GATE_DIR / "verification_chains.json").read_text())

    tok, model, resolved = load_checkpoint("base", local_dir=merged)
    tmp = OUT / "identity_gate" / f"extract_{arm}"
    extract_activations(model, tok, chains, layers=list(GATE_LAYERS),
                        save_dir=tmp, behaviours=list(BEHAVIOURS),
                        n_preceding=1, n_execution=10, pooling="mean",
                        sweep_modes=[], clip_to_sentence_end=False,
                        keep_in_memory=False)

    new_pos = key_positions(load_row_keys(tmp))
    idx = [i for i, k in enumerate(ref_keys) if k in new_pos]
    coverage = len(idx) / len(ref_keys)
    cos_same, cos_opp, norms = {}, {}, {}
    for layer in GATE_LAYERS:
        mats = {b: np.load(tmp / f"{b}_layer{layer}.npy", mmap_mode="r")
                for b in BEHAVIOURS}
        new_rows = np.stack([mats[ref_keys[i][0]][new_pos[ref_keys[i]]]
                             for i in idx]).astype(np.float32)
        base_rows = ref[f"base_l{layer}"][idx].astype(np.float32)
        stored_same = (ref[f"arm_{arm}_l{layer}"][idx].astype(np.float32)
                       - base_rows).mean(axis=0)
        stored_opp = (ref[f"arm_{opposite_arm(arm)}_l{layer}"][idx]
                      .astype(np.float32) - base_rows).mean(axis=0)
        new_dir = (new_rows - base_rows).mean(axis=0)
        cos = lambda u, v: float(np.dot(u, v)
                                 / (np.linalg.norm(u) * np.linalg.norm(v)))
        cos_same[str(layer)] = cos(new_dir, stored_same)
        cos_opp[str(layer)] = cos(new_dir, stored_opp)
        norms[str(layer)] = {"new": float(np.linalg.norm(new_dir)),
                             "stored": float(np.linalg.norm(stored_same))}

    verdict = gate_verdict(cos_same, cos_opp, coverage)
    report = {"arm": arm, "verdict": verdict, "cos_same_seed": cos_same,
              "cos_opposite_recipe": cos_opp, "direction_norms": norms,
              "row_coverage": coverage, "n_rows_compared": len(idx),
              "n_rows_reference": len(ref_keys),
              "resolved_checkpoint": resolved,
              "reference_npz_sha256": sha256(GATE_DIR / "reference.npz"),
              "extracted_npy_sha256": {
                  f"{b}_layer{layer}.npy": sha256(tmp / f"{b}_layer{layer}.npy")
                  for b in BEHAVIOURS for layer in GATE_LAYERS}}
    _atomic_json(report, GATE_DIR / f"gate_{arm}.json")
    if verdict["verdict"] == "STOP":
        del model
        fail(f"gate_{arm}", verdict["reason"])
    return report, tok, model


def generate_arm(arm: str, tok, model) -> None:
    from ph2_executor import manifest_tasks
    from src.ph2_stages import (HS_SITE, RUN_SEED, battery_record,
                                frame_clamp_model, identity_reload_gate)
    from src.steered_inference import default_max_new_tokens

    tasks = manifest_tasks()
    cap = default_max_new_tokens()
    out_path = BAT_DIR / f"{arm}.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["task_id"] for r in results}
    todo = [t for t in tasks if t["id"] not in done]
    if not todo:
        print(f"[skip] {arm} battery complete")
        return

    gate = identity_reload_gate(model, tok, tasks, HS_SITE)
    if not gate["pass"]:
        fail(f"generate_{arm}", f"identity/reload gate failed: {gate}")
    engine = frame_clamp_model(model, tok, [], HS_SITE)
    BATCH = 32   # mirrors ph2_executor.BATTERY_BATCH
    for b0 in range(0, len(todo), BATCH):
        chunk = todo[b0:b0 + BATCH]
        gens = engine.generate_batch([t["prompt"] for t in chunk],
                                     max_new_tokens=cap, temperature=0.0,
                                     seed=RUN_SEED)
        for task, gen in zip(chunk, gens):
            rec = battery_record(arm, "vanilla", task, gen, None, None,
                                 extra={"expected_answer": None,
                                        "ptb1_seed": PTB1_SEED})
            results.append(rec)
        _atomic_json(results, out_path)
        print(f"[{arm}] {len(results)}/{len(tasks)}", flush=True)


def run_arm(arm: str) -> None:
    if arm not in ARMS:
        fail("arm", f"unknown arm {arm!r}")
    if not (STATUS / "DONE_preflight.json").exists():
        fail(f"arm_{arm}", "preflight has not passed on this host")
    merged = train_arm(arm)
    report, tok, model = identity_gate(arm, merged)
    generate_arm(arm, tok, model)
    del model
    write_provenance(f"arm_{arm}", {
        "authorised": True, "gate": report["verdict"],
        "cos_same_seed": report["cos_same_seed"],
        "generation_config": {"do_sample": False, "temperature": 0.0,
                              "max_new_tokens": 8192, "seed": 20260808,
                              "samples_per_cell": 1,
                              "protocol": "E8 sealed settings"}})
    shutil.rmtree(CKPT_ROOT / f"ptb1_{arm}", ignore_errors=True)
    done_marker(f"arm_{arm}", {"gate": report["verdict"],
                               "n_rows": len(json.loads(
                                   (BAT_DIR / f"{arm}.json").read_text()))})
    print(f"ARM {arm} COMPLETE ({report['verdict']['verdict']})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight")
    p_arm = sub.add_parser("arm")
    p_arm.add_argument("name", choices=sorted(ARMS))
    args = ap.parse_args()
    STATUS.mkdir(parents=True, exist_ok=True)
    BAT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cmd == "preflight":
        preflight()
    else:
        run_arm(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
