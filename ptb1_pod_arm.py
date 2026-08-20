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

#: Amendment 2 gate: >= MIN_PREFIX_PROBES probes must share a leading
#: identical run of >= MIN_PREFIX_CHARS characters with their stored chain.
MIN_PREFIX_CHARS = 200
MIN_PREFIX_PROBES = 2
MANIFEST_SHA256 = "69cbe32dc7991c42ee58c8b5fae683750abb6a8812444ab320e68e23b8214333"
PROMPT_TEXT_SHA256 = "f6b91123016196596f265a4a910e780508f7c7277325c1c87344dc63e7252211"
BASE_BATTERY_SHA256 = "fab5a0d1366a8158b77c2260147a8daa2928827d3752be784e452759d6983832"


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def preflight() -> None:
    """Amendment 2 gate: direct prompt-identity hashes + prefix agreement.

    Exact chain reproduction is a DIAGNOSTIC, not a gate: greedy decoding is
    hardware/library-sensitive, so a single flipped logit tie separates chains
    irreversibly without any prompt drift. See PTB1_AMENDMENT_2_2026-08-20.md
    (post-hoc, disclosed) for the diagnosis that motivated the replacement.
    """
    import hashlib

    import ph2_manifest
    from ph2_executor import manifest_tasks
    from src.ph2_stages import load_checkpoint

    manifest_path = ROOT / "results/prereg/phase2_task_manifest.json"
    battery_path = ROOT / "results/ph2/battery/base.json"
    doc = json.loads(manifest_path.read_text())
    if ph2_manifest.manifest_hash(doc["tasks"]) != doc["ids_sha256"]:
        fail("preflight", "task manifest ids_sha256 does not re-derive")

    prompt_text = "".join(t["prompt"] for t in
                          sorted(doc["tasks"], key=lambda x: x["id"]))
    hashes = {
        "manifest_file": (sha256(manifest_path), MANIFEST_SHA256),
        "prompt_text": (hashlib.sha256(prompt_text.encode()).hexdigest(),
                        PROMPT_TEXT_SHA256),
        "base_battery": (sha256(battery_path), BASE_BATTERY_SHA256),
    }
    for name, (got, want) in hashes.items():
        if got != want:
            fail("preflight", f"{name} sha256 mismatch: {got} != {want}")

    stored = [r for r in json.loads(battery_path.read_text())
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
        prefix = _common_prefix_len(gen["chain"], row["chain"])
        cases.append({"task_id": row["task_id"],
                      "stored_n_tokens": row["n_tokens"],
                      "identical": gen["chain"] == row["chain"],
                      "common_prefix_chars": prefix,
                      "prefix_ok": prefix >= MIN_PREFIX_CHARS})
    del model
    n_ident = sum(1 for c in cases if c["identical"])
    n_prefix = sum(1 for c in cases if c["prefix_ok"])
    report = {
        "amendment": "PTB1_AMENDMENT_2_2026-08-20.md",
        "hashes_verified": {k: v[0] for k, v in hashes.items()},
        "cases": cases, "n_identical": n_ident, "n_prefix_ok": n_prefix,
        "resolved_checkpoint": resolved,
        "rule": f"PASS if >= {MIN_PREFIX_PROBES}/3 probes share >= "
                f"{MIN_PREFIX_CHARS} identical leading chars AND all three "
                f"hashes match; exact reproduction is diagnostic only",
        "environment_note": "PT-B1 arms are NOT byte-equivalent to the July "
                            "Phase-2 environment; the safety-minus-control "
                            "primary is unaffected (common environment, "
                            "paired), base-referenced comparisons carry a "
                            "disclosed environment confound",
    }
    _atomic_json(report, STATUS / "preflight.json")
    if n_prefix < MIN_PREFIX_PROBES:
        fail("preflight", f"only {n_prefix}/3 probes reached "
                          f"{MIN_PREFIX_CHARS} identical leading chars — "
                          f"prompt drift cannot be excluded")
    done_marker("preflight", report)
    print(json.dumps({"n_prefix_ok": n_prefix, "n_identical": n_ident,
                      "prefix_chars": [c["common_prefix_chars"]
                                       for c in cases]}, indent=1))


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


def _extract_subset(model, tok, chains, save_dir: Path) -> None:
    from src.activation_extraction import extract_activations
    extract_activations(model, tok, chains, layers=list(GATE_LAYERS),
                        save_dir=save_dir, behaviours=list(BEHAVIOURS),
                        n_preceding=1, n_execution=10, pooling="mean",
                        sweep_modes=[], clip_to_sentence_end=False,
                        keep_in_memory=False)


def base_reference_currentenv() -> Path:
    """Amendment 3: extract the BASE checkpoint on the verification subset in
    THIS environment, once per session, and cache it.

    The adapter displacement being gated is ~0.5% of activation norm, while
    cross-environment drift on these activations is ~6x larger. Subtracting a
    July baseline from a today extraction therefore measures drift, not the
    adapter. Both terms of every G1/G2 cosine must come from one environment.
    """
    out = GATE_DIR / "base_currentenv"
    if (out / "row_index.json").exists():
        return out
    from src.ph2_stages import load_checkpoint
    chains = json.loads((GATE_DIR / "verification_chains.json").read_text())
    tok, model, resolved = load_checkpoint("base")
    _extract_subset(model, tok, chains, out)
    del model
    _atomic_json({"resolved_checkpoint": resolved,
                  "amendment": "PTB1_AMENDMENT_3_2026-08-20.md",
                  "purpose": "same-environment G1/G2 baseline"},
                 out / "BASE_REFERENCE_META.json")
    return out


def _subset_rows(act_dir: Path, layer: int, ref_keys: list, idx: list):
    import numpy as np
    pos = key_positions(load_row_keys(act_dir))
    mats = {b: np.load(act_dir / f"{b}_layer{layer}.npy", mmap_mode="r")
            for b in BEHAVIOURS}
    return np.stack([mats[ref_keys[i][0]][pos[ref_keys[i]]]
                     for i in idx]).astype(np.float32)


def identity_gate(arm: str, merged: Path):
    """Sealed G1/G2 gate; returns (verdict_dict, tok, model) on PASS/AMENDED.

    Amendment 3: the arm direction is (arm_new - base_new) with BOTH terms
    extracted in the current environment; the reference direction stays the
    within-July (arm_stored - base_stored). Environment drift cancels on both
    sides, so the cosine compares adapter effects rather than library drift.
    """
    import numpy as np
    from src.ph2_stages import load_checkpoint

    ref = np.load(GATE_DIR / "reference.npz")
    ref_keys = [tuple(k) for k in
                json.loads((GATE_DIR / "reference_keys.json").read_text())]
    chains = json.loads((GATE_DIR / "verification_chains.json").read_text())

    base_env_dir = base_reference_currentenv()
    tok, model, resolved = load_checkpoint("base", local_dir=merged)
    tmp = GATE_DIR / f"extract_{arm}"
    _extract_subset(model, tok, chains, tmp)

    new_pos = key_positions(load_row_keys(tmp))
    base_pos = key_positions(load_row_keys(base_env_dir))
    idx = [i for i, k in enumerate(ref_keys)
           if k in new_pos and k in base_pos]
    coverage = len(idx) / len(ref_keys)
    cos_same, cos_opp, norms = {}, {}, {}
    cos = lambda u, v: float(np.dot(u, v)
                             / (np.linalg.norm(u) * np.linalg.norm(v)))
    for layer in GATE_LAYERS:
        arm_new = _subset_rows(tmp, layer, ref_keys, idx)
        base_new = _subset_rows(base_env_dir, layer, ref_keys, idx)
        base_stored = ref[f"base_l{layer}"][idx].astype(np.float32)
        stored_same = (ref[f"arm_{arm}_l{layer}"][idx].astype(np.float32)
                       - base_stored).mean(axis=0)
        stored_opp = (ref[f"arm_{opposite_arm(arm)}_l{layer}"][idx]
                      .astype(np.float32) - base_stored).mean(axis=0)
        new_dir = (arm_new - base_new).mean(axis=0)          # same-environment
        env_drift = (base_new - base_stored).mean(axis=0)    # diagnostic only
        cos_same[str(layer)] = cos(new_dir, stored_same)
        cos_opp[str(layer)] = cos(new_dir, stored_opp)
        norms[str(layer)] = {
            "new_same_env": float(np.linalg.norm(new_dir)),
            "stored": float(np.linalg.norm(stored_same)),
            "environment_drift_diagnostic": float(np.linalg.norm(env_drift))}

    verdict = gate_verdict(cos_same, cos_opp, coverage)
    report = {"arm": arm, "verdict": verdict, "cos_same_seed": cos_same,
              "cos_opposite_recipe": cos_opp, "direction_norms": norms,
              "baseline": "same-environment (PTB1_AMENDMENT_3_2026-08-20.md)",
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
    # Resume guard: a finished arm (gate passed + full battery on disk) is not
    # retrained. Its merged checkpoint is deleted on completion, so without
    # this the loop would pay the training cost again to reach a no-op.
    bat = BAT_DIR / f"{arm}.json"
    if (STATUS / f"DONE_arm_{arm}.json").exists() and bat.exists():
        from ph2_executor import manifest_tasks
        if len(json.loads(bat.read_text())) >= len(manifest_tasks()):
            print(f"[skip] arm {arm} already complete")
            return
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
